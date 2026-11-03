//! Camera capture on Linux with V4L2: raw syscalls (no libc needed), mmap
//! streaming with four buffers, YUYV converted to BGRA, MJPEG decoded by
//! VISAR's own JPEG decoder. A capture thread waits for buffers with poll,
//! converts each into one BGRA image and hands it to the sink.
//!
//! Cameras are numbered in the order they are found among /dev/video0-63,
//! counting only nodes that capture video (UVC cameras also expose
//! metadata nodes).

const std = @import("std");
const builtin = @import("builtin");
const linux = std.os.linux;
const camera = @import("camera.zig");
const core = @import("../core/core.zig");
const jpeg = @import("../codecs/jpeg.zig");
const Image = core.Image;

const Error = camera.Error;

// ---------------------------------------------------------------------------
// The kernel ABI (include/uapi/linux/videodev2.h), 64-bit layouts
// ---------------------------------------------------------------------------

const Capability = extern struct {
    driver: [16]u8,
    card: [32]u8,
    bus_info: [32]u8,
    version: u32,
    capabilities: u32,
    device_caps: u32,
    reserved: [3]u32,
};

const FmtDesc = extern struct {
    index: u32,
    type: u32,
    flags: u32,
    description: [32]u8,
    pixelformat: u32,
    mbus_code: u32,
    reserved: [3]u32,
};

const FrmSizeEnum = extern struct {
    index: u32,
    pixel_format: u32,
    type: u32,
    /// discrete: width, height; stepwise: min_w, max_w, step_w, min_h, max_h, step_h.
    u: [6]u32,
    reserved: [2]u32,
};

const FrmIvalEnum = extern struct {
    index: u32,
    pixel_format: u32,
    width: u32,
    height: u32,
    type: u32,
    /// discrete: numerator, denominator; stepwise: min, max, step (fractions).
    u: [6]u32,
    reserved: [2]u32,
};

const PixFormat = extern struct {
    width: u32,
    height: u32,
    pixelformat: u32,
    field: u32,
    bytesperline: u32,
    sizeimage: u32,
    colorspace: u32,
    priv: u32,
    flags: u32,
    ycbcr_enc: u32,
    quantization: u32,
    xfer_func: u32,
};

const Format = extern struct {
    type: u32,
    /// The kernel's union is 8-byte aligned (it holds pointers).
    pad: u32 = 0,
    pix: PixFormat,
    rest: [152]u8 = @splat(0),
};

const RequestBuffers = extern struct {
    count: u32,
    type: u32,
    memory: u32,
    capabilities: u32 = 0,
    flags: u8 = 0,
    reserved: [3]u8 = @splat(0),
};

const TimeCode = extern struct { type: u32, flags: u32, frames: u8, seconds: u8, minutes: u8, hours: u8, userbits: [4]u8 };

const Buffer = extern struct {
    index: u32 = 0,
    type: u32 = 0,
    bytesused: u32 = 0,
    flags: u32 = 0,
    field: u32 = 0,
    timestamp: extern struct { sec: c_long, usec: c_long } = .{ .sec = 0, .usec = 0 },
    timecode: TimeCode = std.mem.zeroes(TimeCode),
    sequence: u32 = 0,
    memory: u32 = 0,
    m: extern union { offset: u32, userptr: c_ulong, planes: usize, fd: i32 } = .{ .userptr = 0 },
    length: u32 = 0,
    reserved2: u32 = 0,
    request_fd: i32 = 0,
};

const StreamParm = extern struct {
    type: u32,
    capability: u32 = 0,
    capturemode: u32 = 0,
    numerator: u32 = 0,
    denominator: u32 = 0,
    extendedmode: u32 = 0,
    readbuffers: u32 = 0,
    reserved: [4]u32 = @splat(0),
    rest: [160]u8 = @splat(0),
};

comptime {
    if (@sizeOf(usize) == 8) {
        std.debug.assert(@sizeOf(Capability) == 104);
        std.debug.assert(@sizeOf(FmtDesc) == 64);
        std.debug.assert(@sizeOf(FrmSizeEnum) == 44);
        std.debug.assert(@sizeOf(FrmIvalEnum) == 52);
        std.debug.assert(@sizeOf(Format) == 208);
        std.debug.assert(@sizeOf(RequestBuffers) == 20);
        std.debug.assert(@sizeOf(Buffer) == 88);
        std.debug.assert(@sizeOf(StreamParm) == 204);
    }
}

const IOCTL = linux.IOCTL;
const VIDIOC_QUERYCAP = IOCTL.IOR('V', 0, Capability);
const VIDIOC_ENUM_FMT = IOCTL.IOWR('V', 2, FmtDesc);
const VIDIOC_S_FMT = IOCTL.IOWR('V', 5, Format);
const VIDIOC_REQBUFS = IOCTL.IOWR('V', 8, RequestBuffers);
const VIDIOC_QUERYBUF = IOCTL.IOWR('V', 9, Buffer);
const VIDIOC_QBUF = IOCTL.IOWR('V', 15, Buffer);
const VIDIOC_DQBUF = IOCTL.IOWR('V', 17, Buffer);
const VIDIOC_STREAMON = IOCTL.IOW('V', 18, c_int);
const VIDIOC_STREAMOFF = IOCTL.IOW('V', 19, c_int);
const VIDIOC_G_PARM = IOCTL.IOWR('V', 21, StreamParm);
const VIDIOC_S_PARM = IOCTL.IOWR('V', 22, StreamParm);
const VIDIOC_ENUM_FRAMESIZES = IOCTL.IOWR('V', 74, FrmSizeEnum);
const VIDIOC_ENUM_FRAMEINTERVALS = IOCTL.IOWR('V', 75, FrmIvalEnum);

const BUF_TYPE_VIDEO_CAPTURE: u32 = 1;
const MEMORY_MMAP: u32 = 1;
const CAP_VIDEO_CAPTURE: u32 = 0x1;
const CAP_STREAMING: u32 = 0x0400_0000;
const CAP_DEVICE_CAPS: u32 = 0x8000_0000;
const FRMSIZE_TYPE_DISCRETE: u32 = 1;
const FRMIVAL_TYPE_DISCRETE: u32 = 1;
const FIELD_NONE: u32 = 1;

fn fourcc(s: *const [4]u8) u32 {
    return std.mem.readInt(u32, s, .little);
}
const PIX_YUYV = fourcc("YUYV");
const PIX_MJPG = fourcc("MJPG");

fn ioctl(fd: linux.fd_t, req: u32, arg: anytype) linux.E {
    while (true) {
        const e = linux.errno(linux.ioctl(fd, req, @intFromPtr(arg)));
        if (e != .INTR) return e;
    }
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

fn openNode(n: usize, buf: *[32]u8) ?linux.fd_t {
    const path = std.fmt.bufPrintSentinel(buf, "/dev/video{d}", .{n}, 0) catch return null;
    const rc = linux.open(path, .{ .ACCMODE = .RDWR, .NONBLOCK = true, .CLOEXEC = true }, 0);
    if (linux.errno(rc) != .SUCCESS) return null;
    return @intCast(rc);
}

/// True for a node that captures video by streaming.
fn isCapture(fd: linux.fd_t, cap: *Capability) bool {
    if (ioctl(fd, VIDIOC_QUERYCAP, cap) != .SUCCESS) return false;
    const caps = if (cap.capabilities & CAP_DEVICE_CAPS != 0) cap.device_caps else cap.capabilities;
    return caps & CAP_VIDEO_CAPTURE != 0 and caps & CAP_STREAMING != 0;
}

/// The capture formats VISAR can convert (YUYV, MJPEG), with every size
/// and its fastest rate.
fn formats(a: std.mem.Allocator, fd: linux.fd_t) ![]camera.Format {
    var out: std.ArrayList(camera.Format) = .empty;
    var fi: u32 = 0;
    while (fi < 64) : (fi += 1) {
        var desc = std.mem.zeroes(FmtDesc);
        desc.index = fi;
        desc.type = BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(fd, VIDIOC_ENUM_FMT, &desc) != .SUCCESS) break;
        if (desc.pixelformat != PIX_YUYV and desc.pixelformat != PIX_MJPG) continue;
        var si: u32 = 0;
        while (si < 256) : (si += 1) {
            var fs = std.mem.zeroes(FrmSizeEnum);
            fs.index = si;
            fs.pixel_format = desc.pixelformat;
            if (ioctl(fd, VIDIOC_ENUM_FRAMESIZES, &fs) != .SUCCESS) break;
            // Stepwise sizes: list the largest.
            const w = if (fs.type == FRMSIZE_TYPE_DISCRETE) fs.u[0] else fs.u[1];
            const h = if (fs.type == FRMSIZE_TYPE_DISCRETE) fs.u[1] else fs.u[4];
            var f: camera.Format = .{ .width = w, .height = h, .min_fps = 0, .max_fps = 0, .pixel_format = @bitCast(desc.pixelformat) };
            var ii: u32 = 0;
            var min: f64 = std.math.inf(f64);
            while (ii < 64) : (ii += 1) {
                var iv = std.mem.zeroes(FrmIvalEnum);
                iv.index = ii;
                iv.pixel_format = desc.pixelformat;
                iv.width = w;
                iv.height = h;
                if (ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, &iv) != .SUCCESS) break;
                // Discrete: one interval; stepwise: min interval = max rate.
                const fast = rateOf(iv.u[0], iv.u[1]);
                const slow = if (iv.type == FRMIVAL_TYPE_DISCRETE) fast else rateOf(iv.u[2], iv.u[3]);
                f.max_fps = @max(f.max_fps, fast);
                min = @min(min, slow);
                if (iv.type != FRMIVAL_TYPE_DISCRETE) break;
            }
            f.min_fps = if (std.math.isInf(min)) 0 else min;
            if (f.max_fps == 0) f.max_fps = 30; // the driver lists no rates
            try out.append(a, f);
            if (fs.type != FRMSIZE_TYPE_DISCRETE) break;
        }
    }
    return out.items;
}

fn rateOf(num: u32, den: u32) f64 {
    if (num == 0) return 0;
    return @as(f64, @floatFromInt(den)) / @as(f64, @floatFromInt(num));
}

pub fn permission() camera.Permission {
    return .not_applicable;
}

pub fn list(gpa: std.mem.Allocator) Error!camera.DeviceList {
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();
    var devs: std.ArrayList(camera.Device) = .empty;
    var denied = false;
    for (0..64) |n| {
        var pbuf: [32]u8 = undefined;
        const fd = openNode(n, &pbuf) orelse {
            const path = std.fmt.bufPrintSentinel(&pbuf, "/dev/video{d}", .{n}, 0) catch continue;
            const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
            if (linux.errno(rc) == .ACCES) denied = true else if (linux.errno(rc) == .SUCCESS) _ = linux.close(@intCast(rc));
            continue;
        };
        defer _ = linux.close(fd);
        var cap: Capability = undefined;
        if (!isCapture(fd, &cap)) continue;
        const name = std.mem.sliceTo(&cap.card, 0);
        try devs.append(a, .{
            .name = try a.dupe(u8, name),
            .id = try std.fmt.allocPrint(a, "/dev/video{d}", .{n}),
            .is_default = devs.items.len == 0,
            .formats = try formats(a, fd),
        });
    }
    return .{ .arena = arena, .devices = devs.items, .permission = if (denied and devs.items.len == 0) .denied else .not_applicable };
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

const n_buffers = 4;

pub const Camera = struct {
    gpa: std.mem.Allocator,
    fd: linux.fd_t,
    sink: camera.Sink,
    granted: camera.Granted = .{},
    pixfmt: u32 = 0,
    bytesperline: u32 = 0,
    maps: [n_buffers][]align(std.heap.page_size_min) u8 = @splat(&.{}),
    n_mapped: usize = 0,
    bgra: Image(u8),
    thread: ?std.Thread = null,
    stop_flag: std.atomic.Value(bool) = .init(false),
    frame_count: std.atomic.Value(usize) = .init(0),
    bad_frames: std.atomic.Value(usize) = .init(0),

    pub fn open(gpa: std.mem.Allocator, _: std.Io, opts: camera.Options, sink: camera.Sink) Error!*Camera {
        // The opts.index-th capture node.
        var fd: linux.fd_t = -1;
        var found: usize = 0;
        var cap: Capability = undefined;
        var any_denied = false;
        var pbuf: [32]u8 = undefined;
        for (0..64) |n| {
            const f = openNode(n, &pbuf) orelse {
                const path = std.fmt.bufPrintSentinel(&pbuf, "/dev/video{d}", .{n}, 0) catch continue;
                const rc = linux.open(path, .{ .ACCMODE = .RDONLY, .CLOEXEC = true }, 0);
                if (linux.errno(rc) == .ACCES) any_denied = true else if (linux.errno(rc) == .SUCCESS) _ = linux.close(@intCast(rc));
                continue;
            };
            if (isCapture(f, &cap)) {
                if (found == (opts.index orelse 0)) {
                    fd = f;
                    break;
                }
                found += 1;
            }
            _ = linux.close(f);
        }
        if (fd < 0) {
            if (any_denied) {
                camera.setDetail("no access to the /dev/video devices; add yourself to the 'video' group", .{});
                return error.PermissionDenied;
            }
            return if (opts.index != null and found > 0) error.NoSuchCamera else error.NoCamera;
        }
        errdefer _ = linux.close(fd);

        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const fmts = formats(arena.allocator(), fd) catch return error.OutOfMemory;
        const fi = camera.chooseFormat(fmts, opts.width, opts.height, opts.fps) orelse {
            camera.setDetail("the camera offers neither YUYV nor MJPEG", .{});
            return error.ConfigureFailed;
        };
        const want = fmts[fi];

        var fmt: Format = .{ .type = BUF_TYPE_VIDEO_CAPTURE, .pix = std.mem.zeroes(PixFormat) };
        fmt.pix.width = want.width;
        fmt.pix.height = want.height;
        fmt.pix.pixelformat = @bitCast(want.pixel_format);
        fmt.pix.field = FIELD_NONE;
        switch (ioctl(fd, VIDIOC_S_FMT, &fmt)) {
            .SUCCESS => {},
            .BUSY => {
                camera.setDetail("the device is busy (another program is using it)", .{});
                return error.CameraUnavailable;
            },
            else => |e| {
                camera.setDetail("VIDIOC_S_FMT failed: {s}", .{@tagName(e)});
                return error.ConfigureFailed;
            },
        }
        if (fmt.pix.pixelformat != PIX_YUYV and fmt.pix.pixelformat != PIX_MJPG) {
            camera.setDetail("the driver switched to an unsupported pixel format", .{});
            return error.ConfigureFailed;
        }

        // Frame rate: the wanted one, or the format's fastest.
        var parm: StreamParm = .{ .type = BUF_TYPE_VIDEO_CAPTURE };
        const rate = opts.fps orelse want.max_fps;
        parm.numerator = 1000;
        parm.denominator = @intFromFloat(@round(rate * 1000));
        _ = ioctl(fd, VIDIOC_S_PARM, &parm);
        var got: StreamParm = .{ .type = BUF_TYPE_VIDEO_CAPTURE };
        const fps = if (ioctl(fd, VIDIOC_G_PARM, &got) == .SUCCESS and got.numerator > 0) rateOf(got.numerator, got.denominator) else rate;

        const self = gpa.create(Camera) catch return error.OutOfMemory;
        errdefer gpa.destroy(self);
        self.* = .{
            .gpa = gpa,
            .fd = fd,
            .sink = sink,
            .pixfmt = fmt.pix.pixelformat,
            .bytesperline = fmt.pix.bytesperline,
            .bgra = Image(u8).init(gpa, fmt.pix.width, fmt.pix.height, 4) catch return error.OutOfMemory,
        };
        errdefer self.bgra.deinit(gpa);
        self.granted = .{ .width = fmt.pix.width, .height = fmt.pix.height, .fps = fps, .format_max_fps = want.max_fps, .pixel_format = @bitCast(fmt.pix.pixelformat) };
        self.granted.setName(std.mem.sliceTo(&cap.card, 0));
        if (self.bytesperline == 0) self.bytesperline = fmt.pix.width * 2;

        // Buffers.
        var req: RequestBuffers = .{ .count = n_buffers, .type = BUF_TYPE_VIDEO_CAPTURE, .memory = MEMORY_MMAP };
        if (ioctl(fd, VIDIOC_REQBUFS, &req) != .SUCCESS or req.count == 0) {
            camera.setDetail("the device refused memory-mapped buffers", .{});
            return error.CameraUnavailable;
        }
        errdefer self.unmap();
        for (0..@min(req.count, n_buffers)) |i| {
            var b: Buffer = .{ .index = @intCast(i), .type = BUF_TYPE_VIDEO_CAPTURE, .memory = MEMORY_MMAP };
            if (ioctl(fd, VIDIOC_QUERYBUF, &b) != .SUCCESS) return error.CameraUnavailable;
            const rc = linux.mmap(null, b.length, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, b.m.offset);
            if (linux.errno(rc) != .SUCCESS) return error.CameraUnavailable;
            self.maps[i] = @as([*]align(std.heap.page_size_min) u8, @ptrFromInt(rc))[0..b.length];
            self.n_mapped = i + 1;
            if (ioctl(fd, VIDIOC_QBUF, &b) != .SUCCESS) return error.CameraUnavailable;
        }
        return self;
    }

    fn unmap(self: *Camera) void {
        for (self.maps[0..self.n_mapped]) |m| _ = linux.munmap(m.ptr, m.len);
        self.n_mapped = 0;
    }

    pub fn start(self: *Camera) Error!void {
        if (self.thread != null) return;
        var t: c_int = BUF_TYPE_VIDEO_CAPTURE;
        if (ioctl(self.fd, VIDIOC_STREAMON, &t) != .SUCCESS) {
            camera.setDetail("VIDIOC_STREAMON failed", .{});
            return error.CameraUnavailable;
        }
        self.stop_flag.store(false, .release);
        self.thread = std.Thread.spawn(.{}, captureMain, .{self}) catch return error.OutOfMemory;
    }

    pub fn stop(self: *Camera) void {
        const t = self.thread orelse return;
        self.stop_flag.store(true, .release);
        t.join();
        self.thread = null;
        var ty: c_int = BUF_TYPE_VIDEO_CAPTURE;
        _ = ioctl(self.fd, VIDIOC_STREAMOFF, &ty);
    }

    pub fn close(self: *Camera) void {
        self.stop();
        self.unmap();
        _ = linux.close(self.fd);
        self.bgra.deinit(self.gpa);
        self.gpa.destroy(self);
    }

    pub fn frames(self: *const Camera) usize {
        return self.frame_count.load(.monotonic);
    }

    fn captureMain(self: *Camera) void {
        while (!self.stop_flag.load(.acquire)) {
            var pfd = [_]linux.pollfd{.{ .fd = self.fd, .events = linux.POLL.IN, .revents = 0 }};
            const rc = linux.poll(&pfd, 1, 100);
            if (linux.errno(rc) != .SUCCESS or rc == 0) continue;
            var b: Buffer = .{ .type = BUF_TYPE_VIDEO_CAPTURE, .memory = MEMORY_MMAP };
            switch (ioctl(self.fd, VIDIOC_DQBUF, &b)) {
                .SUCCESS => {},
                .AGAIN => continue,
                else => {
                    // Unplugged or failed: stop quietly (the caller notices
                    // that frames stopped).
                    return;
                },
            }
            if (b.index < self.n_mapped) {
                const data = self.maps[b.index][0..@min(b.bytesused, self.maps[b.index].len)];
                if (self.convert(data)) {
                    _ = self.frame_count.fetchAdd(1, .monotonic);
                    const pts = @as(f64, @floatFromInt(b.timestamp.sec)) + @as(f64, @floatFromInt(b.timestamp.usec)) / 1e6;
                    self.sink.func(self.sink.ctx, self.bgra, pts);
                } else _ = self.bad_frames.fetchAdd(1, .monotonic);
            }
            _ = ioctl(self.fd, VIDIOC_QBUF, &b);
        }
    }

    /// The buffer's frame as BGRA in `self.bgra`.
    fn convert(self: *Camera, data: []const u8) bool {
        if (self.pixfmt == PIX_YUYV) {
            const need = @as(usize, self.bytesperline) * (self.bgra.height - 1) + @as(usize, self.bgra.width) * 2;
            if (data.len < need) return false;
            yuyvToBgra(data, self.bytesperline, self.bgra);
            return true;
        }
        var img = jpeg.decode(self.gpa, data, .{ .mode = .color, .order = .bgr }) catch return false;
        defer img.deinit(self.gpa);
        const src = img.as(u8) orelse return false;
        if (src.channels != 3) return false;
        const w = @min(src.width, self.bgra.width);
        const h = @min(src.height, self.bgra.height);
        for (0..h) |y| {
            const s = src.row(y);
            const d = self.bgra.row(y);
            for (0..w) |x| {
                d[x * 4 ..][0..4].* = .{ s[x * 3], s[x * 3 + 1], s[x * 3 + 2], 255 };
            }
        }
        return true;
    }
};

/// YUYV 4:2:2 (BT.601, limited range) to BGRA.
pub fn yuyvToBgra(src: []const u8, stride: usize, dst: Image(u8)) void {
    for (0..dst.height) |y| {
        const s = src[y * stride ..][0 .. @as(usize, dst.width) * 2];
        const d = dst.row(y);
        var x: usize = 0;
        while (x + 1 < dst.width) : (x += 2) {
            const u: i32 = @as(i32, s[x * 2 + 1]) - 128;
            const v: i32 = @as(i32, s[x * 2 + 3]) - 128;
            inline for (0..2) |k| {
                const c: i32 = (@as(i32, s[x * 2 + 2 * k]) - 16) * 298;
                d[(x + k) * 4 ..][0..4].* = .{
                    clampU8((c + 516 * u + 128) >> 8),
                    clampU8((c - 100 * u - 208 * v + 128) >> 8),
                    clampU8((c + 409 * v + 128) >> 8),
                    255,
                };
            }
        }
        if (x < dst.width) {
            const c: i32 = (@as(i32, s[x * 2]) - 16) * 298;
            d[x * 4 ..][0..4].* = .{ clampU8((c + 128) >> 8), clampU8((c + 128) >> 8), clampU8((c + 128) >> 8), 255 };
        }
    }
}

fn clampU8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

test "yuyv to bgra" {
    // Two pixels: Y=235 (white) and Y=16 (black), no chroma.
    const src = [_]u8{ 235, 128, 16, 128 };
    var buf: [8]u8 = undefined;
    const dst = try Image(u8).fromSlice(&buf, 2, 1, 4);
    yuyvToBgra(&src, 4, dst);
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255, 255, 0, 0, 0, 255 }, &buf);
}
