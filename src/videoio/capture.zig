//! Video files, as OpenCV's VideoCapture and VideoWriter have them: open by
//! path, read frames one after another into a reusable BGR, BGRA or gray
//! image, ask for the size, frame rate and frame count, seek by frame.
//!
//!   var cap = try VideoCapture.open(gpa, io, "clip.avi");
//!   defer cap.close();
//!   var frame = try Image(u8).init(gpa, cap.width, cap.height, 3);
//!   while (try cap.read(frame)) { ... }
//!
//! The format comes from the file's first bytes, then its extension:
//!
//!   AVI (RIFF)    Motion-JPEG or uncompressed (avi.zig), every platform
//!   YUV4MPEG2     .y4m (y4m.zig), every platform
//!   MP4 / QuickTime  .mp4 .mov .m4v: macOS only, through AVFoundation
//!                 (video_macos.zig, hardware decoding of H.264 and HEVC);
//!                 elsewhere error.Unsupported, and `detail()` says how to
//!                 convert the file with ffmpeg
//!
//! VideoWriter writes .avi (Motion-JPEG by default, OpenCV's 'MJPG'; or
//! uncompressed with fourcc 0 / 'DIB ') and .y4m.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const avi = @import("avi.zig");
const y4m = @import("y4m.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const is_macos = builtin.os.tag == .macos;
const macos = if (is_macos) @import("video_macos.zig") else struct {
    pub const Reader = void;
};

/// Whether .mp4 / .mov files can be decoded on this platform.
pub const native_supported = is_macos;

pub const Backend = enum {
    avi,
    y4m,
    /// AVFoundation (macOS).
    avfoundation,

    pub fn name(b: Backend) []const u8 {
        return switch (b) {
            .avi => "AVI",
            .y4m => "YUV4MPEG2",
            .avfoundation => "AVFoundation",
        };
    }
};

pub const Error = error{
    FileNotFound,
    AccessDenied,
    /// Opening, reading or writing the file failed in the OS.
    OpenFailed,
    ReadFailed,
    WriteFailed,
    /// Not a video format VISAR knows (from the bytes or the extension).
    UnknownFormat,
    /// A format or codec VISAR cannot handle here; see `detail()`.
    Unsupported,
    InvalidData,
    Truncated,
    TooLarge,
    SizeMismatch,
    UnsupportedChannels,
    /// Seeking needs an index this file does not have.
    NotSeekable,
    /// The system frameworks could not be loaded (macOS).
    RuntimeUnavailable,
    OutOfMemory,
};

var detail_buf: [320]u8 = undefined;
var detail_len: usize = 0;

/// More about the last error, when there is more to say ("" otherwise).
pub fn detail() []const u8 {
    return detail_buf[0..detail_len];
}

fn setDetail(comptime fmt: []const u8, a: anytype) void {
    const s: []const u8 = std.fmt.bufPrint(&detail_buf, fmt, a) catch &detail_buf;
    detail_len = s.len;
}

fn mapOpen(e: anyerror) Error {
    return switch (e) {
        error.FileNotFound => error.FileNotFound,
        error.AccessDenied, error.PermissionDenied => error.AccessDenied,
        error.OutOfMemory => error.OutOfMemory,
        else => error.OpenFailed,
    };
}

/// The video kind a path's extension names, if any.
pub const Kind = enum { avi, y4m, mp4 };

pub fn kindFromExtension(path: []const u8) ?Kind {
    const dot = std.mem.lastIndexOfScalar(u8, path, '.') orelse return null;
    const ext = path[dot + 1 ..];
    const table = .{
        .{ "avi", Kind.avi }, .{ "y4m", Kind.y4m }, .{ "mp4", Kind.mp4 }, .{ "mov", Kind.mp4 },
        .{ "m4v", Kind.mp4 }, .{ "qt", Kind.mp4 },
    };
    inline for (table) |t| if (std.ascii.eqlIgnoreCase(ext, t[0])) return t[1];
    return null;
}

/// Whether a path names a video file VISAR may read (by extension).
pub fn isVideoPath(path: []const u8) bool {
    return kindFromExtension(path) != null;
}

/// The kind from a file's first bytes.
pub fn detectKind(head: []const u8) ?Kind {
    if (avi.detect(head)) return .avi;
    if (y4m.detect(head)) return .y4m;
    if (head.len >= 8 and (std.mem.eql(u8, head[4..8], "ftyp") or std.mem.eql(u8, head[4..8], "moov") or std.mem.eql(u8, head[4..8], "mdat") or std.mem.eql(u8, head[4..8], "wide"))) return .mp4;
    return null;
}

const Y4mIn = struct {
    file_reader: Io.File.Reader = undefined,
    mem: Io.Reader = undefined,
    in_memory: bool,
    buf: [64 * 1024]u8 = undefined,
    r: y4m.Reader = undefined,
    header_len: u64 = 0,
    /// Bytes per frame with a bare "FRAME\n" line.
    stride: u64 = 0,

    fn reader(self: *Y4mIn) *Io.Reader {
        return if (self.in_memory) &self.mem else &self.file_reader.interface;
    }

    fn seekTo(self: *Y4mIn, off: u64) Error!void {
        if (self.in_memory) {
            if (off > self.mem.end) return error.InvalidData;
            self.mem.seek = @intCast(off);
        } else self.file_reader.seekTo(off) catch return error.ReadFailed;
    }
};

pub const VideoCapture = struct {
    gpa: Allocator,
    io: ?Io,
    backend: Backend,
    file: ?Io.File = null,
    width: u32,
    height: u32,
    /// Frames per second (0 when the file does not say).
    fps: f64,
    /// Frames in the file (null when unknown).
    frame_count: ?u64,
    /// The codec: 'MJPG', 'I420', 'avc1', ...
    fourcc: [4]u8,
    /// The frame `read` returns next.
    pos: u64 = 0,
    impl: union(enum) {
        avi: avi.Reader,
        y4m: *Y4mIn,
        avf: if (is_macos) *macos.Reader else void,
    },

    /// Opens a video file.
    pub fn open(gpa: Allocator, io: Io, path: []const u8) Error!VideoCapture {
        detail_len = 0;
        const file = Io.Dir.cwd().openFile(io, path, .{}) catch |e| return mapOpen(e);
        var keep = false;
        defer if (!keep) file.close(io);
        var head: [16]u8 = @splat(0);
        const n = file.readPositionalAll(io, &head, 0) catch return error.ReadFailed;
        const kind = detectKind(head[0..n]) orelse kindFromExtension(path) orelse {
            setDetail("VISAR reads .avi (Motion-JPEG or uncompressed), .y4m{s}", .{if (native_supported) ", .mp4 and .mov" else ""});
            return error.UnknownFormat;
        };
        var cap: VideoCapture = switch (kind) {
            .avi => try openAvi(gpa, .{ .file = .{ .io = io, .file = file } }),
            .y4m => try openY4m(gpa, file, io, null),
            .mp4 => try openNative(gpa, io, path),
        };
        cap.io = io;
        if (kind != .mp4) {
            cap.file = file;
            keep = true;
        }
        return cap;
    }

    /// .mp4 / .mov through the platform's decoder (AVFoundation opens the
    /// file itself).
    fn openNative(gpa: Allocator, io: Io, path: []const u8) Error!VideoCapture {
        if (comptime !is_macos) {
            setDetail("decoding .mp4/.mov needs macOS (AVFoundation); convert it: ffmpeg -i {s} -c:v mjpeg -q:v 3 out.avi (or -pix_fmt yuv420p out.y4m)", .{std.fs.path.basename(path)});
            return error.Unsupported;
        } else {
            const r = macos.Reader.open(gpa, path) catch |e| {
                setDetail("{s}", .{macos.detail()});
                return e;
            };
            return .{
                .gpa = gpa,
                .io = io,
                .backend = .avfoundation,
                .width = r.width,
                .height = r.height,
                .fps = r.fps,
                .frame_count = if (r.frame_count > 0) r.frame_count else null,
                .fourcc = r.fourcc,
                .impl = .{ .avf = r },
            };
        }
    }

    /// Opens an AVI or YUV4MPEG2 video held in memory (`bytes` must outlive
    /// the capture).
    pub fn openMemory(gpa: Allocator, bytes: []const u8) Error!VideoCapture {
        detail_len = 0;
        return switch (detectKind(bytes) orelse return error.UnknownFormat) {
            .avi => openAvi(gpa, .{ .memory = bytes }),
            .y4m => openY4m(gpa, null, null, bytes),
            .mp4 => error.Unsupported,
        };
    }

    fn openAvi(gpa: Allocator, src: avi.Source) Error!VideoCapture {
        const r = avi.Reader.init(gpa, src) catch |e| {
            if (e == error.Unsupported) setDetail("VISAR reads AVI with Motion-JPEG, uncompressed RGB, I420/YV12 or Y800 video; convert it: ffmpeg -i in.avi -c:v mjpeg -q:v 3 out.avi", .{});
            return e;
        };
        return .{
            .gpa = gpa,
            .io = null,
            .backend = .avi,
            .width = r.width,
            .height = r.height,
            .fps = r.fps(),
            .frame_count = r.frameCount(),
            .fourcc = r.fourcc,
            .impl = .{ .avi = r },
        };
    }

    fn openY4m(gpa: Allocator, file: ?Io.File, io: ?Io, bytes: ?[]const u8) Error!VideoCapture {
        const s = try gpa.create(Y4mIn);
        errdefer gpa.destroy(s);
        s.* = .{ .in_memory = bytes != null };
        var size: u64 = 0;
        if (bytes) |b| {
            s.mem = .fixed(b);
            size = b.len;
        } else {
            s.file_reader = file.?.reader(io.?, &s.buf);
            size = file.?.length(io.?) catch return error.ReadFailed;
        }
        s.r = y4m.Reader.init(gpa, s.reader()) catch |e| return switch (e) {
            error.Unsupported => blk: {
                setDetail("VISAR reads 8-bit YUV4MPEG2; convert it: ffmpeg -i in.y4m -pix_fmt yuv420p out.y4m", .{});
                break :blk error.Unsupported;
            },
            error.ReadFailed => error.ReadFailed,
            error.OutOfMemory => error.OutOfMemory,
            else => error.InvalidData,
        };
        errdefer s.r.deinit();
        s.header_len = if (bytes != null) s.mem.seek else s.file_reader.logicalPos();
        s.stride = 6 + s.r.frameBytes();
        // Bare "FRAME\n" lines make every frame the same size: count and
        // seek by arithmetic (checked when each frame is read).
        const body = size - @min(size, s.header_len);
        const count: ?u64 = if (body % s.stride == 0) body / s.stride else null;
        if (count == null) s.stride = 0;
        const fps: f64 = if (s.r.fps_den > 0) @as(f64, @floatFromInt(s.r.fps_num)) / @as(f64, @floatFromInt(s.r.fps_den)) else 0;
        return .{
            .gpa = gpa,
            .io = io,
            .backend = .y4m,
            .width = s.r.width,
            .height = s.r.height,
            .fps = fps,
            .frame_count = count,
            .fourcc = switch (s.r.chroma) {
                .c420 => "I420".*,
                .c422 => "Y42B".*,
                .c444 => "444P".*,
                .mono => "Y800".*,
            },
            .impl = .{ .y4m = s },
        };
    }

    pub fn close(self: *VideoCapture) void {
        switch (self.impl) {
            .avi => |*r| r.deinit(),
            .y4m => |s| {
                s.r.deinit();
                self.gpa.destroy(s);
            },
            .avf => |r| if (comptime is_macos) r.close(),
        }
        if (self.file) |f| f.close(self.io.?);
        self.* = undefined;
    }

    /// Reads the next frame into `dst`: the video's size, 8-bit, 3 (BGR),
    /// 4 (BGRA) or 1 (gray) channels. False at the end of the video.
    pub fn read(self: *VideoCapture, dst: Image(u8)) Error!bool {
        if (dst.width != self.width or dst.height != self.height) return error.SizeMismatch;
        if (dst.channels != 1 and dst.channels != 3 and dst.channels != 4) return error.UnsupportedChannels;
        const got = switch (self.impl) {
            .avi => |*r| try r.next(dst),
            .y4m => |s| blk: {
                if (dst.channels == 1) {
                    // y4m converts to colour; take the luma from a BGR pass.
                    var tmp = try scratch(self.gpa, dst.width, dst.height);
                    defer tmp.deinit(self.gpa);
                    const ok = try y4mNext(s, tmp);
                    if (ok) toGray(tmp, dst);
                    break :blk ok;
                }
                break :blk try y4mNext(s, dst);
            },
            .avf => |r| if (comptime is_macos) (r.next(dst) catch |e| {
                setDetail("{s}", .{macos.detail()});
                return e;
            }) else unreachable,
        };
        if (got) self.pos += 1;
        return got;
    }

    fn y4mNext(s: *Y4mIn, dst: Image(u8)) Error!bool {
        return s.r.next(dst) catch |e| switch (e) {
            error.Truncated => error.Truncated,
            error.ReadFailed => error.ReadFailed,
            error.OutOfMemory => error.OutOfMemory,
            error.SizeMismatch => error.SizeMismatch,
            error.UnsupportedChannels => error.UnsupportedChannels,
            else => error.InvalidData,
        };
    }

    /// The next `read` returns frame `n` (0-based). Past the end, `read`
    /// returns false.
    pub fn seek(self: *VideoCapture, n: u64) Error!void {
        if (n == self.pos) return;
        switch (self.impl) {
            .avi => |*r| try r.seek(@min(n, r.frameCount())),
            .y4m => |s| {
                if (s.stride == 0) return error.NotSeekable;
                const count = self.frame_count.?;
                try s.seekTo(s.header_len + @min(n, count) * s.stride);
                s.r.frames = @min(n, count);
            },
            .avf => |r| if (comptime is_macos) (r.seek(n) catch |e| {
                setDetail("{s}", .{macos.detail()});
                return e;
            }) else unreachable,
        }
        self.pos = n;
    }

    /// Seconds from the start to frame `pos` (from the frame rate).
    pub fn timeOf(self: *const VideoCapture, frame: u64) f64 {
        return if (self.fps > 0) @as(f64, @floatFromInt(frame)) / self.fps else 0;
    }
};

fn scratch(gpa: Allocator, w: u32, h: u32) Error!Image(u8) {
    return Image(u8).init(gpa, w, h, 3) catch |e| if (e == error.OutOfMemory) error.OutOfMemory else error.InvalidData;
}

fn toGray(src: Image(u8), dst: Image(u8)) void {
    for (0..src.height) |y| {
        const s = src.row(y);
        const o = dst.row(y);
        for (0..src.width) |x| {
            const p = s[x * 3 ..][0..3];
            o[x] = @intCast((@as(u32, p[0]) * 1868 + @as(u32, p[1]) * 9617 + @as(u32, p[2]) * 4899 + 8192) >> 14);
        }
    }
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub const WriterOptions = struct {
    /// The codec as a FourCC: 'MJPG' (default for .avi), 0 or 'DIB '
    /// (uncompressed .avi), 'I420' (the only .y4m codec). null: the
    /// extension's default.
    fourcc: ?[4]u8 = null,
    /// JPEG quality for Motion-JPEG, 1-100.
    quality: u8 = 95,
};

const Y4mOut = struct {
    fw: Io.File.Writer,
    buf: [256 * 1024]u8,
    w: y4m.Writer,
};

pub const VideoWriter = struct {
    gpa: Allocator,
    io: Io,
    file: Io.File,
    width: u32,
    height: u32,
    impl: union(enum) { avi: avi.Writer, y4m: *Y4mOut },

    /// Creates (or replaces) a video file; the format follows the
    /// extension (.avi or .y4m).
    pub fn open(gpa: Allocator, io: Io, path: []const u8, width: u32, height: u32, fps: f64, opts: WriterOptions) Error!VideoWriter {
        detail_len = 0;
        const kind = kindFromExtension(path) orelse {
            setDetail("VISAR writes .avi (Motion-JPEG) and .y4m", .{});
            return error.UnknownFormat;
        };
        const fr = avi.fpsFraction(fps);
        switch (kind) {
            .mp4 => {
                setDetail("VISAR writes .avi (Motion-JPEG) and .y4m, not .mp4; convert afterwards: ffmpeg -i out.avi -c:v libx264 out.mp4", .{});
                return error.Unsupported;
            },
            .y4m => if (opts.fourcc) |f| if (!std.mem.eql(u8, &f, "I420") and !std.mem.eql(u8, &f, "IYUV")) {
                setDetail(".y4m is written as I420 (4:2:0)", .{});
                return error.Unsupported;
            },
            .avi => {},
        }
        var codec: avi.WriteCodec = .mjpeg;
        if (kind == .avi) if (opts.fourcc) |f| {
            var up: [4]u8 = undefined;
            for (&up, f) |*o, c| o.* = std.ascii.toUpper(c);
            if (std.mem.readInt(u32, &f, .little) == 0 or std.mem.eql(u8, &up, "DIB ") or std.mem.eql(u8, &up, "RGB ")) {
                codec = .bgr24;
            } else if (!std.mem.eql(u8, &up, "MJPG")) {
                setDetail("VISAR writes AVI as 'MJPG' (Motion-JPEG) or uncompressed (fourcc 0), not '{s}'", .{std.mem.sliceTo(&f, 0)});
                return error.Unsupported;
            }
        };
        if (width == 0 or height == 0 or width > 16384 or height > 16384) return error.InvalidData;
        const file = Io.Dir.cwd().createFile(io, path, .{}) catch |e| return mapOpen(e);
        errdefer file.close(io);
        var self: VideoWriter = .{ .gpa = gpa, .io = io, .file = file, .width = width, .height = height, .impl = undefined };
        switch (kind) {
            .avi => self.impl = .{ .avi = try avi.Writer.init(gpa, .{ .file = .{ .io = io, .file = file } }, width, height, fr[0], fr[1], .{ .codec = codec, .quality = opts.quality }) },
            .y4m => {
                const o = try gpa.create(Y4mOut);
                errdefer gpa.destroy(o);
                o.fw = file.writer(io, &o.buf);
                o.w = y4m.Writer.init(gpa, &o.fw.interface, width, height, fr[0], fr[1]) catch |e| return switch (e) {
                    error.WriteFailed => error.WriteFailed,
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.InvalidData,
                };
                self.impl = .{ .y4m = o };
            },
            .mp4 => unreachable,
        }
        return self;
    }

    /// Appends a frame (BGR, BGRA or gray; the video's size).
    pub fn write(self: *VideoWriter, frame: Image(u8)) Error!void {
        switch (self.impl) {
            .avi => |*w| try w.write(frame),
            .y4m => |o| {
                if (frame.channels == 1) {
                    var tmp = try scratch(self.gpa, frame.width, frame.height);
                    defer tmp.deinit(self.gpa);
                    for (0..frame.height) |y| for (frame.row(y), 0..) |v, x| {
                        tmp.row(y)[x * 3 ..][0..3].* = .{ v, v, v };
                    };
                    return self.write(tmp);
                }
                o.w.write(frame) catch |e| return switch (e) {
                    error.WriteFailed => error.WriteFailed,
                    error.OutOfMemory => error.OutOfMemory,
                    error.SizeMismatch => error.SizeMismatch,
                    error.UnsupportedChannels => error.UnsupportedChannels,
                    else => error.InvalidData,
                };
            },
        }
    }

    pub fn frameCount(self: *const VideoWriter) u64 {
        return switch (self.impl) {
            .avi => |*w| w.frameCount(),
            .y4m => |o| o.w.frames,
        };
    }

    /// Finishes the file (indexes, headers) and frees the writer, also
    /// when finishing fails.
    pub fn close(self: *VideoWriter) Error!void {
        defer self.* = undefined;
        var result: Error!void = {};
        switch (self.impl) {
            .avi => |*w| {
                result = w.close();
                w.deinit();
            },
            .y4m => |o| {
                o.fw.interface.flush() catch {
                    result = error.WriteFailed;
                };
                o.w.deinit();
                self.gpa.destroy(o);
            },
        }
        self.file.close(self.io);
        return result;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn pattern(gpa: Allocator, w: u32, h: u32, t: usize) !Image(u8) {
    var img = try Image(u8).init(gpa, w, h, 3);
    for (0..h) |y| for (0..w) |x| {
        img.row(y)[x * 3 ..][0..3].* = .{ @intCast((x * 4 + t * 9) & 255), @intCast((y * 5) & 255), @intCast(((x + y) * 2 + t * 30) & 255) };
    };
    return img;
}

test "capture: write and read .avi and .y4m files, seek, properties" {
    const gpa = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    for ([_][]const u8{ "a.avi", "b.y4m", "c.avi" }, 0..) |name, k| {
        const path = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/{s}", .{ tmp.sub_path, name });
        defer gpa.free(path);
        {
            var w = try VideoWriter.open(gpa, io, path, 48, 32, 29.97, .{ .fourcc = if (k == 2) .{ 0, 0, 0, 0 } else null });
            for (0..9) |t| {
                var f = try pattern(gpa, 48, 32, t);
                defer f.deinit(gpa);
                try w.write(f);
            }
            try testing.expectEqual(@as(u64, 9), w.frameCount());
            try w.close();
        }
        var cap = try VideoCapture.open(gpa, io, path);
        defer cap.close();
        try testing.expectEqual(if (k == 1) Backend.y4m else Backend.avi, cap.backend);
        try testing.expectEqual(@as(u32, 48), cap.width);
        try testing.expectEqual(@as(?u64, 9), cap.frame_count);
        try testing.expectApproxEqAbs(@as(f64, 29.97), cap.fps, 0.001);
        var img = try Image(u8).init(gpa, 48, 32, 4);
        defer img.deinit(gpa);
        var n: usize = 0;
        while (try cap.read(img)) n += 1;
        try testing.expectEqual(@as(usize, 9), n);
        try cap.seek(6);
        try testing.expect(try cap.read(img));
        try testing.expectEqual(@as(u64, 7), cap.pos);
        var want = try pattern(gpa, 48, 32, 6);
        defer want.deinit(gpa);
        var err: u64 = 0;
        for (0..32) |y| for (0..48) |x| for (0..3) |c| {
            err += @abs(@as(i32, img.row(y)[x * 4 + c]) - @as(i32, want.row(y)[x * 3 + c]));
        };
        // Uncompressed is exact; JPEG and 4:2:0 are close.
        if (k == 2) try testing.expectEqual(@as(u64, 0), err) else try testing.expect(err / (48 * 32 * 3) < 12);
        var gray = try Image(u8).init(gpa, 48, 32, 1);
        defer gray.deinit(gpa);
        try cap.seek(0);
        try testing.expect(try cap.read(gray));
        try cap.seek(100);
        try testing.expect(!try cap.read(img));
    }
    // Unknown and unsupported.
    try testing.expectError(error.FileNotFound, VideoCapture.open(gpa, io, "/nonexistent/x.avi"));
    const bad = try std.fmt.allocPrint(gpa, ".zig-cache/tmp/{s}/x.mp4", .{tmp.sub_path});
    defer gpa.free(bad);
    try testing.expectError(error.Unsupported, VideoWriter.open(gpa, io, bad, 8, 8, 30, .{}));
    if (!native_supported) {
        try tmp.dir.writeFile(io, .{ .sub_path = "x.mp4", .data = "\x00\x00\x00\x18ftypmp42" });
        try testing.expectError(error.Unsupported, VideoCapture.open(gpa, io, bad));
        try testing.expect(std.mem.indexOf(u8, detail(), "ffmpeg") != null);
    }
}

test "capture: in memory" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    var w = try avi.Writer.init(gpa, .{ .memory = &out }, 16, 16, 10, 1, .{});
    var f = try pattern(gpa, 16, 16, 0);
    defer f.deinit(gpa);
    try w.write(f);
    try w.close();
    w.deinit();
    var cap = try VideoCapture.openMemory(gpa, out.items);
    defer cap.close();
    try testing.expectEqual(@as(?u64, 1), cap.frame_count);
    try testing.expectError(error.UnknownFormat, VideoCapture.openMemory(gpa, "hello world, not a video"));
}
