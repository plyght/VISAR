//! Video file decoding on macOS with AVFoundation (AVAssetReader), reached
//! through the Objective-C runtime at run time like the camera
//! (objc.zig): no SDK and no link-time dependency. H.264, HEVC, ProRes and
//! whatever else the system decodes, in .mp4, .mov and .m4v files, on
//! VideoToolbox's hardware decoder where the machine has one.
//!
//! Frames come out as the decoder's own 4:2:0 ('420v', bi-planar, video
//! range) and are converted to the caller's BGR, BGRA or gray image by
//! VISAR, with the matrix the buffer is tagged with (BT.601 unless it says
//! BT.709) and the nearest chroma sample, as the AVI and YUV4MPEG2 readers
//! do: the same picture whichever backend reads it. Frame n is the frame
//! presented at first_pts + n / fps; seeking starts a new AVAssetReader a
//! little before that time (the decoder starts from the keyframe before it
//! and the frames before the target are skipped). The track's
//! preferredTransform (rotation metadata) is not applied.
//!
//! Only compiled for macOS (see capture.zig).

const std = @import("std");
const objc = @import("objc.zig");
const core = @import("../core/core.zig");
const Image = core.Image;
const id = objc.id;
const send = objc.send;

pub const Error = error{ RuntimeUnavailable, InvalidData, Unsupported, ReadFailed, SizeMismatch, UnsupportedChannels, OutOfMemory };

/// kCVPixelFormatType_32BGRA ('BGRA').
const pixel_format_bgra: u32 = 0x42475241;
/// kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange ('420v') and its full
/// range sibling ('420f').
const pixel_format_420v: u32 = 0x34323076;
const pixel_format_420f: u32 = 0x34323066;
const lock_read_only: u64 = 1;

const CMTimeRange = extern struct { start: objc.CMTime, duration: objc.CMTime };

const Api = struct {
    CMSampleBufferGetImageBuffer: *const fn (?*anyopaque) callconv(.c) ?*anyopaque,
    CMSampleBufferGetPresentationTimeStamp: *const fn (?*anyopaque) callconv(.c) objc.CMTime,
    CVPixelBufferLockBaseAddress: *const fn (?*anyopaque, u64) callconv(.c) i32,
    CVPixelBufferUnlockBaseAddress: *const fn (?*anyopaque, u64) callconv(.c) i32,
    CVPixelBufferGetBaseAddress: *const fn (?*anyopaque) callconv(.c) ?[*]u8,
    CVPixelBufferGetBytesPerRow: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetWidth: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetHeight: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetPixelFormatType: *const fn (?*anyopaque) callconv(.c) u32,
    CVPixelBufferGetBaseAddressOfPlane: *const fn (?*anyopaque, usize) callconv(.c) ?[*]u8,
    CVPixelBufferGetBytesPerRowOfPlane: *const fn (?*anyopaque, usize) callconv(.c) usize,
    CVPixelBufferGetHeightOfPlane: *const fn (?*anyopaque, usize) callconv(.c) usize,
    CVBufferGetAttachment: *const fn (?*anyopaque, id, ?*u32) callconv(.c) id,
    CFRelease: *const fn (?*anyopaque) callconv(.c) void,
    CFEqual: *const fn (id, id) callconv(.c) u8,
    kCVImageBufferYCbCrMatrixKey: id,
    kCVImageBufferYCbCrMatrix_ITU_R_709_2: id,
    kCVPixelBufferPixelFormatTypeKey: id,
    AVMediaTypeVideo: id,
};

var api_storage: ?Api = null;

/// Why the last open or read failed (for error messages).
pub var detail_buf: [256]u8 = undefined;
pub var detail_len: usize = 0;

pub fn detail() []const u8 {
    return detail_buf[0..detail_len];
}

fn setDetail(comptime fmt: []const u8, a: anytype) void {
    const s: []const u8 = std.fmt.bufPrint(&detail_buf, fmt, a) catch &detail_buf;
    detail_len = s.len;
}

fn load() Error!*const Api {
    if (api_storage) |*a| return a;
    api_storage = loadApi() catch {
        setDetail("cannot load {s}", .{objc.missing()});
        return error.RuntimeUnavailable;
    };
    return &api_storage.?;
}

fn loadApi() objc.Error!Api {
    _ = try objc.load();
    _ = try objc.open(.avfoundation);
    return .{
        .CMSampleBufferGetImageBuffer = try objc.func(fn (?*anyopaque) callconv(.c) ?*anyopaque, .coremedia, "CMSampleBufferGetImageBuffer"),
        .CMSampleBufferGetPresentationTimeStamp = try objc.func(fn (?*anyopaque) callconv(.c) objc.CMTime, .coremedia, "CMSampleBufferGetPresentationTimeStamp"),
        .CVPixelBufferLockBaseAddress = try objc.func(fn (?*anyopaque, u64) callconv(.c) i32, .corevideo, "CVPixelBufferLockBaseAddress"),
        .CVPixelBufferUnlockBaseAddress = try objc.func(fn (?*anyopaque, u64) callconv(.c) i32, .corevideo, "CVPixelBufferUnlockBaseAddress"),
        .CVPixelBufferGetBaseAddress = try objc.func(fn (?*anyopaque) callconv(.c) ?[*]u8, .corevideo, "CVPixelBufferGetBaseAddress"),
        .CVPixelBufferGetBytesPerRow = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetBytesPerRow"),
        .CVPixelBufferGetWidth = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetWidth"),
        .CVPixelBufferGetHeight = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetHeight"),
        .CVPixelBufferGetPixelFormatType = try objc.func(fn (?*anyopaque) callconv(.c) u32, .corevideo, "CVPixelBufferGetPixelFormatType"),
        .CVPixelBufferGetBaseAddressOfPlane = try objc.func(fn (?*anyopaque, usize) callconv(.c) ?[*]u8, .corevideo, "CVPixelBufferGetBaseAddressOfPlane"),
        .CVPixelBufferGetBytesPerRowOfPlane = try objc.func(fn (?*anyopaque, usize) callconv(.c) usize, .corevideo, "CVPixelBufferGetBytesPerRowOfPlane"),
        .CVPixelBufferGetHeightOfPlane = try objc.func(fn (?*anyopaque, usize) callconv(.c) usize, .corevideo, "CVPixelBufferGetHeightOfPlane"),
        .CVBufferGetAttachment = try objc.func(fn (?*anyopaque, id, ?*u32) callconv(.c) id, .corevideo, "CVBufferGetAttachment"),
        .CFRelease = try objc.func(fn (?*anyopaque) callconv(.c) void, .corefoundation, "CFRelease"),
        .CFEqual = try objc.func(fn (id, id) callconv(.c) u8, .corefoundation, "CFEqual"),
        .kCVImageBufferYCbCrMatrixKey = try objc.constant(id, .corevideo, "kCVImageBufferYCbCrMatrixKey"),
        .kCVImageBufferYCbCrMatrix_ITU_R_709_2 = try objc.constant(id, .corevideo, "kCVImageBufferYCbCrMatrix_ITU_R_709_2"),
        .kCVPixelBufferPixelFormatTypeKey = try objc.constant(id, .corevideo, "kCVPixelBufferPixelFormatTypeKey"),
        .AVMediaTypeVideo = try objc.constant(id, .avfoundation, "AVMediaTypeVideo"),
    };
}

fn cls(comptime name: [:0]const u8) Error!objc.Class {
    return objc.class(name) catch {
        setDetail("AVFoundation has no class {s}", .{name});
        return error.RuntimeUnavailable;
    };
}

pub const Reader = struct {
    gpa: std.mem.Allocator,
    api: *const Api,
    asset: id = null,
    track: id = null,
    reader: id = null,
    output: id = null,
    width: u32 = 0,
    height: u32 = 0,
    fps: f64 = 0,
    /// Estimated from the duration and the nominal frame rate (as OpenCV's
    /// CAP_PROP_FRAME_COUNT is).
    frame_count: u64 = 0,
    duration: f64 = 0,
    /// The frame `next` returns.
    pos: u64 = 0,
    /// After a seek: frames earlier than this time are skipped.
    skip_before: f64 = -1,
    /// The first frame's presentation time (frame n is at first_pts + n/fps).
    first_pts: f64 = 0,
    /// A sample read ahead (the first, to learn first_pts), owned (+1).
    pending: ?*anyopaque = null,
    /// The codec's FourCC ('avc1', 'hvc1', ...).
    fourcc: [4]u8 = .{ '?', '?', '?', '?' },

    pub fn open(gpa: std.mem.Allocator, path: []const u8) Error!*Reader {
        const a = try load();
        const pool = objc.Pool.push();
        defer pool.pop();
        const self = try gpa.create(Reader);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .api = a };
        errdefer self.teardown();

        const zpath = try gpa.dupeSentinel(u8, path, 0);
        defer gpa.free(zpath);
        const str = objc.nsString(zpath.ptr);
        if (str == null) return error.InvalidData;
        const url = send(id, try cls("NSURL"), "fileURLWithPath:", .{str});
        self.asset = objc.retain(send(id, try cls("AVURLAsset"), "URLAssetWithURL:options:", .{ url, @as(id, null) }));
        if (self.asset == null) {
            setDetail("AVFoundation cannot open the file", .{});
            return error.InvalidData;
        }
        const tracks = send(id, self.asset, "tracksWithMediaType:", .{a.AVMediaTypeVideo});
        if (objc.count(tracks) == 0) {
            setDetail("AVFoundation finds no video track it can read in it", .{});
            return error.Unsupported;
        }
        self.track = objc.retain(objc.at(tracks, 0));
        const size = send(objc.CGSize, self.track, "naturalSize", .{});
        if (!(size.width >= 1 and size.height >= 1 and size.width <= 16384 and size.height <= 16384)) {
            setDetail("the video track reports a size of {d}x{d}", .{ size.width, size.height });
            return error.InvalidData;
        }
        self.width = @intFromFloat(@round(size.width));
        self.height = @intFromFloat(@round(size.height));
        self.fps = send(f32, self.track, "nominalFrameRate", .{});
        self.duration = send(objc.CMTime, self.asset, "duration", .{}).seconds();
        if (self.fps > 0 and self.duration > 0) self.frame_count = @intFromFloat(@round(self.duration * self.fps));
        const descs = send(id, self.track, "formatDescriptions", .{});
        if (objc.count(descs) > 0) {
            const sub = objc.func(fn (?*anyopaque) callconv(.c) u32, .coremedia, "CMFormatDescriptionGetMediaSubType") catch null;
            if (sub) |f| _ = objc.fourcc(f(objc.at(descs, 0)), &self.fourcc);
        }
        try self.start(0);
        self.pending = send(?*anyopaque, self.output, "copyNextSampleBuffer", .{});
        if (self.pending) |sb| {
            const t = a.CMSampleBufferGetPresentationTimeStamp(sb);
            if (t.flags & objc.CMTime.flag_valid != 0) self.first_pts = t.seconds();
        }
        return self;
    }

    /// A reader from `t` seconds on.
    fn start(self: *Reader, t: f64) Error!void {
        const a = self.api;
        self.stopReader();
        var err: id = null;
        const r = send(id, send(id, try cls("AVAssetReader"), "alloc", .{}), "initWithAsset:error:", .{ self.asset, @as(*id, &err) });
        if (r == null) {
            setDetail("{s}", .{objc.errorText(err)});
            return error.ReadFailed;
        }
        self.reader = r;
        if (t > 0) {
            const range: CMTimeRange = .{
                .start = .{ .value = @intFromFloat(@round(t * 90000)), .timescale = 90000, .flags = objc.CMTime.flag_valid, .epoch = 0 },
                // kCMTimePositiveInfinity: valid | positive infinity.
                .duration = .{ .value = 0, .timescale = 0, .flags = 1 | 4, .epoch = 0 },
            };
            send(void, r, "setTimeRange:", .{range});
        }
        const number = try cls("NSNumber");
        var keys = [_]id{a.kCVPixelBufferPixelFormatTypeKey};
        var values = [_]id{send(id, number, "numberWithUnsignedInt:", .{pixel_format_420v})};
        const settings = send(id, try cls("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]const id, &values), @as([*]const id, &keys), @as(objc.NSUInteger, 1) });
        self.output = send(id, send(id, try cls("AVAssetReaderTrackOutput"), "alloc", .{}), "initWithTrack:outputSettings:", .{ self.track, settings });
        if (self.output == null) {
            setDetail("cannot create the track output", .{});
            return error.ReadFailed;
        }
        send(void, self.output, "setAlwaysCopiesSampleData:", .{objc.NO});
        if (send(objc.BOOL, r, "canAddOutput:", .{self.output}) == objc.NO) {
            setDetail("the asset reader refused the track output", .{});
            return error.ReadFailed;
        }
        send(void, r, "addOutput:", .{self.output});
        if (send(objc.BOOL, r, "startReading", .{}) == objc.NO) {
            setDetail("{s}", .{objc.errorText(send(id, r, "error", .{}))});
            return error.ReadFailed;
        }
    }

    fn stopReader(self: *Reader) void {
        if (self.pending) |sb| self.api.CFRelease(sb);
        self.pending = null;
        if (self.reader != null) {
            // Status 1: reading.
            if (send(objc.NSInteger, self.reader, "status", .{}) == 1) send(void, self.reader, "cancelReading", .{});
        }
        objc.release(self.output);
        objc.release(self.reader);
        self.output = null;
        self.reader = null;
    }

    fn teardown(self: *Reader) void {
        self.stopReader();
        objc.release(self.track);
        objc.release(self.asset);
        self.track = null;
        self.asset = null;
    }

    pub fn close(self: *Reader) void {
        const pool = objc.Pool.push();
        defer pool.pop();
        self.teardown();
        self.gpa.destroy(self);
    }

    /// The next `next` returns frame `n`.
    pub fn seek(self: *Reader, n: u64) Error!void {
        const pool = objc.Pool.push();
        defer pool.pop();
        const fps = if (self.fps > 0) self.fps else 30;
        const t = self.first_pts + @as(f64, @floatFromInt(n)) / fps;
        // Start a little early: the range is in presentation time and the
        // frame's own time may be rounded.
        try self.start(@max(0, t - 0.5 / fps));
        self.skip_before = t - 0.5 / fps;
        self.pos = n;
    }

    /// Reads the next frame into `dst` (the video's size; BGR, BGRA or
    /// gray). Returns false at the end.
    pub fn next(self: *Reader, dst: Image(u8)) Error!bool {
        if (dst.width != self.width or dst.height != self.height) return error.SizeMismatch;
        if (dst.channels != 1 and dst.channels != 3 and dst.channels != 4) return error.UnsupportedChannels;
        const a = self.api;
        const pool = objc.Pool.push();
        defer pool.pop();
        if (self.output == null) return false;
        while (true) {
            const pending = self.pending;
            self.pending = null;
            const sb = pending orelse send(?*anyopaque, self.output, "copyNextSampleBuffer", .{}) orelse {
                // Status 3: failed.
                if (send(objc.NSInteger, self.reader, "status", .{}) == 3) {
                    setDetail("{s}", .{objc.errorText(send(id, self.reader, "error", .{}))});
                    return error.ReadFailed;
                }
                return false;
            };
            defer a.CFRelease(sb);
            const pb = a.CMSampleBufferGetImageBuffer(sb) orelse continue;
            if (self.skip_before >= 0) {
                const pts = a.CMSampleBufferGetPresentationTimeStamp(sb).seconds();
                std.debug.print("avf pts {d:.4} skip_before {d:.4} first {d:.4}\n", .{ pts, self.skip_before, self.first_pts });
                if (pts < self.skip_before) continue;
                self.skip_before = -1;
            }
            const format = a.CVPixelBufferGetPixelFormatType(pb);
            if (a.CVPixelBufferLockBaseAddress(pb, lock_read_only) != 0) return error.ReadFailed;
            defer _ = a.CVPixelBufferUnlockBaseAddress(pb, lock_read_only);
            const w = @min(a.CVPixelBufferGetWidth(pb), dst.width);
            const h = @min(a.CVPixelBufferGetHeight(pb), dst.height);
            switch (format) {
                pixel_format_bgra => {
                    const base = a.CVPixelBufferGetBaseAddress(pb) orelse return error.ReadFailed;
                    const bpr = a.CVPixelBufferGetBytesPerRow(pb);
                    if (bpr < w * 4) return error.ReadFailed;
                    copyBgra(base, bpr, w, h, dst);
                },
                pixel_format_420v, pixel_format_420f => {
                    const yp = a.CVPixelBufferGetBaseAddressOfPlane(pb, 0) orelse return error.ReadFailed;
                    const uv = a.CVPixelBufferGetBaseAddressOfPlane(pb, 1) orelse return error.ReadFailed;
                    const ybpr = a.CVPixelBufferGetBytesPerRowOfPlane(pb, 0);
                    const cbpr = a.CVPixelBufferGetBytesPerRowOfPlane(pb, 1);
                    if (ybpr < w or cbpr < (w + 1) / 2 * 2 or a.CVPixelBufferGetHeightOfPlane(pb, 1) < (h + 1) / 2) return error.ReadFailed;
                    const m = a.CVBufferGetAttachment(pb, a.kCVImageBufferYCbCrMatrixKey, null);
                    const bt709 = m != null and a.CFEqual(m, a.kCVImageBufferYCbCrMatrix_ITU_R_709_2) != 0;
                    nv12ToBgr(yp, ybpr, uv, cbpr, w, h, format == pixel_format_420f, bt709, dst);
                },
                else => return error.Unsupported,
            }
            self.pos += 1;
            return true;
        }
    }
};

/// Bi-planar 4:2:0 (Y plane, interleaved CbCr plane) to BGR, BGRA or gray,
/// fixed point with 8 fractional bits like y4m.yuvToBgr, nearest chroma.
fn nv12ToBgr(yp: [*]const u8, ybpr: usize, uv: [*]const u8, cbpr: usize, w: usize, h: usize, full: bool, bt709: bool, dst: Image(u8)) void {
    const ky: i32 = if (full) 256 else 298;
    const y0: i32 = if (full) 0 else 16;
    // [B from U, G from U, G from V, R from V]
    const k: [4]i32 = if (bt709)
        (if (full) .{ 475, 48, 120, 403 } else .{ 541, 55, 136, 459 })
    else
        (if (full) .{ 454, 88, 183, 359 } else .{ 516, 100, 208, 409 });
    const ch: usize = dst.channels;
    for (0..dst.height) |y| {
        const o = dst.row(y);
        if (y >= h) {
            @memset(o, 0);
            continue;
        }
        const yr = yp + y * ybpr;
        const cr = uv + (y / 2) * cbpr;
        if (ch == 1) {
            // Gray: from the luma, as BGR->gray would give it.
            for (0..w) |x| o[x] = clamp((((@as(i32, yr[x]) - y0) * ky) + 128) >> 8);
        } else for (0..w) |x| {
            const l: i32 = (@as(i32, yr[x]) - y0) * ky;
            const u: i32 = @as(i32, cr[(x / 2) * 2]) - 128;
            const v: i32 = @as(i32, cr[(x / 2) * 2 + 1]) - 128;
            o[x * ch ..][0..3].* = .{ clamp((l + k[0] * u + 128) >> 8), clamp((l - k[1] * u - k[2] * v + 128) >> 8), clamp((l + k[3] * v + 128) >> 8) };
            if (ch == 4) o[x * 4 + 3] = 255;
        }
        if (w < dst.width) @memset(o[w * ch ..], 0);
    }
}

fn clamp(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// BGRA rows into a BGR, BGRA or gray image (the part `w` x `h` covers;
/// the rest, if the decoder's buffer is smaller, is cleared).
fn copyBgra(base: [*]const u8, bpr: usize, w: usize, h: usize, dst: Image(u8)) void {
    const ch: usize = dst.channels;
    for (0..dst.height) |y| {
        const o = dst.row(y);
        if (y >= h) {
            @memset(o, 0);
            continue;
        }
        const s = base[y * bpr ..][0 .. w * 4];
        switch (ch) {
            4 => @memcpy(o[0 .. w * 4], s),
            3 => for (0..w) |x| {
                o[x * 3 ..][0..3].* = s[x * 4 ..][0..3].*;
            },
            else => for (0..w) |x| {
                const p = s[x * 4 ..][0..3];
                o[x] = @intCast((@as(u32, p[0]) * 1868 + @as(u32, p[1]) * 9617 + @as(u32, p[2]) * 4899 + 8192) >> 14);
            },
        }
        if (w < dst.width) @memset(o[w * ch ..], 0);
    }
}
