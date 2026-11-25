//! YUV4MPEG2 (.y4m) reading: uncompressed video, one header line and then
//! frames of planar 8-bit YUV, each after a "FRAME" line. Any video becomes
//! one with ffmpeg:
//!
//!   ffmpeg -i clip.mp4 -pix_fmt yuv420p clip.y4m
//!
//! Chroma 420 (every siting variant), 422, 444 and mono are read; frames are
//! converted to BGR or BGRA with the BT.601 matrix, limited range unless
//! the header says XCOLORRANGE=FULL, chroma taken from the nearest sample.
//! High bit depths (C420p10, ...) are not supported.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Chroma = enum { c420, c422, c444, mono };

pub const Error = error{ InvalidData, Unsupported, Truncated, SizeMismatch, UnsupportedChannels, ReadFailed, OutOfMemory };

pub const Reader = struct {
    gpa: Allocator,
    in: *Io.Reader,
    width: u32,
    height: u32,
    /// Frame rate (0/0 when the header has none).
    fps_num: u32 = 0,
    fps_den: u32 = 0,
    chroma: Chroma = .c420,
    full_range: bool = false,
    /// One frame's planes.
    buf: []u8,
    frames: u64 = 0,

    /// Reads the header from `in` (which must outlive the reader).
    pub fn init(gpa: Allocator, in: *Io.Reader) Error!Reader {
        const magic = "YUV4MPEG2";
        const line = in.takeDelimiter('\n') catch |e| return mapErr(e) orelse return error.InvalidData;
        const hdr = line orelse return error.InvalidData;
        if (!std.mem.startsWith(u8, hdr, magic)) return error.InvalidData;
        var r: Reader = .{ .gpa = gpa, .in = in, .width = 0, .height = 0, .buf = &.{} };
        var it = std.mem.tokenizeScalar(u8, hdr[magic.len..], ' ');
        while (it.next()) |tok| {
            const v = tok[1..];
            switch (tok[0]) {
                'W' => r.width = std.fmt.parseInt(u32, v, 10) catch return error.InvalidData,
                'H' => r.height = std.fmt.parseInt(u32, v, 10) catch return error.InvalidData,
                'F' => {
                    var f = std.mem.splitScalar(u8, v, ':');
                    r.fps_num = std.fmt.parseInt(u32, f.next() orelse "", 10) catch return error.InvalidData;
                    r.fps_den = std.fmt.parseInt(u32, f.next() orelse "", 10) catch return error.InvalidData;
                },
                'I' => if (v.len > 0 and v[0] == 'm') return error.Unsupported,
                'C' => r.chroma = if (std.mem.startsWith(u8, v, "420") and (v.len == 3 or std.mem.eql(u8, v[3..], "jpeg") or std.mem.eql(u8, v[3..], "paldv") or std.mem.eql(u8, v[3..], "mpeg2")))
                    .c420
                else if (std.mem.eql(u8, v, "422"))
                    .c422
                else if (std.mem.eql(u8, v, "444"))
                    .c444
                else if (std.mem.eql(u8, v, "mono"))
                    .mono
                else
                    return error.Unsupported,
                'X' => if (std.ascii.eqlIgnoreCase(v, "COLORRANGE=FULL")) {
                    r.full_range = true;
                },
                else => {},
            }
        }
        if (r.width == 0 or r.height == 0 or r.width > 16384 or r.height > 16384) return error.InvalidData;
        r.buf = try gpa.alloc(u8, r.frameBytes());
        return r;
    }

    pub fn deinit(r: *Reader) void {
        r.gpa.free(r.buf);
        r.* = undefined;
    }

    fn chromaSize(r: *const Reader) [2]usize {
        const w: usize = r.width;
        const h: usize = r.height;
        return switch (r.chroma) {
            .c420 => .{ (w + 1) / 2, (h + 1) / 2 },
            .c422 => .{ (w + 1) / 2, h },
            .c444 => .{ w, h },
            .mono => .{ 0, 0 },
        };
    }

    /// Bytes of one frame's planes.
    pub fn frameBytes(r: *const Reader) usize {
        const c = r.chromaSize();
        return @as(usize, r.width) * r.height + 2 * c[0] * c[1];
    }

    /// Reads the next frame into `dst` (the video's size, BGR or BGRA).
    /// Returns false at the end of the video.
    pub fn next(r: *Reader, dst: Image(u8)) Error!bool {
        if (dst.width != r.width or dst.height != r.height) return error.SizeMismatch;
        if (dst.channels != 3 and dst.channels != 4) return error.UnsupportedChannels;
        const line = (r.in.takeDelimiter('\n') catch |e| return mapErr(e) orelse error.InvalidData) orelse return false;
        if (!std.mem.startsWith(u8, line, "FRAME")) return error.InvalidData;
        r.in.readSliceAll(r.buf) catch |e| return switch (e) {
            error.EndOfStream => error.Truncated,
            error.ReadFailed => error.ReadFailed,
        };
        r.frames += 1;
        r.convert(dst);
        return true;
    }

    fn convert(r: *const Reader, dst: Image(u8)) void {
        const w: usize = r.width;
        const h: usize = r.height;
        const c = r.chromaSize();
        const y_plane = r.buf[0 .. w * h];
        const u_plane = r.buf[w * h ..][0 .. c[0] * c[1]];
        const v_plane = r.buf[w * h + c[0] * c[1] ..][0 .. c[0] * c[1]];
        yuvToBgr(r.chroma, r.full_range, y_plane, u_plane, v_plane, dst);
    }
};

/// Planar 8-bit YUV (BT.601; limited range unless `full_range`) to BGR or
/// BGRA `dst`, chroma taken from the nearest sample. The planes are
/// `dst`'s size (Y) and the chroma size `chroma` implies (U, V; ignored
/// for mono), rows packed.
pub fn yuvToBgr(chroma: Chroma, full_range: bool, y_plane: []const u8, u_plane: []const u8, v_plane: []const u8, dst: Image(u8)) void {
    const w: usize = dst.width;
    const h: usize = dst.height;
    const cw: usize = switch (chroma) {
        .c444 => w,
        .mono => 0,
        else => (w + 1) / 2,
    };
    const ch: usize = dst.channels;
    // Fixed point, 8 fractional bits (as camera_linux.yuyvToBgra).
    const ky: i32 = if (full_range) 256 else 298;
    const y0: i32 = if (full_range) 0 else 16;
    const ku_b: i32 = if (full_range) 454 else 516;
    const ku_g: i32 = if (full_range) 88 else 100;
    const kv_g: i32 = if (full_range) 183 else 208;
    const kv_r: i32 = if (full_range) 359 else 409;
    for (0..h) |y| {
        const d = dst.row(y);
        const yr = y_plane[y * w ..][0..w];
        const cy = switch (chroma) {
            .c420 => y / 2,
            else => y,
        };
        for (0..w) |x| {
            const l: i32 = (@as(i32, yr[x]) - y0) * ky;
            var u: i32 = 0;
            var v: i32 = 0;
            if (chroma != .mono) {
                const cx = if (chroma == .c444) x else x / 2;
                u = @as(i32, u_plane[cy * cw + cx]) - 128;
                v = @as(i32, v_plane[cy * cw + cx]) - 128;
            }
            const px = d[x * ch ..][0..3];
            px.* = .{ clampU8((l + ku_b * u + 128) >> 8), clampU8((l - ku_g * u - kv_g * v + 128) >> 8), clampU8((l + kv_r * v + 128) >> 8) };
            if (ch == 4) d[x * 4 + 3] = 255;
        }
    }
}

/// Bytes of a 4:2:0 frame of this size (Y, U and V planes).
pub fn i420Bytes(width: u32, height: u32) usize {
    const cw: usize = (width + 1) / 2;
    const chh: usize = (height + 1) / 2;
    return @as(usize, width) * height + 2 * cw * chh;
}

/// BGR or BGRA `src` to planar 4:2:0 YUV (BT.601, limited range), chroma
/// averaged over each 2x2 block. `out` holds i420Bytes bytes: Y, then U,
/// then V.
pub fn bgrToI420(src: Image(u8), out: []u8) void {
    const w: usize = src.width;
    const h: usize = src.height;
    const cw = (w + 1) / 2;
    const chh = (h + 1) / 2;
    const ch: usize = src.channels;
    const yp = out[0 .. w * h];
    const up = out[w * h ..][0 .. cw * chh];
    const vp = out[w * h + cw * chh ..][0 .. cw * chh];
    for (0..h) |y| {
        const s = src.row(y);
        for (0..w) |x| {
            const b: i32 = s[x * ch];
            const g: i32 = s[x * ch + 1];
            const r: i32 = s[x * ch + 2];
            yp[y * w + x] = @intCast(((66 * r + 129 * g + 25 * b + 128) >> 8) + 16);
        }
    }
    for (0..chh) |cy| {
        const r0 = src.row(2 * cy);
        const r1 = src.row(@min(2 * cy + 1, h - 1));
        for (0..cw) |cx| {
            const x0 = 2 * cx;
            const x1 = @min(2 * cx + 1, w - 1);
            var sum: [3]i32 = .{ 0, 0, 0 };
            for ([_][]const u8{ r0, r1 }) |row| {
                for ([_]usize{ x0, x1 }) |x| {
                    for (0..3) |k| sum[k] += row[x * ch + k];
                }
            }
            const b = @divTrunc(sum[0] + 2, 4);
            const g = @divTrunc(sum[1] + 2, 4);
            const r = @divTrunc(sum[2] + 2, 4);
            up[cy * cw + cx] = clampU8(((-38 * r - 74 * g + 112 * b + 128) >> 8) + 128);
            vp[cy * cw + cx] = clampU8(((112 * r - 94 * g - 18 * b + 128) >> 8) + 128);
        }
    }
}

/// Writes YUV4MPEG2: 4:2:0 (C420jpeg), progressive, limited range.
pub const Writer = struct {
    gpa: Allocator,
    out: *Io.Writer,
    width: u32,
    height: u32,
    buf: []u8,
    frames: u64 = 0,

    pub const WriteError = error{ InvalidData, SizeMismatch, UnsupportedChannels, OutOfMemory, WriteFailed };

    /// Writes the header to `out` (which must outlive the writer).
    pub fn init(gpa: Allocator, out: *Io.Writer, width: u32, height: u32, fps_num: u32, fps_den: u32) WriteError!Writer {
        if (width == 0 or height == 0 or width > 16384 or height > 16384) return error.InvalidData;
        const buf = try gpa.alloc(u8, i420Bytes(width, height));
        errdefer gpa.free(buf);
        try out.print("YUV4MPEG2 W{d} H{d} F{d}:{d} Ip A1:1 C420jpeg XYSCSS=420JPEG\n", .{ width, height, @max(fps_num, 1), @max(fps_den, 1) });
        return .{ .gpa = gpa, .out = out, .width = width, .height = height, .buf = buf };
    }

    pub fn deinit(w: *Writer) void {
        w.gpa.free(w.buf);
        w.* = undefined;
    }

    /// Appends a BGR or BGRA frame of the video's size.
    pub fn write(w: *Writer, frame: Image(u8)) WriteError!void {
        if (frame.width != w.width or frame.height != w.height) return error.SizeMismatch;
        if (frame.channels != 3 and frame.channels != 4) return error.UnsupportedChannels;
        bgrToI420(frame, w.buf);
        try w.out.writeAll("FRAME\n");
        try w.out.writeAll(w.buf);
        w.frames += 1;
    }
};

fn mapErr(e: anyerror) ?Error {
    return switch (e) {
        error.ReadFailed => error.ReadFailed,
        error.StreamTooLong => error.InvalidData,
        else => null,
    };
}

fn clampU8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

/// Whether `bytes` start like a YUV4MPEG2 file.
pub fn detect(bytes: []const u8) bool {
    return std.mem.startsWith(u8, bytes, "YUV4MPEG2 ");
}

test "y4m: header, 420 frames, conversion and the end" {
    const gpa = std.testing.allocator;
    // 4x2 frames: Y 235 (white) on the left, 16 (black) on the right, no
    // chroma; then a mid-gray frame with full red chroma (V = 240).
    const hdr = "YUV4MPEG2 W4 H2 F30:1 Ip A1:1 C420jpeg XYSCSS=420JPEG\n";
    const f1 = "FRAME\n" ++ [_]u8{ 235, 235, 16, 16, 235, 235, 16, 16 } ++ [_]u8{ 128, 128 } ++ [_]u8{ 128, 128 };
    const f2 = "FRAME Ixyz\n" ++ [_]u8{ 126, 126, 126, 126, 126, 126, 126, 126 } ++ [_]u8{ 128, 128 } ++ [_]u8{ 240, 240 };
    var in: Io.Reader = .fixed(hdr ++ f1 ++ f2);
    var r = try Reader.init(gpa, &in);
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 4), r.width);
    try std.testing.expectEqual(@as(u32, 30), r.fps_num);
    var buf: [4 * 2 * 4]u8 = undefined;
    const img = try Image(u8).fromSlice(&buf, 4, 2, 4);
    try std.testing.expect(try r.next(img));
    try std.testing.expectEqualSlices(u8, &.{ 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 255 }, buf[0..16]);
    try std.testing.expect(try r.next(img));
    // Red: B and G low, R high.
    try std.testing.expect(buf[2] > 200 and buf[0] < 140 and buf[1] < 60);
    try std.testing.expect(!try r.next(img));
    try std.testing.expectEqual(@as(u64, 2), r.frames);
}

test "y4m: writer round trip" {
    const gpa = std.testing.allocator;
    var aw: Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    var src = try Image(u8).init(gpa, 6, 4, 3);
    defer src.deinit(gpa);
    for (0..4) |y| for (0..6) |x| {
        src.row(y)[x * 3 ..][0..3].* = .{ @intCast(100 + 10 * x), @intCast(80 + 20 * y), 128 };
    };
    var w = try Writer.init(gpa, &aw.writer, 6, 4, 25, 1);
    try w.write(src);
    try w.write(src);
    w.deinit();
    var in: Io.Reader = .fixed(aw.written());
    var r = try Reader.init(gpa, &in);
    defer r.deinit();
    try std.testing.expectEqual(@as(u32, 25), r.fps_num);
    var out = try Image(u8).init(gpa, 6, 4, 3);
    defer out.deinit(gpa);
    var n: usize = 0;
    while (try r.next(out)) n += 1;
    try std.testing.expectEqual(@as(usize, 2), n);
    // 4:2:0 loses chroma detail; the mean error stays small.
    var err: u64 = 0;
    for (0..4) |y| for (out.row(y), src.row(y)) |a, b| {
        err += @abs(@as(i32, a) - @as(i32, b));
    };
    try std.testing.expect(err / (6 * 4 * 3) < 6);
}

test "y4m: rejects what it cannot read" {
    const gpa = std.testing.allocator;
    var a: Io.Reader = .fixed("YUV4MPEG2 W4 H2 C420p10\n");
    try std.testing.expectError(error.Unsupported, Reader.init(gpa, &a));
    var b: Io.Reader = .fixed("P6 4 2 255\n");
    try std.testing.expectError(error.InvalidData, Reader.init(gpa, &b));
    var c: Io.Reader = .fixed("YUV4MPEG2 W4 H2 Cmono\nFRAME\n\x10\x10");
    var r = try Reader.init(gpa, &c);
    defer r.deinit();
    var buf: [4 * 2 * 3]u8 = undefined;
    try std.testing.expectError(error.Truncated, r.next(try Image(u8).fromSlice(&buf, 4, 2, 3)));
}
