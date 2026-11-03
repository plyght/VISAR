//! Histograms, equalisation and integral images.

const std = @import("std");
const core = @import("../core/core.zig");
const threshold = @import("threshold.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;

pub const Error = core.Error;

/// 256-bin histogram of a single-channel 8-bit image.
pub fn calcHist(src: Image(u8)) Error![256]u32 {
    if (src.channels != 1) return error.UnsupportedChannels;
    var h: [256]u32 = @splat(0);
    threshold.histogram8(src, &h);
    return h;
}

/// equalizeHist, OpenCV's algorithm (LUT from the cumulative histogram,
/// excluding the first occupied bin).
pub fn equalizeHist(exec: Exec, src: Image(u8), dst: Image(u8)) Error!void {
    if (src.channels != 1 or !src.sameShape(dst)) return error.SizeMismatch;
    var hist: [256]u32 = @splat(0);
    threshold.histogram8Exec(exec, src, &hist);
    const total = src.pixelCount();
    var i: usize = 0;
    while (hist[i] == 0) i += 1;
    var lut: [256]u8 = @splat(0);
    if (hist[i] == total) {
        dst.fill(@intCast(i));
        return;
    }
    const scale: f32 = 255.0 / @as(f32, @floatFromInt(total - hist[i]));
    var acc: u32 = 0;
    lut[i] = 0;
    i += 1;
    while (i < 256) : (i += 1) {
        acc += hist[i];
        lut[i] = sat.cast(u8, @as(f32, @floatFromInt(acc)) * scale);
    }
    threshold.applyLut(exec, src, dst, &lut);
}

/// integral(src) -> (h+1) x (w+1) sums, per channel. dst must be i32 (8-bit
/// source) or f64.
pub fn integral(src: anytype, dst: anytype) Error!void {
    const S = @TypeOf(src).Elem;
    const D = @TypeOf(dst).Elem;
    const cn: usize = src.channels;
    if (dst.width != src.width + 1 or dst.height != src.height + 1 or dst.channels != src.channels) return error.SizeMismatch;
    @memset(dst.row(0), 0);
    for (0..src.height) |y| {
        const s = src.row(y);
        const prev = dst.row(y);
        const d = dst.row(y + 1);
        @memset(d[0..cn], 0);
        var run: [4]D = @splat(0);
        for (0..src.width) |x| for (0..cn) |ch| {
            const v: D = if (@typeInfo(D) == .float) (if (@typeInfo(S) == .float) @floatCast(s[x * cn + ch]) else @floatFromInt(s[x * cn + ch])) else @intCast(s[x * cn + ch]);
            run[ch % 4] += v;
            d[(x + 1) * cn + ch] = prev[(x + 1) * cn + ch] + run[ch % 4];
        };
    }
}

test "equalize and integral" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 4, 1, 1);
    defer src.deinit(gpa);
    @memcpy(src.slice(), &[_]u8{ 10, 10, 20, 30 });
    var dst = try Image(u8).initLike(gpa, src);
    defer dst.deinit(gpa);
    try equalizeHist(.serial, src, dst);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 128, 255 }, dst.slice());
    var ii = try Image(i32).init(gpa, 5, 2, 1);
    defer ii.deinit(gpa);
    try integral(src, ii);
    try std.testing.expectEqualSlices(i32, &.{ 0, 10, 20, 40, 70 }, ii.row(1));
}
