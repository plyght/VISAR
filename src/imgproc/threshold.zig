//! Fixed and automatic (Otsu, triangle) thresholding, inRange.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;

pub const Error = core.Error || error{Unsupported};

pub const Type = enum(u8) {
    binary = 0,
    binary_inv = 1,
    trunc = 2,
    tozero = 3,
    tozero_inv = 4,
};

pub const Auto = enum { none, otsu, triangle };

/// Applies the threshold and returns the threshold actually used (the
/// computed one for Otsu/triangle). Like OpenCV, 8-bit thresholds are
/// floored and maxval rounded.
pub fn threshold(exec: Exec, src: anytype, dst: @TypeOf(src), thresh_in: f64, maxval: f64, kind: Type, auto: Auto) Error!f64 {
    const T = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    var thresh = thresh_in;
    if (auto != .none) {
        if (T != u8 or src.channels != 1) return error.Unsupported;
        var hist: [256]u32 = @splat(0);
        histogram8(src, &hist);
        thresh = switch (auto) {
            .otsu => otsu(&hist, src.pixelCount()),
            .triangle => triangle(&hist),
            .none => unreachable,
        };
    }
    if (T == u8) {
        const t: i32 = @intFromFloat(@floor(thresh));
        const m: u8 = sat.cast(u8, maxval);
        var lut: [256]u8 = undefined;
        for (&lut, 0..) |*l, i_| {
            const i: i32 = @intCast(i_);
            const v: u8 = @intCast(i_);
            l.* = switch (kind) {
                .binary => if (i > t) m else 0,
                .binary_inv => if (i > t) 0 else m,
                .trunc => if (i > t) sat.cast(u8, t) else v,
                .tozero => if (i > t) v else 0,
                .tozero_inv => if (i > t) 0 else v,
            };
        }
        if (t < 0 or t >= 255) {
            // OpenCV special-cases out-of-range thresholds; the LUT covers it.
        }
        applyLut(exec, src, dst, &lut);
        return thresh;
    }
    const Ctx = struct { s: @TypeOf(src), d: @TypeOf(src), t: T, m: T, kind: Type };
    const t: T = if (@typeInfo(T) == .float) @floatCast(thresh) else sat.cast(T, @floor(thresh));
    const m: T = sat.cast(T, maxval);
    exec.parallelFor(src.height, 16, Ctx{ .s = src, .d = dst, .t = t, .m = m, .kind = kind }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                const s = c.s.row(y);
                const d = c.d.row(y);
                switch (c.kind) {
                    inline else => |k| for (s, d) |v, *o| {
                        o.* = switch (k) {
                            .binary => if (v > c.t) c.m else 0,
                            .binary_inv => if (v > c.t) 0 else c.m,
                            .trunc => if (v > c.t) c.t else v,
                            .tozero => if (v > c.t) v else 0,
                            .tozero_inv => if (v > c.t) 0 else v,
                        };
                    },
                }
            }
        }
    }.run);
    return thresh;
}

/// dst[i] = lut[src[i]] for 8-bit images (any channel count, same in/out).
pub fn applyLut(exec: Exec, src: Image(u8), dst: Image(u8), lut: *const [256]u8) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), lut: *const [256]u8 };
    exec.parallelFor(src.height, 32, Ctx{ .s = src, .d = dst, .lut = lut }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| for (c.s.row(y), c.d.row(y)) |v, *o| {
                o.* = c.lut[v];
            };
        }
    }.run);
}

pub fn histogram8(src: Image(u8), hist: *[256]u32) void {
    // Four sub-histograms hide store-to-load dependencies on repeated values.
    var h: [4][256]u32 = @splat(@splat(0));
    for (0..src.height) |y| {
        const r = src.row(y);
        var i: usize = 0;
        while (i + 4 <= r.len) : (i += 4) {
            h[0][r[i]] += 1;
            h[1][r[i + 1]] += 1;
            h[2][r[i + 2]] += 1;
            h[3][r[i + 3]] += 1;
        }
        while (i < r.len) : (i += 1) h[0][r[i]] += 1;
    }
    for (hist, 0..) |*o, i| o.* += h[0][i] + h[1][i] + h[2][i] + h[3][i];
}

/// Otsu's method, OpenCV's getThreshVal_Otsu_8u.
fn otsu(hist: *const [256]u32, total: usize) f64 {
    const scale = 1.0 / @as(f64, @floatFromInt(total));
    var mu: f64 = 0;
    for (hist, 0..) |h, i| mu += @as(f64, @floatFromInt(i)) * @as(f64, @floatFromInt(h));
    mu *= scale;
    var mu1: f64 = 0;
    var q1: f64 = 0;
    var max_sigma: f64 = 0;
    var max_val: f64 = 0;
    for (hist, 0..) |h, i| {
        const p_i = @as(f64, @floatFromInt(h)) * scale;
        mu1 *= q1;
        q1 += p_i;
        const q2 = 1.0 - q1;
        if (@min(q1, q2) < std.math.floatEps(f64) or @max(q1, q2) > 1.0 - std.math.floatEps(f64)) continue;
        mu1 = (mu1 + @as(f64, @floatFromInt(i)) * p_i) / q1;
        const mu2 = (mu - q1 * mu1) / q2;
        const sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2);
        if (sigma > max_sigma) {
            max_sigma = sigma;
            max_val = @floatFromInt(i);
        }
    }
    return max_val;
}

/// Triangle method, OpenCV's getThreshVal_Triangle_8u.
fn triangle(hist: *const [256]u32) f64 {
    var left_bound: i32 = 0;
    var right_bound: i32 = 0;
    var max_ind: i32 = 0;
    var max: u32 = 0;
    for (hist, 0..) |h, i| if (h > 0) {
        left_bound = @intCast(i);
        break;
    };
    if (left_bound > 0) left_bound -= 1;
    var i: i32 = 255;
    while (i > 0) : (i -= 1) if (hist[@intCast(i)] > 0) {
        right_bound = i;
        break;
    };
    if (right_bound < 255) right_bound += 1;
    for (hist, 0..) |h, j| if (h > max) {
        max = h;
        max_ind = @intCast(j);
    };
    var flip = false;
    var hcopy = hist.*;
    if (max_ind - left_bound < right_bound - max_ind) {
        flip = true;
        std.mem.reverse(u32, &hcopy);
        left_bound = 255 - right_bound;
        max_ind = 255 - max_ind;
    }
    var thresh = left_bound;
    const a: f64 = @floatFromInt(max);
    const b: f64 = @floatFromInt(left_bound - max_ind);
    var dist: f64 = 0;
    var j = left_bound + 1;
    while (j <= max_ind) : (j += 1) {
        const tempdist = a * @as(f64, @floatFromInt(j)) + b * @as(f64, @floatFromInt(hcopy[@intCast(j)]));
        if (tempdist > dist) {
            dist = tempdist;
            thresh = j;
        }
    }
    thresh -= 1;
    if (flip) thresh = 255 - thresh;
    return @floatFromInt(thresh);
}

/// dst = 255 where lo <= src <= hi for every channel, else 0. dst is 1-channel u8.
pub fn inRange(exec: Exec, src: anytype, lo: []const f64, hi: []const f64, dst: Image(u8)) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameSize(dst) or dst.channels != 1) return error.SizeMismatch;
    if (lo.len < src.channels or hi.len < src.channels) return error.UnsupportedChannels;
    const Ctx = struct { s: @TypeOf(src), d: Image(u8), lo: []const f64, hi: []const f64 };
    exec.parallelFor(src.height, 16, Ctx{ .s = src, .d = dst, .lo = lo, .hi = hi }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn = c.s.channels;
            for (y0..y1) |y| {
                const s = c.s.row(y);
                const d = c.d.row(y);
                for (d, 0..) |*o, x| {
                    var ok = true;
                    for (0..cn) |ch| {
                        const v: f64 = if (@typeInfo(T) == .float) @floatCast(s[x * cn + ch]) else @floatFromInt(s[x * cn + ch]);
                        ok = ok and v >= c.lo[ch] and v <= c.hi[ch];
                    }
                    o.* = if (ok) 255 else 0;
                }
            }
        }
    }.run);
}

test "threshold binary and otsu" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 8, 1, 1);
    defer src.deinit(gpa);
    @memcpy(src.slice(), &[_]u8{ 10, 20, 30, 200, 210, 220, 15, 205 });
    var dst = try Image(u8).init(gpa, 8, 1, 1);
    defer dst.deinit(gpa);
    _ = try threshold(.serial, src, dst, 100, 255, .binary, .none);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 255, 255, 255, 0, 255 }, dst.slice());
    const t = try threshold(.serial, src, dst, 0, 255, .binary, .otsu);
    try std.testing.expect(t >= 30 and t < 200);
}
