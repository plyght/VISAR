//! Fixed and automatic (Otsu, triangle) thresholding, inRange.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const simd = @import("../core/simd.zig");

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
        histogram8Exec(exec, src, &hist);
        thresh = switch (auto) {
            .otsu => otsu(&hist, src.pixelCount()),
            .triangle => triangle(&hist),
            .none => unreachable,
        };
    }
    // Like OpenCV, 8-bit thresholds are floored to an integer; out-of-range
    // ones go through the LUT below.
    const ti: i32 = if (T == u8) @intFromFloat(@floor(thresh)) else 0;
    const t: T = if (T == u8) @intCast(std.math.clamp(ti, 0, 255)) else if (@typeInfo(T) == .float) @floatCast(thresh) else sat.cast(T, @floor(thresh));
    const m: T = sat.cast(T, maxval);
    if (T == u8) {
        if (ti < 0 or ti >= 255) {
            var lut: [256]u8 = undefined;
            for (&lut, 0..) |*l, i_| {
                const i: i32 = @intCast(i_);
                const v: u8 = @intCast(i_);
                l.* = switch (kind) {
                    .binary => if (i > ti) m else 0,
                    .binary_inv => if (i > ti) 0 else m,
                    .trunc => if (i > ti) sat.cast(u8, ti) else v,
                    .tozero => if (i > ti) v else 0,
                    .tozero_inv => if (i > ti) 0 else v,
                };
            }
            applyLut(exec, src, dst, &lut);
            return thresh;
        }
    }
    if (T != u8 and @typeInfo(T) == .int and @floor(thresh) < std.math.minInt(T)) {
        // Every value is above the threshold; like OpenCV, trunc gives 0.
        // (Thresholds above the range saturate to the maximum, which no value
        // exceeds, so the general path handles them.)
        switch (kind) {
            .tozero => if (src.data != dst.data) dst.copyFrom(src),
            .binary => dst.fill(m),
            .binary_inv, .trunc, .tozero_inv => dst.fill(0),
        }
        return thresh;
    }
    const Ctx = struct { s: @TypeOf(src), d: @TypeOf(src), t: T, m: T, kind: Type };
    exec.parallelFor(src.height, @max(1, (1 << 15) / @max(1, src.rowLen())), Ctx{ .s = src, .d = dst, .t = t, .m = m, .kind = kind }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.kind) {
                inline else => |k| for (y0..y1) |y| threshRow(T, k, c.s.row(y), c.d.row(y), c.t, c.m),
            }
        }
    }.run);
    return thresh;
}

inline fn threshOne(comptime T: type, comptime k: Type, v: anytype, t: @TypeOf(v), m: @TypeOf(v)) @TypeOf(v) {
    const V = @TypeOf(v);
    const z: V = if (@typeInfo(V) == .vector) @splat(0) else 0;
    const gt = v > t;
    const sel = struct {
        inline fn f(c: anytype, a: V, b: V) V {
            return if (@typeInfo(V) == .vector) @select(T, c, a, b) else if (c) a else b;
        }
    }.f;
    return switch (k) {
        .binary => sel(gt, m, z),
        .binary_inv => sel(gt, z, m),
        .trunc => sel(gt, t, v),
        .tozero => sel(gt, v, z),
        .tozero_inv => sel(gt, z, v),
    };
}

fn threshRow(comptime T: type, comptime k: Type, s: []const T, d: []T, t: T, m: T) void {
    const N = simd.lanes(T);
    const V = @Vector(N, T);
    const vt: V = @splat(t);
    const vm: V = @splat(m);
    var i: usize = 0;
    while (i + 2 * N <= s.len) : (i += 2 * N) {
        const a: V = s[i..][0..N].*;
        const b: V = s[i + N ..][0..N].*;
        d[i..][0..N].* = threshOne(T, k, a, vt, vm);
        d[i + N ..][0..N].* = threshOne(T, k, b, vt, vm);
    }
    while (i + N <= s.len) : (i += N) {
        const a: V = s[i..][0..N].*;
        d[i..][0..N].* = threshOne(T, k, a, vt, vm);
    }
    while (i < s.len) : (i += 1) d[i] = threshOne(T, k, s[i], t, m);
}

/// dst[i] = lut[src[i]] for 8-bit images (any channel count, same in/out).
pub fn applyLut(exec: Exec, src: Image(u8), dst: Image(u8), lut: *const [256]u8) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), lut: *const [256]u8 };
    exec.parallelFor(src.height, @max(1, (1 << 15) / @max(1, src.rowLen())), Ctx{ .s = src, .d = dst, .lut = lut }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| lutRow(c.s.row(y), c.d.row(y), c.lut);
        }
    }.run);
}

fn lutRow(s: []const u8, d: []u8, lut: *const [256]u8) void {
    var i: usize = 0;
    // 8 bytes per load/store: the gathers are independent of each other.
    while (i + 8 <= s.len) : (i += 8) {
        const v = std.mem.readInt(u64, s[i..][0..8], .little);
        var o: u64 = 0;
        inline for (0..8) |j| o |= @as(u64, lut[@as(u8, @truncate(v >> (8 * j)))]) << (8 * j);
        std.mem.writeInt(u64, d[i..][0..8], o, .little);
    }
    while (i < s.len) : (i += 1) d[i] = lut[s[i]];
}

pub fn histogram8(src: Image(u8), hist: *[256]u32) void {
    histRows(src, 0, src.height, hist);
}

/// histogram8, split over row bands when `exec` has threads.
pub fn histogram8Exec(exec: Exec, src: Image(u8), hist: *[256]u32) void {
    const bands = @min(exec.threads() * 2, 32, src.height / 32);
    if (bands <= 1 or src.pixelCount() < 1 << 16) return histogram8(src, hist);
    var parts: [32][256]u32 = undefined;
    const Ctx = struct { s: Image(u8), parts: *[32][256]u32, bands: usize };
    exec.parallelFor(bands, 1, Ctx{ .s = src, .parts = &parts, .bands = bands }, struct {
        fn run(c: Ctx, b0: usize, b1: usize) void {
            for (b0..b1) |b| {
                c.parts[b] = @splat(0);
                histRows(c.s, c.s.height * b / c.bands, c.s.height * (b + 1) / c.bands, &c.parts[b]);
            }
        }
    }.run);
    for (parts[0..bands]) |*p| for (hist, p) |*o, v| {
        o.* += v;
    };
}

fn histRows(src: Image(u8), y0: usize, y1: usize, hist: *[256]u32) void {
    // Sub-histograms hide store-to-load dependencies on repeated values
    // (common in natural images, where neighbours are often equal).
    var h: [8][256]u32 = @splat(@splat(0));
    for (y0..y1) |y| {
        const r = src.row(y);
        var i: usize = 0;
        while (i + 16 <= r.len) : (i += 16) {
            const a = std.mem.readInt(u64, r[i..][0..8], .little);
            const b = std.mem.readInt(u64, r[i + 8 ..][0..8], .little);
            inline for (0..8) |j| {
                h[j][@as(u8, @truncate(a >> (8 * j)))] += 1;
                h[(j + 4) % 8][@as(u8, @truncate(b >> (8 * j)))] += 1;
            }
        }
        while (i < r.len) : (i += 1) h[0][r[i]] += 1;
    }
    for (hist, 0..) |*o, i| {
        var t: u32 = 0;
        for (&h) |*sub| t += sub[i];
        o.* += t;
    }
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
