//! Linear filtering: a separable engine plus box, Gaussian, Sobel and
//! Scharr filters. 8-bit Gaussian blur uses OpenCV's bit-exact fixed point
//! (8 fractional bits per tap) so results match cv2.

const std = @import("std");
const core = @import("../core/core.zig");
const border_mod = @import("border.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const Allocator = std.mem.Allocator;

pub const Border = border_mod.Border;
pub const Error = core.Error || Allocator.Error || error{InvalidKernel};

/// Describes one separable filter: sample types, kernel type, accumulator
/// types and how a vertical sum becomes an output sample.
fn Spec(comptime S_: type, comptime H_: type, comptime V_: type, comptime D_: type, comptime K_: type, comptime finish_: fn (V_, f64) D_) type {
    return struct {
        const S = S_;
        const H = H_;
        const V = V_;
        const D = D_;
        const K = K_;
        const finish = finish_;
    };
}

/// Runs `kx` horizontally then `ky` vertically. `scale` is passed to the
/// spec's finish function.
fn separable(comptime P: type, exec: Exec, gpa: Allocator, src: Image(P.S), dst: Image(P.D), kx: []const P.K, ky: []const P.K, border: Border, border_value: P.S, scale: f64) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (kx.len == 0 or ky.len == 0 or kx.len % 2 == 0 or ky.len % 2 == 0) return error.InvalidKernel;
    const Ctx = struct { s: Image(P.S), d: Image(P.D), kx: []const P.K, ky: []const P.K, border: Border, bv: P.S, scale: f64, gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(8, (1 << 16) / @max(1, src.rowLen()));
    exec.parallelFor(src.height, grain, Ctx{ .s = src, .d = dst, .kx = kx, .ky = ky, .border = border, .bv = border_value, .scale = scale, .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            chunk(P, c.s, c.d, c.kx, c.ky, c.border, c.bv, c.scale, c.gpa, y0, y1) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

fn chunk(comptime P: type, s: Image(P.S), d: Image(P.D), kx: []const P.K, ky: []const P.K, border: Border, bv: P.S, scale: f64, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const cn: usize = s.channels;
    const n = s.rowLen();
    const rx = kx.len / 2;
    const ry = ky.len / 2;
    const kh = ky.len;
    const pad_len = n + 2 * rx * cn;
    const pad = try gpa.alloc(P.S, pad_len);
    defer gpa.free(pad);
    const ring = try gpa.alloc(P.H, kh * n);
    defer gpa.free(ring);
    const sums = try gpa.alloc(P.V, n);
    defer gpa.free(sums);
    var const_row: ?[]P.H = null;
    defer if (const_row) |r| gpa.free(r);
    if (border == .constant) {
        const r = try gpa.alloc(P.H, n);
        @memset(pad, bv);
        hpass(P, pad, r, kx, cn);
        const_row = r;
    }
    // Border columns (left then right), computed once.
    var left: [32]?usize = undefined;
    var right: [32]?usize = undefined;
    const rxc = @min(rx, 32);
    for (0..rxc) |i| {
        left[i] = border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(rx)), s.width, border);
        right[i] = border_mod.map(@as(isize, @intCast(s.width + i)), s.width, border);
    }

    // Virtual source rows needed: y0-ry .. y1+ry-1. ring slot = v mod kh.
    const v0: isize = @as(isize, @intCast(y0)) - @as(isize, @intCast(ry));
    var next_v: isize = v0;
    var rows_ptr: [128][]const P.H = undefined;
    for (y0..y1) |y| {
        const want_last: isize = @as(isize, @intCast(y + ry));
        while (next_v <= want_last) : (next_v += 1) {
            const slot: usize = @intCast(@mod(next_v - v0, @as(isize, @intCast(kh))));
            const dst_row = ring[slot * n ..][0..n];
            if (border_mod.map(next_v, s.height, border)) |sy| {
                const sr = s.row(sy);
                @memcpy(pad[rx * cn ..][0..n], sr);
                if (rx > 0) {
                    if (rx <= 32) {
                        for (0..rx) |i| {
                            const lp = pad[i * cn ..][0..cn];
                            const rp = pad[(rx + s.width + i) * cn ..][0..cn];
                            if (left[i]) |sx| @memcpy(lp, sr[sx * cn ..][0..cn]) else @memset(lp, bv);
                            if (right[i]) |sx| @memcpy(rp, sr[sx * cn ..][0..cn]) else @memset(rp, bv);
                        }
                    } else {
                        for (0..rx) |i| {
                            const lp = pad[i * cn ..][0..cn];
                            const rp = pad[(rx + s.width + i) * cn ..][0..cn];
                            if (border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(rx)), s.width, border)) |sx| @memcpy(lp, sr[sx * cn ..][0..cn]) else @memset(lp, bv);
                            if (border_mod.map(@intCast(s.width + i), s.width, border)) |sx| @memcpy(rp, sr[sx * cn ..][0..cn]) else @memset(rp, bv);
                        }
                    }
                }
                hpass(P, pad, dst_row, kx, cn);
            } else {
                @memcpy(dst_row, const_row.?);
            }
        }
        // Rows y-ry .. y+ry live in the ring.
        for (0..kh) |j| {
            const v: isize = @as(isize, @intCast(y + j)) - @as(isize, @intCast(ry));
            const slot: usize = @intCast(@mod(v - v0, @as(isize, @intCast(kh))));
            rows_ptr[j] = ring[slot * n ..][0..n];
        }
        vpass(P, rows_ptr[0..kh], ky, sums, d.row(y), scale);
    }
}

inline fn hpass(comptime P: type, pad: []const P.S, out: []P.H, k: []const P.K, cn: usize) void {
    const n = out.len;
    for (out, 0..) |*o, x| o.* = mul(P.H, k[0], pad[x]);
    for (k[1..], 1..) |kv, i| {
        const off = i * cn;
        const src = pad[off..][0..n];
        for (out, src) |*o, v| o.* = acc(P.H, o.*, mul(P.H, kv, v));
    }
}

/// Unsigned integer accumulators saturate (OpenCV's ufixedpoint types);
/// signed and float ones add normally.
inline fn acc(comptime A: type, a: A, b: A) A {
    return if (@typeInfo(A) == .int and @typeInfo(A).int.signedness == .unsigned) a +| b else a + b;
}

inline fn vpass(comptime P: type, rows: []const []const P.H, k: []const P.K, sums: []P.V, out: []P.D, scale: f64) void {
    for (sums, rows[0]) |*a, v| a.* = mul(P.V, k[0], v);
    for (rows[1..], k[1..]) |r, kv| {
        for (sums, r) |*a, v| a.* = acc(P.V, a.*, mul(P.V, kv, v));
    }
    for (out, sums) |*o, a| o.* = P.finish(a, scale);
}

inline fn mul(comptime A: type, k: anytype, v: anytype) A {
    const ka: A = if (@typeInfo(A) == .float) floatOf(A, k) else @intCast(k);
    const va: A = if (@typeInfo(A) == .float) floatOf(A, v) else @intCast(v);
    return ka * va;
}

inline fn floatOf(comptime A: type, v: anytype) A {
    return switch (@typeInfo(@TypeOf(v))) {
        .float => @floatCast(v),
        else => @floatFromInt(v),
    };
}

// ---------------------------------------------------------------- Gaussian

/// OpenCV's getGaussianKernel. With sigma <= 0 and ksize <= 7, the fixed
/// small tables are used; otherwise sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8.
pub fn gaussianKernel(out: []f64, sigma_in: f64) void {
    const n = out.len;
    const small = [_][]const f64{
        &.{1.0},
        &.{ 0.25, 0.5, 0.25 },
        &.{ 0.0625, 0.25, 0.375, 0.25, 0.0625 },
        &.{ 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 },
    };
    if (n % 2 == 1 and n <= 7 and sigma_in <= 0) {
        @memcpy(out, small[n / 2]);
        return;
    }
    const sigma = if (sigma_in > 0) sigma_in else ((@as(f64, @floatFromInt(n)) - 1) * 0.5 - 1) * 0.3 + 0.8;
    const scale2x = -0.5 / (sigma * sigma);
    var sum: f64 = 0;
    for (out, 0..) |*v, i| {
        const x = @as(f64, @floatFromInt(i)) - @as(f64, @floatFromInt(n - 1)) * 0.5;
        v.* = @exp(scale2x * x * x);
        sum += v.*;
    }
    for (out) |*v| v.* /= sum;
}

/// OpenCV's automatic kernel size for a sigma (8-bit images: 3 sigma each
/// side; others 4 sigma), always odd.
pub fn gaussianKsize(sigma: f64, depth_is_u8: bool) u32 {
    const k: f64 = if (depth_is_u8) 3 else 4;
    const r: u32 = @intFromFloat(@round(sigma * k * 2 + 1));
    return r | 1;
}

/// 8-bit kernel in 8.8 fixed point, OpenCV's getGaussianKernelBitExact:
/// taps are rounded independently (they need not sum to exactly 256).
fn gaussianKernelFixed(out: []u16, sigma: f64) void {
    const n = out.len;
    var tmp: [128]f64 = undefined;
    const k = tmp[0..n];
    if (n % 2 == 1 and n <= 7 and sigma <= 0) {
        gaussianKernel(k, sigma);
    } else {
        // Same arithmetic order as OpenCV (softdouble): taps from the edge
        // inwards with x = 1-n, 3-n, ..., centre tap exp(0) = 1.
        const sx = if (sigma > 0) sigma else @as(f64, @floatFromInt(n)) * 0.15 + 0.35;
        const scale2x = -0.125 / (sx * sx);
        const half = (n - 1) / 2;
        var sum: f64 = 0;
        var x: i64 = 1 - @as(i64, @intCast(n));
        for (0..half) |i| {
            const t = @exp(@as(f64, @floatFromInt(x * x)) * scale2x);
            k[i] = t;
            sum += t;
            x += 2;
        }
        sum = sum * 2 + 1;
        if (n % 2 == 0) sum += 1;
        const mul1 = 1.0 / sum;
        for (0..half) |i| {
            k[i] *= mul1;
            k[n - 1 - i] = k[i];
        }
        k[half] = mul1;
        if (n % 2 == 0) k[half + 1] = mul1;
    }
    for (out, k) |*o, v| o.* = @intCast(std.math.clamp(@as(i64, @intFromFloat(sat.rint(v * 256.0))), 0, 65535));
}

fn finishU8Fixed(v: u32, _: f64) u8 {
    return @intCast(@min(255, (v +| (1 << 15)) >> 16));
}

fn finishCast(comptime D: type, comptime V: type) fn (V, f64) D {
    return struct {
        fn f(v: V, _: f64) D {
            return sat.cast(D, v);
        }
    }.f;
}

fn finishScaled(comptime D: type, comptime V: type) fn (V, f64) D {
    return struct {
        fn f(v: V, scale: f64) D {
            return sat.cast(D, floatOf(f64, v) * scale);
        }
    }.f;
}

pub const GaussianOptions = struct {
    /// Kernel width/height; 0 = derive from sigma.
    kw: u32 = 0,
    kh: u32 = 0,
    sigma_x: f64 = 0,
    /// 0 = same as sigma_x.
    sigma_y: f64 = 0,
    border: Border = .default,
};

/// Gaussian blur. Supports u8 (bit-exact fixed point), u16, i16 and f32.
pub fn gaussianBlur(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), o: GaussianOptions) Error!void {
    const T = @TypeOf(src).Elem;
    const sy_in = if (o.sigma_y > 0) o.sigma_y else o.sigma_x;
    const kw = if (o.kw > 0) o.kw else gaussianKsize(o.sigma_x, T == u8);
    const kh = if (o.kh > 0) o.kh else gaussianKsize(sy_in, T == u8);
    if (kw % 2 == 0 or kh % 2 == 0 or kw > 127 or kh > 127) return error.InvalidKernel;
    if (T == u8) {
        var kx: [127]u16 = undefined;
        var ky: [127]u16 = undefined;
        gaussianKernelFixed(kx[0..kw], o.sigma_x);
        gaussianKernelFixed(ky[0..kh], sy_in);
        const P = Spec(u8, u16, u32, u8, u16, finishU8Fixed);
        return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, 1);
    } else {
        var kxd: [127]f64 = undefined;
        var kyd: [127]f64 = undefined;
        gaussianKernel(kxd[0..kw], o.sigma_x);
        gaussianKernel(kyd[0..kh], sy_in);
        var kx: [127]f32 = undefined;
        var ky: [127]f32 = undefined;
        for (kx[0..kw], kxd[0..kw]) |*a, b| a.* = @floatCast(b);
        for (ky[0..kh], kyd[0..kh]) |*a, b| a.* = @floatCast(b);
        const P = Spec(T, f32, f32, T, f32, finishCast(T, f32));
        return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, 1);
    }
}

// --------------------------------------------------------------------- Box

pub const BoxOptions = struct {
    kw: u32,
    kh: u32,
    normalize: bool = true,
    border: Border = .default,
};

/// Box filter / blur. Integer sums, then (normalised) scaled and rounded.
pub fn boxFilter(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), o: BoxOptions) Error!void {
    const T = @TypeOf(src).Elem;
    if (o.kw == 0 or o.kh == 0 or o.kw > 255 or o.kh > 255) return error.InvalidKernel;
    if (o.kw % 2 == 0 or o.kh % 2 == 0) return error.InvalidKernel; // anchor-centred only
    const ones = [_]u8{1} ** 255;
    const scale: f64 = if (o.normalize) 1.0 / @as(f64, @floatFromInt(o.kw * o.kh)) else 1.0;
    if (@typeInfo(T) == .float) {
        const fones = [_]f32{1} ** 255;
        const P = Spec(T, f32, f32, T, f32, finishScaled(T, f32));
        return separable(P, exec, gpa, src, dst, fones[0..o.kw], fones[0..o.kh], o.border, 0, scale);
    }
    const P = Spec(T, u32, u32, T, u8, finishScaled(T, u32));
    return separable(P, exec, gpa, src, dst, ones[0..o.kw], ones[0..o.kh], o.border, 0, scale);
}

/// blur(): normalised box filter.
pub fn blur(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), kw: u32, kh: u32, border: Border) Error!void {
    return boxFilter(exec, gpa, src, dst, .{ .kw = kw, .kh = kh, .border = border });
}

// ------------------------------------------------------------ Derivatives

/// OpenCV's getDerivKernels: smoothing [1 2 1]-style or derivative taps of
/// the given order for an aperture of ksize (1, 3, 5, 7; -1 = Scharr).
pub fn derivKernel(out: []i32, order: u32, ksize: i32) error{InvalidKernel}!void {
    if (ksize == -1) {
        // Scharr (3 taps).
        if (out.len != 3 or order > 1) return error.InvalidKernel;
        if (order == 0) out[0..3].* = .{ 3, 10, 3 } else out[0..3].* = .{ -1, 0, 1 };
        return;
    }
    const n: usize = @intCast(if (ksize == 1) 3 else ksize);
    if (out.len != n) return error.InvalidKernel;
    if (ksize == 1 and order == 0) {
        out[0..3].* = .{ 0, 1, 0 };
        return;
    }
    if (order >= n) return error.InvalidKernel;
    // Pascal smoothing (n - order - 1 times), then differencing (order times).
    var a: [32]i32 = @splat(0);
    a[0] = 1;
    var len: usize = 1;
    for (0..n - order - 1) |_| {
        var b: [32]i32 = @splat(0);
        for (0..len) |j| {
            b[j] += a[j];
            b[j + 1] += a[j];
        }
        a = b;
        len += 1;
    }
    for (0..order) |_| {
        var b: [32]i32 = @splat(0);
        for (0..len) |j| {
            b[j] -= a[j];
            b[j + 1] += a[j];
        }
        a = b;
        len += 1;
    }
    @memcpy(out, a[0..n]);
}

pub const SobelOptions = struct {
    dx: u32,
    dy: u32,
    /// 1, 3, 5, 7, or -1 for Scharr.
    ksize: i32 = 3,
    scale: f64 = 1,
    delta: f64 = 0,
    border: Border = .default,
};

/// Sobel / Scharr derivative. `dst` may be i16 (from u8), or f32.
pub fn sobel(exec: Exec, gpa: Allocator, src: anytype, dst: anytype, o: SobelOptions) Error!void {
    const S = @TypeOf(src).Elem;
    const D = @TypeOf(dst).Elem;
    const n: usize = if (o.ksize <= 1) 3 else @intCast(o.ksize);
    var kx: [31]i32 = undefined;
    var ky: [31]i32 = undefined;
    const kxs = kx[0..if (o.ksize == 1 and o.dx > 0) 3 else n];
    const kys = ky[0..if (o.ksize == 1 and o.dy > 0) 3 else n];
    try derivKernel(kxs, o.dx, if (o.ksize == 1 and o.dx == 0) 1 else if (o.ksize == 1) 3 else o.ksize);
    try derivKernel(kys, o.dy, if (o.ksize == 1 and o.dy == 0) 1 else if (o.ksize == 1) 3 else o.ksize);
    if (o.delta != 0) return error.InvalidKernel; // TODO: delta
    if (@typeInfo(D) == .float or @typeInfo(S) == .float) {
        var fx: [31]f32 = undefined;
        var fy: [31]f32 = undefined;
        for (fx[0..kxs.len], kxs) |*a, b| a.* = @floatFromInt(b);
        for (fy[0..kys.len], kys) |*a, b| a.* = @floatFromInt(b);
        const P = Spec(S, f32, f32, D, f32, finishScaled(D, f32));
        return separable(P, exec, gpa, src, dst, fx[0..kxs.len], fy[0..kys.len], o.border, 0, o.scale);
    }
    if (o.scale == 1) {
        const P = Spec(S, i32, i32, D, i32, finishCast(D, i32));
        return separable(P, exec, gpa, src, dst, kxs, kys, o.border, 0, 1);
    }
    const P = Spec(S, i32, i32, D, i32, finishScaled(D, i32));
    return separable(P, exec, gpa, src, dst, kxs, kys, o.border, 0, o.scale);
}

/// General separable filter with float kernels (sepFilter2D).
pub fn sepFilter2D(exec: Exec, gpa: Allocator, src: anytype, dst: anytype, kx: []const f32, ky: []const f32, border: Border) Error!void {
    const S = @TypeOf(src).Elem;
    const D = @TypeOf(dst).Elem;
    const P = Spec(S, f32, f32, D, f32, finishCast(D, f32));
    return separable(P, exec, gpa, src, dst, kx, ky, border, 0, 1);
}

test "gaussian 3x3 matches hand computation" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).initFill(gpa, 5, 5, 1, 0);
    defer src.deinit(gpa);
    src.set(2, 2, 0, 255);
    var dst = try Image(u8).init(gpa, 5, 5, 1);
    defer dst.deinit(gpa);
    try gaussianBlur(.serial, gpa, src, dst, .{ .kw = 3, .kh = 3 });
    // centre: 255 * 0.5 * 0.5 = 63.75 -> 64
    try std.testing.expectEqual(@as(u8, 64), dst.at(2, 2, 0));
    try std.testing.expectEqual(@as(u8, 32), dst.at(1, 2, 0));
    try std.testing.expectEqual(@as(u8, 16), dst.at(1, 1, 0));
}

test "sobel kernels" {
    var k: [3]i32 = undefined;
    try derivKernel(&k, 1, 3);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, 1 }, &k);
    try derivKernel(&k, 0, 3);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 1 }, &k);
    var k5: [5]i32 = undefined;
    try derivKernel(&k5, 1, 5);
    try std.testing.expectEqualSlices(i32, &.{ -1, -2, 0, 2, 1 }, &k5);
}
