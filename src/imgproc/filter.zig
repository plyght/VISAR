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
fn Spec(comptime S_: type, comptime H_: type, comptime V_: type, comptime D_: type, comptime K_: type, comptime F: type) type {
    return SpecK(S_, H_, V_, D_, K_, F, null, null);
}

/// A Spec whose kernels are known at compile time: tap loops unroll and
/// constant multiplies become shifts and adds.
fn SpecK(comptime S_: type, comptime H_: type, comptime V_: type, comptime D_: type, comptime K_: type, comptime F: type, comptime ckx_: ?[]const K_, comptime cky_: ?[]const K_) type {
    return struct {
        const ckx = ckx_;
        const cky = cky_;
        const S = S_;
        const H = H_;
        const V = V_;
        const D = D_;
        const K = K_;
        fn finish(v: V, scale: f64) D {
            const r: [1]D = F.apply(D, 1, @as(@Vector(1, V), @splat(v)), scale);
            return r[0];
        }
        inline fn finishV(comptime N: usize, v: @Vector(N, V), scale: f64) @Vector(N, D) {
            return F.apply(D, N, v, scale);
        }
    };
}

/// Saturating vector conversion with OpenCV rounding (half to even).
inline fn satV(comptime D: type, comptime N: usize, v: anytype) @Vector(N, D) {
    const S = @typeInfo(@TypeOf(v)).vector.child;
    if (S == D) return v;
    if (@typeInfo(D) == .float) return cvtV(D, N, v);
    const lo = std.math.minInt(D);
    const hi = std.math.maxInt(D);
    if (@typeInfo(S) == .float) {
        const r = sat.rint(v);
        const c = @min(@max(r, @as(@Vector(N, S), @splat(@floatFromInt(lo)))), @as(@Vector(N, S), @splat(@floatFromInt(hi))));
        return @intFromFloat(c);
    }
    const W = if (@bitSizeOf(S) > @bitSizeOf(D)) S else i64;
    const w: @Vector(N, W) = v;
    return core.simd.narrow(D, @min(@max(w, @as(@Vector(N, W), @splat(lo))), @as(@Vector(N, W), @splat(hi))));
}

/// Finish kinds: plain saturating cast, scaled (f32 multiply), u8 fixed point.
const FinishCast = struct {
    inline fn apply(comptime D: type, comptime N: usize, v: anytype, _: f64) @Vector(N, D) {
        return satV(D, N, v);
    }
};
const FinishScaled = struct {
    inline fn apply(comptime D: type, comptime N: usize, v: anytype, scale: f64) @Vector(N, D) {
        const f: @Vector(N, f32) = cvtV(f32, N, v);
        return satV(D, N, f * @as(@Vector(N, f32), @splat(@floatCast(scale))));
    }
};
/// (v + 2^(s-1)) >> s with s passed (as a float) in `scale`.
const FinishShift = struct {
    inline fn apply(comptime D: type, comptime N: usize, v: anytype, scale: f64) @Vector(N, D) {
        const E = @typeInfo(@TypeOf(v)).vector.child;
        const s: std.math.Log2Int(E) = @intFromFloat(scale);
        const one: E = 1;
        const r = if (s == 0) v else (v +| @as(@Vector(N, E), @splat(one << (s - 1)))) >> @splat(s);
        return @intCast(@min(r, @as(@Vector(N, E), @splat(255))));
    }
};

/// Divides out the largest power of two common to all taps (in place;
/// callers pass a copy).
fn reduce(k: []u16) struct { shift: u5, sum: u32 } {
    var tz: u5 = 15;
    for (k) |v| if (v != 0) {
        tz = @min(tz, @as(u5, @intCast(@ctz(v))));
    };
    var sum: u32 = 0;
    for (k) |*v| {
        v.* >>= @intCast(tz);
        sum += v.*;
    }
    return .{ .shift = tz, .sum = sum };
}

const FinishU8Fixed = struct {
    inline fn apply(comptime D: type, comptime N: usize, v: anytype, _: f64) @Vector(N, D) {
        const r = (v +| @as(@Vector(N, u32), @splat(1 << 15))) >> @splat(16);
        return @intCast(@min(r, @as(@Vector(N, u32), @splat(255))));
    }
};

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

fn lanes(comptime T: type) comptime_int {
    return @max(4, std.simd.suggestVectorLength(T) orelse 4);
}

/// Converts a vector of S to a vector of A (int widening, int->float, ...).
inline fn cvtV(comptime A: type, comptime N: usize, v: anytype) @Vector(N, A) {
    const S = @typeInfo(@TypeOf(v)).vector.child;
    if (S == A) return v;
    if (@typeInfo(A) == .float) {
        return if (@typeInfo(S) == .float) @floatCast(v) else @floatFromInt(v);
    }
    return @intCast(v);
}

inline fn kconst(comptime A: type, k: anytype) A {
    return if (@typeInfo(A) == .float) floatOf(A, k) else @intCast(k);
}

inline fn accV(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    const A = @typeInfo(@TypeOf(a)).vector.child;
    return if (@typeInfo(A) == .int and @typeInfo(A).int.signedness == .unsigned) a +| b else a + b;
}

/// Vertical-first separable filtering of rows [y0, y1). For each output
/// row the `ky` taps combine source rows straight into one padded row of
/// P.H (which stays in L1), then the `kx` taps run along that row. With the
/// exact integer accumulators VISAR uses the order of the two passes does
/// not change results.
fn chunk(comptime P: type, s: Image(P.S), d: Image(P.D), kx: []const P.K, ky: []const P.K, border: Border, bv: P.S, scale: f64, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const cn: usize = s.channels;
    const n = s.rowLen();
    const rx = kx.len / 2;
    const ry = ky.len / 2;
    const vrow = try gpa.alloc(P.H, n + 2 * rx * cn);
    defer gpa.free(vrow);
    const const_row = try gpa.alloc(P.S, if (border == .constant) n else 0);
    defer gpa.free(const_row);
    @memset(const_row, bv);
    // Value of a constant-border pixel after the vertical pass.
    var const_v: P.H = 0;
    for (ky) |kv| const_v = acc(P.H, const_v, mul(P.H, kv, bv));
    const vsym = symmetry(ky);
    const hsym = symmetry(kx);
    var rows: [128][*]const P.S = undefined;
    for (y0..y1) |y| {
        for (0..ky.len) |j| {
            const v = @as(isize, @intCast(y + j)) - @as(isize, @intCast(ry));
            rows[j] = if (border_mod.map(v, s.height, border)) |sy| s.row(sy).ptr else const_row.ptr;
        }
        const mid = vrow[rx * cn ..][0..n];
        if (P.cky) |ck| vfirstConst(P, ck, rows[0..ky.len], mid) else switch (vsym) {
            inline else => |sym| vfirst(P, sym, rows[0..ky.len], ky, mid),
        }
        // Horizontal border: replicate the vertically filtered columns.
        for (0..rx) |i| {
            const lp = vrow[i * cn ..][0..cn];
            const rp = vrow[(rx + s.width + i) * cn ..][0..cn];
            if (border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(rx)), s.width, border)) |sx| @memcpy(lp, mid[sx * cn ..][0..cn]) else @memset(lp, const_v);
            if (border_mod.map(@intCast(s.width + i), s.width, border)) |sx| @memcpy(rp, mid[sx * cn ..][0..cn]) else @memset(rp, const_v);
        }
        const out = d.row(y);
        if (P.ckx) |ck| hsecondConst(P, ck, vrow, out, cn, scale) else switch (hsym) {
            inline else => |sym| hsecond(P, sym, vrow, kx, out, cn, scale),
        }
    }
}

/// Signed or float accumulators; unsigned ones saturate like OpenCV's
/// ufixedpoint (never reached with VISAR's kernel bounds).
inline fn acc(comptime A: type, a: A, b: A) A {
    return if (@typeInfo(A) == .int and @typeInfo(A).int.signedness == .unsigned) a +| b else a + b;
}

const Symmetry = enum { none, even, odd };

fn symmetry(k: anytype) Symmetry {
    const n = k.len;
    var even = true;
    var odd = true;
    for (0..n / 2) |i| {
        const a = k[i];
        const b = k[n - 1 - i];
        if (a != b) even = false;
        if (@typeInfo(@TypeOf(a)) == .int and @typeInfo(@TypeOf(a)).int.signedness == .unsigned) {
            odd = false;
        } else if (a != -b) odd = false;
    }
    if (n % 2 == 1 and k[n / 2] != 0) odd = false;
    return if (even) .even else if (odd) .odd else .none;
}

/// Weighted sum of `rows` at x..x+N (as A), folding mirrored taps.
inline fn tapsV(comptime A: type, comptime S: type, comptime N: usize, comptime sym: Symmetry, rows: anytype, k: anytype, x: usize) @Vector(N, A) {
    const VA = @Vector(N, A);
    const r = k.len / 2;
    if (sym == .none) {
        var a: VA = @splat(0);
        for (0..k.len) |j| {
            const v: @Vector(N, S) = rows[j][x..][0..N].*;
            a = accV(a, cvtV(A, N, v) * @as(VA, @splat(kconst(A, k[j]))));
        }
        return a;
    }
    const c: @Vector(N, S) = rows[r][x..][0..N].*;
    var a: VA = if (sym == .even) cvtV(A, N, c) * @as(VA, @splat(kconst(A, k[r]))) else @splat(0);
    for (0..r) |i| {
        const t: @Vector(N, S) = rows[i][x..][0..N].*;
        const b: @Vector(N, S) = rows[k.len - 1 - i][x..][0..N].*;
        const pair = if (sym == .even) accV(cvtV(A, N, t), cvtV(A, N, b)) else cvtV(A, N, b) - cvtV(A, N, t);
        a = accV(a, pair * @as(VA, @splat(kconst(A, if (sym == .even) k[i] else k[k.len - 1 - i]))));
    }
    return a;
}

inline fn tapsS(comptime A: type, rows: anytype, k: anytype, x: usize) A {
    var a: A = mul(A, k[0], rows[0][x]);
    for (1..k.len) |j| a = acc(A, a, mul(A, k[j], rows[j][x]));
    return a;
}

fn vfirst(comptime P: type, comptime sym: Symmetry, rows: []const [*]const P.S, k: []const P.K, out: []P.H) void {
    const N = comptime lanes(P.H);
    // Local copies: the compiler cannot prove `out` does not alias the
    // caller's kernel or row table, and would reload them for every vector.
    var kk: [128]P.K = undefined;
    var rr: [128][*]const P.S = undefined;
    @memcpy(kk[0..k.len], k);
    @memcpy(rr[0..k.len], rows);
    const kl = kk[0..k.len];
    const rl = rr[0..k.len];
    var x: usize = 0;
    while (x + N <= out.len) : (x += N) out[x..][0..N].* = tapsV(P.H, P.S, N, sym, rl, kl, x);
    while (x < out.len) : (x += 1) out[x] = tapsS(P.H, rl, kl, x);
}

fn hsecond(comptime P: type, comptime sym: Symmetry, row: []const P.H, k: []const P.K, out: []P.D, cn: usize, scale: f64) void {
    const N = comptime lanes(P.V);
    // Tap i of output x reads row[x + i*cn]: express as shifted row starts.
    var starts: [128][*]const P.H = undefined;
    for (0..k.len) |i| starts[i] = row.ptr + i * cn;
    var kk: [128]P.K = undefined;
    @memcpy(kk[0..k.len], k);
    const kl = kk[0..k.len];
    const rs = starts[0..k.len];
    var x: usize = 0;
    while (x + N <= out.len) : (x += N) out[x..][0..N].* = P.finishV(N, tapsV(P.V, P.H, N, sym, rs, kl, x), scale);
    while (x < out.len) : (x += 1) out[x] = P.finish(tapsS(P.V, rs, kl, x), scale);
}

/// Compile-time kernels: taps unroll and multiplies become shifts/adds.
fn vfirstConst(comptime P: type, comptime k: []const P.K, rows: []const [*]const P.S, out: []P.H) void {
    const N = comptime lanes(P.H);
    var rs: [k.len][*]const P.S = undefined;
    inline for (0..k.len) |j| rs[j] = rows[j];
    var x: usize = 0;
    while (x + N <= out.len) : (x += N) out[x..][0..N].* = constTaps(P.H, P.S, N, k, &rs, x);
    while (x < out.len) : (x += 1) {
        var a: P.H = 0;
        inline for (k, 0..) |kv, j| a += @as(P.H, kv) * rs[j][x];
        out[x] = a;
    }
}

fn hsecondConst(comptime P: type, comptime k: []const P.K, row: []const P.H, out: []P.D, cn: usize, scale: f64) void {
    const N = comptime lanes(P.V);
    var rs: [k.len][*]const P.H = undefined;
    inline for (0..k.len) |i| rs[i] = row.ptr + i * cn;
    var x: usize = 0;
    while (x + N <= out.len) : (x += N) out[x..][0..N].* = P.finishV(N, constTaps(P.V, P.H, N, k, &rs, x), scale);
    while (x < out.len) : (x += 1) {
        var a: P.V = 0;
        inline for (k, 0..) |kv, j| a += @as(P.V, kv) * rs[j][x];
        out[x] = P.finish(a, scale);
    }
}

inline fn constTaps(comptime A: type, comptime S: type, comptime N: usize, comptime k: anytype, rs: anytype, x: usize) @Vector(N, A) {
    const VA = @Vector(N, A);
    const r = k.len / 2;
    const c: @Vector(N, S) = rs[r][x..][0..N].*;
    var a: VA = cvtV(A, N, c) * @as(VA, @splat(k[r]));
    inline for (0..r) |i| {
        const t: @Vector(N, S) = rs[i][x..][0..N].*;
        const b: @Vector(N, S) = rs[k.len - 1 - i][x..][0..N].*;
        const pair = cvtV(A, N, t) + cvtV(A, N, b);
        a += if (k[i] == 1) pair else pair * @as(VA, @splat(k[i]));
    }
    return a;
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
        // When every tap shares a power-of-two factor (OpenCV's small fixed
        // kernels do: [1 4 6 4 1] * 16), divide it out. If the reduced 2-D
        // sum fits 16 bits the whole filter runs in u16 lanes, twice as
        // wide, and rounding at the end gives exactly the same result.
        var kxr: [127]u16 = undefined;
        var kyr: [127]u16 = undefined;
        @memcpy(kxr[0..kw], kx[0..kw]);
        @memcpy(kyr[0..kh], ky[0..kh]);
        const rx = reduce(kxr[0..kw]);
        const ry = reduce(kyr[0..kh]);
        if (255 * @as(u64, rx.sum) * ry.sum <= std.math.maxInt(u16)) {
            @memcpy(kx[0..kw], kxr[0..kw]);
            @memcpy(ky[0..kh], kyr[0..kh]);
            const shift: u5 = 16 - (rx.shift + ry.shift);
            // OpenCV's fixed small kernels, fully unrolled.
            const k3 = [_]u16{ 1, 2, 1 };
            const k5 = [_]u16{ 1, 4, 6, 4, 1 };
            if (kw == kh and std.mem.eql(u16, kx[0..kw], ky[0..kh])) {
                if (kw == 3 and std.mem.eql(u16, kx[0..3], &k3)) {
                    const P = SpecK(u8, u16, u16, u8, u16, FinishShift, &k3, &k3);
                    return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
                }
                if (kw == 5 and std.mem.eql(u16, kx[0..5], &k5)) {
                    const P = SpecK(u8, u16, u16, u8, u16, FinishShift, &k5, &k5);
                    return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
                }
            }
            const P = Spec(u8, u16, u16, u8, u16, FinishShift);
            return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
        }
        const P = Spec(u8, u16, u32, u8, u16, FinishU8Fixed);
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
        const P = Spec(T, f32, f32, T, f32, FinishCast);
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
    const ones: [255]u8 = @splat(1);
    const scale: f64 = if (o.normalize) 1.0 / @as(f64, @floatFromInt(o.kw * o.kh)) else 1.0;
    if (@typeInfo(T) == .float) {
        const fones: [255]f32 = @splat(1);
        const P = Spec(T, f32, f32, T, f32, FinishScaled);
        return separable(P, exec, gpa, src, dst, fones[0..o.kw], fones[0..o.kh], o.border, 0, scale);
    }
    if (T == u8 and o.border != .constant) return boxU8(exec, gpa, src, dst, o.kw, o.kh, o.border, scale);
    const P = Spec(T, u32, u32, T, u8, FinishScaled);
    return separable(P, exec, gpa, src, dst, ones[0..o.kw], ones[0..o.kh], o.border, 0, scale);
}

/// 8-bit box filter: horizontal sums in u16, then a running column sum in
/// u32 (add the entering row, subtract the leaving one), so the vertical
/// cost does not grow with the kernel height.
fn boxU8(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), kw: u32, kh: u32, border: Border, scale: f64) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { s: Image(u8), d: Image(u8), kw: u32, kh: u32, border: Border, scale: f64, gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(16, (1 << 17) / @max(1, src.rowLen()));
    exec.parallelFor(src.height, grain, Ctx{ .s = src, .d = dst, .kw = kw, .kh = kh, .border = border, .scale = scale, .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            boxChunk(c.s, c.d, c.kw, c.kh, c.border, c.scale, c.gpa, y0, y1) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

fn boxChunk(s: Image(u8), d: Image(u8), kw: u32, kh: u32, border: Border, scale: f64, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const cn: usize = s.channels;
    const n = s.rowLen();
    const rx: usize = kw / 2;
    const ry: usize = kh / 2;
    const pad = try gpa.alloc(u8, n + 2 * rx * cn);
    defer gpa.free(pad);
    const ring = try gpa.alloc(u16, @as(usize, kh) * n);
    defer gpa.free(ring);
    const col = try gpa.alloc(u32, n);
    defer gpa.free(col);
    const N = comptime lanes(u32);
    const V32 = @Vector(N, u32);

    const Row = struct {
        /// Copies source row v (with border) into the padded buffer.
        fn load(img: Image(u8), v: isize, b: Border, p: []u8, r: usize, ch: usize) void {
            const sr = img.row(border_mod.map(v, img.height, b).?);
            @memcpy(p[r * ch ..][0..sr.len], sr);
            for (0..r) |i| {
                const lx = border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(r)), img.width, b).?;
                const rxx = border_mod.map(@intCast(img.width + i), img.width, b).?;
                @memcpy(p[i * ch ..][0..ch], sr[lx * ch ..][0..ch]);
                @memcpy(p[(r + img.width + i) * ch ..][0..ch], sr[rxx * ch ..][0..ch]);
            }
        }
        inline fn hsumV(p: []const u8, x: usize, w: usize, ch: usize) V32 {
            var a: @Vector(N, u16) = @as(@Vector(N, u8), p[x..][0..N].*);
            for (1..w) |i| a += @as(@Vector(N, u8), p[x + i * ch ..][0..N].*);
            return a;
        }
        inline fn hsum(p: []const u8, x: usize, w: usize, ch: usize) u16 {
            var a: u16 = p[x];
            for (1..w) |i| a += p[x + i * ch];
            return a;
        }
    };

    const v0: isize = @as(isize, @intCast(y0)) - @as(isize, @intCast(ry));
    const khi: isize = @intCast(kh);
    // Prime: col = sum of rows y0-ry .. y0+ry-1 (kh-1 rows).
    @memset(col, 0);
    for (0..kh - 1) |j| {
        const v = v0 + @as(isize, @intCast(j));
        Row.load(s, v, border, pad, rx, cn);
        const slot: usize = @intCast(@mod(v - v0, khi));
        const row = ring[slot * n ..][0..n];
        for (0..n) |x| {
            row[x] = Row.hsum(pad, x, kw, cn);
            col[x] += row[x];
        }
    }
    const sc: @Vector(N, f32) = @splat(@floatCast(scale));
    const scs: f32 = @floatCast(scale);
    for (y0..y1) |y| {
        // One pass: horizontal sum of the entering row, emit, slide.
        const vin: isize = @as(isize, @intCast(y + ry));
        const vout: isize = @as(isize, @intCast(y)) - @as(isize, @intCast(ry));
        Row.load(s, vin, border, pad, rx, cn);
        const rin = ring[@as(usize, @intCast(@mod(vin - v0, khi))) * n ..][0..n];
        const rout = ring[@as(usize, @intCast(@mod(vout - v0, khi))) * n ..][0..n];
        const out = d.row(y);
        var x: usize = 0;
        while (x + N <= n) : (x += N) {
            const h = Row.hsumV(pad, x, kw, cn);
            const c = @as(V32, col[x..][0..N].*) + h;
            out[x..][0..N].* = satV(u8, N, cvtV(f32, N, c) * sc);
            // When kh == 1 the entering and leaving rows are the same row.
            const old: V32 = if (kh == 1) h else @as(@Vector(N, u16), rout[x..][0..N].*);
            col[x..][0..N].* = c - old;
            rin[x..][0..N].* = @as(@Vector(N, u16), @intCast(h));
        }
        while (x < n) : (x += 1) {
            const h = Row.hsum(pad, x, kw, cn);
            const c = col[x] + h;
            out[x] = sat.cast(u8, @as(f32, @floatFromInt(c)) * scs);
            col[x] = c - (if (kh == 1) h else rout[x]);
            rin[x] = h;
        }
    }
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
        const P = Spec(S, f32, f32, D, f32, FinishScaled);
        return separable(P, exec, gpa, src, dst, fx[0..kxs.len], fy[0..kys.len], o.border, 0, o.scale);
    }
    var bound: i64 = 255;
    var sx_: i64 = 0;
    var sy_: i64 = 0;
    for (kxs) |v| sx_ += @abs(v);
    for (kys) |v| sy_ += @abs(v);
    bound *= sx_ * sy_;
    if (S == u8 and D == i16 and o.scale == 1 and bound <= std.math.maxInt(i16)) {
        var kx16: [31]i16 = undefined;
        var ky16: [31]i16 = undefined;
        for (kx16[0..kxs.len], kxs) |*a, b| a.* = @intCast(b);
        for (ky16[0..kys.len], kys) |*a, b| a.* = @intCast(b);
        const P = Spec(S, i16, i16, D, i16, FinishCast);
        return separable(P, exec, gpa, src, dst, kx16[0..kxs.len], ky16[0..kys.len], o.border, 0, 1);
    }
    if (o.scale == 1) {
        const P = Spec(S, i32, i32, D, i32, FinishCast);
        return separable(P, exec, gpa, src, dst, kxs, kys, o.border, 0, 1);
    }
    const P = Spec(S, i32, i32, D, i32, FinishScaled);
    return separable(P, exec, gpa, src, dst, kxs, kys, o.border, 0, o.scale);
}

/// General separable filter with float kernels (sepFilter2D).
pub fn sepFilter2D(exec: Exec, gpa: Allocator, src: anytype, dst: anytype, kx: []const f32, ky: []const f32, border: Border) Error!void {
    const S = @TypeOf(src).Elem;
    const D = @TypeOf(dst).Elem;
    const P = Spec(S, f32, f32, D, f32, FinishCast);
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
