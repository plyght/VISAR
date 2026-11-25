//! Linear filtering: a separable engine plus box, Gaussian, Sobel and
//! Scharr filters. 8-bit and 16-bit Gaussian blur use OpenCV's bit-exact
//! fixed point (8 and 16 fractional bits per tap) so results match cv2.

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
        // Rounding constant 2^(s-1), or 0 when s = 0: no per-vector select.
        const half: E = @truncate((@as(u64, 1) << s) >> 1);
        const r = (v +| @as(@Vector(N, E), @splat(half))) >> @splat(s);
        return @intCast(@min(r, @as(@Vector(N, E), @splat(std.math.maxInt(D)))));
    }
};

/// FinishShift with the shift known at compile time: on NEON the round,
/// shift, clamp and narrow become one uqrshrn.
fn FinishShiftC(comptime s: u5) type {
    return struct {
        inline fn apply(comptime D: type, comptime N: usize, v: anytype, _: f64) @Vector(N, D) {
            const E = @typeInfo(@TypeOf(v)).vector.child;
            const r = (v +| @as(@Vector(N, E), @splat(1 << (s - 1)))) >> @splat(s);
            return @intCast(@min(r, @as(@Vector(N, E), @splat(std.math.maxInt(D)))));
        }
    };
}

/// Divides out the largest power of two common to all taps (in place;
/// callers pass a copy).
fn reduce(k: anytype) struct { shift: u5, sum: u64 } {
    var tz: u5 = @bitSizeOf(@TypeOf(k[0])) - 1;
    for (k) |v| if (v != 0) {
        tz = @min(tz, @as(u5, @intCast(@ctz(v))));
    };
    var sum: u64 = 0;
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

/// Lanes per step: at least 32 bytes of `T`, so 128-bit SIMD (NEON, SSE)
/// works on two registers at a time, which keeps wide out-of-order cores
/// such as Apple's busy and lets NEON use its paired widening instructions.
fn lanes(comptime T: type) comptime_int {
    const native = @max(4, std.simd.suggestVectorLength(T) orelse 4);
    return @max(native, 64 / @sizeOf(T));
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

/// OpenCV's softfloat exp (f64_exp in core/src/softfloat.cpp): a 64-entry
/// 2^(i/64) table times a degree-5 polynomial, every step an IEEE double
/// operation, so ordinary f64 arithmetic reproduces it bit for bit. It is
/// not correctly rounded, so std.math.exp differs from it in the last bit
/// for many arguments, which can move a quantised Gaussian tap.
fn cvExp(x: f64) f64 {
    if (std.math.isNan(x)) return x;
    if (std.math.isInf(x)) return if (x > 0) x else 0;
    const tab = comptime blk: {
        const raw = [64]u64{
            0x3ff0000000000000, 0x3ff02c9a3e778061, 0x3ff059b0d3158574, 0x3ff0874518759bc8,
            0x3ff0b5586cf9890f, 0x3ff0e3ec32d3d1a2, 0x3ff11301d0125b51, 0x3ff1429aaea92de0,
            0x3ff172b83c7d517b, 0x3ff1a35beb6fcb75, 0x3ff1d4873168b9aa, 0x3ff2063b88628cd6,
            0x3ff2387a6e756238, 0x3ff26b4565e27cdd, 0x3ff29e9df51fdee1, 0x3ff2d285a6e4030b,
            0x3ff306fe0a31b715, 0x3ff33c08b26416ff, 0x3ff371a7373aa9cb, 0x3ff3a7db34e59ff7,
            0x3ff3dea64c123422, 0x3ff4160a21f72e2a, 0x3ff44e086061892d, 0x3ff486a2b5c13cd0,
            0x3ff4bfdad5362a27, 0x3ff4f9b2769d2ca7, 0x3ff5342b569d4f82, 0x3ff56f4736b527da,
            0x3ff5ab07dd485429, 0x3ff5e76f15ad2148, 0x3ff6247eb03a5585, 0x3ff6623882552225,
            0x3ff6a09e667f3bcd, 0x3ff6dfb23c651a2f, 0x3ff71f75e8ec5f74, 0x3ff75feb564267c9,
            0x3ff7a11473eb0187, 0x3ff7e2f336cf4e62, 0x3ff82589994cce13, 0x3ff868d99b4492ed,
            0x3ff8ace5422aa0db, 0x3ff8f1ae99157736, 0x3ff93737b0cdc5e5, 0x3ff97d829fde4e50,
            0x3ff9c49182a3f090, 0x3ffa0c667b5de565, 0x3ffa5503b23e255d, 0x3ffa9e6b5579fdbf,
            0x3ffae89f995ad3ad, 0x3ffb33a2b84f15fb, 0x3ffb7f76f2fb5e47, 0x3ffbcc1e904bc1d2,
            0x3ffc199bdd85529c, 0x3ffc67f12e57d14b, 0x3ffcb720dcef9069, 0x3ffd072d4a07897c,
            0x3ffd5818dcfba487, 0x3ffda9e603db3285, 0x3ffdfc97337b9b5f, 0x3ffe502ee78b3ff6,
            0x3ffea4afa2a490da, 0x3ffefa1bee615a27, 0x3fff50765b6e4540, 0x3fffa7c1819e90d8,
        };
        var t: [64]f64 = undefined;
        for (&t, raw) |*d, r| d.* = @bitCast(r);
        break :blk t;
    };
    const f = struct {
        fn h(comptime r: u64) f64 {
            return @bitCast(r);
        }
    }.h;
    const a0: f64 = comptime f(0x3f83ce0f3e46f431);
    const p5: f64 = comptime 1.0 / a0;
    const p4: f64 = comptime f(0x3fe62e42fefa39f1) / a0;
    const p3: f64 = comptime f(0x3fcebfbdff82a45a) / a0;
    const p2: f64 = comptime f(0x3fac6b08d81fec75) / a0;
    const p1: f64 = comptime f(0x3f83b2a72b4f3cd3) / a0;
    const p0: f64 = comptime f(0x3f55e7aa1566c2a4) / a0;
    const prescale: f64 = comptime f(0x3ff71547652b82fe) * 64.0;
    const max_val: f64 = 3000.0 * 64.0;
    const bits: u64 = @bitCast(x);
    var x0: f64 = if (((bits >> 52) & 0x7ff) > 1023 + 10) (if (x < 0) -max_val else max_val) else x * prescale;
    const r = sat.rint(x0);
    const v: i32 = @intFromFloat(r);
    const t: u64 = @intCast(std.math.clamp((v >> 6) + 1023, 0, 2047));
    const buf: f64 = @bitCast(t << 52);
    x0 = (x0 - r) * (1.0 / 64.0);
    const poly = ((((p0 * x0 + p1) * x0 + p2) * x0 + p3) * x0 + p4) * x0 + p5;
    return buf * a0 * tab[@intCast(v & 63)] * poly;
}

/// OpenCV's getGaussianKernelBitExact (the kernel behind getGaussianKernel
/// and every GaussianBlur): fixed binomial tables for sigma <= 0 and
/// ksize <= 9, otherwise sigma = 0.15*ksize + 0.35 (one fused multiply-add)
/// and exp(-x^2 / (2 sigma^2)) over the half kernel, normalised by the
/// reciprocal of the sum. Matches cv2.getGaussianKernel(n, sigma, CV_64F)
/// bit for bit.
pub fn gaussianKernel(out: []f64, sigma: f64) void {
    const n = out.len;
    const small = [_][]const f64{
        &.{1.0},
        &.{ 0.25, 0.5, 0.25 },
        &.{ 0.0625, 0.25, 0.375, 0.25, 0.0625 },
        &.{ 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 },
        &.{ 4.0 / 256.0, 13.0 / 256.0, 30.0 / 256.0, 51.0 / 256.0, 60.0 / 256.0, 51.0 / 256.0, 30.0 / 256.0, 13.0 / 256.0, 4.0 / 256.0 },
    };
    if (n % 2 == 1 and n <= 9 and sigma <= 0) {
        @memcpy(out, small[n / 2]);
        return;
    }
    // Taps from the edge inwards with x = 1-n, 3-n, ... (twice the offset,
    // hence -1/8 rather than -1/2), centre tap exp(0) = 1.
    const nf: f64 = @floatFromInt(n);
    const sx = if (sigma > 0) sigma else @mulAdd(f64, nf, 0.15, 0.35);
    const scale2x = -0.125 / (sx * sx);
    const half = (n - 1) / 2;
    var sum: f64 = 0;
    var x: i64 = 1 - @as(i64, @intCast(n));
    for (0..half) |i| {
        const t = cvExp(@as(f64, @floatFromInt(x * x)) * scale2x);
        out[i] = t;
        sum += t;
        x += 2;
    }
    sum = sum * 2 + 1;
    if (n % 2 == 0) sum += 1;
    const mul1 = 1.0 / sum;
    for (0..half) |i| {
        out[i] *= mul1;
        out[n - 1 - i] = out[i];
    }
    out[half] = mul1;
    if (n % 2 == 0) out[half + 1] = mul1;
}

/// OpenCV's automatic kernel size for a sigma (8-bit images: 3 sigma each
/// side; others 4 sigma), always odd.
pub fn gaussianKsize(sigma: f64, depth_is_u8: bool) u32 {
    const k: f64 = if (depth_is_u8) 3 else 4;
    const r: u32 = @intFromFloat(@round(sigma * k * 2 + 1));
    return r | 1;
}

/// The fixed-point kernel of OpenCV's bit-exact GaussianBlur
/// (getGaussianKernelFixedPoint_ED): the half kernel is quantised to
/// `bits` fractional bits from the edge inwards with error diffusion
/// (each rounding error carried into the next tap, half to even), and the
/// centre tap takes whatever makes the taps sum to exactly 2^bits.
fn gaussianKernelFixed(comptime K: type, out: []K, sigma: f64, comptime bits: u5) void {
    const n = out.len;
    var tmp: [128]f64 = undefined;
    const k = tmp[0..n];
    gaussianKernel(k, sigma);
    const m: f64 = @floatFromInt(@as(u64, 1) << bits);
    var err: f64 = 0;
    var sum: u64 = 0;
    for (0..n / 2) |i| {
        const a = k[i] * m + err;
        const v = sat.rint(a);
        err = a - v;
        const q: u64 = @intFromFloat(@max(v, 0));
        out[i] = @intCast(q);
        out[n - 1 - i] = @intCast(q);
        sum += q;
    }
    out[n / 2] = @intCast((@as(u64, 1) << bits) - 2 * sum);
}

const FinishU16Fixed = struct {
    inline fn apply(comptime D: type, comptime N: usize, v: anytype, _: f64) @Vector(N, D) {
        const r = (v +| @as(@Vector(N, u64), @splat(1 << 31))) >> @splat(32);
        return @intCast(@min(r, @as(@Vector(N, u64), @splat(65535))));
    }
};

pub const GaussianOptions = struct {
    /// Kernel width/height; 0 = derive from sigma.
    kw: u32 = 0,
    kh: u32 = 0,
    sigma_x: f64 = 0,
    /// 0 = same as sigma_x.
    sigma_y: f64 = 0,
    border: Border = .default,
};

/// Gaussian blur, following cv2.GaussianBlur: u8 and u16 use OpenCV's
/// bit-exact fixed point (8 and 16 fractional bits per tap, exact integer
/// sums, one rounding at the end), i16 and f32 the f32 kernel.
pub fn gaussianBlur(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), o: GaussianOptions) Error!void {
    const T = @TypeOf(src).Elem;
    const sy_in = if (o.sigma_y > 0) o.sigma_y else o.sigma_x;
    var kw = if (o.kw > 0) o.kw else gaussianKsize(o.sigma_x, T == u8);
    var kh = if (o.kh > 0) o.kh else gaussianKsize(sy_in, T == u8);
    if (kw % 2 == 0 or kh % 2 == 0 or kw > 127 or kh > 127) return error.InvalidKernel;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    // Like OpenCV, a one-pixel-wide (or -high) image is not filtered
    // along that axis unless the border is constant.
    if (o.border != .constant) {
        if (src.width == 1) kw = 1;
        if (src.height == 1) kh = 1;
    }
    if (T == u8) {
        var kx: [127]u16 = undefined;
        var ky: [127]u16 = undefined;
        gaussianKernelFixed(u16, kx[0..kw], o.sigma_x, 8);
        gaussianKernelFixed(u16, ky[0..kh], sy_in, 8);
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
        if (255 * rx.sum * ry.sum <= std.math.maxInt(u16)) {
            @memcpy(kx[0..kw], kxr[0..kw]);
            @memcpy(ky[0..kh], kyr[0..kh]);
            const shift: u5 = 16 - (rx.shift + ry.shift);
            // OpenCV's fixed small kernels, fully unrolled.
            const k3 = [_]u16{ 1, 2, 1 };
            const k5 = [_]u16{ 1, 4, 6, 4, 1 };
            if (kw == kh and std.mem.eql(u16, kx[0..kw], ky[0..kh])) {
                if (kw == 3 and shift == 4 and std.mem.eql(u16, kx[0..3], &k3)) {
                    const P = SpecK(u8, u16, u16, u8, u16, FinishShiftC(4), &k3, &k3);
                    return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
                }
                if (kw == 5 and shift == 8 and std.mem.eql(u16, kx[0..5], &k5)) {
                    const P = SpecK(u8, u16, u16, u8, u16, FinishShiftC(8), &k5, &k5);
                    return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
                }
            }
            const P = Spec(u8, u16, u16, u8, u16, FinishShift);
            return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, @floatFromInt(shift));
        }
        const P = Spec(u8, u16, u32, u8, u16, FinishU8Fixed);
        return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, 1);
    } else if (T == u16) {
        // Taps sum to exactly 2^16, so the vertical sums fit u32 and the
        // horizontal ones u64 without saturating.
        var kx: [127]u32 = undefined;
        var ky: [127]u32 = undefined;
        gaussianKernelFixed(u32, kx[0..kw], o.sigma_x, 16);
        gaussianKernelFixed(u32, ky[0..kh], sy_in, 16);
        // Binomial kernels (sigma 0, ksize <= 9) share a power-of-two
        // factor; divided out, the whole 2-D sum fits u32 lanes.
        var kxr: [127]u32 = undefined;
        var kyr: [127]u32 = undefined;
        @memcpy(kxr[0..kw], kx[0..kw]);
        @memcpy(kyr[0..kh], ky[0..kh]);
        const rx = reduce(kxr[0..kw]);
        const ry = reduce(kyr[0..kh]);
        if (65535 * rx.sum * ry.sum <= std.math.maxInt(u32)) {
            const shift: u6 = 32 - (@as(u6, rx.shift) + ry.shift);
            const k3 = [_]u32{ 1, 2, 1 };
            const k5 = [_]u32{ 1, 4, 6, 4, 1 };
            if (kw == kh and std.mem.eql(u32, kxr[0..kw], kyr[0..kh])) {
                if (kw == 3 and shift == 4 and std.mem.eql(u32, kxr[0..3], &k3)) {
                    const P = SpecK(u16, u32, u32, u16, u32, FinishShiftC(4), &k3, &k3);
                    return separable(P, exec, gpa, src, dst, kxr[0..kw], kyr[0..kh], o.border, 0, @floatFromInt(shift));
                }
                if (kw == 5 and shift == 8 and std.mem.eql(u32, kxr[0..5], &k5)) {
                    const P = SpecK(u16, u32, u32, u16, u32, FinishShiftC(8), &k5, &k5);
                    return separable(P, exec, gpa, src, dst, kxr[0..kw], kyr[0..kh], o.border, 0, @floatFromInt(shift));
                }
            }
            const P = Spec(u16, u32, u32, u16, u32, FinishShift);
            return separable(P, exec, gpa, src, dst, kxr[0..kw], kyr[0..kh], o.border, 0, @floatFromInt(shift));
        }
        const P = Spec(u16, u32, u64, u16, u32, FinishU16Fixed);
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
        // OpenCV's fixed small kernels, unrolled at compile time.
        if (T == f32 and kw == kh and std.mem.eql(f32, kx[0..kw], ky[0..kh])) {
            const t3 = [_]f32{ 0.25, 0.5, 0.25 };
            const t5 = [_]f32{ 0.0625, 0.25, 0.375, 0.25, 0.0625 };
            if (kw == 3 and std.mem.eql(f32, kx[0..3], &t3)) {
                const P = SpecK(T, f32, f32, T, f32, FinishCast, &t3, &t3);
                return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, 1);
            }
            if (kw == 5 and std.mem.eql(f32, kx[0..5], &t5)) {
                const P = SpecK(T, f32, f32, T, f32, FinishCast, &t5, &t5);
                return separable(P, exec, gpa, src, dst, kx[0..kw], ky[0..kh], o.border, 0, 1);
            }
        }
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
    if (@as(u64, kw) * kh <= 257) return boxChunkT(u16, s, d, kw, kh, border, scale, gpa, y0, y1);
    return boxChunkT(u32, s, d, kw, kh, border, scale, gpa, y0, y1);
}

/// Running column sums (add the entering row, subtract the leaving one)
/// in `A`, then the horizontal window sum along that one L1-resident row,
/// scaled and rounded straight into the output.
fn boxChunkT(comptime A: type, s: Image(u8), d: Image(u8), kw: u32, kh: u32, border: Border, scale: f64, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const cn: usize = s.channels;
    const n = s.rowLen();
    const rx: usize = kw / 2;
    const ry: usize = kh / 2;
    const col_buf = try gpa.alloc(A, n + 2 * rx * cn);
    defer gpa.free(col_buf);
    // Wide windows: wrapping prefix sums along the row (per channel), so a
    // window is pre[x + kw*cn] - pre[x] whatever kw. Exact because every
    // window sum fits A.
    const use_prefix = kw >= 5 and cn <= 4;
    const pre = try gpa.alloc(A, if (use_prefix) col_buf.len + cn else 0);
    defer gpa.free(pre);
    const col = col_buf[rx * cn ..][0..n];
    const N = comptime lanes(A);
    const VA = @Vector(N, A);
    const NF = comptime lanes(f32);
    const sc: @Vector(NF, f32) = @splat(@floatCast(scale));
    const scs: f32 = @floatCast(scale);
    // round(sum * scale) as ((sum * M) >> 16 + 1) >> 1, a 16-bit multiply,
    // used only if it equals the f32 path for every possible sum.
    const int_scale: ?u16 = if (A != u16) null else blk: {
        const m_f = @round(scale * 131072.0);
        if (m_f < 1 or m_f > 65535) break :blk null;
        const m: u32 = @intFromFloat(m_f);
        const max_sum: u32 = 255 * kw * kh;
        var t: u32 = 0;
        while (t <= max_sum) : (t += 1) {
            const want = sat.cast(u8, @as(f32, @floatFromInt(t)) * scs);
            const got: u32 = @min(255, (((t * m) >> 16) + 1) >> 1);
            if (got != want) break :blk null;
        }
        break :blk @intCast(m);
    };
    const srow = struct {
        fn f(img: Image(u8), v: isize, b: Border) [*]const u8 {
            return img.row(border_mod.map(v, img.height, b).?).ptr;
        }
    }.f;

    @memset(col, 0);
    const top: isize = @as(isize, @intCast(y0)) - @as(isize, @intCast(ry));
    for (0..kh) |j| {
        const r = srow(s, top + @as(isize, @intCast(j)), border);
        var x: usize = 0;
        while (x + N <= n) : (x += N) col[x..][0..N].* = @as(VA, col[x..][0..N].*) + @as(VA, @as(@Vector(N, u8), r[x..][0..N].*));
        while (x < n) : (x += 1) col[x] += r[x];
    }
    for (y0..y1) |y| {
        if (y > y0) {
            const add = srow(s, @as(isize, @intCast(y + ry)), border);
            const sub = srow(s, @as(isize, @intCast(y)) - @as(isize, @intCast(ry)) - 1, border);
            var x: usize = 0;
            while (x + N <= n) : (x += N) {
                const c: VA = col[x..][0..N].*;
                col[x..][0..N].* = c + @as(VA, @as(@Vector(N, u8), add[x..][0..N].*)) - @as(VA, @as(@Vector(N, u8), sub[x..][0..N].*));
            }
            while (x < n) : (x += 1) col[x] = col[x] + add[x] - sub[x];
        }
        for (0..rx) |i| {
            const lx = border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(rx)), s.width, border).?;
            const rxx = border_mod.map(@intCast(s.width + i), s.width, border).?;
            @memcpy(col_buf[i * cn ..][0..cn], col[lx * cn ..][0..cn]);
            @memcpy(col_buf[(rx + s.width + i) * cn ..][0..cn], col[rxx * cn ..][0..cn]);
        }
        if (use_prefix) switch (cn) {
            inline 1, 2, 3, 4 => |k| prefixRow(A, k, col_buf, pre),
            else => unreachable,
        };
        const win = struct {
            inline fn f(comptime M: usize, cb: []const A, p: []const A, pf: bool, kw_: usize, cn_: usize, x0: usize) @Vector(M, A) {
                if (pf) return @as(@Vector(M, A), p[x0 + kw_ * cn_ ..][0..M].*) -% @as(@Vector(M, A), p[x0..][0..M].*);
                var total: @Vector(M, A) = cb[x0..][0..M].*;
                for (1..kw_) |i| total += cb[x0 + i * cn_ ..][0..M].*;
                return total;
            }
        }.f;
        const out = d.row(y);
        var x: usize = 0;
        // Integer scaling when it reproduces the f32 result for every sum.
        if (A == u16 and int_scale != null) {
            const M: VA = @splat(int_scale.?);
            const one: VA = @splat(1);
            while (x + N <= n) : (x += N) {
                const sum: VA = win(N, col_buf, pre, use_prefix, kw, cn, x);
                const hi: VA = @intCast((@as(@Vector(N, u32), sum) * @as(@Vector(N, u32), M)) >> @splat(16));
                out[x..][0..N].* = @as(@Vector(N, u8), @intCast(@min((hi + one) >> @splat(1), @as(VA, @splat(255)))));
            }
        }
        // Sum in full-width lanes of A, then scale in f32 halves.
        if (N == 2 * NF) {
            while (x + N <= n) : (x += N) {
                const sum: VA = win(N, col_buf, pre, use_prefix, kw, cn, x);
                const arr: [N]A = sum;
                const lo: @Vector(NF, A) = arr[0..NF].*;
                const hi: @Vector(NF, A) = arr[NF..N].*;
                out[x..][0..NF].* = satV(u8, NF, cvtV(f32, NF, lo) * sc);
                out[x + NF ..][0..NF].* = satV(u8, NF, cvtV(f32, NF, hi) * sc);
            }
        }
        while (x + NF <= n) : (x += NF) {
            const sum = win(NF, col_buf, pre, use_prefix, kw, cn, x);
            out[x..][0..NF].* = satV(u8, NF, cvtV(f32, NF, sum) * sc);
        }
        while (x < n) : (x += 1) {
            var sum: u32 = col_buf[x];
            for (1..kw) |i| sum += col_buf[x + i * cn];
            out[x] = sat.cast(u8, @as(f32, @floatFromInt(sum)) * scs);
        }
    }
}

/// Inclusive prefix sums of `src` per channel (stride cn), wrapping, into
/// pre[cn..]; pre[0..cn] = 0, so sum(src[x + i*cn], i < k) =
/// pre[x + k*cn] - pre[x].
fn prefixRow(comptime A: type, comptime cn: usize, src: []const A, pre: []A) void {
    const N = comptime lanes(A);
    const V = @Vector(N, A);
    const zero: V = @splat(0);
    // For lane i, the lane of the previous vector holding the same channel's
    // running total (the last one before this vector).
    const carry_mask = comptime blk: {
        var m: [N]i32 = undefined;
        for (0..N) |i| m[i] = @intCast(N + i - cn * (i / cn + 1));
        break :blk m;
    };
    @memset(pre[0..cn], 0);
    var prev: V = zero;
    var x: usize = 0;
    while (x + N <= src.len) : (x += N) {
        var v: V = src[x..][0..N].*;
        comptime var k = cn;
        inline while (k < N) : (k *= 2) {
            const m = comptime blk: {
                var mm: [N]i32 = undefined;
                for (0..N) |i| mm[i] = if (i >= k) @intCast(i - k) else -1;
                break :blk mm;
            };
            v +%= @shuffle(A, v, zero, m);
        }
        v +%= @shuffle(A, prev, undefined, carry_mask);
        pre[cn + x ..][0..N].* = v;
        prev = v;
    }
    while (x < src.len) : (x += 1) pre[cn + x] = pre[x] +% src[x];
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

test "gaussian kernels match OpenCV bit for bit" {
    // cv2.getGaussianKernel(3, 0.9, CV_64F): softfloat exp, not libm's.
    var k3: [3]f64 = undefined;
    gaussianKernel(&k3, 0.9);
    try std.testing.expectEqual(@as(u64, 0x3fd09b4b1dc450d8), @as(u64, @bitCast(k3[0])));
    try std.testing.expectEqual(@as(u64, 0x3fdec969c4775e50), @as(u64, @bitCast(k3[1])));
    // Fixed point with error diffusion: taps sum to exactly 2^bits
    // (rounding each tap alone gives 1 4 10 19 31 41 45 ..., summing to 257).
    var f8: [13]u16 = undefined;
    gaussianKernelFixed(u16, &f8, 0, 8);
    try std.testing.expectEqualSlices(u16, &.{ 1, 5, 10, 19, 30, 41, 44, 41, 30, 19, 10, 5, 1 }, &f8);
    var w16: [13]u32 = undefined;
    gaussianKernelFixed(u32, &w16, 0, 16);
    try std.testing.expectEqualSlices(u32, &.{ 380, 1075, 2516, 4877, 7823, 10388, 11418, 10388, 7823, 4877, 2516, 1075, 380 }, &w16);
    var f9: [9]u16 = undefined;
    gaussianKernelFixed(u16, &f9, 0, 8);
    try std.testing.expectEqualSlices(u16, &.{ 4, 13, 30, 51, 60, 51, 30, 13, 4 }, &f9);
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

test "box filter: prefix-sum and direct windows match a naive sum" {
    const gpa = std.testing.allocator;
    inline for (.{ 1, 2, 3, 4 }) |cn| {
        var src = try Image(u8).init(gpa, 67, 23, cn);
        defer src.deinit(gpa);
        var seed: u32 = 99;
        for (src.slice()) |*v| {
            seed = seed *% 1103515245 +% 12345;
            v.* = @truncate(seed >> 16);
        }
        for ([_][2]u32{ .{ 3, 3 }, .{ 5, 5 }, .{ 7, 3 }, .{ 9, 1 }, .{ 15, 15 }, .{ 31, 31 }, .{ 1, 9 } }) |k| {
            var dst = try Image(u8).init(gpa, 67, 23, cn);
            defer dst.deinit(gpa);
            try boxFilter(.serial, gpa, src, dst, .{ .kw = k[0], .kh = k[1] });
            const scale: f32 = @floatCast(1.0 / @as(f64, @floatFromInt(k[0] * k[1])));
            for (0..23) |y| for (0..67) |x| for (0..cn) |ch| {
                var total: u32 = 0;
                for (0..k[1]) |j| for (0..k[0]) |i| {
                    const sy = border_mod.map(@as(isize, @intCast(y + j)) - @as(isize, @intCast(k[1] / 2)), 23, .reflect101).?;
                    const sx = border_mod.map(@as(isize, @intCast(x + i)) - @as(isize, @intCast(k[0] / 2)), 67, .reflect101).?;
                    total += src.row(sy)[sx * cn + ch];
                };
                try std.testing.expectEqual(sat.cast(u8, @as(f32, @floatFromInt(total)) * scale), dst.row(y)[x * cn + ch]);
            };
        }
    }
}
