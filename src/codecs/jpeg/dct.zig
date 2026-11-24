//! 8x8 integer DCTs, bit-exact with libjpeg(-turbo)'s "ISLOW" algorithms
//! (jidctint.c / jfdctint.c: Loeffler-Ligtenberg-Moschytz with 13-bit
//! constants and 2 extra bits of precision between passes). Each pass runs
//! on all 8 rows or columns at once as an 8-lane i32 vector.

const std = @import("std");
const simd = @import("../../core/simd.zig");

pub const V = @Vector(8, i32);

const const_bits = 13;
const pass1_bits = 2;

const fix_0_298631336: i32 = 2446;
const fix_0_390180644: i32 = 3196;
const fix_0_541196100: i32 = 4433;
const fix_0_765366865: i32 = 6270;
const fix_0_899976223: i32 = 7373;
const fix_1_175875602: i32 = 9633;
const fix_1_501321110: i32 = 12299;
const fix_1_847759065: i32 = 15137;
const fix_1_961570560: i32 = 16069;
const fix_2_053119869: i32 = 16819;
const fix_2_562915447: i32 = 20995;
const fix_3_072711026: i32 = 25172;

inline fn kv(comptime T: type, c: i32) T {
    return @splat(c);
}

fn transposeMask(comptime L: usize, comptime half: usize) [L]i32 {
    var m: [L]i32 = undefined;
    for (0..L) |j| {
        const g = j / 8;
        const p = j % 8;
        const src: i32 = @intCast(g * 8 + half * 4 + p / 2);
        m[j] = if (p % 2 == 0) src else ~src;
    }
    return m;
}

/// In-place transpose of the 8x8 matrix (or, with 16 lanes, the two
/// side-by-side matrices) held as 8 row vectors: three rounds of perfect
/// shuffles.
pub inline fn transpose(comptime T: type, r: *[8]T) void {
    const L = @typeInfo(T).vector.len;
    const lo = comptime transposeMask(L, 0);
    const hi = comptime transposeMask(L, 1);
    inline for (0..3) |_| {
        var t: [8]T = undefined;
        inline for (0..4) |i| {
            t[2 * i] = @shuffle(i32, r[i], r[i + 4], lo);
            t[2 * i + 1] = @shuffle(i32, r[i], r[i + 4], hi);
        }
        r.* = t;
    }
}

/// a * c where `a` fits in 16 bits: written so LLVM emits a single
/// pmaddwd instead of a 32-bit multiply.
inline fn m16(comptime T: type, a: T, comptime c: i32) T {
    const H = @Vector(@typeInfo(T).vector.len, i16);
    return @as(T, @as(H, @truncate(a))) *% kv(T, c);
}

/// The IDCT's odd-part rotation expanded into one exact linear combination
/// per output: out[o] = sum_j odd_k[o][j] * t[j], t = (x7, x5, x3, x1).
/// (Same values as computing z1..z5 first; only the order of the integer
/// additions changes, so results are identical.)
const odd_k: [4][4]i32 = blk: {
    const diag = [4]i32{ fix_0_298631336, fix_2_053119869, fix_3_072711026, fix_1_501321110 };
    // z1 = t0+t3, z2 = t1+t2, z3 = t0+t2, z4 = t1+t3, z5 = c(t0+t1+t2+t3).
    const zc = [4]i32{ -fix_0_899976223, -fix_2_562915447, -fix_1_961570560, -fix_0_390180644 };
    const members = [4][2]usize{ .{ 0, 3 }, .{ 1, 2 }, .{ 0, 2 }, .{ 1, 3 } };
    const uses = [4][2]usize{ .{ 0, 2 }, .{ 1, 3 }, .{ 1, 2 }, .{ 0, 3 } };
    var k: [4][4]i32 = undefined;
    for (0..4) |o| {
        for (0..4) |j| {
            var v: i32 = fix_1_175875602;
            if (o == j) v += diag[o];
            for (uses[o]) |z| {
                if (members[z][0] == j or members[z][1] == j) v += zc[z];
            }
            k[o][j] = v;
        }
    }
    break :blk k;
};

inline fn odd4(comptime T: type, t: [4]T) [4]T {
    var out: [4]T = undefined;
    inline for (0..4) |o| {
        out[o] = m16(T, t[0], odd_k[o][0]) +% m16(T, t[1], odd_k[o][1]) +%
            m16(T, t[2], odd_k[o][2]) +% m16(T, t[3], odd_k[o][3]);
    }
    return out;
}

/// One 1-D ISLOW IDCT; result descaled by `shift` bits. Inputs must fit
/// in 16 bits, as in libjpeg-turbo's SIMD code.
inline fn idct1d(comptime T: type, x: [8]T, comptime shift: u5) [8]T {
    // Even part: z1 = c541 (x2 + x6), tmp2 = z1 - c1847 x6, tmp3 = z1 + c765 x2.
    const tmp2 = m16(T, x[2], fix_0_541196100) +% m16(T, x[6], fix_0_541196100 - fix_1_847759065);
    const tmp3 = m16(T, x[2], fix_0_541196100 + fix_0_765366865) +% m16(T, x[6], fix_0_541196100);
    const tmp0 = (x[0] +% x[4]) << @splat(const_bits);
    const tmp1 = (x[0] -% x[4]) << @splat(const_bits);
    const tmp10 = tmp0 +% tmp3;
    const tmp13 = tmp0 -% tmp3;
    const tmp11 = tmp1 +% tmp2;
    const tmp12 = tmp1 -% tmp2;
    // Odd part.
    const od = odd4(T, .{ x[7], x[5], x[3], x[1] });
    const rnd = kv(T, 1 << (shift - 1));
    const sh: T = @splat(shift);
    return .{
        (tmp10 +% od[3] +% rnd) >> sh,
        (tmp11 +% od[2] +% rnd) >> sh,
        (tmp12 +% od[1] +% rnd) >> sh,
        (tmp13 +% od[0] +% rnd) >> sh,
        (tmp13 -% od[0] +% rnd) >> sh,
        (tmp12 -% od[1] +% rnd) >> sh,
        (tmp11 -% od[2] +% rnd) >> sh,
        (tmp10 -% od[3] +% rnd) >> sh,
    };
}

/// Inverse DCT of one block of dequantized coefficients (natural order),
/// writing 8x8 clamped samples to `out` with rows `stride` bytes apart.
pub fn idct(in: *const [64]i32, out: [*]u8, stride: usize) void {
    const T = V;
    var r: [8]T = undefined;
    inline for (0..8) |i| r[i] = in[i * 8 ..][0..8].*;
    var ws = idct1d(T, r, const_bits - pass1_bits);
    transpose(T, &ws);
    var o = idct1d(T, ws, const_bits + pass1_bits + 3);
    transpose(T, &o);
    inline for (0..8) |i| out[i * stride ..][0..8].* = toU8(8, o[i]);
}

const V16 = @Vector(16, i32);

/// Two independent blocks at once (16-lane vectors).
pub fn idct2(a: *const [64]i32, oa: [*]u8, sa: usize, b: *const [64]i32, ob: [*]u8, sb: usize) void {
    const T = V16;
    var r: [8]T = undefined;
    inline for (0..8) |i| r[i] = concat(a[i * 8 ..][0..8].*, b[i * 8 ..][0..8].*);
    var ws = idct1d(T, r, const_bits - pass1_bits);
    transpose(T, &ws);
    var o = idct1d(T, ws, const_bits + pass1_bits + 3);
    transpose(T, &o);
    inline for (0..8) |i| {
        const u: [16]u8 = toU8(16, o[i]);
        oa[i * sa ..][0..8].* = u[0..8].*;
        ob[i * sb ..][0..8].* = u[8..16].*;
    }
}

inline fn concat(a: V, b: V) V16 {
    return @shuffle(i32, a, b, blk: {
        var m: [16]i32 = undefined;
        for (0..8) |i| {
            m[i] = i;
            m[8 + i] = ~@as(i32, i);
        }
        break :blk m;
    });
}

inline fn toU8(comptime L: usize, v: @Vector(L, i32)) @Vector(L, u8) {
    const T = @Vector(L, i32);
    return simd.narrow(u8, @min(@max(v +% @as(T, @splat(128)), @as(T, @splat(0))), @as(T, @splat(255))));
}

/// IDCT of a block whose only non-zero coefficient is the (dequantized) DC.
pub inline fn idctDc(dc: i32, out: [*]u8, stride: usize) void {
    const v: u8 = @intCast(std.math.clamp(((dc +% 4) >> 3) +% 128, 0, 255));
    const row: @Vector(8, u8) = @splat(v);
    inline for (0..8) |i| out[i *% stride ..][0..8].* = row;
}

/// Forward ISLOW DCT. `in` holds level-shifted samples (x - 128), natural
/// order; `out` receives coefficients scaled up by 8, as libjpeg's do.
pub fn fdct(in: *const [64]i32, out: *[64]i32) void {
    const T = V;
    var c: [8]T = undefined;
    inline for (0..8) |i| c[i] = in[i * 8 ..][0..8].*;
    transpose(T, &c);
    var o = fdct1d(T, c, true);
    transpose(T, &o);
    const u = fdct1d(T, o, false);
    inline for (0..8) |i| out[i * 8 ..][0..8].* = u[i];
}

/// Forward DCT of two blocks of samples (8 rows each, `stride` bytes
/// apart), level shift included. `out` gets block a in lanes 0..7 of each
/// row vector and block b in lanes 8..15.
pub fn fdct2(pa: [*]const u8, sa: usize, pb: [*]const u8, sb: usize, out: *[8]V16) void {
    const T = V16;
    var c: [8]T = undefined;
    inline for (0..8) |i| {
        const ra: @Vector(8, u8) = pa[i * sa ..][0..8].*;
        const rb: @Vector(8, u8) = pb[i * sb ..][0..8].*;
        c[i] = concat(@as(V, ra), @as(V, rb)) -% @as(T, @splat(128));
    }
    transpose(T, &c);
    var o = fdct1d(T, c, true);
    transpose(T, &o);
    out.* = fdct1d(T, o, false);
}

inline fn fdct1d(comptime T: type, d: [8]T, comptime first: bool) [8]T {
    const tmp0 = d[0] +% d[7];
    var tmp7 = d[0] -% d[7];
    const tmp1 = d[1] +% d[6];
    var tmp6 = d[1] -% d[6];
    const tmp2 = d[2] +% d[5];
    var tmp5 = d[2] -% d[5];
    const tmp3 = d[3] +% d[4];
    var tmp4 = d[3] -% d[4];

    const tmp10 = tmp0 +% tmp3;
    const tmp13 = tmp0 -% tmp3;
    const tmp11 = tmp1 +% tmp2;
    const tmp12 = tmp1 -% tmp2;

    const shift: u5 = if (first) const_bits -% pass1_bits else const_bits +% pass1_bits;
    const rnd = kv(T, 1 << (shift -% 1));
    const sh: T = @splat(shift);
    var o: [8]T = undefined;
    if (first) {
        o[0] = (tmp10 +% tmp11) << @splat(pass1_bits);
        o[4] = (tmp10 -% tmp11) << @splat(pass1_bits);
    } else {
        const r2 = kv(T, 1 << (pass1_bits -% 1));
        o[0] = (tmp10 +% tmp11 +% r2) >> @splat(pass1_bits);
        o[4] = (tmp10 -% tmp11 +% r2) >> @splat(pass1_bits);
    }
    var z1 = (tmp12 +% tmp13) *% kv(T, fix_0_541196100);
    o[2] = (z1 +% tmp13 *% kv(T, fix_0_765366865) +% rnd) >> sh;
    o[6] = (z1 +% tmp12 *% kv(T, -fix_1_847759065) +% rnd) >> sh;

    z1 = tmp4 +% tmp7;
    var z2 = tmp5 +% tmp6;
    var z3 = tmp4 +% tmp6;
    var z4 = tmp5 +% tmp7;
    const z5 = (z3 +% z4) *% kv(T, fix_1_175875602);
    tmp4 *%= kv(T, fix_0_298631336);
    tmp5 *%= kv(T, fix_2_053119869);
    tmp6 *%= kv(T, fix_3_072711026);
    tmp7 *%= kv(T, fix_1_501321110);
    z1 *%= kv(T, -fix_0_899976223);
    z2 *%= kv(T, -fix_2_562915447);
    z3 *%= kv(T, -fix_1_961570560);
    z4 *%= kv(T, -fix_0_390180644);
    z3 +%= z5;
    z4 +%= z5;
    o[7] = (tmp4 +% z1 +% z3 +% rnd) >> sh;
    o[5] = (tmp5 +% z2 +% z4 +% rnd) >> sh;
    o[3] = (tmp6 +% z2 +% z3 +% rnd) >> sh;
    o[1] = (tmp7 +% z1 +% z4 +% rnd) >> sh;
    return o;
}

// ------------------------------------------------------------------------
// 16-bit forward DCT (libjpeg-turbo's NEON layout): one block, a row per
// 8-lane i16 vector, additions in i16 and products widened to i32. Same
// integer values as fdct1d: for 8-bit samples every i16 intermediate is
// bounded by 32768 in magnitude (pass-1 outputs by 4096, and each pass-2
// sum is a +-1 combination of at most eight of them, the column-0 extremes
// being -32768 and 32640), the products and their sums are exact in i32,
// and the descales round the same way.

pub const H16 = @Vector(8, i16);
const W32 = @Vector(8, i32);
pub const arm_neon = simd.llvm_intrinsics and @import("builtin").cpu.arch == .aarch64;

const neon = struct {
    extern fn @"llvm.aarch64.neon.rshrn.v4i16"(@Vector(4, i32), i32) @Vector(4, i16);
    extern fn @"llvm.aarch64.neon.srshl.v8i16"(H16, H16) H16;
};

/// (x + 2^(s-1)) >> s, narrowed to i16 (the result fits).
inline fn descale(x: W32, comptime s: u5) H16 {
    if (comptime arm_neon) {
        const a: [8]i32 = x;
        return std.simd.join(neon.@"llvm.aarch64.neon.rshrn.v4i16"(a[0..4].*, s), neon.@"llvm.aarch64.neon.rshrn.v4i16"(a[4..8].*, s));
    }
    return @truncate((x +% kv(W32, 1 << (s - 1))) >> @splat(s));
}

/// (x + 2) >> 2 in i16 without overflow of the addition.
inline fn rshr2(x: H16) H16 {
    if (comptime arm_neon) return neon.@"llvm.aarch64.neon.srshl.v8i16"(x, @splat(-2));
    return @truncate((@as(W32, x) + kv(W32, 2)) >> @splat(2));
}

inline fn wide(x: H16) W32 {
    return x;
}

inline fn fdct1d16(d: [8]H16, comptime first: bool) [8]H16 {
    const tmp0 = d[0] +% d[7];
    const tmp7 = d[0] -% d[7];
    const tmp1 = d[1] +% d[6];
    const tmp6 = d[1] -% d[6];
    const tmp2 = d[2] +% d[5];
    const tmp5 = d[2] -% d[5];
    const tmp3 = d[3] +% d[4];
    const tmp4 = d[3] -% d[4];

    const tmp10 = tmp0 +% tmp3;
    const tmp13 = tmp0 -% tmp3;
    const tmp11 = tmp1 +% tmp2;
    const tmp12 = tmp1 -% tmp2;

    const sh: u5 = if (first) const_bits - pass1_bits else const_bits + pass1_bits;
    var o: [8]H16 = undefined;
    if (first) {
        o[0] = (tmp10 +% tmp11) << @splat(pass1_bits);
        o[4] = (tmp10 -% tmp11) << @splat(pass1_bits);
    } else {
        o[0] = rshr2(tmp10 +% tmp11);
        o[4] = rshr2(tmp10 -% tmp11);
    }
    const z1 = wide(tmp12 +% tmp13) * kv(W32, fix_0_541196100);
    o[2] = descale(z1 + wide(tmp13) * kv(W32, fix_0_765366865), sh);
    o[6] = descale(z1 + wide(tmp12) * kv(W32, -fix_1_847759065), sh);

    const z5 = wide(tmp4 +% tmp6 +% tmp5 +% tmp7) * kv(W32, fix_1_175875602);
    const q1 = wide(tmp4 +% tmp7) * kv(W32, -fix_0_899976223);
    const q2 = wide(tmp5 +% tmp6) * kv(W32, -fix_2_562915447);
    const q3 = wide(tmp4 +% tmp6) * kv(W32, -fix_1_961570560) + z5;
    const q4 = wide(tmp5 +% tmp7) * kv(W32, -fix_0_390180644) + z5;
    o[7] = descale(wide(tmp4) * kv(W32, fix_0_298631336) + q1 + q3, sh);
    o[5] = descale(wide(tmp5) * kv(W32, fix_2_053119869) + q2 + q4, sh);
    o[3] = descale(wide(tmp6) * kv(W32, fix_3_072711026) + q2 + q3, sh);
    o[1] = descale(wide(tmp7) * kv(W32, fix_1_501321110) + q1 + q4, sh);
    return o;
}

inline fn trnMask(comptime n: usize, comptime odd: usize) [n]i32 {
    var m: [n]i32 = undefined;
    for (0..n / 2) |i| {
        m[2 * i] = @intCast(2 * i + odd);
        m[2 * i + 1] = ~@as(i32, @intCast(2 * i + odd));
    }
    return m;
}

/// trn1 / trn2 on elements of `E` (16, 32 or 64 bits).
inline fn trn(comptime E: type, a: H16, b: H16, comptime odd: usize) H16 {
    const n = 128 / @bitSizeOf(E);
    const T = @Vector(n, E);
    const x: T = @bitCast(a);
    const y: T = @bitCast(b);
    return @bitCast(@shuffle(E, x, y, trnMask(n, odd)));
}

/// 8x8 i16 transpose: trn at 16, 32 and 64 bits.
pub inline fn transpose16(r: *[8]H16) void {
    var x: [8]H16 = undefined;
    inline for (0..4) |i| {
        x[2 * i] = trn(i16, r[2 * i], r[2 * i + 1], 0);
        x[2 * i + 1] = trn(i16, r[2 * i], r[2 * i + 1], 1);
    }
    var y: [8]H16 = undefined;
    inline for (0..2) |h| inline for (0..2) |k| {
        y[4 * h + k] = trn(i32, x[4 * h + k], x[4 * h + k + 2], 0);
        y[4 * h + k + 2] = trn(i32, x[4 * h + k], x[4 * h + k + 2], 1);
    };
    inline for (0..4) |k| {
        r[k] = trn(i64, y[k], y[k + 4], 0);
        r[k + 4] = trn(i64, y[k], y[k + 4], 1);
    }
}

/// Forward ISLOW DCT of one block of samples (8 rows `stride` bytes
/// apart), level shift included: coefficient rows in natural order, the
/// same values as fdct.
pub inline fn fdct16(p: [*]const u8, stride: usize) [8]H16 {
    var c: [8]H16 = undefined;
    inline for (0..8) |i| {
        const r: @Vector(8, u8) = p[i * stride ..][0..8].*;
        c[i] = @as(H16, @intCast(@as(@Vector(8, u16), r))) -% @as(H16, @splat(128));
    }
    transpose16(&c);
    var o = fdct1d16(c, true);
    transpose16(&o);
    return fdct1d16(o, false);
}

test "jpeg fdct16 matches fdct" {
    var seed: u32 = 7;
    for (0..3000) |it| {
        var px: [64]u8 = undefined;
        for (&px, 0..) |*v, i| {
            seed = seed *% 1664525 +% 1013904223;
            v.* = switch (it % 6) {
                0 => 0,
                1 => 255,
                2 => if ((i + i / 8) % 2 == 0) 0 else 255, // checkerboard
                3 => if ((seed >> 24) & 1 == 0) 0 else 255,
                else => @truncate(seed >> 24),
            };
        }
        var in: [64]i32 = undefined;
        for (&in, px) |*v, s| v.* = @as(i32, s) - 128;
        var want: [64]i32 = undefined;
        fdct(&in, &want);
        const got = fdct16(&px, 8);
        for (0..8) |r| for (0..8) |k| try std.testing.expectEqual(want[r * 8 + k], @as(i32, @as([8]i16, got[r])[k]));
    }
    // Extreme sign patterns: every row and column of +-1 signs.
    for (0..256) |a| for (0..256) |b| {
        var px: [64]u8 = undefined;
        for (0..8) |y| for (0..8) |x| {
            const s = ((a >> @intCast(y)) ^ (b >> @intCast(x))) & 1;
            px[y * 8 + x] = if (s == 0) 0 else 255;
        };
        var in: [64]i32 = undefined;
        for (&in, px) |*v, s| v.* = @as(i32, s) - 128;
        var want: [64]i32 = undefined;
        fdct(&in, &want);
        const got = fdct16(&px, 8);
        for (0..8) |r| for (0..8) |k| try std.testing.expectEqual(want[r * 8 + k], @as(i32, @as([8]i16, got[r])[k]));
    };
}

test "jpeg dct transpose" {
    var r: [8]V = undefined;
    for (0..8) |i| {
        var a: [8]i32 = undefined;
        for (0..8) |j| a[j] = @intCast(i * 8 + j);
        r[i] = a;
    }
    transpose(V, &r);
    for (0..8) |i| {
        const a: [8]i32 = r[i];
        for (0..8) |j| try std.testing.expectEqual(@as(i32, @intCast(j * 8 + i)), a[j]);
    }
}

test "jpeg dct round trip" {
    var px: [64]i32 = undefined;
    for (&px, 0..) |*p, i| p.* = @as(i32, @intCast((i * 37 + i / 8 * 11) % 200)) - 100;
    var coef: [64]i32 = undefined;
    fdct(&px, &coef);
    // Undo the factor of 8 the forward DCT leaves in.
    for (&coef) |*c| c.* = @divFloor(c.* + 4, 8);
    var out: [64]u8 = undefined;
    idct(&coef, &out, 8);
    for (px, out) |a, b| try std.testing.expect(@abs(a + 128 - @as(i32, b)) <= 1);
}
