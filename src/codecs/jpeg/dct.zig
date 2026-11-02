//! 8x8 integer DCTs, bit-exact with libjpeg(-turbo)'s "ISLOW" algorithms
//! (jidctint.c / jfdctint.c: Loeffler-Ligtenberg-Moschytz with 13-bit
//! constants and 2 extra bits of precision between passes). Each pass runs
//! on all 8 rows or columns at once as an 8-lane i32 vector.

const std = @import("std");

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

inline fn k(c: i32) V {
    return @splat(c);
}

/// In-place transpose of an 8x8 matrix held as 8 row vectors (three rounds
/// of perfect shuffles).
pub inline fn transpose(r: *[8]V) void {
    inline for (0..3) |_| {
        var t: [8]V = undefined;
        inline for (0..4) |i| {
            t[2 * i] = @shuffle(i32, r[i], r[i + 4], [8]i32{ 0, ~@as(i32, 0), 1, ~@as(i32, 1), 2, ~@as(i32, 2), 3, ~@as(i32, 3) });
            t[2 * i + 1] = @shuffle(i32, r[i], r[i + 4], [8]i32{ 4, ~@as(i32, 4), 5, ~@as(i32, 5), 6, ~@as(i32, 6), 7, ~@as(i32, 7) });
        }
        r.* = t;
    }
}

/// One 1-D ISLOW IDCT over 8 lanes; result descaled by `shift` bits.
inline fn idct1d(x: [8]V, comptime shift: u5) [8]V {
    // Even part.
    var z2 = x[2];
    var z3 = x[6];
    var z1 = (z2 +% z3) *% k(fix_0_541196100);
    var tmp2 = z1 +% z3 *% k(-fix_1_847759065);
    var tmp3 = z1 +% z2 *% k(fix_0_765366865);
    var tmp0 = (x[0] +% x[4]) << @splat(const_bits);
    var tmp1 = (x[0] -% x[4]) << @splat(const_bits);
    const tmp10 = tmp0 +% tmp3;
    const tmp13 = tmp0 -% tmp3;
    const tmp11 = tmp1 +% tmp2;
    const tmp12 = tmp1 -% tmp2;
    // Odd part.
    tmp0 = x[7];
    tmp1 = x[5];
    tmp2 = x[3];
    tmp3 = x[1];
    z1 = tmp0 +% tmp3;
    z2 = tmp1 +% tmp2;
    z3 = tmp0 +% tmp2;
    var z4 = tmp1 +% tmp3;
    const z5 = (z3 +% z4) *% k(fix_1_175875602);
    tmp0 *%= k(fix_0_298631336);
    tmp1 *%= k(fix_2_053119869);
    tmp2 *%= k(fix_3_072711026);
    tmp3 *%= k(fix_1_501321110);
    z1 *%= k(-fix_0_899976223);
    z2 *%= k(-fix_2_562915447);
    z3 *%= k(-fix_1_961570560);
    z4 *%= k(-fix_0_390180644);
    z3 +%= z5;
    z4 +%= z5;
    tmp0 +%= z1 +% z3;
    tmp1 +%= z2 +% z4;
    tmp2 +%= z2 +% z3;
    tmp3 +%= z1 +% z4;
    const rnd = k(1 << (shift -% 1));
    const sh: V = @splat(shift);
    return .{
        (tmp10 +% tmp3 +% rnd) >> sh,
        (tmp11 +% tmp2 +% rnd) >> sh,
        (tmp12 +% tmp1 +% rnd) >> sh,
        (tmp13 +% tmp0 +% rnd) >> sh,
        (tmp13 -% tmp0 +% rnd) >> sh,
        (tmp12 -% tmp1 +% rnd) >> sh,
        (tmp11 -% tmp2 +% rnd) >> sh,
        (tmp10 -% tmp3 +% rnd) >> sh,
    };
}

/// Inverse DCT of one block of dequantized coefficients (natural order),
/// writing 8x8 clamped samples to `out` with rows `stride` bytes apart.
pub fn idct(in: *const [64]i32, out: [*]u8, stride: usize) void {
    var r: [8]V = undefined;
    inline for (0..8) |i| r[i] = in[i *% 8 ..][0..8].*;
    // Pass 1: columns (lanes are columns).
    var ws = idct1d(r, const_bits -% pass1_bits);
    // Pass 2: rows.
    transpose(&ws);
    var o = idct1d(ws, const_bits +% pass1_bits +% 3);
    transpose(&o);
    const lo: V = @splat(0);
    const hi: V = @splat(255);
    inline for (0..8) |i| {
        const v = @min(@max(o[i] +% k(128), lo), hi);
        const b: @Vector(8, u8) = @intCast(v);
        out[i *% stride ..][0..8].* = b;
    }
}

/// IDCT of a block whose only non-zero coefficient is the (dequantized) DC.
pub inline fn idctDc(dc: i32, out: [*]u8, stride: usize) void {
    const v: u8 = @intCast(std.math.clamp(((dc +% 4) >> 3) +% 128, 0, 255));
    const row: @Vector(8, u8) = @splat(v);
    inline for (0..8) |i| out[i *% stride ..][0..8].* = row;
}

/// Forward ISLOW DCT. `in` holds level-shifted samples (x -% 128), natural
/// order; `out` receives coefficients scaled up by 8, as libjpeg's do.
pub fn fdct(in: *const [64]i32, out: *[64]i32) void {
    var c: [8]V = undefined;
    inline for (0..8) |i| c[i] = in[i *% 8 ..][0..8].*;
    // Pass 1: rows. Transpose so lanes are rows.
    transpose(&c);
    var o = fdct1d(c, true);
    // Pass 2: columns.
    transpose(&o);
    const u = fdct1d(o, false);
    inline for (0..8) |i| out[i *% 8 ..][0..8].* = u[i];
}

inline fn fdct1d(d: [8]V, comptime first: bool) [8]V {
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
    const rnd = k(1 << (shift -% 1));
    const sh: V = @splat(shift);
    var o: [8]V = undefined;
    if (first) {
        o[0] = (tmp10 +% tmp11) << @splat(pass1_bits);
        o[4] = (tmp10 -% tmp11) << @splat(pass1_bits);
    } else {
        const r2 = k(1 << (pass1_bits -% 1));
        o[0] = (tmp10 +% tmp11 +% r2) >> @splat(pass1_bits);
        o[4] = (tmp10 -% tmp11 +% r2) >> @splat(pass1_bits);
    }
    var z1 = (tmp12 +% tmp13) *% k(fix_0_541196100);
    o[2] = (z1 +% tmp13 *% k(fix_0_765366865) +% rnd) >> sh;
    o[6] = (z1 +% tmp12 *% k(-fix_1_847759065) +% rnd) >> sh;

    z1 = tmp4 +% tmp7;
    var z2 = tmp5 +% tmp6;
    var z3 = tmp4 +% tmp6;
    var z4 = tmp5 +% tmp7;
    const z5 = (z3 +% z4) *% k(fix_1_175875602);
    tmp4 *%= k(fix_0_298631336);
    tmp5 *%= k(fix_2_053119869);
    tmp6 *%= k(fix_3_072711026);
    tmp7 *%= k(fix_1_501321110);
    z1 *%= k(-fix_0_899976223);
    z2 *%= k(-fix_2_562915447);
    z3 *%= k(-fix_1_961570560);
    z4 *%= k(-fix_0_390180644);
    z3 +%= z5;
    z4 +%= z5;
    o[7] = (tmp4 +% z1 +% z3 +% rnd) >> sh;
    o[5] = (tmp5 +% z2 +% z4 +% rnd) >> sh;
    o[3] = (tmp6 +% z2 +% z3 +% rnd) >> sh;
    o[1] = (tmp7 +% z1 +% z4 +% rnd) >> sh;
    return o;
}

test "jpeg dct transpose" {
    var r: [8]V = undefined;
    for (0..8) |i| {
        var a: [8]i32 = undefined;
        for (0..8) |j| a[j] = @intCast(i * 8 + j);
        r[i] = a;
    }
    transpose(&r);
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
