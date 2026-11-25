//! 8-bit CIE L*a*b* conversions of linear (not gamma-encoded) RGB, bit-exact
//! with OpenCV's integer paths (cvtColor COLOR_LBGR2Lab / COLOR_Lab2LBGR,
//! RGB2Lab_b and Lab2RGBinteger in color_lab.cpp). OpenCV builds its tables
//! with softfloat; the same IEEE operations are replayed here at compile
//! time (its cube root included).

const std = @import("std");

const lab_shift = 12; // xyz_shift
const gamma_shift = 3;
const lab_shift2 = lab_shift + gamma_shift;
const cbrt_tab_size = 256 * 3 / 2 * (1 << gamma_shift);
const lut_base = 1 << 14;
const inv_gamma_shift = 12;

const sRGB2XYZ_D65 = [9]u64{
    0x3fda65a14488c60d, 0x3fd6e297396d0918, 0x3fc71819d2391d58,
    0x3fcb38cda6e75ff6, 0x3fe6e297396d0918, 0x3fb279aae6c8f755,
    0x3f93cc4ac6cdaf4b, 0x3fbe836eb4e98138, 0x3fee68427418d691,
};
const XYZ2sRGB_D65 = [9]u64{
    0x4009ec804102ff8f, 0xbff8982a9930be0e, 0xbfdfe7ff583a53b9,
    0xbfef042528ae74f3, 0x3ffe040f23897204, 0x3fa546d3f9e7b80b,
    0x3fac7de5082cf52c, 0xbfca1e14bdfd2631, 0x3ff0eabef06b3786,
};
const D65 = [3]u64{ 0x3fee6a22b3892ee8, 0x3ff0000000000000, 0x3ff16b8950763a19 };

fn d(bits: u64) f64 {
    return @bitCast(bits);
}

fn roundEven(x: anytype) i32 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return @intFromFloat(2.0 * @round(x / 2.0));
    return @intFromFloat(r);
}

/// OpenCV's softfloat f32_cbrt.
fn cbrt32(x: f32) f32 {
    const xb: u32 = @bitCast(x);
    if (xb & 0x7fffffff == 0) return 0;
    const s: u32 = xb >> 31;
    var ex: i32 = @as(i32, @intCast((xb >> 23) & 0xff)) - 127;
    var shx: i32 = @rem(ex, 3);
    if (shx >= 0) shx -= 3;
    ex = @divTrunc(ex - shx, 3) - 1;
    var fr: f64 = @bitCast((@as(u64, @intCast(shx + 1023)) << 52) | (@as(u64, xb & 0x7fffff) << 29));
    const A1 = d(0x4046a09e6653ba70);
    const A2 = d(0x406808f46c6116e0);
    const A3 = d(0x405dca97439cae14);
    const A4 = d(0x402add70d2827500);
    const A5 = d(0x3fc4f15f83f55d2d);
    const A6 = d(0x402d9e20660edb21);
    const A7 = d(0x4062ff15c0285815);
    const A8 = d(0x406510d06a8112ce);
    const A9 = d(0x4040fecbc9e2c375);
    fr = ((((A1 * fr + A2) * fr + A3) * fr + A4) * fr + A5) /
        ((((A6 * fr + A7) * fr + A8) * fr + A9) * fr + 1.0);
    const frb: u64 = @bitCast(fr);
    const frac: u32 = @intCast((frb & 0xfffffffffffff) >> 29);
    return @bitCast((s << 31) +% (@as(u32, @bitCast(ex + 127)) << 23) +% frac);
}

const cbrt_tab: [cbrt_tab_size]u16 = blk: {
    @setEvalBranchQuota(1_000_000);
    var t: [cbrt_tab_size]u16 = undefined;
    const lthresh: f32 = @as(f32, 216.0) / @as(f32, 24389.0);
    const lscale: f32 = @as(f32, 841.0) / @as(f32, 108.0);
    const lbias: f32 = @as(f32, 16.0) / @as(f32, 116.0);
    const scale: f32 = @as(f32, 1.0) / (@as(f32, 255.0) * @as(f32, 1 << gamma_shift));
    const lshift2: f32 = 1 << lab_shift2;
    for (0..cbrt_tab_size) |i| {
        const x: f32 = scale * @as(f32, @floatFromInt(i));
        const f = if (x < lthresh) @mulAdd(f32, x, lscale, lbias) else cbrt32(x);
        t[i] = @intCast(roundEven(lshift2 * f));
    }
    break :blk t;
};

/// Coefficients of RGB2Lab_b for blueIdx 0 (BGR input order).
const fwd_coeffs: [9]i32 = blk: {
    var c: [9]i32 = undefined;
    const lshift: f64 = 1 << lab_shift;
    for (0..3) |i| {
        const wp = d(D65[i]);
        c[i * 3 + 2] = roundEven(lshift * d(sRGB2XYZ_D65[i * 3]) / wp);
        c[i * 3 + 1] = roundEven(lshift * d(sRGB2XYZ_D65[i * 3 + 1]) / wp);
        c[i * 3 + 0] = roundEven(lshift * d(sRGB2XYZ_D65[i * 3 + 2]) / wp);
    }
    break :blk c;
};

/// Coefficients of Lab2RGBinteger for blueIdx 0 (rows R, G, B).
const inv_coeffs: [9]i32 = blk: {
    var c: [9]i32 = undefined;
    const lshift: f64 = 1 << lab_shift;
    for (0..3) |i| {
        const wp = d(D65[i]);
        for (0..3) |j| c[i + j * 3] = roundEven(lshift * d(XYZ2sRGB_D65[i + j * 3]) * wp);
    }
    break :blk c;
};

/// LabToYF_b: y and ify per L.
const lab_to_yf: [256][2]i32 = blk: {
    @setEvalBranchQuota(100_000);
    var t: [256][2]i32 = undefined;
    for (0..256) |i| {
        const ii: i32 = @intCast(i);
        if (i <= 20) {
            t[i][0] = roundEven(@as(f32, @floatFromInt(ii * lut_base * 20 * 9)) / @as(f32, 17 * 29 * 29 * 29));
            t[i][1] = roundEven(@as(f32, lut_base) * (@as(f32, 16.0) / @as(f32, 116.0) + @as(f32, @floatFromInt(ii * 5)) / @as(f32, 3 * 17 * 29)));
        } else {
            const fy: f32 = @as(f32, @floatFromInt(ii * 100 * lut_base)) / @as(f32, 255 * 116) +
                @as(f32, 16 * lut_base) / @as(f32, 116);
            t[i][1] = roundEven(fy);
            t[i][0] = roundEven(fy * fy * fy / @as(f32, lut_base * lut_base));
        }
    }
    break :blk t;
};

inline fn descale(x: i32, comptime n: u5) i32 {
    return (x + (1 << (n - 1))) >> n;
}

inline fn sat8(x: i32) u8 {
    return @intCast(std.math.clamp(x, 0, 255));
}

/// One linear BGR pixel to 8-bit Lab (L scaled to 0..255, a and b offset
/// by 128), as cvtColor COLOR_LBGR2Lab.
pub fn lbgrToLab(b: u8, g: u8, r: u8) [3]u8 {
    const c = fwd_coeffs;
    const B: i32 = @as(i32, b) << gamma_shift;
    const G: i32 = @as(i32, g) << gamma_shift;
    const R: i32 = @as(i32, r) << gamma_shift;
    const fX: i32 = cbrt_tab[@intCast(descale(B * c[0] + G * c[1] + R * c[2], lab_shift))];
    const fY: i32 = cbrt_tab[@intCast(descale(B * c[3] + G * c[4] + R * c[5], lab_shift))];
    const fZ: i32 = cbrt_tab[@intCast(descale(B * c[6] + G * c[7] + R * c[8], lab_shift))];
    const Lscale = (116 * 255 + 50) / 100;
    const Lshift = -@divTrunc(16 * 255 * (1 << lab_shift2) + 50, 100);
    const L = descale(Lscale * fY + Lshift, lab_shift2);
    const A = descale(500 * (fX - fY) + 128 * (1 << lab_shift2), lab_shift2);
    const Bv = descale(200 * (fY - fZ) + 128 * (1 << lab_shift2), lab_shift2);
    return .{ sat8(L), sat8(A), sat8(Bv) };
}

inline fn abToXZ(i: i32) i32 {
    if (i <= 3390) return @divTrunc(i * 108, 841) - @divTrunc(@divTrunc(lut_base * 16, 116) * 108, 841);
    return @divTrunc(@divTrunc(i * i, lut_base) * i, lut_base);
}

/// One 8-bit Lab pixel to linear BGR, as cvtColor COLOR_Lab2LBGR.
pub fn labToLbgr(ll: u8, aa: u8, bb: u8) [3]u8 {
    const c = inv_coeffs;
    const y = lab_to_yf[ll][0];
    const ify = lab_to_yf[ll][1];
    const base = 1 << 14;
    const adiv = ((5 * @as(i32, aa) * 53687 + (1 << 7)) >> 13) - @divTrunc(128 * base, 500);
    const bdiv = ((@as(i32, bb) * 41943 + (1 << 4)) >> 9) - @divTrunc(128 * base, 200) + 1;
    const x = abToXZ(ify + adiv);
    const z = abToXZ(ify - bdiv);
    const shift = lab_shift + (14 - inv_gamma_shift);
    var rgb: [3]i32 = undefined;
    for (0..3) |k| {
        const v = descale(c[k * 3] *% x +% c[k * 3 + 1] *% y +% c[k * 3 + 2] *% z, shift);
        const q = std.math.clamp(v, 0, (1 << inv_gamma_shift) - 1);
        rgb[k] = ((q << 8) - q) >> inv_gamma_shift;
    }
    return .{ sat8(rgb[2]), sat8(rgb[1]), sat8(rgb[0]) };
}

test "lab round trip of grays" {
    for ([_]u8{ 0, 1, 50, 128, 200, 255 }) |v| {
        const l = lbgrToLab(v, v, v);
        try std.testing.expect(@abs(@as(i32, l[1]) - 128) <= 1);
        try std.testing.expect(@abs(@as(i32, l[2]) - 128) <= 1);
    }
}
