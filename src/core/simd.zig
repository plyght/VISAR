//! Small explicit-SIMD helpers shared by the element-wise kernels. Vectors are
//! sized to one 64-byte register's worth of elements; on narrower machines
//! LLVM splits them, which still beats the scalar loops it fails to vectorise.

const std = @import("std");

/// Elements of `T` per vector (64 bytes' worth).
pub fn lanes(comptime T: type) comptime_int {
    return @max(1, 64 / @sizeOf(T));
}

pub fn Vec(comptime T: type) type {
    return @Vector(lanes(T), T);
}

pub inline fn load(comptime V: type, s: []const @typeInfo(V).vector.child, i: usize) V {
    const n = @typeInfo(V).vector.len;
    return s[i..][0..n].*;
}

pub inline fn store(comptime V: type, d: []@typeInfo(V).vector.child, i: usize, v: V) void {
    const n = @typeInfo(V).vector.len;
    d[i..][0..n].* = v;
}

pub inline fn splat(comptime V: type, x: @typeInfo(V).vector.child) V {
    return @splat(x);
}

/// Round half to even for f32 values already clamped to |x| < 2^22.
const arm_cvt = @import("builtin").zig_backend == .stage2_llvm and @import("builtin").cpu.arch == .aarch64;
extern fn @"llvm.aarch64.neon.fcvtns.v4i32.v4f32"(@Vector(4, f32)) @Vector(4, i32);
const x86_llvm = @import("builtin").zig_backend == .stage2_llvm and @import("builtin").cpu.arch == .x86_64;
const x86_avx_cvt = x86_llvm and std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx);
const x86_sse_cvt = x86_llvm and std.Target.x86.featureSetHas(@import("builtin").cpu.features, .sse2);
extern fn @"llvm.x86.avx.cvt.ps2dq.256"(@Vector(8, f32)) @Vector(8, i32);
extern fn @"llvm.x86.sse2.cvtps2dq"(@Vector(4, f32)) @Vector(4, i32);
extern fn @"llvm.aarch64.neon.sqxtun.v4i16"(@Vector(4, i32)) @Vector(4, u16);
extern fn @"llvm.aarch64.neon.uqxtn.v8i8"(@Vector(8, u16)) @Vector(8, u8);

pub inline fn rintSmall(v: anytype) @TypeOf(v) {
    const m: @TypeOf(v) = @splat(12582912.0);
    return (v + m) - m;
}

/// Rounds f32 lanes (|v| < 2^22) half to even and converts to i32: one
/// fcvtns per 4 lanes on aarch64, the magic-number rint elsewhere.
pub inline fn roundI32(comptime n: comptime_int, v: @Vector(n, f32)) @Vector(n, i32) {
    if (comptime arm_cvt and n % 4 == 0) {
        const fv: [n]f32 = v;
        var r: [n]i32 = undefined;
        inline for (0..n / 4) |k| r[k * 4 ..][0..4].* = @"llvm.aarch64.neon.fcvtns.v4i32.v4f32"(fv[k * 4 ..][0..4].*);
        return r;
    }
    // x86: cvtps2dq rounds with MXCSR's default mode, half to even.
    if (comptime x86_avx_cvt and n % 8 == 0) {
        const fv: [n]f32 = v;
        var r: [n]i32 = undefined;
        inline for (0..n / 8) |k| r[k * 8 ..][0..8].* = @"llvm.x86.avx.cvt.ps2dq.256"(fv[k * 8 ..][0..8].*);
        return r;
    }
    if (comptime x86_sse_cvt and n % 4 == 0) {
        const fv: [n]f32 = v;
        var r: [n]i32 = undefined;
        inline for (0..n / 4) |k| r[k * 4 ..][0..4].* = @"llvm.x86.sse2.cvtps2dq"(fv[k * 4 ..][0..4].*);
        return r;
    }
    return @intFromFloat(rintSmall(v));
}

/// Converts f32 lanes to the integer type `D` with OpenCV's saturate_cast
/// semantics (round half to even, clamp, NaN -> 0). `D` must be at most
/// 16 bits wide so the clamped range is exactly representable.
pub inline fn satF32(comptime D: type, comptime n: comptime_int, v: @Vector(n, f32)) @Vector(n, D) {
    if (comptime arm_cvt and n % 4 == 0) {
        // fcvtns rounds half to even and saturates (NaN -> 0) in one step;
        // the clamp to D then lowers to saturating narrows.
        const fv: [n]f32 = v;
        var r: [n]i32 = undefined;
        inline for (0..n / 4) |k| r[k * 4 ..][0..4].* = @"llvm.aarch64.neon.fcvtns.v4i32.v4f32"(fv[k * 4 ..][0..4].*);
        if (D == u8 and n % 8 == 0) {
            // sqxtun (i32 -> u16, saturating) then uqxtn (u16 -> u8).
            var o: [n]u8 = undefined;
            inline for (0..n / 8) |k| {
                const a = @"llvm.aarch64.neon.sqxtun.v4i16"(r[k * 8 ..][0..4].*);
                const b = @"llvm.aarch64.neon.sqxtun.v4i16"(r[k * 8 + 4 ..][0..4].*);
                o[k * 8 ..][0..8].* = @"llvm.aarch64.neon.uqxtn.v8i8"(std.simd.join(a, b));
            }
            return o;
        }
        const lo: @Vector(n, i32) = @splat(std.math.minInt(D));
        const hi: @Vector(n, i32) = @splat(std.math.maxInt(D));
        return narrow(D, @min(@max(@as(@Vector(n, i32), r), lo), hi));
    }
    const lo: @Vector(n, f32) = @splat(@floatFromInt(std.math.minInt(D)));
    const hi: @Vector(n, f32) = @splat(@floatFromInt(std.math.maxInt(D)));
    // @max/@min return the non-NaN operand, so NaN clamps to the low bound
    // first; fix it to 0 explicitly when 0 is not the low bound.
    var c = @min(@max(v, lo), hi);
    if (std.math.minInt(D) != 0) c = @select(f32, v == v, c, @as(@Vector(n, f32), @splat(0)));
    // The clamped value converts exactly as i32. Unsigned results are then
    // narrowed from u32 (a free bitcast): LLVM (Zig 0.17, AVX2 without
    // AVX-512) can lower i32 -> u8 to signed packs, turning 150 into 127.
    const r = roundI32(n, c);
    if (std.math.minInt(D) == 0) {
        const u: @Vector(n, u32) = @bitCast(r);
        return @intCast(u);
    }
    return @intCast(r);
}

/// Narrows integer lanes that are already within `D`'s range. For unsigned
/// `D` the value goes through the unsigned type of the source width: LLVM
/// (Zig 0.17, AVX2 without AVX-512) can lower a signed -> u8 narrowing to
/// signed packs, turning 150 into 127.
pub inline fn narrow(comptime D: type, v: anytype) @Vector(@typeInfo(@TypeOf(v)).vector.len, D) {
    const n = @typeInfo(@TypeOf(v)).vector.len;
    const S = @typeInfo(@TypeOf(v)).vector.child;
    if (@typeInfo(D).int.signedness == .unsigned and @typeInfo(S).int.signedness == .signed) {
        const U = @Int(.unsigned, @bitSizeOf(S));
        const u: @Vector(n, U) = @bitCast(v);
        return @intCast(u);
    }
    return @intCast(v);
}

/// Widens integer or float lanes to f32.
pub inline fn toF32(comptime n: comptime_int, v: anytype) @Vector(n, f32) {
    const C = @typeInfo(@TypeOf(v)).vector.child;
    if (@typeInfo(C) == .float) return @floatCast(v);
    if (@bitSizeOf(C) < 32) {
        const w: @Vector(n, i32) = v;
        return @floatFromInt(w);
    }
    return @floatFromInt(v);
}

test "satF32" {
    const v: @Vector(8, f32) = .{ -1, 0.5, 1.5, 2.5, 254.5, 255.5, 300, std.math.nan(f32) };
    const r = satF32(u8, 8, v);
    try std.testing.expectEqual(@Vector(8, u8){ 0, 0, 2, 2, 254, 255, 255, 0 }, r);
    const s = satF32(i16, 8, v * @as(@Vector(8, f32), @splat(-200)));
    try std.testing.expectEqual(@Vector(8, i16){ 200, -100, -300, -500, -32768, -32768, -32768, 0 }, s);
}

test "narrow via unsigned (AVX2 miscompile regression)" {
    // 32 lanes of f32 -> i32 -> u8 is the shape LLVM got wrong.
    var x: @Vector(32, f32) = @splat(150);
    _ = &x;
    const r = satF32(u8, 32, x);
    const arr: [32]u8 = r;
    for (arr) |v| try std.testing.expectEqual(@as(u8, 150), v);
    var w: @Vector(32, i32) = @splat(200);
    _ = &w;
    const q: [32]u8 = narrow(u8, w);
    for (q) |v| try std.testing.expectEqual(@as(u8, 200), v);
}
