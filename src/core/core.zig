//! visar.core: containers, execution and numeric helpers shared by every module.

const std = @import("std");
const builtin = @import("builtin");

pub const image = @import("image.zig");
pub const pool = @import("pool.zig");
pub const sat = @import("saturate.zig");
pub const arith = @import("arith.zig");
pub const vec = @import("vec.zig");
pub const fft = @import("fft.zig");
pub const simd = @import("simd.zig");

pub const Image = image.Image;
pub const AnyImage = image.AnyImage;
pub const Depth = image.Depth;
pub const Error = image.Error;
pub const Exec = pool.Exec;
pub const Pool = pool.Pool;

/// SIMD lanes VISAR uses for `T` on the compile target (1 when the target has
/// no vector unit).
pub fn lanes(comptime T: type) comptime_int {
    return std.simd.suggestVectorLength(T) orelse 1;
}

/// A short description of the compile target, e.g. "x86_64-linux (avx2 fma)".
pub const target_description: []const u8 = desc: {
    const cpu = builtin.cpu;
    const feats = switch (cpu.arch) {
        .x86_64 => blk: {
            const x = std.Target.x86.Feature;
            if (std.Target.x86.featureSetHas(cpu.features, x.avx512bw)) break :blk "avx512";
            if (std.Target.x86.featureSetHas(cpu.features, x.avx2)) break :blk "avx2";
            if (std.Target.x86.featureSetHas(cpu.features, x.sse4_1)) break :blk "sse4.1";
            break :blk "sse2";
        },
        .aarch64 => "neon",
        .wasm32, .wasm64 => if (std.Target.wasm.featureSetHas(cpu.features, .simd128)) "simd128" else "scalar",
        else => "scalar",
    };
    break :desc @tagName(cpu.arch) ++ "-" ++ @tagName(builtin.os.tag) ++ " (" ++ feats ++ ")";
};

test {
    std.testing.refAllDecls(@This());
}
