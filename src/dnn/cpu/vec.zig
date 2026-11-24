//! Vector width, channel block and register-tile shapes for the CPU
//! kernels, chosen per target at compile time (after ditch's matmul tile
//! table: one native register per vector, as many accumulators as the
//! register file holds next to the weights and broadcasts).

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("../ir.zig");

const x86 = builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86;
const avx512 = x86 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
const avx = x86 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx);
pub const neon = builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .aarch64_be;

/// Channels per block of the blocked layout = f32 lanes of one register.
pub const B: usize = if (avx512) 16 else if (avx) 8 else 4;
pub const V = @Vector(B, f32);

/// Whether the target fuses multiply-add in hardware (without it `@mulAdd`
/// becomes a libm call per lane).
pub const has_fma = if (x86) std.Target.x86.featureSetHas(builtin.cpu.features, .fma) else switch (builtin.cpu.arch) {
    .aarch64, .aarch64_be => true,
    else => false,
};

pub inline fn fma(a: V, b: V, c: V) V {
    return if (has_fma) @mulAdd(V, a, b, c) else a * b + c;
}

/// On NEON a lane of a loaded vector feeds `fmla v, v, v.s[j]` directly, so
/// the pointwise kernel loads each pixel's channel block once and splats
/// lanes from registers; x86 broadcasts straight from memory instead.
pub const lane_broadcast = neon;

/// Accumulator vectors the pointwise and direct kernels keep live.
const acc_budget: usize = if (avx512) 24 else if (avx) 12 else if (neon) 20 else 8;

/// Output-channel blocks per register tile (at most).
pub const max_nb: usize = 4;

/// Pixels per register tile for a tile `nb` blocks wide.
pub fn tileR(comptime nb: usize) usize {
    return @max(1, @min(12, acc_budget / nb));
}

pub inline fn splat(x: f32) V {
    return @splat(x);
}

pub inline fn load(p: [*]const f32) V {
    return p[0..B].*;
}

pub inline fn store(p: [*]f32, v: V) void {
    p[0..B].* = v;
}

/// Lanes [0, n) set.
pub fn laneMask(n: usize) @Vector(B, bool) {
    const idx = std.simd.iota(u32, B);
    return idx < @as(@Vector(B, u32), @splat(@intCast(n)));
}

pub const ActKind = enum { none, relu, generic };

pub fn actKind(a: ir.Act) ActKind {
    return switch (a) {
        .none => .none,
        .relu => .relu,
        else => .generic,
    };
}

pub inline fn act(comptime kind: ActKind, a: ir.Act, v: V) V {
    return switch (kind) {
        .none => v,
        .relu => @max(v, splat(0)),
        .generic => actGeneric(a, v),
    };
}

pub fn actGeneric(a: ir.Act, v: V) V {
    const zero = splat(0);
    const one = splat(1);
    return switch (a) {
        .none => v,
        .relu => @max(v, zero),
        .leaky_relu => |al| @select(f32, v >= zero, v, v * splat(al)),
        .clip => |c| @min(@max(v, splat(c[0])), splat(c[1])),
        .sigmoid => one / (one + exp(-v)),
        .hard_sigmoid => |p| @min(@max(v * splat(p[0]) + splat(p[1]), zero), one),
        .hard_swish => v * @min(@max(v * splat(1.0 / 6.0) + splat(0.5), zero), one),
        .silu => v / (one + exp(-v)),
        .tanh => blk: {
            var r: V = undefined;
            inline for (0..B) |i| r[i] = std.math.tanh(v[i]);
            break :blk r;
        },
    };
}

/// e^x for every lane (Cephes expf: range reduction by ln 2 in two parts
/// and a degree-6 polynomial; within 2 ulp of the true value; 0 below
/// -87.3 and +inf above 88.7).
pub fn exp(x: V) V {
    const hi = splat(88.72283);
    const lo = splat(-87.33654);
    const xc = @min(@max(x, lo), hi);
    const n = @round(xc * splat(1.44269504088896341));
    var r = xc - n * splat(0.693359375);
    r = r - n * splat(-2.12194440e-4);
    var p = splat(1.9875691500e-4);
    p = p * r + splat(1.3981999507e-3);
    p = p * r + splat(8.3334519073e-3);
    p = p * r + splat(4.1665795894e-2);
    p = p * r + splat(1.6666665459e-1);
    p = p * r + splat(5.0000001201e-1);
    p = p * r * r + r + splat(1);
    const ni: @Vector(B, i32) = @intFromFloat(n);
    const scale: V = @bitCast((ni + @as(@Vector(B, i32), @splat(127))) << @splat(23));
    const y = p * scale;
    const big = x > hi;
    const small = x < lo;
    return @select(f32, big, splat(std.math.inf(f32)), @select(f32, small, splat(0), y));
}

test "vector exp matches @exp" {
    var x: f32 = -87;
    while (x < 88) : (x += 0.37) {
        const want = @exp(x);
        const got = exp(splat(x))[0];
        try std.testing.expectApproxEqRel(want, got, 3e-7);
    }
    try std.testing.expectEqual(@as(f32, 0), exp(splat(-100))[0]);
    try std.testing.expect(std.math.isInf(exp(splat(100))[0]));
}

/// Rows of a blocked tensor or of a ring buffer holding the last `ring`
/// rows of one: row `y` of channel block `cb` starts at `at(cb, y)`.
pub const Rows = struct {
    base: [*]f32,
    cb_stride: usize,
    row_stride: usize,
    /// 0: every row is stored; else rows live at `y % ring`.
    ring: usize = 0,

    pub inline fn at(r: Rows, cb: usize, y: usize) [*]f32 {
        const yy = if (r.ring != 0) y % r.ring else y;
        return r.base + cb * r.cb_stride + yy * r.row_stride;
    }

    /// Image `img` of a blocked tensor at element offset `off`.
    pub fn tensor(base: [*]f32, off: usize, img: usize, blocks: usize, h: usize, w: usize) Rows {
        return .{ .base = base + off + img * blocks * h * w * B, .cb_stride = h * w * B, .row_stride = w * B };
    }

    /// A ring of `rows` rows of `blocks` blocks of width `w` at `buf`,
    /// stored row-major ([row][block][w][B]).
    pub fn ringOf(buf: [*]f32, rows: usize, blocks: usize, w: usize) Rows {
        return .{ .base = buf, .cb_stride = w * B, .row_stride = blocks * w * B, .ring = rows };
    }
};
