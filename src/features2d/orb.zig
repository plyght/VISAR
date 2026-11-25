//! ORB (Rublee et al. 2011): oriented FAST keypoints on a scale pyramid,
//! ranked by the Harris measure, with rotated BRIEF descriptors, following
//! OpenCV's ORB_create step for step:
//!
//!  * the pyramid levels are scaled by scaleFactor^-level with OpenCV's
//!    bit-exact bilinear resize (INTER_LINEAR_EXACT, 8.8 fixed point), each
//!    from the previous level, and padded with BORDER_REFLECT_101;
//!  * FAST-9/16 with non-maximum suppression per level, the border filter,
//!    retainBest(2n) on the FAST score, the 7x7 Harris response in f32 and
//!    retainBest(n) per level, n following OpenCV's geometric split;
//!  * the intensity-centroid angle over the circular patch with OpenCV's
//!    fastAtan2 polynomial;
//!  * descriptors on the level blurred with OpenCV's 7x7, sigma 2 Gaussian
//!    (the f32 separable filter OpenCV runs on a submatrix), sampled with the 256 tests of bit_pattern_31 rotated by
//!    the keypoint angle and rounded to the nearest pixel.
//!
//! Keypoint sets and descriptors match cv2.ORB_create. One deliberate
//! difference: within a pyramid level OpenCV leaves the keypoints in the
//! order std::nth_element shuffled them into (which differs between C++
//! standard libraries); VISAR keeps FAST's row-major order. Levels come in
//! increasing order in both.
//!
//! On aarch64 the f32 expressions (Harris response, angle polynomial,
//! pattern rotation) are evaluated with fused multiply-adds where clang
//! contracts them when it compiles OpenCV for Apple Silicon, so responses,
//! angles and sample positions round the same way there.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const kp_mod = @import("keypoint.zig");
const fast = @import("fast.zig");
const bit_pattern_31 = @import("orb_pattern.zig").bit_pattern_31;
const Image = core.Image;
const simd = core.simd;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const KeyPoint = kp_mod.KeyPoint;
pub const Error = core.Error || Allocator.Error || error{ InvalidArgument, Unsupported };

/// Bytes per descriptor (OpenCV's ORB always produces 32).
pub const descriptor_size = 32;

/// ORB::ScoreType.
pub const ScoreType = enum(u8) { harris = 0, fast = 1 };

/// ORB_create's parameters and defaults.
pub const Options = struct {
    nfeatures: u32 = 500,
    /// OpenCV stores the float it is given as a double: 1.2f by default.
    scale_factor: f64 = @as(f32, 1.2),
    nlevels: u32 = 8,
    edge_threshold: u32 = 31,
    first_level: u32 = 0,
    /// Points per descriptor test: 2 (BRIEF), 3 or 4 (argmax of 2 bits).
    wta_k: u8 = 2,
    score_type: ScoreType = .harris,
    patch_size: u32 = 31,
    fast_threshold: i32 = 20,
};

pub const Features = struct {
    keypoints: []KeyPoint,
    /// keypoints.len * descriptor_size bytes (empty when not requested).
    descriptors: []u8,

    pub fn deinit(self: *Features, gpa: Allocator) void {
        gpa.free(self.keypoints);
        gpa.free(self.descriptors);
        self.* = undefined;
    }
};

const contract = builtin.cpu.arch == .aarch64;

/// a*b - c*d as OpenCV's compiler evaluates it.
inline fn mulSub(a: f32, b: f32, c: f32, d: f32) f32 {
    if (contract) return @mulAdd(f32, a, b, -(c * d));
    return a * b - c * d;
}

/// a*b + c*d as OpenCV's compiler evaluates it.
inline fn mulAdd2(a: f32, b: f32, c: f32, d: f32) f32 {
    if (contract) return @mulAdd(f32, a, b, c * d);
    return a * b + c * d;
}

inline fn madd(a: f32, b: f32, c: f32) f32 {
    if (contract) return @mulAdd(f32, a, b, c);
    return a * b + c;
}

const rad2deg_f32: f32 = @floatCast(@as(f64, 180.0) / @as(f64, std.math.pi));
const atan_p1: f32 = @as(f32, 0.9997878412794807) * rad2deg_f32;
const atan_p3: f32 = @as(f32, -0.3258083974640975) * rad2deg_f32;
const atan_p5: f32 = @as(f32, 0.1555786518463281) * rad2deg_f32;
const atan_p7: f32 = @as(f32, -0.04432655554792128) * rad2deg_f32;
const dbl_eps_f32: f32 = @floatCast(std.math.floatEps(f64));

/// OpenCV's fastAtan2: degrees in [0, 360).
pub fn fastAtan2(y: f32, x: f32) f32 {
    const ax = @abs(x);
    const ay = @abs(y);
    var a: f32 = undefined;
    if (ax >= ay) {
        const c = ay / (ax + dbl_eps_f32);
        const c2 = c * c;
        a = madd(madd(madd(atan_p7, c2, atan_p5), c2, atan_p3), c2, atan_p1) * c;
    } else {
        const c = ax / (ay + dbl_eps_f32);
        const c2 = c * c;
        const p = madd(madd(madd(atan_p7, c2, atan_p5), c2, atan_p3), c2, atan_p1);
        a = if (contract) @mulAdd(f32, -p, c, 90.0) else 90.0 - p * c;
    }
    if (x < 0) a = 180.0 - a;
    if (y < 0) a = 360.0 - a;
    return a;
}

/// cvRound for f32: round half to even.
inline fn roundEven(x: f32) i32 {
    const r = @round(x);
    return @intFromFloat(if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) r - std.math.sign(x) else r);
}

// ------------------------------------------------------------- cv::RNG

const Rng = struct {
    state: u64,

    fn init(s: u64) Rng {
        return .{ .state = if (s != 0) s else 0xffffffff };
    }

    fn next(r: *Rng) u32 {
        r.state = @as(u64, @as(u32, @truncate(r.state))) *% 4164903690 +% (r.state >> 32);
        return @truncate(r.state);
    }

    /// RNG::uniform(int a, int b) for a < b.
    fn uniform(r: *Rng, a: i32, b: i32) i32 {
        if (a == b) return a;
        const span: u32 = @bitCast(b -% a);
        const v = r.next() % span;
        return @bitCast(v +% @as(u32, @bitCast(a)));
    }
};

/// The test points: 512 for WTA_K = 2 (pairs), or 128 tuples of WTA_K.
fn makePattern(gpa: Allocator, patch_size: u32, wta_k: u8) Allocator.Error![][2]i32 {
    var pool: [512][2]i32 = undefined;
    if (patch_size == 31) {
        for (&pool, bit_pattern_31) |*p, q| p.* = .{ q[0], q[1] };
    } else {
        // makeRandomPattern
        var rng = Rng.init(0x34985739);
        const half: i32 = @intCast(patch_size / 2);
        for (&pool) |*p| {
            const x = rng.uniform(-half, half + 1);
            const y = rng.uniform(-half, half + 1);
            p.* = .{ x, y };
        }
    }
    if (wta_k == 2) return gpa.dupe([2]i32, &pool);
    // initializeOrbPattern: tuples of distinct points drawn from the pool.
    const ntuples = descriptor_size * 4;
    const out = try gpa.alloc([2]i32, ntuples * @as(usize, wta_k));
    var rng = Rng.init(0x12345678);
    for (0..ntuples) |i| {
        for (0..wta_k) |k| {
            while (true) {
                const idx: usize = @intCast(rng.uniform(0, 512));
                const pt = pool[idx];
                var dup = false;
                for (0..k) |k1| {
                    if (out[wta_k * i + k1][0] == pt[0] and out[wta_k * i + k1][1] == pt[1]) dup = true;
                }
                if (!dup) {
                    out[wta_k * i + k] = pt;
                    break;
                }
            }
        }
    }
    return out;
}

// ---------------------------------------------------- bit-exact resize

/// One axis of INTER_LINEAR_EXACT: source offsets and 8.8 weights, with
/// the outputs before `lo` and from `hi` on copying the first and last
/// source sample.
const Axis = struct {
    ofs: []u32,
    w0: []u16,
    w1: []u16,
    lo: usize,
    hi: usize,
    /// Blocks of 8 outputs from `lo` for the byte-shuffle path: the source
    /// window start (or `no_block` when the 8 pairs do not fit 16 bytes)
    /// and the 16 byte indices of the (s[i], s[i + 1]) pairs.
    base: []u32 = &.{},
    idx: []@Vector(16, u8) = &.{},

    const no_block = std.math.maxInt(u32);

    fn initBlocks(a: *Axis, gpa: Allocator, src: usize) Allocator.Error!void {
        if (!has_shuffle) return;
        const nb = (a.hi - a.lo) / 8;
        a.base = try gpa.alloc(u32, nb);
        a.idx = try gpa.alloc(@Vector(16, u8), nb);
        for (0..nb) |b| {
            const d = a.lo + 8 * b;
            const b0 = a.ofs[d];
            if (a.ofs[d + 7] + 1 - b0 > 15 or b0 + 16 > src) {
                a.base[b] = no_block;
                continue;
            }
            a.base[b] = b0;
            var ix: [16]u8 = undefined;
            for (0..8) |k| {
                ix[2 * k] = @intCast(a.ofs[d + k] - b0);
                ix[2 * k + 1] = @intCast(a.ofs[d + k] - b0 + 1);
            }
            a.idx[b] = ix;
        }
    }

    fn init(gpa: Allocator, src: usize, dst: usize) Allocator.Error!Axis {
        const ofs = try gpa.alloc(u32, dst);
        errdefer gpa.free(ofs);
        const w0 = try gpa.alloc(u16, dst);
        errdefer gpa.free(w0);
        const w1 = try gpa.alloc(u16, dst);
        var a: Axis = .{ .ofs = ofs, .w0 = w0, .w1 = w1, .lo = 0, .hi = dst };
        const inv = @as(f64, @floatFromInt(dst)) / @as(f64, @floatFromInt(src));
        const scale = 1.0 / inv;
        for (0..dst) |d| {
            const fval = scale * (@as(f64, @floatFromInt(d)) + 0.5) - 0.5;
            const fl = @floor(fval);
            ofs[d] = 0;
            w0[d] = 256;
            w1[d] = 0;
            if (fl >= 0 and src > 1) {
                const ival: usize = @intFromFloat(fl);
                if (ival < src - 1) {
                    ofs[d] = @intCast(ival);
                    const c1: u16 = @intFromFloat(@round(roundTiesEven((fval - fl) * 256.0)));
                    w1[d] = c1;
                    w0[d] = 256 -| c1;
                } else {
                    ofs[d] = @intCast(src - 1);
                    a.hi = @min(a.hi, d);
                }
            } else {
                a.lo = @max(a.lo, d + 1);
            }
        }
        if (a.lo > a.hi) a.hi = a.lo;
        return a;
    }

    fn deinit(a: *Axis, gpa: Allocator) void {
        gpa.free(a.ofs);
        gpa.free(a.w0);
        gpa.free(a.w1);
        gpa.free(a.base);
        gpa.free(a.idx);
    }
};

fn roundTiesEven(x: f64) f64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5 and @mod(r, 2) != 0) return r - std.math.sign(x);
    return r;
}

/// A runtime 16-byte table lookup: pshufb (AVX encoding) on x86, tbl on
/// AArch64 (both need the LLVM backend).
const has_shuffle = builtin.zig_backend == .stage2_llvm and switch (builtin.cpu.arch) {
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx),
    .aarch64 => simd.llvm_intrinsics,
    else => false,
};
extern fn @"llvm.aarch64.neon.tbl1.v16i8"(@Vector(16, u8), @Vector(16, u8)) @Vector(16, u8);

inline fn shuffle16(t: @Vector(16, u8), i: @Vector(16, u8)) @Vector(16, u8) {
    if (builtin.cpu.arch == .aarch64) return @"llvm.aarch64.neon.tbl1.v16i8"(t, i);
    return asm ("vpshufb %[i], %[t], %[r]"
        : [r] "=x" (-> @Vector(16, u8)),
        : [t] "x" (t),
          [i] "x" (i),
    );
}

/// Horizontal pass of one source row into 8.8 fixed point.
fn hline(ax: *const Axis, s: []const u8, out: []u16) void {
    const last: u16 = @as(u16, s[s.len - 1]) << 8;
    const first: u16 = @as(u16, s[0]) << 8;
    for (out[0..ax.lo]) |*o| o.* = first;
    if (has_shuffle) {
        // Eight outputs per 16-byte load and byte shuffle.
        const VU8 = @Vector(8, u16);
        var d = ax.lo;
        for (ax.base, ax.idx) |b0, ix| {
            if (b0 != Axis.no_block) {
                const t: @Vector(16, u8) = s[b0..][0..16].*;
                const pv: VU8 = @bitCast(shuffle16(t, ix));
                const w0: VU8 = ax.w0[d..][0..8].*;
                const w1: VU8 = ax.w1[d..][0..8].*;
                out[d..][0..8].* = w0 * (pv & @as(VU8, @splat(0xff))) + w1 * (pv >> @splat(8));
            } else for (d..d + 8) |e| {
                const i = ax.ofs[e];
                out[e] = ax.w0[e] * @as(u16, s[i]) + ax.w1[e] * @as(u16, s[i + 1]);
            }
            d += 8;
        }
        while (d < ax.hi) : (d += 1) {
            const i = ax.ofs[d];
            out[d] = ax.w0[d] * @as(u16, s[i]) + ax.w1[d] * @as(u16, s[i + 1]);
        }
        for (out[ax.hi..]) |*o| o.* = last;
        return;
    }
    // Sixteen outputs at a time: gather the source pairs as u16 (s[i] in
    // the low byte), then weight them with vector arithmetic.
    const V = 16;
    const VU16 = @Vector(V, u16);
    var d = ax.lo;
    while (d + V <= ax.hi) : (d += V) {
        var pairs: [V]u16 = undefined;
        inline for (0..V) |k| pairs[k] = std.mem.readInt(u16, s[ax.ofs[d + k]..][0..2], .little);
        const pv: VU16 = pairs;
        const w0: VU16 = ax.w0[d..][0..V].*;
        const w1: VU16 = ax.w1[d..][0..V].*;
        out[d..][0..V].* = w0 * (pv & @as(VU16, @splat(0xff))) + w1 * (pv >> @splat(8));
    }
    while (d < ax.hi) : (d += 1) {
        const i = ax.ofs[d];
        out[d] = ax.w0[d] * @as(u16, s[i]) + ax.w1[d] * @as(u16, s[i + 1]);
    }
    for (out[ax.hi..]) |*o| o.* = last;
}

/// OpenCV's resize(..., INTER_LINEAR_EXACT) for one-channel u8.
pub fn resizeLinearExact(gpa: Allocator, src: Image(u8), dst: Image(u8)) Allocator.Error!void {
    const sw: usize = src.width;
    const sh: usize = src.height;
    const dw: usize = dst.width;
    const dh: usize = dst.height;
    if (sw == 2 * dw and sh == 2 * dh) return halve(src, dst);
    var ax = try Axis.init(gpa, sw, dw);
    defer ax.deinit(gpa);
    try ax.initBlocks(gpa, sw);
    var ay = try Axis.init(gpa, sh, dh);
    defer ay.deinit(gpa);
    const buf = try gpa.alloc(u16, 2 * dw);
    defer gpa.free(buf);
    const rows: [2][]u16 = .{ buf[0..dw], buf[dw..] };
    var have: [2]isize = .{ -1, -1 }; // source row held in each buffer
    for (0..dh) |dy| {
        const d = dst.row(dy);
        if (dy < ay.lo or dy >= ay.hi) {
            const sy: usize = if (dy < ay.lo) 0 else sh - 1;
            hline(&ax, src.row(sy), rows[0]);
            have[0] = @intCast(sy);
            for (d, rows[0]) |*o, v| o.* = @intCast((@as(u32, v) + 128) >> 8);
            continue;
        }
        const iy = ay.ofs[dy];
        // Reuse a row already filtered when the window slides by one.
        var r0: usize = 0;
        var r1: usize = 1;
        if (have[1] == iy) {
            r0 = 1;
            r1 = 0;
        } else if (have[0] != iy) {
            hline(&ax, src.row(iy), rows[0]);
            have[0] = @intCast(iy);
        }
        if (have[r1] != iy + 1) {
            hline(&ax, src.row(iy + 1), rows[r1]);
            have[r1] = @intCast(iy + 1);
        }
        vline(rows[r0], rows[r1], ay.w0[dy], ay.w1[dy], d);
    }
}

fn vline(a: []const u16, b: []const u16, w0: u16, w1: u16, d: []u8) void {
    const V = 16;
    const VW = @Vector(V, u32);
    const c0: VW = @splat(w0);
    const c1: VW = @splat(w1);
    const half: VW = @splat(0x8000);
    var x: usize = 0;
    while (x + V <= d.len) : (x += V) {
        const va: VW = @as(@Vector(V, u16), a[x..][0..V].*);
        const vb: VW = @as(@Vector(V, u16), b[x..][0..V].*);
        const r = (va * c0 + vb * c1 + half) >> @splat(16);
        const o: @Vector(V, u8) = @intCast(r);
        d[x..][0..V].* = o;
    }
    while (x < d.len) : (x += 1) {
        d[x] = @intCast((@as(u32, a[x]) * w0 + @as(u32, b[x]) * w1 + 0x8000) >> 16);
    }
}

/// Exact halving: OpenCV switches INTER_LINEAR_EXACT to its 2x2 area
/// average, (a + b + c + d + 2) >> 2.
fn halve(src: Image(u8), dst: Image(u8)) void {
    for (0..dst.height) |y| {
        const a = src.row(2 * y);
        const b = src.row(2 * y + 1);
        const d = dst.row(y);
        for (d, 0..) |*o, x| {
            const s = @as(u32, a[2 * x]) + a[2 * x + 1] + b[2 * x] + b[2 * x + 1];
            o.* = @intCast((s + 2) >> 2);
        }
    }
}

// --------------------------------------------------------------- pyramid

/// A pyramid level stored with `border` pixels of BORDER_REFLECT_101
/// padding on every side (OpenCV's imagePyramid layout).
const Level = struct {
    buf: []u8 = &.{},
    stride: usize = 0,
    w: usize = 0,
    h: usize = 0,
    scale: f32 = 1,

    fn interior(l: Level, border: usize) Image(u8) {
        return .{ .width = @intCast(l.w), .height = @intCast(l.h), .channels = 1, .stride = l.stride, .data = l.buf.ptr + border * l.stride + border };
    }

    /// Pointer to interior pixel (x, y), which may lie in the padding.
    inline fn at(l: Level, border: usize, x: isize, y: isize) [*]const u8 {
        const off: isize = (y + @as(isize, @intCast(border))) * @as(isize, @intCast(l.stride)) + x + @as(isize, @intCast(border));
        return l.buf.ptr + @as(usize, @intCast(off));
    }
};

fn fillBorder(l: Level, border: usize) void {
    const b = border;
    // Source columns of the left and right padding (at most 2 * 4096 + 2).
    var lmap: [4097]u32 = undefined;
    var rmap: [4097]u32 = undefined;
    for (0..b) |i| {
        lmap[i] = @intCast(imgproc.border.map(-@as(isize, @intCast(b - i)), l.w, .reflect101).?);
        rmap[i] = @intCast(imgproc.border.map(@intCast(l.w + i), l.w, .reflect101).?);
    }
    for (0..l.h) |y| {
        const row = l.buf[(y + b) * l.stride ..][0..l.stride];
        const in = row[b..][0..l.w];
        for (row[0..b], lmap[0..b]) |*o, x| o.* = in[x];
        for (row[b + l.w ..][0..b], rmap[0..b]) |*o, x| o.* = in[x];
    }
    for (0..b) |i| {
        const yt = imgproc.border.map(-@as(isize, @intCast(b - i)), l.h, .reflect101).?;
        @memcpy(l.buf[i * l.stride ..][0..l.stride], l.buf[(yt + b) * l.stride ..][0..l.stride]);
        const yb = imgproc.border.map(@intCast(l.h + i), l.h, .reflect101).?;
        @memcpy(l.buf[(b + l.h + i) * l.stride ..][0..l.stride], l.buf[(yb + b) * l.stride ..][0..l.stride]);
    }
}

fn fillMaskBorder(l: Level, border: usize) void {
    // BORDER_CONSTANT 0.
    const b = border;
    @memset(l.buf[0 .. b * l.stride], 0);
    @memset(l.buf[(b + l.h) * l.stride ..], 0);
    for (0..l.h) |y| {
        const row = l.buf[(y + b) * l.stride ..][0..l.stride];
        @memset(row[0..b], 0);
        @memset(row[b + l.w ..], 0);
    }
}

// ------------------------------------------------------------- Gaussian

/// getGaussianKernel(7, 2, CV_32F), bit for bit.
const gauss7 = [7]f32{ 0x1.1f5f62p-4, 0x1.0c70fcp-3, 0x1.869472p-3, 0x1.ba95c0p-3, 0x1.869472p-3, 0x1.0c70fcp-3, 0x1.1f5f62p-4 };

const hw_fma = switch (builtin.cpu.arch) {
    .aarch64 => true,
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    else => false,
};

/// a * b + c rounded once, as OpenCV's v_muladd computes it (with FMA on
/// AVX2 and NEON). Without FMA hardware the f64 sum is exact for these
/// operands (8-bit pixels and the f32 kernel), so one rounding to f32 gives
/// the same result.
inline fn fmaV(comptime n: comptime_int, a: @Vector(n, f32), b: @Vector(n, f32), c: @Vector(n, f32)) @Vector(n, f32) {
    if (hw_fma) return @mulAdd(@Vector(n, f32), a, b, c);
    const A: @Vector(n, f64) = a;
    const B: @Vector(n, f64) = b;
    const C: @Vector(n, f64) = c;
    return @floatCast(A * B + C);
}

/// GaussianBlur(level, 7x7, sigma 2, BORDER_REFLECT_101) on the interior of
/// `src` (reading its padding), written to the interior of `dst` (which may
/// be `src`: row y is written after rows up to y + 3 were read). On a
/// submatrix OpenCV skips its fixed-point Gaussian and runs the separable
/// f32 filter: the row pass accumulates k[i] * p[x + i] with fused
/// multiply-adds from i = 0, the column pass starts from k[3] * r[y] and
/// adds k[3 + j] * (r[y + j] + r[y - j]); the result rounds half to even.
fn blurLevel(gpa: Allocator, src: Level, dst: Level, border: usize) Allocator.Error!void {
    const w = src.w;
    const h = src.h;
    const V = 16;
    const wp = std.mem.alignForward(usize, w, V);
    const ring = try gpa.alloc(f32, 8 * wp + 2 * V);
    defer gpa.free(ring);
    @memset(ring, 0);
    const conv = ring[7 * wp ..]; // one source row as f32 (w + 6 values)
    const VF = @Vector(V, f32);
    const S = struct {
        fn rowFilter(l: Level, b: usize, y: isize, out: []f32, n: usize, f: []f32) void {
            const p = l.at(b, -3, y);
            var x: usize = 0;
            while (x + V <= n + 6) : (x += V) f[x..][0..V].* = simd.toF32(V, @as(@Vector(V, u8), p[x..][0..V].*));
            while (x < n + 6) : (x += 1) f[x] = @floatFromInt(p[x]);
            x = 0;
            while (x + V <= n) : (x += V) {
                var acc: VF = @splat(0);
                inline for (0..7) |k| acc = fmaV(V, f[x + k ..][0..V].*, @splat(gauss7[k]), acc);
                out[x..][0..V].* = acc;
            }
            while (x < n) : (x += 1) {
                var acc: @Vector(1, f32) = @splat(0);
                inline for (0..7) |k| acc = fmaV(1, @splat(f[x + k]), @splat(gauss7[k]), acc);
                out[x] = acc[0];
            }
        }
    };
    var slots: [7][]f32 = undefined;
    for (0..7) |i| slots[i] = ring[i * wp ..][0..wp];
    for (0..6) |i| S.rowFilter(src, border, @as(isize, @intCast(i)) - 3, slots[i], w, conv);
    var next_slot: usize = 6;
    for (0..h) |y| {
        S.rowFilter(src, border, @as(isize, @intCast(y)) + 3, slots[next_slot], w, conv);
        next_slot = (next_slot + 1) % 7;
        // Rows y - 3 .. y + 3 start at next_slot.
        var r: [7][]const f32 = undefined;
        for (0..7) |k| r[k] = slots[(next_slot + k) % 7];
        const d = dst.buf[(y + border) * dst.stride + border ..][0..w];
        var x: usize = 0;
        while (x < w) : (x += V) {
            var s: VF = @as(VF, r[3][x..][0..V].*) * @as(VF, @splat(gauss7[3]));
            inline for (1..4) |j| {
                const pair = @as(VF, r[3 + j][x..][0..V].*) + @as(VF, r[3 - j][x..][0..V].*);
                s = fmaV(V, @splat(gauss7[3 + j]), pair, s);
            }
            const o: [V]u8 = simd.satF32(u8, V, s);
            if (x + V <= w) {
                d[x..][0..V].* = o;
            } else {
                @memcpy(d[x..], o[0 .. w - x]);
            }
        }
    }
}

// ------------------------------------------------------------- measures

/// HarrisResponses with blockSize 7 on the unblurred level.
fn harris(l: Level, border: usize, k: *KeyPoint) void {
    const bs = 7;
    const r = bs / 2;
    const x0: isize = roundEven(k.x);
    const y0: isize = roundEven(k.y);
    const step: isize = @intCast(l.stride);
    var a: i32 = 0;
    var b: i32 = 0;
    var c: i32 = 0;
    for (0..bs) |i| {
        const p = l.at(border, x0 - r, y0 - r + @as(isize, @intCast(i)));
        for (0..bs) |j| {
            const q = p + j;
            const s: usize = @intCast(step);
            const up = q - s;
            const dn = q + s;
            const Ix = (@as(i32, (q + 1)[0]) - @as(i32, (q - 1)[0])) * 2 + (@as(i32, (up + 1)[0]) - @as(i32, (up - 1)[0])) + (@as(i32, (dn + 1)[0]) - @as(i32, (dn - 1)[0]));
            const Iy = (@as(i32, dn[0]) - @as(i32, up[0])) * 2 + (@as(i32, (dn - 1)[0]) - @as(i32, (up - 1)[0])) + (@as(i32, (dn + 1)[0]) - @as(i32, (up + 1)[0]));
            a += Ix * Ix;
            b += Iy * Iy;
            c += Ix * Iy;
        }
    }
    const scale: f32 = 1.0 / (@as(f32, @floatFromInt(4 * bs)) * 255.0);
    const s4 = scale * scale * scale * scale;
    const fa: f32 = @floatFromInt(a);
    const fb: f32 = @floatFromInt(b);
    const fc: f32 = @floatFromInt(c);
    const kk: f32 = 0.04;
    const sum = fa + fb;
    const det = mulSub(fa, fb, fc, fc);
    const resp = if (contract) @mulAdd(f32, -(kk * sum), sum, det) else det - kk * sum * sum;
    k.response = resp * s4;
}

/// End of each row of the circular patch (OpenCV's umax).
fn patchRows(gpa: Allocator, half: usize) Allocator.Error![]i32 {
    const umax = try gpa.alloc(i32, half + 2);
    @memset(umax, 0);
    const hf: f32 = @floatFromInt(half);
    const vmax: usize = @intFromFloat(@floor(hf * @sqrt(@as(f32, 2)) / 2 + 1));
    const vmin: usize = @intFromFloat(@ceil(hf * @sqrt(@as(f32, 2)) / 2));
    const hd: f64 = @floatFromInt(half);
    for (0..@min(vmax, half + 1) + 1) |v| {
        const vd: f64 = @floatFromInt(v);
        umax[v] = @intFromFloat(roundTiesEven(@sqrt(@max(0, hd * hd - vd * vd))));
    }
    // Make it symmetric.
    var v0: usize = 0;
    var v: usize = half;
    while (v >= vmin) : (v -= 1) {
        while (umax[v0] == umax[v0 + 1]) v0 += 1;
        umax[v] = @intCast(v0);
        v0 += 1;
        if (v == 0) break;
    }
    return umax;
}

/// Intensity-centroid orientation (ICAngles) on the unblurred level.
fn icAngle(l: Level, border: usize, x: isize, y: isize, umax: []const i32, half: usize) f32 {
    const c = l.at(border, x, y);
    const s = l.stride;
    var m01: i32 = 0;
    var m10: i32 = 0;
    if (half == 15) {
        const W = 32;
        const VI = @Vector(W, i32);
        const VU32 = @Vector(W, u32);
        const u: VI = std.simd.iota(i32, W) - @as(VI, @splat(15));
        const au: VU32 = @abs(u);
        const lo = c - 15;
        const row0: VI = @as(@Vector(W, u8), lo[0..W].*);
        var a10: VI = @select(i32, au <= @as(VU32, @splat(15)), u * row0, @as(VI, @splat(0)));
        var a01: VI = @splat(0);
        for (1..16) |v| {
            const d: u32 = @intCast(umax[v]);
            const in = au <= @as(VU32, @splat(d));
            const p: VI = @as(@Vector(W, u8), (lo + v * s)[0..W].*);
            const m: VI = @as(@Vector(W, u8), (lo - v * s)[0..W].*);
            const vv: VI = @splat(@intCast(v));
            a10 += @select(i32, in, u * (p + m), @as(VI, @splat(0)));
            a01 += @select(i32, in, vv * (p - m), @as(VI, @splat(0)));
        }
        m10 = @reduce(.Add, a10);
        m01 = @reduce(.Add, a01);
    } else {
        const hk: isize = @intCast(half);
        const step: isize = @intCast(s);
        var u: isize = -hk;
        while (u <= hk) : (u += 1) m10 += @as(i32, @intCast(u)) * pix(c, u);
        var v: isize = 1;
        while (v <= hk) : (v += 1) {
            var v_sum: i32 = 0;
            const d: isize = umax[@intCast(v)];
            u = -d;
            while (u <= d) : (u += 1) {
                const vp = pix(c, u + v * step);
                const vm = pix(c, u - v * step);
                v_sum += vp - vm;
                m10 += @as(i32, @intCast(u)) * (vp + vm);
            }
            m01 += @as(i32, @intCast(v)) * v_sum;
        }
    }
    return fastAtan2(@floatFromInt(m01), @floatFromInt(m10));
}

inline fn pix(base: [*]const u8, off: isize) i32 {
    return if (off >= 0) (base + @as(usize, @intCast(off)))[0] else (base - @as(usize, @intCast(-off)))[0];
}

const deg2rad_f32: f32 = @floatCast(@as(f64, std.math.pi) / @as(f64, @as(f32, 180.0)));

/// The 32-byte descriptor of `k` on its blurred level.
fn describe(l: Level, border: usize, k: KeyPoint, inv: f32, px: []const f32, py: []const f32, wta_k: u8, out: *[descriptor_size]u8) void {
    const angle = k.angle * deg2rad_f32;
    const a: f32 = @floatCast(@cos(@as(f64, angle)));
    const b: f32 = @floatCast(@sin(@as(f64, angle)));
    const cx = roundEven(k.x * inv);
    const cy = roundEven(k.y * inv);
    const center = l.at(border, cx, cy);
    const step: i32 = @intCast(l.stride);
    const n = px.len;
    var vals: [512]u8 = undefined;
    const V = 16;
    const VF = @Vector(V, f32);
    const VI = @Vector(V, i32);
    const va: VF = @splat(a);
    const vb: VF = @splat(b);
    const magic: VF = @splat(12582912.0);
    var i: usize = 0;
    while (i < n) : (i += V) {
        const x0: VF = px[i..][0..V].*;
        const y0: VF = py[i..][0..V].*;
        const x = if (contract) @mulAdd(VF, x0, va, -(y0 * vb)) else x0 * va - y0 * vb;
        const y = if (contract) @mulAdd(VF, x0, vb, y0 * va) else x0 * vb + y0 * va;
        const ix: VI = @intFromFloat((x + magic) - magic);
        const iy: VI = @intFromFloat((y + magic) - magic);
        const off: [V]i32 = iy * @as(VI, @splat(step)) + ix;
        inline for (0..V) |j| vals[i + j] = @intCast(pix(center, off[j]));
    }
    switch (wta_k) {
        2 => {
            inline for (0..8) |c| {
                const v: @Vector(64, u8) = vals[c * 64 ..][0..64].*;
                const ev = @shuffle(u8, v, undefined, std.simd.iota(i32, 32) * @as(@Vector(32, i32), @splat(2)));
                const od = @shuffle(u8, v, undefined, std.simd.iota(i32, 32) * @as(@Vector(32, i32), @splat(2)) + @as(@Vector(32, i32), @splat(1)));
                const bits: u32 = @bitCast(ev < od);
                std.mem.writeInt(u32, out[c * 4 ..][0..4], bits, .little);
            }
        },
        3 => {
            for (0..descriptor_size) |byte| {
                var val: u8 = 0;
                for (0..4) |g| {
                    const t = vals[byte * 12 + g * 3 ..][0..3];
                    const r: u8 = if (t[2] > t[1]) (if (t[2] > t[0]) 2 else 0) else @intFromBool(t[1] > t[0]);
                    val |= r << @intCast(2 * g);
                }
                out[byte] = val;
            }
        },
        else => {
            for (0..descriptor_size) |byte| {
                var val: u8 = 0;
                for (0..4) |g| {
                    const t = vals[byte * 16 + g * 4 ..][0..4];
                    var t0 = t[0];
                    var t2 = t[2];
                    var uu: u8 = 0;
                    var vv: u8 = 2;
                    if (t[1] > t0) {
                        t0 = t[1];
                        uu = 1;
                    }
                    if (t[3] > t2) {
                        t2 = t[3];
                        vv = 3;
                    }
                    const kk = if (t0 > t2) uu else vv;
                    val |= kk << @intCast(2 * g);
                }
                out[byte] = val;
            }
        },
    }
}

// ------------------------------------------------------------- pipeline

fn levelScale(opt: Options, level: usize) f32 {
    const e = @as(f64, @floatFromInt(level)) - @as(f64, @floatFromInt(opt.first_level));
    return @floatCast(std.math.pow(f64, opt.scale_factor, e));
}

const Stage = struct {
    levels: []const Level,
    masks: ?[]const Level,
    border: usize,
    opt: Options,
    per_level: []const usize,
    out: [][]KeyPoint,
    umax: []const i32,
    gpa: Allocator,
    failed: *std.atomic.Value(bool),
};

/// FAST, filtering, Harris ranking and orientation for one level.
fn levelKeypoints(st: Stage, li: usize) Error!void {
    const gpa = st.gpa;
    const l = st.levels[li];
    if (l.w == 0 or l.h == 0) return;
    const kps = try fast.detect(.serial, gpa, l.interior(st.border), .{ .threshold = st.opt.fast_threshold, .nonmax_suppression = true, .type = .type_9_16 });
    var keep = false;
    defer if (!keep) gpa.free(kps);
    var n = kps.len;
    if (st.masks) |ms| {
        const m = ms[li].interior(st.border);
        var j: usize = 0;
        for (kps[0..n]) |k| {
            if (m.at(@intFromFloat(k.x), @intFromFloat(k.y), 0) != 0) {
                kps[j] = k;
                j += 1;
            }
        }
        n = j;
    }
    n = kp_mod.runByImageBorder(kps[0..n], @intCast(l.w), @intCast(l.h), st.opt.edge_threshold);
    const want = st.per_level[li];
    n = try kp_mod.retainBest(gpa, kps[0..n], if (st.opt.score_type == .harris) 2 * want else want);
    const size = @as(f32, @floatFromInt(st.opt.patch_size)) * l.scale;
    for (kps[0..n]) |*k| {
        k.octave = @intCast(li);
        k.size = size;
    }
    if (st.opt.score_type == .harris) {
        for (kps[0..n]) |*k| harris(l, st.border, k);
        n = try kp_mod.retainBest(gpa, kps[0..n], want);
    }
    const half = st.opt.patch_size / 2;
    for (kps[0..n]) |*k| {
        k.angle = icAngle(l, st.border, roundEven(k.x), roundEven(k.y), st.umax, half);
        k.x *= l.scale;
        k.y *= l.scale;
    }
    if (n > 0) {
        const r = try gpa.realloc(kps, n);
        keep = true;
        st.out[li] = r;
    }
}

/// cv2.ORB_create(...).detectAndCompute(img, mask). `src` is u8 with 1, 3
/// (BGR) or 4 (BGRA) channels; `mask` (optional) is one-channel u8 of the
/// same size. Descriptors are computed when `descriptors` is true.
pub fn detectAndCompute(exec: Exec, gpa: Allocator, src: Image(u8), mask: ?Image(u8), opt: Options, descriptors: bool) Error!Features {
    if (!(opt.scale_factor > 1) or opt.nlevels == 0 or opt.patch_size < 2) return error.InvalidArgument;
    if (opt.wta_k < 2 or opt.wta_k > 4) return error.InvalidArgument;
    if (opt.nlevels > 64 or opt.first_level > 64 or opt.patch_size > 255 or opt.edge_threshold > 4000) return error.InvalidArgument;
    if (mask) |m| {
        if (m.channels != 1) return error.UnsupportedChannels;
        if (m.width != src.width or m.height != src.height) return error.SizeMismatch;
    }

    // Gray input.
    var gray_owned: ?Image(u8) = null;
    defer if (gray_owned) |*g| g.deinit(gpa);
    const gray: Image(u8) = switch (src.channels) {
        1 => src,
        3, 4 => blk: {
            gray_owned = try Image(u8).init(gpa, src.width, src.height, 1);
            imgproc.cvtColor(exec, src, gray_owned.?, if (src.channels == 3) .bgr2gray else .bgra2gray) catch return error.Unsupported;
            break :blk gray_owned.?;
        },
        else => return error.UnsupportedChannels,
    };

    const half = opt.patch_size / 2;
    const desc_patch: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(half)) * @sqrt(2.0)));
    const border: usize = @max(opt.edge_threshold, @max(desc_patch, 4)) + 1;
    const nl: usize = opt.nlevels;

    // Level geometry, and one allocation for every level (and mask level).
    const levels = try gpa.alloc(Level, nl);
    defer gpa.free(levels);
    @memset(levels, .{});
    const mlevels: ?[]Level = if (mask != null) try gpa.alloc(Level, nl) else null;
    defer if (mlevels) |m| gpa.free(m);
    var total_bytes: usize = 0;
    for (levels, 0..) |*l, i| {
        const sc = levelScale(opt, i);
        const inv: f32 = 1.0 / sc;
        l.scale = sc;
        const w = roundEven(@as(f32, @floatFromInt(gray.width)) * inv);
        const h = roundEven(@as(f32, @floatFromInt(gray.height)) * inv);
        if (w <= 0 or h <= 0) break;
        l.w = @intCast(w);
        l.h = @intCast(h);
        l.stride = l.w + 2 * border;
        total_bytes += l.stride * (l.h + 2 * border);
    }
    const slab = try gpa.alloc(u8, if (mask != null) 2 * total_bytes else total_bytes);
    defer gpa.free(slab);
    {
        var off: usize = 0;
        for (levels) |*l| {
            if (l.w == 0) break;
            const n = l.stride * (l.h + 2 * border);
            l.buf = slab[off..][0..n];
            off += n;
        }
        if (mlevels) |ms| for (ms, levels) |*m, l| {
            m.* = l;
            if (l.w == 0) continue;
            m.buf = slab[off..][0..l.buf.len];
            off += l.buf.len;
        };
    }

    // The pyramid: each level resized from the previous one (levels up to
    // first_level from the input).
    const first = opt.first_level;
    var prev: Image(u8) = gray;
    var mask_bin: ?Image(u8) = null;
    defer if (mask_bin) |*m| m.deinit(gpa);
    var prev_mask: Image(u8) = undefined;
    if (mask) |m| {
        mask_bin = try Image(u8).init(gpa, m.width, m.height, 1);
        for (0..m.height) |y| for (m.row(y), mask_bin.?.row(y)) |a, *b| {
            b.* = if (a > 0) 255 else 0;
        };
        prev_mask = mask_bin.?;
    }
    for (levels, 0..) |l, i| {
        if (l.w == 0) break;
        const cur = l.interior(border);
        if (i != first) {
            try resizeLinearExact(gpa, prev, cur);
        } else {
            for (0..cur.height) |y| @memcpy(cur.row(y), gray.row(y));
        }
        fillBorder(l, border);
        if (mlevels) |ms| {
            const mc = ms[i].interior(border);
            if (i != first) {
                try resizeLinearExact(gpa, prev_mask, mc);
                if (i > first) for (0..mc.height) |y| for (mc.row(y)) |*v| {
                    if (v.* <= 254) v.* = 0;
                };
            } else {
                for (0..mc.height) |y| @memcpy(mc.row(y), mask_bin.?.row(y));
            }
            fillMaskBorder(ms[i], border);
            if (i > first) prev_mask = mc;
        }
        if (i > first) prev = cur;
    }

    // Features per level: a geometric series summing to nfeatures.
    const per_level = try gpa.alloc(usize, nl);
    defer gpa.free(per_level);
    {
        const factor: f32 = @floatCast(1.0 / opt.scale_factor);
        const nf: f32 = @floatFromInt(opt.nfeatures);
        var desired: f32 = nf * (1 - factor) / (1 - @as(f32, @floatCast(std.math.pow(f64, factor, @floatFromInt(nl)))));
        var sum: usize = 0;
        for (0..nl - 1) |i| {
            per_level[i] = @intCast(@max(0, roundEven(desired)));
            sum += per_level[i];
            desired *= factor;
        }
        per_level[nl - 1] = if (opt.nfeatures > sum) opt.nfeatures - sum else 0;
    }

    const umax = try patchRows(gpa, half);
    defer gpa.free(umax);
    const outs = try gpa.alloc([]KeyPoint, nl);
    defer gpa.free(outs);
    @memset(outs, &.{});
    defer for (outs) |o| if (o.len > 0) gpa.free(o);
    var failed = std.atomic.Value(bool).init(false);
    const st: Stage = .{ .levels = levels, .masks = mlevels, .border = border, .opt = opt, .per_level = per_level, .out = outs, .umax = umax, .gpa = gpa, .failed = &failed };
    exec.parallelFor(nl, 1, st, struct {
        fn run(s: Stage, a: usize, b: usize) void {
            for (a..b) |li| levelKeypoints(s, li) catch s.failed.store(true, .monotonic);
        }
    }.run);
    if (failed.load(.monotonic)) return error.OutOfMemory;

    var total: usize = 0;
    for (outs) |o| total += o.len;
    const kps = try gpa.alloc(KeyPoint, total);
    errdefer gpa.free(kps);
    {
        var i: usize = 0;
        for (outs) |o| {
            @memcpy(kps[i..][0..o.len], o);
            i += o.len;
        }
    }
    if (!descriptors or total == 0) return .{ .keypoints = kps, .descriptors = try gpa.alloc(u8, 0) };

    // Blur every level in place (as OpenCV does: the padding keeps the
    // unblurred pixels). The keypoint stages above needed the raw levels.
    const blurred = levels;
    const BlurCtx = struct { src: []const Level, dst: []const Level, border: usize, gpa: Allocator, failed: *std.atomic.Value(bool) };
    exec.parallelFor(nl, 1, BlurCtx{ .src = levels, .dst = blurred, .border = border, .gpa = gpa, .failed = &failed }, struct {
        fn run(c: BlurCtx, a: usize, b: usize) void {
            for (a..b) |li| {
                if (c.src[li].w == 0) continue;
                blurLevel(c.gpa, c.src[li], c.dst[li], c.border) catch c.failed.store(true, .monotonic);
            }
        }
    }.run);
    if (failed.load(.monotonic)) return error.OutOfMemory;

    const pattern = try makePattern(gpa, opt.patch_size, opt.wta_k);
    defer gpa.free(pattern);
    var px: [512]f32 = @splat(0);
    var py: [512]f32 = @splat(0);
    for (pattern, 0..) |p, i| {
        px[i] = @floatFromInt(p[0]);
        py[i] = @floatFromInt(p[1]);
    }
    const npts = std.mem.alignForward(usize, pattern.len, 16);
    const desc = try gpa.alloc(u8, total * descriptor_size);
    errdefer gpa.free(desc);
    const DescCtx = struct { levels: []const Level, kps: []const KeyPoint, border: usize, px: []const f32, py: []const f32, wta_k: u8, out: []u8 };
    exec.parallelFor(total, 64, DescCtx{ .levels = blurred, .kps = kps, .border = border, .px = px[0..npts], .py = py[0..npts], .wta_k = opt.wta_k, .out = desc }, struct {
        fn run(c: DescCtx, a: usize, b: usize) void {
            for (a..b) |i| {
                const k = c.kps[i];
                const l = c.levels[@intCast(k.octave)];
                describe(l, c.border, k, 1.0 / l.scale, c.px, c.py, c.wta_k, c.out[i * descriptor_size ..][0..descriptor_size]);
            }
        }
    }.run);
    return .{ .keypoints = kps, .descriptors = desc };
}

// ------------------------------------------------------------------ tests

const testing = std.testing;

test "fastAtan2 follows atan2" {
    const pts = [_][2]f32{ .{ 1, 0 }, .{ 1, 1 }, .{ 0, 1 }, .{ -1, 1 }, .{ -1, -1 }, .{ 3, -7 }, .{ -100, 3 } };
    for (pts) |p| {
        var e = std.math.atan2(p[1], p[0]) * 180.0 / std.math.pi;
        if (e < 0) e += 360;
        try testing.expectApproxEqAbs(e, fastAtan2(p[1], p[0]), 0.02);
    }
}

test "OpenCV's Gaussian kernel" {
    // getGaussianKernel(7, 2): exp(-x^2 / 8), normalised, rounded to f32.
    var k: [7]f64 = undefined;
    var sum: f64 = 0;
    for (&k, 0..) |*v, i| {
        const x: f64 = @as(f64, @floatFromInt(i)) - 3;
        v.* = @exp(-x * x / 8.0);
        sum += v.*;
    }
    for (k, gauss7) |v, g| try testing.expectApproxEqRel(@as(f32, @floatCast(v / sum)), g, 1e-6);
}

test "umax of the 31x31 patch" {
    const u = try patchRows(testing.allocator, 15);
    defer testing.allocator.free(u);
    try testing.expectEqualSlices(i32, &.{ 15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3 }, u[0..16]);
}

test "ORB finds keypoints and is deterministic across threads" {
    const gpa = testing.allocator;
    var img = try Image(u8).init(gpa, 320, 240, 1);
    defer img.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(1);
    for (0..240) |y| for (0..320) |x| {
        const v: u32 = @truncate(((x / 6) *% 2654435761) ^ ((y / 6) *% 40503));
        img.set(x, y, 0, @truncate((v >> 9) +% prng.random().int(u8) / 32));
    };
    var a = try detectAndCompute(.serial, gpa, img, null, .{}, true);
    defer a.deinit(gpa);
    try testing.expect(a.keypoints.len > 100);
    try testing.expect(a.keypoints.len <= 520);
    if (!core.pool.threads_supported) return;
    var pool = core.Pool.init(gpa, std.Io.Threaded.global_single_threaded.io(), 4);
    defer pool.deinit();
    var b = try detectAndCompute(.on(&pool), gpa, img, null, .{}, true);
    defer b.deinit(gpa);
    try testing.expectEqualSlices(KeyPoint, a.keypoints, b.keypoints);
    try testing.expectEqualSlices(u8, a.descriptors, b.descriptors);
}
