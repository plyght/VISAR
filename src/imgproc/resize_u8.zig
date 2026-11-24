//! 8-bit resize kernels with OpenCV's fixed-point arithmetic, for 1 to 4
//! interleaved channels: exact 2x2 averaging, and separable K-tap
//! resampling (bilinear K = 2, bicubic K = 4) with a row cache so every
//! source row is resampled horizontally once per worker.

const std = @import("std");
const core = @import("../core/core.zig");
const simd = core.simd;
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const vec = core.vec;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

/// AVX-512 VBMI byte permutes drive the bilinear horizontal pass.
pub const has_vbmi = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi);

/// AArch64 (LLVM backend): NEON tbl over four registers is the same 64-byte
/// table lookup, 16 bytes at a time.
const arm_tbl = core.simd.llvm_intrinsics and builtin.cpu.arch == .aarch64;

/// x86 with AVX2 but no VBMI (LLVM backend): the 64-byte lookup is four
/// in-lane vpshufb, one per 16-byte chunk, with per-chunk indices that are
/// 0x80 (zero) outside the chunk.
const avx2_shuf = !has_vbmi and builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

/// Targets with a fast 64-byte byte permute for the group path.
pub const fast_permute = has_vbmi or arm_tbl or avx2_shuf;
const planar = arm_tbl;
const idx_len = if (avx2_shuf) 128 else 64;

const avx2 = struct {
    const B = @Vector(32, u8);
    // Inline asm rather than LLVM intrinsics: independent of the calling
    // convention (Windows passes vectors by reference to extern functions).
    inline fn pshufb(t: B, i: B) B {
        return asm ("vpshufb %[i], %[t], %[r]"
            : [r] "=x" (-> B),
            : [t] "x" (t),
              [i] "x" (i),
        );
    }
    inline fn pmaddwd(a: @Vector(16, i16), b: @Vector(16, i16)) @Vector(8, i32) {
        return asm ("vpmaddwd %[b], %[a], %[r]"
            : [r] "=x" (-> @Vector(8, i32)),
            : [a] "x" (a),
              [b] "x" (b),
        );
    }
};

const neon = struct {
    const V = @Vector(16, u8);
    extern fn @"llvm.aarch64.neon.tbl4.v16i8"(V, V, V, V, V) V;
    extern fn @"llvm.aarch64.neon.tbl2.v16i8"(V, V, V) V;
    extern fn @"llvm.aarch64.neon.tbl1.v16i8"(V, V) V;
    extern fn @"llvm.aarch64.neon.uaddlp.v8i16.v16i8"(V) @Vector(8, u16);
    extern fn @"llvm.aarch64.neon.addp.v8i16"(@Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
    extern fn @"llvm.aarch64.neon.sqdmulh.v8i16"(@Vector(8, i16), @Vector(8, i16)) @Vector(8, i16);
    extern fn @"llvm.aarch64.neon.uhsub.v8i16"(@Vector(8, u16), @Vector(8, u16)) @Vector(8, u16);
    extern fn @"llvm.aarch64.neon.sqrshrun.v8i8"(@Vector(8, i16), i32) @Vector(8, u8);
    inline fn tbl(comptime k: usize, t: [k]V, idx: V) V {
        return switch (k) {
            1 => @"llvm.aarch64.neon.tbl1.v16i8"(t[0], idx),
            2 => @"llvm.aarch64.neon.tbl2.v16i8"(t[0], t[1], idx),
            4 => @"llvm.aarch64.neon.tbl4.v16i8"(t[0], t[1], t[2], t[3], idx),
            else => unreachable,
        };
    }
    inline fn tbl4(t: [4]V, idx: V) V {
        return @"llvm.aarch64.neon.tbl4.v16i8"(t[0], t[1], t[2], t[3], idx);
    }
};

/// Inline AVX-512 assembly needs the LLVM backend (the self-hosted x86
/// backend used for Debug builds lacks the vector register constraint).
/// The `v` register class (zmm) also needs AVX-512 on the target.
const use_asm = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);
const avx512bw = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bw);
const avx512f = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f);

/// `n` destination pixels of a bilinear row (up to 16 samples): a 64-byte
/// source window at `base`, the byte index of each (sample, tap) pair, and
/// the matching 16-bit weights. Samples are packed, so all 16 output lanes
/// are used for 1, 2 and 4 channels (15 for 3). `ok` is false when the
/// window does not fit.
///
/// Layout: x86 (vpermb + vpmaddwd) interleaves the taps, index in the even
/// byte of each 16-bit lane; on NEON (`planar`) the first 16 indices are
/// tap 0, the next 16 tap 1, so two tbl lookups fetch the samples, and `w`
/// holds the weights split as bytes (w >> 4 for taps 0 and 1, then w & 15)
/// for u8 x u8 widening multiplies (see hrowGroup).
pub const Group = struct {
    idx: [idx_len]u8 align(64),
    w: [32]i16 align(64),
    base: u32,
    n: u8,
    /// Bytes of the window actually used (16, 32 or 64; the same for all
    /// groups of a row): fewer table registers when the pixels are close.
    win: u8,
    ok: bool,
    /// NEON only: the group is 16 consecutive samples of a row whose taps
    /// all have w0 + w1 = 2048 and w1 < 2048 (see sumGroups and
    /// hrowSumNeon), not `n` whole pixels.
    sum: bool = false,
    /// Sum groups: how many groups from this one on are `ok` in a row.
    run: u32 = 0,
};

/// Builds the groups for a bilinear row of `rowlen` source samples. Uses
/// as many pixels per group as fit 16 samples, fewer when strong
/// downscaling would make most windows overflow.
pub fn groups(gpa: Allocator, xt: []const Tap(2), cn: usize, rowlen: usize) Allocator.Error![]Group {
    if (planar) if (try sumGroups(gpa, xt, cn, rowlen)) |g| return g;
    var n: usize = 16 / cn;
    while (n > 1) : (n -= 1) {
        // Fits when the span of n pixels (plus the right tap) is in 64 bytes.
        var bad: usize = 0;
        var gi: usize = 0;
        while ((gi + 1) * n <= xt.len) : (gi += 1) {
            const span = xt[gi * n + n - 1].o[1] + cn - xt[gi * n].o[0];
            if (span > 64) bad += 1;
        }
        if (bad * 16 <= gi) break; // at most 1 in 16 groups on the slow path
    }
    // The smallest window that holds every group that fits at all (vpermb
    // always takes 64 bytes).
    var win: usize = 16;
    if (planar or avx2_shuf) {
        var gi: usize = 0;
        while ((gi + 1) * n <= xt.len) : (gi += 1) {
            const span = xt[gi * n + n - 1].o[1] + cn - xt[gi * n].o[0];
            const need: usize = if (span <= 16) 16 else if (span <= 32) 32 else 64;
            if (span <= 64) win = @max(win, need);
        }
    } else win = 64;
    const g = try gpa.alloc(Group, xt.len / n);
    for (g, 0..) |*gr, gi| {
        const base = xt[gi * n].o[0];
        gr.base = base;
        gr.n = @intCast(n);
        gr.win = @intCast(win);
        gr.ok = base + win <= rowlen;
        gr.idx = @splat(if (avx2_shuf) 0x80 else 0);
        gr.w = @splat(0);
        for (0..n) |p| {
            const t = xt[gi * n + p];
            for (0..cn) |l| for (0..2) |tap| {
                const sample = p * cn + l;
                const off = t.o[tap] + l - base;
                if (planar) {
                    if (off >= win) gr.ok = false else gr.idx[tap * 16 + sample] = @intCast(off);
                    const wb: *[64]u8 = @ptrCast(&gr.w);
                    wb[tap * 16 + sample] = @intCast(t.w[tap] >> 4);
                    wb[32 + tap * 16 + sample] = @intCast(t.w[tap] & 15);
                } else if (avx2_shuf) {
                    // Byte `pair * 2` of each chunk's 32 indices, so the
                    // result is the interleaved pairs as 32 bytes.
                    const pair = sample * 2 + tap;
                    if (off >= win) gr.ok = false else gr.idx[(off / 16) * 32 + pair] = @intCast(off % 16);
                    gr.w[pair] = @intCast(t.w[tap]);
                } else {
                    const pair = sample * 2 + tap;
                    if (off >= 64) gr.ok = false else gr.idx[2 * pair] = @intCast(off);
                    gr.w[pair] = @intCast(t.w[tap]);
                }
            };
        }
    }
    return g;
}

/// NEON groups of 16 consecutive samples (whole pixels or not), when
/// every tap pair sums to 2048 and the windows fit: then
/// (s0 w0 + s1 w1) >> 4 = 128 s0 + floor((s1 - s0) w1 / 16), one sqdmulh
/// per 8 samples (hrowSumNeon). Null when the row does not qualify.
fn sumGroups(gpa: Allocator, xt: []const Tap(2), cn: usize, rowlen: usize) Allocator.Error!?[]Group {
    return switch (cn) {
        inline 1, 2, 3, 4 => |c| sumGroupsN(c, gpa, xt, rowlen),
        else => null,
    };
}

fn sumGroupsN(comptime cn: usize, gpa: Allocator, xt: []const Tap(2), rowlen: usize) Allocator.Error!?[]Group {
    for (xt) |t| if (t.w[0] + t.w[1] != coef_scale or t.w[1] >= coef_scale or t.w[1] < 0) return null;
    const ng = xt.len * cn / 16;
    if (ng == 0) return null;
    const g = try gpa.alloc(Group, ng);
    // Pass 1: each group's window start and span.
    var bad: usize = 0;
    var win: usize = 16;
    for (g, 0..) |*gr, gi| {
        var lo: usize = std.math.maxInt(usize);
        var hi: usize = 0;
        inline for (0..16) |l| {
            const t = xt[(gi * 16 + l) / cn];
            const ch = (gi * 16 + l) % cn;
            lo = @min(lo, t.o[0] + ch);
            hi = @max(hi, t.o[1] + ch + 1);
        }
        const span = hi - lo;
        if (span > 64) bad += 1 else win = @max(win, @as(usize, if (span <= 16) 16 else if (span <= 32) 32 else 64));
        gr.base = @intCast(lo);
    }
    if (bad * 16 > ng) {
        gpa.free(g);
        return null;
    }
    // Pass 2: indices and weights.
    for (g, 0..) |*gr, gi| {
        const base: usize = gr.base;
        gr.* = .{ .idx = @splat(0xff), .w = @splat(0), .base = @intCast(base), .n = 0, .win = @intCast(win), .ok = base + win <= rowlen, .sum = true };
        inline for (0..16) |l| {
            const t = xt[(gi * 16 + l) / cn];
            const ch = (gi * 16 + l) % cn;
            // Lane l of the 16-bit tap vectors takes its sample as the high
            // byte (s << 8); index 0xff (out of range) zeroes the low byte.
            inline for (0..2) |tap| {
                const off = t.o[tap] + ch - base;
                if (off >= win) gr.ok = false else gr.idx[tap * 32 + 2 * l + 1] = @intCast(off);
            }
            gr.w[l] = @intCast(t.w[1] << 4);
        }
    }
    var run: u32 = 0;
    var i = g.len;
    while (i > 0) {
        i -= 1;
        run = if (g[i].ok) run + 1 else 0;
        g[i].run = run;
    }
    return g;
}

inline fn permb(tab: @Vector(64, u8), idx: @Vector(64, u8)) @Vector(64, u8) {
    if (comptime arm_tbl) {
        const ta: [64]u8 = tab;
        const ia: [64]u8 = idx;
        const t: [4]neon.V = .{ ta[0..16].*, ta[16..32].*, ta[32..48].*, ta[48..64].* };
        var r: [64]u8 = undefined;
        inline for (0..4) |q| r[q * 16 ..][0..16].* = neon.tbl4(t, ia[q * 16 ..][0..16].*);
        return r;
    }
    if (comptime !use_asm) {
        const t: [64]u8 = tab;
        const i: [64]u8 = idx;
        var r: [64]u8 = undefined;
        for (&r, i) |*o, k| o.* = t[k & 63];
        return r;
    }
    return asm ("vpermb %[tab], %[idx], %[ret]"
        : [ret] "=v" (-> @Vector(64, u8)),
        : [tab] "v" (tab),
          [idx] "v" (idx),
    );
}

inline fn maddwd(a: @Vector(32, i16), b: @Vector(32, i16)) @Vector(16, i32) {
    if (comptime !use_asm) {
        // Widening multiply, then add even and odd products (NEON:
        // smull/smull2 + uzp + add).
        const p = @as(@Vector(32, i32), a) * @as(@Vector(32, i32), b);
        const ev = comptime blk: {
            var m: [16]i32 = undefined;
            for (0..16) |k| m[k] = 2 * k;
            break :blk m;
        };
        const od = comptime blk: {
            var m: [16]i32 = undefined;
            for (0..16) |k| m[k] = 2 * k + 1;
            break :blk m;
        };
        return @shuffle(i32, p, undefined, ev) + @shuffle(i32, p, undefined, od);
    }
    return asm ("vpmaddwd %[b], %[a], %[ret]"
        : [ret] "=v" (-> @Vector(16, i32)),
        : [a] "v" (a),
          [b] "v" (b),
    );
}

/// NEON bilinear group over R rows sharing the group's indices and
/// weights. (s0 w0 + s1 w1) >> 4 with w = 16 hi + lo (weights are
/// 0..2048) is s0 hi0 + s1 hi1 + ((s0 lo0 + s1 lo1) >> 4), all in u16
/// (umull, umlal, usra); the result is below 2^15, so it is the same i16.
inline fn hrowGroupNeon(comptime R: usize, comptime W: usize, s: [R][]const u8, out: [R][]i16, g: *const Group) void {
    const U = @Vector(16, u16);
    const ix0: neon.V = g.idx[0..16].*;
    const ix1: neon.V = g.idx[16..32].*;
    const wb: *const [64]u8 = @ptrCast(&g.w);
    const h0: U = @as(@Vector(16, u8), wb[0..16].*);
    const h1: U = @as(@Vector(16, u8), wb[16..32].*);
    const l0: U = @as(@Vector(16, u8), wb[32..48].*);
    const l1: U = @as(@Vector(16, u8), wb[48..64].*);
    inline for (0..R) |r| {
        const w = s[r][g.base..][0..W];
        var t: [W / 16]neon.V = undefined;
        inline for (0..W / 16) |k| t[k] = w[k * 16 ..][0..16].*;
        const s0: U = neon.tbl(W / 16, t, ix0);
        const s1: U = neon.tbl(W / 16, t, ix1);
        const hi = s0 * h0 + s1 * h1;
        const lo = s0 * l0 + s1 * l1;
        out[r][0..16].* = @bitCast(hi + (lo >> @splat(4)));
    }
}

/// NEON bilinear sum group (see sumGroups) over R rows: with a = s0 << 8
/// and b = s1 << 8 in u16 lanes, uhsub(b, a) = (s1 - s0) << 7 exactly as
/// i16, sqdmulh by w1 << 4 gives floor((s1 - s0) w1 / 16), and adding
/// a >> 1 = 128 s0 (usra) gives (s0 w0 + s1 w1) >> 4 for w0 = 2048 - w1.
inline fn hrowSumNeon(comptime R: usize, comptime W: usize, s: [R][]const u8, out: [R][*]i16, g: *const Group) void {
    const H = @Vector(8, u16);
    var ix: [4]neon.V = undefined;
    inline for (0..4) |k| ix[k] = g.idx[k * 16 ..][0..16].*;
    const wv = [2]@Vector(8, i16){ g.w[0..8].*, g.w[8..16].* };
    const base = g.base;
    inline for (0..R) |r| {
        const w = s[r][base..][0..W];
        var t: [W / 16]neon.V = undefined;
        inline for (0..W / 16) |k| t[k] = w[k * 16 ..][0..16].*;
        inline for (0..2) |h| {
            const a: H = @bitCast(neon.tbl(W / 16, t, ix[h]));
            const b: H = @bitCast(neon.tbl(W / 16, t, ix[2 + h]));
            const d: @Vector(8, i16) = @bitCast(neon.@"llvm.aarch64.neon.uhsub.v8i16"(b, a));
            const m: H = @bitCast(neon.@"llvm.aarch64.neon.sqdmulh.v8i16"(d, wv[h]));
            out[r][8 * h ..][0..8].* = @bitCast(m +% (a >> @splat(1)));
        }
    }
}

/// Bilinear horizontal pass for one group: 16 samples (some unused) are
/// written, so `out` needs room past the group's own samples.
inline fn hrowGroup(comptime W: usize, s: []const u8, out: []i16, g: Group) void {
    if (comptime planar) return hrowGroupNeon(1, W, .{s}, .{out}, &g);
    if (comptime avx2_shuf) {
        const w = s[g.base..][0..W];
        var bytes: avx2.B = @splat(0);
        inline for (0..W / 16) |k| {
            const c: @Vector(16, u8) = w[k * 16 ..][0..16].*;
            const both = @shuffle(u8, c, undefined, dup16);
            bytes |= avx2.pshufb(both, g.idx[k * 32 ..][0..32].*);
        }
        const a: @Vector(32, i16) = @intCast(bytes);
        const wv: @Vector(32, i16) = g.w;
        const lo = avx2.pmaddwd(@shuffle(i16, a, undefined, lo16), @shuffle(i16, wv, undefined, lo16));
        const hi = avx2.pmaddwd(@shuffle(i16, a, undefined, hi16), @shuffle(i16, wv, undefined, hi16));
        const acc = @shuffle(i32, lo, hi, [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, -1, -2, -3, -4, -5, -6, -7, -8 }) >> @splat(4);
        out[0..16].* = @as(@Vector(16, i16), @truncate(acc));
        return;
    }
    const win: @Vector(64, u8) = s[g.base..][0..64].*;
    const bytes = permb(win, g.idx[0..64].*);
    const a: @Vector(32, i16) = @bitCast(@as(@Vector(32, u16), @bitCast(bytes)) & @as(@Vector(32, u16), @splat(0xff)));
    const acc = maddwd(a, g.w) >> @splat(4);
    out[0..16].* = @as(@Vector(16, i16), @truncate(acc));
}

const lo16 = blk: {
    var m: [16]i32 = undefined;
    for (0..16) |k| m[k] = k;
    break :blk m;
};
const hi16 = blk: {
    var m: [16]i32 = undefined;
    for (0..16) |k| m[k] = 16 + k;
    break :blk m;
};

const dup16 = blk: {
    var m: [32]i32 = undefined;
    for (0..32) |k| m[k] = k % 16;
    break :blk m;
};

const coef_bits = 11;
const coef_scale = 1 << coef_bits;

/// Exact halving in both directions: OpenCV's bilinear coefficients are then
/// all 1024 and its fixed-point rounding reduces to (a + b + c + d + 2) >> 2,
/// the same as INTER_AREA's integer-factor path.
pub fn half(exec: Exec, src: Image(u8), dst: Image(u8)) void {
    const Ctx = struct { s: Image(u8), d: Image(u8) };
    exec.parallelFor(dst.height, 16, Ctx{ .s = src, .d = dst }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| for (y0..y1) |y| halfRow(cn, c.s.row(2 * y), c.s.row(2 * y + 1), c.d.row(y)),
                else => unreachable,
            }
        }
    }.run);
}

fn pairMask(comptime cn: usize, comptime odd: bool) [16 * cn]i32 {
    @setEvalBranchQuota(10000);
    var m: [16 * cn]i32 = undefined;
    for (0..16) |x| for (0..cn) |c| {
        m[x * cn + c] = @intCast((2 * x + @intFromBool(odd)) * cn + c);
    };
    return m;
}

fn halfRow(comptime cn: usize, a: []const u8, b: []const u8, d: []u8) void {
    const L = 32 * cn; // input samples per step (16 output pixels)
    const VI = @Vector(L, u16);
    const VO = @Vector(16 * cn, u16);
    const w = d.len / cn;
    var x: usize = 0;
    if (comptime planar) {
        // NEON: deinterleave (ldN), add neighbouring pixels of a channel
        // pairwise into u16 (uaddlp), round-shift-narrow, interleave (stN).
        const H = @Vector(8, u16);
        while (x + 16 <= w) : (x += 16) {
            const so = 2 * x * cn;
            var half8: [2][cn]@Vector(8, u8) = undefined;
            inline for (0..2) |h| {
                const ca = vec.load(cn, a.ptr + so + h * 16 * cn);
                const cb = vec.load(cn, b.ptr + so + h * 16 * cn);
                inline for (0..cn) |ch| {
                    const sum: H = neon.@"llvm.aarch64.neon.uaddlp.v8i16.v16i8"(ca[ch]) + neon.@"llvm.aarch64.neon.uaddlp.v8i16.v16i8"(cb[ch]);
                    half8[h][ch] = @truncate((sum + @as(H, @splat(2))) >> @splat(2));
                }
            }
            var out: [cn]vec.V16 = undefined;
            inline for (0..cn) |ch| out[ch] = std.simd.join(half8[0][ch], half8[1][ch]);
            vec.store(cn, out, d.ptr + x * cn);
        }
    }
    while (x + 16 <= w) : (x += 16) {
        const so = 2 * x * cn;
        const va: @Vector(L, u8) = a[so..][0..L].*;
        const vb: @Vector(L, u8) = b[so..][0..L].*;
        const v = @as(VI, va) + @as(VI, vb);
        const e: VO = @shuffle(u16, v, undefined, pairMask(cn, false));
        const o: VO = @shuffle(u16, v, undefined, pairMask(cn, true));
        const r: @Vector(16 * cn, u8) = @truncate((e + o + @as(VO, @splat(2))) >> @splat(2));
        d[x * cn ..][0 .. 16 * cn].* = r;
    }
    while (x < w) : (x += 1) {
        inline for (0..cn) |ch| {
            const i = 2 * x * cn + ch;
            d[x * cn + ch] = @intCast((@as(u32, a[i]) + a[i + cn] + b[i] + b[i + cn] + 2) >> 2);
        }
    }
}

pub fn Tap(comptime K: usize) type {
    return struct { o: [K]u32, w: [K]i32 };
}

pub const Kind = enum { linear, cubic };

/// Separable resampling. `xt` holds per destination pixel the source sample
/// offsets of channel 0 (non-decreasing) and fixed-point weights summing to
/// 2048; `yt` the same for rows (offsets are row indices).
pub fn separable(comptime K: usize, comptime kind: Kind, exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), xt: []const Tap(K), yt: []const Tap(K), gs: []const Group) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), xt: []const Tap(K), yt: []const Tap(K), gpa: Allocator, gs: []const Group };
    exec.parallelFor(dst.height, 8, Ctx{ .s = src, .d = dst, .xt = xt, .yt = yt, .gpa = gpa, .gs = gs }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| rows(K, kind, cn, c, y0, y1),
                else => unreachable,
            }
        }
    }.run);
}

fn rows(comptime K: usize, comptime kind: Kind, comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    if (K == 2 and kind == .linear and planar and c.gs.len > 0) return rowsNeon(cn, c, y0, y1);
    const n = c.d.rowLen();
    const stride = n + 16; // room for the 16-lane store of the last group
    // Bilinear rows keep acc >> 4 (all the vertical step uses) as i16.
    const E = if (kind == .linear) i16 else i32;
    const buf = c.gpa.alloc(E, K * stride) catch @panic("out of memory");
    defer c.gpa.free(buf);
    var tag: [K]i64 = @splat(-1);
    for (y0..y1) |y| {
        const t = c.yt[y];
        var ptr: [K][]const E = undefined;
        // Which buffers are still needed by this row.
        var keep: [K]bool = @splat(false);
        var have: [K]?usize = @splat(null);
        for (0..K) |k| {
            for (0..K) |b| if (tag[b] == t.o[k]) {
                have[k] = b;
                keep[b] = true;
            };
        }
        for (0..K) |k| {
            if (have[k] == null) {
                // Rows repeat at the edges: reuse one filled for an earlier tap.
                for (0..k) |j| if (t.o[j] == t.o[k]) {
                    have[k] = have[j];
                };
            }
            if (have[k] == null) {
                var b: usize = 0;
                while (keep[b]) b += 1;
                keep[b] = true;
                tag[b] = t.o[k];
                hrow(K, cn, E, c.s.row(t.o[k]), buf[b * stride ..][0..stride], c.xt, c.gs);
                have[k] = b;
            }
            ptr[k] = buf[have[k].? * stride ..][0..n];
        }
        switch (kind) {
            .linear => vlinear(ptr[0], ptr[1], t.w, c.d.row(y)),
            .cubic => vcubic(ptr, t.w, c.d.row(y)),
        }
    }
}

fn hrow(comptime K: usize, comptime cn: usize, comptime E: type, s: []const u8, out: []E, xt: []const Tap(K), gs: []const Group) void {
    if (E == i16 and fast_permute and gs.len > 0) {
        const gn: usize = gs[0].n;
        switch (gs[0].win) {
            inline 16, 32, 64 => |W| {
                var gi: usize = 0;
                if (comptime planar) {
                    // Two groups per step: the NEON group is only ~12
                    // arithmetic instructions, so loop overhead matters.
                    const step = gn * cn;
                    while (gi + 2 <= gs.len and gs[gi].ok and gs[gi + 1].ok) : (gi += 2) {
                        hrowGroup(W, s, out[gi * step ..], gs[gi]);
                        hrowGroup(W, s, out[(gi + 1) * step ..], gs[gi + 1]);
                    }
                }
                for (gs[gi..], gi..) |g, k| {
                    if (g.ok) hrowGroup(W, s, out[k * gn * cn ..], g) else hrowPixels(K, cn, E, s, out, xt, k * gn, k * gn + gn);
                }
            },
            else => unreachable,
        }
        return hrowPixels(K, cn, E, s, out, xt, gs.len * gn, xt.len);
    }
    hrowPixels(K, cn, E, s, out, xt, 0, xt.len);
}

/// NEON bilinear rows with a four-row cache. Source rows are resampled
/// horizontally two at a time: when an output row needs only one new row,
/// the next new row the following output row needs goes with it, so each
/// group's indices and weights are loaded once for two rows.
fn rowsNeon(comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    const S = 4;
    const n = c.d.rowLen();
    const stride = n + 16;
    const buf = c.gpa.alloc(i16, S * stride) catch @panic("out of memory");
    defer c.gpa.free(buf);
    var tag: [S]i64 = @splat(-1);
    const Find = struct {
        fn slot(tg: [S]i64, row: i64) ?usize {
            for (tg, 0..) |v, i| if (v == row) return i;
            return null;
        }
    };
    for (y0..y1) |y| {
        const t = c.yt[y];
        const need = [2]i64{ t.o[0], t.o[1] };
        var todo: [2]i64 = undefined;
        var nt: usize = 0;
        for (need, 0..) |r, k| {
            if (Find.slot(tag, r) == null and !(k == 1 and r == need[0])) {
                todo[nt] = r;
                nt += 1;
            }
        }
        if (nt == 1 and y + 1 < y1) {
            for (c.yt[y + 1].o) |r| {
                if (r != todo[0] and Find.slot(tag, r) == null) {
                    todo[1] = r;
                    nt = 2;
                    break;
                }
            }
        }
        // Evict the lowest rows the current output row does not read.
        var dst: [2][]i16 = undefined;
        for (0..nt) |k| {
            var best: ?usize = null;
            for (tag, 0..) |v, i| {
                if (v == need[0] or v == need[1] or (k == 1 and v == todo[0])) continue;
                if (best == null or v < tag[best.?]) best = i;
            }
            tag[best.?] = todo[k];
            dst[k] = buf[best.? * stride ..][0..stride];
        }
        switch (nt) {
            2 => hrowNeon(2, cn, .{ c.s.row(@intCast(todo[0])), c.s.row(@intCast(todo[1])) }, dst, c.xt, c.gs),
            1 => hrowNeon(1, cn, .{c.s.row(@intCast(todo[0]))}, .{dst[0]}, c.xt, c.gs),
            else => {},
        }
        const p0 = buf[Find.slot(tag, need[0]).? * stride ..][0..n];
        const p1 = buf[Find.slot(tag, need[1]).? * stride ..][0..n];
        vlinear(p0, p1, t.w, c.d.row(y));
    }
}

/// R rows of the NEON bilinear horizontal pass over the groups.
fn hrowNeon(comptime R: usize, comptime cn: usize, s: [R][]const u8, out: [R][]i16, xt: []const Tap(2), gs: []const Group) void {
    switch (gs[0].win) {
        inline 16, 32, 64 => |W| {
            if (gs[0].sum) {
                var o: [R][*]i16 = undefined;
                inline for (0..R) |r| o[r] = out[r].ptr;
                var k: usize = 0;
                while (k < gs.len) {
                    if (gs[k].ok) {
                        // A run of windows that fit: no per-group checks.
                        for (gs[k..][0..gs[k].run]) |*g| {
                            hrowSumNeon(R, W, s, o, g);
                            inline for (0..R) |r| o[r] += 16;
                        }
                        k += gs[k].run;
                    } else {
                        for (0..R) |r| hrowSamples(cn, s[r], out[r], xt, k * 16, k * 16 + 16);
                        inline for (0..R) |r| o[r] += 16;
                        k += 1;
                    }
                }
                for (0..R) |r| hrowSamples(cn, s[r], out[r], xt, gs.len * 16, xt.len * cn);
                return;
            }
            if (R == 1) return hrow(2, cn, i16, s[0], out[0], xt, gs);
            const gn: usize = gs[0].n;
            const step = gn * cn;
            for (gs, 0..) |*g, k| {
                if (g.ok) {
                    var o: [R][]i16 = undefined;
                    inline for (0..R) |r| o[r] = out[r][k * step ..];
                    hrowGroupNeon(R, W, s, o, g);
                } else for (0..R) |r| hrowPixels(2, cn, i16, s[r], out[r], xt, k * gn, k * gn + gn);
            }
            for (0..R) |r| hrowPixels(2, cn, i16, s[r], out[r], xt, gs.len * gn, xt.len);
        },
        else => unreachable,
    }
}

/// Samples [from, to) of a bilinear row, one at a time.
noinline fn hrowSamples(comptime cn: usize, s: []const u8, out: []i16, xt: []const Tap(2), from: usize, to: usize) void {
    for (from..to) |i| {
        const t = xt[i / cn];
        const ch = i % cn;
        const acc = @as(i32, s[t.o[0] + ch]) * t.w[0] + @as(i32, s[t.o[1] + ch]) * t.w[1];
        out[i] = @intCast(acc >> 4);
    }
}

fn hrowPixels(comptime K: usize, comptime cn: usize, comptime E: type, s: []const u8, out: []E, xt: []const Tap(K), x0: usize, x1: usize) void {
    const shift = if (E == i16) 4 else 0;
    const V4 = @Vector(4, i32);
    for (xt[x0..x1], x0..) |t, x| {
        if (t.o[K - 1] + 4 <= s.len) {
            var acc: V4 = @splat(0);
            inline for (0..K) |k| {
                const p: @Vector(4, u8) = s[t.o[k]..][0..4].*;
                acc += @as(V4, p) * @as(V4, @as(@Vector(4, i16), @splat(@as(i16, @intCast(t.w[k])))));
            }
            out[x * cn ..][0..4].* = @as(@Vector(4, E), @truncate(acc >> @splat(shift)));
        } else {
            inline for (0..cn) |ch| {
                var acc: i32 = 0;
                inline for (0..K) |k| acc += @as(i32, s[t.o[k] + ch]) * t.w[k];
                out[x * cn + ch] = @truncate(acc >> shift);
            }
        }
    }
}

inline fn mulhi(a: @Vector(32, i16), b: @Vector(32, i16)) @Vector(32, i16) {
    if (use_asm and avx512bw)
        return asm ("vpmulhw %[b], %[a], %[ret]"
            : [ret] "=v" (-> @Vector(32, i16)),
            : [a] "v" (a),
              [b] "v" (b),
        );
    const W = @Vector(32, i32);
    return @truncate((@as(W, a) * @as(W, b)) >> @splat(16));
}

/// OpenCV's VResizeLinearVec_32s8u: rows hold (sum >> 4) as i16, each is
/// scaled with a 16-bit high multiply by its weight, then (+2) >> 2.
fn vlinear(r0: []const i16, r1: []const i16, b: [2]i32, d: []u8) void {
    const N = 32;
    const V = @Vector(N, i16);
    const b0: V = @splat(@intCast(b[0]));
    const b1: V = @splat(@intCast(b[1]));
    var x: usize = 0;
    if (comptime arm_tbl) {
        // mulhi(a, b) = sqdmulh(a, b) >> 1: (2ab) >> 16 never saturates
        // for |a| < 2^15 and |b| <= 2^11, and the floors nest. With b even
        // it is sqdmulh(a, b / 2). (t + 2) >> 2 and the clamp to u8 are
        // one sqrshrun.
        x = if ((b[0] | b[1]) & 1 == 0) vlinearNeon(true, r0, r1, b, d) else vlinearNeon(false, r0, r1, b, d);
    }
    while (x + N <= d.len) : (x += N) {
        const a: V = r0[x..][0..N].*;
        const c: V = r1[x..][0..N].*;
        const m0 = mulhi(a, b0);
        const m1 = mulhi(c, b1);
        const t = (m0 + m1 + @as(V, @splat(2))) >> @splat(2);
        const cl = @max(@min(t, @as(V, @splat(255))), @as(V, @splat(0)));
        d[x..][0..N].* = simd.narrow(u8, cl);
    }
    while (x < d.len) : (x += 1) {
        const t = ((@as(i32, r0[x]) * b[0]) >> 16) + ((@as(i32, r1[x]) * b[1]) >> 16);
        d[x] = sat.cast(u8, (t + 2) >> 2);
    }
}

fn vlinearNeon(comptime even: bool, r0: []const i16, r1: []const i16, b: [2]i32, d: []u8) usize {
    const H = @Vector(8, i16);
    const sh = @intFromBool(even);
    const c0: H = @splat(@intCast(b[0] >> sh));
    const c1: H = @splat(@intCast(b[1] >> sh));
    var x: usize = 0;
    while (x + 32 <= d.len) : (x += 32) {
        // All loads before the stores (u8 stores may alias the rows).
        var a: [4]H = undefined;
        var c: [4]H = undefined;
        inline for (0..4) |k| {
            a[k] = r0[x + 8 * k ..][0..8].*;
            c[k] = r1[x + 8 * k ..][0..8].*;
        }
        var o: [4]@Vector(8, u8) = undefined;
        inline for (0..4) |k| {
            const m0 = neon.@"llvm.aarch64.neon.sqdmulh.v8i16"(a[k], c0);
            const m1 = neon.@"llvm.aarch64.neon.sqdmulh.v8i16"(c[k], c1);
            const t = if (even) m0 + m1 else (m0 >> @splat(1)) + (m1 >> @splat(1));
            o[k] = neon.@"llvm.aarch64.neon.sqrshrun.v8i8"(t, 2);
        }
        d[x..][0..16].* = std.simd.join(o[0], o[1]);
        d[x + 16 ..][0..16].* = std.simd.join(o[2], o[3]);
    }
    return x;
}

/// OpenCV's VResizeCubicVec_32s8u: float fma chain with the weights scaled
/// by 2^-22, rounded half to even.
fn vcubic(r: [4][]const i32, b: [4]i32, d: []u8) void {
    const N = 16;
    const F = @Vector(N, f32);
    const I = @Vector(N, i32);
    const scale: f32 = 1.0 / @as(f32, coef_scale * coef_scale);
    var bv: [4]F = undefined;
    for (0..4) |k| bv[k] = @splat(@as(f32, @floatFromInt(b[k])) * scale);
    var x: usize = 0;
    while (x + N <= d.len) : (x += N) {
        var s: [4]F = undefined;
        inline for (0..4) |k| s[k] = @floatFromInt(@as(I, r[k][x..][0..N].*));
        const v = @mulAdd(F, s[0], bv[0], @mulAdd(F, s[1], bv[1], @mulAdd(F, s[2], bv[2], s[3] * bv[3])));
        const q: I = @intFromFloat(@min(@max(sat.rint(v), @as(F, @splat(-1.0))), @as(F, @splat(256.0))));
        const cl = @max(@min(q, @as(I, @splat(255))), @as(I, @splat(0)));
        d[x..][0..N].* = simd.narrow(u8, cl);
    }
    const bs = [4]f32{ bv[0][0], bv[1][0], bv[2][0], bv[3][0] };
    while (x < d.len) : (x += 1) {
        const v = @mulAdd(f32, @floatFromInt(r[0][x]), bs[0], @mulAdd(f32, @floatFromInt(r[1][x]), bs[1], @mulAdd(f32, @floatFromInt(r[2][x]), bs[2], @as(f32, @floatFromInt(r[3][x])) * bs[3])));
        d[x] = sat.cast(u8, v);
    }
}

// ---------------------------------------------------------------------------
// Exact 2x enlargement (both directions). Horizontal taps are then fixed per
// output parity, so rows are filtered sample-parallel on planar channels:
// a source pixel k yields destination pixels 2k (even taps) and 2k+1 (odd).

/// Planar per-channel row of the doubled width, element type `E`.
fn Planar(comptime E: type) type {
    return struct {
        data: []E,
        w2: usize, // doubled width
        fn ch(self: @This(), c: usize) []E {
            return self.data[c * self.w2 ..][0..self.w2];
        }
    };
}

const zip_lo = blk: {
    var m: [16]i32 = undefined;
    for (0..8) |i| {
        m[2 * i] = i;
        m[2 * i + 1] = ~@as(i32, i);
    }
    break :blk m;
};
const zip_hi = blk: {
    var m: [16]i32 = undefined;
    for (0..8) |i| {
        m[2 * i] = 8 + i;
        m[2 * i + 1] = ~@as(i32, 8 + i);
    }
    break :blk m;
};

/// Bilinear 2x. With OpenCV's coefficients (512, 1536) the horizontal sums
/// are 512*h with h = s[k-1] + 3 s[k] (even) or 3 s[k] + s[k+1] (odd), edges
/// replicated; the vertical step ((r>>4)*b>>16) is then (h*(b/512))>>2, so
/// everything fits 16-bit lanes and matches the generic fixed-point path.
pub fn up2Linear(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), beta: []const [2]i32, yofs: []const [2]u32) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), gpa: Allocator, beta: []const [2]i32, yofs: []const [2]u32 };
    exec.parallelFor(dst.height, 8, Ctx{ .s = src, .d = dst, .gpa = gpa, .beta = beta, .yofs = yofs }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| up2Rows(u16, 2, cn, c, y0, y1),
                else => unreachable,
            }
        }
    }.run);
}

/// Bicubic 2x: fixed even/odd 4-tap weights in i32, OpenCV's float vertical.
pub fn up2Cubic(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), we: [4]i32, wo: [4]i32, beta: []const [4]i32, yofs: []const [4]u32) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), gpa: Allocator, beta: []const [4]i32, yofs: []const [4]u32, we: [4]i32, wo: [4]i32 };
    exec.parallelFor(dst.height, 8, Ctx{ .s = src, .d = dst, .gpa = gpa, .beta = beta, .yofs = yofs, .we = we, .wo = wo }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| up2Rows(f32, 4, cn, c, y0, y1),
                else => unreachable,
            }
        }
    }.run);
}

fn up2Rows(comptime E: type, comptime K: usize, comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    const w2: usize = c.d.width;
    const buf = c.gpa.alloc(E, K * cn * w2) catch @panic("out of memory");
    defer c.gpa.free(buf);
    var tag: [K]i64 = @splat(-1);
    var y = y0;
    while (y < y1) {
        const o = c.yofs[y];
        var prow: [K]Planar(E) = undefined;
        var keep: [K]bool = @splat(false);
        var have: [K]?usize = @splat(null);
        for (0..K) |k| for (0..K) |b| if (tag[b] == o[k]) {
            have[k] = b;
            keep[b] = true;
        };
        for (0..K) |k| {
            if (have[k] == null) for (0..k) |j| if (o[j] == o[k]) {
                have[k] = have[j];
            };
            if (have[k] == null) {
                var b: usize = 0;
                while (keep[b]) b += 1;
                keep[b] = true;
                tag[b] = o[k];
                const pl = Planar(E){ .data = buf[b * cn * w2 ..][0 .. cn * w2], .w2 = w2 };
                if (E == u16) up2HLinear(cn, c.s.row(o[k]), pl) else up2HCubic(cn, c.s.row(o[k]), pl, c.we, c.wo);
                have[k] = b;
            }
            prow[k] = .{ .data = buf[have[k].? * cn * w2 ..][0 .. cn * w2], .w2 = w2 };
        }
        // Destination rows 2k+1 and 2k+2 read the same source rows: one pass
        // over the intermediate rows serves both.
        if (y + 1 < y1 and std.mem.eql(u32, &c.yofs[y + 1], &o)) {
            const b = [2][K]i32{ c.beta[y], c.beta[y + 1] };
            const d = [2][]u8{ c.d.row(y), c.d.row(y + 1) };
            if (E == u16) up2VLinear(2, cn, prow[0..2].*, b, d) else up2VCubic(2, cn, prow, b, d);
            y += 2;
        } else {
            const b = [1][K]i32{c.beta[y]};
            const d = [1][]u8{c.d.row(y)};
            if (E == u16) up2VLinear(1, cn, prow[0..2].*, b, d) else up2VCubic(1, cn, prow, b, d);
            y += 1;
        }
    }
}

fn up2HLinear(comptime cn: usize, s: []const u8, out: Planar(u16)) void {
    const W16 = @Vector(16, u16);
    const w = s.len / cn;
    const S = struct {
        inline fn px(row: []const u8, k: isize, ch: usize, ww: usize) u16 {
            const kk: usize = @intCast(std.math.clamp(k, 0, @as(isize, @intCast(ww)) - 1));
            return row[kk * cn + ch];
        }
    };
    var k: usize = 0;
    // Scalar for the first pixel (left edge) and wherever 16 + 1 pixels
    // to the right are unavailable.
    while (k < w) {
        if (k >= 1 and k + 17 <= w) {
            const prev = vec.load(cn, s[(k - 1) * cn ..].ptr);
            const cur = vec.load(cn, s[k * cn ..].ptr);
            const next = vec.load(cn, s[(k + 1) * cn ..].ptr);
            inline for (0..cn) |ch| {
                const c3 = @as(W16, cur[ch]) * @as(W16, @splat(3));
                const e = @as(W16, prev[ch]) + c3;
                const o = c3 + @as(W16, next[ch]);
                const d = out.ch(ch)[2 * k ..];
                d[0..16].* = @shuffle(u16, e, o, zip_lo);
                d[16..32].* = @shuffle(u16, e, o, zip_hi);
            }
            k += 16;
        } else {
            const ki: isize = @intCast(k);
            inline for (0..cn) |ch| {
                const c3 = 3 * S.px(s, ki, ch, w);
                out.ch(ch)[2 * k] = S.px(s, ki - 1, ch, w) + c3;
                out.ch(ch)[2 * k + 1] = c3 + S.px(s, ki + 1, ch, w);
            }
            k += 1;
        }
    }
}

fn up2VLinear(comptime R: usize, comptime cn: usize, r: [2]Planar(u16), b: [R][2]i32, d: [R][]u8) void {
    const W16 = @Vector(16, u16);
    var m: [R][2]u16 = undefined;
    for (0..R) |i| for (0..2) |k| {
        m[i][k] = @intCast(@divExact(b[i][k], 512));
    };
    const w2 = r[0].w2;
    var x: usize = 0;
    while (x + 16 <= w2) : (x += 16) {
        var out: [R][cn]vec.V16 = undefined;
        inline for (0..cn) |ch| {
            const h0: W16 = r[0].ch(ch)[x..][0..16].*;
            const h1: W16 = r[1].ch(ch)[x..][0..16].*;
            inline for (0..R) |i| {
                const t = (((h0 * @as(W16, @splat(m[i][0]))) >> @splat(2)) + ((h1 * @as(W16, @splat(m[i][1]))) >> @splat(2)) + @as(W16, @splat(2))) >> @splat(2);
                out[i][ch] = @intCast(@min(t, @as(W16, @splat(255))));
            }
        }
        inline for (0..R) |i| vec.store(cn, out[i], d[i][x * cn ..].ptr);
    }
    while (x < w2) : (x += 1) {
        inline for (0..R) |i| inline for (0..cn) |ch| {
            const t = ((r[0].ch(ch)[x] * m[i][0]) >> 2) + ((r[1].ch(ch)[x] * m[i][1]) >> 2);
            d[i][x * cn + ch] = @intCast(@min((t + 2) >> 2, 255));
        };
    }
}

fn up2HCubic(comptime cn: usize, s: []const u8, out: Planar(f32), we: [4]i32, wo: [4]i32) void {
    // Integer sums below 2^24, so f32 arithmetic is exact.
    const F = @Vector(16, f32);
    const I = @Vector(16, i32);
    const w = s.len / cn;
    var wef: [4]F = undefined;
    var wof: [4]F = undefined;
    for (0..4) |j| {
        wef[j] = @splat(@floatFromInt(we[j]));
        wof[j] = @splat(@floatFromInt(wo[j]));
    }
    var k: usize = 0;
    while (k < w) {
        if (k >= 2 and k + 18 <= w) {
            // Pixels k-2 .. k+2 shifted by j.
            var p: [5][cn]vec.V16 = undefined;
            inline for (0..5) |j| p[j] = vec.load(cn, s[(k + j - 2) * cn ..].ptr);
            inline for (0..cn) |ch| {
                var f: [5]F = undefined;
                inline for (0..5) |j| f[j] = @floatFromInt(@as(I, p[j][ch]));
                var e = f[0] * wef[0];
                var o = f[1] * wof[0];
                inline for (1..4) |j| {
                    e = @mulAdd(F, f[j], wef[j], e);
                    o = @mulAdd(F, f[j + 1], wof[j], o);
                }
                const d = out.ch(ch)[2 * k ..];
                d[0..16].* = @shuffle(f32, e, o, zip_lo);
                d[16..32].* = @shuffle(f32, e, o, zip_hi);
            }
            k += 16;
        } else {
            const ki: isize = @intCast(k);
            inline for (0..cn) |ch| {
                var e: i32 = 0;
                var o: i32 = 0;
                inline for (0..4) |j| {
                    const ji: isize = @intCast(j);
                    const pe: usize = @intCast(std.math.clamp(ki + ji - 2, 0, @as(isize, @intCast(w)) - 1));
                    const po: usize = @intCast(std.math.clamp(ki + ji - 1, 0, @as(isize, @intCast(w)) - 1));
                    e += @as(i32, s[pe * cn + ch]) * we[j];
                    o += @as(i32, s[po * cn + ch]) * wo[j];
                }
                out.ch(ch)[2 * k] = @floatFromInt(e);
                out.ch(ch)[2 * k + 1] = @floatFromInt(o);
            }
            k += 1;
        }
    }
}

/// Rounds half to even and saturates to u8, like OpenCV's v_round + packs.
inline fn roundU8(v: @Vector(16, f32)) vec.V16 {
    // OpenCV's saturate_cast<uchar>(float): round half to even and clamp.
    // AVX-512: one zmm conversion. Elsewhere simd.satF32 (cvtps2dq /
    // fcvtns and saturating narrows; A/B: Apple Silicon 1.24x, AVX2 1.16x
    // over the magic-number rint).
    if (use_asm and avx512f) {
        const I = @Vector(16, i32);
        const q: I = asm ("vcvtps2dq %[v], %[ret]"
            : [ret] "=v" (-> I),
            : [v] "v" (v),
        );
        return simd.narrow(u8, @min(@max(q, @as(I, @splat(0))), @as(I, @splat(255))));
    }
    return simd.satF32(u8, 16, v);
}

fn up2VCubic(comptime R: usize, comptime cn: usize, r: [4]Planar(f32), b: [R][4]i32, d: [R][]u8) void {
    const N = 16;
    const F = @Vector(N, f32);
    const scale: f32 = 1.0 / @as(f32, coef_scale * coef_scale);
    var bs: [R][4]f32 = undefined;
    for (0..R) |i| for (0..4) |k| {
        bs[i][k] = @as(f32, @floatFromInt(b[i][k])) * scale;
    };
    const w2 = r[0].w2;
    var x: usize = 0;
    while (x + N <= w2) : (x += N) {
        var out: [R][cn]vec.V16 = undefined;
        inline for (0..cn) |ch| {
            var s: [4]F = undefined;
            inline for (0..4) |k| s[k] = r[k].ch(ch)[x..][0..N].*;
            inline for (0..R) |i| {
                const v = @mulAdd(F, s[0], @splat(bs[i][0]), @mulAdd(F, s[1], @splat(bs[i][1]), @mulAdd(F, s[2], @splat(bs[i][2]), s[3] * @as(F, @splat(bs[i][3])))));
                out[i][ch] = roundU8(v);
            }
        }
        inline for (0..R) |i| vec.store(cn, out[i], d[i][x * cn ..].ptr);
    }
    while (x < w2) : (x += 1) {
        inline for (0..R) |i| inline for (0..cn) |ch| {
            const v = @mulAdd(f32, r[0].ch(ch)[x], bs[i][0], @mulAdd(f32, r[1].ch(ch)[x], bs[i][1], @mulAdd(f32, r[2].ch(ch)[x], bs[i][2], r[3].ch(ch)[x] * bs[i][3])));
            d[i][x * cn + ch] = sat.cast(u8, v);
        };
    }
}

/// One INTER_AREA table entry (OpenCV's DecimateAlpha): source cell `si`
/// contributes `alpha` to destination cell `di`.
pub const AreaTab = struct { di: u32, si: u32, alpha: f32 };

/// Most horizontal taps handled by the fixed-tap kernels; wider cells use
/// OpenCV's table loop directly.
const area_max_k = 8;

fn AreaTap(comptime K: usize) type {
    return struct { o: [K]u32, w: [K]f32 };
}

/// Fractional INTER_AREA shrink with OpenCV's arithmetic
/// (ResizeArea_Invoker<uchar, float>): each source row is resampled
/// horizontally into f32 by summing `s * alpha` over the destination cell's
/// taps in table order, then destination rows are `sum(beta * row)` in
/// table order, rounded half to even. The kernels keep that per-element
/// order of operations (plain multiplies and adds, no FMA) and vectorise
/// across channels and pixels, so results are bit-exact; each source row is
/// resampled once per worker. `ystart[dy]..ystart[dy+1]` is the run of
/// `ytab` entries of destination row `dy`.
pub fn area(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), xtab: []const AreaTab, ytab: []const AreaTab, ystart: []const u32) Allocator.Error!void {
    var kx: usize = 0;
    var i: usize = 0;
    while (i < xtab.len) {
        var j = i;
        while (j < xtab.len and xtab[j].di == xtab[i].di) j += 1;
        kx = @max(kx, j - i);
        i = j;
    }
    var ky: usize = 1;
    for (0..dst.height) |y| ky = @max(ky, ystart[y + 1] - ystart[y]);
    switch (kx) {
        inline 1...area_max_k => |K| {
            // Fixed K taps per destination pixel; shorter cells are padded
            // with zero weights on their last source pixel (adding s * 0 to
            // a non-negative partial sum leaves it unchanged).
            const cn: u32 = src.channels;
            const xt = try gpa.alloc(AreaTap(K), dst.width);
            defer gpa.free(xt);
            var e: usize = 0;
            for (xt, 0..) |*t, x| {
                var k: usize = 0;
                while (e < xtab.len and xtab[e].di == x) : (e += 1) {
                    t.o[k] = xtab[e].si * cn;
                    t.w[k] = xtab[e].alpha;
                    k += 1;
                }
                std.debug.assert(k > 0);
                while (k < K) : (k += 1) {
                    t.o[k] = t.o[k - 1];
                    t.w[k] = 0;
                }
            }
            areaRun(K, exec, gpa, src, dst, xt, xtab, ytab, ystart, ky);
        },
        else => areaRun(0, exec, gpa, src, dst, &.{}, xtab, ytab, ystart, ky),
    }
}

fn areaRun(comptime K: usize, exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), xt: []const AreaTap(K), xtab: []const AreaTab, ytab: []const AreaTab, ystart: []const u32, ky: usize) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), xt: []const AreaTap(K), xtab: []const AreaTab, ytab: []const AreaTab, ystart: []const u32, ky: usize, gpa: Allocator };
    const ctx: Ctx = .{ .s = src, .d = dst, .xt = xt, .xtab = xtab, .ytab = ytab, .ystart = ystart, .ky = ky, .gpa = gpa };
    exec.parallelFor(dst.height, 4, ctx, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| areaRows(K, cn, c, y0, y1),
                else => unreachable,
            }
        }
    }.run);
}

const area_vn = 16;

fn areaRows(comptime K: usize, comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    const n = c.d.rowLen();
    // Rows are padded for the 4-lane stores of 3-channel pixels and for
    // whole vectors in the vertical pass.
    const stride = std.mem.alignForward(usize, n + 4, area_vn);
    const slen = c.s.rowLen() + 4;
    const buf = c.gpa.alloc(f32, c.ky * stride + slen) catch @panic("out of memory");
    defer c.gpa.free(buf);
    const fsrc = buf[c.ky * stride ..];
    @memset(buf, 0);
    const tags = c.gpa.alloc(u32, c.ky) catch @panic("out of memory");
    defer c.gpa.free(tags);
    @memset(tags, std.math.maxInt(u32));
    var rows_: [area_max_k][]const f32 = undefined;
    var betas: [area_max_k]f32 = undefined;
    for (y0..y1) |dy| {
        const es = c.ytab[c.ystart[dy]..c.ystart[dy + 1]];
        for (es, 0..) |e, k| {
            const slot = e.si % c.ky;
            const h = buf[slot * stride ..][0..stride];
            if (tags[slot] != e.si) {
                tags[slot] = e.si;
                areaH(K, cn, c.s.row(e.si), fsrc, h, c.xt, c.xtab);
            }
            if (k < area_max_k) {
                rows_[k] = h;
                betas[k] = e.alpha;
            }
        }
        const d = c.d.row(dy);
        switch (es.len) {
            inline 1...4 => |M| areaV(M, rows_[0..M].*, betas[0..M].*, d),
            else => areaVAny(c.ky, es, buf, stride, d),
        }
    }
}

/// Horizontal pass of one source row into `h` (f32, `n` + padding).
fn areaH(comptime K: usize, comptime cn: usize, s: []const u8, f: []f32, h: []f32, xt: []const AreaTap(K), xtab: []const AreaTab) void {
    {
        // u8 -> f32 once per row.
        const V = @Vector(area_vn, u8);
        var i: usize = 0;
        while (i + area_vn <= s.len) : (i += area_vn) {
            const v: V = s[i..][0..area_vn].*;
            f[i..][0..area_vn].* = simd.toF32(area_vn, v);
        }
        while (i < s.len) : (i += 1) f[i] = @floatFromInt(s[i]);
    }
    if (K == 0) {
        // OpenCV's loop over the table.
        @memset(h, 0);
        for (xtab) |t| inline for (0..cn) |ch| {
            h[t.di * cn + ch] += f[t.si * cn + ch] * t.alpha;
        };
        return;
    }
    if (cn == 1) {
        for (xt, 0..) |t, x| {
            var acc = f[t.o[0]] * t.w[0];
            inline for (1..K) |k| acc = acc + f[t.o[k]] * t.w[k];
            h[x] = acc;
        }
        return;
    }
    // One vector per pixel; 3-channel pixels use 4 lanes, the extra lane is
    // overwritten by the next pixel (or lands in the padding).
    const L = if (cn == 3) 4 else cn;
    const V = @Vector(L, f32);
    for (xt, 0..) |t, x| {
        var acc = @as(V, f[t.o[0]..][0..L].*) * @as(V, @splat(t.w[0]));
        inline for (1..K) |k| acc = acc + @as(V, f[t.o[k]..][0..L].*) * @as(V, @splat(t.w[k]));
        h[x * cn ..][0..L].* = acc;
    }
}

/// Vertical pass for a destination row with `M` source rows.
fn areaV(comptime M: usize, r: [M][]const f32, b: [M]f32, d: []u8) void {
    const V = @Vector(area_vn, f32);
    var x: usize = 0;
    while (x < d.len) : (x += area_vn) {
        var acc = @as(V, r[0][x..][0..area_vn].*) * @as(V, @splat(b[0]));
        inline for (1..M) |k| acc = acc + @as(V, r[k][x..][0..area_vn].*) * @as(V, @splat(b[k]));
        const o: [area_vn]u8 = simd.satF32(u8, area_vn, acc);
        if (x + area_vn <= d.len) {
            d[x..][0..area_vn].* = o;
        } else @memcpy(d[x..], o[0 .. d.len - x]);
    }
}

fn areaVAny(ky: usize, es: []const AreaTab, buf: []const f32, stride: usize, d: []u8) void {
    for (d, 0..) |*o, x| {
        var acc: f32 = buf[(es[0].si % ky) * stride + x] * es[0].alpha;
        for (es[1..]) |e| acc = acc + buf[(e.si % ky) * stride + x] * e.alpha;
        o.* = sat.cast(u8, acc);
    }
}

/// Integer-factor INTER_AREA shrink (OpenCV's resizeAreaFast for factors
/// other than 2x2): the integer block sum times the f32 reciprocal of the
/// block area, rounded half to even.
pub fn areaInt(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), ix: u32, iy: u32) void {
    const Ctx = struct { s: Image(u8), d: Image(u8), ix: u32, iy: u32, gpa: Allocator };
    exec.parallelFor(dst.height, 4, Ctx{ .s = src, .d = dst, .ix = ix, .iy = iy, .gpa = gpa }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| areaIntRows(cn, c, y0, y1),
                else => unreachable,
            }
        }
    }.run);
}

fn areaIntRows(comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    if (comptime planar) if (c.ix == 4 and c.iy == 4) return area4Rows(cn, c, y0, y1);
    const n = c.d.rowLen();
    const sn: usize = c.d.width * c.ix * cn; // source elements used per row
    const stride = std.mem.alignForward(usize, n + 4, area_vn);
    const buf = c.gpa.alloc(u32, sn + 4 + stride) catch @panic("out of memory");
    defer c.gpa.free(buf);
    @memset(buf, 0);
    const vs = buf[0 .. sn + 4];
    const hs = buf[sn + 4 ..];
    const scale: f32 = 1.0 / @as(f32, @floatFromInt(c.ix * c.iy));
    const U = @Vector(area_vn, u32);
    for (y0..y1) |dy| {
        // Column sums over the block's rows.
        for (0..c.iy) |k| {
            const s = c.s.row(dy * c.iy + k)[0..sn];
            var i: usize = 0;
            while (i + area_vn <= sn) : (i += area_vn) {
                const v: @Vector(area_vn, u8) = s[i..][0..area_vn].*;
                const prev: U = if (k == 0) @splat(0) else vs[i..][0..area_vn].*;
                vs[i..][0..area_vn].* = prev + @as(U, v);
            }
            while (i < sn) : (i += 1) vs[i] = (if (k == 0) 0 else vs[i]) + s[i];
        }
        // Row sums over the block's columns.
        if (cn == 1) {
            for (0..c.d.width) |x| {
                var acc: u32 = 0;
                for (vs[x * c.ix ..][0..c.ix]) |v| acc += v;
                hs[x] = acc;
            }
        } else {
            const L = if (cn == 3) 4 else cn;
            const W = @Vector(L, u32);
            for (0..c.d.width) |x| {
                var acc: W = @splat(0);
                const base = x * c.ix * cn;
                for (0..c.ix) |k| acc += @as(W, vs[base + k * cn ..][0..L].*);
                hs[x * cn ..][0..L].* = acc;
            }
        }
        const d = c.d.row(dy);
        var x: usize = 0;
        while (x < n) : (x += area_vn) {
            const v: U = hs[x..][0..area_vn].*;
            const f = @as(@Vector(area_vn, f32), @floatFromInt(v)) * @as(@Vector(area_vn, f32), @splat(scale));
            const o: [area_vn]u8 = simd.satF32(u8, area_vn, f);
            if (x + area_vn <= n) {
                d[x..][0..area_vn].* = o;
            } else @memcpy(d[x..], o[0 .. n - x]);
        }
    }
}

/// NEON 4x4 block average (the shape of carotene's resizeAreaU8): per
/// channel, pairwise-add 16 deinterleaved bytes of each of the four rows
/// into u16 (uaddlp), sum the rows, pairwise-add again (addp) and divide by
/// 16 rounding half to even, as the f32 path does (the sums are exact).
fn area4Rows(comptime cn: usize, c: anytype, y0: usize, y1: usize) void {
    @setEvalBranchQuota(10000);
    const H = @Vector(8, u16);
    const w = c.d.width;
    for (y0..y1) |dy| {
        var r: [4][*]const u8 = undefined;
        inline for (0..4) |k| r[k] = c.s.row(4 * dy + k).ptr;
        const d = c.d.row(dy);
        var x: usize = 0;
        while (x + 16 <= w) : (x += 16) {
            // Each half: 8 output pixels from 32 source pixels; acc[g][ch]
            // holds pair sums of source pixels 16g..16g+15 of the half.
            var b: [2][cn]@Vector(8, u8) = undefined;
            inline for (0..2) |h| {
                var acc: [2][cn]H = undefined;
                inline for (0..2) |g| {
                    const off = (4 * x + 32 * h + 16 * g) * cn;
                    inline for (0..4) |k| {
                        const v = vec.load(cn, r[k] + off);
                        inline for (0..cn) |ch| {
                            const p = neon.@"llvm.aarch64.neon.uaddlp.v8i16.v16i8"(v[ch]);
                            acc[g][ch] = if (k == 0) p else acc[g][ch] + p;
                        }
                    }
                }
                inline for (0..cn) |ch| {
                    const sum = neon.@"llvm.aarch64.neon.addp.v8i16"(acc[0][ch], acc[1][ch]);
                    b[h][ch] = @truncate((sum + @as(H, @splat(7)) + ((sum >> @splat(4)) & @as(H, @splat(1)))) >> @splat(4));
                }
            }
            var out: [cn]vec.V16 = undefined;
            inline for (0..cn) |ch| out[ch] = std.simd.join(b[0][ch], b[1][ch]);
            vec.store(cn, out, d.ptr + x * cn);
        }
        while (x < w) : (x += 1) {
            inline for (0..cn) |ch| {
                var s: u32 = 0;
                for (0..4) |k| for (0..4) |j| {
                    s += r[k][(4 * x + j) * cn + ch];
                };
                d[x * cn + ch] = @intCast((s + 7 + ((s >> 4) & 1)) >> 4);
            }
        }
    }
}
