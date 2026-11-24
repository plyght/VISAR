//! Byte-vector helpers for interleaved pixels: split 16 pixels of `cn`
//! channels into per-channel vectors and back, using two-input 16-byte
//! shuffles (pshufb/tbl-sized), which every SIMD target lowers well.

const std = @import("std");
const builtin = @import("builtin");
const simd = @import("simd.zig");

pub const V16 = @Vector(16, u8);

pub const is_arm = builtin.cpu.arch == .aarch64;

/// LLVM's NEON structured-store intrinsics (aarch64, LLVM backend only).
const neon = struct {
    extern fn @"llvm.aarch64.neon.st2.v16i8.p0"(V16, V16, [*]u8) void;
    extern fn @"llvm.aarch64.neon.st3.v16i8.p0"(V16, V16, V16, [*]u8) void;
    extern fn @"llvm.aarch64.neon.st4.v16i8.p0"(V16, V16, V16, V16, [*]u8) void;
    extern fn @"llvm.aarch64.neon.tbl1.v16i8"(V16, V16) V16;
    extern fn @"llvm.aarch64.neon.tbl2.v16i8"(V16, V16, V16) V16;
    extern fn @"llvm.aarch64.neon.tbl3.v16i8"(V16, V16, V16, V16) V16;
    /// tbl over `k` consecutive registers of `t` starting at `lo`.
    inline fn tbl(comptime lo: usize, comptime k: usize, t: anytype, i: V16) V16 {
        return switch (k) {
            1 => @"llvm.aarch64.neon.tbl1.v16i8"(t[lo], i),
            2 => @"llvm.aarch64.neon.tbl2.v16i8"(t[lo], t[lo + 1], i),
            3 => @"llvm.aarch64.neon.tbl3.v16i8"(t[lo], t[lo + 1], t[lo + 2], i),
            else => @compileError("tbl of 1-3 registers"),
        };
    }
    inline fn st2(a: V16, b: V16, p: [*]u8) void {
        @"llvm.aarch64.neon.st2.v16i8.p0"(a, b, p);
    }
    inline fn st3(a: V16, b: V16, c: V16, p: [*]u8) void {
        @"llvm.aarch64.neon.st3.v16i8.p0"(a, b, c, p);
    }
    inline fn st4(a: V16, b: V16, c: V16, d: V16, p: [*]u8) void {
        @"llvm.aarch64.neon.st4.v16i8.p0"(a, b, c, d, p);
    }
};

fn strideMask(comptime cn: usize, comptime c: usize) [16]i32 {
    var m: [16]i32 = undefined;
    for (0..16) |i| m[i] = @intCast(i * cn + c);
    return m;
}

/// Hides a loaded vector from LLVM's load narrowing, which otherwise turns
/// a 16-byte load feeding byte shuffles into many scalar byte inserts.
pub inline fn opaqueV(v: V16) V16 {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => asm (""
            : [ret] "=x" (-> V16),
            : [v] "0" (v),
        ),
        .aarch64 => asm (""
            : [ret] "=w" (-> V16),
            : [v] "0" (v),
        ),
        else => v,
    };
}

/// x86 targets with AVX-512 VBMI: 64-byte two-source byte permutes
/// (vpermt2b) are single instructions, so 64-pixel deinterleaves are cheap.
pub const has_vbmi = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx512vbmi);
const has_avx2 = builtin.cpu.arch == .x86_64 and std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

pub const V64 = @Vector(64, u8);

inline fn opaque64(v: V64) V64 {
    return if (has_vbmi) asm (""
        : [ret] "=v" (-> V64),
        : [v] "0" (v),
    ) else v;
}

/// Like `load`, for 64 pixels at a time (for `has_vbmi` targets).
pub inline fn load64(comptime cn: usize, p: [*]const u8) [cn]V64 {
    var blocks: [cn]V64 = undefined;
    inline for (0..cn) |b| blocks[b] = opaque64(p[64 * b ..][0..64].*);
    if (cn == 1) return blocks;
    var out: [cn]V64 = undefined;
    inline for (0..cn) |c| {
        const masks = comptime blk: {
            @setEvalBranchQuota(100000);
            var m: [cn - 1][64]i32 = undefined;
            for (0..cn - 1) |step| for (0..64) |i| {
                const src = i * cn + c;
                const blk = src / 64;
                const off: i32 = @intCast(src % 64);
                m[step][i] = if (step == 0)
                    (if (blk == 0) off else if (blk == 1) ~off else 0)
                else if (blk == step + 1) ~off else @intCast(i);
            };
            break :blk m;
        };
        var acc = @shuffle(u8, blocks[0], blocks[1], masks[0]);
        inline for (1..cn - 1) |step| acc = @shuffle(u8, acc, blocks[step + 1], masks[step]);
        out[c] = acc;
    }
    return out;
}

/// Like `store`, for 64 pixels at a time (for `has_vbmi` targets).
pub inline fn store64(comptime cn: usize, ch: [cn]V64, p: [*]u8) void {
    if (cn == 1) {
        p[0..64].* = ch[0];
        return;
    }
    inline for (0..cn) |b| {
        const masks = comptime blk: {
            @setEvalBranchQuota(100000);
            var m: [cn - 1][64]i32 = undefined;
            for (0..cn - 1) |step| for (0..64) |t| {
                const g = 64 * b + t;
                const src = (g % cn) * 64 + g / cn;
                const blk = src / 64;
                const off: i32 = @intCast(src % 64);
                m[step][t] = if (step == 0)
                    (if (blk == 0) off else if (blk == 1) ~off else 0)
                else if (blk == step + 1) ~off else @intCast(t);
            };
            break :blk m;
        };
        var acc = @shuffle(u8, ch[0], ch[1], masks[0]);
        inline for (1..cn - 1) |step| acc = @shuffle(u8, acc, ch[step + 1], masks[step]);
        p[64 * b ..][0..64].* = acc;
    }
}

/// Narrows i32 lanes already in [0, 255] to u8. Goes through u32 because
/// LLVM (Zig 0.17, AVX2 without AVX-512) can lower a direct i32 -> u8
/// narrowing to signed packs, turning 150 into 127.
pub inline fn narrowU8(comptime n: usize, v: @Vector(n, i32)) @Vector(n, u8) {
    const u: @Vector(n, u32) = @bitCast(v);
    return @intCast(u);
}

/// Shuffle masks that gather `src(i)` (an index into `nblocks` 16-byte
/// blocks) for each output lane i, as a chain of two-input shuffles:
/// step 0 combines blocks 0 and 1, step j>0 combines the running result
/// with block j+1.
fn chain(comptime nblocks: usize, comptime src: [16]usize) [nblocks - 1][16]i32 {
    @setEvalBranchQuota(10000);
    var masks: [nblocks - 1][16]i32 = undefined;
    for (0..nblocks - 1) |step| {
        for (0..16) |i| {
            const s = src[i];
            const blk = s / 16;
            const off: i32 = @intCast(s % 16);
            masks[step][i] = if (step == 0)
                (if (blk == 0) off else if (blk == 1) ~off else 0)
            else if (blk == step + 1) ~off else @intCast(i);
        }
    }
    return masks;
}

inline fn gather(comptime n: usize, blocks: [n]V16, comptime src: [16]usize) V16 {
    if (n == 1) return @shuffle(u8, blocks[0], undefined, blk: {
        var m: [16]i32 = undefined;
        for (0..16) |i| m[i] = @intCast(src[i]);
        break :blk m;
    });
    const masks = comptime chain(n, src);
    var acc = @shuffle(u8, blocks[0], blocks[1], masks[0]);
    inline for (1..n - 1) |step| acc = @shuffle(u8, acc, blocks[step + 1], masks[step]);
    return acc;
}

/// Loads 16 pixels of `cn` interleaved channels from `p` (16*cn bytes) and
/// returns one vector per channel.
pub inline fn load(comptime cn: usize, p: [*]const u8) [cn]V16 {
    // AArch64 has structured loads (ld2/ld3/ld4): a plain wide load plus
    // stride shuffles lowers to exactly one of them.
    if (comptime is_arm and cn > 1) {
        const v: @Vector(16 * cn, u8) = p[0 .. 16 * cn].*;
        var out: [cn]V16 = undefined;
        inline for (0..cn) |c| out[c] = @shuffle(u8, v, undefined, strideMask(cn, c));
        return out;
    }
    var blocks: [cn]V16 = undefined;
    inline for (0..cn) |b| blocks[b] = opaqueV(p[16 * b ..][0..16].*);
    if (cn == 1) return blocks;
    var out: [cn]V16 = undefined;
    inline for (0..cn) |c| {
        const src = comptime blk: {
            @setEvalBranchQuota(10000);
            var s: [16]usize = undefined;
            for (0..16) |i| s[i] = i * cn + c;
            break :blk s;
        };
        out[c] = gather(cn, blocks, src);
    }
    return out;
}

/// Stores per-channel vectors as 16 interleaved pixels (16*cn bytes).
pub inline fn store(comptime cn: usize, ch: [cn]V16, p: [*]u8) void {
    if (cn == 1) {
        p[0..16].* = ch[0];
        return;
    }
    // AArch64: NEON structured stores. LLVM does not reliably match the
    // interleaving shuffle below to st3/st4, so call the intrinsics.
    if (comptime is_arm and simd.llvm_intrinsics) {
        switch (cn) {
            2 => return neon.st2(ch[0], ch[1], p),
            3 => return neon.st3(ch[0], ch[1], ch[2], p),
            4 => return neon.st4(ch[0], ch[1], ch[2], ch[3], p),
            else => {},
        }
    }
    if (comptime is_arm) {
        // One two-operand shuffle over (first half of the channels, second
        // half), the shape LLVM's interleaved-access pass turns into stN.
        const half = cn / 2;
        const lo: @Vector(16 * half, u8) = @bitCast(ch[0..half].*);
        const hi: @Vector(16 * (cn - half), u8) = @bitCast(ch[half..cn].*);
        const mask = comptime blk: {
            @setEvalBranchQuota(10000);
            var m: [16 * cn]i32 = undefined;
            for (0..16 * cn) |t| {
                const src = (t % cn) * 16 + t / cn;
                m[t] = if (src < 16 * half) @intCast(src) else ~@as(i32, @intCast(src - 16 * half));
            }
            break :blk m;
        };
        p[0 .. 16 * cn].* = @shuffle(u8, lo, hi, mask);
        return;
    }
    inline for (0..cn) |b| {
        // Output byte t of block b is channel (16b+t) % cn of pixel (16b+t) / cn.
        const src = comptime blk: {
            @setEvalBranchQuota(10000);
            var s: [16]usize = undefined;
            for (0..16) |t| {
                const g = 16 * b + t;
                s[t] = (g % cn) * 16 + g / cn;
            }
            break :blk s;
        };
        p[16 * b ..][0..16].* = gather(cn, ch, src);
    }
}

/// Rearranges 16 pixels of `scn` channels at `p` into 16 pixels of `dcn`
/// channels at `o`: output channel c of pixel i is input channel `map[c]`
/// of pixel i, or `fill` where `map[c]` is null. Each output block only
/// touches the input blocks it needs (one or two for typical reorders), so
/// this is cheaper than a full deinterleave/interleave round trip.
pub inline fn remap(comptime scn: usize, comptime dcn: usize, comptime map: [dcn]?u8, fill: u8, p: [*]const u8, o: [*]u8) void {
    var blocks: [scn + 1]V16 = undefined;
    inline for (0..scn) |b| blocks[b] = opaqueV(p[16 * b ..][0..16].*);
    blocks[scn] = @splat(fill);
    inline for (0..dcn) |b| {
        const plan = comptime blk: {
            @setEvalBranchQuota(20000);
            var src: [16]usize = undefined;
            for (0..16) |t| {
                const g = 16 * b + t;
                src[t] = if (map[g % dcn]) |c| (g / dcn) * scn + c else 16 * scn;
            }
            // Blocks referenced, in increasing order.
            var used: [scn + 1]bool = @splat(false);
            for (src) |v| used[v / 16] = true;
            var list: [scn + 1]usize = undefined;
            var n: usize = 0;
            for (used, 0..) |u, k| if (u) {
                list[n] = k;
                n += 1;
            };
            break :blk .{ .src = src, .list = list, .n = n };
        };
        o[16 * b ..][0..16].* = gatherFrom(scn + 1, blocks, plan.list[0..plan.n].*, plan.src);
    }
}

/// Reorders whole rows of pixels with overlapping 16-byte loads and one
/// single-source byte shuffle per step (pshufb / tbl). Each step converts
/// `16 / max(scn, dcn)` pixels; loads and stores may run past the bytes a
/// step uses, so the loop stops while a full 16 bytes still fit on both
/// sides. Returns the number of pixels done; the caller finishes the rest.
pub inline fn reorderRow(comptime scn: usize, comptime dcn: usize, comptime map: [dcn]?u8, fill: u8, s: [*]const u8, d: [*]u8, n: usize) usize {
    if (comptime is_arm and simd.llvm_intrinsics and scn == 3) {
        // AArch64, 3 channels in (BGR <-> RGB, alpha add): each 16-byte
        // output block is one tbl over the one to three consecutive input
        // blocks it draws from, plus an orr for a filled channel (out of
        // range tbl indices give 0), with plain loads and stores. The
        // indices go through opaqueV, or LLVM folds the tbl back into a
        // shuffle and scalarises it.
        const plan = comptime blk: {
            var lo: [dcn]usize = undefined;
            var k: [dcn]usize = undefined;
            var idx: [dcn][16]u8 = undefined;
            var fmask: [dcn][16]bool = undefined;
            for (0..dcn) |b| {
                var mn: usize = 48;
                var mx: usize = 0;
                for (0..16) |t| {
                    const g = 16 * b + t;
                    if (map[g % dcn]) |c| {
                        const src = 3 * (g / dcn) + c;
                        mn = @min(mn, src);
                        mx = @max(mx, src);
                    }
                }
                lo[b] = mn / 16;
                k[b] = mx / 16 - mn / 16 + 1;
                for (0..16) |t| {
                    const g = 16 * b + t;
                    fmask[b][t] = map[g % dcn] == null;
                    idx[b][t] = if (map[g % dcn]) |c| 3 * (g / dcn) + c - 16 * lo[b] else 0xff;
                }
            }
            break :blk .{ .lo = lo, .k = k, .idx = idx, .fmask = fmask };
        };
        var keys: [dcn]V16 = undefined;
        var fills: [dcn]V16 = undefined;
        inline for (0..dcn) |b| {
            keys[b] = opaqueV(plan.idx[b]);
            fills[b] = @select(u8, @as(@Vector(16, bool), plan.fmask[b]), @as(V16, @splat(fill)), @as(V16, @splat(0)));
        }
        const any_fill = comptime blk: {
            for (map) |m| if (m == null) break :blk true;
            break :blk false;
        };
        var x: usize = 0;
        while (x + 16 <= n) : (x += 16) {
            var in: [3]V16 = undefined;
            inline for (0..3) |b| in[b] = s[x * 3 + 16 * b ..][0..16].*;
            inline for (0..dcn) |b| {
                const v = neon.tbl(plan.lo[b], plan.k[b], in, keys[b]);
                d[x * dcn + 16 * b ..][0..16].* = if (any_fill) v | fills[b] else v;
            }
        }
        return x;
    }

    // AArch64: ldN + stN do the whole reorder in two instructions per 16
    // pixels (measured faster than single-register tbl groups for BGR <->
    // RGB on Apple cores), two blocks per iteration.
    if (comptime is_arm) {
        const f: V16 = @splat(fill);
        var x: usize = 0;
        while (x + 32 <= n) : (x += 32) {
            inline for (0..2) |b| {
                const ch = load(scn, s + (x + 16 * b) * scn);
                var out: [dcn]V16 = undefined;
                inline for (0..dcn) |c| out[c] = if (map[c]) |sc| ch[sc] else f;
                store(dcn, out, d + (x + 16 * b) * dcn);
            }
        }
        while (x + 16 <= n) : (x += 16) {
            const ch = load(scn, s + x * scn);
            var out: [dcn]V16 = undefined;
            inline for (0..dcn) |c| out[c] = if (map[c]) |sc| ch[sc] else f;
            store(dcn, out, d + x * dcn);
        }
        return x;
    }
    if (comptime builtin.cpu.arch == .x86_64 and builtin.zig_backend == .stage2_llvm and scn == 3 and dcn == 3)
        return permute3(map, s, d, n);
    if (comptime has_avx2 and scn == 3 and dcn == 4) {
        // AVX2: one in-lane vpshufb over two 16-byte loads 12 bytes apart
        // makes 8 pixels, stored as one 32-byte store.
        const m32 = comptime blk: {
            var m: [32]i32 = undefined;
            for (0..32) |t| {
                const lane = t / 16;
                const p = (t % 16) / 4;
                const c = t % 4;
                m[t] = if (map[c]) |sc| @intCast(16 * lane + p * 3 + sc) else -1;
            }
            break :blk m;
        };
        const f: @Vector(1, u8) = .{fill};
        var x: usize = 0;
        while (x * 3 + 28 <= n * 3) : (x += 8) {
            const lo = opaqueV(s[x * 3 ..][0..16].*);
            const hi = opaqueV(s[x * 3 + 12 ..][0..16].*);
            const v: @Vector(32, u8) = @shuffle(u8, lo, hi, blk: {
                var m: [32]i32 = undefined;
                for (0..16) |i| {
                    m[i] = @intCast(i);
                    m[16 + i] = ~@as(i32, @intCast(i));
                }
                break :blk m;
            });
            d[x * 4 ..][0..32].* = @shuffle(u8, v, f, m32);
        }
        return x;
    }
    const widest: usize = if (scn > dcn) scn else dcn;
    const per: usize = 16 / widest;
    const mask = comptime blk: {
        var m: [16]i32 = undefined;
        for (0..16) |t| {
            const p = t / dcn;
            const c = t % dcn;
            m[t] = if (p >= per) 0 else if (map[c]) |sc| @intCast(p * scn + sc) else -1;
        }
        break :blk m;
    };
    const f: @Vector(1, u8) = .{fill};
    var x: usize = 0;
    // Need 16 readable bytes at x*scn and 16 writable at x*dcn. (Unrolling
    // this by hand makes LLVM scalarise the stores into byte extracts.)
    while (x * scn + 16 <= n * scn and x * dcn + 16 <= n * dcn) : (x += per) {
        const v = opaqueV(s[x * scn ..][0..16].*);
        d[x * dcn ..][0..16].* = @shuffle(u8, v, f, mask);
    }
    return x;
}

/// x86 3 -> 3 channel permutation (BGR <-> RGB). Output byte i is input
/// byte i + map[i % 3] - i % 3, a shift of at most two bytes, so with two
/// unaligned loads 2 bytes before and 2 bytes after the output position
/// every output byte is in the same 16-byte lane of one of them: each
/// output vector is two in-lane pshufbs and an or, stored whole (32 bytes
/// with AVX2). Unlike 5-pixel pshufb steps with overlapping 16-byte
/// stores, this keeps the store count at one per vector. Three vectors
/// (three mask phases) per step, from pixel 1 on so the loads stay inside
/// the row, plus one overlapping step at the row end; pixel 0 is converted
/// here. Returns the pixels done from 0.
inline fn permute3(comptime map: [3]?u8, s: [*]const u8, d: [*]u8, n: usize) usize {
    const W = if (has_avx2) 32 else 16;
    const V = @Vector(W, u8);
    // ma[t][i] / mb[t][i]: lane-local source index of output byte i of
    // vector t in the load at -2 / +2, or a zero lane (index into the
    // second operand).
    const masks = comptime blk: {
        var ma: [3][W]i32 = undefined;
        var mb: [3][W]i32 = undefined;
        for (0..3) |t| for (0..W) |i| {
            const c = (W * t + i) % 3;
            const delta: i32 = @as(i32, map[c].?) - @as(i32, c);
            const li: i32 = @intCast(i % 16);
            const base: i32 = @intCast(i - i % 16);
            if (li + delta + 2 <= 15) {
                ma[t][i] = base + li + delta + 2;
                mb[t][i] = -1;
            } else {
                ma[t][i] = -1;
                mb[t][i] = base + li + delta - 2;
            }
        };
        break :blk .{ ma, mb };
    };
    const end = 3 * n;
    if (end < 3 * W + 5) return 0;
    const z: V = @splat(0);
    const Step = struct {
        inline fn opaque_(v: V) V {
            return asm (""
                : [ret] "=x" (-> V),
                : [v] "0" (v),
            );
        }
        inline fn run(src: [*]const u8, dst: [*]u8, o: usize) void {
            inline for (0..3) |t| {
                const a = opaque_(src[o + W * t - 2 ..][0..W].*);
                const b = opaque_(src[o + W * t + 2 ..][0..W].*);
                dst[o + W * t ..][0..W].* = @shuffle(u8, a, z, masks[0][t]) | @shuffle(u8, b, z, masks[1][t]);
            }
        }
    };
    var o: usize = 3;
    while (o + 3 * W + 2 <= end) : (o += 3 * W) Step.run(s, d, o);
    // One more step ending near the row end, overlapping the last one
    // (same bytes rewritten), leaves at most a pixel or two.
    const last = (end - 3 * W - 2) / 3 * 3;
    if (last + 3 * W > o) {
        Step.run(s, d, last);
        o = last + 3 * W;
    }
    inline for (0..3) |c| d[c] = s[map[c].?];
    return o / 3;
}

/// gather() restricted to the listed blocks.
inline fn gatherFrom(comptime total: usize, blocks: [total]V16, comptime list: anytype, comptime src: [16]usize) V16 {
    const n = list.len;
    var sub: [n]V16 = undefined;
    inline for (0..n) |k| sub[k] = blocks[list[k]];
    const local = comptime blk: {
        var l: [16]usize = undefined;
        for (0..16) |i| {
            const blkno = src[i] / 16;
            for (list, 0..) |b, k| if (b == blkno) {
                l[i] = k * 16 + src[i] % 16;
            };
        }
        break :blk l;
    };
    return gather(n, sub, local);
}

test "remap 3 -> 4 with fill" {
    var buf: [48]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @intCast(i);
    var out: [64]u8 = undefined;
    remap(3, 4, .{ 2, 1, 0, null }, 255, &buf, &out);
    for (0..16) |i| {
        try std.testing.expectEqual(buf[i * 3 + 2], out[i * 4]);
        try std.testing.expectEqual(buf[i * 3 + 1], out[i * 4 + 1]);
        try std.testing.expectEqual(buf[i * 3], out[i * 4 + 2]);
        try std.testing.expectEqual(@as(u8, 255), out[i * 4 + 3]);
    }
}

test "load64/store64 round trip" {
    var buf: [192]u8 = undefined;
    for (&buf, 0..) |*v, i| v.* = @truncate(i * 7);
    const ch = load64(3, &buf);
    var out: [192]u8 = undefined;
    store64(3, ch, &out);
    try std.testing.expectEqualSlices(u8, &buf, &out);
    const c1: [64]u8 = ch[1];
    for (c1, 0..) |v, i| try std.testing.expectEqual(buf[3 * i + 1], v);
}

test "reorderRow matches scalar" {
    var src: [3 * 37]u8 = undefined;
    for (&src, 0..) |*v, i| v.* = @intCast(i);
    var dst: [4 * 37]u8 = @splat(0);
    const done = reorderRow(3, 4, .{ 2, 1, 0, null }, 255, &src, &dst, 37);
    try std.testing.expect(done > 0 and done <= 37);
    for (0..done) |i| {
        try std.testing.expectEqual(src[3 * i + 2], dst[4 * i]);
        try std.testing.expectEqual(src[3 * i], dst[4 * i + 2]);
        try std.testing.expectEqual(@as(u8, 255), dst[4 * i + 3]);
    }
    var d3: [3 * 37]u8 = @splat(0);
    const done3 = reorderRow(3, 3, .{ 2, 1, 0 }, 0, &src, &d3, 37);
    try std.testing.expect(done3 > 0 and done3 <= 37);
    for (0..done3) |i| for (0..3) |c| try std.testing.expectEqual(src[3 * i + 2 - c], d3[3 * i + c]);
}

test "load/store round trip" {
    inline for (.{ 2, 3, 4 }) |cn| {
        var buf: [16 * cn]u8 = undefined;
        for (&buf, 0..) |*v, i| v.* = @intCast(i);
        const ch = load(cn, &buf);
        for (0..cn) |c| {
            const arr: [16]u8 = ch[c];
            for (arr, 0..) |v, i| try std.testing.expectEqual(@as(u8, @intCast(i * cn + c)), v);
        }
        var out: [16 * cn]u8 = undefined;
        store(cn, ch, &out);
        try std.testing.expectEqualSlices(u8, &buf, &out);
    }
}
