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
const arm_tbl = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .aarch64;

/// Targets with a fast 64-byte byte permute for the group path.
pub const fast_permute = has_vbmi or arm_tbl;

const neon = struct {
    const V = @Vector(16, u8);
    extern fn @"llvm.aarch64.neon.tbl4.v16i8"(V, V, V, V, V) V;
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

/// Four destination pixels of a bilinear row: a 64-byte source window at
/// `base`, the byte index of each (pixel, channel-lane, tap) pair, and the
/// matching 16-bit weights. `ok` is false when the window does not fit.
pub const Group = struct {
    idx: [64]u8 align(64),
    w: [32]i16 align(64),
    base: u32,
    ok: bool,
};

/// Builds the groups for a bilinear row of `rowlen` source samples.
pub fn groups(gpa: Allocator, xt: []const Tap(2), cn: usize, rowlen: usize) Allocator.Error![]Group {
    const g = try gpa.alloc(Group, xt.len / 4);
    for (g, 0..) |*gr, gi| {
        const base = xt[gi * 4].o[0];
        gr.base = base;
        gr.ok = base + 64 <= rowlen;
        gr.idx = @splat(0);
        gr.w = @splat(0);
        for (0..4) |p| {
            const t = xt[gi * 4 + p];
            for (0..4) |l| for (0..2) |tap| {
                const pair = (p * 4 + l) * 2 + tap;
                if (l >= cn) continue;
                const off = t.o[tap] + l - base;
                if (off >= 64) gr.ok = false else gr.idx[2 * pair] = @intCast(off);
                gr.w[pair] = @intCast(t.w[tap]);
            };
        }
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

/// Bilinear horizontal pass for one group of 4 destination pixels.
inline fn hrowGroup(comptime cn: usize, s: []const u8, out: []i16, g: Group) void {
    const lane_mask = comptime blk: {
        var m: [4 * cn]i32 = undefined;
        for (0..4) |p| for (0..cn) |l| {
            m[p * cn + l] = @intCast(p * 4 + l);
        };
        break :blk m;
    };
    const win: @Vector(64, u8) = s[g.base..][0..64].*;
    const bytes = permb(win, g.idx);
    const a: @Vector(32, i16) = @bitCast(@as(@Vector(32, u16), @bitCast(bytes)) & @as(@Vector(32, u16), @splat(0xff)));
    const acc = maddwd(a, g.w) >> @splat(4);
    const r: @Vector(16, i16) = @truncate(acc);
    out[0 .. 4 * cn].* = @shuffle(i16, r, undefined, lane_mask);
}

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
    const n = c.d.rowLen();
    const stride = n + 4; // room for the 4-lane store of the last pixel
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
        for (gs, 0..) |g, gi| {
            if (g.ok) hrowGroup(cn, s, out[gi * 4 * cn ..], g) else hrowPixels(K, cn, E, s, out, xt, gi * 4, gi * 4 + 4);
        }
        return hrowPixels(K, cn, E, s, out, xt, gs.len * 4, xt.len);
    }
    hrowPixels(K, cn, E, s, out, xt, 0, xt.len);
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
        d[x..][0..N].* = @as(@Vector(N, u8), @intCast(cl));
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
    const I = @Vector(16, i32);
    const q: I = if (use_asm and avx512f)
        asm ("vcvtps2dq %[v], %[ret]"
            : [ret] "=v" (-> I),
            : [v] "v" (v),
        )
    else
        @intFromFloat(@min(@max(sat.rint(v), @as(@Vector(16, f32), @splat(-1.0))), @as(@Vector(16, f32), @splat(256.0))));
    return simd.narrow(u8, @min(@max(q, @as(I, @splat(0))), @as(I, @splat(255))));
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
