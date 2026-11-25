//! Convolution kernels over the blocked layout ([N][C/B][H][W][B]).
//!
//! - `Pointwise`: 1x1, stride 1, no padding, one group. A GEMM where the
//!   weights (at most a few hundred KB, packed per output block at plan time)
//!   stay in cache while pixel chunks stream past. Register tile: `R`
//!   pixels x `NB` output blocks of accumulators, fed per input channel by
//!   `NB` weight vectors and `R` broadcasts.
//! - `Depthwise`: one group per channel. Each lane of a vector is one
//!   channel, so a 3x3 tap is one multiply-add per pixel; the fast path
//!   produces four pixels per step from six loaded columns per row.
//! - `Direct`: any other single-group conv (strided, KxK, few input
//!   channels such as a network's first layer reading the plain NCHW input).
//! - `Reference`: grouped convolutions that are neither; plain loops.
//!
//! Every kernel applies the fused epilogue (bias, residual, activation) and
//! zeroes the padding lanes of the last output block.

const std = @import("std");
const ir = @import("../ir.zig");
const vec = @import("vec.zig");
const core = @import("../../core/core.zig");
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

const B = vec.B;
const V = vec.V;
const fma = vec.fma;
const ActKind = vec.ActKind;
const Rows = vec.Rows;

/// Geometry of one convolution (one image of the batch).
pub const Geom = struct {
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    pt: usize,
    pl: usize,
    dh: usize,
    dw: usize,
    group: usize,

    pub fn cinb(g: Geom) usize {
        return (g.cin + B - 1) / B;
    }
    pub fn coutb(g: Geom) usize {
        return (g.cout + B - 1) / B;
    }
};

/// Element offsets of the operands in the arena (`res` maxInt if none).
pub const Io = struct {
    x: usize,
    out: usize,
    res: usize = no_res,
    /// Input is plain NCHW instead of blocked (Direct and Reference).
    x_plain: bool = false,
    /// Output (and residual) plain NCHW (Reference only).
    out_plain: bool = false,
};
pub const no_res = std.math.maxInt(usize);

pub const Epilogue = struct {
    act: ir.Act,
    /// `vec.linearSlope(act)`.
    slope: f32,
    residual_after_act: bool,
    /// Lanes of the last output block that hold real channels.
    tail_mask: @Vector(B, bool),
    coutb: usize,
    /// The last block has padding lanes (cout % B != 0).
    tail: bool,

    fn init(c: ir.Conv, cout: usize) Epilogue {
        const rem = cout % B;
        return .{ .act = c.act, .slope = vec.linearSlope(c.act), .residual_after_act = c.residual_after_act, .tail_mask = vec.laneMask(if (rem == 0) B else rem), .coutb = (cout + B - 1) / B, .tail = rem != 0 };
    }

    inline fn apply(e: *const Epilogue, comptime ak: ActKind, acc: V, res: ?[*]const f32, cob: usize) V {
        var v = acc;
        if (res) |r| {
            if (!e.residual_after_act) v += vec.load(r);
        }
        v = vec.act(ak, e.act, e.slope, v);
        if (res) |r| {
            if (e.residual_after_act) v += vec.load(r);
        }
        if (e.tail and cob + 1 == e.coutb) v = @select(f32, e.tail_mask, v, vec.splat(0));
        return v;
    }

    /// Stores an `R` pixels x `NB` blocks register tile through the
    /// epilogue (a `deferred` activation in a second pass over the stored
    /// sums): `o` and `res` point at block `cob`, pixel 0 of the tile;
    /// `os`, `rs` are the distances between their blocks.
    inline fn storeTile(ep: *const Epilogue, comptime ak: ActKind, comptime R: usize, comptime NB: usize, acc: *const [R][NB]V, o: [*]f32, os: usize, res: ?[*]const f32, rs: usize, cob: usize) void {
        // A local copy (the stores could alias `ep`: every one would reload
        // its fields), and the residual tested once per tile.
        const ev = ep.*;
        const e = &ev;
        if (ak != .deferred) {
            if (res) |rr| {
                inline for (0..NB) |nb| {
                    inline for (0..R) |r| vec.store(o + nb * os + r * B, e.apply(ak, acc[r][nb], rr + nb * rs + r * B, cob + nb));
                }
            } else {
                inline for (0..NB) |nb| {
                    inline for (0..R) |r| vec.store(o + nb * os + r * B, e.apply(ak, acc[r][nb], null, cob + nb));
                }
            }
            return;
        }
        inline for (0..NB) |nb| {
            inline for (0..R) |r| vec.store(o + nb * os + r * B, acc[r][nb]);
        }
        for (0..NB) |nb| {
            for (0..R) |r| {
                const rp: ?[*]const f32 = if (res) |rr| rr + nb * rs + r * B else null;
                const p = o + nb * os + r * B;
                vec.store(p, e.apply(ak, vec.load(p), rp, cob + nb));
            }
        }
    }
};

fn packBias(arena: Allocator, bias: ?[]const f32, cout: usize) Allocator.Error![]align(64) f32 {
    const nb = (cout + B - 1) / B;
    const out = try arena.alignedAlloc(f32, .@"64", nb * B);
    @memset(out, 0);
    if (bias) |b| @memcpy(out[0..cout], b);
    return out;
}

// ------------------------------------------------------------- pointwise

pub const Pointwise = struct {
    g: Geom,
    io: Io,
    /// [coutb][cin_pad][B]
    w: []align(64) f32,
    bias: []align(64) f32,
    epi: Epilogue,
    cin_pad: usize,

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Pointwise {
        const cin_pad = g.cinb() * B;
        const cb = g.coutb();
        const wp = try arena.alignedAlloc(f32, .@"64", cb * cin_pad * B);
        @memset(wp, 0);
        for (0..g.cout) |o| {
            for (0..g.cin) |i| wp[((o / B) * cin_pad + i) * B + o % B] = w[o * g.cin + i];
        }
        return .{ .g = g, .io = io, .w = wp, .bias = try packBias(arena, bias, g.cout), .epi = .init(c, g.cout), .cin_pad = cin_pad };
    }

    const chunk = 96;

    const Ctx = struct { k: *const Pointwise, base: [*]f32, img: usize };

    pub fn run(k: *const Pointwise, exec: anytype, base: [*]f32) void {
        const p = k.g.h * k.g.w;
        const chunks = (p + chunk - 1) / chunk;
        for (0..k.g.n) |img| {
            const ctx: Ctx = .{ .k = k, .base = base, .img = img };
            switch (vec.actKind(k.epi.act)) {
                inline else => |ak| exec.parallelFor(chunks, 1, ctx, struct {
                    fn f(c: Ctx, s: usize, e: usize) void {
                        const g = c.k.g;
                        const plane = g.h * g.w * B;
                        const x = c.base + c.k.io.x + c.img * g.cinb() * plane;
                        const out = c.base + c.k.io.out + c.img * g.coutb() * plane;
                        const res: ?[*]f32 = if (c.k.io.res != no_res) c.base + c.k.io.res + c.img * g.coutb() * plane else null;
                        const p1 = @min(g.h * g.w, e * chunk);
                        c.k.sweep(ak, x, plane, out, plane, res, plane, s * chunk, p1);
                    }
                }.f),
            }
        }
    }

    /// One output row of a streamed chain.
    pub fn streamRow(k: *const Pointwise, comptime ak: ActKind, src: Rows, dst: Rows, res: ?Rows, y: usize) void {
        const rp: ?[*]f32 = if (res) |r| r.at(0, y) else null;
        k.sweep(ak, src.at(0, y), src.cb_stride, dst.at(0, y), dst.cb_stride, rp, if (res) |r| r.cb_stride else 0, 0, k.g.w);
    }

    /// Pixels [p0, p1) of the flat pixel index space; `xs`, `os`, `rs`
    /// are the distances between channel blocks of input, output, residual.
    fn sweep(k: *const Pointwise, comptime ak: ActKind, x: [*]const f32, xs: usize, out: [*]f32, os: usize, res: ?[*]const f32, rs: usize, p0: usize, p1: usize) void {
        var cob: usize = 0;
        const coutb = k.g.coutb();
        while (cob < coutb) {
            const nb = @min(vec.max_nb, coutb - cob);
            switch (nb) {
                inline 1...vec.max_nb => |NB| {
                    const R = comptime vec.tileR(NB);
                    // As few tiles as R allows, of near-equal widths: a
                    // narrow tile (a streamed 13- or 26-pixel row's last
                    // one or two pixels) is bound by FMA latency, not
                    // throughput. (Widths from `lo` get a tile of their
                    // own; narrower ones, from short rows, halving tiles.)
                    const lo = comptime @max((R + 1) / 2, R -| 3);
                    const n = p1 - p0;
                    const nt = (n + R - 1) / R;
                    var px = p0;
                    for (0..nt) |i| {
                        const sz = n / nt + @intFromBool(i < n % nt);
                        switch (sz) {
                            inline lo...R => |T| k.tile(T, NB, ak, x, xs, out, os, res, rs, cob, px),
                            else => {
                                var q = px;
                                inline for (.{ 8, 4, 2, 1 }) |T| {
                                    if (T < lo and q + T <= px + sz) {
                                        k.tile(T, NB, ak, x, xs, out, os, res, rs, cob, q);
                                        q += T;
                                    }
                                }
                                while (q < px + sz) : (q += 1) k.tile(1, NB, ak, x, xs, out, os, res, rs, cob, q);
                            },
                        }
                        px += sz;
                    }
                },
                else => unreachable,
            }
            cob += nb;
        }
    }

    fn tile(kp: *const Pointwise, comptime R: usize, comptime NB: usize, comptime ak: ActKind, x: [*]const f32, xs: usize, out: [*]f32, os: usize, res: ?[*]const f32, rs: usize, cob: usize, px: usize) void {
        const kv = kp.*; // (as in Depthwise.row)
        const k = &kv;
        @setEvalBranchQuota(100000);
        const wcs = k.cin_pad * B;
        const w = k.w.ptr + cob * wcs;
        var acc: [R][NB]V = undefined;
        inline for (0..NB) |nb| {
            const bv = vec.load(k.bias.ptr + (cob + nb) * B);
            inline for (0..R) |r| acc[r][nb] = bv;
        }
        const cinb = k.g.cinb();
        for (0..cinb) |cb| {
            const xp = x + cb * xs + px * B;
            const wb = w + cb * B * B;
            var xv: [R]V = undefined;
            if (vec.lane_broadcast) {
                inline for (0..R) |r| xv[r] = vec.load(xp + r * B);
            }
            inline for (0..B) |j| {
                var wv: [NB]V = undefined;
                inline for (0..NB) |nb| wv[nb] = vec.load(wb + nb * wcs + j * B);
                inline for (0..R) |r| {
                    const bc: V = if (vec.lane_broadcast) @shuffle(f32, xv[r], undefined, @as(@Vector(B, i32), @splat(j))) else @splat(xp[r * B + j]);
                    inline for (0..NB) |nb| acc[r][nb] = fma(bc, wv[nb], acc[r][nb]);
                }
            }
        }
        const rp: ?[*]const f32 = if (res) |rr| rr + cob * rs + px * B else null;
        k.epi.storeTile(ak, R, NB, &acc, out + cob * os + px * B, os, rp, rs, cob);
    }
};

// ------------------------------------------------------------- depthwise

pub const Depthwise = struct {
    g: Geom,
    io: Io,
    /// [cb][kh*kw][B]
    w: []align(64) f32,
    bias: []align(64) f32,
    epi: Epilogue,
    /// 3x3, stride 1, pad 1, dilation 1.
    fast3: bool,
    /// 3x3, stride 2, pad 1, dilation 1.
    fast3s2: bool,
    /// One input row of zeros: the fast paths read it for a kernel row
    /// outside the input (adding exact zeros), so the first and last rows
    /// need no checked path.
    zero: []align(64) f32,

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Depthwise {
        const kk = g.kh * g.kw;
        const cb = g.coutb();
        const wp = try arena.alignedAlloc(f32, .@"64", cb * kk * B);
        @memset(wp, 0);
        for (0..g.cout) |ch| {
            for (0..kk) |t| wp[((ch / B) * kk + t) * B + ch % B] = w[ch * kk + t];
        }
        const fast3 = g.kh == 3 and g.kw == 3 and g.sh == 1 and g.sw == 1 and g.pt == 1 and g.pl == 1 and g.dh == 1 and g.dw == 1 and g.oh == g.h and g.ow == g.w;
        const fast3s2 = g.kh == 3 and g.kw == 3 and g.sh == 2 and g.sw == 2 and g.pt == 1 and g.pl == 1 and g.dh == 1 and g.dw == 1;
        const zero = try arena.alignedAlloc(f32, .@"64", if (fast3 or fast3s2) g.w * B else 0);
        @memset(zero, 0);
        return .{ .g = g, .io = io, .w = wp, .bias = try packBias(arena, bias, g.cout), .epi = .init(c, g.cout), .fast3 = fast3, .fast3s2 = fast3s2, .zero = zero };
    }

    const Ctx = struct { k: *const Depthwise, base: [*]f32 };

    pub fn run(k: *const Depthwise, exec: anytype, base: [*]f32) void {
        const rows = k.g.n * k.g.coutb() * k.g.oh;
        const ctx: Ctx = .{ .k = k, .base = base };
        switch (vec.actKind(k.epi.act)) {
            inline else => |ak| exec.parallelFor(rows, 4, ctx, struct {
                fn f(c: Ctx, s: usize, e: usize) void {
                    const g = c.k.g;
                    for (s..e) |r| {
                        const oy = r % g.oh;
                        const cb = (r / g.oh) % g.coutb();
                        const img = r / g.oh / g.coutb();
                        const src = Rows.tensor(c.base, c.k.io.x, img, g.cinb(), g.h, g.w);
                        const dst = Rows.tensor(c.base, c.k.io.out, img, g.coutb(), g.oh, g.ow);
                        const res: ?Rows = if (c.k.io.res != no_res) Rows.tensor(c.base, c.k.io.res, img, g.coutb(), g.oh, g.ow) else null;
                        c.k.row(ak, src, dst, res, cb, oy);
                    }
                }
            }.f),
        }
    }

    pub fn streamRow(k: *const Depthwise, comptime ak: ActKind, src: Rows, dst: Rows, res: ?Rows, y: usize) void {
        for (0..k.g.coutb()) |cb| k.row(ak, src, dst, res, cb, y);
    }

    fn row(k: *const Depthwise, comptime ak: ActKind, src: Rows, dst: Rows, res_rows: ?Rows, cb: usize, oy: usize) void {
        // The residual decided once per row, not at every pixel.
        if (res_rows) |r| k.rowRes(ak, src, dst, r.at(cb, oy), cb, oy) else k.rowRes(ak, src, dst, null, cb, oy);
    }

    inline fn rowRes(kp: *const Depthwise, comptime ak: ActKind, src: Rows, dst: Rows, res: ?[*]const f32, cb: usize, oy: usize) void {
        // A local copy: the stores below could alias the kernel, so LLVM
        // would reload its fields (epilogue flags, slope) at every store.
        const kv = kp.*;
        const k = &kv;
        @setEvalBranchQuota(10000);
        const g = k.g;
        const out = dst.at(cb, oy);
        const kk = g.kh * g.kw;
        const w = k.w.ptr + cb * kk * B;
        const bias = vec.load(k.bias.ptr + cb * B);
        if (k.fast3 and g.ow >= 3) {
            var wv: [9]V = undefined;
            inline for (0..9) |t| wv[t] = vec.load(w + t * B);
            const rows = [3][*]const f32{
                if (oy > 0) src.at(cb, oy - 1) else k.zero.ptr,
                src.at(cb, oy),
                if (oy + 1 < g.h) src.at(cb, oy + 1) else k.zero.ptr,
            };
            k.edge3(ak, rows, &wv, bias, 1, 3, 0, out, res, cb, 0);
            var ox: usize = 1;
            while (ox + 4 < g.ow) : (ox += 4) {
                var acc: [4]V = @splat(bias);
                inline for (0..3) |ky| {
                    const rp = rows[ky] + (ox - 1) * B;
                    var v: [6]V = undefined;
                    inline for (0..6) |i| v[i] = vec.load(rp + i * B);
                    inline for (0..4) |i| {
                        inline for (0..3) |kx| acc[i] = fma(v[i + kx], wv[ky * 3 + kx], acc[i]);
                    }
                }
                inline for (0..4) |i| {
                    const rp: ?[*]const f32 = if (res) |rr| rr + (ox + i) * B else null;
                    vec.store(out + (ox + i) * B, k.epi.apply(ak, acc[i], rp, cb));
                }
            }
            while (ox + 1 < g.ow) : (ox += 1) {
                var acc = bias;
                inline for (0..3) |ky| {
                    const rp = rows[ky] + (ox - 1) * B;
                    inline for (0..3) |kx| acc = fma(vec.load(rp + kx * B), wv[ky * 3 + kx], acc);
                }
                const rp: ?[*]const f32 = if (res) |rr| rr + ox * B else null;
                vec.store(out + ox * B, k.epi.apply(ak, acc, rp, cb));
            }
            k.edge3(ak, rows, &wv, bias, 0, 2, g.w - 2, out, res, cb, g.ow - 1);
            return;
        }
        if (k.fast3s2 and 2 * oy < g.h and g.w >= 2) {
            // Columns 1 .. while 2*ox+1 < w have all three inside.
            var wv: [9]V = undefined;
            inline for (0..9) |t| wv[t] = vec.load(w + t * B);
            const rows = [3][*]const f32{
                if (oy > 0) src.at(cb, 2 * oy - 1) else k.zero.ptr,
                src.at(cb, 2 * oy),
                if (2 * oy + 1 < g.h) src.at(cb, 2 * oy + 1) else k.zero.ptr,
            };
            const ox_hi = @min(g.ow, g.w / 2);
            k.edge3(ak, rows, &wv, bias, 1, 3, 0, out, res, cb, 0);
            var ox: usize = 1;
            while (ox + 4 <= ox_hi) : (ox += 4) {
                var acc: [4]V = @splat(bias);
                inline for (0..3) |ky| {
                    const rp = rows[ky] + (2 * ox - 1) * B;
                    var v: [9]V = undefined;
                    inline for (0..9) |i| v[i] = vec.load(rp + i * B);
                    inline for (0..4) |i| {
                        inline for (0..3) |kx| acc[i] = fma(v[2 * i + kx], wv[ky * 3 + kx], acc[i]);
                    }
                }
                inline for (0..4) |i| {
                    const rp: ?[*]const f32 = if (res) |rr| rr + (ox + i) * B else null;
                    vec.store(out + (ox + i) * B, k.epi.apply(ak, acc[i], rp, cb));
                }
            }
            while (ox < ox_hi) : (ox += 1) {
                var acc = bias;
                inline for (0..3) |ky| {
                    const rp = rows[ky] + (2 * ox - 1) * B;
                    inline for (0..3) |kx| acc = fma(vec.load(rp + kx * B), wv[ky * 3 + kx], acc);
                }
                const rp: ?[*]const f32 = if (res) |rr| rr + ox * B else null;
                vec.store(out + ox * B, k.epi.apply(ak, acc, rp, cb));
            }
            while (ox < g.ow) : (ox += 1) {
                if (2 * ox < g.w) k.edge3(ak, rows, &wv, bias, 0, 2, 2 * ox - 1, out, res, cb, ox) else k.pixel(ak, src, out, res, w, bias, cb, oy, ox);
            }
            return;
        }
        for (0..g.ow) |ox| k.pixel(ak, src, out, res, w, bias, cb, oy, ox);
    }

    /// Output pixel `ox` of a 3x3 fast path whose kernel columns
    /// [kx_lo, kx_hi) lie inside the input, the first at input column `c`
    /// (the same sums in the same order as `pixel`, plus exact zeros for a
    /// zero row).
    inline fn edge3(k: *const Depthwise, comptime ak: ActKind, rows: [3][*]const f32, wv: *const [9]V, bias: V, comptime kx_lo: usize, comptime kx_hi: usize, c: usize, out: [*]f32, res: ?[*]const f32, cb: usize, ox: usize) void {
        var acc = bias;
        inline for (0..3) |ky| {
            inline for (kx_lo..kx_hi) |kx| acc = fma(vec.load(rows[ky] + (c + kx - kx_lo) * B), wv[ky * 3 + kx], acc);
        }
        const rp: ?[*]const f32 = if (res) |rr| rr + ox * B else null;
        vec.store(out + ox * B, k.epi.apply(ak, acc, rp, cb));
    }

    /// One output pixel with bounds checks (any kernel, stride, dilation).
    inline fn pixel(k: *const Depthwise, comptime ak: ActKind, src: Rows, out: [*]f32, res: ?[*]const f32, w: [*]const f32, bias: V, cb: usize, oy: usize, ox: usize) void {
        const g = k.g;
        var acc = bias;
        for (0..g.kh) |ky| {
            const iy = @as(isize, @intCast(oy * g.sh + ky * g.dh)) - @as(isize, @intCast(g.pt));
            if (iy < 0 or iy >= @as(isize, @intCast(g.h))) continue;
            const rp = src.at(cb, @intCast(iy));
            for (0..g.kw) |kx| {
                const ix = @as(isize, @intCast(ox * g.sw + kx * g.dw)) - @as(isize, @intCast(g.pl));
                if (ix < 0 or ix >= @as(isize, @intCast(g.w))) continue;
                acc = fma(vec.load(rp + @as(usize, @intCast(ix)) * B), vec.load(w + (ky * g.kw + kx) * B), acc);
            }
        }
        const rp: ?[*]const f32 = if (res) |rr| rr + ox * B else null;
        vec.store(out + ox * B, k.epi.apply(ak, acc, rp, cb));
    }
};

// ---------------------------------------------------------------- direct

pub const Direct = struct {
    g: Geom,
    io: Io,
    /// [coutb][kh][kw][cin][B]
    w: []align(64) f32,
    bias: []align(64) f32,
    epi: Epilogue,
    /// Element offset of input channel `ci` within one image.
    ci_off: []usize,
    /// Taps of an interior tile: input offset from the window's top-left
    /// sample, and weight offset (both in elements), in (ky, kx, ci) order.
    taps: []Tap,
    /// Distance between horizontally adjacent input pixels, and rows.
    ps: usize,
    rs: usize,
    img_stride: usize,
    /// Output columns and rows whose taps all lie inside the input.
    ox_lo: usize,
    ox_hi: usize,
    oy_lo: usize,
    oy_hi: usize,

    const Tap = struct { x: usize, w: usize };

    fn interior(lo_pad: usize, size: usize, k: usize, s: usize, d: usize, out: usize) [2]usize {
        // o*s - pad >= 0 and o*s - pad + (k-1)*d <= size-1
        const lo = (lo_pad + s - 1) / s;
        const reach = lo_pad + size;
        const span = (k - 1) * d + 1;
        const hi = if (reach >= span) @min(out, (reach - span) / s + 1) else 0;
        return .{ @min(lo, out), @max(hi, @min(lo, out)) };
    }

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Direct {
        const kk = g.kh * g.kw;
        const cb = g.coutb();
        const wp = try arena.alignedAlloc(f32, .@"64", cb * g.cin * kk * B);
        @memset(wp, 0);
        for (0..g.cout) |o| {
            for (0..g.cin) |i| {
                for (0..kk) |t| wp[(((o / B) * kk + t) * g.cin + i) * B + o % B] = w[(o * g.cin + i) * kk + t];
            }
        }
        const ci_off = try arena.alloc(usize, g.cin);
        for (ci_off, 0..) |*o, i| o.* = if (io.x_plain) i * g.h * g.w else (i / B) * g.h * g.w * B + i % B;
        const ps: usize = if (io.x_plain) 1 else B;
        const rs = g.w * ps;
        const taps = try arena.alloc(Tap, kk * g.cin);
        for (0..g.kh) |ky| for (0..g.kw) |kx| for (0..g.cin) |ci| {
            const t = (ky * g.kw + kx) * g.cin + ci;
            taps[t] = .{ .x = ky * g.dh * rs + kx * g.dw * ps + ci_off[ci], .w = t * B };
        };
        const xr = interior(g.pl, g.w, g.kw, g.sw, g.dw, g.ow);
        const yr = interior(g.pt, g.h, g.kh, g.sh, g.dh, g.oh);
        return .{
            .g = g,
            .io = io,
            .w = wp,
            .bias = try packBias(arena, bias, g.cout),
            .epi = .init(c, g.cout),
            .ci_off = ci_off,
            .taps = taps,
            .ps = ps,
            .rs = rs,
            .img_stride = if (io.x_plain) g.cin * g.h * g.w else g.cinb() * g.h * g.w * B,
            .ox_lo = xr[0],
            .ox_hi = xr[1],
            .oy_lo = yr[0],
            .oy_hi = yr[1],
        };
    }

    const Ctx = struct { k: *const Direct, base: [*]f32 };

    pub fn run(k: *const Direct, exec: anytype, base: [*]f32) void {
        const rows = k.g.n * k.g.oh;
        const ctx: Ctx = .{ .k = k, .base = base };
        switch (vec.actKind(k.epi.act)) {
            inline else => |ak| exec.parallelFor(rows, 1, ctx, struct {
                fn f(c: Ctx, s: usize, e: usize) void {
                    const g = c.k.g;
                    for (s..e) |r| {
                        const img = r / g.oh;
                        const dst = Rows.tensor(c.base, c.k.io.out, img, g.coutb(), g.oh, g.ow);
                        const res: ?Rows = if (c.k.io.res != no_res) Rows.tensor(c.base, c.k.io.res, img, g.coutb(), g.oh, g.ow) else null;
                        c.k.row(ak, c.base, img, dst, res, r % g.oh);
                    }
                }
            }.f),
        }
    }

    /// One output row of a streamed chain (image 0; reads the input tensor).
    pub fn streamRow(k: *const Direct, comptime ak: ActKind, base: [*]f32, dst: Rows, res: ?Rows, y: usize) void {
        k.row(ak, base, 0, dst, res, y);
    }

    fn row(k: *const Direct, comptime ak: ActKind, base: [*]f32, img: usize, dst: Rows, res_rows: ?Rows, oy: usize) void {
        const g = k.g;
        const x = base + k.io.x + img * k.img_stride;
        const out = dst.at(0, oy);
        const res: ?[*]const f32 = if (res_rows) |r| r.at(0, oy) else null;
        const rs = if (res_rows) |r| r.cb_stride else 0;
        const inner_y = oy >= k.oy_lo and oy < k.oy_hi;
        const coutb = g.coutb();
        var cob: usize = 0;
        while (cob < coutb) {
            const nb = @min(vec.max_nb, coutb - cob);
            switch (nb) {
                inline 1...vec.max_nb => |NB| {
                    const R = comptime vec.tileR(NB);
                    var ox: usize = 0;
                    if (inner_y) {
                        while (ox < k.ox_lo) : (ox += 1) k.tile(1, NB, ak, true, x, out, dst.cb_stride, res, rs, cob, oy, ox);
                        while (ox + R <= k.ox_hi) : (ox += R) k.tile(R, NB, ak, false, x, out, dst.cb_stride, res, rs, cob, oy, ox);
                        inline for (.{ 8, 4, 2, 1 }) |T| {
                            if (T < R and ox + T <= k.ox_hi) {
                                k.tile(T, NB, ak, false, x, out, dst.cb_stride, res, rs, cob, oy, ox);
                                ox += T;
                            }
                        }
                        while (ox < k.ox_hi) : (ox += 1) k.tile(1, NB, ak, false, x, out, dst.cb_stride, res, rs, cob, oy, ox);
                    }
                    while (ox < g.ow) : (ox += 1) k.tile(1, NB, ak, true, x, out, dst.cb_stride, res, rs, cob, oy, ox);
                },
                else => unreachable,
            }
            cob += nb;
        }
    }

    /// `R` output pixels from `ox` of row `oy`, output blocks [cob, cob+NB).
    /// Unchecked tiles have every tap inside the input.
    fn tile(k: *const Direct, comptime R: usize, comptime NB: usize, comptime ak: ActKind, comptime checked: bool, x: [*]const f32, out: [*]f32, os: usize, res: ?[*]const f32, rs: usize, cob: usize, oy: usize, ox: usize) void {
        @setEvalBranchQuota(100000);
        const g = k.g;
        const kk = g.kh * g.kw;
        const wcs = kk * g.cin * B;
        const w = k.w.ptr + cob * wcs;
        var acc: [R][NB]V = undefined;
        inline for (0..NB) |nb| {
            const bv = vec.load(k.bias.ptr + (cob + nb) * B);
            inline for (0..R) |r| acc[r][nb] = bv;
        }
        const step = g.sw * k.ps;
        if (!checked) {
            const x0 = x + (oy * g.sh - g.pt) * k.rs + (ox * g.sw - g.pl) * k.ps;
            for (k.taps) |t| {
                const xc = x0 + t.x;
                var wv: [NB]V = undefined;
                inline for (0..NB) |nb| wv[nb] = vec.load(w + nb * wcs + t.w);
                inline for (0..R) |r| {
                    const bc: V = @splat(xc[r * step]);
                    inline for (0..NB) |nb| acc[r][nb] = fma(bc, wv[nb], acc[r][nb]);
                }
            }
        } else {
            for (0..g.kh) |ky| {
                const iy = @as(isize, @intCast(oy * g.sh + ky * g.dh)) - @as(isize, @intCast(g.pt));
                if (iy < 0 or iy >= @as(isize, @intCast(g.h))) continue;
                const rp = x + @as(usize, @intCast(iy)) * k.rs;
                for (0..g.kw) |kx| {
                    const ix = @as(isize, @intCast(ox * g.sw + kx * g.dw)) - @as(isize, @intCast(g.pl));
                    if (ix < 0 or ix >= @as(isize, @intCast(g.w))) continue;
                    const xp = rp + @as(usize, @intCast(ix)) * k.ps;
                    const wt = w + (ky * g.kw + kx) * g.cin * B;
                    for (0..g.cin) |ci| {
                        const xc = xp + k.ci_off[ci];
                        var wv: [NB]V = undefined;
                        inline for (0..NB) |nb| wv[nb] = vec.load(wt + nb * wcs + ci * B);
                        inline for (0..R) |r| {
                            const bc: V = @splat(xc[r * step]);
                            inline for (0..NB) |nb| acc[r][nb] = fma(bc, wv[nb], acc[r][nb]);
                        }
                    }
                }
            }
        }
        const rp: ?[*]const f32 = if (res) |rr| rr + cob * rs + ox * B else null;
        k.epi.storeTile(ak, R, NB, &acc, out + cob * os + ox * B, os, rp, rs, cob);
    }
};

// ---------------------------------------------------------------- blocked

/// Single-group convolution of a blocked input with any kernel, stride and
/// dilation (the big 3x3 layers of detection backbones): a GEMM-class
/// kernel cache-blocked over input channels.
///
/// For one input-channel block at a time, the weights of an `NB`-wide group
/// of output blocks (kh*kw*B*B*NB floats) stay in L1 while every register
/// tile of a band of output rows runs over them; partial sums go back to
/// the output and are reloaded for the next input block (2 accesses per
/// kh*kw*B FMAs per accumulator), the epilogue applies after the last.
/// The register tile is `R` pixels x `NB` output blocks, fed per input
/// lane by `NB` weight vectors and `R` broadcasts (from memory on x86,
/// lanes of a loaded vector on NEON).
pub const Blocked = struct {
    g: Geom,
    io: Io,
    /// [coutb][cinb][kh*kw][B ci][B co]
    w: []align(64) f32,
    bias: []align(64) f32,
    epi: Epilogue,
    /// Output columns whose taps all lie inside the input.
    ox_lo: usize,
    ox_hi: usize,
    /// Output rows per band (one unit of work per band and block group).
    band: usize,

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Blocked {
        const kk = g.kh * g.kw;
        const cinb = g.cinb();
        const wp = try arena.alignedAlloc(f32, .@"64", g.coutb() * cinb * kk * B * B);
        @memset(wp, 0);
        for (0..g.cout) |o| for (0..g.cin) |i| for (0..kk) |t| {
            wp[((((o / B) * cinb + i / B) * kk + t) * B + i % B) * B + o % B] = w[(o * g.cin + i) * kk + t];
        };
        const xr = Direct.interior(g.pl, g.w, g.kw, g.sw, g.dw, g.ow);
        return .{
            .g = g,
            .io = io,
            .w = wp,
            .bias = try packBias(arena, bias, g.cout),
            .epi = .init(c, g.cout),
            .ox_lo = xr[0],
            .ox_hi = xr[1],
            .band = @max(1, @min(g.oh, (256 + g.ow - 1) / g.ow)),
        };
    }

    const Ctx = struct { k: *const Blocked, base: [*]f32, band: usize };

    fn groups(k: *const Blocked) usize {
        return (k.g.coutb() + vec.max_nb - 1) / vec.max_nb;
    }

    pub fn run(k: *const Blocked, exec: anytype, base: [*]f32) void {
        // Bands short enough that every thread gets several units.
        var band = k.band;
        while (band > 1 and k.g.n * k.groups() * ((k.g.oh + band - 1) / band) < 4 * exec.threads()) band = (band + 1) / 2;
        const bands = (k.g.oh + band - 1) / band;
        const units = k.g.n * k.groups() * bands;
        const ctx: Ctx = .{ .k = k, .base = base, .band = band };
        switch (vec.actKind(k.epi.act)) {
            inline else => |ak| exec.parallelFor(units, 1, ctx, struct {
                fn f(c: Ctx, s: usize, e: usize) void {
                    const g = c.k.g;
                    const nbands = (g.oh + c.band - 1) / c.band;
                    for (s..e) |u| {
                        const bi = u % nbands;
                        const grp = (u / nbands) % c.k.groups();
                        const img = u / nbands / c.k.groups();
                        const dst = Rows.tensor(c.base, c.k.io.out, img, g.coutb(), g.oh, g.ow);
                        const res: ?Rows = if (c.k.io.res != no_res) Rows.tensor(c.base, c.k.io.res, img, g.coutb(), g.oh, g.ow) else null;
                        const y0 = bi * c.band;
                        c.k.rows(ak, c.base, img, dst, res, grp, y0, @min(g.oh, y0 + c.band));
                    }
                }
            }.f),
        }
    }

    /// One output row of a streamed chain (image 0; reads the input tensor).
    pub fn streamRow(k: *const Blocked, comptime ak: ActKind, base: [*]f32, dst: Rows, res: ?Rows, y: usize) void {
        for (0..k.groups()) |grp| k.rows(ak, base, 0, dst, res, grp, y, y + 1);
    }

    /// Output rows [y0, y1) of output-block group `grp`.
    fn rows(k: *const Blocked, comptime ak: ActKind, base: [*]f32, img: usize, dst: Rows, res: ?Rows, grp: usize, y0: usize, y1: usize) void {
        const coutb = k.g.coutb();
        const cob = grp * vec.max_nb;
        switch (@min(vec.max_nb, coutb - cob)) {
            inline 1...vec.max_nb => |NB| k.rowsNB(NB, ak, base, img, dst, res, cob, y0, y1),
            else => unreachable,
        }
    }

    fn rowsNB(k: *const Blocked, comptime NB: usize, comptime ak: ActKind, base: [*]f32, img: usize, dst: Rows, res_rows: ?Rows, cob: usize, y0: usize, y1: usize) void {
        const g = k.g;
        const R = comptime vec.tileR(NB);
        const cinb = g.cinb();
        const kk = g.kh * g.kw;
        const plane = g.h * g.w * B;
        const x = base + k.io.x + img * cinb * plane;
        for (0..cinb) |cib| {
            const ph: Phase = .{ .first = cib == 0, .last = cib + 1 == cinb };
            const xb = x + cib * plane;
            const wb = k.w.ptr + (cob * cinb + cib) * kk * B * B;
            for (y0..y1) |oy| {
                // Kernel rows inside the input for this output row.
                const top = @as(isize, @intCast(oy * g.sh)) - @as(isize, @intCast(g.pt));
                var ky_lo: usize = 0;
                while (ky_lo < g.kh and top + @as(isize, @intCast(ky_lo * g.dh)) < 0) ky_lo += 1;
                var ky_hi: usize = g.kh;
                while (ky_hi > ky_lo and top + @as(isize, @intCast((ky_hi - 1) * g.dh)) >= @as(isize, @intCast(g.h))) ky_hi -= 1;
                const t: TileArgs = .{
                    .xb = xb,
                    .wb = wb,
                    .out = dst.at(cob, oy),
                    .os = dst.cb_stride,
                    .res = if (res_rows) |r| r.at(cob, oy) else null,
                    .rs = if (res_rows) |r| r.cb_stride else 0,
                    .top = top,
                    .ky_lo = ky_lo,
                    .ky_hi = ky_hi,
                    .cob = cob,
                };
                var ox: usize = 0;
                while (ox < k.ox_lo) : (ox += 1) k.tile(1, NB, ak, true, ph, t, ox);
                while (ox + R <= k.ox_hi) : (ox += R) k.tile(R, NB, ak, false, ph, t, ox);
                inline for (.{ 8, 4, 2, 1 }) |T| {
                    if (T < R and ox + T <= k.ox_hi) {
                        k.tile(T, NB, ak, false, ph, t, ox);
                        ox += T;
                    }
                }
                while (ox < k.ox_hi) : (ox += 1) k.tile(1, NB, ak, false, ph, t, ox);
                while (ox < g.ow) : (ox += 1) k.tile(1, NB, ak, true, ph, t, ox);
            }
        }
    }

    const Phase = struct { first: bool, last: bool };

    const TileArgs = struct {
        /// Input channel block (image plane base) and its weights.
        xb: [*]const f32,
        wb: [*]const f32,
        /// Output row of block `cob`, distance between output blocks.
        out: [*]f32,
        os: usize,
        res: ?[*]const f32,
        rs: usize,
        /// Input row of kernel row 0 (may be negative) and the kernel rows
        /// inside the input.
        top: isize,
        ky_lo: usize,
        ky_hi: usize,
        cob: usize,
    };

    /// `R` output pixels from column `ox`, blocks [cob, cob+NB), one input
    /// channel block. Unchecked tiles have every column tap inside.
    inline fn tile(k: *const Blocked, comptime R: usize, comptime NB: usize, comptime ak: ActKind, comptime checked: bool, ph: Phase, t: TileArgs, ox: usize) void {
        @setEvalBranchQuota(100000);
        const g = k.g;
        const kk = g.kh * g.kw;
        const wcs = g.cinb() * kk * B * B;
        const step = g.sw * B;
        var acc: [R][NB]V = undefined;
        if (ph.first) {
            inline for (0..NB) |nb| {
                const bv = vec.load(k.bias.ptr + (t.cob + nb) * B);
                inline for (0..R) |r| acc[r][nb] = bv;
            }
        } else {
            inline for (0..NB) |nb| {
                inline for (0..R) |r| acc[r][nb] = vec.load(t.out + nb * t.os + (ox + r) * B);
            }
        }
        const left = @as(isize, @intCast(ox * g.sw)) - @as(isize, @intCast(g.pl));
        for (t.ky_lo..t.ky_hi) |ky| {
            const iy: usize = @intCast(t.top + @as(isize, @intCast(ky * g.dh)));
            const row = t.xb + iy * g.w * B;
            for (0..g.kw) |kx| {
                const ix = left + @as(isize, @intCast(kx * g.dw));
                if (checked and (ix < 0 or ix >= @as(isize, @intCast(g.w)))) continue;
                const xp = row + @as(usize, @intCast(ix)) * B;
                const wt = t.wb + (ky * g.kw + kx) * B * B;
                var xv: [R]V = undefined;
                if (vec.lane_broadcast) {
                    inline for (0..R) |r| xv[r] = vec.load(xp + r * step);
                }
                inline for (0..B) |j| {
                    var wv: [NB]V = undefined;
                    inline for (0..NB) |nb| wv[nb] = vec.load(wt + nb * wcs + j * B);
                    inline for (0..R) |r| {
                        const bc: V = if (vec.lane_broadcast) @shuffle(f32, xv[r], undefined, @as(@Vector(B, i32), @splat(j))) else @splat(xp[r * step + j]);
                        inline for (0..NB) |nb| acc[r][nb] = fma(bc, wv[nb], acc[r][nb]);
                    }
                }
            }
        }
        if (ph.last) {
            const rp: ?[*]const f32 = if (t.res) |rr| rr + ox * B else null;
            k.epi.storeTile(ak, R, NB, &acc, t.out + ox * B, t.os, rp, t.rs, t.cob);
        } else {
            inline for (0..NB) |nb| {
                inline for (0..R) |r| vec.store(t.out + nb * t.os + (ox + r) * B, acc[r][nb]);
            }
        }
    }
};

// ------------------------------------------------------------- reference

/// Grouped convolution (neither depthwise nor one group): plain loops.
pub const Reference = struct {
    g: Geom,
    io: Io,
    w: []const f32,
    bias: []align(64) f32,
    c: ir.Conv,

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Reference {
        return .{ .g = g, .io = io, .w = w, .bias = try packBias(arena, bias, g.cout), .c = c };
    }

    const Ctx = struct { k: *const Reference, base: [*]f32 };

    pub fn run(k: *const Reference, exec: anytype, base: [*]f32) void {
        exec.parallelFor(k.g.n * k.g.cout, 1, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                for (s..e) |i| c.k.channel(c.base, i);
            }
        }.f);
    }

    fn xAt(k: *const Reference, x: [*]const f32, ci: usize, iy: usize, ix: usize) f32 {
        const g = k.g;
        if (k.io.x_plain) return x[(ci * g.h + iy) * g.w + ix];
        return x[(((ci / B) * g.h + iy) * g.w + ix) * B + ci % B];
    }

    fn channel(k: *const Reference, base: [*]f32, i: usize) void {
        const g = k.g;
        const img = i / g.cout;
        const co = i % g.cout;
        const cpg = g.cin / g.group;
        const opg = g.cout / g.group;
        const grp = co / opg;
        const x = base + k.io.x + img * (if (k.io.x_plain) g.cin * g.h * g.w else g.cinb() * g.h * g.w * B);
        const ohw = g.oh * g.ow;
        // Output element of channel `co`, pixel `p`, and its lane stride.
        const ob, const ps = if (k.io.out_plain) .{ img * g.cout * ohw + co * ohw, @as(usize, 1) } else .{ img * g.coutb() * ohw * B + (co / B) * ohw * B + co % B, @as(usize, B) };
        const out = base + k.io.out + ob;
        const res: ?[*]const f32 = if (k.io.res != no_res) base + k.io.res + ob else null;
        const pad_lanes = !k.io.out_plain and co % B == 0 and co / B + 1 == g.coutb() and g.cout % B != 0;
        for (0..g.oh) |oy| for (0..g.ow) |ox| {
            var s: f32 = k.bias[co];
            for (0..cpg) |cl| {
                const ci = grp * cpg + cl;
                for (0..g.kh) |ky| {
                    const iy = @as(isize, @intCast(oy * g.sh + ky * g.dh)) - @as(isize, @intCast(g.pt));
                    if (iy < 0 or iy >= @as(isize, @intCast(g.h))) continue;
                    for (0..g.kw) |kx| {
                        const ix = @as(isize, @intCast(ox * g.sw + kx * g.dw)) - @as(isize, @intCast(g.pl));
                        if (ix < 0 or ix >= @as(isize, @intCast(g.w))) continue;
                        s += k.xAt(x, ci, @intCast(iy), @intCast(ix)) * k.w[((co * cpg + cl) * g.kh + ky) * g.kw + kx];
                    }
                }
            }
            const o = (oy * g.ow + ox) * ps;
            if (res) |r| {
                if (!k.c.residual_after_act) s += r[o];
            }
            s = k.c.act.apply(s);
            if (res) |r| {
                if (k.c.residual_after_act) s += r[o];
            }
            out[o] = s;
            // The padding lanes of the last block, written by its first channel.
            if (pad_lanes) {
                for (g.cout % B..B) |l| out[o + l] = 0;
            }
        };
    }
};
