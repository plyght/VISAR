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
    residual_after_act: bool,
    /// Lanes of the last output block that hold real channels.
    tail_mask: @Vector(B, bool),
    coutb: usize,

    fn init(c: ir.Conv, cout: usize) Epilogue {
        const rem = cout % B;
        return .{ .act = c.act, .residual_after_act = c.residual_after_act, .tail_mask = vec.laneMask(if (rem == 0) B else rem), .coutb = (cout + B - 1) / B };
    }

    inline fn apply(e: *const Epilogue, comptime ak: ActKind, acc: V, res: ?[*]const f32, cob: usize) V {
        var v = acc;
        if (res) |r| {
            if (!e.residual_after_act) v += vec.load(r);
        }
        v = vec.act(ak, e.act, v);
        if (res) |r| {
            if (e.residual_after_act) v += vec.load(r);
        }
        if (cob + 1 == e.coutb) v = @select(f32, e.tail_mask, v, vec.splat(0));
        return v;
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
                    var px = p0;
                    while (px + R <= p1) : (px += R) k.tile(R, NB, ak, x, xs, out, os, res, rs, cob, px);
                    // Remainder in halving tiles (a one-pixel tile is bound by
                    // FMA latency, not throughput).
                    inline for (.{ 8, 4, 2, 1 }) |T| {
                        if (T < R and px + T <= p1) {
                            k.tile(T, NB, ak, x, xs, out, os, res, rs, cob, px);
                            px += T;
                        }
                    }
                    while (px < p1) : (px += 1) k.tile(1, NB, ak, x, xs, out, os, res, rs, cob, px);
                },
                else => unreachable,
            }
            cob += nb;
        }
    }

    fn tile(k: *const Pointwise, comptime R: usize, comptime NB: usize, comptime ak: ActKind, x: [*]const f32, xs: usize, out: [*]f32, os: usize, res: ?[*]const f32, rs: usize, cob: usize, px: usize) void {
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
        inline for (0..NB) |nb| {
            const o = out + (cob + nb) * os + px * B;
            inline for (0..R) |r| {
                const rp: ?[*]const f32 = if (res) |rr| rr + (cob + nb) * rs + (px + r) * B else null;
                vec.store(o + r * B, k.epi.apply(ak, acc[r][nb], rp, cob + nb));
            }
        }
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

    pub fn init(arena: Allocator, g: Geom, io: Io, c: ir.Conv, w: []const f32, bias: ?[]const f32) Allocator.Error!Depthwise {
        const kk = g.kh * g.kw;
        const cb = g.coutb();
        const wp = try arena.alignedAlloc(f32, .@"64", cb * kk * B);
        @memset(wp, 0);
        for (0..g.cout) |ch| {
            for (0..kk) |t| wp[((ch / B) * kk + t) * B + ch % B] = w[ch * kk + t];
        }
        const fast3 = g.kh == 3 and g.kw == 3 and g.sh == 1 and g.sw == 1 and g.pt == 1 and g.pl == 1 and g.dh == 1 and g.dw == 1 and g.oh == g.h and g.ow == g.w;
        return .{ .g = g, .io = io, .w = wp, .bias = try packBias(arena, bias, g.cout), .epi = .init(c, g.cout), .fast3 = fast3 };
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
        const g = k.g;
        const out = dst.at(cb, oy);
        const res: ?[*]const f32 = if (res_rows) |r| r.at(cb, oy) else null;
        const kk = g.kh * g.kw;
        const w = k.w.ptr + cb * kk * B;
        const bias = vec.load(k.bias.ptr + cb * B);
        if (k.fast3 and oy > 0 and oy + 1 < g.oh and g.ow >= 3) {
            var wv: [9]V = undefined;
            inline for (0..9) |t| wv[t] = vec.load(w + t * B);
            const rows = [3][*]const f32{ src.at(cb, oy - 1), src.at(cb, oy), src.at(cb, oy + 1) };
            k.pixel(ak, src, out, res, w, bias, cb, oy, 0);
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
            k.pixel(ak, src, out, res, w, bias, cb, oy, g.ow - 1);
            return;
        }
        for (0..g.ow) |ox| k.pixel(ak, src, out, res, w, bias, cb, oy, ox);
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
        inline for (0..NB) |nb| {
            const o = out + (cob + nb) * os + ox * B;
            inline for (0..R) |r| {
                const rp: ?[*]const f32 = if (res) |rr| rr + (cob + nb) * rs + (ox + r) * B else null;
                vec.store(o + r * B, k.epi.apply(ak, acc[r][nb], rp, cob + nb));
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
