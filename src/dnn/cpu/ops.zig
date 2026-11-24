//! Non-convolution CPU kernels: pooling, resize, element-wise, layout
//! reorders, transpose, concat and softmax, on blocked and plain tensors.

const std = @import("std");
const ir = @import("../ir.zig");
const vec = @import("vec.zig");
const core = @import("../../core/core.zig");
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
const Shape = ir.Shape;

const B = vec.B;
const V = vec.V;
const Rows = vec.Rows;

/// Where an operand lives: an element offset in the arena, or constant data.
pub const Ref = struct {
    off: usize = 0,
    cdata: ?[*]const f32 = null,

    pub inline fn ptr(r: Ref, base: [*]f32) [*]f32 {
        return if (r.cdata) |c| @constCast(c) else base + r.off;
    }
};

inline fn ldL(comptime L: usize, p: [*]const f32) @Vector(L, f32) {
    return p[0..L].*;
}

inline fn stL(comptime L: usize, p: [*]f32, v: @Vector(L, f32)) void {
    p[0..L].* = v;
}

inline fn spL(comptime L: usize, x: f32) @Vector(L, f32) {
    return @splat(x);
}

fn cb(c: usize) usize {
    return (c + B - 1) / B;
}

/// Zeroes the padding lanes of a blocked tensor's last channel block.
pub fn fixPadding(base: [*]f32, t: Ref, s: Shape) void {
    const c = s.d[1];
    if (c % B == 0) return;
    const nb = cb(c);
    const hw = s.d[2] * s.d[3];
    const mask = vec.laneMask(c % B);
    const p = t.ptr(base);
    for (0..s.d[0]) |n| {
        const plane = p + ((n * nb) + nb - 1) * hw * B;
        for (0..hw) |i| vec.store(plane + i * B, @select(f32, mask, vec.load(plane + i * B), vec.splat(0)));
    }
}

// ---------------------------------------------------------------- pooling

pub const Pool = struct {
    kind: enum { max, avg },
    /// Accumulator start (-inf for max, 0 for average), kept as data: a
    /// select between two float constants becomes a constant-pool load,
    /// which LLVM cannot lower on wasm without SIMD.
    init: f32,
    plain: bool,
    p: ir.Pool,
    in: Ref,
    out: Ref,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,

    const Ctx = struct { k: *const Pool, base: [*]f32 };

    pub fn run(k: *const Pool, exec: anytype, base: [*]f32) void {
        const planes = k.n * (if (k.plain) k.c else cb(k.c));
        exec.parallelFor(planes * k.oh, 8, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                const kp = c.k;
                for (s..e) |r| {
                    const oy = r % kp.oh;
                    const pc = r / kp.oh;
                    if (kp.plain) {
                        const src: Rows = .{ .base = kp.in.ptr(c.base), .cb_stride = kp.h * kp.w, .row_stride = kp.w };
                        const dst: Rows = .{ .base = kp.out.ptr(c.base), .cb_stride = kp.oh * kp.ow, .row_stride = kp.ow };
                        kp.row(1, src, dst, pc, oy);
                    } else {
                        const src: Rows = .{ .base = kp.in.ptr(c.base), .cb_stride = kp.h * kp.w * B, .row_stride = kp.w * B };
                        const dst: Rows = .{ .base = kp.out.ptr(c.base), .cb_stride = kp.oh * kp.ow * B, .row_stride = kp.ow * B };
                        kp.row(B, src, dst, pc, oy);
                    }
                }
            }
        }.f);
    }

    /// One output row of a streamed chain (blocked, image 0).
    pub fn streamRow(k: *const Pool, src: Rows, dst: Rows, y: usize) void {
        for (0..cb(k.c)) |b| k.row(B, src, dst, b, y);
    }

    /// Output row `oy` of plane `pc` (a channel block, or a channel when
    /// L = 1).
    fn row(k: *const Pool, comptime L: usize, src: Rows, dst: Rows, pc: usize, oy: usize) void {
        const out = dst.at(pc, oy);
        const p = k.p;
        const fast2 = k.kind == .max and p.kernel[0] == 2 and p.kernel[1] == 2 and p.stride[0] == 2 and p.stride[1] == 2 and
            p.pad[0] == 0 and p.pad[1] == 0 and p.dilation[0] == 1 and p.dilation[1] == 1 and 2 * oy + 1 < k.h;
        var ox: usize = 0;
        if (fast2) {
            const r0 = src.at(pc, 2 * oy);
            const r1 = src.at(pc, 2 * oy + 1);
            while (2 * ox + 1 < k.w and ox < k.ow) : (ox += 1) {
                const a = @max(ldL(L, r0 + 2 * ox * L), ldL(L, r0 + (2 * ox + 1) * L));
                const b = @max(ldL(L, r1 + 2 * ox * L), ldL(L, r1 + (2 * ox + 1) * L));
                stL(L, out + ox * L, @max(a, b));
            }
        }
        while (ox < k.ow) : (ox += 1) {
            var acc: @Vector(L, f32) = spL(L, k.init);
            var cnt: usize = 0;
            var cnt_pad: usize = 0;
            for (0..p.kernel[0]) |ky| {
                const iy = @as(isize, @intCast(oy * p.stride[0] + ky * p.dilation[0])) - @as(isize, @intCast(p.pad[0]));
                const in_pad_y = iy < @as(isize, @intCast(k.h + p.pad[2]));
                for (0..p.kernel[1]) |kx| {
                    const ix = @as(isize, @intCast(ox * p.stride[1] + kx * p.dilation[1])) - @as(isize, @intCast(p.pad[1]));
                    if (in_pad_y and ix < @as(isize, @intCast(k.w + p.pad[3]))) cnt_pad += 1;
                    if (iy < 0 or iy >= @as(isize, @intCast(k.h)) or ix < 0 or ix >= @as(isize, @intCast(k.w))) continue;
                    const v = ldL(L, src.at(pc, @intCast(iy)) + @as(usize, @intCast(ix)) * L);
                    acc = if (k.kind == .max) @max(acc, v) else acc + v;
                    cnt += 1;
                }
            }
            if (k.kind == .avg) {
                const d: f32 = @floatFromInt(@max(1, if (p.count_include_pad) cnt_pad else cnt));
                acc /= spL(L, d);
            } else if (cnt == 0) acc = spL(L, 0);
            stL(L, out + ox * L, acc);
        }
    }
};

pub const GlobalAvg = struct {
    plain: bool,
    in: Ref,
    out: Ref,
    planes: usize,
    hw: usize,

    pub fn run(k: *const GlobalAvg, base: [*]f32) void {
        if (k.plain) k.runL(1, base) else k.runL(B, base);
    }

    fn runL(k: *const GlobalAvg, comptime L: usize, base: [*]f32) void {
        const x = k.in.ptr(base);
        const o = k.out.ptr(base);
        const inv = spL(L, 1.0 / @as(f32, @floatFromInt(k.hw)));
        for (0..k.planes) |p| {
            var acc = spL(L, 0);
            for (0..k.hw) |i| acc += ldL(L, x + (p * k.hw + i) * L);
            stL(L, o + p * L, acc * inv);
        }
    }
};

// ----------------------------------------------------------------- resize

pub const Resize = struct {
    linear: bool,
    plain: bool,
    in: Ref,
    out: Ref,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    oh: usize,
    ow: usize,
    /// Nearest: source index. Linear: first source index (second is +1
    /// clamped) and the weight of the second.
    ymap: []u32,
    xmap: []u32,
    yw: []f32,
    xw: []f32,

    fn coord(r: ir.Resize, x: usize, scale: f32, in: usize, out: usize) f32 {
        const xf: f32 = @floatFromInt(x);
        return switch (r.coord) {
            .half_pixel => (xf + 0.5) / scale - 0.5,
            .asymmetric => xf / scale,
            .align_corners => if (out > 1) xf * @as(f32, @floatFromInt(in - 1)) / @as(f32, @floatFromInt(out - 1)) else 0,
            .pytorch_half_pixel => if (out > 1) (xf + 0.5) / scale - 0.5 else 0,
            .tf_half_pixel_for_nn => (xf + 0.5) / scale,
        };
    }

    fn maps(arena: Allocator, r: ir.Resize, scale: f32, in: usize, out: usize, idx: *[]u32, wt: *[]f32) Allocator.Error!void {
        idx.* = try arena.alloc(u32, out);
        wt.* = try arena.alloc(f32, out);
        const last: f32 = @floatFromInt(in - 1);
        for (0..out) |o| {
            const xo = coord(r, o, scale, in, out);
            if (r.mode == .nearest) {
                const fl = @floor(xo);
                const fr = xo - fl;
                const v = switch (r.nearest) {
                    .round_prefer_floor => if (fr == 0.5) fl else @round(xo),
                    .round_prefer_ceil => if (fr == 0.5) fl + 1 else @round(xo),
                    .floor => fl,
                    .ceil => @ceil(xo),
                };
                idx.*[o] = @intFromFloat(std.math.clamp(v, 0, last));
                wt.*[o] = 0;
            } else {
                const c = std.math.clamp(xo, 0, last);
                const fl = @floor(c);
                idx.*[o] = @intFromFloat(fl);
                wt.*[o] = c - fl;
            }
        }
    }

    pub fn init(arena: Allocator, r: ir.Resize, plain: bool, in: Ref, out: Ref, s: Shape, os: Shape) Allocator.Error!Resize {
        var k: Resize = .{ .linear = r.mode == .linear, .plain = plain, .in = in, .out = out, .n = s.d[0], .c = s.d[1], .h = s.d[2], .w = s.d[3], .oh = os.d[2], .ow = os.d[3], .ymap = undefined, .xmap = undefined, .yw = undefined, .xw = undefined };
        try maps(arena, r, r.scale[0], k.h, k.oh, &k.ymap, &k.yw);
        try maps(arena, r, r.scale[1], k.w, k.ow, &k.xmap, &k.xw);
        return k;
    }

    const Ctx = struct { k: *const Resize, base: [*]f32 };

    pub fn run(k: *const Resize, exec: anytype, base: [*]f32) void {
        const planes = k.n * (if (k.plain) k.c else cb(k.c));
        exec.parallelFor(planes * k.oh, 8, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                for (s..e) |r| {
                    if (c.k.plain) c.k.row(1, c.base, r) else c.k.row(B, c.base, r);
                }
            }
        }.f);
    }

    fn row(k: *const Resize, comptime L: usize, base: [*]f32, r: usize) void {
        const oy = r % k.oh;
        const pc = r / k.oh;
        const x = k.in.ptr(base) + pc * k.h * k.w * L;
        const out = k.out.ptr(base) + (pc * k.oh + oy) * k.ow * L;
        if (!k.linear) {
            const src = x + k.ymap[oy] * k.w * L;
            for (0..k.ow) |ox| stL(L, out + ox * L, ldL(L, src + k.xmap[ox] * L));
            return;
        }
        const y0 = k.ymap[oy];
        const y1 = @min(y0 + 1, k.h - 1);
        const fy = spL(L, k.yw[oy]);
        const r0 = x + y0 * k.w * L;
        const r1 = x + y1 * k.w * L;
        for (0..k.ow) |ox| {
            const x0 = k.xmap[ox];
            const x1 = @min(x0 + 1, k.w - 1);
            const fx = spL(L, k.xw[ox]);
            const a = ldL(L, r0 + x0 * L) + (ldL(L, r0 + x1 * L) - ldL(L, r0 + x0 * L)) * fx;
            const b = ldL(L, r1 + x0 * L) + (ldL(L, r1 + x1 * L) - ldL(L, r1 + x0 * L)) * fx;
            stL(L, out + ox * L, a + (b - a) * fy);
        }
    }
};

// ------------------------------------------------------------ element-wise

fn binop(op: ir.Binary, a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return switch (op) {
        .add => a + b,
        .sub => a - b,
        .mul => a * b,
        .div => a / b,
        .max => @max(a, b),
        .min => @min(a, b),
        .pow => if (@TypeOf(a) == f32) std.math.pow(f32, a, b) else blk: {
            var r: @TypeOf(a) = undefined;
            inline for (0..B) |i| r[i] = std.math.pow(f32, a[i], b[i]);
            break :blk r;
        },
    };
}

pub const Act = struct {
    act: ir.Act,
    in: Ref,
    out: Ref,
    /// Stored elements (a multiple of B for blocked tensors).
    count: usize,
    blocked: ?Shape,

    const Ctx = struct { k: *const Act, base: [*]f32 };
    const grain = 4096;

    pub fn run(k: *const Act, exec: anytype, base: [*]f32) void {
        exec.parallelFor((k.count + grain - 1) / grain, 1, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                const x = c.k.in.ptr(c.base);
                const o = c.k.out.ptr(c.base);
                const lo = s * grain;
                const hi = @min(c.k.count, e * grain);
                var i = lo;
                while (i + B <= hi) : (i += B) vec.store(o + i, vec.actGeneric(c.k.act, vec.load(x + i)));
                while (i < hi) : (i += 1) o[i] = c.k.act.apply(x[i]);
            }
        }.f);
        if (k.blocked != null and !k.act.keepsZero()) exec.parallelFor(1, 1, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, _: usize, _: usize) void {
                fixPadding(c.base, c.k.out, c.k.blocked.?);
            }
        }.f);
    }
};

/// y = x * scale[c] + shift[c]
pub const Affine = struct {
    in: Ref,
    out: Ref,
    s: Shape,
    blocked: bool,
    /// Per channel, padded to whole blocks.
    scale: []align(64) f32,
    shift: []align(64) f32,

    pub fn init(arena: Allocator, in: Ref, out: Ref, s: Shape, blocked: bool, scale: []const f32, shift: []const f32) Allocator.Error!Affine {
        const c = s.d[1];
        const sc = try arena.alignedAlloc(f32, .@"64", cb(c) * B);
        const sh = try arena.alignedAlloc(f32, .@"64", cb(c) * B);
        @memset(sc, 0);
        @memset(sh, 0);
        @memcpy(sc[0..c], scale);
        @memcpy(sh[0..c], shift);
        return .{ .in = in, .out = out, .s = s, .blocked = blocked, .scale = sc, .shift = sh };
    }

    pub fn run(k: *const Affine, base: [*]f32) void {
        const x = k.in.ptr(base);
        const o = k.out.ptr(base);
        const c = k.s.d[1];
        var inner: usize = 1;
        for (2..k.s.rank) |i| inner *= k.s.d[i];
        if (k.blocked) {
            const nb = cb(c);
            for (0..k.s.d[0] * nb) |p| {
                const sc = vec.load(k.scale.ptr + (p % nb) * B);
                const sh = vec.load(k.shift.ptr + (p % nb) * B);
                for (0..inner) |i| vec.store(o + (p * inner + i) * B, vec.fma(vec.load(x + (p * inner + i) * B), sc, sh));
            }
            return;
        }
        for (0..k.s.d[0] * c) |p| {
            const sc = k.scale[p % c];
            const sh = k.shift[p % c];
            for (0..inner) |i| o[p * inner + i] = x[p * inner + i] * sc + sh;
        }
    }
};

pub const Binary = struct {
    op: ir.Binary,
    a: Ref,
    b: Ref,
    out: Ref,
    os: Shape,
    mode: union(enum) {
        /// Same shape, any layout: element by element over `count`.
        same: usize,
        /// Blocked tensor op per-channel vector (`a_is_const` swaps sides).
        per_channel: struct { vals: []align(64) f32, const_first: bool },
        /// Plain numpy broadcasting.
        broadcast: struct { as: [ir.max_rank]usize, bs: [ir.max_rank]usize },
    },
    blocked: bool,

    pub fn init(arena: Allocator, op: ir.Binary, a: Ref, b: Ref, out: Ref, as: Shape, bs: Shape, os: Shape, blocked: bool, a_const: ?[]const f32, b_const: ?[]const f32) Allocator.Error!Binary {
        var k: Binary = .{ .op = op, .a = a, .b = b, .out = out, .os = os, .mode = undefined, .blocked = blocked };
        if (blocked and (a_const != null or b_const != null)) {
            const cdata = a_const orelse b_const.?;
            const c = os.d[1];
            const vals = try arena.alignedAlloc(f32, .@"64", cb(c) * B);
            @memset(vals, 0);
            for (0..c) |i| vals[i] = if (cdata.len == 1) cdata[0] else cdata[i];
            k.mode = .{ .per_channel = .{ .vals = vals, .const_first = a_const != null } };
            return k;
        }
        if (as.eql(os) and bs.eql(os)) {
            k.mode = .{ .same = if (blocked) os.d[0] * cb(os.d[1]) * B * os.d[2] * os.d[3] else os.count() };
            return k;
        }
        var st: [2][ir.max_rank]usize = undefined;
        for ([2]Shape{ as, bs }, 0..) |s, j| {
            var stride: usize = 1;
            var i: usize = os.rank;
            while (i > 0) {
                i -= 1;
                const k_in = i + s.rank;
                if (k_in < os.rank) {
                    st[j][i] = 0;
                    continue;
                }
                const d = s.d[k_in - os.rank];
                st[j][i] = if (d == 1) 0 else stride;
                stride *= d;
            }
        }
        k.mode = .{ .broadcast = .{ .as = st[0], .bs = st[1] } };
        return k;
    }

    const Ctx = struct { k: *const Binary, base: [*]f32 };
    const grain = 8192;

    pub fn run(k: *const Binary, exec: anytype, base: [*]f32) void {
        if (k.mode == .same) {
            const n = k.mode.same;
            exec.parallelFor((n + grain - 1) / grain, 1, Ctx{ .k = k, .base = base }, struct {
                fn f(c: Ctx, s: usize, e: usize) void {
                    const m = c.k.mode.same;
                    const a = c.k.a.ptr(c.base);
                    const b = c.k.b.ptr(c.base);
                    const o = c.k.out.ptr(c.base);
                    var i = s * grain;
                    const hi = @min(m, e * grain);
                    switch (c.k.op) {
                        inline else => |op| {
                            while (i + B <= hi) : (i += B) vec.store(o + i, binop(op, vec.load(a + i), vec.load(b + i)));
                            while (i < hi) : (i += 1) o[i] = binop(op, a[i], b[i]);
                        },
                    }
                }
            }.f);
            if (k.blocked) exec.parallelFor(1, 1, Ctx{ .k = k, .base = base }, struct {
                fn f(c: Ctx, _: usize, _: usize) void {
                    fixPadding(c.base, c.k.out, c.k.os);
                }
            }.f);
            return;
        }
        exec.parallelFor(1, 1, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, _: usize, _: usize) void {
                c.k.runSerial(c.base);
            }
        }.f);
    }

    fn runSerial(k: *const Binary, base: [*]f32) void {
        const a = k.a.ptr(base);
        const b = k.b.ptr(base);
        const o = k.out.ptr(base);
        switch (k.mode) {
            .same => unreachable,
            .per_channel => |pc| {
                const x = if (pc.const_first) b else a;
                const nb = cb(k.os.d[1]);
                const hw = k.os.d[2] * k.os.d[3];
                for (0..k.os.d[0] * nb) |p| {
                    const cv = vec.load(pc.vals.ptr + (p % nb) * B);
                    for (0..hw) |i| {
                        const xv = vec.load(x + (p * hw + i) * B);
                        vec.store(o + (p * hw + i) * B, if (pc.const_first) binop(k.op, cv, xv) else binop(k.op, xv, cv));
                    }
                }
            },
            .broadcast => |bc| {
                const r = k.os.rank;
                var idx: [ir.max_rank]usize = @splat(0);
                const n = k.os.count();
                var ia: usize = 0;
                var ib: usize = 0;
                for (0..n) |i| {
                    o[i] = binop(k.op, a[ia], b[ib]);
                    var d = r;
                    while (d > 0) {
                        d -= 1;
                        idx[d] += 1;
                        ia += bc.as[d];
                        ib += bc.bs[d];
                        if (idx[d] < k.os.d[d]) break;
                        ia -= bc.as[d] * idx[d];
                        ib -= bc.bs[d] * idx[d];
                        idx[d] = 0;
                    }
                }
            },
        }
        if (k.blocked) fixPadding(base, k.out, k.os);
    }
};

// ---------------------------------------------------------------- layouts

pub const Reorder = struct {
    kind: enum { to_blocked, to_plain, copy },
    in: Ref,
    out: Ref,
    s: Shape,

    const Ctx = struct { k: *const Reorder, base: [*]f32 };

    pub fn run(k: *const Reorder, exec: anytype, base: [*]f32) void {
        switch (k.kind) {
            .copy => exec.parallelFor(1, 1, Ctx{ .k = k, .base = base }, struct {
                fn f(c: Ctx, _: usize, _: usize) void {
                    const n = c.k.s.count();
                    @memcpy(c.k.out.ptr(c.base)[0..n], c.k.in.ptr(c.base)[0..n]);
                }
            }.f),
            else => exec.parallelFor(k.s.d[0] * cb(k.s.d[1]) * k.s.d[2], 8, Ctx{ .k = k, .base = base }, struct {
                fn f(c: Ctx, s: usize, e: usize) void {
                    for (s..e) |r| c.k.row(c.base, r);
                }
            }.f),
        }
    }

    fn row(k: *const Reorder, base: [*]f32, r: usize) void {
        const c = k.s.d[1];
        const h = k.s.d[2];
        const w = k.s.d[3];
        const nb = cb(c);
        const y = r % h;
        const blk = (r / h) % nb;
        const n = r / h / nb;
        const bl = k.in.ptr(base);
        const o = k.out.ptr(base);
        const lanes = @min(B, c - blk * B);
        if (k.kind == .to_blocked) {
            const dst = o + (((n * nb + blk) * h + y) * w) * B;
            for (0..w) |x| {
                var v: [B]f32 = @splat(0);
                for (0..lanes) |l| v[l] = bl[((n * c + blk * B + l) * h + y) * w + x];
                dst[x * B ..][0..B].* = v;
            }
        } else {
            const src = bl + (((n * nb + blk) * h + y) * w) * B;
            for (0..lanes) |l| {
                const dst = o + ((n * c + blk * B + l) * h + y) * w;
                for (0..w) |x| dst[x] = src[x * B + l];
            }
        }
    }
};

pub const Transpose = struct {
    in: Ref,
    out: Ref,
    /// Input shape and permutation.
    s: Shape,
    perm: [ir.max_rank]u8,

    pub fn run(k: *const Transpose, base: [*]f32) void {
        const x = k.in.ptr(base);
        const o = k.out.ptr(base);
        const r = k.s.rank;
        var in_stride: [ir.max_rank]usize = undefined;
        var st: usize = 1;
        var i: usize = r;
        while (i > 0) {
            i -= 1;
            in_stride[i] = st;
            st *= k.s.d[i];
        }
        var os: [ir.max_rank]usize = undefined; // input stride per output axis
        var od: [ir.max_rank]usize = undefined;
        for (0..r) |a| {
            os[a] = in_stride[k.perm[a]];
            od[a] = k.s.d[k.perm[a]];
        }
        var idx: [ir.max_rank]usize = @splat(0);
        var src: usize = 0;
        const n = k.s.count();
        for (0..n) |j| {
            o[j] = x[src];
            var d = r;
            while (d > 0) {
                d -= 1;
                idx[d] += 1;
                src += os[d];
                if (idx[d] < od[d]) break;
                src -= os[d] * idx[d];
                idx[d] = 0;
            }
        }
    }
};

/// Channels [c0, c1) of a blocked tensor, activated, to plain NHWC or NCHW.
pub const Extract = struct {
    in: Ref,
    out: Ref,
    /// Input shape (NCHW).
    s: Shape,
    c0: usize,
    c1: usize,
    act: ir.Act,
    nhwc: bool,

    const Ctx = struct { k: *const Extract, base: [*]f32 };

    pub fn run(k: *const Extract, exec: anytype, base: [*]f32) void {
        exec.parallelFor(k.s.d[0] * k.s.d[2], 16, Ctx{ .k = k, .base = base }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                for (s..e) |r| c.k.row(c.base, r);
            }
        }.f);
    }

    fn row(k: *const Extract, base: [*]f32, r: usize) void {
        const h = k.s.d[2];
        const w = k.s.d[3];
        const n = r / h;
        const y = r % h;
        const hw = h * w;
        const nb = cb(k.s.d[1]);
        const cn = k.c1 - k.c0;
        const x = k.in.ptr(base);
        const o = k.out.ptr(base);
        var b = k.c0 / B;
        while (b * B < k.c1) : (b += 1) {
            const lo = @max(k.c0, b * B);
            const hi = @min(k.c1, b * B + B);
            const src = x + ((n * nb + b) * hw + y * w) * B;
            for (0..w) |i| {
                const v: [B]f32 = vec.actGeneric(k.act, vec.load(src + i * B));
                if (k.nhwc) {
                    const dst = o + (n * hw + y * w + i) * cn + (lo - k.c0);
                    @memcpy(dst[0 .. hi - lo], v[lo - b * B .. hi - b * B]);
                } else {
                    for (lo..hi) |c| o[(n * cn + c - k.c0) * hw + y * w + i] = v[c - b * B];
                }
            }
        }
    }
};

pub const Concat = struct {
    ins: []Ref,
    in_shapes: []Shape,
    out: Ref,
    os: Shape,
    axis: usize,
    blocked: bool,

    pub fn run(k: *const Concat, base: [*]f32) void {
        const o = k.out.ptr(base);
        if (k.blocked) {
            // Whole blocks per input (all but the last are multiples of B).
            const hw = k.os.d[2] * k.os.d[3];
            const onb = cb(k.os.d[1]);
            for (0..k.os.d[0]) |n| {
                var at: usize = 0;
                for (k.ins, k.in_shapes) |r, s| {
                    const nb = cb(s.d[1]);
                    const len = nb * hw * B;
                    @memcpy(o[(n * onb + at) * hw * B ..][0..len], r.ptr(base)[n * len ..][0..len]);
                    at += nb;
                }
            }
            return;
        }
        var outer: usize = 1;
        for (0..k.axis) |i| outer *= k.os.d[i];
        var inner: usize = 1;
        for (k.axis + 1..k.os.rank) |i| inner *= k.os.d[i];
        const orow = k.os.d[k.axis] * inner;
        var at: usize = 0;
        for (k.ins, k.in_shapes) |r, s| {
            const len = s.d[k.axis] * inner;
            const x = r.ptr(base);
            for (0..outer) |i| @memcpy(o[i * orow + at ..][0..len], x[i * len ..][0..len]);
            at += len;
        }
    }
};

pub const Softmax = struct {
    in: Ref,
    out: Ref,
    s: Shape,
    axis: usize,

    pub fn run(k: *const Softmax, base: [*]f32) void {
        const x = k.in.ptr(base);
        const o = k.out.ptr(base);
        var outer: usize = 1;
        for (0..k.axis) |i| outer *= k.s.d[i];
        var inner: usize = 1;
        for (k.axis + 1..k.s.rank) |i| inner *= k.s.d[i];
        const d = k.s.d[k.axis];
        for (0..outer) |a| for (0..inner) |b| {
            const at = a * d * inner + b;
            var m = -std.math.inf(f32);
            for (0..d) |i| m = @max(m, x[at + i * inner]);
            var sum: f32 = 0;
            for (0..d) |i| {
                const e = @exp(x[at + i * inner] - m);
                o[at + i * inner] = e;
                sum += e;
            }
            for (0..d) |i| o[at + i * inner] /= sum;
        };
    }
};
