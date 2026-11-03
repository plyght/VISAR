//! Morphology: erode, dilate and the compound operations of morphologyEx.
//! Pixels outside the image are ignored (OpenCV's default border value), so
//! a rectangular element is separable into a row pass and a column pass.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
const simd = @import("../core/simd.zig");

pub const Error = core.Error || Allocator.Error || error{InvalidKernel};

pub const Shape = enum(u8) { rect = 0, cross = 1, ellipse = 2 };

/// A structuring element: `w` x `h` mask (row-major, nonzero = on) and an
/// anchor (defaults to the centre).
pub const Element = struct {
    w: u32,
    h: u32,
    mask: []const u8,
    ax: u32,
    ay: u32,

    pub fn isRect(e: Element) bool {
        for (e.mask) |m| if (m == 0) return false;
        return true;
    }
};

/// OpenCV's getStructuringElement. `buf` must hold w*h bytes.
pub fn structuringElement(shape: Shape, w: u32, h: u32, buf: []u8) Element {
    const ax = w / 2;
    const ay = h / 2;
    @memset(buf[0 .. w * h], 0);
    const r = ay;
    const c = ax;
    const inv_r2: f64 = if (r > 0) 1.0 / @as(f64, @floatFromInt(r * r)) else 0;
    for (0..h) |i| {
        var j1: usize = 0;
        var j2: usize = 0;
        if (shape == .rect or (shape == .cross and i == ay)) {
            j2 = w;
        } else if (shape == .cross) {
            j1 = ax;
            j2 = j1 + 1;
        } else {
            const dy: i64 = @as(i64, @intCast(i)) - @as(i64, r);
            if (@abs(dy) <= r) {
                const dx_f = @as(f64, @floatFromInt(c)) * @sqrt(@as(f64, @floatFromInt((r * r) - @as(u64, @intCast(dy * dy)))) * inv_r2);
                const dx: i64 = @intFromFloat(@round(dx_f));
                j1 = @intCast(@max(@as(i64, c) - dx, 0));
                j2 = @intCast(@min(@as(i64, c) + dx + 1, @as(i64, w)));
            }
        }
        for (j1..j2) |j| buf[i * w + j] = 1;
    }
    return .{ .w = w, .h = h, .mask = buf[0 .. w * h], .ax = ax, .ay = ay };
}

const Op = enum { min, max };

pub fn erode(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    return morph(.min, exec, gpa, src, dst, el, iterations);
}

pub fn dilate(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    return morph(.max, exec, gpa, src, dst, el, iterations);
}

fn morph(comptime op: Op, exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (el.w == 0 or el.h == 0) return error.InvalidKernel;
    if (iterations == 0) return dst.copyFrom(src);
    const plan = try Plan.init(gpa, el);
    defer plan.deinit(gpa);
    var cur = src;
    // Passes never run in place: a band reads source rows other bands (and
    // its own later rows) still need.
    var copy: ?Image(T) = null;
    defer if (copy) |*t| t.deinit(gpa);
    if (overlaps(src, dst)) {
        copy = try src.clone(gpa);
        cur = copy.?;
    }
    var tmp: ?Image(T) = null;
    defer if (tmp) |*t| t.deinit(gpa);
    if (iterations > 1) tmp = try Image(T).initLike(gpa, src);
    for (0..iterations) |it| {
        // Alternate between dst and the temporary so the last pass lands in dst.
        const out = if ((iterations - 1 - it) % 2 == 0) dst else tmp.?;
        try runPass(op, exec, gpa, cur, out, el, plan);
        cur = out;
    }
}

fn overlaps(a: anytype, b: @TypeOf(a)) bool {
    const T = @TypeOf(a).Elem;
    const a0 = @intFromPtr(a.data);
    const b0 = @intFromPtr(b.data);
    const a1 = a0 + ((@as(usize, a.height) - 1) * a.stride + a.rowLen()) * @sizeOf(T);
    const b1 = b0 + ((@as(usize, b.height) - 1) * b.stride + b.rowLen()) * @sizeOf(T);
    return a0 < b1 and b0 < a1;
}

/// The element as horizontal runs of "on" cells. Each output row is the
/// min/max over runs of the run-length horizontal min/max of one source row,
/// so every distinct run length is computed once per source row (by doubling:
/// spans 1, 2, 4, ... then one combine for a non-power-of-two length).
const Plan = struct {
    const Run = struct { ky: u32, x0: u32, li: u32 };
    runs: []Run,
    /// Distinct run lengths.
    lens: []u32,
    /// Doubling levels needed (span 2^k for k < levels).
    levels: u32,

    fn init(gpa: Allocator, el: Element) Allocator.Error!Plan {
        var runs: std.ArrayList(Run) = .empty;
        errdefer runs.deinit(gpa);
        var lens: std.ArrayList(u32) = .empty;
        errdefer lens.deinit(gpa);
        var max_len: u32 = 1;
        for (0..el.h) |ky| {
            const m = el.mask[ky * el.w ..][0..el.w];
            var x: u32 = 0;
            while (x < el.w) {
                if (m[x] == 0) {
                    x += 1;
                    continue;
                }
                const x0 = x;
                while (x < el.w and m[x] != 0) x += 1;
                const len = x - x0;
                max_len = @max(max_len, len);
                const li = std.mem.indexOfScalar(u32, lens.items, len) orelse blk: {
                    try lens.append(gpa, len);
                    break :blk lens.items.len - 1;
                };
                try runs.append(gpa, .{ .ky = @intCast(ky), .x0 = x0, .li = @intCast(li) });
            }
        }
        const levels: u32 = std.math.log2_int(u32, max_len) + 1;
        const r = try runs.toOwnedSlice(gpa);
        errdefer gpa.free(r);
        return .{ .runs = r, .lens = try lens.toOwnedSlice(gpa), .levels = levels };
    }

    fn deinit(p: Plan, gpa: Allocator) void {
        gpa.free(p.runs);
        gpa.free(p.lens);
    }
};

inline fn vop(comptime op: Op, a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (op == .min) @min(a, b) else @max(a, b);
}

/// d[i] = op(a[i], b[i]) for i < d.len.
fn combine2(comptime op: Op, comptime T: type, d: []T, a: [*]const T, b: [*]const T) void {
    const N = lanesM(T);
    const V = @Vector(N, T);
    const n = d.len;
    if (n < N) {
        for (d, 0..) |*o, i| o.* = vop(op, a[i], b[i]);
        return;
    }
    var i: usize = 0;
    while (true) : (i += N) {
        if (i + N > n) i = n - N; // overlapping tail: recomputes a few lanes
        const x: V = a[i..][0..N].*;
        const y: V = b[i..][0..N].*;
        d[i..][0..N].* = vop(op, x, y);
        if (i + N == n) break;
    }
}

/// d[i] = op over k of p[k][i]; p.len >= 1.
fn combineN(comptime op: Op, comptime T: type, d: []T, p: []const [*]const T) void {
    switch (p.len) {
        inline 1...9 => |k| return combineK(op, T, k, d, p[0..k].*),
        else => {},
    }
    const N = lanesM(T);
    const V = @Vector(N, T);
    const n = d.len;
    if (n < N) {
        for (d, 0..) |*o, i| {
            var acc = p[0][i];
            for (p[1..]) |q| acc = vop(op, acc, q[i]);
            o.* = acc;
        }
        return;
    }
    var i: usize = 0;
    while (i + 2 * N <= n) : (i += 2 * N) {
        var a0: V = p[0][i..][0..N].*;
        var a1: V = p[0][i + N ..][0..N].*;
        for (p[1..]) |q| {
            a0 = vop(op, a0, @as(V, q[i..][0..N].*));
            a1 = vop(op, a1, @as(V, q[i + N ..][0..N].*));
        }
        d[i..][0..N].* = a0;
        d[i + N ..][0..N].* = a1;
    }
    while (i < n) : (i += N) {
        if (i + N > n) i = n - N;
        var a0: V = p[0][i..][0..N].*;
        for (p[1..]) |q| a0 = vop(op, a0, @as(V, q[i..][0..N].*));
        d[i..][0..N].* = a0;
    }
}

fn combineK(comptime op: Op, comptime T: type, comptime K: usize, d: []T, p: [K][*]const T) void {
    const N = lanesM(T);
    const V = @Vector(N, T);
    const n = d.len;
    if (n < N) {
        for (d, 0..) |*o, i| {
            var acc = p[0][i];
            inline for (1..K) |k| acc = vop(op, acc, p[k][i]);
            o.* = acc;
        }
        return;
    }
    const Body = struct {
        inline fn f(comptime aligned: bool, q: [K][*]const T, o: [*]T, i: usize) void {
            var acc: V = q[0][i..][0..N].*;
            inline for (1..K) |k| acc = vop(op, acc, @as(V, q[k][i..][0..N].*));
            storeV(aligned, T, N, o + i, acc);
        }
    };
    // Unaligned first and last vectors, aligned stores between.
    const mis = (@intFromPtr(d.ptr) % (N * @sizeOf(T))) / @sizeOf(T);
    Body.f(false, p, d.ptr, 0);
    var i: usize = (N - mis) % N;
    if (@intFromPtr(d.ptr) % @sizeOf(T) == 0) {
        while (i + N <= n) : (i += N) Body.f(true, p, d.ptr, i);
    } else {
        while (i + N <= n) : (i += N) Body.f(false, p, d.ptr, i);
    }
    Body.f(false, p, d.ptr, n - N);
}

inline fn storeV(comptime aligned: bool, comptime T: type, comptime N: usize, dst: [*]T, v: @Vector(N, T)) void {
    if (aligned) {
        const pa: *align(N * @sizeOf(T)) [N]T = @ptrCast(@alignCast(dst));
        pa.* = v;
    } else {
        dst[0..N].* = v;
    }
}

fn runPass(comptime op: Op, exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, plan: Plan) Error!void {
    const T = @TypeOf(src).Elem;
    const ident: T = if (op == .min) maxOf(T) else minOf(T);
    if (plan.runs.len == 0) {
        for (0..dst.height) |y| @memset(dst.row(y), ident);
        return;
    }
    var failed: std.atomic.Value(bool) = .init(false);
    const Ctx = struct { s: Image(T), d: Image(T), el: Element, plan: Plan, gpa: Allocator, failed: *std.atomic.Value(bool) };
    exec.parallelFor(src.height, @max(16, 2 * @as(usize, el.h)), Ctx{ .s = src, .d = dst, .el = el, .plan = plan, .gpa = gpa, .failed = &failed }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            band(op, T, c.s, c.d, c.el, c.plan, c.gpa, y0, y1) catch c.failed.store(true, .monotonic);
        }
    }.run);
    if (failed.load(.monotonic)) return error.OutOfMemory;
}

/// Run lengths up to this are min/max-ed tap by tap straight from the source
/// row; longer ones go through a padded copy and the doubling levels.
const direct_max = 8;

/// For every length l in 2..direct_max with outs[l] non-null, and padded
/// pixels p in [ps, pe - l + 1): outs[l][p*cn + ch] = op over source pixels
/// p - ax .. p - ax + l - 1 (channel ch), ignoring pixels outside the row.
/// `outs` are virtual bases (only the range above is touched). All lengths
/// share one pass over the source: the running min/max is stored as it
/// reaches each length.
fn runsDirect(comptime op: Op, comptime T: type, outs: *const [direct_max + 1]?[*]T, ps: usize, pe: usize, s: []const T, cn: usize, ax: usize) void {
    const W = s.len / cn;
    var max_len: usize = 0;
    for (outs, 0..) |o, l| if (o != null) {
        max_len = l;
    };
    if (max_len == 0) return;
    // Interior of the longest run: the whole run lies inside the row.
    const lo = @max(ps, ax) * cn;
    const hi = @max(lo, @min(pe - max_len + 1, if (W + ax + 1 >= max_len) ax + W + 1 - max_len else 0) * cn);
    const N = lanesM(T);
    const V = @Vector(N, T);
    var done = lo; // [lo, done) is handled by the vector loop
    if (hi - lo >= N) {
        const base = s.ptr - ax * cn; // base[p*cn + j*cn + ch] = s[(p - ax + j)*cn + ch]
        switch (max_len) {
            inline 2...direct_max => |ml| {
                const Body = struct {
                    inline fn f(comptime aligned: bool, o: *const [direct_max + 1]?[*]T, b: [*]const T, c: usize, i: usize) void {
                        var acc: V = b[i..][0..N].*;
                        inline for (1..ml) |j| {
                            acc = vop(op, acc, @as(V, b[i + j * c ..][0..N].*));
                            if (o[j + 1]) |q| storeV(aligned, T, N, q + i, acc);
                        }
                    }
                };
                // Unaligned first and last vectors, aligned stores between.
                const ref = outs[max_len].?;
                const mis = (@intFromPtr(ref + lo) % (N * @sizeOf(T))) / @sizeOf(T);
                Body.f(false, outs, base, cn, lo);
                var i = lo + (N - mis) % N;
                while (i + N <= hi) : (i += N) Body.f(true, outs, base, cn, i);
                Body.f(false, outs, base, cn, hi - N);
            },
            else => unreachable,
        }
        done = hi;
    }
    for (outs, 0..) |o, l| if (o) |out| {
        const end = (pe - l + 1) * cn;
        const a = @min(lo, end);
        edgeRun(op, T, out[ps * cn .. a], s, cn, ax, l, ps * cn);
        if (done < end) edgeRun(op, T, out[@max(done, a)..end], s, cn, ax, l, @max(done, a));
    };
}

/// Scalar, clipped version of runsDirect for padded elements
/// [start, start + o.len).
fn edgeRun(comptime op: Op, comptime T: type, o: []T, s: []const T, cn: usize, ax: usize, len: usize, start: usize) void {
    const ident: T = if (op == .min) maxOf(T) else minOf(T);
    const W: i64 = @intCast(s.len / cn);
    for (o, start..) |*v, i| {
        const px: i64 = @as(i64, @intCast(i / cn)) - @as(i64, @intCast(ax));
        const ch = i % cn;
        const a = @max(px, 0);
        const e = @min(px + @as(i64, @intCast(len)), W);
        var acc = ident;
        var c = a;
        while (c < e) : (c += 1) acc = vop(op, acc, s[@as(usize, @intCast(c)) * cn + ch]);
        v.* = acc;
    }
}

/// Bytes of preprocessed rows a band keeps live; wider rows are processed
/// in column strips so the ring stays in L2.
const ring_budget = 192 * 1024;

/// Output rows [y0, y1) of one pass, with a ring of `el.h` preprocessed
/// source rows.
fn band(comptime op: Op, comptime T: type, s: Image(T), d: Image(T), el: Element, plan: Plan, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const ident: T = if (op == .min) maxOf(T) else minOf(T);
    const cn: usize = s.channels;
    const W: usize = s.width;
    const ax: usize = el.ax;
    const nl = plan.lens.len;
    // A length-1 run at the anchor column reads the source row itself. A
    // padded copy is kept only for other length-1 runs and long runs.
    var need_pad = false;
    for (plan.runs) |r| {
        const len = plan.lens[r.li];
        if ((len == 1 and r.x0 != el.ax) or len > direct_max) need_pad = true;
    }
    // arr[li]: which array of a ring slot holds run length plan.lens[li].
    const arr = try gpa.alloc(u32, nl);
    defer gpa.free(arr);
    var per_slot: usize = @intFromBool(need_pad);
    for (plan.lens, arr) |len, *a| {
        if (len == 1) {
            a.* = 0;
        } else {
            a.* = @intCast(per_slot);
            per_slot += 1;
        }
    }
    const slots: usize = el.h;
    // Strip width in pixels (the padded path works on whole rows).
    const strip: usize = if (need_pad or per_slot == 0) W else @min(W, @max(128, ring_budget / (slots * per_slot * cn * @sizeOf(T))));
    const wp = strip + el.w - 1; // padded strip width in pixels
    const plen = std.mem.alignForward(usize, wp * cn, lanesM(T)); // 64-byte aligned arrays
    const levels: usize = if (need_pad) plan.levels else 1;
    const buf = try gpa.alignedAlloc(T, .@"64", (slots * per_slot + levels) * plen);
    defer gpa.free(buf);
    const ptrs = try gpa.alloc([*]const T, plan.runs.len);
    defer gpa.free(ptrs);
    if (need_pad) for (0..slots) |k| {
        const pad = buf[k * per_slot * plen ..][0..plen];
        @memset(pad[0 .. ax * cn], ident);
        @memset(pad[(ax + W) * cn ..], ident);
    };

    // With strips, rows go in blocks so a block's source and output rows
    // stay cached across its strips.
    const block: usize = if (strip == W) y1 - y0 else @max(32, 8 * @as(usize, el.h));
    var yb0 = y0;
    while (yb0 < y1) : (yb0 += block) {
        const yb1 = @min(y1, yb0 + block);
        try blockStrips(op, T, s, d, el, plan, arr, need_pad, per_slot, strip, levels, buf, ptrs, yb0, yb1);
    }
}

fn blockStrips(
    comptime op: Op,
    comptime T: type,
    s: Image(T),
    d: Image(T),
    el: Element,
    plan: Plan,
    arr: []const u32,
    need_pad: bool,
    per_slot: usize,
    strip: usize,
    levels: usize,
    buf: []T,
    ptrs: [][*]const T,
    y0: usize,
    y1: usize,
) Allocator.Error!void {
    const ident: T = if (op == .min) maxOf(T) else minOf(T);
    const cn: usize = s.channels;
    const W: usize = s.width;
    const H: i64 = s.height;
    const n = W * cn;
    const ax: usize = el.ax;
    const ay: i64 = el.ay;
    const slots: usize = el.h;
    const wp = strip + el.w - 1; // padded strip width in pixels
    const plen = std.mem.alignForward(usize, wp * cn, lanesM(T)); // 64-byte aligned arrays
    const scratch = buf[slots * per_slot * plen ..];
    var xs0: usize = 0;
    while (xs0 < W) : (xs0 += strip) {
        const xs1 = @min(W, xs0 + strip);
        const off = xs0 * cn; // slot arrays start at padded pixel xs0
        var next: i64 = @max(@as(i64, @intCast(y0)) - ay, 0);
        for (y0..y1) |y| {
            const yi: i64 = @intCast(y);
            const last = @min(yi + @as(i64, el.h) - 1 - ay, H - 1);
            while (next <= last) : (next += 1) {
                const sy: usize = @intCast(next);
                const srow = s.row(sy);
                const slot = buf[(sy % slots) * per_slot * plen ..][0 .. per_slot * plen];
                if (need_pad) {
                    const pad = slot[0..plen];
                    @memcpy(pad[ax * cn ..][0..n], srow);
                    // Doubling: level k holds the span-2^k min/max (level 0 = pad).
                    var k: usize = 1;
                    while (k < levels) : (k += 1) {
                        const half = @as(usize, 1) << @intCast(k - 1);
                        const span = half * 2;
                        if (span > wp) break;
                        const prev: []const T = if (k == 1) pad else scratch[(k - 2) * plen ..][0..plen];
                        const out = scratch[(k - 1) * plen ..][0 .. (wp - span + 1) * cn];
                        combine2(op, T, out, prev.ptr, prev.ptr + half * cn);
                    }
                }
                var outs: [direct_max + 1]?[*]T = @splat(null);
                for (plan.lens, arr) |len, ai| {
                    if (len == 1) continue;
                    const out = slot[ai * plen ..][0 .. (wp - len + 1) * cn];
                    if (len <= direct_max) {
                        outs[len] = out.ptr - off;
                        continue;
                    }
                    const lv = std.math.log2_int(u32, len);
                    const p = @as(usize, 1) << @intCast(lv);
                    const src_lv: []const T = scratch[(lv - 1) * plen ..][0..plen];
                    if (p == len) @memcpy(out, src_lv[0..out.len]) else combine2(op, T, out, src_lv.ptr, src_lv.ptr + (len - p) * cn);
                }
                runsDirect(op, T, &outs, xs0, xs1 + el.w - 1, srow, cn, ax);
            }
            var kk: usize = 0;
            for (plan.runs) |r| {
                const sy = yi + @as(i64, r.ky) - ay;
                if (sy < 0 or sy >= H) continue;
                const syu: usize = @intCast(sy);
                const len = plan.lens[r.li];
                ptrs[kk] = if (len == 1 and r.x0 == el.ax)
                    s.row(syu).ptr + off
                else
                    buf.ptr + ((syu % slots) * per_slot + arr[r.li]) * plen + @as(usize, r.x0) * cn;
                kk += 1;
            }
            const drow = d.row(y)[off .. xs1 * cn];
            if (kk == 0) @memset(drow, ident) else combineN(op, T, drow, ptrs[0..kk]);
        }
    }
}

fn generalPass(comptime op: Op, exec: Exec, src: anytype, dst: @TypeOf(src), el: Element) Error!void {
    const T = @TypeOf(src).Elem;
    const Ctx = struct { s: Image(T), d: Image(T), el: Element };
    exec.parallelFor(src.height, 8, Ctx{ .s = src, .d = dst, .el = el }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const W: i64 = c.s.width;
            const H: i64 = c.s.height;
            const init: T = if (op == .min) maxOf(T) else minOf(T);
            for (y0..y1) |y| {
                const d = c.d.row(y);
                @memset(d, init);
                for (0..c.el.h) |ky| {
                    const sy = @as(i64, @intCast(y)) + @as(i64, @intCast(ky)) - c.el.ay;
                    if (sy < 0 or sy >= H) continue;
                    const sr = c.s.row(@intCast(sy));
                    for (0..c.el.w) |kx| {
                        if (c.el.mask[ky * c.el.w + kx] == 0) continue;
                        const off = @as(i64, @intCast(kx)) - c.el.ax;
                        const x0: i64 = @max(0, -off);
                        const x1: i64 = @min(W, W - off);
                        if (x1 <= x0) continue;
                        const dd = d[@as(usize, @intCast(x0)) * cn .. @as(usize, @intCast(x1)) * cn];
                        const ss = sr[@as(usize, @intCast(x0 + off)) * cn .. @as(usize, @intCast(x1 + off)) * cn];
                        for (dd, ss) |*o, v| o.* = if (op == .min) @min(o.*, v) else @max(o.*, v);
                    }
                }
            }
        }
    }.run);
}

fn maxOf(comptime T: type) T {
    return if (@typeInfo(T) == .float) std.math.inf(T) else std.math.maxInt(T);
}

fn minOf(comptime T: type) T {
    return if (@typeInfo(T) == .float) -std.math.inf(T) else std.math.minInt(T);
}

pub const MorphOp = enum(u8) { erode = 0, dilate = 1, open = 2, close = 3, gradient = 4, tophat = 5, blackhat = 6 };

/// morphologyEx. Uses one temporary image for compound operations.
pub fn morphologyEx(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), op: MorphOp, el: Element, iterations: u32) Error!void {
    const T = @TypeOf(src).Elem;
    switch (op) {
        .erode => return erode(exec, gpa, src, dst, el, iterations),
        .dilate => return dilate(exec, gpa, src, dst, el, iterations),
        else => {},
    }
    var t = try Image(T).initLike(gpa, src);
    defer t.deinit(gpa);
    switch (op) {
        .open => {
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, t, dst, el, iterations);
        },
        .close => {
            try dilate(exec, gpa, src, t, el, iterations);
            try erode(exec, gpa, t, dst, el, iterations);
        },
        .gradient => {
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, src, dst, el, iterations);
            subInPlace(T, dst, t, dst);
        },
        .tophat => {
            var t2 = try Image(T).initLike(gpa, src);
            defer t2.deinit(gpa);
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, t, t2, el, iterations);
            subInPlace(T, src, t2, dst);
        },
        .blackhat => {
            var t2 = try Image(T).initLike(gpa, src);
            defer t2.deinit(gpa);
            try dilate(exec, gpa, src, t, el, iterations);
            try erode(exec, gpa, t, t2, el, iterations);
            subInPlace(T, t2, src, dst);
        },
        .erode, .dilate => unreachable,
    }
}

/// d = saturate(a - b)
fn subInPlace(comptime T: type, a: Image(T), b: Image(T), d: Image(T)) void {
    const N = lanesM(T);
    const V = @Vector(N, T);
    for (0..d.height) |y| {
        const ra = a.row(y);
        const rb = b.row(y);
        const rd = d.row(y);
        var i: usize = 0;
        while (i + N <= rd.len) : (i += N) {
            const x: V = ra[i..][0..N].*;
            const z: V = rb[i..][0..N].*;
            rd[i..][0..N].* = if (@typeInfo(T) == .float) x - z else @select(T, x > z, x -| z, @as(V, @splat(0)));
        }
        while (i < rd.len) : (i += 1) {
            const x = ra[i];
            const z = rb[i];
            rd[i] = if (@typeInfo(T) == .float) x - z else if (x > z) x -| z else 0;
        }
    }
}

test "erode/dilate rect vs general" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 13, 9, 1);
    defer src.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(1);
    prng.random().bytes(src.slice());
    var buf: [15]u8 = undefined;
    const el = structuringElement(.rect, 5, 3, &buf);
    var a = try Image(u8).initLike(gpa, src);
    defer a.deinit(gpa);
    var b = try Image(u8).initLike(gpa, src);
    defer b.deinit(gpa);
    try erode(.serial, gpa, src, a, el, 1);
    try generalPass(.min, .serial, src, b, el);
    try std.testing.expectEqualSlices(u8, a.slice(), b.slice());
    try dilate(.serial, gpa, src, a, el, 2);
    var c = try Image(u8).initLike(gpa, src);
    defer c.deinit(gpa);
    try generalPass(.max, .serial, src, b, el);
    try generalPass(.max, .serial, b, c, el);
    try std.testing.expectEqualSlices(u8, a.slice(), c.slice());
}

/// Morphology vectors are 32 bytes: nearly every load is unaligned (shifted
/// by whole pixels), and 64-byte ones always split a cache line.
const morph_vec_bytes = 32;
fn lanesM(comptime T: type) comptime_int {
    return @max(1, morph_vec_bytes / @sizeOf(T));
}

test "run decomposition matches the per-tap reference" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    const sizes = [_][3]u32{ .{ 13, 9, 1 }, .{ 100, 21, 3 }, .{ 5, 4, 4 }, .{ 70, 33, 1 } };
    var buf: [21 * 21]u8 = undefined;
    for (sizes) |sz| {
        var src = try Image(u8).init(gpa, sz[0], sz[1], @intCast(sz[2]));
        defer src.deinit(gpa);
        rnd.bytes(src.slice());
        var a = try Image(u8).initLike(gpa, src);
        defer a.deinit(gpa);
        var b = try Image(u8).initLike(gpa, src);
        defer b.deinit(gpa);
        for (0..12) |case| {
            const w: u32 = rnd.intRangeAtMost(u32, 1, 21);
            const h: u32 = rnd.intRangeAtMost(u32, 1, 9);
            var el = structuringElement(@fromBackingInt(@intCast(case % 3)), w, h, &buf);
            if (case >= 9) {
                // Arbitrary masks with gaps and an off-centre anchor.
                for (buf[0 .. w * h]) |*m| m.* = @intFromBool(rnd.boolean());
                el.ax = rnd.uintLessThan(u32, w);
                el.ay = rnd.uintLessThan(u32, h);
            }
            try erode(.serial, gpa, src, a, el, 1);
            try generalPass(.min, .serial, src, b, el);
            try std.testing.expectEqualSlices(u8, b.slice(), a.slice());
            try dilate(.serial, gpa, src, a, el, 1);
            try generalPass(.max, .serial, src, b, el);
            try std.testing.expectEqualSlices(u8, b.slice(), a.slice());
        }
    }
    // In place, iterated.
    var src = try Image(f32).init(gpa, 40, 17, 2);
    defer src.deinit(gpa);
    for (src.slice()) |*v| v.* = rnd.float(f32);
    var ref = try Image(f32).initLike(gpa, src);
    defer ref.deinit(gpa);
    var t = try Image(f32).initLike(gpa, src);
    defer t.deinit(gpa);
    const el = structuringElement(.ellipse, 5, 5, &buf);
    try generalPass(.max, .serial, src, t, el);
    try generalPass(.max, .serial, t, ref, el);
    try dilate(.serial, gpa, src, src, el, 2);
    try std.testing.expectEqualSlices(f32, ref.slice(), src.slice());
}

test "3x3 rect matches a naive min/max" {
    const gpa = std.testing.allocator;
    var buf: [9]u8 = undefined;
    const el = structuringElement(.rect, 3, 3, &buf);
    inline for (.{ 1, 2, 3, 4 }) |cn| {
        for ([_][2]u32{ .{ 1, 1 }, .{ 2, 3 }, .{ 37, 1 }, .{ 1, 9 }, .{ 53, 37 }, .{ 70, 2 } }) |sz| {
            var src = try Image(u8).init(gpa, sz[0], sz[1], cn);
            defer src.deinit(gpa);
            for (src.slice(), 0..) |*v, i| v.* = @truncate((i *% 2654435761) >> 9);
            var dst = try Image(u8).init(gpa, sz[0], sz[1], cn);
            defer dst.deinit(gpa);
            inline for (.{ Op.min, Op.max }) |op| {
                try morph(op, .serial, gpa, src, dst, el, 1);
                for (0..sz[1]) |y| for (0..sz[0]) |x| for (0..cn) |c| {
                    var want: u8 = if (op == .min) 255 else 0;
                    for (0..3) |j| for (0..3) |i| {
                        const sy = @as(isize, @intCast(y + j)) - 1;
                        const sx = @as(isize, @intCast(x + i)) - 1;
                        if (sy < 0 or sx < 0 or sy >= sz[1] or sx >= sz[0]) continue;
                        want = vop(op, want, src.row(@intCast(sy))[@as(usize, @intCast(sx)) * cn + c]);
                    };
                    try std.testing.expectEqual(want, dst.row(y)[x * cn + c]);
                };
            }
        }
    }
}
