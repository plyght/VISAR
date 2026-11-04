//! Gaussian image pyramids: pyrDown, pyrUp and buildPyramid, matching
//! OpenCV's 5-tap [1 4 6 4 1] kernels bit for bit on integer images.
//!
//! pyrDown: dst(x, y) = (sum of the 5x5 kernel over src(2x+i, 2y+j), i, j in
//! -2..2, + 128) >> 8, with src coordinates border-mapped. pyrUp is the same
//! kernel on the image upsampled with zeros (so each output sums to 64 of the
//! 1/4-scaled taps), rounded as (sum + 32) >> 6; the upsampled image is
//! reflected about its own edges, which OpenCV's pyrUp hard-codes.
//!
//! Both run vertical-then-horizontal on one L1-resident row of exact
//! integer sums (u16 for 8-bit images, u32 for 16-bit, i32 for i16, f32 for
//! float), so the pass order does not change integer results.

const std = @import("std");
const core = @import("../core/core.zig");
const border_mod = @import("border.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Border = border_mod.Border;
pub const Error = core.Error || Allocator.Error || error{Unsupported};

/// The exact accumulator for a sample type.
fn Acc(comptime T: type) type {
    return switch (T) {
        u8 => u16,
        u16 => u32,
        i16 => i32,
        f32 => f32,
        else => @compileError("pyramids support u8, u16, i16 and f32, not " ++ @typeName(T)),
    };
}

/// Lanes per step: at least 32 bytes of the accumulator.
fn lanes(comptime A: type) comptime_int {
    const native = std.simd.suggestVectorLength(A) orelse 4;
    return @max(native, 32 / @sizeOf(A));
}

inline fn widen(comptime A: type, comptime N: usize, v: anytype) @Vector(N, A) {
    return v; // integer widening and f32 -> f32 both coerce
}

inline fn splat(comptime A: type, comptime N: usize, x: A) @Vector(N, A) {
    return @splat(x);
}

/// (v + 2^(s-1)) >> s for integers, v / 2^s for floats, narrowed to T.
inline fn finishV(comptime T: type, comptime s: comptime_int, comptime N: usize, v: anytype) @Vector(N, T) {
    const A = @typeInfo(@TypeOf(v)).vector.child;
    if (@typeInfo(A) == .float) return v * splat(A, N, 1.0 / @as(f32, 1 << s));
    const r = (v + splat(A, N, 1 << (s - 1))) >> @splat(s);
    return core.simd.narrow(T, r);
}

inline fn finishS(comptime T: type, comptime s: comptime_int, v: anytype) T {
    const A = @TypeOf(v);
    if (@typeInfo(A) == .float) return v * (1.0 / @as(f32, 1 << s));
    return @intCast((v + (1 << (s - 1))) >> s);
}

// ---------------------------------------------------------------- pyrDown

/// The size pyrDown produces: ((w + 1) / 2, (h + 1) / 2).
pub fn downSize(w: u32, h: u32) [2]u32 {
    return .{ (w + 1) / 2, (h + 1) / 2 };
}

/// Blurs with the 5x5 Gaussian and drops every other row and column. `dst`
/// must be ((w + 1) / 2, (h + 1) / 2) with the same channels. Any border
/// except `.constant` (which OpenCV rejects too).
pub fn pyrDown(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), border: Border) Error!void {
    const T = @TypeOf(src).Elem;
    if (border == .constant) return error.Unsupported;
    const ds = downSize(src.width, src.height);
    if (dst.channels != src.channels or dst.width != ds[0] or dst.height != ds[1]) return error.SizeMismatch;
    const Ctx = struct { s: Image(T), d: Image(T), border: Border, gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(2, (1 << 15) / @max(1, src.rowLen()));
    exec.parallelFor(dst.height, grain, Ctx{ .s = src, .d = dst, .border = border, .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            (switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| downChunk(T, cn, c.s, c.d, c.border, c.gpa, y0, y1),
                else => downChunk(T, 0, c.s, c.d, c.border, c.gpa, y0, y1),
            }) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

/// Rows [y0, y1) of pyrDown. `ccn` is the channel count when known at
/// compile time (1-4), else 0.
fn downChunk(comptime T: type, comptime ccn: usize, s: Image(T), d: Image(T), border: Border, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    if (T == f32 and (ccn == 3 or ccn == 4)) return downChunkPlanarF(ccn, s, d, border, gpa, y0, y1);
    const A = Acc(T);
    const N = comptime lanes(A);
    const cn: usize = if (ccn != 0) ccn else s.channels;
    const w: usize = s.width;
    const n = w * cn;
    const pad = 2 * cn;
    // [2 left pixels][row][2 right pixels][slack the vector loop may read]
    const p = try gpa.alloc(A, n + 2 * pad + 2 * N + cn);
    defer gpa.free(p);
    @memset(p[n + pad ..], 0);
    var lmap: [2]usize = undefined;
    var rmap: [2]usize = undefined;
    for (0..2) |i| {
        lmap[i] = border_mod.map(@as(isize, @intCast(i)) - 2, w, border).?;
        rmap[i] = border_mod.map(@intCast(w + i), w, border).?;
    }
    for (y0..y1) |y| {
        var rows: [5][*]const T = undefined;
        for (0..5) |k| {
            const sy = border_mod.map(@as(isize, @intCast(2 * y + k)) - 2, s.height, border).?;
            rows[k] = s.row(sy).ptr;
        }
        const mid = p[pad..][0..n];
        vert5(T, A, rows, mid);
        for (0..2) |i| {
            @memcpy(p[i * cn ..][0..cn], mid[lmap[i] * cn ..][0..cn]);
            @memcpy(p[pad + n + i * cn ..][0..cn], mid[rmap[i] * cn ..][0..cn]);
        }
        const out = d.row(y);
        if (ccn != 0) hdown(T, A, ccn, p, out) else hdownGeneric(T, A, cn, p, out);
    }
}

const neon_st = core.simd.llvm_intrinsics and @import("builtin").cpu.arch == .aarch64;
const neon = struct {
    extern fn @"llvm.aarch64.neon.st3.v4f32.p0"(@Vector(4, f32), @Vector(4, f32), @Vector(4, f32), [*]f32) void;
    extern fn @"llvm.aarch64.neon.st4.v4f32.p0"(@Vector(4, f32), @Vector(4, f32), @Vector(4, f32), @Vector(4, f32), [*]f32) void;
};

/// downChunkPlanar for f32: deinterleave/interleave through stride
/// shuffles (ld3/st3 .4s on NEON), same plane layout and pass structure.
fn downChunkPlanarF(comptime cn: usize, s: Image(f32), d: Image(f32), border: Border, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const w: usize = s.width;
    const N = 8;
    const V = @Vector(N, f32);
    const plen = w + 4 + 2 * N + 8;
    const buf = try gpa.alloc(f32, cn * plen);
    defer gpa.free(buf);
    @memset(buf, 0);
    var planes: [cn][]f32 = undefined;
    inline for (0..cn) |c| planes[c] = buf[c * plen ..][0..plen];
    var lmap: [2]usize = undefined;
    var rmap: [2]usize = undefined;
    for (0..2) |i| {
        lmap[i] = border_mod.map(@as(isize, @intCast(i)) - 2, w, border).?;
        rmap[i] = border_mod.map(@intCast(w + i), w, border).?;
    }
    const split = comptime blk: {
        var m: [cn][N]i32 = undefined;
        for (0..cn) |c| for (0..N) |i| {
            m[c][i] = @intCast(i * cn + c);
        };
        break :blk m;
    };
    const ev = comptime blk: {
        var m: [N]i32 = undefined;
        for (0..N) |i| m[i] = @intCast(2 * i);
        break :blk m;
    };
    const od = comptime blk: {
        var m: [N]i32 = undefined;
        for (0..N) |i| m[i] = @intCast(2 * i + 1);
        break :blk m;
    };
    const dw: usize = d.width;
    for (y0..y1) |y| {
        var rows: [5][*]const f32 = undefined;
        for (0..5) |k| {
            const sy = border_mod.map(@as(isize, @intCast(2 * y + k)) - 2, s.height, border).?;
            rows[k] = s.row(sy).ptr;
        }
        var x: usize = 0;
        while (x + N <= w) : (x += N) {
            var t: [5][cn]V = undefined;
            inline for (0..5) |k| {
                const wv: @Vector(N * cn, f32) = rows[k][x * cn ..][0 .. N * cn].*;
                inline for (0..cn) |c| t[k][c] = @shuffle(f32, wv, undefined, split[c]);
            }
            inline for (0..cn) |c| planes[c][2 + x ..][0..N].* = (t[0][c] + t[4][c]) + (t[1][c] + t[3][c]) * @as(V, @splat(4)) + t[2][c] * @as(V, @splat(6));
        }
        while (x < w) : (x += 1) {
            inline for (0..cn) |c| {
                const i = x * cn + c;
                planes[c][2 + x] = (rows[0][i] + rows[4][i]) + (rows[1][i] + rows[3][i]) * 4 + rows[2][i] * 6;
            }
        }
        inline for (0..cn) |c| {
            const pl = planes[c];
            for (0..2) |i| {
                pl[i] = pl[2 + lmap[i]];
                pl[2 + w + i] = pl[2 + rmap[i]];
            }
        }
        const out = d.row(y);
        const join = comptime blk: {
            var m: [N * cn]i32 = undefined;
            for (0..N) |i| for (0..cn) |c| {
                m[i * cn + c] = @intCast(c * N + i);
            };
            break :blk m;
        };
        var ox: usize = 0;
        while (ox + N <= dw) : (ox += N) {
            var res: [cn * N]f32 = undefined;
            inline for (0..cn) |c| {
                const pl = planes[c];
                const w0: @Vector(2 * N, f32) = pl[2 * ox ..][0 .. 2 * N].*;
                const w1: @Vector(2 * N, f32) = pl[2 * ox + 2 ..][0 .. 2 * N].*;
                const w2: @Vector(2 * N, f32) = pl[2 * ox + 4 ..][0 .. 2 * N].*;
                const e0: V = @shuffle(f32, w0, undefined, ev);
                const o0: V = @shuffle(f32, w0, undefined, od);
                const e1: V = @shuffle(f32, w1, undefined, ev);
                const o1: V = @shuffle(f32, w1, undefined, od);
                const e2: V = @shuffle(f32, w2, undefined, ev);
                const v = (e0 + e2) + (o0 + o1) * @as(V, @splat(4)) + e1 * @as(V, @splat(6));
                res[c * N ..][0..N].* = finishV(f32, 8, N, v);
            }
            if (comptime neon_st) {
                // LLVM does not match the interleaving shuffle to st3/st4.
                const F4 = @Vector(4, f32);
                inline for (0..N / 4) |h| {
                    var q: [cn]F4 = undefined;
                    inline for (0..cn) |c| q[c] = res[c * N + h * 4 ..][0..4].*;
                    const dst = out.ptr + (ox + h * 4) * cn;
                    if (cn == 3) neon.@"llvm.aarch64.neon.st3.v4f32.p0"(q[0], q[1], q[2], dst) else neon.@"llvm.aarch64.neon.st4.v4f32.p0"(q[0], q[1], q[2], q[3], dst);
                }
            } else {
                const all: @Vector(cn * N, f32) = res;
                out[ox * cn ..][0 .. N * cn].* = @shuffle(f32, all, undefined, join);
            }
        }
        while (ox < dw) : (ox += 1) {
            inline for (0..cn) |c| {
                const pl = planes[c];
                const q = 2 * ox;
                out[ox * cn + c] = finishS(f32, 8, (pl[q] + pl[q + 4]) + (pl[q + 1] + pl[q + 3]) * 4 + pl[q + 2] * 6);
            }
        }
    }
}

/// out = r0 + 4 r1 + 6 r2 + 4 r3 + r4, widened to A.
fn vert5(comptime T: type, comptime A: type, rows: [5][*]const T, out: []A) void {
    const N = comptime lanes(A);
    const V = @Vector(N, A);
    const r = rows;
    var x: usize = 0;
    while (x + N <= out.len) : (x += N) {
        const a: V = widen(A, N, @as(@Vector(N, T), r[0][x..][0..N].*));
        const b: V = widen(A, N, @as(@Vector(N, T), r[1][x..][0..N].*));
        const c: V = widen(A, N, @as(@Vector(N, T), r[2][x..][0..N].*));
        const e: V = widen(A, N, @as(@Vector(N, T), r[3][x..][0..N].*));
        const f: V = widen(A, N, @as(@Vector(N, T), r[4][x..][0..N].*));
        out[x..][0..N].* = (a + f) + (b + e) * splat(A, N, 4) + c * splat(A, N, 6);
    }
    while (x < out.len) : (x += 1) {
        const a: A = r[0][x];
        const b: A = r[1][x];
        const c: A = r[2][x];
        const e: A = r[3][x];
        const f: A = r[4][x];
        out[x] = (a + f) + (b + e) * 4 + c * 6;
    }
}

/// The horizontal 5-tap sum centred on element e + 2cn of the padded row.
inline fn h5(comptime A: type, comptime N: usize, p: []const A, e: usize, cn: usize) @Vector(N, A) {
    const V = @Vector(N, A);
    const a: V = p[e..][0..N].*;
    const b: V = p[e + cn ..][0..N].*;
    const c: V = p[e + 2 * cn ..][0..N].*;
    const d: V = p[e + 3 * cn ..][0..N].*;
    const f: V = p[e + 4 * cn ..][0..N].*;
    return (a + f) + (b + d) * splat(A, N, 4) + c * splat(A, N, 6);
}

inline fn h5s(comptime A: type, p: []const A, e: usize, cn: usize) A {
    return (p[e] + p[e + 4 * cn]) + (p[e + cn] + p[e + 3 * cn]) * 4 + p[e + 2 * cn] * 6;
}

/// Horizontal filter at full resolution over two vectors, then one shuffle
/// keeps the even pixels.
fn hdown(comptime T: type, comptime A: type, comptime cn: usize, p: []const A, out: []T) void {
    const N = comptime lanes(A);
    const mask = comptime blk: {
        var m: [N]i32 = undefined;
        for (0..N) |i| {
            const idx = 2 * (i / cn) * cn + i % cn;
            m[i] = if (idx < N) idx else ~@as(i32, idx - N);
        }
        break :blk m;
    };
    const step = (N / cn) * cn;
    var x: usize = 0;
    while (x + N <= out.len) : (x += step) {
        const lo = h5(A, N, p, 2 * x, cn);
        const hi = h5(A, N, p, 2 * x + N, cn);
        out[x..][0..N].* = finishV(T, 8, N, @shuffle(A, lo, hi, mask));
    }
    while (x < out.len) : (x += 1) out[x] = finishS(T, 8, h5s(A, p, 2 * x - x % cn, cn));
}

fn hdownGeneric(comptime T: type, comptime A: type, cn: usize, p: []const A, out: []T) void {
    for (out, 0..) |*o, x| o.* = finishS(T, 8, h5s(A, p, 2 * x - x % cn, cn));
}

// ------------------------------------------------------------------ pyrUp

/// Upsamples by two (dst is 2w x 2h) and blurs with the 5x5 Gaussian scaled
/// by 4. OpenCV's pyrUp only supports its default border (reflect101).
pub fn pyrUp(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), border: Border) Error!void {
    const T = @TypeOf(src).Elem;
    if (border != .reflect101) return error.Unsupported;
    if (dst.channels != src.channels or dst.width != @as(u64, src.width) * 2 or dst.height != @as(u64, src.height) * 2) return error.SizeMismatch;
    const Ctx = struct { s: Image(T), d: Image(T), gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(1, (1 << 14) / @max(1, src.rowLen()));
    exec.parallelFor(src.height, grain, Ctx{ .s = src, .d = dst, .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            (switch (c.s.channels) {
                inline 1, 2, 3, 4 => |cn| upChunk(T, cn, c.s, c.d, c.gpa, y0, y1),
                else => upChunk(T, 0, c.s, c.d, c.gpa, y0, y1),
            }) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

/// Source rows [y0, y1) of pyrUp: each makes destination rows 2y and 2y+1.
fn upChunk(comptime T: type, comptime ccn: usize, s: Image(T), d: Image(T), gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const A = Acc(T);
    const N = comptime lanes(A);
    const cn: usize = if (ccn != 0) ccn else s.channels;
    const w: usize = s.width;
    const h: usize = s.height;
    const n = w * cn;
    // [1 left pixel][row][1 right pixel][slack]
    const qlen = n + 2 * cn + N;
    const buf = try gpa.alloc(A, 2 * qlen);
    defer gpa.free(buf);
    const qe = buf[0..qlen];
    const qo = buf[qlen..][0..qlen];
    @memset(qe[n + 2 * cn ..], 0);
    @memset(qo[n + 2 * cn ..], 0);
    // The upsampled image reflected about its edges: pixel -1 is pixel 1,
    // pixel w is pixel w - 1 (also for rows).
    const left: usize = if (w > 1) 1 else 0;
    for (y0..y1) |y| {
        const a = s.row(if (y > 0) y - 1 else if (h > 1) 1 else 0).ptr;
        const b = s.row(y).ptr;
        const c = s.row(if (y + 1 < h) y + 1 else y).ptr;
        vert3(T, A, a, b, c, qe[cn..][0..n], qo[cn..][0..n]);
        for ([2][]A{ qe, qo }) |q| {
            @memcpy(q[0..cn], q[cn + left * cn ..][0..cn]);
            @memcpy(q[cn + n ..][0..cn], q[n..][0..cn]);
        }
        if (ccn != 0) {
            hup(T, A, ccn, qe, d.row(2 * y));
            hup(T, A, ccn, qo, d.row(2 * y + 1));
        } else {
            hupGeneric(T, A, cn, qe, d.row(2 * y));
            hupGeneric(T, A, cn, qo, d.row(2 * y + 1));
        }
    }
}

/// Even output row: a + 6 b + c; odd output row: 4 (b + c).
fn vert3(comptime T: type, comptime A: type, a: [*]const T, b: [*]const T, c: [*]const T, even: []A, odd: []A) void {
    const N = comptime lanes(A);
    const V = @Vector(N, A);
    var x: usize = 0;
    while (x + N <= even.len) : (x += N) {
        const va: V = widen(A, N, @as(@Vector(N, T), a[x..][0..N].*));
        const vb: V = widen(A, N, @as(@Vector(N, T), b[x..][0..N].*));
        const vc: V = widen(A, N, @as(@Vector(N, T), c[x..][0..N].*));
        even[x..][0..N].* = (va + vc) + vb * splat(A, N, 6);
        odd[x..][0..N].* = (vb + vc) * splat(A, N, 4);
    }
    while (x < even.len) : (x += 1) {
        const va: A = a[x];
        const vb: A = b[x];
        const vc: A = c[x];
        even[x] = (va + vc) + vb * 6;
        odd[x] = (vb + vc) * 4;
    }
}

/// Even output pixel 2p: q[p-1] + 6 q[p] + q[p+1]; odd 2p+1: 4 (q[p] + q[p+1]).
/// Both halves are computed for a vector of source pixels, then one shuffle
/// interleaves them pixel by pixel.
fn hup(comptime T: type, comptime A: type, comptime cn: usize, q: []const A, out: []T) void {
    const N = comptime lanes(A);
    const V = @Vector(N, A);
    const step = (N / cn) * cn;
    const mask = comptime blk: {
        var m: [2 * step]i32 = undefined;
        for (0..2 * step) |j| {
            const px = j / cn;
            const idx: i32 = @intCast((px / 2) * cn + j % cn);
            m[j] = if (px % 2 == 0) idx else ~idx;
        }
        break :blk m;
    };
    const n = out.len / 2;
    var x: usize = 0;
    while (x + step <= n) : (x += step) {
        const l: V = q[x..][0..N].*;
        const c: V = q[x + cn ..][0..N].*;
        const r: V = q[x + 2 * cn ..][0..N].*;
        const ev = (l + r) + c * splat(A, N, 6);
        const od = (c + r) * splat(A, N, 4);
        out[2 * x ..][0 .. 2 * step].* = finishV(T, 6, 2 * step, @shuffle(A, ev, od, mask));
    }
    while (x < n) : (x += 1) upPixel(T, A, cn, q, out, x);
}

inline fn upPixel(comptime T: type, comptime A: type, cn: usize, q: []const A, out: []T, x: usize) void {
    const o = 2 * x - x % cn; // output element of the even pixel
    out[o] = finishS(T, 6, (q[x] + q[x + 2 * cn]) + q[x + cn] * 6);
    out[o + cn] = finishS(T, 6, (q[x + cn] + q[x + 2 * cn]) * 4);
}

fn hupGeneric(comptime T: type, comptime A: type, cn: usize, q: []const A, out: []T) void {
    for (0..out.len / 2) |x| upPixel(T, A, cn, q, out, x);
}

// ------------------------------------------------------------ buildPyramid

/// A Gaussian pyramid: `levels[0]` is a view of the source, each further
/// level is pyrDown of the one before.
pub fn Pyramid(comptime T: type) type {
    return struct {
        levels: []Image(T),

        pub fn deinit(self: *@This(), gpa: Allocator) void {
            for (self.levels[1..]) |*l| l.deinit(gpa);
            gpa.free(self.levels);
            self.* = undefined;
        }
    };
}

/// buildPyramid: levels 0..max_level (level 0 is a view of `src`, not a
/// copy). Stops early, like repeated pyrDown would fail, once a level would
/// be empty; it never does for images of at least 1x1, since pyrDown of 1x1
/// is 1x1.
pub fn buildPyramid(exec: Exec, gpa: Allocator, src: anytype, max_level: u32, border: Border) Error!Pyramid(@TypeOf(src).Elem) {
    const T = @TypeOf(src).Elem;
    const levels = try gpa.alloc(Image(T), @as(usize, max_level) + 1);
    var built: usize = 1;
    errdefer {
        for (levels[1..built]) |*l| l.deinit(gpa);
        gpa.free(levels);
    }
    levels[0] = src.view();
    while (built <= max_level) : (built += 1) {
        const prev = levels[built - 1];
        const sz = downSize(prev.width, prev.height);
        var next = try Image(T).init(gpa, sz[0], sz[1], prev.channels);
        pyrDown(exec, gpa, prev, next, border) catch |e| {
            next.deinit(gpa);
            return e;
        };
        levels[built] = next;
    }
    return .{ .levels = levels };
}

// ------------------------------------------------------------------ tests

fn refDown(comptime T: type, s: Image(T), x: usize, y: usize, ch: usize, border: Border) f64 {
    const k = [5]i64{ 1, 4, 6, 4, 1 };
    var sum: f64 = 0;
    for (0..5) |j| for (0..5) |i| {
        const sy = border_mod.map(@as(isize, @intCast(2 * y + j)) - 2, s.height, border).?;
        const sx = border_mod.map(@as(isize, @intCast(2 * x + i)) - 2, s.width, border).?;
        const v: f64 = if (@typeInfo(T) == .float) s.at(sx, sy, ch) else @floatFromInt(s.at(sx, sy, ch));
        sum += v * @as(f64, @floatFromInt(k[i] * k[j]));
    };
    return sum;
}

/// pyrUp reference: the 5x5 kernel over the zero-upsampled image reflected
/// about its own edges (reflect101 in upsampled coordinates).
fn refUp(comptime T: type, s: Image(T), x: usize, y: usize, ch: usize) f64 {
    const k = [5]i64{ 1, 4, 6, 4, 1 };
    var sum: f64 = 0;
    for (0..5) |j| for (0..5) |i| {
        const uy = border_mod.map(@as(isize, @intCast(y + j)) - 2, 2 * s.height, .reflect101).?;
        const ux = border_mod.map(@as(isize, @intCast(x + i)) - 2, 2 * s.width, .reflect101).?;
        if (uy % 2 != 0 or ux % 2 != 0) continue;
        const v: f64 = if (@typeInfo(T) == .float) s.at(ux / 2, uy / 2, ch) else @floatFromInt(s.at(ux / 2, uy / 2, ch));
        sum += v * @as(f64, @floatFromInt(k[i] * k[j]));
    };
    return sum;
}

fn fillRandom(comptime T: type, img: Image(T), seed0: u32) void {
    var seed = seed0;
    for (0..img.height) |y| for (img.row(y)) |*v| {
        seed = seed *% 1103515245 +% 12345;
        const r: u16 = @truncate(seed >> 12);
        v.* = switch (T) {
            u8 => @truncate(r),
            u16 => r,
            i16 => @bitCast(r),
            f32 => @as(f32, @floatFromInt(r)) / 65536.0,
            else => unreachable,
        };
    };
}

test "pyrDown and pyrUp match a direct 5x5 sum" {
    const gpa = std.testing.allocator;
    inline for (.{ u8, u16, i16, f32 }) |T| {
        for ([_][2]u32{ .{ 1, 1 }, .{ 2, 3 }, .{ 5, 4 }, .{ 37, 19 }, .{ 70, 9 }, .{ 129, 7 } }) |sz| {
            for ([_]u16{ 1, 2, 3, 4, 5 }) |cn| {
                var src = try Image(T).init(gpa, sz[0], sz[1], cn);
                defer src.deinit(gpa);
                fillRandom(T, src, sz[0] * 31 + cn);
                const ds = downSize(sz[0], sz[1]);
                var down = try Image(T).init(gpa, ds[0], ds[1], cn);
                defer down.deinit(gpa);
                for ([_]Border{ .reflect101, .reflect, .replicate, .wrap }) |b| {
                    try pyrDown(.serial, gpa, src, down, b);
                    for (0..ds[1]) |y| for (0..ds[0]) |x| for (0..cn) |ch| {
                        const s = refDown(T, src, x, y, ch, b);
                        const want: f64 = if (@typeInfo(T) == .float) s / 256 else @floor((s + 128) / 256);
                        const got: f64 = if (@typeInfo(T) == .float) down.at(x, y, ch) else @floatFromInt(down.at(x, y, ch));
                        try std.testing.expectApproxEqAbs(want, got, if (@typeInfo(T) == .float) 1e-5 else 0);
                    };
                }
                var up = try Image(T).init(gpa, sz[0] * 2, sz[1] * 2, cn);
                defer up.deinit(gpa);
                try pyrUp(.serial, gpa, src, up, .reflect101);
                for (0..up.height) |y| for (0..up.width) |x| for (0..cn) |ch| {
                    const s = refUp(T, src, x, y, ch);
                    const want: f64 = if (@typeInfo(T) == .float) s / 64 else @floor((s + 32) / 64);
                    const got: f64 = if (@typeInfo(T) == .float) up.at(x, y, ch) else @floatFromInt(up.at(x, y, ch));
                    try std.testing.expectApproxEqAbs(want, got, if (@typeInfo(T) == .float) 1e-5 else 0);
                };
            }
        }
    }
}

test "buildPyramid sizes" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).initFill(gpa, 101, 50, 1, 7);
    defer src.deinit(gpa);
    var p = try buildPyramid(.serial, gpa, src, 4, .reflect101);
    defer p.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 5), p.levels.len);
    try std.testing.expectEqual(@as(u32, 51), p.levels[1].width);
    try std.testing.expectEqual(@as(u32, 25), p.levels[1].height);
    try std.testing.expectEqual(@as(u32, 7), p.levels[4].width);
    try std.testing.expectEqual(@as(u32, 4), p.levels[4].height);
    try std.testing.expectEqual(@as(u8, 7), p.levels[4].at(3, 2, 0));
    try std.testing.expectError(error.SizeMismatch, pyrDown(.serial, gpa, src, p.levels[2], .reflect101));
}
