//! Corner detection: cornerMinEigenVal, cornerHarris and
//! goodFeaturesToTrack, following OpenCV's corner.cpp and featureselect.cpp.
//!
//! The structure tensor uses Sobel (or Scharr) derivatives scaled by
//! 1 / (2^(ksize-1) * blockSize [* 2 for Scharr] [* 255 for 8-bit]) and an
//! unnormalised blockSize x blockSize box sum of Dx^2, DxDy and Dy^2. With
//! 3x3 apertures (the default, and goodFeaturesToTrack's) the f32
//! derivatives are rounded exactly as OpenCV rounds them and the box sums
//! use f64 like OpenCV's, so responses match bit for bit and corners rank
//! identically. Other 8-bit apertures keep the derivatives and box sums as
//! exact integers and apply the scale once, which agrees with OpenCV to a
//! few f32 ulps of the tensor entries.

const std = @import("std");
const core = @import("../core/core.zig");
const border_mod = @import("border.zig");
const filter = @import("filter.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Border = border_mod.Border;
pub const Error = core.Error || Allocator.Error || error{ InvalidKernel, Unsupported };

pub const Point2f = @import("shape.zig").Point2f;

pub const CornerOptions = struct {
    /// Side of the window the tensor is summed over (any size >= 1; even
    /// sizes anchor like OpenCV, at blockSize / 2).
    block_size: u32 = 3,
    /// Derivative aperture: 1, 3, 5, 7, or -1 for Scharr.
    ksize: i32 = 3,
    border: Border = .default,
};

const Kind = enum { min_eig, harris };

/// The smaller eigenvalue of the structure tensor at every pixel
/// (OpenCV's cornerMinEigenVal). `src` is u8 or f32, one channel; `dst` is f32.
pub fn cornerMinEigenVal(exec: Exec, gpa: Allocator, src: anytype, dst: Image(f32), o: CornerOptions) Error!void {
    return response(exec, gpa, src, dst, o, .min_eig, 0);
}

/// Harris response det(M) - k * trace(M)^2 (OpenCV's cornerHarris).
pub fn cornerHarris(exec: Exec, gpa: Allocator, src: anytype, dst: Image(f32), o: CornerOptions, k: f64) Error!void {
    return response(exec, gpa, src, dst, o, .harris, k);
}

fn response(exec: Exec, gpa: Allocator, src: anytype, dst: Image(f32), o: CornerOptions, kind: Kind, k: f64) Error!void {
    const T = @TypeOf(src).Elem;
    if (T != u8 and T != f32) @compileError("corner detection supports u8 and f32 images");
    if (src.channels != 1 or dst.channels != 1) return error.UnsupportedChannels;
    if (!src.sameSize(dst)) return error.SizeMismatch;
    if (o.block_size == 0 or o.block_size > 255) return error.InvalidKernel;
    if (!(o.ksize == -1 or o.ksize == 1 or o.ksize == 3 or o.ksize == 5 or o.ksize == 7)) return error.InvalidKernel;
    if (o.border == .wrap) return error.Unsupported; // OpenCV's filters reject it too
    var scale: f64 = @floatFromInt((@as(u32, 1) << @intCast((if (o.ksize > 0) o.ksize else 3) - 1)) * o.block_size);
    if (o.ksize < 0) scale *= 2;
    if (T == u8) scale *= 255;
    scale = 1.0 / scale;
    if (o.ksize == 3) return fused3(T, exec, gpa, src, dst, o, kind, k, scale);
    if (T == f32) {
        var dx = try Image(f32).initLike(gpa, src);
        defer dx.deinit(gpa);
        var dy = try Image(f32).initLike(gpa, src);
        defer dy.deinit(gpa);
        try derivs(exec, gpa, src, dx, dy, o, scale);
        return tensor(f32, f64, exec, gpa, dx, dy, dst, o, kind, k, 1);
    }
    // Exact integer derivatives: i16 while they fit, else i32.
    if (o.ksize == 7) {
        var dx = try Image(i32).initLike(gpa, src);
        defer dx.deinit(gpa);
        var dy = try Image(i32).initLike(gpa, src);
        defer dy.deinit(gpa);
        try derivs(exec, gpa, src, dx, dy, o, 1);
        return tensor(i32, f64, exec, gpa, dx, dy, dst, o, kind, k, scale * scale);
    }
    var dx = try Image(i16).initLike(gpa, src);
    defer dx.deinit(gpa);
    var dy = try Image(i16).initLike(gpa, src);
    defer dy.deinit(gpa);
    try derivs(exec, gpa, src, dx, dy, o, 1);
    // |D| <= 255 * (sum of |taps|); the box sum of products must fit i32.
    const bound: u64 = 255 * @as(u64, switch (o.ksize) {
        -1 => 32,
        1 => 2,
        3 => 8,
        else => 96,
    });
    const bs: u64 = o.block_size;
    if (bound * bound * bs * bs <= std.math.maxInt(i32))
        return tensor(i16, i32, exec, gpa, dx, dy, dst, o, kind, k, scale * scale);
    return tensor(i16, f64, exec, gpa, dx, dy, dst, o, kind, k, scale * scale);
}

fn derivs(exec: Exec, gpa: Allocator, src: anytype, dx: anytype, dy: @TypeOf(dx), o: CornerOptions, scale: f64) Error!void {
    filter.sobel(exec, gpa, src, dx, .{ .dx = 1, .dy = 0, .ksize = o.ksize, .scale = scale, .border = o.border }) catch |e| return mapErr(e);
    filter.sobel(exec, gpa, src, dy, .{ .dx = 0, .dy = 1, .ksize = o.ksize, .scale = scale, .border = o.border }) catch |e| return mapErr(e);
}

/// 3x3 apertures: derivatives (sobel3) and tensor in bands of rows, so
/// the derivative rows are still in cache when the box sums read them.
fn fused3(comptime T: type, exec: Exec, gpa: Allocator, src: Image(T), dst: Image(f32), o: CornerOptions, kind: Kind, k: f64, scale: f64) Error!void {
    const Ctx = struct { s: Image(T), d: Image(f32), o: CornerOptions, kind: Kind, k: f32, k0: f32, gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(16, (1 << 16) / @max(1, src.width));
    exec.parallelFor(src.height, grain, Ctx{ .s = src, .d = dst, .o = o, .kind = kind, .k = @floatCast(k), .k0 = @floatCast(scale), .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            (switch (c.kind) {
                inline else => |kd| fused3Chunk(T, kd, c.s, c.d, c.o, c.k, c.k0, c.gpa, y0, y1),
            }) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

fn fused3Chunk(comptime T: type, comptime kind: Kind, s: Image(T), d: Image(f32), o: CornerOptions, k: f32, k0: f32, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const band = 256;
    const w: usize = s.width;
    const h: usize = s.height;
    const bs: usize = o.block_size;
    const anchor = bs / 2;
    var b0 = y0;
    var cap: usize = 0;
    var buf: []f32 = &.{};
    defer gpa.free(buf);
    while (b0 < y1) {
        const b1 = @min(y1, b0 + band);
        // Source rows the band's windows touch.
        var lo: usize = h;
        var hi: usize = 0;
        var yy: isize = @as(isize, @intCast(b0)) - @as(isize, @intCast(anchor)) - 1;
        while (yy <= @as(isize, @intCast(b1 + bs - anchor))) : (yy += 1) {
            if (border_mod.map(yy, h, o.border)) |r| {
                lo = @min(lo, r);
                hi = @max(hi, r);
            }
        }
        if (lo > hi) {
            lo = 0;
            hi = 0;
        }
        const rows = hi - lo + 1;
        if (2 * rows * w > cap) {
            gpa.free(buf);
            buf = &.{};
            buf = try gpa.alloc(f32, 2 * rows * w);
            cap = buf.len;
        }
        const dx: Image(f32) = .{ .width = s.width, .height = @intCast(rows), .channels = 1, .stride = w, .data = buf.ptr };
        const dy: Image(f32) = .{ .width = s.width, .height = @intCast(rows), .channels = 1, .stride = w, .data = buf.ptr + rows * w };
        try sobel3Chunk(T, s, dx, dy, lo, o.border, k0, gpa, lo, hi + 1);
        try tensorChunk(f32, f64, kind, dx, dy, lo, d, o, k, 1, gpa, b0, b1);
        b0 = b1;
    }
}

/// Rows [y0, y1) of the 3x3 Sobel pair (written to row y - orow0) in f32 with the scale folded into the smoothing
/// taps, rounded exactly as OpenCV's filter engine rounds it (its SIMD
/// kernels use fused multiply-adds in this order): Dx is [-1 0 1] along
/// rows, then fma(r0 + r2, s, 2s * r1) down columns; Dy is
/// fma(c, s, fma(b, 2s, s * a)) along rows, then r2 - r0. This makes the
/// default goodFeaturesToTrack (3x3 apertures) rank near-equal corners
/// exactly like OpenCV.
fn sobel3Chunk(comptime T: type, s: Image(T), dx: Image(f32), dy: Image(f32), orow0: usize, border: Border, k0: f32, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const w: usize = s.width;
    const NF = comptime lanes(f32);
    const V = @Vector(NF, f32);
    // Three cached source rows (tagged by row index; `none` is the zero row
    // of a constant border), each as its difference and its smoothing.
    const none = std.math.maxInt(usize);
    const buf = try gpa.alloc(f32, (w + 2) + 6 * w);
    defer gpa.free(buf);
    const p = buf[0 .. w + 2];
    var diff: [3][]f32 = undefined;
    var smooth: [3][]f32 = undefined;
    for (0..3) |j| {
        diff[j] = buf[w + 2 + 2 * j * w ..][0..w];
        smooth[j] = buf[w + 2 + (2 * j + 1) * w ..][0..w];
    }
    var tags = [3]usize{ none - 1, none - 1, none - 1 };
    const k1 = k0 * 2;
    const lx = border_mod.map(-1, w, border);
    const rx = border_mod.map(@intCast(w), w, border);
    for (y0..y1) |y| {
        var want: [3]usize = undefined;
        for (0..3) |j| want[j] = border_mod.map(@as(isize, @intCast(y + j)) - 1, s.height, border) orelse none;
        var slot: [3]usize = undefined;
        var used = [3]bool{ false, false, false };
        var missing = [3]bool{ true, true, true };
        for (0..3) |j| for (0..3) |k| if (!used[k] and tags[k] == want[j]) {
            slot[j] = k;
            used[k] = true;
            missing[j] = false;
            break;
        };
        for (0..3) |j| if (missing[j]) {
            const k = for (0..3) |kk| {
                if (!used[kk]) break kk;
            } else unreachable;
            used[k] = true;
            slot[j] = k;
            tags[k] = want[j];
            const df = diff[k];
            const sm = smooth[k];
            if (want[j] == none) {
                @memset(df, 0);
                @memset(sm, 0);
                continue;
            }
            const row = s.row(want[j]);
            for (p[1 .. w + 1], row) |*d, v| d.* = if (T == f32) v else @floatFromInt(v);
            p[0] = if (lx) |i| p[1 + i] else 0;
            p[w + 1] = if (rx) |i| p[1 + i] else 0;
            var x: usize = 0;
            while (x + NF <= w) : (x += NF) {
                const a: V = p[x..][0..NF].*;
                const b: V = p[x + 1 ..][0..NF].*;
                const c: V = p[x + 2 ..][0..NF].*;
                df[x..][0..NF].* = c - a;
                sm[x..][0..NF].* = @mulAdd(V, c, @splat(k0), @mulAdd(V, b, @splat(k1), a * @as(V, @splat(k0))));
            }
            while (x < w) : (x += 1) {
                df[x] = p[x + 2] - p[x];
                sm[x] = @mulAdd(f32, p[x + 2], k0, @mulAdd(f32, p[x + 1], k1, p[x] * k0));
            }
        };
        const d0s = diff[slot[0]];
        const d1s = diff[slot[1]];
        const d2s = diff[slot[2]];
        const s0s = smooth[slot[0]];
        const s2s = smooth[slot[2]];
        const ox = dx.row(y - orow0);
        const oy = dy.row(y - orow0);
        var x: usize = 0;
        while (x + NF <= w) : (x += NF) {
            const d0: V = d0s[x..][0..NF].*;
            const d1: V = d1s[x..][0..NF].*;
            const d2: V = d2s[x..][0..NF].*;
            ox[x..][0..NF].* = @mulAdd(V, d0 + d2, @splat(k0), d1 * @as(V, @splat(k1)));
            const s0: V = s0s[x..][0..NF].*;
            const s2: V = s2s[x..][0..NF].*;
            oy[x..][0..NF].* = s2 - s0;
        }
        while (x < w) : (x += 1) {
            ox[x] = @mulAdd(f32, d0s[x] + d2s[x], k0, d1s[x] * k1);
            oy[x] = s2s[x] - s0s[x];
        }
    }
}

fn mapErr(e: filter.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidKernel => error.InvalidKernel,
        error.SizeMismatch => error.SizeMismatch,
        error.InvalidSize => error.InvalidSize,
        error.UnsupportedChannels => error.UnsupportedChannels,
        error.OutOfBounds => error.OutOfBounds,
    };
}

/// Lanes per step: at least 32 bytes of the accumulator.
fn lanes(comptime A: type) comptime_int {
    const native = std.simd.suggestVectorLength(A) orelse 4;
    return @max(native, 32 / @sizeOf(A));
}

/// Box sums of the tensor entries (running column sums, like the box
/// filter) and the response, for rows split across threads.
fn tensor(comptime G: type, comptime A: type, exec: Exec, gpa: Allocator, dx: Image(G), dy: Image(G), dst: Image(f32), o: CornerOptions, kind: Kind, k: f64, s2: f64) Error!void {
    const Ctx = struct { dx: Image(G), dy: Image(G), d: Image(f32), o: CornerOptions, kind: Kind, k: f32, s2: f64, gpa: Allocator, err: *?Allocator.Error };
    var err: ?Allocator.Error = null;
    const grain = @max(8, (1 << 16) / @max(1, dx.width));
    exec.parallelFor(dx.height, grain, Ctx{ .dx = dx, .dy = dy, .d = dst, .o = o, .kind = kind, .k = @floatCast(k), .s2 = s2, .gpa = gpa, .err = &err }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            (switch (c.kind) {
                inline else => |kd| tensorChunk(G, A, kd, c.dx, c.dy, 0, c.d, c.o, c.k, c.s2, c.gpa, y0, y1),
            }) catch |e| {
                c.err.* = e;
            };
        }
    }.run);
    if (err) |e| return e;
}

/// dx * dy as A (exact for integers; f32 products widened, as OpenCV
/// multiplies in f32 and sums in f64).
inline fn prod(comptime A: type, comptime N: usize, a: anytype, b: @TypeOf(a)) @Vector(N, A) {
    const G = @typeInfo(@TypeOf(a)).vector.child;
    if (@typeInfo(G) == .float) return @floatCast(a * b);
    if (@typeInfo(A) == .float) {
        const fa: @Vector(N, A) = @floatFromInt(a);
        const fb: @Vector(N, A) = @floatFromInt(b);
        return fa * fb;
    }
    const wa: @Vector(N, A) = a;
    const wb: @Vector(N, A) = b;
    return wa * wb;
}

inline fn prodS(comptime A: type, a: anytype, b: @TypeOf(a)) A {
    const G = @TypeOf(a);
    if (@typeInfo(G) == .float) return @floatCast(a * b);
    if (@typeInfo(A) == .float) return @as(A, @floatFromInt(a)) * @as(A, @floatFromInt(b));
    return @as(A, a) * @as(A, b);
}

/// Adds (sign = 1) or subtracts the products of one derivative row.
fn accumulate(comptime G: type, comptime A: type, sa: []A, sb: []A, sc: []A, rx: []const G, ry: []const G, comptime add: bool) void {
    const N = comptime lanes(A);
    const V = @Vector(N, A);
    var x: usize = 0;
    while (x + N <= sa.len) : (x += N) {
        const gx: @Vector(N, G) = rx[x..][0..N].*;
        const gy: @Vector(N, G) = ry[x..][0..N].*;
        const pa = prod(A, N, gx, gx);
        const pb = prod(A, N, gx, gy);
        const pc = prod(A, N, gy, gy);
        const va: V = sa[x..][0..N].*;
        const vb: V = sb[x..][0..N].*;
        const vc: V = sc[x..][0..N].*;
        sa[x..][0..N].* = if (add) va + pa else va - pa;
        sb[x..][0..N].* = if (add) vb + pb else vb - pb;
        sc[x..][0..N].* = if (add) vc + pc else vc - pc;
    }
    while (x < sa.len) : (x += 1) {
        const pa = prodS(A, rx[x], rx[x]);
        const pb = prodS(A, rx[x], ry[x]);
        const pc = prodS(A, ry[x], ry[x]);
        if (add) {
            sa[x] += pa;
            sb[x] += pb;
            sc[x] += pc;
        } else {
            sa[x] -= pa;
            sb[x] -= pb;
            sc[x] -= pc;
        }
    }
}

inline fn toF32(comptime N: usize, v: anytype, s2: f64) @Vector(N, f32) {
    const A = @typeInfo(@TypeOf(v)).vector.child;
    const f: @Vector(N, f64) = if (@typeInfo(A) == .float) v else @floatFromInt(v);
    return @floatCast(f * @as(@Vector(N, f64), @splat(s2)));
}

inline fn respond(comptime kind: Kind, comptime N: usize, a: @Vector(N, f32), b: @Vector(N, f32), c: @Vector(N, f32), k: f32) @Vector(N, f32) {
    const V = @Vector(N, f32);
    if (kind == .min_eig) {
        const half: V = @splat(0.5);
        const ha = a * half;
        const hc = c * half;
        const t = ha - hc;
        return (ha + hc) - @sqrt(t * t + b * b);
    }
    const ac = a + c;
    return (a * c - b * b) - (@as(V, @splat(k)) * ac) * ac;
}

fn tensorChunk(comptime G: type, comptime A: type, comptime kind: Kind, dx: Image(G), dy: Image(G), row0: usize, d: Image(f32), o: CornerOptions, k: f32, s2: f64, gpa: Allocator, y0: usize, y1: usize) Allocator.Error!void {
    const w: usize = dx.width;
    const h: usize = d.height;
    const bs: usize = o.block_size;
    const anchor = bs / 2;
    const total = w + bs - 1;
    const N = comptime lanes(A);
    const buf = try gpa.alloc(A, 3 * total);
    defer gpa.free(buf);
    const pa = buf[0..total];
    const pb = buf[total..][0..total];
    const pc = buf[2 * total ..][0..total];
    const ma = pa[anchor..][0..w];
    const mb = pb[anchor..][0..w];
    const mc = pc[anchor..][0..w];
    @memset(buf, 0);
    const rowOf = struct {
        fn f(v: isize, n: usize, b: Border) ?usize {
            return border_mod.map(v, n, b);
        }
    }.f;
    const top: isize = @as(isize, @intCast(y0)) - @as(isize, @intCast(anchor));
    for (0..bs) |j| {
        if (rowOf(top + @as(isize, @intCast(j)), h, o.border)) |sy|
            accumulate(G, A, ma, mb, mc, dx.row(sy - row0), dy.row(sy - row0), true);
    }
    for (y0..y1) |y| {
        if (y > y0) {
            const yi: isize = @intCast(y);
            if (rowOf(yi - @as(isize, @intCast(anchor)) + @as(isize, @intCast(bs)) - 1, h, o.border)) |sy|
                accumulate(G, A, ma, mb, mc, dx.row(sy - row0), dy.row(sy - row0), true);
            if (rowOf(yi - @as(isize, @intCast(anchor)) - 1, h, o.border)) |sy|
                accumulate(G, A, ma, mb, mc, dx.row(sy - row0), dy.row(sy - row0), false);
        }
        // Horizontal border of the column sums (constant border: zeros).
        for (0..anchor) |i| {
            const sx = border_mod.map(@as(isize, @intCast(i)) - @as(isize, @intCast(anchor)), w, o.border);
            pa[i] = if (sx) |s| ma[s] else 0;
            pb[i] = if (sx) |s| mb[s] else 0;
            pc[i] = if (sx) |s| mc[s] else 0;
        }
        for (0..bs - 1 - anchor) |i| {
            const sx = border_mod.map(@intCast(w + i), w, o.border);
            pa[anchor + w + i] = if (sx) |s| ma[s] else 0;
            pb[anchor + w + i] = if (sx) |s| mb[s] else 0;
            pc[anchor + w + i] = if (sx) |s| mc[s] else 0;
        }
        const out = d.row(y);
        var x: usize = 0;
        while (x + N <= w) : (x += N) {
            var va: @Vector(N, A) = pa[x..][0..N].*;
            var vb: @Vector(N, A) = pb[x..][0..N].*;
            var vc: @Vector(N, A) = pc[x..][0..N].*;
            for (1..bs) |i| {
                va += pa[x + i ..][0..N].*;
                vb += pb[x + i ..][0..N].*;
                vc += pc[x + i ..][0..N].*;
            }
            out[x..][0..N].* = respond(kind, N, toF32(N, va, s2), toF32(N, vb, s2), toF32(N, vc, s2), k);
        }
        while (x < w) : (x += 1) {
            var sa: A = 0;
            var sb: A = 0;
            var sc: A = 0;
            for (0..bs) |i| {
                sa += pa[x + i];
                sb += pb[x + i];
                sc += pc[x + i];
            }
            const r = respond(kind, 1, toF32(1, @as(@Vector(1, A), @splat(sa)), s2), toF32(1, @as(@Vector(1, A), @splat(sb)), s2), toF32(1, @as(@Vector(1, A), @splat(sc)), s2), k);
            out[x] = r[0];
        }
    }
}

// ----------------------------------------------------- goodFeaturesToTrack

pub const GfttOptions = struct {
    /// Most corners to return (0 = no limit).
    max_corners: u32 = 0,
    /// Corners weaker than quality_level * (strongest response) are dropped.
    quality_level: f64 = 0.01,
    /// Minimum Euclidean distance between returned corners.
    min_distance: f64 = 1,
    /// Only pixels where the mask is non-zero are considered.
    mask: ?Image(u8) = null,
    block_size: u32 = 3,
    gradient_size: i32 = 3,
    use_harris: bool = false,
    k: f64 = 0.04,
};

/// OpenCV's goodFeaturesToTrack: the strongest corners (min-eigenvalue or
/// Harris response), local maxima in 3x3, at least min_distance apart,
/// strongest first. The caller frees the returned slice with `gpa`.
pub fn goodFeaturesToTrack(exec: Exec, gpa: Allocator, src: anytype, o: GfttOptions) Error![]Point2f {
    if (!(o.quality_level > 0) or o.min_distance < 0) return error.InvalidKernel;
    if (o.mask) |m| if (!m.sameSize(src) or m.channels != 1) return error.SizeMismatch;
    var eig = try Image(f32).init(gpa, src.width, src.height, 1);
    defer eig.deinit(gpa);
    const co: CornerOptions = .{ .block_size = o.block_size, .ksize = o.gradient_size };
    if (o.use_harris) try cornerHarris(exec, gpa, src, eig, co, o.k) else try cornerMinEigenVal(exec, gpa, src, eig, co);
    return selectCorners(gpa, eig, o);
}

const Candidate = struct {
    val: f32,
    idx: u32,

    /// Stronger first; equal responses in reverse raster order (OpenCV
    /// sorts pointers into the response image the same way).
    fn before(_: void, a: Candidate, b: Candidate) bool {
        if (a.val != b.val) return a.val > b.val;
        return a.idx > b.idx;
    }
};

/// Candidates strongest first: a full sort when all of them may be needed,
/// else a binary heap popped on demand (the selection usually stops after
/// a few thousand of possibly hundreds of thousands).
const Ordered = struct {
    items: []Candidate,
    sorted: bool,
    pos: usize = 0,
    len: usize,

    fn init(items: []Candidate, sort_all: bool) Ordered {
        if (sort_all) {
            std.sort.pdq(Candidate, items, {}, Candidate.before);
        } else {
            var i = items.len / 2;
            while (i > 0) {
                i -= 1;
                siftDown(items, i);
            }
        }
        return .{ .items = items, .sorted = sort_all, .len = items.len };
    }

    fn siftDown(h: []Candidate, start: usize) void {
        var i = start;
        while (true) {
            const l = 2 * i + 1;
            if (l >= h.len) return;
            var best = l;
            if (l + 1 < h.len and Candidate.before({}, h[l + 1], h[l])) best = l + 1;
            if (!Candidate.before({}, h[best], h[i])) return;
            std.mem.swap(Candidate, &h[i], &h[best]);
            i = best;
        }
    }

    fn next(self: *Ordered) ?Candidate {
        if (self.sorted) {
            if (self.pos >= self.items.len) return null;
            self.pos += 1;
            return self.items[self.pos - 1];
        }
        if (self.len == 0) return null;
        const top = self.items[0];
        self.len -= 1;
        self.items[0] = self.items[self.len];
        siftDown(self.items[0..self.len], 0);
        return top;
    }
};

/// The selection half of goodFeaturesToTrack, on a response image.
pub fn selectCorners(gpa: Allocator, eig: Image(f32), o: GfttOptions) Error![]Point2f {
    const w: usize = eig.width;
    const h: usize = eig.height;
    // minMaxLoc over the mask (0 when the mask is empty).
    var max_val: f32 = -std.math.inf(f32);
    var any = false;
    for (0..h) |y| {
        const r = eig.row(y);
        if (o.mask) |m| {
            for (r, m.row(y)) |v, mv| if (mv != 0) {
                any = true;
                max_val = @max(max_val, v);
            };
        } else {
            any = true;
            max_val = @max(max_val, maxRow(r));
        }
    }
    if (!any) max_val = 0;
    const th: f32 = @floatCast(@as(f64, max_val) * o.quality_level);

    var cands: std.ArrayList(Candidate) = .empty;
    defer cands.deinit(gpa);
    if (w >= 3 and h >= 3) {
        const N = comptime lanes(f32);
        const V = @Vector(N, f32);
        const thv: V = @splat(th);
        for (1..h - 1) |y| {
            const r0 = eig.row(y - 1);
            const r1 = eig.row(y);
            const r2 = eig.row(y + 1);
            const mrow = if (o.mask) |m| m.row(y) else null;
            var x: usize = 1;
            while (x < w - 1) {
                // Skip runs below the threshold a vector at a time.
                if (x + N <= w - 1) {
                    const v: V = r1[x..][0..N].*;
                    if (!@reduce(.Or, v > thv)) {
                        x += N;
                        continue;
                    }
                }
                const end = @min(x + N, w - 1);
                while (x < end) : (x += 1) {
                    const v = r1[x];
                    if (!(v > th) or v == 0) continue;
                    if (mrow) |m| if (m[x] == 0) continue;
                    const t = struct {
                        inline fn f(n: f32, t_: f32) f32 {
                            return if (n > t_) n else 0;
                        }
                    }.f;
                    if (t(r0[x - 1], th) > v or t(r0[x], th) > v or t(r0[x + 1], th) > v or
                        t(r1[x - 1], th) > v or t(r1[x + 1], th) > v or
                        t(r2[x - 1], th) > v or t(r2[x], th) > v or t(r2[x + 1], th) > v) continue;
                    try cands.append(gpa, .{ .val = v, .idx = @intCast(y * w + x) });
                }
            }
        }
    }
    var order = Ordered.init(cands.items, o.max_corners == 0);

    var out: std.ArrayList(Point2f) = .empty;
    errdefer out.deinit(gpa);
    const limit: usize = if (o.max_corners > 0) o.max_corners else std.math.maxInt(usize);
    if (o.min_distance >= 1) {
        const cell: usize = @intFromFloat(core.sat.rint(o.min_distance));
        const gw = (w + cell - 1) / cell;
        const gh = (h + cell - 1) / cell;
        // Accepted corners per grid cell as linked lists through `next`.
        const head = try gpa.alloc(u32, gw * gh);
        defer gpa.free(head);
        @memset(head, std.math.maxInt(u32));
        var next: std.ArrayList(u32) = .empty;
        defer next.deinit(gpa);
        const md2 = o.min_distance * o.min_distance;
        while (order.next()) |c| {
            if (out.items.len >= limit) break;
            const x = c.idx % w;
            const y = c.idx / w;
            const xc = x / cell;
            const yc = y / cell;
            const fx: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);
            var good = true;
            outer: for (yc -| 1..@min(gh - 1, yc + 1) + 1) |yy| {
                for (xc -| 1..@min(gw - 1, xc + 1) + 1) |xx| {
                    var i = head[yy * gw + xx];
                    while (i != std.math.maxInt(u32)) : (i = next.items[i]) {
                        const ddx = fx - out.items[i].x;
                        const ddy = fy - out.items[i].y;
                        if (@as(f64, ddx * ddx + ddy * ddy) < md2) {
                            good = false;
                            break :outer;
                        }
                    }
                }
            }
            if (!good) continue;
            const id: u32 = @intCast(out.items.len);
            try out.append(gpa, .{ .x = fx, .y = fy });
            try next.append(gpa, head[yc * gw + xc]);
            head[yc * gw + xc] = id;
        }
    } else {
        while (order.next()) |c| {
            if (out.items.len >= limit) break;
            try out.append(gpa, .{ .x = @floatFromInt(c.idx % w), .y = @floatFromInt(c.idx / w) });
        }
    }
    return out.toOwnedSlice(gpa);
}

fn maxRow(r: []const f32) f32 {
    const N = comptime lanes(f32);
    var m: @Vector(N, f32) = @splat(-std.math.inf(f32));
    var x: usize = 0;
    while (x + N <= r.len) : (x += N) m = @max(m, @as(@Vector(N, f32), r[x..][0..N].*));
    var s = @reduce(.Max, m);
    while (x < r.len) : (x += 1) s = @max(s, r[x]);
    return s;
}

// ------------------------------------------------------------------ tests

/// Straightforward reference: f64 Sobel, box sum, eigenvalue formula.
fn refResponse(src: Image(u8), x: usize, y: usize, o: CornerOptions, kind: Kind, k: f64) f64 {
    var kx: [7]i32 = undefined;
    var ky: [7]i32 = undefined;
    const n: usize = if (o.ksize <= 1) 3 else @intCast(o.ksize);
    filter.derivKernel(kx[0..n], 1, if (o.ksize == 1) 3 else o.ksize) catch unreachable;
    filter.derivKernel(ky[0..n], 0, if (o.ksize == 1) 1 else o.ksize) catch unreachable;
    var scale: f64 = @floatFromInt((@as(u32, 1) << @intCast((if (o.ksize > 0) o.ksize else 3) - 1)) * o.block_size);
    if (o.ksize < 0) scale *= 2;
    scale = 1.0 / (scale * 255);
    const bs = o.block_size;
    var a: f64 = 0;
    var b: f64 = 0;
    var c: f64 = 0;
    for (0..bs) |j| for (0..bs) |i| {
        const py = border_mod.map(@as(isize, @intCast(y + j)) - @as(isize, @intCast(bs / 2)), src.height, o.border) orelse continue;
        const px = border_mod.map(@as(isize, @intCast(x + i)) - @as(isize, @intCast(bs / 2)), src.width, o.border) orelse continue;
        var gx: f64 = 0;
        var gy: f64 = 0;
        for (0..n) |v| for (0..n) |u| {
            const sy = border_mod.map(@as(isize, @intCast(py + v)) - @as(isize, @intCast(n / 2)), src.height, o.border) orelse continue;
            const sx = border_mod.map(@as(isize, @intCast(px + u)) - @as(isize, @intCast(n / 2)), src.width, o.border) orelse continue;
            const p: f64 = @floatFromInt(src.at(sx, sy, 0));
            gx += p * @as(f64, @floatFromInt(kx[u] * ky[v]));
            gy += p * @as(f64, @floatFromInt(ky[u] * kx[v]));
        };
        gx *= scale;
        gy *= scale;
        a += gx * gx;
        b += gx * gy;
        c += gy * gy;
    };
    if (kind == .harris) return a * c - b * b - k * (a + c) * (a + c);
    return (a + c) / 2 - @sqrt((a - c) * (a - c) / 4 + b * b);
}

test "corner responses match a direct computation" {
    const gpa = std.testing.allocator;
    for ([_][2]u32{ .{ 23, 17 }, .{ 9, 83 } }) |sz| {
        var src = try Image(u8).init(gpa, sz[0], sz[1], 1);
        defer src.deinit(gpa);
        var seed: u32 = 5;
        for (src.slice(), 0..) |*v, i| {
            seed = seed *% 1103515245 +% 12345;
            v.* = @truncate((seed >> 16) % 64 + (if ((i % sz[0]) > sz[0] / 2) @as(u32, 150) else 0));
        }
        var dst = try Image(f32).initLike(gpa, src);
        defer dst.deinit(gpa);
        for ([_]CornerOptions{ .{}, .{ .block_size = 2 }, .{ .block_size = 5, .ksize = 5 }, .{ .ksize = -1, .border = .replicate }, .{ .block_size = 4, .ksize = 7 }, .{ .ksize = 1, .border = .reflect }, .{ .block_size = 7, .border = .constant } }) |o| {
            inline for (.{ Kind.min_eig, Kind.harris }) |kind| {
                if (kind == .min_eig) try cornerMinEigenVal(.serial, gpa, src, dst, o) else try cornerHarris(.serial, gpa, src, dst, o, 0.04);
                var max: f64 = 0;
                for (dst.slice()) |v| max = @max(max, @abs(v));
                for (0..sz[1]) |y| for (0..sz[0]) |x| {
                    const want = refResponse(src, x, y, o, kind, 0.04);
                    try std.testing.expectApproxEqAbs(want, dst.at(x, y, 0), 1e-5 * max + 1e-12);
                };
            }
        }
    }
}

test "goodFeaturesToTrack finds the corners of a square" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).initFill(gpa, 40, 30, 1, 0);
    defer src.deinit(gpa);
    for (8..20) |y| for (10..30) |x| src.set(x, y, 0, 200);
    const pts = try goodFeaturesToTrack(.serial, gpa, src, .{ .max_corners = 4, .min_distance = 5 });
    defer gpa.free(pts);
    try std.testing.expectEqual(@as(usize, 4), pts.len);
    for (pts) |p| {
        const near_x = @abs(p.x - 10) <= 1.5 or @abs(p.x - 29) <= 1.5;
        const near_y = @abs(p.y - 8) <= 1.5 or @abs(p.y - 19) <= 1.5;
        try std.testing.expect(near_x and near_y);
    }
}
