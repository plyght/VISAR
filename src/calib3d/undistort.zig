//! Lens undistortion: initUndistortRectifyMap, undistort and a bilinear
//! remap, following OpenCV's undistort.dispatch.cpp / undistort.simd.hpp
//! and imgwarp.cpp.
//!
//! OpenCV's undistort builds fixed-point maps (CV_16SC2 + CV_16UC1: source
//! coordinates rounded to 1/32 pixel) and remaps with 15-bit integer
//! bilinear weights and a constant (0) border. VISAR computes the same
//! double-precision map and the same integer interpolation, fused per row
//! so no map is stored, and matches OpenCV's output exactly (a map
//! coordinate can round the other way only when it lies within ~1e-12 of a
//! 1/64-pixel boundary).

const std = @import("std");
const core = @import("../core/core.zig");
const linalg = @import("linalg.zig");
const camera = @import("camera.zig");
const Image = core.Image;
const Exec = core.Exec;

pub const Error = core.Error || error{ InvalidArgument, Unsupported };

const tab_bits = 5;
const tab_size = 1 << tab_bits;

/// The per-pixel mapping from destination pixel to distorted source pixel.
pub const MapParams = struct {
    /// inv(newK * R)
    ir: [9]f64,
    fx: f64,
    fy: f64,
    u0: f64,
    v0: f64,
    k: camera.Dist,

    /// OpenCV's initUndistortRectifyMap setup. `R` and `new_k` may be null
    /// (identity / `K`).
    pub fn init(K: [9]f64, dist: []const f64, R: ?[9]f64, new_k: ?[9]f64) Error!MapParams {
        const ar = new_k orelse K;
        const rr = R orelse [9]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        const ir = linalg.inv3(linalg.mul3(ar, rr)) orelse return error.InvalidArgument;
        return .{ .ir = ir, .fx = K[0], .fy = K[4], .u0 = K[2], .v0 = K[5], .k = try camera.expandDist(dist) };
    }

    /// Source coordinates of destination row `i`, columns [x0, x0 + n),
    /// L at a time.
    inline fn row(p: MapParams, comptime L: usize, i: f64, x0: usize, us: *[L]f64, vs: *[L]f64) void {
        const V = @Vector(L, f64);
        const ir = p.ir;
        var j: V = undefined;
        inline for (0..L) |l| j[l] = @floatFromInt(x0 + l);
        const sp = struct {
            inline fn s(v: f64) V {
                return @splat(v);
            }
        }.s;
        const _x = sp(i * ir[1] + ir[2]) + j * sp(ir[0]);
        const _y = sp(i * ir[4] + ir[5]) + j * sp(ir[3]);
        var x = _x;
        var y = _y;
        // With R = I and no skew the projective row is (0, 0, 1): w = 1.
        if (!(ir[6] == 0 and ir[7] == 0 and ir[8] == 1)) {
            const _w = sp(i * ir[7] + ir[8]) + j * sp(ir[6]);
            const w = sp(1) / _w;
            x = _x * w;
            y = _y * w;
        }
        const k = p.k;
        const x2 = x * x;
        const y2 = y * y;
        const r2 = x2 + y2;
        const _2xy = sp(2) * x * y;
        var kr = sp(1) + ((sp(k[4]) * r2 + sp(k[1])) * r2 + sp(k[0])) * r2;
        if (k[5] != 0 or k[6] != 0 or k[7] != 0) kr /= sp(1) + ((sp(k[7]) * r2 + sp(k[6])) * r2 + sp(k[5])) * r2;
        const xd = x * kr + sp(k[2]) * _2xy + sp(k[3]) * (r2 + sp(2) * x2);
        const yd = y * kr + sp(k[2]) * (r2 + sp(2) * y2) + sp(k[3]) * _2xy;
        us.* = sp(p.fx) * xd + sp(p.u0);
        vs.* = sp(p.fy) * yd + sp(p.v0);
    }
};

/// OpenCV's initUndistortRectifyMap with m1type CV_32FC1: `mapx`/`mapy`
/// (f32, one channel, the destination size).
pub fn initUndistortRectifyMap(exec: Exec, p: MapParams, mapx: Image(f32), mapy: Image(f32)) Error!void {
    if (!mapx.sameShape(mapy) or mapx.channels != 1) return error.SizeMismatch;
    const Ctx = struct { p: MapParams, mx: Image(f32), my: Image(f32) };
    exec.parallelFor(mapx.height, 16, Ctx{ .p = p, .mx = mapx, .my = mapy }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const L = 4;
            for (y0..y1) |y| {
                const ox = c.mx.row(y);
                const oy = c.my.row(y);
                const fy: f64 = @floatFromInt(y);
                var x: usize = 0;
                const w = c.mx.width;
                while (x < w) : (x += L) {
                    var us: [L]f64 = undefined;
                    var vs: [L]f64 = undefined;
                    c.p.row(L, fy, x, &us, &vs);
                    for (0..@min(L, w - x)) |l| {
                        ox[x + l] = @floatCast(us[l]);
                        oy[x + l] = @floatCast(vs[l]);
                    }
                }
            }
        }
    }.run);
}

/// Fixed-point source coordinate (1/32 pixel): OpenCV's saturate_cast<int>
/// (round to nearest even) of a double.
inline fn fix(comptime L: usize, v: @Vector(L, f64)) @Vector(L, i32) {
    const s = v * @as(@Vector(L, f64), @splat(tab_size));
    const lim: @Vector(L, f64) = @splat(2147483647.0);
    const c = @min(@max(s, -lim), lim);
    // Round half to even like cvRound (the default FP rounding mode).
    const r = c + @as(@Vector(L, f64), @splat(6755399441055744.0)) - @as(@Vector(L, f64), @splat(6755399441055744.0));
    return @intFromFloat(r);
}

/// Bilinear interpolation of one destination pixel from fixed-point
/// source coordinates (iu, iv in 1/32 pixel) with a constant border:
/// OpenCV's remapBilinear with 15-bit weights.
inline fn sample(comptime cn: usize, src: Image(u8), iu: i32, iv: i32, border: [4]u8, out: *[cn]u8) void {
    const sx = iu >> tab_bits;
    const sy = iv >> tab_bits;
    const a: i32 = iu & (tab_size - 1);
    const b: i32 = iv & (tab_size - 1);
    const w = [4]i32{ (tab_size - a) * (tab_size - b), a * (tab_size - b), (tab_size - a) * b, a * b };
    const W: i32 = @intCast(src.width);
    const H: i32 = @intCast(src.height);
    if (sx >= 0 and sx < W - 1 and sy >= 0 and sy < H - 1) {
        const p0 = src.data + @as(usize, @intCast(sy)) * src.stride + @as(usize, @intCast(sx)) * cn;
        const p1 = p0 + src.stride;
        inline for (0..cn) |ch| {
            const s = @as(i32, p0[ch]) * w[0] + @as(i32, p0[ch + cn]) * w[1] + @as(i32, p1[ch]) * w[2] + @as(i32, p1[ch + cn]) * w[3];
            out[ch] = @intCast((s + 512) >> 10);
        }
        return;
    }
    if (sx >= W or sx + 1 < 0 or sy >= H or sy + 1 < 0) {
        inline for (0..cn) |ch| out[ch] = border[ch];
        return;
    }
    inline for (0..cn) |ch| {
        var s: i32 = 0;
        inline for (0..4) |q| {
            const px = sx + @as(i32, q & 1);
            const py = sy + @as(i32, q >> 1);
            const v: i32 = if (px >= 0 and px < W and py >= 0 and py < H)
                src.data[@as(usize, @intCast(py)) * src.stride + @as(usize, @intCast(px)) * cn + ch]
            else
                border[ch];
            s += v * w[q];
        }
        out[ch] = @intCast((s + 512) >> 10);
    }
}

/// 16 bilinear u8 pixels whose taps are all inside the source, with 1/32
/// pixel fractions `a` (x) and `b` (y): the 15-bit fixed-point blend
/// factored as rows first, (h0 (32 - b) + h1 b + 512) >> 10 with
/// h = p0 (32 - a) + p1 a, which is the same integer.
inline fn linear16(comptime cn: usize, s: Image(u8), sx: @Vector(16, i32), sy: @Vector(16, i32), a: @Vector(16, i32), b: @Vector(16, i32), d: []u8) void {
    const L = 16;
    const U = @Vector(L, u32);
    const ixv: U = @bitCast(sx);
    const iyv: U = @bitCast(sy);
    const off: [L]u32 = iyv * @as(U, @splat(@intCast(s.stride))) + ixv * @as(U, @splat(cn));
    var w0: [L]u64 = undefined;
    var w1: [L]u64 = undefined;
    inline for (0..L) |i| {
        const q = s.data + off[i];
        w0[i] = std.mem.readInt(u64, q[0..8], .little);
        w1[i] = std.mem.readInt(u64, q[s.stride..][0..8], .little);
    }
    const r0: @Vector(L, u64) = w0;
    const r1: @Vector(L, u64) = w1;
    const words = [4]U{ @truncate(r0), @truncate(r0 >> @splat(32)), @truncate(r1), @truncate(r1 >> @splat(32)) };
    const au: U = @bitCast(a);
    const bu: U = @bitCast(b);
    const ia = @as(U, @splat(tab_size)) - au;
    const ib = @as(U, @splat(tab_size)) - bu;
    var outs: [cn]@Vector(L, u8) = undefined;
    inline for (0..cn) |ch| {
        const h0 = byteU(words[0..2], ch) * ia + byteU(words[0..2], cn + ch) * au;
        const h1 = byteU(words[2..4], ch) * ia + byteU(words[2..4], cn + ch) * au;
        outs[ch] = @truncate((h0 * ib + h1 * bu + @as(U, @splat(512))) >> @splat(10));
    }
    if (cn == 1) {
        d[0..L].* = outs[0];
    } else {
        core.vec.store(cn, outs, d.ptr);
    }
}

/// Byte `k` (0..7) of the 64-bit words split into (low, high) halves.
inline fn byteU(w: *const [2]@Vector(16, u32), comptime k: usize) @Vector(16, u32) {
    const v = w[k / 4];
    const sh: u5 = (k % 4) * 8;
    return if (sh == 24) v >> @splat(24) else (v >> @splat(sh)) & @as(@Vector(16, u32), @splat(0xff));
}

/// OpenCV's undistort (u8, 1-4 channels): `dst` has `src`'s shape.
/// `new_k` defaults to `K`.
pub fn undistort(exec: Exec, src: Image(u8), dst: Image(u8), K: [9]f64, dist: []const f64, new_k: ?[9]f64) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (src.channels < 1 or src.channels > 4) return error.Unsupported;
    const k = try camera.expandDist(dist);
    const ar0 = new_k orelse K;
    const Ctx = struct { src: Image(u8), dst: Image(u8), K: [9]f64, ar: [9]f64, k: camera.Dist, stripe: usize };
    // OpenCV maps stripes of (4096 / width) rows, shifting the new camera's
    // principal point by the stripe's first row.
    const stripe = @min(@max(1, 4096 / @max(src.width, 1)), src.height);
    const nstripes = (src.height + stripe - 1) / stripe;
    const ctx = Ctx{ .src = src, .dst = dst, .K = K, .ar = ar0, .k = k, .stripe = stripe };
    exec.parallelFor(nstripes, 4, ctx, struct {
        fn run(c: Ctx, s0: usize, s1: usize) void {
            switch (c.src.channels) {
                inline 1, 2, 3, 4 => |cn| runCn(cn, c, s0, s1),
                else => unreachable,
            }
        }

        fn runCn(comptime cn: usize, c: Ctx, s0: usize, s1: usize) void {
            const L = 16;
            const border: [4]u8 = @splat(0);
            const W: i32 = @intCast(c.src.width);
            const H: i32 = @intCast(c.src.height);
            // Taps are read as 8-byte words: the second row's word must end
            // inside the image.
            const total: i64 = @intCast((c.src.height - 1) * c.src.stride + c.src.rowLen());
            for (s0..s1) |s| {
                const y0 = s * c.stripe;
                const y1 = @min(y0 + c.stripe, c.src.height);
                var ar = c.ar;
                ar[5] = c.ar[5] - @as(f64, @floatFromInt(y0));
                const ir = linalg.inv3(ar) orelse return;
                const p: MapParams = .{ .ir = ir, .fx = c.K[0], .fy = c.K[4], .u0 = c.K[2], .v0 = c.K[5], .k = c.k };
                for (y0..y1) |y| {
                    const d = c.dst.row(y);
                    const fi: f64 = @floatFromInt(y - y0);
                    var x: usize = 0;
                    const w = c.dst.width;
                    while (x < w) : (x += L) {
                        var us: [L]f64 = undefined;
                        var vs: [L]f64 = undefined;
                        p.row(L, fi, x, &us, &vs);
                        const iu = fix(L, us);
                        const iv = fix(L, vs);
                        const n = @min(L, w - x);
                        if (cn != 2 and n == L) {
                            const I = @Vector(L, i32);
                            const sx = iu >> @splat(tab_bits);
                            const sy = iv >> @splat(tab_bits);
                            const off1 = (@as(@Vector(L, i64), sy) + @as(@Vector(L, i64), @splat(1))) * @as(@Vector(L, i64), @splat(@intCast(c.src.stride))) + @as(@Vector(L, i64), sx) * @as(@Vector(L, i64), @splat(cn));
                            const zero: I = @splat(0);
                            const ok = (sx >= zero) & (sy >= zero) & (sx < @as(I, @splat(W - 1))) & (sy < @as(I, @splat(H - 1))) &
                                (off1 + @as(@Vector(L, i64), @splat(8)) <= @as(@Vector(L, i64), @splat(total)));
                            if (@reduce(.And, ok)) {
                                linear16(cn, c.src, sx, sy, iu & @as(I, @splat(tab_size - 1)), iv & @as(I, @splat(tab_size - 1)), d[x * cn ..]);
                                continue;
                            }
                        }
                        const ia: [L]i32 = iu;
                        const ib: [L]i32 = iv;
                        for (0..n) |l| sample(cn, c.src, ia[l], ib[l], border, d[(x + l) * cn ..][0..cn]);
                    }
                }
            }
        }
    }.run);
}

/// Bilinear remap of a u8 image (1-4 channels) through f32 maps with a
/// constant border, as OpenCV 5's remap(INTER_LINEAR, BORDER_CONSTANT) with
/// CV_32FC1 maps: f32 taps blended by a chain of fused multiply-adds and
/// rounded half to even (the float warp kernels OpenCV shares with
/// warpAffine; its fixed-point path is only used for CV_16SC2 maps, as in
/// `undistort`).
pub fn remap(exec: Exec, src: Image(u8), dst: Image(u8), mapx: Image(f32), mapy: Image(f32), border_value: [4]u8) Error!void {
    if (!dst.sameSize(mapx) or !mapx.sameShape(mapy) or mapx.channels != 1) return error.SizeMismatch;
    if (src.channels != dst.channels or src.channels < 1 or src.channels > 4) return error.Unsupported;
    const Ctx = struct { src: Image(u8), dst: Image(u8), mx: Image(f32), my: Image(f32), bv: [4]u8 };
    exec.parallelFor(dst.height, 8, Ctx{ .src = src, .dst = dst, .mx = mapx, .my = mapy, .bv = border_value }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.src.channels) {
                inline 1, 2, 3, 4 => |cn| runCn(cn, c, y0, y1),
                else => unreachable,
            }
        }

        fn runCn(comptime cn: usize, c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                const d = c.dst.row(y);
                const mx = c.mx.row(y);
                const my = c.my.row(y);
                for (0..c.dst.width) |x| linearF32(cn, c.src, mx[x], my[x], c.bv, d[x * cn ..][0..cn]);
            }
        }
    }.run);
}

inline fn linearF32(comptime cn: usize, src: Image(u8), sx: f32, sy: f32, bv: [4]u8, out: *[cn]u8) void {
    const fx = @floor(sx);
    const fy = @floor(sy);
    const ax = sx - fx;
    const ay = sy - fy;
    if (!(fx >= -1 and fy >= -1 and fx < @as(f32, @floatFromInt(src.width)) and fy < @as(f32, @floatFromInt(src.height)))) {
        inline for (0..cn) |ch| out[ch] = bv[ch];
        return;
    }
    const ix: isize = @intFromFloat(fx);
    const iy: isize = @intFromFloat(fy);
    const W: isize = @intCast(src.width);
    const H: isize = @intCast(src.height);
    inline for (0..cn) |ch| {
        var p: [4]f32 = undefined;
        inline for (0..4) |q| {
            const px = ix + (q & 1);
            const py = iy + (q >> 1);
            p[q] = @floatFromInt(if (px >= 0 and px < W and py >= 0 and py < H)
                src.data[@as(usize, @intCast(py)) * src.stride + @as(usize, @intCast(px)) * cn + ch]
            else
                bv[ch]);
        }
        const v0 = @mulAdd(f32, ax, p[1] - p[0], p[0]);
        const v1 = @mulAdd(f32, ax, p[3] - p[2], p[2]);
        const v = @mulAdd(f32, ay, v1 - v0, v0);
        // Round half to even; v is within [0, 255].
        out[ch] = @intFromFloat(@min(@max(v + 12582912.0 - 12582912.0, 0), 255));
    }
}

test "undistort with zero distortion is the identity" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 37, 23, 3);
    defer src.deinit(gpa);
    for (src.slice(), 0..) |*v, i| v.* = @truncate(i * 7);
    var dst = try Image(u8).init(gpa, 37, 23, 3);
    defer dst.deinit(gpa);
    const K = [9]f64{ 40, 0, 18, 0, 40, 11, 0, 0, 1 };
    try undistort(.serial, src, dst, K, &.{}, null);
    // The last row/column interpolate with the border.
    for (0..22) |y| try std.testing.expectEqualSlices(u8, src.row(y)[0 .. 36 * 3], dst.row(y)[0 .. 36 * 3]);
}
