//! Planar transforms: findHomography (all points, RANSAC, LMeDS, then
//! Levenberg-Marquardt refinement), getPerspectiveTransform,
//! getAffineTransform, perspectiveTransform, estimateAffine2D and
//! estimateAffinePartial2D, following OpenCV's fundam.cpp, ptsetreg.cpp
//! and imgwarp.cpp.
//!
//! Points are converted to f32 as OpenCV does; the normalised DLT solves
//! its 9x9 normal equations with OpenCV's Jacobi eigen solver, and RANSAC
//! draws OpenCV's samples (see ransac.zig), so on the same input VISAR
//! returns the same inlier mask and a matrix that agrees to rounding.

const std = @import("std");
const linalg = @import("linalg.zig");
const ransac = @import("ransac.zig");
const Allocator = std.mem.Allocator;

pub const Point2f = @import("../imgproc/shape.zig").Point2f;

pub const Method = enum(u8) {
    /// Least squares over all points.
    all = 0,
    lmeds = 4,
    ransac = 8,
};

pub const Error = Allocator.Error || error{InvalidArgument};

// --------------------------------------------- getPerspective/AffineTransform

/// OpenCV's getPerspectiveTransform (DECOMP_LU). Null when singular.
pub fn getPerspectiveTransform(src: [4]Point2f, dst: [4]Point2f) ?[9]f64 {
    var a: [64]f64 = @splat(0);
    var b: [8]f64 = undefined;
    for (0..4) |i| {
        const r0 = a[i * 8 ..][0..8];
        const r1 = a[(i + 4) * 8 ..][0..8];
        r0[0] = src[i].x;
        r1[3] = src[i].x;
        r0[1] = src[i].y;
        r1[4] = src[i].y;
        r0[2] = 1;
        r1[5] = 1;
        // Products in f32, as OpenCV (Point2f * Point2f).
        r0[6] = -src[i].x * dst[i].x;
        r0[7] = -src[i].y * dst[i].x;
        r1[6] = -src[i].x * dst[i].y;
        r1[7] = -src[i].y * dst[i].y;
        b[i] = dst[i].x;
        b[i + 4] = dst[i].y;
    }
    if (!linalg.solveLU(&a, 8, &b, 1)) return null;
    return .{ b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1 };
}

/// OpenCV's getAffineTransform (DECOMP_LU). Null when singular.
pub fn getAffineTransform(src: [3]Point2f, dst: [3]Point2f) ?[6]f64 {
    var a: [36]f64 = @splat(0);
    var b: [6]f64 = undefined;
    for (0..3) |i| {
        const j = i * 12;
        const k = i * 12 + 6;
        a[j] = src[i].x;
        a[k + 3] = src[i].x;
        a[j + 1] = src[i].y;
        a[k + 4] = src[i].y;
        a[j + 2] = 1;
        a[k + 5] = 1;
        b[i * 2] = dst[i].x;
        b[i * 2 + 1] = dst[i].y;
    }
    if (!linalg.solveLU(&a, 6, &b, 1)) return null;
    return b;
}

/// OpenCV's perspectiveTransform for 2-D f32 points (`src` and `dst` may
/// alias): w below FLT_EPSILON maps to (0, 0).
pub fn perspectiveTransform(src: []const Point2f, dst: []Point2f, m: [9]f64) void {
    const eps: f64 = std.math.floatEps(f32);
    for (src, dst) |p, *o| {
        const x: f64 = p.x;
        const y: f64 = p.y;
        var w: f32 = @floatCast(x * m[6] + y * m[7] + m[8]);
        if (@abs(w) > eps) {
            w = @floatCast(1.0 / @as(f64, w));
            const wd: f64 = w;
            o.* = .{ .x = @floatCast((x * m[0] + y * m[1] + m[2]) * wd), .y = @floatCast((x * m[3] + y * m[4] + m[5]) * wd) };
        } else o.* = .{ .x = 0, .y = 0 };
    }
}

/// perspectiveTransform for f64 points (x, y pairs).
pub fn perspectiveTransform64(src: []const [2]f64, dst: [][2]f64, m: [9]f64) void {
    const eps: f64 = std.math.floatEps(f64);
    for (src, dst) |p, *o| {
        const x = p[0];
        const y = p[1];
        var w = x * m[6] + y * m[7] + m[8];
        if (@abs(w) > eps) {
            w = 1.0 / w;
            o.* = .{ (x * m[0] + y * m[1] + m[2]) * w, (x * m[3] + y * m[4] + m[5]) * w };
        } else o.* = .{ 0, 0 };
    }
}

// ------------------------------------------------------------ homography

/// The normalised DLT over the points `idx` selects (or all when null),
/// as HomographyEstimatorCallback::runKernel. Null when degenerate.
fn homographyDlt(src: []const Point2f, dst: []const Point2f, idx: ?[]const u32) ?[9]f64 {
    const count = if (idx) |ii| ii.len else src.len;
    var cm = [2]f64{ 0, 0 };
    var cM = [2]f64{ 0, 0 };
    var sm = [2]f64{ 0, 0 };
    var sM = [2]f64{ 0, 0 };
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        cm[0] += dst[i].x;
        cm[1] += dst[i].y;
        cM[0] += src[i].x;
        cM[1] += src[i].y;
    }
    const n: f64 = @floatFromInt(count);
    cm[0] /= n;
    cm[1] /= n;
    cM[0] /= n;
    cM[1] /= n;
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        sm[0] += @abs(dst[i].x - cm[0]);
        sm[1] += @abs(dst[i].y - cm[1]);
        sM[0] += @abs(src[i].x - cM[0]);
        sM[1] += @abs(src[i].y - cM[1]);
    }
    const de = std.math.floatEps(f64);
    if (@abs(sm[0]) < de or @abs(sm[1]) < de or @abs(sM[0]) < de or @abs(sM[1]) < de) return null;
    sm = .{ n / sm[0], n / sm[1] };
    sM = .{ n / sM[0], n / sM[1] };
    const inv_hnorm = [9]f64{ 1.0 / sm[0], 0, cm[0], 0, 1.0 / sm[1], cm[1], 0, 0, 1 };
    const hnorm2 = [9]f64{ sM[0], 0, -cM[0] * sM[0], 0, sM[1], -cM[1] * sM[1], 0, 0, 1 };
    // A minimal sample: the 8x8 system with h22 = 1 has the exact solution
    // the eigenvector of LᵀL also gives (up to scale and rounding), at a
    // fraction of the cost of the 9x9 Jacobi sweep.
    if (count == 4) minimal: {
        var a: [64]f64 = undefined;
        var b: [8]f64 = undefined;
        for (0..4) |k| {
            const i = idx.?[k];
            const x = (dst[i].x - cm[0]) * sm[0];
            const y = (dst[i].y - cm[1]) * sm[1];
            const X = (src[i].x - cM[0]) * sM[0];
            const Y = (src[i].y - cM[1]) * sM[1];
            a[k * 16 ..][0..16].* = .{ X, Y, 1, 0, 0, 0, -x * X, -x * Y, 0, 0, 0, X, Y, 1, -y * X, -y * Y };
            b[2 * k] = x;
            b[2 * k + 1] = y;
        }
        if (!linalg.solveLU(&a, 8, &b, 1)) break :minimal;
        const h0 = [9]f64{ b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 1 };
        var h = linalg.mul3(linalg.mul3(inv_hnorm, h0), hnorm2);
        if (!(@abs(h[8]) > 0)) break :minimal;
        const s = 1.0 / h[8];
        for (&h) |*e| e.* *= s;
        return h;
    }
    var ltl: [81]f64 = @splat(0);
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        const x = (dst[i].x - cm[0]) * sm[0];
        const y = (dst[i].y - cm[1]) * sm[1];
        const X = (src[i].x - cM[0]) * sM[0];
        const Y = (src[i].y - cM[1]) * sM[1];
        const lx = [9]f64{ X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
        const ly = [9]f64{ 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
        for (0..9) |j| for (j..9) |c| {
            ltl[j * 9 + c] += lx[j] * lx[c] + ly[j] * ly[c];
        };
    }
    for (0..9) |j| for (0..j) |c| {
        ltl[j * 9 + c] = ltl[c * 9 + j];
    };
    var w: [9]f64 = undefined;
    var v: [81]f64 = undefined;
    linalg.eigenSym(&ltl, 9, &w, &v);
    const h0 = v[72..81].*;
    const ht = linalg.mul3(inv_hnorm, h0);
    var h = linalg.mul3(ht, hnorm2);
    const s = 1.0 / h[8];
    for (&h) |*e| e.* *= s;
    return h;
}

/// Squared reprojection error per point in f32, as
/// HomographyEstimatorCallback::computeError.
fn homographyErrors(src: []const Point2f, dst: []const Point2f, h: [9]f64, err: []f32) void {
    var hf: [9]f32 = undefined;
    for (&hf, h) |*o, e| o.* = @floatCast(e);
    const L = 8;
    const V = @Vector(L, f32);
    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        var mx: V = undefined;
        var my: V = undefined;
        var tx: V = undefined;
        var ty: V = undefined;
        inline for (0..L) |k| {
            mx[k] = src[i + k].x;
            my[k] = src[i + k].y;
            tx[k] = dst[i + k].x;
            ty[k] = dst[i + k].y;
        }
        const one: V = @splat(1);
        const ww = one / (@as(V, @splat(hf[6])) * mx + @as(V, @splat(hf[7])) * my + one);
        const dx = (@as(V, @splat(hf[0])) * mx + @as(V, @splat(hf[1])) * my + @as(V, @splat(hf[2]))) * ww - tx;
        const dy = (@as(V, @splat(hf[3])) * mx + @as(V, @splat(hf[4])) * my + @as(V, @splat(hf[5]))) * ww - ty;
        err[i..][0..L].* = dx * dx + dy * dy;
    }
    while (i < src.len) : (i += 1) {
        const M = src[i];
        const ww = 1.0 / (hf[6] * M.x + hf[7] * M.y + 1.0);
        const dx = (hf[0] * M.x + hf[1] * M.y + hf[2]) * ww - dst[i].x;
        const dy = (hf[3] * M.x + hf[4] * M.y + hf[5]) * ww - dst[i].y;
        err[i] = dx * dx + dy * dy;
    }
}

const HomographyCb = struct {
    src: []const Point2f,
    dst: []const Point2f,
    pub const Model = [9]f64;

    pub fn fit(c: *const HomographyCb, idx: []const u32, out: []Model) usize {
        out[0] = homographyDlt(c.src, c.dst, idx) orelse return 0;
        return 1;
    }

    pub fn check(c: *const HomographyCb, idx: []const u32) bool {
        var s: [4]Point2f = undefined;
        var d: [4]Point2f = undefined;
        for (idx, 0..) |i, k| {
            s[k] = c.src[i];
            d[k] = c.dst[i];
        }
        const n = idx.len;
        if (ransac.haveCollinear(&s, n) or ransac.haveCollinear(&d, n)) return false;
        if (n == 4) {
            const tt = [4][3]usize{ .{ 0, 1, 2 }, .{ 1, 2, 3 }, .{ 0, 2, 3 }, .{ 0, 1, 3 } };
            var negative: usize = 0;
            for (tt) |t| {
                const a = [9]f64{ s[t[0]].x, s[t[0]].y, 1, s[t[1]].x, s[t[1]].y, 1, s[t[2]].x, s[t[2]].y, 1 };
                const b = [9]f64{ d[t[0]].x, d[t[0]].y, 1, d[t[1]].x, d[t[1]].y, 1, d[t[2]].x, d[t[2]].y, 1 };
                negative += @intFromBool(linalg.det3(a) * linalg.det3(b) < 0);
            }
            if (negative != 0 and negative != 4) return false;
        }
        return true;
    }

    pub fn errors(c: *const HomographyCb, m: Model, err: []f32) void {
        homographyErrors(c.src, c.dst, m, err);
    }
};

/// LM refinement callback over the 8 free entries (h22 = 1), as
/// HomographyRefineCallback.
const HomographyRefine = struct {
    src: []const Point2f,
    dst: []const Point2f,

    /// Sums over the points, L at a time: the residual, and (with `jac`)
    /// the distinct entries of JᵀJ and Jᵀr. With u = (X, Y, 1) w, a point's
    /// Jacobian rows are [u, 0, -xi u01] and [0, u, -yi u01].
    fn Acc(comptime L: usize) type {
        const V = @Vector(L, f64);
        return struct {
            s: V = @splat(0),
            mx: V = @splat(0),
            uu: [6]V = @splat(@splat(0)),
            xu: [6]V = @splat(@splat(0)),
            yu: [6]V = @splat(@splat(0)),
            ru: [3]V = @splat(@splat(0)),
            ue: [6]V = @splat(@splat(0)),
            re: [2]V = @splat(@splat(0)),

            inline fn add(acc: *@This(), comptime jac: bool, h: []const f64, X: V, Y: V, tx: V, ty: V) void {
                const one: V = @splat(1);
                var ww = @as(V, @splat(h[6])) * X + @as(V, @splat(h[7])) * Y + one;
                ww = @select(f64, @abs(ww) > @as(V, @splat(std.math.floatEps(f64))), one / ww, @as(V, @splat(0)));
                const xi = (@as(V, @splat(h[0])) * X + @as(V, @splat(h[1])) * Y + @as(V, @splat(h[2]))) * ww;
                const yi = (@as(V, @splat(h[3])) * X + @as(V, @splat(h[4])) * Y + @as(V, @splat(h[5]))) * ww;
                const ex = xi - tx;
                const ey = yi - ty;
                acc.s += ex * ex + ey * ey;
                acc.mx = @max(acc.mx, @max(@abs(ex), @abs(ey)));
                if (!jac) return;
                const u = [3]V{ X * ww, Y * ww, ww };
                const pairs = [6][2]usize{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 1, 1 }, .{ 1, 2 }, .{ 2, 2 } };
                inline for (pairs, 0..) |pq, k| acc.uu[k] += u[pq[0]] * u[pq[1]];
                // u_i * u_j for i in 0..3, j in 0..2, weighted by xi / yi.
                inline for (0..3) |i| inline for (0..2) |j| {
                    acc.xu[i * 2 + j] += xi * u[i] * u[j];
                    acc.yu[i * 2 + j] += yi * u[i] * u[j];
                };
                const r2 = xi * xi + yi * yi;
                acc.ru[0] += r2 * u[0] * u[0];
                acc.ru[1] += r2 * u[0] * u[1];
                acc.ru[2] += r2 * u[1] * u[1];
                inline for (0..3) |i| {
                    acc.ue[i] += u[i] * ex;
                    acc.ue[3 + i] += u[i] * ey;
                }
                const q = xi * ex + yi * ey;
                acc.re[0] += q * u[0];
                acc.re[1] += q * u[1];
            }
        };
    }

    fn eval(c: HomographyRefine, h: []const f64, a: ?[]f64, v: ?[]f64) [2]f64 {
        return if (a != null) c.evalImpl(true, h, a, v) else c.evalImpl(false, h, null, null);
    }

    fn evalImpl(c: HomographyRefine, comptime jac: bool, h: []const f64, a: ?[]f64, v: ?[]f64) [2]f64 {
        const L = 4;
        const V = @Vector(L, f64);
        var acc: Acc(L) = .{};
        var i: usize = 0;
        const n = c.src.len;
        while (i + L <= n) : (i += L) {
            var X: V = undefined;
            var Y: V = undefined;
            var tx: V = undefined;
            var ty: V = undefined;
            inline for (0..L) |k| {
                X[k] = c.src[i + k].x;
                Y[k] = c.src[i + k].y;
                tx[k] = c.dst[i + k].x;
                ty[k] = c.dst[i + k].y;
            }
            acc.add(jac, h, X, Y, tx, ty);
        }
        var tail: Acc(1) = .{};
        while (i < n) : (i += 1) {
            tail.add(jac, h, .{c.src[i].x}, .{c.src[i].y}, .{c.dst[i].x}, .{c.dst[i].y});
        }
        const S = struct {
            fn sum(x: V, t: @Vector(1, f64)) f64 {
                return @reduce(.Add, x) + t[0];
            }
        };
        const s = S.sum(acc.s, tail.s);
        const mx = @max(@reduce(.Max, acc.mx), tail.mx[0]);
        if (jac) {
            const aa = a.?;
            const vv = v.?;
            var uu: [6]f64 = undefined;
            for (&uu, 0..) |*o, k| o.* = S.sum(acc.uu[k], tail.uu[k]);
            var xu: [6]f64 = undefined;
            var yu: [6]f64 = undefined;
            for (0..6) |k| {
                xu[k] = S.sum(acc.xu[k], tail.xu[k]);
                yu[k] = S.sum(acc.yu[k], tail.yu[k]);
            }
            const U = [3][3]f64{ .{ uu[0], uu[1], uu[2] }, .{ uu[1], uu[3], uu[4] }, .{ uu[2], uu[4], uu[5] } };
            @memset(aa[0..64], 0);
            for (0..3) |r| for (0..3) |q| {
                aa[r * 8 + q] = U[r][q];
                aa[(r + 3) * 8 + q + 3] = U[r][q];
            };
            for (0..3) |r| for (0..2) |q| {
                aa[r * 8 + 6 + q] = -xu[r * 2 + q];
                aa[(r + 3) * 8 + 6 + q] = -yu[r * 2 + q];
            };
            aa[6 * 8 + 6] = S.sum(acc.ru[0], tail.ru[0]);
            aa[6 * 8 + 7] = S.sum(acc.ru[1], tail.ru[1]);
            aa[7 * 8 + 7] = S.sum(acc.ru[2], tail.ru[2]);
            for (0..8) |r| for (0..r) |q| {
                aa[r * 8 + q] = aa[q * 8 + r];
            };
            for (0..6) |k| vv[k] = S.sum(acc.ue[k], tail.ue[k]);
            vv[6] = -S.sum(acc.re[0], tail.re[0]);
            vv[7] = -S.sum(acc.re[1], tail.re[1]);
        }
        return .{ s, mx };
    }

    pub fn normal(c: HomographyRefine, x: []const f64, a: []f64, v: []f64) [2]f64 {
        return c.eval(x, a, v);
    }

    pub fn cost(c: HomographyRefine, x: []const f64) [2]f64 {
        return c.eval(x, null, null);
    }
};

pub const HomographyOptions = struct {
    method: Method = .all,
    /// Maximum reprojection error (pixels) of an inlier (RANSAC); <= 0
    /// means OpenCV's default of 3.
    ransac_threshold: f64 = 3,
    max_iters: usize = 2000,
    confidence: f64 = 0.995,
    /// RANSAC/LMeDS sampling seed; the default reproduces OpenCV's draws.
    seed: u64 = ransac.default_seed,
};

/// OpenCV's findHomography. `mask` (optional, len = src.len) receives the
/// inliers. Returns null when no homography was found (OpenCV returns an
/// empty matrix).
pub fn findHomography(gpa: Allocator, src_in: []const Point2f, dst_in: []const Point2f, mask: ?[]u8, o: HomographyOptions) Error!?[9]f64 {
    const n = src_in.len;
    if (dst_in.len != n) return error.InvalidArgument;
    if (n < 4) return error.InvalidArgument;
    const buf = try gpa.alloc(Point2f, 2 * n);
    defer gpa.free(buf);
    const src = buf[0..n];
    const dst = buf[n..];
    @memcpy(src, src_in);
    @memcpy(dst, dst_in);
    const tmask = try gpa.alloc(u8, 2 * n);
    defer gpa.free(tmask);
    const err = try gpa.alloc(f32, n);
    defer gpa.free(err);
    const best_mask = tmask[0..n];
    const scratch: ransac.Scratch = .{ .err = err, .mask = tmask[n..] };
    const thr = if (o.ransac_threshold <= 0) 3 else o.ransac_threshold;
    var cb: HomographyCb = .{ .src = src, .dst = dst };
    var h: ?[9]f64 = null;
    if (o.method == .all or n == 4) {
        @memset(best_mask, 1);
        h = homographyDlt(src, dst, null);
    } else if (o.method == .ransac) {
        h = ransac.ransac(&cb, @intCast(n), 4, thr, o.confidence, o.max_iters, o.seed, best_mask, scratch);
    } else {
        h = ransac.lmeds(&cb, @intCast(n), 4, o.confidence, o.max_iters, o.seed, best_mask, scratch);
    }
    if (h != null and n > 4) {
        var k: usize = 0;
        for (0..n) |i| {
            if (best_mask[i] != 0) {
                src[k] = src[i];
                dst[k] = dst[i];
                k += 1;
            }
        }
        if (k > 0) {
            if (o.method != .all) {
                if (homographyDlt(src[0..k], dst[0..k], null)) |hh| h = hh;
            }
            var x: [8]f64 = h.?[0..8].*;
            _ = linalg.lmSolve(8, &x, 10, HomographyRefine{ .src = src[0..k], .dst = dst[0..k] });
            h.?[0..8].* = x;
        }
    }
    if (h != null) {
        // OpenCV 5 reports the inliers of the refined matrix at the
        // reprojection threshold, whatever the method (LMeDS included).
        homographyErrors(src_in, dst_in, h.?, err);
        const t: f32 = @floatCast(thr * thr);
        for (err, best_mask) |e, *k| k.* = @intFromBool(e <= t);
    }
    if (mask) |m| {
        if (h != null) @memcpy(m[0..n], best_mask) else @memset(m[0..n], 0);
    }
    return h;
}

// ------------------------------------------------------------- affine 2-D

const AffineCb = struct {
    src: []const Point2f,
    dst: []const Point2f,
    pub const Model = [6]f64;

    pub fn fit(c: *const AffineCb, idx: []const u32, out: []Model) usize {
        const f = [3]Point2f{ c.src[idx[0]], c.src[idx[1]], c.src[idx[2]] };
        const t = [3]Point2f{ c.dst[idx[0]], c.dst[idx[1]], c.dst[idx[2]] };
        const x1: f64 = f[0].x;
        const y1: f64 = f[0].y;
        const x2: f64 = f[1].x;
        const y2: f64 = f[1].y;
        const x3: f64 = f[2].x;
        const y3: f64 = f[2].y;
        const X1: f64 = t[0].x;
        const Y1: f64 = t[0].y;
        const X2: f64 = t[1].x;
        const Y2: f64 = t[1].y;
        const X3: f64 = t[2].x;
        const Y3: f64 = t[2].y;
        const d = 1.0 / (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
        out[0] = .{
            d * (X1 * (y2 - y3) + X2 * (y3 - y1) + X3 * (y1 - y2)),
            d * (X1 * (x3 - x2) + X2 * (x1 - x3) + X3 * (x2 - x1)),
            d * (X1 * (x2 * y3 - x3 * y2) + X2 * (x3 * y1 - x1 * y3) + X3 * (x1 * y2 - x2 * y1)),
            d * (Y1 * (y2 - y3) + Y2 * (y3 - y1) + Y3 * (y1 - y2)),
            d * (Y1 * (x3 - x2) + Y2 * (x1 - x3) + Y3 * (x2 - x1)),
            d * (Y1 * (x2 * y3 - x3 * y2) + Y2 * (x3 * y1 - x1 * y3) + Y3 * (x1 * y2 - x2 * y1)),
        };
        return 1;
    }

    pub fn check(c: *const AffineCb, idx: []const u32) bool {
        var s: [3]Point2f = undefined;
        var d: [3]Point2f = undefined;
        for (idx, 0..) |i, k| {
            s[k] = c.src[i];
            d[k] = c.dst[i];
        }
        return !ransac.haveCollinear(&s, idx.len) and !ransac.haveCollinear(&d, idx.len);
    }

    pub fn errors(c: *const AffineCb, m: Model, err: []f32) void {
        affineErrors(c.src, c.dst, m, err);
    }
};

fn affineErrors(src: []const Point2f, dst: []const Point2f, m: [6]f64, err: []f32) void {
    var f: [6]f32 = undefined;
    for (&f, m) |*o, e| o.* = @floatCast(e);
    const L = 8;
    const V = @Vector(L, f32);
    var i: usize = 0;
    while (i + L <= src.len) : (i += L) {
        var fx: V = undefined;
        var fy: V = undefined;
        var tx: V = undefined;
        var ty: V = undefined;
        inline for (0..L) |k| {
            fx[k] = src[i + k].x;
            fy[k] = src[i + k].y;
            tx[k] = dst[i + k].x;
            ty[k] = dst[i + k].y;
        }
        const a = @as(V, @splat(f[0])) * fx + @as(V, @splat(f[1])) * fy + @as(V, @splat(f[2])) - tx;
        const b = @as(V, @splat(f[3])) * fx + @as(V, @splat(f[4])) * fy + @as(V, @splat(f[5])) - ty;
        err[i..][0..L].* = a * a + b * b;
    }
    while (i < src.len) : (i += 1) {
        const p = src[i];
        const a = f[0] * p.x + f[1] * p.y + f[2] - dst[i].x;
        const b = f[3] * p.x + f[4] * p.y + f[5] - dst[i].y;
        err[i] = a * a + b * b;
    }
}

const PartialCb = struct {
    src: []const Point2f,
    dst: []const Point2f,
    pub const Model = [6]f64;

    pub fn fit(c: *const PartialCb, idx: []const u32, out: []Model) usize {
        const x1: f64 = c.src[idx[0]].x;
        const y1: f64 = c.src[idx[0]].y;
        const x2: f64 = c.src[idx[1]].x;
        const y2: f64 = c.src[idx[1]].y;
        const X1: f64 = c.dst[idx[0]].x;
        const Y1: f64 = c.dst[idx[0]].y;
        const X2: f64 = c.dst[idx[1]].x;
        const Y2: f64 = c.dst[idx[1]].y;
        const d = 1.0 / ((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
        const S0 = d * ((X1 - X2) * (x1 - x2) + (Y1 - Y2) * (y1 - y2));
        const S1 = d * ((Y1 - Y2) * (x1 - x2) - (X1 - X2) * (y1 - y2));
        const S2 = d * ((Y1 - Y2) * (x1 * y2 - x2 * y1) - (X1 * y2 - X2 * y1) * (y1 - y2) - (X1 * x2 - X2 * x1) * (x1 - x2));
        const S3 = d * (-(X1 - X2) * (x1 * y2 - x2 * y1) - (Y1 * x2 - Y2 * x1) * (x1 - x2) - (Y1 * y2 - Y2 * y1) * (y1 - y2));
        out[0] = .{ S0, -S1, S2, S1, S0, S3 };
        return 1;
    }

    pub fn check(_: *const PartialCb, _: []const u32) bool {
        return true;
    }

    pub fn errors(c: *const PartialCb, m: Model, err: []f32) void {
        affineErrors(c.src, c.dst, m, err);
    }
};

/// LM over the 6 affine entries, or the 4 of a similarity (a, b, tx, ty).
fn AffineRefine(comptime partial: bool) type {
    return struct {
        src: []const Point2f,
        dst: []const Point2f,
        const np = if (partial) 4 else 6;

        fn eval(c: @This(), h: []const f64, a: ?[]f64, v: ?[]f64) [2]f64 {
            var s: f64 = 0;
            var mx: f64 = 0;
            if (a) |aa| @memset(aa[0 .. np * np], 0);
            if (v) |vv| @memset(vv[0..np], 0);
            for (c.src, c.dst) |M, m| {
                const X: f64 = M.x;
                const Y: f64 = M.y;
                const xi = if (partial) h[0] * X - h[1] * Y + h[2] else h[0] * X + h[1] * Y + h[2];
                const yi = if (partial) h[1] * X + h[0] * Y + h[3] else h[3] * X + h[4] * Y + h[5];
                const ex = xi - m.x;
                const ey = yi - m.y;
                s += ex * ex + ey * ey;
                mx = @max(mx, @max(@abs(ex), @abs(ey)));
                if (a) |aa| {
                    const jx: [np]f64 = if (partial) .{ X, -Y, 1, 0 } else .{ X, Y, 1, 0, 0, 0 };
                    const jy: [np]f64 = if (partial) .{ Y, X, 0, 1 } else .{ 0, 0, 0, X, Y, 1 };
                    for (0..np) |r| {
                        for (0..np) |q| aa[r * np + q] += jx[r] * jx[q] + jy[r] * jy[q];
                        v.?[r] += jx[r] * ex + jy[r] * ey;
                    }
                }
            }
            return .{ s, mx };
        }

        pub fn normal(c: @This(), x: []const f64, a: []f64, v: []f64) [2]f64 {
            return c.eval(x, a, v);
        }

        pub fn cost(c: @This(), x: []const f64) [2]f64 {
            return c.eval(x, null, null);
        }
    };
}

pub const AffineOptions = struct {
    method: Method = .ransac,
    ransac_threshold: f64 = 3,
    max_iters: usize = 2000,
    confidence: f64 = 0.99,
    refine_iters: usize = 10,
    seed: u64 = ransac.default_seed,
};

fn estimateAffineImpl(comptime partial: bool, gpa: Allocator, src_in: []const Point2f, dst_in: []const Point2f, inliers: ?[]u8, o: AffineOptions) Error!?[6]f64 {
    const n = src_in.len;
    if (dst_in.len != n) return error.InvalidArgument;
    if (o.method == .all) return error.InvalidArgument;
    const buf = try gpa.alloc(Point2f, 2 * n);
    defer gpa.free(buf);
    const src = buf[0..n];
    const dst = buf[n..];
    @memcpy(src, src_in);
    @memcpy(dst, dst_in);
    const tmask = try gpa.alloc(u8, 2 * n);
    defer gpa.free(tmask);
    const err = try gpa.alloc(f32, n);
    defer gpa.free(err);
    const best_mask = tmask[0..n];
    const scratch: ransac.Scratch = .{ .err = err, .mask = tmask[n..] };
    const mp = if (partial) 2 else 3;
    const Cb = if (partial) PartialCb else AffineCb;
    var cb: Cb = .{ .src = src, .dst = dst };
    var h: ?[6]f64 = if (o.method == .ransac)
        ransac.ransac(&cb, @intCast(n), mp, o.ransac_threshold, o.confidence, o.max_iters, o.seed, best_mask, scratch)
    else
        ransac.lmeds(&cb, @intCast(n), mp, o.confidence, o.max_iters, o.seed, best_mask, scratch);
    if (h != null and n > mp and o.refine_iters > 0) {
        var k: usize = 0;
        for (0..n) |i| {
            if (best_mask[i] != 0) {
                src[k] = src[i];
                dst[k] = dst[i];
                k += 1;
            }
        }
        if (k > 0) {
            const R = AffineRefine(partial);
            if (partial) {
                var x = [4]f64{ h.?[0], h.?[3], h.?[2], h.?[5] };
                _ = linalg.lmSolve(4, &x, o.refine_iters, R{ .src = src[0..k], .dst = dst[0..k] });
                h = .{ x[0], -x[1], x[2], x[1], x[0], x[3] };
            } else {
                var x = h.?;
                _ = linalg.lmSolve(6, &x, o.refine_iters, R{ .src = src[0..k], .dst = dst[0..k] });
                h = x;
            }
        }
    }
    if (inliers) |m| {
        if (h != null) @memcpy(m[0..n], best_mask) else @memset(m[0..n], 0);
    }
    return h;
}

/// OpenCV's estimateAffine2D (RANSAC or LMeDS, then LM refinement).
pub fn estimateAffine2D(gpa: Allocator, src: []const Point2f, dst: []const Point2f, inliers: ?[]u8, o: AffineOptions) Error!?[6]f64 {
    return estimateAffineImpl(false, gpa, src, dst, inliers, o);
}

/// OpenCV's estimateAffinePartial2D: rotation, uniform scale and
/// translation (4 degrees of freedom).
pub fn estimateAffinePartial2D(gpa: Allocator, src: []const Point2f, dst: []const Point2f, inliers: ?[]u8, o: AffineOptions) Error!?[6]f64 {
    return estimateAffineImpl(true, gpa, src, dst, inliers, o);
}

test "findHomography recovers an exact homography" {
    const gpa = std.testing.allocator;
    const H = [9]f64{ 1.2, 0.1, 30, -0.05, 0.9, 12, 0.0004, -0.0002, 1 };
    var src: [40]Point2f = undefined;
    var dst: [40]Point2f = undefined;
    var rng = linalg.Rng.init(5);
    for (&src) |*p| p.* = .{ .x = @floatFromInt(rng.uniform(0, 640)), .y = @floatFromInt(rng.uniform(0, 480)) };
    perspectiveTransform(&src, &dst, H);
    // Outliers.
    for (0..8) |i| dst[i * 5] = .{ .x = @floatFromInt(rng.uniform(0, 640)), .y = @floatFromInt(rng.uniform(0, 480)) };
    var mask: [40]u8 = undefined;
    for ([_]Method{ .ransac, .lmeds }) |m| {
        const h = (try findHomography(gpa, &src, &dst, &mask, .{ .method = m })).?;
        for (0..9) |i| try std.testing.expectApproxEqAbs(H[i], h[i], 1e-4 * @max(1, @abs(H[i])));
        for (0..40) |i| try std.testing.expectEqual(@as(u8, @intFromBool(i % 5 != 0 or i >= 40)), mask[i]);
    }
    const p = getPerspectiveTransform(src[1..5].*, dst[1..5].*).?;
    for (0..9) |i| try std.testing.expectApproxEqAbs(H[i], p[i], 1e-3 * @max(1, @abs(H[i])));
    const A = [6]f64{ 0.9, -0.2, 5, 0.3, 1.1, -7 };
    for (&src, &dst) |s, *d| d.* = .{ .x = @floatCast(A[0] * s.x + A[1] * s.y + A[2]), .y = @floatCast(A[3] * s.x + A[4] * s.y + A[5]) };
    const a = (try estimateAffine2D(gpa, &src, &dst, null, .{})).?;
    for (0..6) |i| try std.testing.expectApproxEqAbs(A[i], a[i], 1e-4);
}
