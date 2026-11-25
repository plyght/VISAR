//! findFundamentalMat (7-point, 8-point, RANSAC and LMeDS over 7-point
//! samples) and solveCubic, following OpenCV's fundam.cpp. RANSAC draws
//! OpenCV's samples (see ransac.zig).
//!
//! The 7-point solver takes the null space of its 7 x 9 system by
//! Gauss-Jordan elimination rather than OpenCV's full SVD: the basis differs
//! but the cubic's solutions, normalised to F(3,3) = 1, are the same
//! matrices up to rounding (they may come out in another order).

const std = @import("std");
const linalg = @import("linalg.zig");
const ransac = @import("ransac.zig");
const homography = @import("homography.zig");
const Allocator = std.mem.Allocator;
const Point2f = homography.Point2f;

pub const Error = Allocator.Error || error{InvalidArgument};

pub const Method = enum(u8) { seven_point = 1, eight_point = 2, lmeds = 4, ransac = 8 };

/// OpenCV's solveCubic for c0 x³ + c1 x² + c2 x + c3 = 0. Returns the
/// number of real roots (-1: every x).
pub fn solveCubic(c: [4]f64, roots: *[3]f64) i32 {
    var a0 = c[0];
    var a1 = c[1];
    var a2 = c[2];
    var a3 = c[3];
    var x0: f64 = 0;
    var x1: f64 = 0;
    var x2: f64 = 0;
    var n: i32 = 0;
    if (a0 == 0) {
        if (a1 == 0) {
            if (a2 == 0) {
                n = if (a3 == 0) -1 else 0;
            } else {
                x0 = -a3 / a2;
                n = 1;
            }
        } else {
            var d = a2 * a2 - 4 * a1 * a3;
            if (d >= 0) {
                d = @sqrt(d);
                const q1 = (-a2 + d) * 0.5;
                const q2 = (a2 + d) * -0.5;
                if (@abs(q1) > @abs(q2)) {
                    x0 = q1 / a1;
                    x1 = a3 / q1;
                } else {
                    x0 = q2 / a1;
                    x1 = a3 / q2;
                }
                n = if (d > 0) 2 else 1;
            }
        }
    } else {
        a0 = 1 / a0;
        a1 *= a0;
        a2 *= a0;
        a3 *= a0;
        const Q = (a1 * a1 - 3 * a2) * (1.0 / 9.0);
        const R = (a1 * (2 * a1 * a1 - 9 * a2) + 27 * a3) * (1.0 / 54.0);
        const Qcubed = Q * Q * Q;
        var d = (a1 * a1 * (a2 * a2 - 4 * a1 * a3) + 2 * a2 * (9 * a1 * a3 - 2 * a2 * a2) - 27 * a3 * a3) * (1.0 / 108.0);
        if (d > 0) {
            const theta = std.math.acos(std.math.clamp(R / @sqrt(Qcubed), -1, 1));
            const sqrtQ = @sqrt(Q);
            const t0 = -2 * sqrtQ;
            const t1 = theta * (1.0 / 3.0);
            const t2 = a1 * (1.0 / 3.0);
            x0 = t0 * @cos(t1) - t2;
            x1 = t0 * @cos(t1 + (2.0 * std.math.pi / 3.0)) - t2;
            x2 = t0 * @cos(t1 + (4.0 * std.math.pi / 3.0)) - t2;
            n = 3;
        } else if (d == 0) {
            if (R >= 0) {
                x0 = -2 * std.math.cbrt(R) - a1 / 3;
                x1 = std.math.cbrt(R) - a1 / 3;
            } else {
                x0 = 2 * std.math.cbrt(-R) - a1 / 3;
                x1 = -std.math.cbrt(-R) - a1 / 3;
            }
            x2 = 0;
            n = if (x0 == x1) 1 else 2;
            if (x0 == x1) x1 = 0;
        } else {
            d = @sqrt(-d);
            var e = std.math.cbrt(d + @abs(R));
            if (R > 0) e = -e;
            x0 = (e + Q / e) - a1 * (1.0 / 3.0);
            n = 1;
        }
    }
    roots.* = .{ x0, x1, x2 };
    return n;
}

const Norm = struct { c1: [2]f64, c2: [2]f64, s1: f64, s2: f64 };

fn normalization(m1: []const Point2f, m2: []const Point2f, idx: ?[]const u32) ?Norm {
    const count = if (idx) |ii| ii.len else m1.len;
    var c1 = [2]f64{ 0, 0 };
    var c2 = [2]f64{ 0, 0 };
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        c1[0] += m1[i].x;
        c1[1] += m1[i].y;
        c2[0] += m2[i].x;
        c2[1] += m2[i].y;
    }
    const t = 1.0 / @as(f64, @floatFromInt(count));
    c1 = .{ c1[0] * t, c1[1] * t };
    c2 = .{ c2[0] * t, c2[1] * t };
    var s1: f64 = 0;
    var s2: f64 = 0;
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        s1 += std.math.hypot(m1[i].x - c1[0], m1[i].y - c1[1]);
        s2 += std.math.hypot(m2[i].x - c2[0], m2[i].y - c2[1]);
    }
    s1 *= t;
    s2 *= t;
    const fe = std.math.floatEps(f32);
    if (s1 < fe or s2 < fe) return null;
    return .{ .c1 = c1, .c2 = c2, .s1 = @sqrt(2.0) / s1, .s2 = @sqrt(2.0) / s2 };
}

fn row(nm: Norm, p1: Point2f, p2: Point2f) [9]f64 {
    const x1 = (p1.x - nm.c1[0]) * nm.s1;
    const y1 = (p1.y - nm.c1[1]) * nm.s1;
    const x2 = (p2.x - nm.c2[0]) * nm.s2;
    const y2 = (p2.y - nm.c2[1]) * nm.s2;
    return .{ x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1 };
}

/// F = T2ᵀ F0 T1, scaled so F(3,3) = 1.
fn denormalize(nm: Norm, f0: [9]f64) [9]f64 {
    const T1 = [9]f64{ nm.s1, 0, -nm.s1 * nm.c1[0], 0, nm.s1, -nm.s1 * nm.c1[1], 0, 0, 1 };
    const T2 = [9]f64{ nm.s2, 0, -nm.s2 * nm.c2[0], 0, nm.s2, -nm.s2 * nm.c2[1], 0, 0, 1 };
    var f = linalg.mul3(linalg.mul3(linalg.transpose3(T2), f0), T1);
    if (@abs(f[8]) > std.math.floatEps(f32)) {
        const s = 1.0 / f[8];
        for (&f) |*v| v.* *= s;
    }
    return f;
}

fn run8Point(m1: []const Point2f, m2: []const Point2f, idx: ?[]const u32) ?[9]f64 {
    const nm = normalization(m1, m2, idx) orelse return null;
    const count = if (idx) |ii| ii.len else m1.len;
    var a: [81]f64 = @splat(0);
    for (0..count) |k| {
        const i = if (idx) |ii| ii[k] else k;
        const r = row(nm, m1[i], m2[i]);
        for (0..9) |p| for (0..9) |q| {
            a[p * 9 + q] += r[p] * r[q];
        };
    }
    var w: [9]f64 = undefined;
    var v: [81]f64 = undefined;
    linalg.eigenSym(&a, 9, &w, &v);
    var i: usize = 0;
    while (i < 9) : (i += 1) {
        if (@abs(w[i]) < std.math.floatEps(f64)) break;
    }
    if (i < 8) return null;
    const f0 = v[72..81].*;
    // Rank 2.
    const sv = linalg.Svd(3, 3).compute(.{ f0[0..3].*, f0[3..6].*, f0[6..9].* });
    var f: [9]f64 = undefined;
    for (0..3) |r| for (0..3) |c| {
        f[r * 3 + c] = sv.u[r][0] * sv.w[0] * sv.vt[0][c] + sv.u[r][1] * sv.w[1] * sv.vt[1][c];
    };
    return denormalize(nm, f);
}

/// Up to three fundamental matrices through 7 correspondences.
fn run7Point(m1: []const Point2f, m2: []const Point2f, idx: []const u32, out: [][9]f64) usize {
    const nm = normalization(m1, m2, idx) orelse return 0;
    // Null space of the 7 x 9 system by Gauss-Jordan elimination with
    // full pivoting (OpenCV takes the last two right singular vectors; any
    // basis of the same plane gives the same solutions).
    var a: [7][9]f64 = undefined;
    for (idx, 0..) |i, k| a[k] = row(nm, m1[i], m2[i]);
    var cols = [9]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    for (0..7) |k| {
        var br = k;
        var bc = k;
        var bv: f64 = 0;
        for (k..7) |r| for (k..9) |cc| {
            const v = @abs(a[r][cols[cc]]);
            if (v > bv) {
                bv = v;
                br = r;
                bc = cc;
            }
        };
        if (!(bv > 1e-12)) return 0;
        std.mem.swap([9]f64, &a[k], &a[br]);
        std.mem.swap(usize, &cols[k], &cols[bc]);
        const pc = cols[k];
        const inv = 1 / a[k][pc];
        for (&a[k]) |*v| v.* *= inv;
        for (0..7) |r| {
            if (r == k) continue;
            const f = a[r][pc];
            if (f == 0) continue;
            for (0..9) |cc| a[r][cc] -= f * a[k][cc];
        }
    }
    var f1: [9]f64 = @splat(0);
    var f2: [9]f64 = @splat(0);
    f1[cols[7]] = 1;
    f2[cols[8]] = 1;
    for (0..7) |k| {
        f1[cols[k]] = -a[k][cols[7]];
        f2[cols[k]] = -a[k][cols[8]];
    }
    for (0..9) |i| f1[i] -= f2[i];
    var t0 = f2[4] * f2[8] - f2[5] * f2[7];
    var t1 = f2[3] * f2[8] - f2[5] * f2[6];
    var t2 = f2[3] * f2[7] - f2[4] * f2[6];
    var c: [4]f64 = undefined;
    c[3] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2;
    c[2] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2 -
        f1[3] * (f2[1] * f2[8] - f2[2] * f2[7]) +
        f1[4] * (f2[0] * f2[8] - f2[2] * f2[6]) -
        f1[5] * (f2[0] * f2[7] - f2[1] * f2[6]) +
        f1[6] * (f2[1] * f2[5] - f2[2] * f2[4]) -
        f1[7] * (f2[0] * f2[5] - f2[2] * f2[3]) +
        f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);
    t0 = f1[4] * f1[8] - f1[5] * f1[7];
    t1 = f1[3] * f1[8] - f1[5] * f1[6];
    t2 = f1[3] * f1[7] - f1[4] * f1[6];
    c[1] = f2[0] * t0 - f2[1] * t1 + f2[2] * t2 -
        f2[3] * (f1[1] * f1[8] - f1[2] * f1[7]) +
        f2[4] * (f1[0] * f1[8] - f1[2] * f1[6]) -
        f2[5] * (f1[0] * f1[7] - f1[1] * f1[6]) +
        f2[6] * (f1[1] * f1[5] - f1[2] * f1[4]) -
        f2[7] * (f1[0] * f1[5] - f1[2] * f1[3]) +
        f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);
    c[0] = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
    var roots: [3]f64 = undefined;
    const n = solveCubic(c, &roots);
    if (n < 1 or n > 3) return 0;
    for (0..@intCast(n)) |k| {
        var lambda = roots[k];
        var mu: f64 = 1;
        const s = f1[8] * roots[k] + f2[8];
        var f: [9]f64 = undefined;
        if (@abs(s) > std.math.floatEps(f64)) {
            mu = 1 / s;
            lambda *= mu;
            f[8] = 1;
        } else f[8] = 0;
        for (0..8) |i| f[i] = f1[i] * lambda + f2[i] * mu;
        out[k] = denormalize(nm, f);
    }
    return @intCast(n);
}

/// Squared symmetric epipolar distance (the larger of the two), as
/// FMEstimatorCallback::computeError.
fn epipolarErrors(m1: []const Point2f, m2: []const Point2f, F: [9]f64, err: []f32) void {
    const L = 4;
    const V = @Vector(L, f64);
    var i: usize = 0;
    const sp = struct {
        inline fn s(v: f64) V {
            return @splat(v);
        }
    }.s;
    while (i + L <= m1.len) : (i += L) {
        var px: V = undefined;
        var py: V = undefined;
        var qx: V = undefined;
        var qy: V = undefined;
        inline for (0..L) |l| {
            px[l] = m1[i + l].x;
            py[l] = m1[i + l].y;
            qx[l] = m2[i + l].x;
            qy[l] = m2[i + l].y;
        }
        var a = sp(F[0]) * px + sp(F[1]) * py + sp(F[2]);
        var b = sp(F[3]) * px + sp(F[4]) * py + sp(F[5]);
        var c = sp(F[6]) * px + sp(F[7]) * py + sp(F[8]);
        const s2 = sp(1) / (a * a + b * b);
        const d2 = qx * a + qy * b + c;
        a = sp(F[0]) * qx + sp(F[3]) * qy + sp(F[6]);
        b = sp(F[1]) * qx + sp(F[4]) * qy + sp(F[7]);
        c = sp(F[2]) * qx + sp(F[5]) * qy + sp(F[8]);
        const s1 = sp(1) / (a * a + b * b);
        const d1 = px * a + py * b + c;
        const ev: @Vector(L, f32) = @floatCast(@max(d1 * d1 * s1, d2 * d2 * s2));
        err[i..][0..L].* = ev;
    }
    while (i < m1.len) : (i += 1) {
        const p = m1[i];
        const q = m2[i];
        var a = F[0] * p.x + F[1] * p.y + F[2];
        var b = F[3] * p.x + F[4] * p.y + F[5];
        var c = F[6] * p.x + F[7] * p.y + F[8];
        const s2 = 1.0 / (a * a + b * b);
        const d2 = q.x * a + q.y * b + c;
        a = F[0] * q.x + F[3] * q.y + F[6];
        b = F[1] * q.x + F[4] * q.y + F[7];
        c = F[2] * q.x + F[5] * q.y + F[8];
        const s1 = 1.0 / (a * a + b * b);
        const d1 = p.x * a + p.y * b + c;
        err[i] = @floatCast(@max(d1 * d1 * s1, d2 * d2 * s2));
    }
}

const FmCb = struct {
    m1: []const Point2f,
    m2: []const Point2f,
    pub const Model = [9]f64;

    pub fn fit(c: *const FmCb, idx: []const u32, out: []Model) usize {
        if (idx.len == 7) return run7Point(c.m1, c.m2, idx, out);
        out[0] = run8Point(c.m1, c.m2, idx) orelse return 0;
        return 1;
    }

    pub fn check(c: *const FmCb, idx: []const u32) bool {
        var s: [8]Point2f = undefined;
        var d: [8]Point2f = undefined;
        for (idx, 0..) |i, k| {
            s[k] = c.m1[i];
            d[k] = c.m2[i];
        }
        return !ransac.haveCollinear(&s, idx.len) and !ransac.haveCollinear(&d, idx.len);
    }

    pub fn errors(c: *const FmCb, m: Model, err: []f32) void {
        epipolarErrors(c.m1, c.m2, m, err);
    }
};

pub const Options = struct {
    method: Method = .ransac,
    ransac_threshold: f64 = 3,
    confidence: f64 = 0.99,
    max_iters: usize = 1000,
    seed: u64 = ransac.default_seed,
};

/// OpenCV's findFundamentalMat. Writes up to 3 matrices to `out` (only
/// FM_7POINT with exactly 7 points yields several) and returns how many
/// (0: not found). `mask` (n bytes, optional) receives the inliers.
pub fn findFundamentalMat(gpa: Allocator, m1: []const Point2f, m2: []const Point2f, mask: ?[]u8, out: *[3][9]f64, o: Options) Error!usize {
    const n = m1.len;
    if (m2.len != n) return error.InvalidArgument;
    if (n < 7) return 0;
    if (n == 7 or o.method == .eight_point or o.method == .seven_point) {
        if (mask) |m| @memset(m[0..n], 1);
        if (n == 7) {
            var idx: [7]u32 = .{ 0, 1, 2, 3, 4, 5, 6 };
            return run7Point(m1, m2, &idx, out);
        }
        out[0] = run8Point(m1, m2, null) orelse return 0;
        return 1;
    }
    const thr = if (o.ransac_threshold <= 0) 3 else o.ransac_threshold;
    const conf = if (o.confidence < std.math.floatEps(f64) or o.confidence > 1 - std.math.floatEps(f64)) 0.99 else o.confidence;
    const buf = try gpa.alloc(u8, 2 * n);
    defer gpa.free(buf);
    const err = try gpa.alloc(f32, n);
    defer gpa.free(err);
    var cb: FmCb = .{ .m1 = m1, .m2 = m2 };
    const scratch: ransac.Scratch = .{ .err = err, .mask = buf[n..] };
    const f = if (o.method == .ransac and n >= 15)
        ransac.ransac(&cb, @intCast(n), 7, thr, conf, o.max_iters, o.seed, buf[0..n], scratch)
    else
        ransac.lmeds(&cb, @intCast(n), 7, conf, o.max_iters, o.seed, buf[0..n], scratch);
    if (f == null) {
        if (mask) |m| @memset(m[0..n], 0);
        return 0;
    }
    if (mask) |m| @memcpy(m[0..n], buf[0..n]);
    out[0] = f.?;
    return 1;
}

test "findFundamentalMat on synthetic views" {
    const gpa = std.testing.allocator;
    const camera = @import("camera.zig");
    const K = [9]f64{ 800, 0, 320, 0, 800, 240, 0, 0, 1 };
    var obj: [60][3]f64 = undefined;
    var rng = linalg.Rng.init(11);
    for (&obj) |*p| {
        for (p[0..2]) |*v| v.* = @as(f64, @floatFromInt(rng.uniform(0, 2000))) / 1000 - 1;
        p[2] = 4 + @as(f64, @floatFromInt(rng.uniform(0, 2000))) / 1000;
    }
    var a: [60][2]f64 = undefined;
    var b: [60][2]f64 = undefined;
    try camera.projectPoints(&obj, .{ 0, 0, 0 }, .{ 0, 0, 0 }, K, &.{}, &a);
    try camera.projectPoints(&obj, .{ 0.05, -0.1, 0.02 }, .{ -0.5, 0.05, 0.1 }, K, &.{}, &b);
    var p1: [60]Point2f = undefined;
    var p2: [60]Point2f = undefined;
    for (0..60) |i| {
        p1[i] = .{ .x = @floatCast(a[i][0]), .y = @floatCast(a[i][1]) };
        p2[i] = .{ .x = @floatCast(b[i][0]), .y = @floatCast(b[i][1]) };
    }
    p2[3] = .{ .x = 10, .y = 400 };
    var F: [3][9]f64 = undefined;
    var mask: [60]u8 = undefined;
    for ([_]Method{ .ransac, .lmeds }) |m| {
        try std.testing.expectEqual(@as(usize, 1), try findFundamentalMat(gpa, &p1, &p2, &mask, &F, .{ .method = m }));
        try std.testing.expectEqual(@as(u8, 0), mask[3]);
        var e: [60]f32 = undefined;
        epipolarErrors(&p1, &p2, F[0], &e);
        for (0..60) |i| if (i != 3) try std.testing.expect(e[i] < 0.01);
    }
}
