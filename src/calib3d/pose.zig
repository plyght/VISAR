//! Pose estimation: solvePnP (SOLVEPNP_ITERATIVE and SOLVEPNP_EPNP) and
//! solvePnPRansac, following OpenCV's solvepnp.cpp, epnp.cpp and
//! cvFindExtrinsicCameraParams2.
//!
//! ITERATIVE initialises from a homography (planar objects) or a DLT
//! (6+ non-planar points) and refines the reprojection error with
//! Levenberg-Marquardt (20 iterations), like OpenCV. RANSAC fits EPnP to
//! 5-point samples drawn with OpenCV's sampler and seed, then refines on
//! the inliers with ITERATIVE.

const std = @import("std");
const linalg = @import("linalg.zig");
const ransac = @import("ransac.zig");
const camera = @import("camera.zig");
const homography = @import("homography.zig");
const Allocator = std.mem.Allocator;
const Point2f = homography.Point2f;

pub const Error = Allocator.Error || error{InvalidArgument};

pub const Flags = enum(u8) { iterative = 0, epnp = 1 };

pub const PoseResult = struct { rvec: [3]f64, tvec: [3]f64 };

// ----------------------------------------------------------------- EPnP

const Epnp = struct {
    pws: []const [3]f64,
    us: []const [2]f64,
    alphas: [][4]f64,
    pcs: [][3]f64,
    fu: f64,
    fv: f64,
    uc: f64,
    vc: f64,
    cws: [4][3]f64 = undefined,
    ccs: [4][3]f64 = undefined,

    fn chooseControlPoints(e: *Epnp) void {
        const n: f64 = @floatFromInt(e.pws.len);
        e.cws[0] = .{ 0, 0, 0 };
        for (e.pws) |p| for (0..3) |j| {
            e.cws[0][j] += p[j];
        };
        for (0..3) |j| e.cws[0][j] /= n;
        var m: [3][3]f64 = @splat(@splat(0));
        for (e.pws) |p| {
            const d = [3]f64{ p[0] - e.cws[0][0], p[1] - e.cws[0][1], p[2] - e.cws[0][2] };
            for (0..3) |i| for (0..3) |j| {
                m[i][j] += d[i] * d[j];
            };
        }
        const sv = linalg.Svd(3, 3).compute(m);
        for (1..4) |i| {
            const k = @sqrt(sv.w[i - 1] / n);
            for (0..3) |j| e.cws[i][j] = e.cws[0][j] + k * sv.u[j][i - 1];
        }
    }

    fn barycentric(e: *Epnp) void {
        var cc: [9]f64 = undefined;
        for (0..3) |i| for (1..4) |j| {
            cc[3 * i + j - 1] = e.cws[j][i] - e.cws[0][i];
        };
        // cvInvert(CV_SVD): the pseudo-inverse (planar objects leave the
        // last control point on the centroid).
        const sv = linalg.Svd(3, 3).compute(.{ cc[0..3].*, cc[3..6].*, cc[6..9].* });
        var thr: f64 = 0;
        for (sv.w) |w| thr += w;
        thr *= std.math.floatEps(f64) * 2;
        var ci: [9]f64 = @splat(0);
        for (0..3) |k| {
            if (sv.w[k] <= thr) continue;
            for (0..3) |i| for (0..3) |j| {
                ci[i * 3 + j] += sv.vt[k][i] * sv.u[j][k] / sv.w[k];
            };
        }
        for (e.pws, e.alphas) |p, *a| {
            for (0..3) |j| a[1 + j] = ci[3 * j] * (p[0] - e.cws[0][0]) + ci[3 * j + 1] * (p[1] - e.cws[0][1]) + ci[3 * j + 2] * (p[2] - e.cws[0][2]);
            a[0] = 1.0 - a[1] - a[2] - a[3];
        }
    }

    fn computeCcs(e: *Epnp, betas: [4]f64, ut: *const [144]f64) void {
        e.ccs = @splat(@splat(0));
        for (0..4) |i| {
            const v = ut[12 * (11 - i) ..][0..12];
            for (0..4) |j| for (0..3) |k| {
                e.ccs[j][k] += betas[i] * v[3 * j + k];
            };
        }
    }

    fn computePcs(e: *Epnp) void {
        for (e.alphas, e.pcs) |a, *pc| {
            for (0..3) |j| pc[j] = a[0] * e.ccs[0][j] + a[1] * e.ccs[1][j] + a[2] * e.ccs[2][j] + a[3] * e.ccs[3][j];
        }
    }

    fn estimateRt(e: *Epnp, R: *[9]f64, t: *[3]f64) void {
        var pc0 = [3]f64{ 0, 0, 0 };
        var pw0 = [3]f64{ 0, 0, 0 };
        const n: f64 = @floatFromInt(e.pws.len);
        for (e.pcs, e.pws) |pc, pw| for (0..3) |j| {
            pc0[j] += pc[j];
            pw0[j] += pw[j];
        };
        for (0..3) |j| {
            pc0[j] /= n;
            pw0[j] /= n;
        }
        var abt: [3][3]f64 = @splat(@splat(0));
        for (e.pcs, e.pws) |pc, pw| for (0..3) |j| {
            abt[j][0] += (pc[j] - pc0[j]) * (pw[0] - pw0[0]);
            abt[j][1] += (pc[j] - pc0[j]) * (pw[1] - pw0[1]);
            abt[j][2] += (pc[j] - pc0[j]) * (pw[2] - pw0[2]);
        };
        const sv = linalg.Svd(3, 3).compute(abt);
        // R = U V^T
        for (0..3) |i| for (0..3) |j| {
            R[i * 3 + j] = sv.u[i][0] * sv.vt[0][j] + sv.u[i][1] * sv.vt[1][j] + sv.u[i][2] * sv.vt[2][j];
        };
        if (linalg.det3(R.*) < 0) {
            R[6] = -R[6];
            R[7] = -R[7];
            R[8] = -R[8];
        }
        for (0..3) |i| t[i] = pc0[i] - (R[i * 3] * pw0[0] + R[i * 3 + 1] * pw0[1] + R[i * 3 + 2] * pw0[2]);
    }

    fn reprojectionError(e: *const Epnp, R: [9]f64, t: [3]f64) f64 {
        var sum2: f64 = 0;
        for (e.pws, e.us) |pw, u| {
            const Xc = R[0] * pw[0] + R[1] * pw[1] + R[2] * pw[2] + t[0];
            const Yc = R[3] * pw[0] + R[4] * pw[1] + R[5] * pw[2] + t[1];
            const iz = 1.0 / (R[6] * pw[0] + R[7] * pw[1] + R[8] * pw[2] + t[2]);
            const ue = e.uc + e.fu * Xc * iz;
            const ve = e.vc + e.fv * Yc * iz;
            sum2 += @sqrt((u[0] - ue) * (u[0] - ue) + (u[1] - ve) * (u[1] - ve));
        }
        return sum2 / @as(f64, @floatFromInt(e.pws.len));
    }

    fn computeRt(e: *Epnp, ut: *const [144]f64, betas: [4]f64, R: *[9]f64, t: *[3]f64) f64 {
        e.computeCcs(betas, ut);
        e.computePcs();
        if (e.pcs[0][2] < 0) {
            for (&e.ccs) |*c| for (c) |*v| {
                v.* = -v.*;
            };
            for (e.pcs) |*p| for (p) |*v| {
                v.* = -v.*;
            };
        }
        e.estimateRt(R, t);
        return e.reprojectionError(R.*, t.*);
    }

    fn computePose(e: *Epnp, gpa: Allocator) Allocator.Error!PoseMat {
        e.chooseControlPoints();
        e.barycentric();
        var mtm: [144]f64 = @splat(0);
        for (e.alphas, e.us) |a, u| {
            var m1: [12]f64 = undefined;
            var m2: [12]f64 = undefined;
            for (0..4) |i| {
                m1[3 * i] = a[i] * e.fu;
                m1[3 * i + 1] = 0;
                m1[3 * i + 2] = a[i] * (e.uc - u[0]);
                m2[3 * i] = 0;
                m2[3 * i + 1] = a[i] * e.fv;
                m2[3 * i + 2] = a[i] * (e.vc - u[1]);
            }
            for (0..12) |i| for (i..12) |j| {
                mtm[i * 12 + j] += m1[i] * m1[j] + m2[i] * m2[j];
            };
        }
        _ = gpa;
        for (0..12) |i| for (0..i) |j| {
            mtm[i * 12 + j] = mtm[j * 12 + i];
        };
        // cvSVD(MtM, D, Ut, 0, U_T): the transposed input is MtM itself.
        var w: [12]f64 = undefined;
        var vt: [144]f64 = undefined;
        var ut = mtm;
        linalg.jacobiSvd(&ut, 12, 12, 12, &w, &vt);
        var l: [6][10]f64 = undefined;
        computeL6x10(&ut, &l);
        var rho: [6]f64 = undefined;
        const pairs = [6][2]usize{ .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 2 }, .{ 1, 3 }, .{ 2, 3 } };
        for (pairs, 0..) |p, i| {
            var d: f64 = 0;
            for (0..3) |k| d += (e.cws[p[0]][k] - e.cws[p[1]][k]) * (e.cws[p[0]][k] - e.cws[p[1]][k]);
            rho[i] = d;
        }
        var best: PoseMat = undefined;
        var best_err = std.math.inf(f64);
        for (1..4) |N| {
            var betas = switch (N) {
                1 => betasApprox1(l, rho),
                2 => betasApprox2(l, rho),
                else => betasApprox3(l, rho),
            };
            gaussNewton(l, rho, &betas);
            var R: [9]f64 = undefined;
            var t: [3]f64 = undefined;
            const err = e.computeRt(&ut, betas, &R, &t);
            if (N == 1 or err < best_err) {
                best_err = err;
                best = .{ .R = R, .t = t };
            }
        }
        return best;
    }
};

const PoseMat = struct { R: [9]f64, t: [3]f64 };

fn dot3(a: [3]f64, b: [3]f64) f64 {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

fn computeL6x10(ut: *const [144]f64, l: *[6][10]f64) void {
    var dv: [4][6][3]f64 = undefined;
    for (0..4) |i| {
        const v = ut[12 * (11 - i) ..][0..12];
        var a: usize = 0;
        var b: usize = 1;
        for (0..6) |j| {
            for (0..3) |k| dv[i][j][k] = v[3 * a + k] - v[3 * b + k];
            b += 1;
            if (b > 3) {
                a += 1;
                b = a + 1;
            }
        }
    }
    for (0..6) |i| {
        l[i] = .{
            dot3(dv[0][i], dv[0][i]),
            2 * dot3(dv[0][i], dv[1][i]),
            dot3(dv[1][i], dv[1][i]),
            2 * dot3(dv[0][i], dv[2][i]),
            2 * dot3(dv[1][i], dv[2][i]),
            dot3(dv[2][i], dv[2][i]),
            2 * dot3(dv[0][i], dv[3][i]),
            2 * dot3(dv[1][i], dv[3][i]),
            2 * dot3(dv[2][i], dv[3][i]),
            dot3(dv[3][i], dv[3][i]),
        };
    }
}

/// Least squares through the SVD (cvSolve CV_SVD) of a 6 x K system.
fn lstsqSvd(comptime K: usize, a: [6][K]f64, b: [6]f64) [K]f64 {
    const sv = linalg.Svd(6, K).compute(a);
    var x: [K]f64 = @splat(0);
    var thr: f64 = 0;
    for (sv.w) |w| thr += w;
    thr *= std.math.floatEps(f64) * 2;
    for (0..K) |i| {
        if (sv.w[i] <= thr) continue;
        var d: f64 = 0;
        for (0..6) |r| d += sv.u[r][i] * b[r];
        d /= sv.w[i];
        for (0..K) |c| x[c] += d * sv.vt[i][c];
    }
    return x;
}

fn betasApprox1(l: [6][10]f64, rho: [6]f64) [4]f64 {
    var a: [6][4]f64 = undefined;
    for (0..6) |i| a[i] = .{ l[i][0], l[i][1], l[i][3], l[i][6] };
    const b4 = lstsqSvd(4, a, rho);
    if (b4[0] < 0) {
        const b0 = @sqrt(-b4[0]);
        return .{ b0, -b4[1] / b0, -b4[2] / b0, -b4[3] / b0 };
    }
    const b0 = @sqrt(b4[0]);
    return .{ b0, b4[1] / b0, b4[2] / b0, b4[3] / b0 };
}

fn betasApprox2(l: [6][10]f64, rho: [6]f64) [4]f64 {
    var a: [6][3]f64 = undefined;
    for (0..6) |i| a[i] = .{ l[i][0], l[i][1], l[i][2] };
    const b3 = lstsqSvd(3, a, rho);
    var betas: [4]f64 = .{ 0, 0, 0, 0 };
    if (b3[0] < 0) {
        betas[0] = @sqrt(-b3[0]);
        betas[1] = if (b3[2] < 0) @sqrt(-b3[2]) else 0;
    } else {
        betas[0] = @sqrt(b3[0]);
        betas[1] = if (b3[2] > 0) @sqrt(b3[2]) else 0;
    }
    if (b3[1] < 0) betas[0] = -betas[0];
    return betas;
}

fn betasApprox3(l: [6][10]f64, rho: [6]f64) [4]f64 {
    var a: [6][5]f64 = undefined;
    for (0..6) |i| a[i] = l[i][0..5].*;
    const b5 = lstsqSvd(5, a, rho);
    var betas: [4]f64 = .{ 0, 0, 0, 0 };
    if (b5[0] < 0) {
        betas[0] = @sqrt(-b5[0]);
        betas[1] = if (b5[2] < 0) @sqrt(-b5[2]) else 0;
    } else {
        betas[0] = @sqrt(b5[0]);
        betas[1] = if (b5[2] > 0) @sqrt(b5[2]) else 0;
    }
    if (b5[1] < 0) betas[0] = -betas[0];
    betas[2] = b5[3] / betas[0];
    return betas;
}

fn gaussNewton(l: [6][10]f64, rho: [6]f64, betas: *[4]f64) void {
    for (0..5) |_| {
        var a: [6][4]f64 = undefined;
        var b: [6]f64 = undefined;
        const be = betas.*;
        for (0..6) |i| {
            const r = l[i];
            a[i] = .{
                2 * r[0] * be[0] + r[1] * be[1] + r[3] * be[2] + r[6] * be[3],
                r[1] * be[0] + 2 * r[2] * be[1] + r[4] * be[2] + r[7] * be[3],
                r[3] * be[0] + r[4] * be[1] + 2 * r[5] * be[2] + r[8] * be[3],
                r[6] * be[0] + r[7] * be[1] + r[8] * be[2] + 2 * r[9] * be[3],
            };
            b[i] = rho[i] - (r[0] * be[0] * be[0] + r[1] * be[0] * be[1] + r[2] * be[1] * be[1] + r[3] * be[0] * be[2] +
                r[4] * be[1] * be[2] + r[5] * be[2] * be[2] + r[6] * be[0] * be[3] + r[7] * be[1] * be[3] +
                r[8] * be[2] * be[3] + r[9] * be[3] * be[3]);
        }
        const x = qrSolve(a, b) orelse return;
        for (0..4) |i| betas[i] += x[i];
    }
}

/// EPnP's Householder QR least squares (epnp::qr_solve).
fn qrSolve(a_in: [6][4]f64, b_in: [6]f64) ?[4]f64 {
    const nr = 6;
    const nc = 4;
    var a = a_in;
    var b = b_in;
    var a1: [nc]f64 = undefined;
    var a2: [nc]f64 = undefined;
    for (0..nc) |k| {
        // OpenCV scans rows k .. nr-2 for the column maximum.
        var eta = @abs(a[k][k]);
        for (k + 1..nr) |i| eta = @max(eta, @abs(a[i - 1][k]));
        if (eta == 0) return null;
        var sum2: f64 = 0;
        const inv_eta = 1 / eta;
        for (k..nr) |i| {
            a[i][k] *= inv_eta;
            sum2 += a[i][k] * a[i][k];
        }
        var sigma = @sqrt(sum2);
        if (a[k][k] < 0) sigma = -sigma;
        a[k][k] += sigma;
        a1[k] = sigma * a[k][k];
        a2[k] = -eta * sigma;
        for (k + 1..nc) |j| {
            var sum: f64 = 0;
            for (k..nr) |i| sum += a[i][k] * a[i][j];
            const tau = sum / a1[k];
            for (k..nr) |i| a[i][j] -= tau * a[i][k];
        }
    }
    for (0..nc) |j| {
        var tau: f64 = 0;
        for (j..nr) |i| tau += a[i][j] * b[i];
        tau /= a1[j];
        for (j..nr) |i| b[i] -= tau * a[i][j];
    }
    var x: [nc]f64 = undefined;
    x[nc - 1] = b[nc - 1] / a2[nc - 1];
    var i: usize = nc - 1;
    while (i > 0) {
        i -= 1;
        var sum: f64 = 0;
        for (i + 1..nc) |j| sum += a[i][j] * x[j];
        x[i] = (b[i] - sum) / a2[i];
    }
    return x;
}

/// EPnP on undistorted pixel coordinates `us`.
fn epnp(gpa: Allocator, obj: []const [3]f64, us: []const [2]f64, K: [9]f64) Error!PoseResult {
    const n = obj.len;
    const alphas = try gpa.alloc([4]f64, n);
    defer gpa.free(alphas);
    const pcs = try gpa.alloc([3]f64, n);
    defer gpa.free(pcs);
    var e: Epnp = .{ .pws = obj, .us = us, .alphas = alphas, .pcs = pcs, .fu = K[0], .fv = K[4], .uc = K[2], .vc = K[5] };
    const pm = try e.computePose(gpa);
    return .{ .rvec = camera.rodriguesInv(pm.R), .tvec = pm.t };
}

// -------------------------------------------------------------- ITERATIVE

const PnpRefine = struct {
    obj: []const [3]f64,
    img: []const [2]f64,
    K: [9]f64,
    k: camera.Dist,

    fn eval(c: PnpRefine, x: []const f64, a: ?[]f64, v: ?[]f64) f64 {
        const p = camera.Pose.init(x[0..3].*, x[3..6].*, a != null);
        var s: f64 = 0;
        if (a) |aa| {
            @memset(aa[0..36], 0);
            @memset(v.?[0..6], 0);
        }
        for (c.obj, c.img) |X, m| {
            var J: [2][camera.jac_cols]f64 = undefined;
            const pr = camera.project(&p, X, c.K[0], c.K[4], c.K[2], c.K[5], c.k, if (a != null) &J else null);
            const e = [2]f64{ pr[0] - m[0], pr[1] - m[1] };
            s += e[0] * e[0] + e[1] * e[1];
            if (a) |aa| {
                for (0..6) |r| {
                    for (r..6) |q| aa[r * 6 + q] += J[0][r] * J[0][q] + J[1][r] * J[1][q];
                    v.?[r] += J[0][r] * e[0] + J[1][r] * e[1];
                }
            }
        }
        if (a) |aa| for (0..6) |r| for (0..r) |q| {
            aa[r * 6 + q] = aa[q * 6 + r];
        };
        return s;
    }

    pub fn normal(c: PnpRefine, x: []const f64, a: []f64, v: []f64) f64 {
        return c.eval(x, a, v);
    }

    pub fn cost(c: PnpRefine, x: []const f64) f64 {
        return c.eval(x, null, null);
    }
};

/// cvFindExtrinsicCameraParams2.
fn iterative(gpa: Allocator, obj: []const [3]f64, img: []const [2]f64, K: [9]f64, dist: []const f64, guess: ?PoseResult) Error!PoseResult {
    const n = obj.len;
    const k = try camera.expandDist(dist);
    var pose: PoseResult = undefined;
    if (guess) |g| {
        pose = g;
    } else {
        const mn = try gpa.alloc([2]f64, n);
        defer gpa.free(mn);
        try camera.undistortPoints(img, mn, K, dist, null, null);
        var mc = [3]f64{ 0, 0, 0 };
        for (obj) |p| for (0..3) |j| {
            mc[j] += p[j];
        };
        for (&mc) |*v| v.* /= @floatFromInt(n);
        var mm: [3][3]f64 = @splat(@splat(0));
        for (obj) |p| {
            const d = [3]f64{ p[0] - mc[0], p[1] - mc[1], p[2] - mc[2] };
            for (0..3) |i| for (0..3) |j| {
                mm[i][j] += d[i] * d[j];
            };
        }
        const sv = linalg.Svd(3, 3).compute(mm);
        if (sv.w[2] / sv.w[1] < 1e-3) {
            // Planar: homography between the object plane and the image.
            var Rt: [9]f64 = undefined;
            for (0..3) |i| for (0..3) |j| {
                Rt[i * 3 + j] = sv.vt[i][j];
            };
            if (Rt[2] * Rt[2] + Rt[5] * Rt[5] < 1e-10) Rt = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            if (linalg.det3(Rt) < 0) for (&Rt) |*v| {
                v.* = -v.*;
            };
            var T: [3]f64 = undefined;
            for (0..3) |i| T[i] = -(Rt[i * 3] * mc[0] + Rt[i * 3 + 1] * mc[1] + Rt[i * 3 + 2] * mc[2]);
            const buf = try gpa.alloc(Point2f, 2 * n);
            defer gpa.free(buf);
            for (obj, 0..) |p, i| {
                buf[i] = .{
                    .x = @floatCast(Rt[0] * p[0] + Rt[1] * p[1] + Rt[2] * p[2] + T[0]),
                    .y = @floatCast(Rt[3] * p[0] + Rt[4] * p[1] + Rt[5] * p[2] + T[1]),
                };
                buf[n + i] = .{ .x = @floatCast(mn[i][0]), .y = @floatCast(mn[i][1]) };
            }
            var R: [9]f64 = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
            var t = [3]f64{ 0, 0, 0 };
            if (try homography.findHomography(gpa, buf[0..n], buf[n..], null, .{})) |hh| {
                var h = hh;
                const h1n = @sqrt(h[0] * h[0] + h[3] * h[3] + h[6] * h[6]);
                const h2n = @sqrt(h[1] * h[1] + h[4] * h[4] + h[7] * h[7]);
                const s1 = 1 / @max(h1n, std.math.floatEps(f64));
                const s2 = 1 / @max(h2n, std.math.floatEps(f64));
                const s3 = 2 / @max(h1n + h2n, std.math.floatEps(f64));
                for (0..3) |i| {
                    h[i * 3] *= s1;
                    h[i * 3 + 1] *= s2;
                    t[i] = h[i * 3 + 2] * s3;
                }
                // h3 = h1 x h2
                h[2] = h[3] * h[7] - h[6] * h[4];
                h[5] = h[6] * h[1] - h[0] * h[7];
                h[8] = h[0] * h[4] - h[3] * h[1];
                const Hr = camera.rodrigues(camera.rodriguesInv(h), null);
                for (0..3) |i| t[i] += Hr[i * 3] * T[0] + Hr[i * 3 + 1] * T[1] + Hr[i * 3 + 2] * T[2];
                R = linalg.mul3(Hr, Rt);
            }
            pose = .{ .rvec = camera.rodriguesInv(R), .tvec = t };
        } else {
            if (n < 6) return error.InvalidArgument;
            var ll: [12][12]f64 = @splat(@splat(0));
            for (obj, mn) |M, m| {
                const x = -m[0];
                const y = -m[1];
                const r0 = [12]f64{ M[0], M[1], M[2], 1, 0, 0, 0, 0, x * M[0], x * M[1], x * M[2], x };
                const r1 = [12]f64{ 0, 0, 0, 0, M[0], M[1], M[2], 1, y * M[0], y * M[1], y * M[2], y };
                for (0..12) |i| for (i..12) |j| {
                    ll[i][j] += r0[i] * r0[j] + r1[i] * r1[j];
                };
            }
            for (0..12) |i| for (0..i) |j| {
                ll[i][j] = ll[j][i];
            };
            const lsv = linalg.Svd(12, 12).compute(ll);
            var rrt: [12]f64 = lsv.vt[11];
            var RR = [9]f64{ rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10] };
            if (linalg.det3(RR) < 0) {
                for (&rrt) |*v| v.* = -v.*;
                RR = .{ rrt[0], rrt[1], rrt[2], rrt[4], rrt[5], rrt[6], rrt[8], rrt[9], rrt[10] };
            }
            var sc: f64 = 0;
            for (RR) |v| sc += v * v;
            sc = @sqrt(sc);
            if (!(sc > std.math.floatEps(f64))) return error.InvalidArgument;
            const rsv = linalg.Svd(3, 3).compute(.{ RR[0..3].*, RR[3..6].*, RR[6..9].* });
            var R: [9]f64 = undefined;
            for (0..3) |i| for (0..3) |j| {
                R[i * 3 + j] = rsv.u[i][0] * rsv.vt[0][j] + rsv.u[i][1] * rsv.vt[1][j] + rsv.u[i][2] * rsv.vt[2][j];
            };
            var rn: f64 = 0;
            for (R) |v| rn += v * v;
            const f = @sqrt(rn) / sc;
            pose = .{ .rvec = camera.rodriguesInv(R), .tvec = .{ rrt[3] * f, rrt[7] * f, rrt[11] * f } };
        }
    }
    var x: [6]f64 = pose.rvec ++ pose.tvec;
    var work: [3 * 36 + 4 * 6]f64 = undefined;
    linalg.levMarq(6, &x, 20, std.math.floatEps(f32), &work, null, PnpRefine{ .obj = obj, .img = img, .K = K, .k = k });
    return .{ .rvec = x[0..3].*, .tvec = x[3..6].* };
}

/// OpenCV's solvePnP (SOLVEPNP_ITERATIVE or SOLVEPNP_EPNP). `guess`
/// (useExtrinsicGuess) seeds ITERATIVE.
pub fn solvePnP(gpa: Allocator, obj: []const [3]f64, img: []const [2]f64, K: [9]f64, dist: []const f64, flags: Flags, guess: ?PoseResult) Error!PoseResult {
    if (obj.len != img.len) return error.InvalidArgument;
    switch (flags) {
        .iterative => {
            if (obj.len < 4) return error.InvalidArgument;
            return iterative(gpa, obj, img, K, dist, guess);
        },
        .epnp => {
            if (obj.len < 4) return error.InvalidArgument;
            const us = try gpa.alloc([2]f64, obj.len);
            defer gpa.free(us);
            try camera.undistortPoints(img, us, K, dist, null, K);
            return epnp(gpa, obj, us, K);
        },
    }
}

// ------------------------------------------------------------------ RANSAC

const PnpCb = struct {
    gpa: Allocator,
    obj: []const [3]f64,
    img: []const [2]f64,
    /// Undistorted pixel coordinates (EPnP's input).
    und: []const [2]f64,
    K: [9]f64,
    k: camera.Dist,
    pub const Model = PoseResult;

    pub fn fit(c: *const PnpCb, idx: []const u32, out: []Model) usize {
        var o: [5][3]f64 = undefined;
        var u: [5][2]f64 = undefined;
        for (idx, 0..) |i, j| {
            o[j] = c.obj[i];
            u[j] = c.und[i];
        }
        const r = epnp(c.gpa, o[0..idx.len], u[0..idx.len], c.K) catch return 0;
        for (r.rvec ++ r.tvec) |v| if (!std.math.isFinite(v)) return 0;
        out[0] = r;
        return 1;
    }

    pub fn check(_: *const PnpCb, _: []const u32) bool {
        return true;
    }

    pub fn errors(c: *const PnpCb, m: Model, err: []f32) void {
        const p = camera.Pose.init(m.rvec, m.tvec, false);
        for (c.obj, c.img, err) |X, im, *e| {
            const pr = camera.project(&p, X, c.K[0], c.K[4], c.K[2], c.K[5], c.k, null);
            // OpenCV projects into Point2f and takes the f32 dot product.
            const dx: f32 = @as(f32, @floatCast(im[0])) - @as(f32, @floatCast(pr[0]));
            const dy: f32 = @as(f32, @floatCast(im[1])) - @as(f32, @floatCast(pr[1]));
            e.* = dx * dx + dy * dy;
        }
    }
};

pub const RansacOptions = struct {
    iterations: usize = 100,
    reprojection_error: f64 = 8,
    confidence: f64 = 0.99,
    flags: Flags = .iterative,
    seed: u64 = ransac.default_seed,
};

/// OpenCV's solvePnPRansac. `inliers` (n bytes, optional) receives the
/// inlier mask. Null when no pose was found.
pub fn solvePnPRansac(gpa: Allocator, obj_in: []const [3]f64, img_in: []const [2]f64, K: [9]f64, dist: []const f64, inliers: ?[]u8, o: RansacOptions) Error!?PoseResult {
    const n = obj_in.len;
    if (img_in.len != n or n < 5) return error.InvalidArgument;
    const k = try camera.expandDist(dist);
    // OpenCV converts the points to f32 first.
    const obj = try gpa.alloc([3]f64, n);
    defer gpa.free(obj);
    const img = try gpa.alloc([2]f64, 2 * n);
    defer gpa.free(img);
    for (obj_in, obj) |s, *d| for (0..3) |j| {
        d[j] = @as(f32, @floatCast(s[j]));
    };
    for (img_in, img[0..n]) |s, *d| for (0..2) |j| {
        d[j] = @as(f32, @floatCast(s[j]));
    };
    const und = img[n..];
    try camera.undistortPoints(img[0..n], und, K, dist, null, K);
    const mask = try gpa.alloc(u8, 2 * n);
    defer gpa.free(mask);
    const err = try gpa.alloc(f32, n);
    defer gpa.free(err);
    var cb: PnpCb = .{ .gpa = gpa, .obj = obj, .img = img[0..n], .und = und, .K = K, .k = k };
    const model = ransac.ransac(&cb, @intCast(n), 5, o.reprojection_error, o.confidence, o.iterations, o.seed, mask[0..n], .{ .err = err, .mask = mask[n..] }) orelse {
        if (inliers) |m| @memset(m[0..n], 0);
        return null;
    };
    // Refine on the inliers.
    const io = try gpa.alloc([3]f64, n);
    defer gpa.free(io);
    const ii = try gpa.alloc([2]f64, n);
    defer gpa.free(ii);
    var cnt: usize = 0;
    for (0..n) |i| {
        if (mask[i] != 0) {
            io[cnt] = obj[i];
            ii[cnt] = img[i];
            cnt += 1;
        }
    }
    if (inliers) |m| @memcpy(m[0..n], mask[0..n]);
    const r = solvePnP(gpa, io[0..cnt], ii[0..cnt], K, dist, o.flags, if (o.flags == .iterative) model else null) catch return model;
    return r;
}

test "solvePnP recovers a pose" {
    const gpa = std.testing.allocator;
    const K = [9]f64{ 800, 0, 320, 0, 800, 240, 0, 0, 1 };
    const dist = [_]f64{ -0.1, 0.01, 0.001, 0.0005, 0 };
    const rv = [3]f64{ 0.2, -0.3, 0.1 };
    const tv = [3]f64{ 0.1, -0.05, 4 };
    var obj: [30][3]f64 = undefined;
    var rng = linalg.Rng.init(3);
    for (&obj) |*p| for (p) |*v| {
        v.* = @as(f64, @floatFromInt(rng.uniform(0, 2000))) / 1000 - 1;
    };
    var img: [30][2]f64 = undefined;
    try camera.projectPoints(&obj, rv, tv, K, &dist, &img);
    for ([_]Flags{ .iterative, .epnp }) |f| {
        const r = try solvePnP(gpa, &obj, &img, K, &dist, f, null);
        for (0..3) |i| {
            try std.testing.expectApproxEqAbs(rv[i], r.rvec[i], 1e-6);
            try std.testing.expectApproxEqAbs(tv[i], r.tvec[i], 1e-6);
        }
    }
    // planar
    for (&obj) |*p| p[2] = 0;
    try camera.projectPoints(&obj, rv, tv, K, &dist, &img);
    const r = try solvePnP(gpa, &obj, &img, K, &dist, .iterative, null);
    for (0..3) |i| try std.testing.expectApproxEqAbs(rv[i], r.rvec[i], 1e-6);
    img[3] = .{ 10, 10 };
    img[7] = .{ 600, 10 };
    var mask: [30]u8 = undefined;
    const rr = (try solvePnPRansac(gpa, &obj, &img, K, &dist, &mask, .{})).?;
    for (0..3) |i| try std.testing.expectApproxEqAbs(tv[i], rr.tvec[i], 1e-4);
    try std.testing.expectEqual(@as(u8, 0), mask[3]);
    try std.testing.expectEqual(@as(u8, 0), mask[7]);
    try std.testing.expectEqual(@as(u8, 1), mask[4]);
}
