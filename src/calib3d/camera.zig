//! The pinhole camera with OpenCV's lens distortion: Rodrigues,
//! projectPoints (with analytic Jacobians) and undistortPoints, following
//! OpenCV's calibration_base.cpp and undistort.dispatch.cpp.
//!
//! Distortion coefficients use OpenCV's order (k1, k2, p1, p2[, k3[, k4,
//! k5, k6]]): 0, 4, 5 or 8 of them. The thin-prism and tilt terms are not
//! supported.

const std = @import("std");
const linalg = @import("linalg.zig");

pub const Error = error{InvalidArgument};

/// Expanded distortion: k1 k2 p1 p2 k3 k4 k5 k6.
pub const Dist = [8]f64;

pub fn expandDist(d: []const f64) Error!Dist {
    if (d.len != 0 and d.len != 4 and d.len != 5 and d.len != 8) return error.InvalidArgument;
    var k: Dist = @splat(0);
    @memcpy(k[0..d.len], d);
    return k;
}

/// Rotation vector -> matrix, with the 3x9 derivative dR/dr (row i =
/// d(R row-major)/d r_i) when `jac` is given.
pub fn rodrigues(r: [3]f64, jac: ?*[27]f64) [9]f64 {
    const theta = @sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (theta < std.math.floatEps(f64)) {
        if (jac) |j| {
            j.* = @splat(0);
            j[5] = -1;
            j[15] = -1;
            j[19] = -1;
            j[7] = 1;
            j[11] = 1;
            j[21] = 1;
        }
        return .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    }
    const c = @cos(theta);
    const s = @sin(theta);
    const c1 = 1 - c;
    const itheta = if (theta != 0) 1 / theta else 0;
    const rx = r[0] * itheta;
    const ry = r[1] * itheta;
    const rz = r[2] * itheta;
    const rrt = [9]f64{ rx * rx, rx * ry, rx * rz, rx * ry, ry * ry, ry * rz, rx * rz, ry * rz, rz * rz };
    const rx_ = [9]f64{ 0, -rz, ry, rz, 0, -rx, -ry, rx, 0 };
    const I = [9]f64{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    var R: [9]f64 = undefined;
    for (0..9) |k| R[k] = c * I[k] + c1 * rrt[k] + s * rx_[k];
    if (jac) |J| {
        const drrt = [27]f64{
            rx + rx, ry, rz, ry, 0,       0,  rz, 0,  0,
            0,       rx, 0,  rx, ry + ry, rz, 0,  rz, 0,
            0,       0,  rx, 0,  0,       ry, rx, ry, rz + rz,
        };
        const d_r_x_ = [27]f64{
            0, 0,  0, 0, 0, -1, 0,  1, 0,
            0, 0,  1, 0, 0, 0,  -1, 0, 0,
            0, -1, 0, 1, 0, 0,  0,  0, 0,
        };
        for (0..3) |i| {
            const ri = if (i == 0) rx else if (i == 1) ry else rz;
            const a0 = -s * ri;
            const a1 = (s - 2 * c1 * itheta) * ri;
            const a2 = c1 * itheta;
            const a3 = (c - s * itheta) * ri;
            const a4 = s * itheta;
            for (0..9) |k| J[i * 9 + k] = a0 * I[k] + a1 * rrt[k] + a2 * drrt[i * 9 + k] + a3 * rx_[k] + a4 * d_r_x_[i * 9 + k];
        }
    }
    return R;
}

/// Rotation matrix -> vector (OpenCV's Rodrigues: the matrix is first
/// projected onto SO(3) through its SVD).
pub fn rodriguesInv(m: [9]f64) [3]f64 {
    const sv = linalg.Svd(3, 3).compute(.{ m[0..3].*, m[3..6].*, m[6..9].* });
    var R: [9]f64 = undefined;
    for (0..3) |i| for (0..3) |j| {
        R[i * 3 + j] = sv.u[i][0] * sv.vt[0][j] + sv.u[i][1] * sv.vt[1][j] + sv.u[i][2] * sv.vt[2][j];
    };
    var r = [3]f64{ R[7] - R[5], R[2] - R[6], R[3] - R[1] };
    const s = @sqrt((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) * 0.25);
    var c = (R[0] + R[4] + R[8] - 1) * 0.5;
    c = std.math.clamp(c, -1, 1);
    var theta = std.math.acos(c);
    if (s < 1e-5) {
        if (c > 0) return .{ 0, 0, 0 };
        var t = (R[0] + 1) * 0.5;
        r[0] = @sqrt(@max(t, 0));
        t = (R[4] + 1) * 0.5;
        r[1] = @sqrt(@max(t, 0)) * @as(f64, if (R[1] < 0) -1 else 1);
        t = (R[8] + 1) * 0.5;
        r[2] = @sqrt(@max(t, 0)) * @as(f64, if (R[2] < 0) -1 else 1);
        if (@abs(r[0]) < @abs(r[1]) and @abs(r[0]) < @abs(r[2]) and (R[5] > 0) != (r[1] * r[2] > 0)) r[2] = -r[2];
        theta /= @sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        return .{ r[0] * theta, r[1] * theta, r[2] * theta };
    }
    const vth = 1 / (2 * s) * theta;
    return .{ r[0] * vth, r[1] * vth, r[2] * vth };
}

/// Columns of a projected point's Jacobian: rvec (3), tvec (3), fx, fy,
/// cx, cy, k1, k2, p1, p2, k3.
pub const jac_cols = 15;

pub const Pose = struct {
    R: [9]f64,
    /// dR/dr (3x9), when Jacobians are wanted.
    dRdr: [27]f64,
    t: [3]f64,

    pub fn init(rvec: [3]f64, tvec: [3]f64, with_jac: bool) Pose {
        var p: Pose = undefined;
        p.R = rodrigues(rvec, if (with_jac) &p.dRdr else null);
        p.t = tvec;
        return p;
    }
};

/// Projects one 3-D point as OpenCV's projectPoints (k: expanded
/// distortion). With `jac`, fills the 2 x jac_cols derivative.
pub inline fn project(p: *const Pose, X: [3]f64, fx: f64, fy: f64, cx: f64, cy: f64, k: Dist, jac: ?*[2][jac_cols]f64) [2]f64 {
    const R = p.R;
    const xc = R[0] * X[0] + R[1] * X[1] + R[2] * X[2] + p.t[0];
    const yc = R[3] * X[0] + R[4] * X[1] + R[5] * X[2] + p.t[1];
    const zc0 = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + p.t[2];
    const z = if (zc0 != 0) 1 / zc0 else 1;
    const x = xc * z;
    const y = yc * z;
    const r2 = x * x + y * y;
    const r4 = r2 * r2;
    const r6 = r4 * r2;
    const a1 = 2 * x * y;
    const a2 = r2 + 2 * x * x;
    const a3 = r2 + 2 * y * y;
    const cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
    const icdist2 = 1 / (1 + k[5] * r2 + k[6] * r4 + k[7] * r6);
    const xd = x * cdist * icdist2 + k[2] * a1 + k[3] * a2;
    const yd = y * cdist * icdist2 + k[2] * a3 + k[3] * a1;
    if (jac) |J| {
        // d(xd, yd)/d(x, y)
        const D = cdist * icdist2;
        const dcdist = k[0] + 2 * k[1] * r2 + 3 * k[4] * r4;
        const dicd = -icdist2 * icdist2 * (k[5] + 2 * k[6] * r2 + 3 * k[7] * r4);
        const dD = dcdist * icdist2 + cdist * dicd; // dD / dr2
        const dxdx = D + 2 * x * x * dD + 2 * k[2] * y + 6 * k[3] * x;
        const dxdy = 2 * x * y * dD + 2 * k[2] * x + 2 * k[3] * y;
        const dydx = 2 * x * y * dD + 2 * k[2] * x + 2 * k[3] * y;
        const dydy = D + 2 * y * y * dD + 6 * k[2] * y + 2 * k[3] * x;
        // d(x, y)/d(Xc, Yc, Zc)
        const dx = [3]f64{ z, 0, -x * z };
        const dy = [3]f64{ 0, z, -y * z };
        var du: [3]f64 = undefined;
        var dv: [3]f64 = undefined;
        for (0..3) |i| {
            du[i] = fx * (dxdx * dx[i] + dxdy * dy[i]);
            dv[i] = fy * (dydx * dx[i] + dydy * dy[i]);
        }
        // rvec: dXc/dr_i = dR_i X
        for (0..3) |i| {
            const d = p.dRdr[i * 9 ..][0..9];
            const dX = [3]f64{
                d[0] * X[0] + d[1] * X[1] + d[2] * X[2],
                d[3] * X[0] + d[4] * X[1] + d[5] * X[2],
                d[6] * X[0] + d[7] * X[1] + d[8] * X[2],
            };
            J[0][i] = du[0] * dX[0] + du[1] * dX[1] + du[2] * dX[2];
            J[1][i] = dv[0] * dX[0] + dv[1] * dX[1] + dv[2] * dX[2];
        }
        for (0..3) |i| {
            J[0][3 + i] = du[i];
            J[1][3 + i] = dv[i];
        }
        J[0][6] = xd;
        J[1][6] = 0;
        J[0][7] = 0;
        J[1][7] = yd;
        J[0][8] = 1;
        J[1][8] = 0;
        J[0][9] = 0;
        J[1][9] = 1;
        J[0][10] = fx * x * r2 * icdist2;
        J[1][10] = fy * y * r2 * icdist2;
        J[0][11] = fx * x * r4 * icdist2;
        J[1][11] = fy * y * r4 * icdist2;
        J[0][12] = fx * a1;
        J[1][12] = fy * a3;
        J[0][13] = fx * a2;
        J[1][13] = fy * a1;
        J[0][14] = fx * x * r6 * icdist2;
        J[1][14] = fy * y * r6 * icdist2;
    }
    return .{ xd * fx + cx, yd * fy + cy };
}

/// OpenCV's projectPoints.
pub fn projectPoints(obj: []const [3]f64, rvec: [3]f64, tvec: [3]f64, K: [9]f64, dist: []const f64, out: [][2]f64) Error!void {
    if (out.len != obj.len) return error.InvalidArgument;
    const k = try expandDist(dist);
    const p = Pose.init(rvec, tvec, false);
    for (obj, out) |X, *o| o.* = project(&p, X, K[0], K[4], K[2], K[5], k, null);
}

/// Normalised undistorted coordinates of one pixel (OpenCV's
/// cvUndistortPointsInternal with `iters` fixed-point iterations), before
/// the R / P mapping.
pub inline fn undistortNormalized(u: f64, v: f64, fx: f64, fy: f64, cx: f64, cy: f64, k: Dist, iters: usize) [2]f64 {
    const ifx = 1 / fx;
    const ify = 1 / fy;
    var x = (u - cx) * ifx;
    var y = (v - cy) * ify;
    const x0 = x;
    const y0 = y;
    for (0..iters) |_| {
        const r2 = x * x + y * y;
        const icdist = (1 + ((k[7] * r2 + k[6]) * r2 + k[5]) * r2) / (1 + ((k[4] * r2 + k[1]) * r2 + k[0]) * r2);
        if (icdist < 0) {
            x = (u - cx) * ifx;
            y = (v - cy) * ify;
            break;
        }
        const dX = 2 * k[2] * x * y + k[3] * (r2 + 2 * x * x);
        const dY = k[2] * (r2 + 2 * y * y) + 2 * k[3] * x * y;
        x = (x0 - dX) * icdist;
        y = (y0 - dY) * icdist;
    }
    return .{ x, y };
}

/// OpenCV's undistortPoints (default criteria: 5 iterations). `R` and
/// `P` are optional (identity / normalised coordinates when null).
pub fn undistortPoints(src: []const [2]f64, dst: [][2]f64, K: [9]f64, dist: []const f64, R: ?[9]f64, P: ?[9]f64) Error!void {
    if (src.len != dst.len) return error.InvalidArgument;
    const k = try expandDist(dist);
    var RR: [9]f64 = .{ 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    if (R) |r| RR = r;
    if (P) |p| RR = linalg.mul3(p, RR);
    const iters: usize = if (dist.len == 0) 0 else 5;
    for (src, dst) |s, *d| {
        const n = undistortNormalized(s[0], s[1], K[0], K[4], K[2], K[5], k, iters);
        const xx = RR[0] * n[0] + RR[1] * n[1] + RR[2];
        const yy = RR[3] * n[0] + RR[4] * n[1] + RR[5];
        const ww = 1 / (RR[6] * n[0] + RR[7] * n[1] + RR[8]);
        d.* = .{ xx * ww, yy * ww };
    }
}

test "rodrigues round trip and jacobian" {
    const r = [3]f64{ 0.3, -0.2, 0.5 };
    var J: [27]f64 = undefined;
    const R = rodrigues(r, &J);
    const back = rodriguesInv(R);
    for (0..3) |i| try std.testing.expectApproxEqAbs(r[i], back[i], 1e-12);
    const h = 1e-7;
    for (0..3) |i| {
        var rp = r;
        rp[i] += h;
        const Rp = rodrigues(rp, null);
        for (0..9) |k| try std.testing.expectApproxEqAbs((Rp[k] - R[k]) / h, J[i * 9 + k], 1e-5);
    }
    // projection Jacobian against finite differences
    const k: Dist = .{ -0.2, 0.05, 0.001, -0.002, 0.01, 0.0, 0.0, 0.0 };
    const p = Pose.init(r, .{ 0.1, -0.2, 3 }, true);
    var PJ: [2][jac_cols]f64 = undefined;
    const X = [3]f64{ 0.4, -0.3, 0.2 };
    const base = project(&p, X, 800, 780, 320, 240, k, &PJ);
    for (0..6) |i| {
        var rv = r;
        var tv = [3]f64{ 0.1, -0.2, 3 };
        if (i < 3) rv[i] += h else tv[i - 3] += h;
        const q = Pose.init(rv, tv, false);
        const o = project(&q, X, 800, 780, 320, 240, k, null);
        try std.testing.expectApproxEqAbs((o[0] - base[0]) / h, PJ[0][i], 1e-2);
        try std.testing.expectApproxEqAbs((o[1] - base[1]) / h, PJ[1][i], 1e-2);
    }
    for (0..5) |i| {
        var kk = k;
        const map = [5]usize{ 0, 1, 2, 3, 4 };
        kk[map[i]] += h;
        const o = project(&p, X, 800, 780, 320, 240, kk, null);
        try std.testing.expectApproxEqAbs((o[0] - base[0]) / h, PJ[0][10 + i], 1e-2);
        try std.testing.expectApproxEqAbs((o[1] - base[1]) / h, PJ[1][10 + i], 1e-2);
    }
}
