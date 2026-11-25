//! The C ABI of visar.calib3d (declared in include/visar.h). Points are
//! interleaved float (x, y) or (x, y, z) arrays; matrices are row-major
//! doubles; methods and flags use OpenCV's numbering.

const std = @import("std");
const capi = @import("capi.zig");
const visar = @import("visar.zig");
const c3 = visar.calib3d;
const Point2f = c3.Point2f;

const gpa = capi.gpa;
const fail = capi.fail;

fn pts2(p: [*]const f32, n: usize) []const Point2f {
    return @as([*]const Point2f, @ptrCast(@alignCast(p)))[0..n];
}

fn method(m: c_int) !c3.Method {
    return switch (m) {
        0 => .all,
        4 => .lmeds,
        8 => .ransac,
        else => error.Unsupported,
    };
}

/// findHomography: `method` 0 (all points), 4 (LMEDS) or 8 (RANSAC).
/// `mask` (n bytes) may be null. `found` is 0 when no homography exists
/// (OpenCV's empty result); `h` is then left untouched.
export fn visar_find_homography(src: [*]const f32, dst: [*]const f32, n: usize, meth: c_int, ransac_threshold: f64, max_iters: c_int, confidence: f64, mask: ?[*]u8, h: *[9]f64, found: *c_int) c_int {
    found.* = 0;
    const m = method(meth) catch |e| return fail(e);
    if (max_iters < 0) return fail(error.InvalidArgument);
    const r = c3.findHomography(gpa, pts2(src, n), pts2(dst, n), if (mask) |k| k[0..n] else null, .{
        .method = m,
        .ransac_threshold = ransac_threshold,
        .max_iters = @intCast(max_iters),
        .confidence = confidence,
    }) catch |e| return fail(e);
    if (r) |hh| {
        h.* = hh;
        found.* = 1;
    }
    return 0;
}

/// getPerspectiveTransform from 4 point pairs; fails when singular.
export fn visar_get_perspective_transform(src: *const [8]f32, dst: *const [8]f32, m: *[9]f64) c_int {
    const s: *const [4]Point2f = @ptrCast(src);
    const d: *const [4]Point2f = @ptrCast(dst);
    m.* = c3.getPerspectiveTransform(s.*, d.*) orelse return fail(error.InvalidArgument);
    return 0;
}

/// getAffineTransform from 3 point pairs; fails when singular.
export fn visar_get_affine_transform(src: *const [6]f32, dst: *const [6]f32, m: *[6]f64) c_int {
    const s: *const [3]Point2f = @ptrCast(src);
    const d: *const [3]Point2f = @ptrCast(dst);
    m.* = c3.getAffineTransform(s.*, d.*) orelse return fail(error.InvalidArgument);
    return 0;
}

/// perspectiveTransform of n 2-D float points (dst may equal src).
export fn visar_perspective_transform(src: [*]const f32, dst: [*]f32, n: usize, m: *const [9]f64) void {
    const d = @as([*]Point2f, @ptrCast(@alignCast(dst)))[0..n];
    c3.perspectiveTransform(pts2(src, n), d, m.*);
}

/// estimateAffine2D (`partial` = 0) or estimateAffinePartial2D (1);
/// `method` 4 (LMEDS) or 8 (RANSAC).
export fn visar_estimate_affine_2d(src: [*]const f32, dst: [*]const f32, n: usize, partial: c_int, meth: c_int, ransac_threshold: f64, max_iters: c_int, confidence: f64, refine_iters: c_int, inliers: ?[*]u8, m: *[6]f64, found: *c_int) c_int {
    found.* = 0;
    const mt = method(meth) catch |e| return fail(e);
    if (max_iters < 0 or refine_iters < 0) return fail(error.InvalidArgument);
    const o: c3.homography.AffineOptions = .{
        .method = mt,
        .ransac_threshold = ransac_threshold,
        .max_iters = @intCast(max_iters),
        .confidence = confidence,
        .refine_iters = @intCast(refine_iters),
    };
    const in = if (inliers) |k| k[0..n] else null;
    const r = (if (partial != 0) c3.estimateAffinePartial2D(gpa, pts2(src, n), pts2(dst, n), in, o) else c3.estimateAffine2D(gpa, pts2(src, n), pts2(dst, n), in, o)) catch |e| return fail(e);
    if (r) |mm| {
        m.* = mm;
        found.* = 1;
    }
    return 0;
}

// ------------------------------------------------------------ camera, pose

fn distSlice(dist: ?[*]const f64, nd: usize) []const f64 {
    return if (dist) |d| d[0..nd] else &.{};
}

/// Rodrigues: a rotation vector (`n` = 3) to a matrix (9 values out, with
/// the 3x9 Jacobian dR/dr in `jac` when not null), or a matrix (`n` = 9)
/// to a vector (3 values out; `jac` must be null).
export fn visar_rodrigues(src: [*]const f64, n: usize, dst: [*]f64, jac: ?*[27]f64) c_int {
    if (n == 3) {
        dst[0..9].* = c3.rodrigues(src[0..3].*, jac);
    } else if (n == 9 and jac == null) {
        dst[0..3].* = c3.rodriguesInv(src[0..9].*);
    } else return fail(error.InvalidArgument);
    return 0;
}

/// projectPoints of n (x, y, z) doubles into n (u, v) doubles. `jac`
/// (optional, needs nd <= 5) receives OpenCV's 2n x (10 + nd) Jacobian:
/// rvec, tvec, (fx, fy), (cx, cy), then the distortion coefficients.
export fn visar_project_points(obj: [*]const f64, n: usize, rvec: *const [3]f64, tvec: *const [3]f64, k: *const [9]f64, dist: ?[*]const f64, nd: usize, out: [*]f64, jac: ?[*]f64) c_int {
    const kd = c3.camera.expandDist(distSlice(dist, nd)) catch |e| return fail(e);
    if (jac != null and nd > 5) return fail(error.Unsupported);
    const p = c3.camera.Pose.init(rvec.*, tvec.*, jac != null);
    const cols = 10 + nd;
    for (0..n) |i| {
        var J: [2][c3.camera.jac_cols]f64 = undefined;
        const X = obj[3 * i ..][0..3].*;
        const r = c3.camera.project(&p, X, k[0], k[4], k[2], k[5], kd, if (jac != null) &J else null);
        out[2 * i] = r[0];
        out[2 * i + 1] = r[1];
        if (jac) |jj| for (0..2) |row| {
            const o = jj[(2 * i + row) * cols ..][0..cols];
            @memcpy(o[0..10], J[row][0..10]);
            @memcpy(o[10..], J[row][10..][0..nd]);
        };
    }
    return 0;
}

/// undistortPoints of n (u, v) doubles; `r` and `p` (3x3) may be null.
export fn visar_undistort_points(src: [*]const f64, n: usize, dst: [*]f64, k: *const [9]f64, dist: ?[*]const f64, nd: usize, r: ?*const [9]f64, p: ?*const [9]f64) c_int {
    const s = @as([*]const [2]f64, @ptrCast(src))[0..n];
    const d = @as([*][2]f64, @ptrCast(dst))[0..n];
    c3.undistortPoints(s, d, k.*, distSlice(dist, nd), if (r) |m| m.* else null, if (p) |m| m.* else null) catch |e| return fail(e);
    return 0;
}

/// solvePnP: `flags` 0 (SOLVEPNP_ITERATIVE) or 1 (SOLVEPNP_EPNP). With
/// `use_guess`, rvec/tvec seed the iterative solver.
export fn visar_solve_pnp(obj: [*]const f64, img: [*]const f64, n: usize, k: *const [9]f64, dist: ?[*]const f64, nd: usize, flags: c_int, use_guess: c_int, rvec: *[3]f64, tvec: *[3]f64) c_int {
    const f: c3.pose.Flags = switch (flags) {
        0 => .iterative,
        1 => .epnp,
        else => return fail(error.Unsupported),
    };
    const o = @as([*]const [3]f64, @ptrCast(obj))[0..n];
    const im = @as([*]const [2]f64, @ptrCast(img))[0..n];
    const guess: ?c3.pose.PoseResult = if (use_guess != 0) .{ .rvec = rvec.*, .tvec = tvec.* } else null;
    const r = c3.solvePnP(gpa, o, im, k.*, distSlice(dist, nd), f, guess) catch |e| return fail(e);
    rvec.* = r.rvec;
    tvec.* = r.tvec;
    return 0;
}

/// solvePnPRansac (EPnP on 5-point samples, then `flags` on the inliers).
/// `inliers` (n bytes) may be null; `found` is 0 when RANSAC failed.
export fn visar_solve_pnp_ransac(obj: [*]const f64, img: [*]const f64, n: usize, k: *const [9]f64, dist: ?[*]const f64, nd: usize, iterations: c_int, reprojection_error: f64, confidence: f64, flags: c_int, rvec: *[3]f64, tvec: *[3]f64, inliers: ?[*]u8, found: *c_int) c_int {
    found.* = 0;
    const f: c3.pose.Flags = switch (flags) {
        0 => .iterative,
        1 => .epnp,
        else => return fail(error.Unsupported),
    };
    if (iterations < 0) return fail(error.InvalidArgument);
    const o = @as([*]const [3]f64, @ptrCast(obj))[0..n];
    const im = @as([*]const [2]f64, @ptrCast(img))[0..n];
    const r = c3.solvePnPRansac(gpa, o, im, k.*, distSlice(dist, nd), if (inliers) |m| m[0..n] else null, .{
        .iterations = @intCast(iterations),
        .reprojection_error = reprojection_error,
        .confidence = confidence,
        .flags = f,
    }) catch |e| return fail(e);
    if (r) |rr| {
        rvec.* = rr.rvec;
        tvec.* = rr.tvec;
        found.* = 1;
    }
    return 0;
}
