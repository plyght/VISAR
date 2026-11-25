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
