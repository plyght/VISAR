//! The C ABI of visar.features2d (declared in include/visar.h): FAST, ORB,
//! brute-force matching and keypoint drawing. Keypoint and match arrays
//! use OpenCV's cv::KeyPoint / cv::DMatch layouts.

const std = @import("std");
const capi = @import("capi.zig");
const visar = @import("visar.zig");
const f2d = visar.features2d;
const Image = visar.core.Image;
const CImage = capi.CImage;
const gpa = capi.gpa;
const fail = capi.fail;
const run = capi.run;
const exec = capi.exec;
const typed = capi.typed;

const KeyPoint = f2d.KeyPoint;
const DMatch = f2d.DMatch;

comptime {
    std.debug.assert(@sizeOf(KeyPoint) == 28);
    std.debug.assert(@sizeOf(DMatch) == 16);
}

fn give(comptime T: type, items: []T, out: *?[*]T, count: *usize) c_int {
    count.* = items.len;
    if (items.len == 0) {
        gpa.free(items);
        out.* = null;
    } else out.* = items.ptr;
    return 0;
}

/// FAST corners. type: 0 = TYPE_5_8, 1 = TYPE_7_12, 2 = TYPE_9_16.
/// The keypoints are VISAR-owned until visar_keypoints_free.
export fn visar_fast(src: *const CImage, threshold: c_int, nonmax: c_int, kind: c_int, out: *?[*]KeyPoint, count: *usize) c_int {
    out.* = null;
    count.* = 0;
    const t = std.enums.fromInt(f2d.FastType, kind) orelse return fail(error.InvalidArgument);
    const s = typed(u8, src) catch |e| return fail(e);
    const kps = f2d.fast(exec(), gpa, s, .{ .threshold = threshold, .nonmax_suppression = nonmax != 0, .type = t }) catch |e| return fail(e);
    return give(KeyPoint, kps, out, count);
}

export fn visar_keypoints_free(kps: ?[*]KeyPoint, count: usize) void {
    if (kps) |p| gpa.free(p[0..count]);
}

pub const COrbParams = extern struct {
    nfeatures: c_int = 500,
    scale_factor: f32 = 1.2,
    nlevels: c_int = 8,
    edge_threshold: c_int = 31,
    first_level: c_int = 0,
    wta_k: c_int = 2,
    /// 0 = HARRIS_SCORE, 1 = FAST_SCORE.
    score_type: c_int = 0,
    patch_size: c_int = 31,
    fast_threshold: c_int = 20,
};

/// Fills `p` with cv2.ORB_create()'s defaults.
export fn visar_orb_default_params(p: *COrbParams) void {
    p.* = .{};
}

fn nonneg(v: c_int) !u32 {
    if (v < 0) return error.InvalidArgument;
    return @intCast(v);
}

/// ORB detectAndCompute. `mask` and `params` may be null (no mask, the
/// defaults). Keypoints are freed with visar_keypoints_free; descriptors
/// (count * 32 bytes, only when `descriptors` is not null) with visar_free.
export fn visar_orb_detect_and_compute(src: *const CImage, mask: ?*const CImage, params: ?*const COrbParams, kps: *?[*]KeyPoint, count: *usize, descriptors: ?*?[*]u8) c_int {
    kps.* = null;
    count.* = 0;
    if (descriptors) |d| d.* = null;
    const p: COrbParams = if (params) |q| q.* else .{};
    const opt: f2d.OrbOptions = .{
        .nfeatures = nonneg(p.nfeatures) catch |e| return fail(e),
        .scale_factor = p.scale_factor,
        .nlevels = nonneg(p.nlevels) catch |e| return fail(e),
        .edge_threshold = nonneg(p.edge_threshold) catch |e| return fail(e),
        .first_level = nonneg(p.first_level) catch |e| return fail(e),
        .wta_k = std.math.cast(u8, p.wta_k) orelse return fail(error.InvalidArgument),
        .score_type = std.enums.fromInt(f2d.orb.ScoreType, p.score_type) orelse return fail(error.InvalidArgument),
        .patch_size = nonneg(p.patch_size) catch |e| return fail(e),
        .fast_threshold = p.fast_threshold,
    };
    const s = typed(u8, src) catch |e| return fail(e);
    const m: ?Image(u8) = if (mask) |mm| typed(u8, mm) catch |e| return fail(e) else null;
    var r = f2d.orbDetectAndCompute(exec(), gpa, s, m, opt, descriptors != null) catch |e| return fail(e);
    count.* = r.keypoints.len;
    if (r.keypoints.len == 0) {
        r.deinit(gpa);
        return 0;
    }
    kps.* = r.keypoints.ptr;
    if (descriptors) |d| d.* = r.descriptors.ptr else gpa.free(r.descriptors);
    return 0;
}

fn norm(n: c_int) !f2d.Norm {
    return std.enums.fromInt(f2d.Norm, n) orelse error.Unsupported;
}

/// BFMatcher(norm, crossCheck).match(query, train). Descriptors are one
/// row each: u8 (any norm) or f32 (NORM_L1, NORM_L2, NORM_L2SQR). The
/// matches are VISAR-owned until visar_matches_free.
export fn visar_bf_match(query: *const CImage, train: *const CImage, norm_type: c_int, cross_check: c_int, out: *?[*]DMatch, count: *usize) c_int {
    out.* = null;
    count.* = 0;
    const nt = norm(norm_type) catch |e| return fail(e);
    if (query.depth != train.depth) return fail(error.Unsupported);
    const ms = switch (query.depth) {
        inline 0, 5 => |d| blk: {
            const T = if (d == 0) u8 else f32;
            const q = typed(T, query) catch |e| return fail(e);
            const t = typed(T, train) catch |e| return fail(e);
            break :blk f2d.match(T, exec(), gpa, q, t, nt, cross_check != 0) catch |e| return fail(e);
        },
        else => return fail(error.Unsupported),
    };
    return give(DMatch, ms, out, count);
}

export fn visar_matches_free(m: ?[*]DMatch, count: usize) void {
    if (m) |p| gpa.free(p[0..count]);
}

/// BFMatcher(norm).knnMatch(query, train, k): `out` holds query rows * k
/// matches (row q's at q * k), `counts` the number found per query.
export fn visar_bf_knn_match(query: *const CImage, train: *const CImage, k: c_int, norm_type: c_int, out: [*]DMatch, counts: [*]u32) c_int {
    const nt = norm(norm_type) catch |e| return fail(e);
    if (k <= 0) return fail(error.InvalidArgument);
    if (query.depth != train.depth) return fail(error.Unsupported);
    const kk: usize = @intCast(k);
    const nq: usize = query.height;
    return run(switch (query.depth) {
        inline 0, 5 => |d| blk: {
            const T = if (d == 0) u8 else f32;
            const q = typed(T, query) catch |e| return fail(e);
            const t = typed(T, train) catch |e| return fail(e);
            break :blk f2d.knnMatch(T, exec(), gpa, q, t, kk, nt, out[0 .. nq * kk], counts[0..nq]);
        },
        else => error.Unsupported,
    });
}

/// Lowe's ratio test over visar_bf_knn_match output (k >= 2). Writes the
/// kept matches (at most nq) to `out` and their number to `count`.
export fn visar_ratio_test(knn: [*]const DMatch, counts: [*]const u32, nq: usize, k: c_int, ratio: f32, out: [*]DMatch, count: *usize) c_int {
    if (k < 2) return fail(error.InvalidArgument);
    const kk: usize = @intCast(k);
    count.* = f2d.ratioTest(knn[0 .. nq * kk], counts[0..nq], kk, ratio, out[0..nq]);
    return 0;
}

fn optColor(c: ?*const [4]f64) ?[3]u8 {
    const v = c orelse return null;
    if (v[0] < 0) return null;
    var o: [3]u8 = undefined;
    for (&o, v[0..3]) |*d, s| d.* = @intFromFloat(std.math.clamp(@round(s), 0, 255));
    return o;
}

/// drawKeypoints: `out` is BGR u8 of the size of `img` (1, 3 or 4 channels).
/// A null colour (or one with a negative first value) colours each keypoint
/// differently. flags: DrawMatchesFlags (1 DRAW_OVER_OUTIMG, 4 DRAW_RICH_KEYPOINTS).
export fn visar_draw_keypoints(img: *const CImage, kps: ?[*]const KeyPoint, count: usize, out: *const CImage, color: ?*const [4]f64, flags: c_int) c_int {
    const s = typed(u8, img) catch |e| return fail(e);
    const o = typed(u8, out) catch |e| return fail(e);
    const k: []const KeyPoint = if (kps) |p| p[0..count] else &.{};
    return run(f2d.drawKeypoints(exec(), s, k, o, optColor(color), .fromInt(@bitCast(flags))));
}

/// drawMatches: `out` is BGR u8, max(h1, h2) x (w1 + w2). flags as above,
/// plus 2 NOT_DRAW_SINGLE_POINTS.
export fn visar_draw_matches(img1: *const CImage, kp1: ?[*]const KeyPoint, n1: usize, img2: *const CImage, kp2: ?[*]const KeyPoint, n2: usize, matches: ?[*]const DMatch, nm: usize, out: *const CImage, match_color: ?*const [4]f64, single_color: ?*const [4]f64, flags: c_int) c_int {
    const a = typed(u8, img1) catch |e| return fail(e);
    const b = typed(u8, img2) catch |e| return fail(e);
    const o = typed(u8, out) catch |e| return fail(e);
    const k1: []const KeyPoint = if (kp1) |p| p[0..n1] else &.{};
    const k2: []const KeyPoint = if (kp2) |p| p[0..n2] else &.{};
    const m: []const DMatch = if (matches) |p| p[0..nm] else &.{};
    return run(f2d.drawMatches(exec(), a, k1, b, k2, m, o, optColor(match_color), optColor(single_color), .fromInt(@bitCast(flags))));
}
