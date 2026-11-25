//! Robust model fitting as OpenCV's ptsetreg.cpp: RANSAC and LMeDS over a
//! callback that fits a model to a minimal sample and scores every point.
//!
//! Sampling uses OpenCV's RNG (multiply-with-carry) seeded like OpenCV
//! (`(uint64)-1` unless the caller picks another seed) and draws subsets
//! exactly as RANSACPointSetRegistrator::getSubset does, so with the default
//! seed VISAR visits the same minimal samples in the same order as OpenCV.
//! Results are deterministic for a given seed.
//!
//! The callback `cb` provides:
//!   `Model` (a type), `fn fit(cb, idx: []const u32, out: []Model) usize`
//!   (number of models, 0 if degenerate), `fn check(cb, idx: []const u32)
//!   bool` (subset acceptance), and `fn errors(cb, m: Model, err: []f32)`
//!   (squared residual per point).

const std = @import("std");
const linalg = @import("linalg.zig");

pub const default_seed: u64 = std.math.maxInt(u64);

/// OpenCV's RANSACUpdateNumIters.
pub fn updateNumIters(p_in: f64, ep_in: f64, model_points: usize, max_iters: usize) usize {
    const p = std.math.clamp(p_in, 0, 1);
    const ep = std.math.clamp(ep_in, 0, 1);
    const num0 = @max(1 - p, std.math.floatMin(f64));
    const denom0 = 1 - std.math.pow(f64, 1 - ep, @floatFromInt(model_points));
    if (denom0 < std.math.floatMin(f64)) return 0;
    const num = @log(num0);
    const denom = @log(denom0);
    const mi: f64 = @floatFromInt(max_iters);
    if (denom >= 0 or -num >= mi * (-denom)) return max_iters;
    return @intFromFloat(@round(num / denom));
}

fn getSubset(cb: anytype, count: u32, idx: []u32, rng: *linalg.Rng, max_attempts: usize) bool {
    const mp = idx.len;
    var iters: usize = 0;
    var i: usize = 0;
    while (iters < max_attempts) : (iters += 1) {
        i = 0;
        while (i < mp and iters < max_attempts) {
            while (true) {
                const v = rng.uniform(0, count);
                idx[i] = v;
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    if (idx[j] == v) break;
                }
                if (j == i) break;
            }
            i += 1;
        }
        if (i == mp and !cb.check(idx)) continue;
        break;
    }
    return i == mp and iters < max_attempts;
}

pub const Scratch = struct {
    err: []f32,
    mask: []u8,
};

/// RANSAC (OpenCV's RANSACPointSetRegistrator::run). `best_mask` receives
/// the inliers of the returned model. `scratch` holds `count` floats and
/// bytes.
pub fn ransac(cb: anytype, count: u32, comptime model_points: usize, threshold: f64, confidence: f64, max_iters: usize, seed: u64, best_mask: []u8, scratch: Scratch) ?@TypeOf(cb.*).Model {
    const Model = @TypeOf(cb.*).Model;
    if (count < model_points) return null;
    var models: [4]Model = undefined;
    if (count == model_points) {
        var idx: [model_points]u32 = undefined;
        for (&idx, 0..) |*v, k| v.* = @intCast(k);
        if (cb.fit(&idx, &models) == 0) return null;
        @memset(best_mask[0..count], 1);
        return models[0];
    }
    var rng = linalg.Rng.init(seed);
    var niters = @max(max_iters, 1);
    var max_good: usize = 0;
    var best: Model = undefined;
    const thr: f32 = @floatCast(threshold * threshold);
    var iter: usize = 0;
    while (iter < niters) : (iter += 1) {
        var idx: [model_points]u32 = undefined;
        if (!getSubset(cb, count, &idx, &rng, 10000)) {
            if (iter == 0) return null;
            break;
        }
        const nm = cb.fit(&idx, &models);
        for (models[0..nm]) |m| {
            cb.errors(m, scratch.err[0..count]);
            var good: usize = 0;
            for (scratch.err[0..count], scratch.mask[0..count]) |e, *k| {
                const in = e <= thr;
                k.* = @intFromBool(in);
                good += @intFromBool(in);
            }
            if (good > @max(max_good, model_points - 1)) {
                @memcpy(best_mask[0..count], scratch.mask[0..count]);
                best = m;
                max_good = good;
                niters = updateNumIters(confidence, @as(f64, @floatFromInt(count - good)) / @as(f64, @floatFromInt(count)), model_points, niters);
            }
        }
    }
    if (max_good == 0) return null;
    return best;
}

/// LMeDS (OpenCV's LMeDSPointSetRegistrator::run).
pub fn lmeds(cb: anytype, count: u32, comptime model_points: usize, confidence: f64, max_iters: usize, seed: u64, best_mask: []u8, scratch: Scratch) ?@TypeOf(cb.*).Model {
    const Model = @TypeOf(cb.*).Model;
    if (count < model_points) return null;
    var models: [4]Model = undefined;
    if (count == model_points) {
        var idx: [model_points]u32 = undefined;
        for (&idx, 0..) |*v, k| v.* = @intCast(k);
        if (cb.fit(&idx, &models) == 0) return null;
        @memset(best_mask[0..count], 1);
        return models[0];
    }
    var rng = linalg.Rng.init(seed);
    const outlier_ratio = 0.45;
    const niters = @max(updateNumIters(confidence, outlier_ratio, model_points, max_iters), 3);
    var min_median: f64 = std.math.floatMax(f64);
    var best: Model = undefined;
    var iter: usize = 0;
    while (iter < niters) : (iter += 1) {
        var idx: [model_points]u32 = undefined;
        if (!getSubset(cb, count, &idx, &rng, 1000)) {
            if (iter == 0) return null;
            break;
        }
        const nm = cb.fit(&idx, &models);
        for (models[0..nm]) |m| {
            cb.errors(m, scratch.err[0..count]);
            const median: f64 = select(scratch.err[0..count], count / 2);
            if (median < min_median) {
                min_median = median;
                best = m;
            }
        }
    }
    if (min_median == std.math.floatMax(f64)) return null;
    const c: f64 = @floatFromInt(count);
    var sigma = 2.5 * 1.4826 * (1 + 5.0 / (c - @as(f64, model_points))) * @sqrt(min_median);
    sigma = @max(sigma, 0.001);
    cb.errors(best, scratch.err[0..count]);
    const thr: f32 = @floatCast(sigma * sigma);
    var good: usize = 0;
    for (scratch.err[0..count], best_mask[0..count]) |e, *k| {
        k.* = @intFromBool(e <= thr);
        good += k.*;
    }
    if (good < model_points) return null;
    return best;
}

/// The k-th smallest value (reorders `a`), as std::nth_element.
pub fn select(a: []f32, k: usize) f32 {
    var lo: usize = 0;
    var hi: usize = a.len;
    while (hi - lo > 1) {
        const pivot = a[lo + (hi - lo) / 2];
        var lt = lo;
        var i = lo;
        var gt = hi;
        while (i < gt) {
            if (a[i] < pivot) {
                std.mem.swap(f32, &a[lt], &a[i]);
                lt += 1;
                i += 1;
            } else if (a[i] > pivot) {
                gt -= 1;
                std.mem.swap(f32, &a[i], &a[gt]);
            } else i += 1;
        }
        if (k < lt) {
            hi = lt;
        } else if (k >= gt) {
            lo = gt;
        } else return pivot;
    }
    return a[k];
}

/// OpenCV's haveCollinearPoints for the last of `n` points (2-D, f32).
pub fn haveCollinear(pts: anytype, n: usize) bool {
    if (n < 3) return false;
    const i = n - 1;
    const flt_eps: f64 = std.math.floatEps(f32);
    for (0..i) |j| {
        const dx1: f64 = @as(f64, pts[j].x) - pts[i].x;
        const dy1: f64 = @as(f64, pts[j].y) - pts[i].y;
        for (0..j) |k| {
            const dx2: f64 = @as(f64, pts[k].x) - pts[i].x;
            const dy2: f64 = @as(f64, pts[k].y) - pts[i].y;
            if (@abs(dx2 * dy1 - dy2 * dx1) <= flt_eps * (@abs(dx1) + @abs(dy1) + @abs(dx2) + @abs(dy2))) return true;
        }
    }
    return false;
}

test "select and updateNumIters" {
    var a = [_]f32{ 5, 1, 4, 2, 3, 9, 0 };
    try std.testing.expectEqual(@as(f32, 3), select(&a, 3));
    var b = [_]f32{ 2, 2, 2, 1 };
    try std.testing.expectEqual(@as(f32, 2), select(&b, 2));
    try std.testing.expectEqual(@as(usize, 2000), updateNumIters(0.995, 0.99, 4, 2000));
    var r = linalg.Rng.init(std.math.maxInt(u64));
    _ = r.next();
}
