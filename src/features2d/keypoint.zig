//! Keypoints and matches, laid out like OpenCV's cv::KeyPoint and cv::DMatch
//! (so the C ABI can hand arrays of them straight to callers), and the
//! keypoint filters ORB uses (KeyPointsFilter in OpenCV).

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const KeyPoint = extern struct {
    x: f32,
    y: f32,
    /// Diameter of the meaningful neighbourhood.
    size: f32,
    /// Orientation in degrees [0, 360), or -1 when not computed.
    angle: f32 = -1,
    /// Detector response (FAST score, Harris measure).
    response: f32 = 0,
    /// Pyramid level the keypoint was found on.
    octave: i32 = 0,
    class_id: i32 = -1,
};

pub const DMatch = extern struct {
    query_idx: i32,
    train_idx: i32,
    img_idx: i32 = 0,
    distance: f32,
};

/// Keeps the keypoints whose response is at least the `n`-th largest
/// (KeyPointsFilter::retainBest: ties at the boundary all survive, so more
/// than `n` may remain). OpenCV leaves the survivors in the order its
/// nth_element produced, which depends on the C++ standard library; VISAR
/// keeps them in their original relative order. Returns the new length.
pub fn retainBest(gpa: Allocator, kps: []KeyPoint, n: usize) Allocator.Error!usize {
    if (kps.len <= n) return kps.len;
    if (n == 0) return 0;
    const resp = try gpa.alloc(f32, kps.len);
    defer gpa.free(resp);
    for (kps, resp) |k, *r| r.* = k.response;
    const bound = nthLargest(resp, n - 1);
    var m: usize = 0;
    for (kps) |k| {
        if (k.response >= bound) {
            kps[m] = k;
            m += 1;
        }
    }
    return m;
}

/// The (k+1)-th largest value of `v` (quickselect; reorders `v`).
fn nthLargest(v: []f32, k: usize) f32 {
    var lo: usize = 0;
    var hi: usize = v.len;
    while (hi - lo > 16) {
        const a = v[lo];
        const b = v[lo + (hi - lo) / 2];
        const c = v[hi - 1];
        const pivot = @max(@min(a, b), @min(@max(a, b), c));
        // Three-way partition: [lo, lt) > pivot, [lt, gt) == pivot, [gt, hi) < pivot.
        var lt = lo;
        var i = lo;
        var gt = hi;
        while (i < gt) {
            if (v[i] > pivot) {
                std.mem.swap(f32, &v[lt], &v[i]);
                lt += 1;
                i += 1;
            } else if (v[i] < pivot) {
                gt -= 1;
                std.mem.swap(f32, &v[i], &v[gt]);
            } else i += 1;
        }
        if (k < lt) {
            hi = lt;
        } else if (k >= gt) {
            lo = gt;
        } else return pivot;
    }
    std.mem.sort(f32, v[lo..hi], {}, std.sort.desc(f32));
    return v[k];
}

/// Removes keypoints closer than `b` to the border of a w x h image
/// (KeyPointsFilter::runByImageBorder; coordinates truncate to int).
pub fn runByImageBorder(kps: []KeyPoint, w: u32, h: u32, b: u32) usize {
    if (b == 0) return kps.len;
    if (h <= 2 * b or w <= 2 * b) return 0;
    var m: usize = 0;
    for (kps) |k| {
        const x: i64 = @intFromFloat(k.x);
        const y: i64 = @intFromFloat(k.y);
        if (x >= b and y >= b and x < w - b and y < h - b) {
            kps[m] = k;
            m += 1;
        }
    }
    return m;
}

test "retainBest keeps ties at the boundary" {
    var k = [_]KeyPoint{
        .{ .x = 0, .y = 0, .size = 7, .response = 5 },
        .{ .x = 1, .y = 0, .size = 7, .response = 9 },
        .{ .x = 2, .y = 0, .size = 7, .response = 5 },
        .{ .x = 3, .y = 0, .size = 7, .response = 1 },
        .{ .x = 4, .y = 0, .size = 7, .response = 7 },
    };
    const n = try retainBest(std.testing.allocator, &k, 3);
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqual(@as(f32, 5), k[0].response);
    try std.testing.expectEqual(@as(f32, 9), k[1].response);
    try std.testing.expectEqual(@as(f32, 7), k[3].response);
    var r = [_]f32{ 3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5 };
    for (0..r.len) |i| {
        var c = r;
        const got = nthLargest(&c, i);
        var s = r;
        std.mem.sort(f32, &s, {}, std.sort.desc(f32));
        try std.testing.expectEqual(s[i], got);
    }
    _ = &r;
}
