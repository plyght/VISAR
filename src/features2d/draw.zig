//! drawKeypoints and drawMatches (OpenCV's features2d drawing), on top of
//! imgproc's drawing. The layout and flags follow OpenCV; circles and lines
//! are drawn with 8-connected strokes rather than OpenCV's anti-aliased
//! ones, so the pixels are not bit-identical to cv2's.

const std = @import("std");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const kp_mod = @import("keypoint.zig");
const Image = core.Image;
const Exec = core.Exec;

pub const KeyPoint = kp_mod.KeyPoint;
pub const DMatch = kp_mod.DMatch;
pub const Error = core.Error || error{ InvalidArgument, Unsupported };

/// cv::DrawMatchesFlags.
pub const Flags = struct {
    /// Draw on `out` as it is instead of copying the input image(s) into it.
    over_outimg: bool = false,
    /// drawMatches: skip keypoints that have no match.
    not_draw_single_points: bool = false,
    /// Circles of the keypoint's size, with a radius showing its angle.
    rich_keypoints: bool = false,

    pub fn fromInt(v: u32) Flags {
        return .{ .over_outimg = v & 1 != 0, .not_draw_single_points = v & 2 != 0, .rich_keypoints = v & 4 != 0 };
    }
};

/// A colour per index when none is given (OpenCV draws random colours).
fn autoColor(i: usize) [3]u8 {
    var h: u64 = (@as(u64, i) +% 0x9e3779b97f4a7c15) *% 0xbf58476d1ce4e5b9;
    h ^= h >> 31;
    return .{ @truncate(h), @truncate(h >> 8), @truncate(h >> 16) };
}

/// Copies a 1, 3 or 4 channel image into the BGR image `dst` at (x0, 0).
fn toBgr(exec: Exec, src: Image(u8), dst: Image(u8), x0: u32) Error!void {
    const view = try dst.roi(x0, 0, src.width, src.height);
    switch (src.channels) {
        1 => imgproc.cvtColor(exec, src, view, .gray2bgr) catch return error.Unsupported,
        3 => view.copyFrom(src),
        4 => imgproc.cvtColor(exec, src, view, .bgra2bgr) catch return error.Unsupported,
        else => return error.UnsupportedChannels,
    }
}

fn drawOne(img: Image(u8), k: KeyPoint, dx: f32, color: [3]u8, rich: bool) void {
    const cx: i64 = @intFromFloat(@round(k.x + dx));
    const cy: i64 = @intFromFloat(@round(k.y));
    const c: []const u8 = &color;
    if (rich) {
        const r: f32 = @max(1, k.size / 2);
        const ri: i64 = @intFromFloat(@round(r));
        imgproc.draw.circle(img, cx, cy, ri, c, 1);
        if (k.angle != -1) {
            const a = k.angle * std.math.pi / 180.0;
            const ex: i64 = @intFromFloat(@round(k.x + dx + r * @cos(a)));
            const ey: i64 = @intFromFloat(@round(k.y + r * @sin(a)));
            imgproc.draw.line(img, cx, cy, ex, ey, c, 1);
        }
    } else imgproc.draw.circle(img, cx, cy, 3, c, 1);
}

/// drawKeypoints(src, kps, out, color, flags): `out` is BGR u8 of the same
/// size; null `color` gives every keypoint its own colour.
pub fn drawKeypoints(exec: Exec, src: Image(u8), kps: []const KeyPoint, out: Image(u8), color: ?[3]u8, flags: Flags) Error!void {
    if (out.channels != 3) return error.UnsupportedChannels;
    if (!flags.over_outimg) {
        if (out.width != src.width or out.height != src.height) return error.SizeMismatch;
        try toBgr(exec, src, out, 0);
    }
    for (kps, 0..) |k, i| drawOne(out, k, 0, color orelse autoColor(i), flags.rich_keypoints);
}

/// drawMatches(img1, kp1, img2, kp2, matches, out, ...): `out` is BGR u8 of
/// max(h1, h2) rows and w1 + w2 columns, img1 on the left. Each match
/// joins query keypoint kp1[query_idx] to train keypoint kp2[train_idx].
pub fn drawMatches(exec: Exec, img1: Image(u8), kp1: []const KeyPoint, img2: Image(u8), kp2: []const KeyPoint, matches: []const DMatch, out: Image(u8), match_color: ?[3]u8, single_color: ?[3]u8, flags: Flags) Error!void {
    if (out.channels != 3) return error.UnsupportedChannels;
    const w = img1.width + img2.width;
    const h = @max(img1.height, img2.height);
    if (out.width != w or out.height != h) return error.SizeMismatch;
    for (matches) |m| {
        if (m.query_idx < 0 or m.train_idx < 0 or m.query_idx >= kp1.len or m.train_idx >= kp2.len) return error.InvalidArgument;
    }
    if (!flags.over_outimg) {
        for (0..h) |y| @memset(out.row(y), 0);
        try toBgr(exec, img1, out, 0);
        try toBgr(exec, img2, out, img1.width);
    }
    const dx2: f32 = @floatFromInt(img1.width);
    if (!flags.not_draw_single_points) {
        for (kp1, 0..) |k, i| drawOne(out, k, 0, single_color orelse autoColor(i), flags.rich_keypoints);
        for (kp2, 0..) |k, i| drawOne(out, k, dx2, single_color orelse autoColor(kp1.len + i), flags.rich_keypoints);
    }
    for (matches, 0..) |m, i| {
        const c = match_color orelse autoColor(i);
        const a = kp1[@intCast(m.query_idx)];
        const b = kp2[@intCast(m.train_idx)];
        drawOne(out, a, 0, c, flags.rich_keypoints);
        drawOne(out, b, dx2, c, flags.rich_keypoints);
        imgproc.draw.line(out, @intFromFloat(@round(a.x)), @intFromFloat(@round(a.y)), @intFromFloat(@round(b.x + dx2)), @intFromFloat(@round(b.y)), &c, 1);
    }
}

test "drawMatches lays out both images and joins matches" {
    const gpa = std.testing.allocator;
    var a = try Image(u8).initFill(gpa, 40, 30, 1, 50);
    defer a.deinit(gpa);
    var b = try Image(u8).initFill(gpa, 20, 40, 3, 200);
    defer b.deinit(gpa);
    var out = try Image(u8).init(gpa, 60, 40, 3);
    defer out.deinit(gpa);
    const k1 = [_]KeyPoint{.{ .x = 10, .y = 10, .size = 7 }};
    const k2 = [_]KeyPoint{.{ .x = 5, .y = 20, .size = 7 }};
    const m = [_]DMatch{.{ .query_idx = 0, .train_idx = 0, .distance = 1 }};
    try drawMatches(.serial, a, &k1, b, &k2, &m, out, .{ 0, 255, 0 }, null, .{});
    try std.testing.expectEqual(@as(u8, 50), out.at(0, 0, 2));
    try std.testing.expectEqual(@as(u8, 0), out.at(0, 35, 0)); // below img1
    try std.testing.expectEqual(@as(u8, 200), out.at(59, 39, 1));
    // The line ends at the train keypoint.
    try std.testing.expectEqual(@as(u8, 255), out.at(45, 20, 1));
    var o2 = try Image(u8).init(gpa, 40, 30, 3);
    defer o2.deinit(gpa);
    try drawKeypoints(.serial, a, &k1, o2, .{ 0, 0, 255 }, .{ .rich_keypoints = true });
    try std.testing.expectEqual(@as(u8, 255), o2.at(10 + 4, 10, 2));
}
