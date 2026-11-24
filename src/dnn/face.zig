//! `FaceDetector`: YuNet face detection, a port of OpenCV's FaceDetectorYN
//! (objdetect/src/face_detect.cpp) with the same preprocessing, decoding and
//! NMS, so results match `cv2.FaceDetectorYN` on the same model.
//!
//! The image (BGR, u8) is zero-padded on the right and bottom to multiples
//! of 32 and fed as raw 0..255 floats (blobFromImage with no mean or
//! scale). The network has score, objectness, box and landmark heads at
//! strides 8, 16 and 32; each anchor's score is sqrt(cls * obj), boxes are
//! centre offsets plus log sizes in units of the stride.

const std = @import("std");
const core = @import("../core/core.zig");
const net_mod = @import("net.zig");
const Allocator = std.mem.Allocator;
const Exec = core.Exec;
const Image = core.Image;

/// One detection, laid out like a row of OpenCV's result: box x, y, w, h;
/// right eye, left eye, nose tip, right and left mouth corners (x, y each);
/// score.
pub const Face = extern struct {
    box: [4]f32,
    landmarks: [5][2]f32,
    score: f32,
};

pub const Options = struct {
    width: u32,
    height: u32,
    score_threshold: f32 = 0.9,
    nms_threshold: f32 = 0.3,
    top_k: i32 = 5000,
};

const divisor = 32;
const strides = [3]u32{ 8, 16, 32 };
const head_names = [12][]const u8{ "cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32" };

pub const FaceDetector = struct {
    gpa: Allocator,
    net: net_mod.Net,
    opts: Options,
    pad_w: u32 = 0,
    pad_h: u32 = 0,
    heads: [12]usize,
    candidates: std.ArrayList(Face) = .empty,
    order: std.ArrayList(u32) = .empty,
    result: std.ArrayList(Face) = .empty,

    /// Loads a YuNet ONNX model (`model` must outlive the detector) and
    /// plans it for `opts.width` x `opts.height` images.
    pub fn create(gpa: Allocator, model: []const u8, opts: Options) !FaceDetector {
        var net = try net_mod.Net.load(gpa, model);
        errdefer net.deinit();
        var heads: [12]usize = undefined;
        for (head_names, &heads) |name, *h| h.* = net.outputIndex(name) orelse return error.InvalidModel;
        if (net.inputCount() != 1) return error.InvalidModel;
        var d: FaceDetector = .{ .gpa = gpa, .net = net, .opts = opts, .heads = heads };
        try d.setInputSize(opts.width, opts.height);
        return d;
    }

    pub fn deinit(d: *FaceDetector) void {
        d.candidates.deinit(d.gpa);
        d.order.deinit(d.gpa);
        d.result.deinit(d.gpa);
        d.net.deinit();
        d.* = undefined;
    }

    /// Re-plans the network for a new image size.
    pub fn setInputSize(d: *FaceDetector, width: u32, height: u32) !void {
        if (width == 0 or height == 0) return error.InvalidSize;
        d.opts.width = width;
        d.opts.height = height;
        d.pad_w = ((width - 1) / divisor + 1) * divisor;
        d.pad_h = ((height - 1) / divisor + 1) * divisor;
        try d.net.plan(&.{&.{ 1, 3, d.pad_h, d.pad_w }});
    }

    const PrepCtx = struct { img: Image(u8), blob: []f32, pw: usize, ph: usize };

    /// Detects faces in a BGR image of the configured size. The result is
    /// owned by the detector and valid until the next call.
    pub fn detect(d: *FaceDetector, exec: Exec, img: Image(u8)) ![]const Face {
        if (img.width != d.opts.width or img.height != d.opts.height) return error.SizeMismatch;
        if (img.channels != 3) return error.UnsupportedChannels;
        const blob = d.net.input(0);
        exec.parallelFor(d.pad_h, 16, PrepCtx{ .img = img, .blob = blob, .pw = d.pad_w, .ph = d.pad_h }, struct {
            fn f(c: PrepCtx, s: usize, e: usize) void {
                const plane = c.pw * c.ph;
                for (s..e) |y| {
                    const rows = [3][]f32{ c.blob[y * c.pw ..][0..c.pw], c.blob[plane + y * c.pw ..][0..c.pw], c.blob[2 * plane + y * c.pw ..][0..c.pw] };
                    if (y >= c.img.height) {
                        for (rows) |r| @memset(r, 0);
                        continue;
                    }
                    const src = c.img.data[y * c.img.stride ..][0 .. @as(usize, c.img.width) * 3];
                    const w = c.img.width;
                    var x: usize = 0;
                    // 16 pixels at a time: deinterleave B, G, R and widen.
                    while (x + 16 <= w) : (x += 16) {
                        const v: @Vector(48, u8) = src[x * 3 ..][0..48].*;
                        inline for (0..3) |ch| {
                            const idx = comptime blk: {
                                var m: [16]i32 = undefined;
                                for (&m, 0..) |*q, i| q.* = @intCast(i * 3 + ch);
                                break :blk m;
                            };
                            const p: @Vector(16, u8) = @shuffle(u8, v, undefined, idx);
                            rows[ch][x..][0..16].* = @as(@Vector(16, f32), @floatFromInt(p));
                        }
                    }
                    for (x..w) |xx| {
                        inline for (0..3) |ch| rows[ch][xx] = @floatFromInt(src[xx * 3 + ch]);
                    }
                    for (rows) |r| @memset(r[w..], 0);
                }
            }
        }.f);
        d.net.run(exec);
        return d.postProcess();
    }

    fn postProcess(d: *FaceDetector) ![]const Face {
        d.candidates.clearRetainingCapacity();
        for (strides, 0..) |stride, i| {
            const cols = d.pad_w / stride;
            const rows = d.pad_h / stride;
            const cls = d.net.output(d.heads[i]).data;
            const obj = d.net.output(d.heads[i + 3]).data;
            const bbox = d.net.output(d.heads[i + 6]).data;
            const kps = d.net.output(d.heads[i + 9]).data;
            if (cls.len < rows * cols or bbox.len < rows * cols * 4 or kps.len < rows * cols * 10) return error.InvalidModel;
            const sf: f32 = @floatFromInt(stride);
            for (0..rows) |r| for (0..cols) |c| {
                const idx = r * cols + c;
                const cls_s = std.math.clamp(cls[idx], 0, 1);
                const obj_s = std.math.clamp(obj[idx], 0, 1);
                const score = @sqrt(cls_s * obj_s);
                if (score < d.opts.score_threshold) continue;
                const cf: f32 = @floatFromInt(c);
                const rf: f32 = @floatFromInt(r);
                const cx = (cf + bbox[idx * 4 + 0]) * sf;
                const cy = (rf + bbox[idx * 4 + 1]) * sf;
                const w: f32 = @floatCast(@exp(@as(f64, bbox[idx * 4 + 2])) * @as(f64, sf));
                const h: f32 = @floatCast(@exp(@as(f64, bbox[idx * 4 + 3])) * @as(f64, sf));
                var f: Face = .{ .box = .{ cx - w / 2, cy - h / 2, w, h }, .landmarks = undefined, .score = score };
                for (0..5) |n| f.landmarks[n] = .{ (kps[idx * 10 + 2 * n] + cf) * sf, (kps[idx * 10 + 2 * n + 1] + rf) * sf };
                try d.candidates.append(d.gpa, f);
            };
        }
        if (d.candidates.items.len <= 1) return d.candidates.items;
        // dnn::NMSBoxes over integer rects (Rect2i(int(x), int(y), int(w), int(h))).
        const faces = d.candidates.items;
        d.order.clearRetainingCapacity();
        for (faces, 0..) |f, i| {
            if (f.score > d.opts.score_threshold) try d.order.append(d.gpa, @intCast(i));
        }
        std.sort.block(u32, d.order.items, faces, struct {
            fn gt(fs: []const Face, a: u32, b: u32) bool {
                return fs[a].score > fs[b].score;
            }
        }.gt);
        const top_k: usize = if (d.opts.top_k > 0) @intCast(d.opts.top_k) else std.math.maxInt(usize);
        const cand = d.order.items[0..@min(top_k, d.order.items.len)];
        d.result.clearRetainingCapacity();
        var kept: std.ArrayList(u32) = .empty;
        defer kept.deinit(d.gpa);
        for (cand) |idx| {
            const a = rect(faces[idx]);
            var keep = true;
            for (kept.items) |k| {
                if (overlap(a, rect(faces[k])) > d.opts.nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                try kept.append(d.gpa, idx);
                try d.result.append(d.gpa, faces[idx]);
            }
        }
        return d.result.items;
    }
};

const Rect = struct { x: i32, y: i32, w: i32, h: i32 };

fn toInt(v: f32) i32 {
    // C++ int(float): truncation toward zero (saturated here; NaN as 0).
    if (std.math.isNan(v)) return 0;
    return @intFromFloat(std.math.clamp(@trunc(v), -2147483648.0, 2147483520.0));
}

fn rect(f: Face) Rect {
    return .{ .x = toInt(f.box[0]), .y = toInt(f.box[1]), .w = toInt(f.box[2]), .h = toInt(f.box[3]) };
}

/// 1 - jaccardDistance(a, b), as OpenCV computes it for Rect2i.
fn overlap(a: Rect, b: Rect) f32 {
    const aa: i64 = @as(i64, a.w) * a.h;
    const ab: i64 = @as(i64, b.w) * b.h;
    if (aa + ab <= 0) return 1;
    var inter: i64 = 0;
    if (a.w > 0 and a.h > 0 and b.w > 0 and b.h > 0) {
        const x1 = @max(a.x, b.x);
        const y1 = @max(a.y, b.y);
        const w = @min(a.x + a.w, b.x + b.w) - x1;
        const h = @min(a.y + a.h, b.y + b.h) - y1;
        if (w > 0 and h > 0) inter = @as(i64, w) * h;
    }
    const iab: f64 = @floatFromInt(inter);
    const dist = 1.0 - iab / (@as(f64, @floatFromInt(aa + ab)) - iab);
    return 1.0 - @as(f32, @floatCast(dist));
}

test "integer-rect overlap matches OpenCV's jaccard" {
    try std.testing.expectEqual(@as(f32, 1), overlap(.{ .x = 0, .y = 0, .w = 10, .h = 10 }, .{ .x = 0, .y = 0, .w = 10, .h = 10 }));
    try std.testing.expectEqual(@as(f32, 0), overlap(.{ .x = 0, .y = 0, .w = 10, .h = 10 }, .{ .x = 10, .y = 0, .w = 10, .h = 10 }));
    try std.testing.expectApproxEqAbs(@as(f32, 50.0 / 150.0), overlap(.{ .x = 0, .y = 0, .w = 10, .h = 10 }, .{ .x = 5, .y = 0, .w = 10, .h = 10 }), 1e-7);
    try std.testing.expectEqual(@as(i32, -3), toInt(-3.7));
}
