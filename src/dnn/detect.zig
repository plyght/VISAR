//! `ObjectDetector`: general object detection with the OpenCV Model Zoo's
//! NanoDet-Plus and YOLOX models (COCO, 80 classes), a port of the zoo's
//! Python pipelines (models/object_detection_nanodet/nanodet.py and
//! models/object_detection_yolox/yolox.py with their demo.py letterbox and
//! unletterbox steps), so results match running those with cv2.
//!
//! - NanoDet: the BGR image goes to RGB, is letterboxed to the model size
//!   (INTER_AREA resize keeping the aspect ratio, centred, zero border) and
//!   normalised per channel as (v - mean) / std. Each head level gives
//!   per-class sigmoid scores and a distribution over 8 bins per box side
//!   (softmax, expectation, times the stride) around the anchor centre.
//! - YOLOX: the RGB image is resized (INTER_LINEAR) to fit, placed at the
//!   top left of a 114-filled canvas and fed as raw 0..255 values. Each
//!   anchor gives centre offset and log size in units of its stride,
//!   objectness and class scores (score = objectness * class).
//!
//! Then the best class per anchor, `cv2.dnn.NMSBoxes` (NanoDet) or
//! `NMSBoxesBatched` (YOLOX) on double-precision boxes, and the boxes mapped
//! back to the input image.
//!
//! One deviation from the zoo code: its NanoDet post-processing pairs the
//! outputs as returned by `getUnconnectedOutLayersNames()`, which OpenCV 5
//! orders differently from OpenCV 4 (cls8, cls16, cls32, box8, ...), so the
//! demo fails there; outputs are paired by level here, as intended.

const std = @import("std");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const net_mod = @import("net.zig");
const Allocator = std.mem.Allocator;
const Exec = core.Exec;
const Image = core.Image;

/// One detection: box x, y, width, height in input-image pixels, score
/// and COCO class index (see `coco_classes`).
pub const Detection = extern struct {
    box: [4]f32,
    score: f32,
    class_id: i32,
};

pub const Arch = enum(u8) { nanodet, yolox };

pub const Options = struct {
    /// Detected from the model's outputs when null.
    arch: ?Arch = null,
    /// Minimum score (exclusive, as NMSBoxes). Default: the zoo demos'
    /// (NanoDet 0.35, YOLOX 0.5).
    score_threshold: ?f32 = null,
    /// IoU above which the lower-scoring box is dropped. Default: the zoo
    /// demos' (NanoDet 0.6, YOLOX 0.5).
    nms_threshold: ?f32 = null,
};

pub const coco_classes = [80][]const u8{
    "person",        "bicycle",    "car",           "motorcycle",    "airplane",     "bus",
    "train",         "truck",      "boat",          "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",      "bird",          "cat",           "dog",          "horse",
    "sheep",         "cow",        "elephant",      "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",   "handbag",       "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",  "sports ball",   "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",  "tennis racket", "bottle",        "wine glass",   "cup",
    "fork",          "knife",      "spoon",         "bowl",          "banana",       "apple",
    "sandwich",      "orange",     "broccoli",      "carrot",        "hot dog",      "pizza",
    "donut",         "cake",       "chair",         "couch",         "potted plant", "bed",
    "dining table",  "toilet",     "tv",            "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone", "microwave",     "oven",          "toaster",      "sink",
    "refrigerator",  "book",       "clock",         "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush",
};

const nanodet_mean = [3]f32{ 103.53, 116.28, 123.675 };
const nanodet_std = [3]f32{ 57.375, 57.12, 58.395 };
const reg_bins = 8;
/// NanoDet keeps the best `nms_pre` anchors of a level (by best class
/// score) before NMS.
const nms_pre = 1000;
const max_levels = 4;

const Candidate = struct {
    /// x, y, w, h (double, as the boxes NMS sees).
    box: [4]f64,
    score: f32,
    class_id: i32,
};

pub const ObjectDetector = struct {
    gpa: Allocator,
    net: net_mod.Net,
    arch: Arch,
    score_threshold: f32,
    nms_threshold: f32,
    /// Model input size.
    in_w: u32,
    in_h: u32,
    classes: usize,
    /// NanoDet: (cls, box) output indices per level, finest first.
    levels: [max_levels][2]usize = undefined,
    nlevels: usize = 0,
    resized: ?Image(u8) = null,
    candidates: std.ArrayList(Candidate) = .empty,
    level_order: std.ArrayList(u32) = .empty,
    order: std.ArrayList(u32) = .empty,
    kept: std.ArrayList(u32) = .empty,
    nms_boxes: std.ArrayList([4]f64) = .empty,
    result: std.ArrayList(Detection) = .empty,

    /// Loads a NanoDet or YOLOX ONNX model (`model` must outlive the
    /// detector) and plans it for its declared input size.
    pub fn create(gpa: Allocator, model: []const u8, opts: Options) !ObjectDetector {
        var net = try net_mod.Net.load(gpa, model);
        errdefer net.deinit();
        if (net.inputCount() != 1) return error.InvalidModel;
        const dims = net.inputDims(0);
        if (dims.len != 4) return error.InvalidModel;
        const ih: u32 = @intCast(dims[2] orelse return error.InvalidModel);
        const iw: u32 = @intCast(dims[3] orelse return error.InvalidModel);
        try net.plan(&.{&.{ 1, 3, ih, iw }});
        const arch = opts.arch orelse if (net.outputCount() == 1) Arch.yolox else Arch.nanodet;
        var d: ObjectDetector = .{
            .gpa = gpa,
            .net = net,
            .arch = arch,
            .score_threshold = opts.score_threshold orelse if (arch == .nanodet) 0.35 else 0.5,
            .nms_threshold = opts.nms_threshold orelse if (arch == .nanodet) 0.6 else 0.5,
            .in_w = iw,
            .in_h = ih,
            .classes = 0,
        };
        try d.findHeads();
        return d;
    }

    pub fn deinit(d: *ObjectDetector) void {
        if (d.resized) |*r| r.deinit(d.gpa);
        d.candidates.deinit(d.gpa);
        d.level_order.deinit(d.gpa);
        d.order.deinit(d.gpa);
        d.kept.deinit(d.gpa);
        d.nms_boxes.deinit(d.gpa);
        d.result.deinit(d.gpa);
        d.net.deinit();
        d.* = undefined;
    }

    fn findHeads(d: *ObjectDetector) !void {
        const net = &d.net;
        switch (d.arch) {
            .yolox => {
                if (net.outputCount() != 1) return error.InvalidModel;
                const s = net.output(0).shape;
                if (s.rank != 3 or s.d[2] <= 5) return error.InvalidModel;
                d.classes = s.d[2] - 5;
                var n: usize = 0;
                for ([_]u32{ 8, 16, 32 }) |st| n += (d.in_h / st) * (d.in_w / st);
                if (s.d[1] != n) return error.InvalidModel;
            },
            .nanodet => {
                // Score outputs [1, N, classes] and box outputs [1, N, 4 * 8]
                // paired by anchor count, finest level first.
                const no = net.outputCount();
                if (no % 2 != 0 or no / 2 > max_levels) return error.InvalidModel;
                var cls: [max_levels]usize = undefined;
                var box: [max_levels]usize = undefined;
                var nc: usize = 0;
                var nb: usize = 0;
                for (0..no) |i| {
                    const s = net.output(i).shape;
                    if (s.rank != 3) return error.InvalidModel;
                    if (s.d[2] == 4 * reg_bins) {
                        if (nb == max_levels) return error.InvalidModel;
                        box[nb] = i;
                        nb += 1;
                    } else {
                        if (nc == max_levels) return error.InvalidModel;
                        cls[nc] = i;
                        nc += 1;
                        if (d.classes != 0 and d.classes != s.d[2]) return error.InvalidModel;
                        d.classes = s.d[2];
                    }
                }
                if (nc != nb or nc == 0) return error.InvalidModel;
                const Ctx = struct {
                    net: *net_mod.Net,
                    fn gt(c: @This(), a: usize, b: usize) bool {
                        return c.net.output(a).shape.d[1] > c.net.output(b).shape.d[1];
                    }
                };
                std.sort.insertion(usize, cls[0..nc], Ctx{ .net = net }, Ctx.gt);
                std.sort.insertion(usize, box[0..nb], Ctx{ .net = net }, Ctx.gt);
                const strides = [max_levels]u32{ 8, 16, 32, 64 };
                for (0..nc) |l| {
                    const n = (d.in_h / strides[l]) * (d.in_w / strides[l]);
                    if (net.output(cls[l]).shape.d[1] != n or net.output(box[l]).shape.d[1] != n) return error.InvalidModel;
                    d.levels[l] = .{ cls[l], box[l] };
                }
                d.nlevels = nc;
            },
        }
    }

    /// Letterbox geometry: where the resized image sits in the model input.
    const Box = struct { top: u32, left: u32, h: u32, w: u32, ratio: f64 };

    fn letterbox(d: *const ObjectDetector, w: u32, h: u32) Box {
        const W = d.in_w;
        const H = d.in_h;
        const wf: f64 = @floatFromInt(w);
        const hf: f64 = @floatFromInt(h);
        switch (d.arch) {
            .nanodet => {
                if (w == h) return .{ .top = 0, .left = 0, .h = H, .w = W, .ratio = 0 };
                const hw = hf / wf;
                if (hw > 1) {
                    const nw: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(W)) / hw)));
                    return .{ .top = 0, .left = @intFromFloat(@as(f64, @floatFromInt(W - nw)) * 0.5), .h = H, .w = nw, .ratio = 0 };
                }
                const nh: u32 = @max(1, @as(u32, @intFromFloat(@as(f64, @floatFromInt(H)) * hw)));
                return .{ .top = @intFromFloat(@as(f64, @floatFromInt(H - nh)) * 0.5), .left = 0, .h = nh, .w = W, .ratio = 0 };
            },
            .yolox => {
                const r = @min(@as(f64, @floatFromInt(H)) / hf, @as(f64, @floatFromInt(W)) / wf);
                const nw: u32 = @max(1, @as(u32, @intFromFloat(wf * r)));
                const nh: u32 = @max(1, @as(u32, @intFromFloat(hf * r)));
                return .{ .top = 0, .left = 0, .h = @min(nh, H), .w = @min(nw, W), .ratio = r };
            },
        }
    }

    const PrepCtx = struct { d: *const ObjectDetector, src: Image(u8), lb: Box, blob: []f32 };

    /// Detects objects in a BGR image of any size. The result is owned by
    /// the detector and valid until the next call.
    pub fn detect(d: *ObjectDetector, exec: Exec, img: Image(u8)) ![]const Detection {
        if (img.channels != 3) return error.UnsupportedChannels;
        if (img.width == 0 or img.height == 0) return error.InvalidSize;
        const lb = d.letterbox(img.width, img.height);
        var src = img;
        if (lb.w != img.width or lb.h != img.height) {
            if (d.resized == null or d.resized.?.width != lb.w or d.resized.?.height != lb.h) {
                if (d.resized) |*r| r.deinit(d.gpa);
                d.resized = null;
                d.resized = try Image(u8).init(d.gpa, lb.w, lb.h, 3);
            }
            try imgproc.resize(exec, d.gpa, img, d.resized.?, if (d.arch == .nanodet) .area else .linear);
            src = d.resized.?;
        }
        const blob = d.net.input(0);
        exec.parallelFor(d.in_h, 8, PrepCtx{ .d = d, .src = src, .lb = lb, .blob = blob }, struct {
            fn f(c: PrepCtx, y0: usize, y1: usize) void {
                for (y0..y1) |y| c.d.prepRow(c.src, c.lb, c.blob, y);
            }
        }.f);
        d.net.run(exec);
        switch (d.arch) {
            .nanodet => try d.decodeNanoDet(),
            .yolox => try d.decodeYolox(),
        }
        try d.nms();
        d.result.clearRetainingCapacity();
        const wf: f64 = @floatFromInt(img.width);
        const hf: f64 = @floatFromInt(img.height);
        for (d.kept.items) |i| {
            const c = d.candidates.items[i];
            try d.result.append(d.gpa, .{ .box = d.unletterbox(c, lb, wf, hf), .score = c.score, .class_id = c.class_id });
        }
        return d.result.items;
    }

    /// One row of the input blob (RGB planes) from the resized BGR image.
    fn prepRow(d: *const ObjectDetector, src: Image(u8), lb: Box, blob: []f32, y: usize) void {
        const W: usize = d.in_w;
        const plane = W * d.in_h;
        const rows = [3][]f32{ blob[y * W ..][0..W], blob[plane + y * W ..][0..W], blob[2 * plane + y * W ..][0..W] };
        const inside = y >= lb.top and y < lb.top + lb.h;
        const x0: usize = if (inside) lb.left else W;
        const x1: usize = if (inside) lb.left + lb.w else W;
        for (0..3) |ch| {
            // Background: zero pixels normalised (NanoDet) or 114 (YOLOX).
            const bg: f32 = if (d.arch == .nanodet) (0 - nanodet_mean[ch]) / nanodet_std[ch] else 114;
            @memset(rows[ch][0..x0], bg);
            @memset(rows[ch][x1..], bg);
        }
        if (!inside) return;
        const s = src.data[(y - lb.top) * src.stride ..][0 .. @as(usize, lb.w) * 3];
        for (0..3) |ch| {
            const o = rows[ch][x0..x1];
            const sc = 2 - ch; // RGB plane ch from BGR channel 2 - ch
            if (d.arch == .nanodet) {
                const m = nanodet_mean[ch];
                const sd = nanodet_std[ch];
                for (o, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(s[i * 3 + sc])) - m) / sd;
            } else {
                for (o, 0..) |*v, i| v.* = @floatFromInt(s[i * 3 + sc]);
            }
        }
    }

    fn decodeNanoDet(d: *ObjectDetector) !void {
        d.candidates.clearRetainingCapacity();
        const strides = [max_levels]u32{ 8, 16, 32, 64 };
        const nc = d.classes;
        const max_w: f64 = @floatFromInt(d.in_w);
        const max_h: f64 = @floatFromInt(d.in_h);
        for (0..d.nlevels) |l| {
            const stride = strides[l];
            const sf: f64 = @floatFromInt(stride);
            const fw = d.in_w / stride;
            const cls = d.net.output(d.levels[l][0]).data;
            const reg = d.net.output(d.levels[l][1]).data;
            const n = cls.len / nc;
            // Anchors in decreasing order of best score when the level has
            // more than nms_pre (numpy's argsort()[::-1][:nms_pre]).
            d.level_order.clearRetainingCapacity();
            for (0..n) |i| try d.level_order.append(d.gpa, @intCast(i));
            if (n > nms_pre) {
                const Ctx = struct {
                    cls: []const f32,
                    nc: usize,
                    fn best(c: @This(), i: u32) f32 {
                        return std.mem.max(f32, c.cls[i * c.nc ..][0..c.nc]);
                    }
                    fn gt(c: @This(), a: u32, b: u32) bool {
                        return c.best(a) > c.best(b);
                    }
                };
                std.sort.block(u32, d.level_order.items, Ctx{ .cls = cls, .nc = nc }, Ctx.gt);
                d.level_order.shrinkRetainingCapacity(nms_pre);
            }
            for (d.level_order.items) |ai| {
                const i: usize = ai;
                const sc = cls[i * nc ..][0..nc];
                // np.max / np.argmax: the first maximum.
                var best: usize = 0;
                for (sc, 0..) |v, k| {
                    if (v > sc[best]) best = k;
                }
                // Side distances: softmax over the bins in f32 (numpy's
                // pairwise order for the sum of 8), expectation in f64.
                var dist: [4]f64 = undefined;
                for (0..4) |side| {
                    const r = reg[(i * 4 + side) * reg_bins ..][0..reg_bins];
                    var e: [reg_bins]f32 = undefined;
                    for (&e, r) |*o, v| o.* = @exp(v);
                    const sum = ((e[0] + e[1]) + (e[2] + e[3])) + ((e[4] + e[5]) + (e[6] + e[7]));
                    var acc: f64 = 0;
                    for (e, 0..) |v, k| acc += @as(f64, v / sum) * @as(f64, @floatFromInt(k));
                    dist[side] = acc * sf;
                }
                const cx = @as(f64, @floatFromInt((i % fw) * stride)) + 0.5 * (sf - 1);
                const cy = @as(f64, @floatFromInt((i / fw) * stride)) + 0.5 * (sf - 1);
                const x1 = std.math.clamp(cx - dist[0], 0, max_w);
                const y1 = std.math.clamp(cy - dist[1], 0, max_h);
                const x2 = std.math.clamp(cx + dist[2], 0, max_w);
                const y2 = std.math.clamp(cy + dist[3], 0, max_h);
                try d.candidates.append(d.gpa, .{ .box = .{ x1, y1, x2 - x1, y2 - y1 }, .score = sc[best], .class_id = @intCast(best) });
            }
        }
    }

    fn decodeYolox(d: *ObjectDetector) !void {
        d.candidates.clearRetainingCapacity();
        const out = d.net.output(0).data;
        const row = d.classes + 5;
        var i: usize = 0;
        for ([_]u32{ 8, 16, 32 }) |stride| {
            const fw = d.in_w / stride;
            const fh = d.in_h / stride;
            const sf: f64 = @floatFromInt(stride);
            for (0..fh * fw) |a| {
                const p = out[i * row ..][0..row];
                i += 1;
                const gx: f64 = @floatFromInt(a % fw);
                const gy: f64 = @floatFromInt(a / fw);
                // numpy: f32 outputs, offsets and sizes in f64 stored back
                // to f32, corners in f32.
                const cx: f32 = @floatCast((@as(f64, p[0]) + gx) * sf);
                const cy: f32 = @floatCast((@as(f64, p[1]) + gy) * sf);
                const w: f32 = @floatCast(@as(f64, @exp(p[2])) * sf);
                const h: f32 = @floatCast(@as(f64, @exp(p[3])) * sf);
                var best: usize = 0;
                var best_s: f32 = p[4] * p[5];
                for (1..d.classes) |k| {
                    const s = p[4] * p[5 + k];
                    if (s > best_s) {
                        best_s = s;
                        best = k;
                    }
                }
                try d.candidates.append(d.gpa, .{ .box = .{ cx - w / 2.0, cy - h / 2.0, w, h }, .score = best_s, .class_id = @intCast(best) });
            }
        }
    }

    /// cv2.dnn.NMSBoxes over Rect2d (NanoDet), or NMSBoxesBatched (YOLOX:
    /// boxes offset by class so different classes never overlap).
    fn nms(d: *ObjectDetector) !void {
        const cand = d.candidates.items;
        d.nms_boxes.clearRetainingCapacity();
        try d.nms_boxes.ensureTotalCapacity(d.gpa, cand.len);
        if (d.arch == .yolox) {
            var max_coord: f64 = 0;
            for (cand) |c| {
                const b = c.box;
                max_coord = @max(max_coord, b[0], b[1], b[0] + b[2], b[1] + b[3]);
            }
            for (cand) |c| {
                const off = @as(f64, @floatFromInt(c.class_id)) * (max_coord + 1);
                d.nms_boxes.appendAssumeCapacity(.{ c.box[0] + off, c.box[1] + off, c.box[2], c.box[3] });
            }
        } else {
            for (cand) |c| d.nms_boxes.appendAssumeCapacity(c.box);
        }
        d.order.clearRetainingCapacity();
        for (cand, 0..) |c, i| {
            if (c.score > d.score_threshold) try d.order.append(d.gpa, @intCast(i));
        }
        // std::stable_sort by descending score.
        std.sort.block(u32, d.order.items, cand, struct {
            fn gt(cs: []const Candidate, a: u32, b: u32) bool {
                return cs[a].score > cs[b].score;
            }
        }.gt);
        d.kept.clearRetainingCapacity();
        const boxes = d.nms_boxes.items;
        for (d.order.items) |idx| {
            var keep = true;
            for (d.kept.items) |k| {
                if (overlap(boxes[idx], boxes[k]) > d.nms_threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) try d.kept.append(d.gpa, idx);
        }
    }

    /// Back to input-image coordinates, as the zoo demos' unletterbox (the
    /// NanoDet demo then truncates to int; kept fractional here).
    fn unletterbox(d: *const ObjectDetector, c: Candidate, lb: Box, w: f64, h: f64) [4]f32 {
        const b = c.box;
        switch (d.arch) {
            .yolox => {
                // f32 box / ratio, as numpy computes on the f32 candidates.
                const r: f32 = @floatCast(lb.ratio);
                return .{ @as(f32, @floatCast(b[0])) / r, @as(f32, @floatCast(b[1])) / r, @as(f32, @floatCast(b[2])) / r, @as(f32, @floatCast(b[3])) / r };
            },
            .nanodet => {
                var x1 = b[0];
                var y1 = b[1];
                var x2 = b[0] + b[2];
                var y2 = b[1] + b[3];
                if (w == h) {
                    const r = h / @as(f64, @floatFromInt(lb.h));
                    x1 *= r;
                    y1 *= r;
                    x2 *= r;
                    y2 *= r;
                } else {
                    const rh = h / @as(f64, @floatFromInt(lb.h));
                    const rw = w / @as(f64, @floatFromInt(lb.w));
                    const top: f64 = @floatFromInt(lb.top);
                    const left: f64 = @floatFromInt(lb.left);
                    x1 = @max((x1 - left) * rw, 0);
                    y1 = @max((y1 - top) * rh, 0);
                    x2 = @min((x2 - left) * rw, w);
                    y2 = @min((y2 - top) * rh, h);
                }
                return .{ @floatCast(x1), @floatCast(y1), @floatCast(x2 - x1), @floatCast(y2 - y1) };
            },
        }
    }
};

/// 1 - jaccardDistance(a, b) for Rect2d (x, y, w, h), as OpenCV computes it.
fn overlap(a: [4]f64, b: [4]f64) f32 {
    const aa = a[2] * a[3];
    const ab = b[2] * b[3];
    if (aa + ab <= std.math.floatEps(f64)) return 1;
    var inter: f64 = 0;
    const x1 = @max(a[0], b[0]);
    const y1 = @max(a[1], b[1]);
    const iw = @min(a[0] + a[2], b[0] + b[2]) - x1;
    const ih = @min(a[1] + a[3], b[1] + b[3]) - y1;
    if (iw > 0 and ih > 0) inter = iw * ih;
    const dist = 1.0 - inter / (aa + ab - inter);
    return 1.0 - @as(f32, @floatCast(dist));
}

test "Rect2d overlap matches OpenCV's jaccard" {
    try std.testing.expectEqual(@as(f32, 1), overlap(.{ 0, 0, 10, 10 }, .{ 0, 0, 10, 10 }));
    try std.testing.expectEqual(@as(f32, 0), overlap(.{ 0, 0, 10, 10 }, .{ 10, 0, 10, 10 }));
    try std.testing.expectApproxEqAbs(@as(f32, 99.0 / 111.0), overlap(.{ 0, 0, 10.5, 10 }, .{ 0.6, 0, 10.5, 10 }), 1e-6);
}
