//! `FaceDetector`: YuNet face detection, a port of OpenCV's FaceDetectorYN
//! (objdetect/src/face_detect.cpp) with the same preprocessing, decoding and
//! NMS, so results match `cv2.FaceDetectorYN` on the same model.
//!
//! The image (BGR or BGRA, u8) is zero-padded on the right and bottom to multiples
//! of 32 and fed as raw 0..255 floats (blobFromImage with no mean or
//! scale). The network has score, objectness, box and landmark heads at
//! strides 8, 16 and 32; each anchor's score is sqrt(cls * obj), boxes are
//! centre offsets plus log sizes in units of the stride.

const std = @import("std");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const net_mod = @import("net.zig");
const prep = @import("prep.zig");
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
    /// Which backend runs the network (see `Net.Options.backend`).
    backend: net_mod.Backend = .auto,
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
    /// Frame pixels per detector-input pixel of the last `prepare`.
    frame_scale: [2]f32 = .{ 1, 1 },
    /// The resized frame, for sizes `prepare` cannot shrink in one pass.
    scratch: ?Image(u8) = null,

    /// Loads a YuNet ONNX model (`model` must outlive the detector) and
    /// plans it for `opts.width` x `opts.height` images.
    pub fn create(gpa: Allocator, model: []const u8, opts: Options) !FaceDetector {
        var net = try net_mod.Net.load(gpa, model);
        errdefer net.deinit();
        var heads: [12]usize = undefined;
        for (head_names, &heads) |name, *h| h.* = net.outputIndex(name) orelse return error.InvalidModel;
        if (net.inputCount() != 1) return error.InvalidModel;
        net.opts.backend = opts.backend;
        var d: FaceDetector = .{ .gpa = gpa, .net = net, .opts = opts, .heads = heads };
        try d.setInputSize(opts.width, opts.height);
        return d;
    }

    pub fn deinit(d: *FaceDetector) void {
        d.candidates.deinit(d.gpa);
        d.order.deinit(d.gpa);
        d.result.deinit(d.gpa);
        if (d.scratch) |*s| s.deinit(d.gpa);
        d.net.deinit();
        d.* = undefined;
    }

    /// Re-plans the network for a new image size.
    pub fn setInputSize(d: *FaceDetector, width: u32, height: u32) !void {
        if (width == 0 or height == 0) return error.InvalidSize;
        d.opts.width = width;
        d.opts.height = height;
        if (d.scratch) |*s| s.deinit(d.gpa);
        d.scratch = null;
        d.pad_w = ((width - 1) / divisor + 1) * divisor;
        d.pad_h = ((height - 1) / divisor + 1) * divisor;
        try d.net.plan(&.{&.{ 1, 3, d.pad_h, d.pad_w }});
    }

    /// Changes the backend and re-plans for the current size.
    pub fn setBackend(d: *FaceDetector, b: net_mod.Backend) !void {
        d.opts.backend = b;
        d.net.opts.backend = b;
        try d.setInputSize(d.opts.width, d.opts.height);
    }

    /// The backend the network runs on.
    pub fn device(d: *const FaceDetector) net_mod.Device {
        return d.net.device().?;
    }

    /// Detects faces in a BGR or BGRA image of the configured size. The
    /// result is owned by the detector and valid until the next call.
    pub fn detect(d: *FaceDetector, exec: Exec, img: Image(u8)) ![]const Face {
        if (img.width != d.opts.width or img.height != d.opts.height) return error.SizeMismatch;
        return d.detectFrame(exec, img);
    }

    /// Detects faces in a BGR or BGRA frame of any size: the frame is
    /// shrunk to the configured size with INTER_AREA (as cv2.resize, so
    /// the result is that of resize + detect) and the boxes and landmarks
    /// are mapped back to frame pixels. `prepare`, `infer` and `decode`
    /// are its three stages.
    pub fn detectFrame(d: *FaceDetector, exec: Exec, frame: Image(u8)) ![]const Face {
        try d.prepare(exec, frame);
        try d.infer(exec);
        return d.decode();
    }

    /// Stage 1: the frame, shrunk to the detector size, into the network
    /// input. Integer factors (3840x2160 to 640x360 or 960x540, ...) take
    /// one fused pass from the frame to the planar f32 tensor; other sizes
    /// are resized into a scratch image first.
    pub fn prepare(d: *FaceDetector, exec: Exec, frame: Image(u8)) !void {
        if (frame.channels != 3 and frame.channels != 4) return error.UnsupportedChannels;
        const w = d.opts.width;
        const h = d.opts.height;
        const t: prep.Target = .{ .planes = d.net.input(0), .pw = d.pad_w, .ph = d.pad_h, .w = w, .h = h };
        if (prep.factors(frame, w, h)) |f| {
            prep.fill(exec, frame, t, f);
        } else {
            // Any other size: INTER_AREA shrinks, bilinear enlarges (as
            // OpenCV's INTER_AREA does).
            if (d.scratch) |s| if (s.channels != frame.channels) {
                d.scratch.?.deinit(d.gpa);
                d.scratch = null;
            };
            if (d.scratch == null) d.scratch = try Image(u8).init(d.gpa, w, h, frame.channels);
            try imgproc.resize(exec, d.gpa, frame, d.scratch.?, .area);
            prep.fill(exec, d.scratch.?, t, .{ 1, 1 });
        }
        d.frame_scale = .{
            @as(f32, @floatFromInt(frame.width)) / @as(f32, @floatFromInt(w)),
            @as(f32, @floatFromInt(frame.height)) / @as(f32, @floatFromInt(h)),
        };
    }

    /// Stage 2: the network.
    pub fn infer(d: *FaceDetector, exec: Exec) !void {
        try d.net.tryRun(exec);
    }

    /// Stage 3: decoding, NMS, and the faces mapped to the prepared
    /// frame's pixels. Owned by the detector, valid until the next call.
    pub fn decode(d: *FaceDetector) ![]const Face {
        const faces = try d.postProcess();
        const sx, const sy = d.frame_scale;
        if (sx != 1 or sy != 1) for (faces) |*f| {
            f.box = .{ f.box[0] * sx, f.box[1] * sy, f.box[2] * sx, f.box[3] * sy };
            for (&f.landmarks) |*l| l.* = .{ l[0] * sx, l[1] * sy };
        };
        return faces;
    }

    fn postProcess(d: *FaceDetector) ![]Face {
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
