//! `visar faces`: YuNet face detection on an image, or face tracking
//! through a video (a .y4m file, or several images as its frames).
//!
//! An image goes through `visar.dnn.FaceDetector` once. A video goes
//! through track_lib's `FaceTracker`, as `visar track --faces` runs it: the
//! detector every --detect-every frames (postponed while nothing changes,
//! and mostly only around the tracked faces: --change-level), MOSSE filters
//! in between. Faces are printed one per line (box, score and the five
//! landmarks, in input pixels) or as JSON; --out writes the input with the
//! faces drawn on it (an animated GIF for a video).

const std = @import("std");
const visar = @import("visar");
const track = @import("track_lib");
const args = @import("args.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const imageio = @import("imageio.zig");
const term = @import("term.zig");

const Io = std.Io;
const Image = visar.Image;
const Face = visar.dnn.Face;
const draw = visar.imgproc.draw;
const y4m = visar.videoio.y4m;

pub const opts = [_]args.Opt{
    .{ .long = "model", .value = "FILE", .help = "YuNet model (default: face_detection_yunet_2023mar.onnx in VISAR_MODELS)" },
    .{ .long = "out", .value = "FILE", .help = "Write the input with the faces drawn (a GIF for a video); or -o" },
    .{ .long = "face-width", .value = "N", .help = "Detector width (default: image up to 1280, video 640; 0 = full size)" },
    .{ .long = "score", .value = "S", .help = "Lowest face score, 0-1 (default 0.9)" },
    .{ .long = "detect-every", .value = "N", .help = "Video: detector every N frames, MOSSE in between (default 4)" },
    .{ .long = "change-level", .value = "N", .help = "Video: 0 off, 1 skip unchanged frames, 2 also detect near faces only" },
    .{ .long = "full-gray", .help = "Video: whole gray tracking image every frame" },
    .{ .long = "frames", .short = 'n', .value = "N", .help = "Video: stop after N frames" },
};

pub const model_name = "face_detection_yunet_2023mar.onnx";
pub const landmark_names = [5][]const u8{ "right_eye", "left_eye", "nose", "mouth_right", "mouth_left" };

/// GIF frames are shrunk to at most this width.
const gif_max_width = 960;

const Config = struct {
    face_width: ?u32 = null,
    score: f32 = 0.9,
    face: track.FaceOptions = .{},
    frame_limit: ?u64 = null,
};

pub fn run(ctx: *Ctx, p: *args.Parsed) !void {
    const paths = p.positionals.items;
    if (paths.len == 0) {
        ctx.hint("e.g. 'visar faces selfie.jpg', 'visar faces clip.y4m --json' or 'visar faces f1.png f2.png f3.png'", .{});
        return ctx.usage("faces needs an image, a .y4m video or several images (the frames of a video)", .{});
    }
    var cfg: Config = .{};
    if (p.value("face-width")) |v| cfg.face_width = try args.int(u32, ctx.diag, "--face-width", v, 0, 8192);
    if (p.value("score")) |v| {
        const s = std.fmt.parseFloat(f32, v) catch return ctx.usage("--score must be a number from 0 to 1, not '{s}'", .{v});
        if (!(s >= 0 and s <= 1)) return ctx.usage("--score must be from 0 to 1", .{});
        cfg.score = s;
    }
    if (p.value("detect-every")) |v| cfg.face.detect_every = try args.int(u32, ctx.diag, "--detect-every", v, 1, 1000);
    if (p.value("change-level")) |v| cfg.face.level = try args.int(u8, ctx.diag, "--change-level", v, 0, 2);
    if (p.flag("full-gray")) cfg.face.partial_gray = false;
    if (p.value("frames")) |v| cfg.frame_limit = try args.int(u64, ctx.diag, "--frames", v, 1, std.math.maxInt(u32));
    cfg.face.score_threshold = cfg.score;

    if (p.value("out") != null and ctx.output != null) return ctx.usage("give the output once, with --out or -o", .{});
    const out_path = p.value("out") orelse ctx.output;
    for (paths) |path| if (std.mem.eql(u8, path, "-") and paths.len > 1) return ctx.usage("'-' (stdin) reads one image or video, not a frame of several", .{});

    const video = paths.len > 1 or isY4mPath(paths[0]);
    if (!video) for ([_][]const u8{ "detect-every", "change-level", "frames" }) |o| {
        if (p.value(o) != null) ctx.warn("--{s} applies to videos; ignored for an image", .{o});
    };
    if (!video and p.flag("full-gray")) ctx.warn("--full-gray applies to videos; ignored for an image", .{});

    var out: ?imageio.Output = null;
    if (out_path) |path| {
        if (video) {
            if (ctx.format) |f| if (f != .gif) {
                ctx.hint("a video is written as an animated GIF; drop --format", .{});
                return ctx.usage("--format {s} does not apply to a video", .{@tagName(f)});
            };
            if (std.mem.eql(u8, path, "-")) return ctx.usage("--out needs a file name for a video", .{});
            ctx.format = .gif;
        }
        out = try imageio.prepare(ctx, path);
    }

    const model = try loadModel(ctx, p.value("model"));
    defer ctx.gpa.free(model);

    // '-': an image, or a .y4m video, from stdin.
    if (std.mem.eql(u8, paths[0], "-")) {
        const bytes = try imageio.readAll(ctx, "-");
        defer ctx.gpa.free(bytes);
        if (y4m.detect(bytes)) {
            var r: Io.Reader = .fixed(bytes);
            return runVideo(ctx, cfg, model, .{ .stream = &r }, "<stdin>", out);
        }
        var in = try imageio.decodeBytes(ctx, "-", bytes, .color);
        defer in.deinit(ctx.gpa);
        return runImage(ctx, cfg, model, in.img.u8, "<stdin>", out);
    }
    if (paths.len > 1) return runVideo(ctx, cfg, model, .{ .images = paths }, paths[0], out);
    if (video) {
        const file = Io.Dir.cwd().openFile(ctx.io, paths[0], .{}) catch |e| return ctx.fail("cannot read '{s}': {s}", .{ paths[0], cx.describe(e) });
        defer file.close(ctx.io);
        var buf: [64 * 1024]u8 = undefined;
        var fr = file.readerStreaming(ctx.io, &buf);
        return runVideo(ctx, cfg, model, .{ .stream = &fr.interface }, paths[0], out);
    }
    var in = try imageio.load(ctx, paths[0], .color);
    defer in.deinit(ctx.gpa);
    return runImage(ctx, cfg, model, in.img.u8, paths[0], out);
}

fn isY4mPath(path: []const u8) bool {
    return std.ascii.endsWithIgnoreCase(path, ".y4m");
}

/// The model bytes: --model, or the YuNet file in $VISAR_MODELS.
fn loadModel(ctx: *Ctx, given: ?[]const u8) ![]u8 {
    if (given) |path| return imageio.readAll(ctx, path);
    const dir = ctx.env.get("VISAR_MODELS") orelse "";
    if (dir.len == 0) {
        ctx.hint("download it from {s}", .{@import("cmd_track.zig").yunet_url});
        return ctx.usage("give the YuNet model with --model FILE, or set VISAR_MODELS to the directory holding {s}", .{model_name});
    }
    const path = try std.fs.path.join(ctx.arena, &.{ dir, model_name });
    return imageio.readAll(ctx, path);
}

fn createFailed(ctx: *Ctx, e: anyerror) error{Failed} {
    if (e == error.OutOfMemory) return ctx.fail("cannot load the face model: out of memory", .{});
    ctx.hint("the model is face_detection_yunet_2023mar.onnx from {s}", .{@import("cmd_track.zig").yunet_url});
    return ctx.fail("cannot load the model as a YuNet face detector: {s}", .{cx.describe(e)});
}

/// The detector size for a `w` x `h` input: `want` wide (0 = full size; by
/// default the input's width up to `cap`), the height following the aspect
/// ratio.
pub fn detectorSize(w: u32, h: u32, want: ?u32, cap: u32) [2]u32 {
    const dw = std.math.clamp(if (want) |v| (if (v == 0) w else v) else @min(w, cap), 32, @max(32, w));
    const dh_f = @round(@as(f64, @floatFromInt(dw)) * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(w)));
    return .{ dw, @max(16, @as(u32, @intFromFloat(dh_f))) };
}

// ---------------------------------------------------------------------------
// One image
// ---------------------------------------------------------------------------

fn runImage(ctx: *Ctx, cfg: Config, model: []const u8, rgb: Image(u8), name: []const u8, out: ?imageio.Output) !void {
    const gpa = ctx.gpa;
    const exec = ctx.exec();
    var bgr = Image(u8).init(gpa, rgb.width, rgb.height, 3) catch return ctx.fail("out of memory", .{});
    defer bgr.deinit(gpa);
    try visar.imgproc.cvtColor(exec, rgb, bgr, .rgb2bgr);
    const size = detectorSize(rgb.width, rgb.height, cfg.face_width, 1280);
    var det = visar.dnn.FaceDetector.create(gpa, model, .{ .width = size[0], .height = size[1], .score_threshold = cfg.score }) catch |e| return createFailed(ctx, e);
    defer det.deinit();
    const t0 = Io.Timestamp.now(ctx.io, .awake);
    const faces = det.detectFrame(exec, bgr) catch |e| return ctx.fail("face detection failed: {s}", .{cx.describe(e)});
    const ms = term.msSince(ctx.io, t0);

    const w = ctx.out;
    if (ctx.json) {
        var list: std.ArrayList(JsonFace) = .empty;
        for (faces) |f| try list.append(ctx.arena, .of(f, null));
        try std.json.Stringify.value(ImageReport{
            .input = name,
            .width = rgb.width,
            .height = rgb.height,
            .detector = size,
            .ms = ms,
            .faces = list.items,
        }, .{ .whitespace = .indent_2 }, w);
        try w.writeAll("\n");
    } else {
        if (!ctx.quiet) {
            ctx.se.print(ctx.err, .dim, "  {s}: {d} face{s} · {d}x{d}, detector {d}x{d} · {d:.1} ms\n", .{ name, faces.len, if (faces.len == 1) "" else "s", rgb.width, rgb.height, size[0], size[1], ms }) catch {};
            ctx.err.flush() catch {};
        }
        for (faces, 1..) |f, i| try writeFace(w, null, @intCast(i), f);
    }
    if (out) |o| {
        var img = try rgb.clone(gpa);
        defer img.deinit(gpa);
        for (faces, 1..) |f, i| drawFace(img, .rgb, f, @intCast(i), true);
        const saved = try imageio.save(ctx, o, .{ .u8 = img });
        if (!ctx.quiet and !ctx.json) {
            var b: [16]u8 = undefined;
            ctx.se.print(ctx.err, .ok, "  {s} {s}", .{ ctx.se.glyph("✓", "ok"), o.name() }) catch {};
            ctx.se.print(ctx.err, .dim, " · {s} {s}\n", .{ cx.formatName(o.format), term.fmtBytes(&b, saved.bytes) }) catch {};
            ctx.err.flush() catch {};
        }
    }
}

// ---------------------------------------------------------------------------
// A video
// ---------------------------------------------------------------------------

const VideoIn = union(enum) {
    /// A .y4m stream.
    stream: *Io.Reader,
    /// Image files, one per frame.
    images: []const []const u8,
};

/// Frames of a video, as BGRA.
const Frames = struct {
    ctx: *Ctx,
    src: union(enum) { y4m: y4m.Reader, images: []const []const u8 },
    width: u32,
    height: u32,
    fps: f64 = 0,
    index: usize = 0,
    /// The first image, already decoded (image sequences).
    first: ?Image(u8) = null,

    fn open(ctx: *Ctx, in: VideoIn, name: []const u8) !Frames {
        switch (in) {
            .stream => |r| {
                const rd = y4m.Reader.init(ctx.gpa, r) catch |e| {
                    if (e == error.Unsupported) ctx.hint("convert it: ffmpeg -i in.y4m -pix_fmt yuv420p out.y4m", .{});
                    return ctx.fail("cannot read '{s}' as a YUV4MPEG2 video: {s}", .{ name, cx.describe(e) });
                };
                const fps: f64 = if (rd.fps_den > 0) @as(f64, @floatFromInt(rd.fps_num)) / @as(f64, @floatFromInt(rd.fps_den)) else 0;
                return .{ .ctx = ctx, .src = .{ .y4m = rd }, .width = rd.width, .height = rd.height, .fps = fps };
            },
            .images => |paths| {
                var in0 = try imageio.load(ctx, paths[0], .color);
                defer in0.deinit(ctx.gpa);
                const im = in0.img.u8;
                var bgra = try Image(u8).init(ctx.gpa, im.width, im.height, 4);
                errdefer bgra.deinit(ctx.gpa);
                try visar.imgproc.cvtColor(ctx.exec(), im, bgra, .rgb2bgra);
                return .{ .ctx = ctx, .src = .{ .images = paths }, .width = im.width, .height = im.height, .first = bgra };
            },
        }
    }

    fn deinit(self: *Frames) void {
        switch (self.src) {
            .y4m => |*r| r.deinit(),
            .images => {},
        }
        if (self.first) |*f| f.deinit(self.ctx.gpa);
    }

    /// The next frame into `dst` (BGRA, the video's size); false at the end.
    fn next(self: *Frames, dst: Image(u8), name: []const u8) !bool {
        const ctx = self.ctx;
        defer self.index += 1;
        switch (self.src) {
            .y4m => |*r| return r.next(dst) catch |e| ctx.fail("cannot read frame {d} of '{s}': {s}", .{ self.index + 1, name, cx.describe(e) }),
            .images => |paths| {
                if (self.index >= paths.len) return false;
                if (self.index == 0) {
                    dst.copyFrom(self.first.?);
                    return true;
                }
                const path = paths[self.index];
                var in = try imageio.load(ctx, path, .color);
                defer in.deinit(ctx.gpa);
                const im = in.img.u8;
                if (im.width != self.width or im.height != self.height) {
                    ctx.hint("every frame must have the size of the first ({d}x{d})", .{ self.width, self.height });
                    return ctx.fail("'{s}' is {d}x{d}", .{ path, im.width, im.height });
                }
                try visar.imgproc.cvtColor(ctx.exec(), im, dst, .rgb2bgra);
                return true;
            },
        }
    }
};

fn runVideo(ctx: *Ctx, cfg: Config, model: []const u8, in: VideoIn, name: []const u8, out: ?imageio.Output) !void {
    const gpa = ctx.gpa;
    const exec = ctx.exec();
    var frames = try Frames.open(ctx, in, name);
    defer frames.deinit();
    const W = frames.width;
    const H = frames.height;
    if (W < 32 or H < 32) return ctx.fail("'{s}' is {d}x{d}; face tracking needs at least 32x32", .{ name, W, H });
    var fo = cfg.face;
    fo.face_width = detectorSize(W, H, cfg.face_width, 640)[0];
    // The tracking image: the detector's scale (as `visar track` picks it).
    const k: u32 = @max(1, W / fo.face_width);
    var ft = track.FaceTracker.create(gpa, model, W, H, k, fo) catch |e| return createFailed(ctx, e);
    defer ft.deinit();
    var frame = try Image(u8).init(gpa, W, H, 4);
    defer frame.deinit(gpa);
    var small = try Image(u8).init(gpa, W / k, H / k, 1);
    defer small.deinit(gpa);

    var gif: ?Gif = if (out != null) try Gif.init(gpa, W, H) else null;
    defer if (gif) |*g| g.deinit(gpa);
    const delay_cs: u16 = if (frames.fps > 0) @intFromFloat(std.math.clamp(@round(100 / frames.fps), 2, 6000)) else 4;

    if (!ctx.quiet and !ctx.json) {
        ctx.se.print(ctx.err, .dim, "  {s}: {d}x{d}", .{ name, W, H }) catch {};
        if (frames.fps > 0) ctx.se.print(ctx.err, .dim, " at {d:.2} fps", .{frames.fps}) catch {};
        ctx.se.print(ctx.err, .dim, " · detector {d}x{d}, tracking at {d}x{d} on {d} thread{s}\n", .{ ft.det_w, ft.det_h, W / k, H / k, exec.threads(), if (exec.threads() == 1) "" else "s" }) catch {};
        ctx.err.flush() catch {};
    }

    const w = ctx.out;
    var json_frames: std.ArrayList(JsonFrame) = .empty;
    var n: u64 = 0;
    var ms_sum: f64 = 0;
    var ms_max: f64 = 0;
    var with_faces: u64 = 0;
    while (cfg.frame_limit == null or n < cfg.frame_limit.?) {
        if (!try frames.next(frame, name)) break;
        const t0 = Io.Timestamp.now(ctx.io, .awake);
        try ft.trackingImage(exec, frame, small);
        const what = ft.step(exec, null, frame, small) catch |e| return ctx.fail("face tracking failed on frame {d}: {s}", .{ n + 1, cx.describe(e) });
        const ms = term.msSince(ctx.io, t0);
        ms_sum += ms;
        ms_max = @max(ms_max, ms);
        n += 1;
        const tracks = ft.faces();
        if (tracks.len > 0) with_faces += 1;
        if (ctx.json) {
            const list = try ctx.arena.alloc(JsonFace, tracks.len);
            for (tracks, list) |t, *j| j.* = .of(trackFace(t), t.id);
            try json_frames.append(ctx.arena, .{ .frame = n, .step = @tagName(what), .ms = ms, .faces = list });
        } else for (tracks) |t| try writeFace(w, n, t.id, trackFace(t));
        if (gif) |*g| {
            for (tracks) |t| drawFace(frame, .bgr, trackFace(t), t.id, t.ok);
            try g.add(exec, gpa, frame, delay_cs);
        }
    }
    if (n == 0) return ctx.fail("'{s}' has no frames", .{name});
    const s = ft.stats;
    const summary: VideoSummary = .{
        .frames = n,
        .frames_with_faces = with_faces,
        .faces_seen = ft.next_id - 1,
        .full_detections = s.full_runs,
        .crop_detections = s.crop_rounds,
        .postponed = s.postponed,
        .mean_ms = ms_sum / @as(f64, @floatFromInt(n)),
        .max_ms = ms_max,
    };
    if (ctx.json) {
        try std.json.Stringify.value(VideoReport{
            .input = name,
            .width = W,
            .height = H,
            .fps = frames.fps,
            .detector = .{ ft.det_w, ft.det_h },
            .summary = summary,
            .frames = json_frames.items,
        }, .{ .whitespace = .indent_2 }, w);
        try w.writeAll("\n");
    } else if (!ctx.quiet) {
        w.flush() catch {};
        ctx.se.print(ctx.err, .dim, "  {d} frames, faces on {d} · {d} face{s} seen · detector: {d} whole-frame, {d} crop-only, {d} postponed · {d:.2} ms/frame (max {d:.2})\n", .{ n, with_faces, summary.faces_seen, if (summary.faces_seen == 1) "" else "s", s.full_runs, s.crop_rounds, s.postponed, summary.mean_ms, ms_max }) catch {};
        ctx.err.flush() catch {};
    }
    if (out) |o| {
        const bytes = try gif.?.anim.finish();
        defer gpa.free(bytes);
        try imageio.writeBytes(ctx, o, bytes);
        if (!ctx.quiet and !ctx.json) {
            var b: [16]u8 = undefined;
            ctx.se.print(ctx.err, .ok, "  {s} {s}", .{ ctx.se.glyph("✓", "ok"), o.name() }) catch {};
            ctx.se.print(ctx.err, .dim, " · GIF {d}x{d}, {d} frames, {s}\n", .{ gif.?.small.width, gif.?.small.height, n, term.fmtBytes(&b, bytes.len) }) catch {};
            ctx.err.flush() catch {};
        }
    }
}

/// An animated GIF of the annotated frames, at most `gif_max_width` wide.
const Gif = struct {
    anim: visar.codecs.gif.Animation,
    small: Image(u8),

    fn init(gpa: std.mem.Allocator, w: u32, h: u32) !Gif {
        const gw = @min(w, gif_max_width);
        const gh = @max(1, @as(u32, @intCast(@as(u64, h) * gw / w)));
        var small = try Image(u8).init(gpa, gw, gh, 4);
        errdefer small.deinit(gpa);
        return .{ .anim = try .init(gpa, gw, gh, .color, .bgr), .small = small };
    }

    fn deinit(self: *Gif, gpa: std.mem.Allocator) void {
        self.anim.deinit();
        self.small.deinit(gpa);
    }

    fn add(self: *Gif, exec: visar.Exec, gpa: std.mem.Allocator, frame: Image(u8), delay_cs: u16) !void {
        if (self.small.width == frame.width) return self.anim.addFrame(frame, delay_cs);
        try visar.imgproc.resize(exec, gpa, frame, self.small, .area);
        try self.anim.addFrame(self.small, delay_cs);
    }
};

/// A track as a detection: its box, landmarks and last detection score.
fn trackFace(t: track.FaceTrack) Face {
    var f: Face = .{ .box = .{ t.box.x, t.box.y, t.box.w, t.box.h }, .landmarks = undefined, .score = t.score };
    for (&f.landmarks, 0..) |*l, i| l.* = t.landmark(i);
    return f;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

const Point = [2]f64;

const JsonFace = struct {
    id: ?u32 = null,
    score: f64,
    /// x, y, width, height.
    box: [4]f64,
    landmarks: struct { right_eye: Point, left_eye: Point, nose: Point, mouth_right: Point, mouth_left: Point },

    fn of(f: Face, id: ?u32) JsonFace {
        return .{
            .id = id,
            .score = round(f.score, 1000),
            .box = .{ round(f.box[0], 10), round(f.box[1], 10), round(f.box[2], 10), round(f.box[3], 10) },
            .landmarks = .{ .right_eye = rp(f.landmarks[0]), .left_eye = rp(f.landmarks[1]), .nose = rp(f.landmarks[2]), .mouth_right = rp(f.landmarks[3]), .mouth_left = rp(f.landmarks[4]) },
        };
    }

    fn rp(p: [2]f32) Point {
        return .{ round(p[0], 10), round(p[1], 10) };
    }

    /// Without "id" for still images.
    pub fn jsonStringify(self: JsonFace, jw: anytype) !void {
        try jw.beginObject();
        if (self.id) |id| {
            try jw.objectField("id");
            try jw.write(id);
        }
        try jw.objectField("score");
        try jw.write(self.score);
        try jw.objectField("box");
        try jw.write(self.box);
        try jw.objectField("landmarks");
        try jw.write(self.landmarks);
        try jw.endObject();
    }
};

/// `v` to 1/`scale` (so JSON shows 0.953, not 0.9530000090599060).
fn round(v: f32, scale: f64) f64 {
    return @round(@as(f64, v) * scale) / scale;
}

const ImageReport = struct {
    input: []const u8,
    width: u32,
    height: u32,
    detector: [2]u32,
    ms: f64,
    faces: []const JsonFace,
};

const JsonFrame = struct {
    frame: u64,
    /// What the tracker did: full (whole-frame detection), crops (detection
    /// around the tracked faces), tracked (MOSSE only), postponed.
    step: []const u8,
    ms: f64,
    faces: []const JsonFace,
};

const VideoSummary = struct {
    frames: u64,
    frames_with_faces: u64,
    faces_seen: u32,
    full_detections: u64,
    crop_detections: u64,
    postponed: u64,
    mean_ms: f64,
    max_ms: f64,
};

const VideoReport = struct {
    input: []const u8,
    width: u32,
    height: u32,
    fps: f64,
    detector: [2]u32,
    summary: VideoSummary,
    frames: []const JsonFrame,
};

/// One face per line:
///   [frame N  ]face I  score S  box X,Y,W,H  right_eye X,Y  left_eye X,Y  nose X,Y  mouth_right X,Y  mouth_left X,Y
pub fn writeFace(w: *Io.Writer, frame: ?u64, id: u32, f: Face) Io.Writer.Error!void {
    if (frame) |n| try w.print("frame {d}  ", .{n});
    try w.print("face {d}  score {d:.3}  box {d:.1},{d:.1},{d:.1},{d:.1}", .{ id, f.score, f.box[0], f.box[1], f.box[2], f.box[3] });
    for (landmark_names, f.landmarks) |nm, l| try w.print("  {s} {d:.1},{d:.1}", .{ nm, l[0], l[1] });
    try w.writeAll("\n");
}

/// Draws a face's box, its label and the five landmarks (eyes blue and
/// red, nose green, mouth corners magenta and yellow, as in OpenCV's YuNet
/// demo) on a 3- or 4-channel image in `order`. A face whose tracker is
/// unsure (`ok` false) is drawn in orange.
pub fn drawFace(img: Image(u8), order: visar.codecs.Order, f: Face, id: u32, ok: bool) void {
    const big = @max(img.width, img.height);
    const t: u32 = @max(2, big / 640);
    const ls: u32 = @max(1, big / 900);
    const rgb = if (ok) [3]u8{ 0, 255, 0 } else [3]u8{ 255, 150, 0 };
    var col: [4]u8 = undefined;
    setColor(&col, order, rgb);
    const bx: i64 = @intFromFloat(@round(f.box[0]));
    const by: i64 = @intFromFloat(@round(f.box[1]));
    const bw: i64 = @intFromFloat(@round(f.box[2]));
    const bh: i64 = @intFromFloat(@round(f.box[3]));
    draw.rectangle(img, bx, by, bw, bh, &col, t);
    const dots = [5][3]u8{ .{ 255, 0, 0 }, .{ 0, 0, 255 }, .{ 0, 255, 0 }, .{ 255, 0, 255 }, .{ 0, 255, 255 } };
    for (f.landmarks, dots) |l, d| {
        var c: [4]u8 = undefined;
        setColor(&c, order, d);
        draw.circle(img, @intFromFloat(@round(l[0])), @intFromFloat(@round(l[1])), @intCast(t + 1), &c, 0);
    }
    var lb: [32]u8 = undefined;
    const label = std.fmt.bufPrint(&lb, "{d} {d:.2}", .{ id, f.score }) catch "";
    const lh: i64 = 9 * @as(i64, ls);
    const lw: i64 = @as(i64, @intCast(label.len)) * 6 * @as(i64, ls) + 2 * @as(i64, ls);
    const ly = if (by - lh >= 0) by - lh else by;
    draw.fillRect(img, bx, ly, bx + lw, ly + lh, &col);
    _ = draw.text(img, bx + ls, ly + ls, label, ls, &.{ 0, 0, 0, 255 });
}

fn setColor(c: *[4]u8, order: visar.codecs.Order, rgb: [3]u8) void {
    c.* = if (order == .bgr) .{ rgb[2], rgb[1], rgb[0], 255 } else .{ rgb[0], rgb[1], rgb[2], 255 };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

test "faces: detector sizes, text lines and drawing" {
    try testing.expectEqual([2]u32{ 1280, 719 }, detectorSize(2048, 1150, null, 1280));
    try testing.expectEqual([2]u32{ 640, 360 }, detectorSize(3840, 2160, 640, 1280));
    try testing.expectEqual([2]u32{ 320, 240 }, detectorSize(320, 240, null, 1280));
    try testing.expectEqual([2]u32{ 320, 240 }, detectorSize(320, 240, 0, 1280));
    const f: Face = .{ .box = .{ 10, 20, 30, 40 }, .landmarks = .{ .{ 15, 30 }, .{ 30, 30 }, .{ 22, 40 }, .{ 16, 50 }, .{ 29, 50 } }, .score = 0.9532 };
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeFace(&w, 7, 2, f);
    try testing.expectEqualStrings("frame 7  face 2  score 0.953  box 10.0,20.0,30.0,40.0  right_eye 15.0,30.0  left_eye 30.0,30.0  nose 22.0,40.0  mouth_right 16.0,50.0  mouth_left 29.0,50.0\n", w.buffered());
    var img = try Image(u8).init(testing.allocator, 64, 64, 3);
    defer img.deinit(testing.allocator);
    img.fill(0);
    drawFace(img, .rgb, f, 1, true);
    // The box outline is green; the nose dot green too, the right eye red.
    try testing.expectEqualSlices(u8, &.{ 0, 255, 0 }, img.pixel(10, 45));
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0 }, img.pixel(15, 30));
}

/// A Ctx writing to memory, for running the command in tests.
const TestCtx = struct {
    out: Io.Writer.Allocating,
    err: Io.Writer.Allocating,
    env: std.process.Environ.Map,
    arena: std.heap.ArenaAllocator,
    diag: args.Diag = .{},
    ctx: Ctx = undefined,

    fn init(self: *TestCtx) void {
        const gpa = testing.allocator;
        self.* = .{ .out = .init(gpa), .err = .init(gpa), .env = .init(gpa), .arena = .init(gpa) };
        self.ctx = .{
            .gpa = gpa,
            .arena = self.arena.allocator(),
            .io = testing.io,
            .env = &self.env,
            .out = &self.out.writer,
            .err = &self.err.writer,
            .so = .plain,
            .se = .plain,
            .diag = &self.diag,
            .threads = 2,
        };
    }

    fn deinit(self: *TestCtx) void {
        self.ctx.deinit();
        self.out.deinit();
        self.err.deinit();
        self.env.deinit();
        self.arena.deinit();
    }

    /// Runs `visar faces <argv>`.
    fn run(self: *TestCtx, argv: []const []const u8) !void {
        const commands = @import("commands.zig");
        var p = try args.parse(self.ctx.arena, argv, &.{ &commands.global_opts, &commands.output_opts, &opts }, &self.diag, .{});
        self.ctx.json = p.flag("json");
        self.ctx.output = p.value("output");
        try @import("commands.zig").find("faces").?.run(&self.ctx, &p);
        try self.ctx.out.flush();
    }
};

fn modelsDir() ?[]const u8 {
    const dir = testing.environ.getAlloc(testing.allocator, "VISAR_MODELS") catch return null;
    defer testing.allocator.free(dir);
    const path = std.fs.path.join(testing.allocator, &.{ dir, "largest_selfie.jpg" }) catch return null;
    defer testing.allocator.free(path);
    Io.Dir.cwd().access(testing.io, path, .{}) catch return null;
    return std.fmt.bufPrint(&models_buf, "{s}", .{dir}) catch null;
}
var models_buf: [1024]u8 = undefined;

test "faces: usage errors" {
    var t: TestCtx = undefined;
    t.init();
    defer t.deinit();
    try testing.expectError(error.Usage, t.run(&.{}));
    // No --model and no VISAR_MODELS.
    try testing.expectError(error.Usage, t.run(&.{"x.png"}));
    try testing.expectError(error.Usage, t.run(&.{ "x.png", "--score", "2" }));
}

test "faces: an image, as JSON and annotated" {
    // Needs VISAR_MODELS with the YuNet model and largest_selfie.jpg.
    const dir = modelsDir() orelse return error.SkipZigTest;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t: TestCtx = undefined;
    t.init();
    defer t.deinit();
    try t.env.put("VISAR_MODELS", dir);
    const a = t.ctx.arena;
    const img = try std.fs.path.join(a, &.{ dir, "largest_selfie.jpg" });
    const out = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/faces.png", .{tmp.sub_path});
    try t.run(&.{ img, "--json", "--out", out });
    const Parsed = struct { width: u32, height: u32, faces: []const struct { score: f32, box: [4]f32, landmarks: struct { right_eye: Point, left_eye: Point } } };
    const r = try std.json.parseFromSlice(Parsed, testing.allocator, t.out.written(), .{ .ignore_unknown_fields = true });
    defer r.deinit();
    try testing.expectEqual(@as(u32, 2048), r.value.width);
    try testing.expect(r.value.faces.len >= 3);
    for (r.value.faces) |f| {
        try testing.expect(f.score >= 0.9 and f.box[2] > 20 and f.box[3] > 20);
        // The eyes lie inside the box, the right eye (the person's) on the left.
        try testing.expect(f.landmarks.right_eye[0] > f.box[0] and f.landmarks.left_eye[0] < f.box[0] + f.box[2]);
        try testing.expect(f.landmarks.right_eye[0] < f.landmarks.left_eye[0]);
    }
    var written = try visar.codecs.read(testing.allocator, testing.io, out, .{});
    defer written.deinit(testing.allocator);
    try testing.expectEqual(@as(u32, 2048), written.width());

    // Plain text: one line per face, the same faces.
    t.out.clearRetainingCapacity();
    try t.run(&.{img});
    try testing.expectEqual(r.value.faces.len, std.mem.count(u8, t.out.written(), "\n"));
    try testing.expect(std.mem.startsWith(u8, t.out.written(), "face 1  score 0."));
}

test "faces: a y4m video is tracked frame by frame" {
    const dir = modelsDir() orelse return error.SkipZigTest;
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t: TestCtx = undefined;
    t.init();
    defer t.deinit();
    const a = t.ctx.arena;

    // The selfie moving over a 640x360 canvas (track_lib.StillScene),
    // written as 4:4:4 YUV4MPEG2.
    var decoded = try visar.codecs.read(gpa, testing.io, try std.fs.path.join(a, &.{ dir, "largest_selfie.jpg" }), .{ .mode = .color, .order = .bgr });
    defer decoded.deinit(gpa);
    const W = 640;
    const H = 360;
    const n = 24;
    var scene = try track.StillScene.init(gpa, visar.Exec.serial, decoded.u8, W, H);
    defer scene.deinit();
    var frame = try Image(u8).init(gpa, W, H, 4);
    defer frame.deinit(gpa);
    var video: Io.Writer.Allocating = .init(gpa);
    defer video.deinit();
    try video.writer.print("YUV4MPEG2 W{d} H{d} F25:1 Ip A1:1 C444\n", .{ W, H });
    const planes = try gpa.alloc(u8, W * H * 3);
    defer gpa.free(planes);
    for (0..n) |_| {
        _ = scene.render(frame);
        for (0..H) |y| for (0..W) |x| {
            const p = frame.pixel(@intCast(x), @intCast(y));
            const b: f32 = @floatFromInt(p[0]);
            const g: f32 = @floatFromInt(p[1]);
            const r: f32 = @floatFromInt(p[2]);
            const i = y * W + x;
            planes[i] = @intFromFloat(@round(16 + 0.2568 * r + 0.5041 * g + 0.0979 * b));
            planes[W * H + i] = @intFromFloat(@round(128 - 0.1482 * r - 0.2910 * g + 0.4392 * b));
            planes[2 * W * H + i] = @intFromFloat(@round(128 + 0.4392 * r - 0.3678 * g - 0.0714 * b));
        };
        try video.writer.writeAll("FRAME\n");
        try video.writer.writeAll(planes);
    }
    const clip = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/clip.y4m", .{tmp.sub_path});
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "clip.y4m", .data = video.written() });
    const gif = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/clip.gif", .{tmp.sub_path});
    const model = try std.fs.path.join(a, &.{ dir, model_name });
    try t.run(&.{ clip, "--model", model, "--json", "-o", gif, "--detect-every", "2" });

    const Parsed = struct {
        width: u32,
        summary: struct { frames: u64, frames_with_faces: u64, full_detections: u64 },
        frames: []const struct { frame: u64, step: []const u8, faces: []const struct { id: u32, box: [4]f32 } },
    };
    const r = try std.json.parseFromSlice(Parsed, gpa, t.out.written(), .{ .ignore_unknown_fields = true });
    defer r.deinit();
    try testing.expectEqual(@as(u32, W), r.value.width);
    try testing.expectEqual(@as(u64, n), r.value.summary.frames);
    try testing.expectEqual(@as(usize, n), r.value.frames.len);
    try testing.expectEqualStrings("full", r.value.frames[0].step);
    try testing.expect(r.value.summary.frames_with_faces == n);
    // Not every frame ran the detector on the whole frame.
    try testing.expect(r.value.summary.full_detections < n);
    // The same face keeps its id.
    try testing.expectEqual(r.value.frames[0].faces[0].id, r.value.frames[n - 1].faces[0].id);
    const g = try tmp.dir.readFileAlloc(testing.io, "clip.gif", gpa, .limited(1 << 26));
    defer gpa.free(g);
    try testing.expect(std.mem.startsWith(u8, g, "GIF89a"));
}
