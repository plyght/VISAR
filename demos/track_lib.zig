//! The live multi-object tracking demo, as a library: `run` renders a
//! synthetic street scene (textured background, sensor noise, lighting
//! flicker, objects that enter, cross, occlude and leave), runs a VISAR
//! pipeline on every frame and returns an annotated animated GIF together
//! with timing and accuracy statistics:
//!
//!   gray -> Gaussian blur -> background subtraction -> morphology
//!        -> connected components -> multi-object tracker
//!
//! Because the scene is synthetic, the demo scores itself against the ground
//! truth (detection recall and identity switches). Both `zig build
//! demo-track` (demos/track.zig) and `visar demo track` call into this file.

const std = @import("std");
const visar = @import("visar");
const Image = visar.Image;
const ip = visar.imgproc;
const draw = ip.draw;
const Io = std.Io;

const Obj = struct {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    w: f32,
    h: f32,
    color: [3]u8,
    round: bool,
    start: u32,
    /// Ground-truth identity.
    gid: u32,
};

pub const Options = struct {
    frames: u32 = 150,
    width: u32 = 960,
    height: u32 = 540,
    /// Width of the GIF; the scene is scaled down to it (never up).
    gif_width: u32 = 480,
    seed: u64 = 7,

    /// The smallest scene the overlay and the object sizes work at.
    pub const min_width = 160;
    pub const min_height = 90;
};

/// Timing and accuracy of one run.
pub const Stats = struct {
    threads: usize,
    width: u32,
    height: u32,
    gif_width: u32,
    gif_height: u32,
    frames: u32,
    /// Per-frame pipeline time in milliseconds, over the frames after warm-up.
    mean_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    fps: f64,
    /// Fraction (0-1) of visible ground-truth objects matched by a live track.
    recall: f64,
    id_switches: u32,
    gif_bytes: usize,
};

pub const Result = struct {
    /// The encoded GIF, owned by the caller (free with `deinit`).
    gif: []u8,
    stats: Stats,

    pub fn deinit(self: *Result, gpa: std.mem.Allocator) void {
        gpa.free(self.gif);
        self.* = undefined;
    }
};

/// Called after every frame with the number of frames done.
pub const Progress = struct {
    ctx: *anyopaque,
    func: *const fn (ctx: *anyopaque, done: u32, total: u32) void,
};

/// The frames at the start that are not timed or scored (the background
/// model is still settling).
pub const warmup: u32 = 8;

pub fn run(gpa: std.mem.Allocator, io: Io, exec: visar.Exec, opts: Options, progress: ?Progress) !Result {
    const W = opts.width;
    const H = opts.height;

    var scene = try Scene.init(gpa, exec, .{ .width = W, .height = H, .seed = opts.seed });
    defer scene.deinit();
    const sx = scene.sx;

    var frame = try Image(u8).init(gpa, W, H, 3);
    defer frame.deinit(gpa);
    var gray = try Image(u8).init(gpa, W, H, 1);
    defer gray.deinit(gpa);
    const gw = @min(opts.gif_width, W);
    const gh = @max(1, gw * H / W);
    var small = try Image(u8).init(gpa, gw, gh, 3);
    defer small.deinit(gpa);

    var det = try Detector.init(gpa, W, H, .{ .sx = sx });
    defer det.deinit();
    const tracker = &det.tracker;
    var anim = try visar.codecs.gif.Animation.init(gpa, gw, gh, .color, .rgb);
    defer anim.deinit();

    var times: std.ArrayList(f64) = .empty;
    defer times.deinit(gpa);

    // Ground-truth scoring.
    var gt_visible: u64 = 0;
    var gt_found: u64 = 0;
    var id_switches: u32 = 0;
    var last_track_for: [4096]u32 = @splat(0);

    for (0..opts.frames) |f_| {
        const f: u32 = @intCast(f_);
        // ---- render the scene
        scene.render(frame, .exact);

        // ---- the timed VISAR pipeline
        const t0 = Io.Clock.awake.now(io);
        try ip.cvtColor(exec, frame, gray, .rgb2gray);
        try det.detect(exec, gray, null);
        const t1 = Io.Clock.awake.now(io);
        const ms = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
        if (f >= warmup) try times.append(gpa, ms);

        // ---- score against ground truth
        if (f >= warmup) for (scene.objs.items) |o| {
            if (f < o.start) continue;
            const gx = o.x + o.w / 2;
            const gy = o.y + o.h / 2;
            if (gx < 0 or gy < 0 or gx >= @as(f32, @floatFromInt(W)) or gy >= @as(f32, @floatFromInt(H))) continue;
            if (o.x < 0 or o.x + o.w > @as(f32, @floatFromInt(W))) continue; // partly outside
            gt_visible += 1;
            var best: ?u32 = null;
            var best_d: f32 = 30 * sx;
            for (tracker.tracks.items) |t| {
                if (!t.confirmed() or t.missed > 0) continue;
                const d = @sqrt((t.box.cx() - gx) * (t.box.cx() - gx) + (t.box.cy() - gy) * (t.box.cy() - gy));
                if (d < best_d) {
                    best_d = d;
                    best = t.id;
                }
            }
            if (best) |id| {
                gt_found += 1;
                const slot = o.gid % 4096;
                if (last_track_for[slot] != 0 and last_track_for[slot] != id) id_switches += 1;
                last_track_for[slot] = id;
            }
        };

        // ---- visualise
        drawTracks(frame, tracker.*, .{});
        var hud: [96]u8 = undefined;
        draw.fillRect(frame, 0, 0, W, 64, &.{ 10, 12, 20 });
        _ = draw.text(frame, 12, 8, "VISAR LIVE TRACKING", 3, &.{ 255, 255, 255 });
        const line2 = std.fmt.bufPrint(&hud, "FRAME {d:>3}  {d}x{d}  PIPELINE {d:.2} MS ({d:.0} FPS)  TRACKS {d}", .{ f, W, H, ms, 1000.0 / ms, countConfirmed(tracker.*) }) catch "";
        _ = draw.text(frame, 12, 40, line2, 2, &.{ 120, 230, 255 });
        try ip.resize(exec, gpa, frame, small, .area);
        try anim.addFrame(small, 4);
        if (progress) |p| p.func(p.ctx, f + 1, opts.frames);
    }

    const bytes = try anim.finish();
    errdefer gpa.free(bytes);

    std.mem.sort(f64, times.items, {}, std.sort.asc(f64));
    var sum: f64 = 0;
    for (times.items) |t| sum += t;
    const n = times.items.len;
    const mean = if (n > 0) sum / @as(f64, @floatFromInt(n)) else 0;
    return .{ .gif = bytes, .stats = .{
        .threads = exec.threads(),
        .width = W,
        .height = H,
        .gif_width = gw,
        .gif_height = gh,
        .frames = opts.frames,
        .mean_ms = mean,
        .p50_ms = if (n > 0) times.items[n / 2] else 0,
        .p95_ms = if (n > 0) times.items[@min(n - 1, n * 95 / 100)] else 0,
        .fps = if (mean > 0) 1000.0 / mean else 0,
        .recall = @as(f64, @floatFromInt(gt_found)) / @as(f64, @floatFromInt(@max(1, gt_visible))),
        .id_switches = id_switches,
        .gif_bytes = bytes.len,
    } };
}

/// The summary as Markdown (CI posts it to the job page).
pub fn writeMarkdown(w: *Io.Writer, s: Stats, gif_name: []const u8) Io.Writer.Error!void {
    try w.print(
        \\### VISAR live tracking demo
        \\
        \\| Metric | Value |
        \\| --- | --- |
        \\| Target | `{s}` |
        \\| Threads | {d} |
        \\| Resolution | {d}x{d} |
        \\| Frames | {d} |
        \\| Pipeline time per frame (mean / p50 / p95) | {d:.2} / {d:.2} / {d:.2} ms |
        \\| Throughput | **{d:.0} FPS** |
        \\| Detection recall vs ground truth | {d:.1}% |
        \\| Identity switches | {d} |
        \\| GIF | `{s}` ({d} KB) |
        \\
        \\Pipeline: {s}.
        \\
    , .{ visar.core.target_description, s.threads, s.width, s.height, s.frames, s.mean_ms, s.p50_ms, s.p95_ms, s.fps, 100.0 * s.recall, s.id_switches, gif_name, s.gif_bytes / 1024, pipeline_description });
}

pub const pipeline_description = "gray, Gaussian 5x5, background subtraction, open 3x3, dilate 9x9 ellipse, connected components, tracker";

pub fn countConfirmed(t: visar.video.Tracker) usize {
    var n: usize = 0;
    for (t.tracks.items) |tr| n += @intFromBool(tr.confirmed());
    return n;
}

/// The colour of track `id`, as RGB.
pub fn idColor(id: u32) [3]u8 {
    const cols = [_][3]u8{ .{ 255, 80, 80 }, .{ 80, 220, 255 }, .{ 255, 220, 60 }, .{ 120, 255, 120 }, .{ 255, 120, 255 }, .{ 255, 160, 60 }, .{ 160, 160, 255 }, .{ 60, 255, 200 } };
    return cols[id % cols.len];
}

// ---------------------------------------------------------------------------
// The pieces of `run`, shared with `visar track` (live camera tracking).
// ---------------------------------------------------------------------------

/// The synthetic street scene: a textured background and objects that enter,
/// cross, occlude and leave. `render` draws the next frame.
pub const Scene = struct {
    gpa: std.mem.Allocator,
    width: u32,
    height: u32,
    /// Scale of the scene relative to the 960-pixel-wide original.
    sx: f32,
    bg: Image(u8),
    objs: std.ArrayList(Obj) = .empty,
    /// The next frame to render.
    frame: u32 = 0,
    /// Pre-generated sensor noise for `.fast` rendering (empty until used).
    noise_pos: []u8 = &.{},
    noise_neg: []u8 = &.{},
    prng: std.Random.DefaultPrng,

    pub const Config = struct { width: u32, height: u32, seed: u64 = 7 };

    pub fn init(gpa: std.mem.Allocator, exec: visar.Exec, c: Config) !Scene {
        const W = c.width;
        const H = c.height;
        // Static background: sky gradient, road, lane marks, texture.
        var bg = try Image(u8).init(gpa, W, H, 3);
        errdefer bg.deinit(gpa);
        var prng = std.Random.DefaultPrng.init(c.seed);
        const rnd = prng.random();
        for (0..H) |y| {
            const r = bg.row(y);
            const fy: f32 = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(H));
            for (0..W) |x| {
                const t = rnd.intRangeAtMost(i32, -10, 10);
                const base: [3]f32 = if (fy < 0.3) .{ 120 + 80 * fy, 160 + 60 * fy, 210 } else if (fy < 0.42) .{ 70, 110, 60 } else .{ 75, 78, 82 };
                for (0..3) |ch| r[x * 3 + ch] = visar.core.sat.cast(u8, base[ch] + @as(f32, @floatFromInt(t)));
            }
        }
        {
            var x: i64 = 0;
            while (x < W) : (x += 80) draw.fillRect(bg, x, @intFromFloat(@as(f32, @floatFromInt(H)) * 0.7), x + 40, @intFromFloat(@as(f32, @floatFromInt(H)) * 0.7 + 6), &.{ 230, 220, 120 });
            var tmp = try bg.clone(gpa);
            defer tmp.deinit(gpa);
            try ip.gaussianBlur(exec, gpa, tmp, bg, .{ .kw = 3, .kh = 3 });
        }

        // Moving objects.
        var objs: std.ArrayList(Obj) = .empty;
        errdefer objs.deinit(gpa);
        const palette = [_][3]u8{ .{ 230, 60, 50 }, .{ 40, 170, 230 }, .{ 250, 200, 40 }, .{ 120, 220, 90 }, .{ 200, 90, 220 }, .{ 245, 140, 40 }, .{ 240, 240, 240 }, .{ 20, 20, 30 } };
        const sx: f32 = @as(f32, @floatFromInt(W)) / 960.0;
        for (0..12) |k| {
            const lane: f32 = @floatFromInt(k % 4);
            const dir: f32 = if (k % 2 == 0) 1 else -1;
            const speed = (2.5 + rnd.float(f32) * 4.5) * sx;
            const w = (40 + rnd.float(f32) * 70) * sx;
            const h = (28 + rnd.float(f32) * 30) * sx;
            try objs.append(gpa, .{
                .x = if (dir > 0) -w else @as(f32, @floatFromInt(W)),
                .y = @as(f32, @floatFromInt(H)) * (0.46 + 0.12 * lane) + rnd.float(f32) * 10,
                .vx = speed * dir,
                .vy = (rnd.float(f32) - 0.5) * 0.6,
                .w = w,
                .h = h,
                .color = palette[k % palette.len],
                .round = k % 3 == 0,
                .start = @intCast(k * 11),
                .gid = @intCast(k + 1),
            });
        }
        return .{ .gpa = gpa, .width = W, .height = H, .sx = sx, .bg = bg, .objs = objs, .prng = prng };
    }

    pub fn deinit(self: *Scene) void {
        self.bg.deinit(self.gpa);
        self.objs.deinit(self.gpa);
        self.gpa.free(self.noise_pos);
        self.gpa.free(self.noise_neg);
        self.* = undefined;
    }

    /// How the sensor noise is made: `.exact` is the demo's per-pixel
    /// generator (its GIF and scores depend on it); `.fast` adds slices of a
    /// pre-generated noise table with saturating SIMD, several times faster,
    /// for live use at high frame rates.
    pub const Noise = enum { exact, fast };

    /// Renders the next frame into `frame` (3 channels, RGB, scene size).
    pub fn render(self: *Scene, frame: Image(u8), noise: Noise) void {
        std.debug.assert(frame.channels == 3 and frame.width == self.width and frame.height == self.height);
        const f = self.frame;
        self.advance();
        frame.copyFrom(self.bg);
        const flicker: i32 = @intFromFloat(4 * @sin(@as(f32, @floatFromInt(f)) * 0.21));
        for (self.objs.items) |*o| {
            if (f < o.start) continue;
            const xi: i64 = @intFromFloat(o.x);
            const yi: i64 = @intFromFloat(o.y);
            if (o.round) {
                draw.circle(frame, xi + @as(i64, @intFromFloat(o.w / 2)), yi + @as(i64, @intFromFloat(o.h / 2)), @intFromFloat(o.h / 2), &o.color, 0);
            } else {
                draw.fillRect(frame, xi, yi, xi + @as(i64, @intFromFloat(o.w)), yi + @as(i64, @intFromFloat(o.h)), &o.color);
                draw.fillRect(frame, xi + 6, yi + 4, xi + @as(i64, @intFromFloat(o.w * 0.45)), yi + @as(i64, @intFromFloat(o.h * 0.45)), &.{ 170, 210, 235 });
            }
        }
        // Sensor noise and flicker.
        switch (noise) {
            .exact => {
                var state: u32 = f *% 2654435761 +% 12345;
                for (0..frame.height) |y| for (frame.row(y)) |*v| {
                    state = state *% 1664525 +% 1013904223;
                    const n: i32 = @as(i32, @intCast((state >> 24) & 15)) - 8 + flicker;
                    v.* = @intCast(std.math.clamp(@as(i32, v.*) + n, 0, 255));
                };
            },
            .fast => self.fastNoise(frame, flicker),
        }
    }

    /// Moves the objects one frame on without drawing (`render` does this
    /// before it draws).
    pub fn advance(self: *Scene) void {
        const f = self.frame;
        self.frame += 1;
        const Wf: f32 = @floatFromInt(self.width);
        for (self.objs.items) |*o| {
            if (f < o.start) continue;
            o.x += o.vx;
            o.y += o.vy;
            // Wrap around after leaving, as a new arrival.
            if (o.vx > 0 and o.x > Wf + 20) {
                o.x = -o.w - 20;
                o.gid += 100;
            }
            if (o.vx < 0 and o.x < -o.w - 20) {
                o.x = Wf + 20;
                o.gid += 100;
            }
        }
    }

    /// An object of the last rendered frame: its identity (which changes
    /// when it leaves and comes back) and its bounding box.
    pub const Truth = struct { gid: u32, box: visar.video.Box };

    /// The objects entirely inside the last rendered frame.
    pub fn truth(self: *const Scene, out: []Truth) []Truth {
        var n: usize = 0;
        const Wf: f32 = @floatFromInt(self.width);
        const Hf: f32 = @floatFromInt(self.height);
        for (self.objs.items) |o| {
            if (n == out.len) break;
            if (self.frame == 0 or self.frame - 1 < o.start) continue;
            if (o.x < 0 or o.y < 0 or o.x + o.w > Wf or o.y + o.h > Hf) continue;
            // Round objects are circles of diameter h, centred in the box.
            const bx = if (o.round) o.x + (o.w - o.h) / 2 else o.x;
            const bw = if (o.round) o.h else o.w;
            out[n] = .{ .gid = o.gid, .box = .{ .x = bx, .y = o.y, .w = bw, .h = o.h } };
            n += 1;
        }
        return out[0..n];
    }

    const noise_len = 1 << 18;

    fn fastNoise(self: *Scene, frame: Image(u8), flicker: i32) void {
        if (self.noise_pos.len == 0) {
            const pos = self.gpa.alloc(u8, noise_len + 4096) catch return;
            const neg = self.gpa.alloc(u8, noise_len + 4096) catch {
                self.gpa.free(pos);
                return;
            };
            var state: u32 = 12345;
            for (pos, neg) |*p, *q| {
                state = state *% 1664525 +% 1013904223;
                const n: i32 = @as(i32, @intCast((state >> 24) & 15)) - 8;
                p.* = @intCast(@max(n, 0));
                q.* = @intCast(@max(-n, 0));
            }
            self.noise_pos = pos;
            self.noise_neg = neg;
        }
        const N = 32;
        const V = @Vector(N, u8);
        const fp: V = @splat(@intCast(@max(flicker, 0)));
        const fn_: V = @splat(@intCast(@max(-flicker, 0)));
        const rnd = self.prng.random();
        for (0..frame.height) |y| {
            const r = frame.row(y);
            const off = rnd.uintLessThan(usize, noise_len - 4096);
            const pos = self.noise_pos[off..];
            const neg = self.noise_neg[off..];
            var x: usize = 0;
            while (x + N <= r.len) : (x += N) {
                const v: V = r[x..][0..N].*;
                const p: V = pos[x % 4096 ..][0..N].*;
                const q: V = neg[x % 4096 ..][0..N].*;
                r[x..][0..N].* = ((v +| p) -| q) +| fp -| fn_;
            }
            while (x < r.len) : (x += 1) {
                r[x] = ((r[x] +| pos[x % 4096]) -| neg[x % 4096]) +| fp[0] -| fn_[0];
            }
        }
    }
};

/// Per-stage times of one `Detector.detect` call, in milliseconds.
pub const StageTimes = struct {
    blur: f64 = 0,
    background: f64 = 0,
    morphology: f64 = 0,
    components: f64 = 0,
    tracker: f64 = 0,
};

/// The detection and tracking part of the pipeline, on a grayscale frame:
/// Gaussian 5x5, background subtraction, open 3x3, dilate, connected
/// components and the multi-object tracker. Its buffers are allocated once.
pub const Detector = struct {
    gpa: std.mem.Allocator,
    width: u32,
    height: u32,
    blurred: Image(u8),
    mask: Image(u8),
    mask2: Image(u8),
    labels: Image(i32),
    model: visar.video.Background,
    tracker: visar.video.Tracker,
    boxes: std.ArrayList(visar.video.Box) = .empty,
    el3_buf: [9]u8 = undefined,
    el_buf: [31 * 31]u8 = undefined,
    dilate_size: u32,
    min_area: u64,
    threshold: u8,

    pub const Config = struct {
        /// Scale of the frame relative to a 960-pixel-wide one; object sizes,
        /// the tracker's gate and the minimum blob area follow it.
        sx: f32 = 1,
        /// Size of the elliptical dilation that joins the parts of an object
        /// (odd, at most 31).
        dilate: u32 = 9,
        /// Background learning rate, 1 / 2^shift per frame.
        bg_shift: u4 = 4,
        /// Gray-level difference from the background that counts as motion.
        threshold: u8 = 28,
    };

    pub fn init(gpa: std.mem.Allocator, width: u32, height: u32, c: Config) !Detector {
        var blurred = try Image(u8).init(gpa, width, height, 1);
        errdefer blurred.deinit(gpa);
        var mask = try Image(u8).init(gpa, width, height, 1);
        errdefer mask.deinit(gpa);
        var mask2 = try Image(u8).init(gpa, width, height, 1);
        errdefer mask2.deinit(gpa);
        var labels = try Image(i32).init(gpa, width, height, 1);
        errdefer labels.deinit(gpa);
        var model = try visar.video.Background.init(gpa, width, height, c.bg_shift);
        errdefer model.deinit(gpa);
        var tracker = visar.video.Tracker.init(gpa);
        tracker.gate = 40 * c.sx;
        return .{
            .gpa = gpa,
            .width = width,
            .height = height,
            .blurred = blurred,
            .mask = mask,
            .mask2 = mask2,
            .labels = labels,
            .model = model,
            .tracker = tracker,
            .dilate_size = @min(31, c.dilate | 1),
            .min_area = @intFromFloat(250 * c.sx * c.sx),
            .threshold = c.threshold,
        };
    }

    pub fn deinit(self: *Detector) void {
        const gpa = self.gpa;
        self.blurred.deinit(gpa);
        self.mask.deinit(gpa);
        self.mask2.deinit(gpa);
        self.labels.deinit(gpa);
        self.model.deinit(gpa);
        self.tracker.deinit();
        self.boxes.deinit(gpa);
        self.* = undefined;
    }

    /// Forgets the background (it is learnt again from the next frame) and
    /// every track.
    pub fn reset(self: *Detector) void {
        self.model.primed = false;
        self.tracker.tracks.clearRetainingCapacity();
    }

    /// Runs the pipeline on `gray` (1 channel, the detector's size) and
    /// updates `tracker`. With `times` (and an `io` to read the clock), the
    /// stages are timed.
    pub fn detect(self: *Detector, exec: visar.Exec, gray: Image(u8), timing: ?struct { io: Io, times: *StageTimes }) !void {
        const gpa = self.gpa;
        const el3 = ip.structuringElement(.rect, 3, 3, &self.el3_buf);
        const eld = ip.structuringElement(.ellipse, self.dilate_size, self.dilate_size, &self.el_buf);
        var clock: Clock = .{ .io = if (timing) |t| t.io else null };
        _ = clock.lap();
        try ip.gaussianBlur(exec, gpa, gray, self.blurred, .{ .kw = 5, .kh = 5 });
        const t_blur = clock.lap();
        try self.model.apply(exec, self.blurred, self.mask, self.threshold);
        const t_bg = clock.lap();
        try ip.morphologyEx(exec, gpa, self.mask, self.mask2, .open, el3, 1);
        try ip.dilate(exec, gpa, self.mask2, self.mask, eld, 1);
        const t_morph = clock.lap();
        var cc = try ip.connectedComponentsWithStats(gpa, self.mask, self.labels, .eight);
        defer cc.deinit(gpa);
        self.boxes.clearRetainingCapacity();
        for (cc.stats[1..]) |st| {
            if (st.area < self.min_area) continue;
            try self.boxes.append(gpa, .{ .x = @floatFromInt(st.left), .y = @floatFromInt(st.top), .w = @floatFromInt(st.width), .h = @floatFromInt(st.height) });
        }
        const t_cc = clock.lap();
        try self.tracker.update(self.boxes.items);
        const t_track = clock.lap();
        if (timing) |t| t.times.* = .{ .blur = t_blur, .background = t_bg, .morphology = t_morph, .components = t_cc, .tracker = t_track };
    }
};

/// Milliseconds between successive `lap` calls (0 without an `io`).
pub const Clock = struct {
    io: ?Io,
    last: ?Io.Timestamp = null,

    pub fn lap(c: *Clock) f64 {
        const io = c.io orelse return 0;
        const now = Io.Clock.awake.now(io);
        const prev = c.last orelse now;
        c.last = now;
        return @as(f64, @floatFromInt(now.nanoseconds - prev.nanoseconds)) / 1e6;
    }
};

pub const DrawOptions = struct {
    /// Frame pixels per tracker pixel (the tracker may run on a smaller copy).
    scale: f32 = 1,
    /// Channel order of the frame.
    order: enum { rgb, bgr } = .rgb,
    /// Line thickness of boxes and trails, and the size of the ID labels.
    thickness: u32 = 3,
    label_scale: u32 = 2,
};

/// Draws every confirmed track: its trail, its box and an "#id" label.
pub fn drawTracks(frame: Image(u8), tracker: visar.video.Tracker, o: DrawOptions) void {
    const s = o.scale;
    const t3: i64 = o.thickness;
    const ls: i64 = o.label_scale;
    for (tracker.tracks.items) |t| {
        if (!t.confirmed()) continue;
        const rgb = idColor(t.id);
        const col: [4]u8 = switch (o.order) {
            .rgb => .{ rgb[0], rgb[1], rgb[2], 255 },
            .bgr => .{ rgb[2], rgb[1], rgb[0], 255 },
        };
        var k: usize = 1;
        while (k < t.trail_count) : (k += 1) {
            const a = t.trailPoint(k - 1);
            const b = t.trailPoint(k);
            draw.line(frame, @intFromFloat(a[0] * s), @intFromFloat(a[1] * s), @intFromFloat(b[0] * s), @intFromFloat(b[1] * s), &col, o.thickness);
        }
        const bx: i64 = @intFromFloat(t.box.x * s);
        const by: i64 = @intFromFloat(t.box.y * s);
        draw.rectangle(frame, bx - t3, by - t3, @as(i64, @intFromFloat(t.box.w * s)) + 2 * t3, @as(i64, @intFromFloat(t.box.h * s)) + 2 * t3, &col, o.thickness);
        var lbuf: [16]u8 = undefined;
        const label = std.fmt.bufPrint(&lbuf, "#{d}", .{t.id}) catch "#";
        draw.fillRect(frame, bx - t3, by - 12 * ls, bx - t3 + @as(i64, @intCast(label.len)) * 6 * ls + 3 * ls, by - 2 * ls - 1, &col);
        _ = draw.text(frame, bx, by - 12 * ls + @divTrunc(3 * ls, 2), label, o.label_scale, &.{ 0, 0, 0, 255 });
    }
}
