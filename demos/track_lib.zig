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
        var cc = try ip.connectedComponentsWithStats(exec, gpa, self.mask, self.labels, .eight);
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

// ---------------------------------------------------------------------------
// Face tracking (`visar track --faces`): detect, then track.
//
// A YuNet detector (visar.dnn.FaceDetector) runs on a schedule; between its
// runs every face is carried by a MOSSE correlation filter on the small
// grayscale tracking image, so faces are followed at camera rate while the
// network runs on a fraction of the frames. Change-driven inference cuts
// the network's work further, without changing what it would find:
//
//   level 1  a scheduled detection is postponed while the frame has not
//            changed since the last one and every track is healthy
//   level 2  with healthy tracks, a scheduled detection runs only on crops
//            around the tracked faces (at the whole-frame detector's scale,
//            so a face looks to the network exactly as in a whole-frame
//            run); every `full_every`-th detection, and any detection forced
//            by a weak or missing track, still covers the whole frame to
//            find new faces
// ---------------------------------------------------------------------------

pub const Face = visar.dnn.Face;
const Box = visar.video.Box;

pub const FaceOptions = struct {
    /// Width of the whole-frame detector input; the height follows the
    /// frame's aspect ratio.
    face_width: u32 = 320,
    /// A detection is scheduled every this many frames.
    detect_every: u32 = 4,
    /// At most this many faces are tracked.
    max_faces: u32 = 16,
    /// Change-driven inference: 0 = every scheduled detection covers the
    /// whole frame; 1 = also postpone it while the frame is unchanged;
    /// 2 = also run it only around the tracked faces (the default).
    level: u8 = 2,
    /// Level 1: the mean absolute gray-level difference (0-255) between the
    /// tracking image now and at the last detection, over every 4th row,
    /// below which the frame counts as unchanged. Webcam noise at the
    /// tracking resolution (already box-averaged) gives about 1; a face
    /// that moved by a few pixels gives several.
    still_threshold: f32 = 2.0,
    /// Level 1: a detection is never postponed beyond this many frames.
    max_postpone: u32 = 30,
    /// Level 2: every this many detections covers the whole frame.
    full_every: u32 = 4,
    /// Level 2: crop side relative to the larger side of the face box.
    crop_scale: f32 = 2.0,
    score_threshold: f32 = 0.9,
    /// Which backend runs the detector (see `visar.dnn.Net.Options.backend`).
    backend: visar.dnn.Backend = .auto,
    /// A detection and a track overlapping by at least this IoU are the
    /// same face.
    match_iou: f32 = 0.3,
    /// A matched track overlapping its detection by at least this IoU
    /// keeps its filter (moved to the detection and adapted); below it the
    /// filter is trained afresh.
    retrain_iou: f32 = 0.6,
    /// A track missed by this many detections in a row is dropped.
    drop_after: u32 = 2,
};

/// One tracked face. Boxes and landmarks are in frame pixels.
pub const FaceTrack = struct {
    id: u32,
    mosse: visar.video.Mosse,
    box: Box,
    /// Landmarks relative to the box ((x - box.x) / box.w, ...), so they
    /// move and scale with it between detections.
    rel: [5][2]f32,
    /// The score of the last detection.
    score: f32,
    psr: f32 = 0,
    /// The last MOSSE update was confident.
    ok: bool = true,
    /// Detections in a row that did not find this face.
    det_missed: u32 = 0,
    frames: u32 = 0,

    pub fn landmark(t: FaceTrack, i: usize) [2]f32 {
        return .{ t.box.x + t.rel[i][0] * t.box.w, t.box.y + t.rel[i][1] * t.box.h };
    }

    fn healthy(t: FaceTrack) bool {
        return t.ok and t.det_missed == 0;
    }
};

/// Counters of a face-tracking run. Times are sums in milliseconds.
pub const FaceStats = struct {
    frames: u64 = 0,
    /// Frames on which the detector ran (on the whole frame or on crops).
    detect_frames: u64 = 0,
    full_runs: u64 = 0,
    /// Detections that ran only on crops (level 2), and the crops.
    crop_rounds: u64 = 0,
    crops: u64 = 0,
    /// Frames whose scheduled detection was postponed (level 1).
    postponed: u64 = 0,
    /// Detector time of whole-frame and crop rounds, conversion included.
    full_ms: f64 = 0,
    crop_ms: f64 = 0,
    /// The frame to detector input conversion (part of the above; 0 when
    /// the detector shrinks the frame itself, see `FaceTracker.fused`).
    convert_ms: f64 = 0,
    /// MOSSE updates, scheduling, matching and re-initialisation.
    track_ms: f64 = 0,
    tracks_started: u64 = 0,
    tracks_dropped: u64 = 0,
    max_faces: u32 = 0,

    pub fn detectMs(s: FaceStats) f64 {
        return s.full_ms + s.crop_ms;
    }

    pub fn meanDetectMs(s: FaceStats) f64 {
        const n = s.full_runs + s.crop_rounds;
        return if (n > 0) s.detectMs() / @as(f64, @floatFromInt(n)) else 0;
    }

    pub fn meanFullMs(s: FaceStats) f64 {
        return if (s.full_runs > 0) s.full_ms / @as(f64, @floatFromInt(s.full_runs)) else 0;
    }

    pub fn meanCropMs(s: FaceStats) f64 {
        return if (s.crop_rounds > 0) s.crop_ms / @as(f64, @floatFromInt(s.crop_rounds)) else 0;
    }

    /// Detector time level 2 saved: what its crop rounds would have cost as
    /// whole-frame runs (at the measured mean), minus what they cost.
    pub fn cropSavingMs(s: FaceStats) f64 {
        if (s.crop_rounds == 0 or s.full_runs == 0) return 0;
        return @as(f64, @floatFromInt(s.crop_rounds)) * s.meanFullMs() - s.crop_ms;
    }
};

/// What a `FaceTracker.step` did.
pub const FaceStep = enum { tracked, postponed, full, crops };

pub const FaceTracker = struct {
    gpa: std.mem.Allocator,
    opts: FaceOptions,
    /// The ONNX model (owned; the detectors point into it).
    model: []u8,
    frame_w: u32,
    frame_h: u32,
    /// Frame pixels per tracking-image pixel.
    k: u32,
    /// The whole-frame detector input size.
    det_w: u32,
    det_h: u32,
    det_bgra: Image(u8),
    det_bgr: Image(u8),
    /// Frame pixels per detector pixel when both are integers (the
    /// detector then shrinks the frame, or a window of it, straight into
    /// its input; `det_bgra` and `det_bgr` are unused).
    fused: ?[2]u32,
    /// Detectors planned per input size (the whole frame first, then crop
    /// sizes, which are multiples of 32).
    plans: std.ArrayList(Plan) = .empty,
    tracks: std.ArrayList(FaceTrack) = .empty,
    found: std.ArrayList(Face) = .empty,
    next_id: u32 = 1,
    /// Frames since the last detection.
    since: u32 = 0,
    /// Crop-only detections since the last whole-frame one.
    crops_in_row: u32 = 0,
    /// Every 4th row of the tracking image at the last detection (level 1).
    ref: []u8,
    ref_valid: bool = false,
    stats: FaceStats = .{},
    last: FaceStep = .tracked,
    last_detect_ms: f64 = 0,
    use_clock: u64 = 0,

    const Plan = struct { w: u32, h: u32, det: visar.dnn.FaceDetector, used: u64 };
    const max_plans = 8;
    const min_crop = 64;

    /// `model`: YuNet ONNX bytes (copied). Frames are `frame_w` x
    /// `frame_h`; the tracking image is `k` times smaller.
    pub fn create(gpa: std.mem.Allocator, model: []const u8, frame_w: u32, frame_h: u32, k: u32, opts: FaceOptions) !FaceTracker {
        if (frame_w < 32 or frame_h < 32 or k == 0) return error.InvalidSize;
        const owned = try gpa.dupe(u8, model);
        errdefer gpa.free(owned);
        const dw = std.math.clamp(opts.face_width, 32, frame_w);
        const dh_f = @round(@as(f64, @floatFromInt(dw)) * @as(f64, @floatFromInt(frame_h)) / @as(f64, @floatFromInt(frame_w)));
        const dh: u32 = std.math.clamp(@as(u32, @intFromFloat(dh_f)), 16, frame_h);
        var det_bgra = try Image(u8).init(gpa, dw, dh, 4);
        errdefer det_bgra.deinit(gpa);
        var det_bgr = try Image(u8).init(gpa, dw, dh, 3);
        errdefer det_bgr.deinit(gpa);
        const ref = try gpa.alloc(u8, @as(usize, (frame_h / k + 3) / 4) * (frame_w / k));
        errdefer gpa.free(ref);
        var self: FaceTracker = .{
            .gpa = gpa,
            .opts = opts,
            .model = owned,
            .frame_w = frame_w,
            .frame_h = frame_h,
            .k = k,
            .det_w = dw,
            .det_h = dh,
            .det_bgra = det_bgra,
            .det_bgr = det_bgr,
            .fused = visar.dnn.prep.factorsOf(frame_w, frame_h, dw, dh),
            .ref = ref,
        };
        // Plan the whole-frame detector now (this also checks the model).
        errdefer {
            for (self.plans.items) |*p| p.det.deinit();
            self.plans.deinit(gpa);
        }
        _ = try self.plan(dw, dh);
        return self;
    }

    pub fn deinit(self: *FaceTracker) void {
        const gpa = self.gpa;
        for (self.plans.items) |*p| p.det.deinit();
        self.plans.deinit(gpa);
        for (self.tracks.items) |*t| t.mosse.deinit();
        self.tracks.deinit(gpa);
        self.found.deinit(gpa);
        self.det_bgra.deinit(gpa);
        self.det_bgr.deinit(gpa);
        gpa.free(self.ref);
        gpa.free(self.model);
        self.* = undefined;
    }

    /// Forgets every face.
    pub fn reset(self: *FaceTracker) void {
        for (self.tracks.items) |*t| t.mosse.deinit();
        self.tracks.clearRetainingCapacity();
        self.ref_valid = false;
    }

    /// The detector planned for `w` x `h` inputs (planned on first use; the
    /// least recently used crop plan makes room when the cache is full).
    fn plan(self: *FaceTracker, w: u32, h: u32) !*visar.dnn.FaceDetector {
        self.use_clock += 1;
        for (self.plans.items) |*p| if (p.w == w and p.h == h) {
            p.used = self.use_clock;
            return &p.det;
        };
        if (self.plans.items.len == max_plans) {
            var lru: usize = 1; // never the whole-frame plan (index 0)
            for (self.plans.items[1..], 1..) |p, i| {
                if (p.used < self.plans.items[lru].used) lru = i;
            }
            self.plans.items[lru].det.deinit();
            _ = self.plans.swapRemove(lru);
        }
        var det = try visar.dnn.FaceDetector.create(self.gpa, self.model, .{ .width = w, .height = h, .score_threshold = self.opts.score_threshold, .backend = self.opts.backend });
        errdefer det.deinit();
        try self.plans.append(self.gpa, .{ .w = w, .h = h, .det = det, .used = self.use_clock });
        return &self.plans.items[self.plans.items.len - 1].det;
    }

    /// The tracked faces (frame pixels).
    pub fn faces(self: *const FaceTracker) []const FaceTrack {
        return self.tracks.items;
    }

    /// Processes one frame: `frame` (BGRA, the frame size) and `small`, its
    /// grayscale tracking image (1 channel, the frame size / k). With an
    /// `io`, the stages are timed.
    pub fn step(self: *FaceTracker, exec: visar.Exec, io: ?Io, frame: Image(u8), small: Image(u8)) !FaceStep {
        if (frame.channels != 4 or frame.width != self.frame_w or frame.height != self.frame_h) return error.SizeMismatch;
        if (small.channels != 1 or small.width != self.frame_w / self.k or small.height != self.frame_h / self.k) return error.SizeMismatch;
        const s = &self.stats;
        var clock: Clock = .{ .io = io };
        _ = clock.lap();
        s.frames += 1;
        self.since += 1;

        // ---- every face follows its MOSSE filter
        self.updateFilters(exec, small);

        // ---- is a detection due, and of what kind?
        var weak = false;
        var healthy = true;
        for (self.tracks.items) |t| {
            weak = weak or !t.ok;
            healthy = healthy and t.healthy();
        }
        var what: FaceStep = .tracked;
        if (self.tracks.items.len == 0 or weak) {
            what = .full;
        } else if (self.since >= @max(1, self.opts.detect_every)) {
            const crops_ok = self.opts.level >= 2 and healthy and self.crops_in_row + 1 < @max(1, self.opts.full_every);
            what = if (crops_ok) .crops else .full;
            if (self.opts.level >= 1 and healthy and self.ref_valid and self.since < self.opts.max_postpone and
                self.changeSinceRef(small) < self.opts.still_threshold) what = .postponed;
        }
        s.track_ms += clock.lap();
        self.last = what;
        if (what == .postponed) s.postponed += 1;
        if (what != .full and what != .crops) return what;

        // ---- detect
        if (self.fused == null) try self.convert(exec, frame);
        const t_convert = clock.lap();
        self.found.clearRetainingCapacity();
        if (what == .crops and !try self.detectCrops(exec, frame)) what = .full;
        if (what == .full) try self.detectIn(exec, frame, 0, 0, self.det_w, self.det_h);
        const t_det = t_convert + clock.lap();
        s.convert_ms += t_convert;
        s.detect_frames += 1;
        if (what == .full) {
            s.full_runs += 1;
            s.full_ms += t_det;
            self.crops_in_row = 0;
        } else {
            s.crop_rounds += 1;
            s.crop_ms += t_det;
            self.crops_in_row += 1;
        }
        self.last = what;
        self.last_detect_ms = t_det;

        // ---- correct the tracks, start new ones, drop lost ones
        try self.match(small, what == .full);
        self.since = 0;
        self.snapshot(small);
        s.track_ms += clock.lap();
        s.max_faces = @max(s.max_faces, @as(u32, @intCast(self.tracks.items.len)));
        return what;
    }

    /// The frame, area-downscaled to the detector input, as BGR.
    fn convert(self: *FaceTracker, exec: visar.Exec, frame: Image(u8)) !void {
        try ip.resize(exec, self.gpa, frame, self.det_bgra, .area);
        try ip.cvtColor(exec, self.det_bgra, self.det_bgr, .bgra2bgr);
    }

    fn updateFilters(self: *FaceTracker, exec: visar.Exec, small: Image(u8)) void {
        const Job = struct { tracks: []FaceTrack, gray: Image(u8), k: f32 };
        exec.parallelFor(self.tracks.items.len, 1, Job{ .tracks = self.tracks.items, .gray = small, .k = @floatFromInt(self.k) }, struct {
            fn run(j: Job, a: usize, b: usize) void {
                for (j.tracks[a..b]) |*t| {
                    const r = t.mosse.update(j.gray) catch visar.video.mosse.Result{ .box = t.mosse.box(), .psr = 0, .ok = false };
                    t.box = .{ .x = r.box.x * j.k, .y = r.box.y * j.k, .w = r.box.w * j.k, .h = r.box.h * j.k };
                    t.psr = r.psr;
                    t.ok = r.ok;
                    t.frames += 1;
                }
            }
        }.run);
    }

    /// Runs the detector on the (x, y, w, h) window of the detector input
    /// and appends what it finds, in frame pixels.
    fn detectIn(self: *FaceTracker, exec: visar.Exec, frame: Image(u8), x: u32, y: u32, w: u32, h: u32) !void {
        const det = try self.plan(w, h);
        if (self.fused) |f| {
            // The window of the frame, shrunk into the detector's input;
            // results come back in window pixels.
            const res = try det.detectFrame(exec, try frame.roi(x * f[0], y * f[1], w * f[0], h * f[1]));
            const ox: f32 = @floatFromInt(x * f[0]);
            const oy: f32 = @floatFromInt(y * f[1]);
            for (res) |r| {
                var g = r;
                g.box[0] += ox;
                g.box[1] += oy;
                for (&g.landmarks) |*l| l.* = .{ l[0] + ox, l[1] + oy };
                try self.found.append(self.gpa, g);
            }
            return;
        }
        const view = try self.det_bgr.roi(x, y, w, h);
        const res = try det.detect(exec, view);
        const sx = @as(f32, @floatFromInt(self.frame_w)) / @as(f32, @floatFromInt(self.det_w));
        const sy = @as(f32, @floatFromInt(self.frame_h)) / @as(f32, @floatFromInt(self.det_h));
        const ox: f32 = @floatFromInt(x);
        const oy: f32 = @floatFromInt(y);
        for (res) |f| {
            var g = f;
            g.box = .{ (f.box[0] + ox) * sx, (f.box[1] + oy) * sy, f.box[2] * sx, f.box[3] * sy };
            for (&g.landmarks, f.landmarks) |*l, m| l.* = .{ (m[0] + ox) * sx, (m[1] + oy) * sy };
            try self.found.append(self.gpa, g);
        }
    }

    /// Level 2: detects in a square window around each tracked face, at the
    /// whole-frame detector's scale. Returns false (nothing done) when the
    /// windows would not be clearly cheaper than the whole frame.
    fn detectCrops(self: *FaceTracker, exec: visar.Exec, frame: Image(u8)) !bool {
        const sx = @as(f32, @floatFromInt(self.det_w)) / @as(f32, @floatFromInt(self.frame_w));
        const sy = @as(f32, @floatFromInt(self.det_h)) / @as(f32, @floatFromInt(self.frame_h));
        var wins: [64][3]u32 = undefined;
        var n: usize = 0;
        var area: u64 = 0;
        for (self.tracks.items) |t| {
            if (n == wins.len) return false;
            const side = self.opts.crop_scale * @max(t.box.w * sx, t.box.h * sy);
            const s32: u32 = @intFromFloat(@ceil(@max(side, min_crop) / 32) * 32);
            if (s32 > self.det_w or s32 > self.det_h) return false;
            const half = @as(f32, @floatFromInt(s32)) / 2;
            const x = std.math.clamp(@round((t.box.x + t.box.w / 2) * sx - half), 0, @as(f32, @floatFromInt(self.det_w - s32)));
            const y = std.math.clamp(@round((t.box.y + t.box.h / 2) * sy - half), 0, @as(f32, @floatFromInt(self.det_h - s32)));
            wins[n] = .{ @intFromFloat(x), @intFromFloat(y), s32 };
            n += 1;
            area += @as(u64, s32) * s32;
        }
        // Runs on crops covering most of the frame cost more than one run
        // on all of it.
        if (n == 0 or 10 * area > 6 * @as(u64, self.det_w) * self.det_h) return false;
        for (wins[0..n]) |r| try self.detectIn(exec, frame, r[0], r[1], r[2], r[2]);
        self.stats.crops += n;
        // Overlapping crops can find a face twice: keep the best of each.
        const f = &self.found;
        var i: usize = 0;
        while (i < f.items.len) : (i += 1) {
            var j: usize = i + 1;
            while (j < f.items.len) {
                if (iou(faceBox(f.items[i]), faceBox(f.items[j])) > 0.3) {
                    if (f.items[j].score > f.items[i].score) f.items[i] = f.items[j];
                    _ = f.swapRemove(j);
                } else j += 1;
            }
        }
        return true;
    }

    /// Matches `found` to the tracks (greedy by IoU): matched tracks restart
    /// their filter on the detected box (drift correction) and take its
    /// landmarks and score; with `full`, unmatched detections start tracks;
    /// tracks missed too often are dropped.
    fn match(self: *FaceTracker, small: Image(u8), full: bool) !void {
        const gpa = self.gpa;
        const Pair = struct { v: f32, t: u32, d: u32 };
        var pairs: std.ArrayList(Pair) = .empty;
        defer pairs.deinit(gpa);
        for (self.tracks.items, 0..) |t, ti| for (self.found.items, 0..) |f, di| {
            const v = iou(t.box, faceBox(f));
            if (v >= self.opts.match_iou) try pairs.append(gpa, .{ .v = v, .t = @intCast(ti), .d = @intCast(di) });
        };
        std.mem.sort(Pair, pairs.items, {}, struct {
            fn gt(_: void, a: Pair, b: Pair) bool {
                return a.v > b.v;
            }
        }.gt);
        const t_used = try gpa.alloc(bool, self.tracks.items.len);
        defer gpa.free(t_used);
        const d_used = try gpa.alloc(bool, self.found.items.len);
        defer gpa.free(d_used);
        @memset(t_used, false);
        @memset(d_used, false);
        for (pairs.items) |p| {
            if (t_used[p.t] or d_used[p.d]) continue;
            t_used[p.t] = true;
            d_used[p.d] = true;
            const t = &self.tracks.items[p.t];
            const f = self.found.items[p.d];
            const box = self.clampBox(faceBox(f));
            if (p.v >= self.opts.retrain_iou) {
                // Close to where the filter has it: move it there and
                // adapt (cheap). Further off: the filter may have learnt
                // the wrong patch, so train a fresh one.
                try t.mosse.correct(small, self.toSmall(box));
            } else {
                const m = try visar.video.Mosse.init(gpa, small, self.toSmall(box), .{});
                t.mosse.deinit();
                t.mosse = m;
            }
            t.box = box;
            t.rel = relLandmarks(f, box);
            t.score = f.score;
            t.ok = true;
            t.det_missed = 0;
        }
        for (self.tracks.items, t_used) |*t, u| {
            if (!u) t.det_missed += 1;
        }
        var i: usize = 0;
        while (i < self.tracks.items.len) {
            if (self.tracks.items[i].det_missed >= self.opts.drop_after) {
                self.tracks.items[i].mosse.deinit();
                _ = self.tracks.orderedRemove(i);
                self.stats.tracks_dropped += 1;
            } else i += 1;
        }
        if (!full) return;
        for (self.found.items, d_used) |f, u| {
            if (u or self.tracks.items.len >= self.opts.max_faces) continue;
            const box = self.clampBox(faceBox(f));
            if (box.w < 4 or box.h < 4) continue;
            var m = try visar.video.Mosse.init(gpa, small, self.toSmall(box), .{});
            errdefer m.deinit();
            try self.tracks.append(gpa, .{ .id = self.next_id, .mosse = m, .box = box, .rel = relLandmarks(f, box), .score = f.score });
            self.next_id += 1;
            self.stats.tracks_started += 1;
        }
    }

    fn clampBox(self: *const FaceTracker, b: Box) Box {
        const W: f32 = @floatFromInt(self.frame_w);
        const H: f32 = @floatFromInt(self.frame_h);
        const x0 = std.math.clamp(b.x, 0, W);
        const y0 = std.math.clamp(b.y, 0, H);
        const x1 = std.math.clamp(b.x + b.w, 0, W);
        const y1 = std.math.clamp(b.y + b.h, 0, H);
        return .{ .x = x0, .y = y0, .w = x1 - x0, .h = y1 - y0 };
    }

    fn toSmall(self: *const FaceTracker, b: Box) Box {
        const k: f32 = @floatFromInt(self.k);
        return .{ .x = b.x / k, .y = b.y / k, .w = @max(4, b.w / k), .h = @max(4, b.h / k) };
    }

    /// Mean absolute difference between every 4th row of `small` and the
    /// same rows at the last detection.
    fn changeSinceRef(self: *const FaceTracker, small: Image(u8)) f32 {
        const w = small.width;
        var sum: u64 = 0;
        var n: u64 = 0;
        var y: usize = 0;
        var off: usize = 0;
        while (y < small.height) : (y += 4) {
            sum += sad(small.row(y)[0..w], self.ref[off..][0..w]);
            off += w;
            n += w;
        }
        return if (n > 0) @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(n)) else 0;
    }

    fn snapshot(self: *FaceTracker, small: Image(u8)) void {
        const w = small.width;
        var y: usize = 0;
        var off: usize = 0;
        while (y < small.height) : (y += 4) {
            @memcpy(self.ref[off..][0..w], small.row(y)[0..w]);
            off += w;
        }
        self.ref_valid = true;
    }
};

/// Sum of absolute differences.
fn sad(a: []const u8, b: []const u8) u64 {
    const N = 32;
    const V = @Vector(N, u8);
    var sum: u64 = 0;
    var x: usize = 0;
    while (x + N <= a.len) : (x += N) {
        const va: V = a[x..][0..N].*;
        const vb: V = b[x..][0..N].*;
        const d: @Vector(N, u16) = (va -| vb) | (vb -| va);
        sum += @reduce(.Add, d);
    }
    while (x < a.len) : (x += 1) sum += @max(a[x], b[x]) - @min(a[x], b[x]);
    return sum;
}

fn faceBox(f: Face) Box {
    return .{ .x = f.box[0], .y = f.box[1], .w = f.box[2], .h = f.box[3] };
}

fn relLandmarks(f: Face, b: Box) [5][2]f32 {
    var r: [5][2]f32 = undefined;
    for (&r, f.landmarks) |*o, l| o.* = .{ (l[0] - b.x) / @max(b.w, 1), (l[1] - b.y) / @max(b.h, 1) };
    return r;
}

pub fn iou(a: Box, b: Box) f32 {
    const x0 = @max(a.x, b.x);
    const y0 = @max(a.y, b.y);
    const x1 = @min(a.x + a.w, b.x + b.w);
    const y1 = @min(a.y + a.h, b.y + b.h);
    const inter = @max(0, x1 - x0) * @max(0, y1 - y0);
    const uni = a.w * a.h + b.w * b.h - inter;
    return if (uni > 0) inter / uni else 0;
}

// ---------------------------------------------------------------------------
// A still image moving across a plain canvas (`--source image:PATH`): a test
// source with a known motion, to exercise the face tracker anywhere.
// ---------------------------------------------------------------------------

pub const StillScene = struct {
    gpa: std.mem.Allocator,
    width: u32,
    height: u32,
    /// The image, scaled to fit, BGRA.
    img: Image(u8),
    /// The next frame to render.
    frame: u32 = 0,

    /// Fraction of the frame's width and height the image may cover.
    pub const fill = 0.75;
    /// The motion repeats every `period` frames: `moving` frames along a
    /// figure of eight (easing in and out), then still.
    pub const period = 300;
    pub const moving = 240;
    const canvas: u32 = 0xff_3c_40_48; // BGRA (72, 64, 60)

    /// `src`: the image, BGR (3 channels) or BGRA.
    pub fn init(gpa: std.mem.Allocator, exec: visar.Exec, src: Image(u8), width: u32, height: u32) !StillScene {
        if (src.channels != 3 and src.channels != 4) return error.UnsupportedChannels;
        const fw: f64 = @floatFromInt(width);
        const fh: f64 = @floatFromInt(height);
        const s = @min(fill * fw / @as(f64, @floatFromInt(src.width)), fill * fh / @as(f64, @floatFromInt(src.height)));
        const iw: u32 = std.math.clamp(@as(u32, @intFromFloat(@round(s * @as(f64, @floatFromInt(src.width))))), 1, width);
        const ih: u32 = std.math.clamp(@as(u32, @intFromFloat(@round(s * @as(f64, @floatFromInt(src.height))))), 1, height);
        var scaled = try Image(u8).init(gpa, iw, ih, src.channels);
        defer scaled.deinit(gpa);
        try ip.resize(exec, gpa, src, scaled, .area);
        var img = try Image(u8).init(gpa, iw, ih, 4);
        errdefer img.deinit(gpa);
        if (src.channels == 3) try ip.cvtColor(exec, scaled, img, .bgr2bgra) else img.copyFrom(scaled);
        return .{ .gpa = gpa, .width = width, .height = height, .img = img };
    }

    pub fn deinit(self: *StillScene) void {
        self.img.deinit(self.gpa);
        self.* = undefined;
    }

    /// Where the image's top-left corner is in frame `f`.
    pub fn offset(self: *const StillScene, f: u32) [2]i32 {
        const mx = @as(f64, @floatFromInt(self.width - self.img.width)) / 2;
        const my = @as(f64, @floatFromInt(self.height - self.img.height)) / 2;
        const ph = f % period;
        var theta: f64 = 0;
        if (ph < moving) {
            const p = @as(f64, @floatFromInt(ph)) / moving;
            theta = 2 * std.math.pi * p - @sin(2 * std.math.pi * p);
        }
        return .{ @intFromFloat(@round(mx + mx * @sin(theta))), @intFromFloat(@round(my + my * @sin(2 * theta))) };
    }

    /// Renders the next frame into `dst` (BGRA, the scene size) and returns
    /// the image's offset in it.
    pub fn render(self: *StillScene, dst: Image(u8)) [2]i32 {
        std.debug.assert(dst.channels == 4 and dst.width == self.width and dst.height == self.height);
        const off = self.offset(self.frame);
        self.frame += 1;
        const bg = std.mem.nativeToLittle(u32, canvas);
        const ox: usize = @intCast(off[0]);
        const oy: usize = @intCast(off[1]);
        const iw: usize = self.img.width;
        for (0..dst.height) |y| {
            const r = dst.row(y)[0 .. @as(usize, dst.width) * 4];
            const px: []align(1) u32 = std.mem.bytesAsSlice(u32, r);
            if (y < oy or y >= oy + self.img.height) {
                @memset(px, bg);
                continue;
            }
            @memset(px[0..ox], bg);
            @memcpy(r[ox * 4 ..][0 .. iw * 4], self.img.row(y - oy)[0 .. iw * 4]);
            @memset(px[ox + iw ..], bg);
        }
        return off;
    }
};

/// Scores face tracks against a `StillScene`'s known motion: the faces
/// tracked after the first detection, moved with the image, are the ground
/// truth.
pub const StillTruth = struct {
    /// The true faces, relative to the image's top-left corner.
    refs: [16]Box = undefined,
    n: usize = 0,
    frames: u64 = 0,
    /// Frames where some tracked face overlaps a true one by IoU >= 0.5.
    followed: u64 = 0,
    /// Sums over frames of the best track-to-truth IoU, its centre error,
    /// and the number of true faces with a track at IoU >= 0.5.
    iou_sum: f64 = 0,
    err_sum: f64 = 0,
    faces_sum: f64 = 0,

    pub fn score(self: *StillTruth, tracks: []const FaceTrack, off: [2]i32) void {
        const ox: f32 = @floatFromInt(off[0]);
        const oy: f32 = @floatFromInt(off[1]);
        if (self.n == 0) {
            for (tracks) |t| {
                if (self.n == self.refs.len) break;
                self.refs[self.n] = .{ .x = t.box.x - ox, .y = t.box.y - oy, .w = t.box.w, .h = t.box.h };
                self.n += 1;
            }
            if (self.n == 0) return;
        }
        self.frames += 1;
        var best: f32 = 0;
        var best_err: f32 = 0;
        var hits: u32 = 0;
        for (self.refs[0..self.n]) |r0| {
            const r: Box = .{ .x = r0.x + ox, .y = r0.y + oy, .w = r0.w, .h = r0.h };
            var b: f32 = 0;
            for (tracks) |t| {
                const v = iou(r, t.box);
                b = @max(b, v);
                if (v > best) {
                    best = v;
                    best_err = std.math.hypot(r.cx() - t.box.cx(), r.cy() - t.box.cy());
                }
            }
            if (b >= 0.5) hits += 1;
        }
        if (best >= 0.5) self.followed += 1;
        self.iou_sum += best;
        self.err_sum += best_err;
        self.faces_sum += @floatFromInt(hits);
    }
};

test "a still image moving over the canvas" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 40, 20, 3);
    defer src.deinit(gpa);
    src.fill(200);
    var scene = try StillScene.init(gpa, .serial, src, 160, 90);
    defer scene.deinit();
    try std.testing.expectEqual(@as(u32, 120), scene.img.width);
    var frame = try Image(u8).init(gpa, 160, 90, 4);
    defer frame.deinit(gpa);
    var prev = scene.offset(0);
    for (0..StillScene.period + 1) |f| {
        const off = scene.render(frame);
        // Smooth: never more than a few pixels per frame.
        try std.testing.expect(@abs(off[0] - prev[0]) <= 3 and @abs(off[1] - prev[1]) <= 3);
        prev = off;
        try std.testing.expectEqual(@as(u8, 200), frame.at(@intCast(off[0]), @intCast(off[1]), 0));
        if (off[0] > 0) try std.testing.expectEqual(@as(u8, 72), frame.at(@intCast(off[0] - 1), @intCast(off[1]), 0));
        if (f >= StillScene.moving and f < StillScene.period) try std.testing.expectEqual(scene.offset(StillScene.moving), off);
    }
}

test "change measure ignores identical frames" {
    var a: [100]u8 = undefined;
    for (&a, 0..) |*v, i| v.* = @intCast(i);
    var b = a;
    try std.testing.expectEqual(@as(u64, 0), sad(&a, &b));
    b[3] +%= 5;
    b[70] -%= 7;
    try std.testing.expectEqual(@as(u64, 12), sad(&a, &b));
}

/// Reads a file from `VISAR_MODELS` (null if unset or missing).
fn readModelFile(gpa: std.mem.Allocator, name: []const u8) !?[]u8 {
    const dir = std.testing.environ.getAlloc(gpa, "VISAR_MODELS") catch return null;
    defer gpa.free(dir);
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(std.testing.io, path, gpa, .limited(1 << 28)) catch null;
}

test "face tracking follows a moving selfie with few detector runs" {
    // Needs VISAR_MODELS with face_detection_yunet_2023mar.onnx and
    // largest_selfie.jpg (the OpenCV Model Zoo files); skipped otherwise.
    const gpa = std.testing.allocator;
    const model = try readModelFile(gpa, "face_detection_yunet_2023mar.onnx") orelse return error.SkipZigTest;
    defer gpa.free(model);
    const jpg = try readModelFile(gpa, "largest_selfie.jpg") orelse return error.SkipZigTest;
    defer gpa.free(jpg);
    var decoded = try visar.codecs.decode(gpa, jpg, .{ .mode = .color, .order = .bgr });
    defer decoded.deinit(gpa);

    var pool = visar.Pool.init(gpa, std.testing.io, 4);
    defer pool.deinit();
    const exec = visar.Exec.on(&pool);
    const W = 1280;
    const H = 720;
    const k = 2;
    var scene = try StillScene.init(gpa, exec, decoded.u8, W, H);
    defer scene.deinit();
    var ft = try FaceTracker.create(gpa, model, W, H, k, .{ .face_width = 640 });
    defer ft.deinit();
    var frame = try Image(u8).init(gpa, W, H, 4);
    defer frame.deinit(gpa);
    var small = try Image(u8).init(gpa, W / k, H / k, 1);
    defer small.deinit(gpa);
    var truth: StillTruth = .{};
    const n = StillScene.period + 60;
    for (0..n) |_| {
        const off = scene.render(frame);
        try visar.videoio.frame.grayDownsample(exec, frame, small, k);
        _ = try ft.step(exec, std.testing.io, frame, small);
        truth.score(ft.faces(), off);
    }
    const s = ft.stats;
    try std.testing.expect(truth.n >= 1);
    // Every frame has a face that moved with the image.
    try std.testing.expectEqual(@as(u64, n), truth.frames);
    try std.testing.expectEqual(truth.frames, truth.followed);
    try std.testing.expect(truth.err_sum / @as(f64, @floatFromInt(truth.frames)) < 6);
    // The detector ran on far fewer frames than there were.
    try std.testing.expect(s.detect_frames * 3 < n);
    try std.testing.expect(s.crop_rounds > 0);
    try std.testing.expect(s.postponed > 0);
}
