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

    // Static background: sky gradient, road, lane marks, texture.
    var bg = try Image(u8).init(gpa, W, H, 3);
    defer bg.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(opts.seed);
    const rnd = prng.random();
    for (0..H) |y| {
        const r = bg.row(y);
        const fy: f32 = @as(f32, @floatFromInt(y)) / @as(f32, @floatFromInt(H));
        for (0..W) |x| {
            const t = rnd.intRangeAtMost(i32, -10, 10);
            const base: [3]f32 = if (fy < 0.3) .{ 120 + 80 * fy, 160 + 60 * fy, 210 } else if (fy < 0.42) .{ 70, 110, 60 } else .{ 75, 78, 82 };
            for (0..3) |c| r[x * 3 + c] = visar.core.sat.cast(u8, base[c] + @as(f32, @floatFromInt(t)));
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
    defer objs.deinit(gpa);
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

    var frame = try Image(u8).init(gpa, W, H, 3);
    defer frame.deinit(gpa);
    var gray = try Image(u8).init(gpa, W, H, 1);
    defer gray.deinit(gpa);
    var blurred = try Image(u8).init(gpa, W, H, 1);
    defer blurred.deinit(gpa);
    var mask = try Image(u8).init(gpa, W, H, 1);
    defer mask.deinit(gpa);
    var mask2 = try Image(u8).init(gpa, W, H, 1);
    defer mask2.deinit(gpa);
    var labels = try Image(i32).init(gpa, W, H, 1);
    defer labels.deinit(gpa);
    const gw = @min(opts.gif_width, W);
    const gh = @max(1, gw * H / W);
    var small = try Image(u8).init(gpa, gw, gh, 3);
    defer small.deinit(gpa);

    var model = try visar.video.Background.init(gpa, W, H, 4);
    defer model.deinit(gpa);
    var tracker = visar.video.Tracker.init(gpa);
    defer tracker.deinit();
    tracker.gate = 40 * sx;
    var anim = try visar.codecs.gif.Animation.init(gpa, gw, gh, .color, .rgb);
    defer anim.deinit();

    var el3_buf: [9]u8 = undefined;
    var el_buf: [81]u8 = undefined;
    const el3 = ip.structuringElement(.rect, 3, 3, &el3_buf);
    const el9 = ip.structuringElement(.ellipse, 9, 9, &el_buf);

    var times: std.ArrayList(f64) = .empty;
    defer times.deinit(gpa);
    var boxes: std.ArrayList(visar.video.Box) = .empty;
    defer boxes.deinit(gpa);

    // Ground-truth scoring.
    var gt_visible: u64 = 0;
    var gt_found: u64 = 0;
    var id_switches: u32 = 0;
    var last_track_for: [4096]u32 = @splat(0);

    for (0..opts.frames) |f_| {
        const f: u32 = @intCast(f_);
        // ---- render the scene
        frame.copyFrom(bg);
        const flicker: i32 = @intFromFloat(4 * @sin(@as(f32, @floatFromInt(f)) * 0.21));
        for (objs.items) |*o| {
            if (f < o.start) continue;
            o.x += o.vx;
            o.y += o.vy;
            const xi: i64 = @intFromFloat(o.x);
            const yi: i64 = @intFromFloat(o.y);
            if (o.round) {
                draw.circle(frame, xi + @as(i64, @intFromFloat(o.w / 2)), yi + @as(i64, @intFromFloat(o.h / 2)), @intFromFloat(o.h / 2), &o.color, 0);
            } else {
                draw.fillRect(frame, xi, yi, xi + @as(i64, @intFromFloat(o.w)), yi + @as(i64, @intFromFloat(o.h)), &o.color);
                draw.fillRect(frame, xi + 6, yi + 4, xi + @as(i64, @intFromFloat(o.w * 0.45)), yi + @as(i64, @intFromFloat(o.h * 0.45)), &.{ 170, 210, 235 });
            }
            // Wrap around after leaving, as a new arrival.
            const Wf: f32 = @floatFromInt(W);
            if (o.vx > 0 and o.x > Wf + 20) {
                o.x = -o.w - 20;
                o.gid += 100;
            }
            if (o.vx < 0 and o.x < -o.w - 20) {
                o.x = Wf + 20;
                o.gid += 100;
            }
        }
        // Sensor noise and flicker.
        {
            var state: u32 = f *% 2654435761 +% 12345;
            const px = frame.slice();
            for (px) |*v| {
                state = state *% 1664525 +% 1013904223;
                const n: i32 = @as(i32, @intCast((state >> 24) & 15)) - 8 + flicker;
                v.* = @intCast(std.math.clamp(@as(i32, v.*) + n, 0, 255));
            }
        }

        // ---- the timed VISAR pipeline
        const t0 = Io.Clock.awake.now(io);
        try ip.cvtColor(exec, frame, gray, .rgb2gray);
        try ip.gaussianBlur(exec, gpa, gray, blurred, .{ .kw = 5, .kh = 5 });
        try model.apply(exec, blurred, mask, 28);
        try ip.morphologyEx(exec, gpa, mask, mask2, .open, el3, 1);
        try ip.dilate(exec, gpa, mask2, mask, el9, 1);
        var cc = try ip.connectedComponentsWithStats(gpa, mask, labels, .eight);
        defer cc.deinit(gpa);
        boxes.clearRetainingCapacity();
        const min_area: u64 = @intFromFloat(250 * sx * sx);
        for (cc.stats[1..]) |st| {
            if (st.area < min_area) continue;
            try boxes.append(gpa, .{ .x = @floatFromInt(st.left), .y = @floatFromInt(st.top), .w = @floatFromInt(st.width), .h = @floatFromInt(st.height) });
        }
        try tracker.update(boxes.items);
        const t1 = Io.Clock.awake.now(io);
        const ms = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
        if (f >= warmup) try times.append(gpa, ms);

        // ---- score against ground truth
        if (f >= warmup) for (objs.items) |o| {
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
        for (tracker.tracks.items) |t| {
            if (!t.confirmed()) continue;
            const col = idColor(t.id);
            var k: usize = 1;
            while (k < t.trail_count) : (k += 1) {
                const a = t.trailPoint(k - 1);
                const b = t.trailPoint(k);
                draw.line(frame, @intFromFloat(a[0]), @intFromFloat(a[1]), @intFromFloat(b[0]), @intFromFloat(b[1]), &col, 3);
            }
            const bx: i64 = @intFromFloat(t.box.x);
            const by: i64 = @intFromFloat(t.box.y);
            draw.rectangle(frame, bx - 3, by - 3, @as(i64, @intFromFloat(t.box.w)) + 6, @as(i64, @intFromFloat(t.box.h)) + 6, &col, 3);
            var lbuf: [16]u8 = undefined;
            const label = std.fmt.bufPrint(&lbuf, "#{d}", .{t.id}) catch "#";
            draw.fillRect(frame, bx - 3, by - 24, bx - 3 + @as(i64, @intCast(label.len)) * 12 + 6, by - 5, &col);
            _ = draw.text(frame, bx, by - 21, label, 2, &.{ 0, 0, 0 });
        }
        var hud: [96]u8 = undefined;
        draw.fillRect(frame, 0, 0, W, 64, &.{ 10, 12, 20 });
        _ = draw.text(frame, 12, 8, "VISAR LIVE TRACKING", 3, &.{ 255, 255, 255 });
        const line2 = std.fmt.bufPrint(&hud, "FRAME {d:>3}  {d}x{d}  PIPELINE {d:.2} MS ({d:.0} FPS)  TRACKS {d}", .{ f, W, H, ms, 1000.0 / ms, countConfirmed(tracker) }) catch "";
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

fn countConfirmed(t: visar.video.Tracker) usize {
    var n: usize = 0;
    for (t.tracks.items) |tr| n += @intFromBool(tr.confirmed());
    return n;
}

fn idColor(id: u32) [3]u8 {
    const cols = [_][3]u8{ .{ 255, 80, 80 }, .{ 80, 220, 255 }, .{ 255, 220, 60 }, .{ 120, 255, 120 }, .{ 255, 120, 255 }, .{ 255, 160, 60 }, .{ 160, 160, 255 }, .{ 60, 255, 200 } };
    return cols[id % cols.len];
}
