//! `visar track`: live multi-object tracking from a camera (or the
//! synthetic scene of `visar demo track`), shown in a native window with
//! IDs, trails and an FPS HUD.
//!
//! Three stages overlap, passing BGRA frames through a lock-free
//! `FrameRing` that always hands on the newest frame:
//!
//!   capture   camera callback (or the synthetic scene's thread): copies the
//!             frame into a free ring buffer
//!   process   worker thread: gray + downscale to the detection size in one
//!             pass, the tracking pipeline (track_lib.Detector), then boxes,
//!             trails and the HUD drawn on the full-size frame
//!   display   main thread: window events, and the newest processed frame
//!             handed to Core Animation without a copy
//!
//! With --headless there is no window; stats are printed every second and a
//! summary (or --json) at the end.

const std = @import("std");
const builtin = @import("builtin");
const visar = @import("visar");
const track = @import("track_lib");
const args = @import("args.zig");
const term = @import("term.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const imageio = @import("imageio.zig");

const Io = std.Io;
const Image = visar.Image;
const videoio = visar.videoio;
const camera = videoio.camera;
const FrameRing = videoio.FrameRing;
const ui = visar.ui;
const draw = visar.imgproc.draw;

pub const opts = [_]args.Opt{
    .{ .long = "camera", .value = "N", .help = "Camera number from --list-cameras (default: the default camera)" },
    .{ .long = "size", .value = "WxH", .help = "Largest frame size wanted (default 1280x720)" },
    .{ .long = "fps", .value = "N", .help = "Frame rate wanted (default: the camera's fastest; synthetic: 60)" },
    .{ .long = "source", .value = "SRC", .help = "camera (default) or synthetic (the demo scene; runs anywhere)" },
    .{ .long = "headless", .help = "No window: print stats every second and a summary" },
    .{ .long = "frames", .short = 'n', .value = "N", .help = "Stop after N processed frames" },
    .{ .long = "detect-width", .value = "N", .help = "Width the detector works at (default 640; 0 = full size)" },
    .{ .long = "record", .value = "FILE", .help = "Also write the annotated frames to an animated GIF; same as -o" },
    .{ .long = "lock", .value = "X,Y,W,H", .help = "Lock onto the object in this box (frame pixels); repeatable" },
    .{ .long = "mode", .value = "MODE", .help = "both (default), motion or locks (Tab cycles in the window)" },
    .{ .long = "list-cameras", .help = "List the cameras, their sizes and frame rates, and exit" },
};

/// What is tracked: moving objects (background subtraction), the objects
/// the user locked onto (MOSSE correlation filters), or both.
pub const Mode = enum(u8) {
    both,
    motion,
    locks,

    fn next(m: Mode) Mode {
        return switch (m) {
            .both => .motion,
            .motion => .locks,
            .locks => .both,
        };
    }
};

const max_locks = 8;
/// A lock that misses this many frames in a row is shown as lost.
const lost_after = 10;
/// The synthetic scene starts this many frames in, so objects are on screen.
const synthetic_preroll = 100;

pub const Source = enum { camera, synthetic };

const max_record_frames = 900;
/// Without a frame from the camera for this long, something is wrong.
const no_frames_timeout_s = 3.0;

pub fn run(ctx: *Ctx, p: *args.Parsed) !void {
    if (p.positionals.items.len > 0) {
        ctx.hint("'visar track' takes options only, e.g. 'visar track --source synthetic'", .{});
        return ctx.usage("unexpected argument '{s}'", .{p.positionals.items[0]});
    }
    if (p.flag("list-cameras")) return listCameras(ctx);

    var cfg: Config = .{};
    if (p.value("source")) |v| cfg.source = try args.choice(Source, ctx.diag, "source", v);
    if (p.value("camera")) |v| {
        cfg.camera_index = try args.int(usize, ctx.diag, "--camera", v, 0, 63);
        if (cfg.source == .synthetic) ctx.warn("--camera does not apply to --source synthetic; ignored", .{});
    }
    if (p.value("size")) |v| {
        const s = try args.size(ctx.diag, v, false);
        cfg.width = s.w.?;
        cfg.height = s.h.?;
        if (cfg.width < track.Options.min_width or cfg.height < track.Options.min_height) return ctx.usage("--size must be at least {d}x{d}", .{ track.Options.min_width, track.Options.min_height });
        if (cfg.width > 7680 or cfg.height > 4320) return ctx.usage("--size is limited to 7680x4320", .{});
    }
    if (p.value("fps")) |v| {
        const f = try args.positive(ctx.diag, "--fps", v);
        if (f > 1000) return ctx.usage("--fps must be at most 1000", .{});
        cfg.fps = f;
    }
    if (p.value("frames")) |v| cfg.frame_limit = try args.int(usize, ctx.diag, "--frames", v, 1, std.math.maxInt(u32));
    if (p.value("detect-width")) |v| cfg.detect_width = try args.int(u32, ctx.diag, "--detect-width", v, 0, 7680);
    cfg.headless = p.flag("headless");
    if (p.value("mode")) |v| cfg.mode = try args.choice(Mode, ctx.diag, "mode", v);
    for (p.hits.items) |h| {
        if (!std.mem.eql(u8, h.opt.long, "lock")) continue;
        if (cfg.lock_count == max_locks) return ctx.usage("at most {d} --lock boxes", .{max_locks});
        cfg.locks[cfg.lock_count] = try parseBox(ctx, h.value.?);
        cfg.lock_count += 1;
    }
    cfg.threads_given = p.value("threads") != null or (if (ctx.env.get("VISAR_THREADS")) |v| v.len > 0 else false);

    if (p.value("record") != null and ctx.output != null) return ctx.usage("give the recording once, with --record or -o", .{});
    const record_path = p.value("record") orelse ctx.output;
    var record_out: ?imageio.Output = null;
    if (record_path) |path| {
        if (ctx.format) |f| if (f != .gif) {
            ctx.hint("recordings are animated GIFs; drop --format", .{});
            return ctx.usage("--format {s} does not apply to 'visar track'", .{@tagName(f)});
        };
        if (std.mem.eql(u8, path, "-")) return ctx.usage("--record needs a file name", .{});
        ctx.format = .gif;
        record_out = try imageio.prepare(ctx, path);
    }

    if (cfg.source == .camera and !camera.supported) {
        ctx.hint("try 'visar track --source synthetic', which runs the tracker on a generated scene", .{});
        return ctx.fail("no camera support on {s} yet", .{@tagName(builtin.os.tag)});
    }
    if (!cfg.headless and !ui.supported) {
        if (!ctx.quiet) ctx.warn("no window support on {s} yet; running headless (add --headless to silence this)", .{@tagName(builtin.os.tag)});
        cfg.headless = true;
    }
    if (cfg.headless and cfg.frame_limit == null and !visar.core.pool.threads_supported) {
        return ctx.usage("--headless needs --frames N on this platform", .{});
    }
    return runLive(ctx, cfg, record_out);
}

const Config = struct {
    source: Source = .camera,
    camera_index: ?usize = null,
    width: u32 = 1280,
    height: u32 = 720,
    fps: ?f64 = null,
    frame_limit: ?usize = null,
    detect_width: u32 = 640,
    headless: bool = false,
    threads_given: bool = false,
    mode: Mode = .both,
    locks: [max_locks][4]f32 = undefined,
    lock_count: usize = 0,
};

fn parseBox(ctx: *Ctx, v: []const u8) ![4]f32 {
    var out: [4]f32 = undefined;
    var it = std.mem.splitScalar(u8, v, ',');
    for (&out, 0..) |*o, i| {
        const part = it.next() orelse "";
        const n = std.fmt.parseFloat(f32, std.mem.trim(u8, part, " ")) catch {
            ctx.hint("write the box as X,Y,W,H in frame pixels, e.g. --lock 600,380,90,60", .{});
            return ctx.usage("'{s}' is not a box", .{v});
        };
        if (!std.math.isFinite(n) or n < 0 or (i >= 2 and n < 4)) {
            ctx.hint("X and Y are at least 0, W and H at least 4", .{});
            return ctx.usage("'{s}' is not a usable box", .{v});
        }
        o.* = n;
    }
    if (it.next() != null) {
        ctx.hint("write the box as X,Y,W,H in frame pixels, e.g. --lock 600,380,90,60", .{});
        return ctx.usage("'{s}' is not a box", .{v});
    }
    return out;
}

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

const Live = struct {
    gpa: std.mem.Allocator,
    io: Io,
    exec: visar.Exec,
    ring: *FrameRing,
    windowed: bool,
    frame_limit: ?usize,
    /// Integer downscale from the frame to the detector (1 = full size).
    factor: u32,

    stop: std.atomic.Value(bool) = .init(false),
    /// The worker reached the frame limit.
    done: std.atomic.Value(bool) = .init(false),
    worker_error: ?anyerror = null,

    captured: std.atomic.Value(usize) = .init(0),
    processed: std.atomic.Value(usize) = .init(0),
    hud: std.atomic.Value(bool) = .init(true),
    reset: std.atomic.Value(bool) = .init(false),
    /// Rates measured by the main thread, for the HUD (f32 bits).
    input_fps: std.atomic.Value(u32) = .init(0),
    display_fps: std.atomic.Value(u32) = .init(0),
    /// Confirmed tracks in the last processed frame.
    tracks_now: std.atomic.Value(u32) = .init(0),
    /// Frames shown per second over the whole run (main thread).
    display_fps_mean: f64 = 0,
    mode: std.atomic.Value(u8) = .init(0),
    /// Boxes to lock onto (frame pixels), from the window or --lock.
    requests: [max_locks][4]f32 = undefined,
    request_count: usize = 0,
    request_mutex: Io.Mutex = .init,
    clear_locks: std.atomic.Value(bool) = .init(false),
    /// The rectangle being dragged in the window (f32 bits; 0 width = none).
    drag: [4]std.atomic.Value(u32) = @splat(.init(0)),
    locks: std.ArrayList(Lock) = .empty,
    lock_reports: std.ArrayList(LockReport) = .empty,
    next_lock_id: u32 = 1,
    /// Ground truth per ring buffer (synthetic source).
    truth: [FrameRing.capacity][16]track.Scene.Truth = undefined,
    truth_len: [FrameRing.capacity]usize = @splat(0),

    /// Worker-owned.
    det: track.Detector,
    small: Image(u8),
    times: std.ArrayList(f64) = .empty,
    stage_sum: StageSums = .{},
    ema_ms: f64 = 0,
    first_ns: i96 = 0,
    last_ns: i96 = 0,
    recorder: ?Recorder = null,
    /// Synthetic source (the producer thread owns it).
    scene: ?*track.Scene = null,
    synthetic_fps: ?f64 = null,
    hud_label: []const u8 = "",
    hud_cache: Hud,

    fn now(self: *const Live) i96 {
        return Io.Clock.awake.now(self.io).nanoseconds;
    }

    fn setF32(v: *std.atomic.Value(u32), x: f64) void {
        v.store(@bitCast(@as(f32, @floatCast(x))), .monotonic);
    }

    fn getF32(v: *const std.atomic.Value(u32)) f64 {
        return @as(f32, @bitCast(v.load(.monotonic)));
    }

    fn shouldStop(self: *const Live) bool {
        return self.stop.load(.acquire) or interrupted.load(.acquire);
    }
};

const StageSums = struct {
    convert: f64 = 0,
    blur: f64 = 0,
    background: f64 = 0,
    morphology: f64 = 0,
    components: f64 = 0,
    tracker: f64 = 0,
    locks: f64 = 0,
    draw: f64 = 0,
};

const Lock = struct {
    mosse: visar.video.Mosse,
    report: LockReport,
    /// Detector pixels.
    box: visar.video.Box,
    psr: f32 = 0,
    ok: bool = true,
    /// Consecutive frames without a confident match.
    missed: u32 = 0,
    gt_gid: ?u32 = null,
    gt_tried: bool = false,
    psr_sum: f64 = 0,
    iou_sum: f64 = 0,
    err_sum: f64 = 0,

    fn lost(l: Lock) bool {
        return l.missed >= lost_after;
    }
};

const LockReport = struct {
    id: u32,
    /// The box it started from, frame pixels: x, y, w, h.
    start: [4]f32,
    frames: u32 = 0,
    /// Frames without a confident match (PSR below the threshold).
    weak_frames: u32 = 0,
    /// Frames counted as lost (after 10 weak frames in a row).
    lost_frames: u32 = 0,
    mean_psr: f64 = 0,
    /// Against the synthetic scene's ground truth (null for a camera, or
    /// when the start box covers no object).
    ground_truth: ?struct {
        frames: u32 = 0,
        /// Fraction of frames with IoU >= 0.5.
        success: f64 = 0,
        mean_iou: f64 = 0,
        mean_center_error_px: f64 = 0,
    } = null,
};

/// Set by SIGINT (Ctrl-C), so the run ends with its summary.
var interrupted: std.atomic.Value(bool) = .init(false);

fn onSigint(_: std.posix.SIG) callconv(.c) void {
    interrupted.store(true, .release);
}

fn catchSigint() void {
    switch (builtin.os.tag) {
        .linux, .macos, .freebsd, .netbsd, .openbsd, .dragonfly, .illumos => {},
        else => return,
    }
    const act: std.posix.Sigaction = .{
        .handler = .{ .handler = onSigint },
        .mask = std.posix.sigemptyset(),
        .flags = 0,
    };
    std.posix.sigaction(.INT, &act, null);
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// The camera sink: runs on the capture thread for every frame.
fn onCameraFrame(c: *anyopaque, img: Image(u8), pts: f64) void {
    const live: *Live = @ptrCast(@alignCast(c));
    const ring = live.ring;
    const i = ring.acquire() orelse return;
    ring.store(i, img);
    const seq = live.captured.fetchAdd(1, .monotonic);
    ring.meta[i] = .{ .seq = seq, .t_ns = live.now(), .pts = pts };
    ring.publishCaptured(live.io, i);
}

/// Renders the synthetic scene into the ring, paced to `synthetic_fps`
/// (or as fast as it goes without one).
fn syntheticMain(live: *Live) void {
    const scene = live.scene.?;
    var rgb = Image(u8).init(live.gpa, scene.width, scene.height, 3) catch |e| {
        live.worker_error = e;
        live.stop.store(true, .release);
        return;
    };
    defer rgb.deinit(live.gpa);
    const period_ns: i96 = if (live.synthetic_fps) |f| @intFromFloat(1e9 / f) else 0;
    var next = live.now();
    while (!live.shouldStop()) {
        if (period_ns == 0 and live.ring.captured.load(.acquire) != videoio.frame.none) {
            // Unpaced: render the next frame only once the worker took the
            // last one, so rendering frames nobody processes does not take
            // CPU time from the pipeline being measured.
            live.io.sleep(.fromMicroseconds(50), .awake) catch {};
            continue;
        }
        if (!produceSynthetic(live, rgb)) {
            // Every buffer is in use: the consumers are behind.
            live.io.sleep(.fromMicroseconds(200), .awake) catch {};
            continue;
        }
        if (period_ns > 0) {
            next += period_ns;
            const wait = next - live.now();
            if (wait > 0) {
                live.io.sleep(.fromNanoseconds(wait), .awake) catch {};
            } else if (wait < -100 * std.time.ns_per_ms) next = live.now(); // fell far behind: do not burst
        }
    }
}

fn produceSynthetic(live: *Live, rgb: Image(u8)) bool {
    const ring = live.ring;
    const i = ring.acquire() orelse return false;
    live.scene.?.render(rgb, .fast);
    live.truth_len[i] = live.scene.?.truth(&live.truth[i]).len;
    visar.imgproc.cvtColor(.serial, rgb, ring.image(i), .rgb2bgra) catch unreachable;
    const seq = live.captured.fetchAdd(1, .monotonic);
    ring.meta[i] = .{ .seq = seq, .t_ns = live.now(), .pts = 0 };
    ring.publishCaptured(if (visar.core.pool.threads_supported) live.io else null, i);
    return true;
}

// ---------------------------------------------------------------------------
// Processing
// ---------------------------------------------------------------------------

fn workerMain(live: *Live) void {
    while (!live.shouldStop()) {
        const i = live.ring.waitCaptured(live.io, 20 * std.time.ns_per_ms) orelse continue;
        processOne(live, i) catch |e| {
            live.ring.release(i);
            live.worker_error = e;
            live.stop.store(true, .release);
            return;
        };
        if (live.done.load(.acquire)) return;
    }
}

fn processOne(live: *Live, i: u8) !void {
    const ring = live.ring;
    const frame = ring.image(i);
    if (live.reset.swap(false, .acq_rel)) live.det.reset();

    var clock: track.Clock = .{ .io = live.io };
    _ = clock.lap();
    try videoio.frame.grayDownsample(live.exec, frame, live.small, live.factor);
    const t_convert = clock.lap();
    const mode: Mode = @fromBackingInt(@intCast(live.mode.load(.monotonic)));
    var st: track.StageTimes = .{};
    if (mode != .locks) try live.det.detect(live.exec, live.small, .{ .io = live.io, .times = &st });
    _ = clock.lap();
    try updateLocks(live, i);
    const t_locks = clock.lap();
    const k: f32 = @floatFromInt(live.factor);
    const thick: i64 = @max(2, frame.width / 400);
    const ls: u32 = @max(2, frame.width / 640);
    if (mode != .locks) drawTracks32(frame, live.det.tracker, k, thick, ls);
    if (mode != .motion) drawLocks(live, frame, k, thick, ls);
    if (live.windowed) drawDrag(live, frame, thick);
    const ms_so_far = t_convert + st.blur + st.background + st.morphology + st.components + st.tracker + t_locks;
    if (live.hud.load(.monotonic)) drawHud(live, frame, ms_so_far);
    const t_draw = clock.lap();

    const ms = ms_so_far + t_draw;
    const s = &live.stage_sum;
    s.convert += t_convert;
    s.blur += st.blur;
    s.background += st.background;
    s.morphology += st.morphology;
    s.components += st.components;
    s.tracker += st.tracker;
    s.locks += t_locks;
    s.draw += t_draw;
    try live.times.append(live.gpa, ms);
    live.ema_ms = if (live.ema_ms == 0) ms else 0.9 * live.ema_ms + 0.1 * ms;
    const t = live.now();
    if (live.first_ns == 0) live.first_ns = t;
    live.last_ns = t;

    live.tracks_now.store(@intCast(track.countConfirmed(live.det.tracker)), .monotonic);
    if (live.recorder) |*r| r.add(live, frame);

    if (live.windowed) ring.publishProcessed(i) else ring.release(i);
    const n = live.processed.fetchAdd(1, .monotonic) + 1;
    if (live.frame_limit) |lim| if (n >= lim) live.done.store(true, .release);
}

/// Applies clear and lock requests, then moves every lock to the current
/// frame (`live.small`, at detector resolution) and scores it against the
/// ground truth of ring buffer `slot` when there is one.
fn updateLocks(live: *Live, slot: u8) !void {
    const gpa = live.gpa;
    const k: f32 = @floatFromInt(live.factor);
    if (live.clear_locks.swap(false, .acq_rel)) {
        for (live.locks.items) |*l| try retireLock(live, l);
        live.locks.clearRetainingCapacity();
    }
    live.request_mutex.lockUncancelable(live.io);
    const reqs = live.requests;
    const nreq = live.request_count;
    live.request_count = 0;
    live.request_mutex.unlock(live.io);
    for (reqs[0..nreq]) |r| {
        if (live.locks.items.len == max_locks) {
            // The oldest lock makes room.
            try retireLock(live, &live.locks.items[0]);
            _ = live.locks.orderedRemove(0);
        }
        const dw: f32 = @floatFromInt(live.small.width);
        const dh: f32 = @floatFromInt(live.small.height);
        const x = std.math.clamp(r[0] / k, 0, dw - 4);
        const y = std.math.clamp(r[1] / k, 0, dh - 4);
        const box: visar.video.Box = .{ .x = x, .y = y, .w = std.math.clamp(r[2] / k, 4, dw - x), .h = std.math.clamp(r[3] / k, 4, dh - y) };
        const m = try visar.video.Mosse.init(gpa, live.small, box, .{});
        try live.locks.append(gpa, .{ .mosse = m, .box = box, .report = .{ .id = live.next_lock_id, .start = r } });
        live.next_lock_id += 1;
    }
    const truth = live.truth[slot][0..live.truth_len[slot]];
    // The filters are independent: one per pool thread.
    const results = try gpa.alloc(visar.video.mosse.Result, live.locks.items.len);
    defer gpa.free(results);
    const Job = struct { locks: []Lock, results: []visar.video.mosse.Result, gray: Image(u8) };
    live.exec.parallelFor(live.locks.items.len, 1, Job{ .locks = live.locks.items, .results = results, .gray = live.small }, struct {
        fn run(j: Job, a: usize, b: usize) void {
            for (j.locks[a..b], j.results[a..b]) |*l, *r| {
                r.* = l.mosse.update(j.gray) catch .{ .box = l.box, .psr = 0, .ok = false };
            }
        }
    }.run);
    for (live.locks.items, results) |*l, res| {
        l.box = res.box;
        l.psr = res.psr;
        l.ok = res.ok;
        l.missed = if (res.ok) 0 else l.missed + 1;
        l.report.frames += 1;
        if (!res.ok) l.report.weak_frames += 1;
        if (l.lost()) l.report.lost_frames += 1;
        l.psr_sum += res.psr;
        if (live.scene == null) continue;
        const fb: visar.video.Box = .{ .x = l.box.x * k, .y = l.box.y * k, .w = l.box.w * k, .h = l.box.h * k };
        if (!l.gt_tried) {
            // The object the start box covers best is the one to follow.
            l.gt_tried = true;
            const sb: visar.video.Box = .{ .x = l.report.start[0], .y = l.report.start[1], .w = l.report.start[2], .h = l.report.start[3] };
            var best: f32 = 0.3;
            for (truth) |t| {
                const v = iou(sb, t.box);
                if (v > best) {
                    best = v;
                    l.gt_gid = t.gid;
                }
            }
            if (l.gt_gid != null) l.report.ground_truth = .{};
        }
        const gid = l.gt_gid orelse continue;
        for (truth) |t| {
            if (t.gid != gid) continue;
            const g = &l.report.ground_truth.?;
            const v = iou(fb, t.box);
            g.frames += 1;
            if (v >= 0.5) g.success += 1;
            l.iou_sum += v;
            l.err_sum += std.math.hypot(fb.cx() - t.box.cx(), fb.cy() - t.box.cy());
        }
    }
}

/// Finishes a lock's report (averages) and frees its filter.
fn retireLock(live: *Live, l: *Lock) !void {
    var r = l.report;
    if (r.frames > 0) r.mean_psr = l.psr_sum / @as(f64, @floatFromInt(r.frames));
    if (r.ground_truth) |*g| if (g.frames > 0) {
        const n: f64 = @floatFromInt(g.frames);
        g.success /= n;
        g.mean_iou = l.iou_sum / n;
        g.mean_center_error_px = l.err_sum / n;
    };
    try live.lock_reports.append(live.gpa, r);
    l.mosse.deinit();
}

fn iou(a: visar.video.Box, b: visar.video.Box) f32 {
    const x0 = @max(a.x, b.x);
    const y0 = @max(a.y, b.y);
    const x1 = @min(a.x + a.w, b.x + b.w);
    const y1 = @min(a.y + a.h, b.y + b.h);
    const inter = @max(0, x1 - x0) * @max(0, y1 - y0);
    const uni = a.w * a.h + b.w * b.h - inter;
    return if (uni > 0) inter / uni else 0;
}

/// Locks are drawn thick and white-edged, labelled "LOCK n" with their
/// PSR (confidence), or "LOST" in red.
fn drawLocks(live: *Live, frame: Image(u8), k: f32, thick: i64, ls: u32) void {
    const lsi: i64 = ls;
    const t = thick * 2;
    for (live.locks.items) |l| {
        const lost = l.lost();
        const col = if (lost) bgra(255, 60, 60) else if (l.ok) bgra(255, 255, 255) else bgra(255, 200, 0);
        const bx: i64 = @as(i64, @intFromFloat(l.box.x * k)) - t;
        const by: i64 = @as(i64, @intFromFloat(l.box.y * k)) - t;
        const bw: i64 = @as(i64, @intFromFloat(l.box.w * k)) + 2 * t;
        const bh: i64 = @as(i64, @intFromFloat(l.box.h * k)) + 2 * t;
        outline32(frame, bx, by, bw, bh, t, col);
        // Corner ticks make a lock stand out from motion boxes.
        const c = @max(4 * t, @divTrunc(@min(bw, bh), 4));
        fill32(frame, bx - t, by - t, bx + c, by, col);
        fill32(frame, bx - t, by - t, bx, by + c, col);
        fill32(frame, bx + bw - c, by + bh, bx + bw + t, by + bh + t, col);
        fill32(frame, bx + bw, by + bh - c, bx + bw + t, by + bh + t, col);
        var lbuf: [40]u8 = undefined;
        const label = if (lost)
            std.fmt.bufPrint(&lbuf, "LOST {d}", .{l.report.id}) catch "LOST"
        else
            std.fmt.bufPrint(&lbuf, "LOCK {d} PSR {d:.1}", .{ l.report.id, l.psr }) catch "LOCK";
        const lw: i64 = @as(i64, @intCast(label.len)) * 6 * lsi + 3 * lsi;
        fill32(frame, bx, by + bh, bx + lw, by + bh + 10 * lsi, col);
        _ = draw.text(frame, bx + 2 * lsi, by + bh + lsi, label, ls, &.{ 0, 0, 0, 255 });
    }
}

/// The rectangle being dragged in the window.
fn drawDrag(live: *Live, frame: Image(u8), thick: i64) void {
    var r: [4]f32 = undefined;
    for (&r, &live.drag) |*v, *a| v.* = @bitCast(a.load(.monotonic));
    if (r[2] <= 0 or r[3] <= 0) return;
    outline32(frame, @intFromFloat(r[0]), @intFromFloat(r[1]), @intFromFloat(r[2]), @intFromFloat(r[3]), @max(1, @divTrunc(thick, 2)), bgra(255, 255, 255));
}

fn outline32(img: Image(u8), x: i64, y: i64, w: i64, h: i64, t: i64, col: u32) void {
    fill32(img, x, y, x + w, y + t, col);
    fill32(img, x, y + h - t, x + w, y + h, col);
    fill32(img, x, y, x + t, y + h, col);
    fill32(img, x + w - t, y, x + w, y + h, col);
}

/// The HUD is rendered into its own small image four times a second (so
/// its numbers are readable, and text is not drawn every frame) and copied
/// onto each frame.
const Hud = struct {
    img: Image(u8),
    w: u32 = 0,
    h: u32 = 0,
    next_ns: i96 = 0,

    const refresh_ns = 250 * std.time.ns_per_ms;

    fn init(gpa: std.mem.Allocator, frame_w: u32, frame_h: u32) !Hud {
        const s = @max(1, frame_w / 640);
        return .{ .img = try Image(u8).init(gpa, frame_w, @min(frame_h, 40 * s), 4) };
    }

    fn deinit(self: *Hud, gpa: std.mem.Allocator) void {
        self.img.deinit(gpa);
    }
};

fn drawHud(live: *Live, frame: Image(u8), ms_now: f64) void {
    const hud = &live.hud_cache;
    const t = live.now();
    if (t >= hud.next_ns) {
        hud.next_ns = t + Hud.refresh_ns;
        renderHud(live, hud, frame.width, frame.height, ms_now);
    }
    // Copy the HUD's rows onto the frame.
    const row_bytes = @as(usize, hud.w) * 4;
    for (0..@min(hud.h, frame.height)) |y| @memcpy(frame.row(y)[0..row_bytes], hud.img.row(y)[0..row_bytes]);
}

fn renderHud(live: *Live, hud: *Hud, fw: u32, fh: u32, ms_now: f64) void {
    const img = hud.img;
    const s: u32 = @max(1, fw / 640);
    const si: i64 = s;
    const line_h = 9 * si;
    const ms = if (live.ema_ms > 0) live.ema_ms else ms_now;
    var b: [3][128]u8 = undefined;
    const cam_fps = Live.getF32(&live.input_fps);
    const disp_fps = Live.getF32(&live.display_fps);
    const l1 = std.fmt.bufPrint(&b[0], "{s}  {d}x{d}  {d:.0} FPS", .{ live.hud_label, fw, fh, cam_fps }) catch "";
    const l2 = std.fmt.bufPrint(&b[1], "VISAR {d:.2} MS/FRAME = {d:.0} FPS  TRACKS {d}", .{ ms, if (ms > 0) 1000 / ms else 0, track.countConfirmed(live.det.tracker) }) catch "";
    const mode: Mode = @fromBackingInt(@intCast(live.mode.load(.monotonic)));
    const mode_name = switch (mode) {
        .both => "MOTION+LOCKS",
        .motion => "MOTION",
        .locks => "LOCKS",
    };
    const l3 = if (live.windowed)
        std.fmt.bufPrint(&b[2], "DISPLAY {d:.0} FPS  {s}  DRAG LOCK  C CLEAR  TAB MODE  Q QUIT", .{ disp_fps, mode_name }) catch ""
    else
        std.fmt.bufPrint(&b[2], "HEADLESS  {s}  LOCKS {d}", .{ mode_name, live.locks.items.len }) catch "";
    const longest: i64 = @intCast(@max(l1.len, l2.len, l3.len));
    hud.w = @intCast(@min(@as(i64, img.width), 12 * si + longest * 6 * si));
    hud.h = @intCast(@min(@as(i64, img.height), 6 * si + 3 * (line_h + 3 * si)));
    fill32(img, 0, 0, hud.w, hud.h, bgra(10, 12, 20));
    const x = 6 * si;
    var y = 6 * si;
    _ = draw.text(img, x, y, l1, s, &.{ 255, 255, 255, 255 });
    y += line_h + 3 * si;
    _ = draw.text(img, x, y, l2, s, &.{ 120, 230, 255, 255 });
    y += line_h + 3 * si;
    _ = draw.text(img, x, y, l3, s, &.{ 170, 170, 170, 255 });
}

/// A BGRA pixel as one little-endian u32 (alpha 255).
fn bgra(r: u8, g: u8, b: u8) u32 {
    return @as(u32, b) | (@as(u32, g) << 8) | (@as(u32, r) << 16) | 0xff00_0000;
}

/// Fills [x0, x1) x [y0, y1), clipped, in a 4-channel image.
fn fill32(img: Image(u8), x0: i64, y0: i64, x1: i64, y1: i64, color: u32) void {
    const ax: usize = @intCast(std.math.clamp(x0, 0, img.width));
    const bx: usize = @intCast(std.math.clamp(x1, 0, img.width));
    const ay: usize = @intCast(std.math.clamp(y0, 0, img.height));
    const by: usize = @intCast(std.math.clamp(y1, 0, img.height));
    if (ax >= bx or ay >= by) return;
    for (ay..by) |y| {
        const r = img.row(y)[ax * 4 .. bx * 4];
        const px: []align(1) u32 = std.mem.bytesAsSlice(u32, r);
        @memset(px, std.mem.nativeToLittle(u32, color));
    }
}

/// A line with a square brush of `t` pixels.
fn line32(img: Image(u8), x0_in: i64, y0_in: i64, x1: i64, y1: i64, color: u32, t: i64) void {
    var x0 = x0_in;
    var y0 = y0_in;
    const dx: i64 = @intCast(@abs(x1 - x0));
    const dy: i64 = -@as(i64, @intCast(@abs(y1 - y0)));
    const sx: i64 = if (x0 < x1) 1 else -1;
    const sy: i64 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const r = @divTrunc(t, 2);
    while (true) {
        fill32(img, x0 - r, y0 - r, x0 - r + t, y0 - r + t, color);
        if (x0 == x1 and y0 == y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

/// Boxes, trails and "#id" labels of the confirmed tracks, on a BGRA frame
/// `k` times the tracker's resolution (track_lib.drawTracks, with 32-bit
/// fills).
fn drawTracks32(frame: Image(u8), tracker: visar.video.Tracker, k: f32, thick: i64, ls: u32) void {
    const lsi: i64 = ls;
    for (tracker.tracks.items) |t| {
        if (!t.confirmed()) continue;
        const rgb = track.idColor(t.id);
        const col = bgra(rgb[0], rgb[1], rgb[2]);
        var i: usize = 1;
        while (i < t.trail_count) : (i += 1) {
            const a = t.trailPoint(i - 1);
            const b = t.trailPoint(i);
            line32(frame, @intFromFloat(a[0] * k), @intFromFloat(a[1] * k), @intFromFloat(b[0] * k), @intFromFloat(b[1] * k), col, thick);
        }
        const bx: i64 = @as(i64, @intFromFloat(t.box.x * k)) - thick;
        const by: i64 = @as(i64, @intFromFloat(t.box.y * k)) - thick;
        const bw: i64 = @as(i64, @intFromFloat(t.box.w * k)) + 2 * thick;
        const bh: i64 = @as(i64, @intFromFloat(t.box.h * k)) + 2 * thick;
        fill32(frame, bx, by, bx + bw, by + thick, col);
        fill32(frame, bx, by + bh - thick, bx + bw, by + bh, col);
        fill32(frame, bx, by, bx + thick, by + bh, col);
        fill32(frame, bx + bw - thick, by, bx + bw, by + bh, col);
        var lbuf: [16]u8 = undefined;
        const label = std.fmt.bufPrint(&lbuf, "#{d}", .{t.id}) catch "#";
        const lw: i64 = @as(i64, @intCast(label.len)) * 6 * lsi + 3 * lsi;
        fill32(frame, bx, by - 10 * lsi, bx + lw, by, col);
        _ = draw.text(frame, bx + 2 * lsi, by - 9 * lsi, label, ls, &.{ 0, 0, 0, 255 });
    }
}

const Recorder = struct {
    anim: visar.codecs.gif.Animation,
    small: Image(u8),
    delay_cs: u16,
    frames: u32 = 0,
    full: bool = false,
    failed: ?anyerror = null,

    fn init(gpa: std.mem.Allocator, w: u32, h: u32, fps: f64) !Recorder {
        const gw = @min(w, 480);
        const gh = @max(1, gw * h / w);
        var small = try Image(u8).init(gpa, gw, gh, 4);
        errdefer small.deinit(gpa);
        return .{
            .anim = try visar.codecs.gif.Animation.init(gpa, gw, gh, .color, .bgr),
            .small = small,
            .delay_cs = @intFromFloat(std.math.clamp(@round(100 / @max(fps, 1)), 2, 100)),
        };
    }

    fn deinit(self: *Recorder, gpa: std.mem.Allocator) void {
        self.anim.deinit();
        self.small.deinit(gpa);
    }

    fn add(self: *Recorder, live: *Live, frame: Image(u8)) void {
        if (self.full or self.failed != null) return;
        if (self.frames >= max_record_frames) {
            self.full = true;
            return;
        }
        visar.imgproc.resize(live.exec, live.gpa, frame, self.small, .area) catch |e| {
            self.failed = e;
            return;
        };
        self.anim.addFrame(self.small, self.delay_cs) catch |e| {
            self.failed = e;
            return;
        };
        self.frames += 1;
    }
};

// ---------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------

fn runLive(ctx: *Ctx, cfg: Config, record_out: ?imageio.Output) !void {
    const gpa = ctx.gpa;
    const io = ctx.io;
    // The detector's images are small (640x360 by default) and capture,
    // display and the scene need cores too: a few pool threads do best.
    if (!cfg.threads_given and visar.core.pool.threads_supported) {
        const cpus = std.Thread.getCpuCount() catch 1;
        ctx.threads = std.math.clamp(cpus -| 2, 1, 4);
    }
    const exec = ctx.exec();

    // ---- the source
    var cam: ?*camera.Camera = null;
    defer if (cam) |c| c.close();
    var scene: ?track.Scene = null;
    defer if (scene) |*s| s.deinit();
    var W: u32 = cfg.width;
    var H: u32 = cfg.height;
    var input_fps: f64 = 0;
    var device_name: []const u8 = "synthetic scene";
    var pixel_format: []const u8 = "RGB";
    var format_max_fps: f64 = 0;

    // The sink needs the Live, which needs the ring, which needs the size:
    // the camera is opened first and started only once everything exists.
    var live_storage: Live = undefined;
    const live = &live_storage;

    switch (cfg.source) {
        .camera => {
            if (!ctx.quiet and !ctx.json) {
                ctx.se.print(ctx.err, .dim, "  opening the camera\n", .{}) catch {};
                ctx.err.flush() catch {};
            }
            cam = camera.Camera.open(gpa, io, .{ .index = cfg.camera_index, .width = cfg.width, .height = cfg.height, .fps = cfg.fps }, .{ .ctx = live, .func = onCameraFrame }) catch |e| return cameraError(ctx, e, cfg);
            const g = &cam.?.granted;
            W = g.width;
            H = g.height;
            input_fps = g.fps;
            device_name = g.name();
            pixel_format = &g.pixel_format;
            format_max_fps = g.format_max_fps;
            if (W == 0 or H == 0) return ctx.fail("the camera reported no frame size", .{});
        },
        .synthetic => {
            scene = track.Scene.init(gpa, exec, .{ .width = W, .height = H }) catch |e| return ctx.fail("cannot build the synthetic scene: {s}", .{cx.describe(e)});
            // Start with objects on screen (so --lock has something to follow).
            for (0..synthetic_preroll) |_| scene.?.advance();
            input_fps = cfg.fps orelse if (cfg.headless) 0 else 60;
        },
    }

    var ring = FrameRing.init(gpa, W, H, 4) catch |e| return ctx.fail("cannot allocate frame buffers: {s}", .{cx.describe(e)});
    var ring_owned = true;
    defer if (ring_owned) ring.deinit();

    // ---- the detector, at an integer fraction of the frame size
    const factor: u32 = if (cfg.detect_width == 0 or cfg.detect_width >= W) 1 else @max(1, @min(16, (W + cfg.detect_width / 2) / cfg.detect_width));
    const dw = W / factor;
    const dh = H / factor;
    const sx: f32 = @as(f32, @floatFromInt(dw)) / 960.0;
    // Background time constant of about half a second at the camera's rate.
    const rate = if (input_fps > 0) input_fps else 60;
    const shift: u4 = @intFromFloat(std.math.clamp(@round(std.math.log2(rate / 2)), 3, 8));
    var det = track.Detector.init(gpa, dw, dh, .{ .sx = sx, .dilate = @max(3, @as(u32, @intFromFloat(@round(9 * sx)))), .bg_shift = if (cfg.source == .synthetic) 4 else shift }) catch |e| return ctx.fail("cannot allocate the detector: {s}", .{cx.describe(e)});
    defer det.deinit();
    var small = Image(u8).init(gpa, dw, dh, 1) catch |e| return ctx.fail("cannot allocate the detector: {s}", .{cx.describe(e)});
    defer small.deinit(gpa);

    var label_buf: [48]u8 = undefined;
    live.* = .{
        .gpa = gpa,
        .io = io,
        .exec = exec,
        .ring = &ring,
        .windowed = !cfg.headless,
        .frame_limit = cfg.frame_limit,
        .factor = factor,
        .det = det,
        .small = small,
        .scene = if (scene) |*s| s else null,
        .synthetic_fps = if (cfg.source == .synthetic) (if (input_fps > 0) input_fps else null) else null,
        .hud_label = upperLabel(&label_buf, if (cfg.source == .synthetic) "SYNTHETIC" else device_name),
        .hud_cache = Hud.init(gpa, W, H) catch |e| return ctx.fail("out of memory: {s}", .{cx.describe(e)}),
    };
    defer live.hud_cache.deinit(gpa);
    live.mode.store(@backingInt(cfg.mode), .monotonic);
    for (cfg.locks[0..cfg.lock_count]) |r| {
        if (r[0] + r[2] > @as(f32, @floatFromInt(W)) or r[1] + r[3] > @as(f32, @floatFromInt(H))) {
            ctx.hint("the frame is {d}x{d}", .{ W, H });
            return ctx.usage("--lock {d},{d},{d},{d} is outside the frame", .{ r[0], r[1], r[2], r[3] });
        }
        live.requests[live.request_count] = r;
        live.request_count += 1;
    }
    defer {
        for (live.locks.items) |*l| l.mosse.deinit();
        live.locks.deinit(gpa);
        live.lock_reports.deinit(gpa);
    }
    // `det` now lives in `live`; the deferred deinit must see the same state.
    defer det = live.det;
    defer live.times.deinit(gpa);
    if (record_out != null) live.recorder = Recorder.init(gpa, W, H, if (input_fps > 0) input_fps else 30) catch |e| return ctx.fail("cannot start the recording: {s}", .{cx.describe(e)});
    defer if (live.recorder) |*r| r.deinit(gpa);

    if (!ctx.quiet and !ctx.json) {
        const s = ctx.se;
        const w = ctx.err;
        s.print(w, .dim, "  {s} · {d}x{d}", .{ device_name, W, H }) catch {};
        if (cfg.source == .camera) {
            s.print(w, .dim, " {s} at {d:.0} fps", .{ pixel_format, input_fps }) catch {};
            if (cfg.fps) |want| if (@abs(want - input_fps) > 0.5) s.print(w, .dim, " ({d:.0} asked; this format tops out at {d:.0})", .{ want, format_max_fps }) catch {};
        } else if (input_fps > 0) s.print(w, .dim, " at {d:.0} fps", .{input_fps}) catch {} else s.print(w, .dim, " as fast as it renders", .{}) catch {};
        s.print(w, .dim, " · detecting at {d}x{d} on {d} thread{s}\n", .{ dw, dh, exec.threads(), if (exec.threads() == 1) "" else "s" }) catch {};
        if (live.windowed) s.print(w, .dim, "  q or Esc quits, space resets the background, h toggles the HUD\n", .{}) catch {};
        w.flush() catch {};
    }

    catchSigint();
    const t_start = live.now();

    // ---- start
    if (!visar.core.pool.threads_supported) {
        // No threads (wasm): capture and process in turn.
        var rgb = Image(u8).init(gpa, W, H, 3) catch |e| return ctx.fail("out of memory: {s}", .{cx.describe(e)});
        defer rgb.deinit(gpa);
        while (!live.done.load(.acquire)) {
            _ = produceSynthetic(live, rgb);
            const i = ring.takeCaptured() orelse continue;
            processOne(live, i) catch |e| return ctx.fail("tracking failed: {s}", .{cx.describe(e)});
        }
    } else {
        const worker = std.Thread.spawn(.{}, workerMain, .{live}) catch |e| return ctx.fail("cannot start the worker thread: {s}", .{cx.describe(e)});
        var producer: ?std.Thread = null;
        var stopped = false;
        const stopAll = struct {
            fn f(l: *Live, wk: std.Thread, pr: ?std.Thread, c: ?*camera.Camera) void {
                l.stop.store(true, .release);
                if (c) |cc| cc.stop();
                if (pr) |t| t.join();
                // The worker notices `stop` within its 20 ms wait.
                wk.join();
            }
        }.f;
        defer if (!stopped) stopAll(live, worker, producer, cam);

        switch (cfg.source) {
            .camera => cam.?.start() catch |e| return cameraError(ctx, e, cfg),
            .synthetic => producer = std.Thread.spawn(.{}, syntheticMain, .{live}) catch |e| return ctx.fail("cannot start the scene thread: {s}", .{cx.describe(e)}),
        }

        const outcome = if (live.windowed) try windowLoop(ctx, live, device_name, W, H) else headlessLoop(ctx, live);
        stopAll(live, worker, producer, cam);
        stopped = true;
        switch (outcome) {
            .no_frames => {
                ctx.hint("if macOS never asked, allow your terminal app under System Settings > Privacy & Security > Camera; also check that no other app is using the camera", .{});
                return ctx.fail("no frames from the camera after {d:.0} seconds", .{no_frames_timeout_s});
            },
            .ok => {},
        }
        if (outcome == .ok and live.windowed) {
            // The window is closed; wait for Core Animation to hand back the
            // last buffers before they are freed (or keep them if it does not).
            if (!waitIdle(live)) ring_owned = false;
        }
    }
    ring.drain();
    if (live.worker_error) |e| return ctx.fail("tracking failed: {s}", .{cx.describe(e)});
    // Locks still running are reported too.
    for (live.locks.items) |*l| try retireLock(live, l);
    live.locks.clearRetainingCapacity();

    const t_end = live.now();
    var st = summarize(live, t_start, t_end);
    st.source = cfg.source;
    st.device = device_name;
    st.width = W;
    st.height = H;
    st.detect_width = dw;
    st.detect_height = dh;
    st.input_fps_set = input_fps;
    st.fps_requested = cfg.fps;
    st.threads = exec.threads();
    st.pixel_format = pixel_format;
    st.mode = cfg.mode;
    st.locks = live.lock_reports.items;

    if (live.recorder) |*r| {
        if (r.failed) |e| ctx.warn("the recording failed: {s}", .{cx.describe(e)}) else if (r.frames > 0) {
            if (r.full) ctx.warn("the recording stopped at {d} frames", .{max_record_frames});
            const bytes = try r.anim.finish();
            defer gpa.free(bytes);
            try imageio.writeBytes(ctx, record_out.?, bytes);
            st.recorded = r.frames;
        }
    }
    try report(ctx, st, record_out);
}

fn upperLabel(buf: []u8, name: []const u8) []const u8 {
    const n = @min(buf.len, name.len);
    for (name[0..n], buf[0..n]) |c, *d| d.* = if (c < 128) std.ascii.toUpper(c) else '?';
    return buf[0..n];
}

const Outcome = enum { ok, no_frames };

/// Rates over the last second, for the HUD and the headless stats line.
const Meter = struct {
    t: i96,
    captured: usize = 0,
    processed: usize = 0,
    shown: usize = 0,

    const Rates = struct { input: f64, processed: f64, shown: f64 };

    fn tick(m: *Meter, live: *Live, shown: usize) ?Rates {
        const t = live.now();
        const dt = @as(f64, @floatFromInt(t - m.t)) / 1e9;
        if (dt < 1.0) return null;
        const c = live.captured.load(.monotonic);
        const p = live.processed.load(.monotonic);
        const r: Rates = .{
            .input = @as(f64, @floatFromInt(c - m.captured)) / dt,
            .processed = @as(f64, @floatFromInt(p - m.processed)) / dt,
            .shown = @as(f64, @floatFromInt(shown - m.shown)) / dt,
        };
        m.* = .{ .t = t, .captured = c, .processed = p, .shown = shown };
        Live.setF32(&live.input_fps, r.input);
        Live.setF32(&live.display_fps, r.shown);
        return r;
    }
};

const ReleaseToken = struct { ring: *FrameRing, i: u8 };

fn releaseToken(c: ?*anyopaque) callconv(.c) void {
    const t: *ReleaseToken = @ptrCast(@alignCast(c.?));
    t.ring.release(t.i);
}

fn windowLoop(ctx: *Ctx, live: *Live, name: []const u8, W: u32, H: u32) !Outcome {
    if (comptime !ui.supported) return .ok;
    var title_buf: [160]u8 = undefined;
    const title = std.fmt.bufPrintSentinel(&title_buf, "VISAR track · {s}", .{name}, 0) catch "VISAR track";
    // At most 1280 points wide (2560 pixels on a Retina display).
    const scale = @min(1.0, 1280.0 / @as(f64, @floatFromInt(W)));
    const win = ui.Window.open(ctx.gpa, title, @intFromFloat(@as(f64, @floatFromInt(W)) * scale), @intFromFloat(@as(f64, @floatFromInt(H)) * scale)) catch {
        ctx.hint("run with --headless, and please report this at {s}/issues", .{@import("commands.zig").repo_url});
        return ctx.fail("cannot open a window (missing: {s})", .{ui.window_macos.missing()});
    };
    defer win.close();

    var tokens: [FrameRing.capacity]ReleaseToken = undefined;
    for (&tokens, 0..) |*t, i| t.* = .{ .ring = live.ring, .i = @intCast(i) };
    var shown: usize = 0;
    var meter: Meter = .{ .t = live.now() };
    const t0 = meter.t;
    while (!live.shouldStop() and !live.done.load(.acquire)) {
        const in = win.poll(0.001);
        if (in.quit) break;
        if (in.reset) live.reset.store(true, .release);
        if (in.toggle_hud) live.hud.store(!live.hud.load(.monotonic), .monotonic);
        if (in.clear) live.clear_locks.store(true, .release);
        if (in.cycle_mode) live.mode.store(@backingInt(@as(Mode, @fromBackingInt(@intCast(live.mode.load(.monotonic)))).next()), .monotonic);
        if (in.selection) |r| {
            live.request_mutex.lockUncancelable(live.io);
            if (live.request_count < max_locks) {
                live.requests[live.request_count] = r;
                live.request_count += 1;
            }
            live.request_mutex.unlock(live.io);
            // A lock is useless in motion-only mode.
            if (live.mode.load(.monotonic) == @backingInt(Mode.motion)) live.mode.store(@backingInt(Mode.both), .monotonic);
        }
        const d = in.dragging orelse [4]f32{ 0, 0, 0, 0 };
        for (&live.drag, d) |*a, v| a.store(@bitCast(v), .monotonic);
        if (live.ring.takeProcessed()) |i| {
            win.present(live.ring.image(i), .{ .ctx = &tokens[i], .func = releaseToken });
            shown += 1;
        }
        _ = meter.tick(live, shown);
        if (live.captured.load(.monotonic) == 0 and secondsSince(live, t0) > no_frames_timeout_s and live.scene == null) return .no_frames;
    }
    if (live.done.load(.acquire)) {
        // Show the last frame.
        if (live.ring.takeProcessed()) |i| win.present(live.ring.image(i), .{ .ctx = &tokens[i], .func = releaseToken });
        _ = win.poll(0);
    }
    live.display_fps_mean = @as(f64, @floatFromInt(shown)) / @max(1e-9, secondsSince(live, t0));
    return .ok;
}

fn waitIdle(live: *Live) bool {
    for (0..200) |_| {
        var busy = false;
        for (&live.ring.busy) |*b| busy = busy or b.load(.acquire) != 0;
        if (!busy) return true;
        live.io.sleep(.fromMilliseconds(5), .awake) catch {};
    }
    return false;
}

fn secondsSince(live: *const Live, t: i96) f64 {
    return @as(f64, @floatFromInt(live.now() - t)) / 1e9;
}

fn headlessLoop(ctx: *Ctx, live: *Live) Outcome {
    var meter: Meter = .{ .t = live.now() };
    const t0 = meter.t;
    while (!live.shouldStop() and !live.done.load(.acquire)) {
        live.io.sleep(.fromMilliseconds(20), .awake) catch {};
        if (meter.tick(live, 0)) |r| {
            if (!ctx.quiet) {
                const n = live.processed.load(.monotonic);
                ctx.se.print(ctx.err, .dim, "  {d:>5.1} s  frame {d:>6}  input {d:>6.1} fps  processed {d:>6.1} fps  visar {d:.2} ms ({d:.0} fps)  tracks {d}\n", .{
                    secondsSince(live, t0), n, r.input, r.processed, live.ema_ms, if (live.ema_ms > 0) 1000 / live.ema_ms else 0, live.tracks_now.load(.monotonic),
                }) catch {};
                ctx.err.flush() catch {};
            }
        }
        if (live.captured.load(.monotonic) == 0 and secondsSince(live, t0) > no_frames_timeout_s and live.scene == null) return .no_frames;
    }
    return .ok;
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

const Summary = struct {
    source: Source = .camera,
    device: []const u8 = "",
    pixel_format: []const u8 = "",
    target: []const u8 = visar.core.target_description,
    threads: usize = 1,
    width: u32 = 0,
    height: u32 = 0,
    detect_width: u32 = 0,
    detect_height: u32 = 0,
    fps_requested: ?f64 = null,
    /// The rate the camera was set to (synthetic: the pacing; 0 = unpaced).
    input_fps_set: f64 = 0,
    frames: u64 = 0,
    seconds: f64 = 0,
    /// Frames arriving per second, measured.
    input_fps: f64 = 0,
    /// Frames processed per second, measured (end to end).
    output_fps: f64 = 0,
    display_fps: ?f64 = null,
    processing: struct { mean_ms: f64 = 0, p50_ms: f64 = 0, p95_ms: f64 = 0, fps: f64 = 0 } = .{},
    stages_ms: StageSums = .{},
    dropped: u64 = 0,
    tracks: u32 = 0,
    recorded: u32 = 0,
    mode: Mode = .both,
    locks: []const LockReport = &.{},
    pipeline: []const u8 = "gray and area downscale in one pass, Gaussian 5x5, background subtraction, open 3x3, dilate (ellipse), connected components, tracker, drawing",
};

fn summarize(live: *Live, t_start: i96, t_end: i96) Summary {
    var s: Summary = .{};
    const times = live.times.items;
    const n = times.len;
    s.frames = n;
    s.seconds = @as(f64, @floatFromInt(t_end - t_start)) / 1e9;
    const captured = live.captured.load(.monotonic);
    s.input_fps = @as(f64, @floatFromInt(captured)) / @max(1e-9, s.seconds);
    const span = @as(f64, @floatFromInt(live.last_ns - live.first_ns)) / 1e9;
    s.output_fps = if (n > 1 and span > 0) @as(f64, @floatFromInt(n - 1)) / span else 0;
    s.dropped = captured -| n;
    s.tracks = live.det.tracker.next_id - 1;
    if (live.windowed) s.display_fps = live.display_fps_mean;
    if (n == 0) return s;
    // Statistics after a short warm-up (the background is still settling).
    const skip = @min(n / 4, track.warmup);
    const t = times[skip..];
    var sum: f64 = 0;
    for (t) |v| sum += v;
    const mean = sum / @as(f64, @floatFromInt(t.len));
    std.mem.sort(f64, t, {}, std.sort.asc(f64));
    s.processing = .{ .mean_ms = mean, .p50_ms = t[t.len / 2], .p95_ms = t[@min(t.len - 1, t.len * 95 / 100)], .fps = if (mean > 0) 1000 / mean else 0 };
    const nf: f64 = @floatFromInt(n);
    const a = live.stage_sum;
    s.stages_ms = .{ .convert = a.convert / nf, .blur = a.blur / nf, .background = a.background / nf, .morphology = a.morphology / nf, .components = a.components / nf, .tracker = a.tracker / nf, .locks = a.locks / nf, .draw = a.draw / nf };
    return s;
}

fn report(ctx: *Ctx, s: Summary, record_out: ?imageio.Output) !void {
    if (ctx.json) {
        try std.json.Stringify.value(s, .{ .whitespace = .indent_2 }, ctx.out);
        try ctx.out.writeAll("\n");
        return;
    }
    if (ctx.quiet) return;
    const w = ctx.out;
    const st = ctx.so;
    var b: [10][320]u8 = undefined;
    try w.writeAll("  ");
    try st.paint(w, .ok, st.glyph("✓", "ok"));
    try w.print(" tracked {d} frames in {d:.1} s\n\n", .{ s.frames, s.seconds });
    const ms = s.processing;
    const sm = s.stages_ms;
    defer writeLocks(w, st, s.locks) catch {};
    try term.writeFields(w, st, 2, &.{
        .{ "source", std.fmt.bufPrint(&b[0], "{s}, {d}x{d}{s}{s}", .{ s.device, s.width, s.height, if (s.source == .camera) " " else "", if (s.source == .camera) s.pixel_format else "" }) catch "" },
        .{ "input", std.fmt.bufPrint(&b[1], "{d:.1} fps measured{s}", .{ s.input_fps, if (s.source == .synthetic and s.input_fps_set == 0) " (unpaced)" else "" }) catch "" },
        .{ "visar", std.fmt.bufPrint(&b[2], "{d:.2} ms per frame = {d:.0} fps (p50 {d:.2}, p95 {d:.2} ms)", .{ ms.mean_ms, ms.fps, ms.p50_ms, ms.p95_ms }) catch "" },
        .{ "stages", std.fmt.bufPrint(&b[3], "convert {d:.2} · blur {d:.2} · bg {d:.2} · morph {d:.2} · cc {d:.2} · track {d:.2} · locks {d:.2} · draw {d:.2} ms", .{ sm.convert, sm.blur, sm.background, sm.morphology, sm.components, sm.tracker, sm.locks, sm.draw }) catch "" },
        .{ "processed", std.fmt.bufPrint(&b[4], "{d:.1} fps end to end, {d} frames skipped (newest frame wins)", .{ s.output_fps, s.dropped }) catch "" },
        .{ "display", if (s.display_fps) |d| std.fmt.bufPrint(&b[5], "{d:.1} fps", .{d}) catch "" else "none (headless)" },
        .{ "detector", std.fmt.bufPrint(&b[6], "{d}x{d}, {d} thread{s}, {d} objects seen", .{ s.detect_width, s.detect_height, s.threads, if (s.threads == 1) "" else "s", s.tracks }) catch "" },
        .{ "mode", @tagName(s.mode) },
        .{ "recording", if (record_out) |o| std.fmt.bufPrint(&b[7], "{s} ({d} frames)", .{ o.name(), s.recorded }) catch "" else "none" },
    });
}

fn writeLocks(w: *Io.Writer, st: term.Style, locks: []const LockReport) !void {
    if (locks.len == 0) return;
    try w.writeAll("\n");
    for (locks) |l| {
        try w.writeAll("  ");
        try st.print(w, .bold, "lock {d}", .{l.id});
        try st.print(w, .dim, " from {d:.0},{d:.0} {d:.0}x{d:.0}", .{ l.start[0], l.start[1], l.start[2], l.start[3] });
        try w.print("  {d} frames, mean PSR {d:.1}, {d} weak, {d} lost", .{ l.frames, l.mean_psr, l.weak_frames, l.lost_frames });
        if (l.ground_truth) |g| {
            try w.print("; vs ground truth: {d:.0}% of {d} frames on target (IoU >= 0.5), mean IoU {d:.2}, centre error {d:.1} px", .{ 100 * g.success, g.frames, g.mean_iou, g.mean_center_error_px });
        }
        try w.writeAll("\n");
    }
}

// ---------------------------------------------------------------------------
// --list-cameras
// ---------------------------------------------------------------------------

fn listCameras(ctx: *Ctx) !void {
    if (!camera.supported) {
        if (ctx.json) {
            try std.json.Stringify.value(.{ .supported = false, .permission = "not_applicable", .cameras = &[_]struct {}{} }, .{ .whitespace = .indent_2 }, ctx.out);
            try ctx.out.writeAll("\n");
        } else try ctx.out.print("no camera support on {s} yet; 'visar track --source synthetic' runs anywhere\n", .{@tagName(builtin.os.tag)});
        return;
    }
    var lst = camera.list(ctx.gpa) catch |e| {
        if (e == error.RuntimeUnavailable) return ctx.fail("cannot load the camera framework: {s}", .{camera.detail()});
        return ctx.fail("cannot list the cameras: {s}", .{cx.describe(e)});
    };
    defer lst.deinit();
    const a = ctx.arena;

    if (ctx.json) {
        const JFormat = struct { width: u32, height: u32, max_fps: f64, min_fps: f64, pixel_formats: []const []const u8 };
        const JCam = struct { index: usize, name: []const u8, id: []const u8, default: bool, max_fps: f64, formats: []JFormat };
        var cams: std.ArrayList(JCam) = .empty;
        for (lst.devices, 0..) |d, i| {
            const rows = try summarizeFormats(a, d.formats);
            const jf = try a.alloc(JFormat, rows.len);
            var top: f64 = 0;
            for (rows, jf) |r, *j| {
                j.* = .{ .width = r.width, .height = r.height, .max_fps = r.max_fps, .min_fps = r.min_fps, .pixel_formats = r.pixel_formats.items };
                top = @max(top, r.max_fps);
            }
            try cams.append(a, .{ .index = i, .name = d.name, .id = d.id, .default = d.is_default, .max_fps = top, .formats = jf });
        }
        try std.json.Stringify.value(.{ .supported = true, .permission = @tagName(lst.permission), .cameras = cams.items }, .{ .whitespace = .indent_2 }, ctx.out);
        try ctx.out.writeAll("\n");
        return;
    }

    const w = ctx.out;
    const s = ctx.so;
    if (lst.devices.len == 0) {
        try w.writeAll("no cameras found\n");
    }
    for (lst.devices, 0..) |d, i| {
        if (i > 0) try w.writeAll("\n");
        try w.print("  {d}  ", .{i});
        try s.paint(w, .bold, d.name);
        if (d.is_default) try s.paint(w, .dim, "  (default)");
        try w.writeAll("\n");
        const rows = try summarizeFormats(a, d.formats);
        var t = term.Table.init(&.{ .{ .title = "SIZE" }, .{ .title = "MAX FPS", .alignment = .right }, .{ .title = "MIN FPS", .alignment = .right }, .{ .title = "PIXEL FORMATS" } });
        defer t.deinit(a);
        var top: f64 = 0;
        for (rows) |r| {
            top = @max(top, r.max_fps);
            var joined: std.ArrayList(u8) = .empty;
            for (r.pixel_formats.items, 0..) |pf, k| {
                if (k > 0) try joined.append(a, ' ');
                try joined.appendSlice(a, pf);
            }
            try t.addRow(a, &.{
                try std.fmt.allocPrint(a, "{d}x{d}", .{ r.width, r.height }),
                try std.fmt.allocPrint(a, "{d:.0}", .{r.max_fps}),
                try std.fmt.allocPrint(a, "{d:.0}", .{r.min_fps}),
                joined.items,
            });
        }
        try t.write(w, s, 5);
        try s.print(w, .dim, "     fastest: {d:.0} fps{s}\n", .{ top, if (top >= 119.5) "" else " (VISAR itself processes faster than this; the camera sets the pace)" });
    }
    try w.writeAll("\n");
    try s.print(w, .dim, "  camera access: {s}\n", .{switch (lst.permission) {
        .authorized => "allowed",
        .not_determined => "not asked yet (macOS asks on first use)",
        .denied => "denied: allow your terminal under System Settings > Privacy & Security > Camera",
        .restricted => "restricted by a policy on this Mac",
        .not_applicable => "governed by device file permissions",
    }});
}

const FormatRow = struct {
    width: u32,
    height: u32,
    max_fps: f64,
    min_fps: f64,
    pixel_formats: std.ArrayList([]const u8),
};

/// One row per frame size (largest first), with the best rate over its
/// pixel formats.
fn summarizeFormats(a: std.mem.Allocator, formats: []const camera.Format) ![]FormatRow {
    var rows: std.ArrayList(FormatRow) = .empty;
    for (formats) |f| {
        const row = for (rows.items) |*r| {
            if (r.width == f.width and r.height == f.height) break r;
        } else blk: {
            try rows.append(a, .{ .width = f.width, .height = f.height, .max_fps = 0, .min_fps = std.math.inf(f64), .pixel_formats = .empty });
            break :blk &rows.items[rows.items.len - 1];
        };
        row.max_fps = @max(row.max_fps, f.max_fps);
        row.min_fps = @min(row.min_fps, f.min_fps);
        const pf = try a.dupe(u8, std.mem.trimEnd(u8, &f.pixel_format, " "));
        for (row.pixel_formats.items) |x| {
            if (std.mem.eql(u8, x, pf)) break;
        } else try row.pixel_formats.append(a, pf);
    }
    std.mem.sort(FormatRow, rows.items, {}, struct {
        fn gt(_: void, x: FormatRow, y: FormatRow) bool {
            return @as(u64, x.width) * x.height > @as(u64, y.width) * y.height;
        }
    }.gt);
    for (rows.items) |*r| if (std.math.isInf(r.min_fps)) {
        r.min_fps = 0;
    };
    return rows.items;
}

fn cameraError(ctx: *Ctx, e: anyerror, cfg: Config) error{ Failed, Usage } {
    const synth = "or try 'visar track --source synthetic'";
    switch (e) {
        error.PermissionDenied => {
            ctx.hint("allow your terminal app (Terminal, iTerm, ...) under System Settings > Privacy & Security > Camera, then run this again; {s}", .{synth});
            return ctx.fail("camera access was denied", .{});
        },
        error.PermissionRestricted => {
            ctx.hint("a configuration profile or parental control blocks the camera on this Mac; {s}", .{synth});
            return ctx.fail("camera access is restricted", .{});
        },
        error.NoCamera => {
            ctx.hint("connect a camera, {s}", .{synth});
            return ctx.fail("no camera found", .{});
        },
        error.NoSuchCamera => {
            ctx.hint("'visar track --list-cameras' lists them, numbered from 0", .{});
            return ctx.usage("there is no camera {d}", .{cfg.camera_index orelse 0});
        },
        error.CameraUnavailable => {
            ctx.hint("quit other apps that use the camera, {s}", .{synth});
            return ctx.fail("cannot open the camera: {s}", .{camera.detail()});
        },
        error.ConfigureFailed => {
            ctx.hint("try another --size or --fps; 'visar track --list-cameras' shows what it supports", .{});
            return ctx.fail("the camera rejected the configuration: {s}", .{camera.detail()});
        },
        error.RuntimeUnavailable => {
            ctx.hint("{s}", .{synth});
            return ctx.fail("cannot load the system camera libraries: {s}", .{camera.detail()});
        },
        error.Unsupported => {
            ctx.hint("try 'visar track --source synthetic'", .{});
            return ctx.fail("no camera support on {s} yet", .{@tagName(builtin.os.tag)});
        },
        else => return ctx.fail("cannot start the camera: {s}", .{cx.describe(e)}),
    }
}

test "labels and format rows" {
    var b: [8]u8 = undefined;
    try std.testing.expectEqualStrings("FACETIME", upperLabel(&b, "FaceTime HD Camera"));
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const f = [_]camera.Format{
        .{ .width = 640, .height = 480, .min_fps = 1, .max_fps = 30, .pixel_format = "420v".* },
        .{ .width = 1280, .height = 720, .min_fps = 1, .max_fps = 30, .pixel_format = "420v".* },
        .{ .width = 1280, .height = 720, .min_fps = 5, .max_fps = 60, .pixel_format = "2vuy".* },
    };
    const rows = try summarizeFormats(arena.allocator(), &f);
    try std.testing.expectEqual(@as(usize, 2), rows.len);
    try std.testing.expectEqual(@as(u32, 1280), rows[0].width);
    try std.testing.expectEqual(@as(f64, 60), rows[0].max_fps);
    try std.testing.expectEqual(@as(usize, 2), rows[0].pixel_formats.items.len);
}
