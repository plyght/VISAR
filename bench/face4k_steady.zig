//! The change-driven steady state of 4K face tracking (track_lib's
//! FaceTracker, as `visar track --faces` runs it): the OpenCV Model Zoo
//! selfie moving slowly over a 3840x2160 BGRA canvas (track_lib.StillScene:
//! a figure of eight, then still), and per frame the camera-side work a
//! real pipeline does: the gray tracking image (gray + downscale in one
//! pass) and the tracker step (MOSSE on every face; YuNet when scheduled,
//! on the whole frame or only around the faces). Rendering the scene is
//! not timed (a camera delivers the frame). Prints one JSON object. The
//! tracking image comes from FaceTracker.trackingImage, which on frames
//! that will probably run a whole-frame detection also writes the
//! detector input in the same pass (counted in "fused_preps"; the time
//! then sits in "gray_ms", not in the detection).
//!
//!   zig build face4k -Doptimize=ReleaseFast    (bench/face4k.py runs this)
//!   visar-face4k-steady [--models DIR] [--threads N] [--face-width N]
//!       [--level L] [--detect-every N] [--frames N] [--size WxH]
//!       [--backend auto|cpu|metal] [--separate] [--full-gray]
//!       [--still-stride N]
//!   --separate: the tracking image always on its own (grayDownsample).
//!   --full-gray: the whole tracking image every frame (FaceOptions
//!       .partial_gray off; by default frames without a whole-frame
//!       detection compute it only where the tracker reads it).
//! Tracking quality: "followed" frames with a track at IoU >= 0.5 to the
//! face moved with the image, "mean_iou" and "mean_err_px" (centre error
//! at 4K) of the best track; "gray_read" is the share of the tracking
//! image computed (so of the frame read for it). "gray_ms" times
//! trackingImage only: the parts of a partial image the step computes count
//! in "track_ms".
//! Per-frame latency: every timed frame is timed alone; "p50_ms" ..
//! "max_ms" are nearest-rank percentiles over all frames, and "by_step"
//! splits the frames by what the step did: whole-frame detections
//! ("full"), crop-only detections ("crops") and frames without a detection
//! ("tracked", postponed ones included), each with the count, mean, p50,
//! p95, p99 and max ms of the whole frame (gray image + step).
//!   --profile: instead, the CPU program of YuNet step by step at 1 and
//!       --threads threads (each step timed alone).

const std = @import("std");
const visar = @import("visar");
const track = @import("track_lib");
const Io = std.Io;
const Image = visar.Image;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var models: ?[]const u8 = init.environ_map.get("VISAR_MODELS");
    var threads: usize = 1;
    var opts: track.FaceOptions = .{ .face_width = 640, .backend = .cpu };
    var frames: u32 = 2 * track.StillScene.period;
    var W: u32 = 3840;
    var H: u32 = 2160;
    var profile = false;
    var separate = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--profile")) {
            profile = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--separate")) {
            separate = true;
            continue;
        }
        if (std.mem.eql(u8, a, "--full-gray")) {
            opts.partial_gray = false;
            continue;
        }
        const v = if (i + 1 < args.len) args[i + 1] else "";
        i += 1;
        if (std.mem.eql(u8, a, "--models")) {
            models = v;
        } else if (std.mem.eql(u8, a, "--threads")) {
            threads = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--face-width")) {
            opts.face_width = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--backend")) {
            opts.backend = std.meta.stringToEnum(visar.dnn.Backend, v) orelse return error.UnknownBackend;
        } else if (std.mem.eql(u8, a, "--level")) {
            opts.level = try std.fmt.parseInt(u8, v, 10);
        } else if (std.mem.eql(u8, a, "--detect-every")) {
            opts.detect_every = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--still-stride")) {
            opts.still_stride = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--frames")) {
            frames = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--size")) {
            var it = std.mem.splitScalar(u8, v, 'x');
            W = try std.fmt.parseInt(u32, it.next() orelse "", 10);
            H = try std.fmt.parseInt(u32, it.next() orelse "", 10);
        } else return error.UnknownOption;
    }
    const dir = models orelse return error.NoModels;
    const cwd = Io.Dir.cwd();
    const mpath = try std.fs.path.join(gpa, &.{ dir, "face_detection_yunet_2023mar.onnx" });
    defer gpa.free(mpath);
    const model = try cwd.readFileAlloc(io, mpath, gpa, .limited(1 << 26));
    defer gpa.free(model);
    const ipath = try std.fs.path.join(gpa, &.{ dir, "largest_selfie.jpg" });
    defer gpa.free(ipath);
    const jpg = try cwd.readFileAlloc(io, ipath, gpa, .limited(1 << 26));
    defer gpa.free(jpg);
    var decoded = try visar.codecs.decode(gpa, jpg, .{ .mode = .color, .order = .bgr });
    defer decoded.deinit(gpa);

    var pool = visar.Pool.init(gpa, io, if (threads == 0) null else threads);
    defer pool.deinit();
    const exec = visar.Exec.on(&pool);
    if (profile) return profileNet(gpa, io, exec, model, opts.face_width, W, H);
    // The tracking image: the detector's scale (as `visar track` picks it).
    const k: u32 = @max(1, W / @max(1, opts.face_width));
    var scene = try track.StillScene.init(gpa, exec, decoded.u8, W, H);
    defer scene.deinit();
    var ft = try track.FaceTracker.create(gpa, model, W, H, k, opts);
    defer ft.deinit();
    var frame = try Image(u8).init(gpa, W, H, 4);
    defer frame.deinit(gpa);
    var small = try Image(u8).init(gpa, W / k, H / k, 1);
    defer small.deinit(gpa);

    // Warm-up (plans the crop sizes, starts the tracks), then the timed run
    // from the start of the motion.
    var truth: track.StillTruth = .{};
    for (0..track.StillScene.period) |_| {
        _ = scene.render(frame);
        if (separate) try visar.videoio.frame.grayDownsample(exec, frame, small, k) else try ft.trackingImage(exec, frame, small);
        _ = try ft.step(exec, io, frame, small);
    }
    const s0 = ft.stats;
    const times = try gpa.alloc(f64, frames);
    defer gpa.free(times);
    const kinds = try gpa.alloc(track.FaceStep, frames);
    defer gpa.free(kinds);
    var gray_ms: f64 = 0;
    for (times, kinds) |*t, *kind| {
        const off = scene.render(frame);
        const t0 = Io.Clock.awake.now(io);
        if (separate) try visar.videoio.frame.grayDownsample(exec, frame, small, k) else try ft.trackingImage(exec, frame, small);
        const tg = Io.Clock.awake.now(io);
        kind.* = try ft.step(exec, io, frame, small);
        const t1 = Io.Clock.awake.now(io);
        t.* = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
        gray_ms += @as(f64, @floatFromInt(tg.nanoseconds - t0.nanoseconds)) / 1e6;
        truth.score(ft.faces(), off);
    }
    const s = ft.stats;
    // Per kind of step (before `times` is sorted).
    var by: [3]Dist = undefined;
    const groups = [3][]const track.FaceStep{ &.{.full}, &.{.crops}, &.{ .tracked, .postponed } };
    const scratch = try gpa.alloc(f64, frames);
    defer gpa.free(scratch);
    for (groups, &by) |g, *d| {
        var n: usize = 0;
        for (times, kinds) |t, kind| if (std.mem.indexOfScalar(track.FaceStep, g, kind) != null) {
            scratch[n] = t;
            n += 1;
        };
        d.* = .of(scratch[0..n]);
    }
    const all: Dist = .of(times);
    const mean = all.mean;
    var ob: [4096]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &ob);
    const w = &ow.interface;
    try w.print(
        \\{{"size":"{d}x{d}","detector":"{d}x{d}","threads":{d},"level":{d},"detect_every":{d},"frames":{d},
        \\"mean_ms":{d:.3},"p50_ms":{d:.3},"p95_ms":{d:.3},"p99_ms":{d:.3},"max_ms":{d:.3},"fps":{d:.1},
        \\"by_step":{{"full":{f},"crops":{f},"tracked":{f}}},
        \\
    , .{
        W,                 H,
        ft.det_w,          ft.det_h,
        exec.threads(),    opts.level,
        opts.detect_every, frames,
        mean,              all.p50,
        all.p95,           all.p99,
        all.max,           1000 / mean,
        by[0],             by[1],
        by[2],
    });
    try w.print(
        \\"full_runs":{d},"crop_rounds":{d},"crops":{d},"postponed":{d},"mean_full_ms":{d:.3},"mean_crop_ms":{d:.3},
        \\"gray_ms":{d:.3},"track_ms":{d:.3},"detect_ms":{d:.3},
        \\"faces":{d},"followed":{d},"truth_frames":{d},"mean_iou":{d:.4},"mean_err_px":{d:.2},"fused_preps":{d},"fused_unused":{d},
        \\"partial_gray":{},"partial_frames":{d},"gray_read":{d:.4}}}
        \\
    , .{
        s.full_runs - s0.full_runs,
        s.crop_rounds - s0.crop_rounds,
        s.crops - s0.crops,
        s.postponed - s0.postponed,
        meanOr0(s.full_ms - s0.full_ms, s.full_runs - s0.full_runs),
        meanOr0(s.crop_ms - s0.crop_ms, s.crop_rounds - s0.crop_rounds),
        gray_ms / @as(f64, @floatFromInt(frames)),
        (s.track_ms - s0.track_ms) / @as(f64, @floatFromInt(frames)),
        (s.detectMs() - s0.detectMs()) / @as(f64, @floatFromInt(frames)),
        ft.faces().len,
        truth.followed,
        truth.frames,
        truth.iou_sum / @max(1, @as(f64, @floatFromInt(truth.frames))),
        truth.err_sum / @max(1, @as(f64, @floatFromInt(truth.frames))),
        s.fused_preps - s0.fused_preps,
        s.fused_unused - s0.fused_unused,
        opts.partial_gray,
        s.partial_frames - s0.partial_frames,
        @as(f64, @floatFromInt(s.gray_px - s0.gray_px)) / (@as(f64, @floatFromInt(frames)) * @as(f64, @floatFromInt(@as(u64, small.width) * small.height))),
    });
    try w.flush();
}

/// The distribution of per-frame times (ms): nearest-rank percentiles.
const Dist = struct {
    n: usize = 0,
    mean: f64 = 0,
    p50: f64 = 0,
    p95: f64 = 0,
    p99: f64 = 0,
    max: f64 = 0,

    /// Sorts `ts`.
    fn of(ts: []f64) Dist {
        if (ts.len == 0) return .{};
        std.mem.sort(f64, ts, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (ts) |t| sum += t;
        return .{ .n = ts.len, .mean = sum / @as(f64, @floatFromInt(ts.len)), .p50 = rank(ts, 50), .p95 = rank(ts, 95), .p99 = rank(ts, 99), .max = ts[ts.len - 1] };
    }

    /// The smallest time with at least p% of the times at or below it.
    fn rank(sorted: []const f64, p: usize) f64 {
        const i = (sorted.len * p + 99) / 100; // ceil(n p / 100), 1-based
        return sorted[@max(i, 1) - 1];
    }

    pub fn format(d: Dist, w: *Io.Writer) Io.Writer.Error!void {
        try w.print("{{\"n\":{d},\"mean_ms\":{d:.3},\"p50_ms\":{d:.3},\"p95_ms\":{d:.3},\"p99_ms\":{d:.3},\"max_ms\":{d:.3}}}", .{ d.n, d.mean, d.p50, d.p95, d.p99, d.max });
    }
};

test "nearest-rank percentiles" {
    var ts: [100]f64 = undefined;
    for (&ts, 0..) |*t, i| t.* = @floatFromInt(100 - i);
    const d: Dist = .of(&ts);
    try std.testing.expectEqual(@as(f64, 50), d.p50);
    try std.testing.expectEqual(@as(f64, 95), d.p95);
    try std.testing.expectEqual(@as(f64, 99), d.p99);
    try std.testing.expectEqual(@as(f64, 100), d.max);
    try std.testing.expectEqual(@as(f64, 50.5), d.mean);
}

fn meanOr0(sum: f64, n: u64) f64 {
    return if (n > 0) sum / @as(f64, @floatFromInt(n)) else 0;
}

/// Times every item of the CPU program (one kernel or a streamed chain)
/// alone, serially and on `exec`, and prints the table.
fn profileNet(gpa: std.mem.Allocator, io: Io, exec: visar.Exec, model: []const u8, fw: u32, W: u32, H: u32) !void {
    const fh = fw * H / W;
    var det = try visar.dnn.FaceDetector.create(gpa, model, .{ .width = fw, .height = fh, .backend = .cpu });
    defer det.deinit();
    try profileProgram(io, exec, &det);
    // Again without streaming: every step alone (the chains' layers).
    det.net.opts.stream = false;
    try det.setInputSize(fw, fh);
    try profileProgram(io, exec, &det);
}

fn profileProgram(io: Io, exec: visar.Exec, det: *visar.dnn.FaceDetector) !void {
    const c = &det.net.compiled.?;
    const base: [*]f32 = @ptrCast(c.buf.ptr);
    var ob: [4096]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &ob);
    const w = &ow.interface;
    const macs = visar.dnn.metal.encode.convMacs(&c.plan);
    try w.print("YuNet {d}x{d} ({d:.1} M conv MACs): item, steps, 1T ms, {d}T ms, scaling\n", .{ det.pad_w, det.pad_h, @as(f64, @floatFromInt(macs)) / 1e6, exec.threads() });
    var tot = [2]f64{ 0, 0 };
    var whole = [2]f64{ 0, 0 };
    for (0..2) |e| {
        const ex = if (e == 0) visar.Exec.serial else exec;
        var best: f64 = 1e9;
        for (0..30) |_| {
            const t0 = Io.Clock.awake.now(io);
            c.program.run(ex, c.buf);
            best = @min(best, @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds)) / 1e6);
        }
        whole[e] = best;
    }
    // The same items back to back, each its own pool job (no team).
    var per_item: f64 = 1e9;
    for (0..30) |_| {
        const t0 = Io.Clock.awake.now(io);
        for (c.program.items) |*it| it.run(exec, base);
        per_item = @min(per_item, @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds)) / 1e6);
    }
    try w.print("items as separate pool jobs: {d:.3} ms\n", .{per_item});
    var step: usize = 0;
    for (c.program.items, 0..) |*it, n| {
        var ms: [2]f64 = undefined;
        for (0..2) |e| {
            const ex = if (e == 0) visar.Exec.serial else exec;
            var best: f64 = 1e9;
            for (0..30) |_| {
                const t0 = Io.Clock.awake.now(io);
                it.run(ex, base);
                best = @min(best, @as(f64, @floatFromInt(Io.Clock.awake.now(io).nanoseconds - t0.nanoseconds)) / 1e6);
            }
            ms[e] = best;
            tot[e] += best;
        }
        const name: []const u8 = switch (it.*) {
            .kernel => |k| @tagName(k.*),
            .chain => "chain",
        };
        if (ms[0] > 0.02) {
            try w.print("{d:>3} {s:<12} {d:>2} {d:>7.3} {d:>7.3} {d:>5.2}x  ", .{ n, name, it.steps(), ms[0], ms[1], ms[0] / ms[1] });
            for (c.plan.steps[step..][0..it.steps()]) |st| {
                const o = c.plan.tensors[st.outputs[0]].shape;
                const x = c.plan.tensors[st.inputs[0]].shape;
                switch (st.op) {
                    .conv => |cv| try w.print("conv{d}x{d}s{d}g{d} {d}->{d}@{d}x{d}; ", .{ cv.kernel[0], cv.kernel[1], cv.stride[0], cv.group, x.d[1], o.d[1], o.d[2], o.d[3] }),
                    else => try w.print("{s} {d}@{d}x{d}; ", .{ @tagName(st.op), o.d[1], o.d[2], o.d[3] }),
                }
            }
            try w.print("\n", .{});
        }
        step += it.steps();
    }
    try w.print("sum of items {d:.3} / {d:.3} ms; whole program (team run) {d:.3} / {d:.3} ms, {d:.2}x\n", .{ tot[0], tot[1], whole[0], whole[1], whole[0] / whole[1] });
    try w.flush();
}
