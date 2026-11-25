//! The change-driven steady state of 4K face tracking (track_lib's
//! FaceTracker, as `visar track --faces` runs it): the OpenCV Model Zoo
//! selfie moving slowly over a 3840x2160 BGRA canvas (track_lib.StillScene:
//! a figure of eight, then still), and per frame the camera-side work a
//! real pipeline does: the gray tracking image (gray + downscale in one
//! pass) and the tracker step (MOSSE on every face; YuNet when scheduled,
//! on the whole frame or only around the faces). Rendering the scene is
//! not timed (a camera delivers the frame). Prints one JSON object.
//!
//!   zig build face4k -Doptimize=ReleaseFast    (bench/face4k.py runs this)
//!   visar-face4k-steady [--models DIR] [--threads N] [--face-width N]
//!       [--level L] [--detect-every N] [--frames N] [--size WxH]

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
    var opts: track.FaceOptions = .{ .face_width = 640 };
    var frames: u32 = 2 * track.StillScene.period;
    var W: u32 = 3840;
    var H: u32 = 2160;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const v = if (i + 1 < args.len) args[i + 1] else "";
        i += 1;
        if (std.mem.eql(u8, a, "--models")) {
            models = v;
        } else if (std.mem.eql(u8, a, "--threads")) {
            threads = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--face-width")) {
            opts.face_width = try std.fmt.parseInt(u32, v, 10);
        } else if (std.mem.eql(u8, a, "--level")) {
            opts.level = try std.fmt.parseInt(u8, v, 10);
        } else if (std.mem.eql(u8, a, "--detect-every")) {
            opts.detect_every = try std.fmt.parseInt(u32, v, 10);
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
        try visar.videoio.frame.grayDownsample(exec, frame, small, k);
        _ = try ft.step(exec, io, frame, small);
    }
    const s0 = ft.stats;
    const times = try gpa.alloc(f64, frames);
    defer gpa.free(times);
    for (times) |*t| {
        const off = scene.render(frame);
        const t0 = Io.Clock.awake.now(io);
        try visar.videoio.frame.grayDownsample(exec, frame, small, k);
        _ = try ft.step(exec, io, frame, small);
        const t1 = Io.Clock.awake.now(io);
        t.* = @as(f64, @floatFromInt(t1.nanoseconds - t0.nanoseconds)) / 1e6;
        truth.score(ft.faces(), off);
    }
    const s = ft.stats;
    var sum: f64 = 0;
    for (times) |t| sum += t;
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    const mean = sum / @as(f64, @floatFromInt(frames));
    var ob: [2048]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &ob);
    const w = &ow.interface;
    try w.print(
        \\{{"size":"{d}x{d}","detector":"{d}x{d}","threads":{d},"level":{d},"detect_every":{d},"frames":{d},
        \\"mean_ms":{d:.3},"p50_ms":{d:.3},"p95_ms":{d:.3},"max_ms":{d:.3},"fps":{d:.1},
        \\"full_runs":{d},"crop_rounds":{d},"crops":{d},"postponed":{d},"mean_full_ms":{d:.3},"mean_crop_ms":{d:.3},
        \\"faces":{d},"followed":{d},"truth_frames":{d}}}
        \\
    , .{
        W,                                                               H,
        ft.det_w,                                                        ft.det_h,
        exec.threads(),                                                  opts.level,
        opts.detect_every,                                               frames,
        mean,                                                            times[frames / 2],
        times[frames * 95 / 100],                                        times[frames - 1],
        1000 / mean,                                                     s.full_runs - s0.full_runs,
        s.crop_rounds - s0.crop_rounds,                                  s.crops - s0.crops,
        s.postponed - s0.postponed,                                      meanOr0(s.full_ms - s0.full_ms, s.full_runs - s0.full_runs),
        meanOr0(s.crop_ms - s0.crop_ms, s.crop_rounds - s0.crop_rounds), ft.faces().len,
        truth.followed,                                                  truth.frames,
    });
    try w.flush();
}

fn meanOr0(sum: f64, n: u64) f64 {
    return if (n > 0) sum / @as(f64, @floatFromInt(n)) else 0;
}
