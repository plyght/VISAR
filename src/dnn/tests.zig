//! Model tests that need files: run when `VISAR_MODELS` names a directory
//! holding the OpenCV Model Zoo files, skipped otherwise. (Agreement with
//! OpenCV itself is checked by tests/py/test_dnn_vs_opencv.py.)

const std = @import("std");
const builtin = @import("builtin");
const dnn = @import("dnn.zig");
const core = @import("../core/core.zig");
const testing = std.testing;

pub const yunet_names = [_][]const u8{ "face_detection_yunet_2023mar.onnx", "y.onnx" };

fn readModel(gpa: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    const dir = testing.environ.getAlloc(gpa, "VISAR_MODELS") catch return null;
    defer gpa.free(dir);
    for (names) |n| {
        const path = try std.fs.path.join(gpa, &.{ dir, n });
        defer gpa.free(path);
        const bytes = std.Io.Dir.cwd().readFileAlloc(testing.io, path, gpa, .limited(1 << 30)) catch continue;
        return bytes;
    }
    return null;
}

test "YuNet: blocked, fused and threaded plans agree with the plain reference plan" {
    const gpa = testing.allocator;
    const bytes = try readModel(gpa, &yunet_names) orelse return error.SkipZigTest;
    defer gpa.free(bytes);
    var pool = core.Pool.init(gpa, testing.io, 4);
    defer pool.deinit();

    var ref = try dnn.Net.load(gpa, bytes);
    defer ref.deinit();
    ref.opts = .{ .fuse = false, .blocked = false };
    var fast = try dnn.Net.load(gpa, bytes);
    defer fast.deinit();

    var prng = std.Random.DefaultPrng.init(3);
    for ([_][2]usize{ .{ 96, 128 }, .{ 64, 32 } }) |hw| {
        const shape = [_]usize{ 1, 3, hw[0], hw[1] };
        try ref.plan(&.{&shape});
        try fast.plan(&.{&shape});
        for (ref.input(0)) |*v| v.* = @floor(prng.random().float(f32) * 256);
        @memcpy(fast.input(0), ref.input(0));
        ref.run(.serial);
        fast.run(.on(&pool));
        try testing.expectEqual(@as(usize, 12), fast.outputCount());
        for (0..12) |i| {
            const a = ref.output(i);
            const b = fast.output(i);
            try testing.expect(a.shape.eql(b.shape));
            var mx: f32 = 0;
            for (a.data) |v| mx = @max(mx, @abs(v));
            for (a.data, b.data) |x, y| try testing.expectApproxEqAbs(x, y, 1e-4 * @max(1, mx));
        }
    }
}

pub const nanodet_names = [_][]const u8{"object_detection_nanodet_2022nov.onnx"};
pub const yolox_names = [_][]const u8{"object_detection_yolox_2022nov.onnx"};

/// The optimised plan (fused, blocked, streamed, threaded) against the
/// plain unfused reference plan on the same input.
fn checkPlans(bytes: []const u8, shape: []const usize, seed: u64, scale: f32, tol: f32) !void {
    const gpa = testing.allocator;
    var pool = core.Pool.init(gpa, testing.io, 4);
    defer pool.deinit();
    var ref = try dnn.Net.load(gpa, bytes);
    defer ref.deinit();
    ref.opts = .{ .fuse = false, .blocked = false };
    var fast = try dnn.Net.load(gpa, bytes);
    defer fast.deinit();
    try ref.plan(&.{shape});
    try fast.plan(&.{shape});
    var prng = std.Random.DefaultPrng.init(seed);
    for (ref.input(0)) |*v| v.* = (prng.random().float(f32) - 0.5) * scale;
    @memcpy(fast.input(0), ref.input(0));
    ref.run(.serial);
    fast.run(.on(&pool));
    try testing.expectEqual(ref.outputCount(), fast.outputCount());
    for (0..ref.outputCount()) |i| {
        const a = ref.output(i);
        const b = fast.output(i);
        try testing.expect(a.shape.eql(b.shape));
        var mx: f32 = 0;
        for (a.data) |v| mx = @max(mx, @abs(v));
        for (a.data, b.data) |x, y| try testing.expectApproxEqAbs(x, y, tol * @max(1, mx));
    }
}

test "NanoDet: optimised plan (channel-shuffle gathers, views) agrees with the plain reference" {
    const gpa = testing.allocator;
    const bytes = try readModel(gpa, &nanodet_names) orelse return error.SkipZigTest;
    defer gpa.free(bytes);
    try checkPlans(bytes, &.{ 1, 3, 416, 416 }, 5, 4, 1e-4);
}

test "YOLOX: optimised plan (SiLU epilogues, strided slices, in-place concats) agrees with the plain reference" {
    const gpa = testing.allocator;
    const bytes = try readModel(gpa, &yolox_names) orelse return error.SkipZigTest;
    defer gpa.free(bytes);
    // A small input keeps the scalar reference plan quick.
    try checkPlans(bytes, &.{ 1, 3, 96, 128 }, 6, 255, 1e-4);
}

test "ObjectDetector: both models run on any image size" {
    const gpa = testing.allocator;
    inline for (.{ &nanodet_names, &yolox_names }, .{ dnn.detect.Arch.nanodet, dnn.detect.Arch.yolox }) |names, arch| {
        if (try readModel(gpa, names)) |bytes| {
            defer gpa.free(bytes);
            var det = try dnn.ObjectDetector.create(gpa, bytes, .{});
            defer det.deinit();
            try testing.expectEqual(arch, det.arch);
            // YOLOX runs at 640x640 whatever the image size: one image, and
            // none in Debug builds (too slow unoptimised).
            const sizes: []const [2]u32 = if (arch == .nanodet) &.{ .{ 200, 120 }, .{ 120, 200 }, .{ 64, 64 }, .{ 700, 500 } } else if (builtin.mode == .debug) &.{} else &.{.{ 200, 120 }};
            for (sizes) |sz| {
                var img = try core.Image(u8).initFill(gpa, sz[0], sz[1], 3, 0);
                defer img.deinit(gpa);
                // A bright square on dark: whatever is found lies in the image.
                for (sz[1] / 4..sz[1] / 2) |y| @memset(img.data[y * img.stride + sz[0] / 4 * 3 ..][0 .. sz[0] / 4 * 3], 230);
                det.score_threshold = 0.05;
                const dets = try det.detect(.serial, img);
                for (dets, 0..) |d, i| {
                    try testing.expect(d.class_id >= 0 and d.class_id < 80);
                    try testing.expect(d.score > 0.05 and d.score <= 1);
                    try testing.expect(d.box[2] >= 0 and d.box[3] >= 0);
                    if (i > 0) try testing.expect(d.score <= dets[i - 1].score);
                }
            }
        }
    }
}

test "FaceDetector runs on a blank image" {
    const gpa = testing.allocator;
    const bytes = try readModel(gpa, &yunet_names) orelse return error.SkipZigTest;
    defer gpa.free(bytes);
    var det = try dnn.FaceDetector.create(gpa, bytes, .{ .width = 100, .height = 70 });
    defer det.deinit();
    var img = try core.Image(u8).initFill(gpa, 100, 70, 3, 128);
    defer img.deinit(gpa);
    const faces = try det.detect(.serial, img);
    try testing.expectEqual(@as(usize, 0), faces.len);
    try det.setInputSize(64, 64);
    try testing.expectError(error.SizeMismatch, det.detect(.serial, img));
}
