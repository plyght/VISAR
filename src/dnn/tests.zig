//! Model tests that need files: run when `VISAR_MODELS` names a directory
//! holding the OpenCV Model Zoo files, skipped otherwise. (Agreement with
//! OpenCV itself is checked by tests/py/test_dnn_vs_opencv.py.)

const std = @import("std");
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
