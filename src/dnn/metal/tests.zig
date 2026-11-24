//! Metal backend tests: GPU outputs against the CPU backend (the
//! numerical reference). They run where a Metal device exists (macOS) and
//! skip elsewhere, unless VISAR_REQUIRE_METAL is set, which turns a
//! missing device into a failure (CI on a Mac sets it).

const std = @import("std");
const metal = @import("metal.zig");
const net_mod = @import("../net.zig");
const dnn_tests = @import("../tests.zig");
const core = @import("../../core/core.zig");
const testing = std.testing;
const Net = net_mod.Net;

fn readModel(gpa: std.mem.Allocator, names: []const []const u8) !?[]u8 {
    const dir = testing.environ.getAlloc(gpa, "VISAR_MODELS") catch return null;
    defer gpa.free(dir);
    for (names) |n| {
        const path = try std.fs.path.join(gpa, &.{ dir, n });
        defer gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(testing.io, path, gpa, .limited(1 << 30)) catch continue;
    }
    return null;
}

fn requireMetal() !void {
    if (metal.available()) return;
    const want = testing.environ.getAlloc(testing.allocator, "VISAR_REQUIRE_METAL") catch return error.SkipZigTest;
    testing.allocator.free(want);
    std.debug.print("VISAR_REQUIRE_METAL is set but Metal is unavailable: {s}\n", .{metal.unavailableReason()});
    return error.MetalUnavailable;
}

/// Max |a - b| over max(1, max |a|), for every output of two planned nets
/// run on the same input.
fn compare(cpu: *Net, gpu: *Net, tol: f32) !void {
    try testing.expectEqual(cpu.outputCount(), gpu.outputCount());
    for (0..cpu.outputCount()) |i| {
        const a = cpu.output(i);
        const b = gpu.output(i);
        try testing.expect(a.shape.eql(b.shape));
        var mx: f32 = 0;
        for (a.data) |v| mx = @max(mx, @abs(v));
        var err: f32 = 0;
        for (a.data, b.data) |x, y| err = @max(err, @abs(x - y));
        const rel = err / @max(1, mx);
        if (!(rel <= tol)) {
            std.debug.print("output {s}: max rel error {e} (tolerance {e})\n", .{ cpu.outputName(i), rel, tol });
            return error.TestExpectedApproxEqAbs;
        }
    }
}

test "backend selection: auto keeps tiny models on the CPU; metal without a device fails cleanly" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const bytes = try metal.test_model.build(arena_state.allocator());
    var net = try Net.load(testing.allocator, bytes);
    defer net.deinit();
    try testing.expectEqual(@as(?net_mod.Device, null), net.device());
    try net.plan(&.{&.{ 1, 3, 16, 20 }});
    try testing.expectEqual(@as(?net_mod.Device, .cpu), net.device());
    net.opts.backend = .metal;
    if (!metal.available()) {
        try testing.expectError(error.BackendUnavailable, net.plan(&.{&.{ 1, 3, 16, 20 }}));
        try testing.expectEqual(@as(?net_mod.Device, null), net.device());
        return;
    }
    try net.plan(&.{&.{ 1, 3, 16, 20 }});
    try testing.expectEqual(@as(?net_mod.Device, .metal), net.device());
}

test "Metal matches the CPU backend on every kernel" {
    try requireMetal();
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const bytes = try metal.test_model.build(arena_state.allocator());
    var cpu = try Net.load(testing.allocator, bytes);
    defer cpu.deinit();
    var gpu = try Net.load(testing.allocator, bytes);
    defer gpu.deinit();
    var prng = std.Random.DefaultPrng.init(11);
    for ([_]bool{ true, false }) |fuse| {
        cpu.opts = .{ .fuse = fuse, .backend = .cpu };
        gpu.opts = .{ .fuse = fuse, .backend = .metal };
        try cpu.plan(&.{&.{ 1, 3, 16, 20 }});
        try gpu.plan(&.{&.{ 1, 3, 16, 20 }});
        try testing.expectEqual(@as(?net_mod.Device, .metal), gpu.device());
        for (0..3) |_| {
            for (cpu.input(0)) |*v| v.* = (prng.random().float(f32) - 0.5) * 2;
            @memcpy(gpu.input(0), cpu.input(0));
            cpu.run(.serial);
            try gpu.tryRun(.serial);
            try compare(&cpu, &gpu, 1e-4);
        }
    }
}

test "YuNet on Metal matches the CPU backend" {
    try requireMetal();
    const gpa = testing.allocator;
    const bytes = try readModel(gpa, &dnn_tests.yunet_names) orelse return error.SkipZigTest;
    defer gpa.free(bytes);
    var pool = core.Pool.init(gpa, testing.io, 4);
    defer pool.deinit();
    var cpu = try Net.load(gpa, bytes);
    defer cpu.deinit();
    cpu.opts.backend = .cpu;
    var gpu = try Net.load(gpa, bytes);
    defer gpu.deinit();
    gpu.opts.backend = .metal;
    var prng = std.Random.DefaultPrng.init(3);
    for ([_][2]usize{ .{ 96, 128 }, .{ 480, 640 }, .{ 64, 32 } }) |hw| {
        const shape = [_]usize{ 1, 3, hw[0], hw[1] };
        try cpu.plan(&.{&shape});
        try gpu.plan(&.{&shape});
        for (cpu.input(0)) |*v| v.* = @floor(prng.random().float(f32) * 256);
        @memcpy(gpu.input(0), cpu.input(0));
        cpu.run(.on(&pool));
        try gpu.tryRun(.serial);
        try compare(&cpu, &gpu, 1e-4);
    }
}
