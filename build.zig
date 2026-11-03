const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // The library module other Zig projects import:
    //   const visar = b.dependency("visar", .{}).module("visar");
    const visar = b.addModule("visar", .{
        .root_source_file = b.path("src/visar.zig"),
        .target = target,
        .optimize = optimize,
    });

    // The C ABI as a shared and a static library (include/visar.h).
    // A library loaded into a foreign process gets no Zig start-up code, so
    // it uses the system C library (an OS interface, like the kernel ABI)
    // for threads and page-size queries where one exists.
    const os_tag = target.result.os.tag;
    const capi = b.createModule(.{
        .root_source_file = b.path("src/capi.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = os_tag != .freestanding and os_tag != .wasi and os_tag != .emscripten,
    });
    const shared = b.addLibrary(.{ .name = "visar", .linkage = .dynamic, .root_module = capi });
    const static = b.addLibrary(.{ .name = "visar", .linkage = .static, .root_module = capi });
    // Dynamic libraries do not exist on wasm; the static library does.
    if (!target.result.cpu.arch.isWasm()) b.installArtifact(shared);
    b.installArtifact(static);
    b.installFile("include/visar.h", "include/visar.h");

    // The tracking demo's core, shared by `zig build demo-track` and
    // `visar demo track`.
    const track_lib = b.createModule(.{
        .root_source_file = b.path("demos/track_lib.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "visar", .module = visar }},
    });

    // The `visar` command-line tool.
    const cli = b.createModule(.{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "visar", .module = visar },
            .{ .name = "track_lib", .module = track_lib },
        },
    });
    const exe = b.addExecutable(.{ .name = "visar", .root_module = cli });
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the visar CLI").dependOn(&run_cmd.step);

    // Demos: `zig build demo-track -Doptimize=ReleaseFast -- --out track.gif`
    const demo_track = b.addExecutable(.{ .name = "visar-demo-track", .root_module = b.createModule(.{
        .root_source_file = b.path("demos/track.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "visar", .module = visar },
            .{ .name = "track_lib", .module = track_lib },
        },
    }) });
    b.installArtifact(demo_track);
    const run_track = b.addRunArtifact(demo_track);
    run_track.addPassthruArgs();
    b.step("demo-track", "Render the live tracking demo to a GIF").dependOn(&run_track.step);

    // Unit tests.
    const test_filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring") orelse &.{};
    // `-Duse-llvm`: Debug tests through LLVM instead of the self-hosted
    // backend (as on Windows/macOS), which catches LLVM-only miscompiles.
    const use_llvm = b.option(bool, "use-llvm", "Compile tests with the LLVM backend even in Debug") orelse null;
    const tests = b.addTest(.{ .root_module = visar, .filters = test_filters, .use_llvm = use_llvm });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    const capi_tests = b.addTest(.{ .root_module = capi, .filters = test_filters, .use_llvm = use_llvm });
    test_step.dependOn(&b.addRunArtifact(capi_tests).step);
    const cli_tests = b.addTest(.{ .root_module = cli, .filters = test_filters, .use_llvm = use_llvm });
    test_step.dependOn(&b.addRunArtifact(cli_tests).step);

    // Differential tests against OpenCV (needs python3 with numpy and cv2).
    const difftest = b.addSystemCommand(&.{ "python3", "tests/py/run_all.py", "--lib" });
    difftest.addArtifactArg(shared);
    b.step("difftest", "Compare against OpenCV (python3 + cv2)").dependOn(&difftest.step);

    const bench = b.addSystemCommand(&.{ "python3", "bench/bench_vs_opencv.py", "--lib" });
    bench.addArtifactArg(shared);
    bench.addPassthruArgs();
    b.step("bench", "Benchmark against OpenCV (use -Doptimize=ReleaseFast)").dependOn(&bench.step);
}
