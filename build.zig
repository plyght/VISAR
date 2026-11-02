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
    b.installArtifact(shared);
    b.installArtifact(static);
    b.installFile("include/visar.h", "include/visar.h");

    // The `visar` command-line tool.
    const cli = b.createModule(.{
        .root_source_file = b.path("src/cli/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    cli.addImport("visar", visar);
    const exe = b.addExecutable(.{ .name = "visar", .root_module = cli });
    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    b.step("run", "Run the visar CLI").dependOn(&run_cmd.step);

    // Unit tests.
    const test_filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring") orelse &.{};
    const tests = b.addTest(.{ .root_module = visar, .filters = test_filters });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
    const capi_tests = b.addTest(.{ .root_module = capi, .filters = test_filters });
    test_step.dependOn(&b.addRunArtifact(capi_tests).step);

    // Differential tests against OpenCV (needs python3 with numpy and cv2).
    const difftest = b.addSystemCommand(&.{ "python3", "tests/py/test_vs_opencv.py" });
    difftest.setEnvironmentVariable("VISAR_LIB", b.getInstallPath(.lib, "libvisar.so"));
    difftest.step.dependOn(b.getInstallStep());
    b.step("difftest", "Compare against OpenCV (python3 + cv2)").dependOn(&difftest.step);

    const bench = b.addSystemCommand(&.{ "python3", "bench/bench_vs_opencv.py" });
    bench.setEnvironmentVariable("VISAR_LIB", b.getInstallPath(.lib, "libvisar.so"));
    bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| bench.addArgs(args);
    b.step("bench", "Benchmark against OpenCV (use -Doptimize=ReleaseFast)").dependOn(&bench.step);
}
