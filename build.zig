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

    // Unit tests.
    const test_filters = b.option([]const []const u8, "test-filter", "Only run tests whose name contains this substring") orelse &.{};
    const tests = b.addTest(.{ .root_module = visar, .filters = test_filters });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);
}
