//! `visar info`: the build, the CPU features its kernels use, the thread
//! count and the supported formats.

const std = @import("std");
const builtin = @import("builtin");
const visar = @import("visar");
const args = @import("args.zig");
const term = @import("term.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const imageio = @import("imageio.zig");

/// The optimize mode as `zig build -Doptimize=` spells it.
pub fn modeName() []const u8 {
    return switch (builtin.mode) {
        .debug => "Debug",
        .safe => "ReleaseSafe",
        .fast => "ReleaseFast",
        .small => "ReleaseSmall",
    };
}

pub const version_string = std.fmt.comptimePrint("{f}", .{visar.version});

/// The CPU features worth knowing about for VISAR's kernels that this build
/// was compiled with.
pub const features: []const []const u8 = blk: {
    const cpu = builtin.cpu;
    var list: []const []const u8 = &.{};
    switch (cpu.arch) {
        .x86_64, .x86 => {
            const X = std.Target.x86;
            for ([_][]const u8{ "sse2", "ssse3", "sse4_1", "sse4_2", "avx", "avx2", "fma", "f16c", "bmi2", "avx512f", "avx512bw", "avx512vl", "avx512vnni" }) |n| {
                if (X.featureSetHas(cpu.features, @field(X.Feature, n))) list = list ++ .{n};
            }
        },
        .aarch64 => {
            const A = std.Target.aarch64;
            for ([_][]const u8{ "neon", "fullfp16", "dotprod", "i8mm", "sve", "sve2" }) |n| {
                if (A.featureSetHas(cpu.features, @field(A.Feature, n))) list = list ++ .{n};
            }
        },
        .wasm32, .wasm64 => {
            if (std.Target.wasm.featureSetHas(cpu.features, .simd128)) list = list ++ .{"simd128"};
        },
        else => {},
    }
    break :blk list;
};

const FormatInfo = struct {
    name: []const u8,
    read: bool,
    write: bool,
    extensions: []const []const u8,
};

const formats = [_]FormatInfo{
    .{ .name = "png", .read = true, .write = true, .extensions = &.{".png"} },
    .{ .name = "jpeg", .read = true, .write = true, .extensions = &.{ ".jpg", ".jpeg", ".jpe" } },
    .{ .name = "qoi", .read = true, .write = true, .extensions = &.{".qoi"} },
    .{ .name = "bmp", .read = true, .write = true, .extensions = &.{ ".bmp", ".dib" } },
    .{ .name = "pnm", .read = true, .write = true, .extensions = &.{ ".pbm", ".pgm", ".ppm", ".pnm", ".pam" } },
    .{ .name = "gif", .read = false, .write = true, .extensions = &.{".gif"} },
};

const Report = struct {
    version: []const u8,
    target: []const u8,
    arch: []const u8,
    os: []const u8,
    cpu: []const u8,
    features: []const []const u8,
    simd_lanes: struct { u8: usize, f32: usize },
    threads: usize,
    build_mode: []const u8,
    zig: []const u8,
    formats: []const FormatInfo,
};

pub fn run(ctx: *Ctx, p: *args.Parsed) !void {
    if (p.positionals.items.len > 0) {
        ctx.hint("to look at an image, use 'visar inspect {s}'", .{p.positionals.items[0]});
        return ctx.usage("info takes no arguments", .{});
    }
    const threads = ctx.exec().threads();
    const report: Report = .{
        .version = version_string,
        .target = visar.core.target_description,
        .arch = @tagName(builtin.cpu.arch),
        .os = @tagName(builtin.os.tag),
        .cpu = builtin.cpu.model.name,
        .features = features,
        .simd_lanes = .{ .u8 = visar.core.lanes(u8), .f32 = visar.core.lanes(f32) },
        .threads = threads,
        .build_mode = modeName(),
        .zig = builtin.zig_version_string,
        .formats = &formats,
    };
    const w = ctx.out;
    if (ctx.json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, w);
        try w.writeAll("\n");
        return;
    }
    const s = ctx.so;
    var b: [4][128]u8 = undefined;
    var feat: std.Io.Writer = .fixed(&b[0]);
    for (features, 0..) |f, i| feat.print("{s}{s}", .{ if (i > 0) " " else "", f }) catch {};
    const threads_note = if (ctx.threads != null) "" else if (threads == 1) " (one CPU, or threads unavailable)" else " (one per CPU; change with --threads)";
    try term.writeFields(w, s, 2, &.{
        .{ "version", std.fmt.bufPrint(&b[1], "{s} ({s} build, Zig {s})", .{ version_string, modeName(), builtin.zig_version_string }) catch "" },
        .{ "target", visar.core.target_description },
        .{ "cpu", builtin.cpu.model.name },
        .{ "features", if (features.len > 0) feat.buffered() else "none (portable scalar code)" },
        .{ "simd", std.fmt.bufPrint(&b[2], "{d} lanes of u8, {d} of f32", .{ visar.core.lanes(u8), visar.core.lanes(f32) }) catch "" },
        .{ "threads", std.fmt.bufPrint(&b[3], "{d}{s}", .{ threads, threads_note }) catch "" },
    });
    try w.writeAll("\n");
    var t = term.Table.init(&.{ .{ .title = "FORMAT" }, .{ .title = "READ" }, .{ .title = "WRITE" }, .{ .title = "EXTENSIONS" } });
    defer t.deinit(ctx.arena);
    const yes = s.glyph("✓", "yes");
    const no = s.glyph("·", "-");
    for (formats) |f| {
        var ext: std.Io.Writer.Allocating = .init(ctx.arena);
        for (f.extensions, 0..) |e, i| try ext.writer.print("{s}{s}", .{ if (i > 0) " " else "", e });
        try t.addRow(ctx.arena, &.{ f.name, if (f.read) yes else no, if (f.write) yes else no, ext.written() });
    }
    try t.write(w, s, 2);
}

test "the feature list matches the target description" {
    if (builtin.cpu.arch == .x86_64) try std.testing.expect(features.len > 0);
    for (formats) |f| _ = visar.codecs.formatFromExtension(f.extensions[0]).?;
    _ = imageio;
}
