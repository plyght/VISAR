//! `visar demo track`: the live multi-object tracking demo (the same code
//! as `zig build demo-track`, in demos/track_lib.zig), with a progress bar,
//! the overwrite rules of every other command, and a summary or JSON.

const std = @import("std");
const visar = @import("visar");
const track = @import("track_lib");
const args = @import("args.zig");
const term = @import("term.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const imageio = @import("imageio.zig");

pub const opts = [_]args.Opt{
    .{ .long = "out", .value = "FILE", .help = "The GIF to write (default track.gif; '-' = stdout); same as -o" },
    .{ .long = "frames", .short = 'n', .value = "N", .help = "Frames to render, 9-10000 (default 150)" },
    .{ .long = "size", .value = "WxH", .help = "Scene size (default 960x540; the GIF is at most 480 wide)" },
    .{ .long = "seed", .value = "N", .help = "Seed of the scene's random texture (default 7)" },
};

const demos = [_][]const u8{"track"};

pub fn run(ctx: *Ctx, p: *args.Parsed) !void {
    const pos = p.positionals.items;
    if (pos.len == 0) {
        ctx.hint("available: track  (e.g. 'visar demo track')", .{});
        return ctx.usage("demo needs the name of a demo", .{});
    }
    if (!std.mem.eql(u8, pos[0], "track")) {
        if (args.suggest(pos[0], &demos)) |s| ctx.hint("did you mean '{s}'?", .{s}) else ctx.hint("available: track", .{});
        return ctx.usage("unknown demo '{s}'", .{pos[0]});
    }
    if (pos.len > 1) return ctx.usage("unexpected argument '{s}'", .{pos[1]});

    var o: track.Options = .{};
    if (p.value("frames")) |v| o.frames = try args.int(u32, ctx.diag, "--frames", v, track.warmup + 1, 10_000);
    if (p.value("seed")) |v| o.seed = try args.int(u64, ctx.diag, "--seed", v, 0, std.math.maxInt(u64));
    if (p.value("size")) |v| {
        const s = try args.size(ctx.diag, v, false);
        o.width = s.w.?;
        o.height = s.h.?;
        if (o.width < track.Options.min_width or o.height < track.Options.min_height) {
            return ctx.usage("--size must be at least {d}x{d}", .{ track.Options.min_width, track.Options.min_height });
        }
        if (o.width > 7680 or o.height > 4320) return ctx.usage("--size is limited to 7680x4320", .{});
    }
    if (ctx.format) |f| if (f != .gif) {
        ctx.hint("the demo writes an animated GIF; drop --format", .{});
        return ctx.usage("--format {s} does not apply to the demo", .{@tagName(f)});
    };
    if (p.value("out") != null and ctx.output != null) return ctx.usage("give the output once, with --out or -o", .{});
    const path = p.value("out") orelse ctx.output orelse "track.gif";
    ctx.format = .gif;
    const out = try imageio.prepare(ctx, path);

    const Bar = struct {
        bar: term.Progress,
        fn tick(c: *anyopaque, done: u32, total: u32) void {
            const self: *@This() = @ptrCast(@alignCast(c));
            var b: [32]u8 = undefined;
            self.bar.update(done, std.fmt.bufPrint(&b, "frame {d} of {d}", .{ done, total }) catch "");
        }
    };
    var bar: Bar = .{ .bar = ctx.progress(o.frames) };
    const exec = ctx.exec();
    if (!ctx.quiet and !ctx.json) {
        ctx.se.print(ctx.err, .dim, "  tracking a {d}x{d} scene for {d} frames on {d} thread{s}\n", .{ o.width, o.height, o.frames, exec.threads(), if (exec.threads() == 1) "" else "s" }) catch {};
        ctx.err.flush() catch {};
    }
    var result = track.run(ctx.gpa, ctx.io, exec, o, .{ .ctx = &bar, .func = Bar.tick }) catch |e| {
        bar.bar.finish();
        return ctx.fail("the tracking demo failed: {s}", .{cx.describe(e)});
    };
    defer result.deinit(ctx.gpa);
    bar.bar.finish();
    try imageio.writeBytes(ctx, out, result.gif);

    const st = result.stats;
    if (ctx.json) {
        try std.json.Stringify.value(.{
            .demo = "track",
            .output = path,
            .target = visar.core.target_description,
            .pipeline = track.pipeline_description,
            .stats = st,
        }, .{ .whitespace = .indent_2 }, ctx.out);
        try ctx.out.writeAll("\n");
        return;
    }
    if (ctx.quiet) return;
    // The summary is the result of the command: stdout, unless the GIF went there.
    const w = if (out.stdout) ctx.err else ctx.out;
    const s = if (out.stdout) ctx.se else ctx.so;
    var b: [8][64]u8 = undefined;
    try w.writeAll("  ");
    try s.paint(w, .ok, s.glyph("✓", "ok"));
    try w.writeAll(" ");
    try s.paint(w, .bold, out.name());
    try s.print(w, .dim, "  {d}{s}{d} animated GIF · {s}\n\n", .{ st.gif_width, s.glyph("×", "x"), st.gif_height, term.fmtBytes(&b[0], st.gif_bytes) });
    try term.writeFields(w, s, 2, &.{
        .{ "scene", std.fmt.bufPrint(&b[1], "{d}x{d}, {d} frames, {d} thread{s}", .{ st.width, st.height, st.frames, st.threads, if (st.threads == 1) "" else "s" }) catch "" },
        .{ "per frame", std.fmt.bufPrint(&b[2], "{d:.2} ms mean · {d:.2} p50 · {d:.2} p95", .{ st.mean_ms, st.p50_ms, st.p95_ms }) catch "" },
        .{ "throughput", std.fmt.bufPrint(&b[3], "{d:.0} frames per second", .{st.fps}) catch "" },
        .{ "recall", std.fmt.bufPrint(&b[4], "{d:.1}% of visible objects tracked", .{100 * st.recall}) catch "" },
        .{ "id switches", std.fmt.bufPrint(&b[5], "{d}", .{st.id_switches}) catch "" },
        .{ "pipeline", track.pipeline_description },
    });
}
