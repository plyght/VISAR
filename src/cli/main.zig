//! The `visar` command-line tool.
//!
//!   main.zig      start-up, global options, dispatch, exit codes
//!   commands.zig  the command and option tables (parsing and help)
//!   help.zig      help pages        banner.zig  the VISAR banner
//!   args.zig      argument parser   term.zig    colour, tables, progress
//!   ctx.zig       shared context    imageio.zig reading and writing images
//!   ops.zig       image operations  cmd_*.zig   the commands
//!   cmd_track.zig live tracking (camera, window, threads)
//!
//! Exit codes: 0 success, 1 runtime failure, 2 usage error.

const std = @import("std");
const builtin = @import("builtin");
const visar = @import("visar");
const Io = std.Io;
const args = @import("args.zig");
const term = @import("term.zig");
const banner = @import("banner.zig");
const help = @import("help.zig");
const commands = @import("commands.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const cmd_info = @import("cmd_info.zig");

pub fn main(init: std.process.Init) u8 {
    const io = init.io;
    var out_buf: [64 * 1024]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    var err_buf: [4096]u8 = undefined;
    var ew = Io.File.Writer.initStreaming(.stderr(), io, &err_buf);
    const arena = init.arena.allocator();
    const env = init.environ_map;

    const raw = init.minimal.args.toSlice(arena) catch return 1;
    const argv: []const []const u8 = if (raw.len > 0) @ptrCast(raw[1..]) else &.{};
    const head = beforeDashDash(argv);
    const no_color = hasArg(head, "--no-color");
    const color_env = term.ColorEnv.read(env);

    var diag: args.Diag = .{};
    var ctx: Ctx = .{
        .gpa = init.gpa,
        .arena = arena,
        .io = io,
        .env = env,
        .out = &ow.interface,
        .err = &ew.interface,
        .so = term.styleFor(io, .stdout(), env, color_env, no_color),
        .se = term.styleFor(io, .stderr(), env, color_env, no_color),
        .diag = &diag,
    };
    defer ctx.deinit();

    const code: u8 = run(&ctx, argv) catch |e| code: {
        ctx.out.flush() catch {};
        switch (e) {
            error.Usage => {
                ctx.report();
                break :code 2;
            },
            error.Failed => ctx.report(),
            error.AlreadyReported => {},
            // stdout was closed early (`visar info | head -1`): nothing to say.
            error.WriteFailed => {},
            error.OutOfMemory => {
                diag.set("out of memory", .{});
                diag.setHint("try a smaller image, or fewer threads with --threads", .{});
                ctx.report();
            },
            else => {
                diag.set("unexpected error: {s}", .{@errorName(e)});
                diag.setHint("this is a bug; please report it at {s}/issues with the command you ran", .{commands.repo_url});
                ctx.report();
            },
        }
        break :code 1;
    };
    ctx.out.flush() catch {};
    ctx.err.flush() catch {};
    return code;
}

/// The arguments before a `--`.
fn beforeDashDash(argv: []const []const u8) []const []const u8 {
    for (argv, 0..) |a, i| if (std.mem.eql(u8, a, "--")) return argv[0..i];
    return argv;
}

fn hasArg(argv: []const []const u8, name: []const u8) bool {
    for (argv) |a| if (std.mem.eql(u8, a, name)) return true;
    return false;
}

/// The index of the command: the first argument that is neither an option
/// nor the value of a global option.
pub fn findCommand(argv: []const []const u8) ?usize {
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const a = argv[i];
        if (std.mem.eql(u8, a, "--")) return null;
        if (!args.isOption(a)) return i;
        if (std.mem.startsWith(u8, a, "--")) {
            if (std.mem.indexOfScalar(u8, a, '=') != null) continue;
            const o = args.findLong(&.{ &commands.global_opts, &commands.output_opts }, a[2..]) orelse continue;
            if (o.takesValue()) i += 1;
        } else if (a.len == 2) {
            const o = args.findShort(&.{ &commands.global_opts, &commands.output_opts }, a[1]) orelse continue;
            if (o.takesValue()) i += 1;
        }
    }
    return null;
}

pub fn versionLine(w: *Io.Writer) !void {
    try w.print("visar {s} for {s}, {s}, Zig {s}\n", .{ cmd_info.version_string, visar.core.target_description, cmd_info.modeName(), builtin.zig_version_string });
}

fn writeBanner(ctx: *Ctx) !void {
    // Decoration for people at a terminal only; never in a pipe.
    if (!ctx.so.tty) return;
    try banner.write(ctx.out, ctx.so, .{ .version = cmd_info.version_string, .tagline = commands.tagline, .target = visar.core.target_description });
    try ctx.out.writeAll("\n");
}

fn run(ctx: *Ctx, argv: []const []const u8) !u8 {
    const head = beforeDashDash(argv);
    const want_help = hasArg(head, "-h") or hasArg(head, "--help");
    const w = ctx.out;

    if (hasArg(head, "--version")) {
        if (hasArg(head, "--json")) {
            try std.json.Stringify.value(.{
                .version = cmd_info.version_string,
                .target = visar.core.target_description,
                .build_mode = cmd_info.modeName(),
                .zig = builtin.zig_version_string,
            }, .{ .whitespace = .indent_2 }, w);
            try w.writeAll("\n");
        } else try versionLine(w);
        return 0;
    }

    const ci = findCommand(argv) orelse {
        // `visar` alone: a short introduction. `visar --help`: everything.
        for (head) |a| if (args.isOption(a) and !std.mem.eql(u8, a, "-h") and !std.mem.eql(u8, a, "--help")) {
            const known = if (std.mem.startsWith(u8, a, "--"))
                args.findLong(&.{&commands.global_opts}, a[2 .. std.mem.indexOfScalar(u8, a, '=') orelse a.len]) != null
            else
                a.len == 2 and args.findShort(&.{&commands.global_opts}, a[1]) != null;
            if (!known) {
                ctx.hint("options go with a command, e.g. 'visar resize in.png out.png {s} ...'; see 'visar --help'", .{a});
                return ctx.usage("'{s}' needs a command", .{a});
            }
        };
        try writeBanner(ctx);
        if (want_help) try help.writeFull(w, ctx.so) else try help.writeConcise(w, ctx.so);
        return 0;
    };
    const name = argv[ci];

    if (std.mem.eql(u8, name, "help")) {
        const topic = for (argv[ci + 1 ..]) |a| {
            if (!args.isOption(a)) break a;
        } else null;
        const t = topic orelse {
            try writeBanner(ctx);
            try help.writeFull(w, ctx.so);
            return 0;
        };
        const c = commands.find(t) orelse return unknownCommand(ctx, t);
        try help.writeCommand(w, ctx.so, c);
        return 0;
    }

    const cmd = commands.find(name) orelse return unknownCommand(ctx, name);
    if (want_help) {
        try help.writeCommand(w, ctx.so, cmd);
        return 0;
    }

    // Everything but the command name, against the global options, the
    // output options (for commands that write) and the command's own.
    var rest: std.ArrayList([]const u8) = .empty;
    try rest.appendSlice(ctx.arena, argv[0..ci]);
    try rest.appendSlice(ctx.arena, argv[ci + 1 ..]);
    var ctxbuf: [48]u8 = undefined;
    const context = std.fmt.bufPrint(&ctxbuf, " for 'visar {s}'", .{name}) catch "";
    const sets: []const []const args.Opt = if (cmd.writes)
        &.{ &commands.global_opts, &commands.output_opts, cmd.opts }
    else
        &.{ &commands.global_opts, cmd.opts };
    var p = args.parse(ctx.arena, rest.items, sets, ctx.diag, .{
        .split_rest = std.mem.eql(u8, name, "pipeline"),
        .context = context,
    }) catch |e| {
        if (e == error.Usage and !cmd.writes) {
            // A known output option on a command that writes nothing.
            for (rest.items) |a| if (args.isOption(a) and std.mem.startsWith(u8, a, "--")) {
                const n = a[2 .. std.mem.indexOfScalar(u8, a, '=') orelse a.len];
                if (args.findLong(&.{&commands.output_opts}, n) != null and args.findLong(&.{ &commands.global_opts, cmd.opts }, n) == null) {
                    ctx.diag.set("'--{s}' does not apply to 'visar {s}'", .{ n, name });
                    ctx.diag.setHint("output options are for commands that write an image; see 'visar help {s}'", .{name});
                    break;
                }
            };
        }
        return e;
    };
    try applyGlobals(ctx, &p);
    try cmd.run(ctx, &p);
    return 0;
}

fn unknownCommand(ctx: *Ctx, name: []const u8) error{Usage} {
    if (args.suggest(name, &commands.command_names)) |s| {
        ctx.hint("did you mean 'visar {s}'? 'visar help' lists every command", .{s});
    } else if (std.mem.indexOfScalar(u8, name, '.') != null) {
        ctx.hint("to look at an image: 'visar inspect {s}'", .{name});
    } else ctx.hint("'visar help' lists every command", .{});
    return ctx.usage("unknown command '{s}'", .{name});
}

fn applyGlobals(ctx: *Ctx, p: *const args.Parsed) !void {
    const d = ctx.diag;
    ctx.json = p.flag("json");
    ctx.quiet = p.flag("quiet");
    ctx.verbose = p.flag("verbose") and !ctx.quiet;
    ctx.force = p.flag("force");
    ctx.output = p.value("output");
    if (p.value("threads")) |v| {
        ctx.threads = try parseThreads(d, "--threads", v);
    } else if (ctx.env.get("VISAR_THREADS")) |v| if (v.len > 0) {
        ctx.threads = try parseThreads(d, "VISAR_THREADS", v);
    };
    if (p.value("format")) |v| ctx.format = try parseFormat(d, v);
    if (p.value("quality")) |v| {
        ctx.quality = try args.int(u8, d, "--quality", v, 1, 100);
        ctx.quality_set = true;
    }
    if (p.value("compression")) |v| {
        ctx.compression = try args.int(u8, d, "--compression", v, 0, 9);
        ctx.compression_set = true;
    }
}

pub fn parseThreads(d: *args.Diag, what: []const u8, v: []const u8) error{Usage}!?usize {
    if (std.mem.eql(u8, v, "auto")) return null;
    return args.int(usize, d, what, v, 1, 1024) catch |e| {
        d.setHint("use a count from 1 to 1024, or 'auto' for one per CPU", .{});
        return e;
    };
}

pub fn parseFormat(d: *args.Diag, v: []const u8) error{Usage}!visar.codecs.Format {
    var buf: [8]u8 = undefined;
    if (v.len <= buf.len) {
        const l = std.ascii.lowerString(&buf, v);
        if (visar.codecs.formatFromExtension(l)) |f| return f;
        if (std.mem.eql(u8, l, "jpeg")) return .jpeg;
    }
    return args.choice(visar.codecs.Format, d, "format", v);
}

test {
    _ = args;
    _ = term;
    _ = banner;
    _ = help;
    _ = commands;
    _ = cx;
    _ = @import("ops.zig");
    _ = @import("imageio.zig");
    _ = @import("cmd_image.zig");
    _ = @import("cmd_bench.zig");
    _ = @import("cmd_demo.zig");
    _ = @import("cmd_track.zig");
    _ = cmd_info;
}

test "the command is found after global options and their values" {
    try std.testing.expectEqual(@as(?usize, 2), findCommand(&.{ "--threads", "4", "info" }));
    try std.testing.expectEqual(@as(?usize, 1), findCommand(&.{ "-q", "resize", "a.png", "b.png" }));
    try std.testing.expectEqual(@as(?usize, 1), findCommand(&.{ "--threads=4", "bench" }));
    try std.testing.expectEqual(@as(?usize, 2), findCommand(&.{ "-o", "out.png", "gray", "in.png" }));
    try std.testing.expectEqual(@as(?usize, null), findCommand(&.{ "--json", "--", "info" }));
    try std.testing.expectEqual(@as(?usize, null), findCommand(&.{}));
}

test "threads and formats" {
    var d: args.Diag = .{};
    try std.testing.expectEqual(@as(?usize, null), try parseThreads(&d, "--threads", "auto"));
    try std.testing.expectEqual(@as(?usize, 3), try parseThreads(&d, "--threads", "3"));
    try std.testing.expectError(error.Usage, parseThreads(&d, "--threads", "0"));
    try std.testing.expectEqual(visar.codecs.Format.jpeg, try parseFormat(&d, "JPG"));
    try std.testing.expectEqual(visar.codecs.Format.jpeg, try parseFormat(&d, "jpeg"));
    try std.testing.expectEqual(visar.codecs.Format.pnm, try parseFormat(&d, "ppm"));
    try std.testing.expectError(error.Usage, parseFormat(&d, "tiff"));
}
