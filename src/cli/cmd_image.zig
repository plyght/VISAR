//! The commands that read images: `inspect`, `convert`, the single
//! operations (`gray`, `blur`, `resize`, ...) and `pipeline`.

const std = @import("std");
const visar = @import("visar");
const Io = std.Io;
const args = @import("args.zig");
const term = @import("term.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const imageio = @import("imageio.zig");
const ops = @import("ops.zig");

// ---------------------------------------------------------------------------
// inspect
// ---------------------------------------------------------------------------

const Inspection = struct {
    path: []const u8,
    format: []const u8,
    width: u32,
    height: u32,
    channels: u16,
    color: []const u8,
    depth: []const u8,
    bits: usize,
    file_bytes: usize,
};

const Failure = struct { path: []const u8, @"error": []const u8 };

pub fn runInspect(ctx: *Ctx, p: *args.Parsed) !void {
    const paths = p.positionals.items;
    if (paths.len == 0) {
        ctx.hint("e.g. 'visar inspect photo.jpg', or '-' for stdin", .{});
        return ctx.usage("inspect needs at least one image", .{});
    }
    const a = ctx.arena;
    var found: std.ArrayList(Inspection) = .empty;
    var failed: std.ArrayList(Failure) = .empty;
    var stdin_used = false;
    for (paths) |path| {
        if (std.mem.eql(u8, path, "-")) {
            if (stdin_used) return ctx.usage("'-' (stdin) can only be read once", .{});
            stdin_used = true;
        }
        const t0 = Io.Timestamp.now(ctx.io, .awake);
        var in = imageio.load(ctx, path, .unchanged) catch |e| switch (e) {
            error.Failed => {
                // Report and go on with the others.
                if (!ctx.json) ctx.report();
                try failed.append(a, .{ .path = path, .@"error" = try a.dupe(u8, ctx.diag.msg) });
                ctx.diag.clear();
                continue;
            },
            else => return e,
        };
        defer in.deinit(ctx.gpa);
        ctx.detail("{s}: decoded in {d:.2} ms", .{ path, term.msSince(ctx.io, t0) });
        var cb: [32]u8 = undefined;
        try found.append(a, .{
            .path = imageio.displayName(path, "stdin"),
            .format = @tagName(in.format),
            .width = in.img.width(),
            .height = in.img.height(),
            .channels = in.img.channels(),
            .color = try a.dupe(u8, cx.channelName(&cb, in.img.channels())),
            .depth = @tagName(in.img.depth()),
            .bits = in.img.depth().size() * 8,
            .file_bytes = in.file_bytes,
        });
    }

    const w = ctx.out;
    if (ctx.json) {
        var js: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_2 } };
        try js.beginArray();
        for (found.items) |f| try js.write(f);
        for (failed.items) |f| try js.write(f);
        try js.endArray();
        try w.writeAll("\n");
    } else if (found.items.len == 1 and paths.len == 1) {
        const f = found.items[0];
        const s = ctx.so;
        try s.paint(w, .bold, f.path);
        try w.writeAll("\n");
        var b: [4][64]u8 = undefined;
        const mp = @as(f64, @floatFromInt(@as(u64, f.width) * f.height)) / 1e6;
        try term.writeFields(w, s, 2, &.{
            .{ "format", cx.formatName(std.meta.stringToEnum(imageio.Format, f.format).?) },
            .{ "size", std.fmt.bufPrint(&b[0], "{d}{s}{d}  ({d:.2} MP)", .{ f.width, s.glyph(" × ", " x "), f.height, mp }) catch "" },
            .{ "channels", std.fmt.bufPrint(&b[1], "{d} ({s})", .{ f.channels, f.color }) catch "" },
            .{ "depth", std.fmt.bufPrint(&b[2], "{d}-bit ({s})", .{ f.bits, f.depth }) catch "" },
            .{ "file", term.fmtBytes(&b[3], f.file_bytes) },
        });
    } else if (found.items.len > 0) {
        var t = term.Table.init(&.{
            .{ .title = "FILE" },                      .{ .title = "FORMAT" },
            .{ .title = "SIZE", .alignment = .right }, .{ .title = "CHANNELS" },
            .{ .title = "DEPTH" },                     .{ .title = "BYTES", .alignment = .right },
        });
        defer t.deinit(a);
        for (found.items) |f| {
            try t.addRow(a, &.{
                f.path,
                f.format,
                try std.fmt.allocPrint(a, "{d}x{d}", .{ f.width, f.height }),
                try std.fmt.allocPrint(a, "{d} {s}", .{ f.channels, f.color }),
                f.depth,
                term.fmtBytes(try a.alloc(u8, 16), f.file_bytes),
            });
        }
        try t.write(w, ctx.so, 0);
    }
    if (failed.items.len > 0) {
        if (paths.len > 1) {
            return ctx.fail("{d} of {d} images could not be read", .{ failed.items.len, paths.len });
        }
        // The one failure was reported above (or is in the JSON).
        if (ctx.json) return ctx.fail("{s}", .{failed.items[0].@"error"});
        return error.AlreadyReported;
    }
}

// ---------------------------------------------------------------------------
// convert, single operations, pipeline
// ---------------------------------------------------------------------------

pub fn runConvert(ctx: *Ctx, p: *args.Parsed) !void {
    const io_paths = try inOut(ctx, p, "convert", 0);
    try process(ctx, io_paths.in, io_paths.out, &.{});
}

pub fn runOp(ctx: *Ctx, p: *args.Parsed, kind: ops.Kind) !void {
    const takes_arg: usize = switch (kind) {
        .threshold, .flip, .rotate, .morph, .resize => 1,
        else => 0,
    };
    const io_paths = try inOut(ctx, p, @tagName(kind), takes_arg);
    const op = try ops.parse(kind, p, io_paths.rest, ctx.diag);
    try process(ctx, io_paths.in, io_paths.out, &.{op});
}

pub fn runPipeline(ctx: *Ctx, p: *args.Parsed) !void {
    const io_paths = try inOut(ctx, p, "pipeline", 0);
    const rest = p.rest orelse {
        ctx.hint("e.g. 'visar pipeline in.png out.png -- gray , blur --sigma 2'", .{});
        return ctx.usage("pipeline needs its steps after '--'", .{});
    };
    const steps = try parseSteps(ctx.arena, rest, ctx.diag);
    if (steps.len == 0) {
        ctx.hint("e.g. 'visar pipeline in.png out.png -- gray , blur --sigma 2'", .{});
        return ctx.usage("pipeline needs at least one step after '--'", .{});
    }
    try process(ctx, io_paths.in, io_paths.out, steps);
}

/// Splits "gray , blur --sigma 2, threshold" into steps and parses each.
pub fn parseSteps(a: std.mem.Allocator, rest: []const []const u8, d: *args.Diag) ![]ops.Op {
    // Tokens, with commas split off ("blur," and ",blur" are fine too).
    var toks: std.ArrayList([]const u8) = .empty;
    for (rest) |t| {
        var it = std.mem.splitScalar(u8, t, ',');
        var first = true;
        while (it.next()) |piece| {
            if (!first) try toks.append(a, ",");
            first = false;
            if (piece.len > 0) try toks.append(a, piece);
        }
    }
    var steps: std.ArrayList(ops.Op) = .empty;
    var i: usize = 0;
    var n: usize = 0;
    while (i < toks.items.len) {
        var j = i;
        while (j < toks.items.len and !std.mem.eql(u8, toks.items[j], ",")) j += 1;
        const seg = toks.items[i..j];
        i = j + 1;
        if (seg.len == 0) continue;
        n += 1;
        const kind = std.meta.stringToEnum(ops.Kind, seg[0]) orelse {
            d.set("unknown pipeline step '{s}' (step {d})", .{ seg[0], n });
            if (args.suggest(seg[0], ops.names)) |s|
                d.setHint("did you mean '{s}'? Steps: gray, blur, threshold, flip, rotate, edges, equalize, morph, resize", .{s})
            else
                d.setHint("steps: gray, blur, threshold, flip, rotate, edges, equalize, morph, resize", .{});
            return error.Usage;
        };
        var ctxbuf: [64]u8 = undefined;
        const context = std.fmt.bufPrint(&ctxbuf, " in pipeline step {d} ({s})", .{ n, seg[0] }) catch "";
        var p = try args.parse(a, seg[1..], &.{ops.optsFor(kind)}, d, .{ .context = context });
        try steps.append(a, try ops.parse(kind, &p, p.positionals.items, d));
    }
    return steps.items;
}

const InOut = struct { in: []const u8, out: []const u8, rest: []const []const u8 };

/// The input, the output (second argument or -o) and the command's own
/// arguments after them.
fn inOut(ctx: *Ctx, p: *args.Parsed, name: []const u8, own_args: usize) !InOut {
    const pos = p.positionals.items;
    if (pos.len == 0) {
        ctx.hint("e.g. 'visar {s} in.png out.png'; '-' reads stdin", .{name});
        return ctx.usage("{s} needs an input image", .{name});
    }
    if (ctx.output) |o| {
        try tooMany(ctx, p, pos[1..], own_args, name);
        return .{ .in = pos[0], .out = o, .rest = pos[1..] };
    }
    if (pos.len < 2 or (own_args > 0 and pos.len == 2 and looksLikeArgument(pos[1]))) {
        ctx.hint("add an output file, e.g. 'visar {s} {s} out.png{s}{s}', or '-' with --format for stdout", .{ name, pos[0], if (pos.len == 2) " " else "", if (pos.len == 2) pos[1] else "" });
        return ctx.usage("{s} needs an output file", .{name});
    }
    try tooMany(ctx, p, pos[2..], own_args, name);
    return .{ .in = pos[0], .out = pos[1], .rest = pos[2..] };
}

fn tooMany(ctx: *Ctx, p: *args.Parsed, rest: []const []const u8, own_args: usize, name: []const u8) !void {
    if (rest.len <= own_args) return;
    const extra = rest[own_args];
    // `-q 80` reads as --quiet and a stray 80: say what was probably meant.
    if (p.flag("quiet") and std.fmt.parseInt(u8, extra, 10) != error.InvalidCharacter) {
        ctx.hint("-q is --quiet; set the JPEG quality with --quality {s}", .{extra});
    } else ctx.hint("'visar help {s}' shows what it takes", .{name});
    return ctx.usage("unexpected argument '{s}'", .{extra});
}

/// An argument for the operation rather than a file name ("90", "50%", "open").
fn looksLikeArgument(s: []const u8) bool {
    if (s.len == 0 or std.mem.eql(u8, s, "-")) return false;
    if (std.mem.indexOfScalar(u8, s, '.') != null or std.mem.indexOfScalar(u8, s, '/') != null) return false;
    return true;
}

const StepReport = struct {
    op: []const u8,
    ms: f64,
    threshold: ?f64 = null,
    interp: ?[]const u8 = null,
};

const Summary = struct {
    input: []const u8,
    output: []const u8,
    format: []const u8,
    width: u32,
    height: u32,
    channels: u16,
    depth: []const u8,
    bytes: usize,
    steps: []const StepReport,
    ms: f64,
};

/// Reads `in_path`, applies `steps` and writes `out_path`.
fn process(ctx: *Ctx, in_path: []const u8, out_path: []const u8, steps: []const ops.Op) !void {
    const io = ctx.io;
    const t_start = Io.Timestamp.now(io, .awake);
    // Validate the output before doing any work.
    const out = try imageio.prepare(ctx, out_path);
    var needs8 = false;
    for (steps) |s| needs8 = needs8 or ops.needs8bit(s);

    var t = Io.Timestamp.now(io, .awake);
    var in = try imageio.load(ctx, in_path, if (needs8) .any_8bit else .unchanged);
    defer in.deinit(ctx.gpa);
    var cb: [32]u8 = undefined;
    ctx.detail("read {s}: {s} {d}x{d} {s} {s} in {d:.2} ms", .{ imageio.displayName(in_path, "stdin"), cx.formatName(in.format), in.img.width(), in.img.height(), cx.channelName(&cb, in.img.channels()), cx.depthName(in.img.depth()), term.msSince(io, t) });

    const env: ops.Env = .{ .exec = ctx.exec(), .gpa = ctx.gpa };
    var reports = try ctx.arena.alloc(StepReport, steps.len);
    var notes: std.ArrayList(u8) = .empty;
    for (steps, 0..) |op, i| {
        t = Io.Timestamp.now(io, .awake);
        var r: ops.Report = .{};
        var db: [64]u8 = undefined;
        const desc = ops.describe(&db, op);
        ops.apply(env, op, &in.img, &r) catch |e| {
            var cb2: [32]u8 = undefined;
            if (e == error.UnsupportedChannels or e == error.Unsupported)
                ctx.hint("convert it first, e.g. 'visar gray' for a 1-channel image", .{});
            return ctx.fail("cannot {s} a {s} {s} image: {s}", .{ desc, cx.depthName(in.img.depth()), cx.channelName(&cb2, in.img.channels()), cx.describe(e) });
        };
        const ms = term.msSince(io, t);
        reports[i] = .{ .op = try ctx.arena.dupe(u8, desc), .ms = ms, .threshold = r.threshold, .interp = if (r.interp) |ip| @tagName(ip) else null };
        ctx.detail("{s}: {d:.2} ms{s}", .{ desc, ms, if (r.interp != null) "" else "" });
        if (r.threshold) |th| try notes.print(ctx.arena, " · threshold {d}", .{th});
        if (r.interp) |ip| if (op.resize.interp == .auto) try notes.print(ctx.arena, " · {s}", .{@tagName(ip)});
    }

    t = Io.Timestamp.now(io, .awake);
    const saved = try imageio.save(ctx, out, in.img);
    ctx.detail("wrote {s}: {s} in {d:.2} ms", .{ out.name(), cx.formatName(out.format), term.msSince(io, t) });
    const total = term.msSince(io, t_start);

    const img = in.img;
    if (ctx.json) {
        const summary: Summary = .{
            .input = in_path,
            .output = out_path,
            .format = @tagName(out.format),
            .width = img.width(),
            .height = img.height(),
            .channels = img.channels(),
            .depth = @tagName(img.depth()),
            .bytes = saved.bytes,
            .steps = reports,
            .ms = total,
        };
        try std.json.Stringify.value(summary, .{ .whitespace = .indent_2, .emit_null_optional_fields = false }, ctx.out);
        try ctx.out.writeAll("\n");
        return;
    }
    if (ctx.quiet) return;
    // One status line on stderr: what was written, how big, how long.
    const s = ctx.se;
    const w = ctx.err;
    var bb: [32]u8 = undefined;
    var mb: [32]u8 = undefined;
    try w.writeAll("  ");
    try s.paint(w, .ok, s.glyph("✓", "ok"));
    try w.writeAll(" ");
    try s.paint(w, .bold, out.name());
    try s.on(w, .dim);
    try w.print("  {d}{s}{d} {s} {s} · {s} {s} · {s}{s}", .{
        img.width(),
        s.glyph("×", "x"),
        img.height(),
        cx.channelName(&cb, img.channels()),
        cx.depthName(img.depth()),
        cx.formatName(out.format),
        term.fmtBytes(&bb, saved.bytes),
        term.fmtMs(&mb, total, s.unicode),
        notes.items,
    });
    try s.off(w);
    try w.writeAll("\n");
}

test "pipeline steps split on commas however they are spaced" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var d: args.Diag = .{};
    const steps = try parseSteps(arena.allocator(), &.{ "gray,", "blur", "--sigma", "1.5", ",threshold", ",", "morph", "open", "--size", "5" }, &d);
    try std.testing.expectEqual(@as(usize, 4), steps.len);
    try std.testing.expectEqual(ops.Kind.gray, std.meta.activeTag(steps[0]));
    try std.testing.expectEqual(@as(f64, 1.5), steps[1].blur.sigma);
    try std.testing.expectEqual(ops.Kind.threshold, std.meta.activeTag(steps[2]));
    try std.testing.expectEqual(@as(u32, 5), steps[3].morph.size);
}

test "pipeline step errors name the step" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    var d: args.Diag = .{};
    try std.testing.expectError(error.Usage, parseSteps(arena.allocator(), &.{ "gray", ",", "blurr" }, &d));
    try std.testing.expectEqualStrings("unknown pipeline step 'blurr' (step 2)", d.msg);
    try std.testing.expect(std.mem.startsWith(u8, d.hint, "did you mean 'blur'?"));
    try std.testing.expectError(error.Usage, parseSteps(arena.allocator(), &.{ "blur", "--sigmma", "2" }, &d));
    try std.testing.expectEqualStrings("unknown option '--sigmma' in pipeline step 1 (blur)", d.msg);
}

test "an operation argument is not mistaken for an output file" {
    try std.testing.expect(looksLikeArgument("90"));
    try std.testing.expect(looksLikeArgument("open"));
    try std.testing.expect(!looksLikeArgument("out.png"));
    try std.testing.expect(!looksLikeArgument("-"));
}
