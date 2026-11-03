//! Argument parsing for the `visar` CLI.
//!
//! Options are declared as `Opt` tables (the same tables the help is
//! rendered from). The parser accepts `--name value`, `--name=value`, short
//! flags (`-q`, clusters such as `-qf`, `-o FILE`, `-oFILE`), `-` as an
//! ordinary argument (stdin or stdout), negative numbers as arguments, and
//! `--` to end option parsing. Options may come before or after the
//! arguments. Every error is reported through `Diag` with a suggestion when
//! one is close ("did you mean --width?").

const std = @import("std");

pub const Opt = struct {
    /// Long name without the dashes.
    long: []const u8,
    short: u8 = 0,
    /// Placeholder for the value (e.g. "N"); null for a flag.
    value: ?[]const u8 = null,
    help: []const u8,

    pub fn takesValue(o: Opt) bool {
        return o.value != null;
    }
};

/// A message for the user (and an optional hint), filled in where an error
/// is detected and printed once by `main`.
pub const Diag = struct {
    msg_buf: [512]u8 = undefined,
    hint_buf: [512]u8 = undefined,
    msg: []const u8 = "",
    hint: []const u8 = "",

    pub fn set(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        d.msg = std.fmt.bufPrint(&d.msg_buf, fmt, args) catch blk: {
            const end = d.msg_buf.len;
            @memcpy(d.msg_buf[end - 3 ..], "...");
            break :blk d.msg_buf[0..end];
        };
        d.hint = "";
    }

    /// Sets the message and keeps a hint given just before it.
    pub fn setKeepHint(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        const hint = d.hint;
        d.set(fmt, args);
        d.hint = hint;
    }

    pub fn clear(d: *Diag) void {
        d.msg = "";
        d.hint = "";
    }

    pub fn setHint(d: *Diag, comptime fmt: []const u8, args: anytype) void {
        d.hint = std.fmt.bufPrint(&d.hint_buf, fmt, args) catch d.hint_buf[0..];
    }
};

pub const Error = error{Usage} || std.mem.Allocator.Error;

pub const Hit = struct {
    opt: *const Opt,
    value: ?[]const u8,
};

pub const Parsed = struct {
    positionals: std.ArrayList([]const u8) = .empty,
    hits: std.ArrayList(Hit) = .empty,
    /// Everything after `--`, when `--` was given.
    rest: ?[]const []const u8 = null,

    pub fn flag(p: *const Parsed, long: []const u8) bool {
        for (p.hits.items) |h| if (std.mem.eql(u8, h.opt.long, long)) return true;
        return false;
    }

    /// The last value given for `long`, or null.
    pub fn value(p: *const Parsed, long: []const u8) ?[]const u8 {
        var i = p.hits.items.len;
        while (i > 0) {
            i -= 1;
            const h = p.hits.items[i];
            if (std.mem.eql(u8, h.opt.long, long)) return h.value;
        }
        return null;
    }

    /// How many times `long` was given (for flags such as -v).
    pub fn count(p: *const Parsed, long: []const u8) usize {
        var n: usize = 0;
        for (p.hits.items) |h| n += @intFromBool(std.mem.eql(u8, h.opt.long, long));
        return n;
    }
};

pub const ParseOptions = struct {
    /// Stop at `--` and keep what follows in `Parsed.rest`. Without this,
    /// `--` makes every later token an argument.
    split_rest: bool = false,
    /// Shown in "unknown option" messages, e.g. "for 'visar resize'".
    context: []const u8 = "",
};

/// Parses `argv` (without the program name and command) against the option
/// tables in `sets`.
pub fn parse(a: std.mem.Allocator, argv: []const []const u8, sets: []const []const Opt, diag: *Diag, po: ParseOptions) Error!Parsed {
    var p: Parsed = .{};
    var i: usize = 0;
    var only_args = false;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (only_args or !isOption(arg)) {
            try p.positionals.append(a, arg);
            continue;
        }
        if (std.mem.eql(u8, arg, "--")) {
            if (po.split_rest) {
                p.rest = argv[i + 1 ..];
                break;
            }
            only_args = true;
            continue;
        }
        if (std.mem.startsWith(u8, arg, "--")) {
            const body = arg[2..];
            const eq = std.mem.indexOfScalar(u8, body, '=');
            const name = if (eq) |e| body[0..e] else body;
            const opt = findLong(sets, name) orelse {
                diag.set("unknown option '--{s}'{s}", .{ name, po.context });
                if (suggestLong(sets, name)) |s| diag.setHint("did you mean '--{s}'?", .{s}) else diag.setHint("run 'visar help' to see the options", .{});
                return error.Usage;
            };
            if (!opt.takesValue()) {
                if (eq != null) {
                    diag.set("'--{s}' does not take a value", .{name});
                    return error.Usage;
                }
                try p.hits.append(a, .{ .opt = opt, .value = null });
                continue;
            }
            const v = if (eq) |e| body[e + 1 ..] else blk: {
                if (i + 1 >= argv.len) return missingValue(diag, opt);
                i += 1;
                break :blk argv[i];
            };
            try p.hits.append(a, .{ .opt = opt, .value = v });
            continue;
        }
        // A cluster of short options: -q, -qf, -o FILE, -oFILE.
        var j: usize = 1;
        while (j < arg.len) : (j += 1) {
            const opt = findShort(sets, arg[j]) orelse {
                diag.set("unknown option '-{c}'{s}", .{ arg[j], po.context });
                diag.setHint("run 'visar help' to see the options", .{});
                return error.Usage;
            };
            if (!opt.takesValue()) {
                try p.hits.append(a, .{ .opt = opt, .value = null });
                continue;
            }
            const v = if (j + 1 < arg.len) arg[j + 1 ..] else blk: {
                if (i + 1 >= argv.len) return missingValue(diag, opt);
                i += 1;
                break :blk argv[i];
            };
            try p.hits.append(a, .{ .opt = opt, .value = v });
            break;
        }
    }
    return p;
}

fn missingValue(diag: *Diag, opt: *const Opt) error{Usage} {
    diag.set("'--{s}' needs a value ({s})", .{ opt.long, opt.value.? });
    return error.Usage;
}

/// An option rather than an argument: starts with '-', but is not "-" (stdin
/// or stdout) and not a negative number.
pub fn isOption(arg: []const u8) bool {
    if (arg.len < 2 or arg[0] != '-') return false;
    if (std.ascii.isDigit(arg[1]) or arg[1] == '.') return false;
    return true;
}

pub fn findLong(sets: []const []const Opt, name: []const u8) ?*const Opt {
    for (sets) |set| for (set) |*o| if (std.mem.eql(u8, o.long, name)) return o;
    return null;
}

pub fn findShort(sets: []const []const Opt, c: u8) ?*const Opt {
    if (c == 0) return null;
    for (sets) |set| for (set) |*o| if (o.short == c) return o;
    return null;
}

fn suggestLong(sets: []const []const Opt, name: []const u8) ?[]const u8 {
    var best: ?[]const u8 = null;
    var best_d: usize = maxDistance(name);
    for (sets) |set| for (set) |o| {
        const d = editDistance(name, o.long);
        if (d <= best_d) {
            if (d < best_d or best == null) best = o.long;
            best_d = d;
        }
    };
    return best;
}

/// Edits tolerated before a name is considered unrelated.
fn maxDistance(name: []const u8) usize {
    return if (name.len <= 3) 1 else if (name.len <= 6) 2 else 3;
}

/// The closest of `names` to `name`, if any is close enough to suggest.
pub fn suggest(name: []const u8, names: []const []const u8) ?[]const u8 {
    var best: ?[]const u8 = null;
    var best_d: usize = maxDistance(name) + 1;
    for (names) |n| {
        // A prefix of a longer name ("thresh" for "threshold") is a good guess too.
        const d = if (name.len >= 3 and std.mem.startsWith(u8, n, name)) 1 else editDistance(name, n);
        if (d < best_d) {
            best = n;
            best_d = d;
        }
    }
    return best;
}

/// Damerau-Levenshtein distance (adjacent transpositions count as one edit),
/// for names up to 63 bytes; longer names compare as far apart.
pub fn editDistance(a: []const u8, b: []const u8) usize {
    if (a.len >= 64 or b.len >= 64) return 64;
    var d: [64][64]usize = undefined;
    for (0..a.len + 1) |i| d[i][0] = i;
    for (0..b.len + 1) |j| d[0][j] = j;
    for (1..a.len + 1) |i| {
        for (1..b.len + 1) |j| {
            const cost: usize = @intFromBool(std.ascii.toLower(a[i - 1]) != std.ascii.toLower(b[j - 1]));
            var v = @min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
            if (i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]) v = @min(v, d[i - 2][j - 2] + 1);
            d[i][j] = v;
        }
    }
    return d[a.len][b.len];
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

/// Parses an integer option value in [lo, hi].
pub fn int(comptime T: type, diag: *Diag, what: []const u8, s: []const u8, lo: T, hi: T) error{Usage}!T {
    const v = std.fmt.parseInt(T, s, 10) catch {
        diag.set("{s} must be a whole number, not '{s}'", .{ what, s });
        return error.Usage;
    };
    if (v < lo or v > hi) {
        diag.set("{s} must be between {d} and {d}, not {d}", .{ what, lo, hi, v });
        return error.Usage;
    }
    return v;
}

/// Parses a positive, finite number.
pub fn positive(diag: *Diag, what: []const u8, s: []const u8) error{Usage}!f64 {
    const v = std.fmt.parseFloat(f64, s) catch {
        diag.set("{s} must be a number, not '{s}'", .{ what, s });
        return error.Usage;
    };
    if (!std.math.isFinite(v) or v <= 0) {
        diag.set("{s} must be greater than 0, not '{s}'", .{ what, s });
        return error.Usage;
    }
    return v;
}

/// An enum value by name, with the choices listed on error.
pub fn choice(comptime E: type, diag: *Diag, what: []const u8, s: []const u8) error{Usage}!E {
    if (std.meta.stringToEnum(E, s)) |e| return e;
    const names = comptime blk: {
        var list: []const []const u8 = &.{};
        for (std.meta.fieldNames(E)) |n| list = list ++ .{n};
        break :blk list;
    };
    diag.set("unknown {s} '{s}'", .{ what, s });
    if (suggest(s, names)) |n| {
        diag.setHint("did you mean '{s}'? (choose from {s})", .{ n, comptime joined(names) });
    } else diag.setHint("choose from {s}", .{comptime joined(names)});
    return error.Usage;
}

fn joined(comptime names: []const []const u8) []const u8 {
    var s: []const u8 = "";
    for (names, 0..) |n, i| s = s ++ (if (i == 0) "" else ", ") ++ n;
    return s;
}

/// A size such as "640x480"; either side may be left out ("640x", "x480")
/// when `allow_partial`.
pub const Size = struct {
    w: ?u32,
    h: ?u32,
};

pub fn size(diag: *Diag, s: []const u8, allow_partial: bool) error{Usage}!Size {
    const x = std.mem.indexOfAny(u8, s, "xX") orelse {
        diag.set("'{s}' is not a size", .{s});
        diag.setHint("write sizes as WIDTHxHEIGHT, e.g. 640x480", .{});
        return error.Usage;
    };
    const ws = s[0..x];
    const hs = s[x + 1 ..];
    if ((!allow_partial and (ws.len == 0 or hs.len == 0)) or (ws.len == 0 and hs.len == 0)) {
        diag.set("'{s}' is not a size", .{s});
        diag.setHint("write sizes as WIDTHxHEIGHT, e.g. 640x480", .{});
        return error.Usage;
    }
    return .{
        .w = if (ws.len > 0) try int(u32, diag, "the width", ws, 1, max_side) else null,
        .h = if (hs.len > 0) try int(u32, diag, "the height", hs, 1, max_side) else null,
    };
}

/// The largest width or height the CLI accepts.
pub const max_side: u32 = 1 << 16;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const test_globals = [_]Opt{
    .{ .long = "json", .help = "" },
    .{ .long = "quiet", .short = 'q', .help = "" },
    .{ .long = "force", .short = 'f', .help = "" },
    .{ .long = "verbose", .short = 'v', .help = "" },
    .{ .long = "output", .short = 'o', .value = "FILE", .help = "" },
    .{ .long = "threads", .value = "N", .help = "" },
};
const test_resize = [_]Opt{
    .{ .long = "width", .value = "W", .help = "" },
    .{ .long = "height", .value = "H", .help = "" },
};

fn testParse(argv: []const []const u8, diag: *Diag) Error!Parsed {
    return parse(std.testing.allocator, argv, &.{ &test_globals, &test_resize }, diag, .{});
}

fn freeParsed(p: *Parsed) void {
    p.positionals.deinit(std.testing.allocator);
    p.hits.deinit(std.testing.allocator);
}

test "options anywhere, both value forms, clusters" {
    var d: Diag = .{};
    var p = try testParse(&.{ "in.png", "--width", "320", "-qf", "out.png", "--threads=4", "-oX.png", "-v", "-v" }, &d);
    defer freeParsed(&p);
    try std.testing.expectEqual(@as(usize, 2), p.positionals.items.len);
    try std.testing.expectEqualStrings("in.png", p.positionals.items[0]);
    try std.testing.expectEqualStrings("out.png", p.positionals.items[1]);
    try std.testing.expectEqualStrings("320", p.value("width").?);
    try std.testing.expectEqualStrings("4", p.value("threads").?);
    try std.testing.expectEqualStrings("X.png", p.value("output").?);
    try std.testing.expect(p.flag("quiet") and p.flag("force"));
    try std.testing.expectEqual(@as(usize, 2), p.count("verbose"));
    try std.testing.expect(!p.flag("json"));
}

test "dash, negative numbers and the -- terminator are arguments" {
    var d: Diag = .{};
    var p = try testParse(&.{ "-", "-90", "--", "--json" }, &d);
    defer freeParsed(&p);
    try std.testing.expectEqual(@as(usize, 3), p.positionals.items.len);
    try std.testing.expectEqualStrings("-", p.positionals.items[0]);
    try std.testing.expectEqualStrings("-90", p.positionals.items[1]);
    try std.testing.expectEqualStrings("--json", p.positionals.items[2]);
}

test "split_rest keeps what follows --" {
    var d: Diag = .{};
    var p = try parse(std.testing.allocator, &.{ "a", "b", "--", "gray", ",", "blur" }, &.{&test_globals}, &d, .{ .split_rest = true });
    defer freeParsed(&p);
    try std.testing.expectEqual(@as(usize, 2), p.positionals.items.len);
    try std.testing.expectEqual(@as(usize, 3), p.rest.?.len);
}

test "a mistyped option gets a suggestion" {
    var d: Diag = .{};
    try std.testing.expectError(error.Usage, testParse(&.{ "--widht", "3" }, &d));
    try std.testing.expectEqualStrings("unknown option '--widht'", d.msg);
    try std.testing.expectEqualStrings("did you mean '--width'?", d.hint);
    try std.testing.expectError(error.Usage, testParse(&.{"--width"}, &d));
    try std.testing.expectEqualStrings("'--width' needs a value (W)", d.msg);
    try std.testing.expectError(error.Usage, testParse(&.{"--json=1"}, &d));
    try std.testing.expectError(error.Usage, testParse(&.{"-x"}, &d));
}

test "command suggestions" {
    const cmds = [_][]const u8{ "info", "inspect", "convert", "resize", "threshold", "bench" };
    try std.testing.expectEqualStrings("resize", suggest("resze", &cmds).?);
    try std.testing.expectEqualStrings("bench", suggest("bnech", &cmds).?);
    try std.testing.expectEqualStrings("threshold", suggest("thresh", &cmds).?);
    try std.testing.expectEqualStrings("inspect", suggest("inpsect", &cmds).?);
    try std.testing.expectEqual(@as(?[]const u8, null), suggest("frobnicate", &cmds));
}

test "values" {
    var d: Diag = .{};
    try std.testing.expectEqual(@as(u8, 80), try int(u8, &d, "--quality", "80", 1, 100));
    try std.testing.expectError(error.Usage, int(u8, &d, "--quality", "101", 1, 100));
    try std.testing.expectEqualStrings("--quality must be between 1 and 100, not 101", d.msg);
    try std.testing.expectError(error.Usage, int(u8, &d, "--quality", "high", 1, 100));
    try std.testing.expectEqual(@as(f64, 1.5), try positive(&d, "--sigma", "1.5"));
    try std.testing.expectError(error.Usage, positive(&d, "--sigma", "-1"));
    const E = enum { nearest, linear, cubic };
    try std.testing.expectEqual(E.cubic, try choice(E, &d, "interpolation", "cubic"));
    try std.testing.expectError(error.Usage, choice(E, &d, "interpolation", "cubci"));
    try std.testing.expectEqualStrings("did you mean 'cubic'? (choose from nearest, linear, cubic)", d.hint);
}

test "sizes" {
    var d: Diag = .{};
    const s = try size(&d, "640x480", false);
    try std.testing.expectEqual(@as(?u32, 640), s.w);
    try std.testing.expectEqual(@as(?u32, 480), s.h);
    const p = try size(&d, "x200", true);
    try std.testing.expectEqual(@as(?u32, null), p.w);
    try std.testing.expectError(error.Usage, size(&d, "x200", false));
    try std.testing.expectError(error.Usage, size(&d, "640", false));
    try std.testing.expectError(error.Usage, size(&d, "0x10", false));
    try std.testing.expectError(error.Usage, size(&d, "x", true));
}
