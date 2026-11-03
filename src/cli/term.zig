//! Terminal styling for the `visar` CLI: when to use colour, the VISAR
//! palette, the terminal's size and character set, a progress bar and an
//! aligned table. Nothing here writes escape codes unless `Style.color` is
//! set, which `detectColor` only does for a terminal.

const std = @import("std");
const builtin = @import("builtin");
const Io = std.Io;

/// The VISAR palette, from the 256-colour cube so that terminals without
/// 24-bit colour show it too. "Iris" cyan is the brand colour; amber is the
/// colour of a detection box and marks values worth a look.
pub const Role = enum {
    heading,
    command,
    iris,
    violet,
    amber,
    ok,
    warn,
    err,
    dim,
    bold,

    fn code(r: Role) []const u8 {
        return switch (r) {
            .heading => "\x1b[1;38;5;44m",
            .command => "\x1b[38;5;44m",
            .iris => "\x1b[38;5;44m",
            .violet => "\x1b[38;5;141m",
            .amber => "\x1b[38;5;214m",
            .ok => "\x1b[38;5;114m",
            .warn => "\x1b[38;5;221m",
            .err => "\x1b[1;38;5;203m",
            .dim => "\x1b[38;5;245m",
            .bold => "\x1b[1m",
        };
    }
};

const reset = "\x1b[0m";

/// How one output stream (stdout or stderr) is styled.
pub const Style = struct {
    color: bool = false,
    /// Box-drawing and block characters are safe to print.
    unicode: bool = false,
    /// The stream is a terminal (progress lines may be redrawn in place).
    tty: bool = false,
    /// The terminal's width, or a default when it is not a terminal.
    columns: u16 = 80,

    pub const plain: Style = .{};

    pub fn on(s: Style, w: *Io.Writer, r: Role) Io.Writer.Error!void {
        if (s.color) try w.writeAll(r.code());
    }

    pub fn off(s: Style, w: *Io.Writer) Io.Writer.Error!void {
        if (s.color) try w.writeAll(reset);
    }

    /// Writes `text` in role `r`.
    pub fn paint(s: Style, w: *Io.Writer, r: Role, text: []const u8) Io.Writer.Error!void {
        try s.on(w, r);
        try w.writeAll(text);
        try s.off(w);
    }

    /// Prints a formatted string in role `r`.
    pub fn print(s: Style, w: *Io.Writer, r: Role, comptime fmt: []const u8, args: anytype) Io.Writer.Error!void {
        try s.on(w, r);
        try w.print(fmt, args);
        try s.off(w);
    }

    /// A glyph with an ASCII fallback.
    pub fn glyph(s: Style, uni: []const u8, ascii: []const u8) []const u8 {
        return if (s.unicode) uni else ascii;
    }
};

/// What the environment says about colour, decided once.
pub const ColorEnv = struct {
    /// NO_COLOR or VISAR_NO_COLOR is set (and non-empty), or TERM=dumb.
    no_color: bool,
    /// FORCE_COLOR is set (and non-empty, not "0").
    force: bool,

    pub fn read(env: *const std.process.Environ.Map) ColorEnv {
        const force = if (env.get("FORCE_COLOR")) |v| v.len > 0 and !std.mem.eql(u8, v, "0") else false;
        const dumb = if (env.get("TERM")) |t| std.mem.eql(u8, t, "dumb") else false;
        return .{ .no_color = set(env, "NO_COLOR") or set(env, "VISAR_NO_COLOR") or dumb, .force = force };
    }

    fn set(env: *const std.process.Environ.Map, name: []const u8) bool {
        const v = env.get(name) orelse return false;
        return v.len > 0;
    }
};

/// Colour for one stream, following clig.dev: never when --no-color, NO_COLOR
/// or TERM=dumb; always with FORCE_COLOR; otherwise only on a terminal.
pub fn detectColor(tty: bool, env: ColorEnv, no_color_flag: bool) bool {
    if (no_color_flag or env.no_color) return false;
    if (env.force) return true;
    return tty;
}

/// Whether the terminal can be expected to show box-drawing and block
/// characters: a UTF-8 locale, or none named, and not the Linux console.
pub fn unicodeTerminal(env: *const std.process.Environ.Map) bool {
    if (env.get("TERM")) |t| if (std.mem.eql(u8, t, "linux") or std.mem.eql(u8, t, "dumb")) return false;
    if (builtin.os.tag == .windows or builtin.os.tag == .macos) return true;
    for ([_][]const u8{ "LC_ALL", "LC_CTYPE", "LANG" }) |name| {
        const v = env.get(name) orelse continue;
        if (v.len == 0) continue;
        return containsIgnoreCase(v, "utf-8") or containsIgnoreCase(v, "utf8");
    }
    return true;
}

pub fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    for (0..haystack.len - needle.len + 1) |i| {
        if (std.ascii.eqlIgnoreCase(haystack[i..][0..needle.len], needle)) return true;
    }
    return false;
}

/// The width of the terminal `file` is attached to, or null when it is not a
/// terminal or cannot be asked.
pub fn terminalColumns(io: Io, file: Io.File) ?u16 {
    if (builtin.os.tag == .windows) {
        var info = std.os.windows.CONSOLE.USER_IO.GET_SCREEN_BUFFER_INFO;
        const status = info.operate(io, file) catch return null;
        if (status != .SUCCESS) return null;
        const cols = info.Data.dwWindowSize.X;
        return if (cols > 0) @intCast(cols) else null;
    } else if (@hasDecl(std.posix.T, "IOCGWINSZ")) {
        var ws: std.posix.winsize = .{ .row = 0, .col = 0, .xpixel = 0, .ypixel = 0 };
        const result = io.operate(.{ .device_io_control = .{
            .file = file,
            .code = std.posix.T.IOCGWINSZ,
            .arg = &ws,
        } }) catch return null;
        if (result.device_io_control < 0 or ws.col == 0) return null;
        return ws.col;
    } else return null;
}

/// Builds the style of a stream.
pub fn styleFor(io: Io, file: Io.File, env: *const std.process.Environ.Map, color_env: ColorEnv, no_color_flag: bool) Style {
    const tty = file.isTty(io) catch false;
    var columns: u16 = if (tty) terminalColumns(io, file) orelse 80 else 80;
    if (env.get("COLUMNS")) |c| columns = std.fmt.parseInt(u16, c, 10) catch columns;
    return .{
        .color = detectColor(tty, color_env, no_color_flag),
        .unicode = unicodeTerminal(env),
        .tty = tty,
        .columns = @max(columns, 20),
    };
}

// ---------------------------------------------------------------------------
// Number formatting
// ---------------------------------------------------------------------------

/// "1.2 MiB", "830 B".
pub fn fmtBytes(buf: []u8, bytes: u64) []const u8 {
    const b: f64 = @floatFromInt(bytes);
    if (bytes >= 1 << 30) return std.fmt.bufPrint(buf, "{d:.2} GiB", .{b / (1 << 30)}) catch "?";
    if (bytes >= 1 << 20) return std.fmt.bufPrint(buf, "{d:.1} MiB", .{b / (1 << 20)}) catch "?";
    if (bytes >= 1 << 10) return std.fmt.bufPrint(buf, "{d:.1} KiB", .{b / (1 << 10)}) catch "?";
    return std.fmt.bufPrint(buf, "{d} B", .{bytes}) catch "?";
}

/// "850 µs", "12.4 ms", "3.21 s".
pub fn fmtMs(buf: []u8, ms: f64, unicode: bool) []const u8 {
    if (ms < 1.0) return std.fmt.bufPrint(buf, "{d:.0} {s}", .{ ms * 1000, if (unicode) "µs" else "us" }) catch "?";
    if (ms < 100.0) return std.fmt.bufPrint(buf, "{d:.2} ms", .{ms}) catch "?";
    if (ms < 1000.0) return std.fmt.bufPrint(buf, "{d:.1} ms", .{ms}) catch "?";
    return std.fmt.bufPrint(buf, "{d:.2} s", .{ms / 1000}) catch "?";
}

/// Milliseconds from `start` to now.
pub fn msSince(io: Io, start: Io.Timestamp) f64 {
    const ns = start.durationTo(Io.Timestamp.now(io, .awake)).nanoseconds;
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

/// Display width of `s`: UTF-8 code points, not counting ANSI escapes.
pub fn visibleWidth(s: []const u8) usize {
    var n: usize = 0;
    var i: usize = 0;
    while (i < s.len) {
        if (s[i] == 0x1b) {
            while (i < s.len and s[i] != 'm') i += 1;
            i += 1;
            continue;
        }
        if (s[i] & 0xC0 != 0x80) n += 1;
        i += 1;
    }
    return n;
}

fn pad(w: *Io.Writer, n: usize) Io.Writer.Error!void {
    try w.splatByteAll(' ', n);
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------

/// A one-line progress bar on stderr, redrawn in place:
///
///   ━━━━━━━━━━━━━━━━╸━━━━━━━━━━━  14/30  gaussian 5x5   2.1 s
///
/// It draws only on a terminal and never under --quiet or --json; elsewhere
/// it stays silent, so logs and pipes see only the final result.
pub const Progress = struct {
    w: *Io.Writer,
    style: Style,
    io: Io,
    total: usize,
    start: Io.Timestamp,
    enabled: bool,
    drawn: bool = false,

    pub fn init(w: *Io.Writer, style: Style, io: Io, total: usize, enabled: bool) Progress {
        return .{ .w = w, .style = style, .io = io, .total = total, .start = Io.Timestamp.now(io, .awake), .enabled = enabled and style.tty };
    }

    pub fn update(self: *Progress, done: usize, label: []const u8) void {
        if (!self.enabled) return;
        self.draw(done, label) catch {};
    }

    fn draw(self: *Progress, done: usize, label: []const u8) Io.Writer.Error!void {
        const w = self.w;
        const s = self.style;
        const bar_w: usize = if (s.columns >= 90) 30 else if (s.columns >= 60) 20 else 10;
        const total = @max(self.total, 1);
        // Eighths of the bar that are filled.
        const filled = @min(bar_w * 2, done * bar_w * 2 / total);
        try w.writeAll("\r  ");
        if (s.unicode) {
            try s.on(w, .iris);
            for (0..filled / 2) |_| try w.writeAll("━");
            if (filled % 2 == 1) try w.writeAll("╸");
            try s.off(w);
            try s.on(w, .dim);
            // Without colour the rest must differ in shape, not just shade.
            for ((filled + 1) / 2..bar_w) |_| try w.writeAll(if (s.color) "━" else "─");
            try s.off(w);
        } else {
            try w.writeAll("[");
            try w.splatByteAll('#', filled / 2);
            try w.splatByteAll('-', bar_w - filled / 2);
            try w.writeAll("]");
        }
        var nb: [32]u8 = undefined;
        const count = std.fmt.bufPrint(&nb, "{d}/{d}", .{ done, self.total }) catch "";
        try w.writeAll("  ");
        try s.paint(w, .bold, count);
        var tb: [32]u8 = undefined;
        const elapsed = fmtMs(&tb, msSince(self.io, self.start), s.unicode);
        // Fit the label into what is left of the line.
        const used = 2 + bar_w + 2 + count.len + 2 + 2 + elapsed.len + 1;
        const room = if (s.columns > used) s.columns - used else 0;
        const lbl = label[0..@min(label.len, room)];
        try w.writeAll("  ");
        try w.writeAll(lbl);
        try w.writeAll("  ");
        try s.paint(w, .dim, elapsed);
        try w.writeAll("\x1b[K");
        try w.flush();
        self.drawn = true;
    }

    /// Clears the line, leaving the cursor where the bar started.
    pub fn finish(self: *Progress) void {
        if (!self.drawn) return;
        self.w.writeAll("\r\x1b[K") catch {};
        self.w.flush() catch {};
        self.drawn = false;
    }
};

// ---------------------------------------------------------------------------
// Tables
// ---------------------------------------------------------------------------

pub const Align = enum { left, right };

pub const Column = struct {
    title: []const u8,
    alignment: Align = .left,
};

/// A table of preformatted cells, written with aligned columns, a styled
/// header and a thin rule under it, and no vertical borders:
///
///   KERNEL           IMAGE            MEDIAN    MPX/S
///   ───────────────  ───────────────  ────────  ──────
///   gray             1920x1080 rgb     0.41 ms  5057.6
pub const Table = struct {
    columns: []const Column,
    /// Row-major cells; `cells.len` is a multiple of `columns.len`.
    cells: std.ArrayList([]const u8) = .empty,
    /// The role of each row's first cell (e.g. a highlighted name).
    first_role: ?Role = .command,

    pub fn init(columns: []const Column) Table {
        return .{ .columns = columns };
    }

    pub fn deinit(self: *Table, a: std.mem.Allocator) void {
        self.cells.deinit(a);
    }

    pub fn addRow(self: *Table, a: std.mem.Allocator, row: []const []const u8) !void {
        std.debug.assert(row.len == self.columns.len);
        try self.cells.appendSlice(a, row);
    }

    pub fn write(self: *const Table, w: *Io.Writer, s: Style, indent: usize) Io.Writer.Error!void {
        const n = self.columns.len;
        var widths: [16]usize = @splat(0);
        std.debug.assert(n <= widths.len);
        for (self.columns, 0..) |c, i| widths[i] = visibleWidth(c.title);
        for (self.cells.items, 0..) |c, i| widths[i % n] = @max(widths[i % n], visibleWidth(c));

        try pad(w, indent);
        for (self.columns, 0..) |c, i| {
            if (i > 0) try w.writeAll("  ");
            try self.cell(w, s, c.title, widths[i], c.alignment, .heading, i + 1 == n);
        }
        try w.writeAll("\n");
        try pad(w, indent);
        try s.on(w, .dim);
        for (0..n) |i| {
            if (i > 0) try w.writeAll("  ");
            for (0..widths[i]) |_| try w.writeAll(s.glyph("─", "-"));
        }
        try s.off(w);
        try w.writeAll("\n");
        var r: usize = 0;
        while (r * n < self.cells.items.len) : (r += 1) {
            try pad(w, indent);
            for (0..n) |i| {
                if (i > 0) try w.writeAll("  ");
                const role: ?Role = if (i == 0) self.first_role else null;
                try self.cell(w, s, self.cells.items[r * n + i], widths[i], self.columns[i].alignment, role, i + 1 == n);
            }
            try w.writeAll("\n");
        }
    }

    fn cell(_: *const Table, w: *Io.Writer, s: Style, text: []const u8, width: usize, al: Align, role: ?Role, last: bool) Io.Writer.Error!void {
        const gap = width - @min(width, visibleWidth(text));
        if (al == .right) try pad(w, gap);
        if (role) |r| try s.paint(w, r, text) else try w.writeAll(text);
        if (al == .left and !last) try pad(w, gap);
    }
};

/// Writes "  key      value" lines with the keys aligned.
pub fn writeFields(w: *Io.Writer, s: Style, indent: usize, fields: []const [2][]const u8) Io.Writer.Error!void {
    var kw: usize = 0;
    for (fields) |f| kw = @max(kw, f[0].len);
    for (fields) |f| {
        try pad(w, indent);
        try s.paint(w, .dim, f[0]);
        try pad(w, kw - f[0].len + 2);
        try w.writeAll(f[1]);
        try w.writeAll("\n");
    }
}

test "colour follows clig.dev" {
    const none: ColorEnv = .{ .no_color = false, .force = false };
    try std.testing.expect(detectColor(true, none, false));
    try std.testing.expect(!detectColor(false, none, false));
    try std.testing.expect(!detectColor(true, none, true));
    try std.testing.expect(!detectColor(true, .{ .no_color = true, .force = false }, false));
    try std.testing.expect(detectColor(false, .{ .no_color = false, .force = true }, false));
    try std.testing.expect(!detectColor(false, .{ .no_color = false, .force = true }, true));
}

test "byte and duration formatting" {
    var b: [32]u8 = undefined;
    try std.testing.expectEqualStrings("830 B", fmtBytes(&b, 830));
    try std.testing.expectEqualStrings("1.5 KiB", fmtBytes(&b, 1536));
    try std.testing.expectEqualStrings("2.0 MiB", fmtBytes(&b, 2 << 20));
    try std.testing.expectEqualStrings("12.40 ms", fmtMs(&b, 12.4, true));
    try std.testing.expectEqualStrings("500 us", fmtMs(&b, 0.5, false));
    try std.testing.expectEqualStrings("1.50 s", fmtMs(&b, 1500, true));
}

test "a plain table has aligned columns and no escape codes" {
    const a = std.testing.allocator;
    var t = Table.init(&.{ .{ .title = "NAME" }, .{ .title = "MS", .alignment = .right } });
    defer t.deinit(a);
    try t.addRow(a, &.{ "gray", "0.41" });
    try t.addRow(a, &.{ "gaussian", "12.00" });
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try t.write(&w, .plain, 0);
    try std.testing.expectEqualStrings(
        \\NAME         MS
        \\--------  -----
        \\gray       0.41
        \\gaussian  12.00
        \\
    , w.buffered());
}

test "visible width skips escape codes and counts code points" {
    try std.testing.expectEqual(@as(usize, 3), visibleWidth("\x1b[1mabc\x1b[0m"));
    try std.testing.expectEqual(@as(usize, 2), visibleWidth("µs"));
}
