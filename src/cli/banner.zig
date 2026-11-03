//! The VISAR banner: the wordmark as a detection, framed by viewfinder
//! corners with an amber "visar 0.99" label, the way a detector draws a box
//! around what it found. The letters run from cyan to blue. A terminal that
//! is narrow or cannot show block characters gets a one-line form instead.
//! It is only ever written to a terminal, never into machine output.
//!
//!   ┌╴ visar 0.99              ╶┐
//!      █   █ ▀█▀ ▄▀▀▀ ▄▀▀▄ █▀▀▄     visar 0.1.0
//!       █ █   █   ▀▀▄ █▀▀█ █▀█      computer vision in pure Zig
//!        ▀   ▀▀▀ ▀▀▀  ▀  ▀ ▀  ▀     x86_64-linux (avx2)
//!   └╴                         ╶┘

const std = @import("std");
const Io = std.Io;
const term = @import("term.zig");

/// Rows of each letter (V I S A R), drawn with half blocks.
const letters = [5][3][]const u8{
    .{ "█   █", " █ █ ", "  ▀  " },
    .{ "▀█▀", " █ ", "▀▀▀" },
    .{ "▄▀▀▀", " ▀▀▄", "▀▀▀ " },
    .{ "▄▀▀▄", "█▀▀█", "▀  ▀" },
    .{ "█▀▀▄", "█▀█ ", "▀  ▀" },
};

/// Cyan to blue, one shade per letter.
const shades = [5][]const u8{ "\x1b[38;5;51m", "\x1b[38;5;45m", "\x1b[38;5;44m", "\x1b[38;5;39m", "\x1b[38;5;33m" };
const amber = "\x1b[38;5;214m";
const tag_on = "\x1b[1;38;5;16;48;5;214m";
const dim = "\x1b[38;5;245m";
const bold = "\x1b[1m";
const reset = "\x1b[0m";

const word_width = 5 + 1 + 3 + 1 + 4 + 1 + 4 + 1 + 4;
const tag = " visar 0.99 ";
/// Width of the framed art: corner, three columns of margin, the word,
/// three columns of margin, corner.
pub const art_width = 3 + word_width + 3;
/// The art plus the information column beside it.
pub const full_width = art_width + 4 + 34;

pub const Info = struct {
    version: []const u8,
    tagline: []const u8,
    target: []const u8,
};

/// Writes the banner. `s` decides colour, block characters and width.
pub fn write(w: *Io.Writer, s: term.Style, info: Info) Io.Writer.Error!void {
    if (!s.unicode or s.columns < full_width) return writeCompact(w, s.color, info);
    const c = s.color;
    // Top edge: the corner, the label, the other corner.
    try w.writeAll("  ");
    try esc(w, c, amber);
    try w.writeAll("┌╴");
    try esc(w, c, tag_on);
    try w.writeAll(tag);
    try esc(w, c, reset);
    try esc(w, c, amber);
    try w.splatByteAll(' ', art_width - 2 - tag.len - 2);
    try w.writeAll("╶┐");
    try esc(w, c, reset);
    try w.writeAll("\n");

    const side = [3][]const u8{ info.version, info.tagline, info.target };
    for (0..3) |row| {
        try w.writeAll("     ");
        for (letters, 0..) |l, i| {
            if (i > 0) try w.writeAll(" ");
            try esc(w, c, shades[i]);
            try w.writeAll(l[row]);
        }
        try esc(w, c, reset);
        try w.writeAll("       ");
        try esc(w, c, if (row == 0) bold else dim);
        if (row == 0) try w.writeAll("visar ");
        try w.writeAll(side[row]);
        try esc(w, c, reset);
        try w.writeAll("\n");
    }

    try w.writeAll("  ");
    try esc(w, c, amber);
    try w.writeAll("└╴");
    try w.splatByteAll(' ', art_width - 4);
    try w.writeAll("╶┘");
    try esc(w, c, reset);
    try w.writeAll("\n");
}

/// One line: the label, the version and the tagline.
fn writeCompact(w: *Io.Writer, color: bool, info: Info) Io.Writer.Error!void {
    try w.writeAll("  ");
    if (color) {
        try w.print("{s} visar {s} {s}{s}{s}  {s}{s}\n", .{ tag_on, reset, bold, info.version, reset ++ dim, info.tagline, reset });
    } else {
        try w.print("[visar] {s}  {s}\n", .{ info.version, info.tagline });
    }
}

fn esc(w: *Io.Writer, color: bool, code: []const u8) Io.Writer.Error!void {
    if (color) try w.writeAll(code);
}

const test_info: Info = .{ .version = "0.1.0", .tagline = "computer vision in pure Zig", .target = "x86_64-linux (avx2)" };

test "every letter row has the letter's width" {
    for (letters) |l| {
        const wdt = try std.unicode.utf8CountCodepoints(l[0]);
        for (l) |row| try std.testing.expectEqual(wdt, try std.unicode.utf8CountCodepoints(row));
    }
}

test "the plain banner has no escape codes and fits its width" {
    var buf: [4096]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try write(&w, .{ .unicode = true, .columns = 100 }, test_info);
    const out = w.buffered();
    try std.testing.expect(std.mem.indexOfScalar(u8, out, 0x1b) == null);
    var rows = std.mem.splitScalar(u8, out, '\n');
    var n: usize = 0;
    while (rows.next()) |row| : (n += 1) try std.testing.expect(try std.unicode.utf8CountCodepoints(row) <= full_width + 2);
    try std.testing.expectEqual(@as(usize, 6), n);
}

test "a narrow or ASCII terminal gets one line" {
    var buf: [512]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try write(&w, .{ .unicode = true, .columns = 40 }, test_info);
    try std.testing.expectEqualStrings("  [visar] 0.1.0  computer vision in pure Zig\n", w.buffered());
    w = .fixed(&buf);
    try write(&w, .{ .unicode = false, .columns = 200 }, test_info);
    try std.testing.expect(std.mem.count(u8, w.buffered(), "\n") == 1);
}
