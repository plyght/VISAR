//! Rendering the help: the short form for a bare `visar`, the full form for
//! `visar --help`, and one page per command for `visar help <command>` and
//! `visar <command> --help`. Examples come first, as clig.dev suggests.

const std = @import("std");
const visar = @import("visar");
const Io = std.Io;
const term = @import("term.zig");
const args = @import("args.zig");
const commands = @import("commands.zig");
const Command = commands.Command;
const Style = term.Style;

const opt_column = 26;

fn heading(w: *Io.Writer, s: Style, title: []const u8) !void {
    try s.paint(w, .heading, title);
    try w.writeAll("\n");
}

fn example(w: *Io.Writer, s: Style, e: commands.Example) !void {
    try w.writeAll("  ");
    try s.paint(w, .dim, "$ ");
    try w.writeAll(e.cmd);
    try w.writeAll("\n");
    if (e.note.len > 0) {
        try w.writeAll("    ");
        try s.paint(w, .dim, e.note);
        try w.writeAll("\n");
    }
}

fn usageLine(w: *Io.Writer, s: Style, line: []const u8) !void {
    try w.writeAll("  ");
    try s.paint(w, .command, "visar");
    try w.print(" {s}\n", .{line});
}

pub fn writeOpts(w: *Io.Writer, s: Style, opts: []const args.Opt) !void {
    for (opts) |o| {
        var buf: [64]u8 = undefined;
        const short = if (o.short != 0) std.fmt.bufPrint(&buf, "-{c}, ", .{o.short}) catch "" else "    ";
        var lbuf: [96]u8 = undefined;
        const label = if (o.value) |v|
            std.fmt.bufPrint(&lbuf, "{s}--{s} {s}", .{ short, o.long, v }) catch ""
        else
            std.fmt.bufPrint(&lbuf, "{s}--{s}", .{ short, o.long }) catch "";
        try w.writeAll("  ");
        try s.paint(w, .command, label);
        if (label.len + 2 >= opt_column) {
            try w.writeAll("\n");
            try w.splatByteAll(' ', opt_column);
        } else try w.splatByteAll(' ', opt_column - 2 - label.len);
        try w.print("{s}\n", .{o.help});
    }
}

/// The help for a bare `visar`: what it is, how to start, where to look.
pub fn writeConcise(w: *Io.Writer, s: Style) !void {
    try heading(w, s, "Usage");
    try usageLine(w, s, "<command> [options] <args>");
    try w.writeAll("\n");
    try heading(w, s, "Examples");
    try example(w, s, .{ .cmd = "visar inspect photo.jpg", .note = "format, size, channels and depth" });
    try example(w, s, .{ .cmd = "visar resize photo.jpg thumb.jpg 320x", .note = "320 pixels wide, aspect ratio kept" });
    try example(w, s, .{ .cmd = "visar pipeline scan.jpg mask.png -- blur , threshold --otsu", .note = "several operations in one pass" });
    try w.writeAll("\n");
    try heading(w, s, "Commands");
    try writeCommandList(w, s, true);
    try w.writeAll("\n");
    try w.writeAll("Run ");
    try s.paint(w, .command, "visar help <command>");
    try w.writeAll(" for a command's options, or ");
    try s.paint(w, .command, "visar --help");
    try w.writeAll(" for everything.\n");
}

fn writeCommandList(w: *Io.Writer, s: Style, compact: bool) !void {
    if (compact) {
        for (std.enums.values(commands.Group)) |g| {
            try w.writeAll("  ");
            var b: [32]u8 = undefined;
            const title = std.fmt.bufPrint(&b, "{s}:", .{g.title()}) catch "";
            try s.paint(w, .dim, title);
            try w.splatByteAll(' ', 24 - @min(24, title.len));
            var first = true;
            for (commands.list) |c| if (c.group == g) {
                if (!first) try w.writeAll(" ");
                first = false;
                try s.paint(w, .command, c.name);
            };
            try w.writeAll("\n");
        }
        return;
    }
    for (std.enums.values(commands.Group)) |g| {
        try w.writeAll("  ");
        try s.paint(w, .dim, g.title());
        try w.writeAll("\n");
        for (commands.list) |c| if (c.group == g) {
            try w.writeAll("    ");
            try s.paint(w, .command, c.name);
            try w.splatByteAll(' ', 12 - @min(12, c.name.len));
            try w.print("{s}\n", .{c.summary});
        };
    }
}

/// `visar --help`.
pub fn writeFull(w: *Io.Writer, s: Style) !void {
    try w.print("{s}\n\n", .{commands.about});
    try heading(w, s, "Usage");
    try usageLine(w, s, "<command> [options] <args>");
    try usageLine(w, s, "help <command>");
    try w.writeAll("\n");
    try heading(w, s, "Examples");
    try example(w, s, .{ .cmd = "visar info", .note = "what this build supports" });
    try example(w, s, .{ .cmd = "visar inspect *.jpg", .note = "a table of images" });
    try example(w, s, .{ .cmd = "visar convert scan.ppm scan.png", .note = "re-encode" });
    try example(w, s, .{ .cmd = "visar resize photo.jpg half.jpg 50% --quality 85", .note = "half size, smaller file" });
    try example(w, s, .{ .cmd = "cat in.png | visar gray - - --format png > gray.png", .note = "stdin to stdout" });
    try example(w, s, .{ .cmd = "visar pipeline scan.jpg mask.png -- gray , blur , threshold , morph open", .note = "several steps, one decode" });
    try w.writeAll("\n");
    try heading(w, s, "Commands");
    try writeCommandList(w, s, false);
    try w.writeAll("\n");
    try heading(w, s, "Output options (commands that write an image)");
    try writeOpts(w, s, &commands.output_opts);
    try w.writeAll("\n");
    try heading(w, s, "Global options");
    try writeOpts(w, s, &commands.global_opts);
    try w.writeAll("\n");
    try heading(w, s, "Files");
    try w.writeAll(
        \\  '-' reads an image from stdin or writes one to stdout (with --format).
        \\  Input formats are detected from the bytes: PNG, JPEG, QOI, BMP, PNM.
        \\  Output formats follow the extension: .png .jpg .qoi .bmp .ppm/.pgm/.pam .gif
        \\  An existing output is never replaced without -f; writes are atomic.
        \\
    );
    try w.writeAll("\n");
    try heading(w, s, "Environment");
    try w.writeAll(
        \\  NO_COLOR, VISAR_NO_COLOR  no colour         FORCE_COLOR  colour even when piped
        \\  VISAR_THREADS             default --threads COLUMNS      output width
        \\
    );
    try w.writeAll("\n");
    try heading(w, s, "Exit codes");
    try w.writeAll("  0 success   1 failure (unreadable file, codec error...)   2 usage error\n");
    try w.writeAll("\n");
    try w.writeAll("Docs and issues: ");
    try s.paint(w, .iris, commands.repo_url);
    try w.writeAll("\n");
}

/// `visar help <command>` and `visar <command> --help`.
pub fn writeCommand(w: *Io.Writer, s: Style, c: *const Command) !void {
    try s.paint(w, .bold, "visar ");
    try s.paint(w, .heading, c.name);
    try w.print(": {s}\n\n", .{c.summary});
    try heading(w, s, "Usage");
    for (c.usage) |u| try usageLine(w, s, u);
    try w.writeAll("\n");
    if (c.about.len > 0) {
        try writeWrapped(w, c.about, @min(s.columns, 88), 0);
        try w.writeAll("\n");
    }
    try heading(w, s, "Examples");
    for (c.examples) |e| try example(w, s, e);
    if (c.opts.len > 0) {
        try w.writeAll("\n");
        try heading(w, s, "Options");
        try writeOpts(w, s, c.opts);
    }
    if (c.writes) {
        try w.writeAll("\n");
        try heading(w, s, "Output options");
        try writeOpts(w, s, &commands.output_opts);
    }
    try w.writeAll("\n");
    try s.paint(w, .dim, "Global options (--threads, --json, -q, -v, --no-color) work here too; see 'visar --help'.\n");
}

/// Writes `text` as a paragraph wrapped at `width` columns.
pub fn writeWrapped(w: *Io.Writer, text: []const u8, width: usize, indent: usize) !void {
    var col: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ' ');
    while (it.next()) |word| {
        if (col > 0 and col + 1 + word.len > width) {
            try w.writeAll("\n");
            col = 0;
        }
        if (col == 0) {
            try w.splatByteAll(' ', indent);
            col = indent;
        } else {
            try w.writeAll(" ");
            col += 1;
        }
        try w.writeAll(word);
        col += word.len;
    }
    if (col > 0) try w.writeAll("\n");
}

test "help pages render without colour codes on a plain stream" {
    var buf: [32 * 1024]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeFull(&w, .plain);
    try writeConcise(&w, .plain);
    for (&commands.list) |*c| try writeCommand(&w, .plain, c);
    try std.testing.expect(std.mem.indexOfScalar(u8, w.buffered(), 0x1b) == null);
}

test "every help line fits in 100 columns" {
    var buf: [32 * 1024]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeFull(&w, .plain);
    for (&commands.list) |*c| try writeCommand(&w, .plain, c);
    var lines = std.mem.splitScalar(u8, w.buffered(), '\n');
    while (lines.next()) |l| {
        if (term.visibleWidth(l) > 100) {
            std.debug.print("too wide: {s}\n", .{l});
            return error.TestUnexpectedResult;
        }
    }
}

test "paragraphs wrap at the width" {
    var buf: [256]u8 = undefined;
    var w: Io.Writer = .fixed(&buf);
    try writeWrapped(&w, "aaa bbb ccc ddd", 7, 0);
    try std.testing.expectEqualStrings("aaa bbb\nccc ddd\n", w.buffered());
}
