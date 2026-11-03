//! What every command gets: allocators, I/O, the two output streams and
//! their styles, the global options, a lazily started thread pool, and the
//! one place messages and errors are worded.
//!
//! Streams: results (tables, JSON, image bytes for '-') go to stdout;
//! everything else (progress, status lines, warnings, errors) to stderr.

const std = @import("std");
const visar = @import("visar");
const Io = std.Io;
const args = @import("args.zig");
const term = @import("term.zig");

pub const Format = visar.codecs.Format;

/// Raised by commands after `Ctx.usage` (exit code 2) or `Ctx.fail` (exit
/// code 1) has filled in the message.
pub const Error = error{ Usage, Failed };

pub const Ctx = struct {
    gpa: std.mem.Allocator,
    /// Freed when the process exits; for small, long-lived things.
    arena: std.mem.Allocator,
    io: Io,
    env: *const std.process.Environ.Map,
    out: *Io.Writer,
    err: *Io.Writer,
    /// Styles of stdout and stderr (each decided from its own terminal).
    so: term.Style,
    se: term.Style,
    diag: *args.Diag,

    json: bool = false,
    quiet: bool = false,
    verbose: bool = false,
    force: bool = false,
    threads: ?usize = null,
    output: ?[]const u8 = null,
    format: ?Format = null,
    quality: u8 = 95,
    compression: u8 = 1,
    quality_set: bool = false,
    compression_set: bool = false,

    pool: ?visar.Pool = null,

    pub fn deinit(self: *Ctx) void {
        if (self.pool) |*p| p.deinit();
    }

    /// The executor for VISAR kernels; the pool starts on first use, so
    /// commands that never compute (help, inspect) spawn no threads.
    pub fn exec(self: *Ctx) visar.Exec {
        if (self.pool == null) self.pool = visar.Pool.init(self.gpa, self.io, self.threads);
        return visar.Exec.on(&self.pool.?);
    }

    /// A usage error (exit code 2).
    pub fn usage(self: *Ctx, comptime fmt: []const u8, a: anytype) error{Usage} {
        self.diag.setKeepHint(fmt, a);
        return error.Usage;
    }

    /// A runtime failure (exit code 1).
    pub fn fail(self: *Ctx, comptime fmt: []const u8, a: anytype) error{Failed} {
        self.diag.setKeepHint(fmt, a);
        return error.Failed;
    }

    /// A hint for the error about to be raised with `usage` or `fail` (or
    /// just raised: either order works).
    pub fn hint(self: *Ctx, comptime fmt: []const u8, a: anytype) void {
        self.diag.setHint(fmt, a);
    }

    /// A status message on stderr, unless --quiet.
    pub fn note(self: *Ctx, comptime fmt: []const u8, a: anytype) void {
        if (self.quiet) return;
        self.err.print(fmt ++ "\n", a) catch {};
    }

    /// A message shown only with --verbose.
    pub fn detail(self: *Ctx, comptime fmt: []const u8, a: anytype) void {
        if (!self.verbose or self.quiet) return;
        self.se.paint(self.err, .dim, "  · ") catch {};
        self.err.print(fmt ++ "\n", a) catch {};
    }

    /// A warning on stderr (shown even with --quiet: it may explain a result).
    pub fn warn(self: *Ctx, comptime fmt: []const u8, a: anytype) void {
        self.se.paint(self.err, .warn, "visar: warning: ") catch {};
        self.err.print(fmt ++ "\n", a) catch {};
    }

    /// A progress bar on stderr, drawn only where it can be redrawn and
    /// nobody asked for silence or machine output.
    pub fn progress(self: *Ctx, total: usize) term.Progress {
        return .init(self.err, self.se, self.io, total, !self.quiet and !self.json);
    }

    /// Writes the error held in `diag` (and its hint) to stderr:
    ///
    ///   visar: cannot read 'x.png': file not found
    ///     hint: check the path; 'visar inspect' lists what an image contains
    pub fn report(self: *Ctx) void {
        const w = self.err;
        const s = self.se;
        s.paint(w, .err, "visar:") catch {};
        w.print(" {s}\n", .{self.diag.msg}) catch {};
        if (self.diag.hint.len > 0) {
            w.writeAll("  ") catch {};
            s.paint(w, .iris, "hint:") catch {};
            w.print(" {s}\n", .{self.diag.hint}) catch {};
        }
        w.flush() catch {};
        self.diag.clear();
    }
};

/// A short, lower-case description of an error from the file system, a
/// codec or a kernel, for messages such as "cannot read 'x.png': <this>".
pub fn describe(err: anyerror) []const u8 {
    return switch (err) {
        error.FileNotFound => "file not found",
        error.AccessDenied, error.PermissionDenied => "permission denied",
        error.IsDir => "it is a directory",
        error.NotDir => "a parent of it is not a directory",
        error.NameTooLong => "the path is too long",
        error.NoSpaceLeft => "no space left on the device",
        error.ReadOnlyFileSystem => "the file system is read-only",
        error.FileTooBig, error.StreamTooLong => "the file is too large",
        error.InvalidData => "not a supported image, or corrupt",
        error.Truncated => "the file is truncated",
        error.TooLarge => "the image has too many pixels",
        error.Unsupported => "uses a feature VISAR does not support yet",
        error.UnsupportedChannels => "this channel count is not supported here",
        error.InvalidSize => "invalid image size",
        error.SizeMismatch => "image sizes do not match",
        error.InvalidKernel => "invalid kernel size",
        error.OutOfMemory => "out of memory",
        error.BrokenPipe => "the reader closed the pipe",
        error.InputOutput => "an I/O error occurred",
        error.SystemResources => "the system is out of resources",
        else => @errorName(err),
    };
}

/// "gray", "gray+alpha", "RGB", "RGBA" or "N channels".
pub fn channelName(buf: []u8, channels: u16) []const u8 {
    return switch (channels) {
        1 => "gray",
        2 => "gray+alpha",
        3 => "RGB",
        4 => "RGBA",
        else => std.fmt.bufPrint(buf, "{d} channels", .{channels}) catch "?",
    };
}

/// "8-bit", "16-bit", "32-bit float", ...
pub fn depthName(d: visar.Depth) []const u8 {
    return switch (d) {
        .u8 => "8-bit",
        .i8 => "8-bit signed",
        .u16 => "16-bit",
        .i16 => "16-bit signed",
        .i32 => "32-bit signed",
        .f16 => "16-bit float",
        .f32 => "32-bit float",
        .f64 => "64-bit float",
    };
}

/// Upper-case display name of a format.
pub fn formatName(f: Format) []const u8 {
    return switch (f) {
        .png => "PNG",
        .jpeg => "JPEG",
        .qoi => "QOI",
        .pnm => "PNM",
        .bmp => "BMP",
        .gif => "GIF",
    };
}

test "error descriptions are short phrases" {
    try std.testing.expectEqualStrings("file not found", describe(error.FileNotFound));
    try std.testing.expectEqualStrings("SomethingNew", describe(error.SomethingNew));
}
