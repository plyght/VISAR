//! Reading and writing images for the CLI.
//!
//! Inputs are files or '-' (stdin); the format is detected from the bytes.
//! Outputs are files or '-' (stdout); the format comes from the extension or
//! --format. An output is checked before any work is done (`prepare`), an
//! existing file is never replaced without -f/--force, and files are written
//! atomically (a temporary file renamed into place), so an interrupted or
//! failed run never leaves a half-written image behind.

const std = @import("std");
const visar = @import("visar");
const Io = std.Io;
const codecs = visar.codecs;
const AnyImage = visar.AnyImage;
const Ctx = @import("ctx.zig").Ctx;
const cx = @import("ctx.zig");
const args = @import("args.zig");

pub const Format = codecs.Format;

/// Formats VISAR can decode (GIF is write-only for now).
pub const readable = [_]Format{ .png, .jpeg, .qoi, .bmp, .pnm };
pub const read_list = "PNG, JPEG, QOI, BMP and PNM (PBM/PGM/PPM/PAM)";
pub const ext_list = ".png, .jpg, .qoi, .bmp, .pgm/.ppm/.pam or .gif";

/// The name to show for a path: '-' reads as "stdin" or "stdout".
pub fn displayName(path: []const u8, comptime stream: []const u8) []const u8 {
    return if (std.mem.eql(u8, path, "-")) "<" ++ stream ++ ">" else path;
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

pub const Input = struct {
    img: AnyImage,
    format: Format,
    file_bytes: usize,

    pub fn deinit(self: *Input, gpa: std.mem.Allocator) void {
        self.img.deinit(gpa);
    }
};

/// The whole file (or stdin). The caller frees the result.
pub fn readAll(ctx: *Ctx, path: []const u8) ![]u8 {
    const io = ctx.io;
    if (std.mem.eql(u8, path, "-")) {
        if (Io.File.stdin().isTty(io) catch false) {
            ctx.hint("pipe an image in, e.g. 'cat photo.png | visar inspect -'", .{});
            return ctx.usage("'-' reads an image from stdin, but stdin is a terminal", .{});
        }
        var buf: [64 * 1024]u8 = undefined;
        var r = Io.File.stdin().readerStreaming(io, &buf);
        return r.interface.allocRemaining(ctx.gpa, .unlimited) catch |e| switch (e) {
            error.ReadFailed => ctx.fail("cannot read stdin: {s}", .{cx.describe(r.err orelse error.InputOutput)}),
            else => ctx.fail("cannot read stdin: {s}", .{cx.describe(e)}),
        };
    }
    return Io.Dir.cwd().readFileAlloc(io, path, ctx.gpa, .unlimited) catch |e| {
        switch (e) {
            error.FileNotFound => {
                var nb: [256]u8 = undefined;
                if (similarFile(ctx, path, &nb)) |s| {
                    ctx.diag.set("cannot read '{s}': file not found", .{path});
                    ctx.hint("did you mean '{s}'?", .{s});
                    return error.Failed;
                }
                ctx.diag.set("cannot read '{s}': file not found", .{path});
                ctx.hint("check the path (it is relative to {s})", .{"the current directory"});
                return error.Failed;
            },
            error.IsDir => {
                ctx.diag.set("cannot read '{s}': it is a directory", .{path});
                ctx.hint("pass an image file; 'visar inspect {s}/*' looks at every image in it", .{std.mem.trimEnd(u8, path, "/")});
                return error.Failed;
            },
            else => return ctx.fail("cannot read '{s}': {s}", .{ path, cx.describe(e) }),
        }
    };
}

/// A file next to `path` whose name is close to its name (a typo), if any.
fn similarFile(ctx: *Ctx, path: []const u8, out: []u8) ?[]const u8 {
    const io = ctx.io;
    const dir_path = std.fs.path.dirname(path) orelse ".";
    const base = std.fs.path.basename(path);
    var dir = Io.Dir.cwd().openDir(io, dir_path, .{ .iterate = true }) catch return null;
    defer dir.close(io);
    var it = dir.iterate();
    var best_d: usize = if (base.len <= 4) 1 else 2;
    var found = false;
    var len: usize = 0;
    var seen: usize = 0;
    while (it.next(io) catch null) |entry| : (seen += 1) {
        if (seen > 4096) break;
        if (entry.kind == .directory) continue;
        const d = args.editDistance(base, entry.name);
        if (d <= best_d and entry.name.len <= out.len) {
            if (found and d == best_d) continue;
            best_d = d;
            found = true;
            const joined = if (std.fs.path.dirname(path) != null)
                std.fmt.bufPrint(out, "{s}/{s}", .{ dir_path, entry.name }) catch continue
            else
                std.fmt.bufPrint(out, "{s}", .{entry.name}) catch continue;
            len = joined.len;
        }
    }
    return if (found) out[0..len] else null;
}

/// Reads and decodes an image. `mode` = .any_8bit for commands whose kernels
/// work on 8-bit samples.
pub fn load(ctx: *Ctx, path: []const u8, mode: codecs.Mode) !Input {
    const bytes = try readAll(ctx, path);
    defer ctx.gpa.free(bytes);
    return decodeBytes(ctx, path, bytes, mode);
}

pub fn decodeBytes(ctx: *Ctx, path: []const u8, bytes: []const u8, mode: codecs.Mode) !Input {
    const name = displayName(path, "stdin");
    const format = codecs.detect(bytes) orelse {
        if (bytes.len == 0) return ctx.fail("cannot read '{s}': the file is empty", .{name});
        ctx.diag.set("cannot read '{s}': not an image format VISAR knows", .{name});
        if (foreignFormat(bytes)) |f| {
            ctx.hint("{s} is not supported yet; VISAR reads {s}", .{ f, read_list });
        } else ctx.hint("VISAR reads {s}", .{read_list});
        return error.Failed;
    };
    if (format == .gif) {
        ctx.hint("VISAR writes GIFs but cannot read them yet; convert it to PNG first", .{});
        return ctx.fail("cannot read '{s}': GIF decoding is not supported yet", .{name});
    }
    const img = codecs.decode(ctx.gpa, bytes, .{ .mode = mode }) catch |e| {
        if (e == error.Unsupported and format == .jpeg)
            ctx.hint("baseline and progressive 8-bit JPEGs are supported; lossless, arithmetic-coded and 12-bit ones are not yet", .{});
        return ctx.fail("cannot decode '{s}' as {s}: {s}", .{ name, cx.formatName(format), cx.describe(e) });
    };
    return .{ .img = img, .format = format, .file_bytes = bytes.len };
}

/// Names a common image format VISAR does not read, from its magic bytes.
fn foreignFormat(b: []const u8) ?[]const u8 {
    if (b.len >= 4 and (std.mem.eql(u8, b[0..4], "II*\x00") or std.mem.eql(u8, b[0..4], "MM\x00*"))) return "TIFF";
    if (b.len >= 12 and std.mem.eql(u8, b[0..4], "RIFF") and std.mem.eql(u8, b[8..12], "WEBP")) return "WebP";
    if (b.len >= 12 and std.mem.eql(u8, b[4..8], "ftyp")) {
        const brand = b[8..12];
        if (std.mem.eql(u8, brand, "avif") or std.mem.eql(u8, brand, "avis")) return "AVIF";
        if (std.mem.eql(u8, brand, "heic") or std.mem.eql(u8, brand, "heix") or std.mem.eql(u8, brand, "mif1")) return "HEIF";
    }
    if (b.len >= 4 and std.mem.eql(u8, b[0..4], "8BPS")) return "Photoshop (PSD)";
    if (b.len >= 4 and std.mem.eql(u8, b[0..4], "\x76\x2f\x31\x01")) return "OpenEXR";
    if (b.len >= 2 and std.mem.eql(u8, b[0..2], "\xff\x0a")) return "JPEG XL";
    if (b.len >= 4 and std.mem.eql(u8, b[0..4], "%PDF")) return "PDF";
    return null;
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

pub const Output = struct {
    path: []const u8,
    format: Format,
    stdout: bool = false,
    /// An existing device or pipe (e.g. /dev/null): written in place.
    special: bool = false,

    pub fn name(o: Output) []const u8 {
        return displayName(o.path, "stdout");
    }
};

/// Validates an output before any work is done: its format, that it does
/// not replace a file without --force, and that binary data does not go to a
/// terminal.
pub fn prepare(ctx: *Ctx, path: []const u8) !Output {
    if (std.mem.eql(u8, path, "-")) {
        const f = ctx.format orelse {
            ctx.hint("name the format of the bytes, e.g. '--format png'", .{});
            return ctx.usage("writing to stdout ('-') needs --format", .{});
        };
        if (ctx.json) {
            ctx.hint("write the image to a file, or drop --json", .{});
            return ctx.usage("--json and an image on stdout ('-') cannot share stdout", .{});
        }
        if (Io.File.stdout().isTty(ctx.io) catch false) {
            ctx.hint("redirect it ('> out.{s}') or pipe it into another program", .{extension(f)});
            return ctx.fail("refusing to write binary image data to a terminal", .{});
        }
        return .{ .path = path, .format = f, .stdout = true };
    }
    const ext_format = codecs.formatFromExtension(std.fs.path.basename(path));
    if (ctx.format) |f| if (ext_format) |e| if (e != f) {
        ctx.hint("drop --format, or name the file '.{s}'", .{extension(f)});
        return ctx.usage("'--format {s}' does not match the extension of '{s}'", .{ @tagName(f), path });
    };
    const format = ctx.format orelse ext_format orelse {
        const ext = std.fs.path.extension(path);
        if (ext.len > 0) {
            ctx.hint("VISAR writes {s}; or pass --format", .{ext_list});
            return ctx.usage("cannot write '{s}': '{s}' is not a format VISAR writes", .{ path, ext });
        }
        ctx.hint("add an extension ({s}) or pass --format", .{ext_list});
        return ctx.usage("cannot tell the output format of '{s}'", .{path});
    };
    var out: Output = .{ .path = path, .format = format };
    if (ctx.quality_set and format != .jpeg) ctx.warn("--quality only affects JPEG; ignored for {s}", .{cx.formatName(format)});
    if (ctx.compression_set and format != .png) ctx.warn("--compression only affects PNG; ignored for {s}", .{cx.formatName(format)});
    const st = Io.Dir.cwd().statFile(ctx.io, path, .{}) catch |e| switch (e) {
        error.FileNotFound => {
            // The file is new; its directory must exist.
            if (std.fs.path.dirname(path)) |dir| {
                const ds = Io.Dir.cwd().statFile(ctx.io, dir, .{}) catch return writeFailed(ctx, path, error.FileNotFound);
                if (ds.kind != .directory) return ctx.fail("cannot write '{s}': '{s}' is not a directory", .{ path, dir });
            }
            return out;
        },
        else => return out, // reported, if it matters, when writing
    };
    switch (st.kind) {
        .directory => {
            ctx.hint("give a file name, e.g. '{s}/out.{s}'", .{ std.mem.trimEnd(u8, path, "/"), extension(format) });
            return ctx.fail("cannot write '{s}': it is a directory", .{path});
        },
        .file => if (!ctx.force) {
            ctx.hint("pass -f/--force to replace it, or choose another name", .{});
            return ctx.fail("'{s}' already exists", .{path});
        },
        else => out.special = true,
    }
    return out;
}

/// The usual extension of a format.
pub fn extension(f: Format) []const u8 {
    return switch (f) {
        .png => "png",
        .jpeg => "jpg",
        .qoi => "qoi",
        .pnm => "ppm",
        .bmp => "bmp",
        .gif => "gif",
    };
}

/// Writes `bytes` to the output.
pub fn writeBytes(ctx: *Ctx, out: Output, bytes: []const u8) !void {
    const io = ctx.io;
    if (out.stdout) {
        ctx.out.writeAll(bytes) catch return ctx.fail("cannot write to stdout: {s}", .{cx.describe(error.BrokenPipe)});
        ctx.out.flush() catch return ctx.fail("cannot write to stdout: {s}", .{cx.describe(error.BrokenPipe)});
        return;
    }
    if (out.special) {
        var f = Io.Dir.cwd().openFile(io, out.path, .{ .mode = .write_only }) catch |e| return writeFailed(ctx, out.path, e);
        defer f.close(io);
        f.writeStreamingAll(io, bytes) catch |e| return writeFailed(ctx, out.path, e);
        return;
    }
    var af = Io.Dir.cwd().createFileAtomic(io, out.path, .{ .replace = ctx.force }) catch |e| return writeFailed(ctx, out.path, e);
    defer af.deinit(io);
    af.file.writeStreamingAll(io, bytes) catch |e| return writeFailed(ctx, out.path, e);
    if (ctx.force) {
        af.replace(io) catch |e| return writeFailed(ctx, out.path, e);
    } else {
        af.link(io) catch |e| {
            if (e == error.PathAlreadyExists) {
                ctx.hint("pass -f/--force to replace it, or choose another name", .{});
                return ctx.fail("'{s}' already exists (it appeared while visar was working)", .{out.path});
            }
            return writeFailed(ctx, out.path, e);
        };
    }
}

fn writeFailed(ctx: *Ctx, path: []const u8, e: anyerror) error{Failed} {
    if (e == error.FileNotFound) {
        ctx.hint("create the directory '{s}' first", .{std.fs.path.dirname(path) orelse "."});
        return ctx.fail("cannot write '{s}': the directory does not exist", .{path});
    }
    return ctx.fail("cannot write '{s}': {s}", .{ path, cx.describe(e) });
}

pub const Saved = struct {
    bytes: usize,
    /// What had to change for the format to hold the image, if anything.
    adapted: ?[]const u8 = null,
};

/// Encodes `img` in the output's format and writes it. When the format
/// cannot hold the image as it is (16-bit samples in a JPEG, gray in a QOI)
/// it is converted to the nearest thing the format holds, and a warning says
/// so.
pub fn save(ctx: *Ctx, out: Output, img: AnyImage) !Saved {
    const gpa = ctx.gpa;
    const opts: codecs.EncodeOptions = .{ .quality = ctx.quality, .compression = ctx.compression };
    var adapted: ?[]const u8 = null;
    const bytes = codecs.encode(gpa, out.format, img, opts) catch |e| retry: {
        if (e != error.Unsupported and e != error.UnsupportedChannels) return encodeFailed(ctx, out, img, e);
        var fitted = (try fit(gpa, out.format, img, &adapted)) orelse return encodeFailed(ctx, out, img, e);
        defer fitted.deinit(gpa);
        break :retry codecs.encode(gpa, out.format, fitted, opts) catch |e2| return encodeFailed(ctx, out, img, e2);
    };
    defer gpa.free(bytes);
    if (adapted) |a| ctx.warn("{s} cannot hold this image as it is: {s}", .{ cx.formatName(out.format), a });
    try writeBytes(ctx, out, bytes);
    return .{ .bytes = bytes.len, .adapted = adapted };
}

fn encodeFailed(ctx: *Ctx, out: Output, img: AnyImage, e: anyerror) error{Failed} {
    var cb: [32]u8 = undefined;
    ctx.hint("PNG stores every depth and channel count VISAR reads", .{});
    return ctx.fail("cannot encode a {s} {s} image as {s}: {s}", .{ cx.depthName(img.depth()), cx.channelName(&cb, img.channels()), cx.formatName(out.format), cx.describe(e) });
}

/// A copy of `img` that `format` can store (8-bit, and 3 or 4 channels for
/// formats without gray), or null when nothing would change.
fn fit(gpa: std.mem.Allocator, format: Format, img: AnyImage, note: *?[]const u8) !?AnyImage {
    var cur: visar.Image(u8) = switch (img) {
        .u8 => |s| try s.clone(gpa),
        .u16 => |s| blk: {
            var d = try visar.Image(u8).init(gpa, s.width, s.height, s.channels);
            for (0..s.height) |y| {
                for (d.row(y), s.row(y)) |*o, v| o.* = @intCast(v >> 8);
            }
            note.* = "converted 16-bit samples to 8-bit";
            break :blk d;
        },
        else => return null,
    };
    errdefer cur.deinit(gpa);
    const needs_color = format == .qoi or format == .bmp or format == .gif;
    if (needs_color and cur.channels <= 2) {
        const want: u16 = if (cur.channels == 1) 3 else 4;
        const d = try codecs.common.convertChannels(u8, gpa, cur, want);
        cur.deinit(gpa);
        cur = d;
        note.* = if (note.* != null) "converted to 8-bit RGB" else "expanded gray to RGB";
    }
    if (note.* == null) {
        cur.deinit(gpa);
        return null;
    }
    return .{ .u8 = cur };
}

test "foreign formats are named" {
    try std.testing.expectEqualStrings("TIFF", foreignFormat("II*\x00abcd").?);
    try std.testing.expectEqualStrings("WebP", foreignFormat("RIFF\x00\x00\x00\x00WEBPVP8 ").?);
    try std.testing.expectEqual(@as(?[]const u8, null), foreignFormat("hello world!"));
}
