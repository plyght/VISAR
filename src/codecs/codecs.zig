//! visar.codecs: image file formats, all implemented in Zig.

const std = @import("std");
const core = @import("../core/core.zig");
pub const common = @import("common.zig");
pub const png = @import("png.zig");
pub const qoi = @import("qoi.zig");
pub const pnm = @import("pnm.zig");
pub const bmp = @import("bmp.zig");
pub const gif = @import("gif.zig");

pub const Mode = common.Mode;
pub const Order = common.Order;
pub const DecodeOptions = common.DecodeOptions;
pub const EncodeOptions = common.EncodeOptions;
pub const DecodeError = common.DecodeError;
pub const EncodeError = common.EncodeError;
pub const AnyImage = core.AnyImage;

pub const Format = enum {
    png,
    jpeg,
    qoi,
    pnm,
    bmp,
    gif,

    fn module(comptime f: Format) type {
        return switch (f) {
            .png => png,
            .jpeg => jpeg,
            .qoi => qoi,
            .pnm => pnm,
            .bmp => bmp,
            .gif => gif,
        };
    }
};

pub const jpeg = @import("jpeg.zig");

/// Identifies the format from the file's first bytes.
pub fn detect(bytes: []const u8) ?Format {
    inline for (comptime std.enums.values(Format)) |f| {
        if (f.module().detect(bytes)) return f;
    }
    return null;
}

/// Format for a file extension such as ".png" or "jpg" (case-insensitive).
pub fn formatFromExtension(ext_in: []const u8) ?Format {
    const ext = if (std.mem.lastIndexOfScalar(u8, ext_in, '.')) |i| ext_in[i + 1 ..] else ext_in;
    var buf: [8]u8 = undefined;
    if (ext.len > buf.len) return null;
    const e = std.ascii.lowerString(&buf, ext);
    const table = .{
        .{ "png", Format.png }, .{ "jpg", Format.jpeg }, .{ "jpeg", Format.jpeg }, .{ "jpe", Format.jpeg },
        .{ "qoi", Format.qoi }, .{ "pgm", Format.pnm },  .{ "ppm", Format.pnm },   .{ "pnm", Format.pnm },
        .{ "pbm", Format.pnm }, .{ "pam", Format.pnm },  .{ "bmp", Format.bmp },   .{ "dib", Format.bmp },
        .{ "gif", Format.gif },
    };
    inline for (table) |t| if (std.mem.eql(u8, e, t[0])) return t[1];
    return null;
}

/// Decodes an in-memory file of any supported format.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: DecodeOptions) DecodeError!AnyImage {
    const f = detect(bytes) orelse return error.InvalidData;
    return switch (f) {
        inline else => |ff| ff.module().decode(gpa, bytes, opts),
    };
}

/// Encodes to `format`. The caller frees the result.
pub fn encode(gpa: std.mem.Allocator, format: Format, img: AnyImage, opts: EncodeOptions) EncodeError![]u8 {
    return switch (format) {
        inline else => |ff| ff.module().encode(gpa, img, opts),
    };
}

pub const ReadError = DecodeError || std.Io.Dir.ReadFileAllocError;

/// Reads and decodes an image file.
pub fn read(gpa: std.mem.Allocator, io: std.Io, path: []const u8, opts: DecodeOptions) ReadError!AnyImage {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
    defer gpa.free(bytes);
    return decode(gpa, bytes, opts);
}

pub const WriteError = EncodeError || std.Io.Dir.WriteFileError || error{UnknownExtension};

/// Encodes `img` in the format implied by `path`'s extension and writes it.
pub fn write(gpa: std.mem.Allocator, io: std.Io, path: []const u8, img: AnyImage, opts: EncodeOptions) WriteError!void {
    const f = formatFromExtension(path) orelse return error.UnknownExtension;
    const bytes = try encode(gpa, f, img, opts);
    defer gpa.free(bytes);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes });
}

test {
    std.testing.refAllDecls(@This());
}

test "format detection" {
    try std.testing.expectEqual(Format.png, formatFromExtension("a/b.PNG").?);
    try std.testing.expectEqual(Format.jpeg, formatFromExtension(".jpg").?);
    try std.testing.expectEqual(@as(?Format, null), formatFromExtension("x.tiff"));
}
