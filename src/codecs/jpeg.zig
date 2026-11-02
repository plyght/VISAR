//! JPEG codec (JFIF / Exif / Adobe).
//!
//! Decoding covers baseline and extended sequential (SOF0/SOF1) and
//! progressive (SOF2) Huffman-coded 8-bit images with 1, 3 or 4 components,
//! any integral sampling factors and restart markers. Output matches
//! libjpeg-turbo's defaults (ISLOW IDCT, fancy upsampling) and therefore
//! OpenCV's imread. Lossless, arithmetic-coded and 12-bit files return
//! error.Unsupported. EXIF orientation is reported by `orientation`, never
//! applied.
//!
//! Encoding writes baseline JFIF with IJG quality scaling, 4:2:0 chroma
//! subsampling and the standard Huffman tables, like OpenCV's imencode.

const std = @import("std");
const common = @import("common.zig");
const dec = @import("jpeg/decode.zig");
const enc = @import("jpeg/encode.zig");

pub const Info = dec.Info;
pub const info = dec.info;
pub const orientation = dec.orientation;
pub const Subsampling = enc.Subsampling;
pub const EncodeExtra = enc.Options;
pub const encodeWith = enc.encodeWith;

/// True when `bytes` starts with a JPEG SOI marker.
pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 3 and bytes[0] == 0xff and bytes[1] == 0xd8 and bytes[2] == 0xff;
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: common.DecodeOptions) common.DecodeError!common.AnyImage {
    return dec.decode(gpa, bytes, opts);
}

/// Encodes an 8-bit gray, gray+alpha, RGB or RGBA image (colour channels in
/// `opts.order`; alpha is dropped). Caller frees the result.
pub fn encode(gpa: std.mem.Allocator, img: common.AnyImage, opts: common.EncodeOptions) common.EncodeError![]u8 {
    return enc.encode(gpa, img, opts);
}

test {
    _ = @import("jpeg/dct.zig");
}

const testing = std.testing;

fn synthetic(gpa: std.mem.Allocator, w: u32, h: u32, ch: u16) !common.Image(u8) {
    var img = try common.Image(u8).init(gpa, w, h, ch);
    for (0..h) |y| {
        const r = img.row(y);
        for (0..w) |x| {
            for (0..ch) |c| {
                const fx: f32 = @floatFromInt(x);
                const fy: f32 = @floatFromInt(y);
                const fc: f32 = @floatFromInt(c);
                const v = 128 + 90 * @sin(fx * 0.05 + fc) * @cos(fy * 0.07 - fc) + (if ((x / 16 + y / 16) % 2 == 0) @as(f32, 20) else -20);
                r[x * ch + c] = @intFromFloat(std.math.clamp(v, 0, 255));
            }
        }
    }
    return img;
}

fn psnr(a: common.Image(u8), b: common.Image(u8)) f64 {
    var se: f64 = 0;
    for (0..a.height) |y| {
        for (a.row(y), b.row(y)) |p, q| {
            const d = @as(f64, @floatFromInt(p)) - @as(f64, @floatFromInt(q));
            se += d * d;
        }
    }
    const mse = se / @as(f64, @floatFromInt(a.rowLen() * a.height));
    return 10 * std.math.log10(255.0 * 255.0 / @max(mse, 1e-10));
}

test "jpeg round trip color" {
    const gpa = testing.allocator;
    const sizes = [_][2]u32{ .{ 64, 48 }, .{ 17, 9 }, .{ 1, 1 }, .{ 33, 70 } };
    for (sizes) |sz| {
        var img = try synthetic(gpa, sz[0], sz[1], 3);
        defer img.deinit(gpa);
        for ([_]enc.Subsampling{ .s420, .s422, .s444 }) |ss| {
            const bytes = try encodeWith(gpa, .{ .u8 = img }, .{ .quality = 90 }, .{ .subsampling = ss });
            defer gpa.free(bytes);
            try testing.expect(detect(bytes));
            var out = try decode(gpa, bytes, .{});
            defer out.deinit(gpa);
            const o = out.as(u8).?;
            try testing.expectEqual(img.width, o.width);
            try testing.expectEqual(img.height, o.height);
            try testing.expectEqual(@as(u16, 3), o.channels);
            if (sz[0] >= 17) try testing.expect(psnr(img, o) > 28);
        }
    }
}

test "jpeg round trip gray and bgr" {
    const gpa = testing.allocator;
    var img = try synthetic(gpa, 40, 30, 1);
    defer img.deinit(gpa);
    const bytes = try encode(gpa, .{ .u8 = img }, .{ .quality = 95 });
    defer gpa.free(bytes);
    var out = try decode(gpa, bytes, .{});
    defer out.deinit(gpa);
    try testing.expectEqual(@as(u16, 1), out.channels());
    try testing.expect(psnr(img, out.as(u8).?) > 35);

    var rgb = try synthetic(gpa, 40, 30, 3);
    defer rgb.deinit(gpa);
    const b2 = try encode(gpa, .{ .u8 = rgb }, .{ .quality = 95 });
    defer gpa.free(b2);
    var a = try decode(gpa, b2, .{});
    defer a.deinit(gpa);
    var b = try decode(gpa, b2, .{ .order = .bgr });
    defer b.deinit(gpa);
    var g = try decode(gpa, b2, .{ .mode = .gray });
    defer g.deinit(gpa);
    try testing.expectEqual(@as(u16, 1), g.channels());
    const ai = a.as(u8).?;
    const bi = b.as(u8).?;
    for (0..ai.height) |y| {
        const ra = ai.row(y);
        const rb = bi.row(y);
        var x: usize = 0;
        while (x < ra.len) : (x += 3) {
            try testing.expectEqual(ra[x], rb[x + 2]);
            try testing.expectEqual(ra[x + 1], rb[x + 1]);
        }
    }
}

test "jpeg malformed input" {
    const gpa = testing.allocator;
    var img = try synthetic(gpa, 50, 37, 3);
    defer img.deinit(gpa);
    const bytes = try encode(gpa, .{ .u8 = img }, .{ .quality = 75 });
    defer gpa.free(bytes);
    // Every truncation either decodes or fails cleanly.
    var n: usize = 0;
    while (n < bytes.len) : (n += 7) {
        if (decode(gpa, bytes[0..n], .{})) |im| {
            var m = im;
            m.deinit(gpa);
        } else |_| {}
    }
    // Corrupted bytes in the entropy-coded data.
    const buf = try gpa.dupe(u8, bytes);
    defer gpa.free(buf);
    var seed: u32 = 1;
    for (0..200) |_| {
        @memcpy(buf, bytes);
        for (0..4) |_| {
            seed = seed *% 1103515245 +% 12345;
            const i = (seed >> 8) % buf.len;
            buf[i] = @truncate(seed >> 3);
        }
        if (decode(gpa, buf, .{})) |im| {
            var m = im;
            m.deinit(gpa);
        } else |_| {}
    }
    try testing.expectError(error.InvalidData, decode(gpa, "\xff\xd8\xff\xd9", .{}));
    try testing.expectError(error.TooLarge, decode(gpa, bytes, .{ .max_pixels = 100 }));
    try testing.expect(!detect("GIF89a"));
}

test "jpeg exif orientation" {
    const bytes = "\xff\xd8\xff\xe1\x00\x22Exif\x00\x00MM\x00\x2a\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00\x00\x00\x00\x00\xff\xd9";
    try testing.expectEqual(@as(u8, 6), orientation(bytes));
    try testing.expectEqual(@as(u8, 1), orientation("\xff\xd8\xff\xd9"));
}

test "jpeg quality scaling" {
    const t = enc.scaleQuant(&@import("jpeg/tables.zig").std_luma_quant, 50);
    try testing.expectEqual(@as(u16, 16), t[0]);
    const t95 = enc.scaleQuant(&@import("jpeg/tables.zig").std_luma_quant, 95);
    try testing.expectEqual(@as(u16, 2), t95[0]);
}
pub fn dbg() *u8 {
    return &dec.dbg_skip;
}
