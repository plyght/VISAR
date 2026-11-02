//! Windows BMP / DIB. Decodes 1/4/8-bit palette (incl. RLE4/RLE8),
//! 16/24/32-bit (incl. BITFIELDS) images, top-down or bottom-up; encodes
//! 8-bit gray (palette), 24-bit and 32-bit.

const std = @import("std");
const common = @import("common.zig");
const Image = common.Image;
const AnyImage = common.AnyImage;
const DecodeError = common.DecodeError;
const Cursor = common.Cursor;

pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 18 and bytes[0] == 'B' and bytes[1] == 'M';
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: common.DecodeOptions) DecodeError!AnyImage {
    if (!detect(bytes)) return error.InvalidData;
    var c: Cursor = .{ .buf = bytes, .pos = 10 };
    const data_off = try c.int(u32, .little);
    const hsize = try c.int(u32, .little);
    var width: i64 = undefined;
    var height: i64 = undefined;
    var bpp: u16 = undefined;
    var compression: u32 = 0;
    var colors_used: u32 = 0;
    var masks: [4]u32 = .{ 0, 0, 0, 0 };
    if (hsize == 12) {
        width = try c.int(u16, .little);
        height = try c.int(u16, .little);
        _ = try c.int(u16, .little);
        bpp = try c.int(u16, .little);
    } else if (hsize >= 40) {
        width = try c.int(i32, .little);
        height = try c.int(i32, .little);
        _ = try c.int(u16, .little);
        bpp = try c.int(u16, .little);
        compression = try c.int(u32, .little);
        try c.skip(12);
        colors_used = try c.int(u32, .little);
        try c.skip(4);
        if (compression == 3 or compression == 6) {
            // Masks follow the 40-byte header (or are part of a V4/V5 header).
            masks[0] = try c.int(u32, .little);
            masks[1] = try c.int(u32, .little);
            masks[2] = try c.int(u32, .little);
            if (hsize >= 56 or compression == 6) masks[3] = try c.int(u32, .little);
        }
        c.pos = 14 + hsize;
        if (compression == 3 and hsize == 40) c.pos += 12;
        if (compression == 6 and hsize == 40) c.pos += 16;
    } else return error.InvalidData;
    if (width <= 0 or height == 0 or width > std.math.maxInt(u31) or @abs(height) > std.math.maxInt(u31)) return error.InvalidData;
    const top_down = height < 0;
    const w: u32 = @intCast(width);
    const h: u32 = @intCast(@abs(height));
    if (@as(u64, w) * h > opts.max_pixels) return error.TooLarge;

    // Palette.
    var palette: [256][3]u8 = @splat(.{ 0, 0, 0 });
    var gray_palette = true;
    if (bpp <= 8) {
        const entries: usize = if (colors_used > 0 and colors_used <= 256) colors_used else @as(usize, 1) << @intCast(bpp);
        const esz: usize = if (hsize == 12) 3 else 4;
        for (0..entries) |i| {
            const e = try c.take(esz);
            palette[i] = .{ e[2], e[1], e[0] }; // stored BGR
            if (e[0] != e[1] or e[1] != e[2]) gray_palette = false;
        }
    }
    if (data_off >= bytes.len) return error.Truncated;
    const pix = bytes[data_off..];

    const has_alpha = bpp == 32;
    const cn: u16 = if (bpp <= 8 and gray_palette) 1 else if (has_alpha) 4 else 3;
    var img = try Image(u8).init(gpa, w, h, cn);
    errdefer img.deinit(gpa);

    switch (compression) {
        1, 2 => {
            if (bpp != (if (compression == 1) @as(u16, 8) else 4)) return error.InvalidData;
            const idx = try gpa.alloc(u8, @as(usize, w) * h);
            defer gpa.free(idx);
            @memset(idx, 0);
            try rle(pix, idx, w, h, compression == 2);
            for (0..h) |y| {
                const r = img.row(if (top_down) y else h - 1 - y);
                expandPalette(idx[y * w ..][0..w], r, &palette, cn);
            }
        },
        0, 3, 6 => {
            const row_bytes = ((@as(usize, w) * bpp + 31) / 32) * 4;
            if (pix.len < row_bytes * (h - 1) + (@as(usize, w) * bpp + 7) / 8) return error.Truncated;
            if (compression == 0 and bpp == 16) masks = .{ 0x7C00, 0x03E0, 0x001F, 0 };
            if (compression == 0 and bpp == 32) masks = .{ 0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 };
            var tmp: [8192]u8 = undefined;
            for (0..h) |y| {
                const src = pix[y * row_bytes ..];
                const r = img.row(if (top_down) y else h - 1 - y);
                switch (bpp) {
                    1, 2, 4, 8 => {
                        var x: usize = 0;
                        while (x < w) {
                            const n = @min(tmp.len, w - x);
                            for (0..n) |i| {
                                const bit = (x + i) * bpp;
                                const byte = src[bit / 8];
                                const shift: u3 = @intCast(8 - bpp - (bit % 8));
                                tmp[i] = @intCast((byte >> shift) & ((@as(u16, 1) << @intCast(bpp)) - 1));
                            }
                            expandPalette(tmp[0..n], r[x * cn ..], &palette, cn);
                            x += n;
                        }
                    },
                    24 => for (0..w) |x| {
                        r[x * 3 + 0] = src[x * 3 + 2];
                        r[x * 3 + 1] = src[x * 3 + 1];
                        r[x * 3 + 2] = src[x * 3 + 0];
                    },
                    16, 32 => for (0..w) |x| {
                        const v: u32 = if (bpp == 16) std.mem.readInt(u16, src[x * 2 ..][0..2], .little) else std.mem.readInt(u32, src[x * 4 ..][0..4], .little);
                        r[x * cn + 0] = channel(v, masks[0]);
                        r[x * cn + 1] = channel(v, masks[1]);
                        r[x * cn + 2] = channel(v, masks[2]);
                        if (cn == 4) r[x * cn + 3] = if (masks[3] != 0) channel(v, masks[3]) else 255;
                    },
                    else => return error.Unsupported,
                }
            }
        },
        else => return error.Unsupported,
    }
    return common.finish(gpa, .{ .u8 = img }, opts);
}

fn expandPalette(idx: []const u8, out: []u8, pal: *const [256][3]u8, cn: u16) void {
    if (cn == 1) {
        for (idx, 0..) |i, x| out[x] = pal[i][0];
    } else for (idx, 0..) |i, x| {
        out[x * 3 + 0] = pal[i][0];
        out[x * 3 + 1] = pal[i][1];
        out[x * 3 + 2] = pal[i][2];
    }
}

/// Extracts a masked channel and scales it to 8 bits.
fn channel(v: u32, mask: u32) u8 {
    if (mask == 0) return 0;
    const shift: u5 = @intCast(@ctz(mask));
    const bits: u6 = @intCast(@popCount(mask));
    const raw = (v & mask) >> shift;
    if (bits >= 8) return @intCast(raw >> @intCast(bits - 8));
    const maxv = (@as(u32, 1) << @intCast(bits)) - 1;
    return @intCast((raw * 255 + maxv / 2) / maxv);
}

/// RLE8 / RLE4 into a bottom-up index buffer (row 0 = bottom).
fn rle(src: []const u8, idx: []u8, w: u32, h: u32, four: bool) DecodeError!void {
    var x: usize = 0;
    var y: usize = 0;
    var i: usize = 0;
    while (i + 1 < src.len) {
        const a = src[i];
        const b = src[i + 1];
        i += 2;
        if (a > 0) {
            for (0..a) |k| {
                if (x < w and y < h) idx[y * w + x] = if (four) (if (k % 2 == 0) b >> 4 else b & 15) else b;
                x += 1;
            }
            continue;
        }
        switch (b) {
            0 => {
                x = 0;
                y += 1;
            },
            1 => return,
            2 => {
                if (i + 1 >= src.len) return error.Truncated;
                x += src[i];
                y += src[i + 1];
                i += 2;
            },
            else => {
                const n: usize = b;
                const nbytes = if (four) (n + 1) / 2 else n;
                if (i + nbytes > src.len) return error.Truncated;
                for (0..n) |k| {
                    const v = if (four) (if (k % 2 == 0) src[i + k / 2] >> 4 else src[i + k / 2] & 15) else src[i + k];
                    if (x < w and y < h) idx[y * w + x] = v;
                    x += 1;
                }
                i += nbytes;
                if (nbytes % 2 == 1) i += 1; // word alignment
            },
        }
    }
}

pub fn encode(gpa: std.mem.Allocator, img: AnyImage, opts: common.EncodeOptions) common.EncodeError![]u8 {
    const im = img.as(u8) orelse return error.Unsupported;
    const cn: u32 = im.channels;
    if (cn != 1 and cn != 3 and cn != 4) return error.Unsupported;
    const bpp: u32 = cn * 8;
    const row_bytes = ((im.width * bpp + 31) / 32) * 4;
    const pal_size: u32 = if (cn == 1) 1024 else 0;
    const data_off: u32 = 54 + pal_size;
    const total = data_off + row_bytes * im.height;
    const out = try gpa.alloc(u8, total);
    @memset(out, 0);
    out[0] = 'B';
    out[1] = 'M';
    std.mem.writeInt(u32, out[2..6], total, .little);
    std.mem.writeInt(u32, out[10..14], data_off, .little);
    std.mem.writeInt(u32, out[14..18], 40, .little);
    std.mem.writeInt(i32, out[18..22], @intCast(im.width), .little);
    std.mem.writeInt(i32, out[22..26], @intCast(im.height), .little);
    std.mem.writeInt(u16, out[26..28], 1, .little);
    std.mem.writeInt(u16, out[28..30], @intCast(bpp), .little);
    std.mem.writeInt(u32, out[34..38], row_bytes * im.height, .little);
    if (cn == 1) for (0..256) |i| {
        const v: u8 = @intCast(i);
        out[54 + i * 4 ..][0..4].* = .{ v, v, v, 0 };
    };
    const bgr_in = opts.order == .bgr;
    for (0..im.height) |y| {
        const s = im.row(y);
        const d = out[data_off + (im.height - 1 - y) * row_bytes ..];
        if (cn == 1) {
            @memcpy(d[0..im.width], s);
            continue;
        }
        for (0..im.width) |x| {
            const p = s[x * cn ..];
            const o = d[x * cn ..];
            if (bgr_in) {
                o[0] = p[0];
                o[2] = p[2];
            } else {
                o[0] = p[2];
                o[2] = p[0];
            }
            o[1] = p[1];
            if (cn == 4) o[3] = p[3];
        }
    }
    return out;
}

test "bmp round trip" {
    const gpa = std.testing.allocator;
    inline for (.{ 1, 3, 4 }) |cn| {
        var img = try Image(u8).init(gpa, 5, 3, cn);
        defer img.deinit(gpa);
        for (img.slice(), 0..) |*v, i| v.* = @truncate(i * 31 + 7);
        const enc = try encode(gpa, .{ .u8 = img }, .{});
        defer gpa.free(enc);
        var dec = try decode(gpa, enc, .{});
        defer dec.deinit(gpa);
        try std.testing.expectEqualSlices(u8, img.slice(), dec.u8.slice());
    }
}
