//! Drawing: lines, rectangles, circles, polylines and text with a built-in
//! 5x7 bitmap font. Everything clips to the image; colours are given per
//! channel (missing channels use the last value given).

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;

inline fn putPixel(img: Image(u8), x: i64, y: i64, color: []const u8) void {
    if (x < 0 or y < 0 or x >= img.width or y >= img.height) return;
    const p = img.pixel(@intCast(x), @intCast(y));
    for (p, 0..) |*v, c| v.* = color[@min(c, color.len - 1)];
}

/// Filled axis-aligned rectangle [x0, x1) x [y0, y1), clipped.
pub fn fillRect(img: Image(u8), x0: i64, y0: i64, x1: i64, y1: i64, color: []const u8) void {
    const ax: usize = @intCast(std.math.clamp(x0, 0, img.width));
    const bx: usize = @intCast(std.math.clamp(x1, 0, img.width));
    const ay: usize = @intCast(std.math.clamp(y0, 0, img.height));
    const by: usize = @intCast(std.math.clamp(y1, 0, img.height));
    if (ax >= bx or ay >= by) return;
    const cn = img.channels;
    for (ay..by) |y| {
        const r = img.row(y);
        for (ax..bx) |x| for (0..cn) |c| {
            r[x * cn + c] = color[@min(c, color.len - 1)];
        };
    }
}

/// Rectangle outline with the given thickness (drawn inward).
pub fn rectangle(img: Image(u8), x: i64, y: i64, w: i64, h: i64, color: []const u8, thickness: u32) void {
    const t: i64 = @max(1, thickness);
    fillRect(img, x, y, x + w, y + t, color);
    fillRect(img, x, y + h - t, x + w, y + h, color);
    fillRect(img, x, y, x + t, y + h, color);
    fillRect(img, x + w - t, y, x + w, y + h, color);
}

/// Bresenham line; thickness draws a square brush.
pub fn line(img: Image(u8), x0_in: i64, y0_in: i64, x1: i64, y1: i64, color: []const u8, thickness: u32) void {
    var x0 = x0_in;
    var y0 = y0_in;
    const dx: i64 = @intCast(@abs(x1 - x0));
    const dy: i64 = -@as(i64, @intCast(@abs(y1 - y0)));
    const sx: i64 = if (x0 < x1) 1 else -1;
    const sy: i64 = if (y0 < y1) 1 else -1;
    var err = dx + dy;
    const r: i64 = @divTrunc(@as(i64, @max(1, thickness)), 2);
    while (true) {
        if (thickness <= 1) putPixel(img, x0, y0, color) else fillRect(img, x0 - r, y0 - r, x0 - r + thickness, y0 - r + thickness, color);
        if (x0 == x1 and y0 == y1) break;
        const e2 = 2 * err;
        if (e2 >= dy) {
            err += dy;
            x0 += sx;
        }
        if (e2 <= dx) {
            err += dx;
            y0 += sy;
        }
    }
}

/// Circle outline (thickness 0 = filled).
pub fn circle(img: Image(u8), cx: i64, cy: i64, radius: i64, color: []const u8, thickness: u32) void {
    const r2 = radius * radius;
    const inner: i64 = if (thickness == 0) -1 else @max(0, radius - @as(i64, thickness));
    const ir2 = inner * inner;
    var y: i64 = -radius;
    while (y <= radius) : (y += 1) {
        var x: i64 = -radius;
        while (x <= radius) : (x += 1) {
            const d = x * x + y * y;
            if (d <= r2 and (thickness == 0 or d > ir2)) putPixel(img, cx + x, cy + y, color);
        }
    }
}

// Classic 5x7 font, column-major, bit 0 = top row.
const glyphs = blk: {
    var t: [128][5]u8 = @splat(.{ 0, 0, 0, 0, 0 });
    const defs = .{
        .{ '0', .{ 0x3E, 0x51, 0x49, 0x45, 0x3E } }, .{ '1', .{ 0x00, 0x42, 0x7F, 0x40, 0x00 } },
        .{ '2', .{ 0x42, 0x61, 0x51, 0x49, 0x46 } }, .{ '3', .{ 0x21, 0x41, 0x45, 0x4B, 0x31 } },
        .{ '4', .{ 0x18, 0x14, 0x12, 0x7F, 0x10 } }, .{ '5', .{ 0x27, 0x45, 0x45, 0x45, 0x39 } },
        .{ '6', .{ 0x3C, 0x4A, 0x49, 0x49, 0x30 } }, .{ '7', .{ 0x01, 0x71, 0x09, 0x05, 0x03 } },
        .{ '8', .{ 0x36, 0x49, 0x49, 0x49, 0x36 } }, .{ '9', .{ 0x06, 0x49, 0x49, 0x29, 0x1E } },
        .{ 'A', .{ 0x7E, 0x11, 0x11, 0x11, 0x7E } }, .{ 'B', .{ 0x7F, 0x49, 0x49, 0x49, 0x36 } },
        .{ 'C', .{ 0x3E, 0x41, 0x41, 0x41, 0x22 } }, .{ 'D', .{ 0x7F, 0x41, 0x41, 0x22, 0x1C } },
        .{ 'E', .{ 0x7F, 0x49, 0x49, 0x49, 0x41 } }, .{ 'F', .{ 0x7F, 0x09, 0x09, 0x09, 0x01 } },
        .{ 'G', .{ 0x3E, 0x41, 0x49, 0x49, 0x7A } }, .{ 'H', .{ 0x7F, 0x08, 0x08, 0x08, 0x7F } },
        .{ 'I', .{ 0x00, 0x41, 0x7F, 0x41, 0x00 } }, .{ 'J', .{ 0x20, 0x40, 0x41, 0x3F, 0x01 } },
        .{ 'K', .{ 0x7F, 0x08, 0x14, 0x22, 0x41 } }, .{ 'L', .{ 0x7F, 0x40, 0x40, 0x40, 0x40 } },
        .{ 'M', .{ 0x7F, 0x02, 0x0C, 0x02, 0x7F } }, .{ 'N', .{ 0x7F, 0x04, 0x08, 0x10, 0x7F } },
        .{ 'O', .{ 0x3E, 0x41, 0x41, 0x41, 0x3E } }, .{ 'P', .{ 0x7F, 0x09, 0x09, 0x09, 0x06 } },
        .{ 'Q', .{ 0x3E, 0x41, 0x51, 0x21, 0x5E } }, .{ 'R', .{ 0x7F, 0x09, 0x19, 0x29, 0x46 } },
        .{ 'S', .{ 0x46, 0x49, 0x49, 0x49, 0x31 } }, .{ 'T', .{ 0x01, 0x01, 0x7F, 0x01, 0x01 } },
        .{ 'U', .{ 0x3F, 0x40, 0x40, 0x40, 0x3F } }, .{ 'V', .{ 0x1F, 0x20, 0x40, 0x20, 0x1F } },
        .{ 'W', .{ 0x3F, 0x40, 0x38, 0x40, 0x3F } }, .{ 'X', .{ 0x63, 0x14, 0x08, 0x14, 0x63 } },
        .{ 'Y', .{ 0x07, 0x08, 0x70, 0x08, 0x07 } }, .{ 'Z', .{ 0x61, 0x51, 0x49, 0x45, 0x43 } },
        .{ '.', .{ 0x00, 0x60, 0x60, 0x00, 0x00 } }, .{ ':', .{ 0x00, 0x36, 0x36, 0x00, 0x00 } },
        .{ '-', .{ 0x08, 0x08, 0x08, 0x08, 0x08 } }, .{ '#', .{ 0x14, 0x7F, 0x14, 0x7F, 0x14 } },
        .{ '/', .{ 0x20, 0x10, 0x08, 0x04, 0x02 } }, .{ '%', .{ 0x23, 0x13, 0x08, 0x64, 0x62 } },
        .{ '(', .{ 0x00, 0x1C, 0x22, 0x41, 0x00 } }, .{ ')', .{ 0x00, 0x41, 0x22, 0x1C, 0x00 } },
        .{ '=', .{ 0x14, 0x14, 0x14, 0x14, 0x14 } }, .{ '+', .{ 0x08, 0x08, 0x3E, 0x08, 0x08 } },
        .{ ',', .{ 0x00, 0x50, 0x30, 0x00, 0x00 } }, .{ '|', .{ 0x00, 0x00, 0x7F, 0x00, 0x00 } },
        .{ 'x', .{ 0x44, 0x28, 0x10, 0x28, 0x44 } },
    };
    for (defs) |d| t[d[0]] = d[1];
    break :blk t;
};

/// Draws `text` with its top-left corner at (x, y). `scale` multiplies the
/// 5x7 glyphs; characters are 6*scale pixels apart. Lowercase letters other
/// than 'x' are drawn as capitals. Returns the width drawn.
pub fn text(img: Image(u8), x: i64, y: i64, str: []const u8, scale: u32, color: []const u8) i64 {
    const s: i64 = @max(1, scale);
    var cx = x;
    for (str) |ch_in| {
        var ch = ch_in;
        if (ch >= 'a' and ch <= 'z' and ch != 'x') ch -= 32;
        if (ch < 128) {
            const g = glyphs[ch];
            for (g, 0..) |colbits, col| {
                for (0..7) |row| {
                    if ((colbits >> @intCast(row)) & 1 != 0) {
                        const px = cx + @as(i64, @intCast(col)) * s;
                        const py = y + @as(i64, @intCast(row)) * s;
                        fillRect(img, px, py, px + s, py + s, color);
                    }
                }
            }
        }
        cx += 6 * s;
    }
    return cx - x;
}

test "draw primitives clip" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 20, 10, 3, 0);
    defer img.deinit(gpa);
    rectangle(img, -5, -5, 12, 9, &.{ 255, 0, 0 }, 2);
    line(img, 0, 0, 30, 30, &.{255}, 1);
    circle(img, 10, 5, 4, &.{ 0, 255, 0 }, 0);
    _ = text(img, 1, 1, "ID 7", 1, &.{255});
    try std.testing.expectEqual(@as(u8, 0), img.at(10, 5, 0));
    try std.testing.expectEqual(@as(u8, 255), img.at(10, 5, 1));
}
