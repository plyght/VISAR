//! GIF encoder (single images and animations). Colour images are mapped to
//! a fixed 252-colour 6x7x6 RGB cube (plus 4 grays) with optional ordered
//! dithering; gray images use a 256-level gray palette. Decoding is not
//! implemented yet.

const std = @import("std");
const common = @import("common.zig");
const Image = common.Image;
const AnyImage = common.AnyImage;
const Allocator = std.mem.Allocator;

pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 6 and (std.mem.eql(u8, bytes[0..6], "GIF87a") or std.mem.eql(u8, bytes[0..6], "GIF89a"));
}

pub fn decode(_: Allocator, _: []const u8, _: common.DecodeOptions) common.DecodeError!AnyImage {
    return error.Unsupported;
}

/// A single-frame GIF.
pub fn encode(gpa: Allocator, img: AnyImage, opts: common.EncodeOptions) common.EncodeError![]u8 {
    const im = img.as(u8) orelse return error.Unsupported;
    var a = try Animation.init(gpa, im.width, im.height, if (im.channels == 1) .gray else .color, opts.order);
    errdefer a.deinit();
    try a.addFrame(im, 0);
    return a.finish();
}

pub const Palette = enum { color, gray };

/// Streams frames into an in-memory GIF89a animation that loops forever.
pub const Animation = struct {
    gpa: Allocator,
    out: std.ArrayList(u8) = .empty,
    width: u32,
    height: u32,
    palette: Palette,
    order: common.Order,
    dither: bool = true,
    indices: []u8,

    pub fn init(gpa: Allocator, width: u32, height: u32, palette: Palette, order: common.Order) Allocator.Error!Animation {
        var a: Animation = .{ .gpa = gpa, .width = width, .height = height, .palette = palette, .order = order, .indices = try gpa.alloc(u8, @as(usize, width) * height) };
        errdefer gpa.free(a.indices);
        const o = &a.out;
        try o.appendSlice(gpa, "GIF89a");
        try appendU16(o, gpa, @intCast(@min(width, 65535)));
        try appendU16(o, gpa, @intCast(@min(height, 65535)));
        try o.appendSlice(gpa, &.{ 0xF7, 0, 0 }); // global table, 8 bpp, 256 entries
        for (0..256) |i| {
            const rgb = paletteColor(palette, @intCast(i));
            try o.appendSlice(gpa, &rgb);
        }
        // NETSCAPE2.0 looping extension: loop forever.
        try o.appendSlice(gpa, &.{ 0x21, 0xFF, 0x0B });
        try o.appendSlice(gpa, "NETSCAPE2.0");
        try o.appendSlice(gpa, &.{ 0x03, 0x01, 0x00, 0x00, 0x00 });
        return a;
    }

    pub fn deinit(self: *Animation) void {
        self.out.deinit(self.gpa);
        self.gpa.free(self.indices);
    }

    /// Adds a frame (same size as the animation) shown for `delay_cs`
    /// hundredths of a second.
    pub fn addFrame(self: *Animation, img: Image(u8), delay_cs: u16) common.EncodeError!void {
        if (img.width != self.width or img.height != self.height) return error.SizeMismatch;
        if (img.channels != 1 and img.channels != 3 and img.channels != 4) return error.Unsupported;
        self.quantize(img);
        const gpa = self.gpa;
        const o = &self.out;
        try o.appendSlice(gpa, &.{ 0x21, 0xF9, 0x04, 0x00 });
        try appendU16(o, gpa, delay_cs);
        try o.appendSlice(gpa, &.{ 0x00, 0x00 });
        try o.append(gpa, 0x2C);
        try appendU16(o, gpa, 0);
        try appendU16(o, gpa, 0);
        try appendU16(o, gpa, @intCast(@min(self.width, 65535)));
        try appendU16(o, gpa, @intCast(@min(self.height, 65535)));
        try o.append(gpa, 0x00);
        try lzw(gpa, o, self.indices);
    }

    /// Returns the finished file (the Animation is consumed).
    pub fn finish(self: *Animation) Allocator.Error![]u8 {
        try self.out.append(self.gpa, 0x3B);
        self.gpa.free(self.indices);
        self.indices = &.{}; // `deinit` may still run after this.
        return self.out.toOwnedSlice(self.gpa);
    }

    fn quantize(self: *Animation, img: Image(u8)) void {
        const bayer = [4][4]u8{ .{ 0, 8, 2, 10 }, .{ 12, 4, 14, 6 }, .{ 3, 11, 1, 9 }, .{ 15, 7, 13, 5 } };
        const cn = img.channels;
        for (0..img.height) |y| {
            const r = img.row(y);
            const dst = self.indices[y * self.width ..][0..self.width];
            for (dst, 0..) |*o, x| {
                if (self.palette == .gray or cn == 1) {
                    const v: u32 = if (cn == 1) r[x] else (@as(u32, r[x * cn]) + r[x * cn + 1] + r[x * cn + 2]) / 3;
                    o.* = if (self.palette == .gray) @intCast(v) else @intCast(cubeIndex(v, v, v, 0));
                    continue;
                }
                const p = r[x * cn ..];
                const red = if (self.order == .bgr) p[2] else p[0];
                const blue = if (self.order == .bgr) p[0] else p[2];
                const t: u32 = if (self.dither) bayer[y & 3][x & 3] else 8;
                o.* = cubeIndex(red, p[1], blue, t);
            }
        }
    }
};

/// Index into the 6x7x6 cube; `t` is an ordered-dither threshold in 0..15.
inline fn cubeIndex(r: u32, g: u32, b: u32, t: u32) u8 {
    const qr: u32 = @min(5, (r * 5 * 16 + t * 255) / (255 * 16));
    const qg: u32 = @min(6, (g * 6 * 16 + t * 255) / (255 * 16));
    const qb: u32 = @min(5, (b * 5 * 16 + t * 255) / (255 * 16));
    return @intCast(qr * 42 + qg * 6 + qb);
}

fn paletteColor(p: Palette, i: u8) [3]u8 {
    if (p == .gray) return .{ i, i, i };
    if (i < 252) {
        const r: u32 = i / 42;
        const g: u32 = (i / 6) % 7;
        const b: u32 = i % 6;
        return .{ @intCast(r * 255 / 5), @intCast(g * 255 / 6), @intCast(b * 255 / 5) };
    }
    const v: u8 = @intCast((@as(u32, i) - 251) * 51);
    return .{ v, v, v };
}

fn appendU16(o: *std.ArrayList(u8), gpa: Allocator, v: u16) Allocator.Error!void {
    try o.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
}

/// GIF LZW (8-bit minimum code size) into 255-byte sub-blocks.
fn lzw(gpa: Allocator, o: *std.ArrayList(u8), data: []const u8) Allocator.Error!void {
    const min_size = 8;
    const clear: u16 = 1 << min_size;
    const eoi: u16 = clear + 1;
    try o.append(gpa, min_size);

    // Dictionary as a hash table keyed by (prefix code << 8 | byte).
    const table_bits = 14;
    const Entry = struct { key: u32, code: u16 };
    var table: [1 << table_bits]Entry = undefined;
    const empty = std.math.maxInt(u32);
    const reset = struct {
        fn f(t: *[1 << table_bits]Entry) void {
            for (t) |*e| e.key = empty;
        }
    }.f;
    reset(&table);

    var block: [256]u8 = undefined;
    var block_len: usize = 0;
    var bitbuf: u32 = 0;
    var nbits: u5 = 0;
    var code_size: u5 = min_size + 1;
    var next_code: u16 = eoi + 1;

    const W = struct {
        o: *std.ArrayList(u8),
        gpa: Allocator,
        block: *[256]u8,
        block_len: *usize,
        bitbuf: *u32,
        nbits: *u5,
        fn emit(w: @This(), code: u16, size: u5) Allocator.Error!void {
            w.bitbuf.* |= @as(u32, code) << w.nbits.*;
            w.nbits.* += size;
            while (w.nbits.* >= 8) {
                w.block[1 + w.block_len.*] = @truncate(w.bitbuf.*);
                w.block_len.* += 1;
                w.bitbuf.* >>= 8;
                w.nbits.* -= 8;
                if (w.block_len.* == 255) try w.flush();
            }
        }
        fn flush(w: @This()) Allocator.Error!void {
            if (w.block_len.* == 0) return;
            w.block[0] = @intCast(w.block_len.*);
            try w.o.appendSlice(w.gpa, w.block[0 .. w.block_len.* + 1]);
            w.block_len.* = 0;
        }
    };
    const w: W = .{ .o = o, .gpa = gpa, .block = &block, .block_len = &block_len, .bitbuf = &bitbuf, .nbits = &nbits };

    try w.emit(clear, code_size);
    if (data.len == 0) {
        try w.emit(eoi, code_size);
    } else {
        var prefix: u16 = data[0];
        for (data[1..]) |c| {
            const key = (@as(u32, prefix) << 8) | c;
            var h: u32 = (key *% 2654435761) >> (32 - table_bits);
            const found: ?u16 = while (true) : (h = (h + 1) & ((1 << table_bits) - 1)) {
                const e = table[h];
                if (e.key == empty) break null;
                if (e.key == key) break e.code;
            };
            if (found) |code| {
                prefix = code;
                continue;
            }
            try w.emit(prefix, code_size);
            if (next_code < 4096) {
                table[h] = .{ .key = key, .code = next_code };
                // Grow after the code that needs the extra bit is assigned.
                if (@as(u32, next_code) == (@as(u32, 1) << code_size) and code_size < 12) code_size += 1;
                next_code += 1;
            } else {
                try w.emit(clear, code_size);
                reset(&table);
                code_size = min_size + 1;
                next_code = eoi + 1;
            }
            prefix = c;
        }
        try w.emit(prefix, code_size);
        try w.emit(eoi, code_size);
    }
    if (nbits > 0) {
        block[1 + block_len] = @truncate(bitbuf);
        block_len += 1;
    }
    if (block_len > 0) {
        block[0] = @intCast(block_len);
        try o.appendSlice(gpa, block[0 .. block_len + 1]);
    }
    try o.append(gpa, 0);
}

test "gif encodes" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).init(gpa, 64, 48, 3);
    defer img.deinit(gpa);
    for (img.slice(), 0..) |*v, i| v.* = @truncate(i * 7);
    const bytes = try encode(gpa, .{ .u8 = img }, .{});
    defer gpa.free(bytes);
    try std.testing.expect(detect(bytes));
    try std.testing.expectEqual(@as(u8, 0x3B), bytes[bytes.len - 1]);
}
