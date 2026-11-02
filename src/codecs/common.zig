//! Types shared by every codec.

const std = @import("std");
const core = @import("../core/core.zig");

pub const AnyImage = core.AnyImage;
pub const Image = core.Image;

/// Channel order for 3- and 4-channel results. VISAR's native order is RGB;
/// decoders write BGR directly (no extra pass) when asked, which the cv2
/// shim uses.
pub const Order = enum { rgb, bgr };

/// What to convert the decoded pixels to (OpenCV's IMREAD_* flags).
pub const Mode = enum {
    /// Keep the file's channels and depth (IMREAD_UNCHANGED).
    unchanged,
    /// 1 channel, 8-bit (IMREAD_GRAYSCALE).
    gray,
    /// 3 channels, 8-bit (IMREAD_COLOR, the default).
    color,
    /// 4 channels, 8-bit.
    color_alpha,
    /// Keep channel count, but force 8-bit.
    any_8bit,
};

pub const DecodeOptions = struct {
    mode: Mode = .unchanged,
    order: Order = .rgb,
    /// Refuse images with more pixels than this (decompression-bomb guard).
    max_pixels: u64 = 1 << 30,
};

pub const EncodeOptions = struct {
    order: Order = .rgb,
    /// 0-100 for lossy formats (JPEG); ignored elsewhere.
    quality: u8 = 95,
    /// 0-9 for PNG (zlib level); ignored elsewhere.
    compression: u8 = 6,
};

pub const DecodeError = error{
    /// The bytes are not a valid file of this format.
    InvalidData,
    /// Valid, but uses a feature this decoder does not implement yet.
    Unsupported,
    /// Larger than `max_pixels`.
    TooLarge,
    /// Ran out of data.
    Truncated,
} || std.mem.Allocator.Error;

pub const EncodeError = error{
    /// The image's depth or channel count cannot be stored in this format.
    Unsupported,
} || std.mem.Allocator.Error;

/// Applies `mode` and `order` to a freshly decoded 8- or 16-bit image whose
/// colour channels are in RGB(A) order. Takes ownership of `img`; returns a
/// new image (or `img` itself when nothing changes).
pub fn finish(gpa: std.mem.Allocator, img: AnyImage, opts: DecodeOptions) DecodeError!AnyImage {
    var cur = img;
    errdefer cur.deinit(gpa);
    // Depth.
    const want8 = opts.mode != .unchanged;
    if (want8 and cur == .u16) {
        const src = cur.u16;
        var dst = try Image(u8).init(gpa, src.width, src.height, src.channels);
        for (0..src.height) |y| {
            const s = src.row(y);
            const d = dst.row(y);
            for (s, d) |a, *b| b.* = @intCast(a >> 8);
        }
        cur.deinit(gpa);
        cur = .{ .u8 = dst };
    }
    // Channels.
    const want_ch: ?u16 = switch (opts.mode) {
        .gray => 1,
        .color => 3,
        .color_alpha => 4,
        else => null,
    };
    if (want_ch) |wc| if (cur.channels() != wc) {
        switch (cur) {
            .u8 => |src| {
                const dst = try convertChannels(u8, gpa, src, wc);
                cur.deinit(gpa);
                cur = .{ .u8 = dst };
            },
            .u16 => |src| {
                const dst = try convertChannels(u16, gpa, src, wc);
                cur.deinit(gpa);
                cur = .{ .u16 = dst };
            },
            else => return error.Unsupported,
        }
    };
    if (opts.order == .bgr) switch (cur) {
        inline .u8, .u16 => |im| swapRB(@TypeOf(im).Elem, im),
        else => {},
    };
    return cur;
}

/// In-place R<->B swap for 3/4-channel images.
pub fn swapRB(comptime T: type, img: Image(T)) void {
    if (img.channels != 3 and img.channels != 4) return;
    const c = img.channels;
    for (0..img.height) |y| {
        const r = img.row(y);
        var i: usize = 0;
        while (i < r.len) : (i += c) std.mem.swap(T, &r[i], &r[i + 2]);
    }
}

/// Gray/GA/RGB/RGBA -> gray/RGB/RGBA (RGB order), BT.601 weights for gray
/// as OpenCV uses when decoding (R*0.299 + G*0.587 + B*0.114, fixed point).
pub fn convertChannels(comptime T: type, gpa: std.mem.Allocator, src: Image(T), want: u16) !Image(T) {
    var dst = try Image(T).init(gpa, src.width, src.height, want);
    const max: u32 = std.math.maxInt(T);
    for (0..src.height) |y| {
        const s = src.row(y);
        const d = dst.row(y);
        for (0..src.width) |x| {
            const p = s[x * src.channels ..][0..src.channels];
            var r: u32 = undefined;
            var g: u32 = undefined;
            var b: u32 = undefined;
            var a: u32 = max;
            switch (src.channels) {
                1 => {
                    r = p[0];
                    g = p[0];
                    b = p[0];
                },
                2 => {
                    r = p[0];
                    g = p[0];
                    b = p[0];
                    a = p[1];
                },
                else => {
                    r = p[0];
                    g = p[1];
                    b = p[2];
                    if (src.channels >= 4) a = p[3];
                },
            }
            const o = d[x * want ..][0..want];
            switch (want) {
                1 => o[0] = @intCast((r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14),
                3 => {
                    o[0] = @intCast(r);
                    o[1] = @intCast(g);
                    o[2] = @intCast(b);
                },
                4 => {
                    o[0] = @intCast(r);
                    o[1] = @intCast(g);
                    o[2] = @intCast(b);
                    o[3] = @intCast(a);
                },
                else => unreachable,
            }
        }
    }
    return dst;
}

/// Big-endian / little-endian readers over a byte slice with bounds checks.
pub const Cursor = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn remaining(self: Cursor) usize {
        return self.buf.len - self.pos;
    }
    pub fn take(self: *Cursor, n: usize) DecodeError![]const u8 {
        if (self.remaining() < n) return error.Truncated;
        defer self.pos += n;
        return self.buf[self.pos..][0..n];
    }
    pub fn skip(self: *Cursor, n: usize) DecodeError!void {
        _ = try self.take(n);
    }
    pub fn int(self: *Cursor, comptime T: type, endian: std.builtin.Endian) DecodeError!T {
        const b = try self.take(@sizeOf(T));
        return std.mem.readInt(T, b[0..@sizeOf(T)], endian);
    }
};
