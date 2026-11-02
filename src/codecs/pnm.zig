//! Netpbm: PBM (P1/P4), PGM (P2/P5), PPM (P3/P6) and PAM (P7) decode;
//! binary PGM/PPM/PAM encode. Samples above 255 are 16-bit big-endian.

const std = @import("std");
const common = @import("common.zig");
const Image = common.Image;
const AnyImage = common.AnyImage;
const DecodeError = common.DecodeError;

pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 2 and bytes[0] == 'P' and bytes[1] >= '1' and bytes[1] <= '7' and (bytes.len < 3 or std.ascii.isWhitespace(bytes[2]));
}

const Header = struct { kind: u8, w: u32, h: u32, cn: u16, maxval: u32, pos: usize };

fn skipSpaceAndComments(b: []const u8, pos: *usize) void {
    while (pos.* < b.len) {
        if (std.ascii.isWhitespace(b[pos.*])) {
            pos.* += 1;
        } else if (b[pos.*] == '#') {
            while (pos.* < b.len and b[pos.*] != '\n') pos.* += 1;
        } else break;
    }
}

fn readUint(b: []const u8, pos: *usize) DecodeError!u32 {
    skipSpaceAndComments(b, pos);
    const start = pos.*;
    while (pos.* < b.len and std.ascii.isDigit(b[pos.*])) pos.* += 1;
    if (pos.* == start) return if (pos.* >= b.len) error.Truncated else error.InvalidData;
    return std.fmt.parseInt(u32, b[start..pos.*], 10) catch error.InvalidData;
}

fn readToken(b: []const u8, pos: *usize) []const u8 {
    while (pos.* < b.len and (b[pos.*] == ' ' or b[pos.*] == '\t')) pos.* += 1;
    const start = pos.*;
    while (pos.* < b.len and !std.ascii.isWhitespace(b[pos.*])) pos.* += 1;
    return b[start..pos.*];
}

fn header(b: []const u8) DecodeError!Header {
    if (!detect(b)) return error.InvalidData;
    const kind = b[1];
    var pos: usize = 2;
    if (kind == '7') {
        var h: Header = .{ .kind = kind, .w = 0, .h = 0, .cn = 0, .maxval = 0, .pos = 0 };
        while (true) {
            skipSpaceAndComments(b, &pos);
            if (pos >= b.len) return error.Truncated;
            const key = readToken(b, &pos);
            if (std.mem.eql(u8, key, "ENDHDR")) break;
            if (std.mem.eql(u8, key, "TUPLTYPE")) {
                while (pos < b.len and b[pos] != '\n') pos += 1;
                continue;
            }
            const v = try readUint(b, &pos);
            if (std.mem.eql(u8, key, "WIDTH")) h.w = v else if (std.mem.eql(u8, key, "HEIGHT")) h.h = v else if (std.mem.eql(u8, key, "DEPTH")) h.cn = std.math.cast(u16, v) orelse return error.InvalidData else if (std.mem.eql(u8, key, "MAXVAL")) h.maxval = v else return error.InvalidData;
        }
        if (pos >= b.len or b[pos] != '\n') return error.InvalidData;
        h.pos = pos + 1;
        if (h.w == 0 or h.h == 0 or h.cn == 0 or h.cn > 4 or h.maxval == 0 or h.maxval > 65535) return error.InvalidData;
        return h;
    }
    const w = try readUint(b, &pos);
    const hgt = try readUint(b, &pos);
    const maxval: u32 = if (kind == '1' or kind == '4') 1 else try readUint(b, &pos);
    if (w == 0 or hgt == 0 or maxval == 0 or maxval > 65535) return error.InvalidData;
    // Exactly one whitespace byte separates the header from binary data.
    if (pos >= b.len) return error.Truncated;
    if (!std.ascii.isWhitespace(b[pos])) return error.InvalidData;
    pos += 1;
    const cn: u16 = switch (kind) {
        '3', '6' => 3,
        else => 1,
    };
    return .{ .kind = kind, .w = w, .h = hgt, .cn = cn, .maxval = maxval, .pos = pos };
}

pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: common.DecodeOptions) DecodeError!AnyImage {
    const h = try header(bytes);
    if (@as(u64, h.w) * h.h > opts.max_pixels) return error.TooLarge;
    const wide = h.maxval > 255;
    var out: AnyImage = if (wide) .{ .u16 = try Image(u16).init(gpa, h.w, h.h, h.cn) } else .{ .u8 = try Image(u8).init(gpa, h.w, h.h, h.cn) };
    errdefer out.deinit(gpa);
    const data = bytes[h.pos..];
    const n = @as(usize, h.w) * h.h * h.cn;
    // Rescale to full range when maxval is not 255 / 65535 (as OpenCV does).
    const full: u32 = if (wide) 65535 else 255;
    switch (h.kind) {
        '4' => {
            const img = out.u8;
            const row_bytes = (h.w + 7) / 8;
            if (data.len < row_bytes * h.h) return error.Truncated;
            for (0..h.h) |y| {
                const r = img.row(y);
                for (0..h.w) |x| r[x] = if ((data[y * row_bytes + x / 8] >> @intCast(7 - x % 8)) & 1 != 0) 0 else 255;
            }
        },
        '5', '6', '7' => {
            const bps: usize = if (wide) 2 else 1;
            if (data.len < n * bps) return error.Truncated;
            switch (out) {
                .u8 => |img| {
                    @memcpy(img.slice(), data[0..n]);
                    if (h.maxval != 255) for (img.slice()) |*v| {
                        v.* = @intCast(@min(full, (@as(u32, v.*) * full + h.maxval / 2) / h.maxval));
                    };
                },
                .u16 => |img| for (img.slice(), 0..) |*v, i| {
                    const raw = std.mem.readInt(u16, data[2 * i ..][0..2], .big);
                    v.* = if (h.maxval == 65535) raw else @intCast(@min(full, (@as(u32, raw) * full + h.maxval / 2) / h.maxval));
                },
                else => unreachable,
            }
        },
        '1', '2', '3' => {
            var pos: usize = h.pos;
            for (0..n) |i| {
                var v: u32 = undefined;
                if (h.kind == '1') {
                    skipSpaceAndComments(bytes, &pos);
                    if (pos >= bytes.len) return error.Truncated;
                    v = switch (bytes[pos]) {
                        '0' => 255,
                        '1' => 0,
                        else => return error.InvalidData,
                    };
                    pos += 1;
                } else {
                    v = try readUint(bytes, &pos);
                    if (v > h.maxval) return error.InvalidData;
                    if (h.maxval != full) v = (v * full + h.maxval / 2) / h.maxval;
                }
                switch (out) {
                    .u8 => |img| img.slice()[i] = @intCast(v),
                    .u16 => |img| img.slice()[i] = @intCast(v),
                    else => unreachable,
                }
            }
        },
        else => return error.InvalidData,
    }
    return common.finish(gpa, out, opts);
}

/// Binary PGM (1 channel), PPM (3 channels) or PAM (2/4 channels).
pub fn encode(gpa: std.mem.Allocator, img: AnyImage, opts: common.EncodeOptions) common.EncodeError![]u8 {
    var list: std.ArrayList(u8) = .empty;
    errdefer list.deinit(gpa);
    switch (img) {
        inline .u8, .u16 => |im| {
            const T = @TypeOf(im).Elem;
            const cn = im.channels;
            if (cn == 0 or cn > 4) return error.Unsupported;
            const maxval: u32 = std.math.maxInt(T);
            var hdr: [160]u8 = undefined;
            const h = switch (cn) {
                1, 3 => std.fmt.bufPrint(&hdr, "P{c}\n{d} {d}\n{d}\n", .{ @as(u8, if (cn == 1) '5' else '6'), im.width, im.height, maxval }) catch unreachable,
                else => std.fmt.bufPrint(&hdr, "P7\nWIDTH {d}\nHEIGHT {d}\nDEPTH {d}\nMAXVAL {d}\nTUPLTYPE {s}\nENDHDR\n", .{ im.width, im.height, cn, maxval, if (cn == 2) "GRAYSCALE_ALPHA" else "RGB_ALPHA" }) catch unreachable,
            };
            try list.appendSlice(gpa, h);
            const swap = opts.order == .bgr and cn >= 3;
            for (0..im.height) |y| {
                const r = im.row(y);
                const start = list.items.len;
                try list.resize(gpa, start + r.len * @sizeOf(T));
                const dst = list.items[start..];
                for (0..im.width) |x| for (0..cn) |c| {
                    const sc = if (swap and c < 3) 2 - c else c;
                    const v = r[x * cn + sc];
                    const i = x * cn + c;
                    if (T == u8) dst[i] = v else std.mem.writeInt(u16, dst[2 * i ..][0..2], v, .big);
                };
            }
        },
        else => return error.Unsupported,
    }
    return list.toOwnedSlice(gpa);
}

test "pnm round trip" {
    const gpa = std.testing.allocator;
    inline for (.{ u8, u16 }) |T| {
        inline for (.{ 1, 2, 3, 4 }) |cn| {
            var img = try Image(T).init(gpa, 7, 5, cn);
            defer img.deinit(gpa);
            for (img.slice(), 0..) |*v, i| v.* = @truncate(i * 37);
            const enc = try encode(gpa, @unionInit(AnyImage, @typeName(T), img), .{});
            defer gpa.free(enc);
            var dec = try decode(gpa, enc, .{});
            defer dec.deinit(gpa);
            try std.testing.expectEqualSlices(T, img.slice(), dec.as(T).?.slice());
        }
    }
}

test "pnm ascii and bitmap" {
    const gpa = std.testing.allocator;
    var a = try decode(gpa, "P2\n# c\n3 1\n15\n0 15 5\n", .{});
    defer a.deinit(gpa);
    try std.testing.expectEqualSlices(u8, &.{ 0, 255, 85 }, a.u8.slice());
    var b = try decode(gpa, "P4\n9 1\n\xA0\x80", .{});
    defer b.deinit(gpa);
    try std.testing.expectEqualSlices(u8, &.{ 0, 255, 0, 255, 255, 255, 255, 255, 0 }, b.u8.slice());
    try std.testing.expectError(error.Truncated, decode(gpa, "P5\n4 4\n255\n\x00", .{}));
}
