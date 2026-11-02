//! QOI ("Quite OK Image") decoder and encoder, per the QOI specification
//! v1.0 (https://qoiformat.org/qoi-specification.pdf).
//!
//! Files hold 3 (RGB) or 4 (RGBA) channels of 8-bit samples. The colorspace
//! byte is informational only and is ignored on decode; the encoder writes 0
//! (sRGB with linear alpha).

const std = @import("std");
const common = @import("common.zig");

const Image = common.Image;
const AnyImage = common.AnyImage;
const DecodeError = common.DecodeError;
const EncodeError = common.EncodeError;

const magic = "qoif";
const header_len = 14;
const end_marker = [8]u8{ 0, 0, 0, 0, 0, 0, 0, 1 };

const op_index: u8 = 0x00; // 00xxxxxx
const op_diff: u8 = 0x40; // 01xxxxxx
const op_luma: u8 = 0x80; // 10xxxxxx
const op_run: u8 = 0xc0; // 11xxxxxx
const op_rgb: u8 = 0xfe;
const op_rgba: u8 = 0xff;
const mask2: u8 = 0xc0;

/// True when `bytes` starts with the QOI magic "qoif".
pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 4 and std.mem.eql(u8, bytes[0..4], magic);
}

const Px = [4]u8;

inline fn hash(p: Px) u6 {
    return @truncate(@as(u32, p[0]) *% 3 +% @as(u32, p[1]) *% 5 +% @as(u32, p[2]) *% 7 +% @as(u32, p[3]) *% 11);
}

/// Decodes a QOI file into a 3- or 4-channel u8 image (as stored), then
/// applies `opts` via `common.finish`.
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: common.DecodeOptions) DecodeError!AnyImage {
    if (!detect(bytes)) return error.InvalidData;
    if (bytes.len < header_len) return error.Truncated;
    const w = std.mem.readInt(u32, bytes[4..8], .big);
    const h = std.mem.readInt(u32, bytes[8..12], .big);
    const ch = bytes[12];
    if (w == 0 or h == 0 or (ch != 3 and ch != 4) or bytes[13] > 1) return error.InvalidData;
    if (@as(u64, w) * h > opts.max_pixels) return error.TooLarge;

    var img = Image(u8).init(gpa, w, h, ch) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TooLarge,
    };
    const ok = switch (ch) {
        3 => decodePixels(3, bytes[header_len..], img.slice()),
        else => decodePixels(4, bytes[header_len..], img.slice()),
    };
    ok catch |e| {
        img.deinit(gpa);
        return e;
    };
    return common.finish(gpa, .{ .u8 = img }, opts);
}

fn decodePixels(comptime ch: usize, data: []const u8, out: []u8) DecodeError!void {
    var index: [64]Px = @splat(.{ 0, 0, 0, 0 });
    var px: Px = .{ 0, 0, 0, 255 };
    var p: usize = 0;
    var o: usize = 0;
    const n = out.len;
    while (o < n) {
        // Every op is at most 5 bytes; take the checked slow path near the end.
        if (data.len - p < 5) {
            @branchHint(.unlikely);
            if (p >= data.len) return error.Truncated;
            const need: usize = switch (data[p]) {
                op_rgb => 4,
                op_rgba => 5,
                else => if (data[p] & mask2 == op_luma) 2 else 1,
            };
            if (data.len - p < need) return error.Truncated;
        }
        const b1 = data[p];
        p += 1;
        if (b1 == op_rgb) {
            px[0] = data[p];
            px[1] = data[p + 1];
            px[2] = data[p + 2];
            p += 3;
        } else if (b1 == op_rgba) {
            px = data[p..][0..4].*;
            p += 4;
        } else switch (b1 & mask2) {
            op_index => px = index[b1 & 0x3f],
            op_diff => {
                px[0] +%= ((b1 >> 4) & 3) -% 2;
                px[1] +%= ((b1 >> 2) & 3) -% 2;
                px[2] +%= (b1 & 3) -% 2;
            },
            op_luma => {
                const b2 = data[p];
                p += 1;
                const vg = (b1 & 0x3f) -% 32;
                px[0] +%= vg -% 8 +% ((b2 >> 4) & 0x0f);
                px[1] +%= vg;
                px[2] +%= vg -% 8 +% (b2 & 0x0f);
            },
            else => { // run of 1..62 (the 63/64 values are taken by RGB/RGBA)
                const run: usize = (b1 & 0x3f) + 1;
                const end = @min(n, o + run * ch);
                while (o < end) : (o += ch) out[o..][0..ch].* = px[0..ch].*;
                index[hash(px)] = px; // as the reference decoder does
                continue;
            },
        }
        index[hash(px)] = px;
        out[o..][0..ch].* = px[0..ch].*;
        o += ch;
    }
}

/// Encodes a 3- or 4-channel u8 image as QOI. `opts.order` gives the input's
/// channel order (BGR input is swapped while writing). Caller frees.
pub fn encode(gpa: std.mem.Allocator, img: AnyImage, opts: common.EncodeOptions) EncodeError![]u8 {
    const im = img.as(u8) orelse return error.Unsupported;
    return switch (im.channels) {
        3 => if (opts.order == .bgr) encodeTyped(3, true, gpa, im) else encodeTyped(3, false, gpa, im),
        4 => if (opts.order == .bgr) encodeTyped(4, true, gpa, im) else encodeTyped(4, false, gpa, im),
        else => error.Unsupported,
    };
}

fn encodeTyped(comptime ch: usize, comptime bgr: bool, gpa: std.mem.Allocator, img: Image(u8)) EncodeError![]u8 {
    // Worst case: every pixel an RGB(A) op.
    const max_len = header_len + img.pixelCount() * (ch + 1) + end_marker.len;
    const buf = try gpa.alloc(u8, max_len);
    errdefer gpa.free(buf);

    @memcpy(buf[0..4], magic);
    std.mem.writeInt(u32, buf[4..8], img.width, .big);
    std.mem.writeInt(u32, buf[8..12], img.height, .big);
    buf[12] = ch;
    buf[13] = 0;
    var o: usize = header_len;

    var index: [64]Px = @splat(.{ 0, 0, 0, 0 });
    var prev: Px = .{ 0, 0, 0, 255 };
    var run: u8 = 0;
    const total = img.pixelCount();
    var count: usize = 0;
    for (0..img.height) |y| {
        const r = img.row(y);
        var x: usize = 0;
        while (x < r.len) : (x += ch) {
            count += 1;
            var px: Px = .{ r[x], r[x + 1], r[x + 2], 255 };
            if (bgr) std.mem.swap(u8, &px[0], &px[2]);
            if (ch == 4) px[3] = r[x + 3];

            if (@as(u32, @bitCast(px)) == @as(u32, @bitCast(prev))) {
                run += 1;
                if (run == 62 or count == total) {
                    buf[o] = op_run | (run - 1);
                    o += 1;
                    run = 0;
                }
                continue;
            }
            if (run > 0) {
                buf[o] = op_run | (run - 1);
                o += 1;
                run = 0;
            }
            const hi = hash(px);
            if (@as(u32, @bitCast(index[hi])) == @as(u32, @bitCast(px))) {
                buf[o] = op_index | hi;
                o += 1;
            } else {
                index[hi] = px;
                if (px[3] == prev[3]) {
                    const vr: i8 = @bitCast(px[0] -% prev[0]);
                    const vg: i8 = @bitCast(px[1] -% prev[1]);
                    const vb: i8 = @bitCast(px[2] -% prev[2]);
                    const vg_r = vr -% vg;
                    const vg_b = vb -% vg;
                    if (vr > -3 and vr < 2 and vg > -3 and vg < 2 and vb > -3 and vb < 2) {
                        buf[o] = op_diff | @as(u8, @intCast(vr + 2)) << 4 | @as(u8, @intCast(vg + 2)) << 2 | @as(u8, @intCast(vb + 2));
                        o += 1;
                    } else if (vg_r > -9 and vg_r < 8 and vg > -33 and vg < 32 and vg_b > -9 and vg_b < 8) {
                        buf[o] = op_luma | @as(u8, @intCast(vg + 32));
                        buf[o + 1] = @as(u8, @intCast(vg_r + 8)) << 4 | @as(u8, @intCast(vg_b + 8));
                        o += 2;
                    } else {
                        buf[o] = op_rgb;
                        buf[o + 1 ..][0..3].* = px[0..3].*;
                        o += 4;
                    }
                } else {
                    buf[o] = op_rgba;
                    buf[o + 1 ..][0..4].* = px;
                    o += 5;
                }
            }
            prev = px;
        }
    }
    @memcpy(buf[o..][0..end_marker.len], &end_marker);
    o += end_marker.len;
    if (gpa.resize(buf, o)) return buf[0..o];
    const out = try gpa.dupe(u8, buf[0..o]);
    gpa.free(buf);
    return out;
}

// ---------------------------------------------------------------------------
// Tests.

const testing = std.testing;

test "qoi round trip 3/4 channels, random and smooth" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(9);
    const rnd = prng.random();
    const sizes = [_][2]u32{ .{ 1, 1 }, .{ 5, 3 }, .{ 37, 19 }, .{ 128, 70 } };
    for (sizes) |sz| {
        for ([_]u16{ 3, 4 }) |c| {
            for (0..3) |kind| {
                var img = try Image(u8).init(gpa, sz[0], sz[1], c);
                defer img.deinit(gpa);
                for (img.slice(), 0..) |*v, i| v.* = switch (kind) {
                    0 => rnd.int(u8),
                    // Small steps exercise DIFF / LUMA; repeats exercise RUN / INDEX.
                    1 => @truncate((i / c) / 3 * 5 + (i % c) * 7 + rnd.uintLessThan(u8, 3)),
                    else => @truncate(((i / c) / 70) * 40 + (if ((i / c) % 9 == 0) @as(usize, 17) else 0)),
                };
                const bytes = try encode(gpa, .{ .u8 = img }, .{});
                defer gpa.free(bytes);
                try testing.expect(detect(bytes));
                var back = try decode(gpa, bytes, .{});
                defer back.deinit(gpa);
                try testing.expectEqual(c, back.channels());
                try testing.expectEqualSlices(u8, img.slice(), back.u8.slice());
            }
        }
    }
}

test "qoi long runs and bgr" {
    const gpa = testing.allocator;
    var img = try Image(u8).initFill(gpa, 200, 3, 3, 9);
    defer img.deinit(gpa);
    img.set(0, 0, 0, 1);
    img.set(0, 0, 2, 3);
    const bytes = try encode(gpa, .{ .u8 = img }, .{ .order = .bgr });
    defer gpa.free(bytes);
    // LUMA op + RGB op + runs of 62 covering 598 pixels -> 10 run ops.
    try testing.expectEqual(header_len + 2 + 4 + 10 + 8, bytes.len);
    var rgb = try decode(gpa, bytes, .{});
    defer rgb.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 3, 9, 1, 9, 9, 9 }, rgb.u8.row(0)[0..6]);
    var bgr = try decode(gpa, bytes, .{ .order = .bgr });
    defer bgr.deinit(gpa);
    try testing.expectEqualSlices(u8, img.slice(), bgr.u8.slice());
}

test "qoi spec ops decode" {
    const gpa = testing.allocator;
    // 6x1 RGBA: RGBA(10,20,30,40), DIFF(+1,-2,0), LUMA(vg=5, dr=-1, db=+2),
    // INDEX of the first pixel, RUN 2.
    const first: Px = .{ 10, 20, 30, 40 };
    const file = magic ++ [_]u8{ 0, 0, 0, 6, 0, 0, 0, 1, 4, 0 } ++
        [_]u8{ op_rgba, 10, 20, 30, 40 } ++
        [_]u8{op_diff | (3 << 4) | (0 << 2) | 2} ++
        [_]u8{ op_luma | (5 + 32), (7 << 4) | 10 } ++
        [_]u8{op_index | @as(u8, hash(first))} ++
        [_]u8{op_run | 1} ++ end_marker;
    var img = try decode(gpa, file, .{});
    defer img.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{
        10, 20, 30, 40, 11, 18, 30, 40, 15, 23, 37, 40, 10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40,
    }, img.u8.slice());
}

test "qoi malformed input" {
    const gpa = testing.allocator;
    try testing.expectError(error.InvalidData, decode(gpa, "nope", .{}));
    try testing.expectError(error.Truncated, decode(gpa, "qoif\x00\x00", .{}));
    const bad_ch = magic ++ [_]u8{ 0, 0, 0, 1, 0, 0, 0, 1, 2, 0 } ++ end_marker;
    try testing.expectError(error.InvalidData, decode(gpa, bad_ch, .{}));
    const huge = magic ++ [_]u8{ 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 3, 0 } ++ end_marker;
    try testing.expectError(error.TooLarge, decode(gpa, huge, .{}));
    var prng: std.Random.DefaultPrng = .init(1);
    const rnd = prng.random();
    var img = try Image(u8).init(gpa, 31, 7, 4);
    defer img.deinit(gpa);
    for (img.slice(), 0..) |*v, i| v.* = @truncate(i / 9 + rnd.uintLessThan(u8, 4));
    const good = try encode(gpa, .{ .u8 = img }, .{});
    defer gpa.free(good);
    for (0..good.len) |n| {
        if (decode(gpa, good[0..n], .{})) |r| {
            var rr = r;
            rr.deinit(gpa);
        } else |_| {}
    }
    const bad = try gpa.dupe(u8, good);
    defer gpa.free(bad);
    for (0..300) |_| {
        @memcpy(bad, good);
        bad[header_len + rnd.uintLessThan(usize, bad.len - header_len)] ^= rnd.int(u8) | 1;
        if (decode(gpa, bad, .{})) |r| {
            var rr = r;
            rr.deinit(gpa);
        } else |_| {}
    }
    var gray = try Image(u8).init(gpa, 2, 2, 1);
    defer gray.deinit(gpa);
    try testing.expectError(error.Unsupported, encode(gpa, .{ .u8 = gray }, .{}));
}
