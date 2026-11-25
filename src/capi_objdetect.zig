//! The C ABI of visar.objdetect (declared in include/visar.h): QR code
//! encoding, detection and decoding, and ArUco markers. Enumerations use
//! OpenCV's numbering (QRCodeEncoder::CorrectionLevel / EncodeMode,
//! cv::aruco::PredefinedDictionaryType, CornerRefineMethod).

const std = @import("std");
const capi = @import("capi.zig");
const visar = @import("visar.zig");
const od = visar.objdetect;
const Image = visar.core.Image;
const CImage = capi.CImage;
const gpa = capi.gpa;
const fail = capi.fail;
const typed = capi.typed;

fn qrOptions(version: c_int, level: c_int, mode: c_int) !od.QrEncodeOptions {
    if (version < 0 or version > 40) return error.InvalidArgument;
    if (level < 0 or level > 3) return error.InvalidArgument;
    const m: od.QrMode = switch (mode) {
        -1 => .auto,
        1 => .numeric,
        2 => .alphanumeric,
        4 => .byte,
        7 => .eci,
        8 => .kanji,
        else => return error.InvalidArgument,
    };
    return .{ .version = @intCast(version), .level = @fromBackingInt(@intCast(level)), .mode = m };
}

/// cv2.QRCodeEncoder: one symbol as a size x size 8U image (0 dark, 255
/// light, two-module quiet zone). Free `out` with visar_free(out, size*size).
export fn visar_qr_encode(text: [*]const u8, len: usize, version: c_int, level: c_int, mode: c_int, out: *?[*]u8, size: *u32) c_int {
    out.* = null;
    size.* = 0;
    const o = qrOptions(version, level, mode) catch |e| return fail(e);
    var img = od.qrEncode(gpa, text[0..len], o) catch |e| return fail(e);
    defer img.deinit(gpa);
    const buf = gpa.dupe(u8, img.owned.?) catch |e| return fail(e);
    out.* = buf.ptr;
    size.* = img.width;
    return 0;
}

/// encodeStructuredAppend into `n` (2..16) symbols: their images back to
/// back in `out` (free with visar_free(out, sum of sizes[i]^2)), sizes[i]
/// the side of symbol i; `count` receives the number of symbols.
export fn visar_qr_encode_structured_append(text: [*]const u8, len: usize, version: c_int, level: c_int, n: c_int, out: *?[*]u8, sizes: *[16]u32, count: *u32) c_int {
    out.* = null;
    count.* = 0;
    if (n < 1) return fail(error.InvalidArgument);
    const o = qrOptions(version, level, -1) catch |e| return fail(e);
    const imgs = od.qrEncodeStructuredAppend(gpa, text[0..len], o, @intCast(n)) catch |e| return fail(e);
    defer {
        for (imgs) |*im| im.deinit(gpa);
        gpa.free(imgs);
    }
    var total: usize = 0;
    for (imgs) |im| total += @as(usize, im.width) * im.height;
    const buf = gpa.alloc(u8, total) catch |e| return fail(e);
    var off: usize = 0;
    for (imgs, 0..) |im, i| {
        const sz = @as(usize, im.width) * im.height;
        @memcpy(buf[off .. off + sz], im.owned.?[0..sz]);
        off += sz;
        sizes[i] = im.width;
    }
    out.* = buf.ptr;
    count.* = @intCast(imgs.len);
    return 0;
}
