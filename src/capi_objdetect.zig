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

pub const CQrCode = extern struct {
    /// Top-left, top-right, bottom-right, bottom-left of the symbol (x, y).
    corners: [8]f32,
    /// The decoded text: text[offset .. offset + length].
    offset: u32,
    length: u32,
    decoded: i32,
    version: i32,
};

/// QRCodeDetector: detect (decode = 0), detectAndDecode (multi = 0) or
/// detectAndDecodeMulti (multi = 1) on an 8U gray, BGR or BGRA image.
/// Free with visar_qr_free.
export fn visar_qr_detect(src: *const CImage, multi: c_int, decode: c_int, codes: *?[*]CQrCode, count: *usize, text: *?[*]u8, text_len: *usize) c_int {
    codes.* = null;
    count.* = 0;
    text.* = null;
    text_len.* = 0;
    const img = typed(u8, src) catch |e| return fail(e);
    const found = od.qrDetect(gpa, img, .{ .multi = multi != 0, .decode = decode != 0 }) catch |e| return fail(e);
    defer od.freeQrCodes(gpa, found);
    if (found.len == 0) return 0;
    var total: usize = 0;
    for (found) |c| total += c.text.len;
    const cs = gpa.alloc(CQrCode, found.len) catch |e| return fail(e);
    const tb = gpa.alloc(u8, @max(total, 1)) catch |e| {
        gpa.free(cs);
        return fail(e);
    };
    var off: usize = 0;
    for (found, cs) |c, *o| {
        for (0..4) |k| {
            o.corners[2 * k] = c.corners[k].x;
            o.corners[2 * k + 1] = c.corners[k].y;
        }
        @memcpy(tb[off .. off + c.text.len], c.text);
        o.offset = @intCast(off);
        o.length = @intCast(c.text.len);
        o.decoded = @intFromBool(c.decoded);
        o.version = if (c.info) |i| i.version else 0;
        off += c.text.len;
    }
    codes.* = cs.ptr;
    count.* = cs.len;
    text.* = tb.ptr;
    text_len.* = @max(total, 1);
    return 0;
}

export fn visar_qr_free(codes: ?[*]CQrCode, count: usize, text: ?[*]u8, text_len: usize) void {
    if (codes) |c| gpa.free(c[0..count]);
    if (text) |t| gpa.free(t[0..text_len]);
}

// ------------------------------------------------------------------ ArUco

/// cv::aruco::DetectorParameters (OpenCV's field order and types).
pub const CArucoParams = extern struct {
    adaptive_thresh_win_size_min: c_int = 3,
    adaptive_thresh_win_size_max: c_int = 23,
    adaptive_thresh_win_size_step: c_int = 10,
    adaptive_thresh_constant: f64 = 7,
    min_marker_perimeter_rate: f64 = 0.03,
    max_marker_perimeter_rate: f64 = 4.0,
    polygonal_approx_accuracy_rate: f64 = 0.03,
    min_corner_distance_rate: f64 = 0.05,
    min_distance_to_border: c_int = 3,
    min_marker_distance_rate: f64 = 0.125,
    min_group_distance: f32 = 0.21,
    /// 0 NONE, 1 SUBPIX (CONTOUR and APRILTAG are not supported).
    corner_refinement_method: c_int = 0,
    corner_refinement_win_size: c_int = 5,
    relative_corner_refinement_win_size: f32 = 0.3,
    corner_refinement_max_iterations: c_int = 30,
    corner_refinement_min_accuracy: f64 = 0.1,
    marker_border_bits: c_int = 1,
    perspective_remove_pixel_per_cell: c_int = 4,
    perspective_remove_ignored_margin_per_cell: f64 = 0.13,
    max_erroneous_bits_in_border_rate: f64 = 0.35,
    min_otsu_std_dev: f64 = 5.0,
    error_correction_rate: f64 = 0.6,
    valid_bit_id_threshold: f32 = 0.49,
    detect_inverted_marker: c_int = 0,
};

/// Fills `p` with cv2.aruco.DetectorParameters()'s defaults.
export fn visar_aruco_default_params(p: *CArucoParams) void {
    p.* = .{};
}

fn dictionary(id: c_int) !od.ArucoDictionary {
    if (id < 0 or id > @backingInt(od.PredefinedDictionary.DICT_ARUCO_MIP_36h12)) return error.InvalidArgument;
    return od.getPredefinedDictionary(@fromBackingInt(@intCast(id)));
}

/// cv::aruco::generateImageMarker(getPredefinedDictionary(dict), id, side,
/// borderBits) into `dst` (8U, one channel, side x side).
export fn visar_aruco_generate(dict: c_int, id: c_int, side: c_int, border_bits: c_int, dst: *const CImage) c_int {
    const d = dictionary(dict) catch |e| return fail(e);
    const img = typed(u8, dst) catch |e| return fail(e);
    if (id < 0 or border_bits <= 0 or side <= 0 or img.width != @as(u32, @intCast(side))) return fail(error.InvalidArgument);
    od.generateImageMarker(gpa, d, @intCast(id), @intCast(border_bits), img) catch |e| return fail(e);
    return 0;
}

fn toParams(c: CArucoParams) !od.ArucoDetectorParameters {
    const m: od.aruco.CornerRefine = switch (c.corner_refinement_method) {
        0 => .none,
        1 => .subpix,
        2 => .contour,
        3 => .apriltag,
        else => return error.InvalidArgument,
    };
    return .{
        .adaptive_thresh_win_size_min = c.adaptive_thresh_win_size_min,
        .adaptive_thresh_win_size_max = c.adaptive_thresh_win_size_max,
        .adaptive_thresh_win_size_step = c.adaptive_thresh_win_size_step,
        .adaptive_thresh_constant = c.adaptive_thresh_constant,
        .min_marker_perimeter_rate = c.min_marker_perimeter_rate,
        .max_marker_perimeter_rate = c.max_marker_perimeter_rate,
        .polygonal_approx_accuracy_rate = c.polygonal_approx_accuracy_rate,
        .min_corner_distance_rate = c.min_corner_distance_rate,
        .min_distance_to_border = c.min_distance_to_border,
        .min_marker_distance_rate = c.min_marker_distance_rate,
        .min_group_distance = c.min_group_distance,
        .corner_refinement_method = m,
        .corner_refinement_win_size = c.corner_refinement_win_size,
        .relative_corner_refinement_win_size = c.relative_corner_refinement_win_size,
        .corner_refinement_max_iterations = c.corner_refinement_max_iterations,
        .corner_refinement_min_accuracy = c.corner_refinement_min_accuracy,
        .marker_border_bits = c.marker_border_bits,
        .perspective_remove_pixel_per_cell = c.perspective_remove_pixel_per_cell,
        .perspective_remove_ignored_margin_per_cell = c.perspective_remove_ignored_margin_per_cell,
        .max_erroneous_bits_in_border_rate = c.max_erroneous_bits_in_border_rate,
        .min_otsu_std_dev = c.min_otsu_std_dev,
        .error_correction_rate = c.error_correction_rate,
        .valid_bit_id_threshold = c.valid_bit_id_threshold,
        .detect_inverted_marker = c.detect_inverted_marker != 0,
    };
}

fn giveBytes(comptime T: type, items: []T) ?[*]u8 {
    if (items.len == 0) {
        gpa.free(items);
        return null;
    }
    const bytes = std.mem.sliceAsBytes(items);
    const copy = gpa.dupe(u8, bytes) catch {
        gpa.free(items);
        return null;
    };
    gpa.free(items);
    return copy.ptr;
}

/// ArucoDetector(getPredefinedDictionary(dict), params).detectMarkers on
/// an 8U gray, BGR or BGRA image. `params` may be null (the defaults).
/// corners: count * 8 floats (x, y per corner, OpenCV's order), ids:
/// count int32, rejected (optional): nrejected * 8 floats. Free each with
/// visar_free(ptr, bytes).
export fn visar_aruco_detect(src: *const CImage, dict: c_int, params: ?*const CArucoParams, corners: *?[*]u8, ids: *?[*]u8, count: *usize, rejected: ?*?[*]u8, nrejected: ?*usize) c_int {
    corners.* = null;
    ids.* = null;
    count.* = 0;
    if (rejected) |r| r.* = null;
    if (nrejected) |n| n.* = 0;
    const d = dictionary(dict) catch |e| return fail(e);
    const p = toParams(if (params) |q| q.* else .{}) catch |e| return fail(e);
    const img = typed(u8, src) catch |e| return fail(e);
    const res = od.detectMarkers(gpa, img, d, p) catch |e| return fail(e);
    count.* = res.ids.len;
    const nrej = res.rejected.len;
    // Ownership moves to the C side (as plain byte buffers).
    const c_ptr = giveBytes([4]visar.objdetect.aruco.Point2f, res.corners);
    const i_ptr = giveBytes(i32, res.ids);
    if (rejected != null and nrejected != null) {
        rejected.?.* = giveBytes([4]visar.objdetect.aruco.Point2f, res.rejected);
        nrejected.?.* = if (rejected.?.* != null) nrej else 0;
    } else gpa.free(res.rejected);
    if ((count.* > 0 and (c_ptr == null or i_ptr == null))) {
        if (c_ptr) |q| gpa.free(q[0 .. count.* * 32]);
        if (i_ptr) |q| gpa.free(q[0 .. count.* * 4]);
        count.* = 0;
        return fail(error.OutOfMemory);
    }
    corners.* = c_ptr;
    ids.* = i_ptr;
    return 0;
}

/// cv::aruco::drawDetectedMarkers on an 8U gray or BGR image (in place).
/// `ids` may be null; color is (B, G, R).
export fn visar_aruco_draw(img: *const CImage, corners: [*]const f32, ids: ?[*]const i32, n: usize, col: *const [3]f64) c_int {
    const im = typed(u8, img) catch |e| return fail(e);
    const pts: [*]const [4]visar.objdetect.aruco.Point2f = @ptrCast(@alignCast(corners));
    var c: [3]u8 = undefined;
    for (&c, col) |*o, v| o.* = visar.core.sat.cast(u8, v);
    od.drawDetectedMarkers(im, pts[0..n], if (ids) |i| i[0..n] else null, c) catch |e| return fail(e);
    return 0;
}
