//! The C ABI of visar.photo (declared in include/visar.h): non-local means
//! denoising and inpainting, plus the 8-bit linear Lab conversion the
//! colored denoiser uses.

const std = @import("std");
const capi = @import("capi.zig");
const visar = @import("visar.zig");
const photo = visar.photo;
const lab = visar.imgproc.lab;

const gpa = capi.gpa;
const fail = capi.fail;
const run = capi.run;
const exec = capi.exec;
const typed = capi.typed;

fn size(v: c_int) !u32 {
    if (v <= 0) return error.InvalidArgument;
    return @intCast(v);
}

/// fastNlMeansDenoising (NORM_L2): 8U, 1-4 channels; dst may be src.
export fn visar_fast_nl_means_denoising(src: *const capi.CImage, dst: *const capi.CImage, h: f32, template_window_size: c_int, search_window_size: c_int) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    const t = size(template_window_size) catch |e| return fail(e);
    const w = size(search_window_size) catch |e| return fail(e);
    return run(photo.fastNlMeansDenoising(exec(), gpa, s, d, .{ .h = h, .template_window_size = t, .search_window_size = w }));
}

/// fastNlMeansDenoisingColored: 8U BGR or BGRA; dst may be src.
export fn visar_fast_nl_means_denoising_colored(src: *const capi.CImage, dst: *const capi.CImage, h: f32, h_color: f32, template_window_size: c_int, search_window_size: c_int) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    const t = size(template_window_size) catch |e| return fail(e);
    const w = size(search_window_size) catch |e| return fail(e);
    return run(photo.fastNlMeansDenoisingColored(exec(), gpa, s, d, h, h_color, t, w));
}

/// cvtColor COLOR_LBGR2Lab (inverse = 0) or COLOR_Lab2LBGR (inverse = 1),
/// 8U 3-channel.
export fn visar_lab_linear(src: *const capi.CImage, dst: *const capi.CImage, inverse: c_int) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    if (!s.sameShape(d)) return fail(error.SizeMismatch);
    if (s.channels != 3) return fail(error.UnsupportedChannels);
    for (0..s.height) |y| {
        const a = s.row(y);
        const b = d.row(y);
        for (0..s.width) |x| {
            const v = if (inverse != 0) lab.labToLbgr(a[3 * x], a[3 * x + 1], a[3 * x + 2]) else lab.lbgrToLab(a[3 * x], a[3 * x + 1], a[3 * x + 2]);
            b[3 * x ..][0..3].* = v;
        }
    }
    return 0;
}

/// inpaint: src 8U (1 or 3 channels), 16U or 32F (1 channel); mask 8U,
/// 1 channel; dst like src (may be src). flags: 0 INPAINT_NS,
/// 1 INPAINT_TELEA.
export fn visar_inpaint(src: *const capi.CImage, mask: *const capi.CImage, dst: *const capi.CImage, radius: f64, flags: c_int) c_int {
    const m = typed(u8, mask) catch |e| return fail(e);
    const meth = std.enums.fromInt(photo.InpaintMethod, flags) orelse return fail(error.InvalidArgument);
    return switch (src.depth) {
        0 => inpaintT(u8, src, m, dst, radius, meth),
        2 => inpaintT(u16, src, m, dst, radius, meth),
        5 => inpaintT(f32, src, m, dst, radius, meth),
        else => fail(error.Unsupported),
    };
}

fn inpaintT(comptime T: type, src: *const capi.CImage, m: visar.core.Image(u8), dst: *const capi.CImage, radius: f64, meth: photo.InpaintMethod) c_int {
    const s = typed(T, src) catch |e| return fail(e);
    const d = typed(T, dst) catch |e| return fail(e);
    return run(photo.inpaint(T, gpa, s, m, d, radius, meth));
}

/// edgePreservingFilter: BGR VISAR_8U; flags 1 RECURS_FILTER, 2 NORMCONV_FILTER.
export fn visar_edge_preserving_filter(src: *const capi.CImage, dst: *const capi.CImage, flags: c_int, sigma_s: f32, sigma_r: f32) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    const f = std.enums.fromInt(photo.EdgePreservingFlags, flags) orelse return fail(error.InvalidArgument);
    return run(photo.edgePreservingFilter(gpa, s, d, f, sigma_s, sigma_r));
}

/// stylization: BGR VISAR_8U.
export fn visar_stylization(src: *const capi.CImage, dst: *const capi.CImage, sigma_s: f32, sigma_r: f32) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    return run(photo.stylization(exec(), gpa, s, d, sigma_s, sigma_r));
}

/// pencilSketch: BGR VISAR_8U in; gray (1 channel) and color (BGR) out.
export fn visar_pencil_sketch(src: *const capi.CImage, gray: *const capi.CImage, color: *const capi.CImage, sigma_s: f32, sigma_r: f32, shade_factor: f32) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const g = typed(u8, gray) catch |e| return fail(e);
    const c = typed(u8, color) catch |e| return fail(e);
    return run(photo.pencilSketch(gpa, s, g, c, sigma_s, sigma_r, shade_factor));
}

/// bilateralFilter: VISAR_8U, 1 or 3 channels; dst must not be src.
/// border: OpenCV's BORDER_* (not WRAP).
export fn visar_bilateral_filter(src: *const capi.CImage, dst: *const capi.CImage, d: c_int, sigma_color: f64, sigma_space: f64, border: c_int) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const t = typed(u8, dst) catch |e| return fail(e);
    const b = std.enums.fromInt(visar.imgproc.Border, border) orelse return fail(error.InvalidArgument);
    return run(visar.imgproc.bilateralFilter(exec(), gpa, s, t, .{ .d = d, .sigma_color = sigma_color, .sigma_space = sigma_space, .border = b }));
}
