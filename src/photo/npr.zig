//! Domain-transform filters (OpenCV's edgePreservingFilter, stylization and
//! pencilSketch, from Gastal and Oliveira's "Domain Transform for
//! Edge-Aware Image and Video Processing"), following OpenCV's float
//! arithmetic operation by operation.
//!
//! OpenCV runs every 1-D pass along image rows and transposes the image for
//! the vertical passes; VISAR keeps the colour channels in separate planes
//! and runs the recursive passes down columns, many columns per vector, on
//! the transposed planes for the horizontal direction. The float operations
//! per sample are OpenCV's, in its order. Results can still differ from
//! OpenCV's by one level in rare pixels: OpenCV raises the feedback
//! coefficient to per-pixel powers with the C library's powf, and computes
//! the Sobel magnitudes of stylization with fused multiply-adds on some
//! CPUs.

const std = @import("std");
const core = @import("../core/core.zig");
const filter = @import("../imgproc/filter.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error || error{ InvalidArgument, InvalidKernel };

pub const Flags = enum(u8) {
    /// cv::RECURS_FILTER
    recursive = 1,
    /// cv::NORMCONV_FILTER
    normalized_convolution = 2,
};

const VL = 16;
const VF = @Vector(VL, f32);
const VD = @Vector(VL / 2, f64);

/// 2^x for f64 lanes with x <= 0, accurate to a few f64 ulps (so its
/// rounding to f32 is the correctly rounded powf result but for the rarest
/// ties).
inline fn exp2Neg(x: VD) VD {
    const lo: VD = @splat(-1070.0);
    const xc = @max(x, lo);
    const n = @round(xc);
    const f = (xc - n) * @as(VD, @splat(std.math.ln2));
    // e^f, |f| <= ln2/2, Taylor to degree 13.
    var p: VD = @splat(1.0 / 6227020800.0);
    inline for ([_]f64{ 1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0, 1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0 }) |c| {
        p = p * f + @as(VD, @splat(c));
    }
    // Scale by 2^n in two steps (n can be below the normal range).
    const ni: @Vector(VL / 2, i64) = @intFromFloat(n);
    const h1 = ni >> @splat(1);
    const h2 = ni - h1;
    const b1: VD = @bitCast((h1 + @as(@Vector(VL / 2, i64), @splat(1023))) << @splat(52));
    const b2: VD = @bitCast((h2 + @as(@Vector(VL / 2, i64), @splat(1023))) << @splat(52));
    const r = p * b1 * b2;
    return @select(f64, x < lo, @as(VD, @splat(0)), r);
}

/// dst[i] = powf(a, e[i]) for a in (0, 1) and e >= 0.
fn powRow(a: f32, e: []const f32, dst: []f32) void {
    const la: f64 = std.math.log2(@as(f64, a));
    var i: usize = 0;
    while (i + VL <= e.len) : (i += VL) {
        const ev: VF = e[i..][0..VL].*;
        const ea: [VL]f32 = ev;
        var r: [VL]f32 = undefined;
        inline for (0..2) |hh| {
            var d: VD = undefined;
            inline for (0..VL / 2) |q| d[q] = ea[hh * (VL / 2) + q];
            const p = exp2Neg(d * @as(VD, @splat(la)));
            inline for (0..VL / 2) |q| r[hh * (VL / 2) + q] = @floatCast(p[q]);
        }
        dst[i..][0..VL].* = r;
    }
    while (i < e.len) : (i += 1) {
        var d: VD = @splat(@as(f64, e[i]));
        d *= @as(VD, @splat(la));
        dst[i] = @floatCast(exp2Neg(d)[0]);
    }
}

/// A float plane of `rows` x `cols`.
const Plane = struct {
    rows: usize,
    cols: usize,
    d: []f32,

    inline fn row(p: Plane, i: usize) []f32 {
        return p.d[i * p.cols ..][0..p.cols];
    }
};

fn transpose(src: Plane, dst: Plane) void {
    std.debug.assert(src.rows == dst.cols and src.cols == dst.rows);
    const B = 16;
    var i: usize = 0;
    while (i < src.rows) : (i += B) {
        const ie = @min(i + B, src.rows);
        var j: usize = 0;
        while (j < src.cols) : (j += B) {
            const je = @min(j + B, src.cols);
            for (i..ie) |ii| for (j..je) |jj| {
                dst.d[jj * dst.cols + ii] = src.d[ii * src.cols + jj];
            };
        }
    }
}

/// OpenCV's compute_Rfilter on the transpose: the recursive filter down
/// every column, forwards then backwards, with feedback V.
fn recursiveColumns(p: Plane, v: Plane) void {
    const n = p.cols;
    for (1..p.rows) |i| {
        const cur = p.row(i);
        const prev = p.row(i - 1);
        const vr = v.row(i);
        var j: usize = 0;
        while (j + VL <= n) : (j += VL) {
            const c: VF = cur[j..][0..VL].*;
            const pr: VF = prev[j..][0..VL].*;
            const vv: VF = vr[j..][0..VL].*;
            cur[j..][0..VL].* = c + (pr - c) * vv;
        }
        while (j < n) : (j += 1) cur[j] = cur[j] + (prev[j] - cur[j]) * vr[j];
    }
    var i = p.rows - 1;
    while (i > 0) {
        i -= 1;
        const cur = p.row(i);
        const nxt = p.row(i + 1);
        const vr = v.row(i + 1);
        var j: usize = 0;
        while (j + VL <= n) : (j += VL) {
            const c: VF = cur[j..][0..VL].*;
            const nx: VF = nxt[j..][0..VL].*;
            const vv: VF = vr[j..][0..VL].*;
            cur[j..][0..VL].* = c + (nx - c) * vv;
        }
        while (j < n) : (j += 1) cur[j] = cur[j] + (nxt[j] - cur[j]) * vr[j];
    }
}

/// The per-iteration sigma of OpenCV's three-iteration schedule.
fn sigmaH(sigma_s: f32, it: usize) f32 {
    const no_of_iter = 3;
    return @floatCast(@as(f64, sigma_s) * @sqrt(3.0) * std.math.pow(f64, 2, @floatFromInt(no_of_iter - (it + 1))) /
        @sqrt(std.math.pow(f64, 4, no_of_iter) - 1));
}

/// Domain transform derivatives (OpenCV's Domain_Filter::init): horiz and
/// vert, 1 + (sigma_s / sigma_r) * L1 distance to the previous pixel.
fn domainDerivatives(planes: []const Plane, horiz: Plane, vert: Plane, sigma_s: f32, sigma_r: f32) void {
    const h = horiz.rows;
    const w = horiz.cols;
    const k: f32 = sigma_s / sigma_r;
    for (0..h) |i| {
        const hr = horiz.row(i);
        hr[0] = 1.0;
        for (1..w) |j| {
            var d: f32 = 0;
            for (planes) |p| d = d + @abs(p.d[i * w + j] - p.d[i * w + j - 1]);
            hr[j] = 1.0 + d * k;
        }
    }
    @memset(vert.row(0), 1.0);
    for (1..h) |i| {
        const vr = vert.row(i);
        for (0..w) |j| {
            var d: f32 = 0;
            for (planes) |p| d = d + @abs(p.d[i * w + j] - p.d[(i - 1) * w + j]);
            vr[j] = 1.0 + d * k;
        }
    }
}

/// The recursive-filter variant on float planes in place.
fn recursiveFilter(gpa: Allocator, planes: []Plane, sigma_s: f32, sigma_r: f32) !void {
    const h = planes[0].rows;
    const w = planes[0].cols;
    const buf = try gpa.alloc(f32, 5 * h * w);
    defer gpa.free(buf);
    const horiz: Plane = .{ .rows = h, .cols = w, .d = buf[0 .. h * w] };
    const vert: Plane = .{ .rows = h, .cols = w, .d = buf[h * w .. 2 * h * w] };
    const horiz_t: Plane = .{ .rows = w, .cols = h, .d = buf[2 * h * w .. 3 * h * w] };
    const vpow: Plane = .{ .rows = h, .cols = w, .d = buf[3 * h * w .. 4 * h * w] };
    const tmp_t: Plane = .{ .rows = w, .cols = h, .d = buf[4 * h * w .. 5 * h * w] };
    domainDerivatives(planes, horiz, vert, sigma_s, sigma_r);
    transpose(horiz, horiz_t);
    const vpow_t: Plane = .{ .rows = w, .cols = h, .d = vpow.d };
    for (0..3) |it| {
        const sh = sigmaH(sigma_s, it);
        const a: f32 = @floatCast(@exp(-@sqrt(2.0) / @as(f64, sh)));
        // Horizontal: down the columns of the transposed planes.
        powRow(a, horiz_t.d, vpow_t.d);
        for (planes) |p| {
            transpose(p, tmp_t);
            recursiveColumns(tmp_t, vpow_t);
            transpose(tmp_t, p);
        }
        // Vertical.
        powRow(a, vert.d, vpow.d);
        for (planes) |p| recursiveColumns(p, vpow);
    }
}

/// OpenCV's compute_boxfilter for one row: 1-based indices of the first
/// domain positions past `pos - radius` and past `pos + radius`.
fn boxIndices(hz: []const f32, radius: f32, lower: []f32, upper: []f32) void {
    const w = hz.len;
    // domain_row[w] is +inf.
    const dom = struct {
        fn at(z: []const f32, k: usize) f32 {
            return if (k == z.len) std.math.inf(f32) else z[k];
        }
    }.at;
    var tl: f32 = 0;
    var tu: f32 = 0;
    const lp0 = hz[0] - radius;
    const up0 = hz[0] + radius;
    for (0..w) |j| if (dom(hz, j) > lp0) {
        tl = @floatFromInt(j);
        break;
    };
    for (0..w) |j| if (dom(hz, j) > up0) {
        tu = @floatFromInt(j);
        break;
    };
    lower[0] = tl + 1;
    upper[0] = tu + 1;
    var temp: f32 = 0;
    for (1..w) |j| {
        const lp = hz[j] - radius;
        const up = hz[j] + radius;
        var count: f32 = 0;
        var k: usize = @intFromFloat(tl);
        while (k < w + 1) : (k += 1) {
            if (dom(hz, k) > lp) {
                temp = count;
                break;
            }
            count += 1;
        }
        tl = tl + temp;
        count = 0;
        k = @intFromFloat(tu);
        while (k < w + 1) : (k += 1) {
            if (dom(hz, k) > up) {
                temp = count;
                break;
            }
            count += 1;
        }
        tu = tu + temp;
        lower[j] = tl + 1;
        upper[j] = tu + 1;
    }
}

/// OpenCV's compute_NCfilter: the normalized-convolution box filter along
/// rows of 3 planes (rows x cols) with domain `ct` (rows x cols).
///
/// OpenCV packs (channel, column, row) of its summed-area table into a
/// float and decodes it back with integer division; the decoding is done
/// here in closed form. It sends one case elsewhere: a window whose upper
/// end is the row's first sample (only at column 0, when the radius
/// exceeds the whole row's domain) reads the last column of the table's
/// last row, in the next channel, or past the end of the table for the
/// last channel (0 here).
fn ncRows(gpa: Allocator, planes: [3]Plane, ct: Plane, radius: f32) !void {
    const h = ct.rows;
    const w = ct.cols;
    const buf = try gpa.alloc(f32, 2 * w + 6 * (w + 1));
    defer gpa.free(buf);
    const lower = buf[0..w];
    const upper = buf[w .. 2 * w];
    const box = buf[2 * w ..][0 .. 3 * (w + 1)];
    const last = buf[2 * w + 3 * (w + 1) ..][0 .. 3 * (w + 1)];
    prefixRows(planes, h - 1, last, w);
    // Above 2^24 OpenCV's float indices round; follow them literally.
    const exact = 3 * h * (w + 1) < (1 << 24);
    for (0..h) |i| {
        boxIndices(ct.row(i), radius, lower, upper);
        prefixRows(planes, i, box, w);
        for (0..w) |j| {
            const den = upper[j] - lower[j];
            if (exact) {
                const bq: usize = @intFromFloat(upper[j] - 1);
                const lq: usize = @intFromFloat(lower[j] - 1);
                inline for (0..3) |pl| {
                    const hiv: f32 = if (bq != 0) box[pl * (w + 1) + bq] else if (i + 1 == h) 0 else if (pl < 2) last[(pl + 1) * (w + 1) + w] else 0;
                    planes[pl].d[i * w + j] = (hiv - box[pl * (w + 1) + lq]) / den;
                }
            } else {
                inline for (0..3) |pl| {
                    const c = 2 - pl;
                    const hiv = literalRead(box, last, h, w, i, c, upper[j]);
                    const lov = literalRead(box, last, h, w, i, c, lower[j]);
                    planes[pl].d[i * w + j] = (hiv - lov) / den;
                }
            }
        }
    }
}

/// Row i's running sums per plane: t[pl][j] = sum of the first j samples.
fn prefixRows(planes: [3]Plane, i: usize, t: []f32, w: usize) void {
    for (0..3) |pl| {
        const src = planes[pl].row(i);
        const b = t[pl * (w + 1) ..][0 .. w + 1];
        b[0] = 0;
        b[1] = src[0];
        for (2..w + 1) |j| b[j] = src[j - 1] + b[j - 1];
    }
}

/// OpenCV's float index arithmetic and decoding, for tables too large for
/// the closed form. Reads that land outside rows i and h - 1 give 0.
fn literalRead(box: []const f32, last: []const f32, h: usize, w: usize, i: usize, c: usize, idx: f32) f32 {
    const hwf: f32 = @floatFromInt(h * (w + 1));
    const hf: f32 = @floatFromInt(h);
    const cf: f32 = @floatFromInt(c + 1);
    const ind: f32 = @floatFromInt(i + 1);
    const v = (cf - 1) * hwf + (idx - 1) * hf + ind;
    const hw1: i64 = @intCast(h * (w + 1));
    const hi: i64 = @intCast(h);
    const bi: i64 = @intFromFloat(v);
    var r = @divTrunc(bi, hw1);
    const rem = bi - r * hw1;
    var q = @divTrunc(rem, hi);
    var p = rem - q * hi;
    if (q == 0) {
        p = hi;
        q = @intCast(w);
        r -= 1;
    }
    if (p == 0) {
        p = hi;
        q -= 1;
    }
    // Interleaved flat index within OpenCV's (h, w + 1, 3) table.
    const row_len: i64 = @intCast(3 * (w + 1));
    const flat = (p - 1) * row_len + q * 3 + (2 - r);
    if (flat < 0 or flat >= @as(i64, @intCast(h)) * row_len) return 0;
    const row: usize = @intCast(@divTrunc(flat, row_len));
    const k: usize = @intCast(@mod(flat, row_len));
    const t = if (row == i) box else if (row == h - 1) last else return 0;
    return t[(k % 3) * (w + 1) + k / 3];
}

fn normalizedConvolution(gpa: Allocator, planes: [3]Plane, sigma_s: f32, sigma_r: f32) !void {
    const h = planes[0].rows;
    const w = planes[0].cols;
    const buf = try gpa.alloc(f32, 3 * h * w + 3 * h * w);
    defer gpa.free(buf);
    const ct_h: Plane = .{ .rows = h, .cols = w, .d = buf[0 .. h * w] };
    const ct_v: Plane = .{ .rows = h, .cols = w, .d = buf[h * w .. 2 * h * w] };
    const ct_vt: Plane = .{ .rows = w, .cols = h, .d = buf[2 * h * w .. 3 * h * w] };
    var tp: [3]Plane = undefined;
    for (0..3) |c| tp[c] = .{ .rows = w, .cols = h, .d = buf[(3 + c) * h * w ..][0 .. h * w] };
    domainDerivatives(&planes, ct_h, ct_v, sigma_s, sigma_r);
    // Cumulative sums: along rows for ct_H, down columns for ct_V.
    for (0..h) |i| {
        const r = ct_h.row(i);
        for (1..w) |j| r[j] = r[j] + r[j - 1];
    }
    for (1..h) |i| {
        const r = ct_v.row(i);
        const pr = ct_v.row(i - 1);
        for (r, pr) |*o, p| o.* = o.* + p;
    }
    transpose(ct_v, ct_vt);
    for (0..3) |it| {
        const sh = sigmaH(sigma_s, it);
        const radius: f32 = @as(f32, @floatCast(@sqrt(3.0))) * sh;
        try ncRows(gpa, planes, ct_h, radius);
        for (0..3) |c| transpose(planes[c], tp[c]);
        try ncRows(gpa, tp, ct_vt, radius);
        for (0..3) |c| transpose(tp[c], planes[c]);
    }
}

/// cv::edgePreservingFilter on an 8-bit BGR image.
pub fn edgePreservingFilter(gpa: Allocator, src: Image(u8), dst: Image(u8), flags: Flags, sigma_s: f32, sigma_r: f32) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (src.channels != 3) return error.UnsupportedChannels;
    const h: usize = src.height;
    const w: usize = src.width;
    const buf = try gpa.alloc(f32, 3 * h * w);
    defer gpa.free(buf);
    var planes: [3]Plane = undefined;
    for (0..3) |c| planes[c] = .{ .rows = h, .cols = w, .d = buf[c * h * w ..][0 .. h * w] };
    toPlanes(src, &planes);
    switch (flags) {
        .recursive => try recursiveFilter(gpa, &planes, sigma_s, sigma_r),
        .normalized_convolution => try normalizedConvolution(gpa, planes, sigma_s, sigma_r),
    }
    // convertScaleAbs(res, dst, 255).
    for (0..h) |y| {
        const r = dst.row(y);
        for (0..w) |x| inline for (0..3) |c| {
            r[x * 3 + c] = satRound(@abs(planes[c].d[y * w + x] * 255.0));
        };
    }
}

fn toPlanes(src: Image(u8), planes: []Plane) void {
    const scale: f32 = @floatCast(@as(f64, 1.0 / 255.0));
    const cn = planes.len;
    const w: usize = src.width;
    for (0..src.height) |y| {
        const r = src.row(y);
        for (0..w) |x| for (0..cn) |c| {
            planes[c].d[y * w + x] = @as(f32, @floatFromInt(r[x * cn + c])) * scale;
        };
    }
}

/// saturate_cast<uchar>(float): round half to even, clamp.
inline fn satRound(v: f32) u8 {
    if (!(v > -0.5)) return 0;
    if (v >= 255.5) return 255;
    const r = @round(v);
    const re = if (@abs(v - @trunc(v)) == 0.5) 2.0 * @round(v / 2.0) else r;
    return @intFromFloat(std.math.clamp(re, 0, 255));
}

/// cv::stylization on an 8-bit BGR image: the normalized-convolution
/// filter, darkened by its Sobel gradient magnitude.
pub fn stylization(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), sigma_s: f32, sigma_r: f32) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (src.channels != 3) return error.UnsupportedChannels;
    const h: usize = src.height;
    const w: usize = src.width;
    const n = h * w;
    const buf = try gpa.alloc(f32, 6 * n);
    defer gpa.free(buf);
    var planes: [3]Plane = undefined;
    for (0..3) |c| planes[c] = .{ .rows = h, .cols = w, .d = buf[c * n ..][0..n] };
    toPlanes(src, &planes);
    try normalizedConvolution(gpa, planes, sigma_s, sigma_r);
    const gx = buf[3 * n .. 4 * n];
    const gy = buf[4 * n .. 5 * n];
    const mag = buf[5 * n .. 6 * n];
    @memset(mag, 0);
    for (0..3) |c| {
        const p = try Image(f32).fromSlice(planes[c].d, @intCast(w), @intCast(h), 1);
        const ix = try Image(f32).fromSlice(gx, @intCast(w), @intCast(h), 1);
        const iy = try Image(f32).fromSlice(gy, @intCast(w), @intCast(h), 1);
        try filter.sobel(exec, gpa, p, ix, .{ .dx = 1, .dy = 0, .ksize = 3 });
        try filter.sobel(exec, gpa, p, iy, .{ .dx = 0, .dy = 1, .ksize = 3 });
        // mag = mag1 + mag2 + mag3, summed left to right.
        for (mag, gx, gy) |*m, x, y| m.* = if (c == 0) @sqrt(x * x + y * y) else m.* + @sqrt(x * x + y * y);
    }
    for (0..h) |y| {
        const r = dst.row(y);
        for (0..w) |x| {
            const m = 1.0 - mag[y * w + x];
            inline for (0..3) |c| r[x * 3 + c] = satRound(planes[c].d[y * w + x] * m * 255.0);
        }
    }
}

/// cv::pencilSketch on an 8-bit BGR image: `gray` (8-bit, 1 channel) and
/// `color` (BGR) sketches.
pub fn pencilSketch(gpa: Allocator, src: Image(u8), gray: Image(u8), color: Image(u8), sigma_s: f32, sigma_r: f32, shade_factor: f32) Error!void {
    if (!src.sameShape(color) or !src.sameSize(gray) or gray.channels != 1) return error.SizeMismatch;
    if (src.channels != 3) return error.UnsupportedChannels;
    const h: usize = src.height;
    const w: usize = src.width;
    const n = h * w;
    const buf = try gpa.alloc(f32, 3 * n + 3 * n + 2 * @max(h, w) * 2);
    defer gpa.free(buf);
    var planes: [3]Plane = undefined;
    for (0..3) |c| planes[c] = .{ .rows = h, .cols = w, .d = buf[c * n ..][0..n] };
    toPlanes(src, &planes);
    const ct_h: Plane = .{ .rows = h, .cols = w, .d = buf[3 * n .. 4 * n] };
    const ct_v: Plane = .{ .rows = h, .cols = w, .d = buf[4 * n .. 5 * n] };
    const pen: Plane = .{ .rows = h, .cols = w, .d = buf[5 * n .. 6 * n] };
    const tmp = buf[6 * n ..];
    domainDerivatives(&planes, ct_h, ct_v, sigma_s, sigma_r);
    for (0..h) |i| {
        const r = ct_h.row(i);
        for (1..w) |j| r[j] = r[j] + r[j - 1];
    }
    for (1..h) |i| {
        const r = ct_v.row(i);
        const pr = ct_v.row(i - 1);
        for (r, pr) |*o, p| o.* = o.* + p;
    }
    // Only the first iteration's counts reach the outputs.
    const sh = sigmaH(sigma_s, 0);
    const radius: f32 = @as(f32, @floatCast(@sqrt(3.0))) * sh;
    const lo = tmp[0..@max(h, w)];
    const up = tmp[@max(h, w)..][0..@max(h, w)];
    for (0..h) |i| {
        boxIndices(ct_h.row(i), radius, lo[0..w], up[0..w]);
        const pr = pen.row(i);
        for (pr, lo[0..w], up[0..w]) |*o, l, u| o.* = u - l;
    }
    const col = tmp[2 * @max(h, w) ..][0..h];
    for (0..w) |j| {
        for (0..h) |i| col[i] = ct_v.d[i * w + j];
        boxIndices(col, radius, lo[0..h], up[0..h]);
        for (0..h) |i| {
            const p = &pen.d[i * w + j];
            p.* = shade_factor * (p.* + (up[i] - lo[i]));
        }
    }
    for (0..h) |y| {
        const gr = gray.row(y);
        const cr = color.row(y);
        for (0..w) |x| {
            const i = y * w + x;
            const yv = pen.d[i];
            gr[x] = satRound(yv * 255.0);
            // BGR -> YCrCb, Y replaced, back to BGR (OpenCV's float
            // coefficients).
            const b = planes[0].d[i];
            const g = planes[1].d[i];
            const r = planes[2].d[i];
            const y0 = r * 0.299 + g * 0.587 + b * 0.114;
            const cr_ = (r - y0) * 0.713 + 0.5;
            const cb = (b - y0) * 0.564 + 0.5;
            const rr = yv + (cr_ - 0.5) * 1.403;
            const gg = yv + (cb - 0.5) * -0.344 + (cr_ - 0.5) * -0.714;
            const bb = yv + (cb - 0.5) * 1.773;
            cr[x * 3] = satRound(bb * 255.0);
            cr[x * 3 + 1] = satRound(gg * 255.0);
            cr[x * 3 + 2] = satRound(rr * 255.0);
        }
    }
}

test "edge preserving filter keeps a flat image" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 120, 100, 3, 90);
    defer img.deinit(gpa);
    var o = try Image(u8).init(gpa, 120, 100, 3);
    defer o.deinit(gpa);
    for ([_]Flags{ .recursive, .normalized_convolution }) |f| {
        try edgePreservingFilter(gpa, img, o, f, 60, 0.4);
        for (o.slice()) |v| try std.testing.expectEqual(@as(u8, 90), v);
    }
}
