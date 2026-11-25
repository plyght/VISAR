//! QR code detection and decoding in images: OpenCV's QRCodeDetector
//! (detect, detectAndDecode, detectAndDecodeMulti) with its own pipeline.
//!
//! 1. Binarisation by local block means (8x8 blocks, thresholds averaged
//!    over 5x5 blocks), robust to uneven light, blur and noise.
//! 2. Finder patterns: every row is scanned for dark/light runs in the
//!    ratio 1:1:3:1:1, confirmed by vertical, horizontal and diagonal
//!    cross-checks; hits on nearby rows are merged.
//! 3. Triples of finders are scored as right-angled corners; for each, the
//!    module size is measured along the lines between the centres, the
//!    version estimated, the bottom-right alignment pattern searched for
//!    (version 2+) and a homography from the four centres built.
//! 4. Modules are sampled through the homography (bilinear grey level
//!    against the local threshold) and decoded (qr_decode.zig); failing
//!    samplings are retried with neighbouring versions, without the
//!    alignment pattern and with a global threshold.
//!
//! Corners are reported as OpenCV does: the symbol's outer corners (no
//! quiet zone) in the order top-left, top-right, bottom-right, bottom-left
//! of the symbol, in pixel-centre coordinates. Every image is treated as
//! untrusted input.

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core/core.zig");
const Image = core.Image;
const shape = @import("../imgproc/shape.zig");
const homography = @import("../calib3d/homography.zig");
const color = @import("../imgproc/color.zig");
const qr_decode = @import("qr_decode.zig");
const geom = @import("../imgproc/geometry.zig");

pub const Point2f = shape.Point2f;

/// Prints the pipeline's intermediate results (tests only).
pub var debug = false;

pub const Code = struct {
    corners: [4]Point2f,
    /// Decoded contents (owned; empty when not decoded).
    text: []u8,
    decoded: bool,
    /// Version and decode details (when decoded).
    info: ?qr_decode.Info,
};

pub const Error = Allocator.Error || core.Error || error{Unsupported};

pub fn freeCodes(gpa: Allocator, codes: []Code) void {
    for (codes) |c| gpa.free(c.text);
    gpa.free(codes);
}

// ------------------------------------------------------------ binarizer

const block = 8;

const Bin = struct {
    w: usize,
    h: usize,
    /// 1 = dark.
    bits: []u8,
    bw: usize,
    bh: usize,
    /// Per-block threshold: a pixel <= thr is dark.
    thr: []u8,

    inline fn dark(b: *const Bin, x: isize, y: isize) ?bool {
        if (x < 0 or y < 0 or x >= b.w or y >= b.h) return null;
        return b.bits[@as(usize, @intCast(y)) * b.w + @as(usize, @intCast(x))] != 0;
    }

    fn threshold(b: *const Bin, x: f32, y: f32) f32 {
        const bx: usize = @intFromFloat(std.math.clamp(x / block, 0, @as(f32, @floatFromInt(b.bw - 1))));
        const by: usize = @intFromFloat(std.math.clamp(y / block, 0, @as(f32, @floatFromInt(b.bh - 1))));
        return @floatFromInt(b.thr[by * b.bw + bx]);
    }

    fn deinit(b: *Bin, gpa: Allocator) void {
        gpa.free(b.bits);
        gpa.free(b.thr);
    }
};

fn binarize(gpa: Allocator, g: Image(u8)) Allocator.Error!Bin {
    const w: usize = g.width;
    const h: usize = g.height;
    const bits = try gpa.alloc(u8, w * h);
    errdefer gpa.free(bits);
    if (w < block * 5 or h < block * 5) {
        // Too small for local blocks: one global mean.
        var sum: u64 = 0;
        for (0..h) |y| for (g.data[y * g.stride ..][0..w]) |v| {
            sum += v;
        };
        const t: u8 = @intCast(sum / (w * h));
        for (0..h) |y| for (0..w) |x| {
            bits[y * w + x] = @intFromBool(g.data[y * g.stride + x] <= t);
        };
        const thr = try gpa.alloc(u8, 1);
        thr[0] = t;
        return .{ .w = w, .h = h, .bits = bits, .bw = 1, .bh = 1, .thr = thr };
    }
    const bw = (w + block - 1) / block;
    const bh = (h + block - 1) / block;
    const black = try gpa.alloc(u8, bw * bh);
    defer gpa.free(black);
    const thr = try gpa.alloc(u8, bw * bh);
    errdefer gpa.free(thr);
    for (0..bh) |by| {
        const y0 = @min(by * block, h - block);
        for (0..bw) |bx| {
            const x0 = @min(bx * block, w - block);
            var sum: u32 = 0;
            var mn: u8 = 255;
            var mx: u8 = 0;
            for (0..block) |yy| {
                const row: *const [block]u8 = g.data[(y0 + yy) * g.stride + x0 ..][0..block];
                const v: @Vector(block, u8) = row.*;
                sum += @reduce(.Add, @as(@Vector(block, u16), v));
                mn = @min(mn, @reduce(.Min, v));
                mx = @max(mx, @reduce(.Max, v));
            }
            var avg: u32 = sum >> 6;
            if (mx - mn <= 24) {
                avg = mn / 2;
                if (by > 0 and bx > 0) {
                    const nb = (@as(u32, black[(by - 1) * bw + bx]) + 2 * @as(u32, black[by * bw + bx - 1]) + black[(by - 1) * bw + bx - 1]) / 4;
                    if (mn < nb) avg = nb;
                }
            }
            black[by * bw + bx] = @intCast(avg);
        }
    }
    for (0..bh) |by| {
        const top = std.math.clamp(by, 2, bh - 3);
        for (0..bw) |bx| {
            const left = std.math.clamp(bx, 2, bw - 3);
            var sum: u32 = 0;
            for (top - 2..top + 3) |yy| for (left - 2..left + 3) |xx| {
                sum += black[yy * bw + xx];
            };
            thr[by * bw + bx] = @intCast(sum / 25);
        }
    }
    for (0..h) |y| {
        const by = @min(y / block, bh - 1);
        const src = g.data[y * g.stride ..][0..w];
        const dst = bits[y * w ..][0..w];
        var x: usize = 0;
        while (x < w) {
            const bx = @min(x / block, bw - 1);
            const t = thr[by * bw + bx];
            const end = if (bx == bw - 1) w else (bx + 1) * block;
            if (end - x == block) {
                const v: @Vector(block, u8) = src[x..][0..block].*;
                const tv: @Vector(block, u8) = @splat(t);
                dst[x..][0..block].* = @select(u8, v <= tv, @as(@Vector(block, u8), @splat(1)), @as(@Vector(block, u8), @splat(0)));
            } else {
                for (x..end) |i| dst[i] = @intFromBool(src[i] <= t);
            }
            x = end;
        }
    }
    return .{ .w = w, .h = h, .bits = bits, .bw = bw, .bh = bh, .thr = thr };
}

// ------------------------------------------------------- finder patterns

const Finder = struct {
    x: f32,
    y: f32,
    module: f32,
    count: u32,
};

fn ratioOk(c: [5]f32) bool {
    var total: f32 = 0;
    for (c) |v| {
        if (v == 0) return false;
        total += v;
    }
    if (total < 7) return false;
    const m = total / 7;
    const tol = m * 0.6;
    return @abs(m - c[0]) < tol and @abs(m - c[1]) < tol and @abs(3 * m - c[2]) < 3 * tol and
        @abs(m - c[3]) < tol and @abs(m - c[4]) < tol;
}

/// Runs of a 1:1:3:1:1 pattern along (dx, dy) through (x, y) (inside the
/// centre stone): the centre offset along the line and the total length,
/// or null.
fn crossCheck(b: *const Bin, x: isize, y: isize, dx: isize, dy: isize, max_count: f32) ?struct { center: f32, total: f32 } {
    var c: [5]f32 = @splat(0);
    // Backwards: centre dark, light, dark.
    var i: isize = 0;
    while (b.dark(x - i * dx, y - i * dy) orelse false) : (i += 1) c[2] += 1;
    if (b.dark(x - i * dx, y - i * dy) == null) return null;
    while (b.dark(x - i * dx, y - i * dy)) |d| {
        if (d or c[1] > max_count) break;
        c[1] += 1;
        i += 1;
    }
    if (b.dark(x - i * dx, y - i * dy) == null or c[1] > max_count) return null;
    while (b.dark(x - i * dx, y - i * dy)) |d| {
        if (!d or c[0] > max_count) break;
        c[0] += 1;
        i += 1;
    }
    if (c[0] > max_count) return null;
    // Forwards.
    i = 1;
    while (b.dark(x + i * dx, y + i * dy) orelse false) : (i += 1) c[2] += 1;
    if (b.dark(x + i * dx, y + i * dy) == null) return null;
    while (b.dark(x + i * dx, y + i * dy)) |d| {
        if (d or c[3] >= max_count) break;
        c[3] += 1;
        i += 1;
    }
    if (b.dark(x + i * dx, y + i * dy) == null or c[3] >= max_count) return null;
    while (b.dark(x + i * dx, y + i * dy)) |d| {
        if (!d or c[4] >= max_count) break;
        c[4] += 1;
        i += 1;
    }
    if (c[4] >= max_count) return null;
    if (!ratioOk(c)) return null;
    const end: f32 = @floatFromInt(i); // one past the last dark pixel
    return .{ .center = end - c[4] - c[3] - c[2] / 2 - 0.5, .total = c[0] + c[1] + c[2] + c[3] + c[4] };
}

fn addFinder(gpa: Allocator, list: *std.ArrayList(Finder), x: f32, y: f32, m: f32) Allocator.Error!void {
    for (list.items) |*f| {
        if (@abs(f.x - x) <= f.module * 1.5 and @abs(f.y - y) <= f.module * 1.5 and
            @abs(f.module - m) <= @max(1.0, f.module * 0.5))
        {
            const n: f32 = @floatFromInt(f.count);
            f.x = (f.x * n + x) / (n + 1);
            f.y = (f.y * n + y) / (n + 1);
            f.module = (f.module * n + m) / (n + 1);
            f.count += 1;
            return;
        }
    }
    try list.append(gpa, .{ .x = x, .y = y, .module = m, .count = 1 });
}

fn handleCandidate(gpa: Allocator, b: *const Bin, list: *std.ArrayList(Finder), runs: [5]f32, end_x: usize, y: usize) Allocator.Error!void {
    const total = runs[0] + runs[1] + runs[2] + runs[3] + runs[4];
    // Centre of the middle run, in pixel indices.
    const cx_f = @as(f32, @floatFromInt(end_x)) - runs[4] - runs[3] - runs[2] / 2;
    const cx: isize = @intFromFloat(cx_f);
    const iy: isize = @intCast(y);
    const v = crossCheck(b, cx, iy, 0, 1, total) orelse return;
    if (5 * @abs(v.total - total) >= 2 * total) return;
    const cy_f = @as(f32, @floatFromInt(iy)) + v.center;
    const cyi: isize = @intFromFloat(@round(cy_f));
    const hz = crossCheck(b, cx, cyi, 1, 0, total) orelse return;
    if (5 * @abs(hz.total - total) >= 2 * total) return;
    const cx2 = @as(f32, @floatFromInt(cx)) + hz.center;
    // Diagonal check guards against random textures.
    const cxi: isize = @intFromFloat(@round(cx2));
    const dg = crossCheck(b, cxi, cyi, 1, 1, total * 1.5) orelse return;
    _ = dg;
    const module = (hz.total + v.total) / 14;
    // Edge convention: pixel i spans [i, i + 1].
    try addFinder(gpa, list, cx2 + 0.5, cy_f + 0.5, module);
}

fn findFinders(gpa: Allocator, b: *const Bin) Allocator.Error![]Finder {
    var list: std.ArrayList(Finder) = .empty;
    errdefer list.deinit(gpa);
    const w = b.w;
    for (0..b.h) |y| {
        const row = b.bits[y * w ..][0..w];
        // Runs: we keep the last five lengths ending in a dark run.
        var runs: [5]f32 = @splat(0);
        var nruns: usize = 0;
        var x: usize = 0;
        // Skip a leading light stretch.
        while (x < w and row[x] == 0) x += 1;
        while (x < w) {
            const c = row[x];
            var e = x + 1;
            while (e < w and row[e] == c) e += 1;
            const len: f32 = @floatFromInt(e - x);
            std.mem.copyForwards(f32, runs[0..4], runs[1..5]);
            runs[4] = len;
            nruns += 1;
            if (c == 1 and nruns >= 5 and ratioOk(runs)) {
                try handleCandidate(gpa, b, &list, runs, e, y);
            }
            x = e;
        }
    }
    return list.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------- geometry

const P = struct {
    x: f32,
    y: f32,
    fn dist(a: P, b: P) f32 {
        return std.math.hypot(a.x - b.x, a.y - b.y);
    }
};

/// Length of the dark-light-dark run starting at `from` (inside a dark
/// region) towards `to`: zxing's sizeOfBlackWhiteBlackRun.
fn bwbRun(b: *const Bin, fx: f32, fy: f32, tx: f32, ty: f32) ?f32 {
    const dxf = tx - fx;
    const dyf = ty - fy;
    const len = @max(@abs(dxf), @abs(dyf));
    if (len < 1) return null;
    const sx = dxf / len;
    const sy = dyf / len;
    var state: u8 = 0; // 0 dark, 1 light, 2 dark
    var k: f32 = 0;
    while (k <= len) : (k += 1) {
        const px = fx + sx * k;
        const py = fy + sy * k;
        const d = b.dark(@intFromFloat(@floor(px)), @intFromFloat(@floor(py))) orelse return null;
        if ((state == 1) == d) {
            if (state == 2) return std.math.hypot(sx * k, sy * k);
            state += 1;
        }
    }
    if (state == 2) return std.math.hypot(sx * len, sy * len);
    return null;
}

fn moduleAlong(b: *const Bin, a: P, c: P) ?f32 {
    const r1 = bwbRun(b, a.x, a.y, c.x, c.y) orelse return null;
    // The other way: same length beyond `a`, clipped to the image.
    var ox = a.x - (c.x - a.x);
    var oy = a.y - (c.y - a.y);
    const wf: f32 = @floatFromInt(b.w);
    const hf: f32 = @floatFromInt(b.h);
    var scale: f32 = 1;
    if (ox < 0) scale = @min(scale, a.x / (a.x - ox));
    if (ox >= wf) scale = @min(scale, (wf - 1 - a.x) / (ox - a.x));
    if (oy < 0) scale = @min(scale, a.y / (a.y - oy));
    if (oy >= hf) scale = @min(scale, (hf - 1 - a.y) / (oy - a.y));
    ox = a.x + (ox - a.x) * scale;
    oy = a.y + (oy - a.y) * scale;
    const r2 = bwbRun(b, a.x, a.y, ox, oy) orelse return null;
    return (r1 + r2) / 7;
}

fn apply(h: [9]f64, x: f64, y: f64) P {
    const w = h[6] * x + h[7] * y + h[8];
    const iw = if (w != 0) 1 / w else 0;
    return .{ .x = @floatCast((h[0] * x + h[1] * y + h[2]) * iw), .y = @floatCast((h[3] * x + h[4] * y + h[5]) * iw) };
}

/// The alignment pattern nearest to (ex, ey): a dark dot of one module
/// with a light ring, cross-checked vertically.
fn findAlignment(b: *const Bin, ex: f32, ey: f32, m: f32, allowance: f32) ?P {
    const r = allowance * m;
    const x0: isize = @intFromFloat(@max(0, ex - r));
    const x1: isize = @intFromFloat(@min(@as(f32, @floatFromInt(b.w - 1)), ex + r));
    const y0: isize = @intFromFloat(@max(0, ey - r));
    const y1: isize = @intFromFloat(@min(@as(f32, @floatFromInt(b.h - 1)), ey + r));
    if (@as(f32, @floatFromInt(x1 - x0)) < 3 * m or @as(f32, @floatFromInt(y1 - y0)) < 3 * m) return null;
    var best: ?P = null;
    var best_d: f32 = std.math.inf(f32);
    const tol = @max(m * 0.7, 1.0);
    var y = y0;
    while (y <= y1) : (y += 1) {
        var x = x0;
        // Runs light, dark, light.
        var runs: [3]f32 = @splat(0);
        var n: usize = 0;
        while (x <= x1) {
            const d = b.dark(x, y).?;
            var e = x + 1;
            while (e <= x1 and b.dark(e, y).? == d) e += 1;
            runs[0] = runs[1];
            runs[1] = runs[2];
            runs[2] = @floatFromInt(e - x);
            n += 1;
            if (!d and n >= 3 and @abs(runs[1] - m) < tol and @abs(runs[0] - m) < tol and @abs(runs[2] - m) < tol) {
                const cx = @as(f32, @floatFromInt(e)) - runs[2] - runs[1] / 2;
                if (verticalDot(b, @intFromFloat(cx), y, m, tol)) |cy| {
                    const p: P = .{ .x = cx + 0.5, .y = cy + 0.5 };
                    const dd = std.math.hypot(p.x - ex, p.y - ey);
                    if (dd < best_d) {
                        best_d = dd;
                        best = p;
                    }
                }
            }
            x = e;
        }
    }
    return best;
}

fn verticalDot(b: *const Bin, x: isize, y: isize, m: f32, tol: f32) ?f32 {
    if (!(b.dark(x, y) orelse return null)) return null;
    var up: isize = 0;
    while (b.dark(x, y - up - 1) orelse false) up += 1;
    var down: isize = 0;
    while (b.dark(x, y + down + 1) orelse false) down += 1;
    const dark_len: f32 = @floatFromInt(up + down + 1);
    if (@abs(dark_len - m) >= tol) return null;
    // Light above and below.
    var lu: isize = 0;
    while (!(b.dark(x, y - up - 1 - lu) orelse return null)) lu += 1;
    var ld: isize = 0;
    while (!(b.dark(x, y + down + 1 + ld) orelse return null)) ld += 1;
    if (@abs(@as(f32, @floatFromInt(lu)) - m) >= tol or @abs(@as(f32, @floatFromInt(ld)) - m) >= tol) return null;
    return @as(f32, @floatFromInt(y - up)) + dark_len / 2 - 0.5;
}

/// Grey level at an edge-convention point, bilinear.
fn sampleGray(g: Image(u8), x: f32, y: f32) ?f32 {
    const fx = x - 0.5;
    const fy = y - 0.5;
    if (!(fx >= 0 and fy >= 0)) return null;
    const ix: usize = @intFromFloat(fx);
    const iy: usize = @intFromFloat(fy);
    if (ix + 1 >= g.width or iy + 1 >= g.height) {
        if (ix < g.width and iy < g.height) return @floatFromInt(g.data[iy * g.stride + ix]);
        return null;
    }
    const ax = fx - @as(f32, @floatFromInt(ix));
    const ay = fy - @as(f32, @floatFromInt(iy));
    const p = g.data[iy * g.stride + ix ..];
    const q = g.data[(iy + 1) * g.stride + ix ..];
    const top = @as(f32, @floatFromInt(p[0])) * (1 - ax) + @as(f32, @floatFromInt(p[1])) * ax;
    const bot = @as(f32, @floatFromInt(q[0])) * (1 - ax) + @as(f32, @floatFromInt(q[1])) * ax;
    return top * (1 - ay) + bot * ay;
}

const Ctx = struct {
    gpa: Allocator,
    g: Image(u8),
    b: *const Bin,
    vals: []f32,
    thr: []f32,
    mods: []u8,
    text: std.ArrayList(u8) = .empty,
    seen: std.ArrayList(u8) = .empty,
    stack: std.ArrayList([2]u16) = .empty,
    pts: std.ArrayList([2]u16) = .empty,
};

/// Samples a dim x dim grid through `h` into ctx.vals / ctx.thr; false when
/// the grid leaves the image.
fn sampleGrid(ctx: *Ctx, h: [9]f64, dim: usize) bool {
    for (0..dim) |r| for (0..dim) |c| {
        const p = apply(h, @as(f64, @floatFromInt(c)) + 0.5, @as(f64, @floatFromInt(r)) + 0.5);
        const v = sampleGray(ctx.g, p.x, p.y) orelse return false;
        ctx.vals[r * dim + c] = v;
        ctx.thr[r * dim + c] = ctx.b.threshold(p.x, p.y);
    };
    return true;
}

/// The fraction of timing-pattern modules with the expected colour.
fn timingScore(mods: []const u8, dim: usize) f32 {
    var ok: usize = 0;
    var n: usize = 0;
    for (8..dim - 8) |i| {
        const want: u8 = @intFromBool(i % 2 == 0);
        ok += @intFromBool(mods[6 * dim + i] == want);
        ok += @intFromBool(mods[i * dim + 6] == want);
        n += 2;
    }
    return @as(f32, @floatFromInt(ok)) / @as(f32, @floatFromInt(n));
}

fn threshModules(ctx: *Ctx, dim: usize, global: bool) void {
    const n = dim * dim;
    var t: f32 = 0;
    if (global) {
        // Two-means on the sampled values.
        var lo: f32 = 255;
        var hi: f32 = 0;
        for (ctx.vals[0..n]) |v| {
            lo = @min(lo, v);
            hi = @max(hi, v);
        }
        t = (lo + hi) / 2;
        for (0..8) |_| {
            var s0: f32 = 0;
            var c0: f32 = 0;
            var s1: f32 = 0;
            var c1: f32 = 0;
            for (ctx.vals[0..n]) |v| {
                if (v <= t) {
                    s0 += v;
                    c0 += 1;
                } else {
                    s1 += v;
                    c1 += 1;
                }
            }
            if (c0 == 0 or c1 == 0) break;
            t = (s0 / c0 + s1 / c1) / 2;
        }
    }
    for (0..n) |i| {
        const th = if (global) t else ctx.thr[i];
        ctx.mods[i] = @intFromBool(ctx.vals[i] <= th);
    }
}

const Geometry = struct {
    tl: P,
    tr: P,
    bl: P,
    module: f32,
    dim0: usize,
};

fn geometry(b: *const Bin, f: [3]Finder) ?Geometry {
    // The corner finder is opposite the longest side.
    const p = [3]P{ .{ .x = f[0].x, .y = f[0].y }, .{ .x = f[1].x, .y = f[1].y }, .{ .x = f[2].x, .y = f[2].y } };
    const d01 = p[0].dist(p[1]);
    const d02 = p[0].dist(p[2]);
    const d12 = p[1].dist(p[2]);
    var a: usize = 0;
    var bi: usize = 1;
    var ci: usize = 2;
    if (d01 >= d02 and d01 >= d12) {
        a = 2;
        bi = 0;
        ci = 1;
    } else if (d02 >= d01 and d02 >= d12) {
        a = 1;
        bi = 0;
        ci = 2;
    }
    var tl = p[a];
    var tr = p[bi];
    var bl = p[ci];
    const cross = (tr.x - tl.x) * (bl.y - tl.y) - (tr.y - tl.y) * (bl.x - tl.x);
    if (cross < 0) std.mem.swap(P, &tr, &bl);
    _ = &tl;
    // Module size along the lines between the centres.
    var sum: f32 = 0;
    var cnt: f32 = 0;
    for ([_][2]P{ .{ tl, tr }, .{ tr, tl }, .{ tl, bl }, .{ bl, tl } }) |pq| {
        if (moduleAlong(b, pq[0], pq[1])) |m| {
            sum += m;
            cnt += 1;
        }
    }
    const module = if (cnt > 0) sum / cnt else (f[0].module + f[1].module + f[2].module) / 3;
    if (!(module >= 0.8)) return null;
    const est = (tl.dist(tr) + tl.dist(bl)) / 2 / module + 7;
    if (!(est >= 17 and est <= 185)) return null;
    // Nearest 4k + 1.
    const k = @round((est - 17) / 4);
    const dim: usize = @intFromFloat(std.math.clamp(k, 1, 40) * 4 + 17);
    return .{ .tl = tl, .tr = tr, .bl = bl, .module = module, .dim0 = dim };
}

fn homographyFor(g: Geometry, dim: usize, al: ?P) ?[9]f64 {
    const fd: f32 = @floatFromInt(dim);
    var src: [4]Point2f = .{ .{ .x = 3.5, .y = 3.5 }, .{ .x = fd - 3.5, .y = 3.5 }, .{ .x = 3.5, .y = fd - 3.5 }, undefined };
    var dst: [4]Point2f = .{ .{ .x = g.tl.x, .y = g.tl.y }, .{ .x = g.tr.x, .y = g.tr.y }, .{ .x = g.bl.x, .y = g.bl.y }, undefined };
    if (al) |a| {
        src[3] = .{ .x = fd - 6.5, .y = fd - 6.5 };
        dst[3] = .{ .x = a.x, .y = a.y };
    } else {
        src[3] = .{ .x = fd - 3.5, .y = fd - 3.5 };
        dst[3] = .{ .x = g.tr.x + g.bl.x - g.tl.x, .y = g.tr.y + g.bl.y - g.tl.y };
    }
    return homography.getPerspectiveTransform(src, dst);
}

/// The four outer corners of a finder pattern's dark ring (unordered),
/// from a flood fill of the ring: the farthest point from the centre, the
/// farthest from that, and the two extremes across their diagonal.
fn finderCorners(ctx: *Ctx, f: Finder) Allocator.Error!?[4]P {
    const b = ctx.b;
    const cx: isize = @intFromFloat(@floor(f.x));
    const cy: isize = @intFromFloat(@floor(f.y));
    // Walk right from the centre: stone, light ring, dark ring.
    var x = cx;
    var state: u8 = 0;
    const lim = cx + @as(isize, @intFromFloat(f.module * 6 + 3));
    while (x < lim) : (x += 1) {
        const d = b.dark(x, cy) orelse return null;
        if (state == 0 and !d) {
            state = 1;
        } else if (state == 1 and d) break;
    }
    if (x >= lim) return null;
    const r: isize = @intFromFloat(f.module * 6 + 4);
    const x0 = @max(cx - r, 0);
    const y0 = @max(cy - r, 0);
    const x1 = @min(cx + r, @as(isize, @intCast(b.w)) - 1);
    const y1 = @min(cy + r, @as(isize, @intCast(b.h)) - 1);
    const bw: usize = @intCast(x1 - x0 + 1);
    const bh: usize = @intCast(y1 - y0 + 1);
    try ctx.seen.resize(ctx.gpa, bw * bh);
    @memset(ctx.seen.items, 0);
    ctx.stack.clearRetainingCapacity();
    ctx.pts.clearRetainingCapacity();
    try ctx.stack.append(ctx.gpa, .{ @intCast(x - x0), @intCast(cy - y0) });
    ctx.seen.items[@as(usize, @intCast(cy - y0)) * bw + @as(usize, @intCast(x - x0))] = 1;
    while (ctx.stack.pop()) |q| {
        try ctx.pts.append(ctx.gpa, q);
        // Touching the window means the ring is not isolated (or clipped).
        if (q[0] == 0 or q[1] == 0 or q[0] + 1 == bw or q[1] + 1 == bh) return null;
        const nb = [4][2]i32{ .{ -1, 0 }, .{ 1, 0 }, .{ 0, -1 }, .{ 0, 1 } };
        for (nb) |o| {
            const nx: usize = @intCast(@as(i32, q[0]) + o[0]);
            const ny: usize = @intCast(@as(i32, q[1]) + o[1]);
            if (ctx.seen.items[ny * bw + nx] != 0) continue;
            ctx.seen.items[ny * bw + nx] = 1;
            if (b.bits[(@as(usize, @intCast(y0)) + ny) * b.w + @as(usize, @intCast(x0)) + nx] != 0)
                try ctx.stack.append(ctx.gpa, .{ @intCast(nx), @intCast(ny) });
        }
    }
    const pts = ctx.pts.items;
    if (pts.len < 16) return null;
    const ccx = f.x - @as(f32, @floatFromInt(x0)) - 0.5;
    const ccy = f.y - @as(f32, @floatFromInt(y0)) - 0.5;
    var p0: [2]f32 = undefined;
    var best: f32 = -1;
    for (pts) |q| {
        const dx = @as(f32, @floatFromInt(q[0])) - ccx;
        const dy = @as(f32, @floatFromInt(q[1])) - ccy;
        if (dx * dx + dy * dy > best) {
            best = dx * dx + dy * dy;
            p0 = .{ @floatFromInt(q[0]), @floatFromInt(q[1]) };
        }
    }
    var p2: [2]f32 = undefined;
    best = -1;
    for (pts) |q| {
        const dx = @as(f32, @floatFromInt(q[0])) - p0[0];
        const dy = @as(f32, @floatFromInt(q[1])) - p0[1];
        if (dx * dx + dy * dy > best) {
            best = dx * dx + dy * dy;
            p2 = .{ @floatFromInt(q[0]), @floatFromInt(q[1]) };
        }
    }
    const ddx = p2[0] - p0[0];
    const ddy = p2[1] - p0[1];
    var p1 = p0;
    var p3 = p0;
    var cmax: f32 = -std.math.inf(f32);
    var cmin: f32 = std.math.inf(f32);
    for (pts) |q| {
        const qx: f32 = @floatFromInt(q[0]);
        const qy: f32 = @floatFromInt(q[1]);
        const c = (qx - p0[0]) * ddy - (qy - p0[1]) * ddx;
        if (c > cmax) {
            cmax = c;
            p1 = .{ qx, qy };
        }
        if (c < cmin) {
            cmin = c;
            p3 = .{ qx, qy };
        }
    }
    // Pixel centres to the outer pixel corners (edge convention).
    var out: [4]P = undefined;
    for ([4][2]f32{ p0, p1, p2, p3 }, &out) |q, *o| {
        const dx = q[0] - ccx;
        const dy = q[1] - ccy;
        const len = @max(std.math.hypot(dx, dy), 1e-3);
        o.* = .{
            .x = q[0] + @as(f32, @floatFromInt(x0)) + 0.5 + 0.7071 * dx / len,
            .y = q[1] + @as(f32, @floatFromInt(y0)) + 0.5 + 0.7071 * dy / len,
        };
    }
    return out;
}

/// Refines a homography with the finders' outer corners (and an alignment
/// centre), assigned to their module positions through `h0`.
fn refineHomography(ctx: *Ctx, h0: [9]f64, dim: usize, corners: [3]?[4]P, al: ?P) Allocator.Error!?[9]f64 {
    const inv = geom.invert3(h0) orelse return null;
    var src: [20]Point2f = undefined;
    var dst: [20]Point2f = undefined;
    var n: usize = 0;
    const fd: f32 = @floatFromInt(dim);
    const offs = [3][2]f32{ .{ 0, 0 }, .{ fd - 7, 0 }, .{ 0, fd - 7 } };
    for (corners, offs) |cq, off| {
        const q = cq orelse continue;
        var used: [4]bool = @splat(false);
        var src_k: [4]Point2f = undefined;
        var ok = true;
        for (q, 0..) |pt, k| {
            const m = apply(inv, pt.x, pt.y);
            const ix: f32 = if (m.x - off[0] > 3.5) 7 else 0;
            const iy: f32 = if (m.y - off[1] > 3.5) 7 else 0;
            const slot: usize = @as(usize, @intFromFloat(ix / 7)) + 2 * @as(usize, @intFromFloat(iy / 7));
            if (used[slot] or @abs(m.x - off[0] - ix) > 2.5 or @abs(m.y - off[1] - iy) > 2.5) {
                ok = false;
                break;
            }
            used[slot] = true;
            src_k[k] = .{ .x = off[0] + ix, .y = off[1] + iy };
        }
        if (!ok) continue;
        for (q, src_k) |pt, sk| {
            src[n] = sk;
            dst[n] = .{ .x = pt.x, .y = pt.y };
            n += 1;
        }
    }
    if (n < 8) return null;
    if (al) |a| {
        for (0..4) |_| {
            src[n] = .{ .x = fd - 6.5, .y = fd - 6.5 };
            dst[n] = .{ .x = a.x, .y = a.y };
            n += 1;
        }
    }
    return homography.findHomography(ctx.gpa, src[0..n], dst[0..n], null, .{}) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
}

const Attempt = struct {
    h: [9]f64,
    dim: usize,
    decoded: bool,
    info: ?qr_decode.Info,
    timing: f32,
};

/// Tries to read the symbol of a finder triple. Returns the best attempt
/// (decoded, or the best-timed undecoded one), or null.
fn readTriple(ctx: *Ctx, f: [3]Finder, want_decode: bool) Allocator.Error!?Attempt {
    const geo = geometry(ctx.b, f) orelse return null;
    // The finders in the order top-left, top-right, bottom-left.
    var fo: [3]Finder = undefined;
    for ([3]P{ geo.tl, geo.tr, geo.bl }, &fo) |p, *o| {
        for (f) |q| {
            if (q.x == p.x and q.y == p.y) o.* = q;
        }
    }
    var corners: [3]?[4]P = undefined;
    for (fo, &corners) |q, *c| c.* = try finderCorners(ctx, q);
    if (debug) std.debug.print("  triple tl=({d:.1},{d:.1}) tr=({d:.1},{d:.1}) bl=({d:.1},{d:.1}) m={d:.2} dim0={d} corners={any}\n", .{ geo.tl.x, geo.tl.y, geo.tr.x, geo.tr.y, geo.bl.x, geo.bl.y, geo.module, geo.dim0, corners });
    var best: ?Attempt = null;
    const dims = [3]i32{ 0, -4, 4 };
    for (dims) |dd| {
        const di: i32 = @as(i32, @intCast(geo.dim0)) + dd;
        if (di < 21 or di > 177) continue;
        const dim: usize = @intCast(di);
        const fd: f64 = @floatFromInt(dim);
        const h_aff = homographyFor(geo, dim, null) orelse continue;
        const h_fit = try refineHomography(ctx, h_aff, dim, corners, null);
        // Alignment pattern (version 2+), predicted by the best model.
        var al: ?P = null;
        if (dim > 21) {
            const pred = apply(h_fit orelse h_aff, fd - 6.5, fd - 6.5);
            for ([_]f32{ 3, 6, 12 }) |allow| {
                al = findAlignment(ctx.b, pred.x, pred.y, geo.module, allow);
                if (al != null) break;
            }
        }
        var hs: [4]?[9]f64 = .{ null, null, null, null };
        if (al != null) {
            hs[0] = try refineHomography(ctx, h_aff, dim, corners, al);
            hs[1] = homographyFor(geo, dim, al);
        }
        hs[2] = h_fit;
        hs[3] = h_aff;
        for (hs) |hopt| {
            const h = hopt orelse continue;
            if (!sampleGrid(ctx, h, dim)) continue;
            for (0..2) |gi| {
                threshModules(ctx, dim, gi == 1);
                const ts = timingScore(ctx.mods[0 .. dim * dim], dim);
                if (debug) std.debug.print("    dim={d} al={any} timing={d:.2}\n", .{ dim, al, ts });
                if (want_decode) {
                    if (try qr_decode.decode(ctx.gpa, ctx.mods[0 .. dim * dim], dim, &ctx.text)) |info| {
                        return .{ .h = h, .dim = dim, .decoded = true, .info = info, .timing = ts };
                    }
                }
                if (best == null or ts > best.?.timing) best = .{ .h = h, .dim = dim, .decoded = false, .info = null, .timing = ts };
                if (!want_decode and ts >= 0.9) return best;
            }
        }
        if (!want_decode and best != null and best.?.timing >= 0.75) return best;
    }
    if (best) |bb| if (bb.timing >= 0.75) return bb;
    return null;
}

const Triple = struct { i: [3]u16, score: f32 };

fn scoreTriples(gpa: Allocator, fs: []const Finder) Allocator.Error![]Triple {
    var out: std.ArrayList(Triple) = .empty;
    errdefer out.deinit(gpa);
    const n = fs.len;
    for (0..n) |i| for (i + 1..n) |j| for (j + 1..n) |k| {
        const f = [3]Finder{ fs[i], fs[j], fs[k] };
        const mmax = @max(f[0].module, f[1].module, f[2].module);
        const mmin = @min(f[0].module, f[1].module, f[2].module);
        if (mmax > mmin * 1.8) continue;
        const p = [3]P{ .{ .x = f[0].x, .y = f[0].y }, .{ .x = f[1].x, .y = f[1].y }, .{ .x = f[2].x, .y = f[2].y } };
        const d = [3]f32{ p[1].dist(p[2]), p[0].dist(p[2]), p[0].dist(p[1]) };
        var a: usize = 0;
        if (d[1] > d[a]) a = 1;
        if (d[2] > d[a]) a = 2;
        const o1 = (a + 1) % 3;
        const o2 = (a + 2) % 3;
        const l1 = d[o2]; // |a - o1|
        const l2 = d[o1]; // |a - o2|
        if (l1 < 1 or l2 < 1) continue;
        const v1x = p[o1].x - p[a].x;
        const v1y = p[o1].y - p[a].y;
        const v2x = p[o2].x - p[a].x;
        const v2y = p[o2].y - p[a].y;
        const cosang = @abs(v1x * v2x + v1y * v2y) / (l1 * l2);
        if (cosang > 0.5) continue;
        const ratio = @max(l1, l2) / @min(l1, l2);
        if (ratio > 2.2) continue;
        const mavg = (f[0].module + f[1].module + f[2].module) / 3;
        const est = (l1 + l2) / 2 / mavg + 7;
        if (est < 15 or est > 200) continue;
        const score = cosang + @log(ratio) + 0.5 * @log(mmax / mmin);
        try out.append(gpa, .{ .i = .{ @intCast(i), @intCast(j), @intCast(k) }, .score = score });
    };
    std.mem.sort(Triple, out.items, {}, struct {
        fn lt(_: void, x: Triple, y: Triple) bool {
            return x.score < y.score;
        }
    }.lt);
    return out.toOwnedSlice(gpa);
}

pub const Options = struct {
    /// Find every symbol (detectAndDecodeMulti) rather than one.
    multi: bool = false,
    /// Decode the symbols (detect alone otherwise).
    decode: bool = true,
};

/// QR codes in a 1-, 3- (BGR) or 4-channel 8-bit image. Free the result
/// with `freeCodes`.
pub fn detect(gpa: Allocator, src: Image(u8), o: Options) Error![]Code {
    var gray = src;
    var owned: ?Image(u8) = null;
    defer if (owned) |*im| im.deinit(gpa);
    switch (src.channels) {
        1 => {},
        3, 4 => {
            owned = try Image(u8).init(gpa, src.width, src.height, 1);
            try color.cvtColor(core.Exec.serial, src, owned.?, if (src.channels == 3) .bgr2gray else .bgra2gray);
            gray = owned.?;
        },
        else => return error.Unsupported,
    }
    var out: std.ArrayList(Code) = .empty;
    errdefer {
        for (out.items) |c| gpa.free(c.text);
        out.deinit(gpa);
    }
    if (gray.width < 21 or gray.height < 21) return out.toOwnedSlice(gpa);

    var bin = try binarize(gpa, gray);
    defer bin.deinit(gpa);
    const all = try findFinders(gpa, &bin);
    defer gpa.free(all);
    // Strongest candidates first; at most 40 of them.
    std.mem.sort(Finder, all, {}, struct {
        fn gt(_: void, x: Finder, y: Finder) bool {
            return x.count > y.count;
        }
    }.gt);
    var nf = all.len;
    if (nf > 3) {
        var k: usize = 0;
        while (k < nf and all[k].count >= 2) k += 1;
        if (k >= 3) nf = k;
    }
    const fs = all[0..@min(nf, 40)];
    if (debug) for (all) |f| std.debug.print("finder ({d:.1},{d:.1}) m={d:.2} n={d}\n", .{ f.x, f.y, f.module, f.count });
    if (fs.len < 3) return out.toOwnedSlice(gpa);

    const triples = try scoreTriples(gpa, fs);
    defer gpa.free(triples);

    var ctx: Ctx = .{
        .gpa = gpa,
        .g = gray,
        .b = &bin,
        .vals = try gpa.alloc(f32, 177 * 177),
        .thr = try gpa.alloc(f32, 177 * 177),
        .mods = try gpa.alloc(u8, 177 * 177),
    };
    defer {
        gpa.free(ctx.vals);
        gpa.free(ctx.thr);
        gpa.free(ctx.mods);
        ctx.text.deinit(gpa);
        ctx.seen.deinit(gpa);
        ctx.stack.deinit(gpa);
        ctx.pts.deinit(gpa);
    }

    var used: [40]bool = @splat(false);
    var undecoded: ?Code = null;
    const max_tries: usize = if (o.multi) triples.len else @min(triples.len, 12);
    for (triples[0..max_tries]) |t| {
        if (used[t.i[0]] or used[t.i[1]] or used[t.i[2]]) continue;
        const f = [3]Finder{ fs[t.i[0]], fs[t.i[1]], fs[t.i[2]] };
        const att = (try readTriple(&ctx, f, o.decode)) orelse continue;
        var corners: [4]Point2f = undefined;
        const fd: f64 = @floatFromInt(att.dim);
        for ([4][2]f64{ .{ 0, 0 }, .{ fd, 0 }, .{ fd, fd }, .{ 0, fd } }, 0..) |q, ci| {
            const p = apply(att.h, q[0], q[1]);
            corners[ci] = .{ .x = p.x - 0.5, .y = p.y - 0.5 };
        }
        const code: Code = .{
            .corners = corners,
            .text = if (att.decoded) try gpa.dupe(u8, ctx.text.items) else try gpa.alloc(u8, 0),
            .decoded = att.decoded,
            .info = att.info,
        };
        if (!o.multi) {
            if (att.decoded or !o.decode) {
                if (undecoded) |u| gpa.free(u.text);
                undecoded = null;
                try out.append(gpa, code);
                break;
            }
            if (undecoded == null) undecoded = code else gpa.free(code.text);
            continue;
        }
        for (t.i) |ii| used[ii] = true;
        try out.append(gpa, code);
    }
    if (out.items.len == 0) {
        if (undecoded) |u| try out.append(gpa, u);
    } else if (undecoded) |u| gpa.free(u.text);

    if (o.multi and o.decode) try mergeStructuredAppend(gpa, out.items);
    return out.toOwnedSlice(gpa);
}

/// decodeMulti's structured append handling: the full message goes to the
/// sequence's first symbol, the others get empty strings.
fn mergeStructuredAppend(gpa: Allocator, codes: []Code) Allocator.Error!void {
    const n = codes.len;
    const merged = try gpa.alloc(?[]u8, n);
    defer gpa.free(merged);
    @memset(merged, null);
    errdefer for (merged) |m| if (m) |s| gpa.free(s);
    for (codes, 0..) |c, i| {
        const info = c.info orelse continue;
        if (!c.decoded or info.mode != 3) continue;
        if (info.sequence_num != 0) {
            merged[i] = try gpa.alloc(u8, 0);
            continue;
        }
        var buf: std.ArrayList(u8) = .empty;
        errdefer buf.deinit(gpa);
        try buf.appendSlice(gpa, c.text);
        var complete = true;
        for (1..info.total_num) |idx| {
            const found = for (codes) |d| {
                const di = d.info orelse continue;
                if (d.decoded and di.mode == 3 and di.parity == info.parity and di.sequence_num == idx) break d;
            } else null;
            if (found) |d| try buf.appendSlice(gpa, d.text) else {
                complete = false;
                break;
            }
        }
        if (!complete) buf.clearRetainingCapacity();
        merged[i] = try buf.toOwnedSlice(gpa);
    }
    for (codes, merged) |*c, m| if (m) |s| {
        gpa.free(c.text);
        c.text = s;
    };
}

test "qr detect: synthetic symbol" {
    const gpa = std.testing.allocator;
    const enc = @import("qr_encode.zig");
    var q = try enc.encode(gpa, "VISAR objdetect", .{ .level = .M });
    defer q.deinit(gpa);
    const scale = 6;
    const side = q.width * scale + 40;
    var img = try Image(u8).initFill(gpa, side, side, 1, 255);
    defer img.deinit(gpa);
    for (0..q.height * scale) |y| for (0..q.width * scale) |x| {
        img.data[(y + 20) * img.stride + x + 20] = q.data[(y / scale) * q.stride + x / scale];
    };
    const codes = try detect(gpa, img, .{});
    defer freeCodes(gpa, codes);
    try std.testing.expectEqual(@as(usize, 1), codes.len);
    try std.testing.expectEqualStrings("VISAR objdetect", codes[0].text);
    try std.testing.expect(@abs(codes[0].corners[0].x - (20 + 2 * scale)) < 1.5);

    // Noise never panics.
    var prng = std.Random.DefaultPrng.init(3);
    prng.random().bytes(img.data[0 .. img.stride * img.height]);
    const junk = try detect(gpa, img, .{ .multi = true });
    freeCodes(gpa, junk);
}
