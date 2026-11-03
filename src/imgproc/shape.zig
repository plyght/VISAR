//! Shape analysis, ported from OpenCV 5's geometry module so results match
//! it: moments (polygon and raster), Hu moments, contourArea, arcLength,
//! boundingRect, convexHull, convexityDefects, isContourConvex,
//! minAreaRect, boxPoints, minEnclosingCircle, fitEllipse, approxPolyDP,
//! pointPolygonTest and matchShapes.
//!
//! Point sets are slices of `Point` (i32) or `Point2f` (f32), like OpenCV's
//! CV_32SC2 / CV_32FC2 inputs. The arithmetic follows OpenCV's operation
//! order and precision (float vs double) so results agree bit for bit
//! where OpenCV is deterministic.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
pub const Point = @import("contours.zig").Point;

pub const Point2f = extern struct { x: f32, y: f32 };

pub const Rect = extern struct { x: i32, y: i32, width: i32, height: i32 };

/// OpenCV's RotatedRect: centre, size and angle in degrees.
pub const RotatedRect = extern struct { cx: f32, cy: f32, width: f32, height: f32, angle: f32 };

/// cv::Moments: spatial (m), central (mu) and normalised central (nu).
pub const Moments = extern struct {
    m00: f64 = 0,
    m10: f64 = 0,
    m01: f64 = 0,
    m20: f64 = 0,
    m11: f64 = 0,
    m02: f64 = 0,
    m30: f64 = 0,
    m21: f64 = 0,
    m12: f64 = 0,
    m03: f64 = 0,
    mu20: f64 = 0,
    mu11: f64 = 0,
    mu02: f64 = 0,
    mu30: f64 = 0,
    mu21: f64 = 0,
    mu12: f64 = 0,
    mu03: f64 = 0,
    nu20: f64 = 0,
    nu11: f64 = 0,
    nu02: f64 = 0,
    nu30: f64 = 0,
    nu21: f64 = 0,
    nu12: f64 = 0,
    nu03: f64 = 0,
};

fn Coord(comptime S: type) type {
    return @FieldType(std.meta.Elem(S), "x");
}

inline fn toF(x: anytype) f32 {
    return if (@TypeOf(x) == f32) x else @floatFromInt(x);
}

inline fn toD(x: anytype) f64 {
    return if (@TypeOf(x) == f32) x else @floatFromInt(x);
}

/// Subtraction in the coordinate type (wrapping for integers, like C).
inline fn sub(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@TypeOf(a) == f32) a - b else a -% b;
}

inline fn mul(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@TypeOf(a) == f32) a * b else a *% b;
}

inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return if (@TypeOf(a) == f32) a + b else a +% b;
}

inline fn pf(p: anytype) Point2f {
    return .{ .x = toF(p.x), .y = toF(p.y) };
}

const flt_eps: f64 = std.math.floatEps(f32);
const dbl_eps: f64 = std.math.floatEps(f64);

// ---------------------------------------------------------------- moments

fn completeMomentState(m: *Moments) void {
    var cx: f64 = 0;
    var cy: f64 = 0;
    var inv_m00: f64 = 0;
    if (@abs(m.m00) > dbl_eps) {
        inv_m00 = 1.0 / m.m00;
        cx = m.m10 * inv_m00;
        cy = m.m01 * inv_m00;
    }
    const mu20 = m.m20 - m.m10 * cx;
    var mu11 = m.m11 - m.m10 * cy;
    const mu02 = m.m02 - m.m01 * cy;
    m.mu20 = mu20;
    m.mu11 = mu11;
    m.mu02 = mu02;
    m.mu30 = m.m30 - cx * (3 * mu20 + cx * m.m10);
    mu11 += mu11;
    m.mu21 = m.m21 - cx * (mu11 + cx * m.m01) - cy * mu20;
    m.mu12 = m.m12 - cy * (mu11 + cy * m.m10) - cx * mu02;
    m.mu03 = m.m03 - cy * (3 * mu02 + cy * m.m01);
    const inv_sqrt_m00 = @sqrt(@abs(inv_m00));
    const s2 = inv_m00 * inv_m00;
    const s3 = s2 * inv_sqrt_m00;
    m.nu20 = m.mu20 * s2;
    m.nu11 = m.mu11 * s2;
    m.nu02 = m.mu02 * s2;
    m.nu30 = m.mu30 * s3;
    m.nu21 = m.mu21 * s3;
    m.nu12 = m.mu12 * s3;
    m.nu03 = m.mu03 * s3;
}

/// moments() of a polygon (Green's theorem over the closed point list).
pub fn contourMoments(pts: anytype) Moments {
    var m: Moments = .{};
    const n = pts.len;
    if (n == 0) return m;
    var a00: f64 = 0;
    var a10: f64 = 0;
    var a01: f64 = 0;
    var a20: f64 = 0;
    var a11: f64 = 0;
    var a02: f64 = 0;
    var a30: f64 = 0;
    var a21: f64 = 0;
    var a12: f64 = 0;
    var a03: f64 = 0;
    var xi_1 = toD(pts[n - 1].x);
    var yi_1 = toD(pts[n - 1].y);
    var xi_12 = xi_1 * xi_1;
    var yi_12 = yi_1 * yi_1;
    for (pts) |p| {
        const xi = toD(p.x);
        const yi = toD(p.y);
        const xi2 = xi * xi;
        const yi2 = yi * yi;
        const dxy = xi_1 * yi - xi * yi_1;
        const xii_1 = xi_1 + xi;
        const yii_1 = yi_1 + yi;
        a00 += dxy;
        a10 += dxy * xii_1;
        a01 += dxy * yii_1;
        a20 += dxy * (xi_1 * xii_1 + xi2);
        a11 += dxy * (xi_1 * (yii_1 + yi_1) + xi * (yii_1 + yi));
        a02 += dxy * (yi_1 * yii_1 + yi2);
        a30 += dxy * xii_1 * (xi_12 + xi2);
        a03 += dxy * yii_1 * (yi_12 + yi2);
        a21 += dxy * (xi_12 * (3 * yi_1 + yi) + 2 * xi * xi_1 * yii_1 + xi2 * (yi_1 + 3 * yi));
        a12 += dxy * (yi_12 * (3 * xi_1 + xi) + 2 * yi * yi_1 * xii_1 + yi2 * (xi_1 + 3 * xi));
        xi_1 = xi;
        yi_1 = yi;
        xi_12 = xi2;
        yi_12 = yi2;
    }
    if (@abs(a00) > flt_eps) {
        const sgn: f64 = if (a00 > 0) 1 else -1;
        const db1_2 = 0.5 * sgn;
        const db1_6 = 0.16666666666666666666666666666667 * sgn;
        const db1_12 = 0.083333333333333333333333333333333 * sgn;
        const db1_24 = 0.041666666666666666666666666666667 * sgn;
        const db1_20 = 0.05 * sgn;
        const db1_60 = 0.016666666666666666666666666666667 * sgn;
        m.m00 = a00 * db1_2;
        m.m10 = a10 * db1_6;
        m.m01 = a01 * db1_6;
        m.m20 = a20 * db1_12;
        m.m11 = a11 * db1_24;
        m.m02 = a02 * db1_12;
        m.m30 = a30 * db1_20;
        m.m21 = a21 * db1_60;
        m.m12 = a12 * db1_60;
        m.m03 = a03 * db1_20;
        completeMomentState(&m);
    }
    return m;
}

const tile = 32;

/// The ten raw moments of one tile, in OpenCV's momentsInTile order
/// (m00 m10 m01 m20 m11 m02 m30 m21 m12 m03).
fn tileMoments(comptime T: type, img: Image(T), x0: usize, y0: usize, w: usize, h: usize, binary: bool) [10]f64 {
    const integer = T != f32 and T != f64;
    if (integer or binary) {
        var mom: [10]i64 = @splat(0);
        const V = @Vector(tile, u32);
        const ix: V = comptime blk: {
            var v: [tile]u32 = undefined;
            for (&v, 0..) |*e, i| e.* = i;
            break :blk v;
        };
        const ix2 = ix * ix;
        const ix3 = ix2 * ix;
        for (0..h) |yy| {
            const row = img.row(y0 + yy)[x0..][0..w];
            const y: i64 = @intCast(yy);
            var s0: i64 = undefined;
            var s1: i64 = undefined;
            var s2: i64 = undefined;
            var s3: i64 = undefined;
            if (T == u8 or binary) {
                // 8-bit (or 0/255) samples: every per-row sum fits in u32
                // (x^3 * p <= 31^3 * 255 per pixel).
                var p: V = @splat(0);
                if (T == u8 and w == tile) {
                    const bv: @Vector(tile, u8) = row[0..tile].*;
                    p = if (binary) @select(u32, bv != @as(@Vector(tile, u8), @splat(0)), @as(V, @splat(255)), @as(V, @splat(0))) else bv;
                } else {
                    var buf: [tile]u32 = @splat(0);
                    for (row, 0..) |v, i| buf[i] = if (binary) (if (v != 0) 255 else 0) else if (T == u8) v else 0;
                    p = buf;
                }
                s0 = @reduce(.Add, p);
                s1 = @reduce(.Add, p * ix);
                s2 = @reduce(.Add, p * ix2);
                s3 = @reduce(.Add, p * ix3);
            } else if (integer) {
                const L = @Vector(tile, i64);
                var buf: [tile]i64 = @splat(0);
                for (row, 0..) |v, i| buf[i] = v;
                const p: L = buf;
                s0 = @reduce(.Add, p);
                s1 = @reduce(.Add, p * @as(L, ix));
                s2 = @reduce(.Add, p * @as(L, ix2));
                s3 = @reduce(.Add, p * @as(L, ix3));
            } else unreachable;
            const py = y * s0;
            const sy = y * y;
            mom[9] += py * sy;
            mom[8] += s1 * sy;
            mom[7] += s2 * y;
            mom[6] += s3;
            mom[5] += s0 * sy;
            mom[4] += s1 * y;
            mom[3] += s2;
            mom[2] += py;
            mom[1] += s1;
            mom[0] += s0;
        }
        var out: [10]f64 = undefined;
        for (&out, mom) |*o, v| o.* = @floatFromInt(v);
        if (binary) {
            const s: f64 = 1.0 / 255.0;
            for (&out) |*o| o.* *= s;
        }
        return out;
    } else {
        var mom: [10]f64 = @splat(0);
        for (0..h) |yy| {
            const row = img.row(y0 + yy)[x0..][0..w];
            var s0: f64 = 0;
            var s1: f64 = 0;
            var s2: f64 = 0;
            var s3: f64 = 0;
            for (row, 0..) |v, xi| {
                const p: f64 = v;
                const x: f64 = @floatFromInt(xi);
                const xp = x * p;
                s0 += p;
                s1 += xp;
                const xxp = xp * x;
                s2 += xxp;
                s3 += xxp * x;
            }
            const y: f64 = @floatFromInt(yy);
            const py = y * s0;
            const sy = y * y;
            mom[9] += py * sy;
            mom[8] += s1 * sy;
            mom[7] += s2 * y;
            mom[6] += s3;
            mom[5] += s0 * sy;
            mom[4] += s1 * y;
            mom[3] += s2;
            mom[2] += py;
            mom[1] += s1;
            mom[0] += s0;
        }
        return mom;
    }
}

/// moments() of a single-channel image (u8, u16, i16 or f32). With
/// `binary`, every nonzero pixel counts as 1. Tiles are computed in
/// parallel and summed in OpenCV's order, so the result does not depend on
/// the thread count.
pub fn imageMoments(comptime T: type, exec: Exec, gpa: Allocator, img: Image(T), binary: bool) (Allocator.Error || core.Error)!Moments {
    if (img.channels != 1) return error.UnsupportedChannels;
    var m: Moments = .{};
    const tw = (@as(usize, img.width) + tile - 1) / tile;
    const th = (@as(usize, img.height) + tile - 1) / tile;
    const moms = try gpa.alloc([10]f64, tw * th);
    defer gpa.free(moms);
    const Ctx = struct {
        img: Image(T),
        moms: [][10]f64,
        tw: usize,
        binary: bool,
        fn run(c: @This(), start: usize, end: usize) void {
            for (start..end) |ty| {
                const y0 = ty * tile;
                const h = @min(tile, c.img.height - y0);
                for (0..c.tw) |tx| {
                    const x0 = tx * tile;
                    const w = @min(tile, c.img.width - x0);
                    c.moms[ty * c.tw + tx] = tileMoments(T, c.img, x0, y0, w, h, c.binary);
                }
            }
        }
    };
    exec.parallelFor(th, 2, Ctx{ .img = img, .moms = moms, .tw = tw, .binary = binary }, Ctx.run);
    for (0..th) |ty| for (0..tw) |tx| {
        const mom = moms[ty * tw + tx];
        const x: f64 = @floatFromInt(tx * tile);
        const y: f64 = @floatFromInt(ty * tile);
        const xm = x * mom[0];
        const ym = y * mom[0];
        m.m00 += mom[0];
        m.m10 += mom[1] + xm;
        m.m01 += mom[2] + ym;
        m.m20 += mom[3] + x * (mom[1] * 2 + xm);
        m.m11 += mom[4] + x * (mom[2] + ym) + y * mom[1];
        m.m02 += mom[5] + y * (mom[2] * 2 + ym);
        m.m30 += mom[6] + x * (3.0 * mom[3] + x * (3.0 * mom[1] + xm));
        m.m21 += mom[7] + x * (2 * (mom[4] + y * mom[1]) + x * (mom[2] + ym)) + y * mom[3];
        m.m12 += mom[8] + y * (2 * (mom[4] + x * mom[2]) + y * (mom[1] + xm)) + x * mom[5];
        m.m03 += mom[9] + y * (3.0 * mom[5] + y * (3.0 * mom[2] + ym));
    };
    completeMomentState(&m);
    return m;
}

/// HuMoments: the seven rotation-invariant moments.
pub fn huMoments(m: Moments) [7]f64 {
    var hu: [7]f64 = undefined;
    var t0 = m.nu30 + m.nu12;
    var t1 = m.nu21 + m.nu03;
    var q0 = t0 * t0;
    var q1 = t1 * t1;
    const n4 = 4 * m.nu11;
    const s = m.nu20 + m.nu02;
    const d = m.nu20 - m.nu02;
    hu[0] = s;
    hu[1] = d * d + n4 * m.nu11;
    hu[3] = q0 + q1;
    hu[5] = d * (q0 - q1) + n4 * t0 * t1;
    t0 *= q0 - 3 * q1;
    t1 *= 3 * q0 - q1;
    q0 = m.nu30 - 3 * m.nu12;
    q1 = 3 * m.nu21 - m.nu03;
    hu[2] = q0 * q0 + q1 * q1;
    hu[4] = q0 * t0 + q1 * t1;
    hu[6] = q1 * t0 - q0 * t1;
    return hu;
}

/// matchShapes (CONTOURS_MATCH_I1 = 1, I2 = 2, I3 = 3) from two sets of
/// moments.
pub fn matchShapes(ma_m: Moments, mb_m: Moments, method: u8) error{InvalidArgument}!f64 {
    if (method < 1 or method > 3) return error.InvalidArgument;
    const ma = huMoments(ma_m);
    const mb = huMoments(mb_m);
    const eps = 1.0e-5;
    var result: f64 = 0;
    var any_a = false;
    var any_b = false;
    for (0..7) |i| {
        var ama = @abs(ma[i]);
        var amb = @abs(mb[i]);
        if (ama > 0) any_a = true;
        if (amb > 0) any_b = true;
        const sma: f64 = if (ma[i] > 0) 1 else if (ma[i] < 0) -1 else 0;
        const smb: f64 = if (mb[i] > 0) 1 else if (mb[i] < 0) -1 else 0;
        if (ama > eps and amb > eps) {
            switch (method) {
                1 => {
                    ama = 1.0 / (sma * std.math.log10(ama));
                    amb = 1.0 / (smb * std.math.log10(amb));
                    result += @abs(-ama + amb);
                },
                2 => {
                    ama = sma * std.math.log10(ama);
                    amb = smb * std.math.log10(amb);
                    result += @abs(-ama + amb);
                },
                else => {
                    ama = sma * std.math.log10(ama);
                    amb = smb * std.math.log10(amb);
                    const mmm = @abs((ama - amb) / ama);
                    if (result < mmm) result = mmm;
                },
            }
        }
    }
    if (any_a != any_b) result = std.math.floatMax(f64);
    return result;
}

// ------------------------------------------------------ area and length

/// contourArea: the polygon's area (signed when `oriented`).
pub fn contourArea(pts: anytype, oriented: bool) f64 {
    const n = pts.len;
    if (n == 0) return 0;
    var a00: f64 = 0;
    var prev = pf(pts[n - 1]);
    for (pts) |q| {
        const p = pf(q);
        a00 += @as(f64, prev.x) * p.y - @as(f64, prev.y) * p.x;
        prev = p;
    }
    a00 *= 0.5;
    return if (oriented) a00 else @abs(a00);
}

/// arcLength: the perimeter (closed) or length (open) of a curve.
pub fn arcLength(pts: anytype, closed: bool) f64 {
    const n = pts.len;
    if (n <= 1) return 0;
    var perimeter: f64 = 0;
    var prev = pf(pts[if (closed) n - 1 else 0]);
    for (pts) |q| {
        const p = pf(q);
        const dx = p.x - prev.x;
        const dy = p.y - prev.y;
        const len: f32 = @sqrt(dx * dx + dy * dy);
        perimeter += len;
        prev = p;
    }
    return perimeter;
}

// ------------------------------------------------------ bounding rects

fn floorSaturate(v: f32) i32 {
    const d = @floor(@as(f64, v));
    if (!(d > @as(f64, std.math.minInt(i32)))) return std.math.minInt(i32);
    if (d >= @as(f64, std.math.maxInt(i32))) return std.math.maxInt(i32);
    return @intFromFloat(d);
}

/// boundingRect of a point set.
pub fn boundingRect(pts: anytype) Rect {
    if (pts.len == 0) return .{ .x = 0, .y = 0, .width = 0, .height = 0 };
    var xmin: i32 = undefined;
    var xmax: i32 = undefined;
    var ymin: i32 = undefined;
    var ymax: i32 = undefined;
    if (Coord(@TypeOf(pts)) == f32) {
        var a = pts[0].x;
        var b = pts[0].x;
        var c = pts[0].y;
        var d = pts[0].y;
        for (pts) |p| {
            a = @min(a, p.x);
            b = @max(b, p.x);
            c = @min(c, p.y);
            d = @max(d, p.y);
        }
        xmin = floorSaturate(a);
        xmax = floorSaturate(b);
        ymin = floorSaturate(c);
        ymax = floorSaturate(d);
    } else {
        xmin = pts[0].x;
        xmax = pts[0].x;
        ymin = pts[0].y;
        ymax = pts[0].y;
        for (pts) |p| {
            xmin = @min(xmin, p.x);
            xmax = @max(xmax, p.x);
            ymin = @min(ymin, p.y);
            ymax = @max(ymax, p.y);
        }
    }
    const w = @as(i64, xmax) - xmin + 1;
    const h = @as(i64, ymax) - ymin + 1;
    return .{ .x = xmin, .y = ymin, .width = @intCast(@min(w, std.math.maxInt(i32))), .height = @intCast(@min(h, std.math.maxInt(i32))) };
}

/// boundingRect of the nonzero pixels of an 8-bit mask ((0,0,0,0) when
/// there are none).
pub fn maskBoundingRect(img: Image(u8)) Rect {
    var xmin: usize = img.width;
    var xmax: usize = 0;
    var ymin: usize = img.height;
    var ymax: usize = 0;
    var any = false;
    const N = 32;
    const V = @Vector(N, u8);
    const zero: V = @splat(0);
    for (0..img.height) |y| {
        const row = img.row(y);
        // First nonzero from the left, skipping zero chunks.
        var x: usize = 0;
        while (x + N <= row.len and !@reduce(.Or, @as(V, row[x..][0..N].*) != zero)) x += N;
        while (x < row.len and row[x] == 0) x += 1;
        if (x == row.len) continue;
        any = true;
        if (ymin == img.height) ymin = y;
        ymax = y;
        xmin = @min(xmin, x);
        var r = row.len;
        while (r >= x + N and !@reduce(.Or, @as(V, row[r - N ..][0..N].*) != zero)) r -= N;
        while (row[r - 1] == 0) r -= 1;
        xmax = @max(xmax, r - 1);
    }
    if (!any) return .{ .x = 0, .y = 0, .width = 0, .height = 0 };
    return .{ .x = @intCast(xmin), .y = @intCast(ymin), .width = @intCast(xmax - xmin + 1), .height = @intCast(ymax - ymin + 1) };
}

// ------------------------------------------------------------ convex hull

fn sign(x: anytype) i32 {
    return @as(i32, @intFromBool(x > 0)) - @as(i32, @intFromBool(x < 0));
}

fn sklansky(comptime T: type, pts: anytype, order: []const u32, start: i32, end_in: i32, stack: []i32, nsign: i32, sign2: i32) usize {
    const incr: i32 = if (end_in > start) 1 else -1;
    var pprev = start;
    var pcur = pprev + incr;
    var pnext = pcur + incr;
    var stacksize: usize = 3;
    const P = struct {
        fn at(p: anytype, o: []const u32, i: i32) std.meta.Elem(@TypeOf(p)) {
            return p[o[@intCast(i)]];
        }
    };
    const ps = P.at(pts, order, start);
    const pe = P.at(pts, order, end_in);
    if (start == end_in or (ps.x == pe.x and ps.y == pe.y)) {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;
    const end = end_in + incr;
    while (pnext != end) {
        const cur = P.at(pts, order, pcur);
        const nxt = P.at(pts, order, pnext);
        const prv = P.at(pts, order, pprev);
        const by = sub(nxt.y, cur.y);
        if (sign(by) != nsign) {
            var a0 = sub(cur.x, prv.x);
            var a1 = sub(cur.y, prv.y);
            var b0 = sub(nxt.x, cur.x);
            var b1 = by;
            const convexity = if (T == f32) blk: {
                const na = @sqrt(@as(f64, a0) * a0 + @as(f64, a1) * a1);
                const sa: f64 = if (na != 0) 1.0 / na else 0;
                a0 = @floatCast(a0 * sa);
                a1 = @floatCast(a1 * sa);
                const nb = @sqrt(@as(f64, b0) * b0 + @as(f64, b1) * b1);
                const sb: f64 = if (nb != 0) 1.0 / nb else 0;
                b0 = @floatCast(b0 * sb);
                b1 = @floatCast(b1 * sb);
                break :blk @as(f64, a1) * b0 - @as(f64, a0) * b1;
            } else @as(i64, a1) * b0 - @as(i64, a0) * b1;
            if (sign(convexity) == sign2 and (a0 != 0 or a1 != 0)) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize += 1;
            } else if (pprev == start) {
                pcur = pnext;
                stack[1] = pcur;
                pnext += incr;
                stack[2] = pnext;
            } else {
                stack[stacksize - 2] = pnext;
                pcur = pprev;
                pprev = stack[stacksize - 4];
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext;
        }
    }
    return stacksize - 1;
}

/// convexHull: returns the indices (into `pts`) of the hull vertices, in
/// OpenCV's order and orientation. `return_points` selects OpenCV's
/// returnPoints mode, which skips the index-monotonicity fix-up; take
/// `pts[i]` for each index to get the hull points. Caller frees.
pub fn convexHull(gpa: Allocator, pts: anytype, clockwise: bool, return_points: bool) Allocator.Error![]u32 {
    const T = Coord(@TypeOf(pts));
    const total = pts.len;
    if (total == 0) return gpa.alloc(u32, 0);
    const order = try gpa.alloc(u32, total);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const Cmp = struct {
        fn less(p: @TypeOf(pts), a: u32, b: u32) bool {
            const pa = p[a];
            const pb = p[b];
            if (pa.x != pb.x) return pa.x < pb.x;
            if (pa.y != pb.y) return pa.y < pb.y;
            return a < b;
        }
    };
    if (T == i32) {
        // (x, y, index) packed into one integer key: same order, cheaper
        // comparisons.
        const keys = try gpa.alloc(u128, total);
        defer gpa.free(keys);
        for (keys, pts, 0..) |*k, p, i| {
            const kx: u128 = @as(u32, @bitCast(p.x)) ^ 0x8000_0000;
            const ky: u128 = @as(u32, @bitCast(p.y)) ^ 0x8000_0000;
            k.* = (kx << 64) | (ky << 32) | i;
        }
        std.sort.pdq(u128, keys, {}, std.sort.asc(u128));
        for (order, keys) |*o, k| o.* = @truncate(k);
    } else std.sort.pdq(u32, order, pts, Cmp.less);
    var miny: usize = 0;
    var maxy: usize = 0;
    for (1..total) |i| {
        const y = pts[order[i]].y;
        if (pts[order[miny]].y > y) miny = i;
        if (pts[order[maxy]].y < y) maxy = i;
    }
    const stack = try gpa.alloc(i32, total + 2);
    defer gpa.free(stack);
    var hull = try gpa.alloc(i32, total);
    defer gpa.free(hull);
    var nout: usize = 0;
    const first = pts[order[0]];
    const last = pts[order[total - 1]];
    if (first.x == last.x and first.y == last.y) {
        hull[0] = 0;
        nout = 1;
    } else {
        const ti: i32 = @intCast(total - 1);
        var tl_off: usize = 0;
        var tl_count = sklansky(T, pts, order, 0, @intCast(maxy), stack, -1, 1);
        var tr_off = tl_count;
        var tr_count = sklansky(T, pts, order, ti, @intCast(maxy), stack[tr_off..], -1, -1);
        if (!clockwise) {
            std.mem.swap(usize, &tl_off, &tr_off);
            std.mem.swap(usize, &tl_count, &tr_count);
        }
        const tl = stack[tl_off..];
        const tr = stack[tr_off..];
        var i: usize = 0;
        while (i + 1 < tl_count) : (i += 1) {
            hull[nout] = tl[i];
            nout += 1;
        }
        i = tr_count -% 1;
        while (i > 0 and i < tr_count) : (i -= 1) {
            hull[nout] = tr[i];
            nout += 1;
        }
        const stop_idx: i32 = if (tr_count > 2) tr[1] else if (tl_count > 2) tl[tl_count - 2] else -1;

        var bl_off: usize = 0;
        var bl_count = sklansky(T, pts, order, 0, @intCast(miny), stack, 1, -1);
        var br_off = bl_count;
        var br_count = sklansky(T, pts, order, ti, @intCast(miny), stack[br_off..], 1, 1);
        if (clockwise) {
            std.mem.swap(usize, &bl_off, &br_off);
            std.mem.swap(usize, &bl_count, &br_count);
        }
        const bl = stack[bl_off..];
        const br = stack[br_off..];
        if (stop_idx >= 0) {
            const check_idx: i32 = if (bl_count > 2) bl[1] else if (bl_count + br_count > 2) br[2 - bl_count] else -1;
            if (check_idx == stop_idx or (check_idx >= 0 and
                pts[order[@intCast(check_idx)]].x == pts[order[@intCast(stop_idx)]].x and
                pts[order[@intCast(check_idx)]].y == pts[order[@intCast(stop_idx)]].y))
            {
                bl_count = @min(bl_count, 2);
                br_count = @min(br_count, 2);
            }
        }
        i = 0;
        while (i + 1 < bl_count) : (i += 1) {
            hull[nout] = bl[i];
            nout += 1;
        }
        i = br_count -% 1;
        while (i > 0 and i < br_count) : (i -= 1) {
            hull[nout] = br[i];
            nout += 1;
        }

        if (!return_points) {
            // Keep indices monotonous in case of self-intersection.
            for (0..nout) |k| {
                const prev = order[@intCast(hull[(if (k == 0) nout else k) - 1])];
                const next = order[@intCast(hull[(k + 1) % nout])];
                var cur = order[@intCast(hull[k])];
                if ((prev < cur and cur < next) or (prev > cur and cur > next)) continue;
                var j: usize = @intCast(hull[k] + 1);
                while (j < total) : (j += 1) {
                    cur = order[j];
                    const a = pts[order[@intCast(hull[k])]];
                    const b = pts[cur];
                    if (a.x == b.x and a.y == b.y) {
                        if ((prev < cur and cur < next) or (prev > cur and cur > next)) {
                            hull[k] = @intCast(j);
                            break;
                        }
                    } else break;
                }
            }
        }
        for (hull[0..nout]) |*h| h.* = @intCast(order[@intCast(h.*)]);

        // Cyclic shift so the indices ascend or descend where possible.
        if (nout >= 3) {
            var min_idx: usize = 0;
            var max_idx: usize = 0;
            var lt: usize = 0;
            var k: usize = 1;
            while (k < nout) : (k += 1) {
                const idx = hull[k];
                lt += @intFromBool(hull[k - 1] < idx);
                if (lt > 1 and lt + 2 <= k) break;
                if (idx < hull[min_idx]) min_idx = k;
                if (idx > hull[max_idx]) max_idx = k;
            }
            const mmdist = if (max_idx > min_idx) max_idx - min_idx else min_idx - max_idx;
            if ((mmdist == 1 or mmdist == nout - 1) and (lt <= 1 or lt + 2 >= nout)) {
                const ascending = (max_idx + 1) % nout == min_idx;
                const start_i = if (ascending) min_idx else max_idx;
                var j = start_i;
                if (start_i > 0) {
                    var q: usize = 0;
                    while (q < nout) : (q += 1) {
                        const curr = hull[j];
                        stack[q] = curr;
                        const next_j = if (j + 1 < nout) j + 1 else 0;
                        const next_idx = hull[next_j];
                        if (q < nout - 1 and (ascending != (curr < next_idx))) break;
                        j = next_j;
                    }
                    if (q == nout) @memcpy(hull[0..nout], stack[0..nout]);
                }
            }
        }
    }
    const out = try gpa.alloc(u32, nout);
    for (out, hull[0..nout]) |*o, h| o.* = @intCast(h);
    return out;
}

/// convexityDefects: [start index, end index, farthest index, depth * 256]
/// for every hull edge with points beyond it. `hull` holds indices as
/// returned by convexHull(return_points = false). Caller frees.
pub fn convexityDefects(gpa: Allocator, pts: []const Point, hull: []const u32) (Allocator.Error || error{InvalidArgument})![][4]i32 {
    const npoints = pts.len;
    var out: std.ArrayList([4]i32) = .empty;
    errdefer out.deinit(gpa);
    if (npoints <= 3 or hull.len < 3) return out.toOwnedSlice(gpa);
    const hp = hull.len;
    const rev = (@as(u8, @intFromBool(hull[1] > hull[0])) + @intFromBool(hull[2] > hull[1]) + @intFromBool(hull[0] > hull[2])) != 2;
    var hcurr: usize = hull[if (rev) 0 else hp - 1];
    if (hcurr >= npoints) return error.InvalidArgument;
    var increasing: i32 = -1;
    for (0..hp) |i| {
        const hnext: usize = hull[if (rev) hp - i - 1 else i];
        if (hnext >= npoints) return error.InvalidArgument;
        const pt0 = pts[hcurr];
        const pt1 = pts[hnext];
        const lt: i32 = @intFromBool(hcurr < hnext);
        if (increasing < 0) {
            increasing = 1 - lt;
        } else if (increasing != lt) return error.InvalidArgument;
        const dx0: f64 = @floatFromInt(pt1.x - pt0.x);
        const dy0: f64 = @floatFromInt(pt1.y - pt0.y);
        const scale: f64 = if (dx0 == 0 and dy0 == 0) 0 else 1.0 / @sqrt(dx0 * dx0 + dy0 * dy0);
        var deepest: i32 = -1;
        var depth: f64 = 0;
        var is_defect = false;
        var j = hcurr;
        while (true) {
            j += 1;
            if (j >= npoints) j = 0;
            if (j == hnext) break;
            const dx: f64 = @floatFromInt(pts[j].x - pt0.x);
            const dy: f64 = @floatFromInt(pts[j].y - pt0.y);
            const dist = @abs(-dy0 * dx + dx0 * dy) * scale;
            if (dist > depth) {
                depth = dist;
                deepest = @intCast(j);
                is_defect = true;
            }
        }
        if (is_defect) try out.append(gpa, .{ @intCast(hcurr), @intCast(hnext), deepest, @intCast(@as(i64, @intFromFloat(core.sat.rint(depth * 256)))) });
        hcurr = hnext;
    }
    return out.toOwnedSlice(gpa);
}

/// isContourConvex: true when the polygon is convex (and not
/// self-intersecting at a vertex).
pub fn isContourConvex(pts: anytype) bool {
    const n = pts.len;
    if (n == 0) return false;
    var prev = pts[(2 * n - 2) % n];
    var cur = pts[n - 1];
    var dx0 = sub(cur.x, prev.x);
    var dy0 = sub(cur.y, prev.y);
    var orientation: u8 = 0;
    for (pts) |p| {
        prev = cur;
        cur = p;
        const dx = sub(cur.x, prev.x);
        const dy = sub(cur.y, prev.y);
        const dxdy0 = mul(dx, dy0);
        const dydx0 = mul(dy, dx0);
        orientation |= if (dydx0 > dxdy0) 1 else if (dydx0 < dxdy0) 2 else 3;
        if (orientation == 3) return false;
        dx0 = dx;
        dy0 = dy;
    }
    return true;
}

// ------------------------------------------------------- rotating calipers

fn rotatingCalipers(points: []const Point2f, out: *[3]Point2f) void {
    const n = points.len;
    var minarea: f32 = std.math.floatMax(f32);
    var buf_left: usize = 0;
    var buf_bottom: usize = 0;
    var buf: [5]f32 = @splat(0); // base_a, width, base_b, height, area
    var vect_buf: [64]Point2f = undefined;
    var inv_buf: [64]f32 = undefined;
    // Hulls are small; fall back to the heap-free path only when they fit.
    const big = n > vect_buf.len;
    var heap: ?[]u8 = null;
    defer if (heap) |h| std.heap.page_allocator.free(h);
    var vect: []Point2f = vect_buf[0..@min(n, vect_buf.len)];
    var inv_len: []f32 = inv_buf[0..@min(n, inv_buf.len)];
    if (big) {
        const bytes = std.heap.page_allocator.alloc(u8, n * (@sizeOf(Point2f) + @sizeOf(f32)) + 8) catch @panic("out of memory");
        heap = bytes;
        vect = @as([*]Point2f, @ptrCast(@alignCast(bytes.ptr)))[0..n];
        inv_len = @as([*]f32, @ptrCast(@alignCast(bytes.ptr + n * @sizeOf(Point2f))))[0..n];
    }
    var left: usize = 0;
    var bottom: usize = 0;
    var right: usize = 0;
    var top: usize = 0;
    var pt0 = points[0];
    var left_x = pt0.x;
    var right_x = pt0.x;
    var top_y = pt0.y;
    var bottom_y = pt0.y;
    for (0..n) |i| {
        if (pt0.x < left_x) {
            left_x = pt0.x;
            left = i;
        }
        if (pt0.x > right_x) {
            right_x = pt0.x;
            right = i;
        }
        if (pt0.y > top_y) {
            top_y = pt0.y;
            top = i;
        }
        if (pt0.y < bottom_y) {
            bottom_y = pt0.y;
            bottom = i;
        }
        const pt = points[if (i + 1 < n) i + 1 else 0];
        const dx: f64 = pt.x - pt0.x;
        const dy: f64 = pt.y - pt0.y;
        vect[i] = .{ .x = @floatCast(dx), .y = @floatCast(dy) };
        inv_len[i] = @floatCast(1.0 / @sqrt(dx * dx + dy * dy));
        pt0 = pt;
    }
    var base_a: f32 = 1; // orientation (counter-clockwise hull)
    var base_b: f32 = 0;
    var seq = [4]usize{ bottom, right, top, left };
    for (0..n) |_| {
        var rot: [4]Point2f = undefined;
        rot[0] = vect[seq[0]];
        rot[1] = .{ .x = vect[seq[1]].y, .y = -vect[seq[1]].x };
        rot[2] = .{ .x = -vect[seq[2]].x, .y = -vect[seq[2]].y };
        rot[3] = .{ .x = -vect[seq[3]].y, .y = vect[seq[3]].x };
        var main: usize = 0;
        for (1..4) |i| {
            // firstVecIsRight(rot[i], rot[main])
            const t: Point2f = .{ .x = rot[i].y, .y = -rot[i].x };
            if (t.x * rot[main].x + t.y * rot[main].y < 0) main = i;
        }
        const pindex = seq[main];
        const lead_x = vect[pindex].x * inv_len[pindex];
        const lead_y = vect[pindex].y * inv_len[pindex];
        switch (main) {
            0 => {
                base_a = lead_x;
                base_b = lead_y;
            },
            1 => {
                base_a = lead_y;
                base_b = -lead_x;
            },
            2 => {
                base_a = -lead_x;
                base_b = -lead_y;
            },
            else => {
                base_a = -lead_y;
                base_b = lead_x;
            },
        }
        seq[main] += 1;
        if (seq[main] == n) seq[main] = 0;
        var dx = points[seq[1]].x - points[seq[3]].x;
        var dy = points[seq[1]].y - points[seq[3]].y;
        const width = dx * base_a + dy * base_b;
        dx = points[seq[2]].x - points[seq[0]].x;
        dy = points[seq[2]].y - points[seq[0]].y;
        const height = -dx * base_b + dy * base_a;
        const area = width * height;
        if (area <= minarea) {
            minarea = area;
            buf_left = seq[3];
            buf = .{ base_a, width, base_b, height, area };
            buf_bottom = seq[0];
        }
    }
    const A1 = buf[0];
    const B1 = buf[2];
    const A2 = -buf[2];
    const B2 = buf[0];
    const C1 = A1 * points[buf_left].x + points[buf_left].y * B1;
    const C2 = A2 * points[buf_bottom].x + points[buf_bottom].y * B2;
    const idet = 1.0 / (A1 * B2 - A2 * B1);
    const px = (C1 * B2 - C2 * B1) * idet;
    const py = (A1 * C2 - A2 * C1) * idet;
    out[0] = .{ .x = px, .y = py };
    out[1] = .{ .x = A1 * buf[1], .y = B1 * buf[1] };
    out[2] = .{ .x = A2 * buf[3], .y = B2 * buf[3] };
}

/// minAreaRect: the smallest rotated rectangle enclosing the points, with
/// OpenCV's angle convention (angle in [-90, 0)).
pub fn minAreaRect(gpa: Allocator, pts: anytype) Allocator.Error!RotatedRect {
    var box: RotatedRect = .{ .cx = 0, .cy = 0, .width = 0, .height = 0, .angle = 0 };
    var angle: f64 = -std.math.pi / 2.0;
    const idx = try convexHull(gpa, pts, false, true);
    defer gpa.free(idx);
    const hp = try gpa.alloc(Point2f, idx.len);
    defer gpa.free(hp);
    for (idx, hp) |i, *h| h.* = pf(pts[i]);
    const n = hp.len;
    if (n > 2) {
        var out: [3]Point2f = undefined;
        rotatingCalipers(hp, &out);
        box.cx = out[0].x + (out[1].x + out[2].x) * 0.5;
        box.cy = out[0].y + (out[1].y + out[2].y) * 0.5;
        box.width = @floatCast(@sqrt(@as(f64, out[2].x) * out[2].x + @as(f64, out[2].y) * out[2].y));
        box.height = @floatCast(@sqrt(@as(f64, out[1].x) * out[1].x + @as(f64, out[1].y) * out[1].y));
        if (out[1].x == 0 and out[1].y > 0) {
            std.mem.swap(f32, &box.width, &box.height);
        } else {
            angle = -std.math.atan2(@as(f64, out[1].x), @as(f64, out[1].y));
        }
    } else if (n == 2) {
        box.cx = (hp[0].x + hp[1].x) * 0.5;
        box.cy = (hp[0].y + hp[1].y) * 0.5;
        const dx: f64 = hp[0].x - hp[1].x;
        const dy: f64 = hp[0].y - hp[1].y;
        box.width = 0;
        box.height = @floatCast(@sqrt(dx * dx + dy * dy));
        if (dx == 0) {
            std.mem.swap(f32, &box.width, &box.height);
        } else if (dy < 0) {
            angle = std.math.atan2(dy, dx);
            std.mem.swap(f32, &box.width, &box.height);
        } else if (dy > 0) {
            angle = -std.math.atan2(dx, dy);
        }
    } else if (n == 1) {
        box.cx = hp[0].x;
        box.cy = hp[0].y;
    }
    box.angle = @floatCast(angle * 180.0 / std.math.pi);
    return box;
}

/// boxPoints: the four corners of a rotated rectangle (RotatedRect::points).
pub fn boxPoints(box: RotatedRect) [4]Point2f {
    const a_rad = @as(f64, box.angle) * std.math.pi / 180.0;
    const b: f32 = @as(f32, @floatCast(@cos(a_rad))) * 0.5;
    const a: f32 = @as(f32, @floatCast(@sin(a_rad))) * 0.5;
    const ah = a * box.height;
    const aw = a * box.width;
    const bh = b * box.height;
    const bw = b * box.width;
    return .{
        .{ .x = box.cx - ah - bw, .y = box.cy + bh - aw },
        .{ .x = box.cx + ah - bw, .y = box.cy - bh - aw },
        .{ .x = box.cx + ah + bw, .y = box.cy - bh + aw },
        .{ .x = box.cx - ah + bw, .y = box.cy + bh + aw },
    };
}

// ----------------------------------------------------- enclosing circle

const EPS: f32 = 1.0e-4;

/// cv::RNG (multiply-with-carry).
const Rng = struct {
    state: u64,
    fn init(seed: u64) Rng {
        return .{ .state = if (seed != 0) seed else 0xffffffff };
    }
    fn next(r: *Rng) u32 {
        r.state = @as(u64, @as(u32, @truncate(r.state))) *% 4164903690 +% (r.state >> 32);
        return @truncate(r.state);
    }
    fn uniform(r: *Rng, a: i32, b: i32) i32 {
        if (a == b) return a;
        const range: u32 = @bitCast(b -% a);
        return @bitCast(r.next() % range +% @as(u32, @bitCast(a)));
    }
};

inline fn normF(dx: f32, dy: f32) f64 {
    return @sqrt(@as(f64, dx) * dx + @as(f64, dy) * dy);
}

pub const Circle = extern struct { cx: f32, cy: f32, radius: f32 };

fn circle3(p0: Point2f, p1: Point2f, p2: Point2f, c: *Circle) void {
    const v1: Point2f = .{ .x = p1.x - p0.x, .y = p1.y - p0.y };
    const v2: Point2f = .{ .x = p2.x - p0.x, .y = p2.y - p0.y };
    const m1: Point2f = .{ .x = (p0.x + p1.x) / 2.0, .y = (p0.y + p1.y) / 2.0 };
    const c1 = m1.x * v1.x + m1.y * v1.y;
    const m2: Point2f = .{ .x = (p0.x + p2.x) / 2.0, .y = (p0.y + p2.y) / 2.0 };
    const c2 = m2.x * v2.x + m2.y * v2.y;
    const det = v1.x * v2.y - v1.y * v2.x;
    if (@abs(det) <= EPS) return;
    var cx = (c1 * v2.y - c2 * v1.y) / det;
    var cy = (v1.x * c2 - v2.x * c1) / det;
    c.cx = cx;
    c.cy = cy;
    cx -= p0.x;
    cy -= p0.y;
    c.radius = @sqrt(cx * cx + cy * cy) + EPS;
}

fn halfSum(a: anytype, b: @TypeOf(a)) f32 {
    return toF(add(a, b)) / 2.0;
}

fn twoPointCircle(a: anytype, b: @TypeOf(a), c: *Circle) void {
    c.cx = halfSum(b.x, a.x);
    c.cy = halfSum(b.y, a.y);
    const dx = toF(sub(b.x, a.x));
    const dy = toF(sub(b.y, a.y));
    c.radius = @as(f32, @floatCast(normF(dx, dy))) / 2.0 + EPS;
}

/// minEnclosingCircle (Welzl's algorithm with OpenCV's deterministic
/// shuffle). `scratch` must hold pts.len points.
pub fn minEnclosingCircle(gpa: Allocator, pts_in: anytype) Allocator.Error!Circle {
    const P = std.meta.Elem(@TypeOf(pts_in));
    const count = pts_in.len;
    var c: Circle = .{ .cx = 0, .cy = 0, .radius = 0 };
    if (count == 0) return c;
    if (count == 1) return .{ .cx = toF(pts_in[0].x), .cy = toF(pts_in[0].y), .radius = EPS };
    const pts = try gpa.dupe(P, pts_in);
    defer gpa.free(pts);
    if (count > 10) {
        const x0: u32 = @bitCast(toF(pts[0].x));
        const y0: u32 = @bitCast(toF(pts[0].y));
        const xn: u32 = @bitCast(toF(pts[count - 1].x));
        const yn: u32 = @bitCast(toF(pts[count - 1].y));
        const seed: u32 = @as(u32, @truncate(count)) ^ x0 ^ (y0 << 8) ^ (xn << 16) ^ (yn << 24);
        var rng = Rng.init(seed);
        for (1..count) |i| {
            const j: usize = @intCast(rng.uniform(0, @intCast(i + 1)));
            std.mem.swap(P, &pts[i], &pts[j]);
        }
    }
    twoPointCircle(pts[1], pts[0], &c);
    for (2..count) |i| {
        const dx = toF(pts[i].x) - c.cx;
        const dy = toF(pts[i].y) - c.cy;
        const d: f32 = @floatCast(normF(dx, dy));
        if (d < c.radius) continue;
        // findSecondPoint
        twoPointCircle(pts[i], pts[0], &c);
        for (1..i) |j| {
            const dx2 = c.cx - toF(pts[j].x);
            const dy2 = c.cy - toF(pts[j].y);
            if (normF(dx2, dy2) < c.radius) continue;
            // findThirdPoint
            twoPointCircle(pts[i], pts[j], &c);
            for (0..j) |k| {
                const dx3 = c.cx - toF(pts[k].x);
                const dy3 = c.cy - toF(pts[k].y);
                if (normF(dx3, dy3) < c.radius) continue;
                circle3(pf(pts[i]), pf(pts[j]), pf(pts[k]), &c);
            }
        }
    }
    return c;
}

// --------------------------------------------------------- approxPolyDP

/// approxPolyDP (Ramer-Douglas-Peucker as OpenCV implements it). Writes
/// the result to `out` (at least pts.len entries) and returns its length.
pub fn approxPolyDP(gpa: Allocator, pts: anytype, out: []std.meta.Elem(@TypeOf(pts)), epsilon_in: f64, closed: bool) (Allocator.Error || error{InvalidArgument})!usize {
    const P = std.meta.Elem(@TypeOf(pts));
    if (epsilon_in < 0 or !(epsilon_in < 1e30)) return error.InvalidArgument;
    const count0 = pts.len;
    if (count0 == 0) return 0;
    std.debug.assert(out.len >= count0);
    const Range = struct { start: usize, end: usize };
    var stack: std.ArrayList(Range) = .empty;
    defer stack.deinit(gpa);
    try stack.ensureTotalCapacity(gpa, count0);

    const count = count0;
    var init_iters: usize = 3;
    var slice: Range = .{ .start = 0, .end = 0 };
    var right_slice: Range = .{ .start = 0, .end = 0 };
    var start_pt: P = undefined;
    var end_pt: P = undefined;
    var pt: P = undefined;
    var pos: usize = 0;
    var new_count: usize = 0;
    var is_closed = closed;
    var le_eps = false;
    const eps = epsilon_in * epsilon_in;

    if (!is_closed) {
        right_slice.start = count;
        end_pt = pts[0];
        start_pt = pts[count - 1];
        if (start_pt.x != end_pt.x or start_pt.y != end_pt.y) {
            slice = .{ .start = 0, .end = count - 1 };
            try stack.append(gpa, slice);
        } else {
            is_closed = true;
            init_iters = 1;
        }
    }
    if (is_closed) {
        right_slice.start = 0;
        for (0..init_iters) |_| {
            var max_dist: f64 = 0;
            pos = (pos + right_slice.start) % count;
            start_pt = pts[pos];
            pos += 1;
            if (pos >= count) pos = 0;
            for (1..count) |j| {
                pt = pts[pos];
                pos += 1;
                if (pos >= count) pos = 0;
                const dx = toD(sub(pt.x, start_pt.x));
                const dy = toD(sub(pt.y, start_pt.y));
                const dist = dx * dx + dy * dy;
                if (dist > max_dist) {
                    max_dist = dist;
                    right_slice.start = j;
                }
            }
            le_eps = max_dist <= eps;
        }
        if (!le_eps) {
            slice.start = pos % count;
            right_slice.end = slice.start;
            right_slice.start = (right_slice.start + slice.start) % count;
            slice.end = right_slice.start;
            try stack.append(gpa, right_slice);
            try stack.append(gpa, slice);
        } else {
            out[new_count] = start_pt;
            new_count += 1;
        }
    }

    while (stack.pop()) |sl| {
        slice = sl;
        end_pt = pts[slice.end];
        pos = slice.start;
        start_pt = pts[pos];
        pos += 1;
        if (pos >= count) pos = 0;
        if (pos != slice.end) {
            var max_d: f64 = 0;
            const dx = toD(sub(end_pt.x, start_pt.x));
            const dy = toD(sub(end_pt.y, start_pt.y));
            const seg2 = dx * dx + dy * dy;
            while (pos != slice.end) {
                pt = pts[pos];
                pos += 1;
                if (pos >= count) pos = 0;
                const ddx = sub(pt.x, start_pt.x);
                const ddy = sub(pt.y, start_pt.y);
                const projection = toD(ddx) * dx + toD(ddy) * dy;
                var d2: f64 = undefined;
                if (projection < 0) {
                    d2 = toD(add(mul(ddx, ddx), mul(ddy, ddy))) * seg2;
                } else if (projection > seg2) {
                    const ex = sub(pt.x, end_pt.x);
                    const ey = sub(pt.y, end_pt.y);
                    d2 = toD(add(mul(ex, ex), mul(ey, ey))) * seg2;
                } else {
                    const dist = toD(ddy) * dx - toD(ddx) * dy;
                    d2 = dist * dist;
                }
                if (d2 > max_d) {
                    max_d = d2;
                    right_slice.start = (pos + count - 1) % count;
                }
            }
            le_eps = max_d <= eps * seg2;
        } else {
            le_eps = true;
            start_pt = pts[slice.start];
        }
        if (le_eps) {
            out[new_count] = start_pt;
            new_count += 1;
        } else {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            try stack.append(gpa, right_slice);
            try stack.append(gpa, slice);
        }
    }
    if (!is_closed) {
        out[new_count] = pts[count - 1];
        new_count += 1;
    }

    // Final clean-up: drop points on (almost) straight lines.
    is_closed = closed;
    const cnt = new_count;
    const not_closed: usize = @intFromBool(!is_closed);
    pos = if (is_closed) cnt - 1 else 0;
    start_pt = out[pos];
    pos += 1;
    if (pos >= cnt) pos = 0;
    var wpos = pos;
    pt = out[pos];
    pos += 1;
    if (pos >= cnt) pos = 0;
    var i: usize = not_closed;
    while (i < cnt - not_closed and new_count > 2) : (i += 1) {
        end_pt = out[pos];
        pos += 1;
        if (pos >= cnt) pos = 0;
        const dx = toD(sub(end_pt.x, start_pt.x));
        const dy = toD(sub(end_pt.y, start_pt.y));
        const dist = @abs(toD(sub(pt.x, start_pt.x)) * dy - toD(sub(pt.y, start_pt.y)) * dx);
        const sip = toD(add(mul(sub(pt.x, start_pt.x), sub(end_pt.x, pt.x)), mul(sub(pt.y, start_pt.y), sub(end_pt.y, pt.y))));
        if (dist * dist <= 0.5 * eps * (dx * dx + dy * dy) and dx != 0 and dy != 0 and sip >= 0) {
            new_count -= 1;
            start_pt = end_pt;
            out[wpos] = end_pt;
            wpos += 1;
            if (wpos >= cnt) wpos = 0;
            pt = out[pos];
            pos += 1;
            if (pos >= cnt) pos = 0;
            i += 1;
            continue;
        }
        start_pt = pt;
        out[wpos] = pt;
        wpos += 1;
        if (wpos >= cnt) wpos = 0;
        pt = end_pt;
    }
    if (!is_closed) out[wpos] = pt;
    return new_count;
}

// ------------------------------------------------------ pointPolygonTest

/// pointPolygonTest: +1 / -1 / 0 (inside / outside / on the edge), or the
/// signed distance to the nearest edge when `measure_dist`.
pub fn pointPolygonTest(pts: anytype, pt: Point2f, measure_dist: bool) f64 {
    const T = Coord(@TypeOf(pts));
    const total = pts.len;
    if (total == 0) return if (measure_dist) -std.math.floatMax(f64) else -1;
    var counter: u32 = 0;
    const ipx: i32 = @intFromFloat(core.sat.rint(@as(f64, pt.x)));
    const ipy: i32 = @intFromFloat(core.sat.rint(@as(f64, pt.y)));
    if (T == i32 and !measure_dist and @as(f32, @floatFromInt(ipx)) == pt.x and @as(f32, @floatFromInt(ipy)) == pt.y) {
        var v = pts[total - 1];
        for (pts) |q| {
            const v0 = v;
            v = q;
            if ((v0.y <= ipy and v.y <= ipy) or (v0.y > ipy and v.y > ipy) or (v0.x < ipx and v.x < ipx)) {
                if (ipy == v.y and (ipx == v.x or (ipy == v0.y and ((v0.x <= ipx and ipx <= v.x) or (v.x <= ipx and ipx <= v0.x))))) return 0;
                continue;
            }
            var dist = @as(i64, ipy - v0.y) * (v.x - v0.x) - @as(i64, ipx - v0.x) * (v.y - v0.y);
            if (dist == 0) return 0;
            if (v.y < v0.y) dist = -dist;
            counter += @intFromBool(dist > 0);
        }
        return if (counter % 2 == 0) -1 else 1;
    }
    var v = pf(pts[total - 1]);
    if (!measure_dist) {
        for (pts) |q| {
            const v0 = v;
            v = pf(q);
            if ((v0.y <= pt.y and v.y <= pt.y) or (v0.y > pt.y and v.y > pt.y) or (v0.x < pt.x and v.x < pt.x)) {
                if (pt.y == v.y and (pt.x == v.x or (pt.y == v0.y and ((v0.x <= pt.x and pt.x <= v.x) or (v.x <= pt.x and pt.x <= v0.x))))) return 0;
                continue;
            }
            var dist = @as(f64, pt.y - v0.y) * (v.x - v0.x) - @as(f64, pt.x - v0.x) * (v.y - v0.y);
            if (dist == 0) return 0;
            if (v.y < v0.y) dist = -dist;
            counter += @intFromBool(dist > 0);
        }
        return if (counter % 2 == 0) -1 else 1;
    }
    var min_num: f64 = std.math.floatMax(f32);
    var min_den: f64 = 1;
    for (pts) |q| {
        const v0 = v;
        v = pf(q);
        const dx: f64 = v.x - v0.x;
        const dy: f64 = v.y - v0.y;
        const dx1: f64 = pt.x - v0.x;
        const dy1: f64 = pt.y - v0.y;
        const dx2: f64 = pt.x - v.x;
        const dy2: f64 = pt.y - v.y;
        var num: f64 = undefined;
        var den: f64 = 1;
        if (dx1 * dx + dy1 * dy <= 0) {
            num = dx1 * dx1 + dy1 * dy1;
        } else if (dx2 * dx + dy2 * dy >= 0) {
            num = dx2 * dx2 + dy2 * dy2;
        } else {
            num = dy1 * dx - dx1 * dy;
            num *= num;
            den = dx * dx + dy * dy;
        }
        if (num * min_den < min_num * den) {
            min_num = num;
            min_den = den;
            if (min_num == 0) break;
        }
        if ((v0.y <= pt.y and v.y <= pt.y) or (v0.y > pt.y and v.y > pt.y) or (v0.x < pt.x and v.x < pt.x)) continue;
        var d = dy1 * dx - dx1 * dy;
        if (dy < 0) d = -d;
        counter += @intFromBool(d > 0);
    }
    const r = @sqrt(min_num / min_den);
    return if (counter % 2 == 0) -r else r;
}

// ------------------------------------------------------------ fitEllipse

/// Least squares min |A x - b| by Householder QR (A is n x k, row-major,
/// overwritten). Matches OpenCV's SVD solve for full-rank systems.
fn lstsq(comptime k: usize, a: []f64, b: []f64, x: *[k]f64) void {
    const n = b.len;
    for (0..k) |j| {
        var norm: f64 = 0;
        for (j..n) |i| norm += a[i * k + j] * a[i * k + j];
        norm = @sqrt(norm);
        if (norm == 0) continue;
        const alpha = if (a[j * k + j] > 0) -norm else norm;
        // v = a[j..n, j] - alpha e1
        a[j * k + j] -= alpha;
        var vnorm2: f64 = 0;
        for (j..n) |i| vnorm2 += a[i * k + j] * a[i * k + j];
        if (vnorm2 == 0) {
            a[j * k + j] += alpha;
            continue;
        }
        for (j + 1..k) |c| {
            var d: f64 = 0;
            for (j..n) |i| d += a[i * k + j] * a[i * k + c];
            const f = 2 * d / vnorm2;
            for (j..n) |i| a[i * k + c] -= f * a[i * k + j];
        }
        var d: f64 = 0;
        for (j..n) |i| d += a[i * k + j] * b[i];
        const f = 2 * d / vnorm2;
        for (j..n) |i| b[i] -= f * a[i * k + j];
        a[j * k + j] = alpha;
    }
    var j: usize = k;
    while (j > 0) {
        j -= 1;
        var sum = b[j];
        for (j + 1..k) |c| sum -= a[j * k + c] * x[c];
        x[j] = if (a[j * k + j] != 0) sum / a[j * k + j] else 0;
    }
}

/// Singular values (descending) of an n x 5 matrix via the eigenvalues of
/// AᵀA (Jacobi rotations); only used for OpenCV's degeneracy test.
fn singularValues5(a: []const f64, n: usize) [5]f64 {
    var m: [5][5]f64 = @splat(@splat(0));
    for (0..n) |i| for (0..5) |r| for (0..5) |c| {
        m[r][c] += a[i * 5 + r] * a[i * 5 + c];
    };
    for (0..60) |_| {
        var off: f64 = 0;
        for (0..5) |r| for (0..5) |c| {
            if (r != c) off += m[r][c] * m[r][c];
        };
        if (off < 1e-30) break;
        for (0..5) |p| for (p + 1..5) |q| {
            if (m[p][q] == 0) continue;
            const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
            const t = std.math.sign(theta) / (@abs(theta) + @sqrt(theta * theta + 1)) + @as(f64, if (theta == 0) 1 else 0);
            const cs = 1 / @sqrt(t * t + 1);
            const sn = t * cs;
            for (0..5) |r| {
                const mp = m[r][p];
                const mq = m[r][q];
                m[r][p] = cs * mp - sn * mq;
                m[r][q] = sn * mp + cs * mq;
            }
            for (0..5) |c| {
                const mp = m[p][c];
                const mq = m[q][c];
                m[p][c] = cs * mp - sn * mq;
                m[q][c] = sn * mp + cs * mq;
            }
        };
    }
    var w: [5]f64 = undefined;
    for (0..5) |i| w[i] = @sqrt(@max(m[i][i], 0));
    std.sort.insertion(f64, &w, {}, std.sort.desc(f64));
    return w;
}

/// fitEllipse (OpenCV's least-squares fit for 6+ points; 5 points use the
/// same fit, where OpenCV switches to fitEllipseDirect). Needs at least 5
/// points. For nearly degenerate sets OpenCV jitters the points with its
/// global RNG; VISAR uses a fixed seed there, so such results differ.
pub fn fitEllipse(gpa: Allocator, pts: anytype) (Allocator.Error || error{InvalidArgument})!RotatedRect {
    const n = pts.len;
    if (n < 5) return error.InvalidArgument;
    var box: RotatedRect = .{ .cx = 0, .cy = 0, .width = 0, .height = 0, .angle = 0 };
    const min_eps = 1e-8;
    const p = try gpa.alloc(Point2f, n);
    defer gpa.free(p);
    const a = try gpa.alloc(f64, n * 5);
    defer gpa.free(a);
    const b = try gpa.alloc(f64, n);
    defer gpa.free(b);
    var c: Point2f = .{ .x = 0, .y = 0 };
    for (pts, p) |q, *d| {
        d.* = pf(q);
        c.x += d.x;
        c.y += d.y;
    }
    c.x /= @floatFromInt(n);
    c.y /= @floatFromInt(n);
    var s: f64 = 0;
    for (p) |q| s += @abs(q.x - c.x) + @abs(q.y - c.y);
    const scale = 100.0 / (if (s > flt_eps) s else flt_eps);
    const fill = struct {
        fn f(pp: []const Point2f, cc: Point2f, sc: f64, aa: []f64, bb: []f64) void {
            for (pp, 0..) |q, i| {
                const px = @as(f64, q.x - cc.x) * sc;
                const py = @as(f64, q.y - cc.y) * sc;
                bb[i] = 10000.0;
                aa[i * 5 ..][0..5].* = .{ -px * px, -py * py, -px * py, px, py };
            }
        }
    }.f;
    fill(p, c, scale, a, b);
    const w = singularValues5(a, n);
    if (w[0] * flt_eps > w[4]) {
        const eps: f32 = @floatCast(s / @as(f64, @floatFromInt(n * 2)) * 1e-3);
        var rng = Rng.init(0x12345678);
        for (p) |*q| {
            q.x += @as(f32, @floatFromInt(rng.next())) / 4294967296.0 * 2 * eps - eps;
            q.y += @as(f32, @floatFromInt(rng.next())) / 4294967296.0 * 2 * eps - eps;
        }
        fill(p, c, scale, a, b);
    }
    var gfp: [5]f64 = undefined;
    lstsq(5, a, b, &gfp);

    // Centre from the general-form parameters.
    var rp: [5]f64 = @splat(0);
    {
        var a2 = [4]f64{ 2 * gfp[0], gfp[2], gfp[2], 2 * gfp[1] };
        var b2 = [2]f64{ gfp[3], gfp[4] };
        var x2: [2]f64 = undefined;
        lstsq(2, &a2, &b2, &x2);
        rp[0] = x2[0];
        rp[1] = x2[1];
    }
    // Re-fit A..C around that centre.
    for (p, 0..) |q, i| {
        const px = @as(f64, q.x - c.x) * scale;
        const py = @as(f64, q.y - c.y) * scale;
        b[i] = 1.0;
        a[i * 3 ..][0..3].* = .{ (px - rp[0]) * (px - rp[0]), (py - rp[1]) * (py - rp[1]), (px - rp[0]) * (py - rp[1]) };
    }
    var g3: [3]f64 = undefined;
    lstsq(3, a[0 .. n * 3], b, &g3);
    rp[4] = -0.5 * std.math.atan2(g3[2], g3[1] - g3[0]);
    const t = if (@abs(g3[2]) > min_eps) g3[2] / @sin(-2.0 * rp[4]) else g3[1] - g3[0];
    rp[2] = @abs(g3[0] + g3[1] - t);
    if (rp[2] > min_eps) rp[2] = @sqrt(2.0 / rp[2]);
    rp[3] = @abs(g3[0] + g3[1] + t);
    if (rp[3] > min_eps) rp[3] = @sqrt(2.0 / rp[3]);
    box.cx = @as(f32, @floatCast(rp[0] / scale)) + c.x;
    box.cy = @as(f32, @floatCast(rp[1] / scale)) + c.y;
    box.width = @floatCast(rp[2] * 2 / scale);
    box.height = @floatCast(rp[3] * 2 / scale);
    if (box.width > box.height) {
        std.mem.swap(f32, &box.width, &box.height);
        box.angle = @floatCast(90 + rp[4] * 180 / std.math.pi);
    }
    if (box.angle < -180) box.angle += 360;
    if (box.angle > 360) box.angle -= 360;
    return box;
}

// ------------------------------------------------------------------ tests

test "moments, area, length of a rectangle" {
    const sq = [_]Point{ .{ .x = 0, .y = 0 }, .{ .x = 0, .y = 4 }, .{ .x = 6, .y = 4 }, .{ .x = 6, .y = 0 } };
    const m = contourMoments(@as([]const Point, &sq));
    try std.testing.expectEqual(@as(f64, 24), m.m00);
    try std.testing.expectApproxEqAbs(@as(f64, 72), m.m10, 1e-9);
    try std.testing.expectEqual(@as(f64, 24), contourArea(@as([]const Point, &sq), false));
    try std.testing.expectEqual(@as(f64, -24), contourArea(@as([]const Point, &sq), true));
    try std.testing.expectEqual(@as(f64, 20), arcLength(@as([]const Point, &sq), true));
    try std.testing.expect(isContourConvex(@as([]const Point, &sq)));
    try std.testing.expectEqual(Rect{ .x = 0, .y = 0, .width = 7, .height = 5 }, boundingRect(@as([]const Point, &sq)));
    try std.testing.expectEqual(@as(f64, 1), pointPolygonTest(@as([]const Point, &sq), .{ .x = 2, .y = 2 }, false));
    try std.testing.expectEqual(@as(f64, 2), pointPolygonTest(@as([]const Point, &sq), .{ .x = 2, .y = 2 }, true));
}

test "convexHull and minAreaRect" {
    const gpa = std.testing.allocator;
    const pts = [_]Point{ .{ .x = 0, .y = 0 }, .{ .x = 2, .y = 1 }, .{ .x = 4, .y = 0 }, .{ .x = 4, .y = 4 }, .{ .x = 0, .y = 4 } };
    const h = try convexHull(gpa, @as([]const Point, &pts), false, false);
    defer gpa.free(h);
    try std.testing.expectEqual(@as(usize, 4), h.len);
    const r = try minAreaRect(gpa, @as([]const Point, &pts));
    try std.testing.expectEqual(@as(f32, 2), r.cx);
    try std.testing.expectEqual(@as(f32, 4), r.width);
    const c = try minEnclosingCircle(gpa, @as([]const Point, &pts));
    try std.testing.expectApproxEqAbs(@as(f32, 2.8285), c.radius, 1e-3);
}

test "image moments of a mask" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 70, 40, 1, 0);
    defer img.deinit(gpa);
    for (5..15) |y| @memset(img.row(y)[30..50], 7);
    const m = try imageMoments(u8, .serial, gpa, img, true);
    try std.testing.expectEqual(@as(f64, 200), m.m00);
    try std.testing.expectApproxEqAbs(@as(f64, 39.5), m.m10 / m.m00, 1e-12);
    try std.testing.expectEqual(Rect{ .x = 30, .y = 5, .width = 20, .height = 10 }, maskBoundingRect(img));
}
