//! Polygon and line rasterisation that reproduces OpenCV's drawing code
//! pixel for pixel (8- and 4-connected lines, thick lines, filled circles,
//! fillConvexPoly, fillPoly and polylines). drawContours is built on these.
//! Colours are given per channel; missing channels use the last value.
//! Antialiased lines (LINE_AA) are not implemented.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;
const contours = @import("contours.zig");
const Point = contours.Point;

/// OpenCV line types: 8-connected (LINE_8) or 4-connected (LINE_4).
pub const LineType = enum(u8) { four = 4, eight = 8 };

const XY_SHIFT = 16;
const XY_ONE: i64 = 1 << XY_SHIFT;

const P64 = struct { x: i64, y: i64 };

fn cvRound(x: f64) i64 {
    return @intFromFloat(core.sat.rint(x));
}

inline fn hline(img: Image(u8), y: i64, xl: i64, xr: i64, color: []const u8) void {
    if (xl > xr) return;
    const row = img.row(@intCast(y));
    const cn = img.channels;
    if (cn == 1) {
        @memset(row[@intCast(xl)..@intCast(xr + 1)], color[0]);
        return;
    }
    var x: usize = @intCast(xl);
    while (x <= xr) : (x += 1) {
        for (0..cn) |c| row[x * cn + c] = color[@min(c, color.len - 1)];
    }
}

inline fn putPixel(img: Image(u8), x: i64, y: i64, color: []const u8) void {
    const p = img.pixel(@intCast(x), @intCast(y));
    for (p, 0..) |*v, c| v.* = color[@min(c, color.len - 1)];
}

/// clipLine(Size2l, ...): clips the segment to [0, w) x [0, h); false when
/// it lies entirely outside.
fn clipLine(w: i64, h: i64, p1: *P64, p2: *P64) bool {
    const right = w - 1;
    const bottom = h - 1;
    if (w <= 0 or h <= 0) return false;
    const code = struct {
        fn f(p: P64, r: i64, b: i64) u8 {
            return @as(u8, @intFromBool(p.x < 0)) + @as(u8, @intFromBool(p.x > r)) * 2 + @as(u8, @intFromBool(p.y < 0)) * 4 + @as(u8, @intFromBool(p.y > b)) * 8;
        }
    }.f;
    var c1 = code(p1.*, right, bottom);
    var c2 = code(p2.*, right, bottom);
    if ((c1 & c2) == 0 and (c1 | c2) != 0) {
        if (c1 & 12 != 0) {
            const a: i64 = if (c1 < 8) 0 else bottom;
            p1.x += @intFromFloat(@as(f64, @floatFromInt(a - p1.y)) * @as(f64, @floatFromInt(p2.x - p1.x)) / @as(f64, @floatFromInt(p2.y - p1.y)));
            p1.y = a;
            c1 = @as(u8, @intFromBool(p1.x < 0)) + @as(u8, @intFromBool(p1.x > right)) * 2;
        }
        if (c2 & 12 != 0) {
            const a: i64 = if (c2 < 8) 0 else bottom;
            p2.x += @intFromFloat(@as(f64, @floatFromInt(a - p2.y)) * @as(f64, @floatFromInt(p2.x - p1.x)) / @as(f64, @floatFromInt(p2.y - p1.y)));
            p2.y = a;
            c2 = @as(u8, @intFromBool(p2.x < 0)) + @as(u8, @intFromBool(p2.x > right)) * 2;
        }
        if ((c1 & c2) == 0 and (c1 | c2) != 0) {
            if (c1 != 0) {
                const a: i64 = if (c1 == 1) 0 else right;
                p1.y += @intFromFloat(@as(f64, @floatFromInt(a - p1.x)) * @as(f64, @floatFromInt(p2.y - p1.y)) / @as(f64, @floatFromInt(p2.x - p1.x)));
                p1.x = a;
                c1 = 0;
            }
            if (c2 != 0) {
                const a: i64 = if (c2 == 1) 0 else right;
                p2.y += @intFromFloat(@as(f64, @floatFromInt(a - p2.x)) * @as(f64, @floatFromInt(p2.y - p1.y)) / @as(f64, @floatFromInt(p2.x - p1.x)));
                p2.x = a;
                c2 = 0;
            }
        }
    }
    return (c1 | c2) == 0;
}

/// OpenCV's Line(): a LineIterator walk (left to right) with clipping.
fn line(img: Image(u8), pt1_in: P64, pt2_in: P64, color: []const u8, connectivity: u8) void {
    const w: i64 = img.width;
    const h: i64 = img.height;
    // LineIterator works on int points.
    var pt1: P64 = .{ .x = @as(i32, @truncate(pt1_in.x)), .y = @as(i32, @truncate(pt1_in.y)) };
    var pt2: P64 = .{ .x = @as(i32, @truncate(pt2_in.x)), .y = @as(i32, @truncate(pt2_in.y)) };
    if (pt1.x < 0 or pt1.x >= w or pt2.x < 0 or pt2.x >= w or pt1.y < 0 or pt1.y >= h or pt2.y < 0 or pt2.y >= h) {
        if (!clipLine(w, h, &pt1, &pt2)) return;
    }
    var delta_x: i64 = 1;
    var delta_y: i64 = 1;
    var dx = pt2.x - pt1.x;
    var dy = pt2.y - pt1.y;
    if (dx < 0) {
        dx = -dx;
        dy = -dy;
        pt1 = pt2;
    }
    if (dy < 0) {
        dy = -dy;
        delta_y = -1;
    }
    const vert = dy > dx;
    if (vert) {
        std.mem.swap(i64, &dx, &dy);
        std.mem.swap(i64, &delta_x, &delta_y);
    }
    var err: i64 = undefined;
    var plus_delta: i64 = undefined;
    var minus_delta: i64 = undefined;
    var minus_shift: i64 = undefined;
    var plus_shift: i64 = undefined;
    var minus_step: i64 = undefined;
    var plus_step: i64 = undefined;
    var count: i64 = undefined;
    if (connectivity == 8) {
        err = dx - (dy + dy);
        plus_delta = dx + dx;
        minus_delta = -(dy + dy);
        minus_shift = delta_x;
        plus_shift = 0;
        minus_step = 0;
        plus_step = delta_y;
        count = dx + 1;
    } else {
        err = 0;
        plus_delta = (dx + dx) + (dy + dy);
        minus_delta = -(dy + dy);
        minus_shift = delta_x;
        plus_shift = -delta_x;
        minus_step = 0;
        plus_step = delta_y;
        count = dx + dy + 1;
    }
    if (vert) {
        std.mem.swap(i64, &plus_step, &plus_shift);
        std.mem.swap(i64, &minus_step, &minus_shift);
    }
    var x = pt1.x;
    var y = pt1.y;
    var i: i64 = 0;
    while (i < count) : (i += 1) {
        putPixel(img, x, y, color);
        const mask: i64 = if (err < 0) -1 else 0;
        err += minus_delta + (plus_delta & mask);
        x += minus_shift + (plus_shift & mask);
        y += minus_step + (plus_step & mask);
    }
}

/// OpenCV's Line2(): a line between fixed-point (XY_SHIFT) endpoints.
fn line2(img: Image(u8), pt1_in: P64, pt2_in: P64, color: []const u8) void {
    var pt1 = pt1_in;
    var pt2 = pt2_in;
    const w: i64 = img.width;
    const h: i64 = img.height;
    if (!clipLine(w << XY_SHIFT, h << XY_SHIFT, &pt1, &pt2)) return;
    var dx = pt2.x - pt1.x;
    var dy = pt2.y - pt1.y;
    const j: i64 = if (dx < 0) -1 else 0;
    const ax = (dx ^ j) - j;
    const i: i64 = if (dy < 0) -1 else 0;
    const ay = (dy ^ i) - i;
    var x_step: i64 = undefined;
    var y_step: i64 = undefined;
    var ecount: i64 = undefined;
    if (ax > ay) {
        dy = (dy ^ j) - j;
        pt1.x ^= pt2.x & j;
        pt2.x ^= pt1.x & j;
        pt1.x ^= pt2.x & j;
        pt1.y ^= pt2.y & j;
        pt2.y ^= pt1.y & j;
        pt1.y ^= pt2.y & j;
        x_step = XY_ONE;
        y_step = @divTrunc(dy * (1 << XY_SHIFT), ax | 1);
        ecount = (pt2.x - pt1.x) >> XY_SHIFT;
    } else {
        dx = (dx ^ i) - i;
        pt1.x ^= pt2.x & i;
        pt2.x ^= pt1.x & i;
        pt1.x ^= pt2.x & i;
        pt1.y ^= pt2.y & i;
        pt2.y ^= pt1.y & i;
        pt1.y ^= pt2.y & i;
        x_step = @divTrunc(dx * (1 << XY_SHIFT), ay | 1);
        y_step = XY_ONE;
        ecount = (pt2.y - pt1.y) >> XY_SHIFT;
    }
    pt1.x += XY_ONE >> 1;
    pt1.y += XY_ONE >> 1;
    const put = struct {
        fn f(im: Image(u8), x: i64, y: i64, c: []const u8) void {
            if (0 <= x and x < im.width and 0 <= y and y < im.height) putPixel(im, x, y, c);
        }
    }.f;
    put(img, (pt2.x + (XY_ONE >> 1)) >> XY_SHIFT, (pt2.y + (XY_ONE >> 1)) >> XY_SHIFT, color);
    if (ax > ay) {
        pt1.x >>= XY_SHIFT;
        while (ecount >= 0) : (ecount -= 1) {
            put(img, pt1.x, pt1.y >> XY_SHIFT, color);
            pt1.x += 1;
            pt1.y += y_step;
        }
    } else {
        pt1.y >>= XY_SHIFT;
        while (ecount >= 0) : (ecount -= 1) {
            put(img, pt1.x >> XY_SHIFT, pt1.y, color);
            pt1.x += x_step;
            pt1.y += 1;
        }
    }
}

/// OpenCV's FillConvexPoly (non-antialiased). `shift` is the number of
/// fractional bits in the vertices.
fn fillConvexPolyImpl(img: Image(u8), v: []const P64, color: []const u8, line_type: LineType, shift: u5) void {
    const npts = v.len;
    if (npts == 0) return;
    const delta: i64 = (@as(i64, 1) << shift) >> 1;
    const size_w: i64 = img.width;
    const size_h: i64 = img.height;
    const delta1: i64 = XY_ONE >> 1;
    const delta2: i64 = XY_ONE >> 1;
    const sh: u6 = XY_SHIFT - @as(u6, shift);

    var p0 = v[npts - 1];
    p0.x <<= sh;
    p0.y <<= sh;
    var xmin = v[0].x;
    var xmax = v[0].x;
    var ymin = v[0].y;
    var ymax = v[0].y;
    var imin: usize = 0;
    for (v, 0..) |pv, i| {
        var p = pv;
        if (p.y < ymin) {
            ymin = p.y;
            imin = i;
        }
        ymax = @max(ymax, p.y);
        xmax = @max(xmax, p.x);
        xmin = @min(xmin, p.x);
        p.x <<= sh;
        p.y <<= sh;
        if (shift == 0) {
            line(img, .{ .x = p0.x >> XY_SHIFT, .y = p0.y >> XY_SHIFT }, .{ .x = p.x >> XY_SHIFT, .y = p.y >> XY_SHIFT }, color, @backingInt(line_type));
        } else {
            line2(img, p0, p, color);
        }
        p0 = p;
    }
    xmin = (xmin + delta) >> shift;
    xmax = (xmax + delta) >> shift;
    ymin = (ymin + delta) >> shift;
    ymax = (ymax + delta) >> shift;
    if (npts < 3 or @as(i32, @truncate(xmax)) < 0 or @as(i32, @truncate(ymax)) < 0 or @as(i32, @truncate(xmin)) >= size_w or @as(i32, @truncate(ymin)) >= size_h) return;
    ymax = @min(ymax, size_h - 1);

    const Edge = struct { idx: usize, di: usize, x: i64, dx: i64, ye: i64 };
    var edge: [2]Edge = .{
        .{ .idx = imin, .di = 1, .x = -XY_ONE, .dx = 0, .ye = ymin },
        .{ .idx = imin, .di = npts - 1, .x = -XY_ONE, .dx = 0, .ye = ymin },
    };
    var edges: i64 = @intCast(npts);
    var y: i64 = ymin;
    while (true) {
        for (&edge) |*e| {
            if (y >= e.ye) {
                var idx0 = e.idx;
                var idx = idx0 + e.di;
                if (idx >= npts) idx -= npts;
                while (true) {
                    const prev_edges = edges;
                    edges -= 1;
                    if (prev_edges <= 0) break;
                    const ty = (v[idx].y + delta) >> shift;
                    if (ty > y) {
                        var xs = v[idx0].x;
                        var xe = v[idx].x;
                        if (shift != XY_SHIFT) {
                            xs <<= sh;
                            xe <<= sh;
                        }
                        e.ye = ty;
                        e.dx = @divTrunc((xe - xs) * 2 + (ty - y), 2 * (ty - y));
                        e.x = xs;
                        e.idx = idx;
                        break;
                    }
                    idx0 = idx;
                    idx += e.di;
                    if (idx >= npts) idx -= npts;
                }
            }
        }
        if (edges < 0) break;
        if (y >= 0) {
            var left: usize = 0;
            var right: usize = 1;
            if (edge[0].x > edge[1].x) {
                left = 1;
                right = 0;
            }
            var xx1: i64 = @as(i32, @truncate((edge[left].x + delta1) >> XY_SHIFT));
            var xx2: i64 = @as(i32, @truncate((edge[right].x + delta2) >> XY_SHIFT));
            if (xx2 >= 0 and xx1 < size_w) {
                if (xx1 < 0) xx1 = 0;
                if (xx2 >= size_w) xx2 = size_w - 1;
                hline(img, y, xx1, xx2, color);
            }
        }
        edge[0].x += edge[0].dx;
        edge[1].x += edge[1].dx;
        y += 1;
        if (y > ymax) break;
    }
}

/// fillConvexPoly(img, points, color, lineType, shift = 0).
pub fn fillConvexPoly(gpa: Allocator, img: Image(u8), pts: []const Point, color: []const u8, line_type: LineType) Allocator.Error!void {
    const v = try gpa.alloc(P64, pts.len);
    defer gpa.free(v);
    for (pts, v) |p, *q| q.* = .{ .x = p.x, .y = p.y };
    fillConvexPolyImpl(img, v, color, line_type, 0);
}

/// OpenCV's Circle(): a (filled when `fill`) circle with integer centre.
fn circleImpl(img: Image(u8), cx: i64, cy: i64, radius: i64, color: []const u8, fill: bool) void {
    const w: i64 = img.width;
    const h: i64 = img.height;
    var err: i64 = 0;
    var dx: i64 = radius;
    var dy: i64 = 0;
    var plus: i64 = 1;
    var minus: i64 = (radius << 1) - 1;
    const inside = cx >= radius and cx < w - radius and cy >= radius and cy < h - radius;
    while (dx >= dy) {
        const y11 = cy - dy;
        const y12 = cy + dy;
        const y21 = cy - dx;
        const y22 = cy + dx;
        var x11 = cx - dx;
        var x12 = cx + dx;
        var x21 = cx - dy;
        var x22 = cx + dy;
        if (inside) {
            if (!fill) {
                putPixel(img, x11, y11, color);
                putPixel(img, x11, y12, color);
                putPixel(img, x12, y11, color);
                putPixel(img, x12, y12, color);
                putPixel(img, x21, y21, color);
                putPixel(img, x21, y22, color);
                putPixel(img, x22, y21, color);
                putPixel(img, x22, y22, color);
            } else {
                hline(img, y11, x11, x12, color);
                hline(img, y12, x11, x12, color);
                hline(img, y21, x21, x22, color);
                hline(img, y22, x21, x22, color);
            }
        } else if (x11 < w and x12 >= 0 and y21 < h and y22 >= 0) {
            if (fill) {
                x11 = @max(x11, 0);
                x12 = @min(x12, w - 1);
            }
            inline for (.{ y11, y12 }) |yy| {
                if (yy >= 0 and yy < h) {
                    if (!fill) {
                        if (x11 >= 0) putPixel(img, x11, yy, color);
                        if (x12 < w) putPixel(img, x12, yy, color);
                    } else hline(img, yy, x11, x12, color);
                }
            }
            if (x21 < w and x22 >= 0) {
                if (fill) {
                    x21 = @max(x21, 0);
                    x22 = @min(x22, w - 1);
                }
                inline for (.{ y21, y22 }) |yy| {
                    if (yy >= 0 and yy < h) {
                        if (!fill) {
                            if (x21 >= 0) putPixel(img, x21, yy, color);
                            if (x22 < w) putPixel(img, x22, yy, color);
                        } else hline(img, yy, x21, x22, color);
                    }
                }
            }
        }
        dy += 1;
        err += plus;
        plus += 2;
        const mask: i64 = @as(i64, @intFromBool(err <= 0)) - 1;
        err -= minus & mask;
        dx += mask;
        minus -= mask & 2;
    }
}

/// OpenCV's ThickLine (shift = 0 input). `flags` bit 0 / bit 1 round the
/// start / end with a disc.
fn thickLine(img: Image(u8), p0_in: P64, p1_in: P64, color: []const u8, thickness_in: i64, line_type: LineType, flags: u8) void {
    var p0 = p0_in;
    var p1 = p1_in;
    var thickness = thickness_in;
    const w: i64 = img.width;
    const h: i64 = img.height;
    const contains = struct {
        fn f(p: P64, ww: i64, hh: i64) bool {
            return p.x >= 0 and p.y >= 0 and p.x < ww and p.y < hh;
        }
    }.f;
    if (thickness > 1 and (!contains(p0, w, h) or !contains(p1, w, h))) {
        const m = thickness;
        p0.x += m;
        p0.y += m;
        p1.x += m;
        p1.y += m;
        _ = clipLine(w + 2 * m, h + 2 * m, &p0, &p1);
        p0.x -= m;
        p0.y -= m;
        p1.x -= m;
        p1.y -= m;
    }
    p0.x <<= XY_SHIFT;
    p0.y <<= XY_SHIFT;
    p1.x <<= XY_SHIFT;
    p1.y <<= XY_SHIFT;
    if (thickness <= 1) {
        p0.x = (p0.x + (XY_ONE >> 1)) >> XY_SHIFT;
        p0.y = (p0.y + (XY_ONE >> 1)) >> XY_SHIFT;
        p1.x = (p1.x + (XY_ONE >> 1)) >> XY_SHIFT;
        p1.y = (p1.y + (XY_ONE >> 1)) >> XY_SHIFT;
        line(img, p0, p1, color, @backingInt(line_type));
        return;
    }
    const inv: f64 = 1.0 / @as(f64, @floatFromInt(XY_ONE));
    const dxf = @as(f64, @floatFromInt(p0.x - p1.x)) * inv;
    const dyf = @as(f64, @floatFromInt(p1.y - p0.y)) * inv;
    var r = dxf * dxf + dyf * dyf;
    const odd = thickness & 1;
    thickness <<= XY_SHIFT - 1;
    if (@abs(r) > std.math.floatEps(f64)) {
        r = (@as(f64, @floatFromInt(thickness)) + @as(f64, @floatFromInt(odd)) * @as(f64, @floatFromInt(XY_ONE)) * 0.5) / @sqrt(r);
        const dpx = cvRound(dyf * r);
        const dpy = cvRound(dxf * r);
        const pt = [4]P64{
            .{ .x = p0.x + dpx, .y = p0.y + dpy },
            .{ .x = p0.x - dpx, .y = p0.y - dpy },
            .{ .x = p1.x - dpx, .y = p1.y - dpy },
            .{ .x = p1.x + dpx, .y = p1.y + dpy },
        };
        fillConvexPolyImpl(img, &pt, color, line_type, XY_SHIFT);
    }
    for (0..2) |i| {
        if (flags & (@as(u8, 1) << @intCast(i)) != 0) {
            const cx = (p0.x + (XY_ONE >> 1)) >> XY_SHIFT;
            const cy = (p0.y + (XY_ONE >> 1)) >> XY_SHIFT;
            circleImpl(img, cx, cy, (thickness + (XY_ONE >> 1)) >> XY_SHIFT, color, true);
        }
        p0 = p1;
    }
}

/// line(img, pt1, pt2, color, thickness, lineType) as OpenCV draws it.
pub fn drawLine(img: Image(u8), p0: Point, p1: Point, color: []const u8, thickness: u32, line_type: LineType) void {
    thickLine(img, .{ .x = p0.x, .y = p0.y }, .{ .x = p1.x, .y = p1.y }, color, thickness, line_type, 3);
}

/// polylines(img, [pts], is_closed, color, thickness, lineType).
pub fn polylines(img: Image(u8), pts: []const Point, is_closed: bool, color: []const u8, thickness: u32, line_type: LineType) void {
    if (pts.len == 0) return;
    var i: usize = if (is_closed) pts.len - 1 else 0;
    var flags: u8 = if (is_closed) 2 else 3;
    var p0: P64 = .{ .x = pts[i].x, .y = pts[i].y };
    i = @intFromBool(!is_closed);
    while (i < pts.len) : (i += 1) {
        const p: P64 = .{ .x = pts[i].x, .y = pts[i].y };
        thickLine(img, p0, p, color, thickness, line_type, flags);
        p0 = p;
        flags = 2;
    }
}

// ------------------------------------------------------- general polygons

const PolyEdge = struct {
    y0: i32 = 0,
    y1: i32 = 0,
    x: i64 = 0,
    dx: i64 = 0,
    next: ?u32 = null,
};

fn collectPolyEdges(gpa: Allocator, img: Image(u8), v: []const Point, edges: *std.ArrayList(PolyEdge), color: []const u8, line_type: LineType, offset: Point) Allocator.Error!void {
    const count = v.len;
    const delta: i64 = offset.y;
    var pt0: P64 = .{ .x = @as(i64, v[count - 1].x + offset.x) << XY_SHIFT, .y = v[count - 1].y + delta };
    try edges.ensureUnusedCapacity(gpa, count);
    for (v) |vi| {
        const pt1: P64 = .{ .x = @as(i64, vi.x + offset.x) << XY_SHIFT, .y = vi.y + delta };
        defer pt0 = pt1;
        var pt0c = pt0;
        var pt1c = pt1;
        var t0: P64 = .{ .x = (pt0.x + (XY_ONE >> 1)) >> XY_SHIFT, .y = pt0.y };
        var t1: P64 = .{ .x = (pt1.x + (XY_ONE >> 1)) >> XY_SHIFT, .y = pt1.y };
        line(img, t0, t1, color, @backingInt(line_type));
        const w: i64 = img.width;
        const h: i64 = img.height;
        if (t0.x < 0 or t0.x >= w or t1.x < 0 or t1.x >= w or t0.y < 0 or t0.y >= h or t1.y < 0 or t1.y >= h) {
            // clipLine(Size, Point&, Point&) works on int points.
            t0 = .{ .x = @as(i32, @truncate(t0.x)), .y = @as(i32, @truncate(t0.y)) };
            t1 = .{ .x = @as(i32, @truncate(t1.x)), .y = @as(i32, @truncate(t1.y)) };
            _ = clipLine(w, h, &t0, &t1);
            if (t0.y != t1.y) {
                pt0c.y = t0.y;
                pt1c.y = t1.y;
            }
        }
        pt0c.x = t0.x << XY_SHIFT;
        pt1c.x = t1.x << XY_SHIFT;
        if (pt0.y == pt1.y) continue;
        var e: PolyEdge = .{};
        e.dx = @divTrunc(pt1c.x - pt0c.x, pt1c.y - pt0c.y);
        if (pt0.y < pt1.y) {
            e.y0 = @intCast(pt0.y);
            e.y1 = @intCast(pt1.y);
            e.x = pt0c.x + (pt0.y - pt0c.y) * e.dx;
        } else {
            e.y0 = @intCast(pt1.y);
            e.y1 = @intCast(pt0.y);
            e.x = pt1c.x + (pt1.y - pt1c.y) * e.dx;
        }
        edges.appendAssumeCapacity(e);
    }
}

fn edgeLess(_: void, a: PolyEdge, b: PolyEdge) bool {
    if (a.y0 != b.y0) return a.y0 < b.y0;
    if (a.x != b.x) return a.x < b.x;
    return a.dx < b.dx;
}

fn fillEdgeCollection(gpa: Allocator, img: Image(u8), edges: *std.ArrayList(PolyEdge), color: []const u8) Allocator.Error!void {
    const total = edges.items.len;
    const w: i64 = img.width;
    const h: i64 = img.height;
    const delta: i64 = XY_ONE - 1;
    if (total < 2) return;
    var y_max: i32 = std.math.minInt(i32);
    var y_min: i32 = std.math.maxInt(i32);
    var x_max: i64 = -1;
    var x_min: i64 = std.math.maxInt(i64);
    for (edges.items) |e1| {
        std.debug.assert(e1.y0 < e1.y1);
        const x1 = e1.x + (e1.y1 - e1.y0) * e1.dx;
        y_min = @min(y_min, e1.y0);
        y_max = @max(y_max, e1.y1);
        x_min = @min(x_min, e1.x);
        x_max = @max(x_max, e1.x);
        x_min = @min(x_min, x1);
        x_max = @max(x_max, x1);
    }
    if (y_max < 0 or y_min >= h or x_max < 0 or x_min >= (w << XY_SHIFT)) return;
    // OpenCV sorts with std::sort; equal keys are interchangeable here
    // because the scanline fill only depends on x, dx and the y range.
    std.sort.pdq(PolyEdge, edges.items, {}, edgeLess);
    try edges.append(gpa, .{ .y0 = std.math.maxInt(i32) });
    const E = edges.items;
    // Node `total + 1` is the list head (OpenCV's `tmp`).
    var head: PolyEdge = .{};
    const at = struct {
        fn f(es: []PolyEdge, hd: *PolyEdge, i: ?u32) *PolyEdge {
            return if (i) |k| &es[k] else hd;
        }
    }.f;
    var i: u32 = 0;
    var e: u32 = 0;
    y_max = @min(y_max, @as(i32, @intCast(h)));
    var y: i32 = E[0].y0;
    while (y < y_max) : (y += 1) {
        var draw = false;
        const clipline = y < 0;
        var prelast: ?u32 = null; // null = head
        var last: ?u32 = head.next;
        while (last != null or E[e].y0 == y) {
            if (last) |l| {
                if (E[l].y1 == y) {
                    at(E, &head, prelast).next = E[l].next;
                    last = E[l].next;
                    continue;
                }
            }
            const keep_prelast = prelast;
            if (last != null and (E[e].y0 > y or E[last.?].x < E[e].x)) {
                prelast = last;
                last = E[last.?].next;
            } else if (i < total) {
                at(E, &head, prelast).next = e;
                E[e].next = last;
                prelast = e;
                i += 1;
                e = i;
            } else break;
            if (draw) {
                const kp = at(E, &head, keep_prelast);
                const pl = at(E, &head, prelast);
                if (!clipline) {
                    var x1: i64 = undefined;
                    var x2: i64 = undefined;
                    if (kp.x > pl.x) {
                        x1 = @as(i32, @truncate((pl.x + delta) >> XY_SHIFT));
                        x2 = @as(i32, @truncate(kp.x >> XY_SHIFT));
                    } else {
                        x1 = @as(i32, @truncate((kp.x + delta) >> XY_SHIFT));
                        x2 = @as(i32, @truncate(pl.x >> XY_SHIFT));
                    }
                    if (x1 < w and x2 >= 0) {
                        if (x1 < 0) x1 = 0;
                        if (x2 >= w) x2 = w - 1;
                        hline(img, y, x1, x2, color);
                    }
                }
                kp.x += kp.dx;
                pl.x += pl.dx;
            }
            draw = !draw;
        }
        // Bubble-sort the active list by x.
        var keep: ?u32 = null;
        while (true) {
            var pl: ?u32 = null;
            var l: ?u32 = head.next;
            var last_exchange: ?u32 = null;
            var have_exchange = false;
            while (l != keep and l != null and E[l.?].next != null) {
                const te = E[l.?].next.?;
                if (E[l.?].x > E[te].x) {
                    at(E, &head, pl).next = te;
                    E[l.?].next = E[te].next;
                    E[te].next = l;
                    pl = te;
                    last_exchange = pl;
                    have_exchange = true;
                } else {
                    pl = l;
                    l = te;
                }
            }
            if (!have_exchange) break;
            keep = last_exchange;
            if (keep == head.next) break;
        }
    }
}

/// fillPoly(img, polygons, color, lineType, shift = 0, offset): fills the
/// area enclosed by the polygons with the even-odd rule.
pub fn fillPoly(gpa: Allocator, img: Image(u8), polys: []const []const Point, color: []const u8, line_type: LineType, offset: Point) Allocator.Error!void {
    var edges: std.ArrayList(PolyEdge) = .empty;
    defer edges.deinit(gpa);
    var total: usize = 0;
    for (polys) |p| total += p.len;
    try edges.ensureTotalCapacity(gpa, total + 1);
    for (polys) |p| {
        if (p.len > 0) try collectPolyEdges(gpa, img, p, &edges, color, line_type, offset);
    }
    try fillEdgeCollection(gpa, img, &edges, color);
}

/// drawContours(img, contours, contourIdx, color, thickness, lineType,
/// hierarchy, maxLevel, offset). `index` < 0 draws every contour;
/// `thickness` < 0 fills (holes from the hierarchy stay unfilled through
/// the even-odd rule, as in OpenCV). With a hierarchy and `max_level` > 0,
/// contours nested up to `max_level` below the selected ones are drawn too.
pub fn drawContours(gpa: Allocator, img: Image(u8), cs: contours.Contours, index: i32, color: []const u8, thickness: i32, line_type: LineType, hierarchy: ?[]const [4]i32, max_level: i32, offset: Point) (Allocator.Error || error{InvalidArgument})!void {
    const n = cs.len();
    if (n == 0) return;
    if (index >= @as(i64, @intCast(n))) return error.InvalidArgument;
    var start: usize = 0;
    var end: usize = n;
    if (index >= 0) {
        start = @intCast(index);
        end = start + 1;
    }
    var to_fill: std.ArrayList(usize) = .empty;
    defer to_fill.deinit(gpa);
    if (hierarchy == null or max_level == 0) {
        for (start..end) |i| try to_fill.append(gpa, i);
    } else {
        const hier = hierarchy.?;
        if (hier.len < n) return error.InvalidArgument;
        var stack: std.ArrayList(usize) = .empty;
        defer stack.deinit(gpa);
        for (start..end) |i| {
            if (hier[i][3] < 0 or index >= 0) try stack.append(gpa, i);
        }
        while (stack.pop()) |cur| {
            var level: i32 = -1;
            var par: i32 = @intCast(cur);
            while (par >= 0) : (level += 1) par = hier[@intCast(par)][3];
            if (level <= max_level) try to_fill.append(gpa, cur);
            var next = hier[cur][2];
            while (next > 0) : (next = hier[@intCast(next)][0]) try stack.append(gpa, @intCast(next));
        }
    }
    if (thickness < 0) {
        const polys = try gpa.alloc([]const Point, to_fill.items.len);
        defer gpa.free(polys);
        for (to_fill.items, polys) |i, *p| p.* = cs.get(i);
        try fillPoly(gpa, img, polys, color, line_type, offset);
    } else {
        for (to_fill.items) |i| {
            const c = cs.get(i);
            for (c, 0..) |p1, j| {
                const p2 = c[if (j + 1 == c.len) 0 else j + 1];
                thickLine(img, .{ .x = p1.x + offset.x, .y = p1.y + offset.y }, .{ .x = p2.x + offset.x, .y = p2.y + offset.y }, color, thickness, line_type, 2);
            }
        }
    }
}

test "fillPoly square and thick line" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 10, 10, 1, 0);
    defer img.deinit(gpa);
    const sq = [_]Point{ .{ .x = 2, .y = 2 }, .{ .x = 6, .y = 2 }, .{ .x = 6, .y = 5 }, .{ .x = 2, .y = 5 } };
    try fillPoly(gpa, img, &.{&sq}, &.{255}, .eight, .{ .x = 0, .y = 0 });
    var n: usize = 0;
    for (img.slice()) |v| n += @intFromBool(v != 0);
    try std.testing.expectEqual(@as(usize, 20), n);
    @memset(img.slice(), 0);
    drawLine(img, .{ .x = 1, .y = 1 }, .{ .x = 8, .y = 1 }, &.{255}, 3, .eight);
    try std.testing.expectEqual(@as(u8, 255), img.at(4, 2, 0));
    try std.testing.expectEqual(@as(u8, 0), img.at(4, 4, 0));
}
