//! Image inpainting by the fast marching method (OpenCV's cv::inpaint):
//! Telea's method (INPAINT_TELEA) and the Navier-Stokes based method
//! (INPAINT_NS), bit-exact with OpenCV.
//!
//! Both walk the unknown region from its boundary inwards in order of the
//! arrival time T of a front solving |grad T| = 1, and fill each pixel when
//! the front reaches it from the already known pixels within `radius`. The
//! order is fixed by OpenCV's priority queue key (T, then insertion order),
//! which is unique, so any priority queue pops in the same order; every
//! float operation is replayed in OpenCV's order and precision.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error || error{InvalidArgument};

pub const Method = enum(u8) {
    /// Navier-Stokes based method (cv::INPAINT_NS).
    ns = 0,
    /// Alexandru Telea's method (cv::INPAINT_TELEA).
    telea = 1,
};

const KNOWN: u8 = 0;
const BAND: u8 = 1;
const INSIDE: u8 = 2;
const CHANGE: u8 = 3;

/// OpenCV's CvPriorityQueueFloat: a min-queue keyed by (T, insertion order).
const Queue = struct {
    nodes: std.ArrayList(Node) = .empty,
    next_order: u32 = 0,

    const Node = struct { key: u64, i: u32, j: u32 };

    fn makeKey(t: f32, order: u32) u64 {
        var bits: u32 = @bitCast(t);
        if (bits << 1 == 0) bits = 0;
        bits = if (bits & 0x80000000 != 0) ~bits else bits | 0x80000000;
        return (@as(u64, bits) << 32) | order;
    }

    fn deinit(q: *Queue, gpa: Allocator) void {
        q.nodes.deinit(gpa);
    }

    fn push(q: *Queue, gpa: Allocator, i: usize, j: usize, t: f32) !void {
        const node: Node = .{ .key = makeKey(t, q.next_order), .i = @intCast(i), .j = @intCast(j) };
        q.next_order +%= 1;
        try q.nodes.append(gpa, node);
        const n = q.nodes.items;
        var pos = n.len - 1;
        while (pos > 0) {
            const parent = (pos - 1) >> 2;
            if (n[parent].key < node.key) break;
            n[pos] = n[parent];
            pos = parent;
        }
        n[pos] = node;
    }

    fn pop(q: *Queue) ?[2]usize {
        const n = q.nodes.items;
        if (n.len == 0) return null;
        const top = n[0];
        const last = n[n.len - 1];
        const size = n.len - 1;
        q.nodes.items.len = size;
        if (size > 0) {
            var pos: usize = 0;
            while (true) {
                const child = 4 * pos + 1;
                if (child >= size) break;
                var best = child;
                var bk = n[child].key;
                const end = @min(child + 4, size);
                for (child + 1..end) |c| {
                    if (n[c].key < bk) {
                        bk = n[c].key;
                        best = c;
                    }
                }
                if (last.key < bk) break;
                n[pos] = n[best];
                pos = best;
            }
            n[pos] = last;
        }
        return .{ top.i, top.j };
    }
};

/// The working grids, one pixel larger than the image on every side.
const Grid = struct {
    rows: usize,
    cols: usize,
    f: []u8,
    t: []f32,

    inline fn F(g: *const Grid, i: usize, j: usize) u8 {
        return g.f[i * g.cols + j];
    }
    inline fn T(g: *const Grid, i: usize, j: usize) f32 {
        return g.t[i * g.cols + j];
    }
};

fn solve(f: []const u8, t: []const f32, a: usize, b: usize) f32 {
    const a11: f64 = t[a];
    const a22: f64 = t[b];
    const m12 = @min(a11, a22);
    const sol: f64 = if (f[a] != INSIDE)
        (if (f[b] != INSIDE)
            (if (@abs(a11 - a22) >= 1.0) 1 + m12 else (a11 + a22 + @sqrt(2 - (a11 - a22) * (a11 - a22))) * 0.5)
        else
            1 + a11)
    else if (f[b] != INSIDE)
        1 + a22
    else
        1 + m12;
    return @floatCast(sol);
}

fn min4(f: []const u8, t: []const f32, cols: usize, i: usize, j: usize) f32 {
    const c = i * cols + j;
    const up = c - cols;
    const dn = c + cols;
    const a = solve(f, t, up, c - 1);
    const b = solve(f, t, dn, c - 1);
    const cc = solve(f, t, up, c + 1);
    const d = solve(f, t, dn, c + 1);
    return @min(@min(a, b), @min(cc, d));
}

/// OpenCV's icvCalcFMM with negate = true: arrival times outside the mask,
/// stored negated.
fn calcFmmOutside(gpa: Allocator, f: []u8, t: []f32, rows: usize, cols: usize, q: *Queue, ia0: usize, ia1: usize, ja0: usize, ja1: usize) !void {
    while (q.pop()) |p| {
        const ii = p[0];
        const jj = p[1];
        f[ii * cols + jj] = CHANGE;
        for (0..4) |k| {
            const i: isize = @as(isize, @intCast(ii)) + @as(isize, switch (k) {
                0 => -1,
                2 => 1,
                else => 0,
            });
            const j: isize = @as(isize, @intCast(jj)) + @as(isize, switch (k) {
                1 => -1,
                3 => 1,
                else => 0,
            });
            if (i <= 0 or j <= 0 or i > rows or j > cols) continue;
            const iu: usize = @intCast(i);
            const ju: usize = @intCast(j);
            if (f[iu * cols + ju] == INSIDE) {
                const dist = min4(f, t, cols, iu, ju);
                t[iu * cols + ju] = dist;
                f[iu * cols + ju] = BAND;
                try q.push(gpa, iu, ju, dist);
            }
        }
    }
    for (ia0..ia1) |i| for (ja0..ja1) |j| {
        const p = i * cols + j;
        if (f[p] == CHANGE) {
            f[p] = KNOWN;
            t[p] = -t[p];
        }
    };
}

/// ring = max of `f` (0 or INSIDE) over the (2r+1)^2 square, minus `f`
/// and `band` (saturating), for rows [ia0, ia1) and columns [ja0, ja1)
/// of the interior; cv::dilate with a rectangle and two cv::subtract.
fn dilateRing(gpa: Allocator, f: []const u8, band: []const u8, ring: []u8, cols: usize, ia0: usize, ia1: usize, ja0: usize, ja1: usize, r: usize) !void {
    const L = 32;
    const Vb = @Vector(L, u8);
    // Source columns [ja0 - r, ja1 + r) clipped to the grid; the frame is 0.
    const x0 = ja0 -| r;
    const x1 = @min(ja1 + r, cols);
    const rw = ja1 - ja0;
    const rwv = (rw + L - 1) / L * L;
    const nrows = ia1 - ia0;
    const buf = try gpa.alloc(u8, (nrows + 2 * r) * rwv + 2 * rwv + 2 * r + 2 * L);
    defer gpa.free(buf);
    // Row maxima of rows [ia0 - r, ia1 + r), clipped to the grid.
    const hm = buf[0 .. (nrows + 2 * r) * rwv];
    const pad = buf[hm.len..][0 .. rwv + 2 * r + L];
    const acc = buf[hm.len + pad.len ..][0..rwv];
    const y0 = ia0 -| r;
    const y1 = @min(ia1 + r, f.len / cols);
    for (y0..y1) |y| {
        @memset(pad, 0);
        // pad[k] is column ja0 - r + k.
        const off = x0 + r - ja0;
        @memcpy(pad[off..][0 .. x1 - x0], f[y * cols + x0 ..][0 .. x1 - x0]);
        const d = hm[(y - y0) * rwv ..][0..rwv];
        var x: usize = 0;
        while (x < rwv) : (x += L) {
            var m: Vb = pad[x..][0..L].*;
            for (1..2 * r + 1) |k| m = @max(m, @as(Vb, pad[x + k ..][0..L].*));
            d[x..][0..L].* = m;
        }
    }
    for (ia0..ia1) |i| {
        const a = @max(i -| r, y0);
        const b = @min(i + r + 1, y1);
        @memset(acc, 0);
        for (a..b) |yy| {
            const srow = hm[(yy - y0) * rwv ..][0..rwv];
            var x: usize = 0;
            while (x < rwv) : (x += L) acc[x..][0..L].* = @max(@as(Vb, acc[x..][0..L].*), @as(Vb, srow[x..][0..L].*));
        }
        const p = i * cols + ja0;
        for (ring[p..][0..rw], acc[0..rw], f[p..][0..rw], band[p..][0..rw]) |*o, m, fv, bv| o.* = m -| fv -| bv;
    }
}

const Disk = struct {
    range: usize,
    side: usize,
    half_width: []usize,
    dst: []f32,

    fn init(gpa: Allocator, r: usize) !Disk {
        const side = 2 * r + 1;
        const hw = try gpa.alloc(usize, side);
        errdefer gpa.free(hw);
        const dst = try gpa.alloc(f32, side * side);
        const ri: isize = @intCast(r);
        var dk: isize = -ri;
        while (dk <= ri) : (dk += 1) {
            var w: isize = 0;
            while ((w + 1) * (w + 1) + dk * dk <= ri * ri) w += 1;
            hw[@intCast(dk + ri)] = @intCast(w);
        }
        return .{ .range = r, .side = side, .half_width = hw, .dst = dst };
    }

    fn deinit(d: *Disk, gpa: Allocator) void {
        gpa.free(d.half_width);
        gpa.free(d.dst);
    }

    inline fn at(d: *const Disk, dk: isize, dl: isize) f32 {
        const r: isize = @intCast(d.range);
        return d.dst[@as(usize, @intCast(dk + r)) * d.side + @as(usize, @intCast(dl + r))];
    }
};

fn vlen(x: f32, y: f32) f32 {
    return x * x + y * y;
}

/// The pixel type's conversion of the integer or float difference to f32.
inline fn diffF(comptime T: type, a: T, b: T) f32 {
    if (T == f32) return a - b;
    return @floatFromInt(@as(i32, a) - @as(i32, b));
}

inline fn absDiffF(comptime T: type, a: T, b: T) f32 {
    if (T == f32) return @abs(a - b);
    return @floatFromInt(@abs(@as(i32, a) - @as(i32, b)));
}

inline fn toF(comptime T: type, a: T) f32 {
    if (T == f32) return a;
    return @floatFromInt(a);
}

/// cvRound of a double, saturated to T (saturate_cast<T>(double)).
inline fn satD(comptime T: type, v: f64) T {
    if (T == f32) return @floatCast(v);
    const r = @round(v);
    const re = if (@abs(v - @trunc(v)) == 0.5) 2.0 * @round(v / 2.0) else r;
    const lo: f64 = 0;
    const hi: f64 = @floatFromInt(std.math.maxInt(T));
    if (!(re >= lo)) return 0;
    if (re > hi) return std.math.maxInt(T);
    return @intFromFloat(re);
}

fn Inpainter(comptime T: type, comptime cn: usize) type {
    return struct {
        const Self = @This();
        g: Grid,
        out: Image(T),
        disk: Disk,

        inline fn o(self: *const Self, row: usize, col: usize, c: usize) T {
            return self.out.data[row * self.out.stride + col * cn + c];
        }

        fn gradT(self: *const Self, i: usize, j: usize) [2]f32 {
            const g = &self.g;
            var gx: f32 = 0;
            var gy: f32 = 0;
            if (g.F(i, j + 1) != INSIDE) {
                gx = if (g.F(i, j - 1) != INSIDE) (g.T(i, j + 1) - g.T(i, j - 1)) * 0.5 else g.T(i, j + 1) - g.T(i, j);
            } else if (g.F(i, j - 1) != INSIDE) gx = g.T(i, j) - g.T(i, j - 1);
            if (g.F(i + 1, j) != INSIDE) {
                gy = if (g.F(i - 1, j) != INSIDE) (g.T(i + 1, j) - g.T(i - 1, j)) * 0.5 else g.T(i + 1, j) - g.T(i, j);
            } else if (g.F(i - 1, j) != INSIDE) gy = g.T(i, j) - g.T(i - 1, j);
            return .{ gx, gy };
        }

        fn telea(self: *Self, i: usize, j: usize) void {
            const g = &self.g;
            const range = self.disk.range;
            const tij = g.T(i, j);
            const grad = self.gradT(i, j);
            const kbeg = @max(i -| range, 1);
            const kend = @min(i + range, g.rows - 2);
            var ia: [cn]f32 = @splat(0);
            var jx: [cn]f32 = @splat(0);
            var jy: [cn]f32 = @splat(0);
            var s: [cn]f32 = @splat(1.0e-20);
            var k = kbeg;
            while (k <= kend) : (k += 1) {
                const km = k - 1 + @intFromBool(k == 1);
                const kp = k - 1 - @intFromBool(k == g.rows - 2);
                const dk = @as(isize, @intCast(k)) - @as(isize, @intCast(i));
                const rhw = self.disk.half_width[@intCast(dk + @as(isize, @intCast(range)))];
                const lbeg = @max(j -| rhw, 1);
                const lend = @min(j + rhw, g.cols - 2);
                if (lbeg > lend) continue;
                const ry: f32 = @floatFromInt(-dk);
                var l = lbeg;
                while (l <= lend) : (l += 1) {
                    if (g.F(k, l) == INSIDE) continue;
                    const dl = @as(isize, @intCast(l)) - @as(isize, @intCast(j));
                    const rx: f32 = @floatFromInt(-dl);
                    const lev: f32 = @floatCast(1.0 / (1.0 + @as(f64, @abs(g.T(k, l) - tij))));
                    var dir: f32 = rx * grad[0] + ry * grad[1];
                    if (@abs(@as(f64, dir)) <= 0.01) dir = 0.000001;
                    const w: f32 = @abs(self.disk.at(dk, dl) * lev * dir);
                    const lm = l - 1 + @intFromBool(l == 1);
                    const lp = l - 1 - @intFromBool(l == g.cols - 2);
                    const fr = g.F(k, l + 1) != INSIDE;
                    const fl = g.F(k, l - 1) != INSIDE;
                    const fd = g.F(k + 1, l) != INSIDE;
                    const fu = g.F(k - 1, l) != INSIDE;
                    inline for (0..cn) |c| {
                        var gix: f32 = 0;
                        if (fr) {
                            gix = if (fl) diffF(T, self.o(km, lp + 1, c), self.o(km, lm - 1, c)) * 2.0 else diffF(T, self.o(km, lp + 1, c), self.o(km, lm, c));
                        } else if (fl) gix = diffF(T, self.o(km, lp, c), self.o(km, lm - 1, c));
                        var giy: f32 = 0;
                        if (fd) {
                            giy = if (fu) diffF(T, self.o(kp + 1, lm, c), self.o(km - 1, lm, c)) * 2.0 else diffF(T, self.o(kp + 1, lm, c), self.o(km, lm, c));
                        } else if (fu) giy = diffF(T, self.o(kp, lm, c), self.o(km - 1, lm, c));
                        ia[c] += w * toF(T, self.o(k - 1, l - 1, c));
                        jx[c] -= w * (gix * rx);
                        jy[c] -= w * (giy * ry);
                        s[c] += w;
                    }
                }
            }
            inline for (0..cn) |c| {
                const sat: f32 = ia[c] / s[c] + (jx[c] + jy[c]) / (@sqrt(jx[c] * jx[c] + jy[c] * jy[c]) + 1.0e-20);
                self.out.data[(i - 1) * self.out.stride + (j - 1) * cn + c] = if (T == f32) sat else satD(T, @as(f64, sat) + 0.5);
            }
        }

        fn ns(self: *Self, i: usize, j: usize) void {
            const g = &self.g;
            const range = self.disk.range;
            const kbeg = @max(i -| range, 1);
            const kend = @min(i + range, g.rows - 2);
            var ia: [cn]f32 = @splat(0);
            var s: [cn]f32 = @splat(1.0e-20);
            var k = kbeg;
            while (k <= kend) : (k += 1) {
                const km = k - 1 + @intFromBool(k == 1);
                const kp = k - 1 - @intFromBool(k == g.rows - 2);
                const dk = @as(isize, @intCast(k)) - @as(isize, @intCast(i));
                const rhw = self.disk.half_width[@intCast(dk + @as(isize, @intCast(range)))];
                const lbeg = @max(j -| rhw, 1);
                const lend = @min(j + rhw, g.cols - 2);
                const ry: f32 = @floatFromInt(-dk);
                var l = lbeg;
                while (l <= lend) : (l += 1) {
                    if (g.F(k, l) == INSIDE) continue;
                    const dl = @as(isize, @intCast(l)) - @as(isize, @intCast(j));
                    const rx: f32 = @floatFromInt(-dl);
                    const lm = l - 1 + @intFromBool(l == 1);
                    const lp = l - 1 - @intFromBool(l == g.cols - 2);
                    const dst = self.disk.at(dk, dl);
                    const fd = g.F(k + 1, l) != INSIDE;
                    const fu = g.F(k - 1, l) != INSIDE;
                    const fr = g.F(k, l + 1) != INSIDE;
                    const fl = g.F(k, l - 1) != INSIDE;
                    inline for (0..cn) |c| {
                        var gix: f32 = 0;
                        if (fd) {
                            gix = if (fu)
                                absDiffF(T, self.o(kp + 1, lm, c), self.o(kp, lm, c)) + absDiffF(T, self.o(kp, lm, c), self.o(km - 1, lm, c))
                            else
                                absDiffF(T, self.o(kp + 1, lm, c), self.o(kp, lm, c)) * 2.0;
                        } else if (fu) gix = absDiffF(T, self.o(kp, lm, c), self.o(km - 1, lm, c)) * 2.0;
                        var giy: f32 = 0;
                        if (fr) {
                            giy = if (fl)
                                absDiffF(T, self.o(km, lp + 1, c), self.o(km, lm, c)) + absDiffF(T, self.o(km, lm, c), self.o(km, lm - 1, c))
                            else
                                absDiffF(T, self.o(km, lp + 1, c), self.o(km, lm, c)) * 2.0;
                        } else if (fl) giy = absDiffF(T, self.o(km, lm, c), self.o(km, lm - 1, c)) * 2.0;
                        gix = -gix;
                        var dir: f32 = rx * gix + ry * giy;
                        if (@abs(@as(f64, dir)) <= 0.01) {
                            dir = 0.000001;
                        } else {
                            dir = @abs(dir / @sqrt(vlen(rx, ry) * vlen(gix, giy)));
                        }
                        const w = dst * dir;
                        ia[c] += w * toF(T, self.o(k - 1, l - 1, c));
                        s[c] += w;
                    }
                }
            }
            inline for (0..cn) |c| {
                self.out.data[(i - 1) * self.out.stride + (j - 1) * cn + c] = satD(T, @as(f64, ia[c]) / @as(f64, s[c]));
            }
        }

        fn march(self: *Self, gpa: Allocator, q: *Queue, method: Method) !void {
            const g = &self.g;
            while (q.pop()) |p| {
                const ii = p[0];
                const jj = p[1];
                g.f[ii * g.cols + jj] = KNOWN;
                for (0..4) |k| {
                    const i: usize = switch (k) {
                        0 => ii - 1,
                        2 => ii + 1,
                        else => ii,
                    };
                    const j: usize = switch (k) {
                        1 => jj - 1,
                        3 => jj + 1,
                        else => jj,
                    };
                    if (i == 0 or j == 0 or i > g.rows - 1 or j > g.cols - 1) continue;
                    if (g.F(i, j) == INSIDE) {
                        const dist = min4(g.f, g.t, g.cols, i, j);
                        g.t[i * g.cols + j] = dist;
                        switch (method) {
                            .telea => self.telea(i, j),
                            .ns => self.ns(i, j),
                        }
                        g.f[i * g.cols + j] = BAND;
                        try q.push(gpa, i, j, dist);
                    }
                }
            }
        }
    };
}

/// Fills the pixels of `dst` where `mask` is non-zero from their
/// surroundings (cv::inpaint). `src`: 8-bit 1 or 3 channels, or 16-bit or
/// float 1 channel; `dst` of the same shape (may be `src`); `mask` 8-bit,
/// 1 channel. `radius` is OpenCV's inpaintRadius (rounded, clamped to
/// 1..100).
pub fn inpaint(comptime T: type, gpa: Allocator, src: Image(T), mask: Image(u8), dst: Image(T), radius: f64, method: Method) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (!src.sameSize(mask) or mask.channels != 1) return error.SizeMismatch;
    switch (src.channels) {
        1 => return inpaintCn(T, 1, gpa, src, mask, dst, radius, method),
        3 => if (T == u8) return inpaintCn(T, 3, gpa, src, mask, dst, radius, method),
        else => {},
    }
    return error.UnsupportedChannels;
}

fn roundEvenI(x: f64) i64 {
    if (@abs(x - @trunc(x)) == 0.5) return @intFromFloat(2.0 * @round(x / 2.0));
    return @intFromFloat(@round(x));
}

fn inpaintCn(comptime T: type, comptime cn: usize, gpa: Allocator, src: Image(T), mask: Image(u8), dst: Image(T), radius: f64, method: Method) Error!void {
    if (std.math.isNan(radius)) return error.InvalidArgument;
    const range: usize = @intCast(std.math.clamp(roundEvenI(std.math.clamp(radius, -1e9, 1e9)), 1, 100));
    const h: usize = src.height;
    const w: usize = src.width;
    const rows = h + 2;
    const cols = w + 2;
    const n = rows * cols;

    if (src.data != dst.data) for (0..h) |y| @memcpy(dst.row(y), src.row(y));

    const fbuf = try gpa.alloc(u8, 3 * n);
    defer gpa.free(fbuf);
    const f = fbuf[0..n];
    const band = fbuf[n .. 2 * n];
    const ring = fbuf[2 * n .. 3 * n];
    const t = try gpa.alloc(f32, n);
    defer gpa.free(t);
    @memset(f, KNOWN);
    @memset(t, 1.0e6);
    @memset(band, 0);
    var r0: usize = std.math.maxInt(usize);
    var r1: usize = 0;
    var c0: usize = std.math.maxInt(usize);
    var c1: usize = 0;
    var any = false;
    for (0..h) |y| {
        const mr = mask.row(y);
        var x: usize = 0;
        while (x < w) {
            if (x + 64 <= w) {
                const v: @Vector(64, u8) = mr[x..][0..64].*;
                if (@reduce(.Max, v) == 0) {
                    x += 64;
                    continue;
                }
            }
            const e = @min(x + 64, w);
            for (x..e) |xx| if (mr[xx] != 0) {
                f[(y + 1) * cols + xx + 1] = INSIDE;
                any = true;
                r0 = @min(r0, y + 1);
                r1 = y + 1;
                c0 = @min(c0, xx + 1);
                c1 = @max(c1, xx + 1);
            };
            x = e;
        }
    }
    // The band: 4-neighbours of the mask, outside it, not on the frame,
    // queued in raster order with T = 0.
    var heap: Queue = .{};
    defer heap.deinit(gpa);
    var out_q: Queue = .{};
    defer out_q.deinit(gpa);
    if (any) {
        for (@max(r0 - 1, 1)..@min(r1 + 2, rows - 1)) |i| for (@max(c0 - 1, 1)..@min(c1 + 2, cols - 1)) |j| {
            const p = i * cols + j;
            if (f[p] == INSIDE) continue;
            if (f[p - cols] == INSIDE or f[p + cols] == INSIDE or f[p - 1] == INSIDE or f[p + 1] == INSIDE) {
                band[p] = INSIDE;
                t[p] = 0;
                try heap.push(gpa, i, j, 0);
                if (method == .telea) try out_q.push(gpa, i, j, 0);
            }
        };
    }

    if (method == .telea and any) {
        // Ring of pixels within `range` (Chebyshev) of the mask, outside the
        // mask and the band: the outward march gives them negative times.
        @memset(ring, 0);
        const ia0 = @max((r0 + 1) -| (range + 1), 1);
        const ja0 = @max((c0 + 1) -| (range + 1), 1);
        const ia1 = @min(r1 + range + 1, rows - 1);
        const ja1 = @min(c1 + range + 1, cols - 1);
        try dilateRing(gpa, f, band, ring, cols, ia0, ia1, ja0, ja1, range);
        try calcFmmOutside(gpa, ring, t, rows, cols, &out_q, ia0, ia1, ja0, ja1);
    }

    var disk = try Disk.init(gpa, range);
    defer disk.deinit(gpa);
    const ri: isize = @intCast(range);
    var dk: isize = -ri;
    while (dk <= ri) : (dk += 1) {
        var dl: isize = -ri;
        while (dl <= ri) : (dl += 1) {
            const idx = @as(usize, @intCast(dk + ri)) * disk.side + @as(usize, @intCast(dl + ri));
            const ry: f32 = @floatFromInt(-dk);
            const rx: f32 = @floatFromInt(-dl);
            const L = vlen(rx, ry);
            disk.dst[idx] = switch (method) {
                .telea => if (dk == 0 and dl == 0) 0 else if (cn == 3)
                    @floatCast(1.0 / (@as(f64, L) * @sqrt(@as(f64, L))))
                else
                    @floatCast(1.0 / @as(f64, L * @sqrt(L))),
                .ns => 1 / (L * L + 1),
            };
        }
    }

    var ip: Inpainter(T, cn) = .{ .g = .{ .rows = rows, .cols = cols, .f = f, .t = t }, .out = dst, .disk = disk };
    try ip.march(gpa, &heap, method);
}

test "inpaint fills a hole in a flat image" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 20, 15, 1, 100);
    defer img.deinit(gpa);
    var m = try Image(u8).initFill(gpa, 20, 15, 1, 0);
    defer m.deinit(gpa);
    for (5..9) |y| for (6..11) |x| {
        m.set(x, y, 0, 255);
        img.set(x, y, 0, 0);
    };
    for ([_]Method{ .telea, .ns }) |meth| {
        var o = try Image(u8).init(gpa, 20, 15, 1);
        defer o.deinit(gpa);
        try inpaint(u8, gpa, img, m, o, 3, meth);
        for (o.slice()) |v| try std.testing.expectEqual(@as(u8, 100), v);
    }
}
