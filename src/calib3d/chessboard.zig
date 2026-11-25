//! findChessboardCorners: the inner corners of a chessboard calibration
//! pattern, in OpenCV's output order.
//!
//! Like OpenCV, the image is binarised (Otsu, then adaptive mean
//! thresholds), dilated so that the black squares come apart at the
//! corners, and the black squares are found as quadrilateral holes of the
//! white region (findContours + approxPolyDP). Quads whose corners touch
//! are linked; walking the links gives every square a lattice position, and
//! a complete W x H block of linked corners is the board. Each corner
//! starts at the mean of its two quads' vertices and is refined with
//! cornerSubPix (5x5 window), as OpenCV does.
//!
//! The corner order follows OpenCV's conventions: W corners per row, rows
//! oriented so the board is traversed clockwise in the image; when W + H
//! is odd the board's colours fix the first corner (next to a black outer
//! square); otherwise the first row is the one whose last row lies below
//! it, starting at the top-most candidate (OpenCV's choice here comes from
//! its quad traversal; this rule reproduces it on every synthetic view we
//! tested). The detection itself is VISAR's own: it finds the same boards
//! as OpenCV on clean views but may differ on hard ones.

const std = @import("std");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
const Point2f = imgproc.Point2f;
const Point = imgproc.Point;

pub const Error = core.Error || Allocator.Error || error{ InvalidArgument, Unsupported, InvalidKernel };

pub const Flags = struct {
    adaptive_thresh: bool = true,
    normalize_image: bool = true,
    filter_quads: bool = false,
    fast_check: bool = false,
};

const Quad = struct {
    /// Vertices in a consistent rotational order.
    p: [4][2]f32,
    edge2: f32,
    /// Lattice square position once assigned.
    r: i32 = 0,
    c: i32 = 0,
    rot: u2 = 0,
    placed: bool = false,
    /// Linked neighbour per vertex: (quad index, its vertex) or none.
    link: [4]?[2]u32 = .{ null, null, null, null },
};

/// Finds the board's W x H inner corners. Returns true and fills `out`
/// (W*H points) when found.
pub fn findChessboardCorners(exec: Exec, gpa: Allocator, img: Image(u8), pattern_w: u32, pattern_h: u32, out: []Point2f, flags: Flags) Error!bool {
    if (pattern_w < 2 or pattern_h < 2) return error.InvalidArgument;
    if (out.len < pattern_w * pattern_h) return error.InvalidArgument;
    if (img.channels != 1) return error.Unsupported;
    const w = img.width;
    const h = img.height;
    if (w < 8 or h < 8) return false;
    var bin = try Image(u8).init(gpa, w, h, 1);
    defer bin.deinit(gpa);
    var dil = try Image(u8).init(gpa, w, h, 1);
    defer dil.deinit(gpa);
    var integral: []u32 = &.{};
    defer gpa.free(integral);
    var el_buf: [9]u8 = undefined;
    const el = imgproc.morph.structuringElement(.rect, 3, 3, &el_buf);

    // Binarisations: Otsu, then (with adaptive_thresh) adaptive means.
    const n_bin: usize = if (flags.adaptive_thresh) 5 else 1;
    for (0..n_bin) |k| {
        if (k == 0) {
            _ = try imgproc.threshold(exec, img, bin, 0, 255, .binary, .otsu);
        } else {
            if (integral.len == 0) integral = try integralImage(gpa, img);
            const frac: f64 = if (k % 2 == 1) 0.2 else 0.1;
            const block: u32 = @as(u32, @intFromFloat(@round(@as(f64, @floatFromInt(@min(w, h))) * frac))) | 1;
            const c: i32 = @intCast(((k - 1) / 2) * 5);
            adaptiveMean(img, bin, integral, block, c);
        }
        for (0..4) |dilations| {
            if (dilations == 0) {
                dil.copyFrom(bin);
            } else {
                try imgproc.dilate(exec, gpa, if (dilations == 1) bin else dil, dil, el, 1);
            }
            if (try detect(gpa, img, dil, pattern_w, pattern_h, out)) {
                try imgproc.cornerSubPix(gpa, img, out[0 .. pattern_w * pattern_h], .{ .win_w = 2, .win_h = 2, .max_iters = 15, .epsilon = 0.1 });
                return true;
            }
        }
    }
    return false;
}

fn integralImage(gpa: Allocator, img: Image(u8)) Allocator.Error![]u32 {
    const w = img.width + 1;
    const ii = try gpa.alloc(u32, w * (img.height + 1));
    @memset(ii[0..w], 0);
    for (0..img.height) |y| {
        const r = img.row(y);
        var s: u32 = 0;
        ii[(y + 1) * w] = 0;
        for (0..img.width) |x| {
            s += r[x];
            ii[(y + 1) * w + x + 1] = ii[y * w + x + 1] + s;
        }
    }
    return ii;
}

/// dst = 255 where src > mean(block) - c, else 0 (replicated-edge window
/// clipped to the image).
fn adaptiveMean(src: Image(u8), dst: Image(u8), ii: []const u32, block: u32, c: i32) void {
    const w = src.width;
    const h = src.height;
    const r = block / 2;
    const iw = w + 1;
    for (0..h) |y| {
        const y0 = y -| r;
        const y1 = @min(y + r + 1, h);
        const s = src.row(y);
        const d = dst.row(y);
        for (0..w) |x| {
            const x0 = x -| r;
            const x1 = @min(x + r + 1, w);
            const sum = ii[y1 * iw + x1] + ii[y0 * iw + x0] - ii[y0 * iw + x1] - ii[y1 * iw + x0];
            const area: u32 = @intCast((y1 - y0) * (x1 - x0));
            const v: i64 = @as(i64, s[x]) * area;
            d[x] = if (v > @as(i64, sum) - @as(i64, c) * area) 255 else 0;
        }
    }
}

fn cross(a: [2]f32, b: [2]f32, o: [2]f32) f32 {
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

fn dist2(a: [2]f32, b: [2]f32) f32 {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]);
}

/// Black squares of a binarised (and dilated) image as quads.
fn findQuads(gpa: Allocator, bin: Image(u8), quads: *std.ArrayList(Quad)) Error!void {
    // White frame, so squares cut by the image edge still close.
    @memset(bin.row(0), 255);
    @memset(bin.row(bin.height - 1), 255);
    for (0..bin.height) |y| {
        bin.row(y)[0] = 255;
        bin.row(y)[bin.width - 1] = 255;
    }
    var cs = try imgproc.findContours(gpa, bin, .ccomp, .simple, .{ .x = 0, .y = 0 });
    defer cs.deinit(gpa);
    var approx: std.ArrayList(Point) = .empty;
    defer approx.deinit(gpa);
    const min_area = 25.0;
    for (0..cs.len()) |i| {
        // Holes of the white region are the black squares.
        if (cs.hierarchy[i][3] < 0) continue;
        const pts = cs.get(i);
        if (pts.len < 4) continue;
        const area = imgproc.shape.contourArea(pts, false);
        if (area < min_area) continue;
        try approx.resize(gpa, pts.len);
        var n: usize = 0;
        var level: f64 = 1;
        while (level <= 7) : (level += 1) {
            n = try imgproc.shape.approxPolyDP(gpa, pts, approx.items, level, true);
            if (n == 4) break;
        }
        if (n != 4) continue;
        const q = approx.items[0..4];
        if (!imgproc.shape.isContourConvex(q)) continue;
        var p: [4][2]f32 = undefined;
        for (0..4) |k| p[k] = .{ @floatFromInt(q[k].x), @floatFromInt(q[k].y) };
        // Consistent orientation.
        if (cross(p[1], p[2], p[0]) < 0) std.mem.swap([2]f32, &p[1], &p[3]);
        const d1 = dist2(p[0], p[2]);
        const d2 = dist2(p[1], p[3]);
        const e = [4]f32{ dist2(p[0], p[1]), dist2(p[1], p[2]), dist2(p[2], p[3]), dist2(p[3], p[0]) };
        const per = @sqrt(e[0]) + @sqrt(e[1]) + @sqrt(e[2]) + @sqrt(e[3]);
        const fa: f32 = @floatCast(area);
        if (!(e[0] * 4 > e[1] and e[1] * 4 > e[0] and e[0] * e[1] < fa * fa * 1.5 * 1.5 and d1 >= 0.15 * per * per / 4 and d2 >= 0.15 * per * per / 4)) continue;
        try quads.append(gpa, .{ .p = p, .edge2 = @min(@min(e[0], e[1]), @min(e[2], e[3])) });
    }
}

/// Lattice offset of square neighbour through corner direction d (TL, TR,
/// BR, BL in rotational order) and the point offset of that corner.
const nb_off = [4][2]i32{ .{ -1, -1 }, .{ -1, 1 }, .{ 1, 1 }, .{ 1, -1 } };
const pt_off = [4][2]i32{ .{ 0, 0 }, .{ 0, 1 }, .{ 1, 1 }, .{ 1, 0 } };

fn detect(gpa: Allocator, img: Image(u8), bin: Image(u8), pw: u32, ph: u32, out: []Point2f) Error!bool {
    _ = img;
    var quads: std.ArrayList(Quad) = .empty;
    defer quads.deinit(gpa);
    try findQuads(gpa, bin, &quads);
    const need = ((pw + 1) * (ph + 1) + 1) / 2;
    if (quads.items.len < need -| 2) return false;
    const qs = quads.items;
    // Link vertices: mutual nearest within a fraction of the edges.
    for (qs, 0..) |*a, ia| {
        for (0..4) |va| {
            var best: f32 = std.math.inf(f32);
            var bq: u32 = 0;
            var bv: u32 = 0;
            for (qs, 0..) |*b, ib| {
                if (ib == ia) continue;
                for (0..4) |vb| {
                    const d = dist2(a.p[va], b.p[vb]);
                    if (d < best) {
                        best = d;
                        bq = @intCast(ib);
                        bv = @intCast(vb);
                    }
                }
            }
            const lim = @min(a.edge2, qs[bq].edge2) * 0.5;
            if (best <= lim) a.link[va] = .{ bq, bv };
        }
    }
    for (qs, 0..) |*a, ia| for (0..4) |va| {
        if (a.link[va]) |l| {
            const back = qs[l[0]].link[l[1]];
            if (back == null or back.?[0] != ia or back.?[1] != va) a.link[va] = null;
        }
    };
    // Components by BFS, placing squares on the lattice.
    var queue: std.ArrayList(u32) = .empty;
    defer queue.deinit(gpa);
    var points: std.AutoHashMapUnmanaged([2]i32, [2]f32) = .empty;
    defer points.deinit(gpa);
    var counts: std.AutoHashMapUnmanaged([2]i32, u8) = .empty;
    defer counts.deinit(gpa);
    for (qs, 0..) |*seed, is| {
        if (seed.placed) continue;
        seed.placed = true;
        seed.r = 0;
        seed.c = 0;
        seed.rot = 0;
        queue.clearRetainingCapacity();
        try queue.append(gpa, @intCast(is));
        var members: usize = 0;
        points.clearRetainingCapacity();
        counts.clearRetainingCapacity();
        var consistent = true;
        var qi: usize = 0;
        while (qi < queue.items.len) : (qi += 1) {
            const a = &qs[queue.items[qi]];
            members += 1;
            for (0..4) |va| {
                const l = a.link[va] orelse continue;
                const dir: u2 = @truncate(va +% @as(usize, a.rot));
                const b = &qs[l[0]];
                const br = a.r + nb_off[dir][0];
                const bc = a.c + nb_off[dir][1];
                const brot: u2 = @truncate(@as(usize, dir) + 2 + 4 - l[1]);
                if (!b.placed) {
                    b.placed = true;
                    b.r = br;
                    b.c = bc;
                    b.rot = brot;
                    try queue.append(gpa, l[0]);
                } else if (b.r != br or b.c != bc or b.rot != brot) consistent = false;
                // Each link visited from both sides; record once.
                if (queue.items[qi] < l[0]) {
                    const pr = a.r + pt_off[dir][0];
                    const pc = a.c + pt_off[dir][1];
                    const m = [2]f32{ (a.p[va][0] + b.p[l[1]][0]) * 0.5, (a.p[va][1] + b.p[l[1]][1]) * 0.5 };
                    try points.put(gpa, .{ pr, pc }, m);
                }
            }
        }
        if (!consistent or points.count() < pw * ph) continue;
        // Bounding box of the linked corners.
        var r0: i32 = std.math.maxInt(i32);
        var r1: i32 = std.math.minInt(i32);
        var c0: i32 = std.math.maxInt(i32);
        var c1: i32 = std.math.minInt(i32);
        var it = points.keyIterator();
        while (it.next()) |k| {
            r0 = @min(r0, k[0]);
            r1 = @max(r1, k[0]);
            c0 = @min(c0, k[1]);
            c1 = @max(c1, k[1]);
        }
        const R: u32 = @intCast(r1 - r0 + 1);
        const C: u32 = @intCast(c1 - c0 + 1);
        if (!((R == ph and C == pw) or (R == pw and C == ph))) continue;
        if (points.count() != R * C) continue;
        try order(gpa, &points, r0, c0, R, C, pw, ph, out);
        return true;
    }
    return false;
}

/// Writes the grid in OpenCV's order (see the file comment).
fn order(gpa: Allocator, points: *const std.AutoHashMapUnmanaged([2]i32, [2]f32), r0: i32, c0: i32, R: u32, C: u32, pw: u32, ph: u32, out: []Point2f) Allocator.Error!void {
    _ = gpa;
    const Cand = struct { k: u8, first: [2]f32, dy: f32, black: bool };
    var best: ?Cand = null;
    for (0..8) |k| {
        // Map output (row i of ph, col j of pw) to lattice (r, c).
        const tr = k & 4 != 0;
        const rows: u32 = if (tr) C else R;
        const cols: u32 = if (tr) R else C;
        // Rotation by (k & 3) quarter turns of the (rows x cols) grid.
        const rot = k & 3;
        const ow: u32 = if (rot % 2 == 0) cols else rows;
        const oh: u32 = if (rot % 2 == 0) rows else cols;
        if (ow != pw or oh != ph) continue;
        const map = struct {
            fn f(i: u32, j: u32, kk: usize, rr: u32, cc: u32, rr0: i32, cc0: i32, RR: u32, CC: u32) [2]i32 {
                _ = RR;
                _ = CC;
                // (i, j) in the rotated grid -> (a, b) in the (rr x cc) grid.
                var a: u32 = undefined;
                var b: u32 = undefined;
                switch (kk & 3) {
                    0 => {
                        a = i;
                        b = j;
                    },
                    1 => {
                        a = j;
                        b = cc - 1 - i;
                    },
                    2 => {
                        a = rr - 1 - i;
                        b = cc - 1 - j;
                    },
                    else => {
                        a = rr - 1 - j;
                        b = i;
                    },
                }
                // Undo the transpose.
                const lr = if (kk & 4 != 0) b else a;
                const lc = if (kk & 4 != 0) a else b;
                return .{ rr0 + @as(i32, @intCast(lr)), cc0 + @as(i32, @intCast(lc)) };
            }
        }.f;
        const at = struct {
            fn g(pts: *const std.AutoHashMapUnmanaged([2]i32, [2]f32), key: [2]i32) [2]f32 {
                return pts.get(key).?;
            }
        }.g;
        const p0 = at(points, map(0, 0, k, rows, cols, r0, c0, R, C));
        const p1 = at(points, map(0, pw - 1, k, rows, cols, r0, c0, R, C));
        const p2 = at(points, map(1, 0, k, rows, cols, r0, c0, R, C));
        const cr = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0]);
        if (cr <= 0) continue;
        const pl = at(points, map(ph - 1, 0, k, rows, cols, r0, c0, R, C));
        const l0 = map(0, 0, k, rows, cols, r0, c0, R, C);
        // The outer square diagonally beyond the first corner.
        const sq_r = if (l0[0] == r0) r0 - 1 else r0 + @as(i32, @intCast(R)) - 1;
        const sq_c = if (l0[1] == c0) c0 - 1 else c0 + @as(i32, @intCast(C)) - 1;
        const cand: Cand = .{ .k = @intCast(k), .first = p0, .dy = pl[1] - p0[1], .black = @mod(sq_r + sq_c, 2) == 0 };
        if (best == null) {
            best = cand;
            continue;
        }
        const b = best.?;
        const better = if ((pw + ph) % 2 == 1)
            cand.black and !b.black
        else if ((cand.dy >= 0) != (b.dy >= 0))
            cand.dy >= 0
        else
            cand.first[1] < b.first[1] or (cand.first[1] == b.first[1] and cand.first[0] < b.first[0]);
        if (better) best = cand;
    }
    const k = best.?.k;
    const tr = k & 4 != 0;
    const rows: u32 = if (tr) C else R;
    const cols: u32 = if (tr) R else C;
    for (0..ph) |i| for (0..pw) |j| {
        var a: u32 = undefined;
        var b: u32 = undefined;
        const ii: u32 = @intCast(i);
        const jj: u32 = @intCast(j);
        switch (k & 3) {
            0 => {
                a = ii;
                b = jj;
            },
            1 => {
                a = jj;
                b = cols - 1 - ii;
            },
            2 => {
                a = rows - 1 - ii;
                b = cols - 1 - jj;
            },
            else => {
                a = rows - 1 - jj;
                b = ii;
            },
        }
        const lr = if (tr) b else a;
        const lc = if (tr) a else b;
        const p = points.get(.{ r0 + @as(i32, @intCast(lr)), c0 + @as(i32, @intCast(lc)) }).?;
        out[i * pw + j] = .{ .x = p[0], .y = p[1] };
    };
}

test "findChessboardCorners on a synthetic board" {
    const gpa = std.testing.allocator;
    const sq = 20;
    const W = 7; // squares
    const H = 5;
    var img = try Image(u8).init(gpa, W * sq + 60, H * sq + 60, 1);
    defer img.deinit(gpa);
    img.fill(255);
    for (0..H * sq) |y| for (0..W * sq) |x| {
        if (((x / sq) + (y / sq)) % 2 == 0) img.row(y + 30)[x + 30] = 0;
    };
    var out: [24]Point2f = undefined;
    try std.testing.expect(try findChessboardCorners(.serial, gpa, img, 6, 4, &out, .{}));
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), out[0].x, 0.6);
    try std.testing.expectApproxEqAbs(@as(f32, 49.5), out[0].y, 0.6);
    try std.testing.expectApproxEqAbs(@as(f32, 69.5), out[1].x, 0.6);
    try std.testing.expectApproxEqAbs(@as(f32, 69.5), out[6].y, 0.6);
}
