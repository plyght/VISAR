//! Contour extraction (findContours) with Suzuki-Abe border following, a
//! port of OpenCV 5's contour scanner: the same contours, starting points,
//! point order, contour order and hierarchy.
//!
//! Like OpenCV, the input is treated as binary (nonzero = 1) and padded with
//! one pixel of zeros, so objects touching the image border are traced along
//! the border. Contours come out in the order of a depth-first walk of the
//! contour tree where, among siblings, the contour found last in the raster
//! scan comes first (OpenCV's order since 4.9).

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error;

pub const Point = extern struct {
    x: i32,
    y: i32,

    pub inline fn add(a: Point, b: Point) Point {
        return .{ .x = a.x + b.x, .y = a.y + b.y };
    }
};

/// Contour retrieval mode (OpenCV RETR_*).
pub const Mode = enum(u8) { external = 0, list = 1, ccomp = 2, tree = 3 };

/// Contour approximation method (OpenCV CHAIN_APPROX_*).
pub const Method = enum(u8) { none = 1, simple = 2, tc89_l1 = 3, tc89_kcos = 4 };

/// The result of findContours: every contour's points stored back to back.
/// Contour i is `points[starts[i]..starts[i + 1]]`; `hierarchy[i]` is
/// OpenCV's [next, previous, first child, parent] (-1 = none).
pub const Contours = struct {
    points: []Point,
    starts: []usize,
    hierarchy: [][4]i32,

    pub const empty: Contours = .{ .points = &.{}, .starts = &.{}, .hierarchy = &.{} };

    pub fn len(self: Contours) usize {
        return if (self.starts.len == 0) 0 else self.starts.len - 1;
    }

    pub fn get(self: Contours, i: usize) []Point {
        return self.points[self.starts[i]..self.starts[i + 1]];
    }

    pub fn deinit(self: *Contours, gpa: Allocator) void {
        gpa.free(self.points);
        gpa.free(self.starts);
        gpa.free(self.hierarchy);
        self.* = undefined;
    }
};

const chain_deltas = [8]Point{
    .{ .x = 1, .y = 0 },  .{ .x = 1, .y = -1 }, .{ .x = 0, .y = -1 }, .{ .x = -1, .y = -1 },
    .{ .x = -1, .y = 0 }, .{ .x = -1, .y = 1 }, .{ .x = 0, .y = 1 },  .{ .x = 1, .y = 1 },
};

const MASK8_RIGHT: i8 = @bitCast(@as(u8, 0x80));
const MASK8_NEW: i8 = 2;
const MASK8_FLAGS: i32 = -2;
const MASK8_BLACK: i8 = 1;
const MASK8_LVAL: i8 = 0x7F;

const Rect = struct { x: i32, y: i32, w: i32, h: i32 };

const Node = struct {
    parent: i32 = -1,
    first_child: i32 = -1,
    prev: i32 = -1,
    next: i32 = -1,
    ctable_next: i32 = -1,
    is_hole: bool,
    rect: Rect = .{ .x = 0, .y = 0, .w = 0, .h = 0 },
    /// Start pixel in padded-image coordinates.
    origin: Point = .{ .x = 0, .y = 0 },
    pts_start: usize = 0,
    pts_end: usize = 0,
};

const Scanner = struct {
    gpa: Allocator,
    img: []i8,
    step: usize,
    width: i32,
    height: i32,
    mode: Mode,
    method: Method,
    /// User offset minus one (the padding).
    offset: Point,
    pt: Point,
    lnbd: Point,
    nbd: i8,
    nodes: std.ArrayList(Node) = .empty,
    points: std.ArrayList(Point) = .empty,
    codes: std.ArrayList(u8) = .empty,
    approx: std.ArrayList(ApproxItem) = .empty,
    ctable: [128]i32 = @splat(-1),
    delta: [16]isize,

    fn isSimple(self: *const Scanner) bool {
        return self.mode == .external or self.mode == .list;
    }

    inline fn idx(self: *const Scanner, x: i32, y: i32) usize {
        return @as(usize, @intCast(y)) * self.step + @as(usize, @intCast(x));
    }

    inline fn at(self: *const Scanner, x: i32, y: i32) i8 {
        return self.img[self.idx(x, y)];
    }

    inline fn move(self: *const Scanner, i: usize, s: u8) usize {
        return @intCast(@as(isize, @intCast(i)) + self.delta[s]);
    }

    fn addChild(self: *Scanner, parent_idx: i32, child_idx: i32) void {
        const nodes = self.nodes.items;
        const parent = &nodes[@intCast(parent_idx)];
        const child = &nodes[@intCast(child_idx)];
        if (parent.first_child != -1) {
            nodes[@intCast(parent.first_child)].prev = child_idx;
            child.next = parent.first_child;
        }
        parent.first_child = child_idx;
        child.parent = parent_idx;
        child.prev = -1;
    }

    inline fn pushPoint(self: *Scanner, p: Point) Allocator.Error!void {
        const pts = &self.points;
        if (pts.items.len == pts.capacity) try pts.ensureUnusedCapacity(self.gpa, @max(4096, pts.capacity / 2));
        pts.appendAssumeCapacity(p);
    }

    /// Bit d is set when the neighbour of `j` in chain direction d is nonzero.
    inline fn neighbours(self: *const Scanner, j: usize) u8 {
        const img = self.img;
        const up = j - self.step;
        const dn = j + self.step;
        const v: @Vector(8, i8) = .{ img[j + 1], img[up + 1], img[up], img[up - 1], img[j - 1], img[dn - 1], img[dn], img[dn + 1] };
        return @bitCast(v != @as(@Vector(8, i8), @splat(0)));
    }

    /// icvFetchContourEx: follows the border starting at `start`, marks it
    /// with `nbd` and records its points (or chain codes).
    fn fetchContour(self: *Scanner, start: Point, nbd: i8, node: *Node, comptime is_direct: bool, comptime is_chain: bool, comptime need_rect: bool) Allocator.Error!void {
        const img = self.img;
        const j0 = self.idx(start.x, start.y);
        var pt = start;
        var rx = pt.x;
        var ry = pt.y;
        var rmx = pt.x;
        var rmy = pt.y;
        var s_end: u8 = if (node.is_hole) 0 else 4;
        var s = s_end;
        var j1: usize = undefined;
        while (true) {
            s = (s -% 1) & 7;
            j1 = self.move(j0, s);
            if (img[j1] != 0 or s == s_end) break;
        }
        if (s == s_end) {
            img[j0] = nbd | MASK8_RIGHT;
            if (!is_chain) try self.pushPoint(pt.add(self.offset));
        } else {
            var j3 = j0;
            var j4: usize = j0;
            var prev_s: u8 = s ^ 4;
            while (true) {
                // The first nonzero neighbour counter-clockwise from s + 1
                // (OpenCV's `while (s < 15) ++s` search), found without
                // branches from an 8-bit neighbourhood mask.
                s_end = s;
                const nb = self.neighbours(j3);
                const from: u3 = @truncate(s + 1);
                s = (s + 1 + @ctz(std.math.rotr(u8, nb, from))) & 7;
                j4 = self.move(j3, s);
                if (s -% 1 < s_end) {
                    img[j3] = nbd | MASK8_RIGHT;
                } else if (img[j3] == MASK8_BLACK) {
                    img[j3] = nbd;
                }
                if (is_chain) {
                    try self.codes.append(self.gpa, s);
                } else if (s != prev_s or is_direct) {
                    try self.pushPoint(pt.add(self.offset));
                }
                if (need_rect and s != prev_s) {
                    if (pt.x < rx) rx = pt.x else if (pt.x > rmx) rmx = pt.x;
                    if (pt.y < ry) ry = pt.y else if (pt.y > rmy) rmy = pt.y;
                }
                prev_s = s;
                pt = pt.add(chain_deltas[s]);
                if (j4 == j0 and j3 == j1) break;
                j3 = j4;
                s = (s + 4) & 7;
            }
        }
        node.rect = .{ .x = rx, .y = ry, .w = rmx - rx + 1, .h = rmy - ry + 1 };
    }

    /// icvTraceContour: walks an already-marked border from `start` and
    /// reports whether it is the contour that passes through `end` last.
    fn traceContour(self: *const Scanner, start: Point, end: Point, is_hole: bool) bool {
        const img = self.img;
        const stop = self.idx(end.x, end.y);
        const j0 = self.idx(start.x, start.y);
        const s_end: u8 = if (is_hole) 0 else 4;
        var s = s_end;
        var j1: usize = undefined;
        while (true) {
            s = (s -% 1) & 7;
            j1 = self.move(j0, s);
            if (img[j1] != 0 or s == s_end) break;
        }
        var j3 = j0;
        if (s == s_end) return j3 == stop;
        var j4: usize = j0;
        while (true) {
            s = @min(s, 15);
            while (s < 15) {
                s += 1;
                j4 = self.move(j3, s);
                if (img[j4] != 0) break;
            }
            if (j3 == stop) {
                if ((img[j3] & MASK8_RIGHT) == 0) return true;
                var t = s;
                while (true) {
                    t = (t -% 1) & 7;
                    if (img[self.move(j3, t)] != 0) break;
                    if (t == 0) return true;
                }
            }
            if (j4 == j0 and j3 == j1) break;
            j3 = j4;
            s = (s + 4) & 7;
        }
        return false;
    }

    fn makeContour(self: *Scanner, nbd: *i8, is_hole: bool, x: i32, y: i32) Allocator.Error!i32 {
        const is_chain = self.method == .tc89_l1 or self.method == .tc89_kcos;
        const start: Point = .{ .x = x - @intFromBool(is_hole), .y = y };
        const self_idx: i32 = @intCast(self.nodes.items.len);
        var node: Node = .{ .is_hole = is_hole };
        node.pts_start = self.points.items.len;
        self.codes.clearRetainingCapacity();
        if (self.isSimple()) {
            switch (self.method) {
                inline else => |m| try self.fetchContour(start, MASK8_NEW, &node, m == .none, m == .tc89_l1 or m == .tc89_kcos, false),
            }
        } else {
            const lval = nbd.*;
            nbd.* = (nbd.* +% 1) & MASK8_LVAL;
            if (nbd.* == 0) nbd.* = MASK8_BLACK | MASK8_NEW;
            switch (self.method) {
                inline else => |m| try self.fetchContour(start, lval, &node, m == .none, m == .tc89_l1 or m == .tc89_kcos, true),
            }
            const l: usize = @intCast(lval);
            node.ctable_next = self.ctable[l];
            self.ctable[l] = self_idx;
        }
        node.origin = start;
        if (is_chain) try self.approximateTC89(start.add(self.offset));
        node.pts_end = self.points.items.len;
        try self.nodes.append(self.gpa, node);
        return self_idx;
    }

    fn findFirstBoundingContour(self: *const Scanner, last_pos: Point, y: i32, lval: usize) i32 {
        const end: Point = .{ .x = last_pos.x, .y = y };
        var res: i32 = -1;
        var cur = self.ctable[lval];
        const nodes = self.nodes.items;
        while (cur != -1) {
            const ce = &nodes[@intCast(cur)];
            if (last_pos.x - ce.rect.x < ce.rect.w and last_pos.y - ce.rect.y < ce.rect.h) {
                if (res != -1) {
                    const re = &nodes[@intCast(res)];
                    if (self.traceContour(re.origin, end, re.is_hole)) break;
                }
                res = cur;
            }
            cur = ce.ctable_next;
        }
        return res;
    }

    fn contourScan(self: *Scanner, prev: i32, p: i32, last_pos: *Point, x: i32, y: i32) Allocator.Error!bool {
        var is_hole = false;
        if (!(prev == 0 and p == 1)) {
            if (p != 0 or prev < 1) return false;
            if (prev & MASK8_FLAGS != 0) last_pos.x = x - 1;
            is_hole = true;
        }
        if (self.mode == .external and (is_hole or self.at(last_pos.x, last_pos.y) > 0)) return false;

        var main_parent: i32 = -1;
        if (self.isSimple() or (!is_hole and self.mode == .ccomp) or last_pos.x <= 0) {
            main_parent = 0;
        } else {
            const lval: usize = @intCast(self.at(last_pos.x, last_pos.y) & MASK8_LVAL);
            main_parent = self.findFirstBoundingContour(last_pos.*, y, lval);
            if (main_parent < 0) main_parent = 0;
            const mp = &self.nodes.items[@intCast(main_parent)];
            if (mp.is_hole == is_hole) main_parent = if (mp.parent != -1) mp.parent else 0;
        }
        last_pos.x = x - @intFromBool(is_hole);

        var nbd = self.nbd;
        const new_idx = try self.makeContour(&nbd, is_hole, x, y);
        self.addChild(main_parent, new_idx);
        self.pt = .{ .x = x + 1, .y = y };
        self.nbd = nbd;
        return true;
    }

    /// First x >= `x0` (and < width) on row y whose value differs from
    /// `prev`; skips equal runs 32 bytes at a time.
    inline fn findNextX(self: *const Scanner, x0: i32, y: i32, prev: i32, p: *i32) i32 {
        const width = self.width - 1;
        var x = x0;
        if (x >= width) return x;
        const row = self.img[self.idx(0, y)..][0..self.step];
        const pv: i8 = @truncate(prev);
        if (row[@intCast(x)] != pv) {
            p.* = row[@intCast(x)];
            return x;
        }
        const N = 32;
        const V = @Vector(N, i8);
        const vp: V = @splat(pv);
        while (x + N <= width) {
            const v: V = row[@intCast(x)..][0..N].*;
            const ne = v != vp;
            if (@reduce(.Or, ne)) {
                const bits: u32 = @bitCast(ne);
                x += @intCast(@ctz(bits));
                p.* = row[@intCast(x)];
                return x;
            }
            x += N;
        }
        while (x < width) : (x += 1) {
            const v = row[@intCast(x)];
            if (v != pv) {
                p.* = v;
                return x;
            }
        }
        return x;
    }

    fn findNext(self: *Scanner) Allocator.Error!bool {
        var x = self.pt.x;
        var y = self.pt.y;
        const width = self.width - 1;
        const height = self.height - 1;
        var last_pos = self.lnbd;
        var prev: i32 = self.at(x - 1, y);
        while (y < height) : (y += 1) {
            var p: i32 = 0;
            while (x < width) : (x += 1) {
                x = self.findNextX(x, y, prev, &p);
                if (x >= width) break;
                if (try self.contourScan(prev, p, &last_pos, x, y)) {
                    self.lnbd = last_pos;
                    return true;
                }
                prev = p;
                if (prev & MASK8_FLAGS != 0) last_pos.x = x;
            }
            last_pos = .{ .x = 0, .y = y + 1 };
            x = 1;
            prev = 0;
        }
        return false;
    }

    // ---------------------------------------------------- Teh-Chin (TC89)

    const ApproxItem = struct {
        pt: Point,
        k: usize = 0,
        s: i32,
        removed: bool = false,
    };

    const abs_diff = [16]i32{ 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0 };

    /// approximateChainTC89 on the chain codes in `self.codes`.
    fn approximateTC89(self: *Scanner, origin: Point) Allocator.Error!void {
        const chain = self.codes.items;
        if (chain.len == 0) {
            try self.points.append(self.gpa, origin);
            return;
        }
        const ares_list = &self.approx;
        ares_list.clearRetainingCapacity();
        var pt = origin;
        for (chain, 0..) |cur, i| {
            const prev = if (i == 0) chain[chain.len - 1] else chain[i - 1];
            const s = abs_diff[@intCast(@as(i32, cur) - @as(i32, prev) + 7)];
            try ares_list.append(self.gpa, .{ .pt = pt, .s = s, .removed = s == 0 });
            pt = pt.add(chain_deltas[cur]);
        }
        var ares = ares_list.items;
        const kcos = self.method == .tc89_kcos;
        // Pass 1: support regions (and k-cosine curvature).
        for (0..ares.len) |i| {
            if (ares[i].removed) continue;
            ares[i].k = calcSupport(ares, i);
            if (kcos) ares[i].s = calcCosine(ares, i);
        }
        // Pass 2: non-maxima suppression.
        for (0..ares.len) |i| {
            if (calcNmsCleanup(ares, i)) {
                ares[i].s = 0;
                ares[i].removed = true;
            }
        }
        // Pass 3: drop non-dominant points with a 1-length support region.
        for (0..ares.len) |i| {
            if (ares[i].removed) continue;
            if (ares[i].k == 1 and calcDominance(ares, i)) {
                ares[i].s = 0;
                ares[i].removed = true;
            }
        }
        if (!kcos) {
            // Pass 4: clean remaining couples of points.
            var skip = false;
            var new_start: usize = 0;
            const n = ares.len;
            if (ares[0].s != 0 and ares[n - 1].s != 0) {
                // calc_new_start
                var c1: usize = 1;
                while (c1 < n and ares[c1].s != 0) : (c1 += 1) ares[c1 - 1].s = 0;
                if (c1 == n) {
                    skip = true;
                } else {
                    c1 -= 1;
                    var c2: usize = n - 2;
                    while (c2 > 0 and ares[c2].s != 0) : (c2 -= 1) {
                        clearUntil(ares, c2, n);
                        ares[c2 + 1].s = 0;
                    }
                    c2 += 1;
                    if (c1 == 0 and c2 == n - 1) {
                        c1 = nextIdx(ares, 0);
                        var first = ares[0];
                        first.removed = false;
                        try ares_list.append(self.gpa, first);
                        ares = ares_list.items;
                    }
                    new_start = c1;
                }
            }
            if (!skip) passCleanup(ares, new_start);
        }
        for (ares) |it| {
            if (!it.removed) try self.points.append(self.gpa, it.pt);
        }
    }

    fn wrapBack(i: usize, j: usize, n: usize) usize {
        return if (i >= j) i - j else n - j + i;
    }

    fn wrapFwd(i: usize, j: usize, n: usize) usize {
        return if (i + j < n) i + j else i + j - n;
    }

    fn calcSupport(ares: []const ApproxItem, i: usize) usize {
        const n = ares.len;
        var d_num: i32 = 0;
        var l: i32 = 0;
        var k: usize = 1;
        while (true) : (k += 1) {
            std.debug.assert(k <= n);
            const a = ares[wrapBack(i, k, n)].pt;
            const b = ares[wrapFwd(i, k, n)].pt;
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const lk = dx * dx + dy * dy;
            const dk_num = (ares[i].pt.x - a.x) * dy - (ares[i].pt.y - a.y) * dx;
            const df: f32 = @floatCast(@as(f64, @floatFromInt(d_num)) * @as(f64, @floatFromInt(lk)) - @as(f64, @floatFromInt(dk_num)) * @as(f64, @floatFromInt(l)));
            const di: i32 = @bitCast(df);
            if (k > 1 and (l >= lk or ((d_num > 0 and di <= 0) or (d_num < 0 and di >= 0)))) break;
            d_num = dk_num;
            l = lk;
        }
        return k - 1;
    }

    fn calcCosine(ares: []const ApproxItem, i: usize) i32 {
        const k = ares[i].k;
        const n = ares.len;
        var s: i32 = 0;
        var j = k;
        while (j > 0) : (j -= 1) {
            const a = ares[wrapBack(i, j, n)].pt;
            const b = ares[wrapFwd(i, j, n)].pt;
            const dx1 = a.x - ares[i].pt.x;
            const dy1 = a.y - ares[i].pt.y;
            const dx2 = b.x - ares[i].pt.x;
            const dy2 = b.y - ares[i].pt.y;
            if ((dx1 | dy1) == 0 or (dx2 | dy2) == 0) break;
            var t: f64 = @floatFromInt(dx1 * dx2 + dy1 * dy2);
            const fx1: f64 = @floatFromInt(dx1);
            const fy1: f64 = @floatFromInt(dy1);
            const fx2: f64 = @floatFromInt(dx2);
            const fy2: f64 = @floatFromInt(dy2);
            t = @as(f32, @floatCast(t / @sqrt((fx1 * fx1 + fy1 * fy1) * (fx2 * fx2 + fy2 * fy2))));
            const sk: f32 = @floatCast(t + 1.1);
            const si: i32 = @bitCast(sk);
            if (j < k and si <= s) break;
            s = si;
        }
        return s;
    }

    fn calcNmsCleanup(ares: []const ApproxItem, i: usize) bool {
        const k2 = ares[i].k >> 1;
        const s = ares[i].s;
        const n = ares.len;
        var j: usize = 1;
        while (j <= k2) : (j += 1) {
            if (ares[wrapBack(i, j, n)].s > s or ares[wrapFwd(i, j, n)].s > s) break;
        }
        return j <= k2;
    }

    fn calcDominance(ares: []const ApproxItem, i: usize) bool {
        const n = ares.len;
        return ares[i].s <= ares[wrapBack(i, 1, n)].s or ares[i].s <= ares[wrapFwd(i, 1, n)].s;
    }

    fn nextIdx(ares: []const ApproxItem, start: usize) usize {
        var r = start + 1;
        while (r < ares.len and ares[r].removed) r += 1;
        return r;
    }

    fn clearUntil(ares: []ApproxItem, start: usize, finish: usize) void {
        var i = start + 1;
        while (i < finish and i < ares.len) : (i += 1) ares[i].removed = true;
    }

    fn passCleanup(ares: []ApproxItem, start_idx: usize) void {
        var count: i32 = 1;
        const n = ares.len;
        var first = start_idx;
        var prev = start_idx;
        var i = start_idx;
        while (i < n) : (i += 1) {
            if (ares[i].removed) continue;
            const nx = nextIdx(ares, i);
            if (nx == n or nx - i != 1) {
                if (count >= 2) {
                    if (count == 2) {
                        const s1 = ares[prev].s;
                        const s2 = ares[i].s;
                        if (s1 > s2 or (s1 == s2 and ares[prev].k <= ares[i].k)) {
                            ares[i].removed = true;
                        } else {
                            ares[prev].removed = true;
                        }
                    } else {
                        first = nextIdx(ares, first);
                        clearUntil(ares, first, i);
                    }
                }
                first = i;
                count = 1;
            } else {
                count += 1;
            }
            prev = i;
        }
    }
};

/// Pads `src` by one pixel of zeros and binarises it (nonzero -> 1).
fn padBinary(gpa: Allocator, src: Image(u8)) Allocator.Error![]i8 {
    const pw = @as(usize, src.width) + 2;
    const ph = @as(usize, src.height) + 2;
    const buf = try gpa.alloc(i8, pw * ph);
    @memset(buf[0..pw], 0);
    @memset(buf[(ph - 1) * pw ..], 0);
    const N = 32;
    const V = @Vector(N, u8);
    const zero: V = @splat(0);
    const one: V = @splat(1);
    for (0..src.height) |y| {
        const s = src.row(y);
        const d = buf[(y + 1) * pw ..][0..pw];
        d[0] = 0;
        d[pw - 1] = 0;
        const dd: []u8 = @ptrCast(d[1 .. pw - 1]);
        var x: usize = 0;
        while (x + N <= s.len) : (x += N) {
            const v: V = s[x..][0..N].*;
            dd[x..][0..N].* = @select(u8, v != zero, one, zero);
        }
        while (x < s.len) : (x += 1) dd[x] = @intFromBool(s[x] != 0);
    }
    return buf;
}

/// findContours on a single-channel 8-bit image (nonzero pixels are
/// foreground). Points are shifted by `offset`. Free the result with
/// `Contours.deinit`.
pub fn findContours(gpa: Allocator, src: Image(u8), mode: Mode, method: Method, offset: Point) Error!Contours {
    if (src.channels != 1) return error.UnsupportedChannels;
    const img = try padBinary(gpa, src);
    defer gpa.free(img);
    const pw: i32 = @intCast(@as(usize, src.width) + 2);
    const ph: i32 = @intCast(@as(usize, src.height) + 2);
    var sc: Scanner = .{
        .gpa = gpa,
        .img = img,
        .step = @intCast(pw),
        .width = pw,
        .height = ph,
        .mode = mode,
        .method = method,
        .offset = .{ .x = offset.x - 1, .y = offset.y - 1 },
        .pt = .{ .x = 1, .y = 1 },
        .lnbd = .{ .x = 0, .y = 1 },
        .nbd = 2,
        .delta = undefined,
    };
    for (0..16) |s| {
        const d = chain_deltas[s % 8];
        sc.delta[s] = @as(isize, d.x) + @as(isize, d.y) * @as(isize, pw);
    }
    defer {
        sc.nodes.deinit(gpa);
        sc.points.deinit(gpa);
        sc.codes.deinit(gpa);
        sc.approx.deinit(gpa);
    }
    try sc.nodes.append(gpa, .{ .is_hole = true, .rect = .{ .x = 0, .y = 0, .w = pw, .h = ph } });
    while (try sc.findNext()) {}

    const nodes = sc.nodes.items;
    if (nodes.len <= 1) return .empty;
    const total = nodes.len - 1;

    // Depth-first order: a node, then its children starting at first_child.
    const order = try gpa.alloc(u32, total);
    defer gpa.free(order);
    const mapping = try gpa.alloc(i32, nodes.len);
    defer gpa.free(mapping);
    {
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(gpa);
        try stack.append(gpa, 0);
        var n: usize = 0;
        var tmp: std.ArrayList(u32) = .empty;
        defer tmp.deinit(gpa);
        while (stack.pop()) |i| {
            if (i != 0) {
                order[n] = i;
                mapping[i] = @intCast(n);
                n += 1;
            }
            // Push children so that first_child is popped first.
            tmp.clearRetainingCapacity();
            var c = nodes[i].first_child;
            while (c != -1) : (c = nodes[@intCast(c)].next) try tmp.append(gpa, @intCast(c));
            var k = tmp.items.len;
            while (k > 0) {
                k -= 1;
                try stack.append(gpa, tmp.items[k]);
            }
        }
        std.debug.assert(n == total);
    }
    mapping[0] = -1;

    const points = try gpa.alloc(Point, sc.points.items.len);
    errdefer gpa.free(points);
    const starts = try gpa.alloc(usize, total + 1);
    errdefer gpa.free(starts);
    const hierarchy = try gpa.alloc([4]i32, total);
    errdefer gpa.free(hierarchy);
    var pos: usize = 0;
    const map = struct {
        fn f(m: []const i32, i: i32) i32 {
            return if (i < 0) -1 else m[@intCast(i)];
        }
    }.f;
    for (order, 0..) |ni, k| {
        const nd = nodes[ni];
        starts[k] = pos;
        const src_pts = sc.points.items[nd.pts_start..nd.pts_end];
        @memcpy(points[pos..][0..src_pts.len], src_pts);
        pos += src_pts.len;
        hierarchy[k] = .{ map(mapping, nd.next), map(mapping, nd.prev), map(mapping, nd.first_child), map(mapping, nd.parent) };
    }
    starts[total] = pos;
    return .{ .points = points, .starts = starts, .hierarchy = hierarchy };
}

// ------------------------------------------------------------------ tests

fn testImage(comptime rows: []const []const u8) [rows.len * rows[0].len]u8 {
    var out: [rows.len * rows[0].len]u8 = undefined;
    for (rows, 0..) |r, y| for (r, 0..) |c, x| {
        out[y * rows[0].len + x] = if (c == '#') 255 else 0;
    };
    return out;
}

test "findContours: rectangle touching the border, all modes" {
    const gpa = std.testing.allocator;
    var data: [20]u8 = @splat(1);
    const img = try Image(u8).fromSlice(&data, 5, 4, 1);
    var c = try findContours(gpa, img, .tree, .none, .{ .x = 0, .y = 0 });
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), c.len());
    const exp = [_][2]i32{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 0, 3 }, .{ 1, 3 }, .{ 2, 3 }, .{ 3, 3 }, .{ 4, 3 }, .{ 4, 2 }, .{ 4, 1 }, .{ 4, 0 }, .{ 3, 0 }, .{ 2, 0 }, .{ 1, 0 } };
    const got = c.get(0);
    try std.testing.expectEqual(exp.len, got.len);
    for (exp, got) |e, g| try std.testing.expectEqual(e, [2]i32{ g.x, g.y });
    try std.testing.expectEqual([4]i32{ -1, -1, -1, -1 }, c.hierarchy[0]);
}

test "findContours: ring with a hole and an island, tree and simple" {
    const gpa = std.testing.allocator;
    var data = testImage(&.{
        "..........",
        ".#######..",
        ".#.....#..",
        ".#.##..#.#",
        ".#.....#..",
        ".#######..",
    });
    const img = try Image(u8).fromSlice(&data, 10, 6, 1);
    var c = try findContours(gpa, img, .tree, .simple, .{ .x = 0, .y = 0 });
    defer c.deinit(gpa);
    // Found in raster order: ring outer (A), its hole (B), single pixel (C),
    // the island (D). Siblings come out last-found first: C, A, B, D.
    try std.testing.expectEqual(@as(usize, 4), c.len());
    try std.testing.expectEqual(@as(usize, 1), c.get(0).len);
    try std.testing.expectEqual(Point{ .x = 9, .y = 3 }, c.get(0)[0]);
    try std.testing.expectEqual([4]i32{ 1, -1, -1, -1 }, c.hierarchy[0]);
    try std.testing.expectEqual([4]i32{ -1, 0, 2, -1 }, c.hierarchy[1]);
    try std.testing.expectEqual([4]i32{ -1, -1, 3, 1 }, c.hierarchy[2]);
    try std.testing.expectEqual([4]i32{ -1, -1, -1, 2 }, c.hierarchy[3]);
    try std.testing.expectEqual(@as(usize, 4), c.get(1).len);

    var e = try findContours(gpa, img, .external, .simple, .{ .x = 10, .y = 20 });
    defer e.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), e.len());
    try std.testing.expectEqual(Point{ .x = 19, .y = 23 }, e.get(0)[0]);
    try std.testing.expectEqual(Point{ .x = 11, .y = 21 }, e.get(1)[0]);
}

test "findContours: empty image" {
    const gpa = std.testing.allocator;
    var data: [12]u8 = @splat(0);
    const img = try Image(u8).fromSlice(&data, 4, 3, 1);
    var c = try findContours(gpa, img, .list, .simple, .{ .x = 0, .y = 0 });
    defer c.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), c.len());
}
