//! Connected-component labeling with statistics (connectedComponentsWithStats).
//!
//! Works on runs, not pixels. The image is split into bands of rows (one
//! per thread chunk; a single band on `Exec.serial`). Each band finds its
//! nonzero runs with vector skips and joins them to overlapping runs of the
//! row above through a union-find whose ids are the runs' positions in
//! raster order; bands are then stitched along their boundary rows. Final
//! labels follow OpenCV: 4-connected components are numbered in raster
//! order of their first pixel, 8-connected ones (Spaghetti) by their first
//! 2x2 block in block-raster order.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error;

pub const Connectivity = enum(u8) { four = 4, eight = 8 };

/// Per-component statistics. Label 0 is the background.
pub const Stats = struct {
    left: u32,
    top: u32,
    width: u32,
    height: u32,
    area: u64,
    cx: f64,
    cy: f64,
};

pub const Result = struct {
    /// Number of labels including the background (label 0).
    count: u32,
    stats: []Stats,

    pub fn deinit(self: *Result, gpa: Allocator) void {
        gpa.free(self.stats);
        self.* = undefined;
    }
};

/// Union-find invariant: parent[x] <= x, so a root is the first run (in
/// raster order) of its set.
fn find(parent: [*]u32, x_in: u32) u32 {
    var x = x_in;
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/// Read-only find, safe while other threads read the same forest.
fn findConst(parent: [*]const u32, x_in: u32) u32 {
    var x = x_in;
    while (parent[x] != x) x = parent[x];
    return x;
}

fn unite(parent: [*]u32, a: u32, b: u32) u32 {
    const ra = find(parent, a);
    const rb = find(parent, b);
    if (ra == rb) return ra;
    // Keep the smaller (earlier) root so raster order is preserved.
    if (ra < rb) {
        parent[rb] = ra;
        return ra;
    }
    parent[ra] = rb;
    return rb;
}

/// A horizontal run of nonzero pixels [x0, x1] on row y. `label` is the
/// run's root (after resolving).
const Run = struct { y: u32, x0: u32, x1: u32, label: u32 };

/// Appends the nonzero runs of `row` to `out[n.*..]`. Works 64 pixels at
/// a time: the transitions of the nonzero mask (bits that differ from the
/// previous pixel) are the run starts and ends, taken off with ctz. Fails
/// when `out` is full.
fn findRuns(out: []Run, n: *usize, row: []const u8, y: u32) error{Overflow}!void {
    const N = 64;
    const V = @Vector(N, u8);
    const zero: V = @splat(0);
    const w = row.len;
    var k = n.*;
    defer n.* = k;
    var open = false;
    var start: usize = 0;
    var x: usize = 0;
    while (x < w) : (x += N) {
        const v: V = if (x + N <= w) row[x..][0..N].* else blk: {
            // The tail, padded with zeros (which also closes an open run).
            var buf: [N]u8 = @splat(0);
            @memcpy(buf[0 .. w - x], row[x..]);
            break :blk buf;
        };
        const m: u64 = @bitCast(v != zero);
        var t = m ^ ((m << 1) | @intFromBool(open));
        while (t != 0) {
            const i = x + @ctz(t);
            t &= t - 1;
            if (open) {
                if (k == out.len) return error.Overflow;
                out[k] = .{ .y = y, .x0 = @intCast(start), .x1 = @intCast(i - 1), .label = 0 };
                k += 1;
            } else start = i;
            open = !open;
        }
    }
    if (open) {
        if (k == out.len) return error.Overflow;
        out[k] = .{ .y = y, .x0 = @intCast(start), .x1 = @intCast(w - 1), .label = 0 };
        k += 1;
    }
}

/// Joins each run of `cur` to the runs of `prev` (the row above) it
/// touches. `id0` is the id of prev[0]; ids are consecutive.
fn joinRows(parent: [*]u32, prev: []const Run, prev_id: u32, cur: []const Run, cur_id: u32, slack: i64, comptime fresh: bool) void {
    var p: usize = 0;
    for (cur, 0..) |r, i| {
        // Previous-row runs are sorted by x; skip those ending too far left.
        while (p < prev.len and @as(i64, prev[p].x1) + slack < r.x0) p += 1;
        var q = p;
        const id: u32 = cur_id + @as(u32, @intCast(i));
        var root: u32 = if (fresh) id else find(parent, id);
        var joined = false;
        while (q < prev.len and @as(i64, prev[q].x0) <= @as(i64, r.x1) + slack) : (q += 1) {
            const pid = prev_id + @as(u32, @intCast(q));
            if (fresh and !joined) {
                root = find(parent, pid);
                joined = true;
            } else root = unite(parent, root, pid);
        }
        if (fresh) parent[id] = root;
    }
}

/// One band of rows [y0, y1), always starting on an even row so that 2x2
/// blocks never straddle two bands. Its runs occupy ids [off, off + n).
const Band = struct {
    y0: u32,
    y1: u32,
    off: u32,
    cap: u32,
    n: u32 = 0,
    /// Runs of row y0 are [off, off + first_end); runs of row y1 - 1
    /// are [off + last_start, off + n).
    first_end: u32 = 0,
    last_start: u32 = 0,
    overflow: bool = false,
    /// Components whose first run lies in this band, and the final label
    /// of the first of them.
    roots: u32 = 0,
    base: u32 = 0,
    /// Runs whose component starts in an earlier band; their statistics
    /// are added serially. Their ids go to foreign_list[foreign_off..].
    foreign: u32 = 0,
    foreign_off: u32 = 0,
    bg: Bg = .{},
};

/// Background (label 0) partial statistics.
const Bg = struct {
    x0: u64 = std.math.maxInt(u64),
    y0: u64 = std.math.maxInt(u64),
    x1: u64 = 0,
    y1: u64 = 0,
    area: u64 = 0,
    /// Twice the sum of x, and the sum of y.
    sx2: u64 = 0,
    sy: u64 = 0,

    fn add(a: *Bg, b: Bg) void {
        a.x0 = @min(a.x0, b.x0);
        a.y0 = @min(a.y0, b.y0);
        a.x1 = @max(a.x1, b.x1);
        a.y1 = @max(a.y1, b.y1);
        a.area += b.area;
        a.sx2 += b.sx2;
        a.sy += b.sy;
    }
};

/// Per-component accumulator. Sums are kept as integers (twice the x sum),
/// so the result is independent of the order runs are added in.
const Acc = struct {
    x0: u32 = std.math.maxInt(u32),
    y0: u32 = std.math.maxInt(u32),
    x1: u32 = 0,
    y1: u32 = 0,
    area: u64 = 0,
    sx2: u64 = 0,
    sy: u64 = 0,

    inline fn add(ac: *Acc, r: Run) void {
        const len: u64 = r.x1 - r.x0 + 1;
        ac.x0 = @min(ac.x0, r.x0);
        ac.x1 = @max(ac.x1, r.x1);
        ac.y0 = @min(ac.y0, r.y);
        ac.y1 = @max(ac.y1, r.y);
        ac.area += len;
        ac.sx2 += len * (@as(u64, r.x0) + r.x1);
        ac.sy += len * r.y;
    }
};

const Ctx = struct {
    src: Image(u8),
    labels: Image(i32),
    conn: Connectivity,
    slack: i64,
    bands: []Band,
    runs: []Run,
    /// The union-find forest; reused for final labels once resolved.
    parent: []u32,
    /// Per-band scratch (width + 2 entries) for ordering roots by block.
    scratch: []u32,
    accs: []Acc,
    foreign_list: []u32,

    /// Phase 1: runs and in-band joins.
    fn scanBands(c: *const Ctx, s: usize, e: usize) void {
        for (c.bands[s..e]) |*b| c.scan(b) catch {
            b.overflow = true;
        };
    }

    fn scan(c: *const Ctx, b: *Band) error{Overflow}!void {
        const out = c.runs[b.off..][0..b.cap];
        var n: usize = 0;
        var prev_start: usize = 0;
        var prev_end: usize = 0;
        b.first_end = 0;
        b.last_start = 0;
        for (b.y0..b.y1) |y| {
            const cur_start = n;
            try findRuns(out, &n, c.src.row(y), @intCast(y));
            joinRows(c.parent.ptr, out[prev_start..prev_end], b.off + @as(u32, @intCast(prev_start)), out[cur_start..n], b.off + @as(u32, @intCast(cur_start)), c.slack, true);
            if (y == b.y0) b.first_end = @intCast(n);
            prev_start = cur_start;
            prev_end = n;
        }
        // Runs of the last row start at prev_start if that row had any.
        b.last_start = if (prev_end > prev_start and out[prev_start].y == b.y1 - 1) @intCast(prev_start) else @intCast(n);
        b.n = @intCast(n);
    }

    /// Phase 3: each run's root, and counts of roots and foreign runs.
    fn resolveBands(c: *const Ctx, s: usize, e: usize) void {
        const par = c.parent.ptr;
        for (c.bands[s..e]) |*b| {
            var roots: u32 = 0;
            var foreign: u32 = 0;
            const o = b.off;
            for (c.runs[o..][0..b.n], o..) |*r, gi| {
                const g: u32 = @intCast(gi);
                const p = par[g];
                // A parent inside the band precedes this run: already resolved.
                const lab = if (p == g) g else if (p >= o) c.runs[p].label else findConst(par, p);
                r.label = lab;
                roots += @intFromBool(lab == g);
                foreign += @intFromBool(lab < o);
            }
            b.roots = roots;
            b.foreign = foreign;
        }
    }

    /// Phase 4: final labels of roots, stored in `parent` at the root's id.
    fn numberBands(c: *const Ctx, s: usize, e: usize) void {
        for (c.bands[s..e], s..) |*b, bi| {
            const runs = c.runs[b.off..][0..b.n];
            var k = b.base;
            if (c.conn == .four) {
                for (runs, b.off..) |r, g| {
                    if (r.label == g) {
                        c.parent[g] = k;
                        k += 1;
                    }
                }
                continue;
            }
            // 8-connectivity: order the roots of each block row (two rows)
            // by their first block column.
            const w2 = c.src.width + 2;
            const scratch = c.scratch[bi * w2 ..][0..w2];
            var i: usize = 0;
            while (i < runs.len) {
                const by = runs[i].y >> 1;
                const gstart: u32 = b.off + @as(u32, @intCast(i));
                var nr: usize = 0;
                var sorted = true;
                while (i < runs.len and runs[i].y >> 1 == by) : (i += 1) {
                    const r = runs[i];
                    const g = b.off + @as(u32, @intCast(i));
                    const key = r.x0 >> 1;
                    if (r.label == g) {
                        if (nr > 0 and c.parent[scratch[nr - 1]] > key) sorted = false;
                        c.parent[g] = key;
                        scratch[nr] = g;
                        nr += 1;
                    } else if (r.label >= gstart and key < c.parent[r.label]) {
                        c.parent[r.label] = key;
                        sorted = false;
                    }
                }
                const roots = scratch[0..nr];
                if (!sorted) std.sort.pdq(u32, roots, c.parent, struct {
                    fn lt(keys: []u32, a: u32, bb: u32) bool {
                        return keys[a] < keys[bb];
                    }
                }.lt);
                for (roots) |g| {
                    c.parent[g] = k;
                    k += 1;
                }
            }
        }
    }

    /// Phase 5: the label image, and statistics of components that start
    /// in this band; foreign runs are listed for the serial pass.
    fn writeBands(c: *const Ctx, s: usize, e: usize) void {
        const W: u64 = c.src.width;
        for (c.bands[s..e]) |*b| {
            for (c.accs[b.base..][0..b.roots]) |*a| a.* = .{};
            const o = b.off;
            const runs = c.runs[o..][0..b.n];
            var nf: u32 = 0;
            var bg: Bg = .{};
            var k: usize = 0;
            for (b.y0..b.y1) |y| {
                const out = c.labels.row(y);
                var x: usize = 0;
                var fg: u64 = 0;
                var fgx2: u64 = 0;
                const start = k;
                while (k < runs.len and runs[k].y == y) : (k += 1) {
                    const r = runs[k];
                    const l = c.parent[r.label];
                    fill(out[x..r.x0], 0);
                    fill(out[r.x0 .. r.x1 + 1], @intCast(l));
                    x = r.x1 + 1;
                    const len: u64 = r.x1 - r.x0 + 1;
                    fg += len;
                    fgx2 += len * (@as(u64, r.x0) + r.x1);
                    if (r.label >= o) c.accs[l].add(r) else {
                        c.foreign_list[b.foreign_off + nf] = o + @as(u32, @intCast(k));
                        nf += 1;
                    }
                }
                @memset(out[x..], 0);
                const zeros = W - fg;
                if (zeros == 0) continue;
                // First zero: 0 unless a run starts there; last zero likewise.
                const first: u64 = if (start < k and runs[start].x0 == 0) runs[start].x1 + 1 else 0;
                const last: u64 = if (start < k and runs[k - 1].x1 == W - 1) runs[k - 1].x0 - 1 else W - 1;
                bg.x0 = @min(bg.x0, first);
                bg.x1 = @max(bg.x1, last);
                bg.y0 = @min(bg.y0, @as(u64, y));
                bg.y1 = @max(bg.y1, @as(u64, y));
                bg.area += zeros;
                bg.sx2 += W * (W - 1) - fgx2;
                bg.sy += zeros * y;
            }
            b.bg = bg;
        }
    }
};

/// @memset for the short spans of a label row: overlapping vector stores
/// instead of a call per span.
inline fn fill(out: []i32, v: i32) void {
    const len = out.len;
    if (len >= 8) {
        const vv: @Vector(8, i32) = @splat(v);
        var i: usize = 0;
        while (i + 8 <= len) : (i += 8) out[i..][0..8].* = vv;
        out[len - 8 ..][0..8].* = vv;
    } else if (len >= 4) {
        const vv: @Vector(4, i32) = @splat(v);
        out[0..4].* = vv;
        out[len - 4 ..][0..4].* = vv;
    } else for (out) |*o| {
        o.* = v;
    }
}

/// Rows per band and bands for `exec`: one band when serial; otherwise
/// four per thread (chunks are claimed dynamically), at least 32 rows each.
fn bandLayout(exec: Exec, h: u32) struct { rows: u32, count: u32 } {
    const t = exec.threads();
    if (t <= 1 or h < 64) return .{ .rows = @max(h, 1), .count = 1 };
    const want: u32 = @intCast(@min(t * 4, h / 32));
    var rows = (h + want - 1) / want;
    rows += rows & 1; // even, so 2x2 blocks stay inside one band
    return .{ .rows = rows, .count = (h + rows - 1) / rows };
}

/// Labels nonzero pixels of a single-channel u8 image into `labels`
/// (same size, i32). Returns per-label statistics (caller frees). With a
/// pool, bands of rows are labelled in parallel; the result is identical
/// to the serial one.
pub fn connectedComponentsWithStats(exec: Exec, gpa: Allocator, src: Image(u8), labels: Image(i32), conn: Connectivity) Error!Result {
    if (src.channels != 1 or labels.channels != 1 or !src.sameSize(labels)) return error.SizeMismatch;
    const h = src.height;
    const w = src.width;
    const per_row: u64 = (@as(u64, w) + 1) / 2;
    const layout = bandLayout(exec, h);
    const nb = layout.count;

    const bands = try gpa.alloc(Band, nb);
    defer gpa.free(bands);
    for (bands, 0..) |*b, i| {
        const y0: u32 = @intCast(i * layout.rows);
        b.* = .{ .y0 = y0, .y1 = @min(h, y0 + layout.rows), .off = 0, .cap = 0 };
    }

    var c: Ctx = .{
        .src = src,
        .labels = labels,
        .conn = conn,
        .slack = if (conn == .eight) 1 else 0,
        .bands = bands,
        .runs = &.{},
        .parent = &.{},
        .scratch = &.{},
        .accs = &.{},
        .foreign_list = &.{},
    };
    defer gpa.free(c.runs);
    defer gpa.free(c.parent);

    if (nb == 1) {
        // Serial: grow the run list as needed.
        var runs: std.ArrayList(Run) = .empty;
        errdefer runs.deinit(gpa);
        var parent: std.ArrayList(u32) = .empty;
        errdefer parent.deinit(gpa);
        var prev_start: usize = 0;
        var prev_end: usize = 0;
        for (0..h) |y| {
            const cur_start = runs.items.len;
            try runs.ensureUnusedCapacity(gpa, @intCast(per_row));
            var n = cur_start;
            findRuns(runs.allocatedSlice(), &n, src.row(y), @intCast(y)) catch unreachable;
            runs.items.len = n;
            if (n > std.math.maxInt(u32)) return error.OutOfMemory;
            try parent.resize(gpa, n);
            joinRows(parent.items.ptr, runs.items[prev_start..prev_end], @intCast(prev_start), runs.items[cur_start..n], @intCast(cur_start), c.slack, true);
            prev_start = cur_start;
            prev_end = n;
        }
        c.runs = try runs.toOwnedSlice(gpa);
        c.parent = try parent.toOwnedSlice(gpa);
        bands[0].n = @intCast(c.runs.len);
    } else {
        // Parallel: each band gets a fixed slice of the id space, sized by
        // a guess; a band that overflows is redone with room for its worst
        // case (every other pixel set).
        for (bands) |*b| b.cap = @intCast(@min(@min(per_row, @max(64, w / 8)) * (b.y1 - b.y0), std.math.maxInt(u32)));
        while (true) {
            var total: u64 = 0;
            for (bands) |*b| {
                b.off = @intCast(@min(total, std.math.maxInt(u32)));
                total += b.cap;
            }
            if (total > std.math.maxInt(u32)) return error.OutOfMemory;
            const len = std.math.cast(usize, total) orelse return error.OutOfMemory;
            gpa.free(c.runs);
            c.runs = &.{};
            gpa.free(c.parent);
            c.parent = &.{};
            c.runs = try gpa.alloc(Run, len);
            c.parent = try gpa.alloc(u32, len);
            exec.parallelFor(nb, 1, @as(*const Ctx, &c), Ctx.scanBands);
            var again = false;
            for (bands) |*b| {
                if (b.overflow) {
                    b.overflow = false;
                    b.cap = @intCast(@min(per_row * (b.y1 - b.y0), std.math.maxInt(u32)));
                    again = true;
                } else b.cap = b.n;
            }
            if (!again) break;
        }
        // Phase 2: stitch each band to the one above along the boundary.
        const par = c.parent.ptr;
        for (bands[1..], bands[0 .. nb - 1]) |b, a| {
            const prev = c.runs[a.off + a.last_start .. a.off + a.n];
            const cur = c.runs[b.off..][0..b.first_end];
            if (prev.len == 0 or cur.len == 0 or prev[0].y + 1 != b.y0) continue;
            joinRows(par, prev, a.off + a.last_start, cur, b.off, c.slack, false);
        }
    }

    exec.parallelFor(nb, 1, @as(*const Ctx, &c), Ctx.resolveBands);
    var next: u32 = 1;
    var nforeign: u32 = 0;
    for (bands) |*b| {
        b.base = next;
        next += b.roots;
        b.foreign_off = nforeign;
        nforeign += b.foreign;
    }
    if (conn == .eight) c.scratch = try gpa.alloc(u32, @as(usize, nb) * (w + 2));
    defer gpa.free(c.scratch);
    exec.parallelFor(nb, 1, @as(*const Ctx, &c), Ctx.numberBands);

    c.accs = try gpa.alloc(Acc, next);
    defer gpa.free(c.accs);
    c.foreign_list = try gpa.alloc(u32, nforeign);
    defer gpa.free(c.foreign_list);
    exec.parallelFor(nb, 1, @as(*const Ctx, &c), Ctx.writeBands);
    for (c.foreign_list) |g| c.accs[c.parent[c.runs[g].label]].add(c.runs[g]);

    const stats = try gpa.alloc(Stats, next);
    errdefer gpa.free(stats);
    var bg: Bg = .{};
    for (bands) |b| bg.add(b.bg);
    {
        const a: f64 = @floatFromInt(@max(bg.area, 1));
        // No background at all: OpenCV reports an empty box and NaN centroid.
        stats[0] = if (bg.area == 0) .{ .left = std.math.maxInt(u32), .top = std.math.maxInt(u32), .width = 0, .height = 0, .area = 0, .cx = std.math.nan(f64), .cy = std.math.nan(f64) } else .{
            .left = @intCast(bg.x0),
            .top = @intCast(bg.y0),
            .width = @intCast(bg.x1 - bg.x0 + 1),
            .height = @intCast(bg.y1 - bg.y0 + 1),
            .area = bg.area,
            .cx = @as(f64, @floatFromInt(bg.sx2)) / 2.0 / a,
            .cy = @as(f64, @floatFromInt(bg.sy)) / a,
        };
    }
    for (stats[1..], c.accs[1..]) |*st, ac| {
        const a: f64 = @floatFromInt(ac.area);
        st.* = .{ .left = ac.x0, .top = ac.y0, .width = ac.x1 - ac.x0 + 1, .height = ac.y1 - ac.y0 + 1, .area = ac.area, .cx = @as(f64, @floatFromInt(ac.sx2)) / 2.0 / a, .cy = @as(f64, @floatFromInt(ac.sy)) / a };
    }
    return .{ .count = next, .stats = stats };
}

test "two components, raster order" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).initFill(gpa, 6, 4, 1, 0);
    defer src.deinit(gpa);
    // A U shape that merges late (tests union) and a separate dot.
    for ([_][2]u32{ .{ 0, 0 }, .{ 0, 1 }, .{ 0, 2 }, .{ 1, 2 }, .{ 2, 2 }, .{ 2, 1 }, .{ 2, 0 }, .{ 5, 3 } }) |pt| src.set(pt[0], pt[1], 0, 255);
    var lab = try Image(i32).init(gpa, 6, 4, 1);
    defer lab.deinit(gpa);
    var r = try connectedComponentsWithStats(.serial, gpa, src, lab, .eight);
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 3), r.count);
    try std.testing.expectEqual(@as(i32, 1), lab.at(2, 0, 0));
    try std.testing.expectEqual(@as(i32, 2), lab.at(5, 3, 0));
    try std.testing.expectEqual(@as(u64, 7), r.stats[1].area);
    try std.testing.expectEqual(@as(u32, 3), r.stats[1].width);
}

/// Flood-fill reference with OpenCV's numbering (first pixel in raster
/// order for 4-connectivity, first 2x2 block in block-raster order for 8).
fn referenceLabels(gpa: Allocator, src: Image(u8), out: []i32, conn: Connectivity) !u32 {
    const w = src.width;
    const h = src.height;
    @memset(out, 0);
    var stack: std.ArrayList([2]u32) = .empty;
    defer stack.deinit(gpa);
    var next: i32 = 1;
    const bw = (w + 1) / 2;
    const bh = (h + 1) / 2;
    for (0..bh) |by| for (0..bw) |bx| for (0..2) |dy| for (0..2) |dx| {
        // 4-connectivity scans pixels in raster order; emulate by
        // visiting pixel (bx*2+dx, by*2+dy) only on the 8 path.
        if (conn == .four) continue;
        const x = bx * 2 + dx;
        const y = by * 2 + dy;
        if (x >= w or y >= h or src.at(x, y, 0) == 0 or out[y * w + x] != 0) continue;
        try flood(gpa, &stack, src, out, x, y, next, conn);
        next += 1;
    };
    if (conn == .four) for (0..h) |y| for (0..w) |x| {
        if (src.at(x, y, 0) == 0 or out[y * w + x] != 0) continue;
        try flood(gpa, &stack, src, out, x, y, next, conn);
        next += 1;
    };
    return @intCast(next);
}

fn flood(gpa: Allocator, stack: *std.ArrayList([2]u32), src: Image(u8), out: []i32, x: usize, y: usize, l: i32, conn: Connectivity) !void {
    const w = src.width;
    out[y * w + x] = l;
    try stack.append(gpa, .{ @intCast(x), @intCast(y) });
    while (stack.pop()) |p| {
        var dy: i64 = -1;
        while (dy <= 1) : (dy += 1) {
            var dx: i64 = -1;
            while (dx <= 1) : (dx += 1) {
                if (dx == 0 and dy == 0) continue;
                if (conn == .four and dx != 0 and dy != 0) continue;
                const nx = @as(i64, p[0]) + dx;
                const ny = @as(i64, p[1]) + dy;
                if (nx < 0 or ny < 0 or nx >= w or ny >= src.height) continue;
                const ux: usize = @intCast(nx);
                const uy: usize = @intCast(ny);
                if (src.at(ux, uy, 0) == 0 or out[uy * w + ux] != 0) continue;
                out[uy * w + ux] = l;
                try stack.append(gpa, .{ @intCast(ux), @intCast(uy) });
            }
        }
    }
}

test "parallel bands match serial and a flood-fill reference" {
    const gpa = std.testing.allocator;
    var pool = core.Pool.init(gpa, std.testing.io, 4);
    defer pool.deinit();
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    const Pattern = enum { random, sparse, checker, spiral, stripes, zeros, ones, diag };
    const sizes = [_][2]u32{ .{ 1, 1 }, .{ 1, 300 }, .{ 300, 1 }, .{ 37, 129 }, .{ 200, 257 }, .{ 64, 64 }, .{ 33, 400 } };
    for (sizes) |sz| for (std.enums.values(Pattern)) |pat| {
        const w = sz[0];
        const h = sz[1];
        var src = try Image(u8).initFill(gpa, w, h, 1, 0);
        defer src.deinit(gpa);
        for (0..h) |y| for (0..w) |x| {
            const v: bool = switch (pat) {
                .random => rnd.uintLessThan(u8, 10) < 5,
                .sparse => rnd.uintLessThan(u8, 10) == 0,
                .checker => (x + y) % 2 == 0,
                .stripes => x % 3 == 0 or (y % 50 == 0),
                .zeros => false,
                .ones => true,
                .diag => (x + y) % 7 == 0 or (x + 2 * h - y) % 11 == 0,
                .spiral => blk: {
                    // Concentric square rings with one gap each, joined
                    // into a single path that crosses every band.
                    const d = @min(@min(x, y), @min(w - 1 - x, h - 1 - y));
                    break :blk d % 2 == 0 and !(x == d + 1 and y == d and d + 2 < w);
                },
            };
            if (v) src.set(x, y, 0, 255);
        };
        for ([_]Connectivity{ .four, .eight }) |conn| {
            var ls = try Image(i32).init(gpa, w, h, 1);
            defer ls.deinit(gpa);
            var lp = try Image(i32).init(gpa, w, h, 1);
            defer lp.deinit(gpa);
            var rs = try connectedComponentsWithStats(.serial, gpa, src, ls, conn);
            defer rs.deinit(gpa);
            var rp = try connectedComponentsWithStats(.on(&pool), gpa, src, lp, conn);
            defer rp.deinit(gpa);
            try std.testing.expectEqual(rs.count, rp.count);
            try std.testing.expectEqualSlices(i32, ls.slice(), lp.slice());
            for (rs.stats, rp.stats) |a, b| {
                try std.testing.expectEqual(a.left, b.left);
                try std.testing.expectEqual(a.top, b.top);
                try std.testing.expectEqual(a.width, b.width);
                try std.testing.expectEqual(a.height, b.height);
                try std.testing.expectEqual(a.area, b.area);
                try std.testing.expect(a.cx == b.cx or (std.math.isNan(a.cx) and std.math.isNan(b.cx)));
                try std.testing.expect(a.cy == b.cy or (std.math.isNan(a.cy) and std.math.isNan(b.cy)));
            }
            const ref = try gpa.alloc(i32, @as(usize, w) * h);
            defer gpa.free(ref);
            const n = try referenceLabels(gpa, src, ref, conn);
            try std.testing.expectEqual(n, rs.count);
            try std.testing.expectEqualSlices(i32, ref, ls.slice());
        }
    };
}
