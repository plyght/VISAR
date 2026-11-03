//! Connected-component labeling with statistics (connectedComponentsWithStats).
//! Two raster passes with a union-find over provisional labels; final labels
//! are numbered in raster order of each component's first pixel, which is
//! the order OpenCV produces.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
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

fn find(parent: []u32, x_in: u32) u32 {
    var x = x_in;
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

fn unite(parent: []u32, a: u32, b: u32) u32 {
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

/// A horizontal run of nonzero pixels [x0, x1] on row y, with its
/// provisional label.
const Run = struct { y: u32, x0: u32, x1: u32, label: u32 };

/// Appends the nonzero runs of `row` to `runs`. Skips empty 32-byte
/// chunks with one vector compare.
fn findRuns(gpa: Allocator, runs: *std.ArrayList(Run), row: []const u8, y: u32) Allocator.Error!void {
    const N = 32;
    const V = @Vector(N, u8);
    const zero: V = @splat(0);
    var x: usize = 0;
    const w = row.len;
    while (x < w) {
        // Skip zeros.
        while (x + N <= w) {
            const v: V = row[x..][0..N].*;
            if (@reduce(.Or, v != zero)) break;
            x += N;
        }
        while (x < w and row[x] == 0) x += 1;
        if (x >= w) break;
        const start = x;
        // Extend through nonzeros.
        while (x + N <= w) {
            const v: V = row[x..][0..N].*;
            if (@reduce(.Or, v == zero)) break;
            x += N;
        }
        while (x < w and row[x] != 0) x += 1;
        try runs.append(gpa, .{ .y = y, .x0 = @intCast(start), .x1 = @intCast(x - 1), .label = 0 });
    }
}

/// Labels nonzero pixels of a single-channel u8 image into `labels`
/// (same size, i32). Returns per-label statistics (caller frees).
///
/// Works on runs, not pixels: each row's nonzero runs are found with
/// vector skips, joined to overlapping runs of the previous row through a
/// union-find, and statistics accumulate per run. Final labels are
/// numbered in raster order of each component's first pixel (OpenCV's
/// order): a component's first run always gets the smallest provisional
/// label of the component.
pub fn connectedComponentsWithStats(gpa: Allocator, src: Image(u8), labels: Image(i32), conn: Connectivity) Error!Result {
    if (src.channels != 1 or labels.channels != 1 or !src.sameSize(labels)) return error.SizeMismatch;
    const h = src.height;
    var runs: std.ArrayList(Run) = .empty;
    defer runs.deinit(gpa);
    var parent: std.ArrayList(u32) = .empty;
    defer parent.deinit(gpa);
    try parent.append(gpa, 0);
    // 8-connectivity also joins runs touching diagonally.
    const slack: i64 = if (conn == .eight) 1 else 0;

    var prev_start: usize = 0;
    var prev_end: usize = 0;
    for (0..h) |y| {
        const cur_start = runs.items.len;
        try findRuns(gpa, &runs, src.row(y), @intCast(y));
        var p = prev_start;
        for (runs.items[cur_start..]) |*r| {
            // Previous-row runs are sorted by x; skip those ending too far left.
            while (p < prev_end and @as(i64, runs.items[p].x1) + slack < r.x0) p += 1;
            var q = p;
            var lab: u32 = 0;
            while (q < prev_end and @as(i64, runs.items[q].x0) <= @as(i64, r.x1) + slack) : (q += 1) {
                const pl = runs.items[q].label;
                lab = if (lab == 0) pl else unite(parent.items, lab, pl);
            }
            if (lab == 0) {
                lab = @intCast(parent.items.len);
                try parent.append(gpa, lab);
            }
            r.label = lab;
        }
        prev_start = cur_start;
        prev_end = runs.items.len;
    }

    // Resolve and renumber in order of first appearance (raster order).
    const par = parent.items;
    const final = try gpa.alloc(u32, par.len);
    defer gpa.free(final);
    @memset(final, 0);
    var next: u32 = 1;
    for (1..par.len) |i| {
        const r = find(par, @intCast(i));
        if (final[r] == 0) {
            final[r] = next;
            next += 1;
        }
        final[i] = final[r];
    }

    // OpenCV's 8-connected labeling (Spaghetti) scans 2x2 blocks, so it
    // numbers components by their first block in block-raster order rather
    // than by their first pixel. Renumber to match.
    if (conn == .eight and next > 2) {
        const bw: u64 = (@as(u64, src.width) + 1) / 2;
        const keys = try gpa.alloc(u64, next);
        defer gpa.free(keys);
        @memset(keys, std.math.maxInt(u64));
        for (runs.items) |r| {
            const l = final[r.label];
            keys[l] = @min(keys[l], @as(u64, r.y >> 1) * bw + (r.x0 >> 1));
        }
        const order = try gpa.alloc(u32, next - 1);
        defer gpa.free(order);
        for (order, 1..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, keys, struct {
            fn lt(k: []const u64, a_: u32, b_: u32) bool {
                return k[a_] < k[b_];
            }
        }.lt);
        const remap = try gpa.alloc(u32, next);
        defer gpa.free(remap);
        remap[0] = 0;
        for (order, 1..) |old, new_| remap[old] = @intCast(new_);
        for (final) |*f| f.* = remap[f.*];
    }

    // Statistics per run, and the label image.
    const Acc = struct { x0: u32, y0: u32, x1: u32, y1: u32, area: u64, sx: f64, sy: f64 };
    const accs = try gpa.alloc(Acc, next);
    defer gpa.free(accs);
    for (accs) |*a_| a_.* = .{ .x0 = std.math.maxInt(u32), .y0 = std.math.maxInt(u32), .x1 = 0, .y1 = 0, .area = 0, .sx = 0, .sy = 0 };
    for (0..h) |y| @memset(labels.row(y), 0);
    var bg_area: u64 = @as(u64, src.width) * h;
    for (runs.items) |r| {
        const l = final[r.label];
        const len: u64 = r.x1 - r.x0 + 1;
        @memset(labels.row(r.y)[r.x0 .. r.x1 + 1], @intCast(l));
        const ac = &accs[l];
        ac.x0 = @min(ac.x0, r.x0);
        ac.x1 = @max(ac.x1, r.x1);
        ac.y0 = @min(ac.y0, r.y);
        ac.y1 = @max(ac.y1, r.y);
        ac.area += len;
        ac.sx += @as(f64, @floatFromInt(len)) * (@as(f64, @floatFromInt(r.x0)) + @as(f64, @floatFromInt(r.x1))) / 2.0;
        ac.sy += @as(f64, @floatFromInt(len)) * @as(f64, @floatFromInt(r.y));
        bg_area -= len;
    }

    const stats = try gpa.alloc(Stats, next);
    errdefer gpa.free(stats);
    // Background statistics (label 0), derived from the runs: totals minus
    // foreground, and per row the first and last zero pixel from its gaps.
    {
        const W: u64 = src.width;
        var bx0: u64 = std.math.maxInt(u64);
        var by0: u64 = std.math.maxInt(u64);
        var bx1: u64 = 0;
        var by1: u64 = 0;
        var bsx: f64 = 0;
        var bsy: f64 = 0;
        var k: usize = 0;
        const rs = runs.items;
        for (0..h) |y| {
            const start = k;
            var fg: u64 = 0;
            var fgx: f64 = 0;
            while (k < rs.len and rs[k].y == y) : (k += 1) {
                const len: u64 = rs[k].x1 - rs[k].x0 + 1;
                fg += len;
                fgx += @as(f64, @floatFromInt(len)) * (@as(f64, @floatFromInt(rs[k].x0)) + @as(f64, @floatFromInt(rs[k].x1))) / 2.0;
            }
            const zeros = W - fg;
            if (zeros == 0) continue;
            // First zero: 0 unless a run starts there; last zero likewise.
            const first: u64 = if (start < k and rs[start].x0 == 0) rs[start].x1 + 1 else 0;
            const last: u64 = if (start < k and rs[k - 1].x1 == W - 1) rs[k - 1].x0 - 1 else W - 1;
            bx0 = @min(bx0, first);
            bx1 = @max(bx1, last);
            by0 = @min(by0, @as(u64, y));
            by1 = @max(by1, @as(u64, y));
            const allx = @as(f64, @floatFromInt(W)) * @as(f64, @floatFromInt(W - 1)) / 2.0;
            bsx += allx - fgx;
            bsy += @as(f64, @floatFromInt(zeros)) * @as(f64, @floatFromInt(y));
        }
        const a: f64 = @floatFromInt(@max(bg_area, 1));
        // No background at all: OpenCV reports an empty box and NaN centroid.
        stats[0] = if (bg_area == 0) .{ .left = std.math.maxInt(u32), .top = std.math.maxInt(u32), .width = 0, .height = 0, .area = 0, .cx = std.math.nan(f64), .cy = std.math.nan(f64) } else .{ .left = @intCast(bx0), .top = @intCast(by0), .width = @intCast(bx1 - bx0 + 1), .height = @intCast(by1 - by0 + 1), .area = bg_area, .cx = bsx / a, .cy = bsy / a };
    }
    for (stats[1..], accs[1..]) |*st, ac| {
        const a: f64 = @floatFromInt(ac.area);
        st.* = .{ .left = ac.x0, .top = ac.y0, .width = ac.x1 - ac.x0 + 1, .height = ac.y1 - ac.y0 + 1, .area = ac.area, .cx = ac.sx / a, .cy = ac.sy / a };
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
    var r = try connectedComponentsWithStats(gpa, src, lab, .eight);
    defer r.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 3), r.count);
    try std.testing.expectEqual(@as(i32, 1), lab.at(2, 0, 0));
    try std.testing.expectEqual(@as(i32, 2), lab.at(5, 3, 0));
    try std.testing.expectEqual(@as(u64, 7), r.stats[1].area);
    try std.testing.expectEqual(@as(u32, 3), r.stats[1].width);
}
