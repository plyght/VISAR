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

/// Labels nonzero pixels of a single-channel u8 image into `labels`
/// (same size, i32). Returns per-label statistics (caller frees).
pub fn connectedComponentsWithStats(gpa: Allocator, src: Image(u8), labels: Image(i32), conn: Connectivity) Error!Result {
    if (src.channels != 1 or labels.channels != 1 or !src.sameSize(labels)) return error.SizeMismatch;
    const w = src.width;
    const h = src.height;
    // Provisional labels: at most one per run start; bound by pixels/2 + 1.
    var parent: std.ArrayList(u32) = .empty;
    defer parent.deinit(gpa);
    try parent.append(gpa, 0);

    for (0..h) |y| {
        const s = src.row(y);
        const l = labels.row(y);
        const up: ?[]i32 = if (y > 0) labels.row(y - 1) else null;
        for (0..w) |x| {
            if (s[x] == 0) {
                l[x] = 0;
                continue;
            }
            var lab: u32 = 0;
            const left: u32 = if (x > 0) @intCast(l[x - 1]) else 0;
            if (left != 0) lab = left;
            if (up) |u| {
                const cands: [3]u32 = if (conn == .eight) .{
                    if (x > 0) @intCast(u[x - 1]) else 0,
                    @intCast(u[x]),
                    if (x + 1 < w) @intCast(u[x + 1]) else 0,
                } else .{ 0, @intCast(u[x]), 0 };
                for (cands) |c| if (c != 0) {
                    lab = if (lab == 0) c else unite(parent.items, lab, c);
                };
            }
            if (lab == 0) {
                lab = @intCast(parent.items.len);
                try parent.append(gpa, lab);
            }
            l[x] = @intCast(lab);
        }
    }

    // Resolve and renumber in order of first appearance (raster order).
    const p = parent.items;
    const final = try gpa.alloc(u32, p.len);
    defer gpa.free(final);
    @memset(final, 0);
    var next: u32 = 1;
    for (1..p.len) |i| {
        const r = find(p, @intCast(i));
        if (final[r] == 0) {
            final[r] = next;
            next += 1;
        }
        final[i] = final[r];
    }
    // Roots always have smaller ids than their members, so the loop above
    // numbers components by their smallest provisional label. Provisional
    // labels are created in raster order, so that is raster order too.

    const stats = try gpa.alloc(Stats, next);
    errdefer gpa.free(stats);
    var sx = try gpa.alloc(f64, next);
    defer gpa.free(sx);
    var sy = try gpa.alloc(f64, next);
    defer gpa.free(sy);
    var x0 = try gpa.alloc(u32, next);
    defer gpa.free(x0);
    var y0 = try gpa.alloc(u32, next);
    defer gpa.free(y0);
    var x1 = try gpa.alloc(u32, next);
    defer gpa.free(x1);
    var y1 = try gpa.alloc(u32, next);
    defer gpa.free(y1);
    var area = try gpa.alloc(u64, next);
    defer gpa.free(area);
    @memset(sx, 0);
    @memset(sy, 0);
    @memset(x0, std.math.maxInt(u32));
    @memset(y0, std.math.maxInt(u32));
    @memset(x1, 0);
    @memset(y1, 0);
    @memset(area, 0);
    for (0..h) |y| {
        const l = labels.row(y);
        for (l, 0..) |*v, x| {
            const lab: u32 = if (v.* == 0) 0 else final[@intCast(v.*)];
            v.* = @intCast(lab);
            const xi: u32 = @intCast(x);
            const yi: u32 = @intCast(y);
            area[lab] += 1;
            sx[lab] += @floatFromInt(x);
            sy[lab] += @floatFromInt(y);
            x0[lab] = @min(x0[lab], xi);
            y0[lab] = @min(y0[lab], yi);
            x1[lab] = @max(x1[lab], xi);
            y1[lab] = @max(y1[lab], yi);
        }
    }
    for (stats, 0..) |*st, i| {
        if (area[i] == 0) {
            st.* = .{ .left = 0, .top = 0, .width = 0, .height = 0, .area = 0, .cx = 0, .cy = 0 };
            continue;
        }
        const a: f64 = @floatFromInt(area[i]);
        st.* = .{ .left = x0[i], .top = y0[i], .width = x1[i] - x0[i] + 1, .height = y1[i] - y0[i] + 1, .area = area[i], .cx = sx[i] / a, .cy = sy[i] / a };
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
