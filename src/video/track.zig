//! Motion detection and multi-object tracking.
//!
//! `Background` keeps a running-average background model and produces a
//! foreground mask; `Tracker` turns per-frame detections (boxes) into
//! persistent tracks with IDs, velocities and trails, using constant-velocity
//! prediction and greedy nearest-neighbour association with a distance gate.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

/// Running-average background on a single-channel 8-bit image, stored in
/// 8.8 fixed point. `rate` is the per-frame learning rate as a power of two
/// (rate = 1 / 2^shift).
pub const Background = struct {
    model: Image(u16),
    shift: u4,
    primed: bool = false,

    pub fn init(gpa: Allocator, width: u32, height: u32, shift: u4) !Background {
        return .{ .model = try Image(u16).init(gpa, width, height, 1), .shift = shift };
    }

    pub fn deinit(self: *Background, gpa: Allocator) void {
        self.model.deinit(gpa);
    }

    /// Writes 255 to `mask` where |frame - background| > `threshold`, then
    /// updates the model (pixels under the mask learn 4x slower so moving
    /// objects do not burn in).
    pub fn apply(self: *Background, exec: Exec, frame: Image(u8), mask: Image(u8), threshold: u8) !void {
        if (frame.channels != 1 or !frame.sameShape(mask) or !frame.sameSize(self.model)) return error.SizeMismatch;
        if (!self.primed) {
            for (0..frame.height) |y| for (frame.row(y), self.model.row(y)) |v, *m| {
                m.* = @as(u16, v) << 8;
            };
            mask.fill(0);
            self.primed = true;
            return;
        }
        const Ctx = struct { f: Image(u8), m: Image(u8), bg: Image(u16), t: u8, sh: u4 };
        exec.parallelFor(frame.height, 16, Ctx{ .f = frame, .m = mask, .bg = self.model, .t = threshold, .sh = self.shift }, struct {
            fn run(c: Ctx, y0: usize, y1: usize) void {
                const N = 16;
                for (y0..y1) |y| {
                    const f = c.f.row(y);
                    const m = c.m.row(y);
                    const b = c.bg.row(y);
                    var x: usize = 0;
                    while (x + N <= f.len) : (x += N) {
                        const fv: @Vector(N, i32) = @as(@Vector(N, u8), f[x..][0..N].*);
                        const bv: @Vector(N, i32) = @as(@Vector(N, u16), b[x..][0..N].*);
                        const diff = @abs(fv - (bv >> @splat(8)));
                        const fg = diff > @as(@Vector(N, u32), @splat(c.t));
                        m[x..][0..N].* = @select(u8, fg, @as(@Vector(N, u8), @splat(255)), @as(@Vector(N, u8), @splat(0)));
                        // b += (f*256 - b) >> shift (slower under the mask)
                        const sh_fg: @Vector(N, i32) = @splat(@as(i32, c.sh) + 2);
                        const sh_bg: @Vector(N, i32) = @splat(@as(i32, c.sh));
                        const delta = (fv << @splat(8)) - bv;
                        const step = delta >> @intCast(@select(i32, fg, sh_fg, sh_bg));
                        b[x..][0..N].* = @as(@Vector(N, u16), @intCast(bv + step));
                    }
                    while (x < f.len) : (x += 1) {
                        const fv: i32 = f[x];
                        const bv: i32 = b[x];
                        const fg = @abs(fv - (bv >> 8)) > c.t;
                        m[x] = if (fg) 255 else 0;
                        const sh: u5 = @as(u5, c.sh) + @as(u5, if (fg) 2 else 0);
                        b[x] = @intCast(bv + (((fv << 8) - bv) >> sh));
                    }
                }
            }
        }.run);
    }
};

pub const Box = struct {
    x: f32,
    y: f32,
    w: f32,
    h: f32,

    pub fn cx(b: Box) f32 {
        return b.x + b.w / 2;
    }
    pub fn cy(b: Box) f32 {
        return b.y + b.h / 2;
    }
};

pub const trail_len = 48;

pub const Track = struct {
    id: u32,
    box: Box,
    vx: f32 = 0,
    vy: f32 = 0,
    /// Frames since creation and since the last matched detection.
    age: u32 = 0,
    missed: u32 = 0,
    hits: u32 = 1,
    trail: [trail_len][2]f32 = undefined,
    trail_count: usize = 0,
    trail_head: usize = 0,

    fn push(t: *Track, x: f32, y: f32) void {
        t.trail[t.trail_head] = .{ x, y };
        t.trail_head = (t.trail_head + 1) % trail_len;
        t.trail_count = @min(t.trail_count + 1, trail_len);
    }

    /// Trail point i, oldest first.
    pub fn trailPoint(t: Track, i: usize) [2]f32 {
        const start = (t.trail_head + trail_len - t.trail_count) % trail_len;
        return t.trail[(start + i) % trail_len];
    }

    /// A track is reported once it has been seen in a few frames.
    pub fn confirmed(t: Track) bool {
        return t.hits >= 3;
    }
};

pub const Tracker = struct {
    gpa: Allocator,
    tracks: std.ArrayList(Track) = .empty,
    next_id: u32 = 1,
    /// Maximum distance (pixels) between a prediction and a detection.
    gate: f32 = 60,
    /// Tracks unseen for this many frames are dropped.
    max_missed: u32 = 8,

    pub fn init(gpa: Allocator) Tracker {
        return .{ .gpa = gpa };
    }

    pub fn deinit(self: *Tracker) void {
        self.tracks.deinit(self.gpa);
    }

    /// Associates `detections` with existing tracks and updates them.
    pub fn update(self: *Tracker, detections: []const Box) !void {
        const gpa = self.gpa;
        const T = self.tracks.items.len;
        const D = detections.len;
        const used_t = try gpa.alloc(bool, T);
        defer gpa.free(used_t);
        const used_d = try gpa.alloc(bool, D);
        defer gpa.free(used_d);
        @memset(used_t, false);
        @memset(used_d, false);

        // All (track, detection) pairs within the gate, nearest first.
        const Pair = struct { d2: f32, t: u32, d: u32 };
        var pairs: std.ArrayList(Pair) = .empty;
        defer pairs.deinit(gpa);
        for (self.tracks.items, 0..) |tr, ti| {
            const px = tr.box.cx() + tr.vx;
            const py = tr.box.cy() + tr.vy;
            const g = self.gate + 0.5 * @max(tr.box.w, tr.box.h);
            for (detections, 0..) |det, di| {
                const dx = det.cx() - px;
                const dy = det.cy() - py;
                const d2 = dx * dx + dy * dy;
                if (d2 <= g * g) try pairs.append(gpa, .{ .d2 = d2, .t = @intCast(ti), .d = @intCast(di) });
            }
        }
        std.mem.sort(Pair, pairs.items, {}, struct {
            fn lt(_: void, a: Pair, b: Pair) bool {
                return a.d2 < b.d2;
            }
        }.lt);
        for (pairs.items) |p| {
            if (used_t[p.t] or used_d[p.d]) continue;
            used_t[p.t] = true;
            used_d[p.d] = true;
            const tr = &self.tracks.items[p.t];
            const det = detections[p.d];
            // Smooth the velocity; take the new box directly.
            const nvx = det.cx() - tr.box.cx();
            const nvy = det.cy() - tr.box.cy();
            tr.vx = 0.6 * tr.vx + 0.4 * nvx;
            tr.vy = 0.6 * tr.vy + 0.4 * nvy;
            tr.box = det;
            tr.missed = 0;
            tr.hits += 1;
            tr.age += 1;
            tr.push(det.cx(), det.cy());
        }
        // Unmatched tracks coast on their velocity; stale ones are dropped.
        var keep: usize = 0;
        for (self.tracks.items, 0..) |tr_in, ti| {
            var tr = tr_in;
            if (!used_t[ti]) {
                tr.missed += 1;
                tr.age += 1;
                tr.box.x += tr.vx;
                tr.box.y += tr.vy;
                if (tr.missed > self.max_missed or (!tr.confirmed() and tr.missed > 1)) continue;
            }
            self.tracks.items[keep] = tr;
            keep += 1;
        }
        self.tracks.shrinkRetainingCapacity(keep);
        // New tracks for unmatched detections.
        for (detections, 0..) |det, di| {
            if (used_d[di]) continue;
            var t: Track = .{ .id = self.next_id, .box = det };
            t.push(det.cx(), det.cy());
            self.next_id += 1;
            try self.tracks.append(gpa, t);
        }
    }
};

test "tracker keeps ids" {
    const gpa = std.testing.allocator;
    var tr = Tracker.init(gpa);
    defer tr.deinit();
    for (0..10) |f| {
        const x: f32 = @floatFromInt(f * 5);
        try tr.update(&.{ .{ .x = x, .y = 10, .w = 10, .h = 10 }, .{ .x = 200 - x, .y = 100, .w = 12, .h = 12 } });
    }
    try std.testing.expectEqual(@as(usize, 2), tr.tracks.items.len);
    try std.testing.expect(tr.tracks.items[0].id != tr.tracks.items[1].id);
    try std.testing.expect(tr.next_id == 3);
    try std.testing.expect(tr.tracks.items[0].confirmed());
}
