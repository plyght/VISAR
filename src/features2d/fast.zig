//! FAST corner detection (Rosten & Drummond), matching OpenCV's
//! FastFeatureDetector keypoint for keypoint: the same circles, the same
//! pre-test for the 7-12 and 5-8 patterns, the same corner score (the
//! largest threshold at which the pixel is still a corner, as OpenCV's SIMD
//! cornerScore computes it) and the same 3x3 non-maximum suppression.
//! Keypoints come out in row-major order, as OpenCV returns them.
//!
//! The segment test runs on whole vectors of pixels: per circle position
//! one "brighter" and one "darker" mask, then a run of K+1 consecutive set
//! masks around the circle is found by doubling (runs of 2, 4, 8, ...).

const std = @import("std");
const core = @import("../core/core.zig");
const kp_mod = @import("keypoint.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const KeyPoint = kp_mod.KeyPoint;
pub const Error = core.Error || Allocator.Error;

/// OpenCV's FastFeatureDetector::DetectorType values.
pub const Type = enum(u8) {
    type_5_8 = 0,
    type_7_12 = 1,
    type_9_16 = 2,

    pub fn points(t: Type) u8 {
        return switch (t) {
            .type_5_8 => 8,
            .type_7_12 => 12,
            .type_9_16 => 16,
        };
    }
};

pub const Options = struct {
    threshold: i32 = 10,
    nonmax_suppression: bool = true,
    type: Type = .type_9_16,
};

/// Circle offsets (dx, dy), in OpenCV's order (makeOffsets).
fn circle(comptime P: u8) [P][2]i8 {
    return switch (P) {
        16 => .{ .{ 0, 3 }, .{ 1, 3 }, .{ 2, 2 }, .{ 3, 1 }, .{ 3, 0 }, .{ 3, -1 }, .{ 2, -2 }, .{ 1, -3 }, .{ 0, -3 }, .{ -1, -3 }, .{ -2, -2 }, .{ -3, -1 }, .{ -3, 0 }, .{ -3, 1 }, .{ -2, 2 }, .{ -1, 3 } },
        12 => .{ .{ 0, 2 }, .{ 1, 2 }, .{ 2, 1 }, .{ 2, 0 }, .{ 2, -1 }, .{ 1, -2 }, .{ 0, -2 }, .{ -1, -2 }, .{ -2, -1 }, .{ -2, 0 }, .{ -2, 1 }, .{ -1, 2 } },
        8 => .{ .{ 0, 1 }, .{ 1, 1 }, .{ 1, 0 }, .{ 1, -1 }, .{ 0, -1 }, .{ -1, -1 }, .{ -1, 0 }, .{ -1, 1 } },
        else => @compileError("FAST circles have 8, 12 or 16 points"),
    };
}

/// Byte offsets of the circle points from the pixel 3 rows up and 3
/// columns left of the centre (so they are all non-negative).
fn offsets(comptime P: u8, stride: usize) [P]usize {
    var o: [P]usize = undefined;
    inline for (circle(P), 0..) |c, k| {
        o[k] = @as(usize, @intCast(@as(i32, c[1]) + 3)) * stride + @as(usize, @intCast(@as(i32, c[0]) + 3));
    }
    return o;
}

const L = @min(64, @max(16, std.simd.suggestVectorLength(u8) orelse 16));
const VU = @Vector(L, u8);
const Mask = @Int(.unsigned, L);

inline fn sel(b: @Vector(L, bool)) VU {
    return @select(u8, b, @as(VU, @splat(0xff)), @as(VU, @splat(0)));
}

/// For every start around the circle, the minimum of `len` consecutive
/// values, and the largest of those minima (runs built by doubling).
inline fn runMinMax(comptime P: u8, comptime len: u8, m: [P]VU) VU {
    var acc: [P]VU = @splat(@as(VU, @splat(0xff)));
    var pow = m; // pow[k] = min of m[k .. k + span)
    comptime var span: u8 = 1;
    comptime var pos: u8 = 0;
    inline while (span <= len) : (span *= 2) {
        if (len & span != 0) {
            inline for (0..P) |k| acc[k] = @min(acc[k], pow[(k + pos) % P]);
            pos += span;
        }
        if (span * 2 <= len) {
            var next: [P]VU = undefined;
            inline for (0..P) |k| next[k] = @min(pow[k], pow[(k + span) % P]);
            pow = next;
        }
    }
    var r: VU = @splat(0);
    inline for (0..P) |k| r = @max(r, acc[k]);
    return r;
}

/// OpenCV's pre-test (written for 16 points, applied to all patterns with
/// pixel[k] = pixel[k mod P]): every antipodal pair must have a point
/// beyond the threshold on the same side. Here on differences: the
/// smallest over pairs of the larger difference of the pair.
inline fn preTest(comptime P: u8, m: [P]VU) VU {
    var r: VU = @splat(0xff);
    inline for (0..8) |k| r = @min(r, @max(m[k % P], m[(k + 8) % P]));
    return r;
}

/// Corner mask of L pixels starting at `base` + 3 rows + 3 columns, and
/// (when `scores` is given) each corner's score. With b = circle -| centre
/// and d = centre -| circle, a pixel is a corner when some arc of K+1
/// points has min(b) > t or min(d) > t, and its score is the largest such
/// minimum less one: OpenCV's cornerScore (whose extra arcs for the 7-12
/// circle cover the same points as its first arc).
inline fn segmentTest(comptime P: u8, base: [*]const u8, stride: usize, o: *const [P]usize, t: u8, scores: ?*[L]u8) Mask {
    const c: VU = base[3 * stride + 3 ..][0..L].*;
    const tv: VU = @splat(t);
    if (P == 16) {
        // Any run of 9 out of 16 covers two neighbouring points of 0/4/8/12.
        const hi = c +| tv;
        const lo = c -| tv;
        var q: [4]VU = undefined;
        inline for (0..4) |i| q[i] = base[o[4 * i]..][0..L].*;
        var b: [4]VU = undefined;
        var d: [4]VU = undefined;
        inline for (0..4) |i| {
            b[i] = sel(q[i] > hi);
            d[i] = sel(q[i] < lo);
        }
        const any = (b[0] & b[1]) | (b[1] & b[2]) | (b[2] & b[3]) | (b[3] & b[0]) |
            (d[0] & d[1]) | (d[1] & d[2]) | (d[2] & d[3]) | (d[3] & d[0]);
        if (@reduce(.Or, any) == 0) return 0;
    }
    var bd: [P]VU = undefined;
    var dd: [P]VU = undefined;
    inline for (0..P) |k| {
        const x: VU = base[o[k]..][0..L].*;
        bd[k] = x -| c;
        dd[k] = c -| x;
    }
    const K = P / 2;
    const rb = runMinMax(P, K + 1, bd);
    const rd = runMinMax(P, K + 1, dd);
    const corner = if (P == 16)
        @max(rb, rd) > tv
    else
        (@min(rb, preTest(P, bd)) > tv) | (@min(rd, preTest(P, dd)) > tv);
    const m: Mask = @bitCast(corner);
    if (m != 0) if (scores) |s| {
        s.* = @max(rb, rd) -% @as(VU, @splat(1));
    };
    return m;
}

/// The scalar segment test, line for line OpenCV's FAST_t (used for the
/// columns a vector does not cover).
fn isCornerScalar(comptime P: u8, base: [*]const u8, stride: usize, o: *const [P]usize, threshold: i32) bool {
    const K = P / 2;
    const N = P + K + 1;
    const v: i32 = base[3 * stride + 3];
    const S = struct {
        fn tab(d: i32, t: i32) u8 {
            return if (d < -t) 1 else if (d > t) 2 else 0;
        }
    };
    var t16: [16]u8 = undefined;
    inline for (0..16) |k| t16[k] = S.tab(@as(i32, base[o[k % P]]) - v, threshold);
    var d: u8 = t16[0] | t16[8];
    if (d == 0) return false;
    d &= t16[2] | t16[10];
    d &= t16[4] | t16[12];
    d &= t16[6] | t16[14];
    if (d == 0) return false;
    d &= t16[1] | t16[9];
    d &= t16[3] | t16[11];
    d &= t16[5] | t16[13];
    d &= t16[7] | t16[15];
    if (d & 1 != 0) {
        const vt = v - threshold;
        var count: u32 = 0;
        for (0..N) |k| {
            if (@as(i32, base[o[k % P]]) < vt) {
                count += 1;
                if (count > K) return true;
            } else count = 0;
        }
    }
    if (d & 2 != 0) {
        const vt = v + threshold;
        var count: u32 = 0;
        for (0..N) |k| {
            if (@as(i32, base[o[k % P]]) > vt) {
                count += 1;
                if (count > K) return true;
            } else count = 0;
        }
    }
    return false;
}

/// OpenCV's cornerScore (its SIMD form, used on x86 and ARM alike): over
/// arcs of K+1 consecutive differences d = centre - circle, the largest of
/// min(arc) and -max(arc), minus one. The 7-12 pattern reproduces OpenCV's
/// padding of d (entries 19..22 repeat d[0..3]).
pub fn cornerScore(comptime P: u8, base: [*]const u8, stride: usize, o: *const [P]usize) i32 {
    const K = P / 2;
    const N = P + K + 1;
    const v: i16 = base[3 * stride + 3];
    var d: [32]i16 = undefined;
    inline for (0..32) |k| {
        const idx = if (P == 12 and k >= N) (k - N) % P else k % P;
        d[k] = v - @as(i16, base[o[idx]]);
    }
    const V16 = @Vector(16, i16);
    var a: V16 = d[1..17].*;
    var b: V16 = a;
    inline for (2..K + 1) |j| {
        const x: V16 = d[j..][0..16].*;
        a = @min(a, x);
        b = @max(b, x);
    }
    const x0: V16 = d[0..16].*;
    const x1: V16 = d[K + 1 ..][0..16].*;
    const q0 = @max(@min(a, x0), @min(a, x1));
    const q1 = @min(@max(b, x0), @max(b, x1));
    const s = @max(q0, -q1);
    // P = 8 has only 8 starting points; the other lanes repeat them.
    return @as(i32, @reduce(.Max, s)) - 1;
}

const Band = struct {
    img: Image(u8),
    opt: Options,
    out: []std.ArrayList(KeyPoint),
    gpa: Allocator,
    rows_per_band: usize,
    failed: *std.atomic.Value(bool),
};

/// FAST corners of a one-channel u8 image. The caller owns the result.
pub fn detect(exec: Exec, gpa: Allocator, img: Image(u8), opt: Options) Error![]KeyPoint {
    if (img.channels != 1) return error.UnsupportedChannels;
    var o = opt;
    o.threshold = std.math.clamp(o.threshold, 0, 255);
    const rows: usize = img.height;
    if (rows < 7 or img.width < 7) return gpa.alloc(KeyPoint, 0);
    // Keypoint rows 3 .. rows - 4, split into bands.
    const first: usize = 3;
    const last: usize = rows - 3; // exclusive
    const n = last - first;
    const bands = @max(1, @min(n / 32, exec.threads() * 4));
    const per = (n + bands - 1) / bands;
    const out = try gpa.alloc(std.ArrayList(KeyPoint), bands);
    defer gpa.free(out);
    @memset(out, .empty);
    defer for (out) |*l| l.deinit(gpa);
    var failed = std.atomic.Value(bool).init(false);
    const ctx: Band = .{ .img = img, .opt = o, .out = out, .gpa = gpa, .rows_per_band = per, .failed = &failed };
    exec.parallelFor(bands, 1, ctx, struct {
        fn run(c: Band, b0: usize, b1: usize) void {
            for (b0..b1) |b| {
                const y0 = first + b * c.rows_per_band;
                const y1 = @min(c.img.height - 3, y0 + c.rows_per_band);
                if (y0 >= y1) continue;
                band(c, b, y0, y1) catch c.failed.store(true, .monotonic);
            }
        }
    }.run);
    if (failed.load(.monotonic)) return error.OutOfMemory;
    var total: usize = 0;
    for (out) |l| total += l.items.len;
    const res = try gpa.alloc(KeyPoint, total);
    var i: usize = 0;
    for (out) |l| {
        @memcpy(res[i..][0..l.items.len], l.items);
        i += l.items.len;
    }
    return res;
}

fn band(c: Band, b: usize, y0: usize, y1: usize) Allocator.Error!void {
    return switch (c.opt.type) {
        .type_9_16 => bandP(16, c, b, y0, y1),
        .type_7_12 => bandP(12, c, b, y0, y1),
        .type_5_8 => bandP(8, c, b, y0, y1),
    };
}

/// Detects rows [y0, y1). With non-maximum suppression the score rows
/// y0 - 1 and y1 are computed too (rows outside 3 .. h - 4 score 0).
fn bandP(comptime P: u8, c: Band, b: usize, y0: usize, y1: usize) Allocator.Error!void {
    const img = c.img;
    const w: usize = img.width;
    const h: usize = img.height;
    const gpa = c.gpa;
    const list = &c.out[b];
    const o = offsets(P, img.stride);
    const t: u8 = @intCast(c.opt.threshold);
    const nms = c.opt.nonmax_suppression;

    // Ring of three score rows and their corner columns.
    const buf = try gpa.alloc(u8, 3 * w);
    defer gpa.free(buf);
    const pos = try gpa.alloc(u32, 3 * w);
    defer gpa.free(pos);
    var cnt: [3]usize = .{ 0, 0, 0 };
    @memset(buf, 0);

    const ys = if (nms) y0 - 1 else y0;
    const ye = if (nms) y1 + 1 else y1;
    var y = ys;
    while (y < ye) : (y += 1) {
        const slot = y % 3;
        const curr = buf[slot * w ..][0..w];
        const cp = pos[slot * w ..][0..w];
        @memset(curr, 0);
        var nc: usize = 0;
        if (y >= 3 and y < h - 3) {
            const rowbase = img.data + (y - 3) * img.stride;
            const xe = w - 3;
            var x: usize = 3;
            while (x + L <= xe) : (x += L) {
                var sc: [L]u8 = undefined;
                var m = segmentTest(P, rowbase + (x - 3), img.stride, &o, t, if (nms) &sc else null);
                while (m != 0) {
                    const lane = @ctz(m);
                    m &= m - 1;
                    const xx = x + lane;
                    cp[nc] = @intCast(xx);
                    nc += 1;
                    if (nms) curr[xx] = sc[lane];
                }
            }
            if (x < xe) {
                if (xe - 3 >= L) {
                    // One more vector ending at the last column; skip the
                    // lanes already tested.
                    const xs = xe - L;
                    var sc: [L]u8 = undefined;
                    var m = segmentTest(P, rowbase + (xs - 3), img.stride, &o, t, if (nms) &sc else null);
                    m >>= @intCast(x - xs);
                    m <<= @intCast(x - xs);
                    while (m != 0) {
                        const lane = @ctz(m);
                        m &= m - 1;
                        const xx = xs + lane;
                        cp[nc] = @intCast(xx);
                        nc += 1;
                        if (nms) curr[xx] = sc[lane];
                    }
                } else {
                    while (x < xe) : (x += 1) {
                        if (isCornerScalar(P, rowbase + (x - 3), img.stride, &o, c.opt.threshold)) {
                            cp[nc] = @intCast(x);
                            nc += 1;
                            if (nms) curr[x] = @intCast(cornerScore(P, rowbase + (x - 3), img.stride, &o));
                        }
                    }
                }
            }
        }
        cnt[slot] = nc;
        if (!nms) {
            try list.ensureUnusedCapacity(gpa, nc);
            for (cp[0..nc]) |xx| list.appendAssumeCapacity(.{ .x = @floatFromInt(xx), .y = @floatFromInt(y), .size = 7, .response = 0 });
            continue;
        }
        // Suppress row y - 1 against rows y - 2, y - 1 and y.
        if (y == ys) continue;
        const r = y - 1;
        if (r < y0 or r >= y1) continue;
        const prev = buf[(r % 3) * w ..][0..w];
        const pprev = buf[((r + 2) % 3) * w ..][0..w];
        const pc = pos[(r % 3) * w ..][0..cnt[r % 3]];
        try list.ensureUnusedCapacity(gpa, pc.len);
        for (pc) |xx| {
            const j: usize = xx;
            const s = prev[j];
            if (s > prev[j + 1] and s > prev[j - 1] and
                s > pprev[j - 1] and s > pprev[j] and s > pprev[j + 1] and
                s > curr[j - 1] and s > curr[j] and s > curr[j + 1])
            {
                list.appendAssumeCapacity(.{ .x = @floatFromInt(j), .y = @floatFromInt(r), .size = 7, .response = @floatFromInt(s) });
            }
        }
    }
}

// ------------------------------------------------------------------ tests

const testing = std.testing;

/// The textbook definition: a cyclic run of K+1 circle points all brighter
/// than v + t or all darker than v - t.
fn naiveCorner(comptime P: u8, img: Image(u8), x: usize, y: usize, t: i32) bool {
    const v: i32 = img.at(x, y, 0);
    for ([_]i32{ 1, -1 }) |sign| {
        var best: usize = 0;
        var run: usize = 0;
        for (0..2 * P) |k| {
            const cc = circle(P)[k % P];
            const p: i32 = img.at(@intCast(@as(i64, @intCast(x)) + cc[0]), @intCast(@as(i64, @intCast(y)) + cc[1]), 0);
            if ((p - v) * sign > t) {
                run += 1;
                best = @max(best, run);
            } else run = 0;
        }
        if (best > P / 2) return true;
    }
    return false;
}

fn testImage(gpa: Allocator, w: u32, h: u32, seed: u64) !Image(u8) {
    var img = try Image(u8).init(gpa, w, h, 1);
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    for (0..h) |y| for (0..w) |x| {
        // Blocky texture so corners exist at every threshold.
        const bx = x / 3;
        const by = y / 3;
        const v: u32 = @truncate((bx *% 2654435761) ^ (by *% 40503) ^ (seed *% 97));
        img.set(x, y, 0, @truncate((v >> 7) +% r.int(u8) / 16));
    };
    return img;
}

test "FAST-9/16 segment test matches the definition" {
    const gpa = testing.allocator;
    var img = try testImage(gpa, 101, 37, 3);
    defer img.deinit(gpa);
    for ([_]i32{ 0, 10, 40 }) |t| {
        const kps = try detect(.serial, gpa, img, .{ .threshold = t, .nonmax_suppression = false });
        defer gpa.free(kps);
        var i: usize = 0;
        for (3..img.height - 3) |y| for (3..img.width - 3) |x| {
            if (naiveCorner(16, img, x, y, t)) {
                try testing.expect(i < kps.len);
                try testing.expectEqual(@as(f32, @floatFromInt(x)), kps[i].x);
                try testing.expectEqual(@as(f32, @floatFromInt(y)), kps[i].y);
                i += 1;
            }
        };
        try testing.expectEqual(kps.len, i);
    }
}

test "FAST vector and scalar paths agree for every pattern" {
    const gpa = testing.allocator;
    var img = try testImage(gpa, 77, 29, 9);
    defer img.deinit(gpa);
    inline for (.{ 8, 12, 16 }) |P| {
        const o = offsets(P, img.stride);
        for ([_]u8{ 0, 5, 25 }) |t| {
            for (3..img.height - 3) |y| {
                var x: usize = 3;
                while (x + L <= img.width - 3) : (x += L) {
                    const base = img.data + (y - 3) * img.stride + (x - 3);
                    var sc: [L]u8 = undefined;
                    const m = segmentTest(P, base, img.stride, &o, t, &sc);
                    for (0..L) |lane| {
                        const s = isCornerScalar(P, base + lane, img.stride, &o, t);
                        try testing.expectEqual(s, (m >> @intCast(lane)) & 1 == 1);
                        if (s) {
                            // The score is the largest threshold that still
                            // detects the corner (for the 16-point circle).
                            const scs = cornerScore(P, base + lane, img.stride, &o);
                            try testing.expectEqual(scs, sc[lane]);
                            try testing.expect(scs >= t);
                            if (P == 16) {
                                try testing.expect(isCornerScalar(P, base + lane, img.stride, &o, scs));
                                try testing.expect(!isCornerScalar(P, base + lane, img.stride, &o, scs + 1));
                            }
                        }
                    }
                }
            }
        }
    }
}

test "FAST threads give the serial result" {
    if (!core.pool.threads_supported) return;
    const gpa = testing.allocator;
    var img = try testImage(gpa, 300, 211, 5);
    defer img.deinit(gpa);
    var pool = core.Pool.init(gpa, std.Io.Threaded.global_single_threaded.io(), 4);
    defer pool.deinit();
    for ([_]Type{ .type_9_16, .type_7_12, .type_5_8 }) |ty| {
        const a = try detect(.serial, gpa, img, .{ .threshold = 12, .type = ty });
        defer gpa.free(a);
        const b = try detect(.on(&pool), gpa, img, .{ .threshold = 12, .type = ty });
        defer gpa.free(b);
        try testing.expect(a.len > 0);
        try testing.expectEqualSlices(KeyPoint, a, b);
    }
}
