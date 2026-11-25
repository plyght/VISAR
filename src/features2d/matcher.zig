//! Brute-force descriptor matching (OpenCV's BFMatcher): match, knnMatch
//! and cross-checked matching under the Hamming norms (u8 descriptors) and
//! L1 / L2 (u8 or f32 descriptors), plus Lowe's ratio test.
//!
//! Descriptors are one-channel images, one descriptor per row. Candidates
//! are ranked exactly as OpenCV's batchDistance ranks them: by distance,
//! with ties going to the lower train index. Hamming distances are exact;
//! float L1 / L2 sums may differ from OpenCV's in the last bits (the
//! summation order differs), which can reorder candidates whose distances
//! tie to within an ulp.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const kp_mod = @import("keypoint.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const DMatch = kp_mod.DMatch;
pub const Error = core.Error || Allocator.Error || error{ InvalidArgument, Unsupported };

/// cv::NormTypes values.
pub const Norm = enum(u8) {
    l1 = 2,
    l2 = 4,
    l2sqr = 5,
    hamming = 6,
    hamming2 = 7,
};

/// Hamming distance between two byte strings.
pub fn hamming(a: []const u8, b: []const u8) u32 {
    var d: u32 = 0;
    var i: usize = 0;
    const V = @Vector(4, u64);
    while (i + 32 <= a.len) : (i += 32) {
        const x: V = @bitCast(a[i..][0..32].*);
        const y: V = @bitCast(b[i..][0..32].*);
        d += @reduce(.Add, @as(@Vector(4, u32), @popCount(x ^ y)));
    }
    while (i + 8 <= a.len) : (i += 8) {
        d += @popCount(std.mem.readInt(u64, a[i..][0..8], .little) ^ std.mem.readInt(u64, b[i..][0..8], .little));
    }
    while (i < a.len) : (i += 1) d += @popCount(a[i] ^ b[i]);
    return d;
}

/// NORM_HAMMING2: the number of 2-bit cells that differ.
pub fn hamming2(a: []const u8, b: []const u8) u32 {
    var d: u32 = 0;
    var i: usize = 0;
    while (i + 8 <= a.len) : (i += 8) {
        const v = std.mem.readInt(u64, a[i..][0..8], .little) ^ std.mem.readInt(u64, b[i..][0..8], .little);
        d += @popCount((v | (v >> 1)) & 0x5555555555555555);
    }
    while (i < a.len) : (i += 1) {
        const v = a[i] ^ b[i];
        d += @popCount((v | (v >> 1)) & 0x55);
    }
    return d;
}

fn distU8(norm: Norm, a: []const u8, b: []const u8) f32 {
    switch (norm) {
        .hamming => return @floatFromInt(hamming(a, b)),
        .hamming2 => return @floatFromInt(hamming2(a, b)),
        .l1 => {
            var s: u32 = 0;
            for (a, b) |x, y| s += if (x > y) x - y else y - x;
            return @floatFromInt(s);
        },
        .l2, .l2sqr => {
            var s: u32 = 0;
            for (a, b) |x, y| {
                const d: i32 = @as(i32, x) - y;
                s += @intCast(d * d);
            }
            const f: f32 = @floatFromInt(s);
            return if (norm == .l2) @sqrt(f) else f;
        },
    }
}

fn distF32(norm: Norm, a: []const f32, b: []const f32) f32 {
    const V = 8;
    var acc: @Vector(V, f32) = @splat(0);
    var i: usize = 0;
    const l1 = norm == .l1;
    while (i + V <= a.len) : (i += V) {
        const d = @as(@Vector(V, f32), a[i..][0..V].*) - @as(@Vector(V, f32), b[i..][0..V].*);
        acc += if (l1) @abs(d) else d * d;
    }
    var s = @reduce(.Add, acc);
    while (i < a.len) : (i += 1) {
        const d = a[i] - b[i];
        s += if (l1) @abs(d) else d * d;
    }
    return if (norm == .l2) @sqrt(s) else s;
}

const hw_fma = switch (builtin.cpu.arch) {
    .aarch64 => true,
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .fma),
    else => false,
};

/// distF32 of one query against train rows [j0, j0 + out.len), eight rows
/// at a time so each query chunk is loaded once per eight rows.
fn f32Block(norm: Norm, q: []const f32, train: Image(f32), j0: usize, out: []f32) void {
    switch (norm) {
        .l1 => f32BlockT(true, norm, q, train, j0, out),
        else => f32BlockT(false, norm, q, train, j0, out),
    }
}

fn f32BlockT(comptime l1: bool, norm: Norm, q: []const f32, train: Image(f32), j0: usize, out: []f32) void {
    const V = 8;
    const R = 8;
    const VF = @Vector(V, f32);
    const n = q.len;
    var j: usize = 0;
    while (j + R <= out.len) : (j += R) {
        var r: [R][*]const f32 = undefined;
        inline for (0..R) |k| r[k] = train.row(j0 + j + k).ptr;
        var acc: [R]VF = @splat(@as(VF, @splat(0)));
        var i: usize = 0;
        while (i + V <= n) : (i += V) {
            const qv: VF = q[i..][0..V].*;
            inline for (0..R) |k| {
                const dv = qv - @as(VF, r[k][i..][0..V].*);
                if (l1) {
                    acc[k] += @abs(dv);
                } else if (hw_fma) {
                    acc[k] = @mulAdd(VF, dv, dv, acc[k]);
                } else acc[k] += dv * dv;
            }
        }
        inline for (0..R) |k| {
            var s = @reduce(.Add, acc[k]);
            var ii = i;
            while (ii < n) : (ii += 1) {
                const dd = q[ii] - r[k][ii];
                s += if (l1) @abs(dd) else dd * dd;
            }
            out[j + k] = if (norm == .l2) @sqrt(s) else s;
        }
    }
    while (j < out.len) : (j += 1) out[j] = distF32(norm, q, train.row(j0 + j));
}

fn check(comptime T: type, query: Image(T), train: Image(T), norm: Norm) Error!void {
    if (query.channels != 1 or train.channels != 1) return error.UnsupportedChannels;
    if (query.width != train.width) return error.SizeMismatch;
    if (T == f32 and (norm == .hamming or norm == .hamming2)) return error.Unsupported;
}

/// Inserts (d, j) into the sorted top-k list (OpenCV's insertion: a new
/// candidate goes after existing ones with the same distance).
inline fn insert(dist: []f32, idx: []i32, d: f32, j: i32) void {
    const k = dist.len;
    if (!(d < dist[k - 1])) return;
    var p = k - 1;
    while (p > 0 and dist[p - 1] > d) : (p -= 1) {
        dist[p] = dist[p - 1];
        idx[p] = idx[p - 1];
    }
    dist[p] = d;
    idx[p] = j;
}

fn Ctx(comptime T: type) type {
    return struct {
        query: Image(T),
        train: Image(T),
        norm: Norm,
        k: usize,
        dist: []f32,
        idx: []i32,
    };
}

/// Byte-wise popcount plus a horizontal sum where the target has a vector
/// byte popcount (NEON cnt, AVX-512 BITALG); four 64-bit popcnts elsewhere.
const byte_popcount = switch (builtin.cpu.arch) {
    .aarch64 => true,
    .x86_64 => std.Target.x86.featureSetHas(builtin.cpu.features, .avx512bitalg),
    else => false,
};

/// Hamming distances of one 32-byte query to train rows [j0, j0 + out.len).
fn hamming32Block(q: *const [32]u8, train: Image(u8), j0: usize, out: []f32) void {
    if (byte_popcount) {
        const V = @Vector(32, u8);
        const qv: V = q.*;
        for (out, 0..) |*o, j| {
            const tv: V = train.row(j0 + j)[0..32].*;
            const c: V = @popCount(qv ^ tv);
            o.* = @floatFromInt(@reduce(.Add, @as(@Vector(32, u16), c)));
        }
        return;
    }
    const q0 = std.mem.readInt(u64, q[0..8], .little);
    const q1 = std.mem.readInt(u64, q[8..16], .little);
    const q2 = std.mem.readInt(u64, q[16..24], .little);
    const q3 = std.mem.readInt(u64, q[24..32], .little);
    for (out, 0..) |*o, j| {
        const r = train.row(j0 + j)[0..32];
        const d = @as(u32, @popCount(q0 ^ std.mem.readInt(u64, r[0..8], .little))) +
            @popCount(q1 ^ std.mem.readInt(u64, r[8..16], .little)) +
            @popCount(q2 ^ std.mem.readInt(u64, r[16..24], .little)) +
            @popCount(q3 ^ std.mem.readInt(u64, r[24..32], .little));
        o.* = @floatFromInt(d);
    }
}

fn minOf(d: []const f32) f32 {
    const V = 16;
    var acc: @Vector(V, f32) = @splat(std.math.inf(f32));
    var i: usize = 0;
    while (i + V <= d.len) : (i += V) acc = @min(acc, @as(@Vector(V, f32), d[i..][0..V].*));
    var lo = @reduce(.Min, acc);
    while (i < d.len) : (i += 1) lo = @min(lo, d[i]);
    return lo;
}

/// Best `k` train rows for each query row; dist/idx are nq * k, with
/// unused entries left at +inf / -1. Distances are computed a block of
/// train rows at a time, then merged into the sorted list.
fn topK(comptime T: type, exec: Exec, query: Image(T), train: Image(T), norm: Norm, k: usize, dist: []f32, idx: []i32) void {
    @memset(dist, std.math.inf(f32));
    @memset(idx, -1);
    exec.parallelFor(query.height, 16, Ctx(T){ .query = query, .train = train, .norm = norm, .k = k, .dist = dist, .idx = idx }, struct {
        fn run(c: Ctx(T), q0: usize, q1: usize) void {
            const n = c.train.height;
            // A block of train rows small enough to stay in cache while
            // every query of this range is compared with it.
            const row_bytes = @max(1, c.train.width * @sizeOf(T));
            const B = 256;
            const blk = std.math.clamp(32768 / row_bytes, 16, B);
            var buf: [B]f32 = undefined;
            const ham32 = T == u8 and c.norm == .hamming and c.query.width == 32;
            var j0: usize = 0;
            while (j0 < n) : (j0 += blk) {
                const m = @min(blk, n - j0);
                const d = buf[0..m];
                for (q0..q1) |q| {
                    const qd = c.query.row(q);
                    const dl = c.dist[q * c.k ..][0..c.k];
                    const il = c.idx[q * c.k ..][0..c.k];
                    if (ham32) {
                        hamming32Block(qd[0..32], c.train, j0, d);
                    } else if (T == f32) {
                        f32Block(c.norm, qd, c.train, j0, d);
                    } else for (d, 0..) |*o, j| {
                        o.* = distU8(c.norm, qd, c.train.row(j0 + j));
                    }
                    if (c.k == 1) {
                        const lo = minOf(d);
                        if (lo < dl[0]) {
                            const at = std.mem.indexOfScalar(f32, d, lo).?;
                            dl[0] = lo;
                            il[0] = @intCast(j0 + at);
                        }
                        continue;
                    }
                    for (d, 0..) |x, j| {
                        if (x < dl[c.k - 1]) insert(dl, il, x, @intCast(j0 + j));
                    }
                }
            }
        }
    }.run);
}

/// knnMatch: the `k` nearest train descriptors of every query descriptor,
/// nearest first. `out` holds query.height * k matches and `counts` the
/// number found per query (min(k, train.height)).
pub fn knnMatch(comptime T: type, exec: Exec, gpa: Allocator, query: Image(T), train: Image(T), k: usize, norm: Norm, out: []DMatch, counts: []u32) Error!void {
    try check(T, query, train, norm);
    if (k == 0) return error.InvalidArgument;
    const nq: usize = query.height;
    if (out.len < nq * k or counts.len < nq) return error.InvalidArgument;
    const dist = try gpa.alloc(f32, nq * k);
    defer gpa.free(dist);
    const idx = try gpa.alloc(i32, nq * k);
    defer gpa.free(idx);
    topK(T, exec, query, train, norm, k, dist, idx);
    for (0..nq) |q| {
        var c: u32 = 0;
        for (0..k) |j| {
            if (idx[q * k + j] < 0) break;
            out[q * k + j] = .{ .query_idx = @intCast(q), .train_idx = idx[q * k + j], .distance = dist[q * k + j] };
            c += 1;
        }
        counts[q] = c;
    }
}

/// match: the nearest train descriptor of each query descriptor. With
/// `cross_check`, only pairs that are each other's nearest neighbour are
/// kept (BFMatcher(norm, crossCheck=true)). Returns the matches (caller
/// owns), in query order.
pub fn match(comptime T: type, exec: Exec, gpa: Allocator, query: Image(T), train: Image(T), norm: Norm, cross_check: bool) Error![]DMatch {
    try check(T, query, train, norm);
    const nq: usize = query.height;
    const nt: usize = train.height;
    const fd = try gpa.alloc(f32, nq);
    defer gpa.free(fd);
    const fi = try gpa.alloc(i32, nq);
    defer gpa.free(fi);
    topK(T, exec, query, train, norm, 1, fd, fi);
    var keep: ?[]i32 = null;
    defer if (keep) |b| gpa.free(b);
    if (cross_check) {
        const bd = try gpa.alloc(f32, nt);
        defer gpa.free(bd);
        keep = try gpa.alloc(i32, nt);
        topK(T, exec, train, query, norm, 1, bd, keep.?);
    }
    var n: usize = 0;
    for (0..nq) |q| {
        if (fi[q] < 0) continue;
        if (keep) |b| if (b[@intCast(fi[q])] != q) continue;
        n += 1;
    }
    const out = try gpa.alloc(DMatch, n);
    n = 0;
    for (0..nq) |q| {
        if (fi[q] < 0) continue;
        if (keep) |b| if (b[@intCast(fi[q])] != q) continue;
        out[n] = .{ .query_idx = @intCast(q), .train_idx = fi[q], .distance = fd[q] };
        n += 1;
    }
    return out;
}

/// Lowe's ratio test over knnMatch output (k >= 2): keeps the best match
/// of each query whose distance is below `ratio` times the second best.
/// Writes the kept matches to the front of `out` and returns their count.
pub fn ratioTest(knn: []const DMatch, counts: []const u32, k: usize, ratio: f32, out: []DMatch) usize {
    var n: usize = 0;
    for (counts, 0..) |c, q| {
        if (c < 2) continue;
        const m = knn[q * k ..][0..2];
        if (m[0].distance < ratio * m[1].distance) {
            out[n] = m[0];
            n += 1;
        }
    }
    return n;
}

// ------------------------------------------------------------------ tests

const testing = std.testing;

test "hamming distances" {
    var a: [32]u8 = @splat(0);
    a[0] = 0xff;
    a[2] = 0x0f;
    a[3] = 0x01;
    const b: [32]u8 = @splat(0);
    try testing.expectEqual(@as(u32, 13), hamming(&a, &b));
    try testing.expectEqual(@as(u32, 4 + 2 + 1), hamming2(&a, &b));
}

test "knnMatch ranks like batchDistance" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const r = prng.random();
    var qd: [20 * 32]u8 = undefined;
    var td: [50 * 32]u8 = undefined;
    r.bytes(&qd);
    r.bytes(&td);
    // Duplicate train rows give ties: the lower index must win.
    @memcpy(td[40 * 32 ..][0..32], td[3 * 32 ..][0..32]);
    @memcpy(qd[0..32], td[3 * 32 ..][0..32]);
    const q = try Image(u8).fromSlice(&qd, 32, 20, 1);
    const t = try Image(u8).fromSlice(&td, 32, 50, 1);
    var out: [20 * 3]DMatch = undefined;
    var counts: [20]u32 = undefined;
    try knnMatch(u8, .serial, gpa, q, t, 3, .hamming, &out, &counts);
    try testing.expectEqual(@as(i32, 3), out[0].train_idx);
    try testing.expectEqual(@as(i32, 40), out[1].train_idx);
    try testing.expectEqual(@as(f32, 0), out[1].distance);
    for (0..20) |i| {
        try testing.expectEqual(@as(u32, 3), counts[i]);
        // Brute force check of the best.
        var best: u32 = 1000;
        for (0..50) |j| best = @min(best, hamming(qd[i * 32 ..][0..32], td[j * 32 ..][0..32]));
        try testing.expectEqual(@as(f32, @floatFromInt(best)), out[i * 3].distance);
        try testing.expect(out[i * 3].distance <= out[i * 3 + 1].distance);
    }
    const m = try match(u8, .serial, gpa, q, t, .hamming, false);
    defer gpa.free(m);
    try testing.expectEqual(@as(usize, 20), m.len);
    for (m, 0..) |x, i| try testing.expectEqual(out[i * 3].train_idx, x.train_idx);
    const cc = try match(u8, .serial, gpa, q, t, .hamming, true);
    defer gpa.free(cc);
    try testing.expect(cc.len >= 1 and cc.len <= 20);
    try testing.expectEqual(@as(i32, 3), cc[0].train_idx);
}
