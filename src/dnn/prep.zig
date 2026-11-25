//! Frame -> network input: BGR or BGRA u8 pixels straight into a planar
//! f32 tensor (one plane per colour, zero-padded on the right and bottom),
//! optionally shrunk by integer factors on the way, in one pass with no
//! intermediate image.
//!
//! The shrink is INTER_AREA with integer factors (OpenCV's resizeAreaFast
//! for factors other than 2x2, as imgproc's `resize` computes it): the
//! block sum times the f32 reciprocal of the block area, rounded half to
//! even to u8. The u8 value is then the float written, so the tensor is
//! exactly what resize(area) + cvtColor(BGRA2BGR) + blobFromImage give.

const std = @import("std");
const core = @import("../core/core.zig");
const simd = core.simd;
const Exec = core.Exec;
const Image = core.Image;

/// The destination: `planes` holds three `pw` x `ph` planes; the image
/// fills the top-left `w` x `h` of each, the rest is zeroed.
pub const Target = struct {
    planes: []f32,
    pw: usize,
    ph: usize,
    w: u32,
    h: u32,
};

/// The largest integer factor (per axis) and block area the fused shrink
/// takes: the block sums must fit in u16 and a segment's column sums on
/// the stack.
pub const max_factor = 16;
pub const max_area = 257;

/// The integer shrink factors from `src` to `w` x `h`, when `fill` can
/// take them (both exact, within `max_factor` and `max_area`).
pub fn factors(src: Image(u8), w: u32, h: u32) ?[2]u32 {
    return factorsOf(src.width, src.height, w, h);
}

/// `factors` for a `sw` x `sh` source.
pub fn factorsOf(sw: u32, sh: u32, w: u32, h: u32) ?[2]u32 {
    if (w == 0 or h == 0 or sw < w or sh < h) return null;
    const fx = sw / w;
    const fy = sh / h;
    if (fx * w != sw or fy * h != sh) return null;
    if (fx > max_factor or fy > max_factor or fx * fy > max_area) return null;
    return .{ fx, fy };
}

/// Writes `src` (3 or 4 channels; B, G, R first), shrunk by `f` (from
/// `factors`), into `t`.
pub fn fill(exec: Exec, src: Image(u8), t: Target, f: [2]u32) void {
    std.debug.assert(src.channels == 3 or src.channels == 4);
    std.debug.assert(src.width == f[0] * t.w and src.height == f[1] * t.h);
    const Ctx = struct { src: Image(u8), t: Target, fx: u32, fy: u32 };
    // Each output row reads `fy` source rows: a few rows per chunk.
    exec.parallelFor(t.ph, 8, Ctx{ .src = src, .t = t, .fx = f[0], .fy = f[1] }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const plane = c.t.pw * c.t.ph;
            for (y0..y1) |y| {
                const rows = [3][]f32{
                    c.t.planes[y * c.t.pw ..][0..c.t.pw],
                    c.t.planes[plane + y * c.t.pw ..][0..c.t.pw],
                    c.t.planes[2 * plane + y * c.t.pw ..][0..c.t.pw],
                };
                if (y >= c.t.h) {
                    for (rows) |r| @memset(r, 0);
                    continue;
                }
                const w = c.t.w;
                switch (c.src.channels) {
                    inline 3, 4 => |cn| if (c.fx == 1 and c.fy == 1)
                        copyRow(cn, c.src.row(y), rows, w)
                    else
                        shrinkRow(cn, c.src, y, c.fx, c.fy, rows, w),
                    else => unreachable,
                }
                for (rows) |r| @memset(r[w..], 0);
            }
        }
    }.run);
}

/// One row at the same size: deinterleave B, G, R and widen, 16 pixels at
/// a time.
fn copyRow(comptime cn: usize, src: []const u8, rows: [3][]f32, w: usize) void {
    var x: usize = 0;
    while (x + 16 <= w) : (x += 16) {
        const v: @Vector(16 * cn, u8) = src[x * cn ..][0 .. 16 * cn].*;
        inline for (0..3) |ch| {
            const idx = comptime blk: {
                var m: [16]i32 = undefined;
                for (&m, 0..) |*q, i| q.* = @intCast(i * cn + ch);
                break :blk m;
            };
            const p: @Vector(16, u8) = @shuffle(u8, v, undefined, idx);
            rows[ch][x..][0..16].* = @as(@Vector(16, f32), @floatFromInt(p));
        }
    }
    for (x..w) |xx| {
        inline for (0..3) |ch| rows[ch][xx] = @floatFromInt(src[xx * cn + ch]);
    }
}

/// Output pixels per segment: the segment's column sums (at most
/// seg * max_factor pixels) stay on the stack, in L1.
const seg = 32;
/// Source bytes summed per vector step.
const vn = 32;

/// One output row of the integer-factor shrink, segment by segment:
/// column sums over the `fy` source rows (u16, the bandwidth-bound part),
/// then each output pixel's sum over `fx` columns (all four lanes of a
/// pixel at once), then the scale, rounding and deinterleave into the
/// three planes, 8 pixels at a time.
fn shrinkRow(comptime cn: usize, src: Image(u8), y: usize, fx: u32, fy: u32, rows: [3][]f32, w: usize) void {
    const U = @Vector(vn, u16);
    var vs: [seg * max_factor * cn + vn]u16 = undefined;
    var hs: [seg * 4]u16 align(16) = undefined;
    var srows: [max_factor][*]const u8 = undefined;
    for (0..fy) |k| srows[k] = src.row(y * fy + k).ptr;
    const scale: f32 = 1.0 / @as(f32, @floatFromInt(fx * fy));
    const sv: @Vector(8, f32) = @splat(scale);
    const top: @Vector(8, f32) = @splat(255);
    var x0: usize = 0;
    while (x0 < w) : (x0 += seg) {
        const n = @min(seg, w - x0);
        const len = n * fx * cn;
        const base = x0 * fx * cn;
        // Column sums.
        var i: usize = 0;
        while (i + vn <= len) : (i += vn) {
            var acc: U = @splat(0);
            for (srows[0..fy]) |r| acc += @as(U, @as(@Vector(vn, u8), r[base + i ..][0..vn].*));
            vs[i..][0..vn].* = acc;
        }
        while (i < len) : (i += 1) {
            var acc: u16 = 0;
            for (srows[0..fy]) |r| acc += r[base + i];
            vs[i] = acc;
        }
        // The last BGR pixel's 4-lane read below takes one lane past it.
        vs[len] = 0;
        // Block sums, one pixel (4 lanes) at a time.
        for (0..n) |p| {
            var acc: @Vector(4, u16) = @splat(0);
            const o = p * fx * cn;
            for (0..fx) |j| acc += @as(@Vector(4, u16), vs[o + j * cn ..][0..4].*);
            hs[p * 4 ..][0..4].* = acc;
        }
        // Scale, round half to even, deinterleave.
        var p: usize = 0;
        while (p + 8 <= n) : (p += 8) {
            const v: @Vector(32, u16) = hs[p * 4 ..][0..32].*;
            inline for (0..3) |ch| {
                const idx = comptime blk: {
                    var m: [8]i32 = undefined;
                    for (&m, 0..) |*q, k| q.* = @intCast(k * 4 + ch);
                    break :blk m;
                };
                const s: @Vector(8, u16) = @shuffle(u16, v, undefined, idx);
                const fv = simd.toF32(8, s) * sv;
                rows[ch][x0 + p ..][0..8].* = @min(simd.rintSmall(fv), top);
            }
        }
        while (p < n) : (p += 1) {
            inline for (0..3) |ch| {
                const fv = @as(f32, @floatFromInt(hs[p * 4 + ch])) * scale;
                rows[ch][x0 + p] = @min((fv + 12582912.0) - 12582912.0, 255);
            }
        }
    }
}

// ------------------------------------------------------------------ tests

const testing = std.testing;
const imgproc = @import("../imgproc/imgproc.zig");

test "fused shrink equals resize(area) + cvtColor + widening" {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    // (factor x, factor y, out w, out h, channels)
    const cases = [_][5]u32{ .{ 6, 6, 37, 5, 4 }, .{ 4, 4, 40, 3, 3 }, .{ 3, 2, 9, 4, 4 }, .{ 1, 1, 21, 3, 4 }, .{ 1, 1, 35, 2, 3 }, .{ 16, 16, 3, 2, 4 }, .{ 5, 7, 33, 2, 3 } };
    for (cases) |cs| {
        const w = cs[2];
        const h = cs[3];
        const cn: u16 = @intCast(cs[4]);
        var src = try Image(u8).init(gpa, w * cs[0], h * cs[1], cn);
        defer src.deinit(gpa);
        for (src.slice()) |*v| v.* = rnd.int(u8);
        var small = try Image(u8).init(gpa, w, h, cn);
        defer small.deinit(gpa);
        try imgproc.resize(Exec.serial, gpa, src, small, .area);
        const pw = w + 5;
        const ph = h + 2;
        const planes = try gpa.alloc(f32, 3 * pw * ph);
        defer gpa.free(planes);
        @memset(planes, -1);
        const f = factors(src, w, h).?;
        fill(Exec.serial, src, .{ .planes = planes, .pw = pw, .ph = ph, .w = w, .h = h }, f);
        for (0..3) |ch| for (0..ph) |y| for (0..pw) |x| {
            const want: f32 = if (x < w and y < h) @floatFromInt(small.at(x, y, ch)) else 0;
            try testing.expectEqual(want, planes[ch * pw * ph + y * pw + x]);
        };
    }
}

test "factors" {
    var buf: [1]u8 = undefined;
    const img = Image(u8){ .width = 3840, .height = 2160, .channels = 4, .stride = 3840 * 4, .data = &buf };
    try testing.expectEqual([2]u32{ 6, 6 }, factors(img, 640, 360).?);
    try testing.expectEqual([2]u32{ 4, 4 }, factors(img, 960, 540).?);
    try testing.expectEqual(@as(?[2]u32, null), factors(img, 700, 360));
    try testing.expectEqual(@as(?[2]u32, null), factors(img, 120, 90));
}
