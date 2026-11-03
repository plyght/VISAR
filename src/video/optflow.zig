//! Sparse optical flow: pyramidal Lucas-Kanade (OpenCV's
//! calcOpticalFlowPyrLK, following lkpyramid.cpp).
//!
//! OpenCV pads every pyramid level by one window (reflect101 for images,
//! zeros for the Scharr derivatives) and differentiates whole levels;
//! VISAR keeps the levels unpadded and builds each window's samples and
//! 3x3 Scharr derivatives on demand, extrapolating exactly like that
//! padding, so it reads the same values without the full-size buffers.
//! Windows are sampled bilinearly with 14-bit fixed-point weights (the
//! image patch kept at 5 fractional bits) on 16 elements per step with
//! 16-bit multiply-adds. Sums of products are exact integers (OpenCV
//! accumulates them in f32 in an order that depends on its SIMD path), so
//! tracked positions agree with OpenCV's to about 1e-3 px, well inside the
//! termination epsilon; a point whose iteration never converges can end
//! anywhere along its chaotic path in either library.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const imgproc = @import("../imgproc/imgproc.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Point2f = imgproc.corner.Point2f;
pub const Error = core.Error || Allocator.Error || error{ InvalidArgument, Unsupported };

pub const LKOptions = struct {
    win_width: u32 = 21,
    win_height: u32 = 21,
    /// Pyramid levels above the image (0 = single level). Reduced
    /// automatically, like OpenCV, once a level would not be larger than
    /// the window.
    max_level: u32 = 3,
    /// Iterations per level (TermCriteria COUNT, clamped to 0..100).
    max_count: u32 = 30,
    /// Stop once an update moves the point less than this (TermCriteria
    /// EPS, in pixels, clamped to 0..10).
    epsilon: f64 = 0.01,
    /// OPTFLOW_USE_INITIAL_FLOW: start from `next_pts` instead of `prev_pts`.
    use_initial_flow: bool = false,
    /// OPTFLOW_LK_GET_MIN_EIGENVALS: report the spatial gradient matrix's
    /// minimum eigenvalue in `err` instead of the mean absolute difference.
    get_min_eigenvals: bool = false,
    /// Points whose gradient matrix has a smaller minimum eigenvalue (per
    /// window pixel) are lost.
    min_eig_threshold: f32 = 1e-4,
};

const V16 = @Vector(16, i16);
const V8 = @Vector(8, i32);
const VL = @Vector(8, i64);

/// A Lucas-Kanade pyramid (level 0 is a view of the image). Build one per
/// frame and reuse it for the next pair when tracking through a video.
pub const FlowPyramid = struct {
    levels: []Image(u8),
    win_width: u32,
    win_height: u32,

    /// OpenCV's buildOpticalFlowPyramid: levels 0..max_level, stopping
    /// before a level that would not be larger than the window. `img` must
    /// outlive the pyramid (level 0 is a view of it).
    pub fn build(exec: Exec, gpa: Allocator, img: Image(u8), win_width: u32, win_height: u32, max_level: u32) Error!FlowPyramid {
        if (win_width <= 2 or win_height <= 2) return error.InvalidArgument;
        if (img.channels > 4) return error.UnsupportedChannels;
        var n_levels: usize = 1;
        {
            var w: usize = img.width;
            var h: usize = img.height;
            var l: u32 = 0;
            while (l < max_level) : (l += 1) {
                w = (w + 1) / 2;
                h = (h + 1) / 2;
                if (w <= win_width or h <= win_height) break;
                n_levels += 1;
            }
        }
        const levels = try gpa.alloc(Image(u8), n_levels);
        var made: usize = 1;
        errdefer {
            for (levels[1..made]) |*l| l.deinit(gpa);
            gpa.free(levels);
        }
        levels[0] = img.view();
        while (made < n_levels) : (made += 1) {
            const prev = levels[made - 1];
            const sz = imgproc.pyramid.downSize(prev.width, prev.height);
            var next = try Image(u8).init(gpa, sz[0], sz[1], prev.channels);
            imgproc.pyrDown(exec, gpa, prev, next, .reflect101) catch |e| {
                next.deinit(gpa);
                return switch (e) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => unreachable,
                };
            };
            levels[made] = next;
        }
        return .{ .levels = levels, .win_width = win_width, .win_height = win_height };
    }

    pub fn deinit(self: *FlowPyramid, gpa: Allocator) void {
        for (self.levels[1..]) |*l| l.deinit(gpa);
        gpa.free(self.levels);
        self.* = undefined;
    }

    pub fn levelCount(self: FlowPyramid) usize {
        return self.levels.len;
    }
};

/// Lucas-Kanade on two pyramids built by FlowPyramid.build. See
/// calcOpticalFlowPyrLK.
pub fn calcOpticalFlowPyrLKPyr(exec: Exec, gpa: Allocator, prev: *const FlowPyramid, next: *const FlowPyramid, prev_pts: []const Point2f, next_pts: []Point2f, status: []u8, err: ?[]f32, o: LKOptions) Error!void {
    if (next_pts.len != prev_pts.len or status.len != prev_pts.len) return error.SizeMismatch;
    if (err) |e| if (e.len != prev_pts.len) return error.SizeMismatch;
    if (o.win_width <= 2 or o.win_height <= 2) return error.InvalidArgument;
    if (!prev.levels[0].sameShape(next.levels[0])) return error.SizeMismatch;
    const max_level = @min(@as(usize, o.max_level), prev.levels.len - 1, next.levels.len - 1);
    @memset(status, 1);
    if (prev_pts.len == 0) return;
    const eps = std.math.clamp(o.epsilon, 0, 10);
    const Ctx = struct {
        prev: *const FlowPyramid,
        next: *const FlowPyramid,
        pp: []const Point2f,
        np: []Point2f,
        st: []u8,
        err: ?[]f32,
        o: LKOptions,
        max_level: usize,
        eps2: f64,
        gpa: Allocator,
        fail: *?Allocator.Error,
    };
    var fail: ?Allocator.Error = null;
    exec.parallelFor(prev_pts.len, 8, Ctx{ .prev = prev, .next = next, .pp = prev_pts, .np = next_pts, .st = status, .err = err, .o = o, .max_level = max_level, .eps2 = eps * eps, .gpa = gpa, .fail = &fail }, struct {
        fn run(c: Ctx, p0: usize, p1: usize) void {
            trackChunk(c.prev, c.next, c.pp, c.np, c.st, c.err, c.o, c.max_level, c.eps2, c.gpa, p0, p1) catch |e| {
                c.fail.* = e;
            };
        }
    }.run);
    if (fail) |e| return e;
}

/// Pyramidal Lucas-Kanade (OpenCV's calcOpticalFlowPyrLK) for 8-bit images
/// of 1-4 channels. On return `next_pts` holds the tracked positions,
/// `status[i]` is 1 where point i was found, and `err` (optional) the mean
/// absolute difference between the two windows (or the minimum eigenvalue
/// with `get_min_eigenvals`); `err` is left unchanged for points lost on
/// the eigenvalue test, as in OpenCV.
pub fn calcOpticalFlowPyrLK(exec: Exec, gpa: Allocator, prev: Image(u8), next: Image(u8), prev_pts: []const Point2f, next_pts: []Point2f, status: []u8, err: ?[]f32, o: LKOptions) Error!void {
    if (!prev.sameShape(next)) return error.SizeMismatch;
    if (next_pts.len != prev_pts.len or status.len != prev_pts.len) return error.SizeMismatch;
    if (prev_pts.len == 0) return;
    var pp = try FlowPyramid.build(exec, gpa, prev, o.win_width, o.win_height, o.max_level);
    defer pp.deinit(gpa);
    var np = try FlowPyramid.build(exec, gpa, next, o.win_width, o.win_height, o.max_level);
    defer np.deinit(gpa);
    return calcOpticalFlowPyrLKPyr(exec, gpa, &pp, &np, prev_pts, next_pts, status, err, o);
}

const w_bits = 14;
const flt_scale: f32 = 1.0 / @as(f32, 1 << 20);

inline fn cvRound(x: f32) i32 {
    return @intFromFloat(core.sat.rint(x));
}

/// floor(x) as an integer, or null when x is not a sane coordinate.
inline fn ifloor(x: f32) ?i32 {
    if (!(@abs(x) < 1e9)) return null;
    return @intFromFloat(@floor(x));
}

const Weights = struct {
    w00: i32,
    w01: i32,
    w10: i32,
    w11: i32,

    /// [w00 w01 w00 w01 ...] and [w10 w11 ...] for madd.
    fn pair01(w: Weights) V16 {
        return pairs(w.w00, w.w01);
    }

    fn pair23(w: Weights) V16 {
        return pairs(w.w10, w.w11);
    }

    fn pairs(a: i32, b: i32) V16 {
        var r: [16]i16 = undefined;
        for (0..8) |i| {
            r[2 * i] = @intCast(a);
            r[2 * i + 1] = @intCast(b);
        }
        return r;
    }
};

inline fn weights(a: f32, b: f32) Weights {
    const one: f32 = @floatFromInt(1 << w_bits);
    const w00 = cvRound((1 - a) * (1 - b) * one);
    const w01 = cvRound(a * (1 - b) * one);
    const w10 = cvRound((1 - a) * b * one);
    return .{ .w00 = w00, .w01 = w01, .w10 = w10, .w11 = (1 << w_bits) - w00 - w01 - w10 };
}

/// i16 pairs multiplied and summed into i32 lanes: r[i] = a[2i] * b[2i] +
/// a[2i+1] * b[2i+1] (x86 pmaddwd; NEON builds it from widening
/// multiplies). Window sums and bilinear weights are all 16-bit, so the
/// tracker's arithmetic runs on 16 elements per instruction.
inline fn madd(a: V16, b: V16) V8 {
    if (comptime use_asm_madd) {
        return asm ("vpmaddwd %[b], %[a], %[r]"
            : [r] "=x" (-> V8),
            : [a] "x" (a),
              [b] "x" (b),
        );
    }
    const even = comptime blk: {
        var m: [8]i32 = undefined;
        for (&m, 0..) |*v, i| v.* = @intCast(2 * i);
        break :blk m;
    };
    const odd = comptime blk: {
        var m: [8]i32 = undefined;
        for (&m, 0..) |*v, i| v.* = @intCast(2 * i + 1);
        break :blk m;
    };
    const ae: V8 = @shuffle(i16, a, undefined, even);
    const ao: V8 = @shuffle(i16, a, undefined, odd);
    const be: V8 = @shuffle(i16, b, undefined, even);
    const bo: V8 = @shuffle(i16, b, undefined, odd);
    return ae * be + ao * bo;
}

const use_asm_madd = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

/// Interleaves two 8-lane halves: [a0 b0 a1 b1 ... a7 b7].
inline fn zip(a: @Vector(8, i16), b: @Vector(8, i16)) V16 {
    const m = comptime blk: {
        var r: [16]i32 = undefined;
        for (0..8) |i| {
            r[2 * i] = @intCast(i);
            r[2 * i + 1] = ~@as(i32, @intCast(i));
        }
        break :blk r;
    };
    return @shuffle(i16, a, b, m);
}

inline fn lo8(v: anytype) @Vector(8, i16) {
    const a: [16]i16 = v;
    return a[0..8].*;
}

inline fn hi8(v: anytype) @Vector(8, i16) {
    const a: [16]i16 = v;
    return a[8..16].*;
}

inline fn load16(comptime T: type, p: [*]const T, i: usize) V16 {
    const v: @Vector(16, T) = p[i..][0..16].*;
    return v;
}

/// Bilinear samples of 16 consecutive elements at p[x] (neighbours at +cn
/// and one row down), descaled by `shift`: the two halves as i32.
inline fn bilin16(comptime T: type, p: [*]const T, stride: usize, x: usize, cn: usize, w01: V16, w23: V16, comptime shift: comptime_int) [2]V8 {
    if (T == u8) {
        // Interleave the bytes first, then widen: cheaper than zipping
        // 16-bit lanes across register halves.
        const m = comptime blk: {
            var r: [32]i32 = undefined;
            for (0..16) |i| {
                r[2 * i] = @intCast(i);
                r[2 * i + 1] = ~@as(i32, @intCast(i));
            }
            break :blk r;
        };
        const ab: [32]u8 = @shuffle(u8, @as(@Vector(16, u8), p[x..][0..16].*), @as(@Vector(16, u8), p[x + cn ..][0..16].*), m);
        const cd: [32]u8 = @shuffle(u8, @as(@Vector(16, u8), p[x + stride ..][0..16].*), @as(@Vector(16, u8), p[x + stride + cn ..][0..16].*), m);
        const delta: V8 = @splat(1 << (shift - 1));
        const ab_lo: V16 = @as(@Vector(16, u8), ab[0..16].*);
        const ab_hi: V16 = @as(@Vector(16, u8), ab[16..32].*);
        const cd_lo: V16 = @as(@Vector(16, u8), cd[0..16].*);
        const cd_hi: V16 = @as(@Vector(16, u8), cd[16..32].*);
        const lo = madd(ab_lo, w01) + madd(cd_lo, w23) + delta;
        const hi = madd(ab_hi, w01) + madd(cd_hi, w23) + delta;
        return .{ lo >> @splat(shift), hi >> @splat(shift) };
    }
    const a = load16(T, p, x);
    const b = load16(T, p, x + cn);
    const c = load16(T, p, x + stride);
    const d = load16(T, p, x + stride + cn);
    const delta: V8 = @splat(1 << (shift - 1));
    const lo = madd(zip(lo8(a), lo8(b)), w01) + madd(zip(lo8(c), lo8(d)), w23) + delta;
    const hi = madd(zip(hi8(a), hi8(b)), w01) + madd(zip(hi8(c), hi8(d)), w23) + delta;
    return .{ lo >> @splat(shift), hi >> @splat(shift) };
}

inline fn pack(v: [2]V8) V16 {
    return @intCast(std.simd.join(v[0], v[1]));
}

/// Products summed per i32 lane before the lanes are flushed to i64: each
/// madd lane is below 2^27 (two 8160 * 4080 products), so 16 cannot
/// overflow.
const flush_every = 16;

/// A window's source rows: straight from the image when the window (with
/// its bilinear neighbours and vector tail) lies inside, else copied into
/// `scratch` with the reflect101 border OpenCV pads its pyramids with.
fn Patch(comptime T: type) type {
    return struct {
        ptr: [*]const T,
        stride: usize,

        fn get(img: Image(T), x0: isize, y0: isize, cols: usize, rows: usize, scratch: []T) @This() {
            const cn: usize = img.channels;
            if (x0 >= 0 and y0 >= 0 and @as(usize, @intCast(y0)) + rows <= img.height and @as(usize, @intCast(x0)) * cn + cols <= img.rowLen())
                return .{ .ptr = img.data + @as(usize, @intCast(y0)) * img.stride + @as(usize, @intCast(x0)) * cn, .stride = img.stride };
            const px = (cols + cn - 1) / cn;
            const w: isize = @intCast(img.width);
            // Columns [lo, hi) of the patch lie inside the image.
            const lo: usize = @intCast(std.math.clamp(-x0, 0, @as(isize, @intCast(px))));
            const hi: usize = @intCast(std.math.clamp(w - x0, @as(isize, @intCast(lo)), @as(isize, @intCast(px))));
            // Source column of each patch column outside the image.
            var sxs: [256]u32 = undefined;
            const mapped = px <= sxs.len;
            if (mapped) for (0..px) |i| if (i < lo or i >= hi) {
                sxs[i] = @intCast(imgproc.border.map(x0 + @as(isize, @intCast(i)), img.width, .reflect101).?);
            };
            for (0..rows) |j| {
                const out = scratch[j * cols ..][0..cols];
                const row = img.row(imgproc.border.map(y0 + @as(isize, @intCast(j)), img.height, .reflect101).?);
                if (hi > lo) {
                    const start: usize = @intCast(x0 + @as(isize, @intCast(lo)));
                    const n = @min(hi * cn, cols) - lo * cn;
                    @memcpy(out[lo * cn ..][0..n], row[start * cn ..][0..n]);
                }
                for ([2][2]usize{ .{ 0, lo }, .{ hi, px } }) |range| for (range[0]..range[1]) |i| {
                    const sx: usize = if (mapped) sxs[i] else imgproc.border.map(x0 + @as(isize, @intCast(i)), img.width, .reflect101).?;
                    for (0..@min(cn, cols - i * cn)) |c| out[i * cn + c] = row[sx * cn + c];
                };
            }
            return .{ .ptr = scratch.ptr, .stride = cols };
        }
    };
}

/// OpenCV's calcScharrDeriv over a window: Ix = [-1 0 1] x [3 10 3]^T and
/// Iy its transpose (unscaled i16) for `rows` x `cols` elements, from the
/// image patch `m` that has a one-pixel margin on every side (reflect101,
/// as calcScharrDeriv extrapolates). Writes Ix to d[0..] and Iy to
/// d[cols * rows..], rows `cols` apart; d[2 * cols * rows..] is scratch.
/// OpenCV differentiates the whole level, but only the windows' pixels are
/// ever read, so computing them per window gives the same values without
/// full-size buffers.
fn scharrPatch(m: Patch(u8), cn: usize, cols: usize, rows: usize, d: []i16) void {
    const M = 16;
    const V = @Vector(M, i16);
    const mc = cols + 2 * cn;
    // Vertical sums of one row: smoothing in tv[0..mc], difference after.
    const tv = d[2 * cols * rows ..][0 .. 2 * mc];
    for (0..rows) |r| {
        const a = m.ptr + r * m.stride;
        const b = a + m.stride;
        const c = b + m.stride;
        var x: usize = 0;
        while (x + M <= mc) : (x += M) {
            const va: V = @as(@Vector(M, u8), a[x..][0..M].*);
            const vb: V = @as(@Vector(M, u8), b[x..][0..M].*);
            const vc: V = @as(@Vector(M, u8), c[x..][0..M].*);
            tv[x..][0..M].* = (va + vc) * @as(V, @splat(3)) + vb * @as(V, @splat(10));
            tv[mc + x ..][0..M].* = vc - va;
        }
        while (x < mc) : (x += 1) {
            const va: i16 = a[x];
            const vb: i16 = b[x];
            const vc: i16 = c[x];
            tv[x] = (va + vc) * 3 + vb * 10;
            tv[mc + x] = vc - va;
        }
        const ox = d[r * cols ..][0..cols];
        const oy = d[cols * rows + r * cols ..][0..cols];
        x = 0;
        while (x + M <= cols) : (x += M) {
            const l0: V = tv[x..][0..M].*;
            const r0: V = tv[x + 2 * cn ..][0..M].*;
            const l1: V = tv[mc + x ..][0..M].*;
            const c1: V = tv[mc + x + cn ..][0..M].*;
            const r1: V = tv[mc + x + 2 * cn ..][0..M].*;
            ox[x..][0..M].* = r0 - l0;
            oy[x..][0..M].* = (l1 + r1) * @as(V, @splat(3)) + c1 * @as(V, @splat(10));
        }
        while (x < cols) : (x += 1) {
            ox[x] = tv[x + 2 * cn] - tv[x];
            oy[x] = (tv[mc + x] + tv[mc + x + 2 * cn]) * 3 + tv[mc + x + cn] * 10;
        }
    }
}

/// Zeroes the derivatives of window pixels outside the image (OpenCV pads
/// its derivative buffers with zeros).
fn zeroOutside(img: Image(u8), x0: isize, y0: isize, cn: usize, cols: usize, rows: usize, d: []i16) void {
    const w: isize = @intCast(img.width);
    const h: isize = @intCast(img.height);
    const px: isize = @intCast((cols + cn - 1) / cn);
    if (x0 >= 0 and y0 >= 0 and x0 + px <= w and y0 + @as(isize, @intCast(rows)) <= h) return;
    for (0..rows) |r| {
        const y = y0 + @as(isize, @intCast(r));
        for (0..cols) |e| {
            const x = x0 + @as(isize, @intCast(e / cn));
            if (y < 0 or y >= h or x < 0 or x >= w) {
                d[r * cols + e] = 0;
                d[cols * rows + r * cols + e] = 0;
            }
        }
    }
}

fn outside(x: ?i32, y: ?i32, ww: usize, wh: usize, img: Image(u8)) bool {
    return x == null or y == null or x.? < -@as(i32, @intCast(ww)) or x.? >= @as(i32, @intCast(img.width)) or
        y.? < -@as(i32, @intCast(wh)) or y.? >= @as(i32, @intCast(img.height));
}

fn trackChunk(prev: *const FlowPyramid, next: *const FlowPyramid, prev_pts: []const Point2f, next_pts: []Point2f, status: []u8, err: ?[]f32, o: LKOptions, max_level: usize, eps2: f64, gpa: Allocator, p0: usize, p1: usize) Allocator.Error!void {
    const ww: usize = o.win_width;
    const wh: usize = o.win_height;
    const cn: usize = prev.levels[0].channels;
    const e = ww * cn; // elements per window row
    const nv = (e + 15) / 16; // 16-element blocks per window row
    const rl = nv * 16;
    // Patch geometry: window rows plus the row below, elements plus the
    // right neighbour.
    const pcols = rl + cn;
    const prows = wh + 1;
    const iwin = try gpa.alloc(i32, rl * wh);
    defer gpa.free(iwin);
    const gbuf = try gpa.alloc(i16, 2 * rl * wh);
    defer gpa.free(gbuf);
    const ixw = gbuf[0 .. rl * wh];
    const iyw = gbuf[rl * wh ..][0 .. rl * wh];
    const sbuf = try gpa.alloc(u8, pcols * prows);
    defer gpa.free(sbuf);
    // The previous window with a one-pixel margin for the derivatives.
    const mcols = pcols + 2 * cn;
    const mbuf = try gpa.alloc(u8, mcols * (prows + 2));
    defer gpa.free(mbuf);
    const dbuf = try gpa.alloc(i16, 2 * pcols * prows + 2 * mcols);
    defer gpa.free(dbuf);
    // Lanes of the last block past the window's last element.
    var tail_mask: [16]i16 = undefined;
    for (&tail_mask, 0..) |*m, i| m.* = if ((nv - 1) * 16 + i < e) -1 else 0;
    const tmask: V16 = tail_mask;
    const tmask32: [2]V8 = .{ @as(V8, lo8(tmask)), @as(V8, hi8(tmask)) };
    const half_w: f32 = (@as(f32, @floatFromInt(ww)) - 1) * 0.5;
    const half_h: f32 = (@as(f32, @floatFromInt(wh)) - 1) * 0.5;
    const max_count: usize = @min(o.max_count, 100);
    const area: f32 = @floatFromInt(ww * wh);

    for (p0..p1) |pi| {
        var level: usize = max_level + 1;
        while (level > 0) {
            level -= 1;
            const I = prev.levels[level];
            const J = next.levels[level];
            const lscale: f32 = @floatCast(1.0 / @as(f64, @floatFromInt(@as(u64, 1) << @intCast(level))));
            var prev_pt: Point2f = .{ .x = prev_pts[pi].x * lscale, .y = prev_pts[pi].y * lscale };
            var next_pt: Point2f = if (level == max_level)
                (if (o.use_initial_flow) .{ .x = next_pts[pi].x * lscale, .y = next_pts[pi].y * lscale } else prev_pt)
            else
                .{ .x = next_pts[pi].x * 2, .y = next_pts[pi].y * 2 };
            next_pts[pi] = next_pt;

            prev_pt.x -= half_w;
            prev_pt.y -= half_h;
            const ipx = ifloor(prev_pt.x);
            const ipy = ifloor(prev_pt.y);
            if (outside(ipx, ipy, ww, wh, I)) {
                if (level == 0) {
                    status[pi] = 0;
                    if (err) |er| er[pi] = 0;
                }
                continue;
            }
            const ix0: isize = ipx.?;
            const iy0: isize = ipy.?;
            var wt = weights(prev_pt.x - @as(f32, @floatFromInt(ix0)), prev_pt.y - @as(f32, @floatFromInt(iy0)));

            // Sample the previous window and its gradients; sum the
            // gradient matrix.
            {
                const pm = Patch(u8).get(I, ix0 - 1, iy0 - 1, mcols, prows + 2, mbuf);
                scharrPatch(pm, cn, pcols, prows, dbuf);
                zeroOutside(I, ix0, iy0, cn, pcols, prows, dbuf);
                const isrc = pm.ptr + pm.stride + cn;
                var w01 = wt.pair01();
                var w23 = wt.pair23();
                var a11: VL = @splat(0);
                var a12: VL = @splat(0);
                var a22: VL = @splat(0);
                var s11: V8 = @splat(0);
                var s12: V8 = @splat(0);
                var s22: V8 = @splat(0);
                var cnt: usize = 0;
                for (0..wh) |y| {
                    const src = isrc + y * pm.stride;
                    const dxs = dbuf.ptr + y * pcols;
                    const dys = dbuf.ptr + pcols * prows + y * pcols;
                    for (0..nv) |v| {
                        const x = v * 16;
                        const k = y * rl + x;
                        const ival = bilin16(u8, src, pm.stride, x, cn, w01, w23, w_bits - 5);
                        var gx = pack(bilin16(i16, dxs, pcols, x, cn, w01, w23, w_bits));
                        var gy = pack(bilin16(i16, dys, pcols, x, cn, w01, w23, w_bits));
                        if (v == nv - 1) {
                            gx &= tmask;
                            gy &= tmask;
                        }
                        iwin[k..][0..8].* = ival[0];
                        iwin[k + 8 ..][0..8].* = ival[1];
                        ixw[k..][0..16].* = gx;
                        iyw[k..][0..16].* = gy;
                        s11 += madd(gx, gx);
                        s12 += madd(gx, gy);
                        s22 += madd(gy, gy);
                        cnt += 1;
                        if (cnt == flush_every) {
                            a11 += @as(VL, s11);
                            a12 += @as(VL, s12);
                            a22 += @as(VL, s22);
                            s11 = @splat(0);
                            s12 = @splat(0);
                            s22 = @splat(0);
                            cnt = 0;
                        }
                    }
                }
                a11 += @as(VL, s11);
                a12 += @as(VL, s12);
                a22 += @as(VL, s22);
                const A11: f32 = @as(f32, @floatFromInt(@reduce(.Add, a11))) * flt_scale;
                const A12: f32 = @as(f32, @floatFromInt(@reduce(.Add, a12))) * flt_scale;
                const A22: f32 = @as(f32, @floatFromInt(@reduce(.Add, a22))) * flt_scale;
                var D = A11 * A22 - A12 * A12;
                const min_eig = (A22 + A11 - @sqrt((A11 - A22) * (A11 - A22) + 4 * A12 * A12)) / (2 * area);
                if (o.get_min_eigenvals) if (err) |er| {
                    er[pi] = min_eig;
                };
                if (min_eig < o.min_eig_threshold or D < std.math.floatEps(f32)) {
                    if (level == 0) status[pi] = 0;
                    continue;
                }
                D = 1 / D;

                next_pt.x -= half_w;
                next_pt.y -= half_h;
                var prev_delta: Point2f = .{ .x = 0, .y = 0 };
                var j: usize = 0;
                while (j < max_count) : (j += 1) {
                    const inx = ifloor(next_pt.x);
                    const iny = ifloor(next_pt.y);
                    if (outside(inx, iny, ww, wh, J)) {
                        if (level == 0) status[pi] = 0;
                        break;
                    }
                    wt = weights(next_pt.x - @as(f32, @floatFromInt(inx.?)), next_pt.y - @as(f32, @floatFromInt(iny.?)));
                    w01 = wt.pair01();
                    w23 = wt.pair23();
                    const pJ = Patch(u8).get(J, inx.?, iny.?, pcols, prows, sbuf);
                    var b1: VL = @splat(0);
                    var b2: VL = @splat(0);
                    var t1: V8 = @splat(0);
                    var t2: V8 = @splat(0);
                    var tc: usize = 0;
                    for (0..wh) |y| {
                        const jrow = pJ.ptr + y * pJ.stride;
                        for (0..nv) |v| {
                            const x = v * 16;
                            const k = y * rl + x;
                            const jval = bilin16(u8, jrow, pJ.stride, x, cn, w01, w23, w_bits - 5);
                            const diff = pack(.{ jval[0] - @as(V8, iwin[k..][0..8].*), jval[1] - @as(V8, iwin[k + 8 ..][0..8].*) });
                            t1 += madd(diff, ixw[k..][0..16].*);
                            t2 += madd(diff, iyw[k..][0..16].*);
                            tc += 1;
                            if (tc == flush_every) {
                                b1 += @as(VL, t1);
                                b2 += @as(VL, t2);
                                t1 = @splat(0);
                                t2 = @splat(0);
                                tc = 0;
                            }
                        }
                    }
                    b1 += @as(VL, t1);
                    b2 += @as(VL, t2);
                    const B1: f32 = @as(f32, @floatFromInt(@reduce(.Add, b1))) * flt_scale;
                    const B2: f32 = @as(f32, @floatFromInt(@reduce(.Add, b2))) * flt_scale;
                    const delta: Point2f = .{ .x = (A12 * B2 - A22 * B1) * D, .y = (A12 * B1 - A11 * B2) * D };
                    next_pt.x += delta.x;
                    next_pt.y += delta.y;
                    next_pts[pi] = .{ .x = next_pt.x + half_w, .y = next_pt.y + half_h };
                    const dd = @as(f64, delta.x) * delta.x + @as(f64, delta.y) * delta.y;
                    if (dd <= eps2) break;
                    if (j > 0 and @abs(@as(f64, delta.x + prev_delta.x)) < 0.01 and @abs(@as(f64, delta.y + prev_delta.y)) < 0.01) {
                        next_pts[pi].x -= delta.x * 0.5;
                        next_pts[pi].y -= delta.y * 0.5;
                        break;
                    }
                    prev_delta = delta;
                }
            }

            if (status[pi] != 0 and err != null and level == 0 and !o.get_min_eigenvals) {
                const np: Point2f = .{ .x = next_pts[pi].x - half_w, .y = next_pts[pi].y - half_h };
                const inx = ifloor(np.x);
                const iny = ifloor(np.y);
                if (outside(inx, iny, ww, wh, J)) {
                    status[pi] = 0;
                    continue;
                }
                wt = weights(np.x - @as(f32, @floatFromInt(inx.?)), np.y - @as(f32, @floatFromInt(iny.?)));
                const w01 = wt.pair01();
                const w23 = wt.pair23();
                const pJ = Patch(u8).get(J, inx.?, iny.?, pcols, prows, sbuf);
                var acc: V8 = @splat(0);
                for (0..wh) |y| {
                    const jrow = pJ.ptr + y * pJ.stride;
                    for (0..nv) |v| {
                        const x = v * 16;
                        const k = y * rl + x;
                        const jval = bilin16(u8, jrow, pJ.stride, x, cn, w01, w23, w_bits - 5);
                        var d0: V8 = @intCast(@abs(jval[0] - @as(V8, iwin[k..][0..8].*)));
                        var d1: V8 = @intCast(@abs(jval[1] - @as(V8, iwin[k + 8 ..][0..8].*)));
                        if (v == nv - 1) {
                            d0 &= tmask32[0];
                            d1 &= tmask32[1];
                        }
                        acc += d0 + d1;
                    }
                }
                const errval: f32 = @floatFromInt(@reduce(.Add, acc));
                err.?[pi] = errval / @as(f32, @floatFromInt(32 * ww * cn * wh));
            }
        }
    }
}

// ------------------------------------------------------------------ tests

test "pyramidal LK recovers a known shift" {
    const gpa = std.testing.allocator;
    const w = 160;
    const h = 120;
    var a = try Image(u8).init(gpa, w, h, 1);
    defer a.deinit(gpa);
    var b = try Image(u8).init(gpa, w, h, 1);
    defer b.deinit(gpa);
    // A smooth textured pattern, shifted by (3, -2) in the second frame.
    const pat = struct {
        fn f(x: f64, y: f64) u8 {
            const v = 128 + 50 * @sin(x / 5.0) * @cos(y / 7.0) + 40 * @sin((x + 2 * y) / 9.0);
            return @intFromFloat(@round(std.math.clamp(v, 0, 255)));
        }
    }.f;
    for (0..h) |y| for (0..w) |x| {
        a.set(x, y, 0, pat(@floatFromInt(x), @floatFromInt(y)));
        b.set(x, y, 0, pat(@as(f64, @floatFromInt(x)) - 3, @as(f64, @floatFromInt(y)) + 2));
    };
    const pts = [_]Point2f{ .{ .x = 50, .y = 40 }, .{ .x = 80.5, .y = 60.25 }, .{ .x = 110, .y = 70 }, .{ .x = -500, .y = 3 } };
    var out: [pts.len]Point2f = undefined;
    var st: [pts.len]u8 = undefined;
    var er: [pts.len]f32 = undefined;
    try calcOpticalFlowPyrLK(.serial, gpa, a, b, &pts, &out, &st, &er, .{});
    for (pts[0..3], out[0..3], st[0..3]) |p, q, s| {
        try std.testing.expectEqual(@as(u8, 1), s);
        try std.testing.expectApproxEqAbs(p.x + 3, q.x, 0.05);
        try std.testing.expectApproxEqAbs(p.y - 2, q.y, 0.05);
    }
    try std.testing.expectEqual(@as(u8, 0), st[3]);
}
