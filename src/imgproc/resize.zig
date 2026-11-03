//! Resizing. 8-bit bilinear and area resampling use OpenCV's fixed-point
//! coefficients (INTER_RESIZE_COEF_BITS = 11) so results match cv2.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const Allocator = std.mem.Allocator;
const ru8 = @import("resize_u8.zig");

pub const Interp = enum(u8) {
    nearest = 0,
    linear = 1,
    cubic = 2,
    area = 3,
    lanczos4 = 4,
};

pub const Error = core.Error || Allocator.Error || error{Unsupported};

const coef_bits = 11;
const coef_scale = 1 << coef_bits;

/// Resizes `src` into `dst` (the sizes of `dst` define the scale). Both
/// images must have the same channel count.
pub fn resize(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), interp: Interp) Error!void {
    if (src.channels != dst.channels) return error.SizeMismatch;
    if (src.sameSize(dst)) return dst.copyFrom(src);
    if (@TypeOf(src).Elem == u8 and src.channels <= 4) {
        const exact_half = src.width == 2 * dst.width and src.height == 2 * dst.height;
        switch (interp) {
            .linear => return if (exact_half) ru8.half(exec, src, dst) else linearU8(exec, gpa, src, dst, false),
            .area => {
                if (dst.width > src.width or dst.height > src.height) return linearU8(exec, gpa, src, dst, true);
                if (exact_half) return ru8.half(exec, src, dst);
            },
            .cubic => return cubicU8(exec, gpa, src, dst),
            else => {},
        }
    }
    switch (interp) {
        .nearest => return nearest(exec, gpa, src, dst),
        .linear => return linear(exec, gpa, src, dst, false),
        .area => {
            // OpenCV uses INTER_AREA only for shrinking; enlarging is bilinear
            // with area-style coefficients.
            if (dst.width > src.width or dst.height > src.height) return areaUp(exec, gpa, src, dst);
            const ix = src.width / dst.width;
            const iy = src.height / dst.height;
            if (ix * dst.width == src.width and iy * dst.height == src.height) return areaInt(exec, src, dst, ix, iy);
            return areaGeneric(exec, gpa, src, dst);
        },
        .cubic => return cubic(exec, gpa, src, dst),
        .lanczos4 => return error.Unsupported,
    }
}

fn nearest(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src)) Error!void {
    const T = @TypeOf(src).Elem;
    const cn = src.channels;
    const xofs = try gpa.alloc(u32, dst.width);
    defer gpa.free(xofs);
    const ifx = @as(f64, @floatFromInt(src.width)) / @as(f64, @floatFromInt(dst.width));
    const ify = @as(f64, @floatFromInt(src.height)) / @as(f64, @floatFromInt(dst.height));
    for (xofs, 0..) |*o, x| {
        const sx: u32 = @intFromFloat(@floor(@as(f64, @floatFromInt(x)) * ifx));
        o.* = @min(sx, src.width - 1) * cn;
    }
    const Ctx = struct { s: Image(T), d: Image(T), xofs: []const u32, ify: f64 };
    exec.parallelFor(dst.height, 16, Ctx{ .s = src, .d = dst, .xofs = xofs, .ify = ify }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                const sy: u32 = @intFromFloat(@floor(@as(f64, @floatFromInt(y)) * c.ify));
                const sr = c.s.row(@min(sy, c.s.height - 1));
                const dr = c.d.row(y);
                switch (c.s.channels) {
                    inline 1, 2, 3, 4 => |k| for (c.xofs, 0..) |o, x| {
                        inline for (0..k) |ch| dr[x * k + ch] = sr[o + ch];
                    },
                    else => |k| for (c.xofs, 0..) |o, x| @memcpy(dr[x * k ..][0..k], sr[o..][0..k]),
                }
            }
        }
    }.run);
}

/// Source index and fractional weight for one destination coordinate, with
/// OpenCV's half-pixel centres and edge clamping.
fn linearCoord(d: usize, scale: f64, len: u32, area: bool) struct { i: u32, f: f64 } {
    var sx: i64 = undefined;
    var f: f64 = undefined;
    if (area) {
        // INTER_AREA when enlarging: OpenCV's area-mode coefficients.
        const df: f64 = @floatFromInt(d);
        sx = @intFromFloat(@floor(df * scale));
        f = (df + 1) - @as(f64, @floatFromInt(sx + 1)) / scale;
        f = if (f <= 0) 0 else f - @floor(f);
    } else {
        const fx = (@as(f64, @floatFromInt(d)) + 0.5) * scale - 0.5;
        sx = @intFromFloat(@floor(fx));
        f = fx - @as(f64, @floatFromInt(sx));
    }
    if (sx < 0) {
        sx = 0;
        f = 0;
    }
    if (sx >= @as(i64, len) - 1) {
        sx = @as(i64, len) - 1;
        f = 0;
    }
    return .{ .i = @intCast(sx), .f = f };
}

fn linearU8(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), area: bool) Error!void {
    const cn: u32 = src.channels;
    const scale_x = @as(f64, @floatFromInt(src.width)) / @as(f64, @floatFromInt(dst.width));
    const scale_y = @as(f64, @floatFromInt(src.height)) / @as(f64, @floatFromInt(dst.height));
    const W = Weights(u8);
    const xt = try gpa.alloc(ru8.Tap(2), dst.width);
    defer gpa.free(xt);
    const yt = try gpa.alloc(ru8.Tap(2), dst.height);
    defer gpa.free(yt);
    for (xt, 0..) |*t, x| {
        const c = linearCoord(x, scale_x, src.width, area);
        t.* = .{ .o = .{ c.i * cn, @min(c.i + 1, src.width - 1) * cn }, .w = W.coef(c.f) };
    }
    for (yt, 0..) |*t, y| {
        const c = linearCoord(y, scale_y, src.height, area);
        t.* = .{ .o = .{ c.i, @min(c.i + 1, src.height - 1) }, .w = W.coef(c.f) };
    }
    if (!area and dst.width == 2 * src.width and dst.height == 2 * src.height and src.width >= 2) {
        const beta = try gpa.alloc([2]i32, dst.height);
        defer gpa.free(beta);
        const yofs = try gpa.alloc([2]u32, dst.height);
        defer gpa.free(yofs);
        for (yt, beta, yofs) |t, *b, *o| {
            b.* = t.w;
            o.* = t.o;
        }
        return ru8.up2Linear(exec, gpa, src, dst, beta, yofs);
    }
    const gs: []ru8.Group = if (ru8.fast_permute) try ru8.groups(gpa, xt, cn, src.rowLen()) else &.{};
    defer if (ru8.fast_permute) gpa.free(gs);
    ru8.separable(2, .linear, exec, gpa, src, dst, xt, yt, gs);
}

/// OpenCV's fixed-point INTER_CUBIC for u8: f32 coordinates and
/// coefficients rounded to 11 bits, integer horizontal pass.
fn cubicU8(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8)) Error!void {
    const xt = try gpa.alloc(ru8.Tap(4), dst.width);
    defer gpa.free(xt);
    const yt = try gpa.alloc(ru8.Tap(4), dst.height);
    defer gpa.free(yt);
    cubicTab(xt, src.width, dst.width, src.channels);
    cubicTab(yt, src.height, dst.height, 1);
    if (dst.width == 2 * src.width and dst.height == 2 * src.height and src.width >= 4) {
        const beta = try gpa.alloc([4]i32, dst.height);
        defer gpa.free(beta);
        const yofs = try gpa.alloc([4]u32, dst.height);
        defer gpa.free(yofs);
        for (yt, beta, yofs) |t, *b, *o| {
            b.* = t.w;
            o.* = t.o;
        }
        // Interior weights: destination pixels 2 (even) and 3 (odd).
        return ru8.up2Cubic(exec, gpa, src, dst, xt[2].w, xt[3].w, beta, yofs);
    }
    ru8.separable(4, .cubic, exec, gpa, src, dst, xt, yt, &.{});
}

fn cubicTab(taps: []ru8.Tap(4), ssize: u32, dsize: u32, mul: u32) void {
    const scale = 1.0 / (@as(f64, @floatFromInt(dsize)) / @as(f64, @floatFromInt(ssize)));
    for (taps, 0..) |*t, i| {
        var fx: f32 = @floatCast((@as(f64, @floatFromInt(i)) + 0.5) * scale - 0.5);
        const sx = @floor(fx);
        fx -= sx;
        const isx: i64 = @intFromFloat(sx);
        const cf = cubicCoeffs(fx);
        for (0..4) |k| {
            t.w[k] = sat.cast(i16, cf[k] * coef_scale);
            const p = std.math.clamp(isx - 1 + @as(i64, @intCast(k)), 0, @as(i64, ssize) - 1);
            t.o[k] = @as(u32, @intCast(p)) * mul;
        }
    }
}

fn linear(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), area: bool) Error!void {
    const T = @TypeOf(src).Elem;
    const cn: usize = src.channels;
    const W = Weights(T);
    const scale_x = @as(f64, @floatFromInt(src.width)) / @as(f64, @floatFromInt(dst.width));
    const scale_y = @as(f64, @floatFromInt(src.height)) / @as(f64, @floatFromInt(dst.height));
    // Horizontal taps per destination sample (channel-expanded).
    const xofs = try gpa.alloc(u32, dst.rowLen());
    defer gpa.free(xofs);
    const alpha = try gpa.alloc([2]W.Coef, dst.rowLen());
    defer gpa.free(alpha);
    for (0..dst.width) |x| {
        const c = linearCoord(x, scale_x, src.width, area);
        const a = W.coef(c.f);
        for (0..cn) |ch| {
            xofs[x * cn + ch] = @intCast(c.i * cn + ch);
            alpha[x * cn + ch] = a;
        }
    }
    const step_x: u32 = @intCast(cn);
    const yofs = try gpa.alloc(u32, dst.height);
    defer gpa.free(yofs);
    const beta = try gpa.alloc([2]W.Coef, dst.height);
    defer gpa.free(beta);
    for (0..dst.height) |y| {
        const c = linearCoord(y, scale_y, src.height, area);
        yofs[y] = c.i;
        beta[y] = W.coef(c.f);
    }
    const Ctx = struct { s: Image(T), d: Image(T), xofs: []const u32, alpha: []const [2]W.Coef, yofs: []const u32, beta: []const [2]W.Coef, step: u32, gpa: Allocator };
    const ctx = Ctx{ .s = src, .d = dst, .xofs = xofs, .alpha = alpha, .yofs = yofs, .beta = beta, .step = step_x, .gpa = gpa };
    exec.parallelFor(dst.height, 8, ctx, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const n = c.d.rowLen();
            var stack: [2][4096]W.Acc = undefined;
            var heap: ?[]W.Acc = null;
            defer if (heap) |h| c.gpa.free(h);
            var rows: [2][]W.Acc = if (n <= 4096) .{ stack[0][0..n], stack[1][0..n] } else blk: {
                heap = c.gpa.alloc(W.Acc, 2 * n) catch @panic("out of memory");
                break :blk .{ heap.?[0..n], heap.?[n..] };
            };
            var have: [2]i64 = .{ -1, -1 };
            const last_x = c.s.rowLen() - c.step;
            for (y0..y1) |y| {
                const sy = c.yofs[y];
                const sy1 = @min(sy + 1, c.s.height - 1);
                // Reuse horizontally resampled rows when they carry over.
                if (have[0] != sy) {
                    if (have[1] == sy) {
                        std.mem.swap([]W.Acc, &rows[0], &rows[1]);
                        have[0] = sy;
                        have[1] = -1;
                    } else {
                        hrow(T, c.s.row(sy), rows[0], c.xofs, c.alpha, c.step, last_x);
                        have[0] = sy;
                    }
                }
                if (have[1] != sy1) {
                    hrow(T, c.s.row(sy1), rows[1], c.xofs, c.alpha, c.step, last_x);
                    have[1] = sy1;
                }
                W.vrow(rows[0], rows[1], c.beta[y], c.d.row(y));
            }
        }
    }.run);
}

inline fn hrow(comptime T: type, s: []const T, out: []Weights(T).Acc, xofs: []const u32, alpha: []const [2]Weights(T).Coef, step: u32, last: usize) void {
    const W = Weights(T);
    for (out, xofs, alpha) |*o, xi, a| {
        const x1 = if (xi >= last) xi else xi + step;
        o.* = W.toAcc(s[xi]) * a[0] + W.toAcc(s[x1]) * a[1];
    }
}

fn Weights(comptime T: type) type {
    if (T == u8) return struct {
        const Coef = i32;
        const Acc = i32;
        fn coef(f: f64) [2]Coef {
            const a0: i32 = sat.cast(i16, (1.0 - f) * coef_scale);
            return .{ a0, coef_scale - a0 };
        }
        inline fn toAcc(v: u8) i32 {
            return v;
        }
        /// OpenCV's VResizeLinear<uchar,int,short> rounding: each product is
        /// pre-shifted by 4 and post-shifted by 16, then (+2) >> 2.
        fn vrow(r0: []const i32, r1: []const i32, b: [2]Coef, d: []u8) void {
            const N = 8;
            var x: usize = 0;
            const b0v: @Vector(N, i32) = @splat(b[0]);
            const b1v: @Vector(N, i32) = @splat(b[1]);
            const two: @Vector(N, i32) = @splat(2);
            while (x + N <= d.len) : (x += N) {
                const a: @Vector(N, i32) = r0[x..][0..N].*;
                const c: @Vector(N, i32) = r1[x..][0..N].*;
                const t = (mulhi(a >> @splat(4), b0v) + mulhi(c >> @splat(4), b1v) + two) >> @splat(2);
                const cl = @max(@min(t, @as(@Vector(N, i32), @splat(255))), @as(@Vector(N, i32), @splat(0)));
                const out: @Vector(N, u8) = @intCast(cl);
                d[x..][0..N].* = out;
            }
            while (x < d.len) : (x += 1) {
                const t = (((r0[x] >> 4) * b[0]) >> 16) + (((r1[x] >> 4) * b[1]) >> 16);
                d[x] = sat.cast(u8, (t + 2) >> 2);
            }
        }
        inline fn mulhi(a: @Vector(8, i32), b: @Vector(8, i32)) @Vector(8, i32) {
            return (a * b) >> @splat(16);
        }
    };
    return struct {
        const Coef = f32;
        const Acc = f32;
        fn coef(f: f64) [2]Coef {
            return .{ @floatCast(1.0 - f), @floatCast(f) };
        }
        inline fn toAcc(v: T) f32 {
            return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
        }
        fn vrow(r0: []const f32, r1: []const f32, b: [2]Coef, d: []T) void {
            for (d, r0, r1) |*o, a, c| o.* = sat.cast(T, a * b[0] + c * b[1]);
        }
    };
}

/// Integer-factor area shrink: plain block average with round-half-up.
fn areaInt(exec: Exec, src: anytype, dst: @TypeOf(src), ix: u32, iy: u32) Error!void {
    const T = @TypeOf(src).Elem;
    const Ctx = struct { s: Image(T), d: Image(T), ix: u32, iy: u32 };
    exec.parallelFor(dst.height, 8, Ctx{ .s = src, .d = dst, .ix = ix, .iy = iy }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const area = c.ix * c.iy;
            for (y0..y1) |y| {
                const dr = c.d.row(y);
                if (T == u8 and c.ix == 2 and c.iy == 2) {
                    const a = c.s.row(2 * y);
                    const b = c.s.row(2 * y + 1);
                    for (0..c.d.width) |x| for (0..cn) |ch| {
                        const i = 2 * x * cn + ch;
                        dr[x * cn + ch] = @intCast((@as(u32, a[i]) + a[i + cn] + b[i] + b[i + cn] + 2) >> 2);
                    };
                    continue;
                }
                for (0..c.d.width) |x| for (0..cn) |ch| {
                    if (@typeInfo(T) == .float or @typeInfo(T).int.signedness == .signed) {
                        var sum: f64 = 0;
                        for (0..c.iy) |dy| {
                            const r = c.s.row(y * c.iy + dy);
                            for (0..c.ix) |dx| sum += toF(T, r[(x * c.ix + dx) * cn + ch]);
                        }
                        dr[x * cn + ch] = sat.cast(T, sum / @as(f64, @floatFromInt(area)));
                    } else {
                        var sum: u64 = 0;
                        for (0..c.iy) |dy| {
                            const r = c.s.row(y * c.iy + dy);
                            for (0..c.ix) |dx| sum += r[(x * c.ix + dx) * cn + ch];
                        }
                        dr[x * cn + ch] = @intCast((sum + area / 2) / area);
                    }
                };
            }
        }
    }.run);
}

const AreaTab = struct { di: u32, si: u32, alpha: f32 };

/// OpenCV's computeResizeAreaTab: the overlap of each destination cell with
/// source cells, as (destination index, source index, weight) triples.
fn areaTab(gpa: Allocator, ssize: u32, dsize: u32, scale: f64) Allocator.Error![]AreaTab {
    var list: std.ArrayList(AreaTab) = .empty;
    errdefer list.deinit(gpa);
    for (0..dsize) |dx| {
        const fsx1 = @as(f64, @floatFromInt(dx)) * scale;
        const fsx2 = fsx1 + scale;
        const cellw = @min(scale, @as(f64, @floatFromInt(ssize)) - fsx1);
        var sx1: i64 = @intFromFloat(@ceil(fsx1));
        const sx2: i64 = @intFromFloat(@floor(fsx2));
        const sx2c: i64 = @min(sx2, @as(i64, ssize) - 1);
        sx1 = @min(sx1, @as(i64, ssize) - 1);
        if (@as(f64, @floatFromInt(sx1)) - fsx1 > 1e-3) {
            try list.append(gpa, .{ .di = @intCast(dx), .si = @intCast(sx1 - 1), .alpha = @floatCast((@as(f64, @floatFromInt(sx1)) - fsx1) / cellw) });
        }
        var sx = sx1;
        while (sx < sx2) : (sx += 1) {
            try list.append(gpa, .{ .di = @intCast(dx), .si = @intCast(sx), .alpha = @floatCast(1.0 / cellw) });
        }
        if (fsx2 - @as(f64, @floatFromInt(sx2)) > 1e-3) {
            try list.append(gpa, .{ .di = @intCast(dx), .si = @intCast(sx2c), .alpha = @floatCast(@min(@min(fsx2 - @as(f64, @floatFromInt(sx2)), 1.0), cellw) / cellw) });
        }
    }
    return list.toOwnedSlice(gpa);
}

/// Non-integer area shrink (OpenCV's ResizeArea_Invoker), float accumulation.
fn areaGeneric(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src)) Error!void {
    const T = @TypeOf(src).Elem;
    const sx = @as(f64, @floatFromInt(src.width)) / @as(f64, @floatFromInt(dst.width));
    const sy = @as(f64, @floatFromInt(src.height)) / @as(f64, @floatFromInt(dst.height));
    const xtab = try areaTab(gpa, src.width, dst.width, sx);
    defer gpa.free(xtab);
    const ytab = try areaTab(gpa, src.height, dst.height, sy);
    defer gpa.free(ytab);
    // Start of each destination row's run in ytab.
    const ystart = try gpa.alloc(u32, dst.height + 1);
    defer gpa.free(ystart);
    {
        var k: u32 = 0;
        for (0..dst.height) |dy| {
            ystart[dy] = k;
            while (k < ytab.len and ytab[k].di == dy) k += 1;
        }
        ystart[dst.height] = k;
    }
    const Ctx = struct { s: Image(T), d: Image(T), xtab: []const AreaTab, ytab: []const AreaTab, ystart: []const u32, gpa: Allocator };
    exec.parallelFor(dst.height, 4, Ctx{ .s = src, .d = dst, .xtab = xtab, .ytab = ytab, .ystart = ystart, .gpa = gpa }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const n = c.d.rowLen();
            const buf = c.gpa.alloc(f32, 2 * n) catch @panic("out of memory");
            defer c.gpa.free(buf);
            const hbuf = buf[0..n];
            const sum = buf[n..];
            for (y0..y1) |dy| {
                @memset(sum, 0);
                for (c.ytab[c.ystart[dy]..c.ystart[dy + 1]]) |yt| {
                    @memset(hbuf, 0);
                    const sr = c.s.row(yt.si);
                    for (c.xtab) |xt| {
                        for (0..cn) |ch| hbuf[xt.di * cn + ch] += toF(T, sr[xt.si * cn + ch]) * xt.alpha;
                    }
                    for (sum, hbuf) |*s, h| s.* += h * yt.alpha;
                }
                for (c.d.row(dy), sum) |*o, s| o.* = sat.cast(T, s);
            }
        }
    }.run);
}

/// Enlarging with INTER_AREA: bilinear with OpenCV's area-style weights
/// (floor-based coordinates, weight = clamp((x+1) - (sx+1)/scale, 0, 1)).
fn areaUp(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src)) Error!void {
    return linear(exec, gpa, src, dst, true);
}

inline fn toF(comptime T: type, v: T) f32 {
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

/// Bicubic (A = -0.75, OpenCV's INTER_CUBIC), computed in f32.
fn cubic(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src)) Error!void {
    const T = @TypeOf(src).Elem;
    const cn: usize = src.channels;
    const scale_x = @as(f64, @floatFromInt(src.width)) / @as(f64, @floatFromInt(dst.width));
    const scale_y = @as(f64, @floatFromInt(src.height)) / @as(f64, @floatFromInt(dst.height));
    const Tap = struct { o: [4]u32, w: [4]f32 };
    const xt = try gpa.alloc(Tap, dst.width);
    defer gpa.free(xt);
    const yt = try gpa.alloc(Tap, dst.height);
    defer gpa.free(yt);
    const fill = struct {
        fn f(taps: []Tap, scale: f64, len: u32, mul: usize) void {
            for (taps, 0..) |*t, i| {
                const fx = (@as(f64, @floatFromInt(i)) + 0.5) * scale - 0.5;
                const sx: i64 = @intFromFloat(@floor(fx));
                const f_: f32 = @floatCast(fx - @as(f64, @floatFromInt(sx)));
                t.w = cubicCoeffs(f_);
                for (0..4) |k| {
                    const p = std.math.clamp(sx - 1 + @as(i64, @intCast(k)), 0, @as(i64, len) - 1);
                    t.o[k] = @intCast(@as(usize, @intCast(p)) * mul);
                }
            }
        }
    }.f;
    fill(xt, scale_x, src.width, cn);
    fill(yt, scale_y, src.height, 1);
    const Ctx = struct { s: Image(T), d: Image(T), xt: []const Tap, yt: []const Tap, gpa: Allocator };
    exec.parallelFor(dst.height, 4, Ctx{ .s = src, .d = dst, .xt = xt, .yt = yt, .gpa = gpa }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const ch: usize = c.s.channels;
            const n = c.d.rowLen();
            const h = c.gpa.alloc(f32, 4 * n) catch @panic("out of memory");
            defer c.gpa.free(h);
            for (y0..y1) |y| {
                const t = c.yt[y];
                for (0..4) |k| {
                    const sr = c.s.row(t.o[k]);
                    const hr = h[k * n ..][0..n];
                    for (c.xt, 0..) |tx, x| for (0..ch) |cc| {
                        var acc: f32 = 0;
                        inline for (0..4) |j| acc += toF(T, sr[tx.o[j] + cc]) * tx.w[j];
                        hr[x * ch + cc] = acc;
                    };
                }
                const dr = c.d.row(y);
                for (dr, 0..) |*o, i| {
                    const v = h[i] * t.w[0] + h[n + i] * t.w[1] + h[2 * n + i] * t.w[2] + h[3 * n + i] * t.w[3];
                    o.* = sat.cast(T, v);
                }
            }
        }
    }.run);
}

fn cubicCoeffs(x: f32) [4]f32 {
    const A: f32 = -0.75;
    var c: [4]f32 = undefined;
    c[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    c[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    c[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    c[3] = 1 - c[0] - c[1] - c[2];
    return c;
}

test "resize identity-ish" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).initFill(gpa, 10, 10, 3, 100);
    defer src.deinit(gpa);
    inline for (.{ Interp.nearest, Interp.linear, Interp.area, Interp.cubic }) |m| {
        var dst = try Image(u8).init(gpa, 7, 13, 3);
        defer dst.deinit(gpa);
        try resize(.serial, gpa, src, dst, m);
        for (dst.slice()) |v| try std.testing.expectEqual(@as(u8, 100), v);
    }
}

test "u8 fast paths match the generic fixed-point kernels" {
    const gpa = std.testing.allocator;
    inline for (.{ 1, 2, 3, 4 }) |cn| {
        var src = try Image(u8).init(gpa, 150, 64, cn);
        defer src.deinit(gpa);
        var seed: u32 = 12345;
        for (src.slice()) |*v| {
            seed = seed *% 1103515245 +% 12345;
            v.* = @truncate(seed >> 16);
        }
        // Halving, exact 2x, and a generic ratio (VBMI groups when present).
        // Generic ratios cover every group size, strong downscales (smaller
        // groups, window overflows) and upscales.
        for ([_][2]u32{ .{ 75, 32 }, .{ 300, 128 }, .{ 113, 47 }, .{ 120, 90 }, .{ 112, 48 }, .{ 37, 20 }, .{ 20, 9 }, .{ 9, 5 }, .{ 199, 70 }, .{ 151, 64 }, .{ 1, 1 }, .{ 2, 64 } }) |sz| {
            var a = try Image(u8).init(gpa, sz[0], sz[1], cn);
            defer a.deinit(gpa);
            var b = try Image(u8).init(gpa, sz[0], sz[1], cn);
            defer b.deinit(gpa);
            try resize(.serial, gpa, src, a, .linear);
            try linear(.serial, gpa, src, b, false);
            try std.testing.expectEqualSlices(u8, b.slice(), a.slice());
        }
    }
}
