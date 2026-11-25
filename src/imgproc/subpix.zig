//! cornerSubPix and the getRectSubPix sampling it relies on, following
//! OpenCV's cornersubpix.cpp and samplers.cpp operation by operation (f32
//! bilinear window, f64 normal equations), so refined corners match
//! OpenCV's to float rounding.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Point2f = @import("shape.zig").Point2f;

pub const Error = error{ InvalidArgument, Unsupported };

/// getRectSubPix of a one-channel u8 or f32 image into an f32 window of
/// `ww` x `wh` centred at (cx, cy), with replicated borders.
pub fn getRectSubPix(src: anytype, ww: usize, wh: usize, cx: f32, cy: f32, dst: []f32) void {
    const T = @TypeOf(src).Elem;
    const center_x = cx - @as(f32, @floatFromInt(ww - 1)) * 0.5;
    const center_y = cy - @as(f32, @floatFromInt(wh - 1)) * 0.5;
    const ipx: i64 = @intFromFloat(@floor(center_x));
    const ipy: i64 = @intFromFloat(@floor(center_y));
    const sw: i64 = src.width;
    const sh: i64 = src.height;
    const wwi: i64 = @intCast(ww);
    const whi: i64 = @intCast(wh);
    if (T == u8 and 0 <= ipx and ipx + wwi < sw and 0 <= ipy and ipy + whi < sh) {
        // getRectSubPix_8u32f
        var a = center_x - @as(f32, @floatFromInt(ipx));
        const b = center_y - @as(f32, @floatFromInt(ipy));
        a = @max(a, 0.0001);
        const a12 = a * (1 - b);
        const a22 = a * b;
        const b1 = 1 - b;
        const b2 = b;
        const s: f64 = (1.0 - @as(f64, a)) / @as(f64, a);
        for (0..wh) |i| {
            const r0 = src.data[@as(usize, @intCast(ipy)) * src.stride + i * src.stride + @as(usize, @intCast(ipx)) ..];
            const r1 = r0[src.stride..];
            const d = dst[i * ww ..][0..ww];
            var prev: f32 = (1 - a) * (b1 * @as(f32, @floatFromInt(r0[0])) + b2 * @as(f32, @floatFromInt(r1[0])));
            for (0..ww) |j| {
                const t = a12 * @as(f32, @floatFromInt(r0[j + 1])) + a22 * @as(f32, @floatFromInt(r1[j + 1]));
                d[j] = prev + t;
                prev = @floatCast(@as(f64, t) * s);
            }
        }
        return;
    }
    // getRectSubPix_Cn_ (one channel)
    const a = center_x - @as(f32, @floatFromInt(ipx));
    const b = center_y - @as(f32, @floatFromInt(ipy));
    const a11 = (1 - a) * (1 - b);
    const a12 = a * (1 - b);
    const a21 = (1 - a) * b;
    const a22 = a * b;
    const b1 = 1 - b;
    const b2 = b;
    const at = struct {
        inline fn f(img: anytype, x: i64, y: i64) f32 {
            const v = img.data[@as(usize, @intCast(y)) * img.stride + @as(usize, @intCast(x))];
            return if (@typeInfo(@TypeOf(v)) == .float) v else @floatFromInt(v);
        }
    }.f;
    if (0 <= ipx and ipx < sw - wwi and 0 <= ipy and ipy < sh - whi) {
        for (0..wh) |i| {
            const y = ipy + @as(i64, @intCast(i));
            for (0..ww) |j| {
                const x = ipx + @as(i64, @intCast(j));
                dst[i * ww + j] = at(src, x, y) * a11 + at(src, x + 1, y) * a12 + at(src, x, y + 1) * a21 + at(src, x + 1, y + 1) * a22;
            }
        }
        return;
    }
    // adjustRect: the in-image part of the window and the row pointers.
    var rx: i64 = 0;
    var rw: i64 = 0;
    var ry: i64 = 0;
    var rh: i64 = 0;
    var base_x: i64 = 0;
    var base_y: i64 = 0;
    if (ipx >= 0) {
        base_x = ipx;
        rx = 0;
    } else {
        rx = -ipx;
        if (rx > wwi) rx = wwi;
    }
    if (ipx < sw - wwi) {
        rw = wwi;
    } else {
        rw = sw - ipx - 1;
        if (rw < 0) {
            base_x += rw;
            rw = 0;
        }
    }
    if (ipy >= 0) {
        base_y = ipy;
        ry = 0;
    } else ry = -ipy;
    if (ipy < sh - whi) {
        rh = whi;
    } else {
        rh = sh - ipy - 1;
        if (rh < 0) {
            base_y += rh;
            rh = 0;
        }
    }
    // `src` points at (base_x - rx, base_y); rows advance while i < rh.
    var row_y = base_y;
    const x0 = base_x - rx;
    for (0..wh) |iu| {
        const i: i64 = @intCast(iu);
        var row2_y = row_y + 1;
        if (i < ry or i >= rh) row2_y -= 1;
        const d = dst[iu * ww ..][0..ww];
        var s0 = at(src, x0 + rx, row_y) * b1 + at(src, x0 + rx, row2_y) * b2;
        for (0..@intCast(rx)) |j| d[j] = s0;
        s0 = at(src, x0 + rw, row_y) * b1 + at(src, x0 + rw, row2_y) * b2;
        for (@intCast(rw)..ww) |j| d[j] = s0;
        var j = rx;
        while (j < rw) : (j += 1) {
            d[@intCast(j)] = at(src, x0 + j, row_y) * a11 + at(src, x0 + j + 1, row_y) * a12 + at(src, x0 + j, row2_y) * a21 + at(src, x0 + j + 1, row2_y) * a22;
        }
        if (i < rh) row_y = row2_y;
    }
}

pub const SubPixOptions = struct {
    /// Half window (OpenCV's winSize): the window is (2w+1) x (2h+1).
    win_w: u32 = 5,
    win_h: u32 = 5,
    /// Half size of the dead zone in the middle (-1: none).
    zero_w: i32 = -1,
    zero_h: i32 = -1,
    /// TermCriteria: iteration cap (0 = no count criterion) and epsilon
    /// (< 0 = no epsilon criterion).
    max_iters: u32 = 30,
    epsilon: f64 = 0.001,
};

/// OpenCV's cornerSubPix on a one-channel u8 or f32 image; refines
/// `corners` in place.
pub fn cornerSubPix(gpa: std.mem.Allocator, src: anytype, corners: []Point2f, o: SubPixOptions) (Error || std.mem.Allocator.Error)!void {
    if (src.channels != 1) return error.Unsupported;
    if (o.win_w == 0 or o.win_h == 0) return error.InvalidArgument;
    if (src.width < o.win_w * 2 + 5 or src.height < o.win_h * 2 + 5) return error.InvalidArgument;
    const max_iters: u32 = if (o.max_iters > 0) @min(@max(o.max_iters, 1), 100) else 100;
    var eps: f64 = if (o.epsilon >= 0) o.epsilon else 0;
    eps *= eps;
    const ww: usize = o.win_w * 2 + 1;
    const wh: usize = o.win_h * 2 + 1;
    const mask = try gpa.alloc(f32, ww * wh + (ww + 2) * (wh + 2));
    defer gpa.free(mask);
    const buf = mask[ww * wh ..];
    for (0..wh) |i| {
        const y: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(i)) - o.win_h)) / @as(f32, @floatFromInt(o.win_h));
        const vy = @exp(-y * y);
        for (0..ww) |j| {
            const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(j)) - o.win_w)) / @as(f32, @floatFromInt(o.win_w));
            mask[i * ww + j] = vy * @exp(-x * x);
        }
    }
    if (o.zero_w >= 0 and o.zero_h >= 0 and o.zero_w * 2 + 1 < ww and o.zero_h * 2 + 1 < wh) {
        const zw: usize = @intCast(o.zero_w);
        const zh: usize = @intCast(o.zero_h);
        for (o.win_h - zh..o.win_h + zh + 1) |i| for (o.win_w - zw..o.win_w + zw + 1) |j| {
            mask[i * ww + j] = 0;
        };
    }
    const fw: f32 = @floatFromInt(src.width);
    const fh: f32 = @floatFromInt(src.height);
    for (corners) |*pt| {
        const cT = pt.*;
        if (!(cT.x >= 0 and cT.y >= 0 and cT.x < fw and cT.y < fh)) return error.InvalidArgument;
        var cI = cT;
        var iter: u32 = 0;
        var err: f64 = 0;
        while (true) {
            getRectSubPix(src, ww + 2, wh + 2, cI.x, cI.y, buf);
            var a: f64 = 0;
            var b: f64 = 0;
            var c: f64 = 0;
            var bb1: f64 = 0;
            var bb2: f64 = 0;
            const st = ww + 2;
            for (0..wh) |i| {
                const py: f64 = @floatFromInt(@as(i64, @intCast(i)) - o.win_h);
                for (0..ww) |j| {
                    const q = (i + 1) * st + 1 + j;
                    const m: f64 = mask[i * ww + j];
                    const tgx: f64 = buf[q + 1] - buf[q - 1];
                    const tgy: f64 = buf[q + st] - buf[q - st];
                    const gxx = tgx * tgx * m;
                    const gxy = tgx * tgy * m;
                    const gyy = tgy * tgy * m;
                    const px: f64 = @floatFromInt(@as(i64, @intCast(j)) - o.win_w);
                    a += gxx;
                    b += gxy;
                    c += gyy;
                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }
            const det = a * c - b * b;
            if (@abs(det) <= std.math.floatEps(f64) * std.math.floatEps(f64)) break;
            const scale = 1.0 / det;
            const cI2 = Point2f{
                .x = @floatCast(cI.x + c * scale * bb1 - b * scale * bb2),
                .y = @floatCast(cI.y - b * scale * bb1 + a * scale * bb2),
            };
            const dx = cI2.x - cI.x;
            const dy = cI2.y - cI.y;
            err = dx * dx + dy * dy;
            cI = cI2;
            if (cI.x < 0 or cI.x >= fw or cI.y < 0 or cI.y >= fh) break;
            iter += 1;
            if (!(iter < max_iters and err > eps)) break;
        }
        if (@abs(cI.x - cT.x) > @as(f32, @floatFromInt(o.win_w)) or @abs(cI.y - cT.y) > @as(f32, @floatFromInt(o.win_h))) cI = cT;
        pt.* = cI;
    }
}

test "cornerSubPix finds a checkerboard corner" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).init(gpa, 40, 40, 1);
    defer img.deinit(gpa);
    for (0..40) |y| for (0..40) |x| {
        img.row(y)[x] = if ((x < 20) == (y < 20)) 200 else 30;
    };
    var pts = [_]Point2f{.{ .x = 21.3, .y = 18.6 }};
    try cornerSubPix(gpa, img, &pts, .{});
    try std.testing.expectApproxEqAbs(@as(f32, 19.5), pts[0].x, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 19.5), pts[0].y, 0.05);
}
