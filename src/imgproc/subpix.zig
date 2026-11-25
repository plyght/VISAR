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
        const L = 8;
        const VF = @Vector(L, f32);
        const VD = @Vector(L, f64);
        for (0..wh) |i| {
            const r0 = src.data[@as(usize, @intCast(ipy)) * src.stride + i * src.stride + @as(usize, @intCast(ipx)) ..];
            const r1 = r0[src.stride..];
            const d = dst[i * ww ..][0..ww];
            var prev: f32 = (1 - a) * (b1 * @as(f32, @floatFromInt(r0[0])) + b2 * @as(f32, @floatFromInt(r1[0])));
            // dst[j] = t[j-1] * s (rounded to f32; prev for j = 0) + t[j]:
            // the same values as OpenCV's running loop, L at a time.
            var j: usize = 0;
            while (j + L <= ww) : (j += L) {
                var top: VF = undefined;
                var bot: VF = undefined;
                inline for (0..L) |l| {
                    top[l] = @floatFromInt(r0[j + 1 + l]);
                    bot[l] = @floatFromInt(r1[j + 1 + l]);
                }
                const t = @as(VF, @splat(a12)) * top + @as(VF, @splat(a22)) * bot;
                const ts: VF = @floatCast(@as(VD, t) * @as(VD, @splat(s)));
                const shifted = std.simd.shiftElementsRight(ts, 1, prev);
                d[j..][0..L].* = shifted + t;
                prev = ts[L - 1];
            }
            while (j < ww) : (j += 1) {
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
    // The mask rows are padded with zeros to whole vectors; the window
    // buffer gets a vector of slack for the padded reads.
    const L = 4;
    const V = @Vector(L, f64);
    const VF = @Vector(L, f32);
    const wwp = (ww + L - 1) / L * L;
    const mask = try gpa.alloc(f32, wwp * wh + (ww + 2) * (wh + 2) + L);
    defer gpa.free(mask);
    @memset(mask, 0);
    const buf = mask[wwp * wh ..];
    for (0..wh) |i| {
        const y: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(i)) - o.win_h)) / @as(f32, @floatFromInt(o.win_h));
        const vy = @exp(-y * y);
        for (0..ww) |j| {
            const x: f32 = @as(f32, @floatFromInt(@as(i64, @intCast(j)) - o.win_w)) / @as(f32, @floatFromInt(o.win_w));
            mask[i * wwp + j] = vy * @exp(-x * x);
        }
    }
    if (o.zero_w >= 0 and o.zero_h >= 0 and o.zero_w * 2 + 1 < ww and o.zero_h * 2 + 1 < wh) {
        const zw: usize = @intCast(o.zero_w);
        const zh: usize = @intCast(o.zero_h);
        for (o.win_h - zh..o.win_h + zh + 1) |i| for (o.win_w - zw..o.win_w + zw + 1) |j| {
            mask[i * wwp + j] = 0;
        };
    }
    var pxs: [L]f64 = undefined;
    for (&pxs, 0..) |*v, l| v.* = @floatFromInt(l);
    const px0: V = pxs;
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
            // OpenCV sums these in f64 one pixel at a time; VISAR keeps L
            // partial sums (the same terms, added in a different order).
            var va: V = @splat(0);
            var vb: V = @splat(0);
            var vc: V = @splat(0);
            var vb1: V = @splat(0);
            var vb2: V = @splat(0);
            const st = ww + 2;
            for (0..wh) |i| {
                const py: V = @splat(@floatFromInt(@as(i64, @intCast(i)) - o.win_h));
                var j: usize = 0;
                while (j < ww) : (j += L) {
                    const q = (i + 1) * st + 1 + j;
                    const m: V = @floatCast(@as(VF, mask[i * wwp + j ..][0..L].*));
                    const r: VF = buf[q + 1 ..][0..L].*;
                    const l: VF = buf[q - 1 ..][0..L].*;
                    const dn: VF = buf[q + st ..][0..L].*;
                    const up: VF = buf[q - st ..][0..L].*;
                    const tgx: V = @floatCast(r - l);
                    const tgy: V = @floatCast(dn - up);
                    const gxx = tgx * tgx * m;
                    const gxy = tgx * tgy * m;
                    const gyy = tgy * tgy * m;
                    const px = px0 + @as(V, @splat(@floatFromInt(@as(i64, @intCast(j)) - o.win_w)));
                    va += gxx;
                    vb += gxy;
                    vc += gyy;
                    vb1 += gxx * px + gxy * py;
                    vb2 += gxy * px + gyy * py;
                }
            }
            const a = @reduce(.Add, va);
            const b = @reduce(.Add, vb);
            const c = @reduce(.Add, vc);
            const bb1 = @reduce(.Add, vb1);
            const bb2 = @reduce(.Add, vb2);
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
