//! Geometric transforms: flip, rotate by multiples of 90 degrees,
//! transpose, borders, affine and perspective warps.

const std = @import("std");
const core = @import("../core/core.zig");
const border_mod = @import("border.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;

pub const Border = border_mod.Border;
pub const Error = core.Error || error{Unsupported};

pub const FlipCode = enum { vertical, horizontal, both };

/// flip(src, dst, code): vertical = around the x axis (OpenCV code 0),
/// horizontal = around the y axis (1), both = (-1).
pub fn flip(exec: Exec, src: anytype, dst: @TypeOf(src), code: FlipCode) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { s: Image(T), d: Image(T), code: FlipCode };
    exec.parallelFor(src.height, 32, Ctx{ .s = src, .d = dst, .code = code }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const h = c.s.height;
            for (y0..y1) |y| {
                const sy = if (c.code == .horizontal) y else h - 1 - y;
                const s = c.s.row(sy);
                const d = c.d.row(y);
                if (c.code == .vertical) {
                    @memcpy(d, s);
                    continue;
                }
                const w = c.s.width;
                switch (cn) {
                    inline 1, 2, 3, 4 => |k| for (0..w) |x| {
                        inline for (0..k) |ch| d[x * k + ch] = s[(w - 1 - x) * k + ch];
                    },
                    else => for (0..w) |x| @memcpy(d[x * cn ..][0..cn], s[(w - 1 - x) * cn ..][0..cn]),
                }
            }
        }
    }.run);
}

/// dst (h x w) = src (w x h) transposed.
pub fn transpose(exec: Exec, src: anytype, dst: @TypeOf(src)) Error!void {
    return rot(exec, src, dst, .transpose);
}

pub const Rotate = enum(u8) { cw90 = 0, r180 = 1, ccw90 = 2 };

pub fn rotate(exec: Exec, src: anytype, dst: @TypeOf(src), r: Rotate) Error!void {
    return switch (r) {
        .r180 => flip(exec, src, dst, .both),
        .cw90 => rot(exec, src, dst, .cw),
        .ccw90 => rot(exec, src, dst, .ccw),
    };
}

const RotKind = enum { transpose, cw, ccw };

/// Blocked so both source reads and destination writes stay in cache.
fn rot(exec: Exec, src: anytype, dst: @TypeOf(src), comptime kind: RotKind) Error!void {
    const T = @TypeOf(src).Elem;
    if (src.width != dst.height or src.height != dst.width or src.channels != dst.channels) return error.SizeMismatch;
    const B = 32;
    const Ctx = struct { s: Image(T), d: Image(T) };
    const blocks_y = (dst.height + B - 1) / B;
    exec.parallelFor(blocks_y, 1, Ctx{ .s = src, .d = dst }, struct {
        fn run(c: Ctx, b0: usize, b1: usize) void {
            const cn: usize = c.s.channels;
            const sw = c.s.width;
            const sh = c.s.height;
            for (b0..b1) |by| {
                const dy0 = by * B;
                const dy1 = @min(dy0 + B, c.d.height);
                var dx0: usize = 0;
                while (dx0 < c.d.width) : (dx0 += B) {
                    const dx1 = @min(dx0 + B, c.d.width);
                    for (dy0..dy1) |dy| {
                        const d = c.d.row(dy);
                        for (dx0..dx1) |dx| {
                            // dst(dx, dy) <- src(sx, sy)
                            const sx, const sy = switch (kind) {
                                .transpose => .{ dy, dx },
                                .cw => .{ dy, sh - 1 - dx },
                                .ccw => .{ sw - 1 - dy, dx },
                            };
                            const sp = c.s.data + sy * c.s.stride + sx * cn;
                            switch (cn) {
                                inline 1, 3, 4 => |k| inline for (0..k) |ch| {
                                    d[dx * k + ch] = sp[ch];
                                },
                                else => @memcpy(d[dx * cn ..][0..cn], sp[0..cn]),
                            }
                        }
                    }
                }
            }
        }
    }.run);
}

/// copyMakeBorder: dst is src padded by (top, bottom, left, right).
pub fn copyMakeBorder(src: anytype, dst: @TypeOf(src), top: u32, left: u32, border: Border, value: []const f64) Error!void {
    const T = @TypeOf(src).Elem;
    const cn: usize = src.channels;
    if (dst.channels != src.channels or dst.width < src.width + left or dst.height < src.height + top) return error.SizeMismatch;
    var cval: [4]T = undefined;
    for (0..@min(cn, 4)) |ch| cval[ch] = sat.cast(T, if (ch < value.len) value[ch] else 0);
    for (0..dst.height) |y| {
        const d = dst.row(y);
        const sy = border_mod.map(@as(isize, @intCast(y)) - @as(isize, @intCast(top)), src.height, border);
        for (0..dst.width) |x| {
            const sx = border_mod.map(@as(isize, @intCast(x)) - @as(isize, @intCast(left)), src.width, border);
            const o = d[x * cn ..][0..cn];
            if (sx != null and sy != null) {
                @memcpy(o, src.pixel(sx.?, sy.?));
            } else for (o, 0..) |*p, ch| {
                p.* = if (ch < 4) cval[ch] else 0;
            }
        }
    }
}

pub const WarpInterp = enum { nearest, linear };

/// warpAffine with the forward 2x3 matrix `m` (dst = M * src), like OpenCV
/// without WARP_INVERSE_MAP. Computed in f64 coordinates, f32 blending.
pub fn warpAffine(exec: Exec, src: anytype, dst: @TypeOf(src), m: [6]f64, interp: WarpInterp, border: Border, value: f64) Error!void {
    const inv = invertAffine(m) orelse return error.Unsupported;
    return warpGeneric(exec, src, dst, .{ inv[0], inv[1], inv[2], inv[3], inv[4], inv[5], 0, 0, 1 }, false, interp, border, value);
}

/// warpPerspective with the forward 3x3 matrix `m`.
pub fn warpPerspective(exec: Exec, src: anytype, dst: @TypeOf(src), m: [9]f64, interp: WarpInterp, border: Border, value: f64) Error!void {
    const inv = invert3(m) orelse return error.Unsupported;
    return warpGeneric(exec, src, dst, inv, true, interp, border, value);
}

pub fn invertAffine(m: [6]f64) ?[6]f64 {
    const det = m[0] * m[4] - m[1] * m[3];
    if (det == 0) return null;
    const d = 1.0 / det;
    const a11 = m[4] * d;
    const a22 = m[0] * d;
    const a12 = -m[1] * d;
    const a21 = -m[3] * d;
    const b1 = -a11 * m[2] - a12 * m[5];
    const b2 = -a21 * m[2] - a22 * m[5];
    return .{ a11, a12, b1, a21, a22, b2 };
}

pub fn invert3(m: [9]f64) ?[9]f64 {
    const det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (det == 0) return null;
    const d = 1.0 / det;
    return .{
        (m[4] * m[8] - m[5] * m[7]) * d, (m[2] * m[7] - m[1] * m[8]) * d, (m[1] * m[5] - m[2] * m[4]) * d,
        (m[5] * m[6] - m[3] * m[8]) * d, (m[0] * m[8] - m[2] * m[6]) * d, (m[2] * m[3] - m[0] * m[5]) * d,
        (m[3] * m[7] - m[4] * m[6]) * d, (m[1] * m[6] - m[0] * m[7]) * d, (m[0] * m[4] - m[1] * m[3]) * d,
    };
}

fn warpGeneric(exec: Exec, src: anytype, dst: @TypeOf(src), inv: [9]f64, persp: bool, interp: WarpInterp, border: Border, value: f64) Error!void {
    const T = @TypeOf(src).Elem;
    if (src.channels != dst.channels) return error.SizeMismatch;
    const Ctx = struct { s: Image(T), d: Image(T), m: [9]f64, persp: bool, interp: WarpInterp, border: Border, value: T };
    exec.parallelFor(dst.height, 8, Ctx{ .s = src, .d = dst, .m = inv, .persp = persp, .interp = interp, .border = border, .value = sat.cast(T, value) }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            for (y0..y1) |y| {
                const d = c.d.row(y);
                const fy: f64 = @floatFromInt(y);
                for (0..c.d.width) |x| {
                    const fx: f64 = @floatFromInt(x);
                    var sx = c.m[0] * fx + c.m[1] * fy + c.m[2];
                    var sy = c.m[3] * fx + c.m[4] * fy + c.m[5];
                    if (c.persp) {
                        const w = c.m[6] * fx + c.m[7] * fy + c.m[8];
                        const iw = if (w != 0) 1.0 / w else 0;
                        sx *= iw;
                        sy *= iw;
                    }
                    const o = d[x * cn ..][0..cn];
                    if (c.interp == .nearest) {
                        const ix: isize = @intFromFloat(@round(std.math.clamp(sx, -1e9, 1e9)));
                        const iy: isize = @intFromFloat(@round(std.math.clamp(sy, -1e9, 1e9)));
                        const mx = border_mod.map(ix, c.s.width, c.border);
                        const my = border_mod.map(iy, c.s.height, c.border);
                        if (mx != null and my != null) @memcpy(o, c.s.pixel(mx.?, my.?)) else @memset(o, c.value);
                        continue;
                    }
                    const x0f = @floor(std.math.clamp(sx, -1e9, 1e9));
                    const y0f = @floor(std.math.clamp(sy, -1e9, 1e9));
                    const ax: f32 = @floatCast(sx - x0f);
                    const ay: f32 = @floatCast(sy - y0f);
                    const ix: isize = @intFromFloat(x0f);
                    const iy: isize = @intFromFloat(y0f);
                    // Entirely outside with a constant border: fill.
                    if (c.border == .constant and (ix < -1 or iy < -1 or ix >= c.s.width or iy >= c.s.height)) {
                        @memset(o, c.value);
                        continue;
                    }
                    const xs = [2]?usize{ border_mod.map(ix, c.s.width, c.border), border_mod.map(ix + 1, c.s.width, c.border) };
                    const ys = [2]?usize{ border_mod.map(iy, c.s.height, c.border), border_mod.map(iy + 1, c.s.height, c.border) };
                    const w = [4]f32{ (1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay };
                    for (0..cn) |ch| {
                        var acc: f32 = 0;
                        inline for (0..4) |k| {
                            const px = xs[k & 1];
                            const py = ys[k >> 1];
                            const v: f32 = if (px != null and py != null) toF(T, c.s.at(px.?, py.?, ch)) else toF(T, c.value);
                            acc += v * w[k];
                        }
                        o[ch] = sat.cast(T, acc);
                    }
                }
            }
        }
    }.run);
}

inline fn toF(comptime T: type, v: T) f32 {
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

/// OpenCV's getRotationMatrix2D: angle in degrees (counter-clockwise).
pub fn rotationMatrix2D(cx: f64, cy: f64, angle_deg: f64, scale: f64) [6]f64 {
    const a = angle_deg * std.math.pi / 180.0;
    const alpha = @cos(a) * scale;
    const beta = @sin(a) * scale;
    return .{ alpha, beta, (1 - alpha) * cx - beta * cy, -beta, alpha, beta * cx + (1 - alpha) * cy };
}

test "rotate and flip" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 3, 2, 1);
    defer src.deinit(gpa);
    @memcpy(src.slice(), &[_]u8{ 1, 2, 3, 4, 5, 6 });
    var r = try Image(u8).init(gpa, 2, 3, 1);
    defer r.deinit(gpa);
    try rotate(.serial, src, r, .cw90);
    try std.testing.expectEqualSlices(u8, &.{ 4, 1, 5, 2, 6, 3 }, r.slice());
    try rotate(.serial, src, r, .ccw90);
    try std.testing.expectEqualSlices(u8, &.{ 3, 6, 2, 5, 1, 4 }, r.slice());
    try transpose(.serial, src, r);
    try std.testing.expectEqualSlices(u8, &.{ 1, 4, 2, 5, 3, 6 }, r.slice());
    var f = try Image(u8).init(gpa, 3, 2, 1);
    defer f.deinit(gpa);
    try flip(.serial, src, f, .horizontal);
    try std.testing.expectEqualSlices(u8, &.{ 3, 2, 1, 6, 5, 4 }, f.slice());
    try flip(.serial, src, f, .both);
    try std.testing.expectEqualSlices(u8, &.{ 6, 5, 4, 3, 2, 1 }, f.slice());
}
