//! Morphology: erode, dilate and the compound operations of morphologyEx.
//! Pixels outside the image are ignored (OpenCV's default border value), so
//! a rectangular element is separable into a row pass and a column pass.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error || error{InvalidKernel};

pub const Shape = enum(u8) { rect = 0, cross = 1, ellipse = 2 };

/// A structuring element: `w` x `h` mask (row-major, nonzero = on) and an
/// anchor (defaults to the centre).
pub const Element = struct {
    w: u32,
    h: u32,
    mask: []const u8,
    ax: u32,
    ay: u32,

    pub fn isRect(e: Element) bool {
        for (e.mask) |m| if (m == 0) return false;
        return true;
    }
};

/// OpenCV's getStructuringElement. `buf` must hold w*h bytes.
pub fn structuringElement(shape: Shape, w: u32, h: u32, buf: []u8) Element {
    const ax = w / 2;
    const ay = h / 2;
    @memset(buf[0 .. w * h], 0);
    const r = ay;
    const c = ax;
    const inv_r2: f64 = if (r > 0) 1.0 / @as(f64, @floatFromInt(r * r)) else 0;
    for (0..h) |i| {
        var j1: usize = 0;
        var j2: usize = 0;
        if (shape == .rect or (shape == .cross and i == ay)) {
            j2 = w;
        } else if (shape == .cross) {
            j1 = ax;
            j2 = j1 + 1;
        } else {
            const dy: i64 = @as(i64, @intCast(i)) - @as(i64, r);
            if (@abs(dy) <= r) {
                const dx_f = @as(f64, @floatFromInt(c)) * @sqrt(@as(f64, @floatFromInt((r * r) - @as(u64, @intCast(dy * dy)))) * inv_r2);
                const dx: i64 = @intFromFloat(@round(dx_f));
                j1 = @intCast(@max(@as(i64, c) - dx, 0));
                j2 = @intCast(@min(@as(i64, c) + dx + 1, @as(i64, w)));
            }
        }
        for (j1..j2) |j| buf[i * w + j] = 1;
    }
    return .{ .w = w, .h = h, .mask = buf[0 .. w * h], .ax = ax, .ay = ay };
}

const Op = enum { min, max };

pub fn erode(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    return morph(.min, exec, gpa, src, dst, el, iterations);
}

pub fn dilate(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    return morph(.max, exec, gpa, src, dst, el, iterations);
}

fn morph(comptime op: Op, exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element, iterations: u32) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (el.w == 0 or el.h == 0) return error.InvalidKernel;
    if (iterations == 0) return dst.copyFrom(src);
    var cur = src;
    var tmp: ?Image(T) = null;
    defer if (tmp) |*t| t.deinit(gpa);
    for (0..iterations) |it| {
        // Ping-pong through a temporary when iterating.
        const out = if (it + 1 == iterations) dst else blk: {
            if (tmp == null) tmp = try Image(T).initLike(gpa, src);
            break :blk if (cur.data == tmp.?.data) dst else tmp.?;
        };
        if (el.isRect()) try rectPass(op, exec, gpa, cur, out, el) else try generalPass(op, exec, cur, out, el);
        cur = out;
    }
}

fn rectPass(comptime op: Op, exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), el: Element) Error!void {
    const T = @TypeOf(src).Elem;
    // Horizontal pass into a temporary, then vertical pass into dst.
    var h = try Image(T).initLike(gpa, src);
    defer h.deinit(gpa);
    const Ctx = struct { s: Image(T), d: Image(T), el: Element, gpa: Allocator };
    exec.parallelFor(src.height, 16, Ctx{ .s = src, .d = h, .el = el, .gpa = gpa }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const n = c.s.rowLen();
            const left = c.el.ax;
            const right = c.el.w - 1 - c.el.ax;
            const pad = c.gpa.alloc(T, n + (left + right) * cn) catch @panic("out of memory");
            defer c.gpa.free(pad);
            for (y0..y1) |y| {
                const r = c.s.row(y);
                @memcpy(pad[left * cn ..][0..n], r);
                // Replicating the edge is equivalent to ignoring outside pixels.
                for (0..left) |i| @memcpy(pad[i * cn ..][0..cn], r[0..cn]);
                for (0..right) |i| @memcpy(pad[(left + c.s.width + i) * cn ..][0..cn], r[n - cn ..][0..cn]);
                const d = c.d.row(y);
                @memcpy(d, pad[0..n]);
                for (1..c.el.w) |k| {
                    const sp = pad[k * cn ..][0..n];
                    for (d, sp) |*o, v| o.* = if (op == .min) @min(o.*, v) else @max(o.*, v);
                }
            }
        }
    }.run);
    const VCtx = struct { s: Image(T), d: Image(T), el: Element };
    exec.parallelFor(src.height, 16, VCtx{ .s = h, .d = dst, .el = el }, struct {
        fn run(c: VCtx, y0: usize, y1: usize) void {
            const hgt: i64 = c.s.height;
            for (y0..y1) |y| {
                const d = c.d.row(y);
                const top = @max(@as(i64, @intCast(y)) - c.el.ay, 0);
                const bot = @min(@as(i64, @intCast(y)) - c.el.ay + c.el.h, hgt);
                @memcpy(d, c.s.row(@intCast(top)));
                var yy = top + 1;
                while (yy < bot) : (yy += 1) {
                    for (d, c.s.row(@intCast(yy))) |*o, v| o.* = if (op == .min) @min(o.*, v) else @max(o.*, v);
                }
            }
        }
    }.run);
}

fn generalPass(comptime op: Op, exec: Exec, src: anytype, dst: @TypeOf(src), el: Element) Error!void {
    const T = @TypeOf(src).Elem;
    const Ctx = struct { s: Image(T), d: Image(T), el: Element };
    exec.parallelFor(src.height, 8, Ctx{ .s = src, .d = dst, .el = el }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const cn: usize = c.s.channels;
            const W: i64 = c.s.width;
            const H: i64 = c.s.height;
            const init: T = if (op == .min) maxOf(T) else minOf(T);
            for (y0..y1) |y| {
                const d = c.d.row(y);
                @memset(d, init);
                for (0..c.el.h) |ky| {
                    const sy = @as(i64, @intCast(y)) + @as(i64, @intCast(ky)) - c.el.ay;
                    if (sy < 0 or sy >= H) continue;
                    const sr = c.s.row(@intCast(sy));
                    for (0..c.el.w) |kx| {
                        if (c.el.mask[ky * c.el.w + kx] == 0) continue;
                        const off = @as(i64, @intCast(kx)) - c.el.ax;
                        const x0: i64 = @max(0, -off);
                        const x1: i64 = @min(W, W - off);
                        if (x1 <= x0) continue;
                        const dd = d[@as(usize, @intCast(x0)) * cn .. @as(usize, @intCast(x1)) * cn];
                        const ss = sr[@as(usize, @intCast(x0 + off)) * cn .. @as(usize, @intCast(x1 + off)) * cn];
                        for (dd, ss) |*o, v| o.* = if (op == .min) @min(o.*, v) else @max(o.*, v);
                    }
                }
            }
        }
    }.run);
}

fn maxOf(comptime T: type) T {
    return if (@typeInfo(T) == .float) std.math.inf(T) else std.math.maxInt(T);
}

fn minOf(comptime T: type) T {
    return if (@typeInfo(T) == .float) -std.math.inf(T) else std.math.minInt(T);
}

pub const MorphOp = enum(u8) { erode = 0, dilate = 1, open = 2, close = 3, gradient = 4, tophat = 5, blackhat = 6 };

/// morphologyEx. Uses one temporary image for compound operations.
pub fn morphologyEx(exec: Exec, gpa: Allocator, src: anytype, dst: @TypeOf(src), op: MorphOp, el: Element, iterations: u32) Error!void {
    const T = @TypeOf(src).Elem;
    switch (op) {
        .erode => return erode(exec, gpa, src, dst, el, iterations),
        .dilate => return dilate(exec, gpa, src, dst, el, iterations),
        else => {},
    }
    var t = try Image(T).initLike(gpa, src);
    defer t.deinit(gpa);
    switch (op) {
        .open => {
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, t, dst, el, iterations);
        },
        .close => {
            try dilate(exec, gpa, src, t, el, iterations);
            try erode(exec, gpa, t, dst, el, iterations);
        },
        .gradient => {
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, src, dst, el, iterations);
            subInPlace(T, dst, t, dst);
        },
        .tophat => {
            var t2 = try Image(T).initLike(gpa, src);
            defer t2.deinit(gpa);
            try erode(exec, gpa, src, t, el, iterations);
            try dilate(exec, gpa, t, t2, el, iterations);
            subInPlace(T, src, t2, dst);
        },
        .blackhat => {
            var t2 = try Image(T).initLike(gpa, src);
            defer t2.deinit(gpa);
            try dilate(exec, gpa, src, t, el, iterations);
            try erode(exec, gpa, t, t2, el, iterations);
            subInPlace(T, t2, src, dst);
        },
        .erode, .dilate => unreachable,
    }
}

/// d = saturate(a - b)
fn subInPlace(comptime T: type, a: Image(T), b: Image(T), d: Image(T)) void {
    for (0..d.height) |y| for (a.row(y), b.row(y), d.row(y)) |x, z, *o| {
        o.* = if (@typeInfo(T) == .float) x - z else if (x > z) x - z else 0;
    };
}

test "erode/dilate rect vs general" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 13, 9, 1);
    defer src.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(1);
    prng.random().bytes(src.slice());
    var buf: [15]u8 = undefined;
    const el = structuringElement(.rect, 5, 3, &buf);
    var a = try Image(u8).initLike(gpa, src);
    defer a.deinit(gpa);
    var b = try Image(u8).initLike(gpa, src);
    defer b.deinit(gpa);
    try erode(.serial, gpa, src, a, el, 1);
    try generalPass(.min, .serial, src, b, el);
    try std.testing.expectEqualSlices(u8, a.slice(), b.slice());
    try dilate(.serial, gpa, src, a, el, 2);
    var c = try Image(u8).initLike(gpa, src);
    defer c.deinit(gpa);
    try generalPass(.max, .serial, src, b, el);
    try generalPass(.max, .serial, b, c, el);
    try std.testing.expectEqualSlices(u8, a.slice(), c.slice());
}
