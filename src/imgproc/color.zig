//! Colour space conversions (cvtColor). Integer paths reproduce OpenCV's
//! fixed-point arithmetic so 8-bit results match cv2 exactly.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;

pub const Error = core.Error || error{Unsupported};

/// Conversion codes. Names follow OpenCV's COLOR_* constants.
pub const Code = enum {
    bgr2rgb,
    rgb2bgr,
    bgr2bgra,
    rgb2rgba,
    bgra2bgr,
    rgba2rgb,
    bgr2rgba,
    rgb2bgra,
    rgba2bgr,
    bgra2rgb,
    bgra2rgba,
    rgba2bgra,
    bgr2gray,
    rgb2gray,
    bgra2gray,
    rgba2gray,
    gray2bgr,
    gray2rgb,
    gray2bgra,
    gray2rgba,
    bgr2hsv,
    rgb2hsv,
    hsv2bgr,
    hsv2rgb,
    bgr2ycrcb,
    rgb2ycrcb,
    ycrcb2bgr,
    ycrcb2rgb,

    /// Channels in and out, and where blue sits (0 = BGR order, 2 = RGB).
    const Info = struct { scn: u16, dcn: u16, sb: u8 = 0, db: u8 = 0, kind: Kind };
    const Kind = enum { reorder, gray, from_gray, hsv, from_hsv, ycrcb, from_ycrcb };

    fn info(comptime c: Code) Info {
        return switch (c) {
            .bgr2rgb, .rgb2bgr => .{ .scn = 3, .dcn = 3, .sb = 0, .db = 2, .kind = .reorder },
            .bgr2bgra, .rgb2rgba => .{ .scn = 3, .dcn = 4, .kind = .reorder },
            .bgra2bgr, .rgba2rgb => .{ .scn = 4, .dcn = 3, .kind = .reorder },
            .bgr2rgba, .rgb2bgra => .{ .scn = 3, .dcn = 4, .sb = 0, .db = 2, .kind = .reorder },
            .bgra2rgb, .rgba2bgr => .{ .scn = 4, .dcn = 3, .sb = 0, .db = 2, .kind = .reorder },
            .bgra2rgba, .rgba2bgra => .{ .scn = 4, .dcn = 4, .sb = 0, .db = 2, .kind = .reorder },
            .bgr2gray => .{ .scn = 3, .dcn = 1, .sb = 0, .kind = .gray },
            .rgb2gray => .{ .scn = 3, .dcn = 1, .sb = 2, .kind = .gray },
            .bgra2gray => .{ .scn = 4, .dcn = 1, .sb = 0, .kind = .gray },
            .rgba2gray => .{ .scn = 4, .dcn = 1, .sb = 2, .kind = .gray },
            .gray2bgr, .gray2rgb => .{ .scn = 1, .dcn = 3, .kind = .from_gray },
            .gray2bgra, .gray2rgba => .{ .scn = 1, .dcn = 4, .kind = .from_gray },
            .bgr2hsv => .{ .scn = 3, .dcn = 3, .sb = 0, .kind = .hsv },
            .rgb2hsv => .{ .scn = 3, .dcn = 3, .sb = 2, .kind = .hsv },
            .hsv2bgr => .{ .scn = 3, .dcn = 3, .db = 0, .kind = .from_hsv },
            .hsv2rgb => .{ .scn = 3, .dcn = 3, .db = 2, .kind = .from_hsv },
            .bgr2ycrcb => .{ .scn = 3, .dcn = 3, .sb = 0, .kind = .ycrcb },
            .rgb2ycrcb => .{ .scn = 3, .dcn = 3, .sb = 2, .kind = .ycrcb },
            .ycrcb2bgr => .{ .scn = 3, .dcn = 3, .db = 0, .kind = .from_ycrcb },
            .ycrcb2rgb => .{ .scn = 3, .dcn = 3, .db = 2, .kind = .from_ycrcb },
        };
    }

    pub fn srcChannels(c: Code) u16 {
        return switch (c) {
            inline else => |cc| comptime cc.info().scn,
        };
    }

    pub fn dstChannels(c: Code) u16 {
        return switch (c) {
            inline else => |cc| comptime cc.info().dcn,
        };
    }
};

/// Converts `src` to `dst` (same size, channel counts given by `code`).
/// Supported depths: u8, u16 and f32 (HSV/YCrCb: u8 and f32).
pub fn cvtColor(exec: Exec, src: anytype, dst: @TypeOf(src), code: Code) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameSize(dst)) return error.SizeMismatch;
    switch (code) {
        inline else => |c| {
            const inf = comptime c.info();
            if (src.channels != inf.scn or dst.channels != inf.dcn) return error.UnsupportedChannels;
            if (comptime !supported(T, inf.kind)) return error.Unsupported;
            const Ctx = struct { s: Image(T), d: Image(T) };
            const grain = @max(1, (1 << 15) / src.width);
            exec.parallelFor(src.height, grain, Ctx{ .s = src, .d = dst }, struct {
                fn run(ctx: Ctx, y0: usize, y1: usize) void {
                    for (y0..y1) |y| convertRow(T, inf, ctx.s.row(y).ptr, ctx.d.row(y).ptr, ctx.s.width);
                }
            }.run);
        },
    }
}

/// Depths each conversion family implements.
fn supported(comptime T: type, comptime kind: Code.Kind) bool {
    return switch (kind) {
        .reorder, .from_gray => true,
        .gray => T == u8 or T == u16 or @typeInfo(T) == .float,
        .hsv, .from_hsv, .ycrcb, .from_ycrcb => T == u8 or T == f32,
    };
}

fn maxVal(comptime T: type) comptime_float {
    return if (@typeInfo(T) == .float) 1.0 else @floatFromInt(std.math.maxInt(T));
}

// Fixed-point constants: OpenCV's yuv_shift = 14 (YCrCb) and gray_shift =
// 15 (RGB -> gray, 8- and 16-bit).
const shift = 14;
const R2Y = 4899;
const G2Y = 9617;
const B2Y = 1868;
const gray_shift = 15;
const R2G = 9798;
const G2G = 19235;
const B2G = 3735;

inline fn convertRow(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) void {
    switch (inf.kind) {
        .reorder => reorderRow(T, inf, s, d, width),
        .gray => grayRow(T, inf, s, d, width),
        .from_gray => {
            const alpha: T = if (@typeInfo(T) == .float) 1.0 else std.math.maxInt(T);
            for (0..width) |x| {
                const v = s[x];
                inline for (0..3) |c| d[x * inf.dcn + c] = v;
                if (inf.dcn == 4) d[x * 4 + 3] = alpha;
            }
        },
        .hsv => for (0..width) |x| {
            const p = s + x * 3;
            const b = p[inf.sb];
            const g = p[1];
            const r = p[2 - inf.sb];
            const o = d + x * 3;
            if (T == u8) rgb2hsvU8(r, g, b, o) else if (T == f32) rgb2hsvF32(r, g, b, o) else @compileError("HSV needs u8 or f32");
        },
        .from_hsv => for (0..width) |x| {
            const p = s + x * 3;
            var rgb: [3]f32 = undefined;
            if (T == u8) {
                hsv2rgbF32(@as(f32, @floatFromInt(p[0])), 6.0 / 180.0, @as(f32, @floatFromInt(p[1])) * (1.0 / 255.0), @as(f32, @floatFromInt(p[2])) * (1.0 / 255.0), &rgb);
                d[x * 3 + inf.db] = truncU8(rgb[2] * 255.0);
                d[x * 3 + 1] = truncU8(rgb[1] * 255.0);
                d[x * 3 + 2 - inf.db] = truncU8(rgb[0] * 255.0);
            } else if (T == f32) {
                hsv2rgbF32(p[0], 6.0 / 360.0, p[1], p[2], &rgb);
                d[x * 3 + inf.db] = rgb[2];
                d[x * 3 + 1] = rgb[1];
                d[x * 3 + 2 - inf.db] = rgb[0];
            } else @compileError("HSV needs u8 or f32");
        },
        .ycrcb => for (0..width) |x| {
            const p = s + x * 3;
            const o = d + x * 3;
            if (T == u8) {
                const b: i32 = p[inf.sb];
                const g: i32 = p[1];
                const r: i32 = p[2 - inf.sb];
                const y_ = (b * B2Y + g * G2Y + r * R2Y + (1 << (shift - 1))) >> shift;
                const delta: i32 = 128 << shift;
                o[0] = sat.cast(u8, y_);
                o[1] = sat.cast(u8, ((r - y_) * 11682 + delta + (1 << (shift - 1))) >> shift);
                o[2] = sat.cast(u8, ((b - y_) * 9241 + delta + (1 << (shift - 1))) >> shift);
            } else if (T == f32) {
                const b = p[inf.sb];
                const g = p[1];
                const r = p[2 - inf.sb];
                const y_ = b * 0.114 + g * 0.587 + r * 0.299;
                o[0] = y_;
                o[1] = (r - y_) * 0.713 + 0.5;
                o[2] = (b - y_) * 0.564 + 0.5;
            } else @compileError("YCrCb needs u8 or f32");
        },
        .from_ycrcb => for (0..width) |x| {
            const p = s + x * 3;
            const o = d + x * 3;
            if (T == u8) {
                const y_: i32 = p[0];
                const cr: i32 = @as(i32, p[1]) - 128;
                const cb: i32 = @as(i32, p[2]) - 128;
                const half = 1 << (shift - 1);
                o[inf.db] = sat.cast(u8, y_ + ((cb * 29049 + half) >> shift));
                o[1] = sat.cast(u8, y_ + ((cb * -5636 + cr * -11698 + half) >> shift));
                o[2 - inf.db] = sat.cast(u8, y_ + ((cr * 22987 + half) >> shift));
            } else if (T == f32) {
                const y_ = p[0];
                const cr = p[1] - 0.5;
                const cb = p[2] - 0.5;
                o[inf.db] = y_ + cb * 1.773;
                o[1] = y_ + cb * -0.344 + cr * -0.714;
                o[2 - inf.db] = y_ + cr * 1.403;
            } else @compileError("YCrCb needs u8 or f32");
        },
    }
}

/// Channel reorder / alpha add or drop. SIMD for u8 via shuffles.
inline fn reorderRow(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) void {
    const swap = inf.sb != inf.db;
    const alpha: T = if (@typeInfo(T) == .float) 1.0 else std.math.maxInt(T);
    var x: usize = 0;
    const N = 16;
    if (T == u8) {
        // Build one shuffle that maps N source pixels to N destination pixels.
        const mask = comptime blk: {
            var m: [N * inf.dcn]i32 = undefined;
            for (0..N) |i| for (0..inf.dcn) |c| {
                m[i * inf.dcn + c] = if (c < 3)
                    @intCast(i * inf.scn + (if (swap and c != 1) 2 - c else c))
                else if (inf.scn == 4) @intCast(i * inf.scn + 3) else -1;
            };
            break :blk m;
        };
        const fill: @Vector(1, u8) = @splat(alpha);
        while (x + N <= width) : (x += N) {
            const v: @Vector(N * inf.scn, u8) = s[x * inf.scn ..][0 .. N * inf.scn].*;
            const out: @Vector(N * inf.dcn, u8) = if (inf.dcn == 4 and inf.scn == 3)
                @shuffle(u8, v, fill, mask)
            else
                @shuffle(u8, v, undefined, mask);
            d[x * inf.dcn ..][0 .. N * inf.dcn].* = out;
        }
    }
    while (x < width) : (x += 1) {
        const p = s + x * inf.scn;
        const o = d + x * inf.dcn;
        if (swap) {
            o[0] = p[2];
            o[1] = p[1];
            o[2] = p[0];
        } else {
            o[0] = p[0];
            o[1] = p[1];
            o[2] = p[2];
        }
        if (inf.dcn == 4) o[3] = if (inf.scn == 4) p[3] else alpha;
    }
}

inline fn grayRow(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) void {
    const bi = inf.sb;
    const ri = 2 - inf.sb;
    var x: usize = 0;
    if (T == u8 or T == u16) {
        const N = 16;
        const W = u32;
        const sel = struct {
            fn mask(comptime c: usize) [N]i32 {
                var m: [N]i32 = undefined;
                for (0..N) |i| m[i] = @intCast(i * inf.scn + c);
                return m;
            }
        };
        const half: @Vector(N, W) = @splat(1 << (gray_shift - 1));
        while (x + N <= width) : (x += N) {
            const v: @Vector(N * inf.scn, T) = s[x * inf.scn ..][0 .. N * inf.scn].*;
            const b: @Vector(N, W) = @shuffle(T, v, undefined, sel.mask(bi));
            const g: @Vector(N, W) = @shuffle(T, v, undefined, sel.mask(1));
            const r: @Vector(N, W) = @shuffle(T, v, undefined, sel.mask(ri));
            const y_ = (b * @as(@Vector(N, W), @splat(B2G)) + g * @as(@Vector(N, W), @splat(G2G)) + r * @as(@Vector(N, W), @splat(R2G)) + half) >> @splat(gray_shift);
            const out: @Vector(N, T) = @intCast(y_);
            d[x..][0..N].* = out;
        }
        while (x < width) : (x += 1) {
            const p = s + x * inf.scn;
            const y_ = (@as(u32, p[bi]) * B2G + @as(u32, p[1]) * G2G + @as(u32, p[ri]) * R2G + (1 << (gray_shift - 1))) >> gray_shift;
            d[x] = @intCast(y_);
        }
    } else {
        while (x < width) : (x += 1) {
            const p = s + x * inf.scn;
            d[x] = @floatCast(@as(f32, @floatCast(p[bi])) * 0.114 + @as(f32, @floatCast(p[1])) * 0.587 + @as(f32, @floatCast(p[ri])) * 0.299);
        }
    }
}

// OpenCV's RGB2HSV_b tables (hsv_shift = 12, hrange = 180).
const hsv_shift = 12;
const hsv_tables = blk: {
    @setEvalBranchQuota(20000);
    var sdiv: [256]i32 = undefined;
    var hdiv: [256]i32 = undefined;
    sdiv[0] = 0;
    hdiv[0] = 0;
    for (1..256) |i| {
        sdiv[i] = @intFromFloat(@round(@as(f64, 255 << hsv_shift) / @as(f64, @floatFromInt(i))));
        hdiv[i] = @intFromFloat(@round(@as(f64, 180 << hsv_shift) / (6.0 * @as(f64, @floatFromInt(i)))));
    }
    break :blk .{ .sdiv = sdiv, .hdiv = hdiv };
};

inline fn rgb2hsvU8(r8: u8, g8: u8, b8: u8, o: [*]u8) void {
    const r: i32 = r8;
    const g: i32 = g8;
    const b: i32 = b8;
    const v = @max(r, g, b);
    const vmin = @min(r, g, b);
    const diff = v - vmin;
    const vr: i32 = if (v == r) -1 else 0;
    const vg: i32 = if (v == g) -1 else 0;
    const half = 1 << (hsv_shift - 1);
    const s = (diff * hsv_tables.sdiv[@intCast(v)] + half) >> hsv_shift;
    var h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = (h * hsv_tables.hdiv[@intCast(diff)] + half) >> hsv_shift;
    if (h < 0) h += 180;
    o[0] = sat.cast(u8, h);
    o[1] = @intCast(s);
    o[2] = @intCast(v);
}

inline fn rgb2hsvF32(r: f32, g: f32, b: f32, o: [*]f32) void {
    const eps = std.math.floatEps(f32);
    const v = @max(r, g, b);
    const vmin = @min(r, g, b);
    var diff = v - vmin;
    const s = diff / (@abs(v) + eps);
    diff = 60.0 / (diff + eps);
    var h: f32 = if (v == r) (g - b) * diff else if (v == g) (b - r) * diff + 120.0 else (r - g) * diff + 240.0;
    if (h < 0) h += 360.0;
    o[0] = h;
    o[1] = s;
    o[2] = v;
}

/// OpenCV's HSV2RGB: h scaled by `hscale` into [0, 6), s and v in [0, 1];
/// writes r, g, b. `1 - s*h` is fused (as OpenCV's SIMD path computes it).
inline fn hsv2rgbF32(h_in: f32, hscale: f32, s: f32, v: f32, out: *[3]f32) void {
    if (s == 0) {
        out.* = .{ v, v, v };
        return;
    }
    var h = h_in * hscale;
    while (h < 0) h += 6;
    while (h >= 6) h -= 6;
    var sector: i32 = @intFromFloat(@floor(h));
    h -= @floatFromInt(sector);
    if (sector < 0 or sector >= 6) {
        sector = 0;
        h = 0;
    }
    const tab = [4]f32{ v, v * (1 - s), v * @mulAdd(f32, -s, h, 1), v * @mulAdd(f32, -s, 1 - h, 1) };
    const sd = [6][3]u8{ .{ 1, 3, 0 }, .{ 1, 0, 2 }, .{ 3, 0, 1 }, .{ 0, 2, 1 }, .{ 0, 1, 3 }, .{ 2, 1, 0 } };
    const k = sd[@intCast(sector)];
    // OpenCV's table gives b, g, r.
    out.* = .{ tab[k[2]], tab[k[1]], tab[k[0]] };
}

inline fn truncU8(x: f32) u8 {
    return @intFromFloat(std.math.clamp(@trunc(x), 0, 255));
}

test "gray and swap" {
    const gpa = std.testing.allocator;
    var src = try Image(u8).init(gpa, 37, 3, 3);
    defer src.deinit(gpa);
    for (src.slice(), 0..) |*p, i| p.* = @truncate(i * 7);
    var gray = try Image(u8).init(gpa, 37, 3, 1);
    defer gray.deinit(gpa);
    try cvtColor(.serial, src, gray, .rgb2gray);
    const p = src.pixel(20, 1);
    const want = (@as(u32, p[0]) * R2G + @as(u32, p[1]) * G2G + @as(u32, p[2]) * B2G + (1 << 14)) >> 15;
    try std.testing.expectEqual(@as(u8, @intCast(want)), gray.at(20, 1, 0));
    var bgra = try Image(u8).init(gpa, 37, 3, 4);
    defer bgra.deinit(gpa);
    try cvtColor(.serial, src, bgra, .rgb2bgra);
    for (0..37) |x| {
        try std.testing.expectEqual(src.at(x, 2, 0), bgra.at(x, 2, 2));
        try std.testing.expectEqual(src.at(x, 2, 2), bgra.at(x, 2, 0));
        try std.testing.expectEqual(@as(u8, 255), bgra.at(x, 2, 3));
    }
}
