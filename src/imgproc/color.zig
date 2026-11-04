//! Colour space conversions (cvtColor). Integer paths reproduce OpenCV's
//! fixed-point arithmetic so 8-bit results match cv2 exactly.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const vec = core.vec;

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
        .hsv => for (hsvU8(T, inf, s, d, width)..width) |x| {
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
        .ycrcb => for (ycrcbU8(T, inf, s, d, width)..width) |x| {
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

/// Rounding high multiply (a * b + 2^14) >> 15 in one instruction: AVX2
/// vpmulhrsw (inline asm), NEON sqrdmulh (LLVM intrinsic, not on Windows).
const rhm_x86 = @import("builtin").zig_backend == .stage2_llvm and @import("builtin").cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx2);
const ycc_i32 = @import("builtin").cpu.arch == .x86_64 and std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx512bw);
const rhm_arm = core.simd.llvm_intrinsics and @import("builtin").cpu.arch == .aarch64;
extern fn @"llvm.aarch64.neon.sqrdmulh.v8i16"(@Vector(8, i16), @Vector(8, i16)) @Vector(8, i16);

/// 16 pixels at a time for u8; returns how many pixels it handled.
inline fn ycrcbU8(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) usize {
    if (T != u8) return 0;
    const I = @Vector(16, i32);
    const half: I = @splat(1 << (shift - 1));
    const delta: I = @splat(128 << shift);
    var x: usize = 0;
    // 16-bit lanes, exact: Y with byte-split coefficients (c = 256 hi + lo,
    // (256 H + L + 2^13) >> 14 == (H + (L >> 8) + 32) >> 6), and
    // (d * c + 2^13) >> 14 == (d * 2c + 2^14) >> 15, a rounding high
    // multiply (pmulhrsw / sqrdmulh) with 2c < 2^15.
    const W = @Vector(16, u16);
    const S = @Vector(16, i16);
    const rhm = struct {
        inline fn f(v: @Vector(16, i16), comptime c2: i32) @Vector(16, i16) {
            const cv: @Vector(16, i16) = @splat(c2);
            if (comptime rhm_x86) return asm ("vpmulhrsw %[b], %[a], %[r]"
                : [r] "=x" (-> @Vector(16, i16)),
                : [a] "x" (v),
                  [b] "x" (cv),
            );
            if (comptime rhm_arm) {
                const va: [16]i16 = v;
                const ca: [16]i16 = cv;
                const lo = @"llvm.aarch64.neon.sqrdmulh.v8i16"(va[0..8].*, ca[0..8].*);
                const hi = @"llvm.aarch64.neon.sqrdmulh.v8i16"(va[8..16].*, ca[8..16].*);
                return std.simd.join(lo, hi);
            }
            const V32 = @Vector(16, i32);
            return @truncate((@as(V32, v) * @as(V32, cv) + @as(V32, @splat(1 << 14))) >> @splat(15));
        }
    }.f;
    // AVX-512 fills a whole register with the i32 form below; elsewhere
    // the 16-bit form wins (A/B: Apple Silicon 1.53x, AVX2 1.33x, AVX-512
    // 0.95x).
    if (comptime !ycc_i32) while (x + 16 <= width) : (x += 16) {
        const ch = vec.load(3, s + x * 3);
        const b: W = ch[inf.sb];
        const g: W = ch[1];
        const r: W = ch[2 - inf.sb];
        const hs = b * @as(W, @splat(B2Y >> 8)) + g * @as(W, @splat(G2Y >> 8)) + r * @as(W, @splat(R2Y >> 8));
        const ls = b * @as(W, @splat(B2Y & 0xff)) + g * @as(W, @splat(G2Y & 0xff)) + r * @as(W, @splat(R2Y & 0xff));
        const y_ = (hs + (ls >> @splat(8)) + @as(W, @splat(32))) >> @splat(6);
        const ys: S = @bitCast(y_);
        const cr = rhm(@as(S, @bitCast(r)) - ys, 2 * 11682) + @as(S, @splat(128));
        const cb = rhm(@as(S, @bitCast(b)) - ys, 2 * 9241) + @as(S, @splat(128));
        const lo: S = @splat(0);
        const hi: S = @splat(255);
        vec.store(3, .{
            @as(vec.V16, @truncate(y_)),
            core.simd.narrow(u8, @min(@max(cr, lo), hi)),
            core.simd.narrow(u8, @min(@max(cb, lo), hi)),
        }, d + x * 3);
    };
    while (x + 16 <= width) : (x += 16) {
        const ch = vec.load(3, s + x * 3);
        const b: I = ch[inf.sb];
        const g: I = ch[1];
        const r: I = ch[2 - inf.sb];
        const y_ = (b * @as(I, @splat(B2Y)) + g * @as(I, @splat(G2Y)) + r * @as(I, @splat(R2Y)) + half) >> @splat(shift);
        const cr = ((r - y_) * @as(I, @splat(11682)) + delta + half) >> @splat(shift);
        const cb = ((b - y_) * @as(I, @splat(9241)) + delta + half) >> @splat(shift);
        const lo: I = @splat(0);
        const hi: I = @splat(255);
        vec.store(3, .{
            vec.narrowU8(16, @min(@max(y_, lo), hi)),
            vec.narrowU8(16, @min(@max(cr, lo), hi)),
            vec.narrowU8(16, @min(@max(cb, lo), hi)),
        }, d + x * 3);
    }
    return x;
}

/// Channel reorder / alpha add or drop. SIMD for u8 via shuffles.
inline fn reorderRow(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) void {
    const swap = inf.sb != inf.db;
    const alpha: T = if (@typeInfo(T) == .float) 1.0 else std.math.maxInt(T);
    var x: usize = 0;
    if (T == u8) {
        const map = comptime blk: {
            var m: [inf.dcn]?u8 = undefined;
            m[0] = if (swap) 2 else 0;
            m[1] = 1;
            m[2] = if (swap) 0 else 2;
            if (inf.dcn == 4) m[3] = if (inf.scn == 4) 3 else null;
            break :blk m;
        };
        x = vec.reorderRow(inf.scn, inf.dcn, map, 255, s, d, width);
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
        const U = @Vector(16, u32);
        if (T == u8) {
            if (vec.has_vbmi) {
                const U64 = @Vector(64, u32);
                while (x + 64 <= width) : (x += 64) {
                    const ch = vec.load64(inf.scn, s + x * inf.scn);
                    const b: U64 = ch[bi];
                    const g: U64 = ch[1];
                    const r: U64 = ch[ri];
                    const y_ = (b * @as(U64, @splat(B2G)) + g * @as(U64, @splat(G2G)) + r * @as(U64, @splat(R2G)) + @as(U64, @splat(1 << (gray_shift - 1)))) >> @splat(gray_shift);
                    d[x..][0..64].* = @as(vec.V64, @intCast(y_));
                }
            }
            if (!vec.has_vbmi) {
                // Coefficients split into bytes, c = 256 * hi + lo: both
                // partial sums fit u16 and come from 8-bit multiply-
                // accumulates (umull/umlal), and
                //   (256 H + L + 2^14) >> 15 == (H + (L >> 8) + 64) >> 7
                // exactly, i.e. usra then rshrn.
                const W = @Vector(16, u16);
                const hi = struct {
                    fn f(c: u16) W {
                        return @splat(c >> 8);
                    }
                }.f;
                const lo = struct {
                    fn f(c: u16) W {
                        return @splat(c & 0xff);
                    }
                }.f;
                while (x + 16 <= width) : (x += 16) {
                    const ch = vec.load(inf.scn, s + x * inf.scn);
                    const b: W = ch[bi];
                    const g: W = ch[1];
                    const r: W = ch[ri];
                    const hs = b * hi(B2G) + g * hi(G2G) + r * hi(R2G);
                    const ls = b * lo(B2G) + g * lo(G2G) + r * lo(R2G);
                    const t = hs + (ls >> @splat(8));
                    d[x..][0..16].* = @as(vec.V16, @truncate((t + @as(W, @splat(64))) >> @splat(7)));
                }
            }
            while (x + 16 <= width) : (x += 16) {
                const ch = vec.load(inf.scn, s + x * inf.scn);
                const b: U = ch[bi];
                const g: U = ch[1];
                const r: U = ch[ri];
                const y_ = (b * @as(U, @splat(B2G)) + g * @as(U, @splat(G2G)) + r * @as(U, @splat(R2G)) + @as(U, @splat(1 << (gray_shift - 1)))) >> @splat(gray_shift);
                d[x..][0..16].* = @as(vec.V16, @intCast(y_));
            }
        } else {
            while (x + 16 <= width) : (x += 16) {
                var b: U = undefined;
                var g: U = undefined;
                var r: U = undefined;
                inline for (0..16) |i| {
                    b[i] = s[(x + i) * inf.scn + bi];
                    g[i] = s[(x + i) * inf.scn + 1];
                    r[i] = s[(x + i) * inf.scn + ri];
                }
                const y_ = (b * @as(U, @splat(B2G)) + g * @as(U, @splat(G2G)) + r * @as(U, @splat(R2G)) + @as(U, @splat(1 << (gray_shift - 1)))) >> @splat(gray_shift);
                d[x..][0..16].* = @as(@Vector(16, u16), @intCast(y_));
            }
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

/// Both table entries (sdiv[v] and hdiv[diff]) from one f32 reciprocal of
/// v*diff and two multiplies: exactly the table values for every pair
/// with diff <= v (checked exhaustively in a test), at half the cost of two
/// divisions. When diff is 0 both products are 0 or unused.
inline fn hsvDivs(comptime N: usize, v: @Vector(N, i32), diff: @Vector(N, i32)) struct { sdiv: @Vector(N, i32), hdiv: @Vector(N, i32) } {
    const F = @Vector(N, f32);
    const prod: F = @floatFromInt(@max(v * diff, @as(@Vector(N, i32), @splat(1))));
    const recip = @as(F, @splat(1.0)) / prod;
    const fd: F = @floatFromInt(diff);
    const fv: F = @floatFromInt(v);
    const sd = fd * @as(F, @splat(255 << hsv_shift)) * recip;
    const hd = fv * @as(F, @splat((180 << hsv_shift) / 6)) * recip;
    return .{ .sdiv = core.simd.roundI32(N, sd), .hdiv = core.simd.roundI32(N, hd) };
}

test "single-reciprocal HSV divisions match the tables" {
    var v: i32 = 1;
    while (v < 256) : (v += 1) {
        var d: i32 = 1;
        while (d <= v) : (d += 16) {
            var dv: @Vector(16, i32) = undefined;
            inline for (0..16) |i| dv[i] = @min(d + @as(i32, i), v);
            const r = hsvDivs(16, @splat(v), dv);
            const sd: [16]i32 = r.sdiv;
            const hd: [16]i32 = r.hdiv;
            const ds: [16]i32 = dv;
            for (0..16) |i| {
                try std.testing.expectEqual(hsv_tables.sdiv[@intCast(v)], sd[i]);
                try std.testing.expectEqual(hsv_tables.hdiv[@intCast(ds[i])], hd[i]);
            }
        }
    }
}

/// OpenCV's RGB2HSV_b on N-lane vectors (N = 16 or 64).
inline fn hsvBlock(comptime N: usize, b8: @Vector(N, u8), g8: @Vector(N, u8), r8: @Vector(N, u8)) [3]@Vector(N, u8) {
    // Comparisons, selects and the hue numerator fit i16 (|h| <= 4 * 255 +
    // 255); only the two products need 32 bits.
    const W = @Vector(N, i16);
    const I = @Vector(N, i32);
    const half: I = @splat(1 << (hsv_shift - 1));
    const b: W = b8;
    const g: W = g8;
    const r: W = r8;
    const v = @max(r, @max(g, b));
    const vmin = @min(r, @min(g, b));
    const diff = v - vmin;
    const zero: W = @splat(0);
    const ones: W = @splat(-1);
    const vr = @select(i16, v == r, ones, zero);
    const vg = @select(i16, v == g, ones, zero);
    const h16 = (vr & (g - b)) + (~vr & ((vg & (b - r + diff + diff)) + (~vg & (r - g + @as(W, @splat(4)) * diff))));
    const vi: I = v;
    const di: I = diff;
    const t = hsvDivs(N, vi, di);
    const sat_ = (di * t.sdiv + half) >> @splat(hsv_shift);
    var h = (@as(I, h16) * t.hdiv + half) >> @splat(hsv_shift);
    h += @select(i32, h < @as(I, @splat(0)), @as(I, @splat(180)), @as(I, @splat(0)));
    return .{ core.simd.narrow(u8, h), core.simd.narrow(u8, sat_), core.simd.narrow(u8, v) };
}

/// 16 (or, with AVX-512 VBMI, 64) pixels at a time of OpenCV's RGB2HSV_b;
/// returns pixels handled.
inline fn hsvU8(comptime T: type, comptime inf: Code.Info, s: [*]const T, d: [*]T, width: usize) usize {
    if (T != u8) return 0;
    var x: usize = 0;
    if (vec.has_vbmi) {
        while (x + 64 <= width) : (x += 64) {
            const ch = vec.load64(3, s + x * 3);
            vec.store64(3, hsvBlock(64, ch[inf.sb], ch[1], ch[2 - inf.sb]), d + x * 3);
        }
    }
    // 32 pixels per step: two deinterleaving loads, one wide block.
    while (x + 32 <= width) : (x += 32) {
        const c0 = vec.load(3, s + x * 3);
        const c1 = vec.load(3, s + (x + 16) * 3);
        const r = hsvBlock(32, std.simd.join(c0[inf.sb], c1[inf.sb]), std.simd.join(c0[1], c1[1]), std.simd.join(c0[2 - inf.sb], c1[2 - inf.sb]));
        var lo: [3]vec.V16 = undefined;
        var hi: [3]vec.V16 = undefined;
        inline for (0..3) |k| {
            const a: [32]u8 = r[k];
            lo[k] = a[0..16].*;
            hi[k] = a[16..32].*;
        }
        vec.store(3, lo, d + x * 3);
        vec.store(3, hi, d + (x + 16) * 3);
    }
    while (x + 16 <= width) : (x += 16) {
        const ch = vec.load(3, s + x * 3);
        vec.store(3, hsvBlock(16, ch[inf.sb], ch[1], ch[2 - inf.sb]), d + x * 3);
    }
    return x;
}

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

test "vector BGR -> HSV matches the scalar reference on every colour" {
    // All 2^24 colours would be slow in Debug; a 1-in-7 lattice in each
    // channel plus the extremes covers every branch and rounding case.
    const gpa = std.testing.allocator;
    var vals: [40]u8 = undefined;
    var nv: usize = 0;
    var c: u32 = 0;
    while (c < 256) : (c += 7) {
        vals[nv] = @intCast(c);
        nv += 1;
    }
    vals[nv] = 255;
    nv += 1;
    const n = nv * nv * nv;
    var src = try Image(u8).init(gpa, @intCast(n), 1, 3);
    defer src.deinit(gpa);
    var dst = try Image(u8).init(gpa, @intCast(n), 1, 3);
    defer dst.deinit(gpa);
    var i: usize = 0;
    for (vals[0..nv]) |b| for (vals[0..nv]) |g| for (vals[0..nv]) |r| {
        src.slice()[i * 3 ..][0..3].* = .{ b, g, r };
        i += 1;
    };
    try cvtColor(.serial, src, dst, .bgr2hsv);
    for (0..n) |p| {
        const px = src.slice()[p * 3 ..][0..3];
        var want: [3]u8 = undefined;
        rgb2hsvU8(px[2], px[1], px[0], &want);
        try std.testing.expectEqualSlices(u8, &want, dst.slice()[p * 3 ..][0..3]);
    }
}

test "vector BGR/BGRA -> gray matches the scalar formula" {
    const gpa = std.testing.allocator;
    inline for (.{ Code.bgr2gray, Code.rgb2gray, Code.bgra2gray }) |code| {
        const scn: u16 = if (code == .bgra2gray) 4 else 3;
        const n: u32 = 52 * 52 * 52;
        var src = try Image(u8).init(gpa, n, 1, scn);
        defer src.deinit(gpa);
        var dst = try Image(u8).init(gpa, n, 1, 1);
        defer dst.deinit(gpa);
        var i: usize = 0;
        for (0..52) |b| for (0..52) |g| for (0..52) |r| {
            const px = src.slice()[i * scn ..];
            px[0] = @intCast(@min(255, b * 5));
            px[1] = @intCast(@min(255, g * 5));
            px[2] = @intCast(@min(255, r * 5));
            if (scn == 4) px[3] = 7;
            i += 1;
        };
        try cvtColor(.serial, src, dst, code);
        const sb: usize = if (code == .rgb2gray) 2 else 0;
        for (0..n) |p| {
            const px = src.slice()[p * scn ..];
            const want = (@as(u32, px[sb]) * B2G + @as(u32, px[1]) * G2G + @as(u32, px[2 - sb]) * R2G + (1 << 14)) >> 15;
            try std.testing.expectEqual(@as(u8, @intCast(want)), dst.slice()[p]);
        }
    }
}

test "vector BGR -> YCrCb matches OpenCV's fixed-point formula" {
    const gpa = std.testing.allocator;
    const n: u32 = 52 * 52 * 52;
    var src = try Image(u8).init(gpa, n, 1, 3);
    defer src.deinit(gpa);
    var dst = try Image(u8).init(gpa, n, 1, 3);
    defer dst.deinit(gpa);
    var i: usize = 0;
    for (0..52) |b| for (0..52) |g| for (0..52) |r| {
        src.slice()[i * 3 ..][0..3].* = .{ @intCast(@min(255, b * 5)), @intCast(@min(255, g * 5)), @intCast(@min(255, r * 5)) };
        i += 1;
    };
    try cvtColor(.serial, src, dst, .bgr2ycrcb);
    for (0..n) |p| {
        const px = src.slice()[p * 3 ..][0..3];
        const b: i32 = px[0];
        const g: i32 = px[1];
        const r: i32 = px[2];
        const half = 1 << (shift - 1);
        const y = (b * B2Y + g * G2Y + r * R2Y + half) >> shift;
        const cr = ((r - y) * 11682 + (128 << shift) + half) >> shift;
        const cb = ((b - y) * 9241 + (128 << shift) + half) >> shift;
        const want = [3]u8{ @intCast(std.math.clamp(y, 0, 255)), @intCast(std.math.clamp(cr, 0, 255)), @intCast(std.math.clamp(cb, 0, 255)) };
        try std.testing.expectEqualSlices(u8, &want, dst.slice()[p * 3 ..][0..3]);
    }
}
