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
            const h = c.s.height;
            for (y0..y1) |y| {
                const sy = if (c.code == .horizontal) y else h - 1 - y;
                const s = c.s.row(sy);
                const d = c.d.row(y);
                if (c.code == .vertical) {
                    @memcpy(d, s);
                    continue;
                }
                switch (c.s.channels) {
                    inline 1, 2, 3, 4 => |k| flipRow(T, k, s, d),
                    else => |cn| {
                        const w = c.s.width;
                        for (0..w) |x| @memcpy(d[x * cn ..][0..cn], s[(w - 1 - x) * cn ..][0..cn]);
                    },
                }
            }
        }
    }.run);
}

const has_vbmi = @import("builtin").cpu.arch == .x86_64 and std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx512vbmi);

/// Inline asm with the `v` (zmm) register class: LLVM backend, x86-64 with
/// AVX-512. Elsewhere the helpers below are plain no-ops.
const zmm_asm = @import("builtin").zig_backend == .stage2_llvm and @import("builtin").cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(@import("builtin").cpu.features, .avx512f);

/// Keeps a loaded 64-byte vector opaque so a following byte shuffle is
/// lowered as one permute instead of per-byte inserts.
inline fn opaque64(v: @Vector(64, u8)) @Vector(64, u8) {
    return if (!zmm_asm) v else asm (""
        : [ret] "=v" (-> @Vector(64, u8)),
        : [v] "0" (v),
    );
}

/// Reverses the pixel order of one row, `cn` interleaved channels.
inline fn flipRow(comptime T: type, comptime cn: usize, s: []const T, d: []T) void {
    const w = s.len / cn;
    // About 64 bytes per step.
    const N = @max(4, 64 / (@sizeOf(T) * (if (cn == 3) 4 else cn)));
    const L = N * cn;
    const mask = comptime blk: {
        @setEvalBranchQuota(100000);
        var m: [L]i32 = undefined;
        for (0..N) |i| for (0..cn) |ch| {
            m[i * cn + ch] = @intCast((N - 1 - i) * cn + ch);
        };
        break :blk m;
    };
    var x: usize = 0;
    if (T == u8 and cn == 3 and has_vbmi) {
        // 21 pixels per 64-byte byte permute; same overlap scheme as below.
        const m21 = comptime blk: {
            var m: [64]i32 = undefined;
            for (0..21) |i| for (0..3) |ch| {
                m[i * 3 + ch] = @intCast(1 + 3 * (20 - i) + ch);
            };
            m[63] = 0;
            break :blk m;
        };
        while (x + 22 <= w) : (x += 21) {
            const v = opaque64(s[(w - x - 21) * 3 - 1 ..][0..64].*);
            d[x * 3 ..][0..64].* = @shuffle(u8, v, undefined, m21);
        }
    }
    if (T == u8 and cn == 3) {
        // 5 pixels per 16-byte shuffle (lane-local, so a single pshufb); the
        // load ends at the group's last byte and the store spills one byte
        // into the next group, which is rewritten on the next step.
        const m3 = comptime blk: {
            var m: [16]i32 = undefined;
            for (0..5) |i| for (0..3) |ch| {
                m[i * 3 + ch] = @intCast(1 + 3 * (4 - i) + ch);
            };
            m[15] = 0;
            break :blk m;
        };
        while (x + 6 <= w) : (x += 5) {
            const v: @Vector(16, u8) = s[(w - x - 5) * 3 - 1 ..][0..16].*;
            d[x * 3 ..][0..16].* = @shuffle(u8, v, undefined, m3);
        }
    }
    while (x + N <= w) : (x += N) {
        const v: @Vector(L, T) = s[(w - x - N) * cn ..][0..L].*;
        d[x * cn ..][0..L].* = @shuffle(T, v, undefined, mask);
    }
    while (x < w) : (x += 1) {
        inline for (0..cn) |ch| d[x * cn + ch] = s[(w - 1 - x) * cn + ch];
    }
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

/// Source coordinates of destination pixel (dx, dy).
inline fn rotSrc(comptime kind: RotKind, dx: usize, dy: usize, sw: usize, sh: usize) [2]usize {
    return switch (kind) {
        .transpose => .{ dy, dx },
        .cw => .{ dy, sh - 1 - dx },
        .ccw => .{ sw - 1 - dy, dx },
    };
}

/// 16x16 tiles of pixels up to 4 bytes wide are transposed in registers as
/// 32-bit lanes; edges and wider pixels go through a scalar blocked loop.
fn rot(exec: Exec, src: anytype, dst: @TypeOf(src), comptime kind: RotKind) Error!void {
    const T = @TypeOf(src).Elem;
    if (src.width != dst.height or src.height != dst.width or src.channels != dst.channels) return error.SizeMismatch;
    const M = 64; // destination rows per band
    const Ctx = struct { s: Image(T), d: Image(T) };
    const blocks_y = (dst.height + M - 1) / M;
    exec.parallelFor(blocks_y, 1, Ctx{ .s = src, .d = dst }, struct {
        fn run(c: Ctx, b0: usize, b1: usize) void {
            const px_bytes = @as(usize, c.s.channels) * @sizeOf(T);
            for (b0..b1) |by| {
                const dy0 = by * M;
                const dy1 = @min(dy0 + M, c.d.height);
                switch (px_bytes) {
                    inline 1, 2, 3, 4 => |pb| rotMacroRow(T, pb, kind, c.s, c.d, dy0, dy1),
                    else => rotScalar(T, kind, c.s, c.d, dy0, dy1, 0, c.d.width),
                }
            }
        }
    }.run);
}

/// Destination rows [dy0, dy1) (at most 64): for each column of 16x16
/// tiles, the 16 source rows it reads are prefetched one column ahead, since
/// each tile's stores depend on 16 loads from 16 different pages.
fn rotMacroRow(comptime T: type, comptime pb: usize, comptime kind: RotKind, s: Image(T), d: Image(T), dy0: usize, dy1: usize) void {
    const B = 16;
    const dw = d.width;
    const full_y = dy0 + (dy1 - dy0) / B * B;
    const full_x = dw / B * B;
    const sb: [*]const u8 = @ptrCast(s.data);
    const ss = s.stride * @sizeOf(T);
    const span = (full_y - dy0) * pb; // source bytes per row of one column
    const PD = 4; // prefetch distance in tile columns
    var tx: usize = 0;
    while (tx < full_x) : (tx += B) {
        if (tx + PD * B < full_x and span > 0) {
            for (0..B) |i| {
                const sy, const sx = switch (kind) {
                    .transpose => .{ tx + PD * B + i, dy0 },
                    .cw => .{ s.height - 1 - (tx + PD * B) - i, dy0 },
                    .ccw => .{ tx + PD * B + i, s.width - (full_y - dy0) - dy0 },
                };
                const p = sb + sy * ss + sx * pb;
                var o: usize = 0;
                while (o < span + 63) : (o += 64) @prefetch(p + @min(o, span - 1), .{});
            }
        }
        var ty = dy0;
        while (ty < full_y) : (ty += B) rotTile(T, pb, kind, s, d, ty, tx);
    }
    if (full_x < dw) rotScalar(T, kind, s, d, dy0, full_y, full_x, dw);
    if (full_y < dy1) rotScalar(T, kind, s, d, full_y, dy1, 0, dw);
}

fn rotScalar(comptime T: type, comptime kind: RotKind, s: Image(T), d: Image(T), dy0: usize, dy1: usize, dx0: usize, dx1: usize) void {
    const cn: usize = s.channels;
    for (dy0..dy1) |dy| {
        const dr = d.row(dy);
        for (dx0..dx1) |dx| {
            const sxy = rotSrc(kind, dx, dy, s.width, s.height);
            const sp = s.data + sxy[1] * s.stride + sxy[0] * cn;
            @memcpy(dr[dx * cn ..][0..cn], sp[0..cn]);
        }
    }
}

const Lanes = @Vector(16, u32);

/// Keeps LLVM from folding the lane transpose into the byte-level load and
/// store shuffles, which it otherwise lowers poorly.
inline fn opaqueLanes(v: Lanes) Lanes {
    return if (!zmm_asm) v else asm (""
        : [ret] "=v" (-> Lanes),
        : [v] "0" (v),
    );
}

fn expand3(comptime skip: usize) [16]i32 {
    var m: [16]i32 = undefined;
    for (0..4) |i| {
        for (0..3) |k| m[i * 4 + k] = @intCast(skip + i * 3 + k);
        m[i * 4 + 3] = -1;
    }
    return m;
}

fn iotaMask(comptime n: usize) [n]i32 {
    @setEvalBranchQuota(10000);
    var m: [n]i32 = undefined;
    for (0..n) |i| m[i] = if (i < n / 2) @intCast(i) else ~@as(i32, @intCast(i - n / 2));
    return m;
}

fn laneMask(comptime k: usize) [16]i32 {
    var m: [16]i32 = undefined;
    for (0..16) |i| m[i] = @intCast(16 * k + i);
    return m;
}

const compact3 = blk: {
    var m: [16]i32 = undefined;
    for (0..4) |i| for (0..3) |k| {
        m[i * 3 + k] = @intCast(i * 4 + k);
    };
    for (12..16) |i| m[i] = 0;
    break :blk m;
};

inline fn loadLanes(comptime pb: usize, p: [*]const u8) Lanes {
    switch (pb) {
        1 => {
            const v: @Vector(16, u8) = p[0..16].*;
            return v;
        },
        2 => {
            const v: @Vector(16, u16) = @bitCast(@as(@Vector(32, u8), p[0..32].*));
            return v;
        },
        3 => {
            // Four 12-byte groups, each widened inside a 16-byte lane by one
            // pshufb; the last group is loaded from offset 32 so nothing is
            // read past the 48 bytes.
            const V16 = core.vec.V16;
            const g0 = core.vec.opaqueV(p[0..16].*);
            const g1 = core.vec.opaqueV(p[12..28].*);
            const g2 = core.vec.opaqueV(p[24..40].*);
            const g3 = core.vec.opaqueV(p[32..48].*);
            const z: V16 = @splat(0);
            const w0 = @shuffle(u8, g0, z, expand3(0));
            const w1 = @shuffle(u8, g1, z, expand3(0));
            const w2 = @shuffle(u8, g2, z, expand3(0));
            const w3 = @shuffle(u8, g3, z, expand3(4));
            const lo: @Vector(32, u8) = @shuffle(u8, w0, w1, iotaMask(32));
            const hi: @Vector(32, u8) = @shuffle(u8, w2, w3, iotaMask(32));
            return @bitCast(@as(@Vector(64, u8), @shuffle(u8, lo, hi, iotaMask(64))));
        },
        4 => return @bitCast(@as(@Vector(64, u8), p[0..64].*)),
        else => unreachable,
    }
}

inline fn storeLanes(comptime pb: usize, p: [*]u8, v: Lanes) void {
    switch (pb) {
        1 => p[0..16].* = @as([16]u8, @as(@Vector(16, u8), @truncate(v))),
        2 => p[0..32].* = @as(@Vector(32, u8), @bitCast(@as(@Vector(16, u16), @truncate(v)))),
        3 => {
            // Compact each 16-byte lane to 12 bytes; stores overlap forward
            // and the last group is written as 8 + 4 bytes.
            const b: @Vector(64, u8) = @bitCast(v);
            const V16 = core.vec.V16;
            inline for (0..4) |k| {
                const lane: V16 = @shuffle(u8, b, undefined, laneMask(k));
                const c = @shuffle(u8, lane, undefined, compact3);
                if (k < 3) {
                    p[12 * k ..][0..16].* = c;
                } else {
                    const arr: [16]u8 = c;
                    p[36..48].* = arr[0..12].*;
                }
            }
        },
        4 => p[0..64].* = @as(@Vector(64, u8), @bitCast(v)),
        else => unreachable,
    }
}

/// In-register transpose of 16 rows of 16 lanes (block-swap network).
inline fn transpose16(r: *[16]Lanes) void {
    inline for (.{ 8, 4, 2, 1 }) |b| {
        const ma = comptime blk: {
            @setEvalBranchQuota(10000);
            var m: [16]i32 = undefined;
            for (0..16) |col| m[col] = if (col & b == 0) @intCast(col) else ~@as(i32, @intCast(col - b));
            break :blk m;
        };
        const mb = comptime blk: {
            @setEvalBranchQuota(10000);
            var m: [16]i32 = undefined;
            for (0..16) |col| m[col] = if (col & b == 0) @intCast(col + b) else ~@as(i32, @intCast(col));
            break :blk m;
        };
        inline for (0..16) |i| {
            if (i & b == 0) {
                const lo = r[i];
                const hi = r[i + b];
                r[i] = @shuffle(u32, lo, hi, ma);
                r[i + b] = @shuffle(u32, lo, hi, mb);
            }
        }
    }
}

/// One 16x16 destination tile at (dx0, dy0).
inline fn rotTile(comptime T: type, comptime pb: usize, comptime kind: RotKind, s: Image(T), d: Image(T), dy0: usize, dx0: usize) void {
    const B = 16;
    const sb: [*]const u8 = @ptrCast(s.data);
    const db: [*]u8 = @ptrCast(d.data);
    const ss = s.stride * @sizeOf(T);
    const ds = d.stride * @sizeOf(T);
    const sw: usize = s.width;
    const sh: usize = s.height;
    var r: [16]Lanes = undefined;
    // r[i] holds 16 source pixels along a row, ordered so that after the
    // transpose r[j] is a run of destination pixels.
    inline for (0..16) |i| {
        const sy, const sx = switch (kind) {
            .transpose => .{ dx0 + i, dy0 },
            .cw => .{ sh - 1 - dx0 - i, dy0 },
            .ccw => .{ dx0 + i, sw - B - dy0 },
        };
        r[i] = opaqueLanes(loadLanes(pb, sb + sy * ss + sx * pb));
    }
    transpose16(&r);
    inline for (0..16) |j| {
        const dy = if (kind == .ccw) dy0 + B - 1 - j else dy0 + j;
        storeLanes(pb, db + dy * ds + dx0 * pb, r[j]);
    }
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
/// without WARP_INVERSE_MAP. Follows OpenCV 5's float kernels exactly: the
/// inverse matrix (computed in f64) is rounded to f32, source coordinates are
/// sx = fma(m0, x, m1*y + m2), bilinear blending is a chain of f32 fmas, and
/// results round half to even. Nearest takes the rounded coordinates.
pub fn warpAffine(exec: Exec, src: anytype, dst: @TypeOf(src), m: [6]f64, interp: WarpInterp, border: Border, value: f64) Error!void {
    const T = @TypeOf(src).Elem;
    const inv = invertAffine(m) orelse return error.Unsupported;
    if (src.channels != dst.channels) return error.SizeMismatch;
    var mf: [6]f32 = undefined;
    for (&mf, inv) |*o, v| o.* = @floatCast(v);
    const Ctx = struct { s: Image(T), d: Image(T), m: [6]f32, interp: WarpInterp, border: Border, value: T };
    const ctx = Ctx{ .s = src, .d = dst, .m = mf, .interp = interp, .border = border, .value = sat.cast(T, value) };
    exec.parallelFor(dst.height, 4, ctx, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            switch (c.s.channels) {
                inline 1, 3, 4 => |k| warpRows(T, k, c, y0, y1),
                else => warpRows(T, 0, c, y0, y1),
            }
        }
    }.run);
}

const WN = 16;
const VF = @Vector(WN, f32);

/// `kcn` is the channel count when known at compile time, else 0.
fn warpRows(comptime T: type, comptime kcn: usize, c: anytype, y0: usize, y1: usize) void {
    const cn: usize = if (kcn != 0) kcn else c.s.channels;
    const iota = comptime blk: {
        var a: [WN]f32 = undefined;
        for (0..WN) |i| a[i] = @floatFromInt(i);
        break :blk a;
    };
    const iv: VF = iota;
    const sw: f32 = @floatFromInt(c.s.width);
    const sh: f32 = @floatFromInt(c.s.height);
    // Fast u8 path: all four taps inside, rows read as 8-byte words.
    const fast = T == u8 and kcn != 0 and c.s.stride * c.s.height < (1 << 31);
    const row_bytes = c.s.rowLen();
    // A word read at the last source row must stay inside the row.
    const xsafe_last: f32 = if (row_bytes >= 8) @floatFromInt((row_bytes - 8) / cn) else -1;
    for (y0..y1) |y| {
        const d = c.d.row(y);
        const yf: f32 = @floatFromInt(y);
        const bx = c.m[1] * yf + c.m[2];
        const by = c.m[4] * yf + c.m[5];
        var x: usize = 0;
        while (x < c.d.width) : (x += WN) {
            const xv = @as(VF, @splat(@floatFromInt(x))) + iv;
            const sx = @mulAdd(VF, @splat(c.m[0]), xv, @splat(bx));
            const sy = @mulAdd(VF, @splat(c.m[3]), xv, @splat(by));
            const n = @min(WN, c.d.width - x);
            if (c.interp == .nearest) {
                const sxa: [WN]f32 = sx;
                const sya: [WN]f32 = sy;
                for (0..n) |i| nearestPixel(T, c, sxa[i], sya[i], d[(x + i) * cn ..][0..cn]);
                continue;
            }
            const fx = @floor(sx);
            const fy = @floor(sy);
            if (fast and n == WN) {
                const zero: VF = @splat(0);
                const ok = (fx >= zero) & (fy >= zero) & (fx <= @as(VF, @splat(sw - 2))) & (fy <= @as(VF, @splat(sh - 2))) &
                    ((fy < @as(VF, @splat(sh - 2))) | (fx <= @as(VF, @splat(xsafe_last))));
                if (@reduce(.And, ok)) {
                    warpLinear16(kcn, c.s, fx, fy, sx - fx, sy - fy, d[x * cn ..]);
                    continue;
                }
            }
            if (c.border == .constant) {
                // Whole block outside: every tap is the border value.
                const out = (fx < @as(VF, @splat(-1))) | (fy < @as(VF, @splat(-1))) | (fx >= @as(VF, @splat(sw))) | (fy >= @as(VF, @splat(sh)));
                if (@reduce(.And, out)) {
                    @memset(d[x * cn ..][0 .. n * cn], c.value);
                    continue;
                }
            }
            const sxa: [WN]f32 = sx;
            const sya: [WN]f32 = sy;
            for (0..n) |i| linearPixel(T, c, sxa[i], sya[i], d[(x + i) * cn ..][0..cn]);
        }
    }
}

/// Clamps a float coordinate into a range where it converts safely; any
/// value this far out is a border pixel anyway.
inline fn toIdx(f: f32) isize {
    return @intFromFloat(@min(@max(f, -1.0e9), 1.0e9));
}

fn nearestPixel(comptime T: type, c: anytype, sx: f32, sy: f32, o: []T) void {
    const ix = toIdx(sat.rint(@min(@max(sx, -4.0e6), 4.0e6)));
    const iy = toIdx(sat.rint(@min(@max(sy, -4.0e6), 4.0e6)));
    const mx = border_mod.map(ix, c.s.width, c.border);
    const my = border_mod.map(iy, c.s.height, c.border);
    if (mx != null and my != null) @memcpy(o, c.s.pixel(mx.?, my.?)) else @memset(o, c.value);
}

inline fn roundTo(comptime T: type, v: f32) T {
    return if (@typeInfo(T) == .float) @floatCast(v) else sat.cast(T, v);
}

fn linearPixel(comptime T: type, c: anytype, sx: f32, sy: f32, o: []T) void {
    const fx = @floor(sx);
    const fy = @floor(sy);
    const ax = sx - fx;
    const ay = sy - fy;
    const ix = toIdx(fx);
    const iy = toIdx(fy);
    const cv = toF(T, c.value);
    if (c.border == .constant and (ix < -1 or iy < -1 or ix >= c.s.width or iy >= c.s.height)) {
        @memset(o, c.value);
        return;
    }
    const xs = [2]?usize{ border_mod.map(ix, c.s.width, c.border), border_mod.map(ix + 1, c.s.width, c.border) };
    const ys = [2]?usize{ border_mod.map(iy, c.s.height, c.border), border_mod.map(iy + 1, c.s.height, c.border) };
    for (o, 0..) |*out, ch| {
        var p: [4]f32 = undefined;
        inline for (0..4) |k| {
            const px = xs[k & 1];
            const py = ys[k >> 1];
            p[k] = if (px != null and py != null) toF(T, c.s.at(px.?, py.?, ch)) else cv;
        }
        const v0 = @mulAdd(f32, ax, p[1] - p[0], p[0]);
        const v1 = @mulAdd(f32, ax, p[3] - p[2], p[2]);
        out.* = roundTo(T, @mulAdd(f32, ay, v1 - v0, v0));
    }
}

/// 16 bilinear u8 pixels whose taps are all inside the source.
inline fn warpLinear16(comptime cn: usize, s: Image(u8), fx: VF, fy: VF, ax: VF, ay: VF, d: []u8) void {
    const U32 = @Vector(WN, u32);
    const ixv: U32 = @intFromFloat(fx);
    const iyv: U32 = @intFromFloat(fy);
    const off: [WN]u32 = iyv * @as(U32, @splat(@intCast(s.stride))) + ixv * @as(U32, @splat(cn));
    const base = s.data;
    const stride = s.stride;
    var w0: [WN]u64 = undefined;
    var w1: [WN]u64 = undefined;
    inline for (0..WN) |i| {
        const p = base + off[i];
        w0[i] = std.mem.readInt(u64, p[0..8], .little);
        w1[i] = std.mem.readInt(u64, p[stride..][0..8], .little);
    }
    const r0: @Vector(WN, u64) = w0;
    const r1: @Vector(WN, u64) = w1;
    // Low and high 32-bit halves of each word, as single-register vectors.
    const words = [4]U32{ @truncate(r0), @truncate(r0 >> @splat(32)), @truncate(r1), @truncate(r1 >> @splat(32)) };
    var outs: [cn]@Vector(WN, u8) = undefined;
    inline for (0..cn) |ch| {
        const p00 = byteF(words[0..2], ch);
        const p01 = byteF(words[0..2], cn + ch);
        const p10 = byteF(words[2..4], ch);
        const p11 = byteF(words[2..4], cn + ch);
        const v0 = @mulAdd(VF, ax, p01 - p00, p00);
        const v1 = @mulAdd(VF, ax, p11 - p10, p10);
        const v = @mulAdd(VF, ay, v1 - v0, v0);
        // Round half to even via the 1.5*2^23 trick; v is within [0, 255].
        const bits: @Vector(WN, i32) = @bitCast(v + @as(VF, @splat(12582912.0)));
        const q = bits - @as(@Vector(WN, i32), @splat(0x4B400000));
        const cl = @min(@max(q, @as(@Vector(WN, i32), @splat(0))), @as(@Vector(WN, i32), @splat(255)));
        outs[ch] = core.simd.narrow(u8, cl);
    }
    if (cn == 1) {
        d[0..16].* = outs[0];
    } else {
        core.vec.store(cn, outs, d.ptr);
    }
}

/// Byte `k` (0..7) of the 64-bit words split into (low, high) halves.
inline fn byteF(w: *const [2]@Vector(WN, u32), comptime k: usize) VF {
    const v = w[k / 4];
    const sh: u5 = (k % 4) * 8;
    const b = if (sh == 24) v >> @splat(24) else (v >> @splat(sh)) & @as(@Vector(WN, u32), @splat(0xff));
    const i: @Vector(WN, i32) = @bitCast(b);
    return @floatFromInt(i);
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

test "tiled rotate and SIMD flip match scalar loops" {
    const gpa = std.testing.allocator;
    inline for (.{ 1, 2, 3, 4 }) |cn| {
        var src = try Image(u8).init(gpa, 83, 70, cn);
        defer src.deinit(gpa);
        for (src.slice(), 0..) |*v, i| v.* = @truncate(i *% 31 +% (i >> 7));
        var a = try Image(u8).init(gpa, 70, 83, cn);
        defer a.deinit(gpa);
        var b = try Image(u8).init(gpa, 70, 83, cn);
        defer b.deinit(gpa);
        inline for (.{ RotKind.transpose, RotKind.cw, RotKind.ccw }) |k| {
            try rot(.serial, src, a, k);
            rotScalar(u8, k, src, b, 0, b.height, 0, b.width);
            try std.testing.expectEqualSlices(u8, b.slice(), a.slice());
        }
        var f = try Image(u8).init(gpa, 83, 70, cn);
        defer f.deinit(gpa);
        try flip(.serial, src, f, .horizontal);
        for (0..70) |y| for (0..83) |x| {
            try std.testing.expectEqualSlices(u8, src.pixel(@intCast(82 - x), @intCast(y)), f.pixel(@intCast(x), @intCast(y)));
        };
    }
}
