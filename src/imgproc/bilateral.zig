//! Bilateral filter (cv::bilateralFilter) for 8-bit images with 1 or 3
//! channels.
//!
//! Each output pixel is the average of the pixels within `radius`, weighted
//! by exp(-r^2 / (2 sigma_space^2)) * exp(-d^2 / (2 sigma_color^2)), with r
//! the distance and d the colour difference (summed over channels, as in
//! OpenCV). OpenCV's own result depends on its build: IPP on x86, its SIMD
//! exp approximation and fused multiply-adds elsewhere; VISAR evaluates the
//! colour weight with a vectorised exp of about one ulp and sums the taps
//! in OpenCV's order. Outputs agree to within one level.

const std = @import("std");
const core = @import("../core/core.zig");
const border_mod = @import("border.zig");
const Border = border_mod.Border;
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error;

pub const Options = struct {
    /// Diameter of the neighbourhood; <= 0 derives it from sigma_space.
    d: i32 = 9,
    sigma_color: f64 = 75,
    sigma_space: f64 = 75,
    /// BORDER_WRAP is not supported (as in OpenCV).
    border: Border = .default,
};

const L = 16;
const VF = @Vector(L, f32);

const builtin = @import("builtin");
const x86_gather = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

/// tbl[idx] for every lane (AVX2 gathers on x86).
inline fn gatherF(tbl: [*]const f32, idx: @Vector(L, u32)) VF {
    const ia: [L]u32 = idx;
    var r: [L]f32 = undefined;
    if (comptime x86_gather) {
        inline for (0..L / 8) |k| {
            const iv: @Vector(8, u32) = ia[k * 8 ..][0..8].*;
            var m: @Vector(8, i32) = undefined;
            r[k * 8 ..][0..8].* = asm ("vpcmpeqd %[m], %[m], %[m]\n\tvxorps %[r], %[r], %[r]\n\tvgatherdps %[m], (%[b],%[i],4), %[r]"
                : [r] "=&x" (-> @Vector(8, f32)),
                  [m] "=&x" (m),
                : [b] "r" (tbl),
                  [i] "x" (iv),
                : .{ .memory = true });
        }
        return r;
    }
    inline for (0..L) |k| r[k] = tbl[ia[k]];
    return r;
}

/// cvRound of a non-negative double (half to even), clamped to the range.
fn roundEven(x: f64) usize {
    if (!(x > 0)) return 0;
    const c = @min(x, 1e9);
    const r = if (@abs(c - @trunc(c)) == 0.5) 2.0 * @round(c / 2.0) else @round(c);
    return @intFromFloat(r);
}

/// saturate_cast<uchar>(float) for in-range values: round half to even.
inline fn roundU8(v: VF) @Vector(L, u8) {
    const m: VF = @splat(12582912.0);
    const r = (v + m) - m;
    const c = @min(@max(r, @as(VF, @splat(0))), @as(VF, @splat(255)));
    const u: @Vector(L, u32) = @intFromFloat(c);
    return @intCast(u);
}

pub fn bilateralFilter(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), o: Options) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const cn: usize = src.channels;
    if (cn != 1 and cn != 3) return error.UnsupportedChannels;
    if (o.border == .wrap) return error.UnsupportedChannels;
    const eps = 1e-6;
    if (o.sigma_color <= eps or o.sigma_space <= eps or std.math.isNan(o.sigma_color) or std.math.isNan(o.sigma_space)) {
        if (src.data != dst.data) for (0..src.height) |y| @memcpy(dst.row(y), src.row(y));
        return;
    }
    if (src.data == dst.data) return error.SizeMismatch;
    const gcc: f32 = @floatCast(-0.5 / (o.sigma_color * o.sigma_color));
    const gsc: f32 = @floatCast(-0.5 / (o.sigma_space * o.sigma_space));
    var radius: usize = if (o.d <= 0) roundEven(o.sigma_space * 1.5) else @intCast(@divTrunc(o.d, 2));
    radius = @max(radius, 1);
    // OpenCV caps nothing; keep the tap list bounded.
    if (radius > 1000) return error.InvalidSize;

    const w: usize = src.width;
    const h: usize = src.height;
    const pw = w + 2 * radius;
    const ph = h + 2 * radius;
    const pws = (pw + L) / L * L + L;
    // Padded planes, as f32.
    const planes = try gpa.alloc(f32, cn * ph * pws);
    defer gpa.free(planes);
    // Lanes past the row end read zeros (their colour index stays small).
    @memset(planes, 0);
    // OpenCV's colour table: exp(float(i * i) * coeff).
    const cw = try gpa.alloc(f32, 256 * cn);
    defer gpa.free(cw);
    for (cw, 0..) |*v, i| v.* = @floatCast(@exp(@as(f64, @as(f32, @floatFromInt(i * i)) * gcc)));
    for (0..ph) |py| {
        const sy = border_mod.map(@as(isize, @intCast(py)) - @as(isize, @intCast(radius)), h, o.border) orelse continue;
        const srow = src.row(sy);
        for (0..cn) |c| {
            const prow = planes[(c * ph + py) * pws ..][0..pw];
            for (prow[radius..][0..w], 0..) |*v, x| v.* = @floatFromInt(srow[x * cn + c]);
            for (0..radius) |k| {
                const l = border_mod.map(@as(isize, @intCast(k)) - @as(isize, @intCast(radius)), w, o.border);
                prow[k] = if (l) |q| @floatFromInt(srow[q * cn + c]) else 0;
                const r = border_mod.map(@as(isize, @intCast(w + k)), w, o.border);
                prow[radius + w + k] = if (r) |q| @floatFromInt(srow[q * cn + c]) else 0;
            }
        }
    }
    // Taps in OpenCV's order, with their spatial weights.
    const side = 2 * radius + 1;
    const taps = try gpa.alloc(Tap, side * side);
    defer gpa.free(taps);
    var ntap: usize = 0;
    const ri: isize = @intCast(radius);
    var dy: isize = -ri;
    while (dy <= ri) : (dy += 1) {
        var dx: isize = -ri;
        while (dx <= ri) : (dx += 1) {
            const r = @sqrt(@as(f64, @floatFromInt(dy * dy + dx * dx)));
            if (r > @as(f64, @floatFromInt(radius))) continue;
            taps[ntap] = .{ .dy = @intCast(dy + ri), .dx = @intCast(dx + ri), .sw = @floatCast(@exp(r * r * @as(f64, gsc))) };
            ntap += 1;
        }
    }
    const ctx: Ctx = .{
        .planes = planes,
        .pws = pws,
        .ph = ph,
        .cn = cn,
        .w = w,
        .taps = taps[0..ntap],
        .cw = cw,
        .radius = radius,
        .dst = dst,
        .gpa = gpa,
    };
    var failed = std.atomic.Value(bool).init(false);
    exec.parallelFor(h, 8, Run{ .c = &ctx, .failed = &failed }, Run.f);
    if (failed.load(.monotonic)) return error.OutOfMemory;
}

const Tap = struct { dy: usize, dx: usize, sw: f32 };

const Ctx = struct {
    planes: []const f32,
    pws: usize,
    ph: usize,
    cn: usize,
    w: usize,
    taps: []const Tap,
    cw: []const f32,
    radius: usize,
    dst: Image(u8),
    gpa: Allocator,
};

const Run = struct {
    c: *const Ctx,
    failed: *std.atomic.Value(bool),

    fn f(r: Run, y0: usize, y1: usize) void {
        const c = r.c;
        const wv = (c.w + L - 1) / L * L;
        const acc = c.gpa.alloc(f32, 4 * wv) catch {
            r.failed.store(true, .monotonic);
            return;
        };
        defer c.gpa.free(acc);
        for (y0..y1) |y| {
            if (c.cn == 1) row(1, c, y, acc, wv) else row(3, c, y, acc, wv);
        }
    }
};

fn row(comptime cn: usize, c: *const Ctx, y: usize, acc: []f32, wv: usize) void {
    _ = acc;
    const pws = c.pws;
    const ph = c.ph;
    const radius = c.radius;
    var center: [cn][*]const f32 = undefined;
    inline for (0..cn) |ch| center[ch] = c.planes.ptr + (ch * ph + y + radius) * pws + radius;

    const out = c.dst.row(y);
    // A chunk of L pixels at a time, its sums in registers over all taps.
    var x: usize = 0;
    while (x < wv) : (x += L) {
        var c0: [cn]VF = undefined;
        inline for (0..cn) |ch| c0[ch] = center[ch][x..][0..L].*;
        var ws: VF = @splat(0);
        var sum: [cn]VF = @splat(@splat(0));
        for (c.taps) |t| {
            var d: VF = @splat(0);
            var v: [cn]VF = undefined;
            inline for (0..cn) |ch| {
                v[ch] = (c.planes.ptr + (ch * ph + y + t.dy) * pws + t.dx + x)[0..L].*;
                d += @abs(v[ch] - c0[ch]);
            }
            // OpenCV's colour table holds exp(float(d * d) * coeff).
            const wgt = @as(VF, @splat(t.sw)) * gatherF(c.cw.ptr, @intFromFloat(d));
            ws += wgt;
            inline for (0..cn) |ch| sum[ch] += v[ch] * wgt;
        }
        var o: [cn][L]u8 = undefined;
        inline for (0..cn) |ch| {
            // OpenCV: sum / wsum for one channel, sum * (1 / wsum) for three.
            o[ch] = if (cn == 1) roundU8(sum[ch] / ws) else roundU8(sum[ch] * (@as(VF, @splat(1.0)) / ws));
        }
        const n = @min(L, c.w - x);
        for (0..n) |q| inline for (0..cn) |ch| {
            out[(x + q) * cn + ch] = o[ch][q];
        };
    }
}

test "bilateral filter keeps a flat image" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 33, 17, 3, 77);
    defer img.deinit(gpa);
    var o = try Image(u8).init(gpa, 33, 17, 3);
    defer o.deinit(gpa);
    try bilateralFilter(.serial, gpa, img, o, .{});
    for (o.slice()) |v| try std.testing.expectEqual(@as(u8, 77), v);
}
