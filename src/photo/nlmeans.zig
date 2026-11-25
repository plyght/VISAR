//! Non-local means denoising (OpenCV's fastNlMeansDenoising and
//! fastNlMeansDenoisingColored), bit-exact with OpenCV.
//!
//! OpenCV weighs every pixel q of the search window around p by
//! `w = table[SSD(p, q) >> shift]`, where SSD is the integer sum of squared
//! differences of the template windows around p and q, and `table` holds
//! `round(fixed * exp(-dist / h^2))` with `dist` the SSD divided by an
//! approximation of the template area (a power of two). The estimate is the
//! integer `(sum w*I(q) + sum w / 2) / sum w`. Everything after the exp
//! table is integer arithmetic on exact sums, so any evaluation order gives
//! OpenCV's result.
//!
//! OpenCV walks the image pixel by pixel, keeping column sums of the patch
//! distances for all 441 offsets of a 21x21 window. VISAR walks offset by
//! offset instead, over strips of rows, and uses the symmetry of the patch
//! distance (SSD(p, p+o) == SSD(p+o, p)): each distance (and each table
//! lookup) serves both pixels, so only half the offsets are visited. Per
//! offset, the distances of a strip are box sums of the squared-difference
//! image: a running column sum down the strip, then a 7-wide row sum, all
//! in vectors.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const border_mod = @import("../imgproc/border.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;

pub const Error = core.Error || Allocator.Error || error{InvalidArgument};

pub const Options = struct {
    /// Filter strength: larger removes more noise (and detail).
    h: f32 = 3,
    /// Side of the compared patches (odd; OpenCV's templateWindowSize).
    template_window_size: u32 = 7,
    /// Side of the window searched for similar patches (odd; OpenCV's
    /// searchWindowSize).
    search_window_size: u32 = 21,
};

/// Lanes of the i32 kernels.
const V = 16;
const VI = @Vector(V, i32);
const VU = @Vector(V, u32);

/// OpenCV's cvRound for doubles: round half to even.
fn roundEven(x: f64) i64 {
    const r = @round(x);
    if (@abs(x - @trunc(x)) == 0.5) return @intFromFloat(2.0 * @round(x / 2.0));
    return @intFromFloat(r);
}

/// The almost_dist2weight table of OpenCV's FastNlMeansDenoisingInvoker,
/// truncated after its last non-zero entry (every distance at or above
/// `max_idx` weighs table[max_idx]).
const Weights = struct {
    tbl: []i32,
    shift: u5,
    max_idx: u32,

    fn init(gpa: Allocator, h: f32, cn: u32, tsize: u32, ssize: u32) !Weights {
        const max_est: i32 = @intCast(ssize * ssize * 255);
        const fixed: i32 = @divTrunc(std.math.maxInt(i32), max_est);
        const tsq = tsize * tsize;
        var shift: u5 = 0;
        while ((@as(u32, 1) << shift) < tsq) shift += 1;
        const mult = @as(f64, @floatFromInt(@as(u32, 1) << shift)) / @as(f64, @floatFromInt(tsq));
        const max_dist: u32 = 255 * 255 * cn;
        const almost_max: u32 = @intFromFloat(@as(f64, @floatFromInt(max_dist)) / mult + 1);
        // OpenCV: h[0]*h[0] * channels in float, then the double division.
        const den: f64 = @floatCast((h * h) * @as(f32, @floatFromInt(cn)));
        var n: u32 = almost_max;
        var tbl = try gpa.alloc(i32, almost_max);
        errdefer gpa.free(tbl);
        for (0..almost_max) |i| {
            const dist = @as(f64, @floatFromInt(i)) * mult;
            var w = @exp(-dist / den);
            if (std.math.isNan(w)) w = 1.0;
            var weight: i32 = @intCast(roundEven(@as(f64, @floatFromInt(fixed)) * w));
            if (@as(f64, @floatFromInt(weight)) < 0.001 * @as(f64, @floatFromInt(fixed))) weight = 0;
            tbl[i] = weight;
            // The weights never increase with the distance: past the first
            // zero they all are.
            if (weight == 0) {
                n = @intCast(i + 1);
                break;
            }
        }
        if (n < almost_max) tbl = try gpa.realloc(tbl, n);
        return .{ .tbl = tbl, .shift = shift, .max_idx = n - 1 };
    }

    fn deinit(self: *Weights, gpa: Allocator) void {
        gpa.free(self.tbl);
    }
};

const x86_gather = builtin.zig_backend == .stage2_llvm and builtin.cpu.arch == .x86_64 and
    std.Target.x86.featureSetHas(builtin.cpu.features, .avx2);

/// tbl[idx] for every lane.
inline fn lookup(tbl: [*]const i32, idx: VU) VI {
    if (comptime x86_gather) {
        const ia: [V]u32 = idx;
        var r: [V]i32 = undefined;
        inline for (0..V / 8) |k| {
            const iv: @Vector(8, u32) = ia[k * 8 ..][0..8].*;
            var m: @Vector(8, i32) = undefined;
            r[k * 8 ..][0..8].* = asm ("vpcmpeqd %[m], %[m], %[m]\n\tvpxor %[r], %[r], %[r]\n\tvpgatherdd %[m], (%[b],%[i],4), %[r]"
                : [r] "=&x" (-> @Vector(8, i32)),
                  [m] "=&x" (m),
                : [b] "r" (tbl),
                  [i] "x" (iv),
                : .{ .memory = true });
        }
        return r;
    }
    const ia: [V]u32 = idx;
    var r: [V]i32 = undefined;
    inline for (0..V) |k| r[k] = tbl[ia[k]];
    return r;
}

/// Geometry shared by all strips.
fn Plan(comptime cn: usize) type {
    return struct {
        const Self = @This();
        w: usize,
        h: usize,
        s: usize, // search half size
        t: usize, // template half size
        /// Extended source planes: rows -(s+t) .. h+s+t, columns
        /// -(2s+t) .. stride-(2s+t); OpenCV's reflect-101 border of s+t,
        /// zeros beyond.
        planes: [cn][]u8,
        es: usize,
        /// Center columns (from -s) rounded up to vectors.
        wcv: usize,
        /// Distance row width (from -s-t), a multiple of V.
        dwv: usize,
        /// Accumulator row width (from -2s).
        aw: usize,
        /// Center rows per strip.
        strip: usize,
        weights: *const Weights,
        /// Image-sized sums: one per channel, then the weight sums.
        acc: [cn + 1][]i32,
        /// Contributions past the end of each strip, s rows per strip.
        spill: [cn + 1][]i32,

        inline fn eRow(self: *const Self, ch: usize, y: isize) [*]const u8 {
            const r: usize = @intCast(y + @as(isize, @intCast(self.s + self.t)));
            return self.planes[ch].ptr + r * self.es;
        }

        fn nStrips(self: *const Self) usize {
            return (self.h + self.s + self.strip - 1) / self.strip;
        }

        /// Processes strip `si`: center rows [y0, y0+strip) with y0 =
        /// si*strip - s.
        fn runStrip(self: *const Self, si: usize, scratch: []u8) void {
            const s = self.s;
            const t = self.t;
            const P = self.strip;
            const y0: isize = @as(isize, @intCast(si * P)) - @as(isize, @intCast(s));
            const y1: isize = @min(y0 + @as(isize, @intCast(P)), @as(isize, @intCast(self.h)));
            const rows: usize = @intCast(y1 - y0);
            const dwv = self.dwv;
            const aw = self.aw;
            // Scratch layout: distance rows, column sums, local sums.
            const fl: [*]u32 = @ptrCast(@alignCast(scratch.ptr));
            const drows = rows + 2 * t;
            const dbuf = fl[0 .. drows * dwv];
            const colsum = fl[drows * dwv ..][0..dwv];
            const arows = rows + s;
            const accp: [*]i32 = @ptrCast(@alignCast(fl + drows * dwv + dwv));
            var acc: [cn + 1][]i32 = undefined;
            for (0..cn + 1) |c| {
                acc[c] = accp[c * arows * aw ..][0 .. arows * aw];
                @memset(acc[c], 0);
            }
            const tbl = self.weights.tbl.ptr;
            const shift: @Vector(V, u5) = @splat(self.weights.shift);
            const max_idx: VU = @splat(self.weights.max_idx);
            const ntap = 2 * t + 1;

            // Offsets (dy, dx) with dy > 0, or dy == 0 and dx > 0.
            const si_: isize = @intCast(s);
            var dy: isize = 0;
            while (dy <= si_) : (dy += 1) {
                var dx: isize = if (dy == 0) 1 else -si_;
                while (dx <= si_) : (dx += 1) {
                    // Squared differences, rows y0-t .. y1+t, columns -s-t ..
                    for (0..drows) |r| {
                        const y = y0 - @as(isize, @intCast(t)) + @as(isize, @intCast(r));
                        const drow = dbuf[r * dwv ..][0..dwv];
                        var a: [cn][*]const u8 = undefined;
                        var b: [cn][*]const u8 = undefined;
                        inline for (0..cn) |c| {
                            // Column -s-t of the distance row is plane column s.
                            a[c] = self.eRow(c, y) + s;
                            b[c] = self.eRow(c, y + dy) + @as(usize, @intCast(@as(isize, @intCast(s)) + dx));
                        }
                        var x: usize = 0;
                        while (x < dwv) : (x += V) {
                            var acc_d: VU = @splat(0);
                            inline for (0..cn) |c| {
                                const av: @Vector(V, u8) = a[c][x..][0..V].*;
                                const bv: @Vector(V, u8) = b[c][x..][0..V].*;
                                const d: @Vector(V, u16) = @max(av, bv) - @min(av, bv);
                                acc_d += @as(VU, d * d);
                            }
                            drow[x..][0..V].* = acc_d;
                        }
                    }
                    // Column sums of the first center row.
                    @memcpy(colsum, dbuf[0..dwv]);
                    for (1..ntap) |r| {
                        var x: usize = 0;
                        while (x < dwv) : (x += V) {
                            const cv: VU = colsum[x..][0..V].*;
                            const dv: VU = dbuf[r * dwv + x ..][0..V].*;
                            colsum[x..][0..V].* = cv + dv;
                        }
                    }
                    const dyu: usize = @intCast(dy);
                    for (0..rows) |k| {
                        if (k > 0) {
                            const add = dbuf[(k + 2 * t) * dwv ..][0..dwv];
                            const sub = dbuf[(k - 1) * dwv ..][0..dwv];
                            var x: usize = 0;
                            while (x < dwv) : (x += V) {
                                const cv: VU = colsum[x..][0..V].*;
                                const av: VU = add[x..][0..V].*;
                                const sv: VU = sub[x..][0..V].*;
                                colsum[x..][0..V].* = cv +% av -% sv;
                            }
                        }
                        const y = y0 + @as(isize, @intCast(k));
                        // Both pixels outside the image: nothing to add.
                        if (y + dy < 0) continue;
                        var ep: [cn][*]const u8 = undefined;
                        var eq: [cn][*]const u8 = undefined;
                        inline for (0..cn) |c| {
                            // Center column -s is plane column s + t.
                            ep[c] = self.eRow(c, y) + s + t;
                            eq[c] = self.eRow(c, y + dy) + @as(usize, @intCast(@as(isize, @intCast(s + t)) + dx));
                        }
                        // Center column x is accumulator column x + s.
                        const ap = k * aw + s;
                        const aq: usize = @intCast(@as(isize, @intCast((k + dyu) * aw + s)) + dx);
                        var x: usize = 0;
                        while (x < self.wcv) : (x += V) {
                            var ssd: VU = colsum[x..][0..V].*;
                            inline for (1..8) |tx| {
                                if (tx < ntap) ssd += colsum[x + tx ..][0..V].*;
                            }
                            var tx: usize = 8;
                            while (tx < ntap) : (tx += 1) ssd += colsum[x + tx ..][0..V].*;
                            const idx = @min(ssd >> shift, max_idx);
                            const w = lookup(tbl, idx);
                            inline for (0..cn) |c| {
                                const pv: @Vector(V, u8) = ep[c][x..][0..V].*;
                                const qv: @Vector(V, u8) = eq[c][x..][0..V].*;
                                const pi: VI = @intCast(@as(@Vector(V, u16), pv));
                                const qi: VI = @intCast(@as(@Vector(V, u16), qv));
                                const a0: VI = acc[c][ap + x ..][0..V].*;
                                acc[c][ap + x ..][0..V].* = a0 +% w * qi;
                                const a1: VI = acc[c][aq + x ..][0..V].*;
                                acc[c][aq + x ..][0..V].* = a1 +% w * pi;
                            }
                            const w0: VI = acc[cn][ap + x ..][0..V].*;
                            acc[cn][ap + x ..][0..V].* = w0 +% w;
                            const w1: VI = acc[cn][aq + x ..][0..V].*;
                            acc[cn][aq + x ..][0..V].* = w1 +% w;
                        }
                    }
                }
            }
            // Own rows go to the image sums, the s rows after the strip to
            // its spill buffer.
            for (0..arows) |k| {
                const y = y0 + @as(isize, @intCast(k));
                if (y < 0 or y >= @as(isize, @intCast(self.h))) continue;
                const yu: usize = @intCast(y);
                for (0..cn + 1) |c| {
                    // Image column 0 is accumulator column 2s.
                    const src = acc[c][k * aw + 2 * s ..][0..self.w];
                    if (k < rows) {
                        @memcpy(self.acc[c][yu * self.w ..][0..self.w], src);
                    } else {
                        @memcpy(self.spill[c][(si * s + (k - rows)) * self.w ..][0..self.w], src);
                    }
                }
            }
        }

        fn scratchBytes(self: *const Self) usize {
            const drows = self.strip + 2 * self.t;
            const arows = self.strip + self.s;
            return 4 * (drows * self.dwv + self.dwv + (cn + 1) * arows * self.aw);
        }
    };
}

fn roundUp(x: usize, m: usize) usize {
    return (x + m - 1) / m * m;
}

/// Denoises the planes `src` (each w*h, row-major) into `dst` planes.
fn denoisePlanes(comptime cn: usize, exec: Exec, gpa: Allocator, src: [cn][]const u8, dst: [cn][]u8, w: usize, h: usize, hval: f32, tsize_in: u32, ssize_in: u32) Error!void {
    const t: usize = tsize_in / 2;
    const s: usize = ssize_in / 2;
    const tsize: u32 = @intCast(2 * t + 1);
    const ssize: u32 = @intCast(2 * s + 1);
    var weights = try Weights.init(gpa, hval, cn, tsize, ssize);
    defer weights.deinit(gpa);

    const wc = w + 2 * s;
    const wcv = roundUp(wc, V);
    const dwv = roundUp(wcv + 2 * t, V);
    const es = roundUp(dwv + 2 * s + V, 64);
    const eh = h + 2 * (s + t);
    const aw = roundUp(wcv + 2 * s + V, V);
    const strip: usize = @max(s, 32);

    var plan: Plan(cn) = undefined;
    plan.w = w;
    plan.h = h;
    plan.s = s;
    plan.t = t;
    plan.es = es;
    plan.wcv = wcv;
    plan.dwv = dwv;
    plan.aw = aw;
    plan.strip = strip;
    plan.weights = &weights;

    // Extended planes.
    const ebuf = try gpa.alignedAlloc(u8, .@"64", cn * es * eh);
    defer gpa.free(ebuf);
    const bsz: isize = @intCast(s + t);
    const ox: isize = @intCast(2 * s + t);
    for (0..cn) |c| {
        const plane = ebuf[c * es * eh ..][0 .. es * eh];
        plan.planes[c] = plane;
        for (0..eh) |er| {
            const y = @as(isize, @intCast(er)) - bsz;
            const sy = border_mod.map(y, h, .reflect101).?;
            const srow = src[c][sy * w ..][0..w];
            const drow = plane[er * es ..][0..es];
            for (drow, 0..) |*o, ec| {
                const x = @as(isize, @intCast(ec)) - ox;
                o.* = if (x < -bsz or x >= @as(isize, @intCast(w)) + bsz) 0 else srow[border_mod.map(x, w, .reflect101).?];
            }
        }
    }

    const nstrips = plan.nStrips();
    const accbuf = try gpa.alignedAlloc(i32, .@"64", (cn + 1) * w * h);
    defer gpa.free(accbuf);
    const spillbuf = try gpa.alignedAlloc(i32, .@"64", (cn + 1) * nstrips * s * w);
    defer gpa.free(spillbuf);
    @memset(spillbuf, 0);
    for (0..cn + 1) |c| {
        plan.acc[c] = accbuf[c * w * h ..][0 .. w * h];
        plan.spill[c] = spillbuf[c * nstrips * s * w ..][0 .. nstrips * s * w];
    }

    const Ctx = struct {
        plan: *const Plan(cn),
        gpa: Allocator,
        failed: *std.atomic.Value(bool),
    };
    var failed = std.atomic.Value(bool).init(false);
    exec.parallelFor(nstrips, 1, Ctx{ .plan = &plan, .gpa = gpa, .failed = &failed }, struct {
        fn f(ctx: Ctx, a: usize, b: usize) void {
            const bytes = ctx.plan.scratchBytes();
            const scratch = ctx.gpa.alignedAlloc(u8, .@"64", bytes) catch {
                ctx.failed.store(true, .monotonic);
                return;
            };
            defer ctx.gpa.free(scratch);
            for (a..b) |si| ctx.plan.runStrip(si, scratch);
        }
    }.f);
    if (failed.load(.monotonic)) return error.OutOfMemory;

    // Add the spills: strip si's spill covers rows y1 .. y1+s.
    for (0..nstrips) |si| {
        const y1 = @as(isize, @intCast((si + 1) * strip)) - @as(isize, @intCast(s));
        for (0..s) |k| {
            const y = y1 + @as(isize, @intCast(k));
            if (y < 0 or y >= @as(isize, @intCast(h))) continue;
            const yu: usize = @intCast(y);
            for (0..cn + 1) |c| {
                const d = plan.acc[c][yu * w ..][0..w];
                const sp = plan.spill[c][(si * s + k) * w ..][0..w];
                for (d, sp) |*o, v| o.* +%= v;
            }
        }
    }

    // The center pixel (distance 0) and the division.
    const w0: i32 = weights.tbl[0];
    const Fin = struct {
        plan: *const Plan(cn),
        src: [cn][]const u8,
        dst: [cn][]u8,
        w0: i32,
    };
    exec.parallelFor(h, 16, Fin{ .plan = &plan, .src = src, .dst = dst, .w0 = w0 }, struct {
        fn f(ctx: Fin, a: usize, b: usize) void {
            const ww = ctx.plan.w;
            for (a * ww..b * ww) |i| {
                const ws: u32 = @bitCast(ctx.plan.acc[cn][i] +% ctx.w0);
                inline for (0..cn) |c| {
                    const e: u32 = @bitCast(ctx.plan.acc[c][i] +% ctx.w0 * @as(i32, ctx.src[c][i]));
                    const v = (e + ws / 2) / ws;
                    ctx.dst[c][i] = @intCast(@min(v, 255));
                }
            }
        }
    }.f);
}

fn checkWindows(o: Options) Error!void {
    if (o.template_window_size == 0 or o.search_window_size == 0) return error.InvalidArgument;
    if (o.template_window_size > 46340) return error.InvalidArgument;
}

/// Denoises a 1- to 4-channel 8-bit image with one filter strength for all
/// channels (OpenCV's fastNlMeansDenoising with NORM_L2). `dst` may be
/// `src`.
pub fn fastNlMeansDenoising(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), opt: Options) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    try checkWindows(opt);
    return switch (src.channels) {
        1 => denoiseImage(1, exec, gpa, src, dst, opt),
        2 => denoiseImage(2, exec, gpa, src, dst, opt),
        3 => denoiseImage(3, exec, gpa, src, dst, opt),
        4 => denoiseImage(4, exec, gpa, src, dst, opt),
        else => error.UnsupportedChannels,
    };
}

fn denoiseImage(comptime cn: usize, exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), opt: Options) Error!void {
    const w: usize = src.width;
    const h: usize = src.height;
    const buf = try gpa.alloc(u8, 2 * cn * w * h);
    defer gpa.free(buf);
    var sp: [cn][]u8 = undefined;
    var dp: [cn][]u8 = undefined;
    for (0..cn) |c| {
        sp[c] = buf[c * w * h ..][0 .. w * h];
        dp[c] = buf[(cn + c) * w * h ..][0 .. w * h];
    }
    for (0..h) |y| {
        const r = src.row(y);
        for (0..w) |x| inline for (0..cn) |c| {
            sp[c][y * w + x] = r[x * cn + c];
        };
    }
    var spc: [cn][]const u8 = undefined;
    for (0..cn) |c| spc[c] = sp[c];
    try denoisePlanes(cn, exec, gpa, spc, dp, w, h, opt.h, opt.template_window_size, opt.search_window_size);
    for (0..h) |y| {
        const r = dst.row(y);
        for (0..w) |x| inline for (0..cn) |c| {
            r[x * cn + c] = dp[c][y * w + x];
        };
    }
}

/// Denoises a BGR (or BGRA) 8-bit image as OpenCV's
/// fastNlMeansDenoisingColored: the luminance of its (linear) Lab
/// conversion with strength `h`, the a/b planes together with
/// `h_color`, and back to BGR. A 4-channel image gets alpha 255, as in
/// OpenCV. `dst` may be `src`.
pub fn fastNlMeansDenoisingColored(exec: Exec, gpa: Allocator, src: Image(u8), dst: Image(u8), h: f32, h_color: f32, template_window_size: u32, search_window_size: u32) Error!void {
    if (!src.sameShape(dst)) return error.SizeMismatch;
    if (src.channels != 3 and src.channels != 4) return error.UnsupportedChannels;
    try checkWindows(.{ .template_window_size = template_window_size, .search_window_size = search_window_size });
    const lab = @import("../imgproc/lab.zig");
    const w: usize = src.width;
    const hh: usize = src.height;
    const n = w * hh;
    const buf = try gpa.alloc(u8, 6 * n);
    defer gpa.free(buf);
    const l = buf[0..n];
    const a = buf[n .. 2 * n];
    const b = buf[2 * n .. 3 * n];
    const lo = buf[3 * n .. 4 * n];
    const ao = buf[4 * n .. 5 * n];
    const bo = buf[5 * n .. 6 * n];
    const sc: usize = src.channels;
    for (0..hh) |y| {
        const r = src.row(y);
        for (0..w) |x| {
            const v = lab.lbgrToLab(r[x * sc], r[x * sc + 1], r[x * sc + 2]);
            l[y * w + x] = v[0];
            a[y * w + x] = v[1];
            b[y * w + x] = v[2];
        }
    }
    try denoisePlanes(1, exec, gpa, .{l}, .{lo}, w, hh, h, template_window_size, search_window_size);
    try denoisePlanes(2, exec, gpa, .{ a, b }, .{ ao, bo }, w, hh, h_color, template_window_size, search_window_size);
    for (0..hh) |y| {
        const r = dst.row(y);
        for (0..w) |x| {
            const i = y * w + x;
            const v = lab.labToLbgr(lo[i], ao[i], bo[i]);
            r[x * sc] = v[0];
            r[x * sc + 1] = v[1];
            r[x * sc + 2] = v[2];
            if (sc == 4) r[x * sc + 3] = 255;
        }
    }
}

test "weights table matches OpenCV's constants" {
    var w = try Weights.init(std.testing.allocator, 10, 1, 7, 21);
    defer w.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(u5, 6), w.shift);
    try std.testing.expectEqual(@as(i32, 19096), w.tbl[0]);
    try std.testing.expectEqual(@as(i32, 0), w.tbl[w.max_idx]);
}

test "denoise constant image is identity" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 37, 23, 1, 77);
    defer img.deinit(gpa);
    try fastNlMeansDenoising(.serial, gpa, img, img, .{ .h = 10 });
    for (img.slice()) |v| try std.testing.expectEqual(@as(u8, 77), v);
}
