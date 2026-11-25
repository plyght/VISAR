//! MOSSE correlation-filter tracker (Bolme et al., "Visual Object Tracking
//! using Adaptive Correlation Filters", CVPR 2010), with a three-scale
//! search. Follows one selected object (no motion needed, unlike
//! background subtraction) at very high frame rates: a frame costs a few
//! 64x64 FFTs.
//!
//!   var t = try Mosse.init(gpa, gray, .{ .x = 100, .y = 80, .w = 40, .h = 30 }, .{});
//!   defer t.deinit();
//!   const r = try t.update(next_gray); // r.box, r.psr, r.ok

const std = @import("std");
const core = @import("../core/core.zig");
const fft = @import("../core/fft.zig");
const track = @import("track.zig");
const Image = core.Image;
const Complex = fft.Complex;
const Allocator = std.mem.Allocator;

pub const Box = track.Box;

pub const Options = struct {
    /// Side of the square correlation patch (power of two).
    size: usize = 64,
    /// Search region relative to the box.
    padding: f32 = 2.0,
    /// Learning rate for the running filter.
    rate: f32 = 0.125,
    /// Width (in patch pixels) of the desired Gaussian response.
    sigma: f32 = 2.0,
    /// Peak-to-sidelobe ratio below which the target counts as lost.
    min_psr: f32 = 7.0,
    /// Search three scales each frame (box size follows the object).
    scale: bool = true,
    scale_step: f32 = 1.05,
};

pub const Result = struct {
    box: Box,
    /// Peak-to-sidelobe ratio of the correlation response (confidence).
    psr: f32,
    /// False when the response is too weak: the target is probably lost
    /// or occluded. The filter is not updated on such frames.
    ok: bool,
};

pub const Mosse = struct {
    gpa: Allocator,
    o: Options,
    plan: fft.Plan2D,
    /// Numerator (complex) and denominator (real) of the filter.
    a: []Complex,
    b: []f32,
    g: []Complex,
    /// The filter A / B, refreshed whenever A and B change.
    filt: []Complex,
    window: []f32,
    /// Per-column sample positions of an axis-aligned patch.
    col_x0: []u32,
    col_fx: []f32,
    buf: []Complex,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,

    pub fn init(gpa: Allocator, frame: Image(u8), start: Box, o: Options) !Mosse {
        if (frame.channels != 1) return error.UnsupportedChannels;
        const n = o.size;
        const nn = n * n;
        var self: Mosse = .{
            .gpa = gpa,
            .o = o,
            .plan = try fft.Plan2D.init(gpa, n, n),
            .a = try gpa.alloc(Complex, nn),
            .b = try gpa.alloc(f32, nn),
            .g = try gpa.alloc(Complex, nn),
            .filt = try gpa.alloc(Complex, nn),
            .col_x0 = try gpa.alloc(u32, n),
            .col_fx = try gpa.alloc(f32, n),
            .window = try gpa.alloc(f32, nn),
            .buf = try gpa.alloc(Complex, nn),
            .cx = start.cx(),
            .cy = start.cy(),
            .w = @max(start.w, 4),
            .h = @max(start.h, 4),
        };
        // Hann window and the desired response, centred.
        const nf: f32 = @floatFromInt(n);
        const c = nf / 2;
        for (0..n) |y| for (0..n) |x| {
            const fx: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);
            self.window[y * n + x] = (0.5 - 0.5 * @cos(2 * std.math.pi * fx / (nf - 1))) * (0.5 - 0.5 * @cos(2 * std.math.pi * fy / (nf - 1)));
            const d2 = (fx - c) * (fx - c) + (fy - c) * (fy - c);
            self.g[y * n + x] = .{ .re = @exp(-d2 / (2 * o.sigma * o.sigma)), .im = 0 };
        };
        self.plan.forward(self.g);
        @memset(self.a, .{ .re = 0, .im = 0 });
        @memset(self.b, 0);
        // Train on small rotations and scalings of the first patch.
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const rnd = prng.random();
        for (0..8) |i| {
            const ang: f32 = if (i == 0) 0 else (rnd.float(f32) - 0.5) * 0.2;
            const sc: f32 = if (i == 0) 1 else 1 + (rnd.float(f32) - 0.5) * 0.1;
            self.patch(frame, self.cx, self.cy, 1.0, ang, sc);
            self.accumulate(1.0);
        }
        return self;
    }

    pub fn deinit(self: *Mosse) void {
        self.plan.deinit(self.gpa);
        self.gpa.free(self.a);
        self.gpa.free(self.b);
        self.gpa.free(self.g);
        self.gpa.free(self.filt);
        self.gpa.free(self.col_x0);
        self.gpa.free(self.col_fx);
        self.gpa.free(self.window);
        self.gpa.free(self.buf);
    }

    pub fn box(self: Mosse) Box {
        return .{ .x = self.cx - self.w / 2, .y = self.cy - self.h / 2, .w = self.w, .h = self.h };
    }

    /// Locates the target in `frame` (gray, same size as before), updates
    /// the filter when confident, and returns the new box.
    pub fn update(self: *Mosse, frame: Image(u8)) !Result {
        if (frame.channels != 1) return error.UnsupportedChannels;
        const scales = [_]f32{ 1.0 / self.o.scale_step, 1.0, self.o.scale_step };
        var best: Peak = .{ .dx = 0, .dy = 0, .value = -std.math.inf(f32), .psr = 0 };
        var best_scale: f32 = 1;
        for (scales) |s| {
            if (!self.o.scale and s != 1.0) continue;
            self.patch(frame, self.cx, self.cy, s, 0, 1);
            self.plan.forward(self.buf);
            // R = F * A / B
            for (self.buf, self.filt) |*f, h| f.* = f.*.mul(h);
            self.plan.inverse(self.buf);
            const p = self.peak();
            // Prefer the unscaled match unless another scale is clearly better.
            const v = if (s == 1.0) p.value else p.value * 0.98;
            if (v > best.value) {
                best = p;
                best.value = v;
                best_scale = s;
            }
        }
        const ok = best.psr >= self.o.min_psr;
        if (ok) {
            const n: f32 = @floatFromInt(self.o.size);
            self.cx += best.dx * self.w * self.o.padding * best_scale / n;
            self.cy += best.dy * self.h * self.o.padding * best_scale / n;
            self.w *= best_scale;
            self.h *= best_scale;
            self.patch(frame, self.cx, self.cy, 1.0, 0, 1);
            self.accumulate(self.o.rate);
        }
        return .{ .box = self.box(), .psr = best.psr, .ok = ok };
    }

    /// Moves the target to `b` (an outside detection: drift correction)
    /// and adapts the filter to the patch there at the learning rate, one
    /// FFT instead of the eight of a fresh `init`.
    pub fn correct(self: *Mosse, frame: Image(u8), b: Box) !void {
        if (frame.channels != 1) return error.UnsupportedChannels;
        self.cx = b.cx();
        self.cy = b.cy();
        self.w = @max(b.w, 4);
        self.h = @max(b.h, 4);
        self.patch(frame, self.cx, self.cy, 1.0, 0, 1);
        self.accumulate(self.o.rate);
    }

    /// Adds the current patch (in `buf`, spatial) to the running filter
    /// with weight `rate`.
    fn accumulate(self: *Mosse, rate: f32) void {
        self.plan.forward(self.buf);
        const keep = 1 - rate;
        const first = rate == 1.0;
        for (self.buf, self.g, self.a, self.b) |f, g, *a, *b| {
            const fc = f.conjugate();
            const na = g.mul(fc);
            const nb = f.re * f.re + f.im * f.im;
            if (first) {
                a.* = a.*.add(na);
                b.* += nb;
            } else {
                a.* = .{ .re = keep * a.re + rate * na.re, .im = keep * a.im + rate * na.im };
                b.* = keep * b.* + rate * nb;
            }
        }
        for (self.filt, self.a, self.b) |*h, a, b| {
            const inv = 1 / (b + 1e-4);
            h.* = .{ .re = a.re * inv, .im = a.im * inv };
        }
    }

    /// Samples the search region around (cx, cy) into `buf` (bilinear,
    /// optionally rotated by `ang` and scaled by `zoom`), then applies
    /// log(1+x), zero-mean / unit-variance normalisation and the window.
    fn patch(self: *Mosse, frame: Image(u8), cx: f32, cy: f32, region_scale: f32, ang: f32, zoom: f32) void {
        const n = self.o.size;
        const nf: f32 = @floatFromInt(n);
        const rw = self.w * self.o.padding * region_scale / zoom;
        const rh = self.h * self.o.padding * region_scale / zoom;
        const ca = @cos(ang);
        const sa = @sin(ang);
        const maxx: f32 = @floatFromInt(frame.width - 1);
        const maxy: f32 = @floatFromInt(frame.height - 1);
        var sum: f32 = 0;
        var sum2: f32 = 0;
        if (ang == 0 and n % L == 0) {
            // Axis-aligned (every update): sample positions separate into
            // per-column and per-row ones; log and sums are vector-wide.
            for (self.col_x0, self.col_fx, 0..) |*x0, *fx, x| {
                const u = (@as(f32, @floatFromInt(x)) + 0.5) / nf - 0.5;
                const sx = std.math.clamp(cx + u * rw, 0, maxx);
                x0.* = @intFromFloat(@floor(sx));
                fx.* = sx - @as(f32, @floatFromInt(x0.*));
            }
            var vs: VF = @splat(0);
            var vs2: VF = @splat(0);
            const one: VF = @splat(1);
            for (0..n) |y| {
                const v = (@as(f32, @floatFromInt(y)) + 0.5) / nf - 0.5;
                const sy = std.math.clamp(cy + v * rh, 0, maxy);
                const y0: usize = @intFromFloat(@floor(sy));
                const y1 = @min(y0 + 1, frame.height - 1);
                const fy: VF = @splat(sy - @as(f32, @floatFromInt(y0)));
                const r0 = frame.row(y0);
                const r1 = frame.row(y1);
                var x: usize = 0;
                while (x < n) : (x += L) {
                    var p: [4][L]f32 = undefined;
                    for (0..L) |k| {
                        const x0 = self.col_x0[x + k];
                        const x1 = @min(x0 + 1, frame.width - 1);
                        p[0][k] = @floatFromInt(r0[x0]);
                        p[1][k] = @floatFromInt(r0[x1]);
                        p[2][k] = @floatFromInt(r1[x0]);
                        p[3][k] = @floatFromInt(r1[x1]);
                    }
                    const fx: VF = self.col_fx[x..][0..L].*;
                    const p00: VF = p[0];
                    const p10: VF = p[1];
                    const p01: VF = p[2];
                    const p11: VF = p[3];
                    const val = logv(one + (p00 * (one - fx) * (one - fy) + p10 * fx * (one - fy) + p01 * (one - fx) * fy + p11 * fx * fy));
                    const arr: [L]f32 = val;
                    for (self.buf[y * n + x ..][0..L], arr) |*b, a| b.* = .{ .re = a, .im = 0 };
                    vs += val;
                    vs2 += val * val;
                }
            }
            sum = @reduce(.Add, vs);
            sum2 = @reduce(.Add, vs2);
        } else for (0..n) |y| for (0..n) |x| {
            const u = (@as(f32, @floatFromInt(x)) + 0.5) / nf - 0.5;
            const v = (@as(f32, @floatFromInt(y)) + 0.5) / nf - 0.5;
            const ox = u * rw;
            const oy = v * rh;
            const sx = std.math.clamp(cx + ca * ox - sa * oy, 0, maxx);
            const sy = std.math.clamp(cy + sa * ox + ca * oy, 0, maxy);
            const x0: usize = @intFromFloat(@floor(sx));
            const y0: usize = @intFromFloat(@floor(sy));
            const x1 = @min(x0 + 1, frame.width - 1);
            const y1 = @min(y0 + 1, frame.height - 1);
            const fx = sx - @as(f32, @floatFromInt(x0));
            const fy = sy - @as(f32, @floatFromInt(y0));
            const p00: f32 = @floatFromInt(frame.at(x0, y0, 0));
            const p10: f32 = @floatFromInt(frame.at(x1, y0, 0));
            const p01: f32 = @floatFromInt(frame.at(x0, y1, 0));
            const p11: f32 = @floatFromInt(frame.at(x1, y1, 0));
            const val = @log(1 + (p00 * (1 - fx) * (1 - fy) + p10 * fx * (1 - fy) + p01 * (1 - fx) * fy + p11 * fx * fy));
            self.buf[y * n + x] = .{ .re = val, .im = 0 };
            sum += val;
            sum2 += val * val;
        };
        const cnt: f32 = @floatFromInt(n * n);
        const mean = sum / cnt;
        const sd = @sqrt(@max(sum2 / cnt - mean * mean, 1e-6));
        const isd = 1 / sd;
        for (self.buf, self.window) |*b, wv| b.* = .{ .re = (b.re - mean) * isd * wv, .im = 0 };
    }

    const Peak = struct { dx: f32, dy: f32, value: f32, psr: f32 };

    /// Peak of the (spatial) response in `buf`: sub-pixel offset from the
    /// patch centre, value and peak-to-sidelobe ratio.
    fn peak(self: *Mosse) Peak {
        const n = self.o.size;
        var bi: usize = 0;
        var bv: f32 = -std.math.inf(f32);
        var s: f32 = 0;
        var s2: f32 = 0;
        for (self.buf, 0..) |v, i| {
            if (v.re > bv) {
                bv = v.re;
                bi = i;
            }
            s += v.re;
            s2 += v.re * v.re;
        }
        const px = bi % n;
        const py = bi / n;
        // Sidelobe: everything outside an 11x11 window around the peak
        // (the totals minus the window).
        var cnt: f32 = @floatFromInt(n * n);
        for (py -| 5..@min(n, py + 6)) |y| for (px -| 5..@min(n, px + 6)) |x| {
            const v = self.buf[y * n + x].re;
            s -= v;
            s2 -= v * v;
            cnt -= 1;
        };
        const mean = s / cnt;
        const sd = @sqrt(@max(s2 / cnt - mean * mean, 1e-12));
        // Parabolic refinement along each axis.
        const at = struct {
            fn f(buf: []const Complex, nn: usize, x: usize, y: usize) f32 {
                return buf[(y % nn) * nn + (x % nn)].re;
            }
        }.f;
        const l = at(self.buf, n, px + n - 1, py);
        const r = at(self.buf, n, px + 1, py);
        const u = at(self.buf, n, px, py + n - 1);
        const d = at(self.buf, n, px, py + 1);
        const subx = parabola(l, bv, r);
        const suby = parabola(u, bv, d);
        const half: f32 = @floatFromInt(n / 2);
        return .{
            .dx = @as(f32, @floatFromInt(px)) + subx - half,
            .dy = @as(f32, @floatFromInt(py)) + suby - half,
            .value = bv,
            .psr = (bv - mean) / sd,
        };
    }
};

const L = 8;
const VF = @Vector(L, f32);

/// Natural logarithm of positive normal floats, vector-wide (Cephes logf,
/// within a couple of ulp).
fn logv(x: VF) VF {
    const VU = @Vector(L, u32);
    const VI = @Vector(L, i32);
    const bits: VU = @bitCast(x);
    var e: VI = @as(VI, @bitCast(bits >> @splat(23))) - @as(VI, @splat(127));
    var m: VF = @bitCast((bits & @as(VU, @splat(0x7fffff))) | @as(VU, @splat(0x3f800000)));
    const big = m > @as(VF, @splat(std.math.sqrt2));
    m = @select(f32, big, m * @as(VF, @splat(0.5)), m);
    e += @select(i32, big, @as(VI, @splat(1)), @as(VI, @splat(0)));
    const f = m - @as(VF, @splat(1));
    const z = f * f;
    var y: VF = @splat(7.0376836292e-2);
    inline for (.{ -1.1514610310e-1, 1.1676998740e-1, -1.2420140846e-1, 1.4249322787e-1, -1.6668057665e-1, 2.0000714765e-1, -2.4999993993e-1, 3.3333331174e-1 }) |c|
        y = y * f + @as(VF, @splat(c));
    y = y * f * z;
    const ef: VF = @floatFromInt(e);
    y += ef * @as(VF, @splat(-2.12194440e-4));
    y -= z * @as(VF, @splat(0.5));
    return f + y + ef * @as(VF, @splat(0.693359375));
}

test "logv matches @log" {
    var x: VF = .{ 1, 1.5, 2, 3.7, 10, 100.25, 255, 256 };
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const got: [L]f32 = logv(x);
        const xs: [L]f32 = x;
        for (got, xs) |g, v| try std.testing.expectApproxEqRel(@log(v), g, 1e-6);
        x = x * @as(VF, @splat(1.013));
    }
}

fn parabola(l: f32, c: f32, r: f32) f32 {
    const den = l - 2 * c + r;
    if (@abs(den) < 1e-12) return 0;
    return std.math.clamp(0.5 * (l - r) / den, -0.5, 0.5);
}

test "mosse follows a moving textured patch" {
    const gpa = std.testing.allocator;
    const W = 320;
    const H = 240;
    var frame = try Image(u8).init(gpa, W, H, 1);
    defer frame.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    // Static textured background and a distinctive 30x30 object.
    var bg: [W * H]u8 = undefined;
    for (&bg) |*v| v.* = 90 + rnd.uintLessThan(u8, 30);
    var obj: [30 * 30]u8 = undefined;
    for (&obj, 0..) |*v, i| v.* = if ((i / 30 / 5 + (i % 30) / 5) % 2 == 0) 230 else 20;
    const render = struct {
        fn f(img: Image(u8), b: []const u8, o: []const u8, ox: usize, oy: usize) void {
            @memcpy(img.slice(), b);
            for (0..30) |y| for (0..30) |x| img.set(ox + x, oy + y, 0, o[y * 30 + x]);
        }
    }.f;
    var ox: usize = 60;
    var oy: usize = 50;
    render(frame, &bg, &obj, ox, oy);
    var t = try Mosse.init(gpa, frame, .{ .x = @floatFromInt(ox), .y = @floatFromInt(oy), .w = 30, .h = 30 }, .{});
    defer t.deinit();
    for (0..60) |_| {
        ox += 3;
        oy += 2;
        render(frame, &bg, &obj, ox, oy);
        const r = try t.update(frame);
        try std.testing.expect(r.ok);
        try std.testing.expect(@abs(r.box.cx() - (@as(f32, @floatFromInt(ox)) + 15)) < 3);
        try std.testing.expect(@abs(r.box.cy() - (@as(f32, @floatFromInt(oy)) + 15)) < 3);
    }
    // Remove the object: the tracker must report it lost.
    @memcpy(frame.slice(), &bg);
    var lost = false;
    for (0..3) |_| {
        const r = try t.update(frame);
        if (!r.ok) lost = true;
    }
    try std.testing.expect(lost);
}
