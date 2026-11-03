//! Fast Fourier transforms for power-of-two sizes: in-place iterative
//! radix-2 on interleaved complex f32, and 2-D transforms of square or
//! rectangular power-of-two grids. Used by correlation trackers and
//! frequency-domain filtering; small and allocation-free.

const std = @import("std");

pub const Complex = std.math.Complex(f32);

/// Precomputed twiddles and bit-reversal for one length.
pub const Plan = struct {
    n: usize,
    log2n: u6,
    /// exp(-2*pi*i*k/n) for k < n/2.
    tw: []Complex,
    rev: []u32,

    pub fn init(gpa: std.mem.Allocator, n: usize) !Plan {
        if (n == 0 or !std.math.isPowerOfTwo(n)) return error.NotPowerOfTwo;
        const log2n: u6 = @intCast(std.math.log2_int(usize, n));
        const tw = try gpa.alloc(Complex, @max(1, n / 2));
        errdefer gpa.free(tw);
        for (tw, 0..) |*t, k| {
            const a = -2.0 * std.math.pi * @as(f64, @floatFromInt(k)) / @as(f64, @floatFromInt(n));
            t.* = .{ .re = @floatCast(@cos(a)), .im = @floatCast(@sin(a)) };
        }
        const rev = try gpa.alloc(u32, n);
        for (rev, 0..) |*r, i| r.* = if (log2n == 0) 0 else @intCast(@bitReverse(@as(u32, @intCast(i))) >> @intCast(32 - @as(u32, log2n)));
        return .{ .n = n, .log2n = log2n, .tw = tw, .rev = rev };
    }

    pub fn deinit(self: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(self.tw);
        gpa.free(self.rev);
    }

    /// In-place transform of `x` (length n, elements `stride` apart).
    /// `inverse` uses conjugate twiddles and scales by 1/n.
    pub fn run(self: Plan, x: [*]Complex, stride: usize, inverse: bool) void {
        const n = self.n;
        for (0..n) |i| {
            const j = self.rev[i];
            if (j > i) std.mem.swap(Complex, &x[i * stride], &x[j * stride]);
        }
        var len: usize = 2;
        while (len <= n) : (len <<= 1) {
            const half = len / 2;
            const step = n / len;
            var start: usize = 0;
            while (start < n) : (start += len) {
                for (0..half) |k| {
                    var w = self.tw[k * step];
                    if (inverse) w.im = -w.im;
                    const a = &x[(start + k) * stride];
                    const b = &x[(start + k + half) * stride];
                    const t = b.*.mul(w);
                    b.* = a.*.sub(t);
                    a.* = a.*.add(t);
                }
            }
        }
        if (inverse) {
            const s = 1.0 / @as(f32, @floatFromInt(n));
            for (0..n) |i| {
                x[i * stride].re *= s;
                x[i * stride].im *= s;
            }
        }
    }
};

/// 2-D transform of a `w` x `h` grid stored row-major.
pub const Plan2D = struct {
    rows: Plan,
    cols: Plan,

    pub fn init(gpa: std.mem.Allocator, w: usize, h: usize) !Plan2D {
        var rows = try Plan.init(gpa, w);
        errdefer rows.deinit(gpa);
        return .{ .rows = rows, .cols = try Plan.init(gpa, h) };
    }

    pub fn deinit(self: *Plan2D, gpa: std.mem.Allocator) void {
        self.rows.deinit(gpa);
        self.cols.deinit(gpa);
    }

    pub fn forward(self: Plan2D, x: []Complex) void {
        self.apply(x, false);
    }

    pub fn inverse(self: Plan2D, x: []Complex) void {
        self.apply(x, true);
    }

    fn apply(self: Plan2D, x: []Complex, inv: bool) void {
        const w = self.rows.n;
        const h = self.cols.n;
        std.debug.assert(x.len == w * h);
        for (0..h) |y| self.rows.run(x.ptr + y * w, 1, inv);
        for (0..w) |c| self.cols.run(x.ptr + c, w, inv);
    }
};

test "fft matches a direct DFT and inverts" {
    const gpa = std.testing.allocator;
    var plan = try Plan.init(gpa, 16);
    defer plan.deinit(gpa);
    var x: [16]Complex = undefined;
    for (&x, 0..) |*v, i| v.* = .{ .re = @floatFromInt(i % 5), .im = @floatFromInt((i * 3) % 7) };
    const orig = x;
    plan.run(&x, 1, false);
    for (0..16) |k| {
        var s: Complex = .{ .re = 0, .im = 0 };
        for (orig, 0..) |v, t| {
            const a = -2.0 * std.math.pi * @as(f32, @floatFromInt(k * t)) / 16.0;
            s = s.add(v.mul(.{ .re = @cos(a), .im = @sin(a) }));
        }
        try std.testing.expectApproxEqAbs(s.re, x[k].re, 1e-3);
        try std.testing.expectApproxEqAbs(s.im, x[k].im, 1e-3);
    }
    plan.run(&x, 1, true);
    for (orig, x) |a, b| {
        try std.testing.expectApproxEqAbs(a.re, b.re, 1e-4);
        try std.testing.expectApproxEqAbs(a.im, b.im, 1e-4);
    }
}

test "2-D round trip" {
    const gpa = std.testing.allocator;
    var p = try Plan2D.init(gpa, 8, 4);
    defer p.deinit(gpa);
    var x: [32]Complex = undefined;
    for (&x, 0..) |*v, i| v.* = .{ .re = @floatFromInt(i), .im = 0 };
    const orig = x;
    p.forward(&x);
    try std.testing.expectApproxEqAbs(@as(f32, 496), x[0].re, 1e-3); // sum 0..31
    p.inverse(&x);
    for (orig, x) |a, b| try std.testing.expectApproxEqAbs(a.re, b.re, 1e-3);
}
