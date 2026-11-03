//! Frames in flight between a capture source, a processing thread and a
//! display, plus the conversion that feeds detection.
//!
//! `FrameRing` owns a fixed set of BGRA buffers and passes them between
//! threads without locks:
//!
//!   producer (camera callback)    acquire -> fill -> publishCaptured
//!   worker (processing)           takeCaptured -> process -> publishProcessed
//!   display (main thread)         takeProcessed -> show -> release
//!
//! Each mailbox holds only the newest frame: publishing over an unclaimed
//! frame releases the stale one (counted as dropped), so a slow consumer
//! always sees the latest picture and never builds up latency. Every
//! operation is wait-free, so the camera callback (a GCD thread on macOS)
//! never blocks.

const std = @import("std");
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const none: u8 = 0xff;

pub const Meta = struct {
    /// Capture sequence number (0, 1, 2, ...).
    seq: u64 = 0,
    /// Capture time on the awake clock, nanoseconds (0 if unknown).
    t_ns: i96 = 0,
    /// Presentation time stamp from the source, seconds (0 if unknown).
    pts: f64 = 0,
};

pub const FrameRing = struct {
    pub const capacity = 8;

    gpa: Allocator,
    width: u32,
    height: u32,
    /// Bytes per pixel (4: BGRA).
    channels: u16,
    storage: []align(64) u8,
    meta: [capacity]Meta = @splat(.{}),
    /// 0 = free, 1 = owned by a producer, worker or display.
    busy: [capacity]std.atomic.Value(u8) = @splat(.init(0)),
    captured: std.atomic.Value(u8) = .init(none),
    processed: std.atomic.Value(u8) = .init(none),
    /// Bumped on every publishCaptured; a futex for the worker to wait on.
    captured_epoch: std.atomic.Value(u32) = .init(0),
    /// Frames published but replaced before anyone took them.
    dropped_captured: std.atomic.Value(usize) = .init(0),
    dropped_processed: std.atomic.Value(usize) = .init(0),
    /// Frames the producer could not store because every buffer was busy.
    starved: std.atomic.Value(usize) = .init(0),

    pub fn init(gpa: Allocator, width: u32, height: u32, channels: u16) !FrameRing {
        if (width == 0 or height == 0 or channels == 0) return error.InvalidSize;
        const per = std.math.mul(usize, @as(usize, width) * height, channels) catch return error.InvalidSize;
        const storage = try gpa.alignedAlloc(u8, .@"64", per * capacity);
        return .{ .gpa = gpa, .width = width, .height = height, .channels = channels, .storage = storage };
    }

    pub fn deinit(self: *FrameRing) void {
        self.gpa.free(self.storage);
        self.* = undefined;
    }

    pub fn frameBytes(self: *const FrameRing) usize {
        return @as(usize, self.width) * self.height * self.channels;
    }

    /// The image in buffer `i` (contiguous).
    pub fn image(self: *const FrameRing, i: u8) Image(u8) {
        const n = self.frameBytes();
        return Image(u8).fromSlice(self.storage[@as(usize, i) * n ..][0..n], self.width, self.height, self.channels) catch unreachable;
    }

    /// A free buffer, now owned by the caller, or null if all are in use.
    pub fn acquire(self: *FrameRing) ?u8 {
        for (&self.busy, 0..) |*b, i| {
            if (b.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) return @intCast(i);
        }
        _ = self.starved.fetchAdd(1, .monotonic);
        return null;
    }

    pub fn release(self: *FrameRing, i: u8) void {
        std.debug.assert(i < capacity);
        self.busy[i].store(0, .release);
    }

    /// Hands a filled buffer to the worker, replacing (and releasing) a
    /// frame it has not taken yet.
    pub fn publishCaptured(self: *FrameRing, io: ?Io, i: u8) void {
        const old = self.captured.swap(i, .acq_rel);
        if (old != none) {
            self.release(old);
            _ = self.dropped_captured.fetchAdd(1, .monotonic);
        }
        _ = self.captured_epoch.fetchAdd(1, .release);
        if (io) |o| o.futexWake(u32, &self.captured_epoch.raw, 1);
    }

    pub fn takeCaptured(self: *FrameRing) ?u8 {
        const i = self.captured.swap(none, .acq_rel);
        return if (i == none) null else i;
    }

    /// Waits up to `timeout_ns` for a captured frame and takes it.
    pub fn waitCaptured(self: *FrameRing, io: Io, timeout_ns: i96) ?u8 {
        const epoch = self.captured_epoch.load(.acquire);
        if (self.takeCaptured()) |i| return i;
        io.futexWaitTimeout(u32, &self.captured_epoch.raw, epoch, .{ .duration = .{ .raw = .fromNanoseconds(timeout_ns), .clock = .awake } }) catch {};
        return self.takeCaptured();
    }

    /// Hands a processed buffer to the display, replacing (and releasing)
    /// one it has not shown yet.
    pub fn publishProcessed(self: *FrameRing, i: u8) void {
        const old = self.processed.swap(i, .acq_rel);
        if (old != none) {
            self.release(old);
            _ = self.dropped_processed.fetchAdd(1, .monotonic);
        }
    }

    pub fn takeProcessed(self: *FrameRing) ?u8 {
        const i = self.processed.swap(none, .acq_rel);
        return if (i == none) null else i;
    }

    /// Releases whatever is still waiting in the mailboxes.
    pub fn drain(self: *FrameRing) void {
        if (self.takeCaptured()) |i| self.release(i);
        if (self.takeProcessed()) |i| self.release(i);
    }

    /// Copies `src` (any stride, at least the ring's channel count) into
    /// buffer `i`, cropping or padding with black if the sizes differ.
    pub fn store(self: *FrameRing, i: u8, src: Image(u8)) void {
        const dst = self.image(i);
        const w = @min(src.width, dst.width);
        const h = @min(src.height, dst.height);
        const row_bytes = @as(usize, w) * dst.channels;
        const same = src.channels == dst.channels;
        if (same and src.isContiguous() and src.width == dst.width and src.height == dst.height) {
            @memcpy(dst.slice(), src.slice());
            return;
        }
        if (w < dst.width or h < dst.height) dst.fill(0);
        for (0..h) |y| {
            if (same) {
                @memcpy(dst.row(y)[0..row_bytes], src.row(y)[0..row_bytes]);
            } else {
                const s = src.row(y);
                const d = dst.row(y);
                for (0..w) |x| @memcpy(d[x * dst.channels ..][0..dst.channels], s[x * src.channels ..][0..dst.channels]);
            }
        }
    }
};

/// Grayscale (BT.601 weights, as cvtColor) of a BGRA or BGR image, shrunk
/// by an integer factor `k` with box (area) averaging, in one pass: no
/// full-size gray image is ever written (k is 1 to 16). `dst` must be (src.width / k) x
/// (src.height / k), 1 channel; leftover columns and rows are ignored.
pub fn grayDownsample(exec: Exec, src: Image(u8), dst: Image(u8), k: u32) error{SizeMismatch}!void {
    if (k == 0 or k > 16 or dst.channels != 1 or (src.channels != 3 and src.channels != 4)) return error.SizeMismatch;
    if (dst.width != src.width / k or dst.height != src.height / k) return error.SizeMismatch;
    const c: KernelCtx = .{ .src = src, .dst = dst, .k = k };
    if (src.channels == 4) exec.parallelFor(dst.height, 8, c, Kernel(4).run) else exec.parallelFor(dst.height, 8, c, Kernel(3).run);
}

const KernelCtx = struct { src: Image(u8), dst: Image(u8), k: u32 };

/// BT.601 luma weights in 14-bit fixed point (OpenCV's), for B, G and R.
const wb = 1868;
const wg = 9617;
const wr = 4899;

fn Kernel(comptime cn: usize) type {
    return struct {
        // Per output row: the k source rows are summed per channel into
        // `acc` (u16; at most 16 * 255 per sample), then each output pixel
        // sums k columns of it, weights B, G and R, and divides by k * k.
        // The luma of the block mean equals the mean of the lumas, so this
        // matches gray-then-area up to rounding.
        const chunk_px = 2048;

        fn run(c: KernelCtx, y0: usize, y1: usize) void {
            var acc_buf: [chunk_px * cn]u16 align(64) = undefined;
            const k: usize = c.k;
            const dw: usize = c.dst.width;
            const per_chunk = (chunk_px / k) * k;
            for (y0..y1) |oy| {
                const out_row = c.dst.row(oy);
                var x0: usize = 0; // source pixels
                while (x0 < dw * k) {
                    const npx = @min(dw * k - x0, per_chunk);
                    const acc = acc_buf[0 .. npx * cn];
                    sumRows(c.src, oy * k, k, x0 * cn, acc);
                    const out = out_row[x0 / k ..][0 .. npx / k];
                    if (k == 2) pairs(acc, out) else blocks(acc, out, k);
                    x0 += npx;
                }
            }
        }

        fn sumRows(src: Image(u8), y: usize, k: usize, off: usize, acc: []u16) void {
            const N = 32;
            const first = src.row(y)[off..][0..acc.len];
            var i: usize = 0;
            while (i + N <= acc.len) : (i += N) {
                var v: @Vector(N, u16) = @as(@Vector(N, u8), first[i..][0..N].*);
                for (1..k) |dy| v += @as(@Vector(N, u8), src.row(y + dy)[off + i ..][0..N].*);
                acc[i..][0..N].* = v;
            }
            while (i < acc.len) : (i += 1) {
                var v: u16 = first[i];
                for (1..k) |dy| v += src.row(y + dy)[off + i];
                acc[i] = v;
            }
        }

        /// k = 2: eight output pixels at a time; the block sum of 4 is
        /// weighted (sum <= 4 * 255 * 2^14 = 2^24) and shifted by 16.
        fn pairs(acc: []const u16, out: []u8) void {
            const N = 8;
            const even_b = comptime idx(N, 0, 0);
            const odd_b = comptime idx(N, 1, 0);
            const even_g = comptime idx(N, 0, 1);
            const odd_g = comptime idx(N, 1, 1);
            const even_r = comptime idx(N, 0, 2);
            const odd_r = comptime idx(N, 1, 2);
            const V = @Vector(N, u32);
            var x: usize = 0;
            while (x + N <= out.len) : (x += N) {
                const a: @Vector(2 * N * cn, u16) = acc[x * 2 * cn ..][0 .. 2 * N * cn].*;
                const b: V = @as(@Vector(N, u16), @shuffle(u16, a, undefined, even_b)) + @as(@Vector(N, u16), @shuffle(u16, a, undefined, odd_b));
                const g: V = @as(@Vector(N, u16), @shuffle(u16, a, undefined, even_g)) + @as(@Vector(N, u16), @shuffle(u16, a, undefined, odd_g));
                const r: V = @as(@Vector(N, u16), @shuffle(u16, a, undefined, even_r)) + @as(@Vector(N, u16), @shuffle(u16, a, undefined, odd_r));
                const l = (b * @as(V, @splat(wb)) + g * @as(V, @splat(wg)) + r * @as(V, @splat(wr)) + @as(V, @splat(1 << 15))) >> @splat(16);
                out[x..][0..N].* = core.simd.narrow(u8, l);
            }
            while (x < out.len) : (x += 1) {
                const p = acc[x * 2 * cn ..];
                const l = (@as(u32, p[0]) + p[cn]) * wb + (@as(u32, p[1]) + p[cn + 1]) * wg + (@as(u32, p[2]) + p[cn + 2]) * wr;
                out[x] = @intCast((l + (1 << 15)) >> 16);
            }
        }

        fn blocks(acc: []const u16, out: []u8, k: usize) void {
            // l <= 256 * 255 * 2^14 < 2^30; dividing in f32 is exact enough
            // (the result is at most 255, the error below 1e-4).
            const inv: f32 = 1.0 / @as(f32, @floatFromInt(k * k * (1 << 14)));
            for (out, 0..) |*o, x| {
                var b: u32 = 0;
                var g: u32 = 0;
                var r: u32 = 0;
                for (acc[x * k * cn ..][0 .. k * cn], 0..) |v, j| switch (j % cn) {
                    0 => b += v,
                    1 => g += v,
                    2 => r += v,
                    else => {},
                };
                const l: f32 = @floatFromInt(b * wb + g * wg + r * wr);
                o.* = @intFromFloat(@min(255, l * inv + 0.5));
            }
        }

        fn idx(comptime n: usize, comptime odd: usize, comptime ch: usize) [n]i32 {
            var r: [n]i32 = undefined;
            for (0..n) |i| r[i] = @intCast((2 * i + odd) * cn + ch);
            return r;
        }
    };
}

test "ring passes the newest frame and drops stale ones" {
    const gpa = std.testing.allocator;
    var ring = try FrameRing.init(gpa, 4, 2, 4);
    defer ring.deinit();
    const a = ring.acquire().?;
    ring.publishCaptured(null, a);
    const b = ring.acquire().?;
    try std.testing.expect(a != b);
    ring.publishCaptured(null, b); // a was never taken: dropped and freed
    try std.testing.expectEqual(@as(usize, 1), ring.dropped_captured.load(.monotonic));
    try std.testing.expectEqual(b, ring.takeCaptured().?);
    try std.testing.expectEqual(@as(?u8, null), ring.takeCaptured());
    ring.publishProcessed(b);
    try std.testing.expectEqual(b, ring.takeProcessed().?);
    ring.release(b);
    // Every buffer is free again.
    var got: [FrameRing.capacity]u8 = undefined;
    for (&got) |*g| g.* = ring.acquire().?;
    try std.testing.expectEqual(@as(?u8, null), ring.acquire());
    for (got) |g| ring.release(g);
}

test "ring store crops, pads and converts channel counts" {
    const gpa = std.testing.allocator;
    var ring = try FrameRing.init(gpa, 3, 2, 4);
    defer ring.deinit();
    var src_buf: [2 * 8 * 4]u8 = undefined;
    for (&src_buf, 0..) |*v, i| v.* = @intCast(i);
    // A 2x2 BGRA source with a stride of 8 pixels: the third column is black.
    const src = try Image(u8).fromSliceStrided(&src_buf, 2, 2, 4, 32);
    ring.store(0, src);
    const img = ring.image(0);
    try std.testing.expectEqualSlices(u8, src_buf[0..8], img.row(0)[0..8]);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 0, 0 }, img.row(0)[8..12]);
    try std.testing.expectEqualSlices(u8, src_buf[32..40], img.row(1)[0..8]);
}

test "grayDownsample matches cvtColor + area resize" {
    const gpa = std.testing.allocator;
    const imgproc = @import("../imgproc/imgproc.zig");
    inline for (.{ 4, 3 }) |cn| for ([_]u32{ 1, 2, 3 }) |k| {
        const W = 70 * k + 1;
        const H = 9 * k;
        var src = try Image(u8).init(gpa, W, H, cn);
        defer src.deinit(gpa);
        var prng = std.Random.DefaultPrng.init(k);
        prng.random().bytes(src.slice());
        var dst = try Image(u8).init(gpa, W / k, H / k, 1);
        defer dst.deinit(gpa);
        try grayDownsample(.serial, src, dst, k);
        // Reference: per-pixel luma at full size, then the k x k mean.
        for (0..dst.height) |oy| for (0..dst.width) |ox| {
            var sum: f64 = 0;
            for (0..k) |dy| for (0..k) |dx| {
                const p = src.pixel(ox * k + dx, oy * k + dy);
                sum += (@as(f64, @floatFromInt(p[0])) * 1868 + @as(f64, @floatFromInt(p[1])) * 9617 + @as(f64, @floatFromInt(p[2])) * 4899) / 16384.0;
            };
            const want = sum / @as(f64, @floatFromInt(k * k));
            const got: f64 = @floatFromInt(dst.at(ox, oy, 0));
            try std.testing.expect(@abs(got - want) <= 0.51);
        };
        if (k == 1) {
            // At k = 1 it is cvtColor's gray (within rounding).
            var ref = try Image(u8).init(gpa, W, H, 1);
            defer ref.deinit(gpa);
            try imgproc.cvtColor(.serial, src, ref, if (cn == 4) .bgra2gray else .bgr2gray);
            for (ref.slice(), dst.slice()) |r, d| try std.testing.expect(@abs(@as(i32, r) - d) <= 1);
        }
    };
}
