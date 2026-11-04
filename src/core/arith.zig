//! Per-element arithmetic, logic, conversion and reductions (OpenCV core).
//! Integer results saturate; float-to-int conversions round half to even.

const std = @import("std");
const image = @import("image.zig");
const pool = @import("pool.zig");
const sat = @import("saturate.zig");
const simd = @import("simd.zig");
const Image = image.Image;
const Exec = pool.Exec;

pub const Error = image.Error;

pub const BinOp = enum { add, sub, absdiff, mul, div, min, max, bit_and, bit_or, bit_xor };

/// dst = a (op) b, element-wise, saturating.
pub fn binary(exec: Exec, op: BinOp, a: anytype, b: @TypeOf(a), dst: @TypeOf(a)) Error!void {
    const T = @TypeOf(a).Elem;
    if (!a.sameShape(b) or !a.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { a: Image(T), b: Image(T), d: Image(T) };
    switch (op) {
        inline else => |o| {
            if (@typeInfo(T) == .float and (o == .bit_and or o == .bit_or or o == .bit_xor)) return error.UnsupportedChannels;
            exec.parallelFor(a.height, rowsGrain(a), Ctx{ .a = a, .b = b, .d = dst }, struct {
                fn run(c: Ctx, y0: usize, y1: usize) void {
                    for (y0..y1) |y| binRow(T, o, c.a.row(y), c.b.row(y), c.d.row(y));
                }
            }.run);
        },
    }
}

fn rowsGrain(img: anytype) usize {
    return @max(1, (1 << 16) / @max(1, img.rowLen()));
}

inline fn binRow(comptime T: type, comptime op: BinOp, a: []const T, b: []const T, d: []T) void {
    const N = comptime simd.lanes(T);
    const V = @Vector(N, T);
    var i: usize = 0;
    const is_int = @typeInfo(T) == .int;
    if (op != .mul and op != .div) {
        while (i + N <= d.len) : (i += N) {
            const x: V = a[i..][0..N].*;
            const y: V = b[i..][0..N].*;
            d[i..][0..N].* = switch (op) {
                .add => if (is_int) x +| y else x + y,
                .sub => if (is_int) x -| y else x - y,
                .absdiff => if (is_int) @max(x, y) -% @min(x, y) else @abs(x - y),
                .min => @min(x, y),
                .max => @max(x, y),
                .bit_and => if (is_int) x & y else unreachable,
                .bit_or => if (is_int) x | y else unreachable,
                .bit_xor => if (is_int) x ^ y else unreachable,
                else => unreachable,
            };
        }
    }
    while (i < d.len) : (i += 1) {
        const x = a[i];
        const y = b[i];
        d[i] = switch (op) {
            .add => if (is_int) x +| y else x + y,
            .sub => if (is_int) x -| y else x - y,
            .absdiff => if (is_int) @max(x, y) -% @min(x, y) else @abs(x - y),
            .min => @min(x, y),
            .max => @max(x, y),
            .bit_and => if (is_int) x & y else unreachable,
            .bit_or => if (is_int) x | y else unreachable,
            .bit_xor => if (is_int) x ^ y else unreachable,
            .mul => if (is_int) sat.cast(T, @as(i64, x) * @as(i64, y)) else x * y,
            .div => if (is_int) (if (y == 0) 0 else sat.cast(T, @as(f64, @floatFromInt(x)) / @as(f64, @floatFromInt(y)))) else x / y,
        };
    }
}

/// dst = ~src (integer images).
pub fn bitwiseNot(exec: Exec, src: anytype, dst: @TypeOf(src)) Error!void {
    const T = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { s: Image(T), d: Image(T) };
    exec.parallelFor(src.height, rowsGrain(src), Ctx{ .s = src, .d = dst }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| for (c.s.row(y), c.d.row(y)) |v, *o| {
                o.* = ~v;
            };
        }
    }.run);
}

/// dst = saturate(src * alpha + beta) with any source/destination depth
/// (convertTo).
pub fn convertTo(exec: Exec, src: anytype, dst: anytype, alpha: f64, beta: f64) Error!void {
    const S = @TypeOf(src).Elem;
    const D = @TypeOf(dst).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { s: Image(S), d: Image(D), a: f64, b: f64 };
    exec.parallelFor(src.height, rowsGrain(src), Ctx{ .s = src, .d = dst, .a = alpha, .b = beta }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            const identity = c.a == 1 and c.b == 0;
            for (y0..y1) |y| {
                const s = c.s.row(y);
                const d = c.d.row(y);
                if (identity) {
                    if (S == D) {
                        @memcpy(d, s);
                        continue;
                    }
                    var i: usize = 0;
                    if (comptime vecConvertible(S) and vecConvertible(D)) {
                        const n = 32;
                        while (i + n <= d.len) : (i += n) {
                            const v: @Vector(n, S) = s[i..][0..n].*;
                            d[i..][0..n].* = identityVec(S, D, n, v);
                        }
                    }
                    for (s[i..], d[i..]) |v, *o| o.* = sat.cast(D, v);
                } else if (@typeInfo(D) == .float or (@sizeOf(S) <= 2 and @typeInfo(S) == .int)) {
                    // f32 math matches OpenCV's 8/16-bit and float paths.
                    const a: f32 = @floatCast(c.a);
                    const b: f32 = @floatCast(c.b);
                    var i: usize = 0;
                    if (comptime vecConvertible(S) and vecConvertible(D)) {
                        const n = 32;
                        const va: @Vector(n, f32) = @splat(a);
                        const vb: @Vector(n, f32) = @splat(b);
                        while (i + n <= d.len) : (i += n) {
                            const v: @Vector(n, S) = s[i..][0..n].*;
                            d[i..][0..n].* = fromF32(D, n, simd.toF32(n, v) * va + vb);
                        }
                    }
                    for (s[i..], d[i..]) |v, *o| o.* = sat.cast(D, toF32(S, v) * a + b);
                } else {
                    for (s, d) |v, *o| o.* = sat.cast(D, toF64(S, v) * c.a + c.b);
                }
            }
        }
    }.run);
}

/// Types the f32 vector paths handle: integers up to 16 bits and f32.
fn vecConvertible(comptime T: type) bool {
    return T == f32 or (@typeInfo(T) == .int and @bitSizeOf(T) <= 16);
}

/// saturate_cast<D> of f32 lanes (D as in vecConvertible).
inline fn fromF32(comptime D: type, comptime n: comptime_int, v: @Vector(n, f32)) @Vector(n, D) {
    return if (D == f32) v else simd.satF32(D, n, v);
}

inline fn identityVec(comptime S: type, comptime D: type, comptime n: comptime_int, v: @Vector(n, S)) @Vector(n, D) {
    if (S == f32 or D == f32) return fromF32(D, n, simd.toF32(n, v));
    const w: @Vector(n, i32) = v;
    const lo: @Vector(n, i32) = @splat(std.math.minInt(D));
    const hi: @Vector(n, i32) = @splat(std.math.maxInt(D));
    return simd.narrow(D, @min(@max(w, lo), hi));
}

/// dst = saturate_u8(|src * alpha + beta|)
pub fn convertScaleAbs(exec: Exec, src: anytype, dst: Image(u8), alpha: f64, beta: f64) Error!void {
    const S = @TypeOf(src).Elem;
    if (!src.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { s: Image(S), d: Image(u8), a: f32, b: f32 };
    exec.parallelFor(src.height, rowsGrain(src), Ctx{ .s = src, .d = dst, .a = @floatCast(alpha), .b = @floatCast(beta) }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                const s = c.s.row(y);
                const d = c.d.row(y);
                var i: usize = 0;
                if (comptime vecConvertible(S)) {
                    const n = 32;
                    const va: @Vector(n, f32) = @splat(c.a);
                    const vb: @Vector(n, f32) = @splat(c.b);
                    while (i + n <= d.len) : (i += n) {
                        const v: @Vector(n, S) = s[i..][0..n].*;
                        d[i..][0..n].* = simd.satF32(u8, n, @abs(simd.toF32(n, v) * va + vb));
                    }
                }
                for (s[i..], d[i..]) |v, *o| o.* = sat.cast(u8, @abs(toF32(S, v) * c.a + c.b));
            }
        }
    }.run);
}

/// dst = saturate(a*alpha + b*beta + gamma), evaluated in f32 as
/// fma(a, alpha, fma(b, beta, gamma)) like OpenCV's SIMD path.
pub fn addWeighted(exec: Exec, a: anytype, alpha: f64, b: @TypeOf(a), beta: f64, gamma: f64, dst: @TypeOf(a)) Error!void {
    const T = @TypeOf(a).Elem;
    if (!a.sameShape(b) or !a.sameShape(dst)) return error.SizeMismatch;
    const Ctx = struct { a: Image(T), b: Image(T), d: Image(T), al: f32, be: f32, ga: f32 };
    exec.parallelFor(a.height, rowsGrain(a), Ctx{ .a = a, .b = b, .d = dst, .al = @floatCast(alpha), .be = @floatCast(beta), .ga = @floatCast(gamma) }, struct {
        fn run(c: Ctx, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                const ra = c.a.row(y);
                const rb = c.b.row(y);
                const d = c.d.row(y);
                var i: usize = 0;
                if (comptime vecConvertible(T)) {
                    const n = 64;
                    const F = @Vector(n, f32);
                    const al: F = @splat(c.al);
                    const be: F = @splat(c.be);
                    const ga: F = @splat(c.ga);
                    while (i + n <= d.len) : (i += n) {
                        const x = simd.toF32(n, @as(@Vector(n, T), ra[i..][0..n].*));
                        const z = simd.toF32(n, @as(@Vector(n, T), rb[i..][0..n].*));
                        d[i..][0..n].* = fromF32(T, n, @mulAdd(F, x, al, @mulAdd(F, z, be, ga)));
                    }
                }
                for (ra[i..], rb[i..], d[i..]) |x, z, *o| {
                    o.* = sat.cast(T, @mulAdd(f32, toF32(T, x), c.al, @mulAdd(f32, toF32(T, z), c.be, c.ga)));
                }
            }
        }
    }.run);
}

inline fn toF32(comptime T: type, v: T) f32 {
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

inline fn toF64(comptime T: type, v: T) f64 {
    return if (@typeInfo(T) == .float) @floatCast(v) else @floatFromInt(v);
}

/// Per-channel sums (up to 4 channels).
pub fn sum(src: anytype) [4]f64 {
    const T = @TypeOf(src).Elem;
    var out: [4]f64 = @splat(0);
    const cn: usize = src.channels;
    for (0..src.height) |y| {
        const r = src.row(y);
        if (@typeInfo(T) == .int and @sizeOf(T) <= 2 and src.rowLen() < (1 << 15)) {
            // Exact integer row sums, then one float add per channel.
            var acc: [4]u64 = @splat(0);
            var iacc: [4]i64 = @splat(0);
            for (r, 0..) |v, i| {
                if (@typeInfo(T).int.signedness == .unsigned) acc[(i % cn) % 4] += v else iacc[(i % cn) % 4] += v;
            }
            for (0..@min(cn, 4)) |ch| out[ch] += @as(f64, @floatFromInt(acc[ch])) + @as(f64, @floatFromInt(iacc[ch]));
        } else {
            for (r, 0..) |v, i| if (i % cn < 4) {
                out[i % cn] += toF64(T, v);
            };
        }
    }
    return out;
}

pub fn mean(src: anytype) [4]f64 {
    var s = sum(src);
    const n: f64 = @floatFromInt(src.pixelCount());
    for (&s) |*v| v.* /= n;
    return s;
}

pub const MinMax = struct { min: f64, max: f64, min_x: u32, min_y: u32, max_x: u32, max_y: u32 };

/// minMaxLoc for single-channel images.
pub fn minMaxLoc(src: anytype) Error!MinMax {
    const T = @TypeOf(src).Elem;
    if (src.channels != 1) return error.UnsupportedChannels;
    var r: MinMax = .{ .min = toF64(T, src.at(0, 0, 0)), .max = toF64(T, src.at(0, 0, 0)), .min_x = 0, .min_y = 0, .max_x = 0, .max_y = 0 };
    var mn = src.at(0, 0, 0);
    var mx = mn;
    for (0..src.height) |y| {
        const row = src.row(y);
        // Fast path: skip rows that cannot change the result.
        var rmn = row[0];
        var rmx = row[0];
        for (row) |v| {
            rmn = @min(rmn, v);
            rmx = @max(rmx, v);
        }
        if (rmn < mn) {
            mn = rmn;
            r.min_y = @intCast(y);
            r.min_x = @intCast(std.mem.indexOfScalar(T, row, rmn).?);
        }
        if (rmx > mx) {
            mx = rmx;
            r.max_y = @intCast(y);
            r.max_x = @intCast(std.mem.indexOfScalar(T, row, rmx).?);
        }
    }
    r.min = toF64(T, mn);
    r.max = toF64(T, mx);
    return r;
}

pub fn countNonZero(src: anytype) Error!usize {
    if (src.channels != 1) return error.UnsupportedChannels;
    var n: usize = 0;
    for (0..src.height) |y| for (src.row(y)) |v| {
        n += @intFromBool(v != 0);
    };
    return n;
}

/// Splits an interleaved image into single-channel planes.
pub fn split(src: anytype, planes: []const @TypeOf(src)) Error!void {
    if (planes.len != src.channels) return error.UnsupportedChannels;
    for (planes) |p| if (!p.sameSize(src) or p.channels != 1) return error.SizeMismatch;
    const cn: usize = src.channels;
    for (0..src.height) |y| {
        const s = src.row(y);
        for (planes, 0..) |p, ch| {
            const d = p.row(y);
            for (d, 0..) |*o, x| o.* = s[x * cn + ch];
        }
    }
}

/// Interleaves single-channel planes into `dst`.
pub fn merge(planes: anytype, dst: @TypeOf(planes[0])) Error!void {
    if (planes.len != dst.channels) return error.UnsupportedChannels;
    for (planes) |p| if (!p.sameSize(dst) or p.channels != 1) return error.SizeMismatch;
    const cn: usize = dst.channels;
    for (0..dst.height) |y| {
        const d = dst.row(y);
        for (planes, 0..) |p, ch| {
            const s = p.row(y);
            for (s, 0..) |v, x| d[x * cn + ch] = v;
        }
    }
}

test "binary ops saturate" {
    const gpa = std.testing.allocator;
    var a = try Image(u8).initFill(gpa, 40, 2, 1, 200);
    defer a.deinit(gpa);
    var b = try Image(u8).initFill(gpa, 40, 2, 1, 100);
    defer b.deinit(gpa);
    var d = try Image(u8).initLike(gpa, a);
    defer d.deinit(gpa);
    try binary(.serial, .add, a, b, d);
    try std.testing.expectEqual(@as(u8, 255), d.at(39, 1, 0));
    try binary(.serial, .sub, b, a, d);
    try std.testing.expectEqual(@as(u8, 0), d.at(5, 0, 0));
    try binary(.serial, .absdiff, b, a, d);
    try std.testing.expectEqual(@as(u8, 100), d.at(5, 0, 0));
    try addWeighted(.serial, a, 0.5, b, 0.5, 0, d);
    try std.testing.expectEqual(@as(u8, 150), d.at(5, 0, 0));
    const m = mean(a);
    try std.testing.expectEqual(@as(f64, 200), m[0]);
}

test {
    _ = simd;
}
