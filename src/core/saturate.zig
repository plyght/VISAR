//! Saturating conversions with OpenCV's semantics: float to integer rounds
//! half to even (cvRound) and clamps to the destination range.

const std = @import("std");

/// Round half to even, as cvRound/lrint do in the default FP mode. Exact for
/// |x| < 2^22, which covers every integer destination VISAR saturates to.
pub inline fn rint(x: anytype) @TypeOf(x) {
    const T = @TypeOf(x);
    const S = switch (@typeInfo(T)) {
        .vector => |v| v.child,
        else => T,
    };
    const magic: S = if (S == f64) 6755399441055744.0 else 12582912.0;
    const m: T = if (@typeInfo(T) == .vector) @splat(magic) else magic;
    return (x + m) - m;
}

/// Converts `x` to `D`, clamping and (for float sources) rounding to nearest even.
pub inline fn cast(comptime D: type, x: anytype) D {
    const S = @TypeOf(x);
    if (S == D) return x;
    switch (@typeInfo(D)) {
        .float => return switch (@typeInfo(S)) {
            .float => @floatCast(x),
            else => @floatFromInt(x),
        },
        .int => {
            const lo = std.math.minInt(D);
            const hi = std.math.maxInt(D);
            switch (@typeInfo(S)) {
                .float => {
                    if (std.math.isNan(x)) return 0;
                    const r = if (S == f64) rint(x) else rint(@as(f64, x));
                    if (r <= @as(f64, lo)) return lo;
                    if (r >= @as(f64, hi)) return hi;
                    return @intFromFloat(r);
                },
                .int, .comptime_int => {
                    if (x < lo) return lo;
                    if (x > hi) return hi;
                    return @intCast(x);
                },
                else => @compileError("cast from " ++ @typeName(S)),
            }
        },
        else => @compileError("cast to " ++ @typeName(D)),
    }
}

/// Clamp a value of any integer type into u8.
pub inline fn u8c(x: anytype) u8 {
    return @intCast(std.math.clamp(x, 0, 255));
}

test "saturate" {
    try std.testing.expectEqual(@as(u8, 255), cast(u8, @as(i32, 300)));
    try std.testing.expectEqual(@as(u8, 0), cast(u8, @as(i32, -5)));
    try std.testing.expectEqual(@as(u8, 2), cast(u8, @as(f32, 2.5)));
    try std.testing.expectEqual(@as(u8, 4), cast(u8, @as(f32, 3.5)));
    try std.testing.expectEqual(@as(i16, -32768), cast(i16, @as(f32, -1e9)));
    try std.testing.expectEqual(@as(f32, 3), cast(f32, @as(u8, 3)));
}
