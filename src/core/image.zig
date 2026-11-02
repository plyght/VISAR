//! The image container. An `Image(T)` is a 2-D grid of pixels with
//! `channels` interleaved samples of type `T` per pixel (HWC, the OpenCV
//! layout). Rows may be padded: `stride` is the distance between the starts
//! of two rows, in elements. Views (`roi`, `view`) share the parent's memory
//! and never own it; only images made by `init`/`clone` own their storage and
//! must be released with `deinit` using the same allocator.

const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Error = error{
    /// Width, height or channel count of zero, or an overflowing size.
    InvalidSize,
    /// The operation needs inputs of the same width, height and channels.
    SizeMismatch,
    /// The operation does not support this channel count.
    UnsupportedChannels,
    /// The region lies (partly) outside the image.
    OutOfBounds,
};

/// Maximum channels per pixel. OpenCV allows up to 512 (CV_CN_MAX); VISAR
/// keeps the same limit.
pub const max_channels = 512;

pub fn Image(comptime T: type) type {
    return struct {
        const Self = @This();
        pub const Elem = T;

        width: u32,
        height: u32,
        channels: u16,
        /// Elements between the start of row y and row y+1 (>= width*channels).
        stride: usize,
        /// Pointer to the first sample of pixel (0, 0).
        data: [*]T,
        /// The allocation this image owns, or null for a view.
        owned: ?[]T = null,

        /// A new, uninitialised, contiguous image.
        pub fn init(gpa: Allocator, width: u32, height: u32, channels: u16) (Allocator.Error || Error)!Self {
            const n = try elemCount(width, height, channels);
            const buf = try gpa.alignedAlloc(T, .@"64", n);
            return .{ .width = width, .height = height, .channels = channels, .stride = @as(usize, width) * channels, .data = buf.ptr, .owned = buf };
        }

        /// A new contiguous image with every sample set to `value`.
        pub fn initFill(gpa: Allocator, width: u32, height: u32, channels: u16, value: T) (Allocator.Error || Error)!Self {
            const img = try init(gpa, width, height, channels);
            @memset(img.owned.?, value);
            return img;
        }

        /// A new image with the same size and channel count as `other`.
        pub fn initLike(gpa: Allocator, other: anytype) (Allocator.Error || Error)!Self {
            return init(gpa, other.width, other.height, other.channels);
        }

        pub fn deinit(self: *Self, gpa: Allocator) void {
            if (self.owned) |buf| gpa.free(@as([]align(64) T, @alignCast(buf)));
            self.* = undefined;
        }

        /// Wraps caller memory (not owned) holding a contiguous image.
        pub fn fromSlice(data: []T, width: u32, height: u32, channels: u16) Error!Self {
            return fromSliceStrided(data, width, height, channels, @as(usize, width) * channels);
        }

        /// Wraps caller memory (not owned) with rows `stride` elements apart.
        pub fn fromSliceStrided(data: []T, width: u32, height: u32, channels: u16, stride: usize) Error!Self {
            const row_len = (try elemCount(width, 1, channels));
            if (stride < row_len) return error.InvalidSize;
            const need = (@as(usize, height) - 1) * stride + row_len;
            if (data.len < need) return error.OutOfBounds;
            return .{ .width = width, .height = height, .channels = channels, .stride = stride, .data = data.ptr };
        }

        /// Samples per row (width * channels), excluding padding.
        pub inline fn rowLen(self: Self) usize {
            return @as(usize, self.width) * self.channels;
        }

        pub inline fn row(self: Self, y: usize) []T {
            return self.data[y * self.stride ..][0..self.rowLen()];
        }

        pub inline fn pixel(self: Self, x: usize, y: usize) []T {
            return self.data[y * self.stride + x * self.channels ..][0..self.channels];
        }

        pub inline fn at(self: Self, x: usize, y: usize, c: usize) T {
            return self.data[y * self.stride + x * self.channels + c];
        }

        pub inline fn set(self: Self, x: usize, y: usize, c: usize, v: T) void {
            self.data[y * self.stride + x * self.channels + c] = v;
        }

        /// True when rows are packed back to back with no padding.
        pub inline fn isContiguous(self: Self) bool {
            return self.stride == self.rowLen() or self.height <= 1;
        }

        /// All samples as one slice. Only valid for contiguous images.
        pub fn slice(self: Self) []T {
            std.debug.assert(self.isContiguous());
            return self.data[0 .. self.rowLen() * self.height];
        }

        pub inline fn pixelCount(self: Self) usize {
            return @as(usize, self.width) * self.height;
        }

        /// A borrowed view of the whole image.
        pub fn view(self: Self) Self {
            var v = self;
            v.owned = null;
            return v;
        }

        /// A borrowed view of the rectangle (x, y, w, h).
        pub fn roi(self: Self, x: u32, y: u32, w: u32, h: u32) Error!Self {
            if (w == 0 or h == 0) return error.InvalidSize;
            if (@as(u64, x) + w > self.width or @as(u64, y) + h > self.height) return error.OutOfBounds;
            return .{
                .width = w,
                .height = h,
                .channels = self.channels,
                .stride = self.stride,
                .data = self.data + @as(usize, y) * self.stride + @as(usize, x) * self.channels,
            };
        }

        /// An owned, contiguous deep copy.
        pub fn clone(self: Self, gpa: Allocator) (Allocator.Error || Error)!Self {
            var out = try init(gpa, self.width, self.height, self.channels);
            out.copyFrom(self);
            return out;
        }

        /// Copies pixels from `src`, which must have the same shape.
        pub fn copyFrom(self: Self, src: Self) void {
            std.debug.assert(self.sameShape(src));
            if (self.isContiguous() and src.isContiguous()) {
                @memcpy(self.slice(), src.slice());
                return;
            }
            for (0..self.height) |y| @memcpy(self.row(y), src.row(y));
        }

        pub fn fill(self: Self, v: T) void {
            if (self.isContiguous()) return @memset(self.slice(), v);
            for (0..self.height) |y| @memset(self.row(y), v);
        }

        pub fn sameShape(self: Self, other: anytype) bool {
            return self.width == other.width and self.height == other.height and self.channels == other.channels;
        }

        pub fn sameSize(self: Self, other: anytype) bool {
            return self.width == other.width and self.height == other.height;
        }

        /// Raw bytes of a contiguous image.
        pub fn bytes(self: Self) []u8 {
            return std.mem.sliceAsBytes(self.slice());
        }
    };
}

fn elemCount(width: u32, height: u32, channels: u16) Error!usize {
    if (width == 0 or height == 0 or channels == 0 or channels > max_channels) return error.InvalidSize;
    const a = std.math.mul(usize, width, height) catch return error.InvalidSize;
    return std.math.mul(usize, a, channels) catch return error.InvalidSize;
}

/// Sample depth, numbered like OpenCV's CV_8U..CV_16F so the C ABI and the
/// cv2 shim can map 1:1.
pub const Depth = enum(u8) {
    u8 = 0,
    i8 = 1,
    u16 = 2,
    i16 = 3,
    i32 = 4,
    f32 = 5,
    f64 = 6,
    f16 = 7,

    pub fn Type(comptime d: Depth) type {
        return switch (d) {
            .u8 => u8,
            .i8 => i8,
            .u16 => u16,
            .i16 => i16,
            .i32 => i32,
            .f32 => f32,
            .f64 => f64,
            .f16 => f16,
        };
    }

    pub fn of(comptime T: type) Depth {
        return switch (T) {
            u8 => .u8,
            i8 => .i8,
            u16 => .u16,
            i16 => .i16,
            i32 => .i32,
            f32 => .f32,
            f64 => .f64,
            f16 => .f16,
            else => @compileError("unsupported sample type " ++ @typeName(T)),
        };
    }

    pub fn size(d: Depth) usize {
        return switch (d) {
            inline else => |t| @sizeOf(t.Type()),
        };
    }
};

/// An image whose sample type is known only at run time (decoders, the C
/// ABI, bindings). Switch on it with `inline else` to reach typed kernels.
pub const AnyImage = union(Depth) {
    u8: Image(u8),
    i8: Image(i8),
    u16: Image(u16),
    i16: Image(i16),
    i32: Image(i32),
    f32: Image(f32),
    f64: Image(f64),
    f16: Image(f16),

    pub fn init(gpa: Allocator, d: Depth, w: u32, h: u32, c: u16) (Allocator.Error || Error)!AnyImage {
        return switch (d) {
            inline else => |t| @unionInit(AnyImage, @tagName(t), try Image(t.Type()).init(gpa, w, h, c)),
        };
    }

    pub fn deinit(self: *AnyImage, gpa: Allocator) void {
        switch (self.*) {
            inline else => |*img| img.deinit(gpa),
        }
    }

    pub fn depth(self: AnyImage) Depth {
        return self;
    }

    pub fn width(self: AnyImage) u32 {
        return switch (self) {
            inline else => |img| img.width,
        };
    }

    pub fn height(self: AnyImage) u32 {
        return switch (self) {
            inline else => |img| img.height,
        };
    }

    pub fn channels(self: AnyImage) u16 {
        return switch (self) {
            inline else => |img| img.channels,
        };
    }

    /// The typed image, or null when the depth differs.
    pub fn as(self: AnyImage, comptime T: type) ?Image(T) {
        return switch (self) {
            inline else => |img| if (@TypeOf(img).Elem == T) img else null,
        };
    }
};

test "image basics" {
    const gpa = std.testing.allocator;
    var img = try Image(u8).initFill(gpa, 4, 3, 3, 7);
    defer img.deinit(gpa);
    try std.testing.expect(img.isContiguous());
    img.set(1, 2, 0, 9);
    try std.testing.expectEqual(@as(u8, 9), img.at(1, 2, 0));
    const r = try img.roi(1, 1, 2, 2);
    try std.testing.expect(!r.isContiguous());
    try std.testing.expectEqual(@as(u8, 9), r.at(0, 1, 0));
    var c = try r.clone(gpa);
    defer c.deinit(gpa);
    try std.testing.expect(c.isContiguous());
    try std.testing.expectEqual(@as(u8, 9), c.at(0, 1, 0));
    try std.testing.expectError(error.OutOfBounds, img.roi(3, 0, 2, 1));
    try std.testing.expectError(error.InvalidSize, Image(u8).init(gpa, 0, 1, 1));
}

test "any image" {
    const gpa = std.testing.allocator;
    var a = try AnyImage.init(gpa, .f32, 2, 2, 1);
    defer a.deinit(gpa);
    try std.testing.expectEqual(Depth.f32, a.depth());
    try std.testing.expect(a.as(f32) != null);
    try std.testing.expect(a.as(u8) == null);
}
