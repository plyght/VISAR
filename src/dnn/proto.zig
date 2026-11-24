//! A minimal protobuf wire-format reader: enough to walk the messages of an
//! ONNX model. Zero-copy: length-delimited fields are slices of the input.
//! No schema is involved; callers switch on field numbers.

const std = @import("std");

pub const Error = error{InvalidData};

pub const Wire = enum(u3) { varint = 0, fixed64 = 1, len = 2, start_group = 3, end_group = 4, fixed32 = 5, _ };

pub const Field = struct {
    num: u32,
    wire: Wire,
    /// The varint, or the fixed32/fixed64 bits.
    int: u64 = 0,
    /// Payload of a length-delimited field.
    bytes: []const u8 = &.{},

    pub fn asI64(f: Field) i64 {
        return @bitCast(f.int);
    }

    pub fn asF32(f: Field) f32 {
        return @bitCast(@as(u32, @truncate(f.int)));
    }

    pub fn asString(f: Field) Error![]const u8 {
        if (f.wire != .len) return error.InvalidData;
        return f.bytes;
    }
};

pub const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(buf: []const u8) Reader {
        return .{ .buf = buf };
    }

    pub fn varint(r: *Reader) Error!u64 {
        var v: u64 = 0;
        var shift: u32 = 0;
        while (true) {
            if (r.pos >= r.buf.len or shift >= 64) return error.InvalidData;
            const c = r.buf[r.pos];
            r.pos += 1;
            v |= @as(u64, c & 0x7f) << @intCast(shift);
            if (c < 0x80) return v;
            shift += 7;
        }
    }

    fn fixed(r: *Reader, comptime n: usize) Error!u64 {
        if (r.buf.len - r.pos < n) return error.InvalidData;
        const v = std.mem.readInt(@Int(.unsigned, n * 8), r.buf[r.pos..][0..n], .little);
        r.pos += n;
        return v;
    }

    /// The next field, or null at the end of the message.
    pub fn next(r: *Reader) Error!?Field {
        if (r.pos >= r.buf.len) return null;
        const key = try r.varint();
        const num: u32 = @intCast(key >> 3);
        if (num == 0 or key >> 3 > std.math.maxInt(u29)) return error.InvalidData;
        const wire: Wire = @fromBackingInt(@intCast(@as(u3, @truncate(key))));
        var f: Field = .{ .num = num, .wire = wire };
        switch (wire) {
            .varint => f.int = try r.varint(),
            .fixed64 => f.int = try r.fixed(8),
            .fixed32 => f.int = try r.fixed(4),
            .len => {
                const n = try r.varint();
                if (n > r.buf.len - r.pos) return error.InvalidData;
                f.bytes = r.buf[r.pos..][0..@intCast(n)];
                r.pos += @intCast(n);
            },
            // Groups are deprecated and never used by ONNX.
            else => return error.InvalidData,
        }
        return f;
    }
};

/// Iterates a repeated scalar field that may be packed (one length-delimited
/// field holding many values) or unpacked (one field per value).
pub fn ScalarIter(comptime kind: enum { varint, fixed32, fixed64 }) type {
    return struct {
        r: Reader,
        single: ?u64,

        const Self = @This();

        pub fn init(f: Field) Self {
            if (f.wire == .len) return .{ .r = .init(f.bytes), .single = null };
            return .{ .r = .init(&.{}), .single = f.int };
        }

        pub fn next(it: *Self) Error!?u64 {
            if (it.single) |v| {
                it.single = null;
                return v;
            }
            if (it.r.pos >= it.r.buf.len) return null;
            return switch (kind) {
                .varint => try it.r.varint(),
                .fixed32 => try it.r.fixed(4),
                .fixed64 => try it.r.fixed(8),
            };
        }
    };
}

test "reader decodes varints, fixed and length-delimited fields" {
    // field 1 varint 300, field 2 bytes "hi", field 3 fixed32 1.5f, field 4 packed varints [1, 150]
    const msg = [_]u8{ 0x08, 0xac, 0x02, 0x12, 0x02, 'h', 'i', 0x1d, 0x00, 0x00, 0xc0, 0x3f, 0x22, 0x03, 0x01, 0x96, 0x01 };
    var r = Reader.init(&msg);
    const a = (try r.next()).?;
    try std.testing.expectEqual(@as(u64, 300), a.int);
    const b = (try r.next()).?;
    try std.testing.expectEqualStrings("hi", b.bytes);
    const c = (try r.next()).?;
    try std.testing.expectEqual(@as(f32, 1.5), c.asF32());
    const d = (try r.next()).?;
    var it = ScalarIter(.varint).init(d);
    try std.testing.expectEqual(@as(?u64, 1), try it.next());
    try std.testing.expectEqual(@as(?u64, 150), try it.next());
    try std.testing.expectEqual(@as(?u64, null), try it.next());
    try std.testing.expectEqual(@as(?Field, null), try r.next());
    // Truncated input is rejected.
    var bad = Reader.init(&[_]u8{ 0x12, 0x05, 'h' });
    try std.testing.expectError(error.InvalidData, bad.next());
}
