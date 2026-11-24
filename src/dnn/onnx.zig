//! ONNX model reader: decodes the parts of ModelProto that inference needs
//! (graph, nodes, attributes, initializers, value infos) straight from the
//! protobuf bytes. Names and raw tensor bytes are slices of the input, so the
//! model bytes must outlive the result; everything else is allocated in the
//! caller's arena. Works from memory, so a model can be `@embedFile`d.

const std = @import("std");
const proto = @import("proto.zig");
const Allocator = std.mem.Allocator;

pub const Error = proto.Error || Allocator.Error || error{Unsupported};

/// TensorProto.DataType (numbering from onnx.proto).
pub const DataType = enum(i32) {
    undefined = 0,
    float = 1,
    uint8 = 2,
    int8 = 3,
    uint16 = 4,
    int16 = 5,
    int32 = 6,
    int64 = 7,
    string = 8,
    bool = 9,
    float16 = 10,
    double = 11,
    uint32 = 12,
    uint64 = 13,
    bfloat16 = 16,
    _,

    pub fn size(t: DataType) usize {
        return switch (t) {
            .uint8, .int8, .bool => 1,
            .uint16, .int16, .float16, .bfloat16 => 2,
            .float, .int32, .uint32 => 4,
            .int64, .double, .uint64 => 8,
            else => 0,
        };
    }
};

pub const Tensor = struct {
    name: []const u8 = "",
    dims: []const i64 = &.{},
    data_type: DataType = .undefined,
    /// raw_data (little-endian, possibly unaligned), when present.
    raw: ?[]const u8 = null,
    float_data: []const f32 = &.{},
    /// int32_data also carries int8/16/uint8/16/bool/float16 values.
    int32_data: []const i32 = &.{},
    int64_data: []const i64 = &.{},
    double_data: []const f64 = &.{},
    /// Set when the payload lives in an external file (unsupported).
    external: bool = false,

    pub fn elemCount(t: Tensor) usize {
        var n: usize = 1;
        for (t.dims) |d| n *= @intCast(@max(d, 0));
        return n;
    }

    /// The values converted to f32 (any numeric type).
    pub fn toF32(t: Tensor, arena: Allocator) Error![]f32 {
        return convert(f32, t, arena);
    }

    /// The values converted to i64 (any numeric type).
    pub fn toI64(t: Tensor, arena: Allocator) Error![]i64 {
        return convert(i64, t, arena);
    }

    fn convert(comptime T: type, t: Tensor, arena: Allocator) Error![]T {
        if (t.external) return error.Unsupported;
        const n = t.elemCount();
        const out = try arena.alloc(T, n);
        if (t.raw) |raw| {
            const es = t.data_type.size();
            if (es == 0) return error.Unsupported;
            if (raw.len != n * es) return error.InvalidData;
            for (out, 0..) |*o, i| o.* = rawElem(T, t.data_type, raw[i * es ..][0..es]);
            return out;
        }
        switch (t.data_type) {
            .float => {
                if (t.float_data.len != n) return error.InvalidData;
                for (out, t.float_data) |*o, v| o.* = cast(T, v);
            },
            .int64 => {
                if (t.int64_data.len != n) return error.InvalidData;
                for (out, t.int64_data) |*o, v| o.* = cast(T, v);
            },
            .double => {
                if (t.double_data.len != n) return error.InvalidData;
                for (out, t.double_data) |*o, v| o.* = cast(T, v);
            },
            .int32, .int16, .int8, .uint8, .uint16, .bool => {
                if (t.int32_data.len != n) return error.InvalidData;
                for (out, t.int32_data) |*o, v| o.* = cast(T, v);
            },
            .float16 => {
                if (t.int32_data.len != n) return error.InvalidData;
                for (out, t.int32_data) |*o, v| o.* = cast(T, @as(f16, @bitCast(@as(u16, @truncate(@as(u32, @bitCast(v)))))));
            },
            else => return error.Unsupported,
        }
        return out;
    }
};

fn cast(comptime T: type, v: anytype) T {
    const V = @TypeOf(v);
    if (T == f32) {
        return switch (@typeInfo(V)) {
            .float => @floatCast(v),
            .int => @floatFromInt(v),
            else => unreachable,
        };
    }
    return switch (@typeInfo(V)) {
        .float => if (std.math.isFinite(v)) @intFromFloat(std.math.clamp(@trunc(v), -9.2e18, 9.2e18)) else 0,
        .int => std.math.cast(T, v) orelse if (v < 0) std.math.minInt(T) else std.math.maxInt(T),
        else => unreachable,
    };
}

fn rawElem(comptime T: type, dt: DataType, b: []const u8) T {
    return switch (dt) {
        .float => cast(T, @as(f32, @bitCast(std.mem.readInt(u32, b[0..4], .little)))),
        .double => cast(T, @as(f64, @bitCast(std.mem.readInt(u64, b[0..8], .little)))),
        .float16 => cast(T, @as(f16, @bitCast(std.mem.readInt(u16, b[0..2], .little)))),
        .bfloat16 => cast(T, @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[0..2], .little)) << 16))),
        .int64 => cast(T, std.mem.readInt(i64, b[0..8], .little)),
        .uint64 => cast(T, std.mem.readInt(u64, b[0..8], .little)),
        .int32 => cast(T, std.mem.readInt(i32, b[0..4], .little)),
        .uint32 => cast(T, std.mem.readInt(u32, b[0..4], .little)),
        .int16 => cast(T, std.mem.readInt(i16, b[0..2], .little)),
        .uint16 => cast(T, std.mem.readInt(u16, b[0..2], .little)),
        .int8 => cast(T, @as(i8, @bitCast(b[0]))),
        .uint8, .bool => cast(T, b[0]),
        else => cast(T, @as(i64, 0)),
    };
}

pub const AttrType = enum(i32) {
    undefined = 0,
    float = 1,
    int = 2,
    string = 3,
    tensor = 4,
    graph = 5,
    floats = 6,
    ints = 7,
    strings = 8,
    tensors = 9,
    graphs = 10,
    _,
};

pub const Attribute = struct {
    name: []const u8 = "",
    type: AttrType = .undefined,
    f: f32 = 0,
    i: i64 = 0,
    s: []const u8 = "",
    t: ?Tensor = null,
    floats: []const f32 = &.{},
    ints: []const i64 = &.{},
    strings: []const []const u8 = &.{},
};

pub const Node = struct {
    name: []const u8 = "",
    op_type: []const u8 = "",
    domain: []const u8 = "",
    inputs: []const []const u8 = &.{},
    outputs: []const []const u8 = &.{},
    attrs: []const Attribute = &.{},

    pub fn attr(n: Node, name: []const u8) ?Attribute {
        for (n.attrs) |a| if (std.mem.eql(u8, a.name, name)) return a;
        return null;
    }

    pub fn int(n: Node, name: []const u8, default: i64) i64 {
        return if (n.attr(name)) |a| a.i else default;
    }

    pub fn float(n: Node, name: []const u8, default: f32) f32 {
        return if (n.attr(name)) |a| a.f else default;
    }

    pub fn string(n: Node, name: []const u8, default: []const u8) []const u8 {
        return if (n.attr(name)) |a| a.s else default;
    }

    pub fn ints(n: Node, name: []const u8) ?[]const i64 {
        return if (n.attr(name)) |a| a.ints else null;
    }
};

pub const Dim = union(enum) { value: i64, param: []const u8, unknown };

pub const ValueInfo = struct {
    name: []const u8 = "",
    elem_type: DataType = .undefined,
    /// Null when the shape is not given at all.
    shape: ?[]const Dim = null,
};

pub const Graph = struct {
    name: []const u8 = "",
    nodes: []const Node = &.{},
    initializers: []const Tensor = &.{},
    inputs: []const ValueInfo = &.{},
    outputs: []const ValueInfo = &.{},
    value_info: []const ValueInfo = &.{},
};

pub const Model = struct {
    ir_version: i64 = 0,
    /// Version of the default ("" / "ai.onnx") operator set.
    opset: i64 = 0,
    producer: []const u8 = "",
    graph: Graph = .{},
};

/// Parses a serialized ModelProto. `bytes` must outlive the result.
pub fn parse(arena: Allocator, bytes: []const u8) Error!Model {
    var m: Model = .{};
    var have_graph = false;
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => m.ir_version = f.asI64(),
        2 => m.producer = try f.asString(),
        7 => {
            m.graph = try parseGraph(arena, try f.asString());
            have_graph = true;
        },
        8 => {
            var domain: []const u8 = "";
            var version: i64 = 0;
            var o = proto.Reader.init(try f.asString());
            while (try o.next()) |g| switch (g.num) {
                1 => domain = try g.asString(),
                2 => version = g.asI64(),
                else => {},
            };
            if (domain.len == 0 or std.mem.eql(u8, domain, "ai.onnx")) m.opset = version;
        },
        else => {},
    };
    if (!have_graph) return error.InvalidData;
    return m;
}

fn List(comptime T: type) type {
    return std.ArrayList(T);
}

fn parseGraph(arena: Allocator, bytes: []const u8) Error!Graph {
    var nodes: List(Node) = .empty;
    var inits: List(Tensor) = .empty;
    var inputs: List(ValueInfo) = .empty;
    var outputs: List(ValueInfo) = .empty;
    var infos: List(ValueInfo) = .empty;
    var g: Graph = .{};
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => try nodes.append(arena, try parseNode(arena, try f.asString())),
        2 => g.name = try f.asString(),
        5 => try inits.append(arena, try parseTensor(arena, try f.asString())),
        11 => try inputs.append(arena, try parseValueInfo(arena, try f.asString())),
        12 => try outputs.append(arena, try parseValueInfo(arena, try f.asString())),
        13 => try infos.append(arena, try parseValueInfo(arena, try f.asString())),
        15 => return error.Unsupported, // sparse initializers
        else => {},
    };
    g.nodes = nodes.items;
    g.initializers = inits.items;
    g.inputs = inputs.items;
    g.outputs = outputs.items;
    g.value_info = infos.items;
    return g;
}

fn parseNode(arena: Allocator, bytes: []const u8) Error!Node {
    var ins: List([]const u8) = .empty;
    var outs: List([]const u8) = .empty;
    var attrs: List(Attribute) = .empty;
    var n: Node = .{};
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => try ins.append(arena, try f.asString()),
        2 => try outs.append(arena, try f.asString()),
        3 => n.name = try f.asString(),
        4 => n.op_type = try f.asString(),
        5 => try attrs.append(arena, try parseAttribute(arena, try f.asString())),
        7 => n.domain = try f.asString(),
        else => {},
    };
    n.inputs = ins.items;
    n.outputs = outs.items;
    n.attrs = attrs.items;
    return n;
}

fn parseAttribute(arena: Allocator, bytes: []const u8) Error!Attribute {
    var a: Attribute = .{};
    var floats: List(f32) = .empty;
    var ints: List(i64) = .empty;
    var strings: List([]const u8) = .empty;
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => a.name = try f.asString(),
        2 => a.f = f.asF32(),
        3 => a.i = f.asI64(),
        4 => a.s = try f.asString(),
        5 => a.t = try parseTensor(arena, try f.asString()),
        7 => {
            var it = proto.ScalarIter(.fixed32).init(f);
            while (try it.next()) |v| try floats.append(arena, @bitCast(@as(u32, @truncate(v))));
        },
        8 => {
            var it = proto.ScalarIter(.varint).init(f);
            while (try it.next()) |v| try ints.append(arena, @bitCast(v));
        },
        9 => try strings.append(arena, try f.asString()),
        20 => a.type = @fromBackingInt(@intCast(@as(i32, @truncate(f.asI64())))),
        else => {},
    };
    a.floats = floats.items;
    a.ints = ints.items;
    a.strings = strings.items;
    return a;
}

fn parseTensor(arena: Allocator, bytes: []const u8) Error!Tensor {
    var t: Tensor = .{};
    var dims: List(i64) = .empty;
    var floats: List(f32) = .empty;
    var i32s: List(i32) = .empty;
    var i64s: List(i64) = .empty;
    var f64s: List(f64) = .empty;
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => {
            var it = proto.ScalarIter(.varint).init(f);
            while (try it.next()) |v| try dims.append(arena, @bitCast(v));
        },
        2 => t.data_type = @fromBackingInt(@intCast(@as(i32, @truncate(f.asI64())))),
        4 => {
            var it = proto.ScalarIter(.fixed32).init(f);
            while (try it.next()) |v| try floats.append(arena, @bitCast(@as(u32, @truncate(v))));
        },
        5 => {
            var it = proto.ScalarIter(.varint).init(f);
            while (try it.next()) |v| try i32s.append(arena, @truncate(@as(i64, @bitCast(v))));
        },
        7 => {
            var it = proto.ScalarIter(.varint).init(f);
            while (try it.next()) |v| try i64s.append(arena, @bitCast(v));
        },
        8 => t.name = try f.asString(),
        9 => t.raw = try f.asString(),
        10 => {
            var it = proto.ScalarIter(.fixed64).init(f);
            while (try it.next()) |v| try f64s.append(arena, @bitCast(v));
        },
        13 => t.external = true,
        14 => if (f.int == 1) {
            t.external = true;
        },
        else => {},
    };
    t.dims = dims.items;
    t.float_data = floats.items;
    t.int32_data = i32s.items;
    t.int64_data = i64s.items;
    t.double_data = f64s.items;
    return t;
}

fn parseValueInfo(arena: Allocator, bytes: []const u8) Error!ValueInfo {
    var v: ValueInfo = .{};
    var r = proto.Reader.init(bytes);
    while (try r.next()) |f| switch (f.num) {
        1 => v.name = try f.asString(),
        2 => {
            // TypeProto.tensor_type (1) -> elem_type (1), shape (2) -> dim (1)
            var tp = proto.Reader.init(try f.asString());
            while (try tp.next()) |g| if (g.num == 1) {
                var tt = proto.Reader.init(try g.asString());
                while (try tt.next()) |h| switch (h.num) {
                    1 => v.elem_type = @fromBackingInt(@intCast(@as(i32, @truncate(h.asI64())))),
                    2 => {
                        var dims: List(Dim) = .empty;
                        var sh = proto.Reader.init(try h.asString());
                        while (try sh.next()) |d| if (d.num == 1) {
                            var dim: Dim = .unknown;
                            var dr = proto.Reader.init(try d.asString());
                            while (try dr.next()) |e| switch (e.num) {
                                1 => dim = .{ .value = e.asI64() },
                                2 => dim = .{ .param = try e.asString() },
                                else => {},
                            };
                            try dims.append(arena, dim);
                        };
                        v.shape = dims.items;
                    },
                    else => {},
                };
            };
        },
        else => {},
    };
    return v;
}

// ------------------------------------------------------------------ tests

/// Writes protobuf messages for tests (and for tools that build small models).
pub const Writer = struct {
    buf: std.ArrayList(u8) = .empty,
    gpa: Allocator,

    pub fn varint(w: *Writer, v: u64) Allocator.Error!void {
        var x = v;
        while (x >= 0x80) : (x >>= 7) try w.buf.append(w.gpa, @as(u8, @truncate(x)) | 0x80);
        try w.buf.append(w.gpa, @truncate(x));
    }

    pub fn key(w: *Writer, num: u32, wire: proto.Wire) Allocator.Error!void {
        try w.varint((@as(u64, num) << 3) | @as(u64, @backingInt(wire)));
    }

    pub fn int(w: *Writer, num: u32, v: i64) Allocator.Error!void {
        try w.key(num, .varint);
        try w.varint(@bitCast(v));
    }

    pub fn float(w: *Writer, num: u32, v: f32) Allocator.Error!void {
        try w.key(num, .fixed32);
        try w.buf.appendSlice(w.gpa, &std.mem.toBytes(std.mem.nativeToLittle(u32, @bitCast(v))));
    }

    pub fn bytes(w: *Writer, num: u32, b: []const u8) Allocator.Error!void {
        try w.key(num, .len);
        try w.varint(b.len);
        try w.buf.appendSlice(w.gpa, b);
    }

    pub fn deinit(w: *Writer) void {
        w.buf.deinit(w.gpa);
    }
};

test "parse a hand-built model" {
    const gpa = std.testing.allocator;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // TensorProto: dims [2], float, raw_data [1.5, -2]
    var t: Writer = .{ .gpa = arena };
    try t.int(1, 2);
    try t.int(2, 1);
    try t.bytes(8, "w");
    const raw = std.mem.toBytes([2]f32{ 1.5, -2 });
    try t.bytes(9, &raw);
    // AttributeProto: name "alpha", f 0.25, type FLOAT; and ints [1, 2]
    var a1: Writer = .{ .gpa = arena };
    try a1.bytes(1, "alpha");
    try a1.float(2, 0.25);
    try a1.int(20, 1);
    var a2: Writer = .{ .gpa = arena };
    try a2.bytes(1, "pads");
    try a2.int(8, 1);
    try a2.int(8, -2);
    var n: Writer = .{ .gpa = arena };
    try n.bytes(1, "x");
    try n.bytes(1, "w");
    try n.bytes(2, "y");
    try n.bytes(4, "Mul");
    try n.bytes(5, a1.buf.items);
    try n.bytes(5, a2.buf.items);
    var g: Writer = .{ .gpa = arena };
    try g.bytes(1, n.buf.items);
    try g.bytes(5, t.buf.items);
    var m: Writer = .{ .gpa = arena };
    try m.int(1, 7);
    var op: Writer = .{ .gpa = arena };
    try op.int(2, 13);
    try m.bytes(8, op.buf.items);
    try m.bytes(7, g.buf.items);

    const model = try parse(arena, m.buf.items);
    try std.testing.expectEqual(@as(i64, 7), model.ir_version);
    try std.testing.expectEqual(@as(i64, 13), model.opset);
    try std.testing.expectEqual(@as(usize, 1), model.graph.nodes.len);
    const node = model.graph.nodes[0];
    try std.testing.expectEqualStrings("Mul", node.op_type);
    try std.testing.expectEqual(@as(f32, 0.25), node.float("alpha", 0));
    try std.testing.expectEqualSlices(i64, &.{ 1, -2 }, node.ints("pads").?);
    const w = try model.graph.initializers[0].toF32(arena);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, -2 }, w);
    try std.testing.expectError(error.InvalidData, parse(arena, &.{ 0x3a, 0x05, 0x00 }));
}
