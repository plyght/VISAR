//! The graph IR: values (tensors, possibly constant) and nodes (typed ops
//! with parsed attributes). Imported once from ONNX; every plan works on its
//! own copy, where shape inference, constant folding and the optimisation
//! passes rewrite it for one fixed input shape.

const std = @import("std");
const onnx = @import("onnx.zig");
const Allocator = std.mem.Allocator;

pub const Error = Allocator.Error || error{ Unsupported, InvalidModel, ShapeMismatch };

pub const max_rank = 6;

pub const Shape = struct {
    rank: u8 = 0,
    d: [max_rank]usize = @splat(0),

    pub fn init(ds: []const usize) Shape {
        var s: Shape = .{ .rank = @intCast(ds.len) };
        @memcpy(s.d[0..ds.len], ds);
        return s;
    }

    pub fn dims(s: *const Shape) []const usize {
        return s.d[0..s.rank];
    }

    pub fn count(s: Shape) usize {
        var n: usize = 1;
        for (s.d[0..s.rank]) |x| n *= x;
        return n;
    }

    pub fn eql(a: Shape, b: Shape) bool {
        return a.rank == b.rank and std.mem.eql(usize, a.d[0..a.rank], b.d[0..b.rank]);
    }

    pub fn format(s: Shape, w: *std.Io.Writer) std.Io.Writer.Error!void {
        try w.writeAll("[");
        for (s.d[0..s.rank], 0..) |x, i| {
            if (i > 0) try w.writeAll(",");
            try w.print("{d}", .{x});
        }
        try w.writeAll("]");
    }
};

pub const DType = enum { f32, i64 };

pub const Const = union(DType) {
    f32: []const f32,
    i64: []const i64,
};

pub const ValueId = u32;
/// An omitted optional input.
pub const none: ValueId = std.math.maxInt(ValueId);

pub const Value = struct {
    name: []const u8,
    dtype: DType = .f32,
    shape: Shape = .{},
    /// Known at plan time (initializers, Constant nodes, folded subgraphs).
    data: ?Const = null,
    /// Shape known (inference reached it).
    shaped: bool = false,
};

/// Activation applied element-wise, standalone or fused into an epilogue.
pub const Act = union(enum) {
    none,
    relu,
    leaky_relu: f32,
    clip: [2]f32,
    sigmoid,
    /// max(0, min(1, alpha * x + beta))
    hard_sigmoid: [2]f32,
    hard_swish,
    tanh,
    /// x * sigmoid(x)
    silu,

    pub fn apply(a: Act, x: f32) f32 {
        return switch (a) {
            .none => x,
            .relu => @max(x, 0),
            .leaky_relu => |al| if (x >= 0) x else x * al,
            .clip => |c| @min(@max(x, c[0]), c[1]),
            .sigmoid => 1.0 / (1.0 + @exp(-x)),
            .hard_sigmoid => |p| @min(@max(p[0] * x + p[1], 0), 1),
            .hard_swish => x * @min(@max(x / 6.0 + 0.5, 0), 1),
            .tanh => std.math.tanh(x),
            .silu => x / (1.0 + @exp(-x)),
        };
    }

    /// Whether act(0) == 0 (padding lanes of blocked tensors stay zero).
    pub fn keepsZero(a: Act) bool {
        // Spelled out rather than apply(0) == 0, which LLVM turns into a
        // select among float constants (a constant-pool load wasm cannot
        // lower without SIMD).
        return switch (a) {
            .none, .relu, .leaky_relu, .hard_swish, .tanh, .silu => true,
            .clip => |c| c[0] <= 0 and c[1] >= 0,
            .sigmoid => false,
            .hard_sigmoid => |p| p[1] <= 0,
        };
    }
};

pub const Conv = struct {
    kernel: [2]u32 = .{ 1, 1 },
    stride: [2]u32 = .{ 1, 1 },
    /// top, left, bottom, right
    pad: [4]u32 = @splat(0),
    dilation: [2]u32 = .{ 1, 1 },
    group: u32 = 1,
    /// Fused epilogue: y = act(conv + bias [+ residual]) or, with
    /// `residual_after_act`, act(conv + bias) + residual.
    act: Act = .none,
    residual: bool = false,
    residual_after_act: bool = false,
};

pub const Pool = struct {
    kernel: [2]u32 = .{ 1, 1 },
    stride: [2]u32 = .{ 1, 1 },
    pad: [4]u32 = @splat(0),
    dilation: [2]u32 = .{ 1, 1 },
    ceil_mode: bool = false,
    count_include_pad: bool = false,
};

pub const Resize = struct {
    mode: enum { nearest, linear } = .nearest,
    coord: enum { half_pixel, asymmetric, align_corners, pytorch_half_pixel, tf_half_pixel_for_nn } = .half_pixel,
    nearest: enum { round_prefer_floor, round_prefer_ceil, floor, ceil } = .round_prefer_floor,
    /// Resolved at shape inference: per-axis scale (output / input) of the
    /// two spatial axes, as ONNX defines it for the coordinate transform.
    scale: [2]f32 = .{ 1, 1 },
};

pub const Binary = enum { add, sub, mul, div, max, min, pow };

pub const Op = union(enum) {
    /// inputs: x, w, [b], [residual]
    conv: Conv,
    /// inputs: x, scale, bias, mean, var (folded into a preceding conv
    /// where possible, else run as a per-channel affine)
    batch_norm: f32,
    act: Act,
    binary: Binary,
    max_pool: Pool,
    average_pool: Pool,
    global_average_pool,
    /// inputs: x, [roi], [scales], [sizes]
    resize: Resize,
    concat: i64,
    /// inputs: x, shape
    reshape: bool,
    flatten: i64,
    transpose: [max_rank]u8,
    /// axes as attribute (opset < 13) or input
    squeeze: []const i64,
    unsqueeze: []const i64,
    softmax: i64,
    /// Channels [start, end) of a rank-4 NCHW tensor (made by the passes).
    channel_slice: [2]usize,
    identity,
    // Evaluated by constant folding only.
    shape,
    gather: i64,
    cast: DType,
    /// inputs: data, starts, ends, [axes], [steps]
    slice,
};

pub const Node = struct {
    name: []const u8,
    op: Op,
    inputs: []ValueId,
    outputs: []ValueId,
    dead: bool = false,
};

pub const Graph = struct {
    values: std.ArrayList(Value) = .empty,
    nodes: std.ArrayList(Node) = .empty,
    inputs: []ValueId = &.{},
    outputs: []ValueId = &.{},

    pub fn value(g: *Graph, id: ValueId) *Value {
        return &g.values.items[id];
    }

    pub fn find(g: *const Graph, name: []const u8) ?ValueId {
        for (g.values.items, 0..) |v, i| if (std.mem.eql(u8, v.name, name)) return @intCast(i);
        return null;
    }

    pub fn addValue(g: *Graph, arena: Allocator, v: Value) Allocator.Error!ValueId {
        try g.values.append(arena, v);
        return @intCast(g.values.items.len - 1);
    }

    /// A deep copy (values, nodes and id lists; constant data is shared).
    pub fn clone(g: *const Graph, arena: Allocator) Allocator.Error!Graph {
        var c: Graph = .{};
        try c.values.appendSlice(arena, g.values.items);
        try c.nodes.ensureTotalCapacity(arena, g.nodes.items.len);
        for (g.nodes.items) |n| {
            var m = n;
            m.inputs = try arena.dupe(ValueId, n.inputs);
            m.outputs = try arena.dupe(ValueId, n.outputs);
            c.nodes.appendAssumeCapacity(m);
        }
        c.inputs = try arena.dupe(ValueId, g.inputs);
        c.outputs = try arena.dupe(ValueId, g.outputs);
        return c;
    }

    pub fn isOutput(g: *const Graph, id: ValueId) bool {
        return std.mem.indexOfScalar(ValueId, g.outputs, id) != null;
    }

    /// Live nodes reading `id`.
    pub fn consumers(g: *const Graph, id: ValueId) usize {
        var n: usize = 0;
        for (g.nodes.items) |node| {
            if (node.dead) continue;
            for (node.inputs) |i| {
                if (i == id) n += 1;
            }
        }
        return n;
    }

    /// The live node producing `id`, if any.
    pub fn producer(g: *const Graph, id: ValueId) ?usize {
        for (g.nodes.items, 0..) |node, i| {
            if (node.dead) continue;
            if (std.mem.indexOfScalar(ValueId, node.outputs, id) != null) return i;
        }
        return null;
    }

    /// Makes every reader of `from` read `to` instead (graph outputs too).
    pub fn replaceUses(g: *Graph, from: ValueId, to: ValueId) void {
        for (g.nodes.items) |*node| {
            for (node.inputs) |*i| {
                if (i.* == from) i.* = to;
            }
        }
        for (g.outputs) |*o| {
            if (o.* == from) o.* = to;
        }
    }
};

// ------------------------------------------------------------------ import

fn ints2(a: ?[]const i64, default: u32) Error![2]u32 {
    const v = a orelse return .{ default, default };
    if (v.len != 2) return error.Unsupported;
    return .{ try pos(v[0]), try pos(v[1]) };
}

fn pos(x: i64) Error!u32 {
    if (x < 0 or x > std.math.maxInt(u31)) return error.InvalidModel;
    return @intCast(x);
}

fn pads4(n: onnx.Node) Error![4]u32 {
    const auto = n.string("auto_pad", "NOTSET");
    if (!std.mem.eql(u8, auto, "NOTSET") and !std.mem.eql(u8, auto, "VALID")) return error.Unsupported;
    const p = n.ints("pads") orelse return @splat(0);
    if (p.len != 4) return error.Unsupported;
    // ONNX order: x1_begin, x2_begin, x1_end, x2_end = top, left, bottom, right
    return .{ try pos(p[0]), try pos(p[1]), try pos(p[2]), try pos(p[3]) };
}

fn importNode(arena: Allocator, n: onnx.Node, opset: i64) Error!?Op {
    const t = n.op_type;
    const eq = struct {
        fn f(a: []const u8, b: []const u8) bool {
            return std.mem.eql(u8, a, b);
        }
    }.f;
    if (n.domain.len != 0 and !eq(n.domain, "ai.onnx")) return error.Unsupported;
    if (eq(t, "Conv")) {
        const k = n.ints("kernel_shape") orelse return error.Unsupported; // inferred later if absent
        if (k.len != 2) return error.Unsupported;
        return .{ .conv = .{
            .kernel = try ints2(k, 1),
            .stride = try ints2(n.ints("strides"), 1),
            .dilation = try ints2(n.ints("dilations"), 1),
            .pad = try pads4(n),
            .group = try pos(n.int("group", 1)),
        } };
    }
    if (eq(t, "Relu")) return .{ .act = .relu };
    if (eq(t, "Sigmoid")) return .{ .act = .sigmoid };
    if (eq(t, "Tanh")) return .{ .act = .tanh };
    if (eq(t, "HardSwish")) return .{ .act = .hard_swish };
    if (eq(t, "LeakyRelu")) return .{ .act = .{ .leaky_relu = n.float("alpha", 0.01) } };
    if (eq(t, "HardSigmoid")) return .{ .act = .{ .hard_sigmoid = .{ n.float("alpha", 0.2), n.float("beta", 0.5) } } };
    if (eq(t, "Clip")) return .{ .act = .{ .clip = .{ n.float("min", -std.math.floatMax(f32)), n.float("max", std.math.floatMax(f32)) } } };
    if (eq(t, "Add")) return .{ .binary = .add };
    if (eq(t, "Sub")) return .{ .binary = .sub };
    if (eq(t, "Mul")) return .{ .binary = .mul };
    if (eq(t, "Div")) return .{ .binary = .div };
    if (eq(t, "Max")) return .{ .binary = .max };
    if (eq(t, "Min")) return .{ .binary = .min };
    if (eq(t, "Pow")) return .{ .binary = .pow };
    if (eq(t, "BatchNormalization")) return .{ .batch_norm = n.float("epsilon", 1e-5) };
    if (eq(t, "MaxPool") or eq(t, "AveragePool")) {
        const k = n.ints("kernel_shape") orelse return error.InvalidModel;
        if (k.len != 2) return error.Unsupported;
        if (n.outputs.len > 1 and n.outputs[1].len > 0) return error.Unsupported; // indices
        const p: Pool = .{
            .kernel = try ints2(k, 1),
            .stride = try ints2(n.ints("strides"), 1),
            .dilation = try ints2(n.ints("dilations"), 1),
            .pad = try pads4(n),
            .ceil_mode = n.int("ceil_mode", 0) != 0,
            .count_include_pad = n.int("count_include_pad", 0) != 0,
        };
        return if (eq(t, "MaxPool")) .{ .max_pool = p } else .{ .average_pool = p };
    }
    if (eq(t, "GlobalAveragePool")) return .global_average_pool;
    if (eq(t, "Resize") or eq(t, "Upsample")) {
        var r: Resize = .{};
        const mode = n.string("mode", "nearest");
        r.mode = if (eq(mode, "nearest")) .nearest else if (eq(mode, "linear") or eq(mode, "bilinear")) .linear else return error.Unsupported;
        if (eq(t, "Upsample")) {
            r.coord = .asymmetric;
            r.nearest = .floor;
        } else if (opset < 11) {
            r.coord = .asymmetric;
            r.nearest = .floor;
        } else {
            const c = n.string("coordinate_transformation_mode", "half_pixel");
            r.coord = if (eq(c, "half_pixel")) .half_pixel else if (eq(c, "asymmetric")) .asymmetric else if (eq(c, "align_corners")) .align_corners else if (eq(c, "pytorch_half_pixel")) .pytorch_half_pixel else if (eq(c, "tf_half_pixel_for_nn")) .tf_half_pixel_for_nn else return error.Unsupported;
            const nm = n.string("nearest_mode", "round_prefer_floor");
            r.nearest = if (eq(nm, "round_prefer_floor")) .round_prefer_floor else if (eq(nm, "round_prefer_ceil")) .round_prefer_ceil else if (eq(nm, "floor")) .floor else if (eq(nm, "ceil")) .ceil else return error.Unsupported;
        }
        return .{ .resize = r };
    }
    if (eq(t, "Concat")) return .{ .concat = n.int("axis", 0) };
    if (eq(t, "Reshape")) return .{ .reshape = n.int("allowzero", 0) != 0 };
    if (eq(t, "Flatten")) return .{ .flatten = n.int("axis", 1) };
    if (eq(t, "Transpose")) {
        var perm: [max_rank]u8 = @splat(255);
        if (n.ints("perm")) |p| {
            if (p.len > max_rank) return error.Unsupported;
            for (p, 0..) |x, i| perm[i] = @intCast(try pos(x));
        }
        return .{ .transpose = perm };
    }
    if (eq(t, "Squeeze")) return .{ .squeeze = try arena.dupe(i64, n.ints("axes") orelse &.{}) };
    if (eq(t, "Unsqueeze")) return .{ .unsqueeze = try arena.dupe(i64, n.ints("axes") orelse &.{}) };
    if (eq(t, "Softmax")) return .{ .softmax = n.int("axis", if (opset >= 13) -1 else 1) };
    if (eq(t, "Identity") or eq(t, "Dropout")) return .identity;
    if (eq(t, "Shape")) return .shape;
    if (eq(t, "Gather")) return .{ .gather = n.int("axis", 0) };
    if (eq(t, "Slice")) return .slice;
    if (eq(t, "Cast")) {
        const to: onnx.DataType = @fromBackingInt(@intCast(@as(i32, @intCast(n.int("to", 1)))));
        return .{ .cast = switch (to) {
            .float, .double, .float16 => .f32,
            .int64, .int32, .int16, .int8, .uint8, .bool => .i64,
            else => return error.Unsupported,
        } };
    }
    if (eq(t, "Constant")) return null;
    return error.Unsupported;
}

fn constFromTensor(arena: Allocator, t: onnx.Tensor) Error!Value {
    var v: Value = .{ .name = t.name, .shaped = true };
    if (t.dims.len > max_rank) return error.Unsupported;
    v.shape.rank = @intCast(t.dims.len);
    for (t.dims, 0..) |d, i| v.shape.d[i] = @intCast(@max(d, 0));
    switch (t.data_type) {
        .float, .double, .float16, .bfloat16 => {
            v.dtype = .f32;
            v.data = .{ .f32 = t.toF32(arena) catch |e| return mapErr(e) };
        },
        .int64, .int32, .int16, .int8, .uint8, .uint16, .uint32, .uint64, .bool => {
            v.dtype = .i64;
            v.data = .{ .i64 = t.toI64(arena) catch |e| return mapErr(e) };
        },
        else => return error.Unsupported,
    }
    return v;
}

fn mapErr(e: onnx.Error) Error {
    return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        error.Unsupported => error.Unsupported,
        error.InvalidData => error.InvalidModel,
    };
}

pub const Imported = struct {
    graph: Graph,
    /// Name, element type and declared shape (null dims unknown) per input.
    input_dims: []const []const ?usize,
};

/// Builds the IR from a parsed ONNX model.
pub fn import(arena: Allocator, model: onnx.Model) Error!Imported {
    var g: Graph = .{};
    var names: std.StringHashMapUnmanaged(ValueId) = .empty;
    const og = model.graph;
    for (og.initializers) |t| {
        const id = try g.addValue(arena, try constFromTensor(arena, t));
        try names.put(arena, t.name, id);
    }
    var inputs: std.ArrayList(ValueId) = .empty;
    var input_dims: std.ArrayList([]const ?usize) = .empty;
    for (og.inputs) |vi| {
        if (names.contains(vi.name)) continue; // initializer listed as input (old exporters)
        if (vi.elem_type != .float and vi.elem_type != .undefined) return error.Unsupported;
        const id = try g.addValue(arena, .{ .name = vi.name });
        try names.put(arena, vi.name, id);
        try inputs.append(arena, id);
        const shape = vi.shape orelse &.{};
        const dims = try arena.alloc(?usize, shape.len);
        for (shape, dims) |d, *o| o.* = switch (d) {
            .value => |x| if (x > 0) @intCast(x) else null,
            else => null,
        };
        try input_dims.append(arena, dims);
    }
    for (og.nodes) |n| {
        const op = try importNode(arena, n, model.opset) orelse {
            // Constant: becomes a constant value.
            if (n.outputs.len != 1) return error.InvalidModel;
            var v: Value = undefined;
            if (n.attr("value")) |a| {
                v = try constFromTensor(arena, a.t orelse return error.InvalidModel);
            } else if (n.attr("value_float")) |a| {
                v = .{ .name = "", .dtype = .f32, .shaped = true, .data = .{ .f32 = try arena.dupe(f32, &.{a.f}) } };
            } else if (n.attr("value_floats")) |a| {
                v = .{ .name = "", .dtype = .f32, .shaped = true, .shape = .init(&.{a.floats.len}), .data = .{ .f32 = a.floats } };
            } else if (n.attr("value_int")) |a| {
                v = .{ .name = "", .dtype = .i64, .shaped = true, .data = .{ .i64 = try arena.dupe(i64, &.{a.i}) } };
            } else if (n.attr("value_ints")) |a| {
                v = .{ .name = "", .dtype = .i64, .shaped = true, .shape = .init(&.{a.ints.len}), .data = .{ .i64 = a.ints } };
            } else return error.Unsupported;
            v.name = n.outputs[0];
            const id = try g.addValue(arena, v);
            try names.put(arena, n.outputs[0], id);
            continue;
        };
        var op2 = op;
        // Opset-dependent attribute/input forms.
        if (op2 == .act and op2.act == .clip and n.inputs.len > 1) op2.act = .{ .clip = .{ -std.math.floatMax(f32), std.math.floatMax(f32) } }; // min/max inputs, resolved by folding
        const ins = try arena.alloc(ValueId, n.inputs.len);
        for (n.inputs, ins) |name, *o| {
            if (name.len == 0) {
                o.* = none;
                continue;
            }
            o.* = names.get(name) orelse return error.InvalidModel;
        }
        const outs = try arena.alloc(ValueId, @min(n.outputs.len, 1));
        for (outs, 0..) |*o, i| {
            o.* = try g.addValue(arena, .{ .name = n.outputs[i] });
            try names.put(arena, n.outputs[i], o.*);
        }
        try g.nodes.append(arena, .{ .name = n.name, .op = op2, .inputs = ins, .outputs = outs });
    }
    const outs = try arena.alloc(ValueId, og.outputs.len);
    for (og.outputs, outs) |vi, *o| o.* = names.get(vi.name) orelse return error.InvalidModel;
    g.inputs = inputs.items;
    g.outputs = outs;
    return .{ .graph = g, .input_dims = input_dims.items };
}

// ---------------------------------------------------- shape inference + folding

fn normAxis(axis: i64, rank: usize) Error!usize {
    const r: i64 = @intCast(rank);
    const a = if (axis < 0) axis + r else axis;
    if (a < 0 or a >= @max(r, 1)) return error.InvalidModel;
    return @intCast(a);
}

fn constI64(g: *Graph, id: ValueId) Error![]const i64 {
    if (id == none) return error.InvalidModel;
    const v = g.value(id);
    const d = v.data orelse return error.Unsupported; // dynamic shape tensors
    return switch (d) {
        .i64 => |x| x,
        .f32 => error.InvalidModel,
    };
}

fn constF32(g: *Graph, id: ValueId) Error![]const f32 {
    const v = g.value(id);
    const d = v.data orelse return error.Unsupported;
    return switch (d) {
        .f32 => |x| x,
        .i64 => error.InvalidModel,
    };
}

fn spatialOut(in: usize, k: u32, s: u32, d: u32, p0: u32, p1: u32, ceil_mode: bool) Error!usize {
    const ek = @as(usize, d) * (k - 1) + 1;
    const padded = in + p0 + p1;
    if (padded < ek or s == 0) return error.ShapeMismatch;
    var o = if (ceil_mode) (padded - ek + s - 1) / s + 1 else (padded - ek) / s + 1;
    // A window may not start in the end padding (ONNX/PyTorch rule).
    if (ceil_mode and (o - 1) * s >= in + p0) o -= 1;
    return o;
}

pub fn broadcastShapes(a: Shape, b: Shape) Error!Shape {
    const r = @max(a.rank, b.rank);
    var s: Shape = .{ .rank = r };
    for (0..r) |i| {
        const da = if (i + a.rank >= r) a.d[i + a.rank - r] else 1;
        const db = if (i + b.rank >= r) b.d[i + b.rank - r] else 1;
        if (da != db and da != 1 and db != 1) return error.ShapeMismatch;
        s.d[i] = if (da == 1) db else da;
    }
    return s;
}

/// Infers every value's shape for the given input shapes and folds every
/// node whose inputs are all constant (and Shape nodes), in place.
pub fn inferShapes(g: *Graph, arena: Allocator, input_shapes: []const Shape) Error!void {
    if (input_shapes.len != g.inputs.len) return error.ShapeMismatch;
    for (g.inputs, input_shapes) |id, s| {
        g.value(id).shape = s;
        g.value(id).shaped = true;
    }
    for (g.nodes.items) |*n| {
        if (n.dead) continue;
        try inferNode(g, arena, n);
    }
}

fn inShape(g: *Graph, n: *const Node, i: usize) Error!Shape {
    if (i >= n.inputs.len or n.inputs[i] == none) return error.InvalidModel;
    const v = g.value(n.inputs[i]);
    if (!v.shaped) return error.InvalidModel;
    return v.shape;
}

fn allConst(g: *Graph, n: *const Node) bool {
    for (n.inputs) |i| {
        if (i != none and g.value(i).data == null) return false;
    }
    return true;
}

fn inferNode(g: *Graph, arena: Allocator, n: *Node) Error!void {
    const out = g.value(n.outputs[0]);
    const x = try inShape(g, n, 0);
    const xv = g.value(n.inputs[0]);
    out.dtype = xv.dtype;
    var s: Shape = x;
    switch (n.op) {
        .conv => |*c| {
            const w = try inShape(g, n, 1);
            if (x.rank != 4 or w.rank != 4) return error.Unsupported;
            if (c.group == 0 or x.d[1] != w.d[1] * c.group or w.d[0] % c.group != 0) return error.ShapeMismatch;
            if (w.d[2] != c.kernel[0] or w.d[3] != c.kernel[1]) return error.ShapeMismatch;
            s.d[1] = w.d[0];
            s.d[2] = try spatialOut(x.d[2], c.kernel[0], c.stride[0], c.dilation[0], c.pad[0], c.pad[2], false);
            s.d[3] = try spatialOut(x.d[3], c.kernel[1], c.stride[1], c.dilation[1], c.pad[1], c.pad[3], false);
            if (n.inputs.len > 2 and n.inputs[2] != none) {
                const b = try inShape(g, n, 2);
                if (b.count() != w.d[0]) return error.ShapeMismatch;
            }
        },
        .max_pool, .average_pool => |p| {
            if (x.rank != 4) return error.Unsupported;
            s.d[2] = try spatialOut(x.d[2], p.kernel[0], p.stride[0], p.dilation[0], p.pad[0], p.pad[2], p.ceil_mode);
            s.d[3] = try spatialOut(x.d[3], p.kernel[1], p.stride[1], p.dilation[1], p.pad[1], p.pad[3], p.ceil_mode);
        },
        .global_average_pool => {
            if (x.rank < 3) return error.Unsupported;
            for (2..x.rank) |i| s.d[i] = 1;
        },
        .batch_norm => {},
        .act => |*a| {
            // Clip with min/max inputs (opset >= 11).
            if (a.* == .clip and n.inputs.len > 1) {
                var lo: f32 = -std.math.floatMax(f32);
                var hi: f32 = std.math.floatMax(f32);
                if (n.inputs[1] != none) lo = (try constF32(g, n.inputs[1]))[0];
                if (n.inputs.len > 2 and n.inputs[2] != none) hi = (try constF32(g, n.inputs[2]))[0];
                a.* = .{ .clip = .{ lo, hi } };
                n.inputs = n.inputs[0..1];
            }
        },
        .binary => {
            s = try broadcastShapes(x, try inShape(g, n, 1));
        },
        .resize => |*r| {
            if (x.rank != 4) return error.Unsupported;
            // Upsample(x, scales) / Resize(x, roi, scales, sizes) / Resize-10(x, scales)
            var scales: ?[]const f32 = null;
            var sizes: ?[]const i64 = null;
            if (n.inputs.len == 2) {
                scales = try constF32(g, n.inputs[1]);
            } else {
                if (n.inputs.len > 2 and n.inputs[2] != none and g.value(n.inputs[2]).shape.count() > 0) scales = try constF32(g, n.inputs[2]);
                if (n.inputs.len > 3 and n.inputs[3] != none) sizes = try constI64(g, n.inputs[3]);
            }
            if (sizes) |sz| {
                if (sz.len != 4) return error.Unsupported;
                for (0..4) |i| s.d[i] = @intCast(@max(sz[i], 0));
                r.scale = .{
                    @as(f32, @floatFromInt(s.d[2])) / @as(f32, @floatFromInt(x.d[2])),
                    @as(f32, @floatFromInt(s.d[3])) / @as(f32, @floatFromInt(x.d[3])),
                };
            } else if (scales) |sc| {
                if (sc.len != 4) return error.Unsupported;
                for (0..4) |i| s.d[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(x.d[i])) * sc[i]));
                r.scale = .{ sc[2], sc[3] };
            } else return error.InvalidModel;
            if (s.d[0] != x.d[0] or s.d[1] != x.d[1]) return error.Unsupported;
        },
        .concat => |axis| {
            const a = try normAxis(axis, x.rank);
            s.d[a] = 0;
            for (0..n.inputs.len) |i| {
                const si = try inShape(g, n, i);
                if (si.rank != x.rank) return error.ShapeMismatch;
                for (0..x.rank) |k| {
                    if (k != a and si.d[k] != x.d[k]) return error.ShapeMismatch;
                }
                s.d[a] += si.d[a];
            }
        },
        .reshape => |allow_zero| {
            const want = try constI64(g, n.inputs[1]);
            if (want.len > max_rank) return error.Unsupported;
            s = .{ .rank = @intCast(want.len) };
            var known: usize = 1;
            var infer: ?usize = null;
            for (want, 0..) |d, i| {
                if (d == -1) {
                    if (infer != null) return error.InvalidModel;
                    infer = i;
                } else if (d == 0 and !allow_zero) {
                    if (i >= x.rank) return error.InvalidModel;
                    s.d[i] = x.d[i];
                    known *= x.d[i];
                } else if (d < 0) return error.InvalidModel else {
                    s.d[i] = @intCast(d);
                    known *= s.d[i];
                }
            }
            if (infer) |i| {
                if (known == 0) return error.InvalidModel;
                s.d[i] = x.count() / known;
            }
            if (s.count() != x.count()) return error.ShapeMismatch;
        },
        .flatten => |axis| {
            const a: usize = if (axis < 0) @intCast(axis + @as(i64, x.rank)) else @intCast(axis);
            if (a > x.rank) return error.InvalidModel;
            var lo: usize = 1;
            for (0..a) |i| lo *= x.d[i];
            s = .init(&.{ lo, x.count() / @max(lo, 1) });
        },
        .transpose => |*perm| {
            if (perm[0] == 255) {
                for (0..x.rank) |i| perm[i] = @intCast(x.rank - 1 - i);
            }
            var seen: [max_rank]bool = @splat(false);
            for (0..x.rank) |i| {
                const p = perm[i];
                if (p >= x.rank or seen[p]) return error.InvalidModel;
                seen[p] = true;
                s.d[i] = x.d[p];
            }
        },
        .squeeze, .unsqueeze => {
            const sq = n.op == .squeeze;
            var axes: []const i64 = if (sq) n.op.squeeze else n.op.unsqueeze;
            if (n.inputs.len > 1 and n.inputs[1] != none) axes = try constI64(g, n.inputs[1]);
            if (sq) {
                s = .{};
                for (0..x.rank) |i| {
                    var drop = false;
                    if (axes.len == 0) {
                        drop = x.d[i] == 1;
                    } else for (axes) |ax| {
                        if (try normAxis(ax, x.rank) == i) drop = true;
                    }
                    if (!drop) {
                        s.d[s.rank] = x.d[i];
                        s.rank += 1;
                    }
                }
            } else {
                const r = x.rank + axes.len;
                if (r > max_rank) return error.Unsupported;
                s = .{ .rank = @intCast(r) };
                var mark: [max_rank]bool = @splat(false);
                for (axes) |ax| mark[try normAxis(ax, r)] = true;
                var k: usize = 0;
                for (0..r) |i| {
                    if (mark[i]) s.d[i] = 1 else {
                        s.d[i] = x.d[k];
                        k += 1;
                    }
                }
            }
        },
        .softmax => |axis| _ = try normAxis(axis, x.rank),
        .channel_slice => |r| {
            if (x.rank != 4 or r[1] > x.d[1] or r[0] >= r[1]) return error.InvalidModel;
            s.d[1] = r[1] - r[0];
        },
        .identity => {},
        .shape => {
            s = .init(&.{x.rank});
            out.dtype = .i64;
            const d = try arena.alloc(i64, x.rank);
            for (d, 0..) |*o, i| o.* = @intCast(x.d[i]);
            out.data = .{ .i64 = d };
        },
        .gather => |axis| {
            const idx = try inShape(g, n, 1);
            const a = try normAxis(axis, x.rank);
            s = .{};
            for (0..a) |i| {
                s.d[s.rank] = x.d[i];
                s.rank += 1;
            }
            for (idx.dims()) |d| {
                s.d[s.rank] = d;
                s.rank += 1;
            }
            for (a + 1..x.rank) |i| {
                s.d[s.rank] = x.d[i];
                s.rank += 1;
            }
            if (!allConst(g, n)) return error.Unsupported;
        },
        .cast => |to| {
            out.dtype = to;
            if (!allConst(g, n)) return error.Unsupported;
        },
        .slice => {
            if (!allConst(g, n)) return error.Unsupported;
        },
    }
    out.shape = s;
    out.shaped = true;
    if (out.data == null and allConst(g, n)) try fold(g, arena, n);
}

/// Evaluates a node with constant inputs (shape arithmetic in exported
/// graphs). Only the ops such subgraphs use are supported; anything else is
/// left to run at inference time.
fn fold(g: *Graph, arena: Allocator, n: *Node) Error!void {
    const out = g.value(n.outputs[0]);
    const xin = g.value(n.inputs[0]);
    switch (n.op) {
        .identity, .reshape, .flatten, .squeeze, .unsqueeze => out.data = xin.data,
        .cast => |to| {
            const cnt = xin.shape.count();
            switch (to) {
                .f32 => {
                    const d = try arena.alloc(f32, cnt);
                    switch (xin.data.?) {
                        .f32 => |v| @memcpy(d, v),
                        .i64 => |v| for (d, v) |*o, a| {
                            o.* = @floatFromInt(a);
                        },
                    }
                    out.data = .{ .f32 = d };
                },
                .i64 => {
                    const d = try arena.alloc(i64, cnt);
                    switch (xin.data.?) {
                        .i64 => |v| @memcpy(d, v),
                        .f32 => |v| for (d, v) |*o, a| {
                            o.* = @intFromFloat(@trunc(a));
                        },
                    }
                    out.data = .{ .i64 = d };
                },
            }
        },
        .gather => |axis| {
            const a = try normAxis(axis, xin.shape.rank);
            const idx = try constI64(g, n.inputs[1]);
            var outer: usize = 1;
            for (0..a) |i| outer *= xin.shape.d[i];
            var inner: usize = 1;
            for (a + 1..xin.shape.rank) |i| inner *= xin.shape.d[i];
            const dim = xin.shape.d[a];
            switch (xin.data.?) {
                inline else => |v, tag| {
                    const T = @TypeOf(v[0]);
                    const d = try arena.alloc(T, outer * idx.len * inner);
                    for (0..outer) |o| for (idx, 0..) |ix, j| {
                        const k: usize = @intCast(if (ix < 0) ix + @as(i64, @intCast(dim)) else ix);
                        if (k >= dim) return error.InvalidModel;
                        @memcpy(d[(o * idx.len + j) * inner ..][0..inner], v[(o * dim + k) * inner ..][0..inner]);
                    };
                    out.data = @unionInit(Const, @tagName(tag), d);
                },
            }
        },
        .concat => |axis| {
            if (out.dtype != .i64 or out.shape.rank != 1) return;
            _ = axis;
            var d: std.ArrayList(i64) = .empty;
            for (n.inputs) |i| try d.appendSlice(arena, try constI64(g, i));
            out.data = .{ .i64 = d.items };
        },
        .binary => |op| {
            if (out.dtype != .i64 or xin.shape.count() != out.shape.count()) return;
            const a = try constI64(g, n.inputs[0]);
            const b = try constI64(g, n.inputs[1]);
            const d = try arena.alloc(i64, a.len);
            for (d, 0..) |*o, i| {
                const bb = b[if (b.len == 1) 0 else i];
                o.* = switch (op) {
                    .add => a[i] + bb,
                    .sub => a[i] - bb,
                    .mul => a[i] * bb,
                    .div => if (bb == 0) return error.InvalidModel else @divTrunc(a[i], bb),
                    .max => @max(a[i], bb),
                    .min => @min(a[i], bb),
                    .pow => return,
                };
            }
            out.data = .{ .i64 = d };
        },
        .slice => {
            if (xin.shape.rank != 1) return error.Unsupported;
            const v = try constI64(g, n.inputs[0]);
            const st = (try constI64(g, n.inputs[1]))[0];
            const en = (try constI64(g, n.inputs[2]))[0];
            if (n.inputs.len > 4 and n.inputs[4] != none and (try constI64(g, n.inputs[4]))[0] != 1) return error.Unsupported;
            const len: i64 = @intCast(v.len);
            const lo: usize = @intCast(std.math.clamp(if (st < 0) st + len else st, 0, len));
            const hi: usize = @intCast(std.math.clamp(if (en < 0) en + len else en, 0, len));
            out.data = .{ .i64 = if (hi > lo) v[lo..hi] else &.{} };
            out.shape = .init(&.{if (hi > lo) hi - lo else 0});
        },
        else => {},
    }
}

test "shape inference for conv, pool, resize and reshape" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var g: Graph = .{};
    const x = try g.addValue(arena, .{ .name = "x" });
    const w = try g.addValue(arena, .{ .name = "w", .shaped = true, .shape = .init(&.{ 8, 3, 3, 3 }), .data = .{ .f32 = try arena.alloc(f32, 8 * 27) } });
    const y = try g.addValue(arena, .{ .name = "y" });
    const p = try g.addValue(arena, .{ .name = "p" });
    const sc = try g.addValue(arena, .{ .name = "sc", .shaped = true, .shape = .init(&.{4}), .data = .{ .f32 = &.{ 1, 1, 2, 2 } } });
    const r = try g.addValue(arena, .{ .name = "r" });
    const shp = try g.addValue(arena, .{ .name = "shp", .shaped = true, .shape = .init(&.{3}), .data = .{ .i64 = &.{ 1, -1, 2 } } });
    const z = try g.addValue(arena, .{ .name = "z" });
    try g.nodes.append(arena, .{ .name = "c", .op = .{ .conv = .{ .kernel = .{ 3, 3 }, .stride = .{ 2, 2 }, .pad = @splat(1) } }, .inputs = try arena.dupe(ValueId, &.{ x, w }), .outputs = try arena.dupe(ValueId, &.{y}) });
    try g.nodes.append(arena, .{ .name = "m", .op = .{ .max_pool = .{ .kernel = .{ 2, 2 }, .stride = .{ 2, 2 } } }, .inputs = try arena.dupe(ValueId, &.{y}), .outputs = try arena.dupe(ValueId, &.{p}) });
    try g.nodes.append(arena, .{ .name = "u", .op = .{ .resize = .{} }, .inputs = try arena.dupe(ValueId, &.{ p, sc }), .outputs = try arena.dupe(ValueId, &.{r}) });
    try g.nodes.append(arena, .{ .name = "s", .op = .{ .reshape = false }, .inputs = try arena.dupe(ValueId, &.{ r, shp }), .outputs = try arena.dupe(ValueId, &.{z}) });
    g.inputs = try arena.dupe(ValueId, &.{x});
    try inferShapes(&g, arena, &.{.init(&.{ 1, 3, 33, 20 })});
    try std.testing.expect(g.value(y).shape.eql(.init(&.{ 1, 8, 17, 10 })));
    try std.testing.expect(g.value(p).shape.eql(.init(&.{ 1, 8, 8, 5 })));
    try std.testing.expect(g.value(r).shape.eql(.init(&.{ 1, 8, 16, 10 })));
    try std.testing.expect(g.value(z).shape.eql(.init(&.{ 1, 640, 2 })));
}
