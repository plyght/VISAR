//! An ONNX model, built in memory, whose plan uses every kernel of the
//! Metal backend: the convolution variants (plain-input direct, blocked
//! direct, grouped, depthwise with a residual, pointwise), fused and
//! standalone activations of every kind, both poolings, global average
//! pooling, linear and nearest resize, per-channel / same-shape /
//! broadcast binary ops, blocked and plain concat (with a constant), the
//! layout reorders, transpose, channel extraction (merged sibling convs)
//! and softmax. Used by the Metal-vs-CPU tests.

const std = @import("std");
const onnx = @import("../onnx.zig");
const Allocator = std.mem.Allocator;
const Writer = onnx.Writer;

const Node = struct {
    w: Writer,

    fn ints(n: *Node, name: []const u8, vals: []const i64) !void {
        var t: Writer = .{ .gpa = n.w.gpa };
        try t.bytes(1, name);
        for (vals) |v| try t.int(8, v);
        try t.int(20, 7);
        try n.w.bytes(5, t.buf.items);
    }

    fn int(n: *Node, name: []const u8, v: i64) !void {
        var t: Writer = .{ .gpa = n.w.gpa };
        try t.bytes(1, name);
        try t.int(3, v);
        try t.int(20, 2);
        try n.w.bytes(5, t.buf.items);
    }

    fn float(n: *Node, name: []const u8, v: f32) !void {
        var t: Writer = .{ .gpa = n.w.gpa };
        try t.bytes(1, name);
        try t.float(2, v);
        try t.int(20, 1);
        try n.w.bytes(5, t.buf.items);
    }

    fn str(n: *Node, name: []const u8, v: []const u8) !void {
        var t: Writer = .{ .gpa = n.w.gpa };
        try t.bytes(1, name);
        try t.bytes(4, v);
        try t.int(20, 3);
        try n.w.bytes(5, t.buf.items);
    }
};

const Builder = struct {
    a: Allocator,
    g: Writer,
    rnd: std.Random,

    fn tensor(b: *Builder, name: []const u8, dims: []const i64, vals: []const f32) !void {
        var t: Writer = .{ .gpa = b.a };
        for (dims) |d| try t.int(1, d);
        try t.int(2, 1);
        try t.bytes(8, name);
        try t.bytes(9, std.mem.sliceAsBytes(vals));
        try b.g.bytes(5, t.buf.items);
    }

    fn random(b: *Builder, name: []const u8, dims: []const i64, lo: f32, hi: f32) !void {
        var n: usize = 1;
        for (dims) |d| n *= @intCast(d);
        const vals = try b.a.alloc(f32, n);
        for (vals) |*v| v.* = lo + (hi - lo) * b.rnd.float(f32);
        try b.tensor(name, dims, vals);
    }

    fn ints(b: *Builder, name: []const u8, vals: []const i64) !void {
        var t: Writer = .{ .gpa = b.a };
        try t.int(1, @intCast(vals.len));
        try t.int(2, 7);
        try t.bytes(8, name);
        for (vals) |v| try t.int(7, v);
        try b.g.bytes(5, t.buf.items);
    }

    fn node(b: *Builder, op: []const u8, ins: []const []const u8, out: []const u8) !Node {
        var n: Node = .{ .w = .{ .gpa = b.a } };
        for (ins) |i| try n.w.bytes(1, i);
        try n.w.bytes(2, out);
        try n.w.bytes(3, out);
        try n.w.bytes(4, op);
        return n;
    }

    fn add(b: *Builder, n: Node) !void {
        try b.g.bytes(1, n.w.buf.items);
    }

    fn simple(b: *Builder, op: []const u8, ins: []const []const u8, out: []const u8) !void {
        try b.add(try b.node(op, ins, out));
    }

    fn conv(b: *Builder, x: []const u8, w: []const u8, bias: []const u8, out: []const u8, k: i64, stride: i64, pad: i64, group: i64) !void {
        var n = try b.node("Conv", &.{ x, w, bias }, out);
        try n.ints("kernel_shape", &.{ k, k });
        try n.ints("strides", &.{ stride, stride });
        try n.ints("pads", &.{ pad, pad, pad, pad });
        if (group != 1) try n.int("group", group);
        try b.add(n);
    }
};

/// Input "x": [1, 3, H, W] with H and W multiples of 4. Outputs: see the
/// names in `output_names`.
pub fn build(a: Allocator) ![]const u8 {
    var prng = std.Random.DefaultPrng.init(1234);
    var b: Builder = .{ .a = a, .g = .{ .gpa = a }, .rnd = prng.random() };

    try b.random("w0", &.{ 8, 3, 3, 3 }, -0.02, 0.02);
    try b.random("b0", &.{8}, -0.5, 0.5);
    try b.random("w1", &.{ 12, 8, 3, 3 }, -0.15, 0.15);
    try b.random("b1", &.{12}, -0.2, 0.2);
    try b.random("wg", &.{ 12, 6, 3, 3 }, -0.2, 0.2);
    try b.random("bg", &.{12}, -0.2, 0.2);
    try b.random("wd", &.{ 12, 1, 3, 3 }, -0.4, 0.4);
    try b.random("bd", &.{12}, -0.2, 0.2);
    try b.random("wp", &.{ 16, 12, 1, 1 }, -0.4, 0.4);
    try b.random("bp", &.{16}, -0.2, 0.2);
    try b.random("bn_s", &.{16}, 0.5, 1.5);
    try b.random("bn_b", &.{16}, -0.3, 0.3);
    try b.random("bn_m", &.{16}, -0.2, 0.2);
    try b.random("bn_v", &.{16}, 0.5, 1.5);
    try b.random("cvec", &.{ 1, 16, 1, 1 }, -1.5, 1.5);
    try b.random("dvec", &.{ 16, 1, 1 }, 0.5, 2);
    try b.random("row", &.{ 1, 1, 1, 10 }, -1, 1);
    try b.random("ws", &.{ 4, 32, 1, 1 }, -0.3, 0.3);
    try b.random("bs", &.{4}, -0.1, 0.1);
    try b.random("wt", &.{ 4, 32, 1, 1 }, -0.3, 0.3);
    try b.random("bt", &.{4}, -0.1, 0.1);
    try b.random("wq", &.{ 5, 32, 1, 1 }, -0.3, 0.3);
    try b.random("tail", &.{ 1, 3 }, -1, 1);
    try b.tensor("one", &.{1}, &.{1.0});
    try b.tensor("two", &.{1}, &.{2.0});
    try b.tensor("scales", &.{4}, &.{ 1, 1, 2, 2 });
    try b.ints("flat", &.{ 1, -1 });

    // Convolutions: plain-input direct + leaky relu, strided blocked direct
    // + clip, grouped + hard swish, depthwise + residual + relu, pointwise
    // + tanh.
    try b.conv("x", "w0", "b0", "c0", 3, 1, 1, 1);
    {
        var n = try b.node("LeakyRelu", &.{"c0"}, "a0");
        try n.float("alpha", 0.1);
        try b.add(n);
    }
    try b.conv("a0", "w1", "b1", "c1", 3, 2, 1, 1);
    {
        var n = try b.node("Clip", &.{"c1"}, "a1");
        try n.float("min", -1);
        try n.float("max", 2);
        try b.add(n);
    }
    try b.conv("a1", "wg", "bg", "cg", 3, 1, 1, 2);
    try b.simple("HardSwish", &.{"cg"}, "ag");
    try b.conv("ag", "wd", "bd", "cd", 3, 1, 1, 12);
    try b.simple("Add", &.{ "cd", "ag" }, "rd");
    try b.simple("Relu", &.{"rd"}, "ad");
    try b.conv("ad", "wp", "bp", "cp", 1, 1, 0, 1);
    try b.simple("Tanh", &.{"cp"}, "ap");
    // Average pool (count_include_pad), batch norm (unfoldable: affine),
    // max pool, linear resize back up.
    {
        var n = try b.node("AveragePool", &.{"ap"}, "avg");
        try n.ints("kernel_shape", &.{ 3, 3 });
        try n.ints("strides", &.{ 1, 1 });
        try n.ints("pads", &.{ 1, 1, 1, 1 });
        try n.int("count_include_pad", 1);
        try b.add(n);
    }
    try b.simple("BatchNormalization", &.{ "avg", "bn_s", "bn_b", "bn_m", "bn_v" }, "bn");
    {
        var n = try b.node("MaxPool", &.{"bn"}, "mp");
        try n.ints("kernel_shape", &.{ 2, 2 });
        try n.ints("strides", &.{ 2, 2 });
        try b.add(n);
    }
    {
        var n = try b.node("Resize", &.{ "mp", "", "scales" }, "up");
        try n.str("mode", "linear");
        try b.add(n);
    }
    // Per-channel constants (either side), then same-shape blocked ops.
    try b.simple("Mul", &.{ "up", "cvec" }, "m");
    try b.simple("Sub", &.{ "one", "m" }, "s");
    try b.simple("Div", &.{ "s", "dvec" }, "dv");
    try b.simple("Min", &.{ "dv", "m" }, "mn");
    {
        var n = try b.node("Concat", &.{ "mn", "ap" }, "cat");
        try n.int("axis", 1);
        try b.add(n);
    }
    {
        var n = try b.node("HardSigmoid", &.{"cat"}, "hs");
        try n.float("alpha", 0.3);
        try n.float("beta", 0.4);
        try b.add(n);
    }
    // Head 1: global average pool -> flatten -> softmax -> concat with a
    // constant (plain).
    try b.simple("GlobalAveragePool", &.{"hs"}, "gap");
    try b.simple("Reshape", &.{ "gap", "flat" }, "gf");
    {
        var n = try b.node("Softmax", &.{"gf"}, "sm");
        try n.int("axis", 1);
        try b.add(n);
    }
    {
        var n = try b.node("Concat", &.{ "sm", "tail" }, "out_cat");
        try n.int("axis", 1);
        try b.add(n);
    }
    // Head 2: a general transpose (plain).
    {
        var n = try b.node("Transpose", &.{"hs"}, "out_tr");
        try n.ints("perm", &.{ 0, 3, 1, 2 });
        try b.add(n);
    }
    // Head 3: plain broadcasting add, then a conv reading the plain result
    // (reorder to blocked) with a sigmoid output.
    try b.simple("Add", &.{ "hs", "row" }, "bc");
    try b.conv("bc", "wq", "", "q", 1, 1, 0, 1);
    try b.simple("Sigmoid", &.{"q"}, "out_q");
    // Head 4: scalar pow, max of two blocked tensors.
    try b.simple("Pow", &.{ "hs", "two" }, "pw2");
    try b.simple("Max", &.{ "pw2", "cat" }, "out_max");
    // Head 5: sibling 1x1 convs (merged, then extracted) -> softmax.
    try b.conv("cat", "ws", "bs", "hs1", 1, 1, 0, 1);
    try b.conv("cat", "wt", "bt", "hs2", 1, 1, 0, 1);
    {
        var n = try b.node("Softmax", &.{"hs1"}, "out_s1");
        try n.int("axis", 1);
        try b.add(n);
    }
    {
        var n = try b.node("Softmax", &.{"hs2"}, "out_s2");
        try n.int("axis", 1);
        try b.add(n);
    }
    // Nearest resize of a blocked tensor (head 6).
    {
        var n = try b.node("Resize", &.{ "ad", "", "scales" }, "out_near");
        try n.str("coordinate_transformation_mode", "asymmetric");
        try n.str("nearest_mode", "floor");
        try b.add(n);
    }

    var vi: Writer = .{ .gpa = a };
    try vi.bytes(1, "x");
    try b.g.bytes(11, vi.buf.items);
    for (output_names) |o| {
        var vo: Writer = .{ .gpa = a };
        try vo.bytes(1, o);
        try b.g.bytes(12, vo.buf.items);
    }
    var m: Writer = .{ .gpa = a };
    try m.int(1, 7);
    var op: Writer = .{ .gpa = a };
    try op.int(2, 13);
    try m.bytes(8, op.buf.items);
    try m.bytes(7, b.g.buf.items);
    return m.buf.items;
}

pub const output_names = [_][]const u8{ "out_cat", "out_tr", "out_q", "out_max", "out_s1", "out_s2", "out_near" };
