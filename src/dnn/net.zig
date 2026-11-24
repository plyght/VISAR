//! `Net`: an ONNX model loaded into the IR, compiled for a fixed input shape
//! (shape inference, folding, fusion, layout and memory planning, then the
//! CPU program), and run with no allocation per inference.
//!
//!     var net = try dnn.Net.load(gpa, model_bytes);
//!     defer net.deinit();
//!     try net.plan(&.{&.{ 1, 3, 480, 640 }});
//!     @memcpy(net.input(0), blob);
//!     net.run(exec);
//!     const scores = net.output(0); // .shape, .data

const std = @import("std");
const onnx = @import("onnx.zig");
const ir = @import("ir.zig");
const passes = @import("passes.zig");
const plan_mod = @import("plan.zig");
const cpu = @import("cpu/backend.zig");
const core = @import("../core/core.zig");
const Allocator = std.mem.Allocator;
const Exec = core.Exec;

pub const Error = ir.Error || onnx.Error || error{NotPlanned};

pub const Tensor = struct {
    shape: ir.Shape,
    data: []f32,
};

pub const Options = struct {
    /// Run the fusion passes (off to compare against the unfused graph).
    fuse: bool = true,
    /// Blocked channel layout for the CPU kernels (off: every tensor plain
    /// NCHW, run by the reference kernels).
    blocked: bool = true,
    /// Stream chains of convolutions and pools row by row.
    stream: bool = true,
};

const Compiled = struct {
    arena: std.heap.ArenaAllocator,
    plan: plan_mod.Plan,
    program: cpu.Program,
    buf: []align(64) u8,
};

pub const Net = struct {
    gpa: Allocator,
    arena: std.heap.ArenaAllocator,
    graph: ir.Graph,
    input_dims: []const []const ?usize,
    opts: Options = .{},
    compiled: ?Compiled = null,

    /// Parses an ONNX model. `bytes` must stay valid while the Net lives
    /// (names alias it; weights are copied).
    pub fn load(gpa: Allocator, bytes: []const u8) Error!Net {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        const model = try onnx.parse(a, bytes);
        const imp = try ir.import(a, model);
        return .{ .gpa = gpa, .arena = arena, .graph = imp.graph, .input_dims = imp.input_dims };
    }

    pub fn deinit(net: *Net) void {
        net.unplan();
        net.arena.deinit();
        net.* = undefined;
    }

    fn unplan(net: *Net) void {
        if (net.compiled) |*c| {
            net.gpa.free(c.buf);
            c.arena.deinit();
        }
        net.compiled = null;
    }

    pub fn inputCount(net: *const Net) usize {
        return net.graph.inputs.len;
    }

    pub fn outputCount(net: *const Net) usize {
        return net.graph.outputs.len;
    }

    pub fn outputName(net: *const Net, i: usize) []const u8 {
        return net.graph.values.items[net.graph.outputs[i]].name;
    }

    pub fn outputIndex(net: *const Net, name: []const u8) ?usize {
        for (0..net.outputCount()) |i| if (std.mem.eql(u8, net.outputName(i), name)) return i;
        return null;
    }

    /// The model's declared input dims (null where symbolic).
    pub fn inputDims(net: *const Net, i: usize) []const ?usize {
        return net.input_dims[i];
    }

    /// Compiles the network for these input shapes (one per input),
    /// replacing any previous plan. All buffers are allocated here.
    pub fn plan(net: *Net, shapes: []const []const usize) Error!void {
        net.unplan();
        var c: Compiled = .{ .arena = .init(net.gpa), .plan = undefined, .program = undefined, .buf = &.{} };
        errdefer c.arena.deinit();
        const a = c.arena.allocator();
        if (shapes.len != net.graph.inputs.len) return error.ShapeMismatch;
        const ss = try a.alloc(ir.Shape, shapes.len);
        for (shapes, ss) |s, *o| {
            if (s.len > ir.max_rank) return error.Unsupported;
            o.* = .init(s);
        }
        var g = try net.graph.clone(a);
        try ir.inferShapes(&g, a, ss);
        try passes.run(&g, a, .{ .fuse = net.opts.fuse });
        c.plan = try plan_mod.build(a, &g, if (net.opts.blocked) cpu.block else 0, if (net.opts.stream) &cpu.streamable else null);
        c.program = try cpu.Program.init(a, &c.plan, .{ .stream = net.opts.stream });
        c.buf = try net.gpa.alignedAlloc(u8, .@"64", @max(c.plan.arena_bytes, 64));
        @memset(c.buf, 0);
        net.compiled = c;
    }

    pub fn planned(net: *const Net) ?*const plan_mod.Plan {
        return if (net.compiled) |*c| &c.plan else null;
    }

    fn tensor(net: *Net, t: plan_mod.TensorId) Tensor {
        const c = &net.compiled.?;
        const x = c.plan.tensors[t];
        const off = c.plan.offset(t);
        const p: [*]f32 = @ptrCast(@alignCast(c.buf.ptr + off));
        return .{ .shape = x.shape, .data = p[0..x.shape.count()] };
    }

    /// The `i`th input buffer (plain NCHW), to be filled before `run`.
    pub fn input(net: *Net, i: usize) []f32 {
        return net.tensor(net.compiled.?.plan.inputs[i]).data;
    }

    pub fn run(net: *Net, exec: Exec) void {
        const c = &net.compiled.?;
        c.program.run(exec, c.buf);
    }

    /// The `i`th output (plain row-major), valid until the next `run`/`plan`.
    pub fn output(net: *Net, i: usize) Tensor {
        return net.tensor(net.compiled.?.plan.outputs[i]);
    }
};

// ------------------------------------------------------------------ tests

const testing = std.testing;
const Writer = onnx.Writer;

/// Builds a small ONNX model in memory: x[1,3,9,11] -> Conv3x3 s2 (3->5) ->
/// Relu -> Conv1x1 (5->20) -> depthwise 3x3 -> MaxPool 2x2 -> Resize x2 ->
/// Add(skip) -> Transpose(0,2,3,1) -> Reshape [1,-1,4] -> Sigmoid.
fn buildTestModel(a: Allocator) ![]const u8 {
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    var g: Writer = .{ .gpa = a };
    const Init = struct {
        fn tensor(al: Allocator, w: *Writer, name: []const u8, dims: []const i64, r: std.Random) !void {
            var t: Writer = .{ .gpa = al };
            for (dims) |d| try t.int(1, d);
            try t.int(2, 1);
            try t.bytes(8, name);
            var n: usize = 1;
            for (dims) |d| n *= @intCast(d);
            const vals = try al.alloc(f32, n);
            for (vals) |*v| v.* = r.float(f32) - 0.5;
            try t.bytes(9, std.mem.sliceAsBytes(vals));
            try w.bytes(5, t.buf.items);
        }
        fn ints(al: Allocator, w: *Writer, name: []const u8, vals: []const i64) !void {
            var t: Writer = .{ .gpa = al };
            try t.int(1, @intCast(vals.len));
            try t.int(2, 7);
            try t.bytes(8, name);
            for (vals) |v| try t.int(7, v);
            try w.bytes(5, t.buf.items);
        }
        fn floats(al: Allocator, w: *Writer, name: []const u8, vals: []const f32) !void {
            var t: Writer = .{ .gpa = al };
            try t.int(1, @intCast(vals.len));
            try t.int(2, 1);
            try t.bytes(8, name);
            try t.bytes(9, std.mem.sliceAsBytes(vals));
            try w.bytes(5, t.buf.items);
        }
        fn attrInts(al: Allocator, n: *Writer, name: []const u8, vals: []const i64) !void {
            var t: Writer = .{ .gpa = al };
            try t.bytes(1, name);
            for (vals) |v| try t.int(8, v);
            try t.int(20, 7);
            try n.bytes(5, t.buf.items);
        }
        fn attrInt(al: Allocator, n: *Writer, name: []const u8, v: i64) !void {
            var t: Writer = .{ .gpa = al };
            try t.bytes(1, name);
            try t.int(3, v);
            try t.int(20, 2);
            try n.bytes(5, t.buf.items);
        }
        fn attrStr(al: Allocator, n: *Writer, name: []const u8, v: []const u8) !void {
            var t: Writer = .{ .gpa = al };
            try t.bytes(1, name);
            try t.bytes(4, v);
            try t.int(20, 3);
            try n.bytes(5, t.buf.items);
        }
        fn node(al: Allocator, w: *Writer, op: []const u8, ins: []const []const u8, out: []const u8) !Writer {
            var n: Writer = .{ .gpa = al };
            for (ins) |i| try n.bytes(1, i);
            try n.bytes(2, out);
            try n.bytes(4, op);
            _ = w;
            return n;
        }
    };
    try Init.tensor(a, &g, "w0", &.{ 5, 3, 3, 3 }, rnd);
    try Init.tensor(a, &g, "b0", &.{5}, rnd);
    try Init.tensor(a, &g, "w1", &.{ 20, 5, 1, 1 }, rnd);
    try Init.tensor(a, &g, "b1", &.{20}, rnd);
    try Init.tensor(a, &g, "w2", &.{ 20, 1, 3, 3 }, rnd);
    try Init.tensor(a, &g, "b2", &.{20}, rnd);
    try Init.floats(a, &g, "scales", &.{ 1, 1, 2, 2 });
    try Init.ints(a, &g, "shape", &.{ 1, -1, 4 });
    {
        var n = try Init.node(a, &g, "Conv", &.{ "x", "w0", "b0" }, "c0");
        try Init.attrInts(a, &n, "kernel_shape", &.{ 3, 3 });
        try Init.attrInts(a, &n, "strides", &.{ 2, 2 });
        try Init.attrInts(a, &n, "pads", &.{ 1, 1, 1, 1 });
        try g.bytes(1, n.buf.items);
    }
    {
        const n = try Init.node(a, &g, "Relu", &.{"c0"}, "r0");
        try g.bytes(1, n.buf.items);
    }
    {
        var n = try Init.node(a, &g, "Conv", &.{ "r0", "w1", "b1" }, "c1");
        try Init.attrInts(a, &n, "kernel_shape", &.{ 1, 1 });
        try g.bytes(1, n.buf.items);
    }
    {
        var n = try Init.node(a, &g, "Conv", &.{ "c1", "w2", "b2" }, "c2");
        try Init.attrInts(a, &n, "kernel_shape", &.{ 3, 3 });
        try Init.attrInts(a, &n, "pads", &.{ 1, 1, 1, 1 });
        try Init.attrInt(a, &n, "group", 20);
        try g.bytes(1, n.buf.items);
    }
    {
        var n = try Init.node(a, &g, "MaxPool", &.{"c2"}, "p");
        try Init.attrInts(a, &n, "kernel_shape", &.{ 2, 2 });
        try Init.attrInts(a, &n, "strides", &.{ 2, 2 });
        try g.bytes(1, n.buf.items);
    }
    {
        var n = try Init.node(a, &g, "Resize", &.{ "p", "", "scales" }, "u");
        try Init.attrStr(a, &n, "coordinate_transformation_mode", "asymmetric");
        try Init.attrStr(a, &n, "nearest_mode", "floor");
        try g.bytes(1, n.buf.items);
    }
    {
        const n = try Init.node(a, &g, "Add", &.{ "u", "c2" }, "s");
        try g.bytes(1, n.buf.items);
    }
    {
        var n = try Init.node(a, &g, "Transpose", &.{"s"}, "t");
        try Init.attrInts(a, &n, "perm", &.{ 0, 2, 3, 1 });
        try g.bytes(1, n.buf.items);
    }
    {
        const n = try Init.node(a, &g, "Reshape", &.{ "t", "shape" }, "rs");
        try g.bytes(1, n.buf.items);
    }
    {
        const n = try Init.node(a, &g, "Sigmoid", &.{"rs"}, "y");
        try g.bytes(1, n.buf.items);
    }
    var vi: Writer = .{ .gpa = a };
    try vi.bytes(1, "x");
    try g.bytes(11, vi.buf.items);
    var vo: Writer = .{ .gpa = a };
    try vo.bytes(1, "y");
    try g.bytes(12, vo.buf.items);
    var m: Writer = .{ .gpa = a };
    try m.int(1, 7);
    var op: Writer = .{ .gpa = a };
    try op.int(2, 13);
    try m.bytes(8, op.buf.items);
    try m.bytes(7, g.buf.items);
    return m.buf.items;
}

/// Straightforward NCHW evaluation of the test model for input [1,3,H,W].
fn referenceTestModel(a: Allocator, net: *Net, x: []const f32, H: usize, W: usize) ![]f32 {
    const G = &net.graph;
    const get = struct {
        fn f(gr: *ir.Graph, name: []const u8) []const f32 {
            return gr.values.items[gr.find(name).?].data.?.f32;
        }
    }.f;
    const oh = (H + 2 - 3) / 2 + 1;
    const ow = (W + 2 - 3) / 2 + 1;
    const hw = oh * ow;
    const r0 = try a.alloc(f32, 5 * hw);
    const w0 = get(G, "w0");
    const b0 = get(G, "b0");
    for (0..5) |o| for (0..oh) |y| for (0..ow) |xx| {
        var s: f32 = b0[o];
        for (0..3) |i| for (0..3) |ky| for (0..3) |kx| {
            const iy = @as(isize, @intCast(y * 2 + ky)) - 1;
            const ix = @as(isize, @intCast(xx * 2 + kx)) - 1;
            if (iy < 0 or ix < 0 or iy >= H or ix >= W) continue;
            s += x[(i * H + @as(usize, @intCast(iy))) * W + @as(usize, @intCast(ix))] * w0[((o * 3 + i) * 3 + ky) * 3 + kx];
        };
        r0[(o * oh + y) * ow + xx] = @max(s, 0);
    };
    const c1 = try a.alloc(f32, 20 * hw);
    const w1 = get(G, "w1");
    const b1 = get(G, "b1");
    for (0..20) |o| for (0..hw) |p| {
        var s: f32 = b1[o];
        for (0..5) |i| s += r0[i * hw + p] * w1[o * 5 + i];
        c1[o * hw + p] = s;
    };
    const c2 = try a.alloc(f32, 20 * hw);
    const w2 = get(G, "w2");
    const b2 = get(G, "b2");
    for (0..20) |o| for (0..oh) |y| for (0..ow) |xx| {
        var s: f32 = b2[o];
        for (0..3) |ky| for (0..3) |kx| {
            const iy = @as(isize, @intCast(y + ky)) - 1;
            const ix = @as(isize, @intCast(xx + kx)) - 1;
            if (iy < 0 or ix < 0 or iy >= oh or ix >= ow) continue;
            s += c1[(o * oh + @as(usize, @intCast(iy))) * ow + @as(usize, @intCast(ix))] * w2[o * 9 + ky * 3 + kx];
        };
        c2[(o * oh + y) * ow + xx] = s;
    };
    // MaxPool 2x2 then nearest x2 (floor, asymmetric) then + c2, NHWC, sigmoid.
    const out = try a.alloc(f32, 20 * hw);
    for (0..oh) |y| for (0..ow) |xx| for (0..20) |o| {
        const py = y / 2;
        const px = xx / 2;
        var m = -std.math.inf(f32);
        for (0..2) |dy| for (0..2) |dx| {
            m = @max(m, c2[(o * oh + py * 2 + dy) * ow + px * 2 + dx]);
        };
        const s = m + c2[(o * oh + y) * ow + xx];
        out[(y * ow + xx) * 20 + o] = 1.0 / (1.0 + @exp(-s));
    };
    return out;
}

test "net runs a small model and matches a direct evaluation" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const bytes = try buildTestModel(a);
    var net = try Net.load(testing.allocator, bytes);
    defer net.deinit();
    var pool = core.Pool.init(testing.allocator, testing.io, 3);
    defer pool.deinit();
    // 9x11 gives 5x6 after the first conv but a 4x6 upsampled pool: the
    // Add cannot broadcast and planning fails cleanly.
    try testing.expectError(error.ShapeMismatch, net.plan(&.{&.{ 1, 3, 9, 11 }}));
    var prng = std.Random.DefaultPrng.init(7);
    for ([_][2]usize{ .{ 16, 20 }, .{ 32, 44 }, .{ 4, 4 } }) |hw| {
        for ([_]Options{ .{}, .{ .stream = false }, .{ .fuse = false }, .{ .blocked = false } }) |opts| {
            for ([_]Exec{ .serial, .on(&pool) }) |exec| {
                net.opts = opts;
                try net.plan(&.{&.{ 1, 3, hw[0], hw[1] }});
                const x = net.input(0);
                for (x) |*v| v.* = prng.random().float(f32) * 4 - 2;
                const want = try referenceTestModel(a, &net, x, hw[0], hw[1]);
                net.run(exec);
                const y = net.output(0);
                try testing.expectEqual(@as(usize, 3), y.shape.rank);
                try testing.expectEqual(want.len, y.data.len);
                for (want, y.data) |w, v| try testing.expectApproxEqAbs(w, v, 1e-5);
            }
        }
    }
}
