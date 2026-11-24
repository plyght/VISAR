//! The execution plan: the optimised graph for one input shape, lowered to a
//! list of steps over tensors whose shapes, layouts and arena offsets are
//! all fixed. This is plain data and device-neutral: a backend (the CPU
//! executor in `cpu/`, later Metal or Core ML) allocates one buffer of
//! `arena_bytes`, places every intermediate at its offset, prepares each
//! step's kernel (packing weights its own way) and runs the steps in order.
//!
//! Layouts: `plain` is the ONNX row-major layout of the logical shape;
//! `blocked` (rank 4 only) stores NCHW as [N][ceil(C/B)][H][W][B] with B =
//! `Plan.block` channels innermost, the padding channels zero. B = 0 plans
//! everything plain.

const std = @import("std");
const ir = @import("ir.zig");
const Allocator = std.mem.Allocator;
const Shape = ir.Shape;

pub const Layout = enum { plain, blocked };

pub const TensorId = u32;
pub const none: TensorId = std.math.maxInt(TensorId);

pub const Tensor = struct {
    name: []const u8,
    shape: Shape,
    layout: Layout,
    /// Constant data (plain layout); such tensors have no storage.
    data: ?[]const f32 = null,
    /// Index into `Plan.storages` (views share their base's storage).
    storage: u32 = std.math.maxInt(u32),
    /// Byte offset within the storage (a view of whole channel blocks of
    /// a blocked tensor, or a concatenation input written in place).
    view_offset: usize = 0,

    /// Elements stored, including blocked padding channels.
    pub fn storedCount(t: Tensor, block: usize) usize {
        if (t.layout == .plain) return t.shape.count();
        const s = t.shape.d;
        return s[0] * ceilDiv(s[1], block) * block * s[2] * s[3];
    }
};

pub const Storage = struct {
    bytes: usize,
    /// Byte offset in the arena (assigned by the memory planner).
    offset: usize = 0,
    /// First and last step touching it (inputs: 0; outputs: steps.len).
    first: u32 = std.math.maxInt(u32),
    last: u32 = 0,
};

pub const Op = union(enum) {
    /// inputs: x, w (constant, OIHW), b (constant or none), residual (or none)
    conv: ir.Conv,
    /// inputs: x, scale, shift (constants, per channel): y = x * scale + shift
    affine,
    act: ir.Act,
    /// numpy broadcasting; inputs may be constants
    binary: ir.Binary,
    max_pool: ir.Pool,
    average_pool: ir.Pool,
    global_average_pool,
    resize: ir.Resize,
    concat: usize,
    /// Plain tensors only.
    transpose: [ir.max_rank]u8,
    /// Blocked NCHW input -> plain NHWC output of channels [c0, c1), with
    /// an activation applied on the way (Transpose(0, 2, 3, 1) of a channel
    /// slice: detection-head outputs).
    to_nhwc: Extract,
    /// Blocked input -> plain NCHW of channels [c0, c1) with an activation.
    extract: Extract,
    softmax: usize,
    /// Rank-4 inputs (all in the output's layout) -> output channel k is
    /// channel `map[k].channel` of input `map[k].input`.
    gather: Gather,
    /// Plain tensors only: a strided window of the input.
    slice: ir.Strided,
    /// Copies between layouts (the input and output tensors say which).
    reorder,
};

pub const Group = struct { first: u32, len: u32 };

/// Decides which steps a backend can stream: `first` is true for a step
/// that would start a group (it reads a whole tensor).
pub const Streamable = *const fn (p: *const Plan, step: usize, first: bool) bool;

pub const max_group = 24;

pub const Extract = struct { c0: usize, c1: usize, act: ir.Act = .none };

pub const Gather = struct { map: []const ir.ChannelSrc, act: ir.Act = .none };

pub const Step = struct {
    name: []const u8,
    op: Op,
    inputs: []const TensorId,
    outputs: []const TensorId,
};

pub const Plan = struct {
    block: usize,
    tensors: []Tensor,
    storages: []Storage,
    steps: []Step,
    inputs: []const TensorId,
    outputs: []const TensorId,
    output_names: []const []const u8,
    /// Runs of consecutive steps a backend executes together, streaming
    /// rows from one to the next: tensors inside a group get no storage,
    /// and the group's inputs and outputs stay live across all of it.
    groups: []const Group = &.{},
    arena_bytes: usize,

    /// Readers of tensor `t` (steps and plan outputs).
    pub fn readers(p: *const Plan, t: TensorId) usize {
        var n: usize = 0;
        for (p.steps) |s| {
            for (s.inputs) |i| {
                if (i == t) n += 1;
            }
        }
        for (p.outputs) |o| {
            if (o == t) n += 1;
        }
        return n;
    }

    pub fn offset(p: *const Plan, t: TensorId) usize {
        return p.storages[p.tensors[t].storage].offset + p.tensors[t].view_offset;
    }

    /// Whether another tensor shares `t`'s storage (a view of it, or it of
    /// another).
    pub fn shared(p: *const Plan, t: TensorId) bool {
        const st = p.tensors[t].storage;
        for (p.tensors, 0..) |x, i| {
            if (i != t and x.data == null and x.storage == st) return true;
        }
        return false;
    }

    pub fn dump(p: *const Plan, w: *std.Io.Writer) std.Io.Writer.Error!void {
        for (p.steps, 0..) |s, i| {
            try w.print("{d:>3} {s:<18}", .{ i, @tagName(s.op) });
            for (s.inputs) |t| {
                if (t == none) continue;
                const x = p.tensors[t];
                try w.print(" {f}{s}", .{ x.shape, if (x.data != null) "c" else if (x.layout == .blocked) "b" else "" });
            }
            try w.writeAll(" ->");
            for (s.outputs) |t| try w.print(" {f}{s}@{d}", .{ p.tensors[t].shape, if (p.tensors[t].layout == .blocked) "b" else "", p.offset(t) });
            if (s.op == .conv) {
                const c = s.op.conv;
                try w.print("  k{d}x{d} s{d} g{d} {s}{s}", .{ c.kernel[0], c.kernel[1], c.stride[0], c.group, @tagName(c.act), if (c.residual) " +res" else "" });
            }
            try w.writeAll("\n");
        }
        try w.print("arena {d} bytes\n", .{p.arena_bytes});
    }
};

pub fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

const Builder = struct {
    arena: Allocator,
    g: *ir.Graph,
    block: usize,
    tensors: std.ArrayList(Tensor) = .empty,
    storages: std.ArrayList(Storage) = .empty,
    steps: std.ArrayList(Step) = .empty,
    /// Per value: tensor holding it in each layout (if produced).
    plain: []TensorId,
    blocked: []TensorId,
    /// Values not materialised yet: a channel range of a blocked tensor
    /// with an activation, made concrete by whichever consumer reads them.
    lazy: []?Lazy,
    /// Storages of the graph inputs (never aliased).
    inputs: []const u32 = &.{},

    const Lazy = struct { base: TensorId, e: Extract };

    fn newTensor(b: *Builder, name: []const u8, shape: Shape, layout: Layout) Allocator.Error!TensorId {
        try b.storages.append(b.arena, .{ .bytes = 0 });
        try b.tensors.append(b.arena, .{ .name = name, .shape = shape, .layout = layout, .storage = @intCast(b.storages.items.len - 1) });
        return @intCast(b.tensors.items.len - 1);
    }

    fn view(b: *Builder, base: TensorId, name: []const u8, shape: Shape) Allocator.Error!TensorId {
        try b.tensors.append(b.arena, .{ .name = name, .shape = shape, .layout = .plain, .storage = b.tensors.items[base].storage });
        return @intCast(b.tensors.items.len - 1);
    }

    fn constTensor(b: *Builder, v: ir.ValueId) ir.Error!TensorId {
        if (b.plain[v] != none) return b.plain[v];
        const val = b.g.value(v);
        const data: []const f32 = switch (val.data.?) {
            .f32 => |x| x,
            .i64 => |x| blk: {
                const d = try b.arena.alloc(f32, x.len);
                for (d, x) |*o, a| o.* = @floatFromInt(a);
                break :blk d;
            },
        };
        try b.tensors.append(b.arena, .{ .name = val.name, .shape = val.shape, .layout = .plain, .data = data });
        b.plain[v] = @intCast(b.tensors.items.len - 1);
        return b.plain[v];
    }

    fn step(b: *Builder, name: []const u8, op: Op, ins: []const TensorId, outs: []const TensorId) Allocator.Error!void {
        try b.steps.append(b.arena, .{ .name = name, .op = op, .inputs = try b.arena.dupe(TensorId, ins), .outputs = try b.arena.dupe(TensorId, outs) });
    }

    /// The tensor holding value `v` in `layout`, inserting a reorder if it
    /// only exists in the other layout.
    fn get(b: *Builder, v: ir.ValueId, layout: Layout) ir.Error!TensorId {
        const val = b.g.value(v);
        if (val.data != null) return b.constTensor(v);
        if (val.dtype != .f32) return error.Unsupported;
        if (b.lazy[v]) |lz| blk: {
            if (layout != .blocked or val.shape.rank != 4 or b.block == 0) break :blk;
            // A blocked channel range: a view when it is whole blocks of a
            // single image, else a gather (which applies the activation).
            const base = b.tensors.items[lz.base];
            const bs = base.shape;
            const B = b.block;
            const t = if (lz.e.act == .none and bs.d[0] == 1 and lz.e.c0 % B == 0 and (lz.e.c1 % B == 0 or lz.e.c1 == bs.d[1])) v: {
                try b.tensors.append(b.arena, .{ .name = val.name, .shape = val.shape, .layout = .blocked, .storage = base.storage, .view_offset = base.view_offset + (lz.e.c0 / B) * bs.d[2] * bs.d[3] * B * @sizeOf(f32) });
                break :v @as(TensorId, @intCast(b.tensors.items.len - 1));
            } else v: {
                const t = try b.newTensor(val.name, val.shape, .blocked);
                const map = try b.arena.alloc(ir.ChannelSrc, lz.e.c1 - lz.e.c0);
                for (map, lz.e.c0..) |*m, c| m.* = .{ .input = 0, .channel = @intCast(c) };
                try b.step("slice", .{ .gather = .{ .map = map, .act = lz.e.act } }, &.{lz.base}, &.{t});
                break :v t;
            };
            b.blocked[v] = t;
            b.lazy[v] = null;
            return t;
        }
        if (b.lazy[v]) |lz| {
            const t = try b.newTensor(val.name, val.shape, .plain);
            try b.step("extract", .{ .extract = lz.e }, &.{lz.base}, &.{t});
            b.plain[v] = t;
            b.lazy[v] = null;
        }
        const want = if (layout == .blocked and (val.shape.rank != 4 or b.block == 0)) Layout.plain else layout;
        const have = if (want == .plain) b.plain[v] else b.blocked[v];
        if (have != none) return have;
        const other = if (want == .plain) b.blocked[v] else b.plain[v];
        if (other == none) return error.InvalidModel;
        const t = try b.newTensor(val.name, val.shape, want);
        try b.step("reorder", .reorder, &.{other}, &.{t});
        if (want == .plain) b.plain[v] = t else b.blocked[v] = t;
        return t;
    }

    /// The layout `v` currently has (blocked preferred).
    fn layoutOf(b: *Builder, v: ir.ValueId) Layout {
        return if (b.blocked[v] != none or b.lazy[v] != null) .blocked else .plain;
    }

    fn define(b: *Builder, v: ir.ValueId, layout: Layout) Allocator.Error!TensorId {
        const val = b.g.value(v);
        const l = if (layout == .blocked and (val.shape.rank != 4 or b.block == 0)) Layout.plain else layout;
        const t = try b.newTensor(val.name, val.shape, l);
        if (l == .plain) b.plain[v] = t else b.blocked[v] = t;
        return t;
    }

    /// Makes the inputs of a blocked single-image channel concatenation
    /// views into its output, so their producers write it in place and no
    /// copy runs. Only inputs owning their storage alone qualify (not
    /// graph inputs, views, or values already placed in another concat).
    fn concatInPlace(b: *Builder, ins: []const TensorId, out: TensorId) bool {
        const T = b.tensors.items;
        for (ins, 0..) |t, k| {
            if (T[t].data != null or T[t].layout != .blocked or T[t].view_offset != 0) return false;
            for (b.inputs) |gi| {
                if (gi == T[t].storage) return false;
            }
            for (T, 0..) |x, i| {
                if (i != t and x.data == null and x.storage == T[t].storage) return false;
            }
            for (ins[0..k]) |u| {
                if (T[u].storage == T[t].storage) return false;
            }
        }
        const os = T[out].shape;
        var at: usize = 0;
        for (ins) |t| {
            T[t].storage = T[out].storage;
            T[t].view_offset = at * os.d[2] * os.d[3] * @sizeOf(f32);
            at += ceilDiv(T[t].shape.d[1], b.block) * b.block;
        }
        return true;
    }

    fn isConst(b: *Builder, v: ir.ValueId) bool {
        return v != ir.none and b.g.value(v).data != null;
    }

    /// A constant that broadcasts per channel (or as a scalar) against a
    /// rank-4 NCHW tensor: shape [C], [C,1,1], [1,C,1,1] or one element.
    fn perChannelConst(b: *Builder, v: ir.ValueId, c: usize) bool {
        const val = b.g.value(v);
        if (val.data == null) return false;
        const s = val.shape;
        if (s.count() == 1) return true;
        if (s.count() != c) return false;
        const d = s.dims();
        return switch (d.len) {
            3 => d[0] == c,
            4 => d[0] == 1 and d[1] == c,
            else => false,
        };
    }

    fn node(b: *Builder, n: ir.Node) ir.Error!void {
        const g = b.g;
        const out_v = n.outputs[0];
        const x = n.inputs[0];
        const xs = g.value(x).shape;
        switch (n.op) {
            .conv => |c| {
                const in = try b.get(x, if (g.value(x).shape.d[1] < b.block and b.layoutOf(x) == .plain) .plain else .blocked);
                const w = try b.constTensor(n.inputs[1]);
                const bias = if (n.inputs.len > 2 and n.inputs[2] != ir.none) try b.constTensor(n.inputs[2]) else none;
                const res = if (c.residual) try b.get(n.inputs[3], .blocked) else none;
                const out = try b.define(out_v, .blocked);
                try b.step(n.name, .{ .conv = c }, &.{ in, w, bias, res }, &.{out});
            },
            .batch_norm => |eps| {
                const ch = xs.d[1];
                const sc = try b.arena.alloc(f32, ch);
                const sh = try b.arena.alloc(f32, ch);
                const p = [4][]const f32{ g.value(n.inputs[1]).data.?.f32, g.value(n.inputs[2]).data.?.f32, g.value(n.inputs[3]).data.?.f32, g.value(n.inputs[4]).data.?.f32 };
                for (0..ch) |i| {
                    sc[i] = p[0][i] / @sqrt(p[3][i] + eps);
                    sh[i] = p[1][i] - p[2][i] * sc[i];
                }
                try b.tensors.append(b.arena, .{ .name = "", .shape = .init(&.{ch}), .layout = .plain, .data = sc });
                const ts: TensorId = @intCast(b.tensors.items.len - 1);
                try b.tensors.append(b.arena, .{ .name = "", .shape = .init(&.{ch}), .layout = .plain, .data = sh });
                const tb: TensorId = @intCast(b.tensors.items.len - 1);
                const l = b.layoutOf(x);
                const in = try b.get(x, l);
                const out = try b.define(out_v, l);
                try b.step(n.name, .affine, &.{ in, ts, tb }, &.{out});
            },
            .act => |a| {
                if (b.lazy[x]) |lz| {
                    if (lz.e.act == .none and b.g.consumers(x) == 1) {
                        var e = lz.e;
                        e.act = a;
                        b.lazy[out_v] = .{ .base = lz.base, .e = e };
                        return;
                    }
                }
                const l = b.layoutOf(x);
                const in = try b.get(x, l);
                const out = try b.define(out_v, l);
                try b.step(n.name, .{ .act = a }, &.{in}, &.{out});
            },
            .binary => |op| {
                const y = n.inputs[1];
                const ys = g.value(y).shape;
                const os = g.value(out_v).shape;
                var l: Layout = .plain;
                if (os.rank == 4 and b.block > 0) {
                    const xc = b.isConst(x);
                    const yc = b.isConst(y);
                    if (!xc and !yc and xs.eql(os) and ys.eql(os) and (b.layoutOf(x) == .blocked or b.layoutOf(y) == .blocked)) l = .blocked;
                    if (!xc and yc and xs.eql(os) and b.perChannelConst(y, os.d[1]) and b.layoutOf(x) == .blocked) l = .blocked;
                    if (xc and !yc and ys.eql(os) and b.perChannelConst(x, os.d[1]) and b.layoutOf(y) == .blocked) l = .blocked;
                }
                const a_t = if (b.isConst(x)) try b.constTensor(x) else try b.get(x, l);
                const b_t = if (b.isConst(y)) try b.constTensor(y) else try b.get(y, l);
                const out = try b.define(out_v, l);
                try b.step(n.name, .{ .binary = op }, &.{ a_t, b_t }, &.{out});
            },
            .max_pool, .average_pool, .global_average_pool, .resize => {
                const in = try b.get(x, .blocked);
                const out = try b.define(out_v, .blocked);
                const op: Op = switch (n.op) {
                    .max_pool => |p| .{ .max_pool = p },
                    .average_pool => |p| .{ .average_pool = p },
                    .global_average_pool => .global_average_pool,
                    .resize => |r| .{ .resize = r },
                    else => unreachable,
                };
                try b.step(n.name, op, &.{in}, &.{out});
            },
            .concat => |axis| {
                const os = g.value(out_v).shape;
                const a: usize = @intCast(if (axis < 0) axis + @as(i64, os.rank) else axis);
                var l: Layout = .plain;
                if (os.rank == 4 and a == 1 and b.block > 0) {
                    l = .blocked;
                    for (n.inputs, 0..) |i, k| {
                        if (b.isConst(i) or b.layoutOf(i) != .blocked) l = .plain;
                        if (k + 1 < n.inputs.len and g.value(i).shape.d[1] % b.block != 0) l = .plain;
                    }
                }
                var ins: std.ArrayList(TensorId) = .empty;
                for (n.inputs) |i| try ins.append(b.arena, if (b.isConst(i)) try b.constTensor(i) else try b.get(i, l));
                const out = try b.define(out_v, l);
                if (l == .blocked and os.d[0] == 1 and b.concatInPlace(ins.items, out)) return;
                try b.step(n.name, .{ .concat = a }, ins.items, &.{out});
            },
            .transpose => |perm| {
                const nhwc = xs.rank == 4 and perm[0] == 0 and perm[1] == 2 and perm[2] == 3 and perm[3] == 1;
                if (nhwc and b.layoutOf(x) == .blocked) {
                    const src: Lazy = b.lazy[x] orelse .{ .base = b.blocked[x], .e = .{ .c0 = 0, .c1 = xs.d[1] } };
                    const out = try b.define(out_v, .plain);
                    try b.step(n.name, .{ .to_nhwc = src.e }, &.{src.base}, &.{out});
                    return;
                }
                const in = try b.get(x, .plain);
                const out = try b.define(out_v, .plain);
                try b.step(n.name, .{ .transpose = perm }, &.{in}, &.{out});
            },
            .channel_slice => |r| {
                if (b.layoutOf(x) == .blocked) {
                    if (b.lazy[x]) |lz| {
                        if (lz.e.act == .none) {
                            b.lazy[out_v] = .{ .base = lz.base, .e = .{ .c0 = lz.e.c0 + r[0], .c1 = lz.e.c0 + r[1] } };
                            return;
                        }
                    } else {
                        b.lazy[out_v] = .{ .base = b.blocked[x], .e = .{ .c0 = r[0], .c1 = r[1] } };
                        return;
                    }
                }
                if (b.layoutOf(x) == .plain) {
                    // Plain plans: a channel range is a plain gather.
                    const in = try b.get(x, .plain);
                    const out = try b.define(out_v, .plain);
                    const map = try b.arena.alloc(ir.ChannelSrc, r[1] - r[0]);
                    for (map, r[0]..) |*m, c| m.* = .{ .input = 0, .channel = @intCast(c) };
                    try b.step(n.name, .{ .gather = .{ .map = map } }, &.{in}, &.{out});
                    return;
                }
                const in = try b.get(x, .blocked);
                const out = try b.define(out_v, .plain);
                try b.step(n.name, .{ .extract = .{ .c0 = r[0], .c1 = r[1] } }, &.{in}, &.{out});
            },
            .softmax => |axis| {
                const in = try b.get(x, .plain);
                const out = try b.define(out_v, .plain);
                try b.step(n.name, .{ .softmax = try normAxis(axis, xs.rank) }, &.{in}, &.{out});
            },
            .channel_gather => |map| {
                var l: Layout = if (b.block > 0) .blocked else .plain;
                for (n.inputs) |i| {
                    if (b.isConst(i)) l = .plain;
                }
                var ins: std.ArrayList(TensorId) = .empty;
                for (n.inputs) |i| try ins.append(b.arena, if (b.isConst(i)) try b.constTensor(i) else try b.get(i, l));
                const out = try b.define(out_v, l);
                try b.step(n.name, .{ .gather = .{ .map = map } }, ins.items, &.{out});
            },
            .strided_slice => |st| {
                const in = if (b.isConst(x)) try b.constTensor(x) else try b.get(x, .plain);
                const out = try b.define(out_v, .plain);
                try b.step(n.name, .{ .slice = st }, &.{in}, &.{out});
            },
            .reshape, .flatten, .squeeze, .unsqueeze, .identity => {
                // Views of the same plain bytes.
                const in = try b.get(x, .plain);
                const t = try b.view(in, g.value(out_v).name, g.value(out_v).shape);
                b.plain[out_v] = t;
            },
            .shape, .gather, .cast, .slice, .split => return error.Unsupported, // only constant-foldable
        }
    }
};

fn normAxis(axis: i64, rank: usize) ir.Error!usize {
    const r: i64 = @intCast(rank);
    const a = if (axis < 0) axis + r else axis;
    if (a < 0 or a >= r) return error.InvalidModel;
    return @intCast(a);
}

/// Lowers an optimised, shape-inferred graph to a plan with blocked layout
/// of `block` channels (0 = plain only) and assigns arena offsets.
pub fn build(arena: Allocator, g: *ir.Graph, block: usize, streamable: ?Streamable) ir.Error!Plan {
    const nv = g.values.items.len;
    var b: Builder = .{
        .arena = arena,
        .g = g,
        .block = block,
        .plain = try arena.alloc(TensorId, nv),
        .blocked = try arena.alloc(TensorId, nv),
        .lazy = try arena.alloc(?Builder.Lazy, nv),
    };
    @memset(b.lazy, null);
    @memset(b.plain, none);
    @memset(b.blocked, none);
    const inputs = try arena.alloc(TensorId, g.inputs.len);
    for (g.inputs, inputs) |v, *t| t.* = try b.define(v, .plain);
    const in_st = try arena.alloc(u32, inputs.len);
    for (inputs, in_st) |t, *s| s.* = b.tensors.items[t].storage;
    b.inputs = in_st;
    for (g.nodes.items) |n| {
        if (n.dead) continue;
        try b.node(n);
    }
    const outputs = try arena.alloc(TensorId, g.outputs.len);
    const names = try arena.alloc([]const u8, g.outputs.len);
    for (g.outputs, outputs, names) |v, *t, *nm| {
        nm.* = g.value(v).name;
        if (g.value(v).data != null) {
            // A constant output still gets arena storage, filled by a copy.
            const c = try b.constTensor(v);
            const o = try b.newTensor(nm.*, g.value(v).shape, .plain);
            try b.step("const", .reorder, &.{c}, &.{o});
            t.* = o;
        } else t.* = try b.get(v, .plain);
    }
    var p: Plan = .{
        .block = block,
        .tensors = b.tensors.items,
        .storages = b.storages.items,
        .steps = b.steps.items,
        .inputs = inputs,
        .outputs = outputs,
        .output_names = names,
        .arena_bytes = 0,
    };
    if (streamable) |f| p.groups = try findGroups(&p, arena, f);
    try assignMemory(&p, arena);
    return p;
}

/// Maximal runs where each step reads only the previous step's output and
/// nothing else reads it. A pooling step ends a run, so the rows a band of
/// output recomputes at its top stay few.
fn findGroups(p: *const Plan, arena: Allocator, f: Streamable) Allocator.Error![]const Group {
    var out: std.ArrayList(Group) = .empty;
    var i: usize = 0;
    while (i < p.steps.len) {
        var j = i + 1;
        if (f(p, i, true)) {
            while (j < p.steps.len and j - i < max_group) : (j += 1) {
                const prev = p.steps[j - 1];
                if (prev.op == .max_pool or prev.op == .average_pool) break;
                if (!f(p, j, false)) break;
                const t = prev.outputs[0];
                // A tensor sharing its storage (a view, or viewed) is
                // always materialised.
                if (p.steps[j].inputs[0] != t or p.readers(t) != 1 or p.shared(t)) break;
                // Other inputs (residuals) must come from outside the run.
                var inside = false;
                for (p.steps[j].inputs[1..]) |x| {
                    for (p.steps[i..j]) |s| {
                        if (x != none and s.outputs[0] == x) inside = true;
                    }
                }
                if (inside) break;
            }
        }
        if (j - i >= 2) try out.append(arena, .{ .first = @intCast(i), .len = @intCast(j - i) });
        i = j;
    }
    return out.items;
}

/// Liveness-based placement: each storage lives from its first to its last
/// step; storages are placed largest first at the lowest offset that does
/// not overlap any already-placed storage whose lifetime intersects.
fn assignMemory(p: *Plan, arena: Allocator) Allocator.Error!void {
    const align_bytes = 64;
    for (p.tensors) |t| {
        if (t.data != null) continue;
        const s = &p.storages[t.storage];
        s.bytes = @max(s.bytes, std.mem.alignForward(usize, t.view_offset + t.storedCount(p.block) * @sizeOf(f32), align_bytes));
    }
    // Steps of a group count as one long step; tensors passed from one
    // member to the next never exist.
    const span = try arena.alloc([2]u32, p.steps.len);
    for (span, 0..) |*sp, i| sp.* = .{ @intCast(i), @intCast(i) };
    for (p.groups) |gr| {
        for (gr.first..gr.first + gr.len) |i| span[i] = .{ gr.first, gr.first + gr.len - 1 };
        for (p.steps[gr.first .. gr.first + gr.len - 1]) |s| p.storages[p.tensors[s.outputs[0]].storage].bytes = 0;
    }
    for (p.steps, 0..) |s, i| {
        for ([2][]const TensorId{ s.inputs, s.outputs }) |list| for (list) |t| {
            if (t == none or p.tensors[t].data != null) continue;
            const st = &p.storages[p.tensors[t].storage];
            st.first = @min(st.first, span[i][0]);
            st.last = @max(st.last, span[i][1]);
        };
    }
    const end: u32 = @intCast(p.steps.len);
    for (p.inputs) |t| p.storages[p.tensors[t].storage].first = 0;
    for (p.outputs) |t| {
        const st = &p.storages[p.tensors[t].storage];
        st.last = end;
        if (st.first == std.math.maxInt(u32)) st.first = 0;
    }
    const order = try arena.alloc(u32, p.storages.len);
    for (order, 0..) |*o, i| o.* = @intCast(i);
    const S = struct {
        fn lt(st: []Storage, a: u32, c: u32) bool {
            if (st[a].bytes != st[c].bytes) return st[a].bytes > st[c].bytes;
            return a < c;
        }
    };
    std.sort.pdq(u32, order, p.storages, S.lt);
    var placed: std.ArrayList(u32) = .empty;
    var total: usize = 0;
    for (order) |si| {
        const s = &p.storages[si];
        if (s.bytes == 0 or s.first == std.math.maxInt(u32)) continue;
        // Candidate offsets: 0 and the end of every overlapping placed block.
        var best: usize = std.math.maxInt(usize);
        var cands: std.ArrayList(usize) = .empty;
        try cands.append(arena, 0);
        for (placed.items) |pi| {
            const q = p.storages[pi];
            if (q.last < s.first or s.last < q.first) continue;
            try cands.append(arena, q.offset + q.bytes);
        }
        for (cands.items) |off| {
            if (off >= best) continue;
            var ok = true;
            for (placed.items) |pi| {
                const q = p.storages[pi];
                if (q.last < s.first or s.last < q.first) continue;
                if (off < q.offset + q.bytes and q.offset < off + s.bytes) {
                    ok = false;
                    break;
                }
            }
            if (ok) best = off;
        }
        s.offset = best;
        total = @max(total, best + s.bytes);
        try placed.append(arena, si);
    }
    p.arena_bytes = total;
}
