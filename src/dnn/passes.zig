//! Graph rewrites run on a plan's copy of the IR after shape inference:
//! identity removal, BatchNorm folding into convolutions, epilogue fusion
//! (activation and residual add into the producing conv) and dead-code
//! elimination. Each pass is device-neutral: it changes the math the graph
//! describes into fewer, larger ops, never how they run.

const std = @import("std");
const ir = @import("ir.zig");
const Allocator = std.mem.Allocator;
const Graph = ir.Graph;
const ValueId = ir.ValueId;

pub const Options = struct {
    fuse: bool = true,
};

pub fn run(g: *Graph, arena: Allocator, opts: Options) ir.Error!void {
    markConstNodesDead(g);
    removeIdentities(g);
    if (opts.fuse) {
        try foldBatchNorm(g, arena);
        try mergeSiblingConvs(g, arena);
        try mergeSlicedDepthwise(g, arena);
        try pushActsUp(g, arena);
        try fuseEpilogues(g, arena);
    }
    eliminateDead(g);
}

/// Nodes whose outputs were folded to constants no longer need to run.
fn markConstNodesDead(g: *Graph) void {
    for (g.nodes.items) |*n| {
        if (n.dead) continue;
        if (g.value(n.outputs[0]).data != null) n.dead = true;
    }
}

fn removeIdentities(g: *Graph) void {
    for (g.nodes.items) |*n| {
        if (n.dead or n.op != .identity) continue;
        const from = n.outputs[0];
        const to = n.inputs[0];
        // Keep a graph output's name: alias only internal values.
        if (g.isOutput(from) and (g.isOutput(to) or isGraphInput(g, to))) continue;
        n.dead = true;
        g.replaceUses(from, to);
    }
}

fn isGraphInput(g: *const Graph, id: ValueId) bool {
    return std.mem.indexOfScalar(ValueId, g.inputs, id) != null;
}

/// The single live consumer of `id`, if it has exactly one and is not a
/// graph output.
fn soleConsumer(g: *Graph, id: ValueId) ?*ir.Node {
    if (g.isOutput(id)) return null;
    var found: ?*ir.Node = null;
    for (g.nodes.items) |*n| {
        if (n.dead) continue;
        for (n.inputs) |i| {
            if (i != id) continue;
            if (found != null) return null;
            found = n;
        }
    }
    return found;
}

/// conv -> BatchNormalization becomes one conv with scaled weights and a
/// shifted bias.
fn foldBatchNorm(g: *Graph, arena: Allocator) ir.Error!void {
    for (g.nodes.items) |*n| {
        if (n.dead or n.op != .conv) continue;
        if (n.op.conv.act != .none or n.op.conv.residual) continue;
        const bn = soleConsumer(g, n.outputs[0]) orelse continue;
        if (bn.op != .batch_norm or bn.inputs[0] != n.outputs[0]) continue;
        const eps = bn.op.batch_norm;
        const vals = [4]?ir.Const{ g.value(bn.inputs[1]).data, g.value(bn.inputs[2]).data, g.value(bn.inputs[3]).data, g.value(bn.inputs[4]).data };
        var p: [4][]const f32 = undefined;
        for (vals, 0..) |v, i| p[i] = switch (v orelse continue) {
            .f32 => |x| x,
            else => continue,
        };
        const wv = g.value(n.inputs[1]);
        const w = switch (wv.data orelse continue) {
            .f32 => |x| x,
            else => continue,
        };
        const co = wv.shape.d[0];
        const per = w.len / co;
        const nw = try arena.alloc(f32, w.len);
        const nb = try arena.alloc(f32, co);
        const old_b: ?[]const f32 = if (n.inputs.len > 2 and n.inputs[2] != ir.none) g.value(n.inputs[2]).data.?.f32 else null;
        for (0..co) |o| {
            const s = p[0][o] / @sqrt(p[3][o] + eps);
            for (0..per) |k| nw[o * per + k] = w[o * per + k] * s;
            const b0 = if (old_b) |b| b[o] else 0;
            nb[o] = (b0 - p[2][o]) * s + p[1][o];
        }
        const w_id = try g.addValue(arena, .{ .name = wv.name, .shape = wv.shape, .shaped = true, .data = .{ .f32 = nw } });
        const b_id = try g.addValue(arena, .{ .name = "", .shape = .init(&.{co}), .shaped = true, .data = .{ .f32 = nb } });
        const ins = try arena.alloc(ValueId, 3);
        ins[0] = n.inputs[0];
        ins[1] = w_id;
        ins[2] = b_id;
        n.inputs = ins;
        n.outputs[0] = bn.outputs[0];
        bn.dead = true;
    }
}

/// conv -> act, conv -> add(residual) [-> act] fused into the conv's epilogue.
fn fuseEpilogues(g: *Graph, arena: Allocator) Allocator.Error!void {
    for (g.nodes.items) |*n| {
        if (n.dead or n.op != .conv) continue;
        while (true) {
            const c = &n.op.conv;
            const next = soleConsumer(g, n.outputs[0]) orelse break;
            if (next.op == .act and c.act == .none and !c.residual) {
                c.act = next.op.act;
                n.outputs[0] = next.outputs[0];
                next.dead = true;
                continue;
            }
            if (next.op == .act and c.act == .none and c.residual and !c.residual_after_act) {
                c.act = next.op.act;
                n.outputs[0] = next.outputs[0];
                next.dead = true;
                continue;
            }
            if (next.op == .binary and next.op.binary == .add and !c.residual) {
                const other = if (next.inputs[0] == n.outputs[0]) next.inputs[1] else next.inputs[0];
                if (other == n.outputs[0]) break;
                const ov = g.value(other);
                if (ov.data != null or !ov.shape.eql(g.value(n.outputs[0]).shape)) break;
                // The residual must be computed before this conv runs: it
                // is, if its producer comes earlier in the node order.
                if (!producedBefore(g, other, n)) break;
                c.residual = true;
                c.residual_after_act = c.act != .none;
                // inputs: x, w, b (or none), residual
                const ins = try arena.alloc(ValueId, 4);
                ins[0] = n.inputs[0];
                ins[1] = n.inputs[1];
                ins[2] = if (n.inputs.len > 2) n.inputs[2] else ir.none;
                ins[3] = other;
                n.inputs = ins;
                n.outputs[0] = next.outputs[0];
                next.dead = true;
                continue;
            }
            break;
        }
    }
}

fn producedBefore(g: *Graph, id: ValueId, node: *const ir.Node) bool {
    for (g.nodes.items) |*n| {
        if (n == node) return false;
        if (n.dead) continue;
        if (std.mem.indexOfScalar(ValueId, n.outputs, id) != null) return true;
    }
    // Graph inputs and constants are available before everything.
    return true;
}

fn constF32(g: *Graph, id: ValueId) ?[]const f32 {
    if (id == ir.none) return null;
    const d = g.value(id).data orelse return null;
    return switch (d) {
        .f32 => |x| x,
        .i64 => null,
    };
}

fn sameConvAttrs(a: ir.Conv, b: ir.Conv) bool {
    return std.mem.eql(u32, &a.kernel, &b.kernel) and std.mem.eql(u32, &a.stride, &b.stride) and
        std.mem.eql(u32, &a.pad, &b.pad) and std.mem.eql(u32, &a.dilation, &b.dilation) and
        std.meta.eql(a.act, b.act) and !a.residual and !b.residual;
}

/// Concatenates the weights and biases of `convs` along output channels.
fn concatWeights(g: *Graph, arena: Allocator, convs: []const ir.Node) Allocator.Error!?[2]ValueId {
    var w: std.ArrayList(f32) = .empty;
    var b: std.ArrayList(f32) = .empty;
    var shape = g.value(convs[0].inputs[1]).shape;
    shape.d[0] = 0;
    for (convs) |c| {
        const wd = constF32(g, c.inputs[1]) orelse return null;
        const ws = g.value(c.inputs[1]).shape;
        const co = ws.d[0];
        shape.d[0] += co;
        try w.appendSlice(arena, wd);
        if (c.inputs.len > 2 and c.inputs[2] != ir.none) {
            const bd = constF32(g, c.inputs[2]) orelse return null;
            try b.appendSlice(arena, bd);
        } else try b.appendNTimes(arena, 0, co);
    }
    const wid = try g.addValue(arena, .{ .name = "", .shape = shape, .shaped = true, .data = .{ .f32 = w.items } });
    const bid = try g.addValue(arena, .{ .name = "", .shape = .init(&.{shape.d[0]}), .shaped = true, .data = .{ .f32 = b.items } });
    return .{ wid, bid };
}

fn sliceNode(arena: Allocator, from: ValueId, to: ValueId, c0: usize, c1: usize) Allocator.Error!ir.Node {
    return .{ .name = "slice", .op = .{ .channel_slice = .{ c0, c1 } }, .inputs = try arena.dupe(ValueId, &.{from}), .outputs = try arena.dupe(ValueId, &.{to}) };
}

/// Convolutions with the same input and attributes (detection heads)
/// become one convolution over all their output channels, followed by
/// channel slices; per-head padding to whole channel blocks disappears.
fn mergeSiblingConvs(g: *Graph, arena: Allocator) ir.Error!void {
    var i: usize = 0;
    while (i < g.nodes.items.len) : (i += 1) {
        const n = g.nodes.items[i];
        if (n.dead or n.op != .conv or n.op.conv.group != 1) continue;
        if (constF32(g, n.inputs[1]) == null) continue;
        var sib: std.ArrayList(usize) = .empty;
        try sib.append(arena, i);
        for (g.nodes.items[i + 1 ..], i + 1..) |m, j| {
            if (m.dead or m.op != .conv or m.op.conv.group != 1 or m.inputs[0] != n.inputs[0]) continue;
            if (!sameConvAttrs(n.op.conv, m.op.conv) or constF32(g, m.inputs[1]) == null) continue;
            try sib.append(arena, j);
        }
        if (sib.items.len < 2) continue;
        const convs = try arena.alloc(ir.Node, sib.items.len);
        for (sib.items, convs) |j, *c| c.* = g.nodes.items[j];
        const wb = try concatWeights(g, arena, convs) orelse continue;
        var shape = g.value(n.outputs[0]).shape;
        shape.d[1] = g.value(wb[0]).shape.d[0];
        const merged = try g.addValue(arena, .{ .name = "", .shape = shape, .shaped = true });
        var mc = n;
        mc.name = "merged";
        mc.inputs = try arena.dupe(ValueId, &.{ n.inputs[0], wb[0], wb[1] });
        mc.outputs = try arena.dupe(ValueId, &.{merged});
        for (sib.items[1..]) |j| g.nodes.items[j].dead = true;
        g.nodes.items[i] = mc;
        var off: usize = 0;
        for (convs, 0..) |c, k| {
            const co = g.value(c.outputs[0]).shape.d[1];
            try g.nodes.insert(arena, i + 1 + k, try sliceNode(arena, merged, c.outputs[0], off, off + co));
            off += co;
        }
    }
}

fn isDepthwise(g: *Graph, n: ir.Node) bool {
    if (n.dead or n.op != .conv or n.op.conv.residual) return false;
    const x = g.value(n.inputs[0]).shape;
    const w = g.value(n.inputs[1]).shape;
    return x.rank == 4 and n.op.conv.group == x.d[1] and w.d[0] == x.d[1] and w.d[1] == 1 and constF32(g, n.inputs[1]) != null;
}

/// Depthwise convolutions over channel slices that partition one tensor
/// (the heads after `mergeSiblingConvs`) become one depthwise convolution
/// over the whole tensor, followed by the same slices.
fn mergeSlicedDepthwise(g: *Graph, arena: Allocator) ir.Error!void {
    var i: usize = 0;
    while (i < g.nodes.items.len) : (i += 1) {
        const first = g.nodes.items[i];
        if (first.dead or first.op != .channel_slice) continue;
        const base = first.inputs[0];
        const total = g.value(base).shape.d[1];
        // Every live slice of `base`, each feeding exactly one depthwise
        // conv; the slices must tile [0, total) and the convs agree.
        var slices: std.ArrayList(usize) = .empty;
        var dws: std.ArrayList(usize) = .empty;
        var ok = true;
        for (g.nodes.items, 0..) |m, j| {
            if (m.dead or m.op != .channel_slice or m.inputs[0] != base) continue;
            const out = m.outputs[0];
            if (g.isOutput(out) or g.consumers(out) != 1) {
                ok = false;
                break;
            }
            var found: ?usize = null;
            for (g.nodes.items, 0..) |d, k| {
                if (!d.dead and d.inputs.len > 0 and d.inputs[0] == out) found = k;
            }
            const k = found orelse {
                ok = false;
                break;
            };
            const d = g.nodes.items[k];
            if (!isDepthwise(g, d)) {
                ok = false;
                break;
            }
            if (dws.items.len > 0 and !sameConvAttrs(d.op.conv, g.nodes.items[dws.items[0]].op.conv)) {
                ok = false;
                break;
            }
            try slices.append(arena, j);
            try dws.append(arena, k);
        }
        if (!ok or slices.items.len < 2) continue;
        // Order by start channel and check the slices tile the tensor.
        const order = try arena.alloc(usize, slices.items.len);
        for (order, 0..) |*o, k| o.* = k;
        const S = struct {
            s: []const usize,
            g: *Graph,
            fn lt(c: @This(), a: usize, b: usize) bool {
                return c.g.nodes.items[c.s[a]].op.channel_slice[0] < c.g.nodes.items[c.s[b]].op.channel_slice[0];
            }
        };
        std.sort.pdq(usize, order, S{ .s = slices.items, .g = g }, S.lt);
        var at: usize = 0;
        for (order) |o| {
            const r = g.nodes.items[slices.items[o]].op.channel_slice;
            if (r[0] != at) ok = false;
            at = r[1];
        }
        if (!ok or at != total) continue;
        const convs = try arena.alloc(ir.Node, order.len);
        for (order, convs) |o, *c| c.* = g.nodes.items[dws.items[o]];
        const ranges = try arena.alloc([2]usize, order.len);
        for (order, ranges) |o, *r| r.* = g.nodes.items[slices.items[o]].op.channel_slice;
        const wb = try concatWeights(g, arena, convs) orelse continue;
        var first_dw: usize = std.math.maxInt(usize);
        for (dws.items) |k| first_dw = @min(first_dw, k);
        var shape = g.value(convs[0].outputs[0]).shape;
        shape.d[1] = total;
        const merged = try g.addValue(arena, .{ .name = "", .shape = shape, .shaped = true });
        var mc = convs[0];
        mc.name = "merged_dw";
        mc.op.conv.group = @intCast(total);
        mc.inputs = try arena.dupe(ValueId, &.{ base, wb[0], wb[1] });
        mc.outputs = try arena.dupe(ValueId, &.{merged});
        for (slices.items) |j| g.nodes.items[j].dead = true;
        for (dws.items) |k| g.nodes.items[k].dead = true;
        g.nodes.items[first_dw] = mc;
        for (convs, ranges, 0..) |c, r, k| {
            try g.nodes.insert(arena, first_dw + 1 + k, try sliceNode(arena, merged, c.outputs[0], r[0], r[1]));
        }
    }
}

/// act(reshape(x)) = reshape(act(x)), likewise for transposes and the other
/// pure data movements: activations move up towards their producer, where
/// they can join a conv epilogue or an output reorder.
fn pushActsUp(g: *Graph, arena: Allocator) ir.Error!void {
    var changed = true;
    while (changed) {
        changed = false;
        for (g.nodes.items, 0..) |a, ia| {
            if (a.dead or a.op != .act) continue;
            const x = a.inputs[0];
            if (g.isOutput(x) or g.consumers(x) != 1) continue;
            const ip = g.producer(x) orelse continue;
            const p = g.nodes.items[ip];
            switch (p.op) {
                .reshape, .flatten, .squeeze, .unsqueeze, .transpose, .identity => {},
                else => continue,
            }
            const y = p.inputs[0];
            const yv = g.value(y).*;
            const x2 = try g.addValue(arena, .{ .name = "", .shape = yv.shape, .shaped = true, .dtype = yv.dtype });
            var a2 = a;
            a2.inputs = try arena.dupe(ValueId, &.{y});
            a2.outputs = try arena.dupe(ValueId, &.{x2});
            var p2 = p;
            p2.inputs = try arena.dupe(ValueId, p.inputs);
            p2.inputs[0] = x2;
            p2.outputs = try arena.dupe(ValueId, &.{a.outputs[0]});
            g.nodes.items[ip] = a2;
            g.nodes.items[ia] = p2;
            changed = true;
        }
    }
}

/// Drops nodes whose outputs nobody reads.
fn eliminateDead(g: *Graph) void {
    var changed = true;
    while (changed) {
        changed = false;
        var i = g.nodes.items.len;
        while (i > 0) {
            i -= 1;
            const n = &g.nodes.items[i];
            if (n.dead) continue;
            var used = false;
            for (n.outputs) |o| {
                if (g.isOutput(o) or g.consumers(o) > 0) used = true;
            }
            if (!used) {
                n.dead = true;
                changed = true;
            }
        }
    }
}
