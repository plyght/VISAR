//! The CPU backend: turns a `Plan` into a program of prepared kernels
//! (weights packed for the register tiles, index maps precomputed) and runs
//! it over one arena buffer. A complete backend in its own right and the
//! numerical reference for the others.

const std = @import("std");
const ir = @import("../ir.zig");
const plan_mod = @import("../plan.zig");
const vec = @import("vec.zig");
const conv = @import("conv.zig");
const ops = @import("ops.zig");
const core = @import("../../core/core.zig");
const Exec = core.Exec;
const Allocator = std.mem.Allocator;
const Plan = plan_mod.Plan;
const TensorId = plan_mod.TensorId;
const Rows = vec.Rows;

pub const block = vec.B;

const Kernel = union(enum) {
    pointwise: conv.Pointwise,
    depthwise: conv.Depthwise,
    direct: conv.Direct,
    blocked: conv.Blocked,
    reference: conv.Reference,
    pool: ops.Pool,
    global_avg: ops.GlobalAvg,
    resize: ops.Resize,
    act: ops.Act,
    affine: ops.Affine,
    binary: ops.Binary,
    reorder: ops.Reorder,
    transpose: ops.Transpose,
    extract: ops.Extract,
    concat: ops.Concat,
    softmax: ops.Softmax,
    gather: ops.Gather,
    slice: ops.Slice,
};

pub const Options = struct {
    /// Stream chains of row-local ops (convs, pools) through per-band ring
    /// buffers instead of materialising every intermediate tensor.
    stream: bool = true,
};

/// What the program runs, in order: one prepared step or a streamed chain.
pub const Item = union(enum) {
    kernel: *const Kernel,
    chain: Chain,

    pub fn run(it: *const Item, exec: anytype, base: [*]f32) void {
        switch (it.*) {
            .kernel => |k| runKernel(k, exec, base),
            .chain => |*c| c.run(exec, base),
        }
    }

    /// Steps it covers (for profiling).
    pub fn steps(it: *const Item) usize {
        return switch (it.*) {
            .kernel => 1,
            .chain => |c| c.stages.len,
        };
    }
};

pub const Program = struct {
    kernels: []Kernel,
    items: []Item,
    /// One per parallelFor call of a run (team execution).
    slots: []Slot,

    /// Prepares every step. Allocations (packed weights, maps, ring
    /// buffers) go to `arena`.
    pub fn init(arena: Allocator, p: *const Plan, opts: Options) (Allocator.Error || error{Unsupported})!Program {
        if (p.block != 0 and p.block != block) return error.Unsupported;
        const ks = try arena.alloc(Kernel, p.steps.len);
        for (p.steps, ks) |s, *k| k.* = try prepare(arena, p, s);
        var items: std.ArrayList(Item) = .empty;
        var i: usize = 0;
        var gi: usize = 0;
        while (i < ks.len) {
            if (gi < p.groups.len and p.groups[gi].first == i) {
                const len = p.groups[gi].len;
                gi += 1;
                if (opts.stream) {
                    try items.append(arena, .{ .chain = try Chain.init(arena, p, ks, i, len) });
                } else return error.Unsupported; // the plan gave group-internal tensors no storage
                i += len;
                continue;
            }
            try items.append(arena, .{ .kernel = &ks[i] });
            i += 1;
        }
        // Count the run's parallelFor calls (no kernel work happens: it all
        // sits inside parallelFor bodies).
        var counter: Counter = .{};
        var dummy: [64]f32 align(64) = undefined;
        for (items.items) |*it| it.run(&counter, &dummy);
        const slots = try arena.alloc(Slot, counter.calls);
        return .{ .kernels = ks, .items = items.items, .slots = slots };
    }

    /// Runs the program; `arena_buf` holds `plan.arena_bytes` bytes with
    /// the inputs already in place.
    pub fn run(prog: *const Program, exec: Exec, arena_buf: []align(64) u8) void {
        const base: [*]f32 = @ptrCast(arena_buf.ptr);
        if (exec.threads() <= 1) {
            for (prog.items) |*it| it.run(Exec.serial, base);
            return;
        }
        for (prog.slots) |*sl| sl.* = .{};
        const Ctx = struct { prog: *const Program, base: [*]f32, threads: usize };
        exec.parallelFor(exec.threads(), 1, Ctx{ .prog = prog, .base = base, .threads = exec.threads() }, struct {
            fn f(c: Ctx, s: usize, e: usize) void {
                for (s..e) |_| {
                    var team: Team = .{ .slots = c.prog.slots, .nthreads = c.threads };
                    for (c.prog.items) |*it| it.run(&team, c.base);
                }
            }
        }.f);
    }
};

pub fn runKernel(k: *const Kernel, exec: anytype, base: [*]f32) void {
    switch (k.*) {
        inline else => |*x| {
            const K = @TypeOf(x.*);
            if (@typeInfo(@TypeOf(K.run)).@"fn".param_types.len == 3) {
                x.run(exec, base);
            } else {
                // Serial kernels run once, on whichever thread claims them.
                const Ctx = struct { x: *const K, base: [*]f32 };
                exec.parallelFor(1, 1, Ctx{ .x = x, .base = base }, struct {
                    fn f(c: Ctx, _: usize, _: usize) void {
                        c.x.run(c.base);
                    }
                }.f);
            }
        },
    }
}

// ------------------------------------------------------------------ team

/// One op's work distribution in a team run: chunks are claimed through
/// `next` and counted in `done`; on separate cache lines.
const Slot = struct {
    next: std.atomic.Value(usize) align(64) = .init(0),
    done: std.atomic.Value(usize) align(64) = .init(0),
};

/// A whole inference is one pool job of `threads` virtual threads, each
/// running every op in order (SPMD). An op's chunks go to whichever
/// threads ask first, and a thread moves on once all of the op's chunks
/// are done, so a thread that was descheduled (another process, or another
/// library's spinning workers) only delays the run if it holds a chunk,
/// instead of stalling every op until it checks in.
const Team = struct {
    slots: []Slot,
    nthreads: usize,
    call: usize = 0,

    pub fn threads(t: *const Team) usize {
        return t.nthreads;
    }

    pub fn parallelFor(t: *Team, n: usize, grain: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        const slot = &t.slots[t.call];
        t.call += 1;
        if (n == 0) return;
        const max_chunks = @max(1, n / @max(1, grain));
        const chunks = @min(max_chunks, t.nthreads * 4);
        const per = (n + chunks - 1) / chunks;
        while (true) {
            const c = slot.next.fetchAdd(1, .monotonic);
            if (c >= chunks) break;
            const s = c * per;
            if (s < n) func(ctx, s, @min(n, s + per));
            _ = slot.done.fetchAdd(1, .release);
        }
        while (slot.done.load(.acquire) < chunks) std.atomic.spinLoopHint();
    }
};

/// Counts the parallelFor calls of one run (to size the team's slots).
const Counter = struct {
    calls: usize = 0,

    pub fn threads(_: *const Counter) usize {
        return 2;
    }

    pub fn parallelFor(c: *Counter, n: usize, grain: usize, ctx: anytype, comptime func: fn (@TypeOf(ctx), usize, usize) void) void {
        _ = n;
        _ = grain;
        _ = func;
        c.calls += 1;
    }
};

// ------------------------------------------------------------- streaming

/// The plan's grouping predicate for this backend: steps whose CPU kernel
/// computes output rows from a window of input rows (`Direct` only as a
/// group's first step, as it reads a whole tensor).
pub fn streamable(p: *const Plan, i: usize, first: bool) bool {
    const s = p.steps[i];
    const x = p.tensors[s.inputs[0]];
    const o = p.tensors[s.outputs[0]];
    if (o.layout != .blocked or x.shape.d[0] != 1) return false;
    return switch (s.op) {
        .conv => switch (convKind(p, s)) {
            .pointwise, .depthwise => true,
            .direct, .blocked => first,
            .reference => false,
        },
        .max_pool, .average_pool => x.layout == .blocked,
        else => false,
    };
}

const Stage = struct {
    k: *const Kernel,
    /// Input rows (height of the previous stage's output) and window.
    in_h: usize,
    span: usize,
    stride: usize,
    pad: usize,
    out_h: usize,
    out_w: usize,
    out_blocks: usize,
    /// Rows kept of this stage's output (the next stage's window) and
    /// where they live in a band's scratch.
    ring_rows: usize = 0,
    ring_off: usize = 0,
    /// Residual input (element offset) or none.
    res: usize = conv.no_res,
};

/// Consecutive row-local steps run band by band: each band of final output
/// rows pulls the rows it needs through the stages, keeping only a few rows
/// of every intermediate (in cache) instead of whole tensors.
const Chain = struct {
    stages: []Stage,
    in_off: usize,
    in_blocks: usize,
    in_w: usize,
    out_off: usize,
    scratch: []align(64) f32,
    per_band: usize,

    const max_stages = plan_mod.max_group;
    const max_bands = 16;

    fn init(arena: Allocator, p: *const Plan, ks: []const Kernel, i: usize, len: usize) Allocator.Error!Chain {
        const stages = try arena.alloc(Stage, len);
        for (stages, 0..) |*st, j| {
            const k = &ks[i + j];
            const o = p.tensors[p.steps[i + j].outputs[0]].shape;
            const x = p.tensors[p.steps[i + j].inputs[0]].shape;
            st.* = .{ .k = k, .in_h = x.d[2], .span = 1, .stride = 1, .pad = 0, .out_h = o.d[2], .out_w = o.d[3], .out_blocks = plan_mod.ceilDiv(o.d[1], block) };
            switch (k.*) {
                .depthwise => |d| {
                    st.span = (d.g.kh - 1) * d.g.dh + 1;
                    st.stride = d.g.sh;
                    st.pad = d.g.pt;
                    st.res = d.io.res;
                },
                .pointwise => |d| st.res = d.io.res,
                .direct => |d| st.res = d.io.res,
                .blocked => |d| st.res = d.io.res,
                .pool => |d| {
                    st.span = (d.p.kernel[0] - 1) * d.p.dilation[0] + 1;
                    st.stride = d.p.stride[0];
                    st.pad = d.p.pad[0];
                },
                else => unreachable,
            }
        }
        var per: usize = 0;
        for (stages[0 .. len - 1], stages[1..]) |*st, next| {
            st.ring_rows = next.span;
            st.ring_off = per;
            per += std.mem.alignForward(usize, st.ring_rows * st.out_blocks * st.out_w * block, 16);
        }
        const x0 = p.tensors[p.steps[i].inputs[0]];
        return .{
            .stages = stages,
            .in_off = ref(p, p.steps[i].inputs[0]).off,
            .in_blocks = plan_mod.ceilDiv(x0.shape.d[1], block),
            .in_w = x0.shape.d[3],
            .out_off = ref(p, p.steps[i + len - 1].outputs[0]).off,
            .scratch = try arena.alignedAlloc(f32, .@"64", @max(1, per * max_bands)),
            .per_band = per,
        };
    }

    const Ctx = struct { c: *const Chain, base: [*]f32, rows: usize };

    fn run(c: *const Chain, exec: anytype, base: [*]f32) void {
        const last = c.stages[c.stages.len - 1];
        const bands = @min(max_bands, exec.threads(), @max(1, last.out_h / 2));
        const rows = (last.out_h + bands - 1) / bands;
        exec.parallelFor(bands, 1, Ctx{ .c = c, .base = base, .rows = rows }, struct {
            fn f(x: Ctx, s: usize, e: usize) void {
                for (s..e) |b| x.c.band(x.base, b, x.rows);
            }
        }.f);
    }

    fn band(c: *const Chain, base: [*]f32, b: usize, rows: usize) void {
        const n = c.stages.len;
        const last = c.stages[n - 1];
        const y0 = b * rows;
        const y1 = @min(last.out_h, y0 + rows);
        if (y0 >= y1) return;
        var next: [max_stages]usize = undefined;
        // The first row each stage must produce for this band.
        next[n - 1] = y0;
        var s = n - 1;
        while (s > 0) : (s -= 1) {
            const st = c.stages[s];
            next[s - 1] = (next[s] * st.stride) -| st.pad;
        }
        const scratch = c.scratch.ptr + b * c.per_band;
        const st: State = .{ .c = c, .base = base, .scratch = scratch, .next = &next };
        for (y0..y1) |y| st.produce(n - 1, y);
    }

    const State = struct {
        c: *const Chain,
        base: [*]f32,
        scratch: [*]f32,
        next: *[max_stages]usize,

        fn ringOf(st: State, s: usize) Rows {
            const g = st.c.stages[s];
            return Rows.ringOf(st.scratch + g.ring_off, g.ring_rows, g.out_blocks, g.out_w);
        }

        fn produce(st: State, s: usize, y: usize) void {
            const g = st.c.stages[s];
            if (s > 0) {
                const top = y * g.stride + g.span - 1;
                if (top >= g.pad) {
                    const hi = @min(top - g.pad, g.in_h - 1);
                    while (st.next[s - 1] <= hi) {
                        st.produce(s - 1, st.next[s - 1]);
                        st.next[s - 1] += 1;
                    }
                }
            }
            const c = st.c;
            const src: Rows = if (s == 0) Rows.tensor(st.base, c.in_off, 0, c.in_blocks, g.in_h, c.in_w) else st.ringOf(s - 1);
            const dst: Rows = if (s + 1 == c.stages.len) Rows.tensor(st.base, c.out_off, 0, g.out_blocks, g.out_h, g.out_w) else st.ringOf(s);
            const res: ?Rows = if (g.res != conv.no_res) Rows.tensor(st.base, g.res, 0, g.out_blocks, g.out_h, g.out_w) else null;
            switch (g.k.*) {
                .pointwise => |*k| switch (vec.actKind(k.epi.act)) {
                    inline else => |ak| k.streamRow(ak, src, dst, res, y),
                },
                .depthwise => |*k| switch (vec.actKind(k.epi.act)) {
                    inline else => |ak| k.streamRow(ak, src, dst, res, y),
                },
                .direct => |*k| switch (vec.actKind(k.epi.act)) {
                    inline else => |ak| k.streamRow(ak, st.base, dst, res, y),
                },
                .blocked => |*k| switch (vec.actKind(k.epi.act)) {
                    inline else => |ak| k.streamRow(ak, st.base, dst, res, y),
                },
                .pool => |*k| k.streamRow(src, dst, y),
                else => unreachable,
            }
        }
    };
};

fn ref(p: *const Plan, t: TensorId) ops.Ref {
    const x = p.tensors[t];
    if (x.data) |d| return .{ .cdata = d.ptr };
    return .{ .off = p.offset(t) / @sizeOf(f32) };
}

const ConvKind = enum { pointwise, depthwise, direct, blocked, reference };

fn convKind(p: *const Plan, s: plan_mod.Step) ConvKind {
    const c = s.op.conv;
    const x = p.tensors[s.inputs[0]];
    const o = p.tensors[s.outputs[0]];
    const plain = x.layout == .plain;
    if (o.layout != .blocked) return .reference;
    const cin = x.shape.d[1];
    const cout = o.shape.d[1];
    if (c.group == 1 and !plain and c.kernel[0] == 1 and c.kernel[1] == 1 and c.stride[0] == 1 and c.stride[1] == 1 and std.mem.allEqual(u32, &c.pad, 0))
        return .pointwise;
    if (c.group == cin and cout == cin and !plain) return .depthwise;
    if (c.group == 1) return if (plain) .direct else .blocked;
    return .reference;
}

fn prepare(arena: Allocator, p: *const Plan, s: plan_mod.Step) (Allocator.Error || error{Unsupported})!Kernel {
    const T = p.tensors;
    switch (s.op) {
        .conv => |c| {
            const x = T[s.inputs[0]];
            const w = T[s.inputs[1]];
            const o = T[s.outputs[0]];
            const bias: ?[]const f32 = if (s.inputs[2] != plan_mod.none) T[s.inputs[2]].data.? else null;
            const g: conv.Geom = .{
                .n = x.shape.d[0],
                .cin = x.shape.d[1],
                .h = x.shape.d[2],
                .w = x.shape.d[3],
                .cout = o.shape.d[1],
                .oh = o.shape.d[2],
                .ow = o.shape.d[3],
                .kh = c.kernel[0],
                .kw = c.kernel[1],
                .sh = c.stride[0],
                .sw = c.stride[1],
                .pt = c.pad[0],
                .pl = c.pad[1],
                .dh = c.dilation[0],
                .dw = c.dilation[1],
                .group = c.group,
            };
            const io: conv.Io = .{
                .x = ref(p, s.inputs[0]).off,
                .out = ref(p, s.outputs[0]).off,
                .res = if (s.inputs[3] != plan_mod.none) ref(p, s.inputs[3]).off else conv.no_res,
                .x_plain = x.layout == .plain,
                .out_plain = o.layout == .plain,
            };
            const wd = w.data orelse return error.Unsupported;
            return switch (convKind(p, s)) {
                .pointwise => .{ .pointwise = try .init(arena, g, io, c, wd, bias) },
                .depthwise => .{ .depthwise = try .init(arena, g, io, c, wd, bias) },
                .direct => .{ .direct = try .init(arena, g, io, c, wd, bias) },
                .blocked => .{ .blocked = try .init(arena, g, io, c, wd, bias) },
                .reference => .{ .reference = try .init(arena, g, io, c, wd, bias) },
            };
        },
        .affine => {
            const x = T[s.inputs[0]];
            return .{ .affine = try .init(arena, ref(p, s.inputs[0]), ref(p, s.outputs[0]), x.shape, x.layout == .blocked, T[s.inputs[1]].data.?, T[s.inputs[2]].data.?) };
        },
        .act => |a| {
            const o = T[s.outputs[0]];
            return .{ .act = .{ .act = a, .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .count = o.storedCount(p.block), .blocked = if (o.layout == .blocked) o.shape else null } };
        },
        .binary => |op| {
            const a = T[s.inputs[0]];
            const b = T[s.inputs[1]];
            const o = T[s.outputs[0]];
            const blocked = o.layout == .blocked;
            return .{ .binary = try .init(arena, op, ref(p, s.inputs[0]), ref(p, s.inputs[1]), ref(p, s.outputs[0]), a.shape, b.shape, o.shape, blocked, if (blocked) a.data else null, if (blocked) b.data else null) };
        },
        .max_pool, .average_pool => |pl| {
            const x = T[s.inputs[0]].shape;
            const o = T[s.outputs[0]].shape;
            return .{ .pool = .{ .kind = if (s.op == .max_pool) .max else .avg, .init = if (s.op == .max_pool) -std.math.inf(f32) else 0, .plain = T[s.inputs[0]].layout == .plain, .p = pl, .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .n = x.d[0], .c = x.d[1], .h = x.d[2], .w = x.d[3], .oh = o.d[2], .ow = o.d[3] } };
        },
        .global_average_pool => {
            const x = T[s.inputs[0]].shape;
            const plain = T[s.inputs[0]].layout == .plain;
            return .{ .global_avg = .{ .plain = plain, .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .planes = x.d[0] * (if (plain) x.d[1] else plan_mod.ceilDiv(x.d[1], block)), .hw = x.d[2] * x.d[3] } };
        },
        .resize => |r| return .{ .resize = try .init(arena, r, T[s.inputs[0]].layout == .plain, ref(p, s.inputs[0]), ref(p, s.outputs[0]), T[s.inputs[0]].shape, T[s.outputs[0]].shape) },
        .reorder => {
            const x = T[s.inputs[0]];
            const o = T[s.outputs[0]];
            const kind: @FieldType(ops.Reorder, "kind") = if (x.layout == o.layout) .copy else if (o.layout == .blocked) .to_blocked else .to_plain;
            return .{ .reorder = .{ .kind = kind, .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .s = x.shape } };
        },
        .transpose => |perm| {
            const x = T[s.inputs[0]];
            if (x.layout != .plain) return error.Unsupported;
            return .{ .transpose = .{ .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .s = x.shape, .perm = perm } };
        },
        .to_nhwc, .extract => |e| {
            const x = T[s.inputs[0]];
            if (x.layout != .blocked) return error.Unsupported;
            return .{ .extract = .{ .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .s = x.shape, .c0 = e.c0, .c1 = e.c1, .act = e.act, .nhwc = s.op == .to_nhwc } };
        },
        .concat => |axis| {
            const rs = try arena.alloc(ops.Ref, s.inputs.len);
            const ss = try arena.alloc(ir.Shape, s.inputs.len);
            for (s.inputs, rs, ss) |t, *r, *sh| {
                r.* = ref(p, t);
                sh.* = T[t].shape;
            }
            const o = T[s.outputs[0]];
            return .{ .concat = .{ .ins = rs, .in_shapes = ss, .out = ref(p, s.outputs[0]), .os = o.shape, .axis = axis, .blocked = o.layout == .blocked } };
        },
        .softmax => |axis| return .{ .softmax = .{ .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .s = T[s.inputs[0]].shape, .axis = axis } },
        .gather => |ga| {
            const rs = try arena.alloc(ops.Ref, s.inputs.len);
            const ss = try arena.alloc(ir.Shape, s.inputs.len);
            for (s.inputs, rs, ss) |t, *r, *sh| {
                r.* = ref(p, t);
                sh.* = T[t].shape;
            }
            const o = T[s.outputs[0]];
            return .{ .gather = try .init(arena, ga.map, ga.act, rs, ss, ref(p, s.outputs[0]), o.shape, o.layout == .blocked) };
        },
        .slice => |st| {
            if (T[s.inputs[0]].layout != .plain) return error.Unsupported;
            return .{ .slice = .{ .in = ref(p, s.inputs[0]), .out = ref(p, s.outputs[0]), .s = T[s.inputs[0]].shape, .st = st } };
        },
    }
}
