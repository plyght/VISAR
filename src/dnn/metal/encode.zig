//! Lowers a `Plan` to the Metal backend's dispatch list: for every step,
//! which kernel of kernels.metal runs, its parameter block and its grid,
//! plus one buffer of constants (weights packed for the kernels, biases,
//! per-channel vectors, resize maps). Plain data with no Metal calls, so it
//! compiles and is tested on every target; device.zig turns it into
//! pipeline states and one command buffer per inference.
//!
//! The plan must use blocks of `block` = 4 channels (one float4) and no
//! streamed groups. Any step without a kernel makes `encode` fail with
//! error.Unsupported: the whole plan is refused and the caller runs it on
//! the CPU backend instead (no per-op fallback).

const std = @import("std");
const ir = @import("../ir.zig");
const plan_mod = @import("../plan.zig");
const cpu_ops = @import("../cpu/ops.zig");
const Allocator = std.mem.Allocator;
const Plan = plan_mod.Plan;
const TensorId = plan_mod.TensorId;

/// Channels per block of the plans this backend runs.
pub const block: usize = 4;

pub const Error = Allocator.Error || error{Unsupported};

// ------------------------------------------------------------ parameters
//
// Only 4-byte scalar fields and arrays of them, so the MSL structs
// generated from these (`msl_structs`) have the same layout. Offsets are in
// f32 elements (of the arena A, or of the constants K when the matching
// `*k` flag is set).

pub const ConvP = extern struct {
    x: u32,
    out: u32,
    wt: u32,
    bias: u32,
    res: u32,
    has_res: u32,
    res_after: u32,
    act: u32,
    a0: f32,
    a1: f32,
    n: u32,
    cin: u32,
    cinb: u32,
    h: u32,
    iw: u32,
    cout: u32,
    coutb: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    pt: u32,
    pl: u32,
    dh: u32,
    dw: u32,
    group: u32,
    x_plain: u32,
};

pub const PoolP = extern struct {
    in: u32,
    out: u32,
    planes: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    sh: u32,
    sw: u32,
    pt: u32,
    pl: u32,
    pb: u32,
    pr: u32,
    dh: u32,
    dw: u32,
    is_max: u32,
    incl_pad: u32,
    init: f32,
};

pub const GapP = extern struct { in: u32, out: u32, planes: u32, hw: u32 };

pub const ResizeP = extern struct {
    in: u32,
    out: u32,
    planes: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    ymap: u32,
    xmap: u32,
    yw: u32,
    xw: u32,
    linear: u32,
};

pub const ActP = extern struct {
    in: u32,
    out: u32,
    count: u32,
    act: u32,
    a0: f32,
    a1: f32,
    blocked: u32,
    c: u32,
    hw: u32,
    nb: u32,
};

pub const AffineP = extern struct {
    in: u32,
    out: u32,
    count: u32,
    scale: u32,
    shift: u32,
    blocked: u32,
    c: u32,
    inner: u32,
    nb: u32,
};

pub const BinP = extern struct {
    a: u32,
    b: u32,
    out: u32,
    ak: u32,
    bk: u32,
    count: u32,
    op: u32,
    blocked: u32,
    c: u32,
    hw: u32,
    nb: u32,
    const_first: u32,
    rank: u32,
    od: [ir.max_rank]u32,
    sa: [ir.max_rank]u32,
    sb: [ir.max_rank]u32,
};

pub const LayoutP = extern struct {
    in: u32,
    out: u32,
    ik: u32,
    count: u32,
    c: u32,
    h: u32,
    w: u32,
    nb: u32,
};

pub const Copy2dP = extern struct {
    in: u32,
    out: u32,
    ik: u32,
    outer: u32,
    len: u32,
    orow: u32,
    at: u32,
};

pub const TransposeP = extern struct {
    in: u32,
    out: u32,
    ik: u32,
    count: u32,
    rank: u32,
    od: [ir.max_rank]u32,
    is: [ir.max_rank]u32,
};

pub const ExtractP = extern struct {
    in: u32,
    out: u32,
    count: u32,
    c0: u32,
    cn: u32,
    nb: u32,
    h: u32,
    w: u32,
    nhwc: u32,
    act: u32,
    a0: f32,
    a1: f32,
};

pub const SoftmaxP = extern struct { in: u32, out: u32, outer: u32, inner: u32, d: u32 };

/// The kernels of kernels.metal (function name: "k_" ++ tag name).
pub const Kernel = enum {
    conv_pw,
    conv_dw,
    conv_dir_b,
    conv_dir_p,
    conv_ref,
    pool,
    gap,
    resize,
    act,
    affine,
    bin_same,
    bin_pc,
    bin_bc,
    to_blocked,
    to_plain,
    copy,
    copy2d,
    transpose,
    extract,
    softmax,

    pub fn Params(comptime k: Kernel) type {
        return switch (k) {
            .conv_pw, .conv_dw, .conv_dir_b, .conv_dir_p, .conv_ref => ConvP,
            .pool => PoolP,
            .gap => GapP,
            .resize => ResizeP,
            .act => ActP,
            .affine => AffineP,
            .bin_same, .bin_pc, .bin_bc => BinP,
            .to_blocked, .to_plain, .copy => LayoutP,
            .copy2d => Copy2dP,
            .transpose => TransposeP,
            .extract => ExtractP,
            .softmax => SoftmaxP,
        };
    }

    /// The MSL function name (NUL-terminated).
    pub fn name(k: Kernel) [:0]const u8 {
        return switch (k) {
            inline else => |t| "k_" ++ @tagName(t),
        };
    }
};

pub const kernel_count = std.enums.values(Kernel).len;

/// The parameter structs as MSL source, generated from the types above.
pub const msl_structs: []const u8 = blk: {
    const list = .{
        .{ "ConvP", ConvP },     .{ "PoolP", PoolP },           .{ "GapP", GapP },         .{ "ResizeP", ResizeP },
        .{ "ActP", ActP },       .{ "AffineP", AffineP },       .{ "BinP", BinP },         .{ "LayoutP", LayoutP },
        .{ "Copy2dP", Copy2dP }, .{ "TransposeP", TransposeP }, .{ "ExtractP", ExtractP }, .{ "SoftmaxP", SoftmaxP },
    };
    var s: []const u8 = "";
    for (list) |e| s = s ++ mslStruct(e[0], e[1]);
    break :blk s;
};

fn mslScalar(comptime T: type) []const u8 {
    return switch (T) {
        u32 => "uint",
        i32 => "int",
        f32 => "float",
        else => @compileError("Metal params: unsupported field type " ++ @typeName(T)),
    };
}

fn mslStruct(comptime name: []const u8, comptime T: type) []const u8 {
    var s: []const u8 = "struct " ++ name ++ " {\n";
    const info = @typeInfo(T).@"struct";
    for (info.field_names, info.field_types) |fname, ftype| {
        s = s ++ switch (@typeInfo(ftype)) {
            .array => |a| "    " ++ mslScalar(a.child) ++ " " ++ fname ++ std.fmt.comptimePrint("[{d}];\n", .{a.len}),
            else => "    " ++ mslScalar(ftype) ++ " " ++ fname ++ ";\n",
        };
    }
    return s ++ "};\n\n";
}

/// The kernels' body (without the generated structs).
pub const kernels_body = @embedFile("kernels.metal");

/// The complete MSL source handed to newLibraryWithSource.
pub const source: [:0]const u8 = "#include <metal_stdlib>\nusing namespace metal;\n\n" ++ msl_structs ++ kernels_body;

// -------------------------------------------------------------- encoding

pub const Dispatch = struct {
    kernel: Kernel,
    /// The kernel's parameter struct, as bytes.
    params: []align(4) const u8,
    /// Threads in x, y, z (each kernel bounds-checks).
    grid: [3]u32,
};

pub const Encoded = struct {
    dispatches: []Dispatch,
    /// Constants (buffer 1), each blob 64-byte aligned.
    consts: []const f32,
    arena_bytes: usize,
};

/// Rough multiply-adds of the plan's convolutions (to decide whether a
/// model is big enough for the GPU to pay off).
pub fn convMacs(p: *const Plan) u64 {
    var macs: u64 = 0;
    for (p.steps) |s| {
        if (s.op != .conv) continue;
        const c = s.op.conv;
        const x = p.tensors[s.inputs[0]].shape;
        const o = p.tensors[s.outputs[0]].shape;
        if (x.rank != 4 or o.rank != 4 or c.group == 0) continue;
        macs += @as(u64, o.count()) * (x.d[1] / c.group) * c.kernel[0] * c.kernel[1];
    }
    return macs;
}

const Encoder = struct {
    arena: Allocator,
    p: *const Plan,
    consts: std.ArrayList(f32) = .empty,
    dispatches: std.ArrayList(Dispatch) = .empty,
    /// K offset of each constant tensor uploaded as is (maxInt: not yet).
    const_off: []u32,

    fn u(x: usize) Error!u32 {
        return std.math.cast(u32, x) orelse error.Unsupported;
    }

    /// Appends a 64-byte-aligned blob of `n` zeroed floats; its offset.
    fn reserve(e: *Encoder, n: usize) Error!struct { off: u32, data: []f32 } {
        while (e.consts.items.len % 16 != 0) try e.consts.append(e.arena, 0);
        const off = e.consts.items.len;
        try e.consts.appendNTimes(e.arena, 0, @max(n, 1));
        return .{ .off = try u(off), .data = e.consts.items[off..][0..n] };
    }

    fn blob(e: *Encoder, data: []const f32) Error!u32 {
        const r = try e.reserve(data.len);
        @memcpy(r.data, data);
        return r.off;
    }

    /// Where tensor `t` lives: arena offset, or K offset of its constant
    /// data (uploaded once), with the K flag.
    fn ref(e: *Encoder, t: TensorId) Error!struct { off: u32, k: u32 } {
        const x = e.p.tensors[t];
        if (x.data) |d| {
            if (e.const_off[t] == std.math.maxInt(u32)) e.const_off[t] = try e.blob(d);
            return .{ .off = e.const_off[t], .k = 1 };
        }
        const off = e.p.offset(t);
        std.debug.assert(off % 4 == 0);
        return .{ .off = try u(off / 4), .k = 0 };
    }

    /// Arena offset of a tensor that must not be a constant.
    fn arenaOff(e: *Encoder, t: TensorId) Error!u32 {
        const r = try e.ref(t);
        if (r.k != 0) return error.Unsupported;
        return r.off;
    }

    fn emit(e: *Encoder, comptime k: Kernel, params: k.Params(), grid: [3]usize) Error!void {
        var g: [3]u32 = undefined;
        for (grid, &g) |a, *b| b.* = try u(a);
        if (g[0] == 0 or g[1] == 0 or g[2] == 0) return; // nothing to do
        const bytes = try e.arena.alignedAlloc(u8, .@"4", @sizeOf(k.Params()));
        @memcpy(bytes, std.mem.asBytes(&params));
        try e.dispatches.append(e.arena, .{ .kernel = k, .params = bytes, .grid = g });
    }

    fn step(e: *Encoder, s: plan_mod.Step) Error!void {
        const p = e.p;
        const T = p.tensors;
        const op = s.op;
        // An if-chain rather than a switch, so ops added to the plan later
        // are refused (error.Unsupported) instead of breaking the build.
        if (op == .conv) return e.conv(s, op.conv);
        if (op == .max_pool or op == .average_pool) {
            const pl = if (op == .max_pool) op.max_pool else op.average_pool;
            const x = T[s.inputs[0]];
            const o = T[s.outputs[0]];
            if (x.layout != .blocked or o.layout != .blocked) return error.Unsupported;
            const planes = x.shape.d[0] * nblocks(x.shape.d[1]);
            const is_max = op == .max_pool;
            try e.emit(.pool, .{
                .in = try e.arenaOff(s.inputs[0]),
                .out = try e.arenaOff(s.outputs[0]),
                .planes = try u(planes),
                .ih = try u(x.shape.d[2]),
                .iw = try u(x.shape.d[3]),
                .oh = try u(o.shape.d[2]),
                .ow = try u(o.shape.d[3]),
                .kh = pl.kernel[0],
                .kw = pl.kernel[1],
                .sh = pl.stride[0],
                .sw = pl.stride[1],
                .pt = pl.pad[0],
                .pl = pl.pad[1],
                .pb = pl.pad[2],
                .pr = pl.pad[3],
                .dh = pl.dilation[0],
                .dw = pl.dilation[1],
                .is_max = @intFromBool(is_max),
                .incl_pad = @intFromBool(pl.count_include_pad),
                .init = if (is_max) -std.math.inf(f32) else 0,
            }, .{ o.shape.d[3], o.shape.d[2], planes });
            return;
        }
        if (op == .global_average_pool) {
            const x = T[s.inputs[0]];
            if (x.layout != .blocked or T[s.outputs[0]].layout != .blocked) return error.Unsupported;
            const planes = x.shape.d[0] * nblocks(x.shape.d[1]);
            try e.emit(.gap, .{ .in = try e.arenaOff(s.inputs[0]), .out = try e.arenaOff(s.outputs[0]), .planes = try u(planes), .hw = try u(x.shape.d[2] * x.shape.d[3]) }, .{ planes, 1, 1 });
            return;
        }
        if (op == .resize) {
            const x = T[s.inputs[0]];
            const o = T[s.outputs[0]];
            if (x.layout != .blocked or o.layout != .blocked) return error.Unsupported;
            // The CPU backend's maps, so both backends sample alike.
            const rz = try cpu_ops.Resize.init(e.arena, op.resize, false, .{}, .{}, x.shape, o.shape);
            const ym = try e.reserve(rz.ymap.len);
            for (ym.data, rz.ymap) |*d, v| d.* = @bitCast(v);
            const xm = try e.reserve(rz.xmap.len);
            for (xm.data, rz.xmap) |*d, v| d.* = @bitCast(v);
            const yw = try e.blob(rz.yw);
            const xw = try e.blob(rz.xw);
            const planes = x.shape.d[0] * nblocks(x.shape.d[1]);
            try e.emit(.resize, .{
                .in = try e.arenaOff(s.inputs[0]),
                .out = try e.arenaOff(s.outputs[0]),
                .planes = try u(planes),
                .ih = try u(x.shape.d[2]),
                .iw = try u(x.shape.d[3]),
                .oh = try u(o.shape.d[2]),
                .ow = try u(o.shape.d[3]),
                .ymap = ym.off,
                .xmap = xm.off,
                .yw = yw,
                .xw = xw,
                .linear = @intFromBool(rz.linear),
            }, .{ o.shape.d[3], o.shape.d[2], planes });
            return;
        }
        if (op == .act) {
            const o = T[s.outputs[0]];
            const a = try actCode(op.act);
            const blocked = o.layout == .blocked;
            const count = o.storedCount(block);
            try e.emit(.act, .{
                .in = try e.arenaOff(s.inputs[0]),
                .out = try e.arenaOff(s.outputs[0]),
                .count = try u(count),
                .act = a.code,
                .a0 = a.a0,
                .a1 = a.a1,
                .blocked = @intFromBool(blocked),
                .c = if (blocked) try u(o.shape.d[1]) else 0,
                .hw = if (blocked) try u(o.shape.d[2] * o.shape.d[3]) else 1,
                .nb = if (blocked) try u(nblocks(o.shape.d[1])) else 1,
            }, .{ count, 1, 1 });
            return;
        }
        if (op == .affine) {
            const x = T[s.inputs[0]];
            const blocked = x.layout == .blocked;
            const c = x.shape.d[1];
            const width = if (blocked) nblocks(c) * block else c;
            const sc = try e.reserve(width);
            @memcpy(sc.data[0..c], T[s.inputs[1]].data orelse return error.Unsupported);
            const sh = try e.reserve(width);
            @memcpy(sh.data[0..c], T[s.inputs[2]].data orelse return error.Unsupported);
            var inner: usize = 1;
            for (2..x.shape.rank) |i| inner *= x.shape.d[i];
            const count = x.storedCount(block);
            try e.emit(.affine, .{
                .in = try e.arenaOff(s.inputs[0]),
                .out = try e.arenaOff(s.outputs[0]),
                .count = try u(count),
                .scale = sc.off,
                .shift = sh.off,
                .blocked = @intFromBool(blocked),
                .c = try u(c),
                .inner = try u(@max(inner, 1)),
                .nb = try u(nblocks(c)),
            }, .{ count, 1, 1 });
            return;
        }
        if (op == .binary) return e.binary(s, op.binary);
        if (op == .concat) return e.concat(s, op.concat);
        if (op == .transpose) {
            const x = T[s.inputs[0]];
            if (x.layout != .plain or T[s.outputs[0]].layout != .plain) return error.Unsupported;
            const r = x.shape.rank;
            var stride: [ir.max_rank]usize = undefined;
            var st: usize = 1;
            var i: usize = r;
            while (i > 0) {
                i -= 1;
                stride[i] = st;
                st *= x.shape.d[i];
            }
            var prm: TransposeP = .{ .in = 0, .out = 0, .ik = 0, .count = try u(x.shape.count()), .rank = @intCast(r), .od = @splat(1), .is = @splat(0) };
            for (0..r) |a| {
                const pa = op.transpose[a];
                if (pa >= r) return error.Unsupported;
                prm.od[a] = try u(x.shape.d[pa]);
                prm.is[a] = try u(stride[pa]);
            }
            const src = try e.ref(s.inputs[0]);
            prm.in = src.off;
            prm.ik = src.k;
            prm.out = try e.arenaOff(s.outputs[0]);
            try e.emit(.transpose, prm, .{ x.shape.count(), 1, 1 });
            return;
        }
        if (op == .to_nhwc or op == .extract) {
            const ex = if (op == .to_nhwc) op.to_nhwc else op.extract;
            const x = T[s.inputs[0]];
            if (x.layout != .blocked or x.shape.rank != 4 or ex.c1 < ex.c0 or ex.c1 > x.shape.d[1]) return error.Unsupported;
            const a = try actCode(ex.act);
            const cn = ex.c1 - ex.c0;
            const count = x.shape.d[0] * cn * x.shape.d[2] * x.shape.d[3];
            try e.emit(.extract, .{
                .in = try e.arenaOff(s.inputs[0]),
                .out = try e.arenaOff(s.outputs[0]),
                .count = try u(count),
                .c0 = try u(ex.c0),
                .cn = try u(cn),
                .nb = try u(nblocks(x.shape.d[1])),
                .h = try u(x.shape.d[2]),
                .w = try u(x.shape.d[3]),
                .nhwc = @intFromBool(op == .to_nhwc),
                .act = a.code,
                .a0 = a.a0,
                .a1 = a.a1,
            }, .{ count, 1, 1 });
            return;
        }
        if (op == .softmax) {
            const x = T[s.inputs[0]];
            const axis = op.softmax;
            if (x.layout != .plain or axis >= x.shape.rank) return error.Unsupported;
            var outer: usize = 1;
            for (0..axis) |i| outer *= x.shape.d[i];
            var inner: usize = 1;
            for (axis + 1..x.shape.rank) |i| inner *= x.shape.d[i];
            try e.emit(.softmax, .{ .in = try e.arenaOff(s.inputs[0]), .out = try e.arenaOff(s.outputs[0]), .outer = try u(outer), .inner = try u(inner), .d = try u(x.shape.d[axis]) }, .{ inner, outer, 1 });
            return;
        }
        if (op == .reorder) {
            const x = T[s.inputs[0]];
            const o = T[s.outputs[0]];
            const src = try e.ref(s.inputs[0]);
            const out = try e.arenaOff(s.outputs[0]);
            if (x.layout == o.layout) {
                const count = o.storedCount(block);
                try e.emit(.copy, .{ .in = src.off, .out = out, .ik = src.k, .count = try u(count), .c = 0, .h = 0, .w = 0, .nb = 0 }, .{ count, 1, 1 });
                return;
            }
            if (x.shape.rank != 4) return error.Unsupported;
            const d = x.shape.d;
            const prm: LayoutP = .{ .in = src.off, .out = out, .ik = src.k, .count = 0, .c = try u(d[1]), .h = try u(d[2]), .w = try u(d[3]), .nb = try u(nblocks(d[1])) };
            if (o.layout == .blocked) {
                var q = prm;
                const count = o.storedCount(block);
                q.count = try u(count);
                try e.emit(.to_blocked, q, .{ count, 1, 1 });
            } else {
                var q = prm;
                const count = o.shape.count();
                q.count = try u(count);
                try e.emit(.to_plain, q, .{ count, 1, 1 });
            }
            return;
        }
        return error.Unsupported;
    }

    fn conv(e: *Encoder, s: plan_mod.Step, c: ir.Conv) Error!void {
        const T = e.p.tensors;
        const x = T[s.inputs[0]];
        const w = T[s.inputs[1]];
        const o = T[s.outputs[0]];
        if (x.shape.rank != 4 or o.shape.rank != 4 or o.layout != .blocked) return error.Unsupported;
        const wd = w.data orelse return error.Unsupported;
        const plain = x.layout == .plain;
        const cin = x.shape.d[1];
        const cout = o.shape.d[1];
        const kh: usize = c.kernel[0];
        const kw: usize = c.kernel[1];
        if (c.group == 0 or cin % c.group != 0 or cout % c.group != 0) return error.Unsupported;
        const cpg = cin / c.group;
        if (wd.len != cout * cpg * kh * kw) return error.Unsupported;
        const cinb = nblocks(cin);
        const coutb = nblocks(cout);

        const bias = try e.reserve(coutb * block);
        if (s.inputs[2] != plan_mod.none) {
            const bd = T[s.inputs[2]].data orelse return error.Unsupported;
            if (bd.len != cout) return error.Unsupported;
            @memcpy(bias.data[0..cout], bd);
        }
        const res = s.inputs.len > 3 and s.inputs[3] != plan_mod.none;
        if (res) {
            const r = T[s.inputs[3]];
            if (r.layout != .blocked or !r.shape.eql(o.shape)) return error.Unsupported;
        }
        const a = try actCode(c.act);
        var prm: ConvP = .{
            .x = try e.arenaOff(s.inputs[0]),
            .out = try e.arenaOff(s.outputs[0]),
            .wt = 0,
            .bias = bias.off,
            .res = if (res) try e.arenaOff(s.inputs[3]) else 0,
            .has_res = @intFromBool(res),
            .res_after = @intFromBool(c.residual_after_act),
            .act = a.code,
            .a0 = a.a0,
            .a1 = a.a1,
            .n = try u(x.shape.d[0]),
            .cin = try u(cin),
            .cinb = try u(cinb),
            .h = try u(x.shape.d[2]),
            .iw = try u(x.shape.d[3]),
            .cout = try u(cout),
            .coutb = try u(coutb),
            .oh = try u(o.shape.d[2]),
            .ow = try u(o.shape.d[3]),
            .kh = c.kernel[0],
            .kw = c.kernel[1],
            .sh = c.stride[0],
            .sw = c.stride[1],
            .pt = c.pad[0],
            .pl = c.pad[1],
            .dh = c.dilation[0],
            .dw = c.dilation[1],
            .group = c.group,
            .x_plain = @intFromBool(plain),
        };
        const n = x.shape.d[0];
        const ohw = .{ o.shape.d[3], o.shape.d[2], n * coutb };
        const kk = kh * kw;
        if (c.group == 1 and !plain and kh == 1 and kw == 1 and c.stride[0] == 1 and c.stride[1] == 1 and std.mem.allEqual(u32, &c.pad, 0) and x.shape.d[2] == o.shape.d[2] and x.shape.d[3] == o.shape.d[3]) {
            // [coutb][cinb] float4x4, column j = input lane j.
            const wp = try e.reserve(coutb * cinb * 16);
            for (0..cout) |oc| for (0..cin) |ic| {
                wp.data[(((oc / 4) * cinb + ic / 4) * 4 + ic % 4) * 4 + oc % 4] = wd[oc * cin + ic];
            };
            prm.wt = wp.off;
            try e.emit(.conv_pw, prm, .{ ceilDiv(x.shape.d[2] * x.shape.d[3], 4), coutb, n });
        } else if (c.group == cin and cout == cin and !plain) {
            // [blocks][kh*kw] float4.
            const wp = try e.reserve(coutb * kk * 4);
            for (0..cout) |ch| for (0..kk) |t| {
                wp.data[((ch / 4) * kk + t) * 4 + ch % 4] = wd[ch * kk + t];
            };
            prm.wt = wp.off;
            try e.emit(.conv_dw, prm, ohw);
        } else if (c.group == 1 and !plain) {
            // [coutb][cinb][kh][kw] float4x4.
            const wp = try e.reserve(coutb * cinb * kk * 16);
            for (0..cout) |oc| for (0..cin) |ic| for (0..kk) |t| {
                wp.data[((((oc / 4) * cinb + ic / 4) * kk + t) * 4 + ic % 4) * 4 + oc % 4] = wd[(oc * cin + ic) * kk + t];
            };
            prm.wt = wp.off;
            try e.emit(.conv_dir_b, prm, ohw);
        } else if (c.group == 1) {
            // [coutb][cin][kh][kw] float4.
            const wp = try e.reserve(coutb * cin * kk * 4);
            for (0..cout) |oc| for (0..cin) |ic| for (0..kk) |t| {
                wp.data[(((oc / 4) * cin + ic) * kk + t) * 4 + oc % 4] = wd[(oc * cin + ic) * kk + t];
            };
            prm.wt = wp.off;
            try e.emit(.conv_dir_p, prm, ohw);
        } else {
            prm.wt = try e.blob(wd);
            try e.emit(.conv_ref, prm, ohw);
        }
    }

    fn binary(e: *Encoder, s: plan_mod.Step, bop: ir.Binary) Error!void {
        const T = e.p.tensors;
        const a = T[s.inputs[0]];
        const b = T[s.inputs[1]];
        const o = T[s.outputs[0]];
        const blocked = o.layout == .blocked;
        var prm: BinP = .{ .a = 0, .b = 0, .out = try e.arenaOff(s.outputs[0]), .ak = 0, .bk = 0, .count = 0, .op = try binCode(bop), .blocked = @intFromBool(blocked), .c = 0, .hw = 1, .nb = 1, .const_first = 0, .rank = 0, .od = @splat(1), .sa = @splat(0), .sb = @splat(0) };
        if (blocked) {
            if (o.shape.rank != 4) return error.Unsupported;
            prm.c = try u(o.shape.d[1]);
            prm.hw = try u(o.shape.d[2] * o.shape.d[3]);
            prm.nb = try u(nblocks(o.shape.d[1]));
            const count = o.storedCount(block);
            prm.count = try u(count);
            if (a.data != null or b.data != null) {
                // Blocked tensor op per-channel (or scalar) constant.
                const cdata = a.data orelse b.data.?;
                const xt = if (a.data != null) s.inputs[1] else s.inputs[0];
                if (T[xt].layout != .blocked or !T[xt].shape.eql(o.shape)) return error.Unsupported;
                const ch = o.shape.d[1];
                if (cdata.len != 1 and cdata.len != ch) return error.Unsupported;
                const v = try e.reserve(nblocks(ch) * block);
                for (0..ch) |i| v.data[i] = if (cdata.len == 1) cdata[0] else cdata[i];
                prm.a = try e.arenaOff(xt);
                prm.b = v.off;
                prm.const_first = @intFromBool(a.data != null);
                try e.emit(.bin_pc, prm, .{ count, 1, 1 });
                return;
            }
            if (a.layout != .blocked or b.layout != .blocked or !a.shape.eql(o.shape) or !b.shape.eql(o.shape)) return error.Unsupported;
            prm.a = try e.arenaOff(s.inputs[0]);
            prm.b = try e.arenaOff(s.inputs[1]);
            try e.emit(.bin_same, prm, .{ count, 1, 1 });
            return;
        }
        if (a.layout != .plain or b.layout != .plain) return error.Unsupported;
        const ra = try e.ref(s.inputs[0]);
        const rb = try e.ref(s.inputs[1]);
        prm.a = ra.off;
        prm.ak = ra.k;
        prm.b = rb.off;
        prm.bk = rb.k;
        const count = o.shape.count();
        prm.count = try u(count);
        if (a.shape.eql(o.shape) and b.shape.eql(o.shape)) {
            try e.emit(.bin_same, prm, .{ count, 1, 1 });
            return;
        }
        // Numpy broadcasting: right-aligned axes, stride 0 where size 1.
        const r = o.shape.rank;
        if (a.shape.rank > r or b.shape.rank > r) return error.Unsupported;
        prm.rank = @intCast(r);
        for (0..r) |i| prm.od[i] = try u(o.shape.d[i]);
        for ([2]ir.Shape{ a.shape, b.shape }, [2]*[ir.max_rank]u32{ &prm.sa, &prm.sb }) |sh, st| {
            var stride: usize = 1;
            var i: usize = r;
            while (i > 0) {
                i -= 1;
                if (i + sh.rank < r) {
                    st[i] = 0;
                    continue;
                }
                const d = sh.d[i + sh.rank - r];
                if (d != 1 and d != o.shape.d[i]) return error.Unsupported;
                st[i] = if (d == 1) 0 else try u(stride);
                stride *= d;
            }
        }
        try e.emit(.bin_bc, prm, .{ count, 1, 1 });
    }

    fn concat(e: *Encoder, s: plan_mod.Step, axis: usize) Error!void {
        const T = e.p.tensors;
        const o = T[s.outputs[0]];
        const out = try e.arenaOff(s.outputs[0]);
        if (o.layout == .blocked) {
            if (o.shape.rank != 4 or axis != 1) return error.Unsupported;
            const hw4 = o.shape.d[2] * o.shape.d[3] * block;
            const onb = nblocks(o.shape.d[1]);
            var at: usize = 0;
            for (s.inputs, 0..) |t, i| {
                const x = T[t];
                if (x.layout != .blocked or x.data != null) return error.Unsupported;
                // Whole blocks per input: all but the last fill theirs.
                if (i + 1 < s.inputs.len and x.shape.d[1] % block != 0) return error.Unsupported;
                const nb = nblocks(x.shape.d[1]);
                try e.emit(.copy2d, .{ .in = try e.arenaOff(t), .out = out, .ik = 0, .outer = try u(o.shape.d[0]), .len = try u(nb * hw4), .orow = try u(onb * hw4), .at = try u(at * hw4) }, .{ nb * hw4, o.shape.d[0], 1 });
                at += nb;
            }
            return;
        }
        if (axis >= o.shape.rank) return error.Unsupported;
        var outer: usize = 1;
        for (0..axis) |i| outer *= o.shape.d[i];
        var inner: usize = 1;
        for (axis + 1..o.shape.rank) |i| inner *= o.shape.d[i];
        const orow = o.shape.d[axis] * inner;
        var at: usize = 0;
        for (s.inputs) |t| {
            const x = T[t];
            if (x.layout != .plain) return error.Unsupported;
            const len = x.shape.d[axis] * inner;
            const src = try e.ref(t);
            try e.emit(.copy2d, .{ .in = src.off, .out = out, .ik = src.k, .outer = try u(outer), .len = try u(len), .orow = try u(orow), .at = try u(at) }, .{ len, outer, 1 });
            at += len;
        }
    }
};

fn nblocks(c: usize) usize {
    return (c + block - 1) / block;
}

fn ceilDiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

const ActCode = struct { code: u32, a0: f32 = 0, a1: f32 = 0 };

/// The kernels' activation code (act4 in kernels.metal).
pub fn actCode(a: ir.Act) error{Unsupported}!ActCode {
    if (a == .none) return .{ .code = 0 };
    if (a == .relu) return .{ .code = 1 };
    if (a == .leaky_relu) return .{ .code = 2, .a0 = a.leaky_relu };
    if (a == .clip) return .{ .code = 3, .a0 = a.clip[0], .a1 = a.clip[1] };
    if (a == .sigmoid) return .{ .code = 4 };
    if (a == .hard_sigmoid) return .{ .code = 5, .a0 = a.hard_sigmoid[0], .a1 = a.hard_sigmoid[1] };
    if (a == .hard_swish) return .{ .code = 6 };
    if (a == .tanh) return .{ .code = 7 };
    if (a == .silu) return .{ .code = 8 };
    return error.Unsupported;
}

/// The kernels' binary op code (binop in kernels.metal).
fn binCode(op: ir.Binary) error{Unsupported}!u32 {
    const order = [_]ir.Binary{ .add, .sub, .mul, .div, .max, .min, .pow };
    for (order, 0..) |o, i| if (op == o) return @intCast(i);
    return error.Unsupported;
}

/// Encodes every step of `p` (planned with `block` and no groups).
pub fn encode(arena: Allocator, p: *const Plan) Error!Encoded {
    if (p.block != block or p.groups.len != 0) return error.Unsupported;
    if (p.arena_bytes / 4 > std.math.maxInt(u32)) return error.Unsupported;
    const const_off = try arena.alloc(u32, p.tensors.len);
    @memset(const_off, std.math.maxInt(u32));
    var e: Encoder = .{ .arena = arena, .p = p, .const_off = const_off };
    for (p.steps) |s| try e.step(s);
    if (e.consts.items.len > std.math.maxInt(u32)) return error.Unsupported;
    return .{ .dispatches = e.dispatches.items, .consts = e.consts.items, .arena_bytes = p.arena_bytes };
}

// ------------------------------------------------------------------ tests

test "generated MSL structs mirror the Zig parameter blocks" {
    try std.testing.expect(std.mem.indexOf(u8, msl_structs, "struct ConvP {\n    uint x;\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl_structs, "    float a0;\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl_structs, "    uint od[6];\n") != null);
    // Every kernel the enum names is defined in the source.
    inline for (comptime std.enums.values(Kernel)) |k| {
        try std.testing.expect(std.mem.indexOf(u8, kernels_body, "kernel void " ++ comptime k.name() ++ "(") != null);
    }
    try std.testing.expectEqual(@as(u8, 0), source.ptr[source.len]);
}
