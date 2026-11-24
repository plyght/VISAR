//! TEMPORARY per-item profiler (not committed).
const std = @import("std");
const dnn = @import("dnn/dnn.zig");
const cpu = @import("dnn/cpu/backend.zig");
const core = @import("core/core.zig");

fn geomOf(k: anytype) ?@import("dnn/cpu/conv.zig").Geom {
    return switch (k.*) {
        .pointwise => |x| x.g,
        .depthwise => |x| x.g,
        .direct => |x| x.g,
        .blocked => |x| x.g,
        .reference => |x| x.g,
        else => null,
    };
}

fn describe(w: anytype, k: anytype) !f64 {
    try w.print("{s}", .{@tagName(k.*)});
    if (geomOf(k)) |g| {
        try w.print("[{d}->{d} {d}x{d} k{d}x{d} s{d} g{d} out{d}x{d}]", .{ g.cin, g.cout, g.h, g.w, g.kh, g.kw, g.sh, g.group, g.oh, g.ow });
        switch (k.*) {
            inline .pointwise, .depthwise, .direct, .blocked => |x| try w.print("<{s}>", .{@tagName(x.epi.act)}),
            else => {},
        }
        return 2.0 * @as(f64, @floatFromInt(g.cout * g.oh * g.ow * (g.cin / g.group) * g.kh * g.kw));
    }
    return 0;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const gpa = std.heap.page_allocator;
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const path = args[1];
    const size = try std.fmt.parseInt(usize, args[2], 10);
    const iters = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 30;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    var net = try dnn.Net.load(gpa, bytes);
    if (args.len > 4) net.opts.stream = false;
    try net.plan(&.{&.{ 1, 3, size, size }});
    var prng = std.Random.DefaultPrng.init(1);
    for (net.input(0)) |*v| v.* = prng.random().float(f32) * 255;
    const c = &net.compiled.?;
    const prog = &c.program;
    const base: [*]f32 = @ptrCast(c.buf.ptr);
    const times = try a.alloc(u64, prog.items.len);
    @memset(times, std.math.maxInt(u64));
    for (0..3) |_| net.run(.serial);
    var total: u64 = 0;
    for (0..iters) |_| {
        for (prog.items, times) |*it, *t| {
            const t0 = std.Io.Clock.awake.now(io);
            it.run(core.Exec.serial, base);
            const t1 = std.Io.Clock.awake.now(io);
            const d: u64 = @intCast(t0.durationTo(t1).nanoseconds);
            t.* = @min(t.*, d);
        }
    }
    var buf: [1 << 16]u8 = undefined;
    var fw = std.Io.File.Writer.initStreaming(.stdout(), io, &buf);
    const w = &fw.interface;
    for (times) |t| total += t;
    const per_total = @as(f64, @floatFromInt(total)) / 1e6;
    try w.print("total(sum of per-item min) {d:.3} ms\n", .{per_total});
    for (prog.items, times, 0..) |*it, t, i| {
        const ms = @as(f64, @floatFromInt(t)) / 1e6;
        try w.print("{d:3} {d:7.3} ms {d:5.1}% ", .{ i, ms, 100 * ms / per_total });
        var flops: f64 = 0;
        switch (it.*) {
            .kernel => |k| flops += try describe(w, k),
            .chain => |ch| {
                try w.print("chain{{ ", .{});
                for (ch.stages) |st| {
                    flops += try describe(w, st.k);
                    try w.print(" ", .{});
                }
                try w.print("}}", .{});
            },
        }
        if (flops > 0) try w.print("  {d:.1} GFLOP/s", .{flops / (ms * 1e6)});
        try w.print("\n", .{});
    }
    try w.flush();
}
