//! The Metal side of the backend (macOS only): device, command queue and
//! compute pipelines (created once per process), and per plan one arena
//! buffer, one constants buffer and the encoded dispatch list, run as one
//! command buffer per inference.
//!
//! No SDK and no Objective-C source: Metal.framework is opened at run time
//! and driven through the Objective-C runtime (videoio/objc.zig), like
//! VISAR's AVFoundation and AppKit code. The kernels are MSL source
//! (encode.source) compiled at start-up with newLibraryWithSource.
//!
//! Ownership follows Cocoa's rules (no ARC here): MTLCreateSystemDefaultDevice
//! and every `new...` method return +1 references, released by us;
//! commandBuffer, computeCommandEncoder, error and name return autoreleased
//! objects, so every entry point runs inside an autorelease pool. No call
//! returns a struct, so objc_msgSend_stret (x86_64) is never needed; MTLSize
//! (24 bytes) is passed by value, which Zig's C calling convention lays out
//! as clang does for both arm64 and x86_64.

const std = @import("std");
const objc = @import("../../videoio/objc.zig");
const encode = @import("encode.zig");
const Allocator = std.mem.Allocator;

const id = objc.id;
const SEL = objc.SEL;
const NSUInteger = objc.NSUInteger;
const NSInteger = objc.NSInteger;

pub const Error = error{ Unavailable, Unsupported, GpuFailed, OutOfMemory };

pub const MTLSize = extern struct { width: NSUInteger, height: NSUInteger, depth: NSUInteger };

comptime {
    std.debug.assert(@sizeOf(MTLSize) == 3 * @sizeOf(usize));
}

/// MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache |
/// MTLResourceHazardTrackingModeDefault: CPU and GPU share the pages.
const storage_shared: NSUInteger = 0;
/// MTLCommandBufferStatusCompleted.
const status_completed: NSUInteger = 4;
/// MTLMathModeSafe; MTLMathFloatingPointFunctionsPrecise.
const math_mode_safe: NSInteger = 0;
const math_functions_precise: NSInteger = 1;

const Sels = struct {
    command_buffer: SEL,
    compute_encoder: SEL,
    set_buffer: SEL,
    set_pipeline: SEL,
    set_bytes: SEL,
    dispatch: SEL,
    end_encoding: SEL,
    commit: SEL,
    wait: SEL,
    status: SEL,
    err: SEL,

    fn init() Sels {
        return .{
            .command_buffer = objc.sel("commandBuffer"),
            .compute_encoder = objc.sel("computeCommandEncoder"),
            .set_buffer = objc.sel("setBuffer:offset:atIndex:"),
            .set_pipeline = objc.sel("setComputePipelineState:"),
            .set_bytes = objc.sel("setBytes:length:atIndex:"),
            .dispatch = objc.sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            .end_encoding = objc.sel("endEncoding"),
            .commit = objc.sel("commit"),
            .wait = objc.sel("waitUntilCompleted"),
            .status = objc.sel("status"),
            .err = objc.sel("error"),
        };
    }
};

pub const Context = struct {
    device: id,
    queue: id,
    library: id,
    pipelines: [encode.kernel_count]id,
    /// maxTotalThreadsPerThreadgroup of each pipeline.
    max_threads: [encode.kernel_count]u32,
    max_buffer: usize,
    sels: Sels,
    name_buf: [128]u8,
    name_len: usize,

    pub fn name(c: *const Context) []const u8 {
        return c.name_buf[0..c.name_len];
    }
};

var ctx: Context = undefined;
/// 0 untried, 1 initialising, 2 ready, 3 failed.
var state: std.atomic.Value(u8) = .init(0);
var reason_buf: [512]u8 = undefined;
var reason_len: usize = 0;
var gpu_error_buf: [512]u8 = undefined;
var gpu_error_len: usize = 0;

fn noteReason(comptime fmt: []const u8, args: anytype) void {
    const s = std.fmt.bufPrint(&reason_buf, fmt, args) catch reason_buf[0..];
    reason_len = s.len;
}

/// Why the device is unavailable (after `context` returned null).
pub fn unavailableReason() []const u8 {
    return reason_buf[0..reason_len];
}

/// The last command buffer error (after `run` failed).
pub fn lastGpuError() []const u8 {
    return gpu_error_buf[0..gpu_error_len];
}

/// The process-wide Metal context, created on first use (from any thread);
/// null when Metal or a device is unavailable, or the kernels failed to
/// compile.
pub fn context() ?*const Context {
    while (true) {
        switch (state.load(.acquire)) {
            2 => return &ctx,
            3 => return null,
            1 => std.atomic.spinLoopHint(),
            else => if (state.cmpxchgStrong(0, 1, .acquire, .monotonic) == null) {
                const ok = if (initContext(&ctx)) true else |_| false;
                state.store(if (ok) 2 else 3, .release);
            },
        }
    }
}

fn initContext(c: *Context) error{Unavailable}!void {
    _ = objc.load() catch {
        noteReason("cannot load the Objective-C runtime ({s})", .{objc.missing()});
        return error.Unavailable;
    };
    // Command-line processes on older macOS only get a device once
    // CoreGraphics is loaded; harmless elsewhere.
    _ = objc.open(.coregraphics) catch {};
    const create = objc.func(fn () callconv(.c) id, .metal, "MTLCreateSystemDefaultDevice") catch {
        noteReason("cannot load Metal.framework ({s})", .{objc.missing()});
        return error.Unavailable;
    };
    const pool = objc.Pool.push();
    defer pool.pop();

    var dev = create();
    if (dev == null) {
        if (objc.func(fn () callconv(.c) id, .metal, "MTLCopyAllDevices")) |copy_all| {
            const all = copy_all();
            if (objc.count(all) > 0) dev = objc.retain(objc.at(all, 0));
            objc.release(all);
        } else |_| {}
    }
    if (dev == null) {
        noteReason("no Metal device (MTLCreateSystemDefaultDevice returned nil)", .{});
        return error.Unavailable;
    }
    errdefer objc.release(dev);
    const queue = objc.send(id, dev, "newCommandQueue", .{});
    if (queue == null) {
        noteReason("newCommandQueue failed", .{});
        return error.Unavailable;
    }
    errdefer objc.release(queue);

    const opts_cls = objc.class("MTLCompileOptions") catch {
        noteReason("no MTLCompileOptions class", .{});
        return error.Unavailable;
    };
    const opts = objc.new(opts_cls);
    if (opts == null) {
        noteReason("cannot create MTLCompileOptions", .{});
        return error.Unavailable;
    }
    defer objc.release(opts);
    // IEEE-safe arithmetic, close to the CPU reference: mathMode (macOS 15)
    // or the older fastMathEnabled; precise exp/tanh where selectable.
    if (responds(opts, "setMathMode:")) {
        objc.send(void, opts, "setMathMode:", .{math_mode_safe});
    } else {
        objc.send(void, opts, "setFastMathEnabled:", .{objc.NO});
    }
    if (responds(opts, "setMathFloatingPointFunctions:")) objc.send(void, opts, "setMathFloatingPointFunctions:", .{math_functions_precise});

    const src = objc.nsString(encode.source.ptr);
    if (src == null) {
        noteReason("cannot create the kernel source string", .{});
        return error.Unavailable;
    }
    var err: id = null;
    const library = objc.send(id, dev, "newLibraryWithSource:options:error:", .{ src, opts, @as(*id, &err) });
    if (library == null) {
        noteReason("kernel compilation failed: {s}", .{objc.errorText(err)});
        return error.Unavailable;
    }
    errdefer objc.release(library);

    var pipelines: [encode.kernel_count]id = @splat(null);
    var max_threads: [encode.kernel_count]u32 = undefined;
    errdefer for (pipelines) |p| objc.release(p);
    inline for (comptime std.enums.values(encode.Kernel), 0..) |k, i| {
        const fname = objc.nsString(comptime k.name().ptr);
        const func = objc.send(id, library, "newFunctionWithName:", .{fname});
        if (func == null) {
            noteReason("kernel {s} missing from the compiled library", .{comptime k.name()});
            return error.Unavailable;
        }
        defer objc.release(func);
        err = null;
        const pso = objc.send(id, dev, "newComputePipelineStateWithFunction:error:", .{ func, @as(*id, &err) });
        if (pso == null) {
            noteReason("pipeline {s} failed: {s}", .{ comptime k.name(), objc.errorText(err) });
            return error.Unavailable;
        }
        pipelines[i] = pso;
        const mt = objc.send(NSUInteger, pso, "maxTotalThreadsPerThreadgroup", .{});
        max_threads[i] = @intCast(@min(@max(mt, 1), 1024));
    }

    c.* = .{
        .device = dev,
        .queue = queue,
        .library = library,
        .pipelines = pipelines,
        .max_threads = max_threads,
        .max_buffer = objc.send(NSUInteger, dev, "maxBufferLength", .{}),
        .sels = .init(),
        .name_buf = undefined,
        .name_len = 0,
    };
    const nm = objc.utf8(objc.send(id, dev, "name", .{}));
    c.name_len = @min(nm.len, c.name_buf.len);
    @memcpy(c.name_buf[0..c.name_len], nm[0..c.name_len]);
}

fn responds(obj: id, comptime selector: [:0]const u8) bool {
    return objc.send(objc.BOOL, obj, "respondsToSelector:", .{objc.sel(selector)}) != objc.NO;
}

fn pow2Ceil(x: u32) u32 {
    var p: u32 = 1;
    while (p < x and p < (1 << 30)) p <<= 1;
    return p;
}

/// Threads per threadgroup for a grid: up to 128 threads (four SIMD
/// groups), filled along x first.
fn threadgroup(grid: [3]u32, max_total: u32) MTLSize {
    const budget = @max(1, @min(max_total, 128));
    const tx = @min(pow2Ceil(grid[0]), budget);
    const ty = @min(pow2Ceil(grid[1]), @max(1, budget / tx));
    const tz = @min(pow2Ceil(grid[2]), @max(1, budget / (tx * ty)), 64);
    return .{ .width = tx, .height = ty, .depth = tz };
}

pub const Program = struct {
    ctx: *const Context,
    arena_buf: id,
    consts_buf: id,
    items: []const Item,
    mem: []align(64) u8,

    const Item = struct {
        pso: id,
        params: []align(4) const u8,
        groups: MTLSize,
        threads: MTLSize,
    };

    /// Allocates the arena (zeroed) and constant buffers and resolves
    /// every dispatch's pipeline and threadgroup shape.
    pub fn init(arena: Allocator, enc: encode.Encoded) Error!Program {
        const c = context() orelse return error.Unavailable;
        const pool = objc.Pool.push();
        defer pool.pop();
        const abytes = std.mem.alignForward(usize, @max(enc.arena_bytes, 64), 64);
        const cbytes = std.mem.alignForward(usize, @max(enc.consts.len * @sizeOf(f32), 64), 64);
        if (abytes > c.max_buffer or cbytes > c.max_buffer) return error.Unsupported;
        const abuf = newBuffer(c, abytes) orelse return error.OutOfMemory;
        errdefer objc.release(abuf);
        const kbuf = newBuffer(c, cbytes) orelse return error.OutOfMemory;
        errdefer objc.release(kbuf);
        const amem = contents(abuf) orelse return error.OutOfMemory;
        const kmem = contents(kbuf) orelse return error.OutOfMemory;
        @memset(amem[0..abytes], 0);
        @memset(kmem[0..cbytes], 0);
        @memcpy(kmem[0 .. enc.consts.len * @sizeOf(f32)], std.mem.sliceAsBytes(enc.consts));

        const items = try arena.alloc(Item, enc.dispatches.len);
        for (enc.dispatches, items) |d, *it| {
            const ki = @backingInt(d.kernel);
            const tg = threadgroup(d.grid, c.max_threads[ki]);
            it.* = .{
                .pso = c.pipelines[ki],
                .params = d.params,
                .threads = tg,
                .groups = .{
                    .width = (d.grid[0] + tg.width - 1) / tg.width,
                    .height = (d.grid[1] + tg.height - 1) / tg.height,
                    .depth = (d.grid[2] + tg.depth - 1) / tg.depth,
                },
            };
        }
        return .{ .ctx = c, .arena_buf = abuf, .consts_buf = kbuf, .items = items, .mem = @alignCast(amem[0..abytes]) };
    }

    pub fn deinit(p: *Program) void {
        const pool = objc.Pool.push();
        defer pool.pop();
        objc.release(p.arena_buf);
        objc.release(p.consts_buf);
        p.* = undefined;
    }

    /// The arena as host memory (inputs are written and outputs read here,
    /// in place: the buffer is shared with the GPU).
    pub fn memory(p: *const Program) []align(64) u8 {
        return p.mem;
    }

    /// One inference: every dispatch in one compute encoder (serial
    /// dispatch, so each kernel sees its predecessors' writes) in one
    /// command buffer, committed and waited for.
    pub fn run(p: *const Program) Error!void {
        const c = p.ctx;
        const s = &c.sels;
        const pool = objc.Pool.push();
        defer pool.pop();
        const cb = objc.sendSel(id, c.queue, s.command_buffer, .{});
        if (cb == null) return gpuFailed("commandBuffer returned nil");
        const enc = objc.sendSel(id, cb, s.compute_encoder, .{});
        if (enc == null) return gpuFailed("computeCommandEncoder returned nil");
        objc.sendSel(void, enc, s.set_buffer, .{ p.arena_buf, @as(NSUInteger, 0), @as(NSUInteger, 0) });
        objc.sendSel(void, enc, s.set_buffer, .{ p.consts_buf, @as(NSUInteger, 0), @as(NSUInteger, 1) });
        for (p.items) |it| {
            objc.sendSel(void, enc, s.set_pipeline, .{it.pso});
            objc.sendSel(void, enc, s.set_bytes, .{ @as(*const anyopaque, it.params.ptr), @as(NSUInteger, it.params.len), @as(NSUInteger, 2) });
            objc.sendSel(void, enc, s.dispatch, .{ it.groups, it.threads });
        }
        objc.sendSel(void, enc, s.end_encoding, .{});
        objc.sendSel(void, cb, s.commit, .{});
        objc.sendSel(void, cb, s.wait, .{});
        const st = objc.sendSel(NSUInteger, cb, s.status, .{});
        if (st != status_completed) {
            const e = objc.sendSel(id, cb, s.err, .{});
            const text = objc.errorText(e);
            const out = std.fmt.bufPrint(&gpu_error_buf, "command buffer status {d}: {s}", .{ st, text }) catch gpu_error_buf[0..];
            gpu_error_len = out.len;
            return error.GpuFailed;
        }
    }
};

fn gpuFailed(msg: []const u8) error{GpuFailed} {
    const n = @min(msg.len, gpu_error_buf.len);
    @memcpy(gpu_error_buf[0..n], msg[0..n]);
    gpu_error_len = n;
    return error.GpuFailed;
}

fn newBuffer(c: *const Context, bytes: usize) id {
    return objc.send(id, c.device, "newBufferWithLength:options:", .{ @as(NSUInteger, bytes), storage_shared });
}

fn contents(buf: id) ?[*]align(64) u8 {
    const p = objc.send(?*anyopaque, buf, "contents", .{}) orelse return null;
    // Shared buffers are page aligned.
    if (@intFromPtr(p) % 64 != 0) return null;
    return @ptrCast(@alignCast(p));
}

test "threadgroup shapes stay within limits" {
    const t = threadgroup(.{ 1000, 3, 2 }, 1024);
    try std.testing.expectEqual(@as(usize, 128), t.width * t.height * t.depth);
    const v = threadgroup(.{ 1, 1, 500 }, 1024);
    try std.testing.expectEqual(@as(usize, 64), v.depth);
    const w = threadgroup(.{ 5, 1, 1 }, 32);
    try std.testing.expectEqual(@as(usize, 8), w.width);
}
