//! The Metal backend (PRD §8.6): runs a whole planned graph on the GPU.
//!
//! - One shared arena buffer laid out by the plan's memory plan (Apple
//!   silicon has unified memory: the CPU writes inputs and reads outputs in
//!   place, no copies), one buffer of constants uploaded at plan time
//!   (weights packed for the kernels), and one command buffer per inference
//!   encoding every step's dispatch. No per-op round trips.
//! - All or nothing: if any step of the plan has no kernel, `Program.init`
//!   fails with error.Unsupported and the caller plans for the CPU instead.
//! - f32 only, IEEE-safe math; checked against the CPU backend.
//!
//! `encode.zig` (device-neutral, compiled everywhere) lowers the plan to
//! kernels, parameters and grids; `device.zig` (macOS only) drives Metal
//! through the Objective-C runtime with no SDK, compiling kernels.metal at
//! start-up. Elsewhere everything here compiles away and `available()` is
//! false.

const std = @import("std");
const builtin = @import("builtin");
const plan_mod = @import("../plan.zig");
const Allocator = std.mem.Allocator;

pub const encode = @import("encode.zig");
/// An in-memory ONNX model exercising every kernel (for tests).
pub const test_model = @import("test_model.zig");

/// Whether this target has a Metal backend at all.
pub const supported = builtin.os.tag == .macos;

const device = if (supported) @import("device.zig") else struct {};

/// Channel block of the plans this backend runs.
pub const block = encode.block;

/// `auto` picks Metal for plans with at least this many convolution
/// multiply-adds: below it, launch and synchronisation overhead (a few
/// hundred microseconds per inference) outweighs the GPU's throughput.
pub const auto_min_macs: u64 = 50_000_000;

pub const Error = error{
    /// No Metal device on this machine (or not macOS).
    Unavailable,
    /// A step of the plan has no Metal kernel.
    Unsupported,
    /// The command buffer failed (see `lastGpuError`).
    GpuFailed,
    OutOfMemory,
};

/// Whether a Metal device is usable (creates the process-wide context and
/// compiles the kernels on first call).
pub fn available() bool {
    if (!supported) return false;
    return device.context() != null;
}

/// The GPU's name ("" when unavailable).
pub fn deviceName() []const u8 {
    if (!supported) return "";
    const c = device.context() orelse return "";
    return c.name();
}

/// Why Metal is unavailable.
pub fn unavailableReason() []const u8 {
    if (!supported) return "Metal needs macOS";
    if (device.context() != null) return "";
    return device.unavailableReason();
}

/// The last command-buffer error.
pub fn lastGpuError() []const u8 {
    if (!supported) return "";
    return device.lastGpuError();
}

/// A plan prepared for the GPU.
pub const Program = struct {
    impl: if (supported) device.Program else void,

    /// Encodes `p` (planned with `block` channels and no streamed groups)
    /// and uploads its constants. Allocations go to `arena`.
    pub fn init(arena: Allocator, p: *const plan_mod.Plan) Error!Program {
        if (!supported) return error.Unavailable;
        if (!available()) return error.Unavailable;
        const enc = try encode.encode(arena, p);
        return .{ .impl = try device.Program.init(arena, enc) };
    }

    pub fn deinit(prog: *Program) void {
        if (supported) prog.impl.deinit();
    }

    /// The arena (`plan.arena_bytes`, zeroed at init) as host memory.
    pub fn memory(prog: *const Program) []align(64) u8 {
        if (!supported) unreachable;
        return prog.impl.memory();
    }

    /// Runs one inference over the arena and waits for it.
    pub fn run(prog: *const Program) Error!void {
        if (!supported) unreachable;
        return prog.impl.run();
    }
};

test {
    _ = encode;
    _ = test_model;
    if (supported) std.testing.refAllDecls(device);
}
