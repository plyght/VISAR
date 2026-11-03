//! visar.videoio: video input.
//!
//!   camera.zig        live capture, one interface for every platform
//!   camera_macos.zig  AVFoundation through the Objective-C runtime
//!   camera_linux.zig  V4L2
//!   frame.zig         FrameRing (lock-free frame passing between threads)
//!                     and the fused gray + downscale used for detection
//!   objc.zig          the Objective-C runtime loader (macOS)

const std = @import("std");
const builtin = @import("builtin");

pub const camera = @import("camera.zig");
pub const frame = @import("frame.zig");
pub const Camera = camera.Camera;
pub const FrameRing = frame.FrameRing;

/// The Objective-C runtime layer (macOS only).
pub const objc = if (builtin.os.tag == .macos) @import("objc.zig") else struct {};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(camera);
    std.testing.refAllDecls(frame);
    // The ABI helpers are plain Zig and are tested on every host.
    _ = @import("objc.zig");
    if (camera.backend == .v4l2) _ = @import("camera_linux.zig");
}
