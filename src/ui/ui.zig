//! visar.ui: showing images in a native window.
//!
//! macOS: AppKit through the Objective-C runtime, loaded at run time
//! (window_macos.zig). Other platforms have no window backend yet:
//! `supported` is false and `Window` is not available.

const std = @import("std");
const builtin = @import("builtin");

pub const supported = builtin.os.tag == .macos;

pub const window_macos = if (supported) @import("window_macos.zig") else struct {};

/// The native window (macOS only; check `supported`).
pub const Window = if (supported) window_macos.Window else void;
pub const Input = if (supported) window_macos.Input else struct {
    quit: bool = false,
    reset: bool = false,
    toggle_hud: bool = false,
    clear: bool = false,
    cycle_mode: bool = false,
    selection: ?[4]f32 = null,
    dragging: ?[4]f32 = null,
};

test {
    if (supported) std.testing.refAllDecls(window_macos);
}
