//! VISAR: Visual Imaging, Sensing, and Recognition.
//! A dependency-free computer-vision library written in Zig.

const std = @import("std");

pub const version = std.SemanticVersion{ .major = 0, .minor = 1, .patch = 0 };

pub const core = @import("core/core.zig");
pub const imgproc = @import("imgproc/imgproc.zig");
pub const codecs = @import("codecs/codecs.zig");

pub const Image = core.Image;
pub const AnyImage = core.AnyImage;
pub const Depth = core.Depth;
pub const Exec = core.Exec;
pub const Pool = core.Pool;

test {
    std.testing.refAllDecls(@This());
    _ = core;
    _ = imgproc;
    _ = codecs;
}
