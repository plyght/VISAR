//! visar.video: motion analysis and tracking.

const std = @import("std");

pub const track = @import("track.zig");
pub const Background = track.Background;
pub const Tracker = track.Tracker;
pub const Box = track.Box;

test {
    std.testing.refAllDecls(@This());
}
