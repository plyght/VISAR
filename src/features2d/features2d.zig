//! visar.features2d: keypoint detection, description and matching
//! (OpenCV's features2d): FAST, ORB, brute-force matching and drawing.

const std = @import("std");

pub const keypoint = @import("keypoint.zig");
pub const fast_mod = @import("fast.zig");

pub const KeyPoint = keypoint.KeyPoint;
pub const DMatch = keypoint.DMatch;
pub const retainBest = keypoint.retainBest;

pub const FastType = fast_mod.Type;
pub const FastOptions = fast_mod.Options;
pub const fast = fast_mod.detect;

pub const orb = @import("orb.zig");
pub const OrbOptions = orb.Options;
pub const OrbFeatures = orb.Features;
pub const orbDetectAndCompute = orb.detectAndCompute;

pub const matcher = @import("matcher.zig");
pub const Norm = matcher.Norm;
pub const knnMatch = matcher.knnMatch;
pub const match = matcher.match;
pub const ratioTest = matcher.ratioTest;

test {
    std.testing.refAllDecls(@This());
}
