//! visar.calib3d: camera geometry, OpenCV's calib3d equivalent.
//! Homographies and affine estimation (RANSAC / LMeDS + Levenberg-Marquardt),
//! pose estimation, camera calibration and lens undistortion.

const std = @import("std");

pub const linalg = @import("linalg.zig");
pub const ransac = @import("ransac.zig");
pub const homography = @import("homography.zig");
pub const camera = @import("camera.zig");
pub const pose = @import("pose.zig");
pub const undistort_mod = @import("undistort.zig");

pub const Point2f = homography.Point2f;
pub const Method = homography.Method;
pub const findHomography = homography.findHomography;
pub const getPerspectiveTransform = homography.getPerspectiveTransform;
pub const getAffineTransform = homography.getAffineTransform;
pub const perspectiveTransform = homography.perspectiveTransform;
pub const estimateAffine2D = homography.estimateAffine2D;
pub const estimateAffinePartial2D = homography.estimateAffinePartial2D;

pub const rodrigues = camera.rodrigues;
pub const rodriguesInv = camera.rodriguesInv;
pub const projectPoints = camera.projectPoints;
pub const undistortPoints = camera.undistortPoints;
pub const solvePnP = pose.solvePnP;
pub const solvePnPRansac = pose.solvePnPRansac;

pub const undistort = undistort_mod.undistort;
pub const initUndistortRectifyMap = undistort_mod.initUndistortRectifyMap;
pub const remap = undistort_mod.remap;

test {
    std.testing.refAllDecls(@This());
}
