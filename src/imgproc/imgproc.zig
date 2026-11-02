//! visar.imgproc: image processing.

const std = @import("std");

pub const border = @import("border.zig");
pub const color = @import("color.zig");
pub const resize_mod = @import("resize.zig");
pub const filter = @import("filter.zig");
pub const threshold_mod = @import("threshold.zig");
pub const morph = @import("morph.zig");
pub const geometry = @import("geometry.zig");
pub const hist = @import("hist.zig");

pub const Border = border.Border;

pub const ColorCode = color.Code;
pub const cvtColor = color.cvtColor;

pub const Interp = resize_mod.Interp;
pub const resize = resize_mod.resize;

pub const gaussianBlur = filter.gaussianBlur;
pub const boxFilter = filter.boxFilter;
pub const blur = filter.blur;
pub const sobel = filter.sobel;
pub const sepFilter2D = filter.sepFilter2D;

pub const ThresholdType = threshold_mod.Type;
pub const threshold = threshold_mod.threshold;
pub const inRange = threshold_mod.inRange;
pub const applyLut = threshold_mod.applyLut;

pub const MorphShape = morph.Shape;
pub const MorphOp = morph.MorphOp;
pub const structuringElement = morph.structuringElement;
pub const erode = morph.erode;
pub const dilate = morph.dilate;
pub const morphologyEx = morph.morphologyEx;

pub const flip = geometry.flip;
pub const rotate = geometry.rotate;
pub const transpose = geometry.transpose;
pub const copyMakeBorder = geometry.copyMakeBorder;
pub const warpAffine = geometry.warpAffine;
pub const warpPerspective = geometry.warpPerspective;
pub const rotationMatrix2D = geometry.rotationMatrix2D;

pub const calcHist = hist.calcHist;
pub const equalizeHist = hist.equalizeHist;
pub const integral = hist.integral;

test {
    std.testing.refAllDecls(@This());
}
