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
pub const components = @import("components.zig");
pub const draw = @import("draw.zig");
pub const contours = @import("contours.zig");
pub const polygon = @import("polygon.zig");
pub const shape = @import("shape.zig");
pub const pyramid = @import("pyramid.zig");
pub const corner = @import("corner.zig");

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

pub const pyrDown = pyramid.pyrDown;
pub const pyrUp = pyramid.pyrUp;
pub const buildPyramid = pyramid.buildPyramid;
pub const Pyramid = pyramid.Pyramid;

pub const Point2f = corner.Point2f;
pub const cornerMinEigenVal = corner.cornerMinEigenVal;
pub const cornerHarris = corner.cornerHarris;
pub const goodFeaturesToTrack = corner.goodFeaturesToTrack;

pub const connectedComponentsWithStats = components.connectedComponentsWithStats;

pub const Point = contours.Point;
pub const Contours = contours.Contours;
pub const ContourMode = contours.Mode;
pub const ContourMethod = contours.Method;
pub const findContours = contours.findContours;
pub const LineType = polygon.LineType;
pub const drawContours = polygon.drawContours;
pub const fillPoly = polygon.fillPoly;
pub const fillConvexPoly = polygon.fillConvexPoly;
pub const polylines = polygon.polylines;

pub const Rect = shape.Rect;
pub const RotatedRect = shape.RotatedRect;
pub const Moments = shape.Moments;
pub const contourMoments = shape.contourMoments;
pub const imageMoments = shape.imageMoments;
pub const huMoments = shape.huMoments;
pub const matchShapes = shape.matchShapes;
pub const contourArea = shape.contourArea;
pub const arcLength = shape.arcLength;
pub const boundingRect = shape.boundingRect;
pub const maskBoundingRect = shape.maskBoundingRect;
pub const convexHull = shape.convexHull;
pub const convexityDefects = shape.convexityDefects;
pub const isContourConvex = shape.isContourConvex;
pub const minAreaRect = shape.minAreaRect;
pub const boxPoints = shape.boxPoints;
pub const minEnclosingCircle = shape.minEnclosingCircle;
pub const approxPolyDP = shape.approxPolyDP;
pub const pointPolygonTest = shape.pointPolygonTest;
pub const fitEllipse = shape.fitEllipse;

test {
    std.testing.refAllDecls(@This());
}
