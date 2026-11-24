//! visar.dnn: neural-network inference.
//!
//! Pipeline: `onnx` (protobuf model reader) -> `ir` (graph of typed ops;
//! shape inference and constant folding) -> `passes` (BatchNorm folding,
//! epilogue fusion, dead-code elimination) -> `plan` (layouts and a
//! liveness-based memory plan with every shape fixed: device-neutral data a
//! backend consumes whole) -> `cpu` (the CPU backend: SIMD kernels with
//! weights packed at plan time, threaded through `Exec`).
//!
//! `Net` ties the stages together; `FaceDetector` runs YuNet the way
//! OpenCV's FaceDetectorYN does; `ObjectDetector` runs the Model Zoo's
//! NanoDet and YOLOX detectors the way its Python demos do.

const std = @import("std");

pub const proto = @import("proto.zig");
pub const onnx = @import("onnx.zig");
pub const ir = @import("ir.zig");
pub const passes = @import("passes.zig");
pub const plan = @import("plan.zig");
pub const cpu = @import("cpu/backend.zig");
pub const net = @import("net.zig");
pub const face = @import("face.zig");
pub const detect = @import("detect.zig");

pub const Net = net.Net;
pub const Tensor = net.Tensor;
pub const FaceDetector = face.FaceDetector;
pub const Face = face.Face;
pub const ObjectDetector = detect.ObjectDetector;
pub const Detection = detect.Detection;

test {
    std.testing.refAllDecls(@This());
    _ = proto;
    _ = onnx;
    _ = ir;
    _ = plan;
    _ = net;
    _ = face;
    _ = detect;
    _ = @import("tests.zig");
    _ = @import("cpu/vec.zig");
}
