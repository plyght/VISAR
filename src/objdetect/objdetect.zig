//! visar.objdetect: OpenCV's objdetect equivalents. QR code generation,
//! detection and decoding, and ArUco marker dictionaries, generation and
//! detection.

const std = @import("std");

pub const gf256 = @import("gf256.zig");
pub const qr_tables = @import("qr_tables.zig");
pub const qr_encode = @import("qr_encode.zig");
pub const qr_decode = @import("qr_decode.zig");
pub const qr_detect = @import("qr_detect.zig");

pub const QrCode = qr_detect.Code;
pub const QrDetectOptions = qr_detect.Options;
pub const qrDetect = qr_detect.detect;
pub const freeQrCodes = qr_detect.freeCodes;

pub const QrLevel = qr_encode.Level;
pub const QrMode = qr_encode.Mode;
pub const QrEncodeOptions = qr_encode.Options;
pub const qrEncode = qr_encode.encode;
pub const qrEncodeStructuredAppend = qr_encode.encodeStructuredAppend;

pub const aruco_dict = @import("aruco_dict.zig");
pub const aruco = @import("aruco.zig");
pub const ArucoDictionary = aruco.Dictionary;
pub const PredefinedDictionary = aruco.PredefinedDictionary;
pub const getPredefinedDictionary = aruco.getPredefinedDictionary;
pub const generateImageMarker = aruco.generateImageMarker;
pub const ArucoDetectorParameters = aruco.DetectorParameters;
pub const ArucoMarkers = aruco.Markers;
pub const detectMarkers = aruco.detectMarkers;
pub const drawDetectedMarkers = aruco.drawDetectedMarkers;

test {
    _ = @import("dbg_test.zig");
    std.testing.refAllDecls(@This());
}
