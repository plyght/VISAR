//! visar.objdetect: OpenCV's objdetect equivalents. QR code generation,
//! detection and decoding, and ArUco marker dictionaries, generation and
//! detection.

const std = @import("std");

pub const gf256 = @import("gf256.zig");
pub const qr_tables = @import("qr_tables.zig");
pub const qr_encode = @import("qr_encode.zig");
pub const qr_decode = @import("qr_decode.zig");

pub const QrLevel = qr_encode.Level;
pub const QrMode = qr_encode.Mode;
pub const QrEncodeOptions = qr_encode.Options;
pub const qrEncode = qr_encode.encode;
pub const qrEncodeStructuredAppend = qr_encode.encodeStructuredAppend;

test {
    std.testing.refAllDecls(@This());
}
