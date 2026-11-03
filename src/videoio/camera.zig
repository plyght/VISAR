//! Live camera capture, the same interface on every platform:
//!
//!   var cam = try camera.Camera.open(gpa, io, .{ .width = 1280, .height = 720 }, sink);
//!   defer cam.close();
//!   try cam.start();          // frames now arrive at sink.func, on a capture thread
//!
//! Each frame is handed to the sink as a zero-copy `Image(u8)` (4 channels,
//! BGRA, rows `stride` bytes apart) that is valid only during the call; the
//! sink copies what it keeps (see `FrameRing.store`).
//!
//! Backends: AVFoundation on macOS (camera_macos.zig, loaded at run time
//! through the Objective-C runtime), V4L2 on Linux (camera_linux.zig).
//! Elsewhere `supported` is false and `open` returns error.Unsupported.

const std = @import("std");
const builtin = @import("builtin");
const core = @import("../core/core.zig");
const Image = core.Image;

pub const Backend = enum { avfoundation, v4l2, none };

pub const backend: Backend = switch (builtin.os.tag) {
    .macos => .avfoundation,
    // V4L2 with the 64-bit struct layouts.
    .linux => if (@sizeOf(usize) == 8 and !builtin.abi.isAndroid()) .v4l2 else .none,
    else => .none,
};

pub const supported = backend != .none;

const impl = switch (backend) {
    .avfoundation => @import("camera_macos.zig"),
    .v4l2 => @import("camera_linux.zig"),
    .none => Unsupported,
};

pub const Camera = impl.Camera;

/// Every camera, with its formats.
pub fn list(gpa: std.mem.Allocator) Error!DeviceList {
    return impl.list(gpa);
}

/// Whether this process may use the camera (macOS asks the user once).
pub fn permission() Permission {
    return impl.permission();
}

pub const Error = error{
    /// No camera backend for this platform.
    Unsupported,
    /// The system camera libraries could not be loaded (see `detail`).
    RuntimeUnavailable,
    NoCamera,
    /// `Options.index` is past the end of the list.
    NoSuchCamera,
    PermissionDenied,
    PermissionRestricted,
    /// The device exists but could not be opened (in use, unplugged...).
    CameraUnavailable,
    /// The device rejected the capture configuration.
    ConfigureFailed,
    OutOfMemory,
};

/// More about the last error, from the system (e.g. an NSError's text).
pub var detail_buf: [256]u8 = undefined;
pub var detail_len: usize = 0;

pub fn detail() []const u8 {
    return detail_buf[0..detail_len];
}

pub fn setDetail(comptime fmt: []const u8, a: anytype) void {
    const s: []const u8 = std.fmt.bufPrint(&detail_buf, fmt, a) catch &detail_buf;
    detail_len = s.len;
}

pub const Permission = enum {
    authorized,
    /// The user has not been asked yet; opening the camera asks.
    not_determined,
    denied,
    /// Blocked by a policy (MDM, parental controls).
    restricted,
    /// No permission system (Linux: device file permissions apply).
    not_applicable,
};

pub const Options = struct {
    /// Index into `list`; null for the system default camera.
    index: ?usize = null,
    /// The largest frame size wanted; the format with the highest frame
    /// rate at or below it is chosen (see `chooseFormat`).
    width: u32 = 1280,
    height: u32 = 720,
    /// Frame rate wanted; null for the format's maximum.
    fps: ?f64 = null,
};

/// Receives frames on the capture thread.
pub const Sink = struct {
    ctx: *anyopaque,
    /// `img` is BGRA and valid only during the call; `pts` is the source's
    /// presentation time in seconds (0 if unknown).
    func: *const fn (ctx: *anyopaque, img: Image(u8), pts: f64) void,
};

/// What the device actually delivers, after `open`.
pub const Granted = struct {
    width: u32 = 0,
    height: u32 = 0,
    /// The frame rate the device was set to.
    fps: f64 = 0,
    /// The highest rate the chosen format supports.
    format_max_fps: f64 = 0,
    /// The camera's native pixel format (FourCC); frames are converted to BGRA.
    pixel_format: [4]u8 = "????".*,
    name_buf: [128]u8 = undefined,
    name_len: usize = 0,

    pub fn name(g: *const Granted) []const u8 {
        return g.name_buf[0..g.name_len];
    }

    pub fn setName(g: *Granted, n: []const u8) void {
        g.name_len = @min(n.len, g.name_buf.len);
        @memcpy(g.name_buf[0..g.name_len], n[0..g.name_len]);
    }
};

pub const Format = struct {
    width: u32,
    height: u32,
    min_fps: f64,
    max_fps: f64,
    /// FourCC of the native pixel format, e.g. "420v", "YUYV", "MJPG".
    pixel_format: [4]u8,
};

pub const Device = struct {
    name: []const u8,
    /// A stable identifier (AVFoundation uniqueID, /dev/videoN).
    id: []const u8,
    is_default: bool,
    formats: []Format,
};

pub const DeviceList = struct {
    arena: std.heap.ArenaAllocator,
    devices: []Device,
    permission: Permission,

    pub fn deinit(self: *DeviceList) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

/// The format to capture with: among those no larger than `want_w` x
/// `want_h`, the one with the highest frame rate (then the largest); with
/// `want_fps`, the largest that reaches that rate (or else the fastest).
/// If none fits, the smallest. Null for an empty list.
pub fn chooseFormat(formats: []const Format, want_w: u32, want_h: u32, want_fps: ?f64) ?usize {
    if (formats.len == 0) return null;
    var any_fit = false;
    for (formats) |f| any_fit = any_fit or (f.width <= want_w and f.height <= want_h);
    var best: ?usize = null;
    for (formats, 0..) |f, i| {
        if (any_fit and (f.width > want_w or f.height > want_h)) continue;
        const b = best orelse {
            best = i;
            continue;
        };
        if (better(f, formats[b], any_fit, want_fps)) best = i;
    }
    return best;
}

fn better(a: Format, b: Format, fits: bool, want_fps: ?f64) bool {
    const area_a = @as(u64, a.width) * a.height;
    const area_b = @as(u64, b.width) * b.height;
    // Rates compare rounded, so 29.97 and 30 tie.
    const ra = @round(a.max_fps);
    const rb = @round(b.max_fps);
    if (!fits) {
        if (area_a != area_b) return area_a < area_b;
        return ra > rb;
    }
    if (want_fps) |w| {
        const oka = a.max_fps >= w - 0.5;
        const okb = b.max_fps >= w - 0.5;
        if (oka != okb) return oka;
        if (!oka and ra != rb) return ra > rb;
        if (area_a != area_b) return area_a > area_b;
        return ra > rb;
    }
    if (ra != rb) return ra > rb;
    return area_a > area_b;
}

/// The backend for platforms without one.
const Unsupported = struct {
    pub const Camera = struct {
        granted: Granted = .{},

        pub fn open(_: std.mem.Allocator, _: std.Io, _: Options, _: Sink) Error!*@This() {
            return error.Unsupported;
        }
        pub fn start(_: *@This()) Error!void {}
        pub fn stop(_: *@This()) void {}
        pub fn close(_: *@This()) void {}
        pub fn frames(_: *const @This()) usize {
            return 0;
        }
    };

    pub fn list(_: std.mem.Allocator) Error!DeviceList {
        return error.Unsupported;
    }

    pub fn permission() Permission {
        return .not_applicable;
    }
};

test "format choice" {
    const F = struct {
        fn f(w: u32, h: u32, fps: f64) Format {
            return .{ .width = w, .height = h, .min_fps = 1, .max_fps = fps, .pixel_format = "420v".* };
        }
    };
    const fmts = [_]Format{ F.f(1920, 1080, 30), F.f(1280, 720, 30), F.f(1280, 720, 60), F.f(640, 480, 120), F.f(320, 240, 120) };
    // Highest rate at or below 1280x720, then the largest.
    try std.testing.expectEqual(@as(?usize, 3), chooseFormat(&fmts, 1280, 720, null));
    // With a rate: the largest that reaches it.
    try std.testing.expectEqual(@as(?usize, 2), chooseFormat(&fmts, 1280, 720, 60));
    try std.testing.expectEqual(@as(?usize, 2), chooseFormat(&fmts, 1280, 720, 29.97));
    try std.testing.expectEqual(@as(?usize, 0), chooseFormat(&fmts, 1920, 1080, 30));
    // Unreachable rate: the fastest.
    try std.testing.expectEqual(@as(?usize, 3), chooseFormat(&fmts, 1280, 720, 240));
    // Nothing fits: the smallest.
    try std.testing.expectEqual(@as(?usize, 4), chooseFormat(&fmts, 100, 100, null));
    try std.testing.expectEqual(@as(?usize, null), chooseFormat(&.{}, 1280, 720, null));
}
