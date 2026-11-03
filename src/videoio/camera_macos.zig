//! Camera capture on macOS with AVFoundation, reached through the
//! Objective-C runtime at run time (objc.zig): no SDK, no link-time
//! dependency beyond libSystem.
//!
//! Set-up (`Camera.open`):
//!   1. pick the device: the default video device, or entry N of the
//!      discovery list (`list`);
//!   2. check the camera permission, asking the user if it was never asked;
//!   3. AVCaptureSession + AVCaptureDeviceInput + AVCaptureVideoDataOutput
//!      (32BGRA at the chosen format's size, late frames discarded), with a
//!      delegate class registered at run time whose
//!      captureOutput:didOutputSampleBuffer:fromConnection: is a Zig
//!      function, called on a serial dispatch queue;
//!   4. under lockForConfiguration:, the format with the highest frame rate
//!      at or below the requested size, and its fastest frame duration.
//!
//! Per frame (on the dispatch queue): lock the CVPixelBuffer read-only, wrap
//! it as an `Image(u8)` (BGRA, the buffer's own stride, no copy), call the
//! sink, unlock.

const std = @import("std");
const builtin = @import("builtin");
const objc = @import("objc.zig");
const camera = @import("camera.zig");
const core = @import("../core/core.zig");
const Image = core.Image;
const id = objc.id;
const send = objc.send;

const Error = camera.Error;

const CMVideoDimensions = extern struct { width: i32, height: i32 };

/// kCVPixelFormatType_32BGRA ('BGRA').
const pixel_format_bgra: u32 = 0x42475241;
/// kCVPixelBufferLock_ReadOnly.
const lock_read_only: u64 = 1;

const Api = struct {
    CMSampleBufferGetImageBuffer: *const fn (?*anyopaque) callconv(.c) ?*anyopaque,
    CMSampleBufferGetPresentationTimeStamp: *const fn (?*anyopaque) callconv(.c) objc.CMTime,
    CMVideoFormatDescriptionGetDimensions: *const fn (?*anyopaque) callconv(.c) CMVideoDimensions,
    CMFormatDescriptionGetMediaSubType: *const fn (?*anyopaque) callconv(.c) u32,
    CVPixelBufferLockBaseAddress: *const fn (?*anyopaque, u64) callconv(.c) i32,
    CVPixelBufferUnlockBaseAddress: *const fn (?*anyopaque, u64) callconv(.c) i32,
    CVPixelBufferGetBaseAddress: *const fn (?*anyopaque) callconv(.c) ?[*]u8,
    CVPixelBufferGetBytesPerRow: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetWidth: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetHeight: *const fn (?*anyopaque) callconv(.c) usize,
    CVPixelBufferGetPixelFormatType: *const fn (?*anyopaque) callconv(.c) u32,
    dispatch_queue_create: *const fn ([*:0]const u8, ?*anyopaque) callconv(.c) ?*anyopaque,
    dispatch_sync_f: *const fn (?*anyopaque, ?*anyopaque, *const fn (?*anyopaque) callconv(.c) void) callconv(.c) void,
    dispatch_release: *const fn (?*anyopaque) callconv(.c) void,
    kCVPixelBufferPixelFormatTypeKey: id,
    kCVPixelBufferWidthKey: id,
    kCVPixelBufferHeightKey: id,
    AVMediaTypeVideo: id,
};

var api_storage: ?Api = null;

fn load() Error!*const Api {
    if (api_storage) |*a| return a;
    api_storage = loadApi() catch {
        camera.setDetail("cannot load {s}", .{objc.missing()});
        return error.RuntimeUnavailable;
    };
    return &api_storage.?;
}

fn loadApi() objc.Error!Api {
    _ = try objc.load();
    _ = try objc.open(.avfoundation);
    return .{
        .CMSampleBufferGetImageBuffer = try objc.func(fn (?*anyopaque) callconv(.c) ?*anyopaque, .coremedia, "CMSampleBufferGetImageBuffer"),
        .CMSampleBufferGetPresentationTimeStamp = try objc.func(fn (?*anyopaque) callconv(.c) objc.CMTime, .coremedia, "CMSampleBufferGetPresentationTimeStamp"),
        .CMVideoFormatDescriptionGetDimensions = try objc.func(fn (?*anyopaque) callconv(.c) CMVideoDimensions, .coremedia, "CMVideoFormatDescriptionGetDimensions"),
        .CMFormatDescriptionGetMediaSubType = try objc.func(fn (?*anyopaque) callconv(.c) u32, .coremedia, "CMFormatDescriptionGetMediaSubType"),
        .CVPixelBufferLockBaseAddress = try objc.func(fn (?*anyopaque, u64) callconv(.c) i32, .corevideo, "CVPixelBufferLockBaseAddress"),
        .CVPixelBufferUnlockBaseAddress = try objc.func(fn (?*anyopaque, u64) callconv(.c) i32, .corevideo, "CVPixelBufferUnlockBaseAddress"),
        .CVPixelBufferGetBaseAddress = try objc.func(fn (?*anyopaque) callconv(.c) ?[*]u8, .corevideo, "CVPixelBufferGetBaseAddress"),
        .CVPixelBufferGetBytesPerRow = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetBytesPerRow"),
        .CVPixelBufferGetWidth = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetWidth"),
        .CVPixelBufferGetHeight = try objc.func(fn (?*anyopaque) callconv(.c) usize, .corevideo, "CVPixelBufferGetHeight"),
        .CVPixelBufferGetPixelFormatType = try objc.func(fn (?*anyopaque) callconv(.c) u32, .corevideo, "CVPixelBufferGetPixelFormatType"),
        .dispatch_queue_create = try objc.func(fn ([*:0]const u8, ?*anyopaque) callconv(.c) ?*anyopaque, .system, "dispatch_queue_create"),
        .dispatch_sync_f = try objc.func(fn (?*anyopaque, ?*anyopaque, *const fn (?*anyopaque) callconv(.c) void) callconv(.c) void, .system, "dispatch_sync_f"),
        .dispatch_release = try objc.func(fn (?*anyopaque) callconv(.c) void, .system, "dispatch_release"),
        .kCVPixelBufferPixelFormatTypeKey = try objc.constant(id, .corevideo, "kCVPixelBufferPixelFormatTypeKey"),
        .kCVPixelBufferWidthKey = try objc.constant(id, .corevideo, "kCVPixelBufferWidthKey"),
        .kCVPixelBufferHeightKey = try objc.constant(id, .corevideo, "kCVPixelBufferHeightKey"),
        .AVMediaTypeVideo = try objc.constant(id, .avfoundation, "AVMediaTypeVideo"),
    };
}

// ---------------------------------------------------------------------------
// Permission
// ---------------------------------------------------------------------------

/// AVAuthorizationStatus.
fn status(a: *const Api) objc.NSInteger {
    const dev_cls = objc.getClass("AVCaptureDevice") orelse return 0;
    return send(objc.NSInteger, dev_cls, "authorizationStatusForMediaType:", .{a.AVMediaTypeVideo});
}

pub fn permission() camera.Permission {
    const a = load() catch return .not_determined;
    return switch (status(a)) {
        0 => .not_determined,
        1 => .restricted,
        2 => .denied,
        3 => .authorized,
        else => .not_determined,
    };
}

/// The answer to requestAccessForMediaType: (0 pending, 1 yes, 2 no).
var access_answer: std.atomic.Value(u8) = .init(0);

const AccessInvoke = fn (*anyopaque, objc.BOOL) callconv(.c) void;
const AccessBlock = objc.GlobalBlock(AccessInvoke);
const access_descriptor: AccessBlock.Descriptor = .{
    .size = @sizeOf(AccessBlock),
    // void (^)(BOOL): BOOL is 'B' (bool) on arm64 and 'c' on x86_64.
    .signature = if (builtin.cpu.arch == .aarch64) "v12@?0B8" else "v12@?0c8",
};
var access_block: AccessBlock = undefined;

fn accessAnswered(_: *anyopaque, granted: objc.BOOL) callconv(.c) void {
    access_answer.store(if (granted != 0) 1 else 2, .release);
}

/// Asks the user (macOS shows its camera prompt) and waits for the answer.
fn requestAccess(a: *const Api, io: std.Io) Error!void {
    access_block = AccessBlock.init(&accessAnswered, &access_descriptor) catch {
        // No block runtime (should not happen): opening the device asks too.
        return;
    };
    access_answer.store(0, .release);
    const dev_cls = objc.getClass("AVCaptureDevice") orelse return error.RuntimeUnavailable;
    send(void, dev_cls, "requestAccessForMediaType:completionHandler:", .{ a.AVMediaTypeVideo, @as(*anyopaque, @ptrCast(&access_block)) });
    // The user may take a while to read the prompt; give them two minutes.
    var waited: u32 = 0;
    while (access_answer.load(.acquire) == 0 and waited < 2400) : (waited += 1) {
        io.sleep(.fromMilliseconds(50), .awake) catch {};
    }
    switch (access_answer.load(.acquire)) {
        1 => return,
        2 => return error.PermissionDenied,
        else => {
            camera.setDetail("no answer to the camera permission prompt", .{});
            return error.PermissionDenied;
        },
    }
}

// ---------------------------------------------------------------------------
// Devices and formats
// ---------------------------------------------------------------------------

/// The video devices (autoreleased NSArray), in discovery order.
fn videoDevices(a: *const Api) id {
    if (objc.getClass("AVCaptureDeviceDiscoverySession")) |ds_cls| {
        var types: [2]id = undefined;
        var n: usize = 0;
        if (objc.optionalConstant(id, .avfoundation, "AVCaptureDeviceTypeBuiltInWideAngleCamera")) |t| {
            types[n] = t;
            n += 1;
        }
        // External cameras: the macOS 14 name, or the older one (never both).
        if (objc.optionalConstant(id, .avfoundation, "AVCaptureDeviceTypeExternal") orelse
            objc.optionalConstant(id, .avfoundation, "AVCaptureDeviceTypeExternalUnknown")) |t|
        {
            types[n] = t;
            n += 1;
        }
        if (n > 0) {
            const array_cls = objc.getClass("NSArray");
            const arr = send(id, array_cls, "arrayWithObjects:count:", .{ @as([*]const id, &types), @as(objc.NSUInteger, n) });
            const ds = send(id, ds_cls, "discoverySessionWithDeviceTypes:mediaType:position:", .{ arr, a.AVMediaTypeVideo, @as(objc.NSInteger, 0) });
            if (ds != null) return send(id, ds, "devices", .{});
        }
    }
    const dev_cls = objc.getClass("AVCaptureDevice") orelse return null;
    return send(id, dev_cls, "devicesWithMediaType:", .{a.AVMediaTypeVideo});
}

fn defaultDevice(a: *const Api) id {
    const dev_cls = objc.getClass("AVCaptureDevice") orelse return null;
    return send(id, dev_cls, "defaultDeviceWithMediaType:", .{a.AVMediaTypeVideo});
}

fn sameDevice(x: id, y: id) bool {
    if (x == null or y == null) return false;
    if (x == y) return true;
    return std.mem.eql(u8, objc.utf8(send(id, x, "uniqueID", .{})), objc.utf8(send(id, y, "uniqueID", .{})));
}

/// Size, pixel format and frame-rate limits of an AVCaptureDeviceFormat.
fn formatInfo(a: *const Api, fmt: id) camera.Format {
    const desc = send(?*anyopaque, fmt, "formatDescription", .{});
    const dims = if (desc != null) a.CMVideoFormatDescriptionGetDimensions(desc) else CMVideoDimensions{ .width = 0, .height = 0 };
    const sub = if (desc != null) a.CMFormatDescriptionGetMediaSubType(desc) else 0;
    var info: camera.Format = .{
        .width = @intCast(@max(0, dims.width)),
        .height = @intCast(@max(0, dims.height)),
        .min_fps = 0,
        .max_fps = 0,
        .pixel_format = undefined,
    };
    _ = objc.fourcc(sub, &info.pixel_format);
    const ranges = send(id, fmt, "videoSupportedFrameRateRanges", .{});
    var min: f64 = std.math.inf(f64);
    for (0..objc.count(ranges)) |i| {
        const r = objc.at(ranges, i);
        info.max_fps = @max(info.max_fps, send(f64, r, "maxFrameRate", .{}));
        min = @min(min, send(f64, r, "minFrameRate", .{}));
    }
    info.min_fps = if (std.math.isInf(min)) 0 else min;
    return info;
}

pub fn list(gpa: std.mem.Allocator) Error!camera.DeviceList {
    const a = try load();
    const pool = objc.Pool.push();
    defer pool.pop();
    var arena = std.heap.ArenaAllocator.init(gpa);
    errdefer arena.deinit();
    const ar = arena.allocator();
    const devs = videoDevices(a);
    const def = defaultDevice(a);
    const n = objc.count(devs);
    const out = try ar.alloc(camera.Device, n);
    for (out, 0..) |*d, i| {
        const dev = objc.at(devs, i);
        const fmts = send(id, dev, "formats", .{});
        const formats = try ar.alloc(camera.Format, objc.count(fmts));
        for (formats, 0..) |*f, j| f.* = formatInfo(a, objc.at(fmts, j));
        d.* = .{
            .name = try ar.dupe(u8, objc.utf8(send(id, dev, "localizedName", .{}))),
            .id = try ar.dupe(u8, objc.utf8(send(id, dev, "uniqueID", .{}))),
            .is_default = sameDevice(dev, def),
            .formats = formats,
        };
    }
    return .{ .arena = arena, .devices = out, .permission = permission() };
}

// ---------------------------------------------------------------------------
// Capture
// ---------------------------------------------------------------------------

/// The camera the delegate delivers to (one capture at a time).
var active: std.atomic.Value(?*Camera) = .init(null);

pub const Camera = struct {
    gpa: std.mem.Allocator,
    api: *const Api,
    sink: camera.Sink,
    granted: camera.Granted = .{},
    session: id = null,
    device: id = null,
    output: id = null,
    delegate: id = null,
    queue: ?*anyopaque = null,
    /// Held from configuration until the session runs, so the session's
    /// start does not reset the format and frame rate.
    locked: bool = false,
    running: bool = false,
    frame_count: std.atomic.Value(usize) = .init(0),
    late_drops: std.atomic.Value(usize) = .init(0),
    /// Frames that were not BGRA or could not be locked.
    bad_frames: std.atomic.Value(usize) = .init(0),

    pub fn open(gpa: std.mem.Allocator, io: std.Io, opts: camera.Options, sink: camera.Sink) Error!*Camera {
        const a = try load();
        const pool = objc.Pool.push();
        defer pool.pop();

        // 1. The device.
        const devs = videoDevices(a);
        const device = if (opts.index) |i| blk: {
            if (i >= objc.count(devs)) return error.NoSuchCamera;
            break :blk objc.at(devs, i);
        } else defaultDevice(a) orelse (if (objc.count(devs) > 0) objc.at(devs, 0) else null);
        if (device == null) return error.NoCamera;

        // 2. Permission.
        switch (status(a)) {
            1 => return error.PermissionRestricted,
            2 => return error.PermissionDenied,
            0 => try requestAccess(a, io),
            else => {},
        }

        // The format: the fastest at or below the requested size.
        const fmts = send(id, device, "formats", .{});
        const nf = objc.count(fmts);
        if (nf == 0) {
            camera.setDetail("the camera lists no formats", .{});
            return error.ConfigureFailed;
        }
        const infos = try gpa.alloc(camera.Format, nf);
        defer gpa.free(infos);
        for (infos, 0..) |*f, j| f.* = formatInfo(a, objc.at(fmts, j));
        const fi = camera.chooseFormat(infos, opts.width, opts.height, opts.fps).?;
        const fmt = objc.at(fmts, fi);
        const info = infos[fi];

        const self = try gpa.create(Camera);
        errdefer gpa.destroy(self);
        self.* = .{ .gpa = gpa, .api = a, .sink = sink };
        errdefer self.teardown();
        self.device = objc.retain(device);
        self.granted.setName(objc.utf8(send(id, device, "localizedName", .{})));
        self.granted.pixel_format = info.pixel_format;
        self.granted.format_max_fps = info.max_fps;

        // 3. Session, input, output and delegate.
        self.session = objc.new(try cls("AVCaptureSession"));
        var err: id = null;
        const input = send(id, try cls("AVCaptureDeviceInput"), "deviceInputWithDevice:error:", .{ device, @as(*id, &err) });
        if (input == null) {
            camera.setDetail("{s}", .{objc.errorText(err)});
            return error.CameraUnavailable;
        }
        send(void, self.session, "beginConfiguration", .{});
        if (send(objc.BOOL, self.session, "canAddInput:", .{input}) == objc.NO) {
            send(void, self.session, "commitConfiguration", .{});
            camera.setDetail("the capture session refused the camera (is another app using it?)", .{});
            return error.CameraUnavailable;
        }
        send(void, self.session, "addInput:", .{input});

        self.output = objc.new(try cls("AVCaptureVideoDataOutput"));
        const number = try cls("NSNumber");
        var keys = [_]id{ a.kCVPixelBufferPixelFormatTypeKey, a.kCVPixelBufferWidthKey, a.kCVPixelBufferHeightKey };
        var values = [_]id{
            send(id, number, "numberWithUnsignedInt:", .{pixel_format_bgra}),
            send(id, number, "numberWithUnsignedInt:", .{@as(u32, info.width)}),
            send(id, number, "numberWithUnsignedInt:", .{@as(u32, info.height)}),
        };
        const settings = send(id, try cls("NSDictionary"), "dictionaryWithObjects:forKeys:count:", .{ @as([*]const id, &values), @as([*]const id, &keys), @as(objc.NSUInteger, keys.len) });
        send(void, self.output, "setVideoSettings:", .{settings});
        send(void, self.output, "setAlwaysDiscardsLateVideoFrames:", .{objc.YES});
        self.delegate = objc.new(try delegateClass());
        self.queue = a.dispatch_queue_create("org.visar.camera", null);
        send(void, self.output, "setSampleBufferDelegate:queue:", .{ self.delegate, self.queue });
        if (send(objc.BOOL, self.session, "canAddOutput:", .{self.output}) == objc.NO) {
            send(void, self.session, "commitConfiguration", .{});
            camera.setDetail("the capture session refused the video output", .{});
            return error.ConfigureFailed;
        }
        send(void, self.session, "addOutput:", .{self.output});
        send(void, self.session, "commitConfiguration", .{});

        // 4. Format and frame rate (after the input joined the session, or
        // the session preset would override them).
        err = null;
        if (send(objc.BOOL, device, "lockForConfiguration:", .{@as(*id, &err)}) != objc.NO) {
            self.locked = true;
            send(void, device, "setActiveFormat:", .{fmt});
            if (frameDuration(fmt, opts.fps)) |d| {
                send(void, device, "setActiveVideoMinFrameDuration:", .{d});
                send(void, device, "setActiveVideoMaxFrameDuration:", .{d});
            }
        }
        self.readBack();
        return self;
    }

    /// What the device is set to now.
    fn readBack(self: *Camera) void {
        const a = self.api;
        const fmt = send(id, self.device, "activeFormat", .{});
        if (fmt != null) {
            const info = formatInfo(a, fmt);
            self.granted.width = info.width;
            self.granted.height = info.height;
            self.granted.pixel_format = info.pixel_format;
            self.granted.format_max_fps = info.max_fps;
        }
        const d = send(objc.CMTime, self.device, "activeVideoMinFrameDuration", .{});
        const s = d.seconds();
        self.granted.fps = if (s > 0) 1 / s else self.granted.format_max_fps;
    }

    pub fn start(self: *Camera) Error!void {
        if (self.running) return;
        const pool = objc.Pool.push();
        defer pool.pop();
        active.store(self, .release);
        send(void, self.session, "startRunning", .{});
        self.running = true;
        if (self.locked) {
            send(void, self.device, "unlockForConfiguration", .{});
            self.locked = false;
        }
        self.readBack();
        if (send(objc.BOOL, self.session, "isRunning", .{}) == objc.NO) {
            camera.setDetail("the capture session did not start", .{});
            return error.CameraUnavailable;
        }
    }

    pub fn stop(self: *Camera) void {
        if (!self.running) return;
        const pool = objc.Pool.push();
        defer pool.pop();
        send(void, self.session, "stopRunning", .{});
        // Wait for a callback that may still be running on the queue.
        if (self.queue) |q| self.api.dispatch_sync_f(q, null, &noop);
        active.store(null, .release);
        self.running = false;
    }

    pub fn close(self: *Camera) void {
        self.stop();
        self.teardown();
        self.gpa.destroy(self);
    }

    fn teardown(self: *Camera) void {
        const pool = objc.Pool.push();
        defer pool.pop();
        if (active.load(.acquire) == self) active.store(null, .release);
        if (self.locked) send(void, self.device, "unlockForConfiguration", .{});
        self.locked = false;
        if (self.output != null) send(void, self.output, "setSampleBufferDelegate:queue:", .{ @as(id, null), @as(?*anyopaque, null) });
        objc.release(self.session);
        objc.release(self.output);
        objc.release(self.delegate);
        objc.release(self.device);
        if (self.queue) |q| self.api.dispatch_release(q);
        self.session = null;
        self.output = null;
        self.delegate = null;
        self.device = null;
        self.queue = null;
    }

    pub fn frames(self: *const Camera) usize {
        return self.frame_count.load(.monotonic);
    }

    fn onSample(self: *Camera, sample: ?*anyopaque) void {
        const a = self.api;
        const pb = a.CMSampleBufferGetImageBuffer(sample) orelse return;
        if (a.CVPixelBufferGetPixelFormatType(pb) != pixel_format_bgra or a.CVPixelBufferLockBaseAddress(pb, lock_read_only) != 0) {
            _ = self.bad_frames.fetchAdd(1, .monotonic);
            return;
        }
        defer _ = a.CVPixelBufferUnlockBaseAddress(pb, lock_read_only);
        const base = a.CVPixelBufferGetBaseAddress(pb) orelse return;
        const w = a.CVPixelBufferGetWidth(pb);
        const h = a.CVPixelBufferGetHeight(pb);
        const bpr = a.CVPixelBufferGetBytesPerRow(pb);
        if (w == 0 or h == 0 or bpr < w * 4) return;
        const img = Image(u8).fromSliceStrided(base[0 .. bpr * h], @intCast(w), @intCast(h), 4, bpr) catch return;
        const pts = a.CMSampleBufferGetPresentationTimeStamp(sample).seconds();
        _ = self.frame_count.fetchAdd(1, .monotonic);
        self.sink.func(self.sink.ctx, img, pts);
    }
};

fn cls(comptime name: [:0]const u8) Error!objc.Class {
    return objc.class(name) catch {
        camera.setDetail("AVFoundation has no class {s}", .{name});
        return error.RuntimeUnavailable;
    };
}

fn noop(_: ?*anyopaque) callconv(.c) void {}

/// The frame duration for `want_fps` among the format's supported ranges:
/// the exact fastest duration of a range (as AVFoundation reports it, so it
/// is always accepted) unless a slower rate inside a range was asked for.
fn frameDuration(fmt: id, want_fps: ?f64) ?objc.CMTime {
    const ranges = send(id, fmt, "videoSupportedFrameRateRanges", .{});
    // The range to take the fastest duration of: without a wanted rate the
    // fastest; with one, the fastest that does not exceed it (else the slowest).
    var best: id = null;
    var best_max: f64 = 0;
    for (0..objc.count(ranges)) |i| {
        const r = objc.at(ranges, i);
        const max = send(f64, r, "maxFrameRate", .{});
        const min = send(f64, r, "minFrameRate", .{});
        if (want_fps) |w| if (w > min + 0.01 and w < max - 0.01) {
            // A rate strictly inside a continuous range is set as asked.
            return .{ .value = 1000, .timescale = @intFromFloat(@round(w * 1000)), .flags = objc.CMTime.flag_valid, .epoch = 0 };
        };
        if (best == null) {
            best = r;
            best_max = max;
            continue;
        }
        const w = want_fps orelse {
            if (max > best_max) {
                best = r;
                best_max = max;
            }
            continue;
        };
        const fits = max <= w + 0.5;
        const best_fits = best_max <= w + 0.5;
        if ((fits and (!best_fits or max > best_max)) or (!fits and !best_fits and max < best_max)) {
            best = r;
            best_max = max;
        }
    }
    if (best == null) return null;
    return send(objc.CMTime, best, "minFrameDuration", .{});
}

// ---------------------------------------------------------------------------
// The delegate class
// ---------------------------------------------------------------------------

fn didOutput(_: id, _: objc.SEL, _: id, sample: ?*anyopaque, _: id) callconv(.c) void {
    const cam = active.load(.acquire) orelse return;
    cam.onSample(sample);
}

fn didDrop(_: id, _: objc.SEL, _: id, _: ?*anyopaque, _: id) callconv(.c) void {
    const cam = active.load(.acquire) orelse return;
    _ = cam.late_drops.fetchAdd(1, .monotonic);
}

var delegate_class: objc.Class = null;

/// VISARCaptureDelegate: NSObject + AVCaptureVideoDataOutputSampleBufferDelegate,
/// built at run time.
fn delegateClass() Error!objc.Class {
    if (delegate_class) |c| return c;
    const name = "VISARCaptureDelegate";
    if (objc.getClass(name)) |c| {
        delegate_class = c;
        return c;
    }
    const rt = objc.load() catch return error.RuntimeUnavailable;
    const c = rt.allocateClassPair(try cls("NSObject"), name, 0) orelse {
        camera.setDetail("cannot create the capture delegate class", .{});
        return error.RuntimeUnavailable;
    };
    // v@:@^{opaqueCMSampleBuffer=}@ : void (id self, SEL, AVCaptureOutput *, CMSampleBufferRef, AVCaptureConnection *)
    const types = "v@:@^{opaqueCMSampleBuffer=}@";
    _ = rt.addMethod(c, objc.sel("captureOutput:didOutputSampleBuffer:fromConnection:"), @ptrCast(&didOutput), types);
    _ = rt.addMethod(c, objc.sel("captureOutput:didDropSampleBuffer:fromConnection:"), @ptrCast(&didDrop), types);
    if (rt.getProtocol("AVCaptureVideoDataOutputSampleBufferDelegate")) |p| _ = rt.addProtocol(c, p);
    rt.registerClassPair(c);
    delegate_class = c;
    return c;
}
