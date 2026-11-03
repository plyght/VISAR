//! A native window on macOS through AppKit, reached through the Objective-C
//! runtime at run time (no SDK; see videoio/objc.zig).
//!
//! The window's content view hosts a CALayer; each frame becomes a CGImage
//! that wraps the caller's BGRA buffer without copying
//! (kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little) and is set as
//! the layer's contents inside a CATransaction with actions disabled. Core
//! Animation keeps the image as long as it needs it; when it lets go, the
//! data provider's release callback hands the buffer back to the caller
//! (`Release`), so a buffer is never rewritten while it can be on screen.
//!
//! There is no [NSApp run]: the caller owns the loop and calls `poll` (which
//! drains pending events, waiting at most `timeout` seconds for the first)
//! and `present` on the main thread.

const std = @import("std");
const objc = @import("../videoio/objc.zig");
const core = @import("../core/core.zig");
const Image = core.Image;
const id = objc.id;
const send = objc.send;

pub const Error = error{ RuntimeUnavailable, WindowFailed };

/// Called when Core Animation no longer needs a presented buffer.
pub const Release = struct {
    ctx: ?*anyopaque,
    func: *const fn (ctx: ?*anyopaque) callconv(.c) void,
};

/// What happened since the last `poll`.
pub const Input = struct {
    quit: bool = false,
    reset: bool = false,
    toggle_hud: bool = false,
    /// Right click or 'c'.
    clear: bool = false,
    /// Tab.
    cycle_mode: bool = false,
    /// A rectangle the user finished dragging, in pixels of the last
    /// presented image: x, y, w, h.
    selection: ?[4]f32 = null,
    /// The rectangle being dragged, if any (same units).
    dragging: ?[4]f32 = null,
};

const Api = struct {
    CGColorSpaceCreateDeviceRGB: *const fn () callconv(.c) ?*anyopaque,
    CGColorSpaceRelease: *const fn (?*anyopaque) callconv(.c) void,
    CGDataProviderCreateWithData: *const fn (?*anyopaque, ?*const anyopaque, usize, ?*const ReleaseData) callconv(.c) ?*anyopaque,
    CGDataProviderRelease: *const fn (?*anyopaque) callconv(.c) void,
    CGImageCreate: *const fn (usize, usize, usize, usize, usize, ?*anyopaque, u32, ?*anyopaque, ?[*]const objc.CGFloat, bool, i32) callconv(.c) ?*anyopaque,
    CGImageRelease: *const fn (?*anyopaque) callconv(.c) void,
    NSDefaultRunLoopMode: id,
    kCAGravityResizeAspect: id,
};

/// void (*)(void *info, const void *data, size_t size)
const ReleaseData = fn (?*anyopaque, ?*const anyopaque, usize) callconv(.c) void;

const kCGImageAlphaNoneSkipFirst: u32 = 6;
const kCGBitmapByteOrder32Little: u32 = 2 << 12;

const NSWindowStyleMaskTitled: objc.NSUInteger = 1 << 0;
const NSWindowStyleMaskClosable: objc.NSUInteger = 1 << 1;
const NSWindowStyleMaskMiniaturizable: objc.NSUInteger = 1 << 2;
const NSWindowStyleMaskResizable: objc.NSUInteger = 1 << 3;
const NSBackingStoreBuffered: objc.NSUInteger = 2;
const NSApplicationActivationPolicyRegular: objc.NSInteger = 0;
const NSEventTypeKeyDown: objc.NSUInteger = 10;
const NSEventTypeLeftMouseDown: objc.NSUInteger = 1;
const NSEventTypeLeftMouseUp: objc.NSUInteger = 2;
const NSEventTypeRightMouseDown: objc.NSUInteger = 3;
const NSEventTypeLeftMouseDragged: objc.NSUInteger = 6;
const NSEventMaskAny: objc.NSUInteger = std.math.maxInt(objc.NSUInteger);

// Virtual key codes (kVK_ANSI_Q and friends).
const key_q: u16 = 12;
const key_h: u16 = 4;
const key_space: u16 = 49;
const key_escape: u16 = 53;
const key_c: u16 = 8;
const key_tab: u16 = 48;

var api_storage: ?Api = null;

fn load() Error!*const Api {
    if (api_storage) |*a| return a;
    api_storage = loadApi() catch return error.RuntimeUnavailable;
    return &api_storage.?;
}

fn loadApi() objc.Error!Api {
    _ = try objc.load();
    _ = try objc.open(.appkit);
    _ = try objc.open(.quartzcore);
    return .{
        .CGColorSpaceCreateDeviceRGB = try objc.func(fn () callconv(.c) ?*anyopaque, .coregraphics, "CGColorSpaceCreateDeviceRGB"),
        .CGColorSpaceRelease = try objc.func(fn (?*anyopaque) callconv(.c) void, .coregraphics, "CGColorSpaceRelease"),
        .CGDataProviderCreateWithData = try objc.func(fn (?*anyopaque, ?*const anyopaque, usize, ?*const ReleaseData) callconv(.c) ?*anyopaque, .coregraphics, "CGDataProviderCreateWithData"),
        .CGDataProviderRelease = try objc.func(fn (?*anyopaque) callconv(.c) void, .coregraphics, "CGDataProviderRelease"),
        .CGImageCreate = try objc.func(fn (usize, usize, usize, usize, usize, ?*anyopaque, u32, ?*anyopaque, ?[*]const objc.CGFloat, bool, i32) callconv(.c) ?*anyopaque, .coregraphics, "CGImageCreate"),
        .CGImageRelease = try objc.func(fn (?*anyopaque) callconv(.c) void, .coregraphics, "CGImageRelease"),
        .NSDefaultRunLoopMode = try objc.constant(id, .foundation, "NSDefaultRunLoopMode"),
        .kCAGravityResizeAspect = try objc.constant(id, .quartzcore, "kCAGravityResizeAspect"),
    };
}

/// Why `open` failed, for the error message.
pub fn missing() []const u8 {
    return objc.missing();
}

pub const Window = struct {
    gpa: std.mem.Allocator,
    api: *const Api,
    app: id,
    window: id,
    view: id,
    layer: id,
    colorspace: ?*anyopaque,
    /// Classes and selectors used every frame, looked up once.
    date_cls: objc.Class,
    catransaction_cls: objc.Class,
    open_: bool = true,
    /// Size of the last presented image, to map the mouse into it.
    image_w: u32 = 0,
    image_h: u32 = 0,
    /// Where the current drag started and is now (image pixels).
    drag_from: ?[2]f32 = null,
    drag_to: [2]f32 = .{ 0, 0 },

    /// Creates the application (as a regular app with a Dock icon) and a
    /// window whose content area is `width` x `height` points, and brings it
    /// to the front. Main thread only.
    pub fn open(gpa: std.mem.Allocator, title: [*:0]const u8, width: u32, height: u32) Error!*Window {
        const a = try load();
        const pool = objc.Pool.push();
        defer pool.pop();
        const app_cls = objc.class("NSApplication") catch return error.RuntimeUnavailable;
        const app = send(id, app_cls, "sharedApplication", .{});
        if (app == null) return error.WindowFailed;
        _ = send(objc.BOOL, app, "setActivationPolicy:", .{NSApplicationActivationPolicyRegular});
        send(void, app, "finishLaunching", .{});

        const rect: objc.CGRect = .{ .origin = .{ .x = 0, .y = 0 }, .size = .{ .width = @floatFromInt(width), .height = @floatFromInt(height) } };
        const style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable;
        const win_alloc = send(id, objc.class("NSWindow") catch return error.RuntimeUnavailable, "alloc", .{});
        const window = send(id, win_alloc, "initWithContentRect:styleMask:backing:defer:", .{ rect, style, NSBackingStoreBuffered, objc.NO });
        if (window == null) return error.WindowFailed;
        errdefer objc.release(window);
        // We keep the window after it is closed and check isVisible.
        send(void, window, "setReleasedWhenClosed:", .{objc.NO});
        send(void, window, "setTitle:", .{objc.nsString(title)});
        const black = send(id, objc.class("NSColor") catch return error.RuntimeUnavailable, "blackColor", .{});
        send(void, window, "setBackgroundColor:", .{black});

        // A layer-hosting content view: AppKit never draws into the layer,
        // so its contents stay what we set.
        const view = send(id, window, "contentView", .{});
        const layer = objc.retain(send(id, objc.class("CALayer") catch return error.RuntimeUnavailable, "layer", .{}));
        if (layer == null or view == null) return error.WindowFailed;
        errdefer objc.release(layer);
        send(void, layer, "setContentsGravity:", .{a.kCAGravityResizeAspect});
        send(void, layer, "setBackgroundColor:", .{send(?*anyopaque, black, "CGColor", .{})});
        send(void, view, "setLayer:", .{layer});
        send(void, view, "setWantsLayer:", .{objc.YES});

        send(void, window, "center", .{});
        send(void, window, "makeKeyAndOrderFront:", .{@as(id, null)});
        send(void, app, "activateIgnoringOtherApps:", .{objc.YES});

        const self = gpa.create(Window) catch return error.WindowFailed;
        self.* = .{
            .gpa = gpa,
            .api = a,
            .app = app,
            .window = window,
            .view = view,
            .layer = layer,
            .colorspace = a.CGColorSpaceCreateDeviceRGB(),
            .date_cls = objc.getClass("NSDate"),
            .catransaction_cls = objc.getClass("CATransaction"),
        };
        return self;
    }

    pub fn close(self: *Window) void {
        const pool = objc.Pool.push();
        defer pool.pop();
        self.setContents(null);
        send(void, self.window, "close", .{});
        objc.release(self.layer);
        objc.release(self.window);
        self.api.CGColorSpaceRelease(self.colorspace);
        self.gpa.destroy(self);
    }

    /// False once the user closed the window.
    pub fn isOpen(self: *Window) bool {
        if (!self.open_) return false;
        self.open_ = send(objc.BOOL, self.window, "isVisible", .{}) != objc.NO or
            send(objc.BOOL, self.window, "isMiniaturized", .{}) != objc.NO;
        return self.open_;
    }

    /// Handles every pending event, waiting up to `timeout` seconds for the
    /// first. Keys: q or Esc quit, space resets, h toggles the HUD; other
    /// events go to AppKit as usual.
    pub fn poll(self: *Window, timeout: f64) Input {
        const pool = objc.Pool.push();
        defer pool.pop();
        var in: Input = .{};
        var wait = timeout;
        while (true) {
            const until = send(id, self.date_cls, "dateWithTimeIntervalSinceNow:", .{wait});
            const ev = send(id, self.app, "nextEventMatchingMask:untilDate:inMode:dequeue:", .{ NSEventMaskAny, until, self.api.NSDefaultRunLoopMode, objc.YES });
            if (ev == null) break;
            wait = 0;
            const kind = send(objc.NSUInteger, ev, "type", .{});
            if (kind == NSEventTypeKeyDown) {
                // Handled here and not passed on (AppKit would beep).
                switch (send(u16, ev, "keyCode", .{})) {
                    key_q, key_escape => in.quit = true,
                    key_space => in.reset = true,
                    key_h => in.toggle_hud = true,
                    key_c => in.clear = true,
                    key_tab => in.cycle_mode = true,
                    else => {},
                }
                continue;
            }
            if (self.mouse(ev, kind, &in)) continue;
            send(void, self.app, "sendEvent:", .{ev});
        }
        if (self.drag_from) |a| in.dragging = rectOf(a, self.drag_to);
        send(void, self.app, "updateWindows", .{});
        if (!self.isOpen()) in.quit = true;
        return in;
    }

    /// Mouse events over the image: a left drag selects a rectangle, a
    /// right click clears. Returns false for events AppKit should handle
    /// (the title bar, resizing).
    fn mouse(self: *Window, ev: id, kind: objc.NSUInteger, in: *Input) bool {
        switch (kind) {
            NSEventTypeLeftMouseDown, NSEventTypeLeftMouseUp, NSEventTypeRightMouseDown, NSEventTypeLeftMouseDragged => {},
            else => return false,
        }
        const loc = send(objc.CGPoint, ev, "locationInWindow", .{});
        const p = send(objc.CGPoint, self.view, "convertPoint:fromView:", .{ loc, @as(id, null) });
        const bounds = send(objc.CGRect, self.view, "bounds", .{});
        const inside = p.x >= 0 and p.y >= 0 and p.x < bounds.size.width and p.y < bounds.size.height;
        const img = self.toImage(p, bounds.size);
        switch (kind) {
            NSEventTypeLeftMouseDown => {
                if (!inside) return false;
                self.drag_from = img orelse return false;
                self.drag_to = self.drag_from.?;
            },
            NSEventTypeLeftMouseDragged => {
                if (self.drag_from == null) return false;
                if (img) |q| self.drag_to = q;
            },
            NSEventTypeLeftMouseUp => {
                const a = self.drag_from orelse return false;
                if (img) |q| self.drag_to = q;
                self.drag_from = null;
                const r = rectOf(a, self.drag_to);
                if (r[2] >= 4 and r[3] >= 4) in.selection = r;
            },
            NSEventTypeRightMouseDown => {
                if (!inside) return false;
                self.drag_from = null;
                in.clear = true;
            },
            else => unreachable,
        }
        return true;
    }

    /// A point of the content view (points, origin bottom left) in pixels
    /// of the image, which the layer shows aspect-fitted and centred.
    fn toImage(self: *const Window, p: objc.CGPoint, size: objc.CGSize) ?[2]f32 {
        if (self.image_w == 0 or self.image_h == 0 or size.width <= 0 or size.height <= 0) return null;
        const iw: f64 = @floatFromInt(self.image_w);
        const ih: f64 = @floatFromInt(self.image_h);
        const scale = @min(size.width / iw, size.height / ih);
        const ox = (size.width - iw * scale) / 2;
        const oy = (size.height - ih * scale) / 2;
        const x = std.math.clamp((p.x - ox) / scale, 0, iw - 1);
        const y = std.math.clamp((size.height - p.y - oy) / scale, 0, ih - 1);
        return .{ @floatCast(x), @floatCast(y) };
    }

    /// Shows `img` (BGRA, 4 channels), which must stay untouched until
    /// `release` is called (immediately if the image cannot be shown).
    pub fn present(self: *Window, img: Image(u8), release: Release) void {
        std.debug.assert(img.channels == 4);
        const a = self.api;
        self.image_w = img.width;
        self.image_h = img.height;
        const size = (@as(usize, img.height) - 1) * img.stride + img.rowLen();
        const slot = releaseSlot(release) orelse return release.func(release.ctx);
        const provider = a.CGDataProviderCreateWithData(slot, img.data, size, &providerReleased) orelse {
            freeSlot(slot);
            return release.func(release.ctx);
        };
        // From here the provider owns the release: it fires when the image
        // is gone (including if CGImageCreate fails).
        const image = a.CGImageCreate(img.width, img.height, 8, 32, img.stride, self.colorspace, kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little, provider, null, false, 0);
        a.CGDataProviderRelease(provider);
        if (image == null) return;
        const pool = objc.Pool.push();
        defer pool.pop();
        self.setContents(image);
        a.CGImageRelease(image);
    }

    fn setContents(self: *Window, image: ?*anyopaque) void {
        const t = self.catransaction_cls;
        send(void, t, "begin", .{});
        send(void, t, "setDisableActions:", .{objc.YES});
        send(void, self.layer, "setContents:", .{image});
        send(void, t, "commit", .{});
    }
};

// The data provider's release callback gets one pointer; a small pool of
// slots carries the caller's Release through it.
const max_slots = 32;
var slots: [max_slots]Release = undefined;
var slot_used: [max_slots]std.atomic.Value(bool) = @splat(.init(false));

fn releaseSlot(r: Release) ?*anyopaque {
    for (&slot_used, 0..) |*u, i| {
        if (u.cmpxchgStrong(false, true, .acquire, .monotonic) == null) {
            slots[i] = r;
            return @ptrCast(&slots[i]);
        }
    }
    return null;
}

fn freeSlot(p: *anyopaque) void {
    const r: *Release = @ptrCast(@alignCast(p));
    const i = (@intFromPtr(r) - @intFromPtr(&slots[0])) / @sizeOf(Release);
    slot_used[i].store(false, .release);
}

fn providerReleased(info: ?*anyopaque, _: ?*const anyopaque, _: usize) callconv(.c) void {
    const p = info orelse return;
    const r: *Release = @ptrCast(@alignCast(p));
    const copy = r.*;
    freeSlot(p);
    copy.func(copy.ctx);
}

fn rectOf(a: [2]f32, b: [2]f32) [4]f32 {
    return .{ @min(a[0], b[0]), @min(a[1], b[1]), @abs(a[0] - b[0]), @abs(a[1] - b[1]) };
}
