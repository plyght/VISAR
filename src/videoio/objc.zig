//! Apple frameworks through the Objective-C runtime, loaded at run time.
//!
//! Nothing here is linked at build time: libobjc and the frameworks are
//! opened with `dlopen` (they are always present on macOS, in the dyld
//! shared cache) and every function, class and constant is looked up by
//! name. So VISAR builds for macOS from any host without Apple's SDK, as
//! PRD §6 requires, and fails with a clear error if something is missing.
//!
//! Messages go through `objc_msgSend`, cast to a C function pointer of the
//! exact signature for every call (`send` builds that type at compile time).
//! The two ABIs differ for large struct returns: on arm64 the caller passes
//! the result address in x8 and `objc_msgSend` works for everything; on
//! x86_64 a struct of more than 16 bytes (CMTime, CGRect) is returned
//! through a hidden first argument, and the call must go through
//! `objc_msgSend_stret` instead. Struct arguments (CMTime: 24 bytes, CGRect:
//! 32) need nothing special: Zig's C calling convention passes them as
//! clang would for the declared type.
//!
//! Only compiled for macOS (see videoio.zig and ui.zig).

const std = @import("std");
const builtin = @import("builtin");

pub const id = ?*anyopaque;
pub const SEL = ?*anyopaque;
pub const Class = ?*anyopaque;
/// Objective-C BOOL: `bool` on arm64, `signed char` on x86_64; one byte on
/// both, so u8 (0 = NO) is exact either way.
pub const BOOL = u8;
pub const YES: BOOL = 1;
pub const NO: BOOL = 0;
pub const NSInteger = isize;
pub const NSUInteger = usize;
pub const CGFloat = f64;

pub const CGPoint = extern struct { x: CGFloat, y: CGFloat };
pub const CGSize = extern struct { width: CGFloat, height: CGFloat };
pub const CGRect = extern struct { origin: CGPoint, size: CGSize };

/// CoreMedia's time: value / timescale seconds (24 bytes).
pub const CMTime = extern struct {
    value: i64,
    timescale: i32,
    flags: u32,
    epoch: i64,

    pub const flag_valid: u32 = 1;

    pub fn seconds(t: CMTime) f64 {
        if (t.timescale <= 0 or t.flags & flag_valid == 0) return 0;
        return @as(f64, @floatFromInt(t.value)) / @as(f64, @floatFromInt(t.timescale));
    }
};

comptime {
    std.debug.assert(@sizeOf(CMTime) == 24);
    std.debug.assert(@sizeOf(CGRect) == 32);
}

pub const Error = error{
    /// libobjc or a framework could not be opened, or lacks a symbol.
    RuntimeUnavailable,
};

const dlopen = std.c.dlopen;
const dlsym = std.c.dlsym;

/// The libraries the runtime layer opens. `system` is libSystem
/// (libdispatch and the block runtime live there).
pub const Lib = enum {
    objc,
    system,
    foundation,
    appkit,
    avfoundation,
    coremedia,
    corevideo,
    coregraphics,
    quartzcore,
    corefoundation,

    pub fn path(l: Lib) [*:0]const u8 {
        return switch (l) {
            .objc => "/usr/lib/libobjc.A.dylib",
            .system => "/usr/lib/libSystem.B.dylib",
            .foundation => "/System/Library/Frameworks/Foundation.framework/Foundation",
            .appkit => "/System/Library/Frameworks/AppKit.framework/AppKit",
            .avfoundation => "/System/Library/Frameworks/AVFoundation.framework/AVFoundation",
            .coremedia => "/System/Library/Frameworks/CoreMedia.framework/CoreMedia",
            .corevideo => "/System/Library/Frameworks/CoreVideo.framework/CoreVideo",
            .coregraphics => "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics",
            .quartzcore => "/System/Library/Frameworks/QuartzCore.framework/QuartzCore",
            .corefoundation => "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
        };
    }
};

var handles: [std.enums.values(Lib).len]?*anyopaque = @splat(null);
/// The library or symbol that failed last, for error messages.
pub var last_missing: [128]u8 = undefined;
pub var last_missing_len: usize = 0;

pub fn missing() []const u8 {
    return last_missing[0..last_missing_len];
}

fn noteMissing(name: []const u8) void {
    const n = @min(name.len, last_missing.len);
    @memcpy(last_missing[0..n], name[0..n]);
    last_missing_len = n;
}

/// Opens a library (once). Not thread-safe: call from the main thread
/// before starting anything that runs callbacks.
pub fn open(l: Lib) Error!*anyopaque {
    const i = @backingInt(l);
    if (handles[i]) |h| return h;
    const h = dlopen(l.path(), .{ .LAZY = true, .GLOBAL = true }) orelse {
        noteMissing(std.mem.span(l.path()));
        return error.RuntimeUnavailable;
    };
    handles[i] = h;
    return h;
}

/// A function from library `l`, typed as `F` (a C function type).
pub fn func(comptime F: type, l: Lib, comptime name: [:0]const u8) Error!*const F {
    const h = try open(l);
    const p = dlsym(h, name) orelse {
        noteMissing(name);
        return error.RuntimeUnavailable;
    };
    return @ptrCast(@alignCast(p));
}

/// The value of a global constant (e.g. an NSString * such as
/// AVMediaTypeVideo): dlsym returns the address of the variable.
pub fn constant(comptime T: type, l: Lib, comptime name: [:0]const u8) Error!T {
    const h = try open(l);
    const p = dlsym(h, name) orelse {
        noteMissing(name);
        return error.RuntimeUnavailable;
    };
    const ptr: *const T = @ptrCast(@alignCast(p));
    return ptr.*;
}

/// Like `constant`, but null when the symbol does not exist (newer API).
pub fn optionalConstant(comptime T: type, l: Lib, comptime name: [:0]const u8) ?T {
    const h = open(l) catch return null;
    const p = dlsym(h, name) orelse return null;
    const ptr: *const T = @ptrCast(@alignCast(p));
    return ptr.*;
}

/// The address of a data symbol itself (e.g. _NSConcreteGlobalBlock).
pub fn address(l: Lib, comptime name: [:0]const u8) Error!*anyopaque {
    const h = try open(l);
    return dlsym(h, name) orelse {
        noteMissing(name);
        return error.RuntimeUnavailable;
    };
}

// ---------------------------------------------------------------------------
// The runtime
// ---------------------------------------------------------------------------

pub const Runtime = struct {
    getClass: *const fn ([*:0]const u8) callconv(.c) Class,
    registerName: *const fn ([*:0]const u8) callconv(.c) SEL,
    msgSend: *const anyopaque,
    /// x86_64 only (arm64 has no stret variant).
    msgSend_stret: ?*const anyopaque,
    allocateClassPair: *const fn (Class, [*:0]const u8, usize) callconv(.c) Class,
    registerClassPair: *const fn (Class) callconv(.c) void,
    addMethod: *const fn (Class, SEL, *const anyopaque, [*:0]const u8) callconv(.c) BOOL,
    getProtocol: *const fn ([*:0]const u8) callconv(.c) ?*anyopaque,
    addProtocol: *const fn (Class, ?*anyopaque) callconv(.c) BOOL,
    poolPush: *const fn () callconv(.c) ?*anyopaque,
    poolPop: *const fn (?*anyopaque) callconv(.c) void,
};

var runtime: ?Runtime = null;

/// Loads libobjc and Foundation. Call from the main thread first; later
/// calls (from any thread) return the loaded runtime.
pub fn load() Error!*const Runtime {
    if (runtime) |*r| return r;
    runtime = .{
        .getClass = try func(fn ([*:0]const u8) callconv(.c) Class, .objc, "objc_getClass"),
        .registerName = try func(fn ([*:0]const u8) callconv(.c) SEL, .objc, "sel_registerName"),
        .msgSend = try func(anyopaque, .objc, "objc_msgSend"),
        .msgSend_stret = if (builtin.cpu.arch == .x86_64) try func(anyopaque, .objc, "objc_msgSend_stret") else null,
        .allocateClassPair = try func(fn (Class, [*:0]const u8, usize) callconv(.c) Class, .objc, "objc_allocateClassPair"),
        .registerClassPair = try func(fn (Class) callconv(.c) void, .objc, "objc_registerClassPair"),
        .addMethod = try func(fn (Class, SEL, *const anyopaque, [*:0]const u8) callconv(.c) BOOL, .objc, "class_addMethod"),
        .getProtocol = try func(fn ([*:0]const u8) callconv(.c) ?*anyopaque, .objc, "objc_getProtocol"),
        .addProtocol = try func(fn (Class, ?*anyopaque) callconv(.c) BOOL, .objc, "class_addProtocol"),
        .poolPush = try func(fn () callconv(.c) ?*anyopaque, .objc, "objc_autoreleasePoolPush"),
        .poolPop = try func(fn (?*anyopaque) callconv(.c) void, .objc, "objc_autoreleasePoolPop"),
    };
    // NSString, NSArray, NSDictionary, NSNumber, NSDate.
    _ = try open(.foundation);
    return &runtime.?;
}

fn rt() *const Runtime {
    return &runtime.?;
}

/// A class by name, or null if no loaded library defines it.
pub fn getClass(name: [*:0]const u8) Class {
    return rt().getClass(name);
}

/// A class by name, or error.RuntimeUnavailable (with the name noted).
pub fn class(comptime name: [:0]const u8) Error!Class {
    return rt().getClass(name) orelse {
        noteMissing(name);
        return error.RuntimeUnavailable;
    };
}

pub fn sel(name: [*:0]const u8) SEL {
    return rt().registerName(name);
}

/// True when a method returning `R` must go through objc_msgSend_stret:
/// only on x86_64, for structs the SysV ABI returns in memory (> 16 bytes).
pub fn needsStret(comptime R: type) bool {
    if (builtin.cpu.arch != .x86_64) return false;
    return switch (@typeInfo(R)) {
        .@"struct", .@"union", .array => @sizeOf(R) > 16,
        else => false,
    };
}

/// The C function type of a message with return type `R` and arguments
/// of the types in the tuple type `Args`: R (*)(id, SEL, Args...).
pub fn MsgFn(comptime R: type, comptime Args: type) type {
    const types = @typeInfo(Args).@"struct".field_types;
    var params: [types.len + 2]type = undefined;
    params[0] = id;
    params[1] = SEL;
    for (types, 0..) |T, i| {
        if (T == comptime_int or T == comptime_float or T == @TypeOf(null))
            @compileError("objc.send: give argument " ++ std.fmt.comptimePrint("{d}", .{i}) ++ " an explicit C type (@as)");
        params[i + 2] = T;
    }
    const attrs: [types.len + 2]std.builtin.Type.Fn.ParamAttributes = @splat(.{});
    return @Fn(&params, &attrs, R, .{ .@"callconv" = .c });
}

/// Sends `selector` to `receiver` (an object or a class) with `args` (a
/// tuple of C-typed values) and returns an `R`.
pub inline fn send(comptime R: type, receiver: anytype, selector: [*:0]const u8, args: anytype) R {
    return sendSel(R, receiver, sel(selector), args);
}

pub inline fn sendSel(comptime R: type, receiver: anytype, s: SEL, args: anytype) R {
    const F = MsgFn(R, @TypeOf(args));
    const r = rt();
    const target: *const anyopaque = if (comptime needsStret(R)) r.msgSend_stret.? else r.msgSend;
    const f: *const F = @ptrCast(@alignCast(target));
    const recv: id = @ptrCast(receiver);
    return @call(.auto, f, .{ recv, s } ++ args);
}

// ---------------------------------------------------------------------------
// Foundation helpers
// ---------------------------------------------------------------------------

/// An autorelease pool scope: `const p = objc.Pool.push(); defer p.pop();`
pub const Pool = struct {
    token: ?*anyopaque,

    pub fn push() Pool {
        return .{ .token = rt().poolPush() };
    }

    pub fn pop(p: Pool) void {
        rt().poolPop(p.token);
    }
};

pub fn retain(obj: id) id {
    if (obj == null) return null;
    return send(id, obj, "retain", .{});
}

pub fn release(obj: id) void {
    if (obj == null) return;
    send(void, obj, "release", .{});
}

/// [[cls alloc] init], owned by the caller (+1).
pub fn new(cls: Class) id {
    const a = send(id, cls, "alloc", .{});
    return send(id, a, "init", .{});
}

/// An autoreleased NSString from UTF-8 (null on failure).
pub fn nsString(text: [*:0]const u8) id {
    const cls = getClass("NSString") orelse return null;
    return send(id, cls, "stringWithUTF8String:", .{text});
}

/// The UTF-8 contents of an NSString, valid while the string (and the
/// enclosing autorelease pool) lives. Empty for nil.
pub fn utf8(str: id) []const u8 {
    if (str == null) return "";
    const p = send(?[*:0]const u8, str, "UTF8String", .{}) orelse return "";
    return std.mem.span(p);
}

pub fn count(array: id) usize {
    if (array == null) return 0;
    return send(NSUInteger, array, "count", .{});
}

pub fn at(array: id, i: usize) id {
    return send(id, array, "objectAtIndex:", .{@as(NSUInteger, i)});
}

/// The description of an NSError (or "" for nil).
pub fn errorText(err: id) []const u8 {
    if (err == null) return "";
    return utf8(send(id, err, "localizedDescription", .{}));
}

/// Writes a FourCC such as '420v' as text.
pub fn fourcc(code: u32, buf: *[4]u8) []const u8 {
    buf.* = .{ @truncate(code >> 24), @truncate(code >> 16), @truncate(code >> 8), @truncate(code) };
    for (buf) |*c| if (c.* < 32 or c.* > 126) {
        c.* = '?';
    };
    return buf;
}

// ---------------------------------------------------------------------------
// Blocks
// ---------------------------------------------------------------------------

/// A global (capture-free) block literal, laid out as clang emits one, so
/// APIs that take a completion handler can call back into Zig. `Invoke` is
/// the block's function type, whose first parameter is the block itself.
pub fn GlobalBlock(comptime Invoke: type) type {
    return extern struct {
        isa: ?*anyopaque,
        flags: c_int,
        reserved: c_int = 0,
        invoke: *const Invoke,
        descriptor: *const Descriptor,

        pub const Descriptor = extern struct {
            reserved: c_ulong = 0,
            size: c_ulong,
            /// Present because flags has BLOCK_HAS_SIGNATURE.
            signature: [*:0]const u8,
        };

        const BLOCK_IS_GLOBAL: c_int = 1 << 28;
        const BLOCK_HAS_SIGNATURE: c_int = 1 << 30;

        pub fn init(invoke: *const Invoke, descriptor: *const Descriptor) Error!@This() {
            return .{
                .isa = try address(.system, "_NSConcreteGlobalBlock"),
                .flags = BLOCK_IS_GLOBAL | BLOCK_HAS_SIGNATURE,
                .invoke = invoke,
                .descriptor = descriptor,
            };
        }
    };
}

test "message function types" {
    const F = MsgFn(CMTime, @TypeOf(.{ @as(i32, 1), @as(f64, 2) }));
    const info = @typeInfo(F).@"fn";
    try std.testing.expectEqual(4, info.param_types.len);
    try std.testing.expect(info.param_types[0].? == id);
    try std.testing.expect(info.param_types[3].? == f64);
    try std.testing.expect(info.return_type.? == CMTime);
    // Large structs come back via objc_msgSend_stret on x86_64 only.
    try std.testing.expectEqual(builtin.cpu.arch == .x86_64, needsStret(CMTime));
    try std.testing.expectEqual(builtin.cpu.arch == .x86_64, needsStret(CGRect));
    try std.testing.expect(!needsStret(CGSize));
    try std.testing.expect(!needsStret(f64));
    var b: [4]u8 = undefined;
    try std.testing.expectEqualStrings("420v", fourcc(0x34323076, &b));
}
