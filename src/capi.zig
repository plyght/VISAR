//! The stable C ABI (include/visar.h). Images are passed as `visar_image`
//! descriptors over caller-owned memory; functions return 0 on success or a
//! negative `visar_status`, with details in `visar_last_error()`.

const std = @import("std");
const builtin = @import("builtin");
const visar = @import("visar.zig");
const core = visar.core;
const ip = visar.imgproc;
const codecs = visar.codecs;
const Image = core.Image;
const Depth = core.Depth;

/// The system allocator where libc is linked: it reuses freed blocks, where
/// smp_allocator maps fresh pages (and page faults) for every large image.
const gpa = if (builtin.cpu.arch.isWasm()) std.heap.wasm_allocator else if (builtin.link_libc) std.heap.c_allocator else std.heap.smp_allocator;

pub const CImage = extern struct {
    width: u32,
    height: u32,
    channels: u32,
    /// VISAR_8U = 0 ... VISAR_16F = 7 (OpenCV numbering).
    depth: u32,
    /// Bytes between row starts.
    stride: usize,
    data: ?*anyopaque,
};

const Status = enum(c_int) {
    ok = 0,
    invalid_argument = -1,
    size_mismatch = -2,
    unsupported = -3,
    out_of_memory = -4,
    invalid_data = -5,
    internal = -6,
};

threadlocal var last_error: [256:0]u8 = @splat(0);

fn fail(err: anyerror) c_int {
    const msg = @errorName(err);
    const n = @min(msg.len, last_error.len - 1);
    @memcpy(last_error[0..n], msg[0..n]);
    last_error[n] = 0;
    const st: Status = switch (err) {
        error.OutOfMemory => .out_of_memory,
        error.SizeMismatch => .size_mismatch,
        error.Unsupported, error.UnsupportedChannels => .unsupported,
        error.InvalidData, error.Truncated, error.TooLarge => .invalid_data,
        error.InvalidSize, error.OutOfBounds, error.InvalidKernel, error.InvalidArgument => .invalid_argument,
        else => .internal,
    };
    return @backingInt(st);
}

fn run(result: anyerror!void) c_int {
    result catch |e| return fail(e);
    return 0;
}

// ------------------------------------------------------------ execution

var pool_mutex: std.atomic.Mutex = .unlocked;
var pool: ?core.Pool = null;
var want_threads: ?usize = null;

fn exec() core.Exec {
    if (!core.pool.threads_supported) return .serial;
    while (!pool_mutex.tryLock()) std.atomic.spinLoopHint();
    defer pool_mutex.unlock();
    if (pool == null) pool = core.Pool.init(gpa, std.Io.Threaded.global_single_threaded.io(), want_threads);
    return .on(&pool.?);
}

/// Sets the number of threads (0 = one per CPU, 1 = serial).
export fn visar_set_num_threads(n: c_int) void {
    while (!pool_mutex.tryLock()) std.atomic.spinLoopHint();
    defer pool_mutex.unlock();
    if (pool) |*p| p.deinit();
    pool = null;
    want_threads = if (n <= 0) null else @intCast(n);
}

export fn visar_get_num_threads() c_int {
    return @intCast(exec().threads());
}

export fn visar_version() [*:0]const u8 {
    return std.fmt.comptimePrint("{d}.{d}.{d}", .{ visar.version.major, visar.version.minor, visar.version.patch });
}

export fn visar_build_info() [*:0]const u8 {
    return core.target_description ++ "";
}

export fn visar_last_error() [*:0]const u8 {
    return &last_error;
}

// ----------------------------------------------------------- conversion

fn typed(comptime T: type, c: *const CImage) !Image(T) {
    if (c.data == null) return error.InvalidArgument;
    if (c.depth != @backingInt(Depth.of(T))) return error.Unsupported;
    if (c.channels == 0 or c.channels > core.image.max_channels) return error.InvalidArgument;
    if (c.stride % @sizeOf(T) != 0) return error.InvalidArgument;
    const stride = c.stride / @sizeOf(T);
    const len = (@as(usize, c.height) - 1) * stride + @as(usize, c.width) * c.channels;
    const p: [*]T = @ptrCast(@alignCast(c.data.?));
    return Image(T).fromSliceStrided(p[0..len], c.width, c.height, @intCast(c.channels), stride);
}

/// Calls `f(T, a)` with `a` the typed view of `c`, for the depths VISAR's
/// kernels support through the C ABI.
fn dispatch(c: *const CImage, ctx: anytype, comptime f: anytype) anyerror!void {
    return switch (c.depth) {
        0 => f(u8, ctx, try typed(u8, c)),
        2 => f(u16, ctx, try typed(u16, c)),
        3 => f(i16, ctx, try typed(i16, c)),
        5 => f(f32, ctx, try typed(f32, c)),
        else => error.Unsupported,
    };
}

// --------------------------------------------------------------- imgproc

/// Indexed by OpenCV's COLOR_* value.
const color_codes: [56]?ip.ColorCode = blk: {
    var t: [56]?ip.ColorCode = @splat(null);
    for ([_]ip.ColorCode{ .bgr2bgra, .bgra2bgr, .bgr2rgba, .rgba2bgr, .bgr2rgb, .bgra2rgba, .bgr2gray, .rgb2gray, .gray2bgr, .gray2bgra, .bgra2gray, .rgba2gray }, 0..) |c, i| t[i] = c;
    t[36] = .bgr2ycrcb;
    t[37] = .rgb2ycrcb;
    t[38] = .ycrcb2bgr;
    t[39] = .ycrcb2rgb;
    t[40] = .bgr2hsv;
    t[41] = .rgb2hsv;
    t[54] = .hsv2bgr;
    t[55] = .hsv2rgb;
    break :blk t;
};

export fn visar_cvt_color(src: *const CImage, dst: *const CImage, code: c_int) c_int {
    if (code < 0 or code >= color_codes.len) return fail(error.Unsupported);
    const cc = color_codes[@intCast(code)] orelse return fail(error.Unsupported);
    return run(dispatch(src, .{ dst, cc }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.cvtColor(exec(), s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

export fn visar_resize(src: *const CImage, dst: *const CImage, interp: c_int) c_int {
    const m = std.enums.fromInt(ip.Interp, interp) orelse return fail(error.Unsupported);
    return run(dispatch(src, .{ dst, m }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.resize(exec(), gpa, s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

fn border(b: c_int) !ip.Border {
    return std.enums.fromInt(ip.Border, b) orelse error.Unsupported;
}

export fn visar_gaussian_blur(src: *const CImage, dst: *const CImage, kw: c_int, kh: c_int, sx: f64, sy: f64, b: c_int) c_int {
    const bd = border(b) catch |e| return fail(e);
    if (kw < 0 or kh < 0) return fail(error.InvalidArgument);
    const o: ip.filter.GaussianOptions = .{ .kw = @intCast(kw), .kh = @intCast(kh), .sigma_x = sx, .sigma_y = sy, .border = bd };
    return run(dispatch(src, .{ dst, o }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.gaussianBlur(exec(), gpa, s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

export fn visar_box_filter(src: *const CImage, dst: *const CImage, kw: c_int, kh: c_int, normalize: c_int, b: c_int) c_int {
    const bd = border(b) catch |e| return fail(e);
    if (kw <= 0 or kh <= 0) return fail(error.InvalidArgument);
    const o: ip.filter.BoxOptions = .{ .kw = @intCast(kw), .kh = @intCast(kh), .normalize = normalize != 0, .border = bd };
    return run(dispatch(src, .{ dst, o }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.boxFilter(exec(), gpa, s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

export fn visar_sobel(src: *const CImage, dst: *const CImage, dx: c_int, dy: c_int, ksize: c_int, scale: f64, b: c_int) c_int {
    const bd = border(b) catch |e| return fail(e);
    if (dx < 0 or dy < 0) return fail(error.InvalidArgument);
    const o: ip.filter.SobelOptions = .{ .dx = @intCast(dx), .dy = @intCast(dy), .ksize = ksize, .scale = scale, .border = bd };
    return run(dispatch(src, .{ dst, o }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return switch (a[0].depth) {
                3 => if (T == u8) ip.sobel(exec(), gpa, s, try typed(i16, a[0]), a[1]) else error.Unsupported,
                5 => ip.sobel(exec(), gpa, s, try typed(f32, a[0]), a[1]),
                else => error.Unsupported,
            };
        }
    }.f));
}

/// `type` uses OpenCV's encoding: 0-4 plus THRESH_OTSU (8) or THRESH_TRIANGLE (16).
export fn visar_threshold(src: *const CImage, dst: *const CImage, thresh: f64, maxval: f64, kind: c_int, used: ?*f64) c_int {
    const k = std.enums.fromInt(ip.ThresholdType, kind & 7) orelse return fail(error.Unsupported);
    const auto: ip.threshold_mod.Auto = if (kind & 8 != 0) .otsu else if (kind & 16 != 0) .triangle else .none;
    var t: f64 = thresh;
    const r = run(dispatch(src, .{ dst, thresh, maxval, k, auto, &t }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            a[5].* = try ip.threshold(exec(), s, try typed(T, a[0]), a[1], a[2], a[3], a[4]);
        }
    }.f));
    if (used) |u| u.* = t;
    return r;
}

export fn visar_morphology(src: *const CImage, dst: *const CImage, op: c_int, shape: c_int, kw: c_int, kh: c_int, iterations: c_int) c_int {
    const o = std.enums.fromInt(ip.MorphOp, op) orelse return fail(error.Unsupported);
    const sh = std.enums.fromInt(ip.MorphShape, shape) orelse return fail(error.Unsupported);
    if (kw <= 0 or kh <= 0 or kw > 255 or kh > 255 or iterations < 0) return fail(error.InvalidArgument);
    var buf: [255 * 255]u8 = undefined;
    const el = ip.structuringElement(sh, @intCast(kw), @intCast(kh), &buf);
    return run(dispatch(src, .{ dst, o, el, @as(u32, @intCast(iterations)) }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.morphologyEx(exec(), gpa, s, try typed(T, a[0]), a[1], a[2], a[3]);
        }
    }.f));
}

/// OpenCV flip codes: 0 = vertical, 1 = horizontal, -1 = both.
export fn visar_flip(src: *const CImage, dst: *const CImage, code: c_int) c_int {
    const fc: ip.geometry.FlipCode = if (code == 0) .vertical else if (code > 0) .horizontal else .both;
    return run(dispatch(src, .{ dst, fc }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.flip(exec(), s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

/// OpenCV rotate codes: 0 = 90 cw, 1 = 180, 2 = 90 ccw.
export fn visar_rotate(src: *const CImage, dst: *const CImage, code: c_int) c_int {
    const rc = std.enums.fromInt(ip.geometry.Rotate, code) orelse return fail(error.Unsupported);
    return run(dispatch(src, .{ dst, rc }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.rotate(exec(), s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

export fn visar_warp_affine(src: *const CImage, dst: *const CImage, m: *const [6]f64, interp: c_int, b: c_int, value: f64) c_int {
    const bd = border(b) catch |e| return fail(e);
    const wi: ip.geometry.WarpInterp = if (interp == 0) .nearest else .linear;
    return run(dispatch(src, .{ dst, m.*, wi, bd, value }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.warpAffine(exec(), s, try typed(T, a[0]), a[1], a[2], a[3], a[4]);
        }
    }.f));
}

export fn visar_equalize_hist(src: *const CImage, dst: *const CImage) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const d = typed(u8, dst) catch |e| return fail(e);
    return run(ip.equalizeHist(exec(), s, d));
}

// ------------------------------------------------------------------ core

/// op: 0 add, 1 subtract, 2 absdiff, 3 multiply, 4 divide, 5 min, 6 max,
/// 7 and, 8 or, 9 xor.
export fn visar_arith(op: c_int, a: *const CImage, b: *const CImage, dst: *const CImage) c_int {
    const o = std.enums.fromInt(core.arith.BinOp, op) orelse return fail(error.Unsupported);
    return run(dispatch(a, .{ b, dst, o }, struct {
        fn f(comptime T: type, x: anytype, s: Image(T)) !void {
            return core.arith.binary(exec(), x[2], s, try typed(T, x[0]), try typed(T, x[1]));
        }
    }.f));
}

export fn visar_add_weighted(a: *const CImage, alpha: f64, b: *const CImage, beta: f64, gamma: f64, dst: *const CImage) c_int {
    return run(dispatch(a, .{ b, dst, alpha, beta, gamma }, struct {
        fn f(comptime T: type, x: anytype, s: Image(T)) !void {
            return core.arith.addWeighted(exec(), s, x[2], try typed(T, x[0]), x[3], x[4], try typed(T, x[1]));
        }
    }.f));
}

export fn visar_convert_to(src: *const CImage, dst: *const CImage, alpha: f64, beta: f64) c_int {
    return run(dispatch(src, .{ dst, alpha, beta }, struct {
        fn f(comptime S: type, x: anytype, s: Image(S)) !void {
            return dispatch(x[0], .{ s, x[1], x[2] }, struct {
                fn g(comptime D: type, y: anytype, d: Image(D)) !void {
                    return core.arith.convertTo(exec(), y[0], d, y[1], y[2]);
                }
            }.g);
        }
    }.f));
}

// ---------------------------------------------------------------- codecs

/// Decodes an image file held in memory. `mode` is OpenCV's IMREAD flag
/// (-1 unchanged, 0 gray, 1 color); `bgr` != 0 returns BGR order like cv2.
/// On success `out` describes a buffer owned by VISAR: release it with
/// visar_image_free.
export fn visar_imdecode(bytes: [*]const u8, len: usize, mode: c_int, bgr: c_int, out: *CImage) c_int {
    const m: codecs.Mode = switch (mode) {
        -1 => .unchanged,
        0 => .gray,
        1 => .color,
        else => return fail(error.Unsupported),
    };
    var img = codecs.decode(gpa, bytes[0..len], .{ .mode = m, .order = if (bgr != 0) .bgr else .rgb }) catch |e| return fail(e);
    switch (img) {
        inline else => |*t| {
            const T = @TypeOf(t.*).Elem;
            if (!t.isContiguous()) {
                const c = t.clone(gpa) catch |e| {
                    img.deinit(gpa);
                    return fail(e);
                };
                t.deinit(gpa);
                t.* = c;
            }
            out.* = .{ .width = t.width, .height = t.height, .channels = t.channels, .depth = @backingInt(Depth.of(T)), .stride = t.stride * @sizeOf(T), .data = @ptrCast(t.owned.?.ptr) };
        },
    }
    return 0;
}

export fn visar_image_free(img: *CImage) void {
    const d = std.enums.fromInt(Depth, img.depth) orelse return;
    switch (d) {
        inline else => |t| {
            const T = t.Type();
            if (img.data) |p| {
                const n = @as(usize, img.width) * img.height * img.channels;
                const ptr: [*]align(64) T = @ptrCast(@alignCast(p));
                gpa.free(ptr[0..n]);
            }
        },
    }
    img.data = null;
}

/// Encodes to the format named by `ext` (".png", ".jpg", ...). `bgr` != 0
/// means the pixels are in BGR order. Free the result with visar_free.
export fn visar_imencode(ext: [*:0]const u8, img: *const CImage, bgr: c_int, quality: c_int, out: *[*]u8, out_len: *usize) c_int {
    const fmt = codecs.formatFromExtension(std.mem.span(ext)) orelse return fail(error.Unsupported);
    const opts: codecs.EncodeOptions = .{ .order = if (bgr != 0) .bgr else .rgb, .quality = if (quality > 0 and quality <= 100) @intCast(quality) else 95 };
    const any: core.AnyImage = switch (img.depth) {
        0 => .{ .u8 = typed(u8, img) catch |e| return fail(e) },
        2 => .{ .u16 = typed(u16, img) catch |e| return fail(e) },
        5 => .{ .f32 = typed(f32, img) catch |e| return fail(e) },
        else => return fail(error.Unsupported),
    };
    const bytes = codecs.encode(gpa, fmt, any, opts) catch |e| return fail(e);
    out.* = bytes.ptr;
    out_len.* = bytes.len;
    return 0;
}

export fn visar_free(p: ?[*]u8, len: usize) void {
    if (p) |ptr| gpa.free(ptr[0..len]);
}

comptime {
    _ = builtin;
}
