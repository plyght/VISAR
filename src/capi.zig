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

export fn visar_pyr_down(src: *const CImage, dst: *const CImage, b: c_int) c_int {
    const bd = border(b) catch |e| return fail(e);
    return run(dispatch(src, .{ dst, bd }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.pyrDown(exec(), gpa, s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

export fn visar_pyr_up(src: *const CImage, dst: *const CImage, b: c_int) c_int {
    const bd = border(b) catch |e| return fail(e);
    return run(dispatch(src, .{ dst, bd }, struct {
        fn f(comptime T: type, a: anytype, s: Image(T)) !void {
            return ip.pyrUp(exec(), gpa, s, try typed(T, a[0]), a[1]);
        }
    }.f));
}

/// cornerMinEigenVal: u8 or f32 one-channel `src`, f32 `dst`; ksize 1, 3,
/// 5, 7 or -1 (Scharr).
export fn visar_corner_min_eigen_val(src: *const CImage, dst: *const CImage, block_size: c_int, ksize: c_int, b: c_int) c_int {
    return cornerCall(src, dst, block_size, ksize, b, null);
}

/// cornerHarris: as visar_corner_min_eigen_val, with the Harris constant k.
export fn visar_corner_harris(src: *const CImage, dst: *const CImage, block_size: c_int, ksize: c_int, k: f64, b: c_int) c_int {
    return cornerCall(src, dst, block_size, ksize, b, k);
}

fn cornerCall(src: *const CImage, dst: *const CImage, block_size: c_int, ksize: c_int, b: c_int, k: ?f64) c_int {
    const bd = border(b) catch |e| return fail(e);
    if (block_size <= 0) return fail(error.InvalidArgument);
    const d = typed(f32, dst) catch |e| return fail(e);
    const o: ip.corner.CornerOptions = .{ .block_size = @intCast(block_size), .ksize = ksize, .border = bd };
    return run(switch (src.depth) {
        inline 0, 5 => |dep| blk: {
            const T = if (dep == 0) u8 else f32;
            const s = typed(T, src) catch |e| return fail(e);
            break :blk if (k) |kk| ip.cornerHarris(exec(), gpa, s, d, o, kk) else ip.cornerMinEigenVal(exec(), gpa, s, d, o);
        },
        else => error.Unsupported,
    });
}

/// goodFeaturesToTrack. Writes up to `cap` corners as (x, y) float pairs to
/// `corners`, strongest first, and the number found to `count` (which may
/// exceed `cap`). `mask` may be null (else a u8 image of the same size).
export fn visar_good_features_to_track(src: *const CImage, corners: ?[*]f32, cap: usize, count: *usize, max_corners: c_int, quality_level: f64, min_distance: f64, mask: ?*const CImage, block_size: c_int, gradient_size: c_int, use_harris: c_int, k: f64) c_int {
    if (max_corners < 0 or block_size <= 0) return fail(error.InvalidArgument);
    var o: ip.corner.GfttOptions = .{
        .max_corners = @intCast(max_corners),
        .quality_level = quality_level,
        .min_distance = min_distance,
        .block_size = @intCast(block_size),
        .gradient_size = gradient_size,
        .use_harris = use_harris != 0,
        .k = k,
    };
    if (mask) |m| o.mask = typed(u8, m) catch |e| return fail(e);
    const pts = switch (src.depth) {
        inline 0, 5 => |dep| blk: {
            const T = if (dep == 0) u8 else f32;
            const s = typed(T, src) catch |e| return fail(e);
            break :blk ip.goodFeaturesToTrack(exec(), gpa, s, o) catch |e| return fail(e);
        },
        else => return fail(error.Unsupported),
    };
    defer gpa.free(pts);
    count.* = pts.len;
    if (corners) |c| for (pts[0..@min(cap, pts.len)], 0..) |p, i| {
        c[2 * i] = p.x;
        c[2 * i + 1] = p.y;
    };
    return 0;
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

// ----------------------------------------------------------------- video

/// calcOpticalFlowPyrLK on 8-bit images. Points are `n` (x, y) float
/// pairs. `criteria_type`, `flags` use OpenCV's values (COUNT = 1, EPS = 2;
/// OPTFLOW_USE_INITIAL_FLOW = 4, OPTFLOW_LK_GET_MIN_EIGENVALS = 8). `err`
/// may be null.
export fn visar_calc_optical_flow_pyr_lk(prev: *const CImage, next: *const CImage, prev_pts: [*]const f32, next_pts: [*]f32, status: [*]u8, err: ?[*]f32, n: usize, win_w: c_int, win_h: c_int, max_level: c_int, criteria_type: c_int, max_count: c_int, epsilon: f64, flags: c_int, min_eig_threshold: f64) c_int {
    if (win_w <= 2 or win_h <= 2 or max_level < 0) return fail(error.InvalidArgument);
    const a = typed(u8, prev) catch |e| return fail(e);
    const b = typed(u8, next) catch |e| return fail(e);
    const o: visar.video.LKOptions = .{
        .win_width = @intCast(win_w),
        .win_height = @intCast(win_h),
        .max_level = @intCast(max_level),
        .max_count = if (criteria_type & 1 != 0) @intCast(std.math.clamp(max_count, 0, 100)) else 30,
        .epsilon = if (criteria_type & 2 != 0) epsilon else 0.01,
        .use_initial_flow = flags & 4 != 0,
        .get_min_eigenvals = flags & 8 != 0,
        .min_eig_threshold = @floatCast(min_eig_threshold),
    };
    const P = visar.video.optflow.Point2f;
    const pp: [*]const P = @ptrCast(@alignCast(prev_pts));
    const np: [*]P = @ptrCast(@alignCast(next_pts));
    return run(visar.video.calcOpticalFlowPyrLK(exec(), gpa, a, b, pp[0..n], np[0..n], status[0..n], if (err) |e| e[0..n] else null, o));
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

/// connectedComponentsWithStats. `labels` is i32 (VISAR_32S). Writes up to
/// `max` rows of stats (left, top, width, height, area as int32) and
/// centroids (cx, cy as double); returns the label count in `count`.
export fn visar_connected_components(src: *const CImage, labels: *const CImage, connectivity: c_int, stats: ?[*]i32, centroids: ?[*]f64, max: usize, count: *usize) c_int {
    const s = typed(u8, src) catch |e| return fail(e);
    const l = typed(i32, labels) catch |e| return fail(e);
    const conn: ip.components.Connectivity = if (connectivity == 4) .four else .eight;
    var r = ip.connectedComponentsWithStats(gpa, s, l, conn) catch |e| return fail(e);
    defer r.deinit(gpa);
    count.* = r.count;
    for (r.stats[0..@min(max, r.stats.len)], 0..) |st, i| {
        // An empty label (only the background, when every pixel is set)
        // is reported as OpenCV does: left -1, top INT_MAX.
        if (stats) |o| o[i * 5 ..][0..5].* = if (st.area == 0)
            .{ -1, std.math.maxInt(i32), 0, 0, 0 }
        else
            .{ @intCast(st.left), @intCast(st.top), @intCast(st.width), @intCast(st.height), @intCast(st.area) };
        if (centroids) |c| c[i * 2 ..][0..2].* = .{ st.cx, st.cy };
    }
    return 0;
}

// -------------------------------------------------------------- contours

/// Variable-length contour output, owned by VISAR until visar_contours_free.
pub const CContours = extern struct {
    count: usize,
    total: usize,
    points: ?[*]i32,
    starts: ?[*]usize,
    hierarchy: ?[*]i32,
};

/// findContours. mode: RETR_EXTERNAL 0, LIST 1, CCOMP 2, TREE 3; method:
/// CHAIN_APPROX_NONE 1, SIMPLE 2, TC89_L1 3, TC89_KCOS 4.
export fn visar_find_contours(src: *const CImage, mode: c_int, method: c_int, ox: c_int, oy: c_int, out: *CContours) c_int {
    out.* = .{ .count = 0, .total = 0, .points = null, .starts = null, .hierarchy = null };
    const s = typed(u8, src) catch |e| return fail(e);
    const m = std.enums.fromInt(ip.ContourMode, mode) orelse return fail(error.Unsupported);
    const a = std.enums.fromInt(ip.ContourMethod, method) orelse return fail(error.Unsupported);
    const c = ip.findContours(gpa, s, m, a, .{ .x = ox, .y = oy }) catch |e| return fail(e);
    if (c.len() == 0) return 0;
    out.* = .{
        .count = c.len(),
        .total = c.points.len,
        .points = @ptrCast(c.points.ptr),
        .starts = c.starts.ptr,
        .hierarchy = @ptrCast(c.hierarchy.ptr),
    };
    return 0;
}

export fn visar_contours_free(c: *CContours) void {
    if (c.count != 0) {
        var r: ip.Contours = .{
            .points = @as([*]ip.Point, @ptrCast(c.points.?))[0..c.total],
            .starts = c.starts.?[0 .. c.count + 1],
            .hierarchy = @as([*][4]i32, @ptrCast(c.hierarchy.?))[0..c.count],
        };
        r.deinit(gpa);
    }
    c.* = .{ .count = 0, .total = 0, .points = null, .starts = null, .hierarchy = null };
}

fn colorBytes(color: *const [4]f64) [4]u8 {
    var c: [4]u8 = undefined;
    for (color, &c) |v, *o| o.* = core.sat.cast(u8, v);
    return c;
}

fn lineType(t: c_int) !ip.LineType {
    return switch (t) {
        1, 4 => .four,
        0, 8 => .eight,
        else => error.Unsupported, // LINE_AA is not implemented
    };
}

fn contourView(points: [*]const i32, starts: [*]const usize, count: usize) ip.Contours {
    const total = starts[count];
    return .{
        .points = @constCast(@as([*]const ip.Point, @ptrCast(points))[0..total]),
        .starts = @constCast(starts[0 .. count + 1]),
        .hierarchy = &.{},
    };
}

/// drawContours on an 8-bit image. The contours use the visar_contours
/// layout (points, starts with count + 1 entries); hierarchy (count * 4)
/// may be NULL. thickness < 0 fills. line_type: 8 or 4.
export fn visar_draw_contours(img: *const CImage, points: [*]const i32, starts: [*]const usize, count: usize, hierarchy: ?[*]const i32, index: c_int, color: *const [4]f64, thickness: c_int, line_type: c_int, max_level: c_int, ox: c_int, oy: c_int) c_int {
    const im = typed(u8, img) catch |e| return fail(e);
    const lt = lineType(line_type) catch |e| return fail(e);
    const cs = contourView(points, starts, count);
    const h: ?[]const [4]i32 = if (hierarchy) |p| @as([*]const [4]i32, @ptrCast(p))[0..count] else null;
    const col = colorBytes(color);
    return run(ip.drawContours(gpa, im, cs, index, &col, thickness, lt, h, max_level, .{ .x = ox, .y = oy }));
}

/// fillPoly with the same polygon layout as visar_draw_contours.
export fn visar_fill_poly(img: *const CImage, points: [*]const i32, starts: [*]const usize, count: usize, color: *const [4]f64, line_type: c_int, ox: c_int, oy: c_int) c_int {
    const im = typed(u8, img) catch |e| return fail(e);
    const lt = lineType(line_type) catch |e| return fail(e);
    const cs = contourView(points, starts, count);
    const polys = gpa.alloc([]const ip.Point, count) catch |e| return fail(e);
    defer gpa.free(polys);
    for (polys, 0..) |*p, i| p.* = cs.get(i);
    const col = colorBytes(color);
    return run(ip.fillPoly(gpa, im, polys, &col, lt, .{ .x = ox, .y = oy }));
}

/// polylines with the same polygon layout as visar_draw_contours.
export fn visar_polylines(img: *const CImage, points: [*]const i32, starts: [*]const usize, count: usize, closed: c_int, color: *const [4]f64, thickness: c_int, line_type: c_int) c_int {
    const im = typed(u8, img) catch |e| return fail(e);
    const lt = lineType(line_type) catch |e| return fail(e);
    if (thickness < 0) return fail(error.InvalidArgument);
    const cs = contourView(points, starts, count);
    const col = colorBytes(color);
    for (0..count) |i| ip.polylines(im, cs.get(i), closed != 0, &col, @intCast(thickness), lt);
    return 0;
}

// ----------------------------------------------------------------- shape

/// Calls `f(pts, ctx)` with the point set as []const Point (depth
/// VISAR_32S) or []const Point2f (VISAR_32F).
fn withPoints(pts: ?*const anyopaque, n: usize, depth: c_int, ctx: anytype, comptime f: anytype) anyerror!void {
    if (n != 0 and pts == null) return error.InvalidArgument;
    return switch (depth) {
        4 => f(if (n == 0) @as([]const ip.Point, &.{}) else @as([*]const ip.Point, @ptrCast(@alignCast(pts.?)))[0..n], ctx),
        5 => f(if (n == 0) @as([]const ip.Point2f, &.{}) else @as([*]const ip.Point2f, @ptrCast(@alignCast(pts.?)))[0..n], ctx),
        else => error.Unsupported,
    };
}

/// Raster moments of a single-channel image (8U, 16U, 16S or 32F).
export fn visar_moments(img: *const CImage, binary: c_int, out: *ip.Moments) c_int {
    return run(switch (img.depth) {
        0 => blk: {
            const s = typed(u8, img) catch |e| break :blk e;
            break :blk if (ip.imageMoments(u8, exec(), gpa, s, binary != 0)) |m| {
                out.* = m;
            } else |e| e;
        },
        2 => blk: {
            const s = typed(u16, img) catch |e| break :blk e;
            break :blk if (ip.imageMoments(u16, exec(), gpa, s, binary != 0)) |m| {
                out.* = m;
            } else |e| e;
        },
        3 => blk: {
            const s = typed(i16, img) catch |e| break :blk e;
            break :blk if (ip.imageMoments(i16, exec(), gpa, s, binary != 0)) |m| {
                out.* = m;
            } else |e| e;
        },
        5 => blk: {
            const s = typed(f32, img) catch |e| break :blk e;
            break :blk if (ip.imageMoments(f32, exec(), gpa, s, binary != 0)) |m| {
                out.* = m;
            } else |e| e;
        },
        else => error.Unsupported,
    });
}

export fn visar_contour_moments(pts: ?*const anyopaque, n: usize, depth: c_int, out: *ip.Moments) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *ip.Moments) !void {
            o.* = ip.contourMoments(p);
        }
    }.f));
}

export fn visar_hu_moments(m: *const ip.Moments, hu: *[7]f64) void {
    hu.* = ip.huMoments(m.*);
}

export fn visar_contour_area(pts: ?*const anyopaque, n: usize, depth: c_int, oriented: c_int, out: *f64) c_int {
    return run(withPoints(pts, n, depth, .{ out, oriented != 0 }, struct {
        fn f(p: anytype, c: anytype) !void {
            c[0].* = ip.contourArea(p, c[1]);
        }
    }.f));
}

export fn visar_arc_length(pts: ?*const anyopaque, n: usize, depth: c_int, closed: c_int, out: *f64) c_int {
    return run(withPoints(pts, n, depth, .{ out, closed != 0 }, struct {
        fn f(p: anytype, c: anytype) !void {
            c[0].* = ip.arcLength(p, c[1]);
        }
    }.f));
}

export fn visar_bounding_rect(pts: ?*const anyopaque, n: usize, depth: c_int, out: *ip.Rect) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *ip.Rect) !void {
            o.* = ip.boundingRect(p);
        }
    }.f));
}

export fn visar_mask_bounding_rect(img: *const CImage, out: *ip.Rect) c_int {
    const s = typed(u8, img) catch |e| return fail(e);
    if (s.channels != 1) return fail(error.UnsupportedChannels);
    out.* = ip.maskBoundingRect(s);
    return 0;
}

/// Writes up to n hull indices (into the input) to `indices`.
export fn visar_convex_hull(pts: ?*const anyopaque, n: usize, depth: c_int, clockwise: c_int, return_points: c_int, indices: [*]i32, count: *usize) c_int {
    count.* = 0;
    return run(withPoints(pts, n, depth, .{ indices, count, clockwise != 0, return_points != 0 }, struct {
        fn f(p: anytype, c: anytype) !void {
            const h = try ip.convexHull(gpa, p, c[2], c[3]);
            defer gpa.free(h);
            for (h, 0..) |v, i| c[0][i] = @intCast(v);
            c[1].* = h.len;
        }
    }.f));
}

/// Writes up to hull_len defects (start, end, farthest, depth * 256).
export fn visar_convexity_defects(pts: [*]const ip.Point, n: usize, hull: [*]const i32, hull_len: usize, out: [*]i32, count: *usize) c_int {
    count.* = 0;
    const h = gpa.alloc(u32, hull_len) catch |e| return fail(e);
    defer gpa.free(h);
    for (h, hull[0..hull_len]) |*d, s| {
        if (s < 0) return fail(error.InvalidArgument);
        d.* = @intCast(s);
    }
    const d = ip.convexityDefects(gpa, pts[0..n], h) catch |e| return fail(e);
    defer gpa.free(d);
    for (d, 0..) |v, i| out[i * 4 ..][0..4].* = v;
    count.* = d.len;
    return 0;
}

export fn visar_is_contour_convex(pts: ?*const anyopaque, n: usize, depth: c_int, out: *c_int) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *c_int) !void {
            o.* = @intFromBool(ip.isContourConvex(p));
        }
    }.f));
}

export fn visar_min_area_rect(pts: ?*const anyopaque, n: usize, depth: c_int, out: *ip.RotatedRect) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *ip.RotatedRect) !void {
            o.* = try ip.minAreaRect(gpa, p);
        }
    }.f));
}

export fn visar_box_points(box: *const ip.RotatedRect, out: *[4]ip.Point2f) void {
    out.* = ip.boxPoints(box.*);
}

/// out: center x, center y, radius.
export fn visar_min_enclosing_circle(pts: ?*const anyopaque, n: usize, depth: c_int, out: *[3]f32) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *[3]f32) !void {
            const c = try ip.minEnclosingCircle(gpa, p);
            o.* = .{ c.cx, c.cy, c.radius };
        }
    }.f));
}

/// out: room for n points of the input's depth.
export fn visar_approx_poly_dp(pts: ?*const anyopaque, n: usize, depth: c_int, epsilon: f64, closed: c_int, out: *anyopaque, count: *usize) c_int {
    count.* = 0;
    return run(withPoints(pts, n, depth, .{ out, count, epsilon, closed != 0 }, struct {
        fn f(p: anytype, c: anytype) !void {
            const P = std.meta.Elem(@TypeOf(p));
            const o = @as([*]P, @ptrCast(@alignCast(c[0])))[0..p.len];
            c[1].* = try ip.approxPolyDP(gpa, p, o, c[2], c[3]);
        }
    }.f));
}

export fn visar_point_polygon_test(pts: ?*const anyopaque, n: usize, depth: c_int, x: f32, y: f32, measure_dist: c_int, out: *f64) c_int {
    return run(withPoints(pts, n, depth, .{ out, ip.Point2f{ .x = x, .y = y }, measure_dist != 0 }, struct {
        fn f(p: anytype, c: anytype) !void {
            c[0].* = ip.pointPolygonTest(p, c[1], c[2]);
        }
    }.f));
}

/// matchShapes of two contours; method CONTOURS_MATCH_I1..I3 (1..3).
export fn visar_match_shapes(a: ?*const anyopaque, na: usize, da: c_int, b: ?*const anyopaque, nb: usize, db: c_int, method: c_int, out: *f64) c_int {
    var ma: ip.Moments = undefined;
    var mb: ip.Moments = undefined;
    const r1 = visar_contour_moments(a, na, da, &ma);
    if (r1 != 0) return r1;
    const r2 = visar_contour_moments(b, nb, db, &mb);
    if (r2 != 0) return r2;
    if (method < 1 or method > 3) return fail(error.InvalidArgument);
    out.* = ip.matchShapes(ma, mb, @intCast(method)) catch |e| return fail(e);
    return 0;
}

/// fitEllipse (at least 5 points).
export fn visar_fit_ellipse(pts: ?*const anyopaque, n: usize, depth: c_int, out: *ip.RotatedRect) c_int {
    return run(withPoints(pts, n, depth, out, struct {
        fn f(p: anytype, o: *ip.RotatedRect) !void {
            o.* = try ip.fitEllipse(gpa, p);
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
