//! The image operations behind `visar gray`, `blur`, `threshold`, `flip`,
//! `rotate`, `edges`, `equalize`, `morph` and `resize`, and the steps of
//! `visar pipeline`. Each is parsed into an `Op` (from the same option tables
//! the help shows) and applied to an `AnyImage` in place.

const std = @import("std");
const visar = @import("visar");
const ip = visar.imgproc;
const Image = visar.Image;
const AnyImage = visar.AnyImage;
const args = @import("args.zig");
const Opt = args.Opt;
const Diag = args.Diag;

pub const Kind = enum { gray, blur, threshold, flip, rotate, edges, equalize, morph, resize };

pub const names = std.meta.fieldNames(Kind);

pub const Interp = enum { auto, nearest, linear, cubic, area };

pub const Op = union(Kind) {
    gray,
    blur: struct { sigma: f64, ksize: u32 },
    threshold: struct { value: ?f64, auto: ip.threshold_mod.Auto, invert: bool, max: f64 },
    flip: ip.geometry.FlipCode,
    rotate: ip.geometry.Rotate,
    edges: struct { ksize: i32 },
    equalize,
    morph: struct { op: ip.MorphOp, shape: ip.MorphShape, size: u32, iterations: u32 },
    resize: struct { w: ?u32, h: ?u32, scale: ?f64, interp: Interp },
};

// ---------------------------------------------------------------------------
// Option tables (also rendered by the help)
// ---------------------------------------------------------------------------

pub const blur_opts = [_]Opt{
    .{ .long = "sigma", .short = 's', .value = "S", .help = "Gaussian sigma in pixels (default 2; the kernel covers 3 sigma)" },
    .{ .long = "ksize", .short = 'k', .value = "K", .help = "Kernel size, odd, 1-127 (sigma then follows from it)" },
};
pub const threshold_opts = [_]Opt{
    .{ .long = "otsu", .help = "Pick the threshold with Otsu's method (the default without VALUE)" },
    .{ .long = "triangle", .help = "Pick the threshold with the triangle method" },
    .{ .long = "invert", .help = "Dark pixels become white instead" },
    .{ .long = "max", .value = "V", .help = "Value of pixels above the threshold (default 255)" },
};
pub const flip_opts = [_]Opt{
    .{ .long = "horizontal", .short = 'H', .help = "Mirror left to right" },
    .{ .long = "vertical", .short = 'V', .help = "Mirror top to bottom" },
    .{ .long = "both", .help = "Both, the same as rotating by 180" },
};
pub const edges_opts = [_]Opt{
    .{ .long = "ksize", .short = 'k', .value = "K", .help = "Sobel aperture: 1, 3, 5 or 7 (default 3)" },
};
pub const morph_opts = [_]Opt{
    .{ .long = "shape", .value = "SHAPE", .help = "Structuring element: rect, cross or ellipse (default rect)" },
    .{ .long = "size", .value = "N", .help = "Element size in pixels, 1-255 (default 3)" },
    .{ .long = "iterations", .short = 'n', .value = "N", .help = "Times to apply it, 1-100 (default 1)" },
};
pub const resize_opts = [_]Opt{
    .{ .long = "width", .short = 'W', .value = "W", .help = "Target width (height follows the aspect ratio)" },
    .{ .long = "height", .short = 'H', .value = "H", .help = "Target height (width follows the aspect ratio)" },
    .{ .long = "scale", .value = "S", .help = "Scale factor, e.g. 0.5 or 2" },
    .{ .long = "interp", .value = "MODE", .help = "nearest, linear, cubic or area (default: area to shrink, cubic to enlarge)" },
};

pub fn optsFor(k: Kind) []const Opt {
    return switch (k) {
        .gray, .rotate, .equalize => &.{},
        .blur => &blur_opts,
        .threshold => &threshold_opts,
        .flip => &flip_opts,
        .edges => &edges_opts,
        .morph => &morph_opts,
        .resize => &resize_opts,
    };
}

/// Whether the kernels behind `k` work on 8-bit samples only (16-bit inputs
/// are then decoded to 8 bits).
pub fn needs8bit(k: Kind) bool {
    return switch (k) {
        .threshold, .edges, .equalize, .morph => true,
        else => false,
    };
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Builds the op from its options and its own arguments (`pos`, the
/// arguments after the input and output).
pub fn parse(k: Kind, p: *const args.Parsed, pos: []const []const u8, d: *Diag) error{Usage}!Op {
    const max_pos: usize = switch (k) {
        .gray, .blur, .edges, .equalize => 0,
        .threshold, .flip, .rotate, .morph, .resize => 1,
    };
    if (pos.len > max_pos) {
        d.set("unexpected argument '{s}' for {s}", .{ pos[max_pos], @tagName(k) });
        d.setHint("'visar help {s}' shows what it takes", .{@tagName(k)});
        return error.Usage;
    }
    const arg: ?[]const u8 = if (pos.len > 0) pos[0] else null;
    switch (k) {
        .gray => return .gray,
        .equalize => return .equalize,
        .blur => {
            const ksize: u32 = if (p.value("ksize")) |v| try args.int(u32, d, "--ksize", v, 1, 127) else 0;
            if (ksize != 0 and ksize % 2 == 0) {
                d.set("--ksize must be odd, not {d}", .{ksize});
                d.setHint("try {d} or {d}", .{ ksize - 1, ksize + 1 });
                return error.Usage;
            }
            var sigma: f64 = if (p.value("sigma")) |v| try args.positive(d, "--sigma", v) else 0;
            if (sigma == 0 and ksize == 0) sigma = 2;
            if (sigma > 20 and ksize == 0) {
                d.set("--sigma {d} needs a kernel larger than 127 pixels", .{sigma});
                d.setHint("use a sigma of 20 or less, or resize the image down first", .{});
                return error.Usage;
            }
            return .{ .blur = .{ .sigma = sigma, .ksize = ksize } };
        },
        .threshold => {
            const otsu = p.flag("otsu");
            const tri = p.flag("triangle");
            if (otsu and tri) {
                d.set("--otsu and --triangle cannot be combined", .{});
                return error.Usage;
            }
            const value: ?f64 = if (arg) |a| std.fmt.parseFloat(f64, a) catch {
                d.set("the threshold must be a number, not '{s}'", .{a});
                d.setHint("e.g. 'visar threshold in.png out.png 128', or --otsu to pick one", .{});
                return error.Usage;
            } else null;
            if (value != null and (otsu or tri)) {
                d.set("give a threshold value or --{s}, not both", .{if (otsu) "otsu" else "triangle"});
                return error.Usage;
            }
            if (value) |v| if (v < 0 or v > 255) {
                d.set("the threshold must be between 0 and 255, not {d}", .{v});
                return error.Usage;
            };
            const max: f64 = if (p.value("max")) |v| @floatFromInt(try args.int(u8, d, "--max", v, 0, 255)) else 255;
            return .{ .threshold = .{
                .value = value,
                .auto = if (value != null) .none else if (tri) .triangle else .otsu,
                .invert = p.flag("invert"),
                .max = max,
            } };
        },
        .flip => {
            var code: ?ip.geometry.FlipCode = null;
            var n: usize = 0;
            if (p.flag("horizontal")) {
                code = .horizontal;
                n += 1;
            }
            if (p.flag("vertical")) {
                code = .vertical;
                n += 1;
            }
            if (p.flag("both")) {
                code = .both;
                n += 1;
            }
            if (arg) |a| {
                code = if (eqlAny(a, &.{ "h", "horizontal", "x" })) .horizontal else if (eqlAny(a, &.{ "v", "vertical", "y" })) .vertical else if (eqlAny(a, &.{ "both", "xy" })) .both else {
                    d.set("unknown flip direction '{s}'", .{a});
                    d.setHint("use horizontal, vertical or both", .{});
                    return error.Usage;
                };
                n += 1;
            }
            if (n > 1) {
                d.set("choose one flip direction", .{});
                d.setHint("--both mirrors both ways", .{});
                return error.Usage;
            }
            return .{ .flip = code orelse {
                d.set("flip needs a direction", .{});
                d.setHint("add --horizontal, --vertical or --both", .{});
                return error.Usage;
            } };
        },
        .rotate => {
            const a = arg orelse {
                d.set("rotate needs an angle", .{});
                d.setHint("add 90, 180 or 270 (clockwise), e.g. 'visar rotate in.png out.png 90'", .{});
                return error.Usage;
            };
            const r: ip.geometry.Rotate = if (eqlAny(a, &.{ "90", "cw", "-270" })) .cw90 else if (eqlAny(a, &.{ "180", "-180" })) .r180 else if (eqlAny(a, &.{ "270", "-90", "ccw" })) .ccw90 else {
                d.set("cannot rotate by '{s}'", .{a});
                d.setHint("rotate turns by quarter turns: 90, 180 or 270 degrees clockwise (-90 = 270)", .{});
                return error.Usage;
            };
            return .{ .rotate = r };
        },
        .edges => {
            const k_ = if (p.value("ksize")) |v| try args.int(i32, d, "--ksize", v, 1, 7) else 3;
            if (@mod(k_, 2) == 0) {
                d.set("--ksize must be 1, 3, 5 or 7, not {d}", .{k_});
                return error.Usage;
            }
            return .{ .edges = .{ .ksize = k_ } };
        },
        .morph => {
            const a = arg orelse {
                d.set("morph needs an operation", .{});
                d.setHint("one of erode, dilate, open, close, gradient, tophat, blackhat", .{});
                return error.Usage;
            };
            return .{ .morph = .{
                .op = try args.choice(ip.MorphOp, d, "morphology operation", a),
                .shape = if (p.value("shape")) |v| try args.choice(ip.MorphShape, d, "element shape", v) else .rect,
                .size = if (p.value("size")) |v| try args.int(u32, d, "--size", v, 1, 255) else 3,
                .iterations = if (p.value("iterations")) |v| try args.int(u32, d, "--iterations", v, 1, 100) else 1,
            } };
        },
        .resize => return parseResize(p, arg, d),
    }
}

fn eqlAny(s: []const u8, opts: []const []const u8) bool {
    for (opts) |o| if (std.ascii.eqlIgnoreCase(s, o)) return true;
    return false;
}

fn parseResize(p: *const args.Parsed, arg: ?[]const u8, d: *Diag) error{Usage}!Op {
    var w: ?u32 = if (p.value("width")) |v| try args.int(u32, d, "--width", v, 1, args.max_side) else null;
    var h: ?u32 = if (p.value("height")) |v| try args.int(u32, d, "--height", v, 1, args.max_side) else null;
    var scale: ?f64 = if (p.value("scale")) |v| try args.positive(d, "--scale", v) else null;
    if (arg) |a| {
        if (w != null or h != null or scale != null) {
            d.set("give the size either as '{s}' or with --width/--height/--scale, not both", .{a});
            return error.Usage;
        }
        if (std.mem.endsWith(u8, a, "%")) {
            scale = (try args.positive(d, "the percentage", a[0 .. a.len - 1])) / 100;
        } else {
            const s = try args.size(d, a, true);
            w = s.w;
            h = s.h;
        }
    }
    if (scale != null and (w != null or h != null)) {
        d.set("--scale cannot be combined with --width or --height", .{});
        return error.Usage;
    }
    if (scale == null and w == null and h == null) {
        d.set("resize needs a target size", .{});
        d.setHint("e.g. 640x480, 640x (keep aspect), 50%, --width 640 or --scale 0.5", .{});
        return error.Usage;
    }
    if (scale) |s| if (s > 64) {
        d.set("--scale {d} is too large (at most 64)", .{s});
        return error.Usage;
    };
    const interp: Interp = if (p.value("interp")) |v| try args.choice(Interp, d, "interpolation", v) else .auto;
    return .{ .resize = .{ .w = w, .h = h, .scale = scale, .interp = interp } };
}

// ---------------------------------------------------------------------------
// Applying
// ---------------------------------------------------------------------------

pub const Env = struct {
    exec: visar.Exec,
    gpa: std.mem.Allocator,
};

/// What an op found out, for the status line and --json.
pub const Report = struct {
    /// The threshold actually used (Otsu's or the triangle method's pick).
    threshold: ?f64 = null,
    /// The interpolation `auto` resolved to.
    interp: ?Interp = null,
};

pub const ApplyError = error{ Unsupported, UnsupportedChannels, SizeMismatch, InvalidSize, InvalidKernel, OutOfBounds, OutOfMemory, UnsupportedDepth };

/// Applies `op` to `img`, replacing it with the result.
pub fn apply(env: Env, op: Op, img: *AnyImage, report: *Report) ApplyError!void {
    switch (op) {
        .gray => try toGray(env, img),
        .blur => |b| try mapSame(env, img, b, struct {
            fn f(e: Env, s: anytype, d: @TypeOf(s), o: @TypeOf(b)) !void {
                try ip.gaussianBlur(e.exec, e.gpa, s, d, .{ .kw = o.ksize, .kh = o.ksize, .sigma_x = o.sigma });
            }
        }.f),
        .flip => |code| try mapSame(env, img, code, struct {
            fn f(e: Env, s: anytype, d: @TypeOf(s), c: ip.geometry.FlipCode) !void {
                try ip.flip(e.exec, s, d, c);
            }
        }.f),
        .rotate => |r| switch (img.*) {
            inline .u8, .u16, .f32 => |s, tag| {
                const quarter = r != .r180;
                var d = try Image(@TypeOf(s).Elem).init(env.gpa, if (quarter) s.height else s.width, if (quarter) s.width else s.height, s.channels);
                errdefer d.deinit(env.gpa);
                try ip.rotate(env.exec, s, d, r);
                replace(env, img, @unionInit(AnyImage, @tagName(tag), d));
            },
            else => return error.UnsupportedDepth,
        },
        .threshold => |t| {
            try toGray(env, img);
            const s = img.as(u8) orelse return error.UnsupportedDepth;
            var d = try Image(u8).initLike(env.gpa, s);
            errdefer d.deinit(env.gpa);
            const used = try ip.threshold(env.exec, s, d, t.value orelse 0, t.max, if (t.invert) .binary_inv else .binary, t.auto);
            report.threshold = used;
            replace(env, img, .{ .u8 = d });
        },
        .edges => |e| try edges(env, img, e.ksize),
        .equalize => try equalize(env, img),
        .morph => |m| {
            const s = img.as(u8) orelse return error.UnsupportedDepth;
            const buf = try env.gpa.alloc(u8, @as(usize, m.size) * m.size);
            defer env.gpa.free(buf);
            const el = ip.structuringElement(m.shape, m.size, m.size, buf);
            var d = try Image(u8).initLike(env.gpa, s);
            errdefer d.deinit(env.gpa);
            try ip.morphologyEx(env.exec, env.gpa, s, d, m.op, el, m.iterations);
            replace(env, img, .{ .u8 = d });
        },
        .resize => |r| {
            const size = targetSize(img.width(), img.height(), r.w, r.h, r.scale);
            const interp: Interp = if (r.interp != .auto) r.interp else if (size[0] <= img.width() and size[1] <= img.height()) .area else .cubic;
            report.interp = interp;
            const lib: ip.Interp = switch (interp) {
                .nearest => .nearest,
                .linear => .linear,
                .cubic => .cubic,
                .area, .auto => .area,
            };
            switch (img.*) {
                inline .u8, .u16, .f32 => |s, tag| {
                    var d = try Image(@TypeOf(s).Elem).init(env.gpa, size[0], size[1], s.channels);
                    errdefer d.deinit(env.gpa);
                    try ip.resize(env.exec, env.gpa, s, d, lib);
                    replace(env, img, @unionInit(AnyImage, @tagName(tag), d));
                },
                else => return error.UnsupportedDepth,
            }
        },
    }
}

/// Width and height after resizing; a side left out follows the aspect ratio.
pub fn targetSize(sw: u32, sh: u32, w: ?u32, h: ?u32, scale: ?f64) [2]u32 {
    const fw: f64 = @floatFromInt(sw);
    const fh: f64 = @floatFromInt(sh);
    if (scale) |s| return .{ clampSide(fw * s), clampSide(fh * s) };
    if (w != null and h != null) return .{ w.?, h.? };
    if (w) |ww| return .{ ww, clampSide(fh * @as(f64, @floatFromInt(ww)) / fw) };
    const hh = h.?;
    return .{ clampSide(fw * @as(f64, @floatFromInt(hh)) / fh), hh };
}

fn clampSide(v: f64) u32 {
    return @intFromFloat(std.math.clamp(@round(v), 1, @as(f64, args.max_side)));
}

fn replace(env: Env, img: *AnyImage, new: AnyImage) void {
    img.deinit(env.gpa);
    img.* = new;
}

/// Runs `f(env, src, dst, extra)` into a new image of the same shape, for
/// every sample type the kernels support.
fn mapSame(env: Env, img: *AnyImage, extra: anytype, comptime f: anytype) ApplyError!void {
    switch (img.*) {
        inline .u8, .u16, .f32 => |s, tag| {
            var d = try Image(@TypeOf(s).Elem).initLike(env.gpa, s);
            errdefer d.deinit(env.gpa);
            try f(env, s, d, extra);
            replace(env, img, @unionInit(AnyImage, @tagName(tag), d));
        },
        else => return error.UnsupportedDepth,
    }
}

/// Gray (1 channel) from gray, gray+alpha, RGB or RGBA; unchanged if gray.
fn toGray(env: Env, img: *AnyImage) ApplyError!void {
    switch (img.*) {
        inline .u8, .u16, .f32 => |s, tag| {
            if (s.channels == 1) return;
            var d = try Image(@TypeOf(s).Elem).init(env.gpa, s.width, s.height, 1);
            errdefer d.deinit(env.gpa);
            switch (s.channels) {
                2 => for (0..s.height) |y| {
                    const sr = s.row(y);
                    for (d.row(y), 0..) |*o, x| o.* = sr[x * 2];
                },
                3 => try ip.cvtColor(env.exec, s, d, .rgb2gray),
                4 => try ip.cvtColor(env.exec, s, d, .rgba2gray),
                else => return error.UnsupportedChannels,
            }
            replace(env, img, @unionInit(AnyImage, @tagName(tag), d));
        },
        else => return error.UnsupportedDepth,
    }
}

/// Sobel gradient magnitude, sqrt(gx^2 + gy^2), saturated to 8 bits.
fn edges(env: Env, img: *AnyImage, ksize: i32) ApplyError!void {
    try toGray(env, img);
    const s = img.as(u8) orelse return error.UnsupportedDepth;
    var gx = try Image(i16).init(env.gpa, s.width, s.height, 1);
    defer gx.deinit(env.gpa);
    var gy = try Image(i16).init(env.gpa, s.width, s.height, 1);
    defer gy.deinit(env.gpa);
    try ip.sobel(env.exec, env.gpa, s, gx, .{ .dx = 1, .dy = 0, .ksize = ksize });
    try ip.sobel(env.exec, env.gpa, s, gy, .{ .dx = 0, .dy = 1, .ksize = ksize });
    var d = try Image(u8).initLike(env.gpa, s);
    errdefer d.deinit(env.gpa);
    const C = struct { gx: Image(i16), gy: Image(i16), d: Image(u8) };
    env.exec.parallelFor(s.height, 16, C{ .gx = gx, .gy = gy, .d = d }, struct {
        fn run(c: C, y0: usize, y1: usize) void {
            for (y0..y1) |y| {
                for (c.d.row(y), c.gx.row(y), c.gy.row(y)) |*o, a, b| {
                    const fa: f32 = @floatFromInt(a);
                    const fb: f32 = @floatFromInt(b);
                    o.* = @intFromFloat(@min(255, @sqrt(fa * fa + fb * fb) + 0.5));
                }
            }
        }
    }.run);
    replace(env, img, .{ .u8 = d });
}

/// Histogram equalisation. A colour image has its luma (Y of YCrCb)
/// equalised and keeps its hues and its alpha.
fn equalize(env: Env, img: *AnyImage) ApplyError!void {
    const s = img.as(u8) orelse return error.UnsupportedDepth;
    const gpa = env.gpa;
    var d = try Image(u8).initLike(gpa, s);
    errdefer d.deinit(gpa);
    switch (s.channels) {
        1 => try ip.equalizeHist(env.exec, s, d),
        3, 4 => {
            var rgb = if (s.channels == 4) try Image(u8).init(gpa, s.width, s.height, 3) else s.view();
            defer if (s.channels == 4) rgb.deinit(gpa);
            if (s.channels == 4) try ip.cvtColor(env.exec, s, rgb, .rgba2rgb);
            var ycc = try Image(u8).init(gpa, s.width, s.height, 3);
            defer ycc.deinit(gpa);
            try ip.cvtColor(env.exec, rgb, ycc, .rgb2ycrcb);
            var luma = try Image(u8).init(gpa, s.width, s.height, 1);
            defer luma.deinit(gpa);
            for (0..s.height) |y| {
                const r = ycc.row(y);
                for (luma.row(y), 0..) |*o, x| o.* = r[x * 3];
            }
            try ip.equalizeHist(env.exec, luma, luma);
            for (0..s.height) |y| {
                const r = ycc.row(y);
                for (luma.row(y), 0..) |v, x| r[x * 3] = v;
            }
            if (s.channels == 3) {
                try ip.cvtColor(env.exec, ycc, d, .ycrcb2rgb);
            } else {
                try ip.cvtColor(env.exec, ycc, rgb, .ycrcb2rgb);
                for (0..s.height) |y| {
                    const sr = s.row(y);
                    const rr = rgb.row(y);
                    const dr = d.row(y);
                    for (0..s.width) |x| {
                        @memcpy(dr[x * 4 ..][0..3], rr[x * 3 ..][0..3]);
                        dr[x * 4 + 3] = sr[x * 4 + 3];
                    }
                }
            }
        },
        else => return error.UnsupportedChannels,
    }
    replace(env, img, .{ .u8 = d });
}

/// A short description of an op, e.g. "blur sigma=2".
pub fn describe(buf: []u8, op: Op) []const u8 {
    return switch (op) {
        .gray => "gray",
        .equalize => "equalize",
        .blur => |b| if (b.ksize > 0)
            std.fmt.bufPrint(buf, "blur {d}x{d}", .{ b.ksize, b.ksize }) catch "blur"
        else
            std.fmt.bufPrint(buf, "blur sigma={d}", .{b.sigma}) catch "blur",
        .threshold => |t| if (t.value) |v|
            std.fmt.bufPrint(buf, "threshold {d}", .{v}) catch "threshold"
        else
            std.fmt.bufPrint(buf, "threshold ({s})", .{@tagName(t.auto)}) catch "threshold",
        .flip => |f| std.fmt.bufPrint(buf, "flip {s}", .{@tagName(f)}) catch "flip",
        .rotate => |r| switch (r) {
            .cw90 => "rotate 90",
            .r180 => "rotate 180",
            .ccw90 => "rotate 270",
        },
        .edges => |e| std.fmt.bufPrint(buf, "edges (Sobel {d}x{d})", .{ e.ksize, e.ksize }) catch "edges",
        .morph => |m| std.fmt.bufPrint(buf, "{s} {s} {d}x{d}", .{ @tagName(m.op), @tagName(m.shape), m.size, m.size }) catch "morph",
        .resize => "resize",
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn parseFor(k: Kind, argv: []const []const u8, d: *Diag) !Op {
    var p = try args.parse(std.testing.allocator, argv, &.{optsFor(k)}, d, .{});
    defer {
        p.positionals.deinit(std.testing.allocator);
        p.hits.deinit(std.testing.allocator);
    }
    return parse(k, &p, p.positionals.items, d);
}

test "resize sizes keep the aspect ratio" {
    try std.testing.expectEqual([2]u32{ 320, 240 }, targetSize(640, 480, 320, null, null));
    try std.testing.expectEqual([2]u32{ 640, 480 }, targetSize(1280, 960, null, 480, null));
    try std.testing.expectEqual([2]u32{ 320, 240 }, targetSize(640, 480, null, null, 0.5));
    try std.testing.expectEqual([2]u32{ 100, 50 }, targetSize(640, 480, 100, 50, null));
    try std.testing.expectEqual([2]u32{ 1, 1 }, targetSize(10, 10, null, null, 0.001));
}

test "resize arguments" {
    var d: Diag = .{};
    const a = try parseFor(.resize, &.{"640x"}, &d);
    try std.testing.expectEqual(@as(?u32, 640), a.resize.w);
    try std.testing.expectEqual(@as(?u32, null), a.resize.h);
    const b = try parseFor(.resize, &.{"50%"}, &d);
    try std.testing.expectEqual(@as(?f64, 0.5), b.resize.scale);
    const c = try parseFor(.resize, &.{ "--width", "100", "--interp", "nearest" }, &d);
    try std.testing.expectEqual(Interp.nearest, c.resize.interp);
    try std.testing.expectError(error.Usage, parseFor(.resize, &.{}, &d));
    try std.testing.expectError(error.Usage, parseFor(.resize, &.{ "--scale", "2", "--width", "3" }, &d));
    try std.testing.expectError(error.Usage, parseFor(.resize, &.{ "640x480", "--width", "3" }, &d));
}

test "op arguments" {
    var d: Diag = .{};
    try std.testing.expectEqual(ip.geometry.Rotate.ccw90, (try parseFor(.rotate, &.{"-90"}, &d)).rotate);
    try std.testing.expectError(error.Usage, parseFor(.rotate, &.{"45"}, &d));
    try std.testing.expectEqual(ip.geometry.FlipCode.horizontal, (try parseFor(.flip, &.{"--horizontal"}, &d)).flip);
    try std.testing.expectEqual(ip.geometry.FlipCode.vertical, (try parseFor(.flip, &.{"v"}, &d)).flip);
    try std.testing.expectError(error.Usage, parseFor(.flip, &.{}, &d));
    try std.testing.expectError(error.Usage, parseFor(.flip, &.{ "--horizontal", "--vertical" }, &d));
    const t = try parseFor(.threshold, &.{}, &d);
    try std.testing.expectEqual(ip.threshold_mod.Auto.otsu, t.threshold.auto);
    const t2 = try parseFor(.threshold, &.{ "100", "--invert" }, &d);
    try std.testing.expectEqual(@as(?f64, 100), t2.threshold.value);
    try std.testing.expect(t2.threshold.invert);
    try std.testing.expectError(error.Usage, parseFor(.threshold, &.{ "100", "--otsu" }, &d));
    try std.testing.expectError(error.Usage, parseFor(.blur, &.{ "--ksize", "4" }, &d));
    try std.testing.expectEqual(@as(f64, 2), (try parseFor(.blur, &.{}, &d)).blur.sigma);
    const m = try parseFor(.morph, &.{ "open", "--shape", "ellipse", "--size", "5" }, &d);
    try std.testing.expectEqual(ip.MorphOp.open, m.morph.op);
    try std.testing.expectEqual(ip.MorphShape.ellipse, m.morph.shape);
    try std.testing.expectError(error.Usage, parseFor(.morph, &.{"opne"}, &d));
    try std.testing.expectEqualStrings("did you mean 'open'? (choose from erode, dilate, open, close, gradient, tophat, blackhat)", d.hint);
    try std.testing.expectError(error.Usage, parseFor(.gray, &.{"extra"}, &d));
}

test "ops run on a small image" {
    const gpa = std.testing.allocator;
    const env: Env = .{ .exec = .serial, .gpa = gpa };
    var img: AnyImage = .{ .u8 = try Image(u8).init(gpa, 8, 6, 3) };
    defer img.deinit(gpa);
    for (img.u8.slice(), 0..) |*v, i| v.* = @intCast((i * 7) % 256);
    var r: Report = .{};
    try apply(env, .{ .rotate = .cw90 }, &img, &r);
    try std.testing.expectEqual(@as(u32, 6), img.width());
    try apply(env, .equalize, &img, &r);
    try std.testing.expectEqual(@as(u16, 3), img.channels());
    try apply(env, .{ .resize = .{ .w = 12, .h = null, .scale = null, .interp = .auto } }, &img, &r);
    try std.testing.expectEqual(@as(u32, 16), img.height());
    try std.testing.expectEqual(Interp.cubic, r.interp.?);
    try apply(env, .{ .threshold = .{ .value = null, .auto = .otsu, .invert = false, .max = 255 } }, &img, &r);
    try std.testing.expectEqual(@as(u16, 1), img.channels());
    try std.testing.expect(r.threshold != null);
    try apply(env, .{ .edges = .{ .ksize = 3 } }, &img, &r);
    try apply(env, .{ .morph = .{ .op = .close, .shape = .ellipse, .size = 3, .iterations = 2 } }, &img, &r);
    try apply(env, .{ .blur = .{ .sigma = 1, .ksize = 0 } }, &img, &r);
    try apply(env, .{ .flip = .both }, &img, &r);
    try apply(env, .gray, &img, &r);
    try std.testing.expectEqual(@as(u32, 12), img.width());
}
