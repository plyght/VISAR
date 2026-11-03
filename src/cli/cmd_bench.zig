//! `visar bench`: times VISAR's own kernels and codecs on a synthetic image,
//! in Zig, with nothing else in the loop. Each kernel runs once to warm up,
//! then repeatedly until its time budget is spent; the table shows the
//! median and the best run and the throughput in megapixels per second.

const std = @import("std");
const builtin = @import("builtin");
const visar = @import("visar");
const Io = std.Io;
const ip = visar.imgproc;
const Image = visar.Image;
const args = @import("args.zig");
const term = @import("term.zig");
const cx = @import("ctx.zig");
const Ctx = cx.Ctx;
const cmd_info = @import("cmd_info.zig");

pub const opts = [_]args.Opt{
    .{ .long = "size", .value = "WxH", .help = "Image size (default 1920x1080)" },
    .{ .long = "filter", .value = "TEXT", .help = "Only kernels whose name contains TEXT (comma-separated: any of them)" },
    .{ .long = "time", .value = "S", .help = "Seconds to spend on each kernel (default 0.25)" },
    .{ .long = "list", .help = "List the kernels and exit" },
};

/// The images and buffers every kernel works on.
const Bench = struct {
    gpa: std.mem.Allocator,
    exec: visar.Exec,
    rgb: Image(u8),
    gray: Image(u8),
    mask: Image(u8),
    out_rgb: Image(u8),
    out_gray: Image(u8),
    half_rgb: Image(u8),
    rot: Image(u8),
    i16_img: Image(i16),
    labels: Image(i32),
    el3: ip.morph.Element,
    el5: ip.morph.Element,
    el_buf: [9 + 25]u8,
    png: []u8,
    jpeg: []u8,
    qoi: []u8,

    fn deinit(b: *Bench) void {
        inline for (.{ "rgb", "gray", "mask", "out_rgb", "out_gray", "half_rgb", "rot", "i16_img", "labels" }) |f| @field(b, f).deinit(b.gpa);
        b.gpa.free(b.png);
        b.gpa.free(b.jpeg);
        b.gpa.free(b.qoi);
    }
};

const Kernel = struct {
    name: []const u8,
    /// What it runs on, for the table.
    input: enum { rgb, gray, mask },
    run: *const fn (b: *Bench) anyerror!void,
};

fn k(name: []const u8, input: @FieldType(Kernel, "input"), comptime f: fn (b: *Bench) anyerror!void) Kernel {
    return .{ .name = name, .input = input, .run = f };
}

const kernels = [_]Kernel{
    k("gray", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.cvtColor(b.exec, b.rgb, b.out_gray, .rgb2gray);
        }
    }.f),
    k("hsv", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.cvtColor(b.exec, b.rgb, b.out_rgb, .rgb2hsv);
        }
    }.f),
    k("resize/linear 0.5x", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.resize(b.exec, b.gpa, b.rgb, b.half_rgb, .linear);
        }
    }.f),
    k("resize/area 0.5x", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.resize(b.exec, b.gpa, b.rgb, b.half_rgb, .area);
        }
    }.f),
    k("resize/cubic 0.5x", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.resize(b.exec, b.gpa, b.rgb, b.half_rgb, .cubic);
        }
    }.f),
    k("blur/gaussian 5x5", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.gaussianBlur(b.exec, b.gpa, b.rgb, b.out_rgb, .{ .kw = 5, .kh = 5 });
        }
    }.f),
    k("blur/gaussian sigma 3", .gray, struct {
        fn f(b: *Bench) !void {
            try ip.gaussianBlur(b.exec, b.gpa, b.gray, b.out_gray, .{ .sigma_x = 3 });
        }
    }.f),
    k("blur/box 7x7", .gray, struct {
        fn f(b: *Bench) !void {
            try ip.blur(b.exec, b.gpa, b.gray, b.out_gray, 7, 7, ip.Border.default);
        }
    }.f),
    k("sobel/dx 3x3", .gray, struct {
        fn f(b: *Bench) !void {
            try ip.sobel(b.exec, b.gpa, b.gray, b.i16_img, .{ .dx = 1, .dy = 0 });
        }
    }.f),
    k("threshold/binary", .gray, struct {
        fn f(b: *Bench) !void {
            _ = try ip.threshold(b.exec, b.gray, b.out_gray, 127, 255, .binary, .none);
        }
    }.f),
    k("threshold/otsu", .gray, struct {
        fn f(b: *Bench) !void {
            _ = try ip.threshold(b.exec, b.gray, b.out_gray, 0, 255, .binary, .otsu);
        }
    }.f),
    k("equalize", .gray, struct {
        fn f(b: *Bench) !void {
            try ip.equalizeHist(b.exec, b.gray, b.out_gray);
        }
    }.f),
    k("morph/erode 3x3", .mask, struct {
        fn f(b: *Bench) !void {
            try ip.erode(b.exec, b.gpa, b.mask, b.out_gray, b.el3, 1);
        }
    }.f),
    k("morph/dilate 5x5 ellipse", .mask, struct {
        fn f(b: *Bench) !void {
            try ip.dilate(b.exec, b.gpa, b.mask, b.out_gray, b.el5, 1);
        }
    }.f),
    k("morph/open 3x3", .mask, struct {
        fn f(b: *Bench) !void {
            try ip.morphologyEx(b.exec, b.gpa, b.mask, b.out_gray, .open, b.el3, 1);
        }
    }.f),
    k("flip/horizontal", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.flip(b.exec, b.rgb, b.out_rgb, .horizontal);
        }
    }.f),
    k("rotate/90", .rgb, struct {
        fn f(b: *Bench) !void {
            try ip.rotate(b.exec, b.rgb, b.rot, .cw90);
        }
    }.f),
    k("warp/affine 15deg", .rgb, struct {
        fn f(b: *Bench) !void {
            const m = ip.rotationMatrix2D(@as(f64, @floatFromInt(b.rgb.width)) / 2, @as(f64, @floatFromInt(b.rgb.height)) / 2, 15, 1);
            try ip.warpAffine(b.exec, b.rgb, b.out_rgb, m, .linear, .constant, 0);
        }
    }.f),
    k("components/8-connected", .mask, struct {
        fn f(b: *Bench) !void {
            var cc = try ip.connectedComponentsWithStats(b.gpa, b.mask, b.labels, .eight);
            cc.deinit(b.gpa);
        }
    }.f),
    k("png/encode", .rgb, struct {
        fn f(b: *Bench) !void {
            b.gpa.free(try visar.codecs.encode(b.gpa, .png, .{ .u8 = b.rgb }, .{}));
        }
    }.f),
    k("png/decode", .rgb, struct {
        fn f(b: *Bench) !void {
            var img = try visar.codecs.decode(b.gpa, b.png, .{});
            img.deinit(b.gpa);
        }
    }.f),
    k("jpeg/encode q90", .rgb, struct {
        fn f(b: *Bench) !void {
            b.gpa.free(try visar.codecs.encode(b.gpa, .jpeg, .{ .u8 = b.rgb }, .{ .quality = 90 }));
        }
    }.f),
    k("jpeg/decode", .rgb, struct {
        fn f(b: *Bench) !void {
            var img = try visar.codecs.decode(b.gpa, b.jpeg, .{});
            img.deinit(b.gpa);
        }
    }.f),
    k("qoi/encode", .rgb, struct {
        fn f(b: *Bench) !void {
            b.gpa.free(try visar.codecs.encode(b.gpa, .qoi, .{ .u8 = b.rgb }, .{}));
        }
    }.f),
    k("qoi/decode", .rgb, struct {
        fn f(b: *Bench) !void {
            var img = try visar.codecs.decode(b.gpa, b.qoi, .{});
            img.deinit(b.gpa);
        }
    }.f),
};

/// Whether `name` passes the --filter (comma-separated substrings, any of
/// which may match; case-insensitive).
pub fn matches(name: []const u8, filter: ?[]const u8) bool {
    const f = filter orelse return true;
    var it = std.mem.tokenizeAny(u8, f, ", ");
    while (it.next()) |part| {
        if (term.containsIgnoreCase(name, part)) return true;
    }
    return false;
}

/// A deterministic test image: smooth gradients with texture and a few
/// shapes, so codecs and filters see something like a photograph.
fn fillScene(img: Image(u8)) void {
    var state: u32 = 0x9e3779b9;
    for (0..img.height) |y| {
        const r = img.row(y);
        for (0..img.width) |x| {
            state = state *% 1664525 +% 1013904223;
            const n: i32 = @as(i32, @intCast(state >> 28)) - 8;
            const fx: f32 = @floatFromInt(x);
            const fy: f32 = @floatFromInt(y);
            const disc = @mod(@as(i32, @intFromFloat(fx / 97)) + @as(i32, @intFromFloat(fy / 71)), 3) == 0;
            const base = [3]f32{ fx * 255 / @as(f32, @floatFromInt(img.width)), fy * 255 / @as(f32, @floatFromInt(img.height)), if (disc) 200 else 60 };
            for (0..3) |c| r[x * 3 + c] = @intCast(std.math.clamp(@as(i32, @intFromFloat(base[c])) + n, 0, 255));
        }
    }
}

const Row = struct {
    name: []const u8,
    input: []const u8,
    median_ms: f64,
    min_ms: f64,
    mpx_per_s: f64,
    iterations: usize,
};

const Report = struct {
    target: []const u8,
    build_mode: []const u8,
    threads: usize,
    width: u32,
    height: u32,
    kernels: []const Row,
};

pub fn run(ctx: *Ctx, p: *args.Parsed) !void {
    if (p.positionals.items.len > 0) {
        ctx.hint("pick kernels with --filter {s}", .{p.positionals.items[0]});
        return ctx.usage("bench takes no arguments", .{});
    }
    const filter = p.value("filter");
    if (p.flag("list")) {
        for (kernels) |kk| if (matches(kk.name, filter)) try ctx.out.print("{s}\n", .{kk.name});
        return;
    }
    var size: args.Size = .{ .w = 1920, .h = 1080 };
    if (p.value("size")) |v| size = try args.size(ctx.diag, v, false);
    const W = size.w.?;
    const H = size.h.?;
    if (W < 16 or H < 16) return ctx.usage("--size must be at least 16x16", .{});
    if (@as(u64, W) * H > 100_000_000) return ctx.usage("--size {d}x{d} is over 100 megapixels", .{ W, H });
    const budget_s = if (p.value("time")) |v| try args.positive(ctx.diag, "--time", v) else 0.25;

    var selected: std.ArrayList(*const Kernel) = .empty;
    for (&kernels) |*kk| if (matches(kk.name, filter)) try selected.append(ctx.arena, kk);
    if (selected.items.len == 0) {
        ctx.hint("'visar bench --list' shows the kernel names", .{});
        return ctx.usage("no kernel matches --filter '{s}'", .{filter.?});
    }
    if (builtin.mode == .debug) ctx.warn("this is a Debug build; build with -Doptimize=ReleaseFast for representative numbers", .{});

    const gpa = ctx.gpa;
    const exec = ctx.exec();
    var b: Bench = undefined;
    b.gpa = gpa;
    b.exec = exec;
    b.rgb = try Image(u8).init(gpa, W, H, 3);
    fillScene(b.rgb);
    b.gray = try Image(u8).init(gpa, W, H, 1);
    try ip.cvtColor(exec, b.rgb, b.gray, .rgb2gray);
    b.mask = try Image(u8).init(gpa, W, H, 1);
    _ = try ip.threshold(exec, b.gray, b.mask, 0, 255, .binary, .otsu);
    b.out_rgb = try Image(u8).initLike(gpa, b.rgb);
    b.out_gray = try Image(u8).initLike(gpa, b.gray);
    b.half_rgb = try Image(u8).init(gpa, @max(1, W / 2), @max(1, H / 2), 3);
    b.rot = try Image(u8).init(gpa, H, W, 3);
    b.i16_img = try Image(i16).init(gpa, W, H, 1);
    b.labels = try Image(i32).init(gpa, W, H, 1);
    b.el3 = ip.structuringElement(.rect, 3, 3, b.el_buf[0..9]);
    b.el5 = ip.structuringElement(.ellipse, 5, 5, b.el_buf[9..]);
    b.png = try visar.codecs.encode(gpa, .png, .{ .u8 = b.rgb }, .{});
    b.jpeg = try visar.codecs.encode(gpa, .jpeg, .{ .u8 = b.rgb }, .{ .quality = 90 });
    b.qoi = try visar.codecs.encode(gpa, .qoi, .{ .u8 = b.rgb }, .{});
    defer b.deinit();

    if (!ctx.json and !ctx.quiet) {
        ctx.se.print(ctx.err, .dim, "  {d} kernels on a {d}x{d} image, {d} thread{s}, about {d:.1} s\n", .{ selected.items.len, W, H, exec.threads(), if (exec.threads() == 1) "" else "s", budget_s * @as(f64, @floatFromInt(selected.items.len)) }) catch {};
        ctx.err.flush() catch {};
    }

    const io = ctx.io;
    var rows = try ctx.arena.alloc(Row, selected.items.len);
    var samples: std.ArrayList(f64) = .empty;
    var bar = ctx.progress(selected.items.len);
    const mpx = @as(f64, @floatFromInt(@as(u64, W) * H)) / 1e6;
    for (selected.items, 0..) |kk, i| {
        bar.update(i, kk.name);
        kk.run(&b) catch |e| {
            bar.finish();
            return ctx.fail("kernel {s} failed: {s}", .{ kk.name, cx.describe(e) });
        };
        samples.clearRetainingCapacity();
        const start = Io.Timestamp.now(io, .awake);
        while (samples.items.len < 3 or (term.msSince(io, start) < budget_s * 1000 and samples.items.len < 10_000)) {
            const t = Io.Timestamp.now(io, .awake);
            try kk.run(&b);
            try samples.append(ctx.arena, term.msSince(io, t));
        }
        std.mem.sort(f64, samples.items, {}, std.sort.asc(f64));
        const med = samples.items[samples.items.len / 2];
        rows[i] = .{
            .name = kk.name,
            .input = switch (kk.input) {
                .rgb => "rgb",
                .gray => "gray",
                .mask => "mask",
            },
            .median_ms = med,
            .min_ms = samples.items[0],
            .mpx_per_s = mpx / (med / 1000),
            .iterations = samples.items.len,
        };
        ctx.detail("{s}: {d} runs, median {d:.3} ms", .{ kk.name, samples.items.len, med });
    }
    bar.update(selected.items.len, "done");
    bar.finish();

    const w = ctx.out;
    if (ctx.json) {
        try std.json.Stringify.value(Report{
            .target = visar.core.target_description,
            .build_mode = cmd_info.modeName(),
            .threads = exec.threads(),
            .width = W,
            .height = H,
            .kernels = rows,
        }, .{ .whitespace = .indent_2 }, w);
        try w.writeAll("\n");
        return;
    }
    const s = ctx.so;
    const a = ctx.arena;
    var t = term.Table.init(&.{
        .{ .title = "KERNEL" },
        .{ .title = "INPUT" },
        .{ .title = "MEDIAN", .alignment = .right },
        .{ .title = "BEST", .alignment = .right },
        .{ .title = "MPX/S", .alignment = .right },
        .{ .title = "RUNS", .alignment = .right },
    });
    defer t.deinit(a);
    for (rows) |r| {
        try t.addRow(a, &.{
            r.name,
            r.input,
            term.fmtMs(try a.alloc(u8, 24), r.median_ms, s.unicode),
            term.fmtMs(try a.alloc(u8, 24), r.min_ms, s.unicode),
            try std.fmt.allocPrint(a, "{d:.0}", .{r.mpx_per_s}),
            try std.fmt.allocPrint(a, "{d}", .{r.iterations}),
        });
    }
    try t.write(w, s, 0);
    try w.writeAll("\n");
    try s.print(w, .dim, "{d}x{d} ({d:.2} MP), {d} thread{s}, {s}, {s} build\n", .{ W, H, mpx, exec.threads(), if (exec.threads() == 1) "" else "s", visar.core.target_description, cmd_info.modeName() });
}

test "filters match any comma-separated part, case-insensitively" {
    try std.testing.expect(matches("resize/area 0.5x", null));
    try std.testing.expect(matches("resize/area 0.5x", "RESIZE"));
    try std.testing.expect(matches("blur/box 7x7", "resize,blur"));
    try std.testing.expect(!matches("png/encode", "resize,blur"));
}

test "every kernel name is unique" {
    for (kernels, 0..) |a, i| for (kernels[i + 1 ..]) |b| try std.testing.expect(!std.mem.eql(u8, a.name, b.name));
}
