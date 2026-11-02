//! The `visar` command-line tool.

const std = @import("std");
const visar = @import("visar");
const Io = std.Io;

const usage =
    \\usage: visar <command> [args]
    \\
    \\commands:
    \\  info                      version, target and thread count
    \\  inspect <image>           format, size, channels and depth
    \\  convert <in> <out> [-q N] decode <in> and encode to <out>'s format
    \\  resize <in> <out> <w> <h> [nearest|linear|cubic|area]
    \\  gray <in> <out>           convert to grayscale
    \\  blur <in> <out> <sigma>   Gaussian blur
    \\
;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var out_buf: [4096]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &out_buf);
    const out = &ow.interface;
    defer out.flush() catch {};
    var err_buf: [1024]u8 = undefined;
    var ew = Io.File.Writer.initStreaming(.stderr(), io, &err_buf);
    const err = &ew.interface;
    defer err.flush() catch {};

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        try err.writeAll(usage);
        std.process.exit(2);
    }
    const cmd = args[1];
    var pool = visar.Pool.init(gpa, io, null);
    defer pool.deinit();
    const exec = visar.Exec.on(&pool);

    if (std.mem.eql(u8, cmd, "info")) {
        try out.print("visar {f}\ntarget  {s}\nthreads {d}\n", .{ visar.version, visar.core.target_description, exec.threads() });
        return;
    }
    if (args.len < 3) {
        try err.writeAll(usage);
        std.process.exit(2);
    }
    var img = visar.codecs.read(gpa, io, args[2], .{}) catch |e| {
        try err.print("visar: cannot read {s}: {s}\n", .{ args[2], @errorName(e) });
        std.process.exit(1);
    };
    defer img.deinit(gpa);

    if (std.mem.eql(u8, cmd, "inspect")) {
        const bytes = try Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited);
        defer gpa.free(bytes);
        const fmt = visar.codecs.detect(bytes).?;
        try out.print("{s}: {s} {d}x{d} {d} channel(s) {s}\n", .{ args[2], @tagName(fmt), img.width(), img.height(), img.channels(), @tagName(img.depth()) });
        return;
    }
    if (args.len < 4) {
        try err.writeAll(usage);
        std.process.exit(2);
    }
    const dst_path = args[3];
    var result: ?visar.AnyImage = null;
    defer if (result) |*r| r.deinit(gpa);
    var quality: u8 = 95;
    if (std.mem.eql(u8, cmd, "convert")) {
        if (args.len >= 6 and std.mem.eql(u8, args[4], "-q")) quality = try std.fmt.parseInt(u8, args[5], 10);
    } else if (std.mem.eql(u8, cmd, "resize")) {
        if (args.len < 6) {
            try err.writeAll(usage);
            std.process.exit(2);
        }
        const w = try std.fmt.parseInt(u32, args[4], 10);
        const h = try std.fmt.parseInt(u32, args[5], 10);
        const interp: visar.imgproc.Interp = if (args.len >= 7) std.meta.stringToEnum(visar.imgproc.Interp, args[6]) orelse .linear else .linear;
        switch (img) {
            inline .u8, .u16, .f32 => |s, tag| {
                var d = try visar.Image(@TypeOf(s).Elem).init(gpa, w, h, s.channels);
                errdefer d.deinit(gpa);
                try visar.imgproc.resize(exec, gpa, s, d, interp);
                result = @unionInit(visar.AnyImage, @tagName(tag), d);
            },
            else => return error.Unsupported,
        }
    } else if (std.mem.eql(u8, cmd, "gray")) {
        const s = img.as(u8) orelse return error.Unsupported;
        var d = try visar.Image(u8).init(gpa, s.width, s.height, 1);
        errdefer d.deinit(gpa);
        const code: visar.imgproc.ColorCode = switch (s.channels) {
            1 => {
                d.copyFrom(s);
                result = .{ .u8 = d };
                return writeOut(gpa, io, err, dst_path, result.?, quality);
            },
            3 => .rgb2gray,
            4 => .rgba2gray,
            else => return error.Unsupported,
        };
        try visar.imgproc.cvtColor(exec, s, d, code);
        result = .{ .u8 = d };
    } else if (std.mem.eql(u8, cmd, "blur")) {
        const sigma = try std.fmt.parseFloat(f64, args[4]);
        const s = img.as(u8) orelse return error.Unsupported;
        var d = try visar.Image(u8).initLike(gpa, s);
        errdefer d.deinit(gpa);
        try visar.imgproc.gaussianBlur(exec, gpa, s, d, .{ .sigma_x = sigma });
        result = .{ .u8 = d };
    } else {
        try err.print("visar: unknown command '{s}'\n\n{s}", .{ cmd, usage });
        std.process.exit(2);
    }
    try writeOut(gpa, io, err, dst_path, result orelse img, quality);
}

fn writeOut(gpa: std.mem.Allocator, io: Io, err: *Io.Writer, path: []const u8, img: visar.AnyImage, quality: u8) !void {
    visar.codecs.write(gpa, io, path, img, .{ .quality = quality }) catch |e| {
        try err.print("visar: cannot write {s}: {s}\n", .{ path, @errorName(e) });
        std.process.exit(1);
    };
}
