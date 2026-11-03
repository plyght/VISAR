//! Every `visar` command and option, in one table: the parser checks
//! arguments against it and the help is rendered from it, so the two cannot
//! drift apart.

const std = @import("std");
const args = @import("args.zig");
const Opt = args.Opt;
const Ctx = @import("ctx.zig").Ctx;
const ops = @import("ops.zig");
const cmd_info = @import("cmd_info.zig");
const cmd_image = @import("cmd_image.zig");
const cmd_bench = @import("cmd_bench.zig");
const cmd_demo = @import("cmd_demo.zig");
const cmd_track = @import("cmd_track.zig");

pub const tagline = "computer vision in pure Zig";
pub const about = "Inspect, convert and process images with VISAR, a dependency-free computer-vision library.";
pub const repo_url = "https://github.com/plyght/VISAR";

pub const RunFn = *const fn (ctx: *Ctx, p: *args.Parsed) anyerror!void;

pub const Group = enum {
    look,
    transform,
    filter,
    tools,

    pub fn title(g: Group) []const u8 {
        return switch (g) {
            .look => "Look",
            .transform => "Convert and transform",
            .filter => "Filter",
            .tools => "Tools",
        };
    }
};

pub const Example = struct { cmd: []const u8, note: []const u8 };

pub const Command = struct {
    name: []const u8,
    group: Group,
    summary: []const u8,
    /// Usage lines, each following "visar ".
    usage: []const []const u8,
    /// A paragraph or two for `visar help <name>`.
    about: []const u8 = "",
    examples: []const Example,
    opts: []const Opt = &.{},
    /// Takes the output options (-o, -f, --format, --quality, --compression).
    writes: bool = false,
    run: RunFn,
};

pub const global_opts = [_]Opt{
    .{ .long = "threads", .value = "N", .help = "Worker threads, or 'auto' for one per CPU (env VISAR_THREADS)" },
    .{ .long = "json", .help = "Print results as JSON on stdout" },
    .{ .long = "no-color", .help = "Plain output without colour (also NO_COLOR=1)" },
    .{ .long = "quiet", .short = 'q', .help = "Print only results and errors" },
    .{ .long = "verbose", .short = 'v', .help = "Explain each step, with timings" },
    .{ .long = "help", .short = 'h', .help = "Show help (for a command: visar <command> --help)" },
    .{ .long = "version", .help = "Print the version and build target" },
};

pub const output_opts = [_]Opt{
    .{ .long = "output", .short = 'o', .value = "FILE", .help = "Output file, instead of the second argument ('-' = stdout)" },
    .{ .long = "force", .short = 'f', .help = "Replace the output file if it exists" },
    .{ .long = "format", .value = "FMT", .help = "png, jpeg, qoi, bmp, pnm or gif (default: from the extension)" },
    .{ .long = "quality", .value = "N", .help = "JPEG quality, 1-100 (default 95)" },
    .{ .long = "compression", .value = "N", .help = "PNG compression level, 0-9 (default 1, fast)" },
};

fn op(comptime k: ops.Kind) RunFn {
    return struct {
        fn run(ctx: *Ctx, p: *args.Parsed) anyerror!void {
            return cmd_image.runOp(ctx, p, k);
        }
    }.run;
}

pub const list = [_]Command{
    .{
        .name = "info",
        .group = .look,
        .summary = "Show the version, CPU features, threads and formats",
        .usage = &.{"info [--json]"},
        .about = "Reports how this binary was built and what it can do on this machine: the target and the SIMD features VISAR's kernels were compiled for, the thread count it will use, and the image formats it reads and writes.",
        .examples = &.{
            .{ .cmd = "visar info", .note = "everything at a glance" },
            .{ .cmd = "visar info --json | jq .threads", .note = "one field, for scripts" },
        },
        .run = cmd_info.run,
    },
    .{
        .name = "inspect",
        .group = .look,
        .summary = "Show the format, size, channels and depth of images",
        .usage = &.{"inspect <image>... [--json]"},
        .about = "Decodes each image and reports what VISAR sees in it. With several images the report is a table; '-' reads one from stdin. Images that cannot be read are reported and skipped, and the exit code is 1 if any failed.",
        .examples = &.{
            .{ .cmd = "visar inspect photo.jpg", .note = "one image in detail" },
            .{ .cmd = "visar inspect shots/*.png", .note = "a table of many" },
            .{ .cmd = "curl -s https://example.com/a.png | visar inspect -", .note = "from stdin" },
            .{ .cmd = "visar inspect *.png --json", .note = "a JSON array, one object per image" },
        },
        .run = cmd_image.runInspect,
    },
    .{
        .name = "convert",
        .group = .transform,
        .summary = "Re-encode an image in another format",
        .usage = &.{ "convert <in> <out> [--quality N] [--compression N]", "convert <in> - --format FMT" },
        .about = "The output format follows the output's extension, or --format. When the format cannot hold the image as it is (16-bit samples in a JPEG, gray in a QOI), it is converted to the closest thing the format holds, with a warning.",
        .examples = &.{
            .{ .cmd = "visar convert scan.ppm scan.png", .note = "PPM to PNG" },
            .{ .cmd = "visar convert photo.png photo.jpg --quality 85", .note = "a smaller JPEG" },
            .{ .cmd = "visar convert big.png - --format qoi | ssh host 'cat > big.qoi'", .note = "stream to stdout" },
        },
        .writes = true,
        .run = cmd_image.runConvert,
    },
    .{
        .name = "resize",
        .group = .transform,
        .summary = "Scale an image, keeping its aspect ratio unless told otherwise",
        .usage = &.{ "resize <in> <out> <WxH | Wx | xH | N%>", "resize <in> <out> (--width W | --height H | --scale S) [--interp MODE]" },
        .about = "With one side given, the other follows the aspect ratio. The default interpolation is area averaging to shrink (sharp, no aliasing) and bicubic to enlarge.",
        .examples = &.{
            .{ .cmd = "visar resize photo.jpg thumb.jpg 320x", .note = "320 wide, height to match" },
            .{ .cmd = "visar resize photo.jpg half.png 50%", .note = "half size" },
            .{ .cmd = "visar resize icon.png big.png --scale 4 --interp nearest", .note = "crisp pixel art" },
            .{ .cmd = "visar resize in.png out.png 640x480", .note = "exactly 640x480 (may stretch)" },
        },
        .opts = &ops.resize_opts,
        .writes = true,
        .run = op(.resize),
    },
    .{
        .name = "rotate",
        .group = .transform,
        .summary = "Rotate by 90, 180 or 270 degrees clockwise",
        .usage = &.{"rotate <in> <out> <90 | 180 | 270 | -90>"},
        .examples = &.{
            .{ .cmd = "visar rotate scan.png upright.png 90", .note = "a quarter turn clockwise" },
            .{ .cmd = "visar rotate in.png out.png -90", .note = "a quarter turn counter-clockwise" },
        },
        .writes = true,
        .run = op(.rotate),
    },
    .{
        .name = "flip",
        .group = .transform,
        .summary = "Mirror an image horizontally, vertically or both",
        .usage = &.{"flip <in> <out> (--horizontal | --vertical | --both)"},
        .examples = &.{
            .{ .cmd = "visar flip selfie.jpg mirrored.jpg --horizontal", .note = "left to right" },
            .{ .cmd = "visar flip in.png out.png --vertical", .note = "upside down" },
        },
        .opts = &ops.flip_opts,
        .writes = true,
        .run = op(.flip),
    },
    .{
        .name = "gray",
        .group = .filter,
        .summary = "Convert to grayscale",
        .usage = &.{"gray <in> <out>"},
        .about = "Uses the BT.601 luma weights, like OpenCV's COLOR_RGB2GRAY. Alpha is dropped.",
        .examples = &.{
            .{ .cmd = "visar gray photo.jpg gray.png", .note = "" },
        },
        .writes = true,
        .run = op(.gray),
    },
    .{
        .name = "blur",
        .group = .filter,
        .summary = "Gaussian blur",
        .usage = &.{"blur <in> <out> [--sigma S] [--ksize K]"},
        .about = "Without options the sigma is 2 pixels. 8-bit results are bit-exact with OpenCV's GaussianBlur.",
        .examples = &.{
            .{ .cmd = "visar blur photo.jpg soft.jpg --sigma 3", .note = "a stronger blur" },
            .{ .cmd = "visar blur noisy.png clean.png --ksize 5", .note = "a 5x5 kernel" },
        },
        .opts = &ops.blur_opts,
        .writes = true,
        .run = op(.blur),
    },
    .{
        .name = "threshold",
        .group = .filter,
        .summary = "Make a black-and-white mask",
        .usage = &.{"threshold <in> <out> [VALUE | --otsu | --triangle] [--invert] [--max V]"},
        .about = "Pixels brighter than VALUE become --max (255), the rest 0. Without VALUE, Otsu's method picks it from the histogram, and the status line says which it chose. Colour images are converted to grayscale first.",
        .examples = &.{
            .{ .cmd = "visar threshold scan.png mask.png", .note = "automatic (Otsu)" },
            .{ .cmd = "visar threshold scan.png mask.png 128 --invert", .note = "fixed, ink becomes white" },
        },
        .opts = &ops.threshold_opts,
        .writes = true,
        .run = op(.threshold),
    },
    .{
        .name = "edges",
        .group = .filter,
        .summary = "Edge strength (Sobel gradient magnitude)",
        .usage = &.{"edges <in> <out> [--ksize K]"},
        .about = "Computes sqrt(gx^2 + gy^2) from horizontal and vertical Sobel derivatives of the grayscale image, saturated to 8 bits.",
        .examples = &.{
            .{ .cmd = "visar edges photo.jpg edges.png", .note = "" },
            .{ .cmd = "visar pipeline photo.jpg e.png -- blur --sigma 1.5 , edges", .note = "smooth first, fewer noisy edges" },
        },
        .opts = &ops.edges_opts,
        .writes = true,
        .run = op(.edges),
    },
    .{
        .name = "equalize",
        .group = .filter,
        .summary = "Spread out the brightness histogram (more contrast)",
        .usage = &.{"equalize <in> <out>"},
        .about = "Grayscale images are equalised directly. Colour images have their luma (Y of YCrCb) equalised, so hues and alpha are kept.",
        .examples = &.{
            .{ .cmd = "visar equalize dark.jpg better.jpg", .note = "" },
        },
        .writes = true,
        .run = op(.equalize),
    },
    .{
        .name = "morph",
        .group = .filter,
        .summary = "Morphology: erode, dilate, open, close, gradient, tophat, blackhat",
        .usage = &.{"morph <in> <out> <OPERATION> [--shape SHAPE] [--size N] [--iterations N]"},
        .about = "Applies a morphological operation with a square, cross or elliptical element. Most useful on masks from 'visar threshold': open removes specks, close fills holes.",
        .examples = &.{
            .{ .cmd = "visar morph mask.png clean.png open --size 5", .note = "remove small specks" },
            .{ .cmd = "visar morph mask.png solid.png close --shape ellipse --size 9", .note = "fill holes" },
            .{ .cmd = "visar morph text.png bold.png dilate --iterations 2", .note = "thicken strokes" },
        },
        .opts = &ops.morph_opts,
        .writes = true,
        .run = op(.morph),
    },
    .{
        .name = "pipeline",
        .group = .tools,
        .summary = "Run several operations in one pass",
        .usage = &.{"pipeline <in> <out> -- <op> [args] , <op> [args] ..."},
        .about = "Decodes once, runs each step on the result of the last, and encodes once. Steps are separated by commas and take the same arguments as the commands of the same name: gray, blur, threshold, flip, rotate, edges, equalize, morph and resize.",
        .examples = &.{
            .{ .cmd = "visar pipeline scan.jpg mask.png -- gray , blur --sigma 1.5 , threshold --otsu , morph open", .note = "a clean mask" },
            .{ .cmd = "visar pipeline in.jpg thumb.jpg -- resize 25% , equalize", .note = "small and punchy" },
        },
        .writes = true,
        .run = cmd_image.runPipeline,
    },
    .{
        .name = "bench",
        .group = .tools,
        .summary = "Time VISAR's kernels and codecs on this machine",
        .usage = &.{"bench [--size WxH] [--filter TEXT] [--time S] [--list] [--json]"},
        .about = "Runs each kernel on a synthetic image until --time seconds have passed and reports the median time and throughput in megapixels per second. Build with -Doptimize=ReleaseFast for representative numbers.",
        .examples = &.{
            .{ .cmd = "visar bench", .note = "every kernel at 1920x1080" },
            .{ .cmd = "visar bench --filter resize,blur --size 3840x2160", .note = "some kernels, at 4K" },
            .{ .cmd = "visar bench --threads 1 --json > single.json", .note = "single-threaded, for comparison" },
        },
        .opts = &cmd_bench.opts,
        .run = cmd_bench.run,
    },
    .{
        .name = "demo",
        .group = .tools,
        .summary = "Run a demo (track: multi-object tracking, rendered to a GIF)",
        .usage = &.{"demo track [--out FILE] [--frames N] [--size WxH] [--seed N]"},
        .about = "track renders a synthetic street scene, tracks the moving objects with a VISAR pipeline (gray, blur, background subtraction, morphology, connected components, tracker), writes an annotated GIF and scores itself against the scene's ground truth.",
        .examples = &.{
            .{ .cmd = "visar demo track", .note = "writes track.gif" },
            .{ .cmd = "visar demo track --frames 300 --size 1280x720 --out big.gif", .note = "longer, larger" },
        },
        .opts = &cmd_demo.opts,
        .writes = true,
        .run = cmd_demo.run,
    },
    .{
        .name = "track",
        .group = .tools,
        .summary = "Live object tracking from the camera, in a window with IDs, trails and FPS",
        .usage = &.{ "track [--camera N] [--size WxH] [--fps N] [--detect-width N] [--record FILE]", "track --source synthetic [--headless] [--frames N] [--json]", "track --list-cameras [--json]" },
        .about = "Opens the camera and a window showing it live, with every moving object boxed, numbered and trailed. The pipeline is the one of 'visar demo track' (gray, blur, background subtraction, morphology, connected components, tracker), run on a worker thread at --detect-width while the camera captures and the window shows the previous result. The HUD shows the camera's frame rate, VISAR's time per frame (and the rate that allows) and the display rate. The camera format with the highest frame rate at or below --size is chosen; --list-cameras shows what each camera offers. On first use macOS asks whether your terminal may use the camera. --source synthetic tracks the demo's generated scene instead, on any platform; --headless prints stats instead of opening a window.",
        .examples = &.{
            .{ .cmd = "visar track", .note = "the default camera, in a window" },
            .{ .cmd = "visar track --list-cameras", .note = "sizes and frame rates of every camera (can it do 120 fps?)" },
            .{ .cmd = "visar track --size 640x480 --fps 60", .note = "a smaller, faster format" },
            .{ .cmd = "visar track --source synthetic --headless --frames 600 --json", .note = "measure VISAR's throughput anywhere" },
            .{ .cmd = "visar track --frames 300 --record clip.gif", .note = "save what the window showed" },
        },
        .opts = &cmd_track.opts,
        .writes = true,
        .run = cmd_track.run,
    },
};

pub const command_names = blk: {
    var n: [list.len + 1][]const u8 = undefined;
    for (list, 0..) |c, i| n[i] = c.name;
    n[list.len] = "help";
    break :blk n;
};

pub fn find(name: []const u8) ?*const Command {
    for (&list) |*c| if (std.mem.eql(u8, c.name, name)) return c;
    return null;
}

test "every command has a unique name, examples and a usage line" {
    for (list, 0..) |c, i| {
        try std.testing.expect(c.examples.len > 0);
        try std.testing.expect(c.usage.len > 0);
        for (list[i + 1 ..]) |o| try std.testing.expect(!std.mem.eql(u8, c.name, o.name));
        // A command's own options never collide with the global ones.
        for (c.opts) |o| {
            try std.testing.expect(args.findLong(&.{ &global_opts, &output_opts }, o.long) == null);
            if (o.short != 0) try std.testing.expect(args.findShort(&.{ &global_opts, &output_opts }, o.short) == null);
        }
    }
}
