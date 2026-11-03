//! Live multi-object tracking demo: renders the synthetic scene, tracks it
//! with VISAR, writes an annotated GIF and prints a Markdown summary (which
//! CI posts to the job page). The work is done in track_lib.zig, which
//! `visar demo track` runs too.
//!
//!   zig build demo-track -Doptimize=ReleaseFast -- --out track.gif

const std = @import("std");
const visar = @import("visar");
const track = @import("track_lib");
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var opts: track.Options = .{};
    var out_path: []const u8 = "track.gif";
    var threads: ?usize = null;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const next = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, a, "--frames")) {
            opts.frames = try std.fmt.parseInt(u32, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--out")) {
            out_path = next;
            i += 1;
        } else if (std.mem.eql(u8, a, "--threads")) {
            threads = try std.fmt.parseInt(usize, next, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--size")) {
            var it = std.mem.splitScalar(u8, next, 'x');
            opts.width = try std.fmt.parseInt(u32, it.next() orelse "", 10);
            opts.height = try std.fmt.parseInt(u32, it.next() orelse "", 10);
            i += 1;
        }
    }

    var pool = visar.Pool.init(gpa, io, threads);
    defer pool.deinit();
    var result = try track.run(gpa, io, visar.Exec.on(&pool), opts, null);
    defer result.deinit(gpa);
    try Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = result.gif });

    var ob: [4096]u8 = undefined;
    var ow = Io.File.Writer.initStreaming(.stdout(), io, &ob);
    try track.writeMarkdown(&ow.interface, result.stats, out_path);
    try ow.interface.flush();
}
