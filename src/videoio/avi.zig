//! AVI (RIFF) files: Motion-JPEG and uncompressed video, read and written
//! in pure Zig.
//!
//! Reading takes the first video stream and builds a frame index from, in
//! order of preference, the OpenDML index (`indx` super index -> `ix##`
//! chunk indexes, files over 1 GB), the AVI 1.0 `idx1`, or a scan of the
//! `movi` lists (a file whose writer never finished). Every frame is then
//! one positional read, so seeking by frame number is exact and cheap.
//! Frame codecs:
//!
//!   MJPG (also mjpg, AVRN, dmb1, JPEG)   Motion-JPEG, through codecs/jpeg
//!                                        (tables-less frames get the
//!                                        standard Huffman tables)
//!   BI_RGB 24 / 32 bit, 8 bit + palette  uncompressed DIB, either row order
//!   I420 / IYUV / YV12                   planar 4:2:0, BT.601 limited range
//!   Y800 / GREY / Y8                     8-bit gray
//!
//! Writing produces what OpenCV's VideoWriter does with fourcc 'MJPG'
//! (a Motion-JPEG AVI that every player and cv2.VideoCapture opens), or
//! uncompressed top-down BGR (fourcc 0, as ffmpeg writes it). Files grow past the
//! AVI 1.0 limit as OpenDML: the first RIFF holds `idx1` for old readers,
//! every RIFF segment an `ix00` chunk index, and `indx` points at them.
//!
//! The file is untrusted: every chunk size is checked against its parent
//! and the file, index entries pointing outside the file are dropped, and
//! a truncated or corrupt file gives an error (or fewer frames), never a
//! crash.

const std = @import("std");
const core = @import("../core/core.zig");
const jpeg = @import("../codecs/jpeg.zig");
const y4m = @import("y4m.zig");
const Image = core.Image;
const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const Error = error{
    /// Not an AVI file, or a structure that makes no sense.
    InvalidData,
    /// An AVI whose video this reader cannot decode, or no video stream.
    Unsupported,
    /// The file ends inside something it needs.
    Truncated,
    /// The destination image is not the video's size.
    SizeMismatch,
    UnsupportedChannels,
    /// The frame is larger than the limits allow.
    TooLarge,
    ReadFailed,
    WriteFailed,
    OutOfMemory,
};

fn fcc(s: *const [4]u8) u32 {
    return std.mem.readInt(u32, s, .little);
}

/// Largest frame size accepted from a file (in either dimension).
pub const max_dimension = 16384;
/// Largest chunk read into memory (one frame), and the largest header list.
const max_chunk = 256 << 20;
const max_header = 16 << 20;

// ---------------------------------------------------------------------------
// Where the bytes come from
// ---------------------------------------------------------------------------

/// The bytes of an AVI file: in memory, or a file read positionally.
pub const Source = union(enum) {
    memory: []const u8,
    file: struct { io: Io, file: Io.File },

    pub fn size(s: Source) Error!u64 {
        return switch (s) {
            .memory => |m| m.len,
            .file => |f| f.file.length(f.io) catch error.ReadFailed,
        };
    }

    /// Reads up to buf.len bytes at `off`; fewer at the end of the data.
    pub fn readAt(s: Source, buf: []u8, off: u64) Error!usize {
        switch (s) {
            .memory => |m| {
                if (off >= m.len) return 0;
                const n = @min(buf.len, m.len - @as(usize, @intCast(off)));
                @memcpy(buf[0..n], m[@intCast(off)..][0..n]);
                return n;
            },
            .file => |f| return f.file.readPositionalAll(f.io, buf, off) catch error.ReadFailed,
        }
    }

    fn readExact(s: Source, buf: []u8, off: u64) Error!void {
        if (try s.readAt(buf, off) != buf.len) return error.Truncated;
    }
};

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

pub const Codec = enum {
    mjpeg,
    /// BI_RGB, 24 bits per pixel (BGR).
    bgr24,
    /// BI_RGB, 32 bits per pixel (BGRX).
    bgr32,
    /// BI_RGB, 8 bits per pixel through a palette.
    pal8,
    i420,
    yv12,
    gray,
};

pub const IndexKind = enum { odml, idx1, scan };

pub const Frame = struct {
    /// Where the frame's data starts (after its chunk header).
    offset: u64,
    size: u32,
};

pub const Reader = struct {
    gpa: Allocator,
    src: Source,
    file_size: u64,
    width: u32,
    height: u32,
    codec: Codec,
    /// The stream's handler / compression FourCC as in the file.
    fourcc: [4]u8,
    /// Uncompressed rows stored bottom row first (positive biHeight).
    bottom_up: bool = true,
    /// Frame rate as the stream header gives it (rate / scale).
    fps_num: u32 = 0,
    fps_den: u32 = 0,
    /// Index of the video stream among the file's streams.
    stream: u32 = 0,
    /// Every frame of the video stream, in order.
    index: []Frame,
    /// Which index the frames came from.
    index_kind: IndexKind = .idx1,
    /// The next frame `next` returns.
    pos: u64 = 0,
    /// One frame's bytes.
    buf: std.ArrayList(u8) = .empty,
    palette: [256][3]u8 = undefined,

    /// Parses the headers and the index. `src` must outlive the reader.
    pub fn init(gpa: Allocator, src: Source) Error!Reader {
        var p: Parser = .{ .gpa = gpa, .src = src, .file_size = try src.size() };
        defer p.deinit();
        try p.parse();
        const v = p.video orelse return error.Unsupported;
        var r: Reader = .{
            .gpa = gpa,
            .src = src,
            .file_size = p.file_size,
            .width = v.width,
            .height = v.height,
            .codec = v.codec,
            .fourcc = v.fourcc,
            .bottom_up = v.bottom_up,
            .fps_num = v.rate,
            .fps_den = v.scale,
            .stream = v.number,
            .index = &.{},
            .palette = v.palette,
        };
        r.index, r.index_kind = try p.buildIndex(v);
        return r;
    }

    pub fn deinit(r: *Reader) void {
        r.gpa.free(r.index);
        r.buf.deinit(r.gpa);
        r.* = undefined;
    }

    pub fn frameCount(r: *const Reader) u64 {
        return r.index.len;
    }

    /// Frames per second, 0 when the header gives none.
    pub fn fps(r: *const Reader) f64 {
        if (r.fps_num == 0 or r.fps_den == 0) return 0;
        return @as(f64, @floatFromInt(r.fps_num)) / @as(f64, @floatFromInt(r.fps_den));
    }

    /// The next `next` returns frame `n` (n == frameCount: the end).
    pub fn seek(r: *Reader, n: u64) Error!void {
        if (n > r.index.len) return error.InvalidData;
        r.pos = n;
    }

    /// Reads the next frame into `dst` (the video's size; BGR, BGRA or
    /// gray). Returns false at the end of the video.
    pub fn next(r: *Reader, dst: Image(u8)) Error!bool {
        if (r.pos >= r.index.len) return false;
        try r.decodeFrame(@intCast(r.pos), dst);
        r.pos += 1;
        return true;
    }

    /// The raw bytes of frame `i` (a JPEG for Motion-JPEG), valid until the
    /// next read. A dropped (empty) frame gives the nearest earlier frame.
    pub fn frameBytes(r: *Reader, i: usize) Error![]const u8 {
        var k = i;
        while (r.index[k].size == 0) {
            if (k == 0) return error.InvalidData;
            k -= 1;
        }
        const f = r.index[k];
        try r.buf.resize(r.gpa, f.size);
        try r.src.readExact(r.buf.items, f.offset);
        return r.buf.items;
    }

    fn decodeFrame(r: *Reader, i: usize, dst: Image(u8)) Error!void {
        if (dst.width != r.width or dst.height != r.height) return error.SizeMismatch;
        if (dst.channels != 1 and dst.channels != 3 and dst.channels != 4) return error.UnsupportedChannels;
        const data = try r.frameBytes(i);
        switch (r.codec) {
            .mjpeg => jpeg.decodeInto(r.gpa, data, dst, .bgr, @as(u64, max_dimension) * max_dimension) catch |e| return switch (e) {
                error.OutOfMemory => error.OutOfMemory,
                error.SizeMismatch => error.InvalidData,
                error.Unsupported => error.Unsupported,
                error.Truncated => error.Truncated,
                error.TooLarge => error.TooLarge,
                else => error.InvalidData,
            },
            .bgr24, .bgr32, .pal8, .gray => {
                const bpp: usize = switch (r.codec) {
                    .bgr24 => 3,
                    .bgr32 => 4,
                    else => 1,
                };
                // DIB rows are padded to 4 bytes; Y800 rows are not.
                const raw_row = @as(usize, r.width) * bpp;
                const stride = if (r.codec == .gray) raw_row else (raw_row + 3) & ~@as(usize, 3);
                // The last row needs no padding.
                if (data.len < stride * (r.height - 1) + raw_row) return error.Truncated;
                for (0..r.height) |y| {
                    const sy = if (r.bottom_up and r.codec != .gray) r.height - 1 - y else y;
                    const s = data[sy * stride ..][0..raw_row];
                    const o = dst.row(y);
                    const ch: usize = dst.channels;
                    for (0..r.width) |x| {
                        const px: [3]u8 = switch (r.codec) {
                            .bgr24, .bgr32 => s[x * bpp ..][0..3].*,
                            .pal8 => r.palette[s[x]],
                            else => .{ s[x], s[x], s[x] },
                        };
                        if (ch == 1) {
                            o[x] = @intCast((@as(u32, px[0]) * 1868 + @as(u32, px[1]) * 9617 + @as(u32, px[2]) * 4899 + 8192) >> 14);
                        } else {
                            o[x * ch ..][0..3].* = px;
                            if (ch == 4) o[x * 4 + 3] = 255;
                        }
                    }
                }
            },
            .i420, .yv12 => {
                const n = y4m.i420Bytes(r.width, r.height);
                if (data.len < n) return error.Truncated;
                const ys = @as(usize, r.width) * r.height;
                const cs = (n - ys) / 2;
                const a = data[ys..][0..cs];
                const b = data[ys + cs ..][0..cs];
                const u, const v = if (r.codec == .i420) .{ a, b } else .{ b, a };
                if (dst.channels == 1) {
                    for (0..r.height) |y| @memcpy(dst.row(y), data[y * r.width ..][0..r.width]);
                } else y4m.yuvToBgr(.c420, false, data[0..ys], u, v, dst);
            },
        }
    }
};

const Video = struct {
    number: u32,
    width: u32,
    height: u32,
    codec: Codec,
    fourcc: [4]u8,
    bottom_up: bool,
    rate: u32,
    scale: u32,
    /// The `indx` chunk's data, when the stream has one.
    indx: ?[]const u8 = null,
    palette: [256][3]u8,
};

const Parser = struct {
    gpa: Allocator,
    src: Source,
    file_size: u64,
    /// The header list (owned).
    hdrl: []u8 = &.{},
    video: ?Video = null,
    /// `movi` lists: [start of the list's data (the 'movi' FourCC), end).
    movi: std.ArrayList([2]u64) = .empty,
    /// The idx1 chunk's data range.
    idx1: ?[2]u64 = null,

    fn deinit(p: *Parser) void {
        p.gpa.free(p.hdrl);
        p.movi.deinit(p.gpa);
    }

    fn read32(p: *Parser, off: u64) Error!u32 {
        var b: [4]u8 = undefined;
        try p.src.readExact(&b, off);
        return std.mem.readInt(u32, &b, .little);
    }

    fn parse(p: *Parser) Error!void {
        var head: [12]u8 = undefined;
        try p.src.readExact(&head, 0);
        if (!std.mem.eql(u8, head[0..4], "RIFF") or !std.mem.eql(u8, head[8..12], "AVI ")) return error.InvalidData;
        // Top level: RIFF 'AVI ', then RIFF 'AVIX' extensions (OpenDML).
        var off: u64 = 0;
        var first = true;
        while (off + 12 <= p.file_size) {
            try p.src.readExact(&head, off);
            if (!std.mem.eql(u8, head[0..4], "RIFF")) break;
            const size = std.mem.readInt(u32, head[4..8], .little);
            // A size running past the file (a truncated file) is clamped.
            const end = @min(off + 8 + size, p.file_size);
            if (first) {
                if (!std.mem.eql(u8, head[8..12], "AVI ")) return error.InvalidData;
            } else if (!std.mem.eql(u8, head[8..12], "AVIX")) break;
            try p.walk(off + 12, end, first);
            first = false;
            off = end + (end & 1);
            // A size of 0 (writer that never finished) cannot be trusted.
            if (size == 0) break;
        }
        if (p.video == null) return if (p.hdrl.len == 0) error.InvalidData else error.Unsupported;
    }

    /// The chunks of a RIFF between `start` and `end`.
    fn walk(p: *Parser, start: u64, end: u64, first: bool) Error!void {
        var off = start;
        while (off + 8 <= end) {
            var h: [12]u8 = undefined;
            const got = try p.src.readAt(&h, off);
            if (got < 8) return;
            const id = h[0..4];
            const size: u64 = std.mem.readInt(u32, h[4..8], .little);
            const data = off + 8;
            const data_end = @min(data + size, end);
            if (std.mem.eql(u8, id, "LIST") and got >= 12) {
                const kind = h[8..12];
                if (std.mem.eql(u8, kind, "hdrl") and first and p.hdrl.len == 0) {
                    const n = data_end - data;
                    if (n > max_header) return error.TooLarge;
                    if (n < 4) return error.InvalidData;
                    p.hdrl = try p.gpa.alloc(u8, @intCast(n));
                    try p.src.readExact(p.hdrl, data);
                    try p.parseHdrl(p.hdrl[4..]);
                } else if (std.mem.eql(u8, kind, "movi")) {
                    try p.movi.append(p.gpa, .{ data, data_end });
                }
            } else if (std.mem.eql(u8, id, "idx1") and first) {
                p.idx1 = .{ data, data_end };
            }
            // A movi list with size 0 or past the end: the rest is movi.
            if (size == 0 and std.mem.eql(u8, id, "LIST")) {
                if (p.movi.items.len > 0 and p.movi.items[p.movi.items.len - 1][0] == data) p.movi.items[p.movi.items.len - 1][1] = end;
                return;
            }
            off = data + size + (size & 1);
        }
    }

    /// The streams in the header list (after its 'hdrl' FourCC).
    fn parseHdrl(p: *Parser, list: []const u8) Error!void {
        var it: ChunkIter = .{ .data = list };
        var number: u32 = 0;
        while (it.next()) |c| {
            if (!std.mem.eql(u8, &c.id, "LIST") or c.data.len < 4 or !std.mem.eql(u8, c.data[0..4], "strl")) continue;
            defer number += 1;
            if (p.video != null) continue;
            var strh: ?[]const u8 = null;
            var strf: ?[]const u8 = null;
            var indx: ?[]const u8 = null;
            var si: ChunkIter = .{ .data = c.data[4..] };
            while (si.next()) |s| {
                if (std.mem.eql(u8, &s.id, "strh")) strh = s.data;
                if (std.mem.eql(u8, &s.id, "strf")) strf = s.data;
                if (std.mem.eql(u8, &s.id, "indx")) indx = s.data;
            }
            const h = strh orelse continue;
            if (h.len < 36 or !std.mem.eql(u8, h[0..4], "vids")) continue;
            const f = strf orelse return error.InvalidData;
            if (f.len < 40) return error.InvalidData;
            const w = std.mem.readInt(i32, f[4..8], .little);
            const hh = std.mem.readInt(i32, f[8..12], .little);
            const bits = std.mem.readInt(u16, f[14..16], .little);
            const comp = f[16..20].*;
            if (w <= 0 or hh == 0 or w > max_dimension or @abs(hh) > max_dimension) return error.InvalidData;
            var v: Video = .{
                .number = number,
                .width = @intCast(w),
                .height = @intCast(@abs(hh)),
                .codec = undefined,
                .fourcc = if (std.mem.readInt(u32, &comp, .little) == 0) h[4..8].* else comp,
                .bottom_up = hh > 0,
                .scale = std.mem.readInt(u32, h[20..24], .little),
                .rate = std.mem.readInt(u32, h[24..28], .little),
                .indx = indx,
                .palette = undefined,
            };
            v.codec = codecOf(comp, bits) orelse return error.Unsupported;
            if (v.codec == .pal8) {
                const used = std.mem.readInt(u32, f[32..36], .little);
                const hsize = std.mem.readInt(u32, f[0..4], .little);
                const n: usize = if (used == 0 or used > 256) 256 else used;
                const at: usize = @max(40, @min(hsize, f.len));
                for (&v.palette, 0..) |*e, i| {
                    e.* = if (i < n and at + 4 * i + 3 <= f.len) f[at + 4 * i ..][0..3].* else .{ 0, 0, 0 };
                }
            }
            p.video = v;
        }
    }

    fn buildIndex(p: *Parser, v: Video) Error!struct { []Frame, IndexKind } {
        var frames: std.ArrayList(Frame) = .empty;
        errdefer frames.deinit(p.gpa);
        var tag: [2]u8 = undefined;
        if (v.number > 99) return error.Unsupported;
        tag = .{ '0' + @as(u8, @intCast(v.number / 10)), '0' + @as(u8, @intCast(v.number % 10)) };
        if (v.indx) |indx| {
            if (p.odmlIndex(indx, &frames) catch |e| switch (e) {
                error.OutOfMemory => return e,
                else => false,
            }) return .{ try frames.toOwnedSlice(p.gpa), .odml };
            frames.clearRetainingCapacity();
        }
        if (p.idx1) |r| {
            if (try p.idx1Index(r, tag, &frames)) return .{ try frames.toOwnedSlice(p.gpa), .idx1 };
            frames.clearRetainingCapacity();
        }
        for (p.movi.items) |m| try p.scan(m[0] + 4, m[1], tag, &frames, 0);
        return .{ try frames.toOwnedSlice(p.gpa), .scan };
    }

    fn isVideoChunk(id: []const u8, tag: [2]u8) bool {
        return id[0] == tag[0] and id[1] == tag[1] and (id[2] == 'd' or id[2] == 'D') and (id[3] == 'c' or id[3] == 'b');
    }

    /// Adds a frame if it lies inside the file (the rest of a truncated
    /// file's index is dropped: returns false then).
    fn add(p: *Parser, frames: *std.ArrayList(Frame), offset: u64, size: u32) Error!bool {
        if (size > max_chunk or offset > p.file_size or size > p.file_size - offset) return false;
        try frames.append(p.gpa, .{ .offset = offset, .size = size });
        return true;
    }

    /// OpenDML: the super index's entries point at ix## chunks.
    fn odmlIndex(p: *Parser, indx: []const u8, frames: *std.ArrayList(Frame)) Error!bool {
        if (indx.len < 24) return false;
        const longs = std.mem.readInt(u16, indx[0..2], .little);
        const kind = indx[3];
        const n = std.mem.readInt(u32, indx[4..8], .little);
        if (kind != 0 or longs != 4) return false; // AVI_INDEX_OF_INDEXES
        if (n == 0 or 24 + @as(u64, n) * 16 > indx.len) return false;
        for (0..n) |i| {
            const e = indx[24 + 16 * i ..][0..16];
            const off = std.mem.readInt(u64, e[0..8], .little);
            const size = std.mem.readInt(u32, e[8..12], .little);
            if (off > p.file_size or size < 24 or size > max_header) return false;
            var hdr: [32]u8 = undefined;
            if (try p.src.readAt(&hdr, off) < 32) return frames.items.len > 0;
            // 'ix##' chunk: header, then entries of (offset, size).
            const csize = std.mem.readInt(u32, hdr[4..8], .little);
            const elongs = std.mem.readInt(u16, hdr[8..10], .little);
            const etype = hdr[11];
            const count = std.mem.readInt(u32, hdr[12..16], .little);
            const base = std.mem.readInt(u64, hdr[20..28], .little);
            if (etype != 1 or elongs != 2 or csize < 24 or csize > max_header or @as(u64, count) * 8 > csize - 24) return false;
            const bytes = try p.gpa.alloc(u8, count * 8);
            defer p.gpa.free(bytes);
            const got = try p.src.readAt(bytes, off + 32);
            const whole = got / 8;
            for (0..whole) |k| {
                const rel = std.mem.readInt(u32, bytes[k * 8 ..][0..4], .little);
                // Bit 31 of the size marks a non-key frame.
                const sz = std.mem.readInt(u32, bytes[k * 8 + 4 ..][0..4], .little) & 0x7fffffff;
                if (!try p.add(frames, base + rel, sz)) return frames.items.len > 0;
            }
            if (whole < count) return frames.items.len > 0;
        }
        return frames.items.len > 0;
    }

    /// AVI 1.0 index: offsets from the 'movi' FourCC (or, in some files,
    /// from the start of the file) to each chunk's header.
    fn idx1Index(p: *Parser, r: [2]u64, tag: [2]u8, frames: *std.ArrayList(Frame)) Error!bool {
        if (p.movi.items.len == 0) return false;
        const len = r[1] - r[0];
        if (len > max_chunk) return false;
        const bytes = try p.gpa.alloc(u8, @intCast(len - len % 16));
        defer p.gpa.free(bytes);
        try p.src.readExact(bytes, r[0]);
        // Find the base from the first video entry: the chunk id must be
        // where the entry points.
        var base: ?u64 = null;
        var i: usize = 0;
        while (i + 16 <= bytes.len) : (i += 16) {
            const e = bytes[i..][0..16];
            if (!isVideoChunk(e[0..4], tag)) continue;
            const off: u64 = std.mem.readInt(u32, e[8..12], .little);
            for ([_]u64{ p.movi.items[0][0], 0 }) |b| {
                var id: [4]u8 = undefined;
                if (try p.src.readAt(&id, b + off) == 4 and std.mem.eql(u8, &id, e[0..4])) {
                    base = b;
                    break;
                }
            }
            break;
        }
        const b = base orelse return false;
        i = 0;
        while (i + 16 <= bytes.len) : (i += 16) {
            const e = bytes[i..][0..16];
            if (!isVideoChunk(e[0..4], tag)) continue;
            const off: u64 = std.mem.readInt(u32, e[8..12], .little);
            const size = std.mem.readInt(u32, e[12..16], .little);
            if (!try p.add(frames, b + off + 8, size)) break;
        }
        return frames.items.len > 0;
    }

    /// Walks a movi list (or a 'rec ' list inside one) for video chunks.
    fn scan(p: *Parser, start: u64, end: u64, tag: [2]u8, frames: *std.ArrayList(Frame), depth: u32) Error!void {
        var off = start;
        while (off + 8 <= end) {
            var h: [12]u8 = undefined;
            const got = try p.src.readAt(&h, off);
            if (got < 8) return;
            const size = std.mem.readInt(u32, h[4..8], .little);
            const data = off + 8;
            if (std.mem.eql(u8, h[0..4], "LIST")) {
                if (got >= 12 and depth < 4) try p.scan(data + 4, @min(data + size, end), tag, frames, depth + 1);
            } else if (isVideoChunk(h[0..4], tag)) {
                if (data + size > end or !try p.add(frames, data, size)) return;
            }
            off = data + size + (size & 1);
        }
    }
};

fn codecOf(comp: [4]u8, bits: u16) ?Codec {
    const c = std.mem.readInt(u32, &comp, .little);
    if (c == 0) return switch (bits) {
        24 => .bgr24,
        32 => .bgr32,
        8 => .pal8,
        else => null,
    };
    var up: [4]u8 = undefined;
    for (&up, comp) |*o, ch| o.* = std.ascii.toUpper(ch);
    const table = .{
        .{ "MJPG", Codec.mjpeg }, .{ "AVRN", Codec.mjpeg }, .{ "DMB1", Codec.mjpeg }, .{ "JPEG", Codec.mjpeg },
        .{ "I420", Codec.i420 },  .{ "IYUV", Codec.i420 },  .{ "YV12", Codec.yv12 },  .{ "Y800", Codec.gray },
        .{ "GREY", Codec.gray },  .{ "Y8  ", Codec.gray },
    };
    inline for (table) |t| if (std.mem.eql(u8, &up, t[0])) return t[1];
    // 'DIB ' and 'RGB ' name BI_RGB too.
    if (std.mem.eql(u8, &up, "DIB ") or std.mem.eql(u8, &up, "RGB ")) return codecOf(.{ 0, 0, 0, 0 }, bits);
    return null;
}

/// Iterates the chunks of an in-memory list body.
const ChunkIter = struct {
    data: []const u8,
    pos: usize = 0,

    const Chunk = struct { id: [4]u8, data: []const u8 };

    fn next(it: *ChunkIter) ?Chunk {
        if (it.pos + 8 > it.data.len) return null;
        const id = it.data[it.pos..][0..4].*;
        const size: usize = std.mem.readInt(u32, it.data[it.pos + 4 ..][0..4], .little);
        const start = it.pos + 8;
        const n = @min(size, it.data.len - start);
        it.pos = start + n + (n & 1);
        return .{ .id = id, .data = it.data[start..][0..n] };
    }
};

/// Whether `bytes` start like an AVI file.
pub fn detect(bytes: []const u8) bool {
    return bytes.len >= 12 and std.mem.eql(u8, bytes[0..4], "RIFF") and std.mem.eql(u8, bytes[8..12], "AVI ");
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Where a written file goes: a file (written positionally through a
/// buffer) or memory.
pub const Sink = union(enum) {
    memory: *std.ArrayList(u8),
    file: struct { io: Io, file: Io.File },
};

pub const WriteCodec = enum {
    /// Motion-JPEG ('MJPG').
    mjpeg,
    /// Uncompressed top-down BGR (negative biHeight) (BI_RGB, 24 bits).
    bgr24,
};

pub const WriterOptions = struct {
    codec: WriteCodec = .mjpeg,
    /// JPEG quality, 1-100 (OpenCV's default is 95).
    quality: u8 = 95,
    /// A new RIFF segment (OpenDML) starts once one passes this size.
    riff_limit: u64 = 1 << 30,
};

/// Super index entries reserved in the header (each covers one RIFF
/// segment of about 1 GB).
const super_entries = 256;
const avih_size = 56;
const strh_size = 56;
const strf_size = 40;
const indx_size = 24 + 16 * super_entries;
const dmlh_size = 248;

pub const Writer = struct {
    gpa: Allocator,
    sink: Sink,
    width: u32,
    height: u32,
    fps_num: u32,
    fps_den: u32,
    opts: WriterOptions,
    /// Bytes written (the file's size so far).
    pos: u64 = 0,
    /// Pending bytes, flushed to the file at `pos - pending.len`.
    pending: std.ArrayList(u8) = .empty,
    frames: std.ArrayList(Frame) = .empty,
    /// Per RIFF segment: the frames it holds, its ix00 chunk.
    segments: std.ArrayList(Segment) = .empty,
    /// Header offsets patched at the end.
    riff_start: u64 = 0,
    movi_start: u64 = 0,
    seg_first_frame: usize = 0,
    max_chunk_size: u32 = 0,
    first_riff_frames: u32 = 0,
    /// For the uncompressed codec.
    row_buf: []u8 = &.{},
    closed: bool = false,

    const Segment = struct { ix_offset: u64, ix_size: u32, frames: u32 };

    // Header layout offsets (from the file start).
    const off_avih = 12 + 12 + 8;
    const off_strl = off_avih + avih_size;
    const off_strh = off_strl + 12 + 8;
    const off_strf = off_strh + strh_size + 8;
    const off_indx = off_strf + strf_size + 8;
    const off_odml = off_indx + indx_size;
    const off_dmlh = off_odml + 12 + 8;
    const hdrl_end = off_dmlh + dmlh_size;

    pub fn init(gpa: Allocator, sink: Sink, width: u32, height: u32, fps_num: u32, fps_den: u32, opts: WriterOptions) Error!Writer {
        if (width == 0 or height == 0 or width > 65500 or height > 65500) return error.InvalidData;
        if (opts.codec == .mjpeg and (width > 65535 or height > 65535)) return error.TooLarge;
        var w: Writer = .{
            .gpa = gpa,
            .sink = sink,
            .width = width,
            .height = height,
            .fps_num = if (fps_num == 0) 30 else fps_num,
            .fps_den = if (fps_den == 0) 1 else fps_den,
            .opts = opts,
        };
        errdefer w.pending.deinit(gpa);
        if (opts.codec == .bgr24) w.row_buf = try gpa.alloc(u8, w.dibStride() * height);
        errdefer gpa.free(w.row_buf);
        try w.writeHeader();
        return w;
    }

    fn dibStride(w: *const Writer) usize {
        return (@as(usize, w.width) * 3 + 3) & ~@as(usize, 3);
    }

    /// Frees everything; a writer not closed leaves an unfinished file
    /// (still readable: its frames are found by scanning).
    pub fn deinit(w: *Writer) void {
        w.pending.deinit(w.gpa);
        w.frames.deinit(w.gpa);
        w.segments.deinit(w.gpa);
        w.gpa.free(w.row_buf);
        w.* = undefined;
    }

    pub fn frameCount(w: *const Writer) usize {
        return w.frames.items.len;
    }

    fn put(w: *Writer, bytes: []const u8) Error!void {
        try w.pending.appendSlice(w.gpa, bytes);
        w.pos += bytes.len;
        if (w.pending.items.len >= 1 << 20) try w.flush();
    }

    fn put32(w: *Writer, v: u32) Error!void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try w.put(&b);
    }

    fn flush(w: *Writer) Error!void {
        const n = w.pending.items.len;
        if (n == 0) return;
        const at = w.pos - n;
        switch (w.sink) {
            .memory => |m| {
                std.debug.assert(m.items.len == at);
                try m.appendSlice(w.gpa, w.pending.items);
            },
            .file => |f| f.file.writePositionalAll(f.io, w.pending.items, at) catch return error.WriteFailed,
        }
        w.pending.clearRetainingCapacity();
    }

    /// Overwrites bytes already written (after a flush).
    fn patch(w: *Writer, off: u64, bytes: []const u8) Error!void {
        try w.flush();
        switch (w.sink) {
            .memory => |m| @memcpy(m.items[@intCast(off)..][0..bytes.len], bytes),
            .file => |f| f.file.writePositionalAll(f.io, bytes, off) catch return error.WriteFailed,
        }
    }

    fn patch32(w: *Writer, off: u64, v: u32) Error!void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try w.patch(off, &b);
    }

    fn fourcc(w: *const Writer) [4]u8 {
        return switch (w.opts.codec) {
            .mjpeg => "MJPG".*,
            .bgr24 => .{ 0, 0, 0, 0 },
        };
    }

    /// The headers, with the sizes and counts patched at the end.
    fn writeHeader(w: *Writer) Error!void {
        var h: [hdrl_end + 12]u8 = @splat(0);
        const set32 = struct {
            fn f(b: []u8, off: usize, v: u32) void {
                std.mem.writeInt(u32, b[off..][0..4], v, .little);
            }
        }.f;
        @memcpy(h[0..4], "RIFF");
        @memcpy(h[8..12], "AVI ");
        @memcpy(h[12..16], "LIST");
        set32(&h, 16, hdrl_end - 20);
        @memcpy(h[20..24], "hdrl");
        // avih
        @memcpy(h[off_avih - 8 ..][0..4], "avih");
        set32(&h, off_avih - 4, avih_size);
        const us: u64 = @as(u64, 1_000_000) * w.fps_den / w.fps_num;
        set32(&h, off_avih, @intCast(@min(us, std.math.maxInt(u32))));
        set32(&h, off_avih + 12, 0x10); // AVIF_HASINDEX
        set32(&h, off_avih + 24, 1); // streams
        set32(&h, off_avih + 32, w.width);
        set32(&h, off_avih + 36, w.height);
        // strl
        @memcpy(h[off_strl..][0..4], "LIST");
        set32(&h, off_strl + 4, off_odml - off_strl - 8);
        @memcpy(h[off_strl + 8 ..][0..4], "strl");
        @memcpy(h[off_strh - 8 ..][0..4], "strh");
        set32(&h, off_strh - 4, strh_size);
        @memcpy(h[off_strh..][0..4], "vids");
        const cc = w.fourcc();
        @memcpy(h[off_strh + 4 ..][0..4], &cc);
        set32(&h, off_strh + 20, w.fps_den);
        set32(&h, off_strh + 24, w.fps_num);
        set32(&h, off_strh + 40, 0xffffffff); // quality: default
        std.mem.writeInt(u16, h[off_strh + 52 ..][0..2], @intCast(w.width), .little);
        std.mem.writeInt(u16, h[off_strh + 54 ..][0..2], @intCast(w.height), .little);
        @memcpy(h[off_strf - 8 ..][0..4], "strf");
        set32(&h, off_strf - 4, strf_size);
        set32(&h, off_strf, strf_size);
        set32(&h, off_strf + 4, w.width);
        // Uncompressed rows top-down (negative height), as ffmpeg writes them:
        // OpenCV's FFmpeg reader mishandles bottom-up frames.
        set32(&h, off_strf + 8, if (w.opts.codec == .bgr24) @bitCast(-@as(i32, @intCast(w.height))) else w.height);
        std.mem.writeInt(u16, h[off_strf + 12 ..][0..2], 1, .little);
        std.mem.writeInt(u16, h[off_strf + 14 ..][0..2], 24, .little);
        @memcpy(h[off_strf + 16 ..][0..4], &cc);
        set32(&h, off_strf + 20, @intCast(@min(@as(u64, w.width) * w.height * 3, std.math.maxInt(u32))));
        // indx (filled in at the end)
        @memcpy(h[off_indx - 8 ..][0..4], "indx");
        set32(&h, off_indx - 4, indx_size);
        std.mem.writeInt(u16, h[off_indx..][0..2], 4, .little);
        @memcpy(h[off_indx + 8 ..][0..4], "00dc");
        // odml / dmlh
        @memcpy(h[off_odml..][0..4], "LIST");
        set32(&h, off_odml + 4, hdrl_end - off_odml - 8);
        @memcpy(h[off_odml + 8 ..][0..4], "odml");
        @memcpy(h[off_dmlh - 8 ..][0..4], "dmlh");
        set32(&h, off_dmlh - 4, dmlh_size);
        // movi
        @memcpy(h[hdrl_end..][0..4], "LIST");
        @memcpy(h[hdrl_end + 8 ..][0..4], "movi");
        try w.put(&h);
        w.riff_start = 0;
        w.movi_start = hdrl_end + 8;
    }

    /// Appends a frame: BGR, BGRA or gray, the video's size.
    pub fn write(w: *Writer, frame: Image(u8)) Error!void {
        if (w.closed) return error.InvalidData;
        if (frame.width != w.width or frame.height != w.height) return error.SizeMismatch;
        if (frame.channels != 1 and frame.channels != 3 and frame.channels != 4) return error.UnsupportedChannels;
        switch (w.opts.codec) {
            .mjpeg => {
                const bytes = jpeg.encode(w.gpa, .{ .u8 = frame }, .{ .order = .bgr, .quality = std.math.clamp(w.opts.quality, 1, 100) }) catch |e| return switch (e) {
                    error.OutOfMemory => error.OutOfMemory,
                    else => error.Unsupported,
                };
                defer w.gpa.free(bytes);
                try w.writeChunk(bytes);
            },
            .bgr24 => {
                const stride = w.dibStride();
                const ch: usize = frame.channels;
                for (0..w.height) |y| {
                    const o = w.row_buf[y * stride ..][0..stride];
                    @memset(o[@as(usize, w.width) * 3 ..], 0);
                    const s = frame.row(y);
                    for (0..w.width) |x| {
                        o[x * 3 ..][0..3].* = if (ch == 1) .{ s[x], s[x], s[x] } else s[x * ch ..][0..3].*;
                    }
                }
                try w.writeChunk(w.row_buf);
            },
        }
    }

    /// A pre-encoded frame (a JPEG for Motion-JPEG), stored as it is.
    pub fn writeChunk(w: *Writer, data: []const u8) Error!void {
        if (data.len > max_chunk) return error.TooLarge;
        if (w.pos - w.riff_start + data.len + 16 > w.opts.riff_limit and w.frames.items.len > w.seg_first_frame) try w.newSegment();
        try w.put(if (w.opts.codec == .mjpeg) "00dc" else "00db");
        try w.put32(@intCast(data.len));
        try w.frames.append(w.gpa, .{ .offset = w.pos, .size = @intCast(data.len) });
        try w.put(data);
        if (data.len & 1 == 1) try w.put(&.{0});
        w.max_chunk_size = @max(w.max_chunk_size, @as(u32, @intCast(data.len)));
    }

    /// Ends the current RIFF segment (its ix00, idx1 for the first) and
    /// starts an 'AVIX' one.
    fn newSegment(w: *Writer) Error!void {
        if (w.segments.items.len + 1 >= super_entries) return error.TooLarge;
        try w.endSegment();
        w.riff_start = w.pos;
        try w.put("RIFF\x00\x00\x00\x00AVIXLIST\x00\x00\x00\x00movi");
        w.movi_start = w.pos - 4;
        w.seg_first_frame = w.frames.items.len;
    }

    fn endSegment(w: *Writer) Error!void {
        const seg = w.frames.items[w.seg_first_frame..];
        // ix00 (OpenDML standard index), inside the movi list.
        const ix_off = w.pos;
        const base = if (seg.len > 0) seg[0].offset - 8 else w.pos;
        var hdr: [32]u8 = @splat(0);
        @memcpy(hdr[0..4], "ix00");
        std.mem.writeInt(u32, hdr[4..8], @intCast(24 + 8 * seg.len), .little);
        std.mem.writeInt(u16, hdr[8..10], 2, .little);
        hdr[11] = 1; // AVI_INDEX_OF_CHUNKS
        std.mem.writeInt(u32, hdr[12..16], @intCast(seg.len), .little);
        @memcpy(hdr[16..20], if (w.opts.codec == .mjpeg) "00dc" else "00db");
        std.mem.writeInt(u64, hdr[20..28], base, .little);
        try w.put(&hdr);
        for (seg) |f| {
            try w.put32(@intCast(f.offset - base));
            try w.put32(f.size);
        }
        try w.segments.append(w.gpa, .{ .ix_offset = ix_off, .ix_size = @intCast(32 + 8 * seg.len), .frames = @intCast(seg.len) });
        // The movi list's size.
        try w.patch32(w.movi_start - 4, @intCast(w.pos - w.movi_start));
        if (w.segments.items.len == 1) {
            // idx1 for AVI 1.0 readers: offsets from the 'movi' FourCC.
            try w.put("idx1");
            try w.put32(@intCast(16 * seg.len));
            for (seg) |f| {
                try w.put(if (w.opts.codec == .mjpeg) "00dc" else "00db");
                try w.put32(0x10); // AVIIF_KEYFRAME
                try w.put32(@intCast(f.offset - 8 - w.movi_start));
                try w.put32(f.size);
            }
            w.first_riff_frames = @intCast(seg.len);
        }
        try w.patch32(w.riff_start + 4, @intCast(w.pos - w.riff_start - 8));
    }

    /// Writes the indexes and patches the headers. The writer must still
    /// be deinit'ed.
    pub fn close(w: *Writer) Error!void {
        if (w.closed) return;
        w.closed = true;
        try w.endSegment();
        try w.flush();
        const total: u32 = @intCast(w.frames.items.len);
        try w.patch32(off_avih + 16, w.first_riff_frames);
        try w.patch32(off_avih + 28, w.max_chunk_size + 8);
        try w.patch32(off_strh + 32, total);
        try w.patch32(off_strh + 36, w.max_chunk_size + 8);
        try w.patch32(off_dmlh, total);
        var max_bytes: u64 = @as(u64, w.max_chunk_size) * w.fps_num / w.fps_den;
        max_bytes = @min(max_bytes, std.math.maxInt(u32));
        try w.patch32(off_avih + 4, @intCast(max_bytes));
        // indx: the super index over the segments' ix00 chunks.
        const n = w.segments.items.len;
        try w.patch32(off_indx + 4, @intCast(n));
        var e: [16]u8 = undefined;
        for (w.segments.items, 0..) |s, i| {
            std.mem.writeInt(u64, e[0..8], s.ix_offset, .little);
            std.mem.writeInt(u32, e[8..12], s.ix_size, .little);
            std.mem.writeInt(u32, e[12..16], s.frames, .little);
            try w.patch(off_indx + 24 + 16 * i, &e);
        }
    }
};

/// A frame rate as a fraction: integers exactly, NTSC-style rates as
/// N*1000/1001, anything else to a thousandth.
pub fn fpsFraction(fps: f64) [2]u32 {
    if (!(fps > 0) or fps > 1e6) return .{ 30, 1 };
    const r = @round(fps);
    if (@abs(fps - r) < 1e-6) return .{ @intFromFloat(r), 1 };
    const ntsc = @round(fps * 1.001);
    if (@abs(fps - ntsc / 1.001) < 1e-4) return .{ @intFromFloat(ntsc * 1000), 1001 };
    var num: u32 = @intFromFloat(@round(fps * 1000));
    var den: u32 = 1000;
    const g = std.math.gcd(num, den);
    num /= g;
    den /= g;
    return .{ num, den };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

fn testFrame(gpa: Allocator, w: u32, h: u32, ch: u16, t: usize) !Image(u8) {
    var img = try Image(u8).init(gpa, w, h, ch);
    for (0..h) |y| {
        const r = img.row(y);
        for (0..w) |x| {
            const v: [4]u8 = .{
                @intCast((x * 255 / @max(w - 1, 1) + t * 16) & 255),
                @intCast(y * 255 / @max(h - 1, 1)),
                @as(u8, if ((x / 8 + y / 8 + t) % 2 == 0) 200 else 60),
                255,
            };
            @memcpy(r[x * ch ..][0..ch], v[0..ch]);
        }
    }
    return img;
}

fn meanAbsDiff(a: Image(u8), b: Image(u8)) f64 {
    var s: u64 = 0;
    for (0..a.height) |y| {
        for (a.row(y)[0 .. a.width * 3], 0..) |p, i| {
            const x = i / 3;
            const q = b.row(y)[x * b.channels + i % 3];
            s += @abs(@as(i32, p) - @as(i32, q));
        }
    }
    return @as(f64, @floatFromInt(s)) / @as(f64, @floatFromInt(@as(u64, a.width) * a.height * 3));
}

fn writeTest(gpa: Allocator, out: *std.ArrayList(u8), n: usize, w: u32, h: u32, opts: WriterOptions) !void {
    var wr = try Writer.init(gpa, .{ .memory = out }, w, h, 25, 1, opts);
    defer wr.deinit();
    for (0..n) |t| {
        var f = try testFrame(gpa, w, h, 3, t);
        defer f.deinit(gpa);
        try wr.write(f);
    }
    try wr.close();
}

test "avi: MJPEG write and read back, seek" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try writeTest(gpa, &out, 7, 64, 48, .{ .quality = 90 });
    try testing.expect(detect(out.items));
    var r = try Reader.init(gpa, .{ .memory = out.items });
    defer r.deinit();
    try testing.expectEqual(@as(u32, 64), r.width);
    try testing.expectEqual(@as(u32, 48), r.height);
    try testing.expectEqual(Codec.mjpeg, r.codec);
    try testing.expectEqual(@as(u64, 7), r.frameCount());
    try testing.expectEqual(@as(f64, 25), r.fps());
    // The OpenDML index is preferred over idx1.
    try testing.expect(r.index_kind == .odml);
    var img = try Image(u8).init(gpa, 64, 48, 3);
    defer img.deinit(gpa);
    var n: usize = 0;
    while (try r.next(img)) : (n += 1) {
        var want = try testFrame(gpa, 64, 48, 3, n);
        defer want.deinit(gpa);
        try testing.expect(meanAbsDiff(want, img) < 10);
    }
    try testing.expectEqual(@as(usize, 7), n);
    try r.seek(4);
    var bgra = try Image(u8).init(gpa, 64, 48, 4);
    defer bgra.deinit(gpa);
    try testing.expect(try r.next(bgra));
    var want = try testFrame(gpa, 64, 48, 3, 4);
    defer want.deinit(gpa);
    try testing.expect(meanAbsDiff(want, bgra) < 10);
    try testing.expectEqual(@as(u8, 255), bgra.row(3)[7]);
    // Wrong size.
    var small = try Image(u8).init(gpa, 32, 48, 3);
    defer small.deinit(gpa);
    try r.seek(0);
    try testing.expectError(error.SizeMismatch, r.next(small));
}

test "avi: OpenDML segments, idx1 and scan agree" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    // Tiny RIFF segments: several AVIX extensions.
    try writeTest(gpa, &out, 12, 40, 30, .{ .riff_limit = 4096 });
    var r = try Reader.init(gpa, .{ .memory = out.items });
    defer r.deinit();
    try testing.expectEqual(@as(u64, 12), r.frameCount());
    try testing.expect(r.index_kind == .odml);
    // Without the super index: idx1 covers the first segment only; with no
    // index at all a scan of every movi list finds them all.
    var p: Parser = .{ .gpa = gpa, .src = .{ .memory = out.items }, .file_size = out.items.len };
    defer p.deinit();
    try p.parse();
    try testing.expect(p.movi.items.len > 2);
    var v = p.video.?;
    v.indx = null;
    const a, const ka = try p.buildIndex(v);
    defer gpa.free(a);
    try testing.expect(ka == .idx1);
    try testing.expect(a.len < 12 and a.len > 0);
    p.idx1 = null;
    const b, const kb = try p.buildIndex(v);
    defer gpa.free(b);
    try testing.expect(kb == .scan);
    try testing.expectEqualSlices(Frame, r.index, b);
    for (a, r.index[0..a.len]) |x, y| try testing.expectEqual(x, y);
}

test "avi: uncompressed BGR" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try writeTest(gpa, &out, 3, 13, 5, .{ .codec = .bgr24 });
    var r = try Reader.init(gpa, .{ .memory = out.items });
    defer r.deinit();
    try testing.expectEqual(Codec.bgr24, r.codec);
    var img = try Image(u8).init(gpa, 13, 5, 3);
    defer img.deinit(gpa);
    try r.seek(2);
    try testing.expect(try r.next(img));
    var want = try testFrame(gpa, 13, 5, 3, 2);
    defer want.deinit(gpa);
    for (0..5) |y| try testing.expectEqualSlices(u8, want.row(y), img.row(y));
    try testing.expect(!r.bottom_up);
    // The same file with a positive height (bottom-up rows, the classic
    // DIB): the rows come out in reverse.
    std.mem.writeInt(i32, out.items[Writer.off_strf + 8 ..][0..4], 5, .little);
    var b = try Reader.init(gpa, .{ .memory = out.items });
    defer b.deinit();
    try testing.expect(b.bottom_up);
    try b.seek(2);
    try testing.expect(try b.next(img));
    for (0..5) |y| try testing.expectEqualSlices(u8, want.row(4 - y), img.row(y));
}

test "avi: fps fractions" {
    try testing.expectEqual([2]u32{ 30, 1 }, fpsFraction(30));
    try testing.expectEqual([2]u32{ 30000, 1001 }, fpsFraction(30000.0 / 1001.0));
    try testing.expectEqual([2]u32{ 25, 2 }, fpsFraction(12.5));
}

test "avi: truncated and corrupted files never crash" {
    const gpa = testing.allocator;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try writeTest(gpa, &out, 4, 24, 16, .{ .riff_limit = 2048 });
    const good = out.items;
    var img = try Image(u8).init(gpa, 24, 16, 3);
    defer img.deinit(gpa);
    const Try = struct {
        fn all(a: Allocator, bytes: []const u8, dst: Image(u8)) void {
            var r = Reader.init(a, .{ .memory = bytes }) catch return;
            defer r.deinit();
            if (r.width != dst.width or r.height != dst.height) return;
            var k: usize = 0;
            while (k < 64) : (k += 1) {
                const more = r.next(dst) catch {
                    // A bad frame: skip it, as a player would.
                    r.pos += 1;
                    continue;
                };
                if (!more) break;
            }
        }
    };
    // Every truncation length (step 7 keeps it quick).
    var len: usize = 0;
    while (len < good.len) : (len += 7) Try.all(gpa, good[0..len], img);
    // Random byte corruption, repeatable.
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    const bad = try gpa.dupe(u8, good);
    defer gpa.free(bad);
    for (0..300) |_| {
        @memcpy(bad, good);
        for (0..1 + rnd.uintLessThan(usize, 8)) |_| {
            const at = rnd.uintLessThan(usize, bad.len);
            bad[at] = rnd.int(u8);
        }
        // Sometimes a size field gets a huge value.
        if (rnd.boolean()) {
            const at = rnd.uintLessThan(usize, bad.len - 4) & ~@as(usize, 3);
            std.mem.writeInt(u32, bad[at..][0..4], rnd.int(u32), .little);
        }
        Try.all(gpa, bad, img);
    }
    // Not an AVI at all.
    try testing.expectError(error.Truncated, Reader.init(gpa, .{ .memory = "RIFF" }));
    try testing.expectError(error.InvalidData, Reader.init(gpa, .{ .memory = "RIFF\x04\x00\x00\x00WAVEfmt " }));
}
