//! PNG decoder and encoder (ISO/IEC 15948, W3C PNG 2nd/3rd edition).
//!
//! Decoding supports every standard colour type (gray, RGB, palette,
//! gray+alpha, RGBA) at every legal bit depth (1, 2, 4, 8, 16), tRNS
//! transparency, Adam7 interlacing, all five filters and IDAT data split over
//! any number of chunks. CRCs of critical chunks are verified; ancillary
//! chunks with a bad CRC, and unknown ancillary chunks, are skipped. 16-bit
//! files decode to `.u16` images in native endianness; lower depths decode to
//! `.u8` (sub-byte gray is scaled to 0-255, as libpng's expand does).
//!
//! Inflate is a built-in table-driven decoder that streams the IDAT payloads
//! in place (no concatenation) through a small sliding window, so the
//! filtered image never exists in memory as a whole. The zlib Adler-32 is
//! not verified (chunk CRCs already cover the data).
//!
//! Encoding writes 8- or 16-bit images with 1 (gray), 2 (gray+alpha),
//! 3 (RGB) or 4 (RGBA) channels, choosing a filter per row with libpng's
//! minimum-sum-of-absolute-differences heuristic. `compression` selects:
//! 0 = stored blocks (filter None), 1 = run-length + dynamic Huffman
//! (zlib's Z_RLE, OpenCV's default; fast and, on photographic data, as small
//! as level 9), 2-9 = `std.compress.flate` LZ77 at that level.

const std = @import("std");
const simd = @import("../core/simd.zig");
const common = @import("common.zig");
const flate = std.compress.flate;

const Image = common.Image;
const AnyImage = common.AnyImage;
const DecodeError = common.DecodeError;
const EncodeError = common.EncodeError;

/// The 8-byte PNG file signature.
pub const signature = [8]u8{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' };

/// True when `bytes` starts with the PNG signature.
pub fn detect(bytes: []const u8) bool {
    return bytes.len >= signature.len and std.mem.eql(u8, bytes[0..signature.len], &signature);
}

// ---------------------------------------------------------------------------
// CRC-32 (ISO 3309 / ITU-T V.42, as used by PNG), slicing-by-8.

const crc_tables: [8][256]u32 = blk: {
    @setEvalBranchQuota(20000);
    var t: [8][256]u32 = undefined;
    for (0..256) |i| {
        var c: u32 = i;
        for (0..8) |_| c = if (c & 1 != 0) 0xEDB88320 ^ (c >> 1) else c >> 1;
        t[0][i] = c;
    }
    for (0..256) |i| {
        for (1..8) |k| t[k][i] = (t[k - 1][i] >> 8) ^ t[0][t[k - 1][i] & 0xff];
    }
    break :blk t;
};

fn crcUpdate(crc_in: u32, data: []const u8) u32 {
    const t = &crc_tables;
    var c = crc_in;
    var p = data;
    while (p.len >= 8) {
        const lo = std.mem.readInt(u32, p[0..4], .little) ^ c;
        const hi = std.mem.readInt(u32, p[4..8], .little);
        c = t[7][lo & 0xff] ^ t[6][(lo >> 8) & 0xff] ^ t[5][(lo >> 16) & 0xff] ^ t[4][lo >> 24] ^
            t[3][hi & 0xff] ^ t[2][(hi >> 8) & 0xff] ^ t[1][(hi >> 16) & 0xff] ^ t[0][hi >> 24];
        p = p[8..];
    }
    for (p) |b| c = t[0][(c ^ b) & 0xff] ^ (c >> 8);
    return c;
}

/// CRC-32 of `data` as stored in PNG chunks.
pub fn crc32(data: []const u8) u32 {
    return ~crcUpdate(0xffffffff, data);
}

// ---------------------------------------------------------------------------
// Shared format helpers.

fn tag(comptime s: *const [4]u8) u32 {
    return std.mem.readInt(u32, s, .big);
}

const ColorType = enum(u8) {
    gray = 0,
    rgb = 2,
    palette = 3,
    gray_alpha = 4,
    rgba = 6,

    fn samples(c: ColorType) u8 {
        return switch (c) {
            .gray, .palette => 1,
            .gray_alpha => 2,
            .rgb => 3,
            .rgba => 4,
        };
    }
};

const Header = struct {
    width: u32,
    height: u32,
    depth: u8,
    color: ColorType,
    interlaced: bool,

    fn bitsPerPixel(h: Header) usize {
        return @as(usize, h.color.samples()) * h.depth;
    }
    /// Bytes in one unfiltered scanline of `w` pixels (without filter byte).
    fn rowBytes(h: Header, w: usize) usize {
        return (w * h.bitsPerPixel() + 7) / 8;
    }
    /// Byte distance to the "left" pixel used by the filters.
    fn filterBpp(h: Header) usize {
        return @max(1, h.bitsPerPixel() / 8);
    }
};

/// Adam7 passes: x0, y0, dx, dy.
const adam7 = [7][4]u8{
    .{ 0, 0, 8, 8 }, .{ 4, 0, 8, 8 }, .{ 0, 4, 4, 8 }, .{ 2, 0, 4, 4 },
    .{ 0, 2, 2, 4 }, .{ 1, 0, 2, 2 }, .{ 0, 1, 1, 2 },
};

fn passDim(full: u32, start: u8, step: u8) u32 {
    if (full <= start) return 0;
    return (full - start + step - 1) / step;
}

// ---------------------------------------------------------------------------
// Decoder.

/// Decodes a PNG file. The result has 1 (gray), 2 (gray+alpha), 3 (RGB) or
/// 4 (RGBA) channels in RGB order before `common.finish` applies
/// `opts.mode` / `opts.order`. With `.unchanged` or `.color_alpha`, a tRNS
/// chunk adds an alpha channel (palette, gray and RGB images).
pub fn decode(gpa: std.mem.Allocator, bytes: []const u8, opts: common.DecodeOptions) DecodeError!AnyImage {
    if (!detect(bytes)) return error.InvalidData;

    var hdr: ?Header = null;
    var palette: [256][4]u8 = @splat(.{ 0, 0, 0, 255 });
    var pal_len: usize = 0;
    var trns: ?[]const u8 = null;
    var idat_count: usize = 0;
    var first_idat: []const u8 = &.{};
    var first_idat_pos: usize = 0;

    var pos: usize = signature.len;
    chunks: while (true) {
        if (bytes.len - pos < 12) {
            if (idat_count > 0) break;
            return error.Truncated;
        }
        const len = std.mem.readInt(u32, bytes[pos..][0..4], .big);
        if (len > 0x7fff_ffff) return error.InvalidData;
        const type_bytes = bytes[pos + 4 ..][0..4];
        const ty = std.mem.readInt(u32, type_bytes, .big);
        if (bytes.len - pos - 12 < len) {
            // A cut-off ancillary chunk after the image data is harmless.
            if (idat_count > 0 and ty != tag("IDAT")) break;
            return error.Truncated;
        }
        const data = bytes[pos + 8 ..][0..len];
        const stored_crc = std.mem.readInt(u32, bytes[pos + 8 + len ..][0..4], .big);
        const critical = type_bytes[0] & 0x20 == 0;
        const chunk_pos = pos;
        pos += 12 + @as(usize, len);
        if (crc32(bytes[chunk_pos + 4 .. chunk_pos + 8 + len]) != stored_crc) {
            if (critical) return error.InvalidData;
            continue;
        }
        if (hdr == null and ty != tag("IHDR")) return error.InvalidData;
        switch (ty) {
            tag("IHDR") => {
                if (hdr != null or len != 13) return error.InvalidData;
                hdr = try parseHeader(data);
            },
            tag("PLTE") => {
                if (len == 0 or len % 3 != 0 or len / 3 > 256 or idat_count > 0) return error.InvalidData;
                pal_len = len / 3;
                for (0..pal_len) |i| palette[i] = .{ data[3 * i], data[3 * i + 1], data[3 * i + 2], 255 };
            },
            tag("tRNS") => if (idat_count == 0) {
                trns = data;
            },
            tag("IDAT") => {
                if (idat_count == 0) {
                    first_idat = data;
                    first_idat_pos = chunk_pos;
                }
                idat_count += 1;
            },
            tag("IEND") => break :chunks,
            else => if (critical) return error.Unsupported,
        }
    }

    const h = hdr orelse return error.InvalidData;
    if (idat_count == 0) return error.InvalidData;
    if (h.color == .palette and pal_len == 0) return error.InvalidData;
    if (@as(u64, h.width) * h.height > opts.max_pixels) return error.TooLarge;

    // Interpret tRNS (ignored when malformed, as libpng does).
    var trns_gray: ?u16 = null;
    var trns_rgb: ?[3]u16 = null;
    var has_trns = false;
    if (trns) |t| switch (h.color) {
        .palette => if (t.len > 0 and t.len <= pal_len) {
            for (t, 0..) |a, i| palette[i][3] = a;
            has_trns = true;
        },
        .gray => if (t.len == 2) {
            trns_gray = std.mem.readInt(u16, t[0..2], .big);
            has_trns = true;
        },
        .rgb => if (t.len == 6) {
            trns_rgb = .{
                std.mem.readInt(u16, t[0..2], .big),
                std.mem.readInt(u16, t[2..4], .big),
                std.mem.readInt(u16, t[4..6], .big),
            };
            has_trns = true;
        },
        else => {},
    };
    const add_alpha = has_trns and (opts.mode == .unchanged or opts.mode == .color_alpha);

    // The inflater reads the IDAT payloads in place, chunk after chunk.
    const zsrc: ZSource = .{
        .first = first_idat,
        .file = bytes,
        .next = first_idat_pos + 12 + first_idat.len,
    };

    const out_ch: u16 = switch (h.color) {
        .gray => if (add_alpha) 2 else 1,
        .gray_alpha => 2,
        .rgb, .palette => if (add_alpha) 4 else 3,
        .rgba => 4,
    };
    // Write BGR directly unless `finish` still has to convert channels
    // (its gray conversion expects RGB order).
    const want_ch: ?u16 = switch (opts.mode) {
        .gray => 1,
        .color => 3,
        .color_alpha => 4,
        else => null,
    };
    const bgr = opts.order == .bgr and out_ch >= 3 and (want_ch == null or want_ch.? == out_ch);
    var fin = opts;
    if (bgr) {
        fin.order = .rgb;
        for (&palette) |*p| std.mem.swap(u8, &p[0], &p[2]);
    }
    const ctx: Expand = .{
        .depth = h.depth,
        .color = h.color,
        .out_ch = out_ch,
        .palette = &palette,
        .trns_gray = if (add_alpha) trns_gray else null,
        .trns_rgb = if (add_alpha) trns_rgb else null,
        .swap_rb = bgr and h.color != .palette,
    };

    if (h.depth == 16) {
        var img = try allocImage(u16, gpa, h.width, h.height, out_ch);
        reconstruct(u16, gpa, h, &ctx, zsrc, img) catch |e| {
            img.deinit(gpa);
            return e;
        };
        return common.finish(gpa, .{ .u16 = img }, fin);
    } else {
        var img = try allocImage(u8, gpa, h.width, h.height, out_ch);
        reconstruct(u8, gpa, h, &ctx, zsrc, img) catch |e| {
            img.deinit(gpa);
            return e;
        };
        return common.finish(gpa, .{ .u8 = img }, fin);
    }
}

fn parseHeader(d: []const u8) DecodeError!Header {
    const w = std.mem.readInt(u32, d[0..4], .big);
    const h = std.mem.readInt(u32, d[4..8], .big);
    const depth = d[8];
    if (w == 0 or h == 0 or w > 0x7fff_ffff or h > 0x7fff_ffff) return error.InvalidData;
    const color: ColorType = switch (d[9]) {
        0 => .gray,
        2 => .rgb,
        3 => .palette,
        4 => .gray_alpha,
        6 => .rgba,
        else => return error.InvalidData,
    };
    const ok = switch (color) {
        .gray => depth == 1 or depth == 2 or depth == 4 or depth == 8 or depth == 16,
        .palette => depth == 1 or depth == 2 or depth == 4 or depth == 8,
        else => depth == 8 or depth == 16,
    };
    if (!ok) return error.InvalidData;
    if (d[10] != 0 or d[11] != 0) return error.InvalidData; // compression, filter method
    if (d[12] > 1) return error.InvalidData;
    return .{ .width = w, .height = h, .depth = depth, .color = color, .interlaced = d[12] == 1 };
}

fn allocImage(comptime T: type, gpa: std.mem.Allocator, w: u32, h: u32, c: u16) DecodeError!Image(T) {
    return Image(T).init(gpa, w, h, c) catch |e| switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.TooLarge,
    };
}

/// Where the zlib stream lives: the first IDAT payload, then the payloads
/// of any later IDAT chunks of `file` (searched from offset `next`).
const ZSource = struct {
    first: []const u8,
    file: []const u8 = &.{},
    next: usize = 0,
};

/// Inflates the zlib stream `z` into exactly `out.len` bytes (used by
/// tests; the decoder streams through `Inflater` directly). Trailing data
/// is ignored and the Adler-32 trailer is not checked (PNG chunk CRCs
/// already cover the data).
fn inflate(z: []const u8, out: []u8) DecodeError!void {
    var inf: Inflater = .{ .src = z, .zsrc = .{ .first = z }, .out = out };
    try inf.zlibHeader();
    try inf.fill();
    if (inf.op != out.len) return error.Truncated;
}

// ---------------------------------------------------------------------------
// Inflate (RFC 1951), specialised for decoding into one known-size buffer.
//
// A 64-bit bit buffer is refilled branch-free while 8 input bytes remain, and
// Huffman codes are decoded through a two-level table (a root table indexed by
// the next `root` bits, with sub-tables for longer codes). Matches copy 8
// bytes at a time while there is slack at the end of the output.

/// Decode-table entry layout (u32):
///   bits 0-4   bits to consume (code length; `root` for a sub-table link)
///   bits 5-8   extra bits that follow (lengths/distances)
///   bits 9-10  kind (see below)
///   bit  11    invalid code
///   bits 12-15 sub-table index bits (links only)
///   bits 16-31 literal byte, length/distance base, or sub-table offset
const Entry = struct {
    const lit: u32 = 0 << 9;
    const len: u32 = 1 << 9;
    const eob: u32 = 2 << 9;
    const link: u32 = 3 << 9;
    const kind_mask: u32 = 3 << 9;
    const invalid: u32 = 1 << 11;
    const lit_test: u32 = kind_mask | invalid;
};

const len_base = [29]u16{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258 };
const len_extra = [29]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
const dist_base = [30]u16{ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577 };
const dist_extra = [30]u8{ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

const lit_root = 10;
const dist_root = 8;
const clen_root = 7;
// Every code of length > root lives in a sub-table of at most 2^(15-root)
// entries, so these bounds hold even for incomplete codes.
const lit_table_len = (1 << lit_root) + 288 * (1 << (15 - lit_root));
const dist_table_len = (1 << dist_root) + 32 * (1 << (15 - dist_root));

const TableKind = enum { litlen, dist, clen };

/// Builds a canonical-Huffman decode table from code lengths. Unused slots of
/// incomplete codes are marked invalid; over-subscribed codes are rejected.
fn buildTable(table: []u32, lens: []const u4, comptime kind: TableKind) DecodeError!void {
    const root: u5 = switch (kind) {
        .litlen => lit_root,
        .dist => dist_root,
        .clen => clen_root,
    };
    var count: [16]u16 = @splat(0);
    for (lens) |l| count[l] += 1;
    count[0] = 0;
    var left: i32 = 1;
    for (1..16) |l| {
        left = left * 2 - count[l];
        if (left < 0) return error.InvalidData;
    }
    var offs: [16]u16 = undefined;
    offs[1] = 0;
    for (1..15) |l| offs[l + 1] = offs[l] + count[l];
    var sorted: [288]u16 = undefined;
    for (lens, 0..) |l, s| if (l != 0) {
        sorted[offs[l]] = @intCast(s);
        offs[l] += 1;
    };
    const n_codes = offs[15];

    const root_size = @as(usize, 1) << root;
    @memset(table[0..root_size], Entry.invalid);
    var next_sub: usize = root_size;
    var code: u32 = 0; // canonical code, MSB-first
    var cur_len: u5 = 1;
    var used: [16]u16 = @splat(0);
    var sub_prefix: usize = std.math.maxInt(usize);
    var sub_start: usize = 0;
    var sub_bits: u5 = 0;
    for (sorted[0..n_codes]) |sym| {
        const l: u5 = lens[sym];
        if (l != cur_len) {
            code <<= l - cur_len;
            cur_len = l;
        }
        used[l] += 1;
        const rev = @bitReverse(@as(u16, @intCast(code))) >> @intCast(16 - @as(u6, l));
        const e = symbolEntry(kind, sym);
        if (l <= root) {
            var i: usize = rev;
            while (i < root_size) : (i += @as(usize, 1) << l) table[i] = e | l;
        } else {
            const prefix = rev & (root_size - 1);
            if (prefix != sub_prefix) {
                // New sub-table: size it to hold every remaining code that
                // shares this prefix (zlib's inflate_table method).
                sub_prefix = prefix;
                var bits: u5 = l - root;
                var room: i32 = @as(i32, 1) << bits;
                while (bits + root < 15) {
                    room -= count[bits + root] - (if (bits + root == l) used[l] - 1 else 0);
                    if (room <= 0) break;
                    bits += 1;
                    room <<= 1;
                }
                sub_bits = bits;
                sub_start = next_sub;
                const size = @as(usize, 1) << bits;
                if (sub_start + size > table.len) return error.InvalidData;
                @memset(table[sub_start..][0..size], Entry.invalid);
                next_sub += size;
                table[prefix] = Entry.link | (@as(u32, sub_bits) << 12) | (@as(u32, @intCast(sub_start)) << 16) | root;
            }
            const sub_len = l - root;
            var i: usize = rev >> root;
            while (i < (@as(usize, 1) << sub_bits)) : (i += @as(usize, 1) << sub_len) table[sub_start + i] = e | sub_len;
        }
        code += 1;
    }
}

fn symbolEntry(comptime kind: TableKind, sym: u16) u32 {
    switch (kind) {
        .clen => return Entry.lit | (@as(u32, sym) << 16),
        .litlen => {
            if (sym < 256) return Entry.lit | (@as(u32, sym) << 16);
            if (sym == 256) return Entry.eob;
            if (sym > 285) return Entry.invalid;
            const k = sym - 257;
            return Entry.len | (@as(u32, len_extra[k]) << 5) | (@as(u32, len_base[k]) << 16);
        },
        .dist => {
            if (sym > 29) return Entry.invalid;
            return Entry.len | (@as(u32, dist_extra[sym]) << 5) | (@as(u32, dist_base[sym]) << 16);
        },
    }
}

const Inflater = struct {
    /// The input segment being read, and the position in it.
    src: []const u8,
    pos: usize = 0,
    zsrc: ZSource,
    /// Output window. `fill` decodes until `op == out.len` or the stream
    /// ends; the caller may then slide the window (keeping at least the
    /// last 32 KiB as match history) and lower `op` accordingly.
    out: []u8,
    op: usize = 0,
    state: enum { header, stored, huffman, done } = .header,
    final: bool = false,
    have_fixed: bool = false,
    stored_left: usize = 0,
    /// A match cut short by the end of the window.
    pend_len: usize = 0,
    pend_dist: usize = 0,
    bitbuf: u64 = 0,
    bitcount: u32 = 0,
    /// Zero bytes fed past the end of `src` by the slow refill.
    overrun: u32 = 0,
    lit: [lit_table_len]u32 = undefined,
    dist: [dist_table_len]u32 = undefined,

    /// Ensures at least 56 bits are buffered.
    inline fn refill(f: *Inflater) void {
        if (f.src.len - f.pos >= 8) {
            f.bitbuf |= std.mem.readInt(u64, f.src[f.pos..][0..8], .little) << @intCast(f.bitcount);
            f.pos += (63 - f.bitcount) >> 3;
            f.bitcount |= 56;
        } else f.refillSlow();
    }

    fn refillSlow(f: *Inflater) void {
        @branchHint(.unlikely);
        while (f.bitcount < 56) {
            var b: u64 = 0;
            if (f.pos < f.src.len or f.nextSegment()) {
                b = f.src[f.pos];
                f.pos += 1;
            } else f.overrun += 1;
            f.bitbuf |= b << @intCast(f.bitcount);
            f.bitcount += 8;
        }
    }

    /// Moves to the next non-empty IDAT payload; false at the end of data.
    fn nextSegment(f: *Inflater) bool {
        const file = f.zsrc.file;
        var p = f.zsrc.next;
        while (p != 0 and file.len - p >= 12) {
            const len = std.mem.readInt(u32, file[p..][0..4], .big);
            const ty = std.mem.readInt(u32, file[p + 4 ..][0..4], .big);
            if (ty == tag("IEND") or file.len - p - 12 < len) break;
            p += 12 + @as(usize, len);
            if (ty == tag("IDAT") and len > 0) {
                f.zsrc.next = p;
                f.src = file[p - 4 - len ..][0..len];
                f.pos = 0;
                return true;
            }
        }
        f.zsrc.next = 0;
        return false;
    }

    /// Reads and checks the 2-byte zlib header (RFC 1950).
    fn zlibHeader(f: *Inflater) DecodeError!void {
        const cmf = try f.take(8);
        const flg = try f.take(8);
        if (cmf & 0x0f != 8 or cmf >> 4 > 7 or (cmf * 256 + flg) % 31 != 0 or flg & 0x20 != 0)
            return error.InvalidData;
    }

    inline fn bits(f: *Inflater, n: u5) u32 {
        return @intCast(f.bitbuf & ((@as(u64, 1) << n) - 1));
    }

    inline fn consume(f: *Inflater, n: u5) void {
        f.bitbuf >>= n;
        f.bitcount -= n;
    }

    /// Fails when bits past the end of the input have been consumed.
    fn checkOverrun(f: *Inflater) DecodeError!void {
        if (f.overrun * 8 > f.bitcount) return error.Truncated;
    }

    fn take(f: *Inflater, n: u5) DecodeError!u32 {
        f.refill();
        const v = f.bits(n);
        f.consume(n);
        try f.checkOverrun();
        return v;
    }

    /// Looks up the next symbol; the caller has refilled.
    inline fn decodeSym(f: *Inflater, table: []const u32, comptime root: u5) u32 {
        var e = table[f.bits(root)];
        if (e & Entry.kind_mask == Entry.link) {
            f.consume(root);
            const sb: u5 = @intCast((e >> 12) & 0xf);
            e = table[(e >> 16) + f.bits(sb)];
        }
        f.consume(@intCast(e & 31));
        return e;
    }

    /// Decodes until the window is full or the stream has ended.
    fn fill(f: *Inflater) DecodeError!void {
        while (f.op < f.out.len) {
            if (f.pend_len > 0) {
                const n = @min(f.pend_len, f.out.len - f.op);
                for (0..n) |i| f.out[f.op + i] = f.out[f.op + i - f.pend_dist];
                f.op += n;
                f.pend_len -= n;
                continue;
            }
            switch (f.state) {
                .done => return,
                .header => {
                    const hdr = try f.take(3);
                    f.final = hdr & 1 != 0;
                    switch (hdr >> 1) {
                        0 => try f.storedHeader(),
                        1 => {
                            if (!f.have_fixed) {
                                var lens: [288]u4 = undefined;
                                @memset(lens[0..144], 8);
                                @memset(lens[144..256], 9);
                                @memset(lens[256..280], 7);
                                @memset(lens[280..288], 8);
                                try buildTable(&f.lit, &lens, .litlen);
                                const dl: [30]u4 = @splat(5);
                                try buildTable(&f.dist, &dl, .dist);
                                f.have_fixed = true;
                            }
                            f.state = .huffman;
                        },
                        2 => {
                            f.have_fixed = false;
                            try f.dynamicHeader();
                            f.state = .huffman;
                        },
                        else => return error.InvalidData,
                    }
                },
                .stored => {
                    // Whole bytes still in the bit buffer come first.
                    while (f.stored_left > 0 and f.op < f.out.len and f.bitcount >= 8) {
                        if (f.overrun * 8 >= f.bitcount) return error.Truncated;
                        f.out[f.op] = @truncate(f.bitbuf);
                        f.consume(8);
                        f.op += 1;
                        f.stored_left -= 1;
                    }
                    // Drop look-ahead bits beyond `bitcount` (see `refill`):
                    // the bytes are now read directly.
                    if (f.bitcount == 0) f.bitbuf = 0;
                    if (f.stored_left == 0) {
                        f.endBlock();
                        continue;
                    }
                    if (f.op == f.out.len) return;
                    if (f.pos == f.src.len and !f.nextSegment()) return error.Truncated;
                    const n = @min(f.stored_left, f.out.len - f.op, f.src.len - f.pos);
                    @memcpy(f.out[f.op..][0..n], f.src[f.pos..][0..n]);
                    f.op += n;
                    f.pos += n;
                    f.stored_left -= n;
                    if (f.stored_left == 0) f.endBlock();
                },
                .huffman => if (try f.codes()) f.endBlock(),
            }
        }
    }

    fn endBlock(f: *Inflater) void {
        f.state = if (f.final) .done else .header;
    }

    fn storedHeader(f: *Inflater) DecodeError!void {
        f.consume(@intCast(f.bitcount & 7)); // to a byte boundary
        const len = try f.take(16);
        const nlen = try f.take(16);
        if (len != ~nlen & 0xffff) return error.InvalidData;
        f.stored_left = len;
        if (len == 0) f.endBlock() else f.state = .stored;
    }

    fn dynamicHeader(f: *Inflater) DecodeError!void {
        const hlit = try f.take(5) + 257;
        const hdist = try f.take(5) + 1;
        const hclen = try f.take(4) + 4;
        if (hlit > 286 or hdist > 30) return error.InvalidData;
        const order = [19]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
        var clens: [19]u4 = @splat(0);
        for (order[0..hclen]) |i| clens[i] = @intCast(try f.take(3));
        var ctab: [1 << clen_root]u32 = undefined;
        try buildTable(&ctab, &clens, .clen);

        var lens: [286 + 30]u4 = @splat(0);
        const total = hlit + hdist;
        var i: usize = 0;
        while (i < total) {
            f.refill();
            const e = ctab[f.bits(clen_root)];
            if (e & Entry.invalid != 0) return error.InvalidData;
            f.consume(@intCast(e & 31));
            const sym = e >> 16;
            var rep: usize = 0;
            var val: u4 = 0;
            switch (sym) {
                0...15 => {
                    lens[i] = @intCast(sym);
                    i += 1;
                    try f.checkOverrun();
                    continue;
                },
                16 => {
                    if (i == 0) return error.InvalidData;
                    val = lens[i - 1];
                    rep = 3 + f.bits(2);
                    f.consume(2);
                },
                17 => {
                    rep = 3 + f.bits(3);
                    f.consume(3);
                },
                else => {
                    rep = 11 + f.bits(7);
                    f.consume(7);
                },
            }
            try f.checkOverrun();
            if (i + rep > total) return error.InvalidData;
            @memset(lens[i..][0..rep], val);
            i += rep;
        }
        if (lens[256] == 0) return error.InvalidData;
        try buildTable(&f.lit, lens[0..hlit], .litlen);
        try buildTable(&f.dist, lens[hlit..][0..hdist], .dist);
    }

    /// Decodes one compressed block's symbols until end-of-block (returns
    /// true) or until the window is full (returns false). The hot loop keeps the bit buffer in locals (so
    /// output stores cannot alias it) and decodes up to three literals per
    /// refill.
    fn codes(f: *Inflater) DecodeError!bool {
        const out = f.out;
        var src = f.src;
        const lt = &f.lit;
        const dt = &f.dist;
        const lmask: u64 = (1 << lit_root) - 1;
        const dmask: u64 = (1 << dist_root) - 1;
        // Output slack for a maximal match plus the 8-byte over-copy and the
        // literals decoded before it; input slack for two 8-byte refills.
        const fast_out = if (out.len > 258 + 32) out.len - (258 + 32) else 0;
        var fast_in = if (src.len > 16) src.len - 16 else 0;
        var op = f.op;
        var bb = f.bitbuf;
        var bc = f.bitcount;
        var ip = f.pos;
        defer {
            f.op = op;
            f.bitbuf = bb;
            f.bitcount = bc;
            f.pos = ip;
        }
        while (true) {
            if (op >= fast_out or ip > fast_in or f.overrun != 0) {
                @branchHint(.unlikely);
                f.op = op;
                f.bitbuf = bb;
                f.bitcount = bc;
                f.pos = ip;
                const r = try f.slowSymbol();
                src = f.src;
                fast_in = if (src.len > 16) src.len - 16 else 0;
                op = f.op;
                bb = f.bitbuf;
                bc = f.bitcount;
                ip = f.pos;
                switch (r) {
                    .more => continue,
                    .eob => return true,
                    .full => return false,
                }
            }
            bb |= std.mem.readInt(u64, src[ip..][0..8], .little) << @intCast(bc);
            ip += (63 - bc) >> 3;
            bc |= 56;
            var e = lt[@intCast(bb & lmask)];
            if (e & Entry.lit_test == 0) {
                // >= 56 bits buffered: room for three 15-bit literals.
                inline for (0..2) |_| {
                    bb >>= @intCast(e & 31);
                    bc -= e & 31;
                    out[op] = @truncate(e >> 16);
                    op += 1;
                    e = lt[@intCast(bb & lmask)];
                    if (e & Entry.lit_test != 0) break;
                } else {
                    bb >>= @intCast(e & 31);
                    bc -= e & 31;
                    out[op] = @truncate(e >> 16);
                    op += 1;
                    continue;
                }
                // `e` is still a valid lookup; top up before the match.
                bb |= std.mem.readInt(u64, src[ip..][0..8], .little) << @intCast(bc);
                ip += (63 - bc) >> 3;
                bc |= 56;
            }
            if (e & Entry.kind_mask == Entry.link) {
                bb >>= lit_root;
                bc -= lit_root;
                const sb: u6 = @intCast((e >> 12) & 0xf);
                e = lt[(e >> 16) + @as(usize, @intCast(bb & ((@as(u64, 1) << sb) - 1)))];
            }
            bb >>= @intCast(e & 31);
            bc -= e & 31;
            if (e & Entry.lit_test != 0) {
                if (e & Entry.invalid != 0) return error.InvalidData;
                if (e & Entry.kind_mask == Entry.eob) return true;
                // Length code; the 56 buffered bits cover length, distance
                // and both extra fields (15 + 5 + 15 + 13).
                const lx: u6 = @intCast((e >> 5) & 0xf);
                const length: usize = (e >> 16) + @as(usize, @intCast(bb & ((@as(u64, 1) << lx) - 1)));
                bb >>= lx;
                bc -= lx;
                var d = dt[@intCast(bb & dmask)];
                if (d & Entry.kind_mask == Entry.link) {
                    bb >>= dist_root;
                    bc -= dist_root;
                    const sb: u6 = @intCast((d >> 12) & 0xf);
                    d = dt[(d >> 16) + @as(usize, @intCast(bb & ((@as(u64, 1) << sb) - 1)))];
                }
                if (d & Entry.invalid != 0) return error.InvalidData;
                bb >>= @intCast(d & 31);
                bc -= d & 31;
                const dx: u6 = @intCast((d >> 5) & 0xf);
                const dist: usize = (d >> 16) + @as(usize, @intCast(bb & ((@as(u64, 1) << dx) - 1)));
                bb >>= dx;
                bc -= dx;
                if (dist > op) return error.InvalidData;
                copyMatch(out, op, dist, length);
                op += length;
            } else {
                out[op] = @truncate(e >> 16);
                op += 1;
            }
        }
    }

    /// One symbol with full bounds checks (used near the end of the window
    /// or of the input).
    fn slowSymbol(f: *Inflater) DecodeError!enum { more, eob, full } {
        if (f.op == f.out.len) return .full;
        f.refill();
        const e = f.decodeSym(&f.lit, lit_root);
        if (e & Entry.invalid != 0) return error.InvalidData;
        try f.checkOverrun();
        switch (e & Entry.kind_mask) {
            Entry.lit => {
                f.out[f.op] = @truncate(e >> 16);
                f.op += 1;
                return .more;
            },
            Entry.eob => return .eob,
            else => {},
        }
        const lx: u5 = @intCast((e >> 5) & 0xf);
        const length: usize = (e >> 16) + f.bits(lx);
        f.consume(lx);
        f.refill();
        const d = f.decodeSym(&f.dist, dist_root);
        if (d & Entry.invalid != 0) return error.InvalidData;
        const dx: u5 = @intCast((d >> 5) & 0xf);
        const dist: usize = (d >> 16) + f.bits(dx);
        f.consume(dx);
        try f.checkOverrun();
        if (dist > f.op) return error.InvalidData;
        const n = @min(length, f.out.len - f.op);
        for (0..n) |i| f.out[f.op + i] = f.out[f.op + i - dist];
        f.op += n;
        f.pend_len = length - n;
        f.pend_dist = dist;
        return .more;
    }
};

/// Copies `length` bytes from `dist` back. May write up to 15 bytes past the
/// end of the match (the caller guarantees the slack).
inline fn copyMatch(out: []u8, op: usize, dist: usize, length: usize) void {
    const dst = out.ptr + op;
    const src = dst - dist;
    if (dist >= 8) {
        var i: usize = 0;
        while (i < length) : (i += 8) {
            const v = std.mem.readInt(u64, (src + i)[0..8], .little);
            std.mem.writeInt(u64, (dst + i)[0..8], v, .little);
        }
    } else if (dist == 1) {
        const v: u64 = @as(u64, src[0]) * 0x0101010101010101;
        var i: usize = 0;
        while (i < length) : (i += 8) std.mem.writeInt(u64, (dst + i)[0..8], v, .little);
    } else {
        for (0..length) |i| dst[i] = src[i];
    }
}

/// Inflates `zdata` through a sliding window, unfilters each scanline and
/// expands it into `img`. Keeping only a window (32 KiB history plus a few
/// hundred KiB) instead of the whole filtered image stays cache-resident.
fn reconstruct(comptime T: type, gpa: std.mem.Allocator, h: Header, ctx: *const Expand, zsrc: ZSource, img: Image(T)) DecodeError!void {
    const bpp = h.filterBpp();
    const rb_max = h.rowBytes(h.width);
    const win = try gpa.alloc(u8, flate.history_len + rb_max + 1 + 256 * 1024);
    defer gpa.free(win);
    const rows = try gpa.alloc(u8, 2 * rb_max);
    defer gpa.free(rows);
    var cur = rows[0..rb_max];
    var prev = rows[rb_max..];
    const inf = try gpa.create(Inflater);
    defer gpa.destroy(inf);
    inf.* = .{ .src = zsrc.first, .zsrc = zsrc, .out = win };
    try inf.zlibHeader();
    var rpos: usize = 0;

    const ch = img.channels;
    const tmp: []T = if (h.interlaced) try gpa.alloc(T, @as(usize, img.width) * ch) else &.{};
    defer if (h.interlaced) gpa.free(tmp);

    const passes: []const [4]u8 = if (h.interlaced) &adam7 else &.{.{ 0, 0, 1, 1 }};
    for (passes) |p| {
        const pw = passDim(h.width, p[0], p[2]);
        const ph = passDim(h.height, p[1], p[3]);
        if (pw == 0 or ph == 0) continue;
        const rb = h.rowBytes(pw);
        @memset(prev[0..rb], 0);
        for (0..ph) |py| {
            const fr = try nextRow(inf, &rpos, rb + 1);
            @memcpy(cur[0..rb], fr[1..]);
            try unfilterRow(fr[0], cur[0..rb], prev[0..rb], bpp);
            const y = p[1] + py * p[3];
            if (!h.interlaced) {
                ctx.row(T, cur[0..rb], img.row(y));
            } else {
                const t = tmp[0 .. @as(usize, pw) * ch];
                ctx.row(T, cur[0..rb], t);
                const dst = img.row(y);
                for (0..pw) |i| {
                    const x = p[0] + i * p[2];
                    @memcpy(dst[x * ch ..][0..ch], t[i * ch ..][0..ch]);
                }
            }
            std.mem.swap([]u8, &cur, &prev);
        }
    }
}

/// Returns the next `need` bytes of inflated data, decoding (and sliding the
/// window) as required.
fn nextRow(inf: *Inflater, rpos: *usize, need: usize) DecodeError![]const u8 {
    while (inf.op - rpos.* < need) {
        if (inf.op == inf.out.len) {
            const keep_from = @min(rpos.*, inf.op - flate.history_len);
            const keep = inf.op - keep_from;
            std.mem.copyForwards(u8, inf.out[0..keep], inf.out[keep_from..inf.op]);
            inf.op = keep;
            rpos.* -= keep_from;
        }
        const before = inf.op;
        try inf.fill();
        if (inf.op == before) return error.Truncated;
    }
    defer rpos.* += need;
    return inf.out[rpos.*..][0..need];
}

fn unfilterRow(ft: u8, cur: []u8, prev: []const u8, bpp: usize) DecodeError!void {
    switch (bpp) {
        inline 1, 2, 3, 4, 6, 8 => |b| return unfilter(b, ft, cur, prev),
        else => unreachable,
    }
}

/// Reverses one scanline's filter. `prev` is the reconstructed previous
/// scanline (zeros for the first one); `cur.len` is a multiple of `bpp`.
fn unfilter(comptime bpp: usize, ft: u8, cur: []u8, prev: []const u8) DecodeError!void {
    const n = cur.len;
    const V8 = @Vector(bpp, u8);
    const V16 = @Vector(bpp, i16);
    switch (ft) {
        0 => {},
        1 => { // Sub
            var a: V8 = @splat(0);
            var i: usize = 0;
            while (i < n) : (i += bpp) {
                const x: V8 = cur[i..][0..bpp].*;
                a +%= x;
                cur[i..][0..bpp].* = a;
            }
        },
        2 => for (cur, prev[0..n]) |*c, p| { // Up
            c.* +%= p;
        },
        3 => { // Average
            var a: @Vector(bpp, u16) = @splat(0);
            var i: usize = 0;
            while (i < n) : (i += bpp) {
                const x: V8 = cur[i..][0..bpp].*;
                const b: V8 = prev[i..][0..bpp].*;
                const avg: V8 = @intCast((a + @as(@Vector(bpp, u16), b)) >> @splat(1));
                const r = x +% avg;
                cur[i..][0..bpp].* = r;
                a = r;
            }
        },
        4 => { // Paeth
            var a: V16 = @splat(0);
            var c: V16 = @splat(0);
            var i: usize = 0;
            while (i < n) : (i += bpp) {
                const x: V8 = cur[i..][0..bpp].*;
                const bb: V8 = prev[i..][0..bpp].*;
                const b: V16 = bb;
                const pa = @abs(b - c);
                const pb = @abs(a - c);
                const pc = @abs(a + b - c - c);
                const bc = @select(i16, pb <= pc, b, c);
                const pred = @select(i16, pa <= pb, @select(i16, pa <= pc, a, bc), bc);
                const r = x +% simd.narrow(u8, pred);
                cur[i..][0..bpp].* = r;
                a = r;
                c = b;
            }
        },
        else => return error.InvalidData,
    }
}

/// Converts one unfiltered scanline to output samples.
const Expand = struct {
    depth: u8,
    color: ColorType,
    out_ch: u16,
    palette: *const [256][4]u8,
    trns_gray: ?u16,
    trns_rgb: ?[3]u16,
    /// Swap R and B (BGR output) for non-palette colour images.
    swap_rb: bool,

    /// `dst` holds exactly `n * out_ch` samples for `n` pixels.
    fn row(e: *const Expand, comptime T: type, src: []const u8, dst: []T) void {
        if (T == u16) e.row16(src, dst) else e.row8(src, dst);
        if (e.swap_rb) switch (e.out_ch) {
            3 => swapRow(T, 3, dst),
            4 => swapRow(T, 4, dst),
            else => {},
        };
    }

    fn row8(e: *const Expand, src: []const u8, dst: []u8) void {
        const n = dst.len / e.out_ch;
        switch (e.color) {
            .palette => switch (e.depth) {
                8 => if (e.out_ch == 4) {
                    for (0..n) |i| dst[i * 4 ..][0..4].* = e.palette[src[i]];
                } else {
                    for (0..n) |i| dst[i * 3 ..][0..3].* = e.palette[src[i]][0..3].*;
                },
                else => {
                    const d: u3 = @intCast(e.depth);
                    for (0..n) |i| {
                        const idx = subByte(src, i, d);
                        @memcpy(dst[i * e.out_ch ..][0..e.out_ch], e.palette[idx][0..e.out_ch]);
                    }
                },
            },
            .gray => if (e.depth == 8) {
                if (e.trns_gray) |t| {
                    for (0..n) |i| {
                        const v = src[i];
                        dst[2 * i] = v;
                        dst[2 * i + 1] = if (v == t) 0 else 255;
                    }
                } else @memcpy(dst, src[0..n]);
            } else {
                const d: u3 = @intCast(e.depth);
                const scale: u8 = 255 / ((@as(u8, 1) << d) - 1);
                if (e.trns_gray) |t| {
                    for (0..n) |i| {
                        const v = subByte(src, i, d);
                        dst[2 * i] = v * scale;
                        dst[2 * i + 1] = if (v == t) 0 else 255;
                    }
                } else {
                    for (0..n) |i| dst[i] = subByte(src, i, d) * scale;
                }
            },
            .rgb => if (e.trns_rgb) |t| {
                for (0..n) |i| {
                    const p = src[i * 3 ..][0..3];
                    dst[i * 4 ..][0..3].* = p.*;
                    dst[i * 4 + 3] = if (p[0] == t[0] and p[1] == t[1] and p[2] == t[2]) 0 else 255;
                }
            } else @memcpy(dst, src[0..dst.len]),
            .gray_alpha, .rgba => @memcpy(dst, src[0..dst.len]),
        }
    }

    fn row16(e: *const Expand, src: []const u8, dst: []u16) void {
        const n = dst.len / e.out_ch;
        const in_ch: usize = e.color.samples();
        if (in_ch == e.out_ch) {
            for (dst, 0..) |*d, i| d.* = std.mem.readInt(u16, src[2 * i ..][0..2], .big);
            return;
        }
        // Gray or RGB with a tRNS key colour: append alpha.
        for (0..n) |i| {
            var opaque_px = false;
            for (0..in_ch) |k| {
                const v = std.mem.readInt(u16, src[2 * (i * in_ch + k) ..][0..2], .big);
                dst[i * e.out_ch + k] = v;
                const key = if (e.trns_gray) |g| g else e.trns_rgb.?[k];
                if (v != key) opaque_px = true;
            }
            dst[i * e.out_ch + in_ch] = if (opaque_px) 0xffff else 0;
        }
    }
};

/// In-place R<->B swap of one row of `ch`-channel pixels.
fn swapRow(comptime T: type, comptime ch: usize, row: []T) void {
    var i: usize = 0;
    while (i + ch <= row.len) : (i += ch) {
        const p = row[i..][0..ch];
        const r = p[0];
        p[0] = p[2];
        p[2] = r;
    }
}

/// Sample `i` of a packed 1/2/4-bit scanline (most significant bits first).
inline fn subByte(src: []const u8, i: usize, d: u3) u8 {
    const bit = i * d;
    const shift: u3 = @intCast(8 - @as(u4, d) - (bit & 7));
    const mask: u8 = (@as(u8, 1) << d) - 1;
    return (src[bit >> 3] >> shift) & mask;
}

// ---------------------------------------------------------------------------
// Encoder.

/// Encodes an 8- or 16-bit image with 1-4 channels as PNG. `opts.order`
/// gives the channel order of 3/4-channel input (BGR input is swapped while
/// writing; the image is not modified). `opts.compression` 0-9 selects the
/// deflate strategy (see the file comment). Caller frees the result.
pub fn encode(gpa: std.mem.Allocator, img: AnyImage, opts: common.EncodeOptions) EncodeError![]u8 {
    return switch (img) {
        .u8 => |im| encodeTyped(u8, gpa, im, opts),
        .u16 => |im| encodeTyped(u16, gpa, im, opts),
        else => error.Unsupported,
    };
}

fn encodeTyped(comptime T: type, gpa: std.mem.Allocator, img: Image(T), opts: common.EncodeOptions) EncodeError![]u8 {
    const ch = img.channels;
    if (ch < 1 or ch > 4) return error.Unsupported;
    if (img.width > 0x7fff_ffff or img.height > 0x7fff_ffff) return error.Unsupported;
    const bps = @sizeOf(T);
    const bpp: usize = ch * bps;
    const rb = @as(usize, img.width) * bpp;
    const swap = opts.order == .bgr and ch >= 3;

    var out: std.Io.Writer.Allocating = try .initCapacity(gpa, @max(4096, rb * img.height / 2 + 1024));
    defer out.deinit();
    const w = &out.writer;

    // Signature, IHDR, and the IDAT chunk header (length patched later).
    var ihdr: [13]u8 = undefined;
    std.mem.writeInt(u32, ihdr[0..4], img.width, .big);
    std.mem.writeInt(u32, ihdr[4..8], img.height, .big);
    ihdr[8] = bps * 8;
    ihdr[9] = switch (ch) {
        1 => 0,
        2 => 4,
        3 => 2,
        else => 6,
    };
    ihdr[10] = 0;
    ihdr[11] = 0;
    ihdr[12] = 0;
    w.writeAll(&signature) catch return error.OutOfMemory;
    writeChunk(w, "IHDR", &ihdr) catch return error.OutOfMemory;
    const idat_start = w.end;
    w.writeAll(&.{ 0, 0, 0, 0, 'I', 'D', 'A', 'T' }) catch return error.OutOfMemory;

    // Row buffers with `pad` zero bytes in front so "left" reads need no
    // bounds checks: the previous row, the current row and 4 filter outputs.
    const pad = 8;
    const stride = pad + rb + filter_lanes;
    const bufs = try gpa.alloc(u8, stride * 6);
    defer gpa.free(bufs);
    @memset(bufs, 0);
    var prev = bufs[0..stride];
    var cur = bufs[stride..][0..stride];
    const cand = bufs[2 * stride ..];

    const level = @min(opts.compression, 9);
    if (level == 0) {
        // Stored deflate blocks, filter None.
        var st: Stored = .{ .w = w };
        st.begin() catch return error.OutOfMemory;
        for (0..img.height) |y| {
            loadRow(T, img.row(y), cur[pad..][0..rb], ch, swap);
            st.write(&.{0}) catch return error.OutOfMemory;
            st.write(cur[pad..][0..rb]) catch return error.OutOfMemory;
        }
        st.finish() catch return error.OutOfMemory;
    } else if (level == 1) {
        var rd = try RleDeflate.init(gpa, w);
        defer rd.deinit(gpa);
        rd.begin() catch return error.OutOfMemory;
        for (0..img.height) |y| {
            loadRow(T, img.row(y), cur[pad..][0..rb], ch, swap);
            const ft = chooseFilter(cur, prev, cand, stride, pad, rb, bpp);
            const data = if (ft == 0) cur[pad..][0..rb] else cand[(ft - 1) * stride ..][0..rb];
            rd.write(&.{ft}) catch return error.OutOfMemory;
            rd.write(data) catch return error.OutOfMemory;
            std.mem.swap([]u8, &prev, &cur);
        }
        rd.finish() catch return error.OutOfMemory;
    } else {
        const window = try gpa.alloc(u8, flate.max_window_len);
        defer gpa.free(window);
        const comp = try gpa.create(flate.Compress);
        defer gpa.destroy(comp);
        const lv: flate.Compress.Options = switch (level) {
            2 => .level_2,
            3 => .level_3,
            4 => .level_4,
            5 => .level_5,
            6 => .level_6,
            7 => .level_7,
            8 => .level_8,
            else => .level_9,
        };
        comp.* = flate.Compress.init(w, window, .zlib, lv) catch return error.OutOfMemory;
        const cw = &comp.writer;
        for (0..img.height) |y| {
            loadRow(T, img.row(y), cur[pad..][0..rb], ch, swap);
            const ft = chooseFilter(cur, prev, cand, stride, pad, rb, bpp);
            const data = if (ft == 0) cur[pad..][0..rb] else cand[(ft - 1) * stride ..][0..rb];
            cw.writeByte(ft) catch return error.OutOfMemory;
            cw.writeAll(data) catch return error.OutOfMemory;
            std.mem.swap([]u8, &prev, &cur);
        }
        comp.finish() catch return error.OutOfMemory;
    }

    // Patch the IDAT length (splitting into several chunks if it is huge),
    // then append the CRC and IEND.
    const max_chunk = 0x7fff_ffff;
    const idat_len = w.end - idat_start - 8;
    if (idat_len <= max_chunk) {
        const buf = w.buffer;
        std.mem.writeInt(u32, buf[idat_start..][0..4], @intCast(idat_len), .big);
        const crc = crc32(buf[idat_start + 4 .. w.end]);
        var cb: [4]u8 = undefined;
        std.mem.writeInt(u32, &cb, crc, .big);
        w.writeAll(&cb) catch return error.OutOfMemory;
    } else {
        const z = try gpa.dupe(u8, w.buffer[idat_start + 8 .. w.end]);
        defer gpa.free(z);
        w.end = idat_start;
        var off: usize = 0;
        while (off < z.len) : (off += max_chunk) {
            writeChunk(w, "IDAT", z[off..@min(z.len, off + max_chunk)]) catch return error.OutOfMemory;
        }
    }
    writeChunk(w, "IEND", "") catch return error.OutOfMemory;
    return out.toOwnedSlice();
}

fn writeChunk(w: *std.Io.Writer, ty: *const [4]u8, data: []const u8) std.Io.Writer.Error!void {
    try w.writeInt(u32, @intCast(data.len), .big);
    try w.writeAll(ty);
    try w.writeAll(data);
    try w.writeInt(u32, ~crcUpdate(crcUpdate(0xffffffff, ty), data), .big);
}

/// Copies one image row to PNG byte order (big-endian samples, RGB order).
fn loadRow(comptime T: type, src: []const T, dst: []u8, ch: u16, swap: bool) void {
    if (T == u8) {
        @memcpy(dst, src);
        if (swap) {
            var i: usize = 0;
            while (i < dst.len) : (i += ch) std.mem.swap(u8, &dst[i], &dst[i + 2]);
        }
    } else {
        for (src, 0..) |v, i| std.mem.writeInt(u16, dst[2 * i ..][0..2], v, .big);
        if (swap) {
            var i: usize = 0;
            while (i < src.len) : (i += ch) {
                std.mem.swap([2]u8, dst[2 * i ..][0..2], dst[2 * (i + 2) ..][0..2]);
            }
        }
    }
}

const filter_lanes = 32;

/// Computes the Sub, Up, Average and Paeth versions of the row at
/// `cur[pad..][0..rb]` into the four candidate buffers and returns the filter
/// type whose output has the smallest sum of absolute (signed) values.
fn chooseFilter(cur: []u8, prev: []u8, cand: []u8, stride: usize, pad: usize, rb: usize, bpp: usize) u8 {
    const L = filter_lanes;
    const V = @Vector(L, u8);
    const W = @Vector(L, i16);
    const S = @Vector(L, i8);
    var sums: [5]@Vector(L, u32) = @splat(@splat(0));
    const subo = cand[0..];
    const upo = cand[stride..];
    const avgo = cand[2 * stride ..];
    const pao = cand[3 * stride ..];
    // Rows are padded to a multiple of L with zeros beyond `rb`, which
    // contribute 0 to every sum except where a "left" neighbour lies inside
    // the row; mask those lanes out.
    const tail_zero = cur[pad + rb ..][0..L];
    @memset(tail_zero, 0);
    @memset(prev[pad + rb ..][0..L], 0);
    var i: usize = 0;
    while (i < rb) : (i += L) {
        const x: V = cur[pad + i ..][0..L].*;
        const a: V = cur[pad + i - bpp ..][0..L].*;
        const b: V = prev[pad + i ..][0..L].*;
        const c: V = prev[pad + i - bpp ..][0..L].*;
        var valid: V = @splat(0xff);
        if (i + L > rb) {
            const idx = std.simd.iota(u8, L);
            valid = @select(u8, idx < @as(V, @splat(@intCast(rb - i))), valid, @as(V, @splat(0)));
        }
        const sub = (x -% a) & valid;
        const up = (x -% b) & valid;
        const avg_pred: V = @intCast((@as(@Vector(L, u16), a) + b) >> @splat(1));
        const avg = (x -% avg_pred) & valid;
        const aw: W = a;
        const bw: W = b;
        const cw: W = c;
        const pa = @abs(bw - cw);
        const pb = @abs(aw - cw);
        const pc = @abs(aw + bw - cw - cw);
        const bc = @select(u8, pb <= pc, b, c);
        const pred = @select(u8, pa <= pb, @select(u8, pa <= pc, a, bc), bc);
        const pae = (x -% pred) & valid;
        subo[i..][0..L].* = sub;
        upo[i..][0..L].* = up;
        avgo[i..][0..L].* = avg;
        pao[i..][0..L].* = pae;
        sums[0] += @as(@Vector(L, u32), @abs(@as(S, @bitCast(x & valid))));
        sums[1] += @as(@Vector(L, u32), @abs(@as(S, @bitCast(sub))));
        sums[2] += @as(@Vector(L, u32), @abs(@as(S, @bitCast(up))));
        sums[3] += @as(@Vector(L, u32), @abs(@as(S, @bitCast(avg))));
        sums[4] += @as(@Vector(L, u32), @abs(@as(S, @bitCast(pae))));
    }
    var best: u8 = 0;
    var best_sum: u64 = std.math.maxInt(u64);
    for (sums, 0..) |s, k| {
        const t: u64 = @reduce(.Add, @as(@Vector(L, u64), s));
        if (t < best_sum) {
            best_sum = t;
            best = @intCast(k);
        }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Fast deflate: run-length matches (distance 1) + per-block dynamic Huffman
// codes, the scheme of zlib's Z_RLE strategy (OpenCV's PNG default). On
// filtered image data it compresses about as well as level-6 LZ77 at a
// fraction of the cost.

const rle_block = 1 << 16; // input bytes per deflate block

const RleDeflate = struct {
    w: *std.Io.Writer,
    adler: std.hash.Adler32 = .{},
    buf: []u8,
    n: usize = 0,
    prev: ?u8 = null,
    acc: u64 = 0,
    nacc: u6 = 0,
    /// Literal/length tokens of the current block: < 256 literal,
    /// 0x8000 | len for a distance-1 match.
    tokens: []u16,

    fn init(gpa: std.mem.Allocator, w: *std.Io.Writer) !RleDeflate {
        const buf = try gpa.alloc(u8, rle_block);
        errdefer gpa.free(buf);
        const tokens = try gpa.alloc(u16, rle_block);
        return .{ .w = w, .buf = buf, .tokens = tokens };
    }

    fn deinit(d: *RleDeflate, gpa: std.mem.Allocator) void {
        gpa.free(d.buf);
        gpa.free(d.tokens);
    }

    fn begin(d: *RleDeflate) !void {
        try d.w.writeAll(&.{ 0x78, 0x01 });
    }

    fn write(d: *RleDeflate, data: []const u8) !void {
        d.adler.update(data);
        var rest = data;
        while (rest.len > 0) {
            const k = @min(rest.len, d.buf.len - d.n);
            @memcpy(d.buf[d.n..][0..k], rest[0..k]);
            d.n += k;
            rest = rest[k..];
            if (d.n == d.buf.len) try d.block(false);
        }
    }

    fn finish(d: *RleDeflate) !void {
        try d.block(true);
        // Byte-align, then the Adler-32 trailer.
        if (d.nacc > 0) {
            try d.ensure(8);
            d.flushBits(true);
        }
        try d.w.writeInt(u32, d.adler.adler, .big);
    }

    fn ensure(d: *RleDeflate, n: usize) !void {
        _ = try d.w.writableSliceGreedy(n);
    }

    /// Appends `count` (<= 32) bits; the caller has ensured capacity.
    inline fn put(d: *RleDeflate, bits: u32, count: u6) void {
        d.acc |= @as(u64, bits) << d.nacc;
        d.nacc += count;
        if (d.nacc >= 32) {
            std.mem.writeInt(u32, d.w.buffer[d.w.end..][0..4], @truncate(d.acc), .little);
            d.w.end += 4;
            d.acc >>= 32;
            d.nacc -= 32;
        }
    }

    fn flushBits(d: *RleDeflate, all: bool) void {
        while (d.nacc >= 8 or (all and d.nacc > 0)) {
            d.w.buffer[d.w.end] = @truncate(d.acc);
            d.w.end += 1;
            d.acc >>= 8;
            d.nacc -|= 8;
        }
    }

    fn block(d: *RleDeflate, final: bool) !void {
        const data = d.buf[0..d.n];
        var lfreq: [286]u32 = @splat(0);
        var dfreq: [30]u32 = @splat(0);
        var nt: usize = 0;
        var i: usize = 0;
        var prev = d.prev;
        while (i < data.len) {
            const b = data[i];
            if (prev != null and prev.? == b) {
                var r: usize = 1;
                const max = @min(258, data.len - i);
                while (r < max and data[i + r] == b) r += 1;
                if (r >= 3) {
                    d.tokens[nt] = 0x8000 | @as(u16, @intCast(r));
                    nt += 1;
                    lfreq[257 + lenCode(r)] += 1;
                    dfreq[0] += 1;
                    i += r;
                    continue;
                }
            }
            d.tokens[nt] = b;
            nt += 1;
            lfreq[b] += 1;
            prev = b;
            i += 1;
        }
        if (data.len > 0) d.prev = data[data.len - 1];
        d.n = 0;
        lfreq[256] = 1;
        // Keep both trees complete (>= 2 codes) so every inflater accepts them.
        if (std.mem.count(u32, &lfreq, &.{0}) >= lfreq.len - 1) lfreq[if (lfreq[0] == 0) 0 else 1] += 1;
        dfreq[1] = @max(dfreq[1], 1);
        dfreq[0] = @max(dfreq[0], 1);

        var llen: [286]u4 = undefined;
        var dlen: [30]u4 = undefined;
        huffmanLengths(&lfreq, &llen, 15);
        huffmanLengths(&dfreq, &dlen, 15);
        var lcode: [286]u16 = undefined;
        var dcode: [30]u16 = undefined;
        canonicalCodes(&llen, &lcode);
        canonicalCodes(&dlen, &dcode);

        // Header: code lengths run-length coded with symbols 16/17/18.
        var hlit: usize = 286;
        while (hlit > 257 and llen[hlit - 1] == 0) hlit -= 1;
        var hdist: usize = 30;
        while (hdist > 1 and dlen[hdist - 1] == 0) hdist -= 1;
        var all: [316]u4 = undefined;
        @memcpy(all[0..hlit], llen[0..hlit]);
        @memcpy(all[hlit..][0..hdist], dlen[0..hdist]);
        const lens = all[0 .. hlit + hdist];
        var cl_syms: [316]u8 = undefined;
        var cl_extra: [316]u8 = undefined;
        var ncl: usize = 0;
        var cfreq: [19]u32 = @splat(0);
        var j: usize = 0;
        while (j < lens.len) {
            const v = lens[j];
            var r: usize = 1;
            while (j + r < lens.len and lens[j + r] == v) r += 1;
            j += r;
            if (v == 0) {
                while (r >= 11) {
                    const k = @min(r, 138);
                    cl_syms[ncl] = 18;
                    cl_extra[ncl] = @intCast(k - 11);
                    ncl += 1;
                    r -= k;
                }
                if (r >= 3) {
                    cl_syms[ncl] = 17;
                    cl_extra[ncl] = @intCast(r - 3);
                    ncl += 1;
                    r = 0;
                }
            } else {
                cl_syms[ncl] = v;
                ncl += 1;
                r -= 1;
                while (r >= 3) {
                    const k = @min(r, 6);
                    cl_syms[ncl] = 16;
                    cl_extra[ncl] = @intCast(k - 3);
                    ncl += 1;
                    r -= k;
                }
            }
            while (r > 0) : (r -= 1) {
                cl_syms[ncl] = v;
                ncl += 1;
            }
        }
        for (cl_syms[0..ncl]) |c| cfreq[c] += 1;
        if (std.mem.count(u32, &cfreq, &.{0}) >= cfreq.len - 1) cfreq[if (cfreq[0] == 0) 0 else 1] += 1;
        var clen: [19]u4 = undefined;
        huffmanLengths(&cfreq, &clen, 7);
        var ccode: [19]u16 = undefined;
        canonicalCodes(&clen, &ccode);
        const order = [19]u8{ 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
        var hclen: usize = 19;
        while (hclen > 4 and clen[order[hclen - 1]] == 0) hclen -= 1;

        // Worst case: header plus 2 bytes per literal / 5 per match.
        try d.ensure(1024 + nt * 5);
        d.put(@as(u32, @intFromBool(final)) | (2 << 1), 3);
        d.put(@intCast(hlit - 257), 5);
        d.put(@intCast(hdist - 1), 5);
        d.put(@intCast(hclen - 4), 4);
        for (order[0..hclen]) |o| d.put(clen[o], 3);
        for (cl_syms[0..ncl], cl_extra[0..ncl]) |c, e| {
            d.put(ccode[c], clen[c]);
            switch (c) {
                16 => d.put(e, 2),
                17 => d.put(e, 3),
                18 => d.put(e, 7),
                else => {},
            }
        }
        const d0: u32 = dcode[0];
        const d0n: u6 = dlen[0];
        for (d.tokens[0..nt]) |t| {
            if (t < 0x100) {
                d.put(lcode[t], llen[t]);
            } else {
                const len: usize = t & 0x7fff;
                const lc = lenCode(len);
                const sym = 257 + lc;
                d.put(lcode[sym], llen[sym]);
                const xb: u6 = @intCast(len_extra[lc]);
                if (xb > 0) d.put(@intCast(len - len_base[lc]), xb);
                d.put(d0, d0n);
            }
        }
        d.put(lcode[256], llen[256]);
        d.flushBits(false);
    }
};

/// Length (3..258) to length-code index 0..28.
fn lenCode(len: usize) usize {
    const table = comptime blk: {
        var t: [259]u8 = undefined;
        for (0..29) |c| {
            const hi = if (c == 28) 258 else if (c == 27) 257 else len_base[c + 1] - 1;
            for (len_base[c]..hi + 1) |l| t[l] = c;
        }
        t[258] = 28;
        break :blk t;
    };
    return table[len];
}

/// Length-limited Huffman code lengths for `freq` (zero frequency -> length
/// 0): Moffat-Katajainen minimum-redundancy lengths, then clamped to
/// `limit` with the Kraft sum repaired (as miniz does).
fn huffmanLengths(freq: []const u32, out: []u4, limit: u5) void {
    var syms: [288]u16 = undefined;
    var a: [288]u32 = undefined;
    var n: usize = 0;
    for (freq, 0..) |f, s| if (f > 0) {
        syms[n] = @intCast(s);
        n += 1;
    };
    @memset(out, 0);
    if (n == 0) return;
    const Ctx = struct {
        f: []const u32,
        fn lt(c: @This(), x: u16, y: u16) bool {
            return c.f[x] < c.f[y] or (c.f[x] == c.f[y] and x < y);
        }
    };
    std.sort.pdq(u16, syms[0..n], Ctx{ .f = freq }, Ctx.lt);
    if (n == 1) {
        out[syms[0]] = 1;
        return;
    }
    for (0..n) |i| a[i] = freq[syms[i]];
    // In-place minimum-redundancy code lengths (Moffat & Katajainen 1995).
    {
        var root: usize = 0;
        var leaf: usize = 2;
        a[0] += a[1];
        for (1..n - 1) |next| {
            if (leaf >= n or a[root] < a[leaf]) {
                a[next] = a[root];
                a[root] = @intCast(next);
                root += 1;
            } else {
                a[next] = a[leaf];
                leaf += 1;
            }
            if (leaf >= n or (root < next and a[root] < a[leaf])) {
                a[next] += a[root];
                a[root] = @intCast(next);
                root += 1;
            } else {
                a[next] += a[leaf];
                leaf += 1;
            }
        }
        a[n - 2] = 0;
        var next: usize = n - 2;
        while (next > 0) {
            next -= 1;
            a[next] = a[a[next]] + 1;
        }
        var avbl: usize = 1;
        var used: usize = 0;
        var dpth: u32 = 0;
        var r: isize = @intCast(n - 2);
        var nx: isize = @intCast(n - 1);
        while (avbl > 0) {
            while (r >= 0 and a[@intCast(r)] == dpth) {
                used += 1;
                r -= 1;
            }
            while (avbl > used) {
                a[@intCast(nx)] = dpth;
                nx -= 1;
                avbl -= 1;
            }
            avbl = 2 * used;
            dpth += 1;
            used = 0;
        }
    }
    // a[i] is now the length of syms[i] (non-increasing). Count, clamp, fix.
    var count: [33]u32 = @splat(0);
    for (a[0..n]) |l| count[@min(l, 32)] += 1;
    for (limit + 1..33) |l| {
        count[limit] += count[l];
        count[l] = 0;
    }
    var total: u64 = 0;
    for (1..limit + 1) |l| total += @as(u64, count[l]) << @intCast(limit - l);
    const one: u64 = @as(u64, 1) << limit;
    while (total > one) {
        count[limit] -= 1;
        var l: usize = limit - 1;
        while (l > 0) : (l -= 1) if (count[l] > 0) {
            count[l] -= 1;
            count[l + 1] += 2;
            break;
        };
        total -= 1;
    }
    // Least frequent symbols get the longest codes.
    var i: usize = 0;
    var l: usize = limit;
    while (l > 0) : (l -= 1) {
        for (0..count[l]) |_| {
            out[syms[i]] = @intCast(l);
            i += 1;
        }
    }
}

/// Canonical codes, bit-reversed for LSB-first output.
fn canonicalCodes(lens: []const u4, codes: []u16) void {
    var count: [16]u16 = @splat(0);
    for (lens) |l| count[l] += 1;
    count[0] = 0;
    var next: [16]u16 = undefined;
    var c: u16 = 0;
    for (1..16) |b| {
        c = (c + count[b - 1]) << 1;
        next[b] = c;
    }
    for (lens, 0..) |l, s| {
        if (l == 0) {
            codes[s] = 0;
            continue;
        }
        codes[s] = @bitReverse(next[l]) >> @intCast(16 - @as(u5, l));
        next[l] += 1;
    }
}

/// Writes a zlib stream of stored (uncompressed) deflate blocks.
const Stored = struct {
    w: *std.Io.Writer,
    adler: std.hash.Adler32 = .{},
    block: [65535]u8 = undefined,
    n: usize = 0,

    fn begin(s: *Stored) !void {
        try s.w.writeAll(&.{ 0x78, 0x01 });
    }
    fn write(s: *Stored, data: []const u8) !void {
        s.adler.update(data);
        var d = data;
        while (d.len > 0) {
            const k = @min(d.len, s.block.len - s.n);
            @memcpy(s.block[s.n..][0..k], d[0..k]);
            s.n += k;
            d = d[k..];
            if (s.n == s.block.len) try s.flush(false);
        }
    }
    fn flush(s: *Stored, final: bool) !void {
        const len: u16 = @intCast(s.n);
        try s.w.writeByte(@intFromBool(final));
        try s.w.writeInt(u16, len, .little);
        try s.w.writeInt(u16, ~len, .little);
        try s.w.writeAll(s.block[0..s.n]);
        s.n = 0;
    }
    fn finish(s: *Stored) !void {
        try s.flush(true);
        try s.w.writeInt(u32, s.adler.adler, .big);
    }
};

// ---------------------------------------------------------------------------
// Tests.

const testing = std.testing;

fn randomImage(comptime T: type, gpa: std.mem.Allocator, rnd: std.Random, w: u32, h: u32, c: u16, smooth: bool) !Image(T) {
    var img = try Image(T).init(gpa, w, h, c);
    for (img.slice(), 0..) |*v, i| {
        if (smooth) {
            const x = (i / c) % w;
            const y = (i / c) / w;
            const base: u32 = @intCast((x * 3 + y * 2 + (i % c) * 40) % 256);
            const noise = rnd.uintLessThan(u32, 8);
            const v8: u32 = (base + noise) & 0xff;
            v.* = if (T == u8) @intCast(v8) else @intCast(v8 * 257 ^ rnd.uintLessThan(u32, 4));
        } else v.* = rnd.int(T);
    }
    return img;
}

fn expectSameImage(comptime T: type, a: Image(T), b: Image(T)) !void {
    try testing.expectEqual(a.width, b.width);
    try testing.expectEqual(a.height, b.height);
    try testing.expectEqual(a.channels, b.channels);
    for (0..a.height) |y| try testing.expectEqualSlices(T, a.row(y), b.row(y));
}

test "png round trip u8/u16, 1-4 channels, all levels" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(42);
    const rnd = prng.random();
    const sizes = [_][2]u32{ .{ 1, 1 }, .{ 7, 3 }, .{ 33, 17 }, .{ 64, 40 } };
    inline for (.{ u8, u16 }) |T| {
        for (sizes) |sz| {
            for (1..5) |c| {
                for ([_]u8{ 0, 1, 6, 9 }) |lvl| {
                    for ([_]bool{ false, true }) |smooth| {
                        var img = try randomImage(T, gpa, rnd, sz[0], sz[1], @intCast(c), smooth);
                        defer img.deinit(gpa);
                        const bytes = try encode(gpa, @unionInit(AnyImage, @typeName(T), img), .{ .compression = lvl });
                        defer gpa.free(bytes);
                        try testing.expect(detect(bytes));
                        var back = try decode(gpa, bytes, .{});
                        defer back.deinit(gpa);
                        try expectSameImage(T, img, back.as(T).?);
                    }
                }
            }
        }
    }
}

test "png bgr order on encode and decode" {
    const gpa = testing.allocator;
    var img = try Image(u8).init(gpa, 2, 1, 3);
    defer img.deinit(gpa);
    @memcpy(img.slice(), &[_]u8{ 1, 2, 3, 4, 5, 6 });
    const bytes = try encode(gpa, .{ .u8 = img }, .{ .order = .bgr });
    defer gpa.free(bytes);
    var rgb = try decode(gpa, bytes, .{});
    defer rgb.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 3, 2, 1, 6, 5, 4 }, rgb.u8.slice());
    var bgr = try decode(gpa, bytes, .{ .order = .bgr });
    defer bgr.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6 }, bgr.u8.slice());
    // The input must be untouched.
    try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6 }, img.slice());
}

test "png roi input (strided rows)" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(7);
    var img = try randomImage(u8, gpa, prng.random(), 20, 10, 3, false);
    defer img.deinit(gpa);
    const view = try img.roi(3, 2, 9, 5);
    const bytes = try encode(gpa, .{ .u8 = view }, .{});
    defer gpa.free(bytes);
    var back = try decode(gpa, bytes, .{});
    defer back.deinit(gpa);
    try expectSameImage(u8, view, back.u8);
}

/// Builds a PNG from already-filtered scanline data (test helper).
fn buildPng(gpa: std.mem.Allocator, ihdr: [13]u8, extra: []const []const u8, filtered: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.writeAll(&signature);
    try writeChunk(w, "IHDR", &ihdr);
    for (extra) |e| try writeChunk(w, e[0..4], e[4..]);
    var z: std.Io.Writer.Allocating = .init(gpa);
    defer z.deinit();
    var st: Stored = .{ .w = &z.writer };
    try st.begin();
    try st.write(filtered);
    try st.finish();
    try writeChunk(w, "IDAT", z.written());
    try writeChunk(w, "IEND", "");
    return out.toOwnedSlice();
}

fn ihdrBytes(w: u32, h: u32, depth: u8, color: u8, interlace: u8) [13]u8 {
    var b: [13]u8 = undefined;
    std.mem.writeInt(u32, b[0..4], w, .big);
    std.mem.writeInt(u32, b[4..8], h, .big);
    b[8] = depth;
    b[9] = color;
    b[10] = 0;
    b[11] = 0;
    b[12] = interlace;
    return b;
}

test "png 2-bit palette with tRNS" {
    const gpa = testing.allocator;
    // 5x1, indices 0 1 2 3 0 -> bits 00 01 10 11 | 00 000000
    const raw = [_]u8{ 0, 0b00011011, 0b00000000 };
    const plte = "PLTE" ++ [_]u8{ 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42 };
    const trns = "tRNS" ++ [_]u8{ 0, 128 };
    const bytes = try buildPng(gpa, ihdrBytes(5, 1, 2, 3, 0), &.{ plte, trns }, &raw);
    defer gpa.free(bytes);
    var img = try decode(gpa, bytes, .{});
    defer img.deinit(gpa);
    try testing.expectEqual(@as(u16, 4), img.channels());
    try testing.expectEqualSlices(u8, &.{
        10, 11, 12, 0, 20, 21, 22, 128, 30, 31, 32, 255, 40, 41, 42, 255, 10, 11, 12, 0,
    }, img.u8.slice());
    // Without alpha requested, tRNS is dropped.
    var rgb = try decode(gpa, bytes, .{ .mode = .color });
    defer rgb.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41, 42, 10, 11, 12 }, rgb.u8.slice());
}

test "png 1-bit gray scales to 0/255, gray tRNS adds alpha" {
    const gpa = testing.allocator;
    const raw = [_]u8{ 0, 0b10100000 }; // 3x1: 1 0 1
    const trns = "tRNS" ++ [_]u8{ 0, 1 };
    const bytes = try buildPng(gpa, ihdrBytes(3, 1, 1, 0, 0), &.{trns}, &raw);
    defer gpa.free(bytes);
    var img = try decode(gpa, bytes, .{});
    defer img.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 0, 255, 255, 0 }, img.u8.slice());
    var g = try decode(gpa, bytes, .{ .mode = .gray });
    defer g.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 255, 0, 255 }, g.u8.slice());
}

test "png 16-bit RGB tRNS and native endianness" {
    const gpa = testing.allocator;
    const raw = [_]u8{ 0, 0x12, 0x34, 0, 1, 0xff, 0xff, 0, 0, 0, 0, 0, 0 }; // 2x1
    const trns = "tRNS" ++ [_]u8{ 0x12, 0x34, 0, 1, 0xff, 0xff };
    const bytes = try buildPng(gpa, ihdrBytes(2, 1, 16, 2, 0), &.{trns}, &raw);
    defer gpa.free(bytes);
    var img = try decode(gpa, bytes, .{});
    defer img.deinit(gpa);
    try testing.expectEqualSlices(u16, &.{ 0x1234, 1, 0xffff, 0, 0, 0, 0, 0xffff }, img.u16.slice());
    var c8 = try decode(gpa, bytes, .{ .mode = .color });
    defer c8.deinit(gpa);
    try testing.expectEqualSlices(u8, &.{ 0x12, 0, 0xff, 0, 0, 0 }, c8.u8.slice());
}

test "png interlaced 1x1 and 3x3 gray" {
    const gpa = testing.allocator;
    {
        const bytes = try buildPng(gpa, ihdrBytes(1, 1, 8, 0, 1), &.{}, &.{ 0, 77 });
        defer gpa.free(bytes);
        var img = try decode(gpa, bytes, .{});
        defer img.deinit(gpa);
        try testing.expectEqualSlices(u8, &.{77}, img.u8.slice());
    }
    {
        // 3x3: pass1 (0,0); pass4 (2,0); pass5 (0,2),(2,2); pass6 (1,0),(1,2); pass7 row1: 3 px.
        const raw = [_]u8{
            0, 1, // pass 1
            0, 3, // pass 4
            0, 7, 9, // pass 5
            0, 2, // pass 6 row 0
            0, 8, // pass 6 row 2
            1, 4, 1, 1, // pass 7, Sub filter: 4 5 6
        };
        const bytes = try buildPng(gpa, ihdrBytes(3, 3, 8, 0, 1), &.{}, &raw);
        defer gpa.free(bytes);
        var img = try decode(gpa, bytes, .{});
        defer img.deinit(gpa);
        try testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, img.u8.slice());
    }
}

test "png malformed input returns errors" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(3);
    const rnd = prng.random();
    var img = try randomImage(u8, gpa, rnd, 23, 11, 3, true);
    defer img.deinit(gpa);
    const good = try encode(gpa, .{ .u8 = img }, .{});
    defer gpa.free(good);

    try testing.expectError(error.InvalidData, decode(gpa, "not a png", .{}));
    // Every truncation must fail cleanly (no panic).
    for (0..good.len - 12) |n| {
        if (decode(gpa, good[0..n], .{})) |r| {
            var rr = r;
            rr.deinit(gpa);
        } else |_| {}
    }
    // Bad CRC on IHDR.
    const bad = try gpa.dupe(u8, good);
    defer gpa.free(bad);
    bad[20] ^= 1;
    try testing.expectError(error.InvalidData, decode(gpa, bad, .{}));
    // Random corruption anywhere: no panic.
    for (0..300) |_| {
        @memcpy(bad, good);
        for (0..3) |_| bad[rnd.uintLessThan(usize, bad.len)] ^= rnd.int(u8) | 1;
        if (decode(gpa, bad, .{})) |r| {
            var rr = r;
            rr.deinit(gpa);
        } else |_| {}
    }
    // Pixel limit.
    try testing.expectError(error.TooLarge, decode(gpa, good, .{ .max_pixels = 100 }));
    // Bad filter type.
    const badf = try buildPng(gpa, ihdrBytes(1, 1, 8, 0, 0), &.{}, &.{ 5, 0 });
    defer gpa.free(badf);
    try testing.expectError(error.InvalidData, decode(gpa, badf, .{}));
    // Too little image data.
    const short = try buildPng(gpa, ihdrBytes(4, 4, 8, 0, 0), &.{}, &.{ 0, 1, 2 });
    defer gpa.free(short);
    try testing.expectError(error.Truncated, decode(gpa, short, .{}));
}

test "png crc32" {
    try testing.expectEqual(@as(u32, 0xCBF43926), crc32("123456789"));
    try testing.expectEqual(std.hash.Crc32.hash("The quick brown fox jumps over the lazy dog!!"), crc32("The quick brown fox jumps over the lazy dog!!"));
}

test "png inflate matches std.compress.flate output" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(11);
    const rnd = prng.random();
    const window = try gpa.alloc(u8, flate.max_window_len);
    defer gpa.free(window);
    const comp = try gpa.create(flate.Compress);
    defer gpa.destroy(comp);
    for ([_]usize{ 0, 1, 7, 300, 5000, 70000, 300000 }) |n| {
        const data = try gpa.alloc(u8, n);
        defer gpa.free(data);
        // Mix of literal noise, short repeats and long runs.
        var i: usize = 0;
        while (i < n) {
            const k = @min(n - i, rnd.uintLessThan(usize, 300) + 1);
            switch (rnd.uintLessThan(u8, 4)) {
                0 => rnd.bytes(data[i..][0..k]),
                1 => @memset(data[i..][0..k], rnd.int(u8)),
                2 => for (data[i..][0..k], 0..) |*b, j| {
                    b.* = if (i >= 1 + j % 7) data[i - 1 - j % 7] else 0;
                },
                else => for (data[i..][0..k], 0..) |*b, j| {
                    b.* = @truncate(j / 3);
                },
            }
            i += k;
        }
        for ([_]u8{ 0, 1, 4, 6, 9 }) |lvl| {
            var z: std.Io.Writer.Allocating = try .initCapacity(gpa, 64);
            defer z.deinit();
            if (lvl == 0) {
                var st: Stored = .{ .w = &z.writer };
                try st.begin();
                try st.write(data);
                try st.finish();
            } else {
                const opts: flate.Compress.Options = switch (lvl) {
                    1 => .level_1,
                    4 => .level_4,
                    6 => .level_6,
                    else => .level_9,
                };
                comp.* = try flate.Compress.init(&z.writer, window, .zlib, opts);
                try comp.writer.writeAll(data);
                try comp.finish();
            }
            const out = try gpa.alloc(u8, n);
            defer gpa.free(out);
            try inflate(z.written(), out);
            try testing.expectEqualSlices(u8, data, out);
            // Truncated streams must fail, never panic or succeed silently.
            if (n > 0) {
                const zs = z.written();
                for ([_]usize{ 1, 2, zs.len / 2, zs.len - 4 }) |cut| {
                    if (cut >= zs.len or cut == 0) continue;
                    if (inflate(zs[0 .. zs.len - cut], out)) {
                        // Only possible when the cut removed the trailer and at
                        // most the end-of-block code (tiny streams).
                        try testing.expect(cut <= 4 or n < 300);
                    } else |_| {}
                }
            }
        }
    }
}

test "png rle deflate output is valid zlib (checked with std inflate)" {
    const gpa = testing.allocator;
    var prng: std.Random.DefaultPrng = .init(5);
    const rnd = prng.random();
    for ([_]usize{ 0, 1, 2, 3, 1000, 200_000 }) |n| {
        const data = try gpa.alloc(u8, n);
        defer gpa.free(data);
        // Fibonacci-skewed byte frequencies force the 15-bit length limit;
        // runs exercise the length codes.
        var i: usize = 0;
        while (i < n) {
            var sym: u8 = 0;
            var r = rnd.int(u32);
            while (r & 1 == 1 and sym < 40) : (r >>= 1) sym += 1;
            const run = if (rnd.uintLessThan(u8, 8) == 0) rnd.uintLessThan(usize, 600) + 1 else 1;
            const k = @min(run, n - i);
            @memset(data[i..][0..k], sym *% 7);
            i += k;
        }
        var z: std.Io.Writer.Allocating = .init(gpa);
        defer z.deinit();
        var rd = try RleDeflate.init(gpa, &z.writer);
        defer rd.deinit(gpa);
        try rd.begin();
        // Uneven writes cross block boundaries.
        var off: usize = 0;
        while (off < n) {
            const k = @min(n - off, rnd.uintLessThan(usize, 90_000) + 1);
            try rd.write(data[off..][0..k]);
            off += k;
        }
        try rd.finish();

        var in: std.Io.Reader = .fixed(z.written());
        var dec: flate.Decompress = .init(&in, .zlib, &.{});
        var back: std.Io.Writer.Allocating = .init(gpa);
        defer back.deinit();
        _ = try dec.reader.streamRemaining(&back.writer);
        try testing.expectEqualSlices(u8, data, back.written());
        const adler = std.hash.Adler32.hash(data);
        try testing.expectEqual(adler, std.mem.readInt(u32, z.written()[z.written().len - 4 ..][0..4], .big));

        const out = try gpa.alloc(u8, n);
        defer gpa.free(out);
        try inflate(z.written(), out);
        try testing.expectEqualSlices(u8, data, out);
    }
}
