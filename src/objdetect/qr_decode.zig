//! QR code decoding from a module grid: format information (BCH), unmasking,
//! codeword extraction, Reed-Solomon correction and segment parsing. The
//! segment parser and the string it returns follow OpenCV's QRCodeDecoder
//! and QRDecode::decodingProcess (5.0), so decoded strings match cv2's,
//! including its ECI handling and its Latin-1 to UTF-8 fallback for byte
//! mode. Unlike OpenCV, the version information modules are excluded
//! exactly (OpenCV's mask is one column off, which costs it some error
//! correction capacity from version 7 on). Every input is treated as
//! untrusted: malformed streams fail instead of asserting.

const std = @import("std");
const Allocator = std.mem.Allocator;
const tables = @import("qr_tables.zig");
const gf = @import("gf256.zig");
const Level = @import("qr_encode.zig").Level;

pub const Info = struct {
    version: u8,
    level: Level,
    mask: u3,
    /// The first segment's mode (OpenCV's QRDecode::mode): 1 numeric,
    /// 2 alphanumeric, 3 structured append, 4 byte, 7 ECI, 8 kanji.
    mode: u4,
    /// The first ECI assignment (0 when none).
    eci: u32 = 0,
    /// Structured append header (total = 1 when absent).
    sequence_num: u8 = 0,
    total_num: u8 = 1,
    parity: u8 = 0,
};

/// The format word's (level, mask), or null when more than three bits are
/// wrong. `raw` holds the 15 read bits (dark = 1), bit i as OpenCV reads it.
fn correctFormat(raw: u16) ?u16 {
    for (tables.format_lut) |w| {
        if (@popCount(w ^ raw) <= 3) return w ^ 0b101010000010010;
    }
    return null;
}

const Grid = struct {
    d: []const u8,
    n: usize,
    transposed: bool,

    inline fn at(g: Grid, r: usize, c: usize) u1 {
        return @intCast(if (g.transposed) g.d[c * g.n + r] & 1 else g.d[r * g.n + c] & 1);
    }
};

fn readFormat(g: Grid) ?struct { level: Level, mask: u3 } {
    const n = g.n;
    var a: u16 = 0;
    for (0..6) |i| a |= @as(u16, g.at(i, 8)) << @intCast(i);
    a |= @as(u16, g.at(7, 8)) << 6;
    a |= @as(u16, g.at(8, 8)) << 7;
    a |= @as(u16, g.at(8, 7)) << 8;
    for (9..15) |i| a |= @as(u16, g.at(8, 14 - i)) << @intCast(i);
    const fa = correctFormat(a);
    var b: u16 = 0;
    for (0..8) |i| b |= @as(u16, g.at(8, n - 1 - i)) << @intCast(i);
    for (0..7) |i| b |= @as(u16, g.at(n - 7 + i, 8)) << @intCast(i + 8);
    const fb = correctFormat(b);
    var f: u16 = undefined;
    if (fb) |x| {
        if (fa != null and fa.? != x) return null;
        f = x;
    } else f = fa orelse return null;
    const level: Level = switch ((f >> 13) & 3) {
        0 => .M,
        1 => .L,
        2 => .H,
        else => .Q,
    };
    return .{ .level = level, .mask = @intCast((f >> 10) & 7) };
}

fn maskHit(mt: u3, i: usize, j: usize) bool {
    return switch (mt) {
        0 => (i + j) % 2 == 0,
        1 => i % 2 == 0,
        2 => j % 3 == 0,
        3 => (i + j) % 3 == 0,
        4 => ((i / 2) + (j / 3)) % 2 == 0,
        5 => (i * j) % 2 + (i * j) % 3 == 0,
        6 => ((i * j) % 2 + (i * j) % 3) % 2 == 0,
        7 => ((i * j) % 3 + (i + j) % 2) % 2 == 0,
    };
}

/// Marks the function modules of a version (1 = function pattern).
fn functionMap(version: usize, func: []u8) void {
    const n = 17 + 4 * version;
    @memset(func, 0);
    const set = struct {
        fn f(m: []u8, nn: usize, r0: usize, r1: usize, c0: usize, c1: usize) void {
            for (r0..r1) |r| @memset(m[r * nn + c0 .. r * nn + c1], 1);
        }
    }.f;
    set(func, n, 0, 9, 0, 9);
    set(func, n, 0, 9, n - 8, n);
    set(func, n, n - 8, n, 0, 9);
    set(func, n, 6, 7, 0, n);
    set(func, n, 0, n, 6, 7);
    if (version >= 7) {
        set(func, n, 0, 6, n - 11, n - 8);
        set(func, n, n - 11, n - 8, 0, 6);
    }
    const al = tables.versions[version].align_;
    var na: usize = 0;
    while (na < 7 and al[na] != 0) na += 1;
    for (0..na) |i| for (0..na) |j| {
        if ((i == 0 and j == 0) or (i == 0 and j == na - 1) or (i == na - 1 and j == 0)) continue;
        set(func, n, al[i] - 2, al[i] + 3, al[j] - 2, al[j] + 3);
    };
}

/// MSB-first reader over the corrected data codewords, failing where
/// OpenCV's Bitstream would throw.
const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    fn empty(r: *const Reader) bool {
        return r.pos >= r.data.len * 8;
    }

    fn next(r: *Reader, nbits: u5) error{Truncated}!u32 {
        const total = r.data.len * 8;
        if (r.pos >= total) return error.Truncated;
        const avail = total - r.pos;
        const take: usize = @min(nbits, avail);
        if (nbits - take >= 8) return error.Truncated;
        var v: u32 = 0;
        for (0..take) |_| {
            v = (v << 1) | ((r.data[r.pos >> 3] >> @intCast(7 - (r.pos & 7))) & 1);
            r.pos += 1;
        }
        return v << @intCast(nbits - take);
    }
};

const alpha_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:";

const Parser = struct {
    gpa: Allocator,
    r: Reader,
    version: usize,
    out: *std.ArrayList(u8),
    info: *Info,
    depth: u32 = 0,

    fn symbols(p: *Parser) !void {
        if (p.r.empty()) return error.Truncated;
        p.out.clearRetainingCapacity();
        while (!p.r.empty()) {
            const m: u4 = @intCast(try p.r.next(4));
            if (p.info.mode == 0) p.info.mode = m;
            if (m == 0 or p.r.empty()) return;
            switch (m) {
                1 => try p.numeric(),
                2 => try p.alpha(),
                4 => try p.byte(),
                7 => return p.eci(),
                8 => try p.kanji(),
                3 => {
                    p.info.sequence_num = @intCast(try p.r.next(4));
                    p.info.total_num = @intCast(1 + try p.r.next(4));
                    p.info.parity = @intCast(try p.r.next(8));
                },
                else => return error.InvalidData,
            }
        }
    }

    fn numeric(p: *Parser) !void {
        const v = p.version;
        const count = try p.r.next(if (v <= 9) 10 else if (v <= 26) 12 else 14);
        for (0..count / 3) |_| {
            const t = try p.r.next(10);
            try p.out.appendSlice(p.gpa, &.{ @truncate('0' + t / 100), @truncate('0' + (t / 10) % 10), @truncate('0' + t % 10) });
        }
        const rem = count % 3;
        if (rem != 0) {
            const t = try p.r.next(if (rem == 1) 4 else 7);
            if (rem == 2) try p.out.append(p.gpa, @truncate('0' + (t / 10) % 10));
            try p.out.append(p.gpa, @truncate('0' + t % 10));
        }
    }

    fn alpha(p: *Parser) !void {
        const v = p.version;
        const count = try p.r.next(if (v <= 9) 9 else if (v <= 26) 11 else 13);
        for (0..count / 2) |_| {
            const t = try p.r.next(11);
            // OpenCV 5.0 indexes its table without a range check; reject.
            if (t >= 45 * 45) return error.InvalidData;
            try p.out.appendSlice(p.gpa, &.{ alpha_map[t / 45], alpha_map[t % 45] });
        }
        if (count % 2 != 0) {
            const t = try p.r.next(6);
            if (t >= 45) return error.InvalidData;
            try p.out.append(p.gpa, alpha_map[t]);
        }
    }

    fn byte(p: *Parser) !void {
        const count = try p.r.next(if (p.version <= 9) 8 else 16);
        for (0..count) |_| try p.out.append(p.gpa, @truncate(try p.r.next(8)));
    }

    fn kanji(p: *Parser) !void {
        const v = p.version;
        const count = try p.r.next(if (v <= 9) 8 else if (v <= 26) 10 else 12);
        for (0..count) |_| {
            const d = try p.r.next(13);
            const hi = d / 0xC0;
            var sym = (hi << 8) + (d - hi * 0xC0);
            if (sym <= 0x9FFC - 0x8140) {
                sym += 0x8140;
            } else if (sym >= 0xE040 - 0xC140 and sym <= 0xEBBF - 0xC140) {
                sym += 0xC140;
            }
            try p.out.appendSlice(p.gpa, &.{ @truncate(sym >> 8), @truncate(sym) });
        }
    }

    fn eci(p: *Parser) anyerror!void {
        var value = try p.r.next(8);
        for (0..8) |i| {
            if (value & (@as(u32, 1) << @intCast(7 - i)) != 0) {
                if (i >= 3) return error.InvalidData; // would overflow 32 bits
                value |= (try p.r.next(8)) << @intCast((i + 1) * 8);
            } else break;
        }
        if (p.info.eci == 0) p.info.eci = value;
        p.depth += 1;
        if (p.depth > 64) return error.InvalidData;
        return p.symbols();
    }
};

fn isAscii(s: []const u8) bool {
    for (s) |c| if (c >= 0x80) return false;
    return true;
}

fn isUtf8(s: []const u8) bool {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        const b = s[i];
        if (b < 0x80) continue;
        const n: usize = if (b & 0xe0 == 0xc0) 2 else if (b & 0xf0 == 0xe0) 3 else if (b & 0xf8 == 0xf0) 4 else return false;
        for (1..n) |j| if (i + j >= s.len or s[i + j] & 0xc0 != 0x80) return false;
        i += n - 1;
    }
    return true;
}

/// One attempt on the grid (possibly transposed): OpenCV's run().
fn run(gpa: Allocator, g: Grid, version: usize, out: *std.ArrayList(u8), info: *Info) !bool {
    const n = g.n;
    const fmt = readFormat(g) orelse return false;
    const vi = tables.versions[version];
    const blk = vi.ecc[@backingInt(fmt.level)];
    const total: usize = vi.total;

    var func_buf: [177 * 177]u8 = undefined;
    const func = func_buf[0 .. n * n];
    functionMap(version, func);

    // Codewords in the zig-zag order, unmasked.
    var cw_buf: [3706]u8 = undefined;
    const cw = cw_buf[0..total];
    @memset(cw, 0);
    var bit: usize = 0;
    var upward = true;
    var right: isize = @intCast(n - 1);
    outer: while (right >= 1) : (right -= 2) {
        if (right == 6) right = 5;
        for (0..n) |k| {
            const row = if (upward) n - 1 - k else k;
            for (0..2) |dx| {
                const col: usize = @intCast(right - @as(isize, @intCast(dx)));
                if (func[row * n + col] != 0) continue;
                var v: u8 = g.at(row, col);
                if (maskHit(fmt.mask, row, col)) v ^= 1;
                cw[bit >> 3] |= v << @intCast(7 - (bit & 7));
                bit += 1;
                if (bit == total * 8) break :outer;
            }
        }
        upward = !upward;
    }
    if (bit < total * 8) return false;

    // De-interleave, correct and concatenate the blocks.
    const nblocks: usize = @as(usize, blk.n1) + blk.n2;
    const necc: usize = blk.ecc;
    const ndata: usize = @as(usize, blk.n1) * blk.d1 + @as(usize, blk.n2) * blk.d2;
    var data_buf: [3706]u8 = undefined;
    const data = data_buf[0..ndata];
    var block: [256]u8 = undefined;
    const min_d: usize = if (blk.n1 > 0) blk.d1 else blk.d2;
    var doff: usize = 0;
    for (0..nblocks) |b| {
        const dlen: usize = if (b < blk.n1) blk.d1 else blk.d2;
        var k: usize = 0;
        for (0..min_d) |i| {
            block[k] = cw[i * nblocks + b];
            k += 1;
        }
        if (dlen > min_d) {
            // The longer blocks' last data codewords follow, one per long block.
            block[k] = cw[min_d * nblocks + (b - blk.n1)];
            k += 1;
        }
        for (0..necc) |i| {
            block[k] = cw[ndata + i * nblocks + b];
            k += 1;
        }
        if (!gf.rsCorrect(block[0..k], necc)) return false;
        @memcpy(data[doff .. doff + dlen], block[0..dlen]);
        doff += dlen;
    }

    info.* = .{ .version = @intCast(version), .level = fmt.level, .mask = fmt.mask, .mode = 0 };
    var p: Parser = .{ .gpa = gpa, .r = .{ .data = data }, .version = version, .out = out, .info = info };
    p.symbols() catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return false,
    };
    return true;
}

/// Decodes an n x n module grid (`modules[r * n + c] & 1` = dark), trying
/// its transpose (a mirrored symbol) when the grid fails, as OpenCV does.
/// The decoded bytes go to `out`; null when the symbol does not decode or
/// OpenCV would reject its contents.
pub fn decode(gpa: Allocator, modules: []const u8, n: usize, out: *std.ArrayList(u8)) Allocator.Error!?Info {
    if (n < 21 or n > 177 or (n - 21) % 4 != 0 or modules.len < n * n) return null;
    const version = (n - 21) / 4 + 1;
    var info: Info = undefined;
    out.clearRetainingCapacity();
    var ok = try run(gpa, .{ .d = modules, .n = n, .transposed = false }, version, out, &info);
    if (!ok) {
        out.clearRetainingCapacity();
        ok = try run(gpa, .{ .d = modules, .n = n, .transposed = true }, version, out, &info);
    }
    if (!ok) {
        out.clearRetainingCapacity();
        return null;
    }
    // QRDecode::decodingProcess: the returned string by first mode.
    switch (info.mode) {
        1, 2, 3, 7, 8 => {},
        4 => {
            if (info.eci == 26) {
                if (!isUtf8(out.items)) return null;
            } else if (info.eci == 25) {
                return null;
            } else if (!isAscii(out.items) and !isUtf8(out.items)) {
                // Latin-1 to UTF-8.
                var extra: usize = 0;
                for (out.items) |c| extra += @intFromBool(c >= 0x80);
                const old = out.items.len;
                try out.resize(gpa, old + extra);
                var w = old + extra;
                var i = old;
                while (i > 0) {
                    i -= 1;
                    const c = out.items[i];
                    if (c >= 0x80) {
                        w -= 2;
                        out.items[w] = 0xc0 | (c >> 6);
                        out.items[w + 1] = 0x80 | (c & 0x3f);
                    } else {
                        w -= 1;
                        out.items[w] = c;
                    }
                }
            }
        },
        else => {
            out.clearRetainingCapacity();
            return null;
        },
    }
    return info;
}

test "decode: round trip through the encoder" {
    const gpa = std.testing.allocator;
    const enc = @import("qr_encode.zig");
    var long: [300]u8 = undefined;
    for (&long, 0..) |*c, i| c.* = @intCast('a' + i % 26);
    const texts = [_][]const u8{ "0123456789", "HELLO WORLD $%*", "Hello, world! \xc3\xa9", &long };
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    for (texts) |t| {
        for ([_]Level{ .L, .M, .Q, .H }) |lvl| {
            var img = try enc.encode(gpa, t, .{ .level = lvl });
            defer img.deinit(gpa);
            const n = img.width - 4;
            const mods = try gpa.alloc(u8, n * n);
            defer gpa.free(mods);
            for (0..n) |r| for (0..n) |c| {
                mods[r * n + c] = @intFromBool(img.data[(r + 2) * img.stride + c + 2] == 0);
            };
            // A few module errors.
            mods[n * 10 + 12] ^= 1;
            mods[n * (n - 3) + n - 2] ^= 1;
            const info = (try decode(gpa, mods, n, &out)).?;
            try std.testing.expectEqual(lvl, info.level);
            try std.testing.expectEqualStrings(t, out.items);
        }
    }
    // Garbage never panics.
    var prng = std.Random.DefaultPrng.init(7);
    const junk = try gpa.alloc(u8, 177 * 177);
    defer gpa.free(junk);
    for (0..200) |k| {
        prng.random().bytes(junk);
        const n = 21 + 4 * (k % 40);
        _ = try decode(gpa, junk, n, &out);
    }
}
