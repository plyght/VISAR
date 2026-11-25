//! QR code generation, OpenCV's QRCodeEncoder: the output is bit-identical
//! to cv2.QRCodeEncoder.encode (versions 1-40, levels L/M/Q/H, numeric,
//! alphanumeric, byte, ECI (UTF-8) and kanji modes, structured append).
//! The steps follow OpenCV's implementation, including its version choice
//! (estimated from the automatic mode's bit count), its reading of the
//! ISO/IEC 18004 mask penalties and its quiet zone of two modules.

const std = @import("std");
const Allocator = std.mem.Allocator;
const core = @import("../core/core.zig");
const Image = core.Image;
const tables = @import("qr_tables.zig");
const gf = @import("gf256.zig");

pub const Level = enum(u2) { L = 0, M = 1, Q = 2, H = 3 };

/// OpenCV's QRCodeEncoder::EncodeMode values.
pub const Mode = enum(i8) {
    auto = -1,
    numeric = 1,
    alphanumeric = 2,
    structured_append = 3,
    byte = 4,
    eci = 7,
    kanji = 8,
};

pub const Options = struct {
    /// 0 picks the smallest version that holds the text.
    version: u8 = 0,
    level: Level = .L,
    mode: Mode = .auto,
};

pub const Error = Allocator.Error || core.Error || error{
    /// The text does not fit the chosen version, or any version.
    DataTooLong,
    /// The text has characters the chosen mode cannot encode, or the
    /// options are out of range.
    InvalidArgument,
};

const invalid: u8 = 110;
const max_version = 40;

/// A square module matrix (0 = dark, 255 = light) as OpenCV's Mats.
const Grid = struct {
    n: usize,
    d: []u8,

    fn at(g: Grid, r: usize, c: usize) *u8 {
        return &g.d[r * g.n + c];
    }
};

fn mapSymbol(c: u8) i32 {
    if (c >= '0' and c <= '9') return c - '0';
    if (c >= 'A' and c <= 'Z') return @as(i32, c - 'A') + 10;
    return switch (c) {
        ' ' => 36,
        '$' => 37,
        '%' => 38,
        '*' => 39,
        '+' => 40,
        '-' => 41,
        '.' => 42,
        '/' => 43,
        ':' => 44,
        else => -1,
    };
}

pub fn isNumeric(s: []const u8) bool {
    for (s) |c| if (c < '0' or c > '9') return false;
    return true;
}

pub fn isAlphanumeric(s: []const u8) bool {
    for (s) |c| if (mapSymbol(c) < 0) return false;
    return true;
}

fn autoMode(s: []const u8) Mode {
    if (isNumeric(s)) return .numeric;
    if (isAlphanumeric(s)) return .alphanumeric;
    return .byte;
}

const Bits = std.ArrayList(u8);

const Encoder = struct {
    gpa: Allocator,
    /// The version used for the length fields (OpenCV's version_level).
    version: u32,
    level: Level,
    mode: Mode,
    sequence_num: u8 = 0,
    total_num: u8 = 0,
    parity: u8 = 0,

    fn put(e: *Encoder, out: *Bits, value: u32, nbits: u32) Allocator.Error!void {
        var i: u32 = nbits;
        while (i > 0) {
            i -= 1;
            try out.append(e.gpa, @intCast((value >> @intCast(i)) & 1));
        }
    }

    fn encodeNumeric(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 1, 4);
        const len_bits: u32 = if (e.version >= 27) 14 else if (e.version >= 10) 12 else 10;
        try e.put(out, @truncate(s.len), len_bits);
        var i: usize = 0;
        while (i + 3 <= s.len) : (i += 3) {
            if (!isNumeric(s[i .. i + 3])) return false;
            try e.put(out, @as(u32, s[i] - '0') * 100 + @as(u32, s[i + 1] - '0') * 10 + (s[i + 2] - '0'), 10);
        }
        if (i + 2 == s.len) {
            if (!isNumeric(s[i..])) return false;
            try e.put(out, @as(u32, s[i] - '0') * 10 + (s[i + 1] - '0'), 7);
        } else if (i + 1 == s.len) {
            if (!isNumeric(s[i..])) return false;
            try e.put(out, s[i] - '0', 4);
        }
        return true;
    }

    fn encodeAlpha(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 2, 4);
        const len_bits: u32 = if (e.version < 10) 9 else if (e.version < 27) 11 else 13;
        try e.put(out, @truncate(s.len), len_bits);
        var i: usize = 0;
        while (i + 1 < s.len) : (i += 2) {
            const a = mapSymbol(s[i]);
            const b = mapSymbol(s[i + 1]);
            if (a < 0 or b < 0) return false;
            try e.put(out, @intCast(a * 45 + b), 11);
        }
        if (s.len % 2 != 0) {
            const a = mapSymbol(s[s.len - 1]);
            if (a < 0) return false;
            try e.put(out, @intCast(a), 6);
        }
        return true;
    }

    fn encodeByte(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 4, 4);
        try e.put(out, @truncate(s.len), if (e.version > 9) 16 else 8);
        for (s) |c| try e.put(out, c, 8);
        return true;
    }

    fn encodeEci(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 7, 4);
        // ECI 26 (UTF-8) fits one codeword: a 0 bit and seven value bits.
        try e.put(out, 0, 1);
        try e.put(out, 26, 7);
        return e.encodeByte(s, out);
    }

    fn encodeKanji(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 8, 4);
        // OpenCV's length field: 10 bits from version 10 on (its 12-bit
        // branch for versions 27+ is unreachable).
        try e.put(out, @truncate(s.len / 2), if (e.version >= 10) 10 else 8);
        var i: usize = 0;
        while (i < (s.len / 2) * 2) : (i += 2) {
            var ch: u16 = (@as(u16, s[i]) << 8) + s[i + 1];
            if (ch >= 0x8140 and ch <= 0x9FFC) {
                ch -= 0x8140;
            } else if (ch >= 0xE040 and ch <= 0xEBBF) {
                ch -= 0xC140;
            }
            const hi: u16 = ch >> 8;
            const v: u16 = hi *% 0xC0 +% (ch & 0xFF);
            try e.put(out, v, 13);
        }
        return true;
    }

    fn encodeAuto(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!Mode {
        const m = autoMode(s);
        _ = switch (m) {
            .numeric => try e.encodeNumeric(s, out),
            .alphanumeric => try e.encodeAlpha(s, out),
            else => try e.encodeByte(s, out),
        };
        return m;
    }

    fn encodeStructure(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        try e.put(out, 3, 4);
        try e.put(out, e.sequence_num, 4);
        try e.put(out, e.total_num, 4);
        try e.put(out, e.parity, 8);
        _ = try e.encodeAuto(s, out);
        return true;
    }

    fn stringToBits(e: *Encoder, s: []const u8, out: *Bits) Allocator.Error!bool {
        return switch (e.mode) {
            .numeric => e.encodeNumeric(s, out),
            .alphanumeric => e.encodeAlpha(s, out),
            .structured_append => e.encodeStructure(s, out),
            .byte => e.encodeByte(s, out),
            .eci => e.encodeEci(s, out),
            .kanji => e.encodeKanji(s, out),
            .auto => {
                _ = try e.encodeAuto(s, out);
                return true;
            },
        };
    }

    fn capacity(e: *const Encoder, version: usize, mode: Mode) u32 {
        const col: usize = switch (mode) {
            .numeric => 0,
            .alphanumeric => 1,
            .byte => 2,
            else => 3,
        };
        return tables.capacity[version][@backingInt(e.level)][col];
    }

    /// OpenCV's versionAuto: -1 when the text fits no version.
    fn versionAuto(e: *Encoder, s: []const u8) Allocator.Error!i32 {
        var tmp: Bits = .empty;
        defer tmp.deinit(e.gpa);
        const mode = try e.encodeAuto(s, &tmp);
        if (s.len > e.capacity(max_version, mode)) return -1;
        var v: usize = max_version;
        while (v > 0) : (v -= 1) {
            if (s.len > e.capacity(v, mode)) break;
        }
        if (v < max_version) v += 1;
        var nbits = tmp.items.len;
        if (e.mode == .structured_append) nbits += 4 + 4 + 8 + 4;
        const lvl = @backingInt(e.level);
        for ([2]usize{ v, v + 1 }, 0..) |cand, k| {
            if (k == 1 and v >= max_version) break;
            const b = tables.versions[cand].ecc[lvl];
            const data: usize = @as(usize, b.d1) * b.n1 + @as(usize, b.d2) * b.n2;
            if (data * 8 >= nbits) return @intCast(cand);
        }
        return -1;
    }
};

/// One QR symbol of `version` from the bit stream (OpenCV's
/// generatingProcess after stringToBits), as an image with a two-module
/// quiet zone.
fn generate(gpa: Allocator, version: u32, level: Level, payload: *Bits) Error!Image(u8) {
    const vi = &tables.versions[version];
    const p = vi.ecc[@backingInt(level)];
    const nblocks: usize = @as(usize, p.n1) + p.n2;
    const ec: usize = p.ecc;

    // padBitStream
    const total_data: usize = (@as(usize, vi.total) - ec * nblocks) * 8;
    if (payload.items.len < total_data) {
        const pad = total_data - payload.items.len;
        if (pad <= 4) {
            try payload.appendNTimes(gpa, 0, pad);
        } else {
            try payload.appendNTimes(gpa, 0, 4);
            const r = payload.items.len % 8;
            if (r != 0) try payload.appendNTimes(gpa, 0, 8 - r);
            if (payload.items.len < total_data) {
                const num = (total_data - payload.items.len) / 8;
                for (0..num) |j| {
                    const pat: u8 = if (j % 2 == 0) 236 else 17;
                    var b: u32 = 8;
                    while (b > 0) {
                        b -= 1;
                        try payload.append(gpa, (pat >> @intCast(b)) & 1);
                    }
                }
            }
        }
    }

    // eccGenerate + rearrangeBlocks: the data and Reed-Solomon codewords
    // of each block, interleaved column by column.
    const gen = gf.generator(ec);
    const data = try gpa.alloc(u8, total_data / 8);
    defer gpa.free(data);
    for (data, 0..) |*d, i| {
        var v: u8 = 0;
        for (0..8) |k| v = (v << 1) | payload.items[i * 8 + k];
        d.* = v;
    }
    const eccs = try gpa.alloc(u8, nblocks * ec);
    defer gpa.free(eccs);
    var off: usize = 0;
    for (0..nblocks) |bi| {
        const len: usize = if (bi < p.n1) p.d1 else p.d2;
        gf.rsRemainder(data[off .. off + len], gen[0 .. ec + 1], eccs[bi * ec .. (bi + 1) * ec]);
        off += len;
    }
    const cw = try gpa.alloc(u8, vi.total);
    defer gpa.free(cw);
    {
        var k: usize = 0;
        const max_d: usize = @max(p.d1, p.d2);
        for (0..max_d) |col| {
            var boff: usize = 0;
            for (0..nblocks) |bi| {
                const len: usize = if (bi < p.n1) p.d1 else p.d2;
                if (col < len) {
                    cw[k] = data[boff + col];
                    k += 1;
                }
                boff += len;
            }
        }
        for (0..ec) |col| {
            for (0..nblocks) |bi| {
                cw[k] = eccs[bi * ec + col];
                k += 1;
            }
        }
        std.debug.assert(k == vi.total);
    }

    const n: usize = 21 + (version - 1) * 4;
    const original: Grid = .{ .n = n, .d = try gpa.alloc(u8, n * n) };
    defer gpa.free(original.d);
    const masked: Grid = .{ .n = n, .d = try gpa.alloc(u8, n * n) };
    defer gpa.free(masked.d);
    @memset(original.d, 255);
    @memset(masked.d, 255);
    writeReservedArea(original, masked, version, vi);
    writeData(original, masked, cw);

    // findAutoMaskType: during the search the version blocks are dark
    // (OpenCV fills them from a not-yet-computed array).
    const test_grid: Grid = .{ .n = n, .d = try gpa.alloc(u8, n * n) };
    defer gpa.free(test_grid.d);
    const tposed: Grid = .{ .n = n, .d = try gpa.alloc(u8, n * n) };
    defer gpa.free(tposed.d);
    const dark_version: [18]u8 = @splat(1);
    var best: u3 = 0;
    var lowest: i64 = std.math.maxInt(i64);
    for (0..8) |mt| {
        @memcpy(test_grid.d, masked.d);
        maskData(original, @intCast(mt), test_grid);
        fillReserved(test_grid, formatBits(level, @intCast(mt)), version, dark_version);
        for (0..n) |r| for (0..n) |c| {
            tposed.at(c, r).* = test_grid.at(r, c).*;
        };
        const pen = penalty(test_grid) + penalty(tposed) + block2x2(test_grid);
        var dark: usize = 0;
        for (test_grid.d) |v| dark += @intFromBool(v == 0);
        const pct: i64 = @intCast(dark * 100 / (n * n));
        const diff: i64 = @intCast(@min(@abs(pct - 45), @abs(pct - 55)));
        const total = pen + @divTrunc(diff, 5) * 10;
        if (total < lowest) {
            lowest = total;
            best = @intCast(mt);
        }
    }
    maskData(original, best, masked);
    fillReserved(masked, formatBits(level, best), version, versionBits(version));

    var img = try Image(u8).initFill(gpa, @intCast(n + 4), @intCast(n + 4), 1, 255);
    for (0..n) |r| @memcpy(img.data[(r + 2) * img.stride + 2 ..][0..n], masked.d[r * n ..][0..n]);
    return img;
}

fn writeReservedArea(original: Grid, masked: Grid, version: u32, vi: *const tables.Version) void {
    const n: i32 = @intCast(original.n);
    const loc = [2]i32{ 3, n - 1 - 3 };
    for (0..2) |a| for (0..2) |b| {
        if (a == 1 and b == 1) continue;
        const x = loc[a];
        const y = loc[b];
        var i: i32 = -5;
        while (i <= 5) : (i += 1) {
            var j: i32 = -5;
            while (j <= 5) : (j += 1) {
                if (x + i < 0 or x + i >= n or y + j < 0 or y + j >= n) continue;
                const ai = @abs(i);
                const aj = @abs(j);
                const r: usize = @intCast(x + i);
                const c: usize = @intCast(y + j);
                if (!(aj == 2 and ai <= 2) and !(aj <= 2 and ai == 2) and ai != 4 and aj != 4)
                    masked.at(r, c).* = 0;
                if ((y == loc[1] and j == -5) or (x == loc[1] and i == -5)) continue;
                original.at(r, c).* = invalid;
            }
        }
    };
    // The dark module.
    {
        const r: usize = @intCast(loc[1] - 4);
        const c: usize = @intCast(loc[0] + 5);
        masked.at(r, c).* = 0;
        original.at(r, c).* = invalid;
    }
    const un: usize = original.n;
    if (version >= 7) {
        for (0..6) |i| for (un - 11..un - 7) |j| {
            original.at(i, j).* = invalid;
            original.at(j, i).* = invalid;
        };
    }
    // Timing patterns.
    for (0..un) |i| for (0..un) |j| {
        if (original.at(i, j).* == invalid) continue;
        if (i == 6 or j == 6) {
            original.at(i, j).* = invalid;
            const ii: i64 = @intCast(i);
            const jj: i64 = @intCast(j);
            if (!(i == 6 and @rem(jj - 7, 2) == 0) and !(j == 6 and @rem(ii - 7, 2) == 0))
                masked.at(i, j).* = 0;
        }
    };
    // Alignment patterns, except where they overlap a finder region.
    const finder = [3][4]i32{ .{ 0, 0, 9, 9 }, .{ 0, n - 8, 9, n }, .{ n - 8, 0, n, 9 } };
    for (vi.align_) |xa| {
        if (xa == 0) break;
        for (vi.align_) |ya| {
            if (ya == 0) break;
            const x: i32 = xa;
            const y: i32 = ya;
            var in_finder = false;
            for (finder) |f| {
                if (x >= f[0] and x <= f[2] and y >= f[1] and y <= f[3]) {
                    in_finder = true;
                    break;
                }
            }
            if (in_finder) continue;
            var i: i32 = -2;
            while (i <= 2) : (i += 1) {
                var j: i32 = -2;
                while (j <= 2) : (j += 1) {
                    const r: usize = @intCast(x + i);
                    const c: usize = @intCast(y + j);
                    original.at(r, c).* = invalid;
                    if ((i == 0 and j == 0) or @abs(j) == 2 or @abs(i) == 2) masked.at(r, c).* = 0;
                }
            }
        }
    }
}

fn writeData(original: Grid, masked: Grid, cw: []const u8) void {
    const n: i32 = @intCast(original.n);
    var y: i32 = n - 1;
    var x: i32 = n - 1;
    var dir: i32 = -1;
    var count: usize = 0;
    var value: u8 = cw[0];
    const limit = cw.len * 8;
    outer: while (x > 0) {
        if (x == 6) x -= 1;
        for (0..2) |i| {
            const bit_one = (value & (@as(u8, 0x80) >> @intCast(count % 8))) != 0;
            const r: usize = @intCast(y);
            const c: usize = @intCast(x - @as(i32, @intCast(i)));
            if (original.at(r, c).* == invalid) continue;
            const v: u8 = if (bit_one) 0 else 255;
            original.at(r, c).* = v;
            masked.at(r, c).* = v;
            count += 1;
            if (count == limit) break :outer;
            if (count % 8 == 0) value = cw[count / 8];
        }
        y += dir;
        if (y < 0 or y >= n) {
            dir = -dir;
            x -= 2;
            y += dir;
        }
    }
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

/// XORs the data modules of `original` selected by mask `mt` into `masked`.
fn maskData(original: Grid, mt: u3, masked: Grid) void {
    const n = original.n;
    for (0..n) |i| for (0..n) |j| {
        const v = original.at(i, j).*;
        if (v == invalid) continue;
        if (maskHit(mt, i, j)) masked.at(i, j).* = v ^ 255;
    };
}

/// The 15 format bits, bit i = 1 for a dark module.
fn formatBits(level: Level, mt: u3) [15]u8 {
    const code: u16 = switch (level) {
        .L => 1,
        .M => 0,
        .Q => 3,
        .H => 2,
    };
    const w = tables.format_lut[(code << 3) | mt];
    var out: [15]u8 = undefined;
    for (&out, 0..) |*b, i| b.* = @intCast((w >> @intCast(i)) & 1);
    return out;
}

/// The 18 version bits (BCH(18,6)), bit k = coefficient of x^k.
fn versionBits(version: u32) [18]u8 {
    var rem: u32 = version << 12;
    var k: u5 = 17;
    while (k >= 12) : (k -= 1) {
        if ((rem >> k) & 1 != 0) rem ^= @as(u32, 0x1F25) << (k - 12);
    }
    const word = (version << 12) | rem;
    var out: [18]u8 = undefined;
    for (&out, 0..) |*b, i| b.* = @intCast((word >> @intCast(i)) & 1);
    return out;
}

fn fillReserved(g: Grid, fmt: [15]u8, version: u32, ver: [18]u8) void {
    const n = g.n;
    for (0..7) |i| g.at(n - 1 - i, 8).* = if (fmt[14 - i] == 0) 255 else 0;
    for (0..8) |i| g.at(8, n - 8 + i).* = if (fmt[14 - (7 + i)] == 0) 255 else 0;
    const xs = [15]usize{ 8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0 };
    const ys = [15]usize{ 0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8 };
    var i: usize = 15;
    while (i > 0) {
        i -= 1;
        g.at(ys[i], xs[i]).* = if (fmt[i] == 0) 255 else 0;
    }
    if (version >= 7) {
        for (0..18) |j| {
            const a = 5 - j / 3;
            const b = n - 9 - j % 3;
            const v: u8 = if (ver[17 - j] == 0) 255 else 0;
            g.at(b, a).* = v; // bottom-left block: (row b, col a)
            g.at(a, b).* = v; // top-right block
        }
    }
}

/// OpenCV's rules 1 and 3 along the rows of `g`.
fn penalty(g: Grid) i64 {
    const n = g.n;
    var p: i64 = 0;
    for (0..n) |i| {
        const row = g.d[i * n ..][0..n];
        var color = row[0];
        var run: i64 = 1;
        for (1..n) |j| {
            if (row[j] == color) run += 1;
            if (row[j] != color or j + 1 == n) {
                color = row[j];
                if (run >= 5) p += 3 + run - 5;
                run = 1;
            }
        }
        const pat0 = [11]u8{ 255, 255, 255, 255, 0, 255, 0, 0, 0, 255, 0 };
        const pat1 = [11]u8{ 0, 255, 0, 0, 0, 255, 0, 255, 255, 255, 255 };
        if (n >= 11) {
            for (0..n - 10) |j| {
                if (std.mem.eql(u8, row[j..][0..11], &pat0)) p += 40;
                if (std.mem.eql(u8, row[j..][0..11], &pat1)) p += 40;
            }
        }
    }
    return p;
}

fn block2x2(g: Grid) i64 {
    const n = g.n;
    var p: i64 = 0;
    for (0..n - 1) |i| for (0..n - 1) |j| {
        const c = g.at(i, j).*;
        if (c == g.at(i, j + 1).* and c == g.at(i + 1, j + 1).* and c == g.at(i + 1, j).*) p += 3;
    };
    return p;
}

/// cv2.QRCodeEncoder.create(params).encode(text): one symbol, 0 = dark,
/// 255 = light, one pixel per module with a two-module quiet zone.
/// Free with `deinit(gpa)`.
pub fn encode(gpa: Allocator, text: []const u8, o: Options) Error!Image(u8) {
    if (o.mode == .structured_append) return error.InvalidArgument;
    const imgs = try encodeImpl(gpa, text, o, 1);
    defer gpa.free(imgs);
    return imgs[0];
}

/// cv2.QRCodeEncoder (MODE_STRUCTURED_APPEND, structure_number = n)
/// .encodeStructuredAppend(text): the text split over n (2..16) symbols.
/// Free each image and then the slice.
pub fn encodeStructuredAppend(gpa: Allocator, text: []const u8, o: Options, n: u32) Error![]Image(u8) {
    var opt = o;
    opt.mode = .structured_append;
    return encodeImpl(gpa, text, opt, n);
}

fn encodeImpl(gpa: Allocator, text: []const u8, o: Options, struct_num_in: u32) Error![]Image(u8) {
    if (o.version > max_version) return error.InvalidArgument;
    if (struct_num_in == 0) return error.InvalidArgument;
    var e: Encoder = .{ .gpa = gpa, .version = o.version, .level = o.level, .mode = o.mode };
    var struct_num = struct_num_in;
    if (struct_num > 1) {
        for (text) |c| e.parity ^= c;
        struct_num = @min(struct_num, 16);
        e.total_num = @intCast(struct_num - 1);
    }
    const out = try gpa.alloc(Image(u8), struct_num);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |*im| im.deinit(gpa);
        gpa.free(out);
    }
    var pos: usize = 0;
    var i: u32 = struct_num;
    while (i > 0) : (i -= 1) {
        e.sequence_num = @intCast(struct_num - i);
        const seg_len = (text.len - pos) / i;
        const seg = text[pos .. pos + seg_len];
        pos += seg_len;
        const detected = try e.versionAuto(seg);
        if (detected < 0) return error.DataTooLong;
        var version: u32 = o.version;
        if (version == 0) {
            version = @intCast(detected);
        } else if (version < detected) return error.DataTooLong;
        var payload: Bits = .empty;
        defer payload.deinit(gpa);
        const saved = e.version;
        e.version = version;
        const ok = try e.stringToBits(seg, &payload);
        e.version = saved;
        if (!ok) return error.InvalidArgument;
        out[made] = try generate(gpa, version, o.level, &payload);
        made += 1;
    }
    return out;
}

test "encode: version 1 numeric" {
    const gpa = std.testing.allocator;
    var img = try encode(gpa, "01234567", .{ .level = .M });
    defer img.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 25), img.width);
    // Finder pattern corner and the quiet zone.
    try std.testing.expectEqual(@as(u8, 0), img.data[2 * img.stride + 2]);
    try std.testing.expectEqual(@as(u8, 255), img.data[0]);
}

test "encode: too long" {
    const gpa = std.testing.allocator;
    const long = try gpa.alloc(u8, 3000);
    defer gpa.free(long);
    @memset(long, 'a');
    try std.testing.expectError(error.DataTooLong, encode(gpa, long, .{ .level = .H }));
    try std.testing.expectError(error.DataTooLong, encode(gpa, "hello world, this is long", .{ .version = 1, .level = .H }));
}
