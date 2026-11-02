//! JPEG decoder: baseline / extended sequential and progressive Huffman
//! JPEG, 8-bit, 1/3/4 components, any integral sampling factors, restart
//! markers. Pixel output is bit-exact with libjpeg-turbo's defaults (ISLOW
//! IDCT, "fancy" upsampling, fixed-point YCbCr->RGB), which is what
//! OpenCV's imread uses.
//!
//! Sequential images whose single scan holds every component are decoded
//! in one streaming pass: each MCU row is entropy-decoded, inverse
//! transformed into a small per-component ring of rows, then the previous
//! MCU row is upsampled and colour converted while it is still in cache.
//! Everything else (progressive, multi-scan sequential) is decoded into a
//! coefficient buffer first and then output through the same row pipeline.

const std = @import("std");
const common = @import("../common.zig");
const tables = @import("tables.zig");
const dct = @import("dct.zig");

pub var dbg_skip: u8 = 0;
const Image = common.Image;
const AnyImage = common.AnyImage;
const DecodeError = common.DecodeError;
const Allocator = std.mem.Allocator;
const natural_order = tables.natural_order;

/// Bits looked up at once by the Huffman decoder.
const fast_bits = 9;
const fast_size = 1 << fast_bits;

// ---------------------------------------------------------------------------
// Huffman tables

const Huff = struct {
    /// (code length << 8) | symbol for every code of at most `fast_bits`
    /// bits, indexed by the next `fast_bits` bits of input; 0 = longer code.
    lut: [fast_size]u16,
    /// AC fast path: for a code plus its magnitude bits fitting in
    /// `fast_bits`: (value << 16) | (run << 8) | total length. Run 64 marks
    /// end-of-block. 0 = take the slow path.
    fast_ac: [fast_size]i32,
    /// Largest code of each length (-1 when none), for lengths 1..16.
    maxcode: [18]i32,
    /// Index into `vals` of a length-l code is code + valoff[l].
    valoff: [17]i32,
    vals: [256]u8,

    fn build(self: *Huff, bits: *const [16]u8, vals: []const u8) DecodeError!void {
        var total: usize = 0;
        for (bits) |b| total += b;
        if (total > 256 or total > vals.len) return error.InvalidData;
        @memset(&self.vals, 0);
        @memcpy(self.vals[0..total], vals[0..total]);
        @memset(&self.lut, 0);
        @memset(&self.fast_ac, 0);
        var code: u32 = 0;
        var idx: u32 = 0;
        for (1..17) |l| {
            const n = bits[l - 1];
            self.valoff[l] = @as(i32, @intCast(idx)) - @as(i32, @intCast(code));
            for (0..n) |_| {
                if (code >= (@as(u32, 1) << @intCast(l))) return error.InvalidData;
                if (l <= fast_bits) {
                    const shift: u5 = @intCast(fast_bits - l);
                    const base = code << shift;
                    const sym = self.vals[idx];
                    for (0..(@as(u32, 1) << shift)) |i| self.lut[base + i] = @intCast((l << 8) | sym);
                }
                idx += 1;
                code += 1;
            }
            self.maxcode[l] = if (n > 0) @as(i32, @intCast(code)) - 1 else -1;
            code <<= 1;
        }
        self.maxcode[0] = -1;
        self.maxcode[17] = std.math.maxInt(i32);
        // AC fast table.
        for (0..fast_size) |p| {
            const e = self.lut[p];
            if (e == 0) continue;
            const len: u32 = e >> 8;
            const rs: u32 = e & 0xff;
            const r = rs >> 4;
            const s = rs & 15;
            if (s == 0) {
                if (r == 0) self.fast_ac[p] = @intCast((64 << 8) | len);
                continue;
            }
            if (len + s > fast_bits) continue;
            const raw: i32 = @intCast((p >> @intCast(fast_bits - len - s)) & ((@as(u32, 1) << @intCast(s)) - 1));
            const v = extend(raw, @intCast(s));
            self.fast_ac[p] = (v << 16) | @as(i32, @intCast((r << 8) | (len + s)));
        }
    }
};

inline fn extend(v: i32, s: u5) i32 {
    return if (v < (@as(i32, 1) << (s - 1))) v - (@as(i32, 1) << s) + 1 else v;
}

// ---------------------------------------------------------------------------
// Bit reader

const BitReader = struct {
    data: []const u8,
    pos: usize,
    /// Unconsumed bits, MSB first; bits below the top `n` are zero.
    acc: u64 = 0,
    n: u32 = 0,
    /// Zero bits appended past a marker or the end of the data.
    fake: u32 = 0,
    /// A marker (or the end of the data) has been reached.
    hit_end: bool = false,

    inline fn refill(br: *BitReader) void {
        if (br.n > 56) return;
        if (!br.hit_end and br.pos + 8 <= br.data.len) {
            const w = std.mem.readInt(u64, br.data[br.pos..][0..8], .big);
            const x = ~w;
            if (((x -% 0x0101010101010101) & ~x & 0x8080808080808080) == 0) {
                const nb = (63 - br.n) >> 3;
                const keep = br.n + nb * 8;
                br.acc |= (w >> @intCast(br.n)) & ~(~@as(u64, 0) >> @intCast(keep));
                br.pos += nb;
                br.n = keep;
                return;
            }
        }
        br.* = refillSlow(br.*);
    }

    fn refillSlow(b: BitReader) BitReader {
        var br = b;
        while (br.n <= 56) {
            var byte: u64 = 0;
            if (!br.hit_end) {
                if (br.pos < br.data.len) {
                    const c = br.data[br.pos];
                    if (c == 0xff) {
                        if (br.pos + 1 < br.data.len and br.data[br.pos + 1] == 0) {
                            byte = 0xff;
                            br.pos += 2;
                        } else {
                            br.hit_end = true;
                        }
                    } else {
                        byte = c;
                        br.pos += 1;
                    }
                } else br.hit_end = true;
            }
            if (br.hit_end) br.fake += 8;
            br.acc |= byte << @intCast(56 - br.n);
            br.n += 8;
        }
        return br;
    }

    inline fn consume(br: *BitReader, k: u32) void {
        br.acc <<= @intCast(k);
        br.n -= k;
    }

    /// Next `s` (1..16) bits as an unsigned value.
    inline fn bits(br: *BitReader, s: u32) i32 {
        if (br.n < s) br.refill();
        const v: i32 = @intCast(br.acc >> @intCast(64 - s));
        br.consume(s);
        return v;
    }

    inline fn bit(br: *BitReader) bool {
        if (br.n < 1) br.refill();
        const v = (br.acc >> 63) != 0;
        br.consume(1);
        return v;
    }

    /// Receives and sign-extends an `s`-bit (1..16) magnitude.
    inline fn receive(br: *BitReader, s: u32) i32 {
        return extend(br.bits(s), @intCast(s));
    }

    /// Decodes one Huffman symbol. Needs at least 16 bits buffered.
    inline fn huff(br: *BitReader, h: *const Huff) u8 {
        const e = h.lut[@intCast(br.acc >> (64 - fast_bits))];
        if (e != 0) {
            br.consume(e >> 8);
            return @truncate(e);
        }
        const r = huffSlow(h, br.acc);
        br.consume(r.len);
        return r.sym;
    }

    /// True when bits beyond the real data have been consumed.
    inline fn overran(br: *const BitReader) bool {
        return br.n < br.fake;
    }
};

const SlowResult = struct { sym: u8, len: u32 };

fn huffSlow(h: *const Huff, acc: u64) SlowResult {
    const code16: i32 = @intCast(acc >> 48);
    var l: u32 = fast_bits + 1;
    while (l <= 16) : (l += 1) {
        const c = code16 >> @intCast(16 - l);
        if (c <= h.maxcode[l]) {
            const idx: usize = @intCast((c + h.valoff[l]) & 0xff);
            return .{ .sym = h.vals[idx], .len = l };
        }
    }
    // Bad code: libjpeg warns and returns symbol 0.
    return .{ .sym = 0, .len = 16 };
}

// ---------------------------------------------------------------------------
// Decoder state

const ColorSpace = enum { gray, ycc, rgb, cmyk, ycck };

const Upsample = enum { full, h2v1, h1v2, h2v2, box };

const Comp = struct {
    id: u8 = 0,
    h: u8 = 1,
    v: u8 = 1,
    tq: u8 = 0,
    td: u8 = 0,
    ta: u8 = 0,
    /// Blocks per row / column, including MCU padding.
    bw: u32 = 0,
    bh: u32 = 0,
    /// Size of the (downsampled) component in samples.
    dw: u32 = 0,
    dh: u32 = 0,
    pred: i32 = 0,
    latched: bool = false,
    /// Dequantization table in zigzag order (padded) and natural order.
    qz: [80]i32 = undefined,
    qn: [64]i32 = undefined,
    coefs: []i16 = &.{},
    // Output pipeline.
    needed: bool = false,
    hr: u8 = 1,
    vr: u8 = 1,
    method: Upsample = .full,
    ring: []u8 = &.{},
    ring_stride: usize = 0,
    ring_h: u32 = 0,
    tmp: []u8 = &.{},
};

const Scan = struct {
    ns: u8,
    cidx: [4]u8,
    ss: u8,
    se: u8,
    ah: u8,
    al: u8,
};

const Decoder = struct {
    arena: Allocator,
    gpa: Allocator,
    data: []const u8,
    pos: usize = 0,
    opts: common.DecodeOptions,

    qt: [4][64]u16 = undefined,
    qt_set: [4]bool = .{ false, false, false, false },
    dc_tab: [4]Huff = undefined,
    ac_tab: [4]Huff = undefined,
    restart_interval: u32 = 0,

    have_frame: bool = false,
    progressive: bool = false,
    width: u32 = 0,
    height: u32 = 0,
    ncomp: u8 = 0,
    comps: [4]Comp = .{ .{}, .{}, .{}, .{} },
    hmax: u8 = 1,
    vmax: u8 = 1,
    mcux: u32 = 0,
    mcuy: u32 = 0,

    jfif: bool = false,
    adobe: bool = false,
    adobe_transform: u8 = 0,

    streamed: bool = false,
    buffered: bool = false,
    scans: u32 = 0,
    eobrun: u32 = 0,

    img: ?Image(u8) = null,
    cs: ColorSpace = .gray,
    out_ch: u16 = 1,
    out_bgr: bool = false,
    pipeline_ready: bool = false,
    colsum: []i16 = &.{},

    fn readMarkerSegment(d: *Decoder) DecodeError![]const u8 {
        if (d.pos + 2 > d.data.len) return error.Truncated;
        const len = std.mem.readInt(u16, d.data[d.pos..][0..2], .big);
        if (len < 2) return error.InvalidData;
        if (d.pos + len > d.data.len) return error.Truncated;
        const seg = d.data[d.pos + 2 .. d.pos + len];
        d.pos += len;
        return seg;
    }

    /// Finds the next marker at or after `pos`; returns its code (pos is
    /// left after it), or null at the end of the data.
    fn nextMarker(d: *Decoder) ?u8 {
        while (d.pos + 1 < d.data.len) {
            if (d.data[d.pos] != 0xff) {
                d.pos += 1;
                continue;
            }
            const c = d.data[d.pos + 1];
            if (c == 0xff) {
                d.pos += 1;
                continue;
            }
            d.pos += 2;
            if (c == 0) continue;
            return c;
        }
        d.pos = d.data.len;
        return null;
    }

    fn run(d: *Decoder) DecodeError!void {
        if (d.data.len < 2 or d.data[0] != 0xff or d.data[1] != 0xd8) return error.InvalidData;
        d.pos = 2;
        // The standard tables stand in for undefined ones (Motion-JPEG).
        try d.dc_tab[0].build(&tables.std_dc_luma.bits, tables.std_dc_luma.vals);
        try d.dc_tab[1].build(&tables.std_dc_chroma.bits, tables.std_dc_chroma.vals);
        try d.ac_tab[0].build(&tables.std_ac_luma.bits, tables.std_ac_luma.vals);
        try d.ac_tab[1].build(&tables.std_ac_chroma.bits, tables.std_ac_chroma.vals);
        d.dc_tab[2] = d.dc_tab[0];
        d.dc_tab[3] = d.dc_tab[1];
        d.ac_tab[2] = d.ac_tab[0];
        d.ac_tab[3] = d.ac_tab[1];

        while (true) {
            const m = d.nextMarker() orelse break;
            switch (m) {
                0xd9 => break, // EOI
                0xd8, 0xd0...0xd7, 0x01 => {},
                0xc0, 0xc1, 0xc2 => try d.readFrame(d.readMarkerSegment() catch |e| return d.lateError(e), m == 0xc2),
                0xc3, 0xc5...0xc7, 0xc9...0xcb, 0xcd...0xcf => return error.Unsupported,
                0xc4 => try d.readDht(d.readMarkerSegment() catch |e| return d.lateError(e)),
                0xdb => try d.readDqt(d.readMarkerSegment() catch |e| return d.lateError(e)),
                0xdd => {
                    const seg = d.readMarkerSegment() catch |e| return d.lateError(e);
                    if (seg.len < 2) return error.InvalidData;
                    d.restart_interval = std.mem.readInt(u16, seg[0..2], .big);
                },
                0xda => {
                    const seg = d.readMarkerSegment() catch |e| return d.lateError(e);
                    try d.readScan(seg);
                    if (d.streamed) break;
                },
                0xe0 => {
                    const seg = d.readMarkerSegment() catch |e| return d.lateError(e);
                    if (seg.len >= 5 and std.mem.eql(u8, seg[0..5], "JFIF\x00")) d.jfif = true;
                },
                0xee => {
                    const seg = d.readMarkerSegment() catch |e| return d.lateError(e);
                    if (seg.len >= 12 and std.mem.eql(u8, seg[0..5], "Adobe")) {
                        d.adobe = true;
                        d.adobe_transform = seg[11];
                    }
                },
                else => _ = d.readMarkerSegment() catch |e| return d.lateError(e),
            }
        }
        if (d.scans == 0) return if (d.have_frame) error.Truncated else error.InvalidData;
        if (d.buffered) try d.outputBuffered();
    }

    /// Running out of data after the first scan still yields an image.
    fn lateError(d: *Decoder, e: DecodeError) DecodeError!void {
        if (d.scans > 0 and e == error.Truncated) {
            d.pos = d.data.len;
            return;
        }
        return e;
    }

    fn readDqt(d: *Decoder, seg: []const u8) DecodeError!void {
        var i: usize = 0;
        while (i < seg.len) {
            const pq = seg[i] >> 4;
            const tq = seg[i] & 15;
            i += 1;
            if (tq > 3 or pq > 1) return error.InvalidData;
            const need: usize = if (pq == 1) 128 else 64;
            if (i + need > seg.len) return error.InvalidData;
            for (0..64) |k| {
                const v: u16 = if (pq == 1) std.mem.readInt(u16, seg[i + 2 * k ..][0..2], .big) else seg[i + k];
                d.qt[tq][natural_order[k]] = v;
            }
            d.qt_set[tq] = true;
            i += need;
        }
    }

    fn readDht(d: *Decoder, seg: []const u8) DecodeError!void {
        var i: usize = 0;
        while (i < seg.len) {
            if (i + 17 > seg.len) return error.InvalidData;
            const tc = seg[i] >> 4;
            const th = seg[i] & 15;
            if (tc > 1 or th > 3) return error.InvalidData;
            const bits: *const [16]u8 = seg[i + 1 ..][0..16];
            var total: usize = 0;
            for (bits) |b| total += b;
            i += 17;
            if (total > 256 or i + total > seg.len) return error.InvalidData;
            const t = if (tc == 0) &d.dc_tab[th] else &d.ac_tab[th];
            try t.build(bits, seg[i .. i + total]);
            i += total;
        }
    }

    fn readFrame(d: *Decoder, seg: []const u8, progressive: bool) DecodeError!void {
        if (d.have_frame) return error.InvalidData;
        if (seg.len < 6) return error.InvalidData;
        if (seg[0] != 8) return error.Unsupported; // 12-bit
        const h = std.mem.readInt(u16, seg[1..3], .big);
        const w = std.mem.readInt(u16, seg[3..5], .big);
        const nc = seg[5];
        if (h == 0) return error.Unsupported; // DNL
        if (w == 0) return error.InvalidData;
        if (nc != 1 and nc != 3 and nc != 4) return error.Unsupported;
        if (seg.len < 6 + 3 * @as(usize, nc)) return error.InvalidData;
        if (@as(u64, w) * h > d.opts.max_pixels) return error.TooLarge;
        d.width = w;
        d.height = h;
        d.ncomp = nc;
        d.progressive = progressive;
        for (0..nc) |i| {
            const p = seg[6 + 3 * i ..][0..3];
            const c = &d.comps[i];
            c.id = p[0];
            c.h = p[1] >> 4;
            c.v = p[1] & 15;
            c.tq = p[2];
            if (c.h < 1 or c.h > 4 or c.v < 1 or c.v > 4 or c.tq > 3) return error.InvalidData;
            for (0..i) |j| if (d.comps[j].id == c.id) return error.InvalidData;
            if (nc == 1) {
                c.h = 1;
                c.v = 1;
            }
            d.hmax = @max(d.hmax, c.h);
            d.vmax = @max(d.vmax, c.v);
        }
        d.mcux = ceilDiv(w, 8 * @as(u32, d.hmax));
        d.mcuy = ceilDiv(h, 8 * @as(u32, d.vmax));
        for (d.comps[0..nc]) |*c| {
            if (d.hmax % c.h != 0 or d.vmax % c.v != 0) return error.Unsupported;
            c.bw = d.mcux * c.h;
            c.bh = d.mcuy * c.v;
            c.dw = ceilDiv(@as(u32, w) * c.h, d.hmax);
            c.dh = ceilDiv(@as(u32, h) * c.v, d.vmax);
        }
        d.have_frame = true;
    }

    fn readScan(d: *Decoder, seg: []const u8) DecodeError!void {
        if (!d.have_frame) return error.InvalidData;
        if (seg.len < 1) return error.InvalidData;
        const ns = seg[0];
        if (ns < 1 or ns > 4 or seg.len < 4 + 2 * @as(usize, ns)) return error.InvalidData;
        var sc: Scan = .{ .ns = ns, .cidx = undefined, .ss = 0, .se = 0, .ah = 0, .al = 0 };
        var blocks: u32 = 0;
        for (0..ns) |i| {
            const cid = seg[1 + 2 * i];
            const t = seg[2 + 2 * i];
            const ci = for (d.comps[0..d.ncomp], 0..) |c, j| {
                if (c.id == cid) break j;
            } else return error.InvalidData;
            for (0..i) |j| if (sc.cidx[j] == ci) return error.InvalidData;
            sc.cidx[i] = @intCast(ci);
            const c = &d.comps[ci];
            c.td = t >> 4;
            c.ta = t & 15;
            if (c.td > 3 or c.ta > 3) return error.InvalidData;
            blocks += @as(u32, c.h) * c.v;
        }
        if (ns > 1 and blocks > 10) return error.InvalidData;
        const p = seg[1 + 2 * @as(usize, ns) ..][0..3];
        sc.ss = p[0];
        sc.se = p[1];
        sc.ah = p[2] >> 4;
        sc.al = p[2] & 15;
        if (d.progressive) {
            if (sc.ss == 0) {
                if (sc.se != 0) return error.InvalidData;
            } else {
                if (sc.se < sc.ss or sc.se > 63 or ns != 1) return error.InvalidData;
            }
            if (sc.ah > 13 or sc.al > 13) return error.InvalidData;
        } else {
            sc.ss = 0;
            sc.se = 63;
            sc.ah = 0;
            sc.al = 0;
        }
        // Latch quantization tables on a component's first scan.
        for (sc.cidx[0..ns]) |ci| {
            const c = &d.comps[ci];
            if (!c.latched) {
                if (!d.qt_set[c.tq]) return error.InvalidData;
                const q = &d.qt[c.tq];
                for (0..64) |kk| {
                    c.qn[kk] = q[kk];
                    c.qz[kk] = q[natural_order[kk]];
                }
                for (64..80) |kk| c.qz[kk] = c.qz[63];
                c.latched = true;
            }
        }
        for (sc.cidx[0..ns]) |ci| d.comps[ci].pred = 0;
        d.eobrun = 0;

        if (d.scans == 0 and !d.progressive and ns == d.ncomp) {
            d.scans += 1;
            try d.setupOutput();
            d.streamed = true;
            try d.decodeStreaming(sc);
            return;
        }
        if (!d.buffered) {
            for (d.comps[0..d.ncomp]) |*c| {
                const n = @as(usize, c.bw) * c.bh * 64;
                c.coefs = try d.arena.alloc(i16, n);
                @memset(c.coefs, 0);
            }
            d.buffered = true;
        }
        d.scans += 1;
        try d.decodeBufferedScan(sc);
    }

    // -----------------------------------------------------------------------
    // Output pipeline setup

    fn setupOutput(d: *Decoder) DecodeError!void {
        if (d.pipeline_ready) return;
        d.pipeline_ready = true;
        d.cs = switch (d.ncomp) {
            1 => .gray,
            3 => blk: {
                if (d.jfif) break :blk .ycc;
                if (d.adobe) break :blk if (d.adobe_transform == 0) .rgb else .ycc;
                const c = d.comps;
                if (c[0].id == 'R' and c[1].id == 'G' and c[2].id == 'B') break :blk .rgb;
                break :blk .ycc;
            },
            else => if (d.adobe and d.adobe_transform == 2) .ycck else .cmyk,
        };
        const want_gray = d.opts.mode == .gray;
        d.out_ch = switch (d.cs) {
            .gray => 1,
            .ycc, .rgb => if (want_gray) 1 else 3,
            .cmyk, .ycck => 3,
        };
        d.out_bgr = d.out_ch == 3 and d.opts.order == .bgr;
        for (d.comps[0..d.ncomp], 0..) |*c, i| {
            c.needed = !(want_gray and d.cs == .ycc and i != 0);
        }
        d.img = try Image(u8).init(d.gpa, d.width, d.height, d.out_ch);

        var max_dw: u32 = 0;
        for (d.comps[0..d.ncomp]) |*c| {
            c.hr = d.hmax / c.h;
            c.vr = d.vmax / c.v;
            c.method = if (c.hr == 1 and c.vr == 1)
                .full
            else if (c.hr == 2 and c.vr == 1)
                (if (c.dw > 2) .h2v1 else .box)
            else if (c.hr == 1 and c.vr == 2)
                .h1v2
            else if (c.hr == 2 and c.vr == 2)
                (if (c.dw > 2) .h2v2 else .box)
            else
                .box;
            if (!c.needed) continue;
            c.ring_stride = @as(usize, c.bw) * 8 + 64;
            c.ring_h = 3 * @as(u32, c.v) * 8;
            c.ring = try d.arena.alignedAlloc(u8, .@"64", c.ring_stride * c.ring_h);
            @memset(c.ring, 0);
            if (c.method != .full) {
                c.tmp = try d.arena.alignedAlloc(u8, .@"64", @as(usize, d.mcux) * d.hmax * 8 + 128);
            }
            max_dw = @max(max_dw, c.dw);
        }
        d.colsum = try d.arena.alloc(i16, max_dw + 64);
        @memset(d.colsum, 0);
    }

    inline fn ringRow(c: *const Comp, r: i64) [*]u8 {
        const rr: u32 = @intCast(std.math.clamp(r, 0, @as(i64, c.dh) - 1));
        return c.ring.ptr + (rr % c.ring_h) * c.ring_stride;
    }

    /// The component's samples for output row `y`, upsampled to full width.
    fn compOutRow(d: *Decoder, c: *const Comp, y: u32) [*]const u8 {
        const yi: i64 = y;
        switch (c.method) {
            .full => return ringRow(c, yi),
            .h2v1 => {
                upH2v1(ringRow(c, yi), c.dw, d.colsum, c.tmp.ptr);
                return c.tmp.ptr;
            },
            .h1v2 => {
                const i = yi >> 1;
                const odd = (y & 1) != 0;
                upH1v2(ringRow(c, i), ringRow(c, if (odd) i + 1 else i - 1), c.dw, if (odd) 2 else 1, c.tmp.ptr);
                return c.tmp.ptr;
            },
            .h2v2 => {
                const i = yi >> 1;
                const odd = (y & 1) != 0;
                upH2v2(ringRow(c, i), ringRow(c, if (odd) i + 1 else i - 1), c.dw, d.colsum, c.tmp.ptr);
                return c.tmp.ptr;
            },
            .box => {
                const src = ringRow(c, @divFloor(yi, c.vr));
                if (c.hr == 1) return src;
                const hr = c.hr;
                for (0..c.dw) |x| @memset(c.tmp[x * hr ..][0..hr], src[x]);
                return c.tmp.ptr;
            },
        }
    }

    fn emitRows(d: *Decoder, y0: u32, y1: u32) void {
        const img = d.img.?;
        const w = d.width;
        var y = y0;
        while (y < y1) : (y += 1) {
            const out = img.row(y);
            var src: [4][*]const u8 = undefined;
            for (d.comps[0..d.ncomp], 0..) |*c, i| {
                if (c.needed) src[i] = d.compOutRow(c, y);
            }
            switch (d.cs) {
                .gray => @memcpy(out, src[0][0..w]),
                .ycc => if (d.out_ch == 1)
                    @memcpy(out, src[0][0..w])
                else if (d.out_bgr)
                    yccRow(true, src[0], src[1], src[2], out, w)
                else
                    yccRow(false, src[0], src[1], src[2], out, w),
                .rgb => if (d.out_ch == 1) rgbGrayRow(src[0], src[1], src[2], out, w) else {
                    const o = if (d.out_bgr) [3]usize{ 2, 1, 0 } else [3]usize{ 0, 1, 2 };
                    for (0..w) |x| {
                        out[3 * x + o[0]] = src[0][x];
                        out[3 * x + o[1]] = src[1][x];
                        out[3 * x + o[2]] = src[2][x];
                    }
                },
                .cmyk, .ycck => cmykRow(d.cs == .ycck, d.out_bgr, src, out, w),
            }
        }
    }

    /// Output rows belonging to MCU row `m`.
    fn emitMcuRow(d: *Decoder, m: u32) void {
        const rows = 8 * @as(u32, d.vmax);
        const y0 = m * rows;
        const y1 = @min(y0 + rows, d.height);
        if (dbg_skip & 1 != 0) return;
        if (y0 < y1) d.emitRows(y0, y1);
    }

    // -----------------------------------------------------------------------
    // Sequential, single-scan streaming decode

    fn decodeStreaming(d: *Decoder, sc: Scan) DecodeError!void {
        var br: BitReader = .{ .data = d.data, .pos = d.pos };
        var blk: [64]i32 align(64) = @splat(0);
        const ri = d.restart_interval;
        var since: u32 = 0;
        var dead = false;
        var my: u32 = 0;
        while (my < d.mcuy) : (my += 1) {
            var mx: u32 = 0;
            while (mx < d.mcux) : (mx += 1) {
                if (ri != 0) {
                    if (since == ri) {
                        dead = !restart(&br);
                        for (sc.cidx[0..sc.ns]) |ci| d.comps[ci].pred = 0;
                        since = 0;
                    }
                    since += 1;
                }
                for (sc.cidx[0..sc.ns]) |ci| {
                    const c = &d.comps[ci];
                    const dc_tab = &d.dc_tab[c.td];
                    const ac_tab = &d.ac_tab[c.ta];
                    const ring_row0 = (my % 3) * @as(usize, c.v) * 8;
                    for (0..c.v) |bv| {
                        for (0..c.h) |bh| {
                            var has_ac = false;
                            if (!dead) has_ac = try decodeBlock(&br, dc_tab, ac_tab, &c.qz, &blk, &c.pred);
                            if (c.needed and dbg_skip & 2 == 0) {
                                const dst = c.ring.ptr + (ring_row0 + bv * 8) * c.ring_stride + (@as(usize, mx) * c.h + bh) * 8;
                                if (has_ac) dct.idct(&blk, dst, c.ring_stride) else dct.idctDc(blk[0], dst, c.ring_stride);
                            }
                            if (has_ac) blk = @splat(0) else blk[0] = 0;
                        }
                    }
                }
                if (!dead and br.overran()) dead = true;
            }
            if (my > 0) d.emitMcuRow(my - 1);
        }
        d.emitMcuRow(d.mcuy - 1);
        d.pos = br.pos;
    }

    // -----------------------------------------------------------------------
    // Buffered (progressive / multi-scan) decode

    fn decodeBufferedScan(d: *Decoder, sc: Scan) DecodeError!void {
        var br: BitReader = .{ .data = d.data, .pos = d.pos };
        const ri = d.restart_interval;
        var since: u32 = 0;
        var dead = false;
        const kind: BlockKind = if (!d.progressive) .seq else if (sc.ss == 0)
            (if (sc.ah == 0) .dc_first else .dc_refine)
        else if (sc.ah == 0) .ac_first else .ac_refine;

        if (sc.ns == 1) {
            const c = &d.comps[sc.cidx[0]];
            const nbx = ceilDiv(c.dw, 8);
            const nby = ceilDiv(c.dh, 8);
            for (0..nby) |by| {
                for (0..nbx) |bx| {
                    if (ri != 0) {
                        if (since == ri) {
                            dead = !restart(&br);
                            c.pred = 0;
                            d.eobrun = 0;
                            since = 0;
                        }
                        since += 1;
                    }
                    if (dead) continue;
                    const blk = c.coefs[(by * c.bw + bx) * 64 ..][0..64];
                    try d.decodeCoefBlock(kind, &br, c, sc, blk);
                    if (br.overran()) dead = true;
                }
            }
        } else {
            for (0..d.mcuy) |my| {
                for (0..d.mcux) |mx| {
                    if (ri != 0) {
                        if (since == ri) {
                            dead = !restart(&br);
                            for (sc.cidx[0..sc.ns]) |ci| d.comps[ci].pred = 0;
                            d.eobrun = 0;
                            since = 0;
                        }
                        since += 1;
                    }
                    if (dead) continue;
                    for (sc.cidx[0..sc.ns]) |ci| {
                        const c = &d.comps[ci];
                        for (0..c.v) |bv| {
                            for (0..c.h) |bh| {
                                const by = my * c.v + bv;
                                const bx = mx * c.h + bh;
                                const blk = c.coefs[(by * c.bw + bx) * 64 ..][0..64];
                                try d.decodeCoefBlock(kind, &br, c, sc, blk);
                            }
                        }
                    }
                    if (br.overran()) dead = true;
                }
            }
        }
        d.pos = br.pos;
    }

    const BlockKind = enum { seq, dc_first, dc_refine, ac_first, ac_refine };

    fn decodeCoefBlock(d: *Decoder, kind: BlockKind, br: *BitReader, c: *Comp, sc: Scan, blk: *[64]i16) DecodeError!void {
        switch (kind) {
            .seq => {
                var tmp: [64]i32 = @splat(0);
                _ = try decodeBlock(br, &d.dc_tab[c.td], &d.ac_tab[c.ta], &ones80, &tmp, &c.pred);
                for (blk, tmp) |*o, v| o.* = @truncate(v);
            },
            .dc_first => {
                br.refill();
                const t = br.huff(&d.dc_tab[c.td]);
                var diff: i32 = 0;
                if (t != 0) {
                    if (t > 16) return error.InvalidData;
                    diff = br.receive(t);
                }
                c.pred +%= diff;
                blk[0] = @truncate(c.pred *% (@as(i32, 1) << @intCast(sc.al)));
            },
            .dc_refine => {
                if (br.bit()) blk[0] |= @as(i16, 1) << @intCast(sc.al);
            },
            .ac_first => {
                if (d.eobrun > 0) {
                    d.eobrun -= 1;
                    return;
                }
                const ac = &d.ac_tab[c.ta];
                var k: u32 = sc.ss;
                while (k <= sc.se) : (k += 1) {
                    br.refill();
                    const rs = br.huff(ac);
                    const r: u32 = rs >> 4;
                    const s: u32 = rs & 15;
                    if (s != 0) {
                        k += r;
                        const v = br.receive(s);
                        blk[natural_order[k]] = @truncate(v *% (@as(i32, 1) << @intCast(sc.al)));
                    } else if (r == 15) {
                        k += 15;
                    } else {
                        d.eobrun = @as(u32, 1) << @intCast(r);
                        if (r != 0) d.eobrun += @intCast(br.bits(r));
                        d.eobrun -= 1;
                        break;
                    }
                }
            },
            .ac_refine => {
                const p1: i16 = @as(i16, 1) << @intCast(sc.al);
                const m1: i16 = @as(i16, -1) *% p1;
                var k: u32 = sc.ss;
                if (d.eobrun == 0) {
                    const ac = &d.ac_tab[c.ta];
                    while (k <= sc.se) : (k += 1) {
                        br.refill();
                        const rs = br.huff(ac);
                        var r: i32 = rs >> 4;
                        var s: i16 = 0;
                        if ((rs & 15) != 0) {
                            s = if (br.bit()) p1 else m1;
                        } else if (r != 15) {
                            d.eobrun = @as(u32, 1) << @intCast(r);
                            if (r != 0) d.eobrun += @intCast(br.bits(@intCast(r)));
                            break;
                        }
                        while (k <= sc.se) {
                            const pos = natural_order[k];
                            const cur = blk[pos];
                            if (cur != 0) {
                                if (br.bit()) {
                                    if ((cur & p1) == 0) blk[pos] = if (cur >= 0) cur +% p1 else cur +% m1;
                                }
                            } else {
                                r -= 1;
                                if (r < 0) break;
                            }
                            k += 1;
                        }
                        if (s != 0) blk[natural_order[k]] = s;
                    }
                }
                if (d.eobrun > 0) {
                    while (k <= sc.se) : (k += 1) {
                        const pos = natural_order[k];
                        const cur = blk[pos];
                        if (cur != 0) {
                            if (br.bit()) {
                                if ((cur & p1) == 0) blk[pos] = if (cur >= 0) cur +% p1 else cur +% m1;
                            }
                        }
                    }
                    d.eobrun -= 1;
                }
            },
        }
    }

    fn outputBuffered(d: *Decoder) DecodeError!void {
        try d.setupOutput();
        var blk: [64]i32 align(64) = undefined;
        for (0..d.mcuy) |m| {
            for (d.comps[0..d.ncomp]) |*c| {
                if (!c.needed) continue;
                const ring_row0 = (m % 3) * @as(usize, c.v) * 8;
                for (0..c.v) |bv| {
                    const by = m * c.v + bv;
                    for (0..c.bw) |bx| {
                        const src = c.coefs[(by * c.bw + bx) * 64 ..][0..64];
                        const dst = c.ring.ptr + (ring_row0 + bv * 8) * c.ring_stride + bx * 8;
                        var any: @Vector(8, i16) = @splat(0);
                        inline for (0..8) |i| {
                            var s: @Vector(8, i16) = src[i * 8 ..][0..8].*;
                            const q: @Vector(8, i32) = c.qn[i * 8 ..][0..8].*;
                            blk[i * 8 ..][0..8].* = @as(@Vector(8, i32), s) *% q;
                            if (i == 0) s[0] = 0; // ignore DC
                            any |= s;
                        }
                        if (@reduce(.Or, any) != 0) dct.idct(&blk, dst, c.ring_stride) else dct.idctDc(blk[0], dst, c.ring_stride);
                    }
                }
            }
            if (m > 0) d.emitMcuRow(@intCast(m - 1));
        }
        d.emitMcuRow(d.mcuy - 1);
    }
};

const ones80: [80]i32 = @splat(1);

inline fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// Ends a restart interval: drops buffered bits and consumes the RSTn
/// marker. Returns false when a different marker (or the end of the data)
/// was found, in which case the rest of the scan reads as zeros.
fn restart(br: *BitReader) bool {
    br.acc = 0;
    br.n = 0;
    br.fake = 0;
    br.hit_end = false;
    const data = br.data;
    var p = br.pos;
    while (p + 1 < data.len) {
        if (data[p] != 0xff) {
            p += 1;
            continue;
        }
        const c = data[p + 1];
        if (c == 0xff) {
            p += 1;
            continue;
        }
        if (c == 0) {
            p += 2;
            continue;
        }
        if (c >= 0xd0 and c <= 0xd7) {
            br.pos = p + 2;
            return true;
        }
        br.pos = p;
        br.hit_end = true;
        return false;
    }
    br.pos = data.len;
    br.hit_end = true;
    return false;
}

/// Decodes one sequential block, dequantizing with `q` (zigzag order) into
/// `blk` (natural order), which must be zero on entry. Returns whether any
/// AC coefficient was coded.
inline fn decodeBlock(br: *BitReader, dc: *const Huff, ac: *const Huff, q: *const [80]i32, blk: *[64]i32, pred: *i32) DecodeError!bool {
    if (br.n < 32) br.refill();
    const t = br.huff(dc);
    if (t != 0) {
        if (t > 16) return error.InvalidData;
        pred.* +%= br.receive(t);
    }
    blk[0] = @as(i32, @as(i16, @truncate(pred.*))) *% q[0];
    var k: u32 = 1;
    while (k < 64) {
        if (br.n < 32) br.refill();
        const fe = ac.fast_ac[@intCast(br.acc >> (64 - fast_bits))];
        if (fe != 0) {
            br.consume(@intCast(fe & 0xff));
            const r: u32 = @intCast((fe >> 8) & 0xff);
            if (r == 64) break;
            k += r;
            blk[natural_order[k]] = (fe >> 16) *% q[k];
            k += 1;
            continue;
        }
        const rs = br.huff(ac);
        const r: u32 = rs >> 4;
        const s: u32 = rs & 15;
        if (s == 0) {
            if (r != 15) break;
            k += 16;
            continue;
        }
        k += r;
        blk[natural_order[k]] = br.receive(s) *% q[k];
        k += 1;
    }
    return k > 1;
}

// ---------------------------------------------------------------------------
// Upsampling (libjpeg "fancy" triangular filters) and colour conversion

const V16 = @Vector(16, i16);
const U16 = @Vector(16, u8);

inline fn load16(p: [*]const u8) V16 {
    const b: U16 = p[0..16].*;
    return b;
}

inline fn interleave2(a: U16, b: U16) [32]u8 {
    const mask = comptime blk: {
        var m: [32]i32 = undefined;
        for (0..16) |i| {
            m[2 * i] = i;
            m[2 * i + 1] = ~@as(i32, i);
        }
        break :blk m;
    };
    return @shuffle(u8, a, b, mask);
}

inline fn lo8(v: V16) U16 {
    return @truncate(@as(@Vector(16, u16), @bitCast(v)));
}

/// h2v1: out[2i] = (3c +% l +% 1) >> 2, out[2i+1] = (3c +% r +% 2) >> 2.
fn upH2v1(in: [*]const u8, dw: u32, cs: []i16, out: [*]u8) void {
    var i: usize = 0;
    while (i < dw) : (i += 16) cs[1 + i ..][0..16].* = load16(in + i);
    cs[0] = cs[1];
    cs[dw + 1] = cs[dw];
    i = 0;
    while (i < dw) : (i += 16) {
        const c: V16 = cs[1 + i ..][0..16].*;
        const l: V16 = cs[i..][0..16].*;
        const r: V16 = cs[2 + i ..][0..16].*;
        const c3 = c *% @as(V16, @splat(3));
        const e = lo8((c3 +% l +% @as(V16, @splat(1))) >> @splat(2));
        const o = lo8((c3 +% r +% @as(V16, @splat(2))) >> @splat(2));
        out[2 * i ..][0..32].* = interleave2(e, o);
    }
}

/// h1v2: out = (3 near + far + bias) >> 2.
fn upH1v2(near: [*]const u8, far: [*]const u8, dw: u32, bias: i16, out: [*]u8) void {
    var i: usize = 0;
    while (i < dw) : (i += 16) {
        const v = (load16(near + i) *% @as(V16, @splat(3)) +% load16(far + i) +% @as(V16, @splat(bias))) >> @splat(2);
        out[i..][0..16].* = lo8(v);
    }
}

/// h2v2: column sums 3 near + far, then out[2i] = (3c +% l +% 8) >> 4,
/// out[2i+1] = (3c +% r +% 7) >> 4.
fn upH2v2(near: [*]const u8, far: [*]const u8, dw: u32, cs: []i16, out: [*]u8) void {
    var i: usize = 0;
    while (i < dw) : (i += 16) {
        cs[1 + i ..][0..16].* = load16(near + i) *% @as(V16, @splat(3)) +% load16(far + i);
    }
    cs[0] = cs[1];
    cs[dw + 1] = cs[dw];
    i = 0;
    while (i < dw) : (i += 16) {
        const c: V16 = cs[1 + i ..][0..16].*;
        const l: V16 = cs[i..][0..16].*;
        const r: V16 = cs[2 + i ..][0..16].*;
        const c3 = c *% @as(V16, @splat(3));
        const e = lo8((c3 +% l +% @as(V16, @splat(8))) >> @splat(4));
        const o = lo8((c3 +% r +% @as(V16, @splat(7))) >> @splat(4));
        out[2 * i ..][0..32].* = interleave2(e, o);
    }
}

// libjpeg jdcolor.c constants (16-bit fixed point).
const fix_1_40200: i32 = 91881;
const fix_1_77200: i32 = 116130;
const fix_0_71414: i32 = 46802;
const fix_0_34414: i32 = 22554;
const one_half: i32 = 1 << 15;

const VI = @Vector(16, i32);

inline fn interleave3(a: U16, b: U16, c: U16) [48]u8 {
    const ab = @shuffle(u8, a, b, blk: {
        var m: [32]i32 = undefined;
        for (0..16) |i| {
            m[i] = i;
            m[16 + i] = ~@as(i32, i);
        }
        break :blk m;
    });
    const cc = @shuffle(u8, c, c, blk: {
        var m: [32]i32 = undefined;
        for (0..32) |i| m[i] = i % 16;
        break :blk m;
    });
    return @shuffle(u8, ab, cc, blk: {
        var m: [48]i32 = undefined;
        for (0..16) |p| {
            m[3 * p] = p;
            m[3 * p + 1] = 16 + p;
            m[3 * p + 2] = ~@as(i32, p);
        }
        break :blk m;
    });
}

inline fn clampU8(v: VI) U16 {
    return @intCast(@min(@max(v, @as(VI, @splat(0))), @as(VI, @splat(255))));
}

fn yccRow(comptime bgr: bool, yp: [*]const u8, cbp: [*]const u8, crp: [*]const u8, out: []u8, w: usize) void {
    var x: usize = 0;
    while (x + 16 <= w) : (x += 16) {
        const y: VI = @as(U16, yp[x..][0..16].*);
        const cb = @as(VI, @as(U16, cbp[x..][0..16].*)) - @as(VI, @splat(128));
        const cr = @as(VI, @as(U16, crp[x..][0..16].*)) - @as(VI, @splat(128));
        const r = y + ((cr * @as(VI, @splat(fix_1_40200)) + @as(VI, @splat(one_half))) >> @splat(16));
        const g = y + ((cb * @as(VI, @splat(-fix_0_34414)) + cr * @as(VI, @splat(-fix_0_71414)) + @as(VI, @splat(one_half))) >> @splat(16));
        const b = y + ((cb * @as(VI, @splat(fix_1_77200)) + @as(VI, @splat(one_half))) >> @splat(16));
        out[3 * x ..][0..48].* = if (bgr) interleave3(clampU8(b), clampU8(g), clampU8(r)) else interleave3(clampU8(r), clampU8(g), clampU8(b));
    }
    while (x < w) : (x += 1) {
        const rgb = yccPixel(yp[x], cbp[x], crp[x]);
        if (bgr) {
            out[3 * x] = rgb[2];
            out[3 * x + 1] = rgb[1];
            out[3 * x + 2] = rgb[0];
        } else out[3 * x ..][0..3].* = rgb;
    }
}

inline fn yccPixel(yv: u8, cbv: u8, crv: u8) [3]u8 {
    const y: i32 = yv;
    const cb: i32 = @as(i32, cbv) - 128;
    const cr: i32 = @as(i32, crv) - 128;
    return .{
        clamp8(y + ((fix_1_40200 * cr + one_half) >> 16)),
        clamp8(y + ((-fix_0_34414 * cb - fix_0_71414 * cr + one_half) >> 16)),
        clamp8(y + ((fix_1_77200 * cb + one_half) >> 16)),
    };
}

inline fn clamp8(v: i32) u8 {
    return @intCast(std.math.clamp(v, 0, 255));
}

fn rgbGrayRow(r: [*]const u8, g: [*]const u8, b: [*]const u8, out: []u8, w: usize) void {
    for (0..w) |x| {
        const v = @as(u32, r[x]) * 19595 + @as(u32, g[x]) * 38470 + @as(u32, b[x]) * 7471 + 32768;
        out[x] = @intCast(v >> 16);
    }
}

/// CMYK / YCCK as libjpeg delivers them (Adobe-inverted), then OpenCV's
/// CMYK->BGR formula.
fn cmykRow(ycck: bool, bgr: bool, src: [4][*]const u8, out: []u8, w: usize) void {
    for (0..w) |x| {
        var c: i32 = src[0][x];
        var m: i32 = src[1][x];
        var ye: i32 = src[2][x];
        const k: i32 = src[3][x];
        if (ycck) {
            const rgb = yccPixel(src[0][x], src[1][x], src[2][x]);
            c = 255 - @as(i32, rgb[0]);
            m = 255 - @as(i32, rgb[1]);
            ye = 255 - @as(i32, rgb[2]);
        }
        const r: u8 = @intCast(k - (((255 - c) * k) >> 8));
        const g: u8 = @intCast(k - (((255 - m) * k) >> 8));
        const b: u8 = @intCast(k - (((255 - ye) * k) >> 8));
        if (bgr) {
            out[3 * x] = b;
            out[3 * x + 1] = g;
            out[3 * x + 2] = r;
        } else {
            out[3 * x] = r;
            out[3 * x + 1] = g;
            out[3 * x + 2] = b;
        }
    }
}

// ---------------------------------------------------------------------------
// Entry points

pub fn decode(gpa: Allocator, bytes: []const u8, opts: common.DecodeOptions) DecodeError!AnyImage {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const d = try arena_state.allocator().create(Decoder);
    d.* = .{ .arena = arena_state.allocator(), .gpa = gpa, .data = bytes, .opts = opts };
    errdefer if (d.img) |*im| im.deinit(gpa);
    try d.run();
    var fo = opts;
    if (d.out_bgr) fo.order = .rgb;
    return common.finish(gpa, .{ .u8 = d.img.? }, fo);
}

/// Reads width, height and component count without decoding.
pub const Info = struct { width: u32, height: u32, components: u8, progressive: bool };

pub fn info(bytes: []const u8) DecodeError!Info {
    if (bytes.len < 4 or bytes[0] != 0xff or bytes[1] != 0xd8) return error.InvalidData;
    var pos: usize = 2;
    while (pos + 4 <= bytes.len) {
        if (bytes[pos] != 0xff) {
            pos += 1;
            continue;
        }
        const m = bytes[pos + 1];
        if (m == 0xff or m == 0x01 or m == 0xd8 or (m >= 0xd0 and m <= 0xd7)) {
            pos += if (m == 0xff) 1 else 2;
            continue;
        }
        if (m == 0xd9 or m == 0xda) break;
        const len = std.mem.readInt(u16, bytes[pos + 2 ..][0..2], .big);
        if (len < 2) return error.InvalidData;
        if (m >= 0xc0 and m <= 0xcf and m != 0xc4 and m != 0xc8 and m != 0xcc) {
            if (pos + 10 > bytes.len) return error.Truncated;
            return .{
                .height = std.mem.readInt(u16, bytes[pos + 5 ..][0..2], .big),
                .width = std.mem.readInt(u16, bytes[pos + 7 ..][0..2], .big),
                .components = bytes[pos + 9],
                .progressive = m == 0xc2 or m == 0xc6 or m == 0xca or m == 0xce,
            };
        }
        pos += 2 + @as(usize, len);
    }
    return error.Truncated;
}

/// EXIF orientation (1-8) from the APP1 segment, or 1 when absent/invalid.
pub fn orientation(bytes: []const u8) u8 {
    if (bytes.len < 4 or bytes[0] != 0xff or bytes[1] != 0xd8) return 1;
    var pos: usize = 2;
    while (pos + 4 <= bytes.len) {
        if (bytes[pos] != 0xff) return 1;
        const m = bytes[pos + 1];
        if (m == 0xff) {
            pos += 1;
            continue;
        }
        if (m == 0xd9 or m == 0xda) return 1;
        if (m == 0x01 or (m >= 0xd0 and m <= 0xd8)) {
            pos += 2;
            continue;
        }
        const len = std.mem.readInt(u16, bytes[pos + 2 ..][0..2], .big);
        if (len < 2 or pos + 2 + len > bytes.len) return 1;
        const seg = bytes[pos + 4 .. pos + 2 + len];
        if (m == 0xe1 and seg.len >= 6 and std.mem.eql(u8, seg[0..6], "Exif\x00\x00")) {
            return exifOrientation(seg[6..]) orelse 1;
        }
        pos += 2 + @as(usize, len);
    }
    return 1;
}

fn exifOrientation(t: []const u8) ?u8 {
    if (t.len < 8) return null;
    const endian: std.builtin.Endian = if (std.mem.eql(u8, t[0..2], "II")) .little else if (std.mem.eql(u8, t[0..2], "MM")) .big else return null;
    if (std.mem.readInt(u16, t[2..4], endian) != 42) return null;
    const ifd = std.mem.readInt(u32, t[4..8], endian);
    if (ifd > t.len or t.len - ifd < 2) return null;
    const n = std.mem.readInt(u16, t[ifd..][0..2], endian);
    for (0..n) |i| {
        const e = ifd + 2 + 12 * i;
        if (e + 12 > t.len) return null;
        const tag = std.mem.readInt(u16, t[e..][0..2], endian);
        if (tag != 0x0112) continue;
        const typ = std.mem.readInt(u16, t[e + 2 ..][0..2], endian);
        if (typ != 3) return null;
        const v = std.mem.readInt(u16, t[e + 8 ..][0..2], endian);
        return if (v >= 1 and v <= 8) @intCast(v) else null;
    }
    return null;
}
