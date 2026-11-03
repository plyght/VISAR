//! Baseline JPEG encoder, matching libjpeg's defaults as OpenCV uses them:
//! JFIF header, IJG quality scaling of the Annex K tables, standard Huffman
//! tables, YCbCr 4:2:0 for colour, ISLOW forward DCT, libjpeg's fixed-point
//! RGB->YCbCr and its 2x2 box downsampling with alternating rounding bias.

const std = @import("std");
const common = @import("../common.zig");
const tables = @import("tables.zig");
const dct = @import("dct.zig");
const decode = @import("decode.zig");

const Allocator = std.mem.Allocator;
const EncodeError = common.EncodeError;
const natural_order = tables.natural_order;
const zigzag_mask: [64]i32 = blk: {
    var m: [64]i32 = undefined;
    for (0..64) |i| m[i] = natural_order[i];
    break :blk m;
};

pub const Subsampling = enum { s420, s422, s444 };

pub const Options = struct {
    subsampling: Subsampling = .s420,
};

/// Scales an Annex K table for `quality` (1..100) the way libjpeg does
/// (jpeg_quality_scaling + jpeg_add_quant_table with force_baseline).
pub fn scaleQuant(base: *const [64]u16, quality: u8) [64]u16 {
    const q: u32 = std.math.clamp(quality, 1, 100);
    const scale: u32 = if (q < 50) 5000 / q else 200 - q * 2;
    var out: [64]u16 = undefined;
    for (base, &out) |b, *o| {
        const t = (@as(u32, b) * scale + 50) / 100;
        o.* = @intCast(std.math.clamp(t, 1, 255));
    }
    return out;
}

const HuffEnc = struct {
    code: [256]u16 = @splat(0),
    size: [256]u8 = @splat(0),

    fn init(spec: tables.HuffSpec) HuffEnc {
        var h: HuffEnc = .{};
        var code: u16 = 0;
        var idx: usize = 0;
        for (spec.bits, 1..) |n, l| {
            for (0..n) |_| {
                const sym = spec.vals[idx];
                h.code[sym] = code;
                h.size[sym] = @intCast(l);
                code += 1;
                idx += 1;
            }
            code <<= 1;
        }
        return h;
    }
};

const Out = struct {
    gpa: Allocator,
    buf: []u8,
    len: usize = 0,
    /// Pending bits, right-aligned; 64 - `free` of them are valid.
    acc: u64 = 0,
    free: i32 = 64,

    fn ensure(o: *Out, extra: usize) Allocator.Error!void {
        if (o.len + extra <= o.buf.len) return;
        const new_len = @max(o.buf.len * 2, o.len + extra);
        o.buf = try o.gpa.realloc(o.buf, new_len);
    }

    fn bytes(o: *Out, b: []const u8) Allocator.Error!void {
        try o.ensure(b.len);
        @memcpy(o.buf[o.len..][0..b.len], b);
        o.len += b.len;
    }

    fn u16be(o: *Out, v: u16) Allocator.Error!void {
        try o.bytes(&.{ @intCast(v >> 8), @truncate(v) });
    }

    /// Appends `len` (1..32) bits of `code`; the buffer goes out 8 bytes
    /// at a time (libjpeg-turbo's scheme). Room must have been ensured.
    inline fn put(o: *Out, code: u32, len: u32) void {
        o.free -= @intCast(len);
        if (o.free < 0) {
            const take: u32 = @intCast(@as(i32, @intCast(len)) + o.free);
            const rest: u5 = @intCast(-o.free);
            o.acc = (o.acc << @intCast(take)) | (code >> rest);
            o.emit64(o.acc);
            o.free += 64;
            o.acc = code;
        } else {
            o.acc = (o.acc << @intCast(len)) | code;
        }
    }

    inline fn emit64(o: *Out, w: u64) void {
        const x = ~w;
        if (((x -% 0x0101010101010101) & ~x & 0x8080808080808080) == 0) {
            std.mem.writeInt(u64, o.buf[o.len..][0..8], w, .big);
            o.len += 8;
        } else o.stuff(w, 8);
    }

    fn stuff(o: *Out, w: u64, n: u32) void {
        for (0..n) |i| {
            const b: u8 = @truncate(w >> @intCast(56 - 8 * i));
            o.buf[o.len] = b;
            o.len += 1;
            if (b == 0xff) {
                o.buf[o.len] = 0;
                o.len += 1;
            }
        }
    }

    /// Pads the last byte with 1-bits and flushes.
    fn flushBits(o: *Out) Allocator.Error!void {
        try o.ensure(32);
        const used: u32 = @intCast(64 - o.free);
        const pad = (8 - (used & 7)) & 7;
        if (pad != 0) o.put((@as(u32, 1) << @intCast(pad)) - 1, pad);
        const n: u32 = @intCast(64 - o.free);
        if (n == 0) return;
        o.stuff(o.acc << @intCast(64 - n), n / 8);
        o.acc = 0;
        o.free = 64;
    }
};

const CompEnc = struct {
    /// 1 / (8q) and 4q + 0.5 per coefficient (natural order): with them
    /// floor((|x| + bias) * recip) equals libjpeg's (|x| + 4q) / 8q
    /// exactly for every |x| < 2^17 (checked exhaustively).
    recip: [64]f32,
    bias: [64]f32,
    dc: *const HuffEnc,
    ac: *const HuffEnc,
    pred: i32 = 0,
};

fn makeRecip(q: *const [64]u16, c: *CompEnc) void {
    for (0..64) |i| {
        const d: u32 = @as(u32, q[i]) * 8;
        c.recip[i] = 1.0 / @as(f32, @floatFromInt(d));
        c.bias[i] = @as(f32, @floatFromInt(d / 2)) + 0.5;
    }
}

/// A block waiting to be transformed: its component and top-left sample.
const Pending = struct { c: *CompEnc, px: [*]const u8, stride: usize };

/// Collects blocks in coding order and transforms them two at a time.
const Pairer = struct {
    pending: ?Pending = null,

    inline fn add(p: *Pairer, o: *Out, b: Pending) void {
        if (p.pending) |a| {
            encodePair(o, a, b, true);
            p.pending = null;
        } else p.pending = b;
    }

    fn flush(p: *Pairer, o: *Out) void {
        if (p.pending) |a| encodePair(o, a, a, false);
        p.pending = null;
    }
};

/// FDCT and quantize two blocks at once, then Huffman-code them in order
/// (the second only when `both`).
fn encodePair(o: *Out, a: Pending, b: Pending, both: bool) void {
    var co: [8]@Vector(16, i32) = undefined;
    dct.fdct2(a.px, a.stride, b.px, b.stride, &co);
    // Quantize: round half away from zero of coef / (8q), as libjpeg does.
    const F = @Vector(16, f32);
    const I = @Vector(16, i32);
    var qa: [64]i16 align(64) = undefined;
    var qb: [64]i16 align(64) = undefined;
    inline for (0..8) |r| {
        const v = co[r];
        const h: F = cat(a.c.bias[r * 8 ..][0..8].*, b.c.bias[r * 8 ..][0..8].*);
        const rc: F = cat(a.c.recip[r * 8 ..][0..8].*, b.c.recip[r * 8 ..][0..8].*);
        const m: I = @intFromFloat((@as(F, @floatFromInt(@abs(v))) + h) * rc);
        const q: [16]i16 = @as(@Vector(16, i16), @truncate(@select(i32, v < @as(I, @splat(0)), -m, m)));
        qa[r * 8 ..][0..8].* = q[0..8].*;
        qb[r * 8 ..][0..8].* = q[8..16].*;
    }
    huffBlock(o, a.c, &qa);
    if (both) huffBlock(o, b.c, &qb);
}

inline fn cat(x: @Vector(8, f32), y: @Vector(8, f32)) @Vector(16, f32) {
    return @shuffle(f32, x, y, [16]i32{ 0, 1, 2, 3, 4, 5, 6, 7, ~@as(i32, 0), ~@as(i32, 1), ~@as(i32, 2), ~@as(i32, 3), ~@as(i32, 4), ~@as(i32, 5), ~@as(i32, 6), ~@as(i32, 7) });
}

/// Huffman-codes one block of quantized coefficients (natural order).
fn huffBlock(out: *Out, c: *CompEnc, q: *const [64]i16) void {
    // Work on a local copy so the bit buffer stays in registers (stores
    // through `buf` could otherwise alias it).
    var ow = out.*;
    defer out.* = ow;
    const o = &ow;
    const Z = @Vector(64, i16);
    // Zigzag reorder as one 64-lane shuffle.
    const zv: Z = @shuffle(i16, @as(Z, q.*), undefined, zigzag_mask);
    var mask: u64 = @bitCast(zv != @as(Z, @splat(0)));
    const U = @Vector(64, u16);
    const mag: @Vector(64, u32) = @abs(zv);
    const nb: U = @intCast(@as(@Vector(64, u32), @splat(32)) - @clz(mag));
    const ones: U = @splat(1);
    const raw: U = @bitCast(zv +% (zv >> @splat(15)));
    const sh: @Vector(64, u4) = @intCast(nb & @as(U, @splat(15)));
    const vb: U = raw & ((ones << sh) -% ones);
    const nbits: [64]u16 = nb;
    const vbits: [64]u16 = vb;

    // DC.
    const dc: i32 = zv[0];
    const diff = dc - c.pred;
    c.pred = dc;
    {
        const s = magBits(diff);
        o.put((@as(u32, c.dc.code[s]) << @intCast(s)) | valBits(diff, s), c.dc.size[s] + s);
    }
    mask &= ~@as(u64, 1);
    var last: u32 = 0;
    while (mask != 0) {
        const k: u32 = @ctz(mask);
        mask &= mask - 1;
        var run = k - last - 1;
        while (run >= 16) : (run -= 16) o.put(c.ac.code[0xf0], c.ac.size[0xf0]);
        const s: u32 = nbits[k];
        const sym = (run << 4) | s;
        o.put((@as(u32, c.ac.code[sym]) << @intCast(s)) | vbits[k], c.ac.size[sym] + s);
        last = k;
    }
    if (last != 63) o.put(c.ac.code[0], c.ac.size[0]);
}

inline fn magBits(v: i32) u32 {
    const a: u32 = @abs(v);
    return 32 - @clz(a);
}

inline fn valBits(v: i32, s: u32) u32 {
    const u: u32 = @bitCast(if (v < 0) v - 1 else v);
    return u & ((@as(u32, 1) << @intCast(s)) - 1);
}

// libjpeg jccolor.c constants.
const VI = @Vector(16, i32);
const U16 = @Vector(16, u8);

fn rgbToYcc(comptime bgr: bool, comptime ch: usize, src: []const u8, w: usize, y: []u8, cb: []u8, cr: []u8) void {
    var x: usize = 0;
    if (ch == 3) {
        while (x + 16 <= w) : (x += 16) {
            const v0: U16 = src[3 * x ..][0..16].*;
            const v1: U16 = src[3 * x + 16 ..][0..16].*;
            const v2: U16 = src[3 * x + 32 ..][0..16].*;
            const p0 = decode.shuf3(v0, v1, v2, comptime deint(0));
            const p1 = decode.shuf3(v0, v1, v2, comptime deint(1));
            const p2 = decode.shuf3(v0, v1, v2, comptime deint(2));
            const r: VI = if (bgr) p2 else p0;
            const g: VI = p1;
            const b: VI = if (bgr) p0 else p2;
            const s16: VI = @splat(16);
            const yy = (r * @as(VI, @splat(19595)) + g * @as(VI, @splat(38470)) + b * @as(VI, @splat(7471)) + @as(VI, @splat(32768))) >> s16;
            const off: VI = @splat((128 << 16) + 32767);
            const cbv = (r * @as(VI, @splat(-11059)) + g * @as(VI, @splat(-21709)) + b * @as(VI, @splat(32768)) + off) >> s16;
            const crv = (r * @as(VI, @splat(32768)) + g * @as(VI, @splat(-27439)) + b * @as(VI, @splat(-5329)) + off) >> s16;
            y[x..][0..16].* = @as(U16, @intCast(yy));
            cb[x..][0..16].* = @as(U16, @intCast(cbv));
            cr[x..][0..16].* = @as(U16, @intCast(crv));
        }
    }
    while (x < w) : (x += 1) {
        const p = src[ch * x ..][0..3];
        const r: i32 = if (bgr) p[2] else p[0];
        const g: i32 = p[1];
        const b: i32 = if (bgr) p[0] else p[2];
        y[x] = @intCast((r * 19595 + g * 38470 + b * 7471 + 32768) >> 16);
        cb[x] = @intCast((r * -11059 + g * -21709 + b * 32768 + (128 << 16) + 32767) >> 16);
        cr[x] = @intCast((r * 32768 + g * -27439 + b * -5329 + (128 << 16) + 32767) >> 16);
    }
}

/// libjpeg h2v2_downsample: 2x2 average with bias alternating 1, 2.
fn downsample2x2(a: []const u8, b: []const u8, d: []u8) void {
    const W = @Vector(16, u16);
    const bias: W = comptime blk: {
        var m: [16]u16 = undefined;
        for (0..16) |i| m[i] = if (i % 2 == 0) 1 else 2;
        break :blk m;
    };
    var x: usize = 0;
    while (x + 16 <= d.len) : (x += 16) {
        // Little-endian pairs: low byte = even pixel, high byte = odd.
        const va: W = @as(*align(1) const [16]u16, @ptrCast(a[2 * x ..].ptr)).*;
        const vb: W = @as(*align(1) const [16]u16, @ptrCast(b[2 * x ..].ptr)).*;
        const lo: W = @splat(0xff);
        const ae = va & lo;
        const ao = va >> @splat(8);
        const be = vb & lo;
        const bo = vb >> @splat(8);
        d[x..][0..16].* = @as(@Vector(16, u8), @intCast((ae + ao + be + bo + bias) >> @splat(2)));
    }
    while (x < d.len) : (x += 1) {
        const bias1: u32 = if (x % 2 == 0) 1 else 2;
        d[x] = @intCast((@as(u32, a[2 * x]) + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + bias1) >> 2);
    }
}

/// libjpeg h2v1_downsample: horizontal pair average with bias 0, 1.
fn downsample2x1(a: []const u8, d: []u8) void {
    for (0..d.len) |x| {
        const bias1: u32 = @intCast(x % 2);
        d[x] = @intCast((@as(u32, a[2 * x]) + a[2 * x + 1] + bias1) >> 1);
    }
}

fn deint(comptime c: usize) [16]u8 {
    var m: [16]u8 = undefined;
    for (0..16) |i| m[i] = @intCast(3 * i + c);
    return m;
}

pub fn encode(gpa: Allocator, img: common.AnyImage, opts: common.EncodeOptions) EncodeError![]u8 {
    return encodeWith(gpa, img, opts, .{});
}

pub fn encodeWith(gpa: Allocator, img: common.AnyImage, opts: common.EncodeOptions, eo: Options) EncodeError![]u8 {
    const src = img.as(u8) orelse return error.Unsupported;
    const ch: usize = src.channels;
    if (ch < 1 or ch > 4) return error.Unsupported;
    if (src.width > 65535 or src.height > 65535) return error.Unsupported;
    const color = ch >= 3;
    const w: usize = src.width;
    const h: usize = src.height;
    const bgr = opts.order == .bgr;

    const hs: usize = if (!color or eo.subsampling == .s444) 1 else 2;
    const vs: usize = if (!color or eo.subsampling != .s420) 1 else 2;
    const mcu_w = 8 * hs;
    const mcu_h = 8 * vs;
    const mcux = (w + mcu_w - 1) / mcu_w;
    const mcuy = (h + mcu_h - 1) / mcu_h;
    const pw = mcux * mcu_w; // padded full-resolution width
    const cw = pw / hs; // padded chroma width

    const ql = scaleQuant(&tables.std_luma_quant, opts.quality);
    const qc = scaleQuant(&tables.std_chroma_quant, opts.quality);
    const dc_l = HuffEnc.init(tables.std_dc_luma);
    const ac_l = HuffEnc.init(tables.std_ac_luma);
    const dc_c = HuffEnc.init(tables.std_dc_chroma);
    const ac_c = HuffEnc.init(tables.std_ac_chroma);

    var o: Out = .{ .gpa = gpa, .buf = try gpa.alloc(u8, @max(4096, w * h * ch / 4)) };
    errdefer gpa.free(o.buf);

    // Headers.
    try o.bytes(&.{ 0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00 });
    const nq: usize = if (color) 2 else 1;
    for (0..nq) |t| {
        const q = if (t == 0) &ql else &qc;
        try o.bytes(&.{ 0xff, 0xdb, 0x00, 67, @intCast(t) });
        var zzq: [64]u8 = undefined;
        for (0..64) |k| zzq[k] = @intCast(q[natural_order[k]]);
        try o.bytes(&zzq);
    }
    const nc: usize = if (color) 3 else 1;
    try o.bytes(&.{ 0xff, 0xc0 });
    try o.u16be(@intCast(8 + 3 * nc));
    try o.bytes(&.{8});
    try o.u16be(@intCast(h));
    try o.u16be(@intCast(w));
    try o.bytes(&.{@intCast(nc)});
    for (0..nc) |i| {
        const hv: u8 = if (i == 0) @intCast(hs << 4 | vs) else 0x11;
        try o.bytes(&.{ @intCast(i + 1), hv, if (i == 0) 0 else 1 });
    }
    const specs = [_]struct { tc: u8, th: u8, s: tables.HuffSpec }{
        .{ .tc = 0, .th = 0, .s = tables.std_dc_luma },
        .{ .tc = 1, .th = 0, .s = tables.std_ac_luma },
        .{ .tc = 0, .th = 1, .s = tables.std_dc_chroma },
        .{ .tc = 1, .th = 1, .s = tables.std_ac_chroma },
    };
    for (specs[0 .. 2 * nq]) |sp| {
        try o.bytes(&.{ 0xff, 0xc4 });
        try o.u16be(@intCast(2 + 17 + sp.s.vals.len));
        try o.bytes(&.{(sp.tc << 4) | sp.th});
        try o.bytes(&sp.s.bits);
        try o.bytes(sp.s.vals);
    }
    try o.bytes(&.{ 0xff, 0xda });
    try o.u16be(@intCast(6 + 2 * nc));
    try o.bytes(&.{@intCast(nc)});
    for (0..nc) |i| try o.bytes(&.{ @intCast(i + 1), if (i == 0) 0x00 else 0x11 });
    try o.bytes(&.{ 0, 63, 0 });

    // Component state.
    var comps: [3]CompEnc = undefined;
    comps[0] = .{ .recip = undefined, .bias = undefined, .dc = &dc_l, .ac = &ac_l };
    makeRecip(&ql, &comps[0]);
    for (1..3) |i| {
        comps[i] = .{ .recip = undefined, .bias = undefined, .dc = &dc_c, .ac = &ac_c };
        makeRecip(&qc, &comps[i]);
    }

    // Row buffers for one MCU row: full-res planes (Y, Cb, Cr) and
    // downsampled chroma.
    const planes = try gpa.alloc(u8, 3 * pw * mcu_h + 2 * cw * 8 + 64);
    defer gpa.free(planes);
    const yp = planes[0 .. pw * mcu_h];
    const cbf = planes[pw * mcu_h .. 2 * pw * mcu_h];
    const crf = planes[2 * pw * mcu_h .. 3 * pw * mcu_h];
    const cbd = planes[3 * pw * mcu_h ..][0 .. cw * 8];
    const crd = planes[3 * pw * mcu_h + cw * 8 ..][0 .. cw * 8];

    for (0..mcuy) |my| {
        // Colour convert (and pad) the MCU row.
        for (0..mcu_h) |r| {
            const sy = @min(my * mcu_h + r, h - 1);
            const s = src.row(sy);
            const yr = yp[r * pw ..][0..pw];
            if (!color) {
                if (ch == 1) {
                    @memcpy(yr[0..w], s[0..w]);
                } else {
                    for (0..w) |x| yr[x] = s[x * ch];
                }
            } else {
                const cbr = cbf[r * pw ..][0..pw];
                const crr = crf[r * pw ..][0..pw];
                switch (ch) {
                    3 => if (bgr) rgbToYcc(true, 3, s, w, yr, cbr, crr) else rgbToYcc(false, 3, s, w, yr, cbr, crr),
                    else => if (bgr) rgbToYcc(true, 4, s, w, yr, cbr, crr) else rgbToYcc(false, 4, s, w, yr, cbr, crr),
                }
                @memset(cbr[w..], cbr[w - 1]);
                @memset(crr[w..], crr[w - 1]);
            }
            @memset(yr[w..], yr[w - 1]);
        }
        // Downsample chroma.
        var cbp: []const u8 = cbf;
        var crp: []const u8 = crf;
        if (color and (hs != 1 or vs != 1)) {
            for (0..8) |r| {
                for ([2][]const u8{ cbf, crf }, [2][]u8{ cbd, crd }) |full, dst| {
                    const a = full[r * vs * pw ..][0..pw];
                    const b = full[(r * vs + vs - 1) * pw ..][0..pw];
                    const d = dst[r * cw ..][0..cw];
                    if (vs == 2) downsample2x2(a, b, d) else downsample2x1(a, d);
                }
            }
            cbp = cbd;
            crp = crd;
        }
        try o.ensure(mcux * (2 + hs * vs) * 64 * 8 + 64);
        var pr: Pairer = .{};
        for (0..mcux) |mx| {
            for (0..vs) |bv| {
                for (0..hs) |bh| {
                    pr.add(&o, .{ .c = &comps[0], .px = yp.ptr + bv * 8 * pw + mx * mcu_w + bh * 8, .stride = pw });
                }
            }
            if (color) {
                pr.add(&o, .{ .c = &comps[1], .px = cbp.ptr + mx * 8, .stride = cw });
                pr.add(&o, .{ .c = &comps[2], .px = crp.ptr + mx * 8, .stride = cw });
            }
        }
        pr.flush(&o);
    }
    try o.flushBits();
    try o.bytes(&.{ 0xff, 0xd9 });
    return gpa.realloc(o.buf, o.len);
}
