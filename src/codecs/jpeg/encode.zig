//! Baseline JPEG encoder, matching libjpeg's defaults as OpenCV uses them:
//! JFIF header, IJG quality scaling of the Annex K tables, standard Huffman
//! tables, YCbCr 4:2:0 for colour, ISLOW forward DCT, libjpeg's fixed-point
//! RGB->YCbCr and its 2x2 box downsampling with alternating rounding bias.

const std = @import("std");
const common = @import("../common.zig");
const tables = @import("tables.zig");
const dct = @import("dct.zig");

const Allocator = std.mem.Allocator;
const EncodeError = common.EncodeError;
const natural_order = tables.natural_order;

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
    acc: u64 = 0,
    n: u32 = 0,

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

    /// Appends `len` bits of `code`. Room must have been ensured.
    inline fn put(o: *Out, code: u32, len: u32) void {
        o.acc = (o.acc << @intCast(len)) | code;
        o.n += len;
        if (o.n >= 32) {
            o.n -= 32;
            const w: u32 = @truncate(o.acc >> @intCast(o.n));
            o.emit32(w);
        }
    }

    inline fn emit32(o: *Out, w: u32) void {
        const x = ~w;
        if (((x -% 0x01010101) & ~x & 0x80808080) == 0) {
            std.mem.writeInt(u32, o.buf[o.len..][0..4], w, .big);
            o.len += 4;
            return;
        }
        var i: u5 = 24;
        while (true) : (i -= 8) {
            const b: u8 = @truncate(w >> i);
            o.buf[o.len] = b;
            o.len += 1;
            if (b == 0xff) {
                o.buf[o.len] = 0;
                o.len += 1;
            }
            if (i == 0) break;
        }
    }

    /// Pads the last byte with 1-bits and flushes.
    fn flushBits(o: *Out) Allocator.Error!void {
        try o.ensure(16);
        const pad = (8 - (o.n & 7)) & 7;
        if (pad != 0) {
            o.acc = (o.acc << @intCast(pad)) | ((@as(u64, 1) << @intCast(pad)) - 1);
            o.n += pad;
        }
        while (o.n >= 8) {
            o.n -= 8;
            const b: u8 = @truncate(o.acc >> @intCast(o.n));
            o.buf[o.len] = b;
            o.len += 1;
            if (b == 0xff) {
                o.buf[o.len] = 0;
                o.len += 1;
            }
        }
    }
};

const CompEnc = struct {
    /// Reciprocal of 8*q per coefficient (natural order) and q*4 rounding.
    recip: [64]f64,
    half: [64]f64,
    dc: *const HuffEnc,
    ac: *const HuffEnc,
    pred: i32 = 0,
};

fn makeRecip(q: *const [64]u16, c: *CompEnc) void {
    for (0..64) |i| {
        const d: f64 = @floatFromInt(@as(u32, q[i]) * 8);
        c.recip[i] = 1.0 / d;
        c.half[i] = @floatFromInt(@as(u32, q[i]) * 4);
    }
}

/// FDCT, quantize and Huffman-code one block of samples (8 rows, `stride`
/// bytes apart).
fn encodeBlock(o: *Out, c: *CompEnc, px: [*]const u8, stride: usize) void {
    var in: [64]i32 align(64) = undefined;
    inline for (0..8) |r| {
        const row: @Vector(8, u8) = px[r * stride ..][0..8].*;
        in[r * 8 ..][0..8].* = @as(@Vector(8, i32), row) - @as(@Vector(8, i32), @splat(128));
    }
    var coef: [64]i32 align(64) = undefined;
    dct.fdct(&in, &coef);
    // Quantize: round half away from zero of coef / (8q), as libjpeg does.
    const F = @Vector(8, f64);
    var qv: [64]i32 align(64) = undefined;
    inline for (0..8) |r| {
        const v: @Vector(8, i32) = coef[r * 8 ..][0..8].*;
        const a: F = @floatFromInt(@abs(v));
        const h: F = c.half[r * 8 ..][0..8].*;
        const rc: F = c.recip[r * 8 ..][0..8].*;
        const qa: @Vector(8, i32) = @intFromFloat(@floor((a + h) * rc + @as(F, @splat(1e-9))));
        qv[r * 8 ..][0..8].* = @select(i32, v < @as(@Vector(8, i32), @splat(0)), -qa, qa);
    }
    var zz: [64]i32 align(64) = undefined;
    inline for (0..64) |k| zz[k] = qv[natural_order[k]];
    var mask: u64 = 0;
    inline for (0..8) |r| {
        const v: @Vector(8, i32) = zz[r * 8 ..][0..8].*;
        const nz: u8 = @bitCast(v != @as(@Vector(8, i32), @splat(0)));
        mask |= @as(u64, nz) << (r * 8);
    }
    // DC.
    const diff = zz[0] - c.pred;
    c.pred = zz[0];
    {
        const s = magBits(diff);
        o.put(c.dc.code[s], c.dc.size[s]);
        if (s != 0) o.put(valBits(diff, s), s);
    }
    mask &= ~@as(u64, 1);
    var last: u32 = 0;
    while (mask != 0) {
        const k: u32 = @ctz(mask);
        mask &= mask - 1;
        var run = k - last - 1;
        while (run >= 16) : (run -= 16) o.put(c.ac.code[0xf0], c.ac.size[0xf0]);
        const v = zz[k];
        const s = magBits(v);
        const sym = (run << 4) | s;
        o.put((@as(u32, c.ac.code[sym]) << @intCast(s)) | valBits(v, s), c.ac.size[sym] + s);
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
            const v: @Vector(48, u8) = src[3 * x ..][0..48].*;
            const p0: U16 = @shuffle(u8, v, undefined, deint(0));
            const p1: U16 = @shuffle(u8, v, undefined, deint(1));
            const p2: U16 = @shuffle(u8, v, undefined, deint(2));
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

fn deint(comptime c: usize) [16]i32 {
    var m: [16]i32 = undefined;
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
    comps[0] = .{ .recip = undefined, .half = undefined, .dc = &dc_l, .ac = &ac_l };
    makeRecip(&ql, &comps[0]);
    for (1..3) |i| {
        comps[i] = .{ .recip = undefined, .half = undefined, .dc = &dc_c, .ac = &ac_c };
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
                    var bias: u32 = 1;
                    if (vs == 2) {
                        for (0..cw) |x| {
                            d[x] = @intCast((@as(u32, a[2 * x]) + a[2 * x + 1] + b[2 * x] + b[2 * x + 1] + bias) >> 2);
                            bias ^= 3;
                        }
                    } else {
                        bias = 0;
                        for (0..cw) |x| {
                            d[x] = @intCast((@as(u32, a[2 * x]) + a[2 * x + 1] + bias) >> 1);
                            bias ^= 1;
                        }
                    }
                }
            }
            cbp = cbd;
            crp = crd;
        }
        try o.ensure(mcux * (2 + hs * vs) * 64 * 8 + 64);
        for (0..mcux) |mx| {
            for (0..vs) |bv| {
                for (0..hs) |bh| {
                    encodeBlock(&o, &comps[0], yp.ptr + bv * 8 * pw + mx * mcu_w + bh * 8, pw);
                }
            }
            if (color) {
                encodeBlock(&o, &comps[1], cbp.ptr + mx * 8, cw);
                encodeBlock(&o, &comps[2], crp.ptr + mx * 8, cw);
            }
        }
    }
    try o.flushBits();
    try o.bytes(&.{ 0xff, 0xd9 });
    return gpa.realloc(o.buf, o.len);
}
