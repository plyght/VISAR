//! Arithmetic in GF(256) with the QR code polynomial x^8 + x^4 + x^3 + x^2 + 1
//! (0x11d) and Reed-Solomon encoding / decoding over it (generator roots
//! alpha^0 .. alpha^(n-1), as in ISO/IEC 18004).

const std = @import("std");

pub const exp_table: [512]u8 = blk: {
    var t: [512]u8 = undefined;
    var x: u16 = 1;
    for (0..255) |i| {
        t[i] = @intCast(x);
        x <<= 1;
        if (x & 0x100 != 0) x ^= 0x11d;
    }
    for (255..512) |i| t[i] = t[i - 255];
    break :blk t;
};

pub const log_table: [256]u8 = blk: {
    var t: [256]u8 = @splat(0);
    for (0..255) |i| t[exp_table[i]] = @intCast(i);
    break :blk t;
};

pub inline fn mul(a: u8, b: u8) u8 {
    if (a == 0 or b == 0) return 0;
    return exp_table[@as(usize, log_table[a]) + log_table[b]];
}

pub inline fn div(a: u8, b: u8) u8 {
    if (a == 0 or b == 0) return 0;
    return exp_table[@as(usize, log_table[a]) + 255 - log_table[b]];
}

/// alpha^e for e >= 0.
pub inline fn pow2(e: usize) u8 {
    return exp_table[e % 255];
}

pub const max_ecc = 68;

/// The generator polynomial prod (x - alpha^i), i < n, highest degree
/// first (g[0] = 1), n <= max_ecc.
pub fn generator(n: usize) [max_ecc + 1]u8 {
    var g: [max_ecc + 1]u8 = @splat(0);
    g[0] = 1;
    var len: usize = 1;
    for (0..n) |i| {
        // g *= (x + alpha^i)
        const r = pow2(i);
        g[len] = 0;
        var k = len;
        while (k > 0) : (k -= 1) g[k] ^= mul(g[k - 1], r);
        len += 1;
    }
    return g;
}

/// The Reed-Solomon parity of `data` (data * x^n mod g), highest degree
/// first, into `out` (n = g.len - 1 bytes).
pub fn rsRemainder(data: []const u8, g: []const u8, out: []u8) void {
    const n = g.len - 1;
    std.debug.assert(out.len == n);
    @memset(out, 0);
    for (data) |d| {
        const f = d ^ out[0];
        std.mem.copyForwards(u8, out[0 .. n - 1], out[1..n]);
        out[n - 1] = 0;
        if (f != 0) {
            const lf: usize = log_table[f];
            for (out, g[1..]) |*o, gv| {
                if (gv != 0) o.* ^= exp_table[lf + log_table[gv]];
            }
        }
    }
}

/// OpenCV's QRCodeDecoder error correction of one block (data codewords
/// followed by `necc` parity codewords) in place: Berlekamp-Massey,
/// a Chien search and Forney's algorithm over an even number of
/// syndromes. False when the block is uncorrectable.
pub fn rsCorrect(cw: []u8, necc: usize) bool {
    var nsyn = necc;
    if (nsyn % 2 == 1) nsyn -= 1;
    if (nsyn == 0 or cw.len == 0 or cw.len > 255) return nsyn == 0;
    var syn: [max_ecc]u8 = undefined;
    var has_error = false;
    for (0..nsyn) |i| {
        var s: u8 = cw[0];
        const r = pow2(i);
        for (cw[1..]) |c| s = mul(s, r) ^ c;
        syn[i] = s;
        has_error = has_error or s != 0;
    }
    if (!has_error) return true;

    // Berlekamp-Massey: the error locator C.
    var C: [max_ecc + 1]u8 = @splat(0);
    var B: [max_ecc + 1]u8 = @splat(0);
    C[0] = 1;
    B[0] = 1;
    var L: usize = 0;
    var m: usize = 1;
    var b: u8 = 1;
    for (0..nsyn) |i| {
        if (m + L > nsyn) return false;
        var d = syn[i];
        for (1..L + 1) |j| d ^= mul(C[j], syn[i - j]);
        if (d == 0) {
            m += 1;
        } else {
            const copy = C;
            const t = mul(d, div(1, b));
            for (0..L) |j| C[m + j] ^= mul(t, B[j]);
            if (2 * L <= i) {
                L = i + 1 - L;
                B = copy;
                b = d;
                m = 1;
            } else m += 1;
        }
    }
    if (L == 0) return false;

    // Chien search over the block's positions.
    var locs: [max_ecc]usize = undefined;
    var nloc: usize = 0;
    for (0..cw.len) |i| {
        var v: u8 = 1;
        const p = pow2(i);
        for (1..L + 1) |j| v = mul(v, p) ^ C[j];
        if (v == 0) {
            if (nloc == L) return false;
            locs[nloc] = cw.len - 1 - i;
            nloc += 1;
        }
    }
    if (nloc != L) return false;

    // Forney: the evaluator is C * S (low terms), as OpenCV computes it.
    var ev: [2 * max_ecc + 1]u8 = @splat(0);
    for (0..nsyn) |j| {
        if (syn[j] == 0) continue;
        for (0..nsyn) |i| {
            if (C[i] == 0) continue;
            ev[i + j] ^= mul(C[i], syn[j]);
        }
    }
    var xs: [max_ecc]u8 = undefined;
    for (0..nloc) |j| xs[j] = pow2(cw.len - 1 - locs[j]);
    for (0..nloc) |i| {
        const x = xs[i];
        const inv_x = div(1, x);
        var num: u8 = 0;
        for (0..L) |j| num = mul(num, inv_x) ^ ev[L - 1 - j];
        var den: u8 = 1;
        for (0..nloc) |j| {
            if (i == j) continue;
            den = mul(den, 1 ^ mul(inv_x, xs[j]));
        }
        cw[locs[i]] ^= div(num, den);
    }
    return true;
}

test "rs: encode and correct" {
    var data: [20]u8 = undefined;
    for (&data, 0..) |*d, i| d.* = @intCast(i * 7 + 3);
    const g = generator(10);
    var block: [30]u8 = undefined;
    @memcpy(block[0..20], &data);
    rsRemainder(&data, g[0..11], block[20..30]);
    var bad = block;
    bad[3] ^= 0x55;
    bad[17] ^= 0x01;
    bad[25] ^= 0xff;
    try std.testing.expect(rsCorrect(&bad, 10));
    try std.testing.expectEqualSlices(u8, &block, &bad);
    // Six errors exceed the capacity of ten parity bytes.
    for (0..6) |k| bad[k * 4] ^= 0x33;
    _ = rsCorrect(&bad, 10);
}
