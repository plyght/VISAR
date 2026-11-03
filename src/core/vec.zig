//! Byte-vector helpers for interleaved pixels: split 16 pixels of `cn`
//! channels into per-channel vectors and back, using two-input 16-byte
//! shuffles (pshufb/tbl-sized), which every SIMD target lowers well.

const std = @import("std");
const builtin = @import("builtin");

pub const V16 = @Vector(16, u8);

/// Hides a loaded vector from LLVM's load narrowing, which otherwise turns
/// a 16-byte load feeding byte shuffles into many scalar byte inserts.
pub inline fn opaqueV(v: V16) V16 {
    return switch (builtin.cpu.arch) {
        .x86_64, .x86 => asm ("" : [ret] "=x" (-> V16) : [v] "0" (v)),
        .aarch64, .aarch64_be => asm ("" : [ret] "=w" (-> V16) : [v] "0" (v)),
        else => v,
    };
}

/// Shuffle masks that gather `src(i)` (an index into `nblocks` 16-byte
/// blocks) for each output lane i, as a chain of two-input shuffles:
/// step 0 combines blocks 0 and 1, step j>0 combines the running result
/// with block j+1.
fn chain(comptime nblocks: usize, comptime src: [16]usize) [nblocks - 1][16]i32 {
    @setEvalBranchQuota(10000);
    var masks: [nblocks - 1][16]i32 = undefined;
    for (0..nblocks - 1) |step| {
        for (0..16) |i| {
            const s = src[i];
            const blk = s / 16;
            const off: i32 = @intCast(s % 16);
            masks[step][i] = if (step == 0)
                (if (blk == 0) off else if (blk == 1) ~off else 0)
            else if (blk == step + 1) ~off else @intCast(i);
        }
    }
    return masks;
}

inline fn gather(comptime n: usize, blocks: [n]V16, comptime src: [16]usize) V16 {
    if (n == 1) return @shuffle(u8, blocks[0], undefined, blk: {
        var m: [16]i32 = undefined;
        for (0..16) |i| m[i] = @intCast(src[i]);
        break :blk m;
    });
    const masks = comptime chain(n, src);
    var acc = @shuffle(u8, blocks[0], blocks[1], masks[0]);
    inline for (1..n - 1) |step| acc = @shuffle(u8, acc, blocks[step + 1], masks[step]);
    return acc;
}

/// Loads 16 pixels of `cn` interleaved channels from `p` (16*cn bytes) and
/// returns one vector per channel.
pub inline fn load(comptime cn: usize, p: [*]const u8) [cn]V16 {
    var blocks: [cn]V16 = undefined;
    inline for (0..cn) |b| blocks[b] = opaqueV(p[16 * b ..][0..16].*);
    if (cn == 1) return blocks;
    var out: [cn]V16 = undefined;
    inline for (0..cn) |c| {
        const src = comptime blk: {
            @setEvalBranchQuota(10000);
            var s: [16]usize = undefined;
            for (0..16) |i| s[i] = i * cn + c;
            break :blk s;
        };
        out[c] = gather(cn, blocks, src);
    }
    return out;
}

/// Stores per-channel vectors as 16 interleaved pixels (16*cn bytes).
pub inline fn store(comptime cn: usize, ch: [cn]V16, p: [*]u8) void {
    if (cn == 1) {
        p[0..16].* = ch[0];
        return;
    }
    inline for (0..cn) |b| {
        // Output byte t of block b is channel (16b+t) % cn of pixel (16b+t) / cn.
        const src = comptime blk: {
            @setEvalBranchQuota(10000);
            var s: [16]usize = undefined;
            for (0..16) |t| {
                const g = 16 * b + t;
                s[t] = (g % cn) * 16 + g / cn;
            }
            break :blk s;
        };
        p[16 * b ..][0..16].* = gather(cn, ch, src);
    }
}

test "load/store round trip" {
    inline for (.{ 2, 3, 4 }) |cn| {
        var buf: [16 * cn]u8 = undefined;
        for (&buf, 0..) |*v, i| v.* = @intCast(i);
        const ch = load(cn, &buf);
        for (0..cn) |c| {
            const arr: [16]u8 = ch[c];
            for (arr, 0..) |v, i| try std.testing.expectEqual(@as(u8, @intCast(i * cn + c)), v);
        }
        var out: [16 * cn]u8 = undefined;
        store(cn, ch, &out);
        try std.testing.expectEqualSlices(u8, &buf, &out);
    }
}
