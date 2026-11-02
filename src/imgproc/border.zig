//! Border extrapolation, numbered like OpenCV's BORDER_* constants.

const std = @import("std");

pub const Border = enum(u8) {
    /// iiiiii|abcdefgh|iiiiiii with a caller-chosen i.
    constant = 0,
    /// aaaaaa|abcdefgh|hhhhhhh
    replicate = 1,
    /// fedcba|abcdefgh|hgfedcb
    reflect = 2,
    /// cdefgh|abcdefgh|abcdefg
    wrap = 3,
    /// gfedcb|abcdefgh|gfedcba (OpenCV's BORDER_DEFAULT)
    reflect101 = 4,

    pub const default: Border = .reflect101;
};

/// Maps a possibly out-of-range coordinate `p` into [0, len). Returns null
/// for `.constant` when `p` is outside (the caller supplies the value).
pub fn map(p: isize, len: usize, border: Border) ?usize {
    const n: isize = @intCast(len);
    if (p >= 0 and p < n) return @intCast(p);
    switch (border) {
        .constant => return null,
        .replicate => return if (p < 0) 0 else len - 1,
        .reflect, .reflect101 => {
            if (n == 1) return 0;
            const delta: isize = if (border == .reflect101) 1 else 0;
            var q = p;
            while (q < 0 or q >= n) {
                if (q < 0) q = -q - 1 + delta else q = n - 1 - (q - n) - delta;
            }
            return @intCast(q);
        },
        .wrap => return @intCast(@mod(p, n)),
    }
}

test "border map" {
    try std.testing.expectEqual(@as(?usize, 1), map(-1, 5, .reflect101));
    try std.testing.expectEqual(@as(?usize, 0), map(-1, 5, .reflect));
    try std.testing.expectEqual(@as(?usize, 3), map(5, 5, .reflect101));
    try std.testing.expectEqual(@as(?usize, 4), map(5, 5, .reflect));
    try std.testing.expectEqual(@as(?usize, 0), map(5, 5, .wrap));
    try std.testing.expectEqual(@as(?usize, 4), map(9, 5, .replicate));
    try std.testing.expectEqual(@as(?usize, null), map(-1, 5, .constant));
}
