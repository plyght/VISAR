//! calibrateCamera: Zhang's initialisation (OpenCV's
//! initIntrinsicParams2D: principal point at the image centre, focal
//! lengths from the views' vanishing points), per-view extrinsics from
//! solvePnP, then Levenberg-Marquardt over all intrinsics (fx, fy, cx, cy,
//! k1, k2, p1, p2, k3) and every view's pose at once (30 iterations, like
//! OpenCV's CvLevMarq in cvCalibrateCamera2Internal).

const std = @import("std");
const linalg = @import("linalg.zig");
const camera = @import("camera.zig");
const pose = @import("pose.zig");
const homography = @import("homography.zig");
const Allocator = std.mem.Allocator;
const Point2f = homography.Point2f;

pub const Error = Allocator.Error || error{ InvalidArgument, Unsupported };

/// OpenCV's CALIB_* bits that VISAR supports.
pub const Flags = struct {
    use_intrinsic_guess: bool = false,
    fix_principal_point: bool = false,
    zero_tangent_dist: bool = false,
    fix_focal_length: bool = false,
    fix_k1: bool = false,
    fix_k2: bool = false,
    fix_k3: bool = false,

    pub fn fromBits(b: u32) error{Unsupported}!Flags {
        const known: u32 = 1 | 4 | 8 | 16 | 32 | 64 | 128;
        if (b & ~known != 0) return error.Unsupported;
        return .{
            .use_intrinsic_guess = b & 1 != 0,
            .fix_principal_point = b & 4 != 0,
            .zero_tangent_dist = b & 8 != 0,
            .fix_focal_length = b & 16 != 0,
            .fix_k1 = b & 32 != 0,
            .fix_k2 = b & 64 != 0,
            .fix_k3 = b & 128 != 0,
        };
    }
};

pub const View = struct {
    obj: []const [3]f64,
    img: []const [2]f64,
};

pub const Result = struct {
    rms: f64,
    K: [9]f64,
    dist: [5]f64,
};

/// OpenCV's initIntrinsicParams2D (aspect ratio free).
fn initIntrinsics(gpa: Allocator, views: []const View, w: u32, h: u32) Error![9]f64 {
    var a = [9]f64{ 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    a[2] = if (w == 0) 0.5 else (@as(f64, @floatFromInt(w)) - 1) * 0.5;
    a[5] = if (h == 0) 0.5 else (@as(f64, @floatFromInt(h)) - 1) * 0.5;
    // Normal equations of the 2n x 2 system A f = b.
    var ata = [4]f64{ 0, 0, 0, 0 };
    var atb = [2]f64{ 0, 0 };
    for (views) |v| {
        const n = v.obj.len;
        const buf = try gpa.alloc(Point2f, 2 * n);
        defer gpa.free(buf);
        for (v.obj, v.img, 0..) |o, im, i| {
            buf[i] = .{ .x = @floatCast(o[0]), .y = @floatCast(o[1]) };
            buf[n + i] = .{ .x = @floatCast(im[0]), .y = @floatCast(im[1]) };
        }
        var H = (try homography.findHomography(gpa, buf[0..n], buf[n..], null, .{})) orelse return error.InvalidArgument;
        H[0] -= H[6] * a[2];
        H[1] -= H[7] * a[2];
        H[2] -= H[8] * a[2];
        H[3] -= H[6] * a[5];
        H[4] -= H[7] * a[5];
        H[5] -= H[8] * a[5];
        var hh: [3]f64 = undefined;
        var vv: [3]f64 = undefined;
        var d1: [3]f64 = undefined;
        var d2: [3]f64 = undefined;
        var nn = [4]f64{ 0, 0, 0, 0 };
        for (0..3) |j| {
            const t0 = H[j * 3];
            const t1 = H[j * 3 + 1];
            hh[j] = t0;
            vv[j] = t1;
            d1[j] = (t0 + t1) * 0.5;
            d2[j] = (t0 - t1) * 0.5;
            nn[0] += t0 * t0;
            nn[1] += t1 * t1;
            nn[2] += d1[j] * d1[j];
            nn[3] += d2[j] * d2[j];
        }
        for (&nn) |*x| x.* = 1 / @sqrt(x.*);
        for (0..3) |j| {
            hh[j] *= nn[0];
            vv[j] *= nn[1];
            d1[j] *= nn[2];
            d2[j] *= nn[3];
        }
        const rows = [2][2]f64{ .{ hh[0] * vv[0], hh[1] * vv[1] }, .{ d1[0] * d2[0], d1[1] * d2[1] } };
        const bs = [2]f64{ -hh[2] * vv[2], -d1[2] * d2[2] };
        for (rows, bs) |r, b| {
            ata[0] += r[0] * r[0];
            ata[1] += r[0] * r[1];
            ata[3] += r[1] * r[1];
            atb[0] += r[0] * b;
            atb[1] += r[1] * b;
        }
    }
    ata[2] = ata[1];
    var f: [2]f64 = undefined;
    linalg.solveSymEig(&ata, 2, &atb, &f);
    a[0] = @sqrt(@abs(1 / f[0]));
    a[4] = @sqrt(@abs(1 / f[1]));
    return a;
}

const Problem = struct {
    views: []const View,
    nint: usize = 9,

    /// Parameters: fx fy cx cy k1 k2 p1 p2 k3, then (rvec, tvec) per view.
    fn eval(p: Problem, x: []const f64, a: ?[]f64, v: ?[]f64) f64 {
        const np = x.len;
        const ni = p.nint;
        if (a) |aa| {
            @memset(aa, 0);
            @memset(v.?, 0);
        }
        var k: camera.Dist = @splat(0);
        k[0..5].* = x[4..9].*;
        var s: f64 = 0;
        for (p.views, 0..) |view, vi| {
            const base = ni + vi * 6;
            const ps = camera.Pose.init(x[base..][0..3].*, x[base + 3 ..][0..3].*, a != null);
            for (view.obj, view.img) |X, m| {
                var J: [2][camera.jac_cols]f64 = undefined;
                const pr = camera.project(&ps, X, x[0], x[1], x[2], x[3], k, if (a != null) &J else null);
                const e = [2]f64{ pr[0] - m[0], pr[1] - m[1] };
                s += e[0] * e[0] + e[1] * e[1];
                const aa = a orelse continue;
                const vv = v.?;
                // Columns: intrinsics 6..14 of J map to params 0..8; the
                // pose columns 0..5 map to base..base+5.
                var idx: [15]usize = undefined;
                for (0..9) |c| idx[c] = c;
                for (0..6) |c| idx[9 + c] = base + c;
                var col: [2][15]f64 = undefined;
                for (0..2) |r| {
                    for (0..9) |c| col[r][c] = J[r][6 + c];
                    for (0..6) |c| col[r][9 + c] = J[r][c];
                }
                for (0..15) |i| {
                    const gi = idx[i];
                    vv[gi] += col[0][i] * e[0] + col[1][i] * e[1];
                    for (i..15) |j| {
                        const gj = idx[j];
                        aa[gi * np + gj] += col[0][i] * col[0][j] + col[1][i] * col[1][j];
                    }
                }
            }
        }
        if (a) |aa| for (0..np) |i| for (0..i) |j| {
            // Entries were added to (min, max) index order: mirror.
            aa[i * np + j] = aa[j * np + i];
        };
        return s;
    }

    pub fn normal(p: Problem, x: []const f64, a: []f64, v: []f64) f64 {
        return p.eval(x, a, v);
    }

    pub fn cost(p: Problem, x: []const f64) f64 {
        return p.eval(x, null, null);
    }
};

/// OpenCV's calibrateCamera for planar targets (object points with z = 0
/// unless `flags.use_intrinsic_guess`). `K` and `dist` are read as the
/// initial guess with use_intrinsic_guess. `rvecs`/`tvecs` (one per view)
/// receive the poses.
pub fn calibrateCamera(gpa: Allocator, views: []const View, width: u32, height: u32, K_guess: [9]f64, dist_guess: [5]f64, flags: Flags, rvecs: [][3]f64, tvecs: [][3]f64) Error!Result {
    const nv = views.len;
    if (nv == 0 or rvecs.len < nv or tvecs.len < nv) return error.InvalidArgument;
    var total: usize = 0;
    for (views) |v| {
        if (v.obj.len != v.img.len or v.obj.len < 4) return error.InvalidArgument;
        total += v.obj.len;
    }
    var K: [9]f64 = undefined;
    var dist: [5]f64 = undefined;
    if (flags.use_intrinsic_guess) {
        K = K_guess;
        dist = dist_guess;
        if (K[0] <= 0 or K[4] <= 0) return error.InvalidArgument;
    } else {
        for (views) |v| for (v.obj) |o| {
            if (o[2] != 0) return error.InvalidArgument;
        };
        K = try initIntrinsics(gpa, views, width, height);
        dist = @splat(0);
    }
    if (flags.zero_tangent_dist) {
        dist[2] = 0;
        dist[3] = 0;
    }
    const np = 9 + 6 * nv;
    const x = try gpa.alloc(f64, np);
    defer gpa.free(x);
    x[0] = K[0];
    x[1] = K[4];
    x[2] = K[2];
    x[3] = K[5];
    @memcpy(x[4..9], &dist);
    for (views, 0..) |v, i| {
        const r = try pose.solvePnP(gpa, v.obj, v.img, K, &dist, .iterative, null);
        x[9 + 6 * i ..][0..3].* = r.rvec;
        x[9 + 6 * i + 3 ..][0..3].* = r.tvec;
    }
    const mask = try gpa.alloc(bool, np);
    defer gpa.free(mask);
    @memset(mask, true);
    if (flags.fix_focal_length) {
        mask[0] = false;
        mask[1] = false;
    }
    if (flags.fix_principal_point) {
        mask[2] = false;
        mask[3] = false;
    }
    if (flags.fix_k1) mask[4] = false;
    if (flags.fix_k2) mask[5] = false;
    if (flags.zero_tangent_dist) {
        mask[6] = false;
        mask[7] = false;
    }
    if (flags.fix_k3) mask[8] = false;
    const work = try gpa.alloc(f64, 3 * np * np + 4 * np);
    defer gpa.free(work);
    const prob: Problem = .{ .views = views };
    linalg.levMarq(np, x, 30, std.math.floatEps(f64), work, mask, prob);
    const err = prob.cost(x);
    for (0..nv) |i| {
        rvecs[i] = x[9 + 6 * i ..][0..3].*;
        tvecs[i] = x[9 + 6 * i + 3 ..][0..3].*;
    }
    return .{
        .rms = @sqrt(err / @as(f64, @floatFromInt(total))),
        .K = .{ x[0], 0, x[2], 0, x[1], x[3], 0, 0, 1 },
        .dist = x[4..9].*,
    };
}

test "calibrateCamera recovers synthetic intrinsics" {
    const gpa = std.testing.allocator;
    const K = [9]f64{ 820, 0, 640, 0, 800, 360, 0, 0, 1 };
    const dist = [5]f64{ -0.2, 0.05, 0.001, -0.0005, 0 };
    var obj: [54][3]f64 = undefined;
    for (0..6) |i| for (0..9) |j| {
        obj[i * 9 + j] = .{ @floatFromInt(j), @floatFromInt(i), 0 };
    };
    const poses = [_][6]f64{
        .{ 0.1, 0.2, 0.05, -4, -2.5, 14 },
        .{ -0.3, 0.1, 0.2, -3, -3, 12 },
        .{ 0.2, -0.35, -0.1, -5, -2, 15 },
        .{ 0.4, 0.3, 0.3, -4, -3, 13 },
        .{ -0.2, -0.3, -0.2, -3.5, -2, 16 },
    };
    var imgs: [5][54][2]f64 = undefined;
    var views: [5]View = undefined;
    for (poses, 0..) |p, i| {
        try camera.projectPoints(&obj, p[0..3].*, p[3..6].*, K, &dist, &imgs[i]);
        views[i] = .{ .obj = &obj, .img = &imgs[i] };
    }
    var rv: [5][3]f64 = undefined;
    var tv: [5][3]f64 = undefined;
    const r = try calibrateCamera(gpa, &views, 1280, 720, undefined, undefined, .{}, &rv, &tv);
    try std.testing.expect(r.rms < 1e-6);
    for ([_]usize{ 0, 2, 4, 5 }) |i| try std.testing.expectApproxEqAbs(K[i], r.K[i], 1e-4);
    for (0..5) |i| try std.testing.expectApproxEqAbs(dist[i], r.dist[i], 1e-6);
}
