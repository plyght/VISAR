//! ArUco markers, OpenCV's cv::aruco (objdetect, 5.0): the predefined
//! dictionaries, generateImageMarker, ArucoDetector::detectMarkers and
//! drawDetectedMarkers.
//!
//! Detection follows OpenCV step for step, so ids and corners match cv2:
//! adaptive mean thresholds at each window size, contours (LIST, NONE),
//! polygon approximation and convexity / size filters, clockwise corner
//! order, grouping of near candidates (OpenCV's candidate tree), bit
//! extraction through a nearest-neighbour perspective warp reproducing
//! OpenCV 5's float warp kernels, Otsu binarisation, per-cell white ratios,
//! border checks, dictionary lookup with error correction, and optional
//! cornerSubPix refinement. Dictionary lookup uses bit masks (equivalent to
//! OpenCV's per-cell comparison, but word-parallel).

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;
const core = @import("../core/core.zig");
const Image = core.Image;
const Exec = core.Exec;
const sat = core.sat;
const shape = @import("../imgproc/shape.zig");
const contours_mod = @import("../imgproc/contours.zig");
const threshold_mod = @import("../imgproc/threshold.zig");
const color = @import("../imgproc/color.zig");
const resize_mod = @import("../imgproc/resize.zig");
const polygon = @import("../imgproc/polygon.zig");
const draw = @import("../imgproc/draw.zig");
const subpix = @import("../imgproc/subpix.zig");
const geometry = @import("../imgproc/geometry.zig");
const homography = @import("../calib3d/homography.zig");
const dict_tables = @import("aruco_dict.zig");

pub const Point2f = shape.Point2f;
const Point = contours_mod.Point;

pub const Error = Allocator.Error || core.Error || error{ Unsupported, InvalidArgument, InvalidKernel };

// ---------------------------------------------------------- dictionaries

/// cv::aruco::PredefinedDictionaryType.
pub const PredefinedDictionary = enum(u8) {
    DICT_4X4_50 = 0,
    DICT_4X4_100,
    DICT_4X4_250,
    DICT_4X4_1000,
    DICT_5X5_50,
    DICT_5X5_100,
    DICT_5X5_250,
    DICT_5X5_1000,
    DICT_6X6_50,
    DICT_6X6_100,
    DICT_6X6_250,
    DICT_6X6_1000,
    DICT_7X7_50,
    DICT_7X7_100,
    DICT_7X7_250,
    DICT_7X7_1000,
    DICT_ARUCO_ORIGINAL,
    DICT_APRILTAG_16h5,
    DICT_APRILTAG_25h9,
    DICT_APRILTAG_36h10,
    DICT_APRILTAG_36h11,
    DICT_ARUCO_MIP_36h12,
};

pub const Dictionary = struct {
    /// Each marker's bits (rotation 0): cell (r, c) is bit n*n-1-(r*n+c).
    bits: []const u64,
    marker_size: u8,
    max_correction_bits: u8,

    pub fn count(d: Dictionary) usize {
        return d.bits.len;
    }

    /// The marker's bits rotated like OpenCV's byte list rotation `rot`
    /// (0..3, 90 degrees anticlockwise each).
    pub fn rotated(d: Dictionary, id: usize, rot: u2) u64 {
        const n: usize = d.marker_size;
        const v = d.bits[id];
        if (rot == 0) return v;
        var out: u64 = 0;
        for (0..n) |r| for (0..n) |c| {
            const sr, const sc = switch (rot) {
                1 => .{ c, n - 1 - r },
                2 => .{ n - 1 - r, n - 1 - c },
                3 => .{ n - 1 - c, r },
                0 => unreachable,
            };
            const bit = (v >> @intCast(n * n - 1 - (sr * n + sc))) & 1;
            out |= bit << @intCast(n * n - 1 - (r * n + c));
        };
        return out;
    }

    /// Cell (r, c) of marker `id` (rotation 0).
    pub fn cell(d: Dictionary, id: usize, r: usize, c: usize) u1 {
        const n: usize = d.marker_size;
        return @intCast((d.bits[id] >> @intCast(n * n - 1 - (r * n + c))) & 1);
    }
};

/// cv::aruco::getPredefinedDictionary.
pub fn getPredefinedDictionary(name: PredefinedDictionary) Dictionary {
    const e = dict_tables.predefined[@backingInt(name)];
    return .{ .bits = e.bits[0..e.count], .marker_size = e.size, .max_correction_bits = e.max_correction };
}

// ------------------------------------------------------------ generation

/// Dictionary::generateImageMarker into `dst` (side x side, one channel):
/// black border of `border_bits` cells, then the marker's bits, scaled
/// with nearest-neighbour interpolation.
pub fn generateImageMarker(gpa: Allocator, d: Dictionary, id: usize, border_bits: u32, dst: Image(u8)) Error!void {
    const n: u32 = d.marker_size;
    const ts = n + 2 * border_bits;
    if (border_bits == 0 or id >= d.bits.len or dst.width != dst.height or dst.width < ts or dst.channels != 1) return error.InvalidArgument;
    var tiny = try Image(u8).initFill(gpa, ts, ts, 1, 0);
    defer tiny.deinit(gpa);
    for (0..n) |r| for (0..n) |c| {
        tiny.data[(r + border_bits) * tiny.stride + c + border_bits] = @as(u8, d.cell(id, r, c)) * 255;
    };
    resize_mod.resize(Exec.serial, gpa, tiny, dst, .nearest) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidArgument,
    };
}

// ------------------------------------------------------------- detection

pub const CornerRefine = enum(u8) { none = 0, subpix = 1, contour = 2, apriltag = 3 };

/// cv::aruco::DetectorParameters (the defaults are OpenCV's).
pub const DetectorParameters = struct {
    adaptive_thresh_win_size_min: i32 = 3,
    adaptive_thresh_win_size_max: i32 = 23,
    adaptive_thresh_win_size_step: i32 = 10,
    adaptive_thresh_constant: f64 = 7,
    min_marker_perimeter_rate: f64 = 0.03,
    max_marker_perimeter_rate: f64 = 4.0,
    polygonal_approx_accuracy_rate: f64 = 0.03,
    min_corner_distance_rate: f64 = 0.05,
    min_distance_to_border: i32 = 3,
    min_marker_distance_rate: f64 = 0.125,
    min_group_distance: f32 = 0.21,
    corner_refinement_method: CornerRefine = .none,
    corner_refinement_win_size: i32 = 5,
    relative_corner_refinement_win_size: f32 = 0.3,
    corner_refinement_max_iterations: i32 = 30,
    corner_refinement_min_accuracy: f64 = 0.1,
    marker_border_bits: i32 = 1,
    perspective_remove_pixel_per_cell: i32 = 4,
    perspective_remove_ignored_margin_per_cell: f64 = 0.13,
    max_erroneous_bits_in_border_rate: f64 = 0.35,
    min_otsu_std_dev: f64 = 5.0,
    error_correction_rate: f64 = 0.6,
    valid_bit_id_threshold: f32 = 0.49,
    detect_inverted_marker: bool = false,
};

pub const Markers = struct {
    corners: [][4]Point2f,
    ids: []i32,
    rejected: [][4]Point2f,

    pub fn deinit(m: *Markers, gpa: Allocator) void {
        gpa.free(m.corners);
        gpa.free(m.ids);
        gpa.free(m.rejected);
        m.* = undefined;
    }
};

const Candidate = struct {
    corners: [4]Point2f,
    /// Index of the contour in the detector's contour store.
    contour: u32,
    perimeter: f32 = 0,
    parent: i32 = -1,
    depth: i32 = 0,
    /// Range of close contours in the shared list.
    close_start: u32 = 0,
    close_len: u32 = 0,
};

fn normSq(a: Point2f, b: Point2f) f32 {
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return dx * dx + dy * dy;
}

fn perimeterOf(c: [4]Point2f) f32 {
    var p: f32 = 0;
    for (0..4) |i| p += @sqrt(normSq(c[i], c[(i + 1) % 4]));
    return p;
}

fn averageModuleSize(c: [4]Point2f, marker_size: i32, border_bits: i32) f32 {
    var s: f32 = 0;
    for (0..4) |i| s += @sqrt(normSq(c[i], c[(i + 1) % 4]));
    const nm: f32 = @floatFromInt(marker_size + border_bits * 2);
    return s / (4 * nm);
}

fn averageDistance(a: [4]Point2f, b: [4]Point2f) f32 {
    var min_sq: f32 = std.math.floatMax(f32);
    for (0..4) |fc| {
        var d: f32 = 0;
        for (0..4) |c| d += normSq(a[(c + fc) % 4], b[c]);
        d /= 4;
        min_sq = @min(min_sq, d);
    }
    return @sqrt(min_sq);
}

/// A store of contours (points back to back).
const ContourStore = struct {
    pts: std.ArrayList(Point) = .empty,
    starts: std.ArrayList(u32) = .empty,

    fn add(s: *ContourStore, gpa: Allocator, c: []const Point) Allocator.Error!u32 {
        if (s.starts.items.len == 0) try s.starts.append(gpa, 0);
        try s.pts.appendSlice(gpa, c);
        try s.starts.append(gpa, @intCast(s.pts.items.len));
        return @intCast(s.starts.items.len - 2);
    }

    fn get(s: *const ContourStore, i: u32) []const Point {
        return s.pts.items[s.starts.items[i]..s.starts.items[i + 1]];
    }

    fn deinit(s: *ContourStore, gpa: Allocator) void {
        s.pts.deinit(gpa);
        s.starts.deinit(gpa);
    }
};

/// _findMarkerContours on one thresholded image.
fn findMarkerContours(gpa: Allocator, bin: Image(u8), p: DetectorParameters, out: *std.ArrayList(Candidate), store: *ContourStore, approx_buf: *std.ArrayList(Point)) Error!void {
    const maxdim: f64 = @floatFromInt(@max(bin.width, bin.height));
    const min_perim: u64 = @intFromFloat(p.min_marker_perimeter_rate * maxdim);
    const max_perim: u64 = @intFromFloat(p.max_marker_perimeter_rate * maxdim);
    var cs = try contours_mod.findContours(gpa, bin, .list, .none, .{ .x = 0, .y = 0 });
    defer cs.deinit(gpa);
    for (0..cs.len()) |i| {
        const c = cs.get(i);
        if (c.len < min_perim or c.len > max_perim) continue;
        try approx_buf.resize(gpa, c.len);
        const n = shape.approxPolyDP(gpa, c, approx_buf.items, @as(f64, @floatFromInt(c.len)) * p.polygonal_approx_accuracy_rate, true) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            else => continue,
        };
        if (n != 4) continue;
        const ap = approx_buf.items[0..4];
        if (!shape.isContourConvex(ap)) continue;
        var min_sq: f64 = maxdim * maxdim;
        for (0..4) |j| {
            const dx: f64 = @floatFromInt(ap[j].x - ap[(j + 1) % 4].x);
            const dy: f64 = @floatFromInt(ap[j].y - ap[(j + 1) % 4].y);
            min_sq = @min(min_sq, dx * dx + dy * dy);
        }
        const min_dist = @as(f64, @floatFromInt(c.len)) * p.min_corner_distance_rate;
        if (min_sq < min_dist * min_dist) continue;
        var cand: Candidate = .{ .corners = undefined, .contour = try store.add(gpa, c) };
        for (0..4) |j| cand.corners[j] = .{ .x = @floatFromInt(ap[j].x), .y = @floatFromInt(ap[j].y) };
        try out.append(gpa, cand);
    }
}

/// OpenCV's filterTooCloseCandidates: groups candidates whose corners are
/// close, keeps one per group (the largest, or the smallest when inverted
/// markers are searched) with the others as fall-backs, drops those near
/// the image border and builds the containment tree.
fn filterTooClose(gpa: Allocator, w: u32, h: u32, cands: []Candidate, p: DetectorParameters, marker_size: i32, closes: *std.ArrayList(Candidate)) Error![]Candidate {
    for (cands) |*c| c.perimeter = perimeterOf(c.corners);
    // stable_sort by perimeter, descending.
    std.mem.sort(Candidate, cands, {}, struct {
        fn lt(_: void, a: Candidate, b: Candidate) bool {
            return a.perimeter > b.perimeter;
        }
    }.lt);
    const n = cands.len;
    const group_id = try gpa.alloc(i32, n);
    defer gpa.free(group_id);
    @memset(group_id, -1);
    const selected = try gpa.alloc(bool, n);
    defer gpa.free(selected);
    @memset(selected, true);
    var groups: std.ArrayList(std.ArrayList(u32)) = .empty;
    defer {
        for (groups.items) |*g| g.deinit(gpa);
        groups.deinit(gpa);
    }
    const rate: f32 = @floatCast(p.min_marker_distance_rate);
    for (0..n) |i| {
        for (i + 1..n) |j| {
            const md = averageDistance(cands[i].corners, cands[j].corners);
            if (md < cands[j].perimeter * rate) {
                selected[i] = false;
                selected[j] = false;
                if (group_id[i] < 0 and group_id[j] < 0) {
                    group_id[i] = @intCast(groups.items.len);
                    group_id[j] = @intCast(groups.items.len);
                    var g: std.ArrayList(u32) = .empty;
                    try g.append(gpa, @intCast(i));
                    try g.append(gpa, @intCast(j));
                    try groups.append(gpa, g);
                } else if (group_id[i] > -1 and group_id[j] == -1) {
                    group_id[j] = group_id[i];
                    try groups.items[@intCast(group_id[i])].append(gpa, @intCast(j));
                } else if (group_id[j] > -1 and group_id[i] == -1) {
                    group_id[i] = group_id[j];
                    try groups.items[@intCast(group_id[j])].append(gpa, @intCast(i));
                }
            }
        }
        if (selected[i]) {
            selected[i] = false;
            group_id[i] = @intCast(groups.items.len);
            var g: std.ArrayList(u32) = .empty;
            try g.append(gpa, @intCast(i));
            try groups.append(gpa, g);
        }
    }
    // Close contours, attached to the group's first member.
    var close_lists = try gpa.alloc(std.ArrayList(Candidate), n);
    defer {
        for (close_lists) |*l| l.deinit(gpa);
        gpa.free(close_lists);
    }
    for (close_lists) |*l| l.* = .empty;
    const mdb: f32 = @floatFromInt(p.min_distance_to_border);
    const wf: f32 = @floatFromInt(w);
    const hf: f32 = @floatFromInt(h);
    for (groups.items) |*g| {
        const items = g.items;
        if (p.detect_inverted_marker) {
            std.mem.sort(u32, items, {}, std.sort.desc(u32));
        } else {
            std.mem.sort(u32, items, {}, std.sort.asc(u32));
        }
        var curr = items[0];
        var near_border = false;
        for (cands[curr].corners) |c| {
            if (c.x < mdb or c.y < mdb or c.x > wf - 1 - mdb or c.y > hf - 1 - mdb) {
                near_border = true;
                break;
            }
        }
        if (near_border) continue;
        selected[curr] = true;
        for (items[1..]) |id| {
            const dist = averageDistance(cands[id].corners, cands[curr].corners);
            const ms = averageModuleSize(cands[id].corners, marker_size, p.marker_border_bits);
            if (dist > p.min_group_distance * ms) {
                curr = id;
                try close_lists[items[0]].append(gpa, cands[id]);
            }
        }
    }
    var sel: std.ArrayList(Candidate) = .empty;
    errdefer sel.deinit(gpa);
    for (0..n) |i| {
        if (!selected[i]) continue;
        var c = cands[i];
        c.close_start = @intCast(closes.items.len);
        c.close_len = @intCast(close_lists[i].items.len);
        try closes.appendSlice(gpa, close_lists[i].items);
        try sel.append(gpa, c);
    }
    // Hierarchy: a candidate's parent is the nearest earlier (larger)
    // candidate containing all its corners.
    const s = sel.items;
    var i: isize = @as(isize, @intCast(s.len)) - 1;
    while (i >= 0) : (i -= 1) {
        const iu: usize = @intCast(i);
        var j = i - 1;
        while (j >= 0) : (j -= 1) {
            const ju: usize = @intCast(j);
            if (inside(s[iu].corners, s[ju].corners)) {
                s[iu].parent = @intCast(j);
                s[ju].depth = @max(s[ju].depth, s[iu].depth + 1);
                break;
            }
        }
    }
    return sel.toOwnedSlice(gpa);
}

fn inside(m1: [4]Point2f, m2: [4]Point2f) bool {
    for (m1) |pt| if (shape.pointPolygonTest(&m2, pt, false) < 0) return false;
    return true;
}

/// The nearest-neighbour inverse-mapped coordinates of OpenCV 5's
/// warpPerspective kernel: f32 matrix, fma along the row and a division,
/// rounded half to even; the last (width % lanes) pixels of each row take
/// the scalar formula.
const warp_lanes: usize = switch (builtin.cpu.arch) {
    .x86_64 => if (std.Target.x86.featureSetHas(builtin.cpu.features, .avx512f)) 32 else 16,
    .aarch64 => 8,
    else => 8,
};

fn warpNearest(src: Image(u8), dst: []u8, side: usize, minv: [9]f64) void {
    var m: [9]f32 = undefined;
    for (&m, minv) |*o, v| o.* = @floatCast(v);
    const vec_end = side - side % warp_lanes;
    for (0..side) |y| {
        const yf: f32 = @floatFromInt(y);
        const mx = yf * m[1] + m[2];
        const my = yf * m[4] + m[5];
        const mw = yf * m[7] + m[8];
        for (0..side) |x| {
            const xf: f32 = @floatFromInt(x);
            var sx: f32 = undefined;
            var sy: f32 = undefined;
            if (x < vec_end) {
                const w = @mulAdd(f32, m[6], xf, mw);
                sx = @mulAdd(f32, m[0], xf, mx) / w;
                sy = @mulAdd(f32, m[3], xf, my) / w;
            } else {
                const w = xf * m[6] + yf * m[7] + m[8];
                sx = (xf * m[0] + yf * m[1] + m[2]) / w;
                sy = (xf * m[3] + yf * m[4] + m[5]) / w;
            }
            const rx = sat.rint(std.math.clamp(sx, -1e9, 1e9));
            const ry = sat.rint(std.math.clamp(sy, -1e9, 1e9));
            var v: u8 = 0;
            if (rx >= 0 and ry >= 0 and rx < @as(f32, @floatFromInt(src.width)) and ry < @as(f32, @floatFromInt(src.height))) {
                const ix: usize = @intFromFloat(rx);
                const iy: usize = @intFromFloat(ry);
                v = src.data[iy * src.stride + ix];
            }
            dst[y * side + x] = v;
        }
    }
}

const Scratch = struct {
    warp: std.ArrayList(u8) = .empty,
    ratio: std.ArrayList(f32) = .empty,

    fn deinit(s: *Scratch, gpa: Allocator) void {
        s.warp.deinit(gpa);
        s.ratio.deinit(gpa);
    }
};

/// _extractCellPixelRatio into s.ratio (msb x msb).
fn extractCellPixelRatio(gpa: Allocator, img: Image(u8), corners: [4]Point2f, marker_size: i32, p: DetectorParameters, s: *Scratch) Error!bool {
    const msb: usize = @intCast(marker_size + 2 * p.marker_border_bits);
    const cs: usize = @intCast(p.perspective_remove_pixel_per_cell);
    const margin: usize = @intFromFloat(p.perspective_remove_ignored_margin_per_cell * @as(f64, @floatFromInt(cs)));
    const side = msb * cs;
    const sf: f32 = @floatFromInt(side - 1);
    const dst_c = [4]Point2f{ .{ .x = 0, .y = 0 }, .{ .x = sf, .y = 0 }, .{ .x = sf, .y = sf }, .{ .x = 0, .y = sf } };
    const t = homography.getPerspectiveTransform(corners, dst_c) orelse return false;
    const inv = geometry.invert3(t) orelse return false;
    try s.warp.resize(gpa, side * side);
    try s.ratio.resize(gpa, msb * msb);
    const wimg = s.warp.items;
    warpNearest(img, wimg, side, inv);
    // meanStdDev over the inner region.
    const a = cs / 2;
    const b = side - cs / 2;
    var sum: u64 = 0;
    var sq: u64 = 0;
    for (a..b) |y| for (a..b) |x| {
        const v: u64 = wimg[y * side + x];
        sum += v;
        sq += v * v;
    };
    const cnt: f64 = @floatFromInt((b - a) * (b - a));
    const mean = @as(f64, @floatFromInt(sum)) / cnt;
    const variance = @max(@as(f64, @floatFromInt(sq)) / cnt - mean * mean, 0);
    if (@sqrt(variance) < p.min_otsu_std_dev) {
        @memset(s.ratio.items, if (mean > 127) 1 else 0);
        return true;
    }
    // Otsu over the whole warped image.
    const wv = Image(u8).fromSlice(wimg, @intCast(side), @intCast(side), 1) catch return false;
    _ = threshold_mod.threshold(Exec.serial, wv, wv, 125, 255, .binary, .otsu) catch return false;
    const inner = cs - 2 * margin;
    const total: f32 = @floatFromInt(inner * inner);
    for (0..msb) |cy| for (0..msb) |cx| {
        var nz: usize = 0;
        for (0..inner) |yy| {
            const row = wimg[(cy * cs + margin + yy) * side + cx * cs + margin ..][0..inner];
            for (row) |v| nz += @intFromBool(v != 0);
        }
        s.ratio.items[cy * msb + cx] = @as(f32, @floatFromInt(nz)) / total;
    };
    return true;
}

fn borderErrors(ratio: []const f32, msb: usize, bb: usize, thr: f32, invert: bool) i32 {
    var e: i32 = 0;
    const f = struct {
        inline fn bad(r: f32, t: f32, inv: bool) bool {
            return (if (inv) 1.0 - r else r) > t;
        }
    }.bad;
    for (0..msb) |y| for (0..bb) |k| {
        e += @intFromBool(f(ratio[y * msb + k], thr, invert));
        e += @intFromBool(f(ratio[y * msb + msb - 1 - k], thr, invert));
    };
    for (bb..msb - bb) |x| for (0..bb) |k| {
        e += @intFromBool(f(ratio[k * msb + x], thr, invert));
        e += @intFromBool(f(ratio[(msb - 1 - k) * msb + x], thr, invert));
    };
    return e;
}

const Identified = struct { id: i32, rotation: u2 };

/// _identifyOneCandidate: the marker id and rotation, or null.
fn identifyOne(gpa: Allocator, img: Image(u8), corners: [4]Point2f, d: Dictionary, rots: []const u64, p: DetectorParameters, s: *Scratch) Error!?Identified {
    const n: i32 = d.marker_size;
    if (!try extractCellPixelRatio(gpa, img, corners, n, p, s)) return null;
    const bb: usize = @intCast(p.marker_border_bits);
    const nu: usize = @intCast(n);
    const msb = nu + 2 * bb;
    const ratio = s.ratio.items;
    const max_border: i32 = @intFromFloat(@as(f64, @floatFromInt(n * n)) * p.max_erroneous_bits_in_border_rate);
    var errs = borderErrors(ratio, msb, bb, p.valid_bit_id_threshold, false);
    if (p.detect_inverted_marker) {
        const inv_errs = borderErrors(ratio, msb, bb, p.valid_bit_id_threshold, true);
        if (inv_errs < errs) {
            errs = inv_errs;
            for (ratio) |*r| r.* = 1.0 - r.*;
        }
    }
    if (errs > max_border) return null;
    // Bit masks of the inner cells: not0 (|r - 0| > t), not1 (|r - 1| > t).
    var not0: u64 = 0;
    var not1: u64 = 0;
    const t = p.valid_bit_id_threshold;
    for (0..nu) |r| for (0..nu) |c| {
        const v = ratio[(r + bb) * msb + c + bb];
        not0 = (not0 << 1) | @intFromBool(@abs(v) > t);
        not1 = (not1 << 1) | @intFromBool(@abs(v - 1.0) > t);
    };
    const max_corr: i32 = @intFromFloat(@as(f64, @floatFromInt(d.max_correction_bits)) * p.error_correction_rate);
    for (0..d.bits.len) |m| {
        var best: i32 = n * n + 1;
        var best_rot: u2 = 0;
        for (0..4) |r| {
            const bits = rots[m * 4 + r];
            const ham: i32 = @popCount((not0 & ~bits) | (not1 & bits));
            if (ham < best) {
                best = ham;
                best_rot = @intCast(r);
            }
        }
        if (best <= max_corr) return .{ .id = @intCast(m), .rotation = best_rot };
    }
    return null;
}

/// ArucoDetector(dictionary, params).detectMarkers(image) on an 8-bit
/// gray, BGR or BGRA image. Free the result with `deinit`.
pub fn detectMarkers(gpa: Allocator, image: Image(u8), d: Dictionary, p: DetectorParameters) Error!Markers {
    if (p.adaptive_thresh_win_size_min < 3 or p.adaptive_thresh_win_size_max < 3 or
        p.adaptive_thresh_win_size_max < p.adaptive_thresh_win_size_min or p.adaptive_thresh_win_size_step <= 0 or
        p.marker_border_bits <= 0 or p.perspective_remove_pixel_per_cell <= 0 or
        !(p.perspective_remove_ignored_margin_per_cell >= 0 and p.perspective_remove_ignored_margin_per_cell <= 0.5) or
        !(p.min_marker_perimeter_rate > 0 and p.max_marker_perimeter_rate > 0 and p.polygonal_approx_accuracy_rate > 0) or
        p.min_corner_distance_rate < 0 or p.min_marker_distance_rate < 0 or p.min_distance_to_border < 0)
        return error.InvalidArgument;
    if (p.corner_refinement_method == .contour or p.corner_refinement_method == .apriltag) return error.Unsupported;

    var grey = image;
    var owned: ?Image(u8) = null;
    defer if (owned) |*im| im.deinit(gpa);
    switch (image.channels) {
        1 => {},
        3, 4 => {
            owned = try Image(u8).init(gpa, image.width, image.height, 1);
            try color.cvtColor(Exec.serial, image, owned.?, if (image.channels == 3) .bgr2gray else .bgra2gray);
            grey = owned.?;
        },
        else => return error.Unsupported,
    }

    var store: ContourStore = .{};
    defer store.deinit(gpa);
    var cands: std.ArrayList(Candidate) = .empty;
    defer cands.deinit(gpa);
    var approx_buf: std.ArrayList(Point) = .empty;
    defer approx_buf.deinit(gpa);
    {
        var thr = try Image(u8).init(gpa, grey.width, grey.height, 1);
        defer thr.deinit(gpa);
        const nscales: usize = @intCast(@divTrunc(p.adaptive_thresh_win_size_max - p.adaptive_thresh_win_size_min, p.adaptive_thresh_win_size_step) + 1);
        for (0..nscales) |i| {
            var win: u32 = @intCast(p.adaptive_thresh_win_size_min + @as(i32, @intCast(i)) * p.adaptive_thresh_win_size_step);
            if (win % 2 == 0) win += 1;
            try threshold_mod.adaptiveThreshold(Exec.serial, gpa, grey, thr, 255, .mean, .binary_inv, win, p.adaptive_thresh_constant);
            try findMarkerContours(gpa, thr, p, &cands, &store, &approx_buf);
        }
    }
    // _reorderCandidatesCorners: clockwise.
    for (cands.items) |*c| {
        const dx1: f64 = c.corners[1].x - c.corners[0].x;
        const dy1: f64 = c.corners[1].y - c.corners[0].y;
        const dx2: f64 = c.corners[2].x - c.corners[0].x;
        const dy2: f64 = c.corners[2].y - c.corners[0].y;
        if (dx1 * dy2 - dy1 * dx2 < 0) std.mem.swap(Point2f, &c.corners[1], &c.corners[3]);
    }

    var closes: std.ArrayList(Candidate) = .empty;
    defer closes.deinit(gpa);
    const sel = try filterTooClose(gpa, grey.width, grey.height, cands.items, p, d.marker_size, &closes);
    defer gpa.free(sel);

    // Dictionary rotations.
    const rots = try gpa.alloc(u64, d.bits.len * 4);
    defer gpa.free(rots);
    for (0..d.bits.len) |m| for (0..4) |r| {
        rots[m * 4 + r] = d.rotated(m, @intCast(r));
    };

    // identifyCandidates: depth by depth; a marker's ancestors are skipped.
    const nc = sel.len;
    const valid = try gpa.alloc(bool, nc);
    defer gpa.free(valid);
    @memset(valid, false);
    const was = try gpa.alloc(bool, nc);
    defer gpa.free(was);
    @memset(was, false);
    const ids_tmp = try gpa.alloc(Identified, nc);
    defer gpa.free(ids_tmp);
    var max_depth: i32 = 0;
    for (sel) |c| max_depth = @max(max_depth, c.depth);
    var scratch: Scratch = .{};
    defer scratch.deinit(gpa);
    var depth: i32 = 0;
    var counter: usize = 0;
    while (counter < nc and depth <= max_depth) : (depth += 1) {
        for (sel, 0..) |*c, v| {
            if (c.depth != depth) continue;
            // OpenCV identifies every candidate of this depth (even those
            // already marked as ancestors of a marker).
            was[v] = true;
            if (try identifyOne(gpa, grey, c.corners, d, rots, p, &scratch)) |idn| {
                valid[v] = true;
                ids_tmp[v] = idn;
            } else {
                for (closes.items[c.close_start..][0..c.close_len]) |cc| {
                    if (try identifyOne(gpa, grey, cc.corners, d, rots, p, &scratch)) |idn| {
                        valid[v] = true;
                        ids_tmp[v] = idn;
                        c.corners = cc.corners;
                        c.contour = cc.contour;
                        break;
                    }
                }
            }
        }
        for (sel, 0..) |c, v| {
            if (c.depth != depth) continue;
            if (valid[v]) {
                var parent = c.parent;
                while (parent != -1) {
                    const pu: usize = @intCast(parent);
                    if (!was[pu]) {
                        was[pu] = true;
                        counter += 1;
                    }
                    parent = sel[pu].parent;
                }
            }
            counter += 1;
        }
    }

    var corners: std.ArrayList([4]Point2f) = .empty;
    errdefer corners.deinit(gpa);
    var ids: std.ArrayList(i32) = .empty;
    errdefer ids.deinit(gpa);
    var rejected: std.ArrayList([4]Point2f) = .empty;
    errdefer rejected.deinit(gpa);
    for (sel, 0..) |c, v| {
        if (valid[v]) {
            const rot = ids_tmp[v].rotation;
            var q: [4]Point2f = undefined;
            for (0..4) |k| q[k] = c.corners[(k + 4 - @as(usize, rot)) % 4];
            try corners.append(gpa, q);
            try ids.append(gpa, ids_tmp[v].id);
        } else try rejected.append(gpa, c.corners);
    }

    if (p.corner_refinement_method == .subpix) {
        if (p.corner_refinement_win_size <= 0 or p.corner_refinement_max_iterations <= 0 or !(p.corner_refinement_min_accuracy > 0))
            return error.InvalidArgument;
        for (corners.items) |*q| {
            const ms = averageModuleSize(q.*, d.marker_size, p.marker_border_bits);
            var win: i32 = @max(1, @as(i32, @intFromFloat(sat.rint(@as(f64, p.relative_corner_refinement_win_size * ms)))));
            win = @min(win, p.corner_refinement_win_size);
            subpix.cornerSubPix(gpa, grey, q, .{
                .win_w = @intCast(win),
                .win_h = @intCast(win),
                .max_iters = @intCast(p.corner_refinement_max_iterations),
                .epsilon = p.corner_refinement_min_accuracy,
            }) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                else => {},
            };
        }
    }

    return .{
        .corners = try corners.toOwnedSlice(gpa),
        .ids = try ids.toOwnedSlice(gpa),
        .rejected = try rejected.toOwnedSlice(gpa),
    };
}

/// cv::aruco::drawDetectedMarkers on a gray or BGR image: the sides in
/// `border` colour, a square on the first corner and "id=N" at the centre.
/// Sides match OpenCV; the corner square is drawn without anti-aliasing
/// and the label uses VISAR's bitmap font (no Hershey fonts).
pub fn drawDetectedMarkers(img: Image(u8), corners: []const [4]Point2f, ids: ?[]const i32, border: [3]u8) error{InvalidArgument}!void {
    if (img.channels != 1 and img.channels != 3) return error.InvalidArgument;
    if (ids) |i| if (i.len != corners.len) return error.InvalidArgument;
    const cn = img.channels;
    const text_c = [3]u8{ border[1], border[0], border[2] };
    const corner_c = [3]u8{ border[0], border[2], border[1] };
    for (corners, 0..) |q, k| {
        var pts: [4]Point = undefined;
        for (q, &pts) |pt, *o| o.* = .{ .x = @intFromFloat(sat.rint(std.math.clamp(pt.x, -1e9, 1e9))), .y = @intFromFloat(sat.rint(std.math.clamp(pt.y, -1e9, 1e9))) };
        for (0..4) |j| polygon.drawLine(img, pts[j], pts[(j + 1) % 4], border[0..cn], 1, .eight);
        const c0 = pts[0];
        const r = [4]Point{ .{ .x = c0.x - 3, .y = c0.y - 3 }, .{ .x = c0.x + 3, .y = c0.y - 3 }, .{ .x = c0.x + 3, .y = c0.y + 3 }, .{ .x = c0.x - 3, .y = c0.y + 3 } };
        for (0..4) |j| polygon.drawLine(img, r[j], r[(j + 1) % 4], corner_c[0..cn], 1, .eight);
        if (ids) |idl| {
            var buf: [32]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "id={d}", .{idl[k]}) catch unreachable;
            var cx: i64 = 0;
            var cy: i64 = 0;
            for (pts) |pt| {
                cx += pt.x;
                cy += pt.y;
            }
            _ = draw.text(img, @divFloor(cx, 4), @divFloor(cy, 4) - 7, s, 1, text_c[0..cn]);
        }
    }
}

test "aruco: generate and detect" {
    const gpa = std.testing.allocator;
    const d = getPredefinedDictionary(.DICT_4X4_50);
    var img = try Image(u8).initFill(gpa, 200, 160, 1, 255);
    defer img.deinit(gpa);
    var m = try Image(u8).init(gpa, 60, 60, 1);
    defer m.deinit(gpa);
    try generateImageMarker(gpa, d, 7, 1, m);
    for (0..60) |y| @memcpy(img.data[(y + 40) * img.stride + 50 ..][0..60], m.data[y * m.stride ..][0..60]);
    var res = try detectMarkers(gpa, img, d, .{});
    defer res.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 1), res.ids.len);
    try std.testing.expectEqual(@as(i32, 7), res.ids[0]);
    try std.testing.expectEqual(@as(f32, 50), res.corners[0][0].x);
    try std.testing.expectEqual(@as(f32, 40), res.corners[0][0].y);
    // Noise never panics.
    var prng = std.Random.DefaultPrng.init(1);
    prng.random().bytes(img.data[0 .. img.stride * img.height]);
    var res2 = try detectMarkers(gpa, img, d, .{ .corner_refinement_method = .subpix, .detect_inverted_marker = true });
    res2.deinit(gpa);
}
