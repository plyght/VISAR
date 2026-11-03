"""Minimal ctypes binding to libvisar for tests and benchmarks.

Arrays are NumPy arrays in OpenCV layout (H x W or H x W x C). This is not
the Python package (that comes later); it exists to drive the C ABI.
"""
import ctypes
import os
import sys

import numpy as np

_DEPTH = {np.uint8: 0, np.int8: 1, np.uint16: 2, np.int16: 3, np.int32: 4, np.float32: 5, np.float64: 6, np.float16: 7}
_DTYPE = {v: k for k, v in _DEPTH.items()}


class Image(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("depth", ctypes.c_uint32),
        ("stride", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
    ]


def _find_lib():
    env = os.environ.get("VISAR_LIB")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    name = {"darwin": "libvisar.dylib", "win32": "visar.dll"}.get(sys.platform, "libvisar.so")
    return os.path.join(here, "..", "..", "zig-out", "lib", name)


lib = ctypes.CDLL(_find_lib())
lib.visar_version.restype = ctypes.c_char_p
lib.visar_build_info.restype = ctypes.c_char_p
lib.visar_last_error.restype = ctypes.c_char_p


class VisarError(RuntimeError):
    pass


def _check(rc):
    if rc != 0:
        raise VisarError(f"{rc}: {lib.visar_last_error().decode()}")


def desc(a):
    """visar_image describing a NumPy array (rows may be strided; pixels must be packed)."""
    if a.ndim == 2:
        h, w = a.shape
        c = 1
    elif a.ndim == 3:
        h, w, c = a.shape
    else:
        raise ValueError("expected 2-D or 3-D array")
    t = a.dtype.type
    if t not in _DEPTH:
        raise ValueError(f"unsupported dtype {a.dtype}")
    if a.strides[-1] != a.itemsize or (a.ndim == 3 and a.strides[1] != a.itemsize * c):
        raise ValueError("pixels must be packed")
    return Image(w, h, c, _DEPTH[t], a.strides[0], a.ctypes.data)


def _out(shape, dtype):
    return np.empty(shape, dtype=dtype)


def version():
    return lib.visar_version().decode()


def set_num_threads(n):
    lib.visar_set_num_threads(n)


def _shape(a, w=None, h=None, c=None):
    H, W = a.shape[:2]
    C = a.shape[2] if a.ndim == 3 else 1
    w = W if w is None else w
    h = H if h is None else h
    c = C if c is None else c
    return (h, w) if c == 1 else (h, w, c)


_COLOR_CN = {  # code: (src channels, dst channels)
    0: (3, 4), 1: (4, 3), 2: (3, 4), 3: (4, 3), 4: (3, 3), 5: (4, 4), 6: (3, 1), 7: (3, 1),
    8: (1, 3), 9: (1, 4), 10: (4, 1), 11: (4, 1), 36: (3, 3), 37: (3, 3), 38: (3, 3), 39: (3, 3),
    40: (3, 3), 41: (3, 3), 54: (3, 3), 55: (3, 3),
}


def cvt_color(a, code, dst=None):
    dst = _out(_shape(a, c=_COLOR_CN[code][1]), a.dtype) if dst is None else dst
    _check(lib.visar_cvt_color(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), code))
    return dst


def resize(a, size, interpolation=1, dst=None):
    dst = _out(_shape(a, w=size[0], h=size[1]), a.dtype) if dst is None else dst
    _check(lib.visar_resize(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), interpolation))
    return dst


def gaussian_blur(a, ksize, sigma_x, sigma_y=0, border=4, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_gaussian_blur(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), ksize[0], ksize[1],
                                   ctypes.c_double(sigma_x), ctypes.c_double(sigma_y), border))
    return dst


def box_filter(a, ksize, normalize=True, border=4, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_box_filter(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), ksize[0], ksize[1], int(normalize), border))
    return dst


def sobel(a, ddepth, dx, dy, ksize=3, scale=1.0, border=4, dst=None):
    dst = np.empty(a.shape, dtype=ddepth) if dst is None else dst
    _check(lib.visar_sobel(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), dx, dy, ksize, ctypes.c_double(scale), border))
    return dst


def pyr_down(a, border=4, dst=None):
    dst = _out(_shape(a, w=(a.shape[1] + 1) // 2, h=(a.shape[0] + 1) // 2), a.dtype) if dst is None else dst
    _check(lib.visar_pyr_down(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), border))
    return dst


def pyr_up(a, border=4, dst=None):
    dst = _out(_shape(a, w=a.shape[1] * 2, h=a.shape[0] * 2), a.dtype) if dst is None else dst
    _check(lib.visar_pyr_up(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), border))
    return dst


def corner_min_eigen_val(a, block_size, ksize=3, border=4, dst=None):
    dst = np.empty(a.shape[:2], np.float32) if dst is None else dst
    _check(lib.visar_corner_min_eigen_val(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), block_size, ksize, border))
    return dst


def corner_harris(a, block_size, ksize, k, border=4, dst=None):
    dst = np.empty(a.shape[:2], np.float32) if dst is None else dst
    _check(lib.visar_corner_harris(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), block_size, ksize,
                                   ctypes.c_double(k), border))
    return dst


def good_features_to_track(a, max_corners, quality_level, min_distance, mask=None, block_size=3, gradient_size=3,
                           use_harris=False, k=0.04, cap=None):
    """Corners as an (n, 2) float32 array, like cv2.goodFeaturesToTrack(...).reshape(-1, 2)."""
    if cap is None:
        cap = max_corners if max_corners > 0 else a.shape[0] * a.shape[1]
    out = np.empty((max(cap, 1), 2), np.float32)
    count = ctypes.c_size_t(0)
    m = ctypes.byref(desc(mask)) if mask is not None else None
    _check(lib.visar_good_features_to_track(ctypes.byref(desc(a)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            ctypes.c_size_t(cap), ctypes.byref(count), max_corners, ctypes.c_double(quality_level),
                                            ctypes.c_double(min_distance), m, block_size, gradient_size,
                                            int(use_harris), ctypes.c_double(k)))
    return out[:min(count.value, cap)].copy()


def calc_optical_flow_pyr_lk(prev, nxt, prev_pts, next_pts=None, win_size=(21, 21), max_level=3,
                             criteria=(3, 30, 0.01), flags=0, min_eig_threshold=1e-4):
    """Like cv2.calcOpticalFlowPyrLK: returns (next_pts, status, err) as (n, 2), (n,), (n,) arrays."""
    pp = np.ascontiguousarray(np.asarray(prev_pts, np.float32).reshape(-1, 2))
    n = len(pp)
    if next_pts is None:
        npts = pp.copy()
    else:
        npts = np.ascontiguousarray(np.asarray(next_pts, np.float32).reshape(-1, 2)).copy()
    st = np.zeros(n, np.uint8)
    err = np.zeros(n, np.float32)
    fp = ctypes.POINTER(ctypes.c_float)
    _check(lib.visar_calc_optical_flow_pyr_lk(
        ctypes.byref(desc(prev)), ctypes.byref(desc(nxt)), pp.ctypes.data_as(fp), npts.ctypes.data_as(fp),
        st.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), err.ctypes.data_as(fp), ctypes.c_size_t(n), win_size[0], win_size[1],
        max_level, criteria[0], criteria[1], ctypes.c_double(criteria[2]), flags, ctypes.c_double(min_eig_threshold)))
    return npts, st, err


def threshold(a, thresh, maxval, kind, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    used = ctypes.c_double(0)
    _check(lib.visar_threshold(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), ctypes.c_double(thresh),
                               ctypes.c_double(maxval), kind, ctypes.byref(used)))
    return used.value, dst


def morphology(a, op, shape, ksize, iterations=1, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_morphology(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), op, shape, ksize[0], ksize[1], iterations))
    return dst


def flip(a, code, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_flip(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), code))
    return dst


def rotate(a, code, dst=None):
    if dst is None:
        dst = np.empty_like(a) if code == 1 else np.empty(_shape(a, w=a.shape[0], h=a.shape[1]), a.dtype)
    _check(lib.visar_rotate(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), code))
    return dst


def warp_affine(a, m, size, interpolation=1, border=0, value=0.0, dst=None):
    dst = _out(_shape(a, w=size[0], h=size[1]), a.dtype) if dst is None else dst
    arr = (ctypes.c_double * 6)(*np.asarray(m, dtype=np.float64).ravel())
    _check(lib.visar_warp_affine(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), arr, interpolation, border, ctypes.c_double(value)))
    return dst


def equalize_hist(a, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_equalize_hist(ctypes.byref(desc(a)), ctypes.byref(desc(dst))))
    return dst


ARITH = {"add": 0, "subtract": 1, "absdiff": 2, "multiply": 3, "divide": 4, "min": 5, "max": 6, "and": 7, "or": 8, "xor": 9}


def arith(op, a, b, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_arith(ARITH[op], ctypes.byref(desc(a)), ctypes.byref(desc(b)), ctypes.byref(desc(dst))))
    return dst


def add_weighted(a, alpha, b, beta, gamma, dst=None):
    dst = np.empty_like(a) if dst is None else dst
    _check(lib.visar_add_weighted(ctypes.byref(desc(a)), ctypes.c_double(alpha), ctypes.byref(desc(b)),
                                  ctypes.c_double(beta), ctypes.c_double(gamma), ctypes.byref(desc(dst))))
    return dst


def convert_to(a, dtype, alpha=1.0, beta=0.0, dst=None):
    dst = np.empty(a.shape, dtype=dtype) if dst is None else dst
    _check(lib.visar_convert_to(ctypes.byref(desc(a)), ctypes.byref(desc(dst)), ctypes.c_double(alpha), ctypes.c_double(beta)))
    return dst


def imdecode(buf, mode=-1, bgr=True):
    data = np.frombuffer(buf, dtype=np.uint8)
    img = Image()
    _check(lib.visar_imdecode(data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), len(data), mode, int(bgr), ctypes.byref(img)))
    try:
        dt = np.dtype(_DTYPE[img.depth])
        shape = (img.height, img.width) if img.channels == 1 else (img.height, img.width, img.channels)
        n = img.stride * img.height
        raw = ctypes.string_at(img.data, n)
        return np.frombuffer(raw, dtype=dt).reshape(shape).copy()
    finally:
        lib.visar_image_free(ctypes.byref(img))


def imencode(ext, a, bgr=True, quality=95):
    out = ctypes.POINTER(ctypes.c_uint8)()
    n = ctypes.c_size_t(0)
    a = np.ascontiguousarray(a)
    _check(lib.visar_imencode(ext.encode(), ctypes.byref(desc(a)), int(bgr), quality, ctypes.byref(out), ctypes.byref(n)))
    try:
        return ctypes.string_at(out, n.value)
    finally:
        lib.visar_free(out, n)


def connected_components_with_stats(a, connectivity=8):
    h, w = a.shape[:2]
    labels = np.empty((h, w), np.int32)
    count = ctypes.c_size_t(0)
    cap = h * w + 1
    stats = np.empty((min(cap, 1 << 20), 5), np.int32)
    cents = np.empty((stats.shape[0], 2), np.float64)
    _check(lib.visar_connected_components(ctypes.byref(desc(a)), ctypes.byref(desc(labels)), connectivity,
                                          stats.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                          cents.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                          stats.shape[0], ctypes.byref(count)))
    n = count.value
    return n, labels, stats[:n].copy(), cents[:n].copy()


# ------------------------------------------------------------- contours

class Contours(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_size_t),
        ("total", ctypes.c_size_t),
        ("points", ctypes.POINTER(ctypes.c_int32)),
        ("starts", ctypes.POINTER(ctypes.c_size_t)),
        ("hierarchy", ctypes.POINTER(ctypes.c_int32)),
    ]


def find_contours(a, mode, method, offset=(0, 0), raw=False):
    """Returns (contours, hierarchy) shaped like cv2.findContours: a tuple of
    N x 1 x 2 int32 arrays and a 1 x N x 4 hierarchy (None when empty).
    With raw=True returns (points (P x 2), starts (N + 1), hierarchy (N x 4))."""
    out = Contours()
    _check(lib.visar_find_contours(ctypes.byref(desc(a)), mode, method, offset[0], offset[1], ctypes.byref(out)))
    try:
        n, total = out.count, out.total
        if n == 0:
            if raw:
                return np.zeros((0, 2), np.int32), np.zeros(1, np.uintp), np.zeros((0, 4), np.int32)
            return (), None
        pts = np.ctypeslib.as_array(out.points, (total * 2,)).reshape(-1, 2).copy()
        starts = np.ctypeslib.as_array(out.starts, (n + 1,)).copy()
        hier = np.ctypeslib.as_array(out.hierarchy, (n * 4,)).reshape(n, 4).copy()
    finally:
        lib.visar_contours_free(ctypes.byref(out))
    if raw:
        return pts, starts, hier
    cs = tuple(pts[starts[i]:starts[i + 1]].reshape(-1, 1, 2) for i in range(n))
    return cs, hier.reshape(1, n, 4)


def _poly_args(polys):
    """(points, starts) arrays for a list of point arrays (any N x ... x 2 shape)."""
    arrs = [np.ascontiguousarray(np.asarray(p, np.int32).reshape(-1, 2)) for p in polys]
    starts = np.zeros(len(arrs) + 1, np.uintp)
    if arrs:
        starts[1:] = np.cumsum([len(a) for a in arrs])
    pts = np.ascontiguousarray(np.concatenate(arrs) if arrs else np.zeros((0, 2), np.int32))
    return pts, starts


def _color(c):
    c = list(c) if isinstance(c, (tuple, list)) else [c]
    c = (c + [c[-1]] * 4)[:4] if len(c) < 4 else c[:4]
    return (ctypes.c_double * 4)(*c)


def _iptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _sptr(a):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))


def draw_contours(img, contours, idx, color, thickness=1, line_type=8, hierarchy=None, max_level=2**31 - 1, offset=(0, 0)):
    """In place, like cv2.drawContours."""
    pts, starts = _poly_args(contours)
    h = None if hierarchy is None else np.ascontiguousarray(np.asarray(hierarchy, np.int32).reshape(-1, 4))
    _check(lib.visar_draw_contours(ctypes.byref(desc(img)), _iptr(pts), _sptr(starts), len(contours),
                                   None if h is None else _iptr(h), idx, _color(color), thickness, line_type,
                                   max_level, offset[0], offset[1]))
    return img


def fill_poly(img, polys, color, line_type=8, offset=(0, 0)):
    pts, starts = _poly_args(polys)
    _check(lib.visar_fill_poly(ctypes.byref(desc(img)), _iptr(pts), _sptr(starts), len(polys), _color(color),
                               line_type, offset[0], offset[1]))
    return img


def polylines(img, polys, closed, color, thickness=1, line_type=8):
    pts, starts = _poly_args(polys)
    _check(lib.visar_polylines(ctypes.byref(desc(img)), _iptr(pts), _sptr(starts), len(polys), int(closed),
                               _color(color), thickness, line_type))
    return img


# ---------------------------------------------------------------- shape

MOMENT_KEYS = ["m00", "m10", "m01", "m20", "m11", "m02", "m30", "m21", "m12", "m03",
               "mu20", "mu11", "mu02", "mu30", "mu21", "mu12", "mu03",
               "nu20", "nu11", "nu02", "nu30", "nu21", "nu12", "nu03"]


class Moments(ctypes.Structure):
    _fields_ = [(k, ctypes.c_double) for k in MOMENT_KEYS]

    def as_dict(self):
        return {k: getattr(self, k) for k in MOMENT_KEYS}


class RotatedRect(ctypes.Structure):
    _fields_ = [("cx", ctypes.c_float), ("cy", ctypes.c_float), ("w", ctypes.c_float),
                ("h", ctypes.c_float), ("angle", ctypes.c_float)]


def _pts(p):
    """(pointer, count, depth) for a point set given as an int32 or float32 array of N (x, y) pairs."""
    a = np.asarray(p)
    if a.dtype not in (np.int32, np.float32):
        raise ValueError("points must be int32 or float32")
    a = np.ascontiguousarray(a.reshape(-1, 2))
    return a, ctypes.c_void_p(a.ctypes.data if len(a) else None), len(a), 4 if a.dtype == np.int32 else 5


def moments(a, binary=False):
    """cv2.moments: a point set (int32/float32 N x 2) or a single-channel image."""
    m = Moments()
    a = np.asarray(a)
    if a.dtype in (np.int32, np.float32) and ((a.ndim == 3 and a.shape[1:] == (1, 2)) or (a.ndim == 2 and a.shape[1] == 2)):
        keep, ptr, n, d = _pts(a)
        _check(lib.visar_contour_moments(ptr, ctypes.c_size_t(n), d, ctypes.byref(m)))
    else:
        _check(lib.visar_moments(ctypes.byref(desc(a)), int(binary), ctypes.byref(m)))
    return m.as_dict()


def hu_moments(md):
    m = Moments(*[md[k] for k in MOMENT_KEYS])
    hu = (ctypes.c_double * 7)()
    lib.visar_hu_moments(ctypes.byref(m), hu)
    return np.array(hu[:]).reshape(7, 1)


def _scalar(fn, p, *args, ctype=ctypes.c_double):
    keep, ptr, n, d = _pts(p)
    out = ctype()
    _check(fn(ptr, ctypes.c_size_t(n), d, *args, ctypes.byref(out)))
    return out.value


def contour_area(p, oriented=False):
    return _scalar(lib.visar_contour_area, p, int(oriented))


def arc_length(p, closed):
    return _scalar(lib.visar_arc_length, p, int(closed))


def is_contour_convex(p):
    return bool(_scalar(lib.visar_is_contour_convex, p, ctype=ctypes.c_int))


def point_polygon_test(p, pt, measure_dist):
    return _scalar(lib.visar_point_polygon_test, p, ctypes.c_float(pt[0]), ctypes.c_float(pt[1]), int(measure_dist))


def bounding_rect(a):
    r = (ctypes.c_int32 * 4)()
    a = np.asarray(a)
    if a.dtype == np.uint8:
        _check(lib.visar_mask_bounding_rect(ctypes.byref(desc(a)), r))
    else:
        keep, ptr, n, d = _pts(a)
        _check(lib.visar_bounding_rect(ptr, ctypes.c_size_t(n), d, r))
    return tuple(r[:])


def convex_hull(p, clockwise=False, return_points=True):
    keep, ptr, n, d = _pts(p)
    idx = np.empty(max(n, 1), np.int32)
    cnt = ctypes.c_size_t(0)
    _check(lib.visar_convex_hull(ptr, ctypes.c_size_t(n), d, int(clockwise), int(return_points),
                                 idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.byref(cnt)))
    idx = idx[:cnt.value]
    if return_points:
        return keep[idx].reshape(-1, 1, 2)
    return idx.reshape(-1, 1)


def convexity_defects(p, hull):
    keep, ptr, n, d = _pts(p)
    h = np.ascontiguousarray(np.asarray(hull, np.int32).ravel())
    out = np.empty((max(len(h), 1), 4), np.int32)
    cnt = ctypes.c_size_t(0)
    _check(lib.visar_convexity_defects(ptr, ctypes.c_size_t(n), h.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                       ctypes.c_size_t(len(h)), out.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                                       ctypes.byref(cnt)))
    if cnt.value == 0:
        return None
    return out[:cnt.value].copy()


def min_area_rect(p):
    keep, ptr, n, d = _pts(p)
    r = RotatedRect()
    _check(lib.visar_min_area_rect(ptr, ctypes.c_size_t(n), d, ctypes.byref(r)))
    return (r.cx, r.cy), (r.w, r.h), r.angle


def box_points(box):
    (cx, cy), (w, h), a = box
    r = RotatedRect(cx, cy, w, h, a)
    out = (ctypes.c_float * 8)()
    lib.visar_box_points(ctypes.byref(r), out)
    return np.array(out[:], np.float32).reshape(4, 2)


def min_enclosing_circle(p):
    keep, ptr, n, d = _pts(p)
    out = (ctypes.c_float * 3)()
    _check(lib.visar_min_enclosing_circle(ptr, ctypes.c_size_t(n), d, out))
    return (out[0], out[1]), out[2]


def approx_poly_dp(p, epsilon, closed):
    keep, ptr, n, d = _pts(p)
    out = np.empty((max(n, 1), 2), keep.dtype)
    cnt = ctypes.c_size_t(0)
    _check(lib.visar_approx_poly_dp(ptr, ctypes.c_size_t(n), d, ctypes.c_double(epsilon), int(closed),
                                    ctypes.c_void_p(out.ctypes.data), ctypes.byref(cnt)))
    return out[:cnt.value].reshape(-1, 1, 2)


def match_shapes(a, b, method):
    ka, pa, na, da = _pts(a)
    kb, pb, nb, db = _pts(b)
    out = ctypes.c_double()
    _check(lib.visar_match_shapes(pa, ctypes.c_size_t(na), da, pb, ctypes.c_size_t(nb), db, method, ctypes.byref(out)))
    return out.value


def fit_ellipse(p):
    keep, ptr, n, d = _pts(p)
    r = RotatedRect()
    _check(lib.visar_fit_ellipse(ptr, ctypes.c_size_t(n), d, ctypes.byref(r)))
    return (r.cx, r.cy), (r.w, r.h), r.angle
