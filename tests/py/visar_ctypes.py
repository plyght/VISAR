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
