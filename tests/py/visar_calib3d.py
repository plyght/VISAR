"""ctypes binding to libvisar's calib3d entry points (see visar_ctypes.py).

Functions mirror cv2's signatures loosely: points are N x 2 / N x 3 arrays,
matrices are returned as NumPy float64 arrays, and "not found" results
come back as None like OpenCV's empty matrices.
"""
import ctypes

import numpy as np

import visar_ctypes as vs

lib = vs.lib
_check = vs._check
c_double = ctypes.c_double
c_int = ctypes.c_int
c_size_t = ctypes.c_size_t


def _f32(a, d=2):
    return np.ascontiguousarray(np.asarray(a, np.float32).reshape(-1, d))


def _f64(a, shape=None):
    a = np.ascontiguousarray(np.asarray(a, np.float64))
    return a.reshape(shape) if shape is not None else a


def _p(a):
    return a.ctypes.data_as(ctypes.c_void_p)


def find_homography(src, dst, method=0, thr=3.0, max_iters=2000, confidence=0.995):
    s, d = _f32(src), _f32(dst)
    n = len(s)
    mask = np.zeros(n, np.uint8)
    h = np.zeros(9, np.float64)
    found = c_int(0)
    _check(lib.visar_find_homography(_p(s), _p(d), c_size_t(n), method, c_double(thr), max_iters, c_double(confidence),
                                     _p(mask), _p(h), ctypes.byref(found)))
    return (h.reshape(3, 3) if found.value else None), mask.reshape(-1, 1)


def get_perspective_transform(src, dst):
    s, d = _f32(src), _f32(dst)
    m = np.zeros(9, np.float64)
    _check(lib.visar_get_perspective_transform(_p(s), _p(d), _p(m)))
    return m.reshape(3, 3)


def get_affine_transform(src, dst):
    s, d = _f32(src), _f32(dst)
    m = np.zeros(6, np.float64)
    _check(lib.visar_get_affine_transform(_p(s), _p(d), _p(m)))
    return m.reshape(2, 3)


def perspective_transform(pts, m):
    s = _f32(pts)
    out = np.empty_like(s)
    mm = _f64(m)
    lib.visar_perspective_transform(_p(s), _p(out), c_size_t(len(s)), _p(mm))
    return out


def estimate_affine_2d(src, dst, partial=False, method=8, thr=3.0, max_iters=2000, confidence=0.99, refine_iters=10):
    s, d = _f32(src), _f32(dst)
    n = len(s)
    inl = np.zeros(n, np.uint8)
    m = np.zeros(6, np.float64)
    found = c_int(0)
    _check(lib.visar_estimate_affine_2d(_p(s), _p(d), c_size_t(n), int(partial), method, c_double(thr), max_iters,
                                        c_double(confidence), refine_iters, _p(inl), _p(m), ctypes.byref(found)))
    return (m.reshape(2, 3) if found.value else None), inl.reshape(-1, 1)
