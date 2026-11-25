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


def _dist(d):
    if d is None:
        return None, 0
    d = _f64(d).ravel()
    return d, len(d)


def rodrigues(src, jacobian=False):
    s = _f64(src).ravel()
    if s.size == 3:
        out = np.zeros(9)
        jac = np.zeros(27) if jacobian else None
        _check(lib.visar_rodrigues(_p(s), c_size_t(3), _p(out), _p(jac) if jac is not None else None))
        return (out.reshape(3, 3), jac.reshape(3, 9)) if jacobian else out.reshape(3, 3)
    out = np.zeros(3)
    _check(lib.visar_rodrigues(_p(s), c_size_t(9), _p(out), None))
    return out.reshape(3, 1)


def project_points(obj, rvec, tvec, K, dist=None, jacobian=False):
    o = _f64(obj).reshape(-1, 3)
    n = len(o)
    d, nd = _dist(dist)
    out = np.zeros((n, 2))
    jac = np.zeros((2 * n, 10 + nd)) if jacobian else None
    _check(lib.visar_project_points(_p(o), c_size_t(n), _p(_f64(rvec).ravel()), _p(_f64(tvec).ravel()), _p(_f64(K).ravel()),
                                    _p(d) if d is not None else None, c_size_t(nd), _p(out),
                                    _p(jac) if jac is not None else None))
    return (out, jac) if jacobian else out


def undistort_points(pts, K, dist=None, R=None, P=None):
    s = _f64(pts).reshape(-1, 2)
    out = np.zeros_like(s)
    d, nd = _dist(dist)
    r = _f64(R).ravel() if R is not None else None
    pp = _f64(P)[:3, :3].ravel().copy() if P is not None else None
    _check(lib.visar_undistort_points(_p(s), c_size_t(len(s)), _p(out), _p(_f64(K).ravel()), _p(d) if d is not None else None,
                                      c_size_t(nd), _p(r) if r is not None else None, _p(pp) if pp is not None else None))
    return out


def solve_pnp(obj, img, K, dist=None, flags=0, rvec=None, tvec=None):
    o = _f64(obj).reshape(-1, 3)
    i = _f64(img).reshape(-1, 2)
    d, nd = _dist(dist)
    rv = np.zeros(3) if rvec is None else _f64(rvec).ravel().copy()
    tv = np.zeros(3) if tvec is None else _f64(tvec).ravel().copy()
    _check(lib.visar_solve_pnp(_p(o), _p(i), c_size_t(len(o)), _p(_f64(K).ravel()), _p(d) if d is not None else None,
                               c_size_t(nd), flags, int(rvec is not None), _p(rv), _p(tv)))
    return rv.reshape(3, 1), tv.reshape(3, 1)


def solve_pnp_ransac(obj, img, K, dist=None, iterations=100, reprojection_error=8.0, confidence=0.99, flags=0):
    o = _f64(obj).reshape(-1, 3)
    i = _f64(img).reshape(-1, 2)
    d, nd = _dist(dist)
    rv = np.zeros(3)
    tv = np.zeros(3)
    inl = np.zeros(len(o), np.uint8)
    found = c_int(0)
    _check(lib.visar_solve_pnp_ransac(_p(o), _p(i), c_size_t(len(o)), _p(_f64(K).ravel()), _p(d) if d is not None else None,
                                      c_size_t(nd), iterations, c_double(reprojection_error), c_double(confidence), flags,
                                      _p(rv), _p(tv), _p(inl), ctypes.byref(found)))
    if not found.value:
        return False, None, None, None
    return True, rv.reshape(3, 1), tv.reshape(3, 1), np.flatnonzero(inl).reshape(-1, 1).astype(np.int32)
