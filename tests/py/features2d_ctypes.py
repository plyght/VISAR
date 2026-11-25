"""ctypes binding of VISAR's features2d C ABI (FAST, ORB, BFMatcher),
on top of visar_ctypes. Keypoints come back as an (N, 7) structured view:
x, y, size, angle, response, octave, class_id (cv::KeyPoint's fields)."""
import ctypes

import numpy as np

import visar_ctypes as vs
from visar_ctypes import _check, desc, lib


class KeyPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float), ("size", ctypes.c_float),
                ("angle", ctypes.c_float), ("response", ctypes.c_float),
                ("octave", ctypes.c_int32), ("class_id", ctypes.c_int32)]


class DMatch(ctypes.Structure):
    _fields_ = [("queryIdx", ctypes.c_int32), ("trainIdx", ctypes.c_int32),
                ("imgIdx", ctypes.c_int32), ("distance", ctypes.c_float)]


class OrbParams(ctypes.Structure):
    _fields_ = [("nfeatures", ctypes.c_int), ("scale_factor", ctypes.c_float), ("nlevels", ctypes.c_int),
                ("edge_threshold", ctypes.c_int), ("first_level", ctypes.c_int), ("wta_k", ctypes.c_int),
                ("score_type", ctypes.c_int), ("patch_size", ctypes.c_int), ("fast_threshold", ctypes.c_int)]


KP_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("size", "<f4"), ("angle", "<f4"), ("response", "<f4"),
                     ("octave", "<i4"), ("class_id", "<i4")])
DM_DTYPE = np.dtype([("queryIdx", "<i4"), ("trainIdx", "<i4"), ("imgIdx", "<i4"), ("distance", "<f4")])

lib.visar_keypoints_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.visar_matches_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.visar_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
lib.visar_ratio_test.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_float,
                                 ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]


def _take_keypoints(ptr, n):
    if n == 0:
        return np.zeros(0, KP_DTYPE)
    buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * (n * KP_DTYPE.itemsize))).contents
    out = np.frombuffer(bytes(buf), KP_DTYPE).copy()
    lib.visar_keypoints_free(ptr, n)
    return out


def fast(img, threshold=10, nonmax=True, type=2):
    """cv2.FastFeatureDetector_create(threshold, nonmax, type).detect(img)."""
    p = ctypes.c_void_p()
    n = ctypes.c_size_t()
    _check(lib.visar_fast(ctypes.byref(desc(img)), int(threshold), int(nonmax), int(type), ctypes.byref(p), ctypes.byref(n)))
    return _take_keypoints(p, n.value)


def orb_params(**kw):
    p = OrbParams()
    lib.visar_orb_default_params(ctypes.byref(p))
    names = {"nfeatures": "nfeatures", "scaleFactor": "scale_factor", "nlevels": "nlevels",
             "edgeThreshold": "edge_threshold", "firstLevel": "first_level", "WTA_K": "wta_k",
             "scoreType": "score_type", "patchSize": "patch_size", "fastThreshold": "fast_threshold"}
    for k, v in kw.items():
        setattr(p, names.get(k, k), v)
    return p


def orb(img, mask=None, descriptors=True, **kw):
    """cv2.ORB_create(**kw).detectAndCompute(img, mask) -> (keypoints, descriptors)."""
    p = orb_params(**kw)
    kp = ctypes.c_void_p()
    n = ctypes.c_size_t()
    d = ctypes.c_void_p()
    m = ctypes.byref(desc(mask)) if mask is not None else None
    _check(lib.visar_orb_detect_and_compute(ctypes.byref(desc(img)), m, ctypes.byref(p), ctypes.byref(kp),
                                            ctypes.byref(n), ctypes.byref(d) if descriptors else None))
    k = _take_keypoints(kp, n.value)
    if not descriptors:
        return k, None
    if n.value == 0:
        return k, np.zeros((0, 32), np.uint8)
    nb = n.value * 32
    buf = ctypes.cast(d, ctypes.POINTER(ctypes.c_uint8 * nb)).contents
    out = np.frombuffer(bytes(buf), np.uint8).reshape(n.value, 32).copy()
    lib.visar_free(d, nb)
    return k, out


NORM_L1, NORM_L2, NORM_L2SQR, NORM_HAMMING, NORM_HAMMING2 = 2, 4, 5, 6, 7


def bf_match(query, train, norm=NORM_HAMMING, cross_check=False):
    """BFMatcher(norm, crossCheck).match(query, train) as a DM_DTYPE array."""
    p = ctypes.c_void_p()
    n = ctypes.c_size_t()
    _check(lib.visar_bf_match(ctypes.byref(desc(query)), ctypes.byref(desc(train)), norm, int(cross_check),
                              ctypes.byref(p), ctypes.byref(n)))
    if n.value == 0:
        return np.zeros(0, DM_DTYPE)
    buf = ctypes.cast(p, ctypes.POINTER(ctypes.c_uint8 * (n.value * 16))).contents
    out = np.frombuffer(bytes(buf), DM_DTYPE).copy()
    lib.visar_matches_free(p, n.value)
    return out


def bf_knn_match(query, train, k, norm=NORM_HAMMING, out=None, counts=None):
    """BFMatcher(norm).knnMatch(query, train, k) -> (matches (Q, k), counts (Q,))."""
    q = query.shape[0]
    out = np.zeros((q, k), DM_DTYPE) if out is None else out
    counts = np.zeros(q, np.uint32) if counts is None else counts
    _check(lib.visar_bf_knn_match(ctypes.byref(desc(query)), ctypes.byref(desc(train)), int(k), norm,
                                  out.ctypes.data_as(ctypes.c_void_p), counts.ctypes.data_as(ctypes.c_void_p)))
    return out, counts


def ratio_test(knn, counts, ratio=0.75):
    q, k = knn.shape
    out = np.zeros(q, DM_DTYPE)
    n = ctypes.c_size_t()
    _check(lib.visar_ratio_test(knn.ctypes.data_as(ctypes.c_void_p), counts.ctypes.data_as(ctypes.c_void_p), q, k,
                                ratio, out.ctypes.data_as(ctypes.c_void_p), ctypes.byref(n)))
    return out[:n.value]


__all__ = ["fast", "orb", "bf_match", "bf_knn_match", "ratio_test", "vs"]
