"""ctypes binding of VISAR's objdetect C ABI (QR codes, ArUco), on top of
visar_ctypes. Results mirror cv2's: QR corner arrays (4, 2) float32,
ArUco corners as a list of (1, 4, 2) float32 arrays and an (N, 1) int32 id
array."""
import ctypes

import numpy as np

from visar_ctypes import _check, desc, lib

lib.visar_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]


def _take(ptr, n, dtype=np.uint8):
    if n == 0 or not ptr:
        return np.zeros(0, dtype)
    nbytes = n * np.dtype(dtype).itemsize
    buf = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8 * nbytes)).contents
    out = np.frombuffer(bytes(buf), dtype).copy()
    lib.visar_free(ptr, nbytes)
    return out


# ------------------------------------------------------------------ QR


def qr_encode(text, version=0, level=0, mode=-1):
    """cv2.QRCodeEncoder.create(params).encode(text) -> (N, N) uint8."""
    b = text.encode() if isinstance(text, str) else bytes(text)
    p = ctypes.c_void_p()
    n = ctypes.c_uint32()
    _check(lib.visar_qr_encode(b, len(b), int(version), int(level), int(mode), ctypes.byref(p), ctypes.byref(n)))
    return _take(p, n.value * n.value).reshape(n.value, n.value)


def qr_encode_structured_append(text, n, version=0, level=0):
    b = text.encode() if isinstance(text, str) else bytes(text)
    p = ctypes.c_void_p()
    sizes = (ctypes.c_uint32 * 16)()
    count = ctypes.c_uint32()
    _check(lib.visar_qr_encode_structured_append(b, len(b), int(version), int(level), int(n), ctypes.byref(p), sizes,
                                                 ctypes.byref(count)))
    total = sum(sizes[i] ** 2 for i in range(count.value))
    flat = _take(p, total)
    out, off = [], 0
    for i in range(count.value):
        s = sizes[i]
        out.append(flat[off:off + s * s].reshape(s, s))
        off += s * s
    return out


class QrCode(ctypes.Structure):
    _fields_ = [("corners", ctypes.c_float * 8), ("offset", ctypes.c_uint32), ("length", ctypes.c_uint32),
                ("decoded", ctypes.c_int32), ("version", ctypes.c_int32)]


if hasattr(lib, "visar_qr_free"):
    lib.visar_qr_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]


def _qr_call(img, multi, decode):
    codes = ctypes.POINTER(QrCode)()
    count = ctypes.c_size_t()
    text = ctypes.c_void_p()
    tlen = ctypes.c_size_t()
    _check(lib.visar_qr_detect(ctypes.byref(desc(np.ascontiguousarray(img))), int(multi), int(decode), ctypes.byref(codes),
                               ctypes.byref(count), ctypes.byref(text), ctypes.byref(tlen)))
    raw = ctypes.string_at(text, tlen.value) if tlen.value else b""
    out = []
    for i in range(count.value):
        c = codes[i]
        pts = np.array(list(c.corners), np.float32).reshape(4, 2)
        s = raw[c.offset:c.offset + c.length].decode("utf-8", "surrogateescape") if c.decoded else ""
        out.append((s, pts, bool(c.decoded)))
    lib.visar_qr_free(codes, count.value, text, tlen.value)
    return out


def qr_detect(img):
    """cv2.QRCodeDetector().detect(img) -> (found, points (4, 2) or None)."""
    r = _qr_call(img, 0, 0)
    return (True, r[0][1]) if r else (False, None)


def qr_detect_and_decode(img):
    """cv2.QRCodeDetector().detectAndDecode(img) -> (text, points or None)."""
    r = _qr_call(img, 0, 1)
    return (r[0][0], r[0][1]) if r else ("", None)


def qr_detect_and_decode_multi(img):
    """cv2.QRCodeDetector().detectAndDecodeMulti(img) -> (ok, texts, points (N, 4, 2))."""
    r = _qr_call(img, 1, 1)
    if not r:
        return False, (), None
    return True, tuple(t for t, _, _ in r), np.stack([p for _, p, _ in r])


# --------------------------------------------------------------- ArUco


class ArucoParams(ctypes.Structure):
    _fields_ = [
        ("adaptive_thresh_win_size_min", ctypes.c_int), ("adaptive_thresh_win_size_max", ctypes.c_int),
        ("adaptive_thresh_win_size_step", ctypes.c_int), ("adaptive_thresh_constant", ctypes.c_double),
        ("min_marker_perimeter_rate", ctypes.c_double), ("max_marker_perimeter_rate", ctypes.c_double),
        ("polygonal_approx_accuracy_rate", ctypes.c_double), ("min_corner_distance_rate", ctypes.c_double),
        ("min_distance_to_border", ctypes.c_int), ("min_marker_distance_rate", ctypes.c_double),
        ("min_group_distance", ctypes.c_float), ("corner_refinement_method", ctypes.c_int),
        ("corner_refinement_win_size", ctypes.c_int), ("relative_corner_refinement_win_size", ctypes.c_float),
        ("corner_refinement_max_iterations", ctypes.c_int), ("corner_refinement_min_accuracy", ctypes.c_double),
        ("marker_border_bits", ctypes.c_int), ("perspective_remove_pixel_per_cell", ctypes.c_int),
        ("perspective_remove_ignored_margin_per_cell", ctypes.c_double),
        ("max_erroneous_bits_in_border_rate", ctypes.c_double), ("min_otsu_std_dev", ctypes.c_double),
        ("error_correction_rate", ctypes.c_double), ("valid_bit_id_threshold", ctypes.c_float),
        ("detect_inverted_marker", ctypes.c_int),
    ]


_PARAM_NAMES = {
    "adaptiveThreshWinSizeMin": "adaptive_thresh_win_size_min", "adaptiveThreshWinSizeMax": "adaptive_thresh_win_size_max",
    "adaptiveThreshWinSizeStep": "adaptive_thresh_win_size_step", "adaptiveThreshConstant": "adaptive_thresh_constant",
    "minMarkerPerimeterRate": "min_marker_perimeter_rate", "maxMarkerPerimeterRate": "max_marker_perimeter_rate",
    "polygonalApproxAccuracyRate": "polygonal_approx_accuracy_rate", "minCornerDistanceRate": "min_corner_distance_rate",
    "minDistanceToBorder": "min_distance_to_border", "minMarkerDistanceRate": "min_marker_distance_rate",
    "minGroupDistance": "min_group_distance", "cornerRefinementMethod": "corner_refinement_method",
    "cornerRefinementWinSize": "corner_refinement_win_size",
    "relativeCornerRefinmentWinSize": "relative_corner_refinement_win_size",
    "cornerRefinementMaxIterations": "corner_refinement_max_iterations",
    "cornerRefinementMinAccuracy": "corner_refinement_min_accuracy", "markerBorderBits": "marker_border_bits",
    "perspectiveRemovePixelPerCell": "perspective_remove_pixel_per_cell",
    "perspectiveRemoveIgnoredMarginPerCell": "perspective_remove_ignored_margin_per_cell",
    "maxErroneousBitsInBorderRate": "max_erroneous_bits_in_border_rate", "minOtsuStdDev": "min_otsu_std_dev",
    "errorCorrectionRate": "error_correction_rate", "validBitIdThreshold": "valid_bit_id_threshold",
    "detectInvertedMarker": "detect_inverted_marker",
}


def aruco_params(**kw):
    p = ArucoParams()
    lib.visar_aruco_default_params(ctypes.byref(p))
    for k, v in kw.items():
        setattr(p, _PARAM_NAMES.get(k, k), v)
    return p


def aruco_generate(dict_id, marker_id, side, border_bits=1):
    """cv2.aruco.generateImageMarker(getPredefinedDictionary(dict_id), id, side, borderBits=...)."""
    out = np.empty((side, side), np.uint8)
    _check(lib.visar_aruco_generate(int(dict_id), int(marker_id), int(side), int(border_bits), ctypes.byref(desc(out))))
    return out


def aruco_detect(img, dict_id, rejected=False, **kw):
    """cv2.aruco.ArucoDetector(dict, params).detectMarkers(img) -> (corners, ids[, rejected])."""
    p = aruco_params(**kw)
    corners = ctypes.c_void_p()
    ids = ctypes.c_void_p()
    n = ctypes.c_size_t()
    rej = ctypes.c_void_p()
    nrej = ctypes.c_size_t()
    _check(lib.visar_aruco_detect(ctypes.byref(desc(np.ascontiguousarray(img))), int(dict_id), ctypes.byref(p),
                                  ctypes.byref(corners), ctypes.byref(ids), ctypes.byref(n),
                                  ctypes.byref(rej) if rejected else None, ctypes.byref(nrej) if rejected else None))
    c = _take(corners, n.value * 8, np.float32).reshape(-1, 1, 4, 2)
    i = _take(ids, n.value, np.int32).reshape(-1, 1)
    out = (tuple(c[k] for k in range(len(c))), i if n.value else None)
    if rejected:
        r = _take(rej, nrej.value * 8, np.float32).reshape(-1, 1, 4, 2)
        out = out + (tuple(r[k] for k in range(len(r))),)
    return out


def aruco_draw(img, corners, ids=None, color=(0, 255, 0)):
    """cv2.aruco.drawDetectedMarkers(img, corners, ids, borderColor) in place."""
    n = len(corners)
    flat = np.ascontiguousarray(np.array(corners, np.float32).reshape(n, 8)) if n else np.zeros((0, 8), np.float32)
    idp = None
    if ids is not None:
        ida = np.ascontiguousarray(np.array(ids, np.int32).reshape(-1))
        idp = ida.ctypes.data
    col = (ctypes.c_double * 3)(*[float(x) for x in color])
    _check(lib.visar_aruco_draw(ctypes.byref(desc(img)), ctypes.c_void_p(flat.ctypes.data), ctypes.c_void_p(idp),
                                ctypes.c_size_t(n), col))
    return img


def adaptive_threshold(img, maxval, method, kind, block, delta):
    """cv2.adaptiveThreshold(img, maxval, method, kind, block, delta)."""
    out = np.empty_like(img)
    _check(lib.visar_adaptive_threshold(ctypes.byref(desc(img)), ctypes.byref(desc(out)), ctypes.c_double(maxval),
                                        int(method), int(kind), int(block), ctypes.c_double(delta)))
    return out
