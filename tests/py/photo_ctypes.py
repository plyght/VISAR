"""ctypes binding of VISAR's photo C ABI (non-local means denoising,
inpainting, edge-preserving filters), on top of visar_ctypes."""
import ctypes

import numpy as np

from visar_ctypes import _check, desc, lib

lib.visar_fast_nl_means_denoising.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.c_int]
lib.visar_fast_nl_means_denoising_colored.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                                                      ctypes.c_int, ctypes.c_int]
lib.visar_lab_linear.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


def _run(fn, src, *args, dst=None):
    src = np.ascontiguousarray(src)
    out = np.empty_like(src) if dst is None else dst
    _check(fn(ctypes.byref(desc(src)), ctypes.byref(desc(out)), *args))
    return out


def fast_nl_means_denoising(src, h=3.0, template_window_size=7, search_window_size=21):
    """cv2.fastNlMeansDenoising(src, None, h, templateWindowSize, searchWindowSize)."""
    return _run(lib.visar_fast_nl_means_denoising, src, float(h), int(template_window_size), int(search_window_size))


def fast_nl_means_denoising_colored(src, h=3.0, h_color=3.0, template_window_size=7, search_window_size=21):
    """cv2.fastNlMeansDenoisingColored(src, None, h, hColor, templateWindowSize, searchWindowSize)."""
    return _run(lib.visar_fast_nl_means_denoising_colored, src, float(h), float(h_color), int(template_window_size),
                int(search_window_size))


def lab_linear(src, inverse=False):
    """cv2.cvtColor(src, COLOR_LBGR2Lab) or (inverse) COLOR_Lab2LBGR, 8-bit."""
    return _run(lib.visar_lab_linear, src, int(inverse))

lib.visar_inpaint.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_double, ctypes.c_int]


def inpaint(src, mask, radius, flags):
    """cv2.inpaint(src, mask, inpaintRadius, flags): flags 0 INPAINT_NS, 1 INPAINT_TELEA."""
    src = np.ascontiguousarray(src)
    mask = np.ascontiguousarray(mask)
    out = np.empty_like(src)
    _check(lib.visar_inpaint(ctypes.byref(desc(src)), ctypes.byref(desc(mask)), ctypes.byref(desc(out)), float(radius),
                             int(flags)))
    return out


lib.visar_edge_preserving_filter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]
lib.visar_stylization.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
lib.visar_pencil_sketch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_float,
                                    ctypes.c_float]


def edge_preserving_filter(src, flags=1, sigma_s=60, sigma_r=0.4):
    """cv2.edgePreservingFilter(src, flags=flags, sigma_s=sigma_s, sigma_r=sigma_r)."""
    return _run(lib.visar_edge_preserving_filter, src, int(flags), float(sigma_s), float(sigma_r))


def stylization(src, sigma_s=60, sigma_r=0.45):
    """cv2.stylization(src, sigma_s=sigma_s, sigma_r=sigma_r)."""
    return _run(lib.visar_stylization, src, float(sigma_s), float(sigma_r))


def pencil_sketch(src, sigma_s=60, sigma_r=0.07, shade_factor=0.02):
    """cv2.pencilSketch(src, ...) -> (gray, color)."""
    src = np.ascontiguousarray(src)
    gray = np.empty(src.shape[:2], np.uint8)
    color = np.empty_like(src)
    _check(lib.visar_pencil_sketch(ctypes.byref(desc(src)), ctypes.byref(desc(gray)), ctypes.byref(desc(color)),
                                   float(sigma_s), float(sigma_r), float(shade_factor)))
    return gray, color


lib.visar_bilateral_filter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                       ctypes.c_int]


def bilateral_filter(src, d, sigma_color, sigma_space, border=4):
    """cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace, borderType=border)."""
    return _run(lib.visar_bilateral_filter, src, int(d), float(sigma_color), float(sigma_space), int(border))
