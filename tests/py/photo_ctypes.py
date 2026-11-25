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
