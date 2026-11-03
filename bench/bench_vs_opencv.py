"""Benchmarks VISAR (C ABI) against OpenCV on the same machine and inputs.

Usage: zig build bench -Doptimize=ReleaseFast [-- --threads N] [--quick] [--md FILE]
Each case reports the best wall time of both libraries, measured in alternation, and the speedup
(OpenCV time / VISAR time; > 1 means VISAR is faster). OpenCV and VISAR use
the same thread count.
"""
import argparse
import os
import statistics
import sys
import time

import cv2

# `--lib PATH` (passed by `zig build difftest/bench`) selects the library.
if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "py"))
import visar_ctypes as vs  # noqa: E402
from test_core_vs_opencv import natural  # noqa: E402


def timeit_pair(a, b, min_time=0.4, max_rounds=400):
    """Times two functions in alternation and returns the fastest run of
    each, so background load affects both libraries equally."""
    a()
    b()
    ta, tb = [], []
    start = time.perf_counter()
    while len(ta) < 7 or (time.perf_counter() - start < min_time and len(ta) < max_rounds):
        t0 = time.perf_counter()
        a()
        t1 = time.perf_counter()
        b()
        t2 = time.perf_counter()
        ta.append(t1 - t0)
        tb.append(t2 - t1)
    return min(ta), min(tb)


def cases(h, w):
    bgr = natural(h, w, 3)
    gray = natural(h, w, 1)
    f = natural(h, w, 3, np.float32)
    other = bgr[::-1].copy()
    out_gray = np.empty((h, w), np.uint8)
    out_bgr = np.empty_like(bgr)
    out_bgra = np.empty((h, w, 4), np.uint8)
    out_i16 = np.empty((h, w), np.int16)
    half = np.empty((h // 2, w // 2, 3), np.uint8)
    dbl = np.empty((h * 2, w * 2, 3), np.uint8)
    odd = np.empty((h * 2 // 3, w * 3 // 4, 3), np.uint8)
    yield "cvtColor BGR2GRAY", lambda: vs.cvt_color(bgr, 6, out_gray), lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY, dst=out_gray)
    yield "cvtColor BGR2RGB", lambda: vs.cvt_color(bgr, 4, out_bgr), lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=out_bgr)
    yield "cvtColor BGR2BGRA", lambda: vs.cvt_color(bgr, 0, out_bgra), lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA, dst=out_bgra)
    yield "cvtColor BGR2HSV", lambda: vs.cvt_color(bgr, 40, out_bgr), lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV, dst=out_bgr)
    yield "cvtColor BGR2YCrCb", lambda: vs.cvt_color(bgr, 36, out_bgr), lambda: cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb, dst=out_bgr)
    yield "resize linear 1/2", lambda: vs.resize(bgr, (w // 2, h // 2), 1, half), lambda: cv2.resize(bgr, (w // 2, h // 2), dst=half, interpolation=cv2.INTER_LINEAR)
    yield "resize linear 2x", lambda: vs.resize(bgr, (w * 2, h * 2), 1, dbl), lambda: cv2.resize(bgr, (w * 2, h * 2), dst=dbl, interpolation=cv2.INTER_LINEAR)
    yield "resize linear 0.75", lambda: vs.resize(bgr, (odd.shape[1], odd.shape[0]), 1, odd), lambda: cv2.resize(bgr, (odd.shape[1], odd.shape[0]), dst=odd, interpolation=cv2.INTER_LINEAR)
    yield "resize area 1/2", lambda: vs.resize(bgr, (w // 2, h // 2), 3, half), lambda: cv2.resize(bgr, (w // 2, h // 2), dst=half, interpolation=cv2.INTER_AREA)
    yield "resize nearest 2x", lambda: vs.resize(bgr, (w * 2, h * 2), 0, dbl), lambda: cv2.resize(bgr, (w * 2, h * 2), dst=dbl, interpolation=cv2.INTER_NEAREST)
    yield "resize cubic 2x", lambda: vs.resize(bgr, (w * 2, h * 2), 2, dbl), lambda: cv2.resize(bgr, (w * 2, h * 2), dst=dbl, interpolation=cv2.INTER_CUBIC)
    yield "GaussianBlur 5x5", lambda: vs.gaussian_blur(bgr, (5, 5), 0, dst=out_bgr), lambda: cv2.GaussianBlur(bgr, (5, 5), 0, dst=out_bgr)
    yield "GaussianBlur s=3", lambda: vs.gaussian_blur(bgr, (0, 0), 3.0, dst=out_bgr), lambda: cv2.GaussianBlur(bgr, (0, 0), 3.0, dst=out_bgr)
    yield "blur 3x3", lambda: vs.box_filter(bgr, (3, 3), dst=out_bgr), lambda: cv2.blur(bgr, (3, 3), dst=out_bgr)
    yield "blur 15x15", lambda: vs.box_filter(bgr, (15, 15), dst=out_bgr), lambda: cv2.blur(bgr, (15, 15), dst=out_bgr)
    yield "Sobel dx 3x3 ->i16", lambda: vs.sobel(gray, np.int16, 1, 0, 3, dst=out_i16), lambda: cv2.Sobel(gray, cv2.CV_16S, 1, 0, dst=out_i16, ksize=3)
    yield "threshold binary", lambda: vs.threshold(bgr, 127, 255, 0, out_bgr), lambda: cv2.threshold(bgr, 127, 255, cv2.THRESH_BINARY, dst=out_bgr)
    yield "threshold otsu", lambda: vs.threshold(gray, 0, 255, 8, out_gray), lambda: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU, dst=out_gray)
    el3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    el7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    yield "erode 3x3 rect", lambda: vs.morphology(bgr, 0, 0, (3, 3), 1, out_bgr), lambda: cv2.erode(bgr, el3, dst=out_bgr)
    yield "dilate 7x7 ellipse", lambda: vs.morphology(bgr, 1, 2, (7, 7), 1, out_bgr), lambda: cv2.dilate(bgr, el7, dst=out_bgr)
    yield "flip horizontal", lambda: vs.flip(bgr, 1, out_bgr), lambda: cv2.flip(bgr, 1, dst=out_bgr)
    rot = np.empty((w, h, 3), np.uint8)
    yield "rotate 90cw", lambda: vs.rotate(bgr, 0, rot), lambda: cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE, dst=rot)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 0.9)
    yield "warpAffine linear", lambda: vs.warp_affine(bgr, m, (w, h), 1, 0, 0, out_bgr), lambda: cv2.warpAffine(bgr, m, (w, h), dst=out_bgr, flags=cv2.INTER_LINEAR)
    yield "equalizeHist", lambda: vs.equalize_hist(gray, out_gray), lambda: cv2.equalizeHist(gray, dst=out_gray)
    yield "add u8", lambda: vs.arith("add", bgr, other, out_bgr), lambda: cv2.add(bgr, other, dst=out_bgr)
    yield "absdiff u8", lambda: vs.arith("absdiff", bgr, other, out_bgr), lambda: cv2.absdiff(bgr, other, dst=out_bgr)
    yield "addWeighted u8", lambda: vs.add_weighted(bgr, 0.3, other, 0.7, 0, out_bgr), lambda: cv2.addWeighted(bgr, 0.3, other, 0.7, 0, dst=out_bgr)
    out_fg = np.empty_like(f)
    yield "GaussianBlur 5x5 f32", lambda: vs.gaussian_blur(f, (5, 5), 0, dst=out_fg), lambda: cv2.GaussianBlur(f, (5, 5), 0, dst=out_fg)
    ok, png = cv2.imencode(".png", bgr)
    pngb = png.tobytes()
    yield "imdecode PNG", lambda: vs.imdecode(pngb), lambda: cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    yield "imencode PNG", lambda: vs.imencode(".png", bgr), lambda: cv2.imencode(".png", bgr)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpgb = jpg.tobytes()
    yield "imdecode JPEG", lambda: vs.imdecode(jpgb), lambda: cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)
    yield "imencode JPEG q90", lambda: vs.imencode(".jpg", bgr, quality=90), lambda: cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--md", help="write a Markdown table here")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()
    w, h = map(int, args.size.split("x"))
    threads = args.threads or os.cpu_count()
    cv2.setNumThreads(threads)
    vs.set_num_threads(threads)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}, {w}x{h}, {threads} thread(s)")
    rows = []
    for name, ours, theirs in cases(h, w):
        if args.filter and args.filter.lower() not in name.lower():
            continue
        try:
            t_v, t_o = timeit_pair(ours, theirs)
        except vs.VisarError as e:
            print(f"  {name:<24} VISAR error: {e}")
            continue
        rows.append((name, t_v, t_o))
        print(f"  {name:<24} visar {t_v * 1e3:8.3f} ms   opencv {t_o * 1e3:8.3f} ms   speedup {t_o / t_v:5.2f}x", flush=True)
    sp = [o / v for _, v, o in rows]
    med = statistics.median(sp) if sp else 0
    worst = min(sp) if sp else 0
    print(f"\nmedian speedup {med:.2f}x, worst {worst:.2f}x over {len(rows)} cases")
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(f"VISAR {vs.version()} vs OpenCV {cv2.__version__}, {w}x{h}, {threads} thread(s), "
                     f"{vs.lib.visar_build_info().decode()}\n\n")
            fh.write("| Operation | VISAR (ms) | OpenCV (ms) | Speedup |\n| --- | ---: | ---: | ---: |\n")
            for name, v, o in rows:
                fh.write(f"| {name} | {v * 1e3:.3f} | {o * 1e3:.3f} | {o / v:.2f}x |\n")
            fh.write(f"\nMedian speedup **{med:.2f}x**, worst {worst:.2f}x.\n")


if __name__ == "__main__":
    main()
