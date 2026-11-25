"""Differential tests for photo: fastNlMeansDenoising(Colored), inpaint
(Telea, NS) and the 8-bit linear Lab conversion, VISAR (through its C ABI)
against OpenCV. Run: zig build difftest

All of these are bit-exact with OpenCV: non-local means is integer
arithmetic after OpenCV's weight table, and inpainting replays OpenCV's
fast-marching order and float operations.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing the core suite handles `--lib` and provides the helpers.
import test_core_vs_opencv as base  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import photo_ctypes as ph  # noqa: E402
import visar_ctypes as vs  # noqa: E402

natural = base.natural
run = base.run
results = base.results


def noisy(h, w, c=1, sigma=15, seed=0):
    """natural() plus Gaussian noise: what a denoiser is for."""
    rng = np.random.default_rng(seed)
    img = natural(h, w, c).astype(np.float32) + rng.normal(0, sigma, (h, w, c) if c > 1 else (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)


def scratches(h, w, seed=0, frac=0.05):
    """An inpainting mask of dots, discs and strokes covering about `frac`
    of the image, touching the borders too."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    while np.count_nonzero(m) < frac * h * w:
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        if rng.random() < 0.5:
            cv2.circle(m, (x, y), int(rng.integers(1, max(2, min(h, w) // 30))), 255, -1)
        else:
            cv2.line(m, (x, y), (x + int(rng.integers(-60, 60)), y + int(rng.integers(-60, 60))), 255,
                     int(rng.integers(1, 4)))
    m[0, : min(3, w)] = 255
    m[-1, -min(2, w):] = 255
    return m


def lab_cases():
    v = np.arange(1 << 24, dtype=np.uint32)
    every = np.stack([v & 255, (v >> 8) & 255, v >> 16], -1).astype(np.uint8).reshape(4096, 4096, 3)
    run("cvtColor LBGR2Lab every color", lambda: (ph.lab_linear(every), cv2.cvtColor(every, cv2.COLOR_LBGR2Lab)))
    run("cvtColor Lab2LBGR every value", lambda: (ph.lab_linear(every, True), cv2.cvtColor(every, cv2.COLOR_Lab2LBGR)))


def nlm_cases():
    sizes = [(480, 640), (37, 53), (5, 9), (1, 1), (7, 1), (1, 30), (100, 100)]
    for (h, w) in sizes:
        g = noisy(h, w, 1, seed=h * w)
        for (hh, t, s) in [(10, 7, 21), (3, 7, 21), (25, 5, 11), (6.5, 3, 7), (0, 7, 21), (100, 7, 21), (10, 1, 1)]:
            run(f"fastNlMeansDenoising {w}x{h} h={hh} t={t} s={s}",
                lambda g=g, hh=hh, t=t, s=s: (ph.fast_nl_means_denoising(g, hh, t, s),
                                              cv2.fastNlMeansDenoising(g, None, hh, t, s)))
    # Even window sizes round down to odd ones, as in OpenCV.
    g = noisy(60, 80, 1, seed=3)
    run("fastNlMeansDenoising even windows 6/20", lambda: (ph.fast_nl_means_denoising(g, 10, 6, 20),
                                                           cv2.fastNlMeansDenoising(g, None, 10, 6, 20)))
    for c in (2, 3, 4):
        img = noisy(90, 120, c, seed=c)
        run(f"fastNlMeansDenoising {c}ch", lambda img=img: (ph.fast_nl_means_denoising(img, 12),
                                                            cv2.fastNlMeansDenoising(img, None, 12)))
    for (h, w) in [(480, 640), (37, 53), (1, 1)]:
        for c in (3, 4):
            img = noisy(h, w, c, seed=w + c)
            for (hh, hc) in [(10, 10), (3, 15), (20, 5)]:
                run(f"fastNlMeansDenoisingColored {w}x{h} {c}ch h={hh} hColor={hc}",
                    lambda img=img, hh=hh, hc=hc: (ph.fast_nl_means_denoising_colored(img, hh, hc),
                                                   cv2.fastNlMeansDenoisingColored(img, None, hh, hc)))
    big = noisy(720, 1280, 1, seed=9)
    run("fastNlMeansDenoising 1280x720 h=10", lambda: (ph.fast_nl_means_denoising(big, 10),
                                                       cv2.fastNlMeansDenoising(big, None, 10)))
    a = ph.fast_nl_means_denoising(big, 10)
    vs.set_num_threads(1)
    b = ph.fast_nl_means_denoising(big, 10)
    vs.set_num_threads(0)
    run("fastNlMeansDenoising 1 thread == N threads", lambda: (a, b))


def inpaint_cases():
    for (h, w) in [(480, 640), (37, 53), (5, 9), (3, 3), (1, 1), (720, 1280)]:
        bgr = noisy(h, w, 3, seed=h)
        gray = bgr[:, :, 1].copy()
        m = scratches(h, w, seed=w)
        imgs = (("8UC1", gray), ("8UC3", bgr), ("16UC1", gray.astype(np.uint16) * 257),
                ("32FC1", gray.astype(np.float32) / 7))
        radii = (3, 1, 5.5, 12) if w < 1000 else (3,)
        for flags, fname in ((cv2.INPAINT_TELEA, "TELEA"), (cv2.INPAINT_NS, "NS")):
            for r in radii:
                for name, img in imgs:
                    run(f"inpaint {fname} {w}x{h} {name} r={r}",
                        lambda img=img, m=m, r=r, flags=flags: (ph.inpaint(img, m, r, flags), cv2.inpaint(img, m, r, flags)))
    # Empty and full masks.
    g = noisy(40, 50, 1, seed=1)
    for name, m in (("empty", np.zeros((40, 50), np.uint8)), ("full", np.full((40, 50), 255, np.uint8))):
        for flags in (cv2.INPAINT_TELEA, cv2.INPAINT_NS):
            run(f"inpaint {flags} {name} mask", lambda m=m, flags=flags: (ph.inpaint(g, m, 3, flags), cv2.inpaint(g, m, 3, flags)))


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    lab_cases()
    nlm_cases()
    inpaint_cases()
    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[3]]
    for name, mx, frac, ok in results:
        tag = "FAIL" if not ok else "PASS"
        if tag != "PASS" or os.environ.get("VERBOSE"):
            print(f"{tag}  {name:<{width}}  max={mx:<8} differ={frac}")
    print(f"\n{len(results) - len(failed)}/{len(results)} cases match OpenCV within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
