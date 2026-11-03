"""Differential tests for pyramids, corner detection and optical flow:
VISAR (through its C ABI) against OpenCV. Run: zig build difftest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing the core suite handles `--lib` and provides the helpers.
import test_core_vs_opencv as base  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import visar_ctypes as vs  # noqa: E402

natural = base.natural
run = base.run
results = base.results

SIZES = [(480, 640), (37, 53), (1, 1), (2, 2), (3, 5), (7, 1), (1, 9), (17, 16)]


def pyramids():
    for (h, w) in SIZES:
        tag = f"{w}x{h}"
        for cn in (1, 2, 3, 4):
            for dt, tol in ((np.uint8, 0), (np.uint16, 0), (np.int16, 0), (np.float32, 1e-5)):
                img = natural(h, w, cn, dt)
                if dt == np.uint16 and cn == 2:
                    img = img.copy()
                    img[..., 1] = 65535  # sums at the top of the u32 range
                nm = np.dtype(dt).name
                run(f"pyrDown {nm} c{cn} {tag}", lambda: (vs.pyr_down(img), cv2.pyrDown(img)), tol)
                run(f"pyrUp {nm} c{cn} {tag}", lambda: (vs.pyr_up(img), cv2.pyrUp(img)), tol)
        img = natural(h, w, 3)
        for border in (cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT):
            run(f"pyrDown border{border} u8 {tag}", lambda: (vs.pyr_down(img, border), cv2.pyrDown(img, borderType=border)))
    # Saturated extremes: every intermediate at its maximum.
    white = np.full((33, 47), 255, np.uint8)
    run("pyrDown u8 all-255", lambda: (vs.pyr_down(white), cv2.pyrDown(white)))
    run("pyrUp u8 all-255", lambda: (vs.pyr_up(white), cv2.pyrUp(white)))
    # A pyramid built level by level.
    img = natural(271, 389, 1)
    a, b = img, img
    for lvl in range(1, 6):
        a, b = vs.pyr_down(a), cv2.pyrDown(b)
        run(f"pyramid level {lvl}", lambda: (a, b))


def rel_run(name, fn, rel, floor=1e-7):
    """run() with a tolerance relative to the reference's largest magnitude
    (at least `floor`: with blockSize 1 the tensor has rank one and the
    minimum eigenvalue is rounding noise around 0)."""
    ours, ref = fn()
    tol = max(rel * (float(np.abs(ref).max()) if ref.size else 0.0), floor)
    run(name, lambda: (ours, ref), tol)


def corners():
    # The tensor entries agree to a few f32 ulps (OpenCV scales its
    # derivative kernels in f32, VISAR sums exact integers); the minimum
    # eigenvalue subtracts nearly equal numbers where the gradient is
    # one-dimensional, so the difference is bounded relative to the image's
    # largest response.
    for (h, w) in [(480, 640), (37, 53), (3, 5), (1, 1), (2, 7), (17, 16)]:
        tag = f"{w}x{h}"
        gray = natural(h, w, 1)
        f = natural(h, w, 1, np.float32)
        for bs, ks in [(3, 3), (2, 3), (5, 3), (3, 5), (3, -1), (7, 7), (3, 1), (4, 5), (1, 3)]:
            for src, nm in ((gray, "u8"), (f, "f32")):
                rel_run(f"cornerMinEigenVal b{bs} k{ks} {nm} {tag}",
                        lambda: (vs.corner_min_eigen_val(src, bs, ks), cv2.cornerMinEigenVal(src, bs, ksize=ks)), 2e-5)
                rel_run(f"cornerHarris b{bs} k{ks} {nm} {tag}",
                        lambda: (vs.corner_harris(src, bs, ks, 0.04), cv2.cornerHarris(src, bs, ks, 0.04)), 2e-5)
        for border in (cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT):
            rel_run(f"cornerMinEigenVal border{border} u8 {tag}",
                    lambda: (vs.corner_min_eigen_val(gray, 3, 3, border), cv2.cornerMinEigenVal(gray, 3, ksize=3, borderType=border)), 2e-5)


def gftt():
    # Same corners, same order: compared exactly.
    def pair(img, mc, q, md, mask=None, bs=3, gs=3, harris=False, k=0.04):
        ref = cv2.goodFeaturesToTrack(img, mc, q, md, mask=mask, blockSize=bs, gradientSize=gs,
                                      useHarrisDetector=harris, k=k)
        ref = np.zeros((0, 2), np.float32) if ref is None else ref.reshape(-1, 2)
        ours = vs.good_features_to_track(img, mc, q, md, mask, bs, gs, harris, k)
        return ours, ref

    for (h, w) in [(480, 640), (720, 1280), (97, 131), (3, 3), (2, 9), (1, 1)]:
        tag = f"{w}x{h}"
        gray = natural(h, w, 1)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        for img, nm in ((gray, "noisy"), (blur, "smooth")):
            for mc, q, md in [(0, 0.01, 0), (100, 0.01, 10), (1000, 0.01, 5), (500, 0.05, 1), (300, 0.001, 7.5),
                              (0, 0.1, 0.5), (25, 0.3, 30)]:
                run(f"goodFeaturesToTrack {nm} max{mc} q{q} d{md} {tag}", lambda: pair(img, mc, q, md))
            run(f"goodFeaturesToTrack harris {nm} {tag}", lambda: pair(img, 400, 0.01, 8, harris=True))
            run(f"goodFeaturesToTrack b7 g5 {nm} {tag}", lambda: pair(img, 400, 0.01, 8, bs=7, gs=5))
            mask = np.zeros((h, w), np.uint8)
            mask[h // 4: 3 * h // 4 + 1, w // 3: w] = 1
            run(f"goodFeaturesToTrack mask {nm} {tag}", lambda: pair(img, 200, 0.02, 6, mask=mask))
        f = natural(h, w, 1, np.float32)
        run(f"goodFeaturesToTrack f32 {tag}", lambda: pair(f, 300, 0.01, 6))
    # A synthetic scene with exact ties (symmetric squares): order of equal
    # responses must follow OpenCV's (reverse raster order).
    img = np.zeros((120, 160), np.uint8)
    for (x, y) in [(20, 20), (90, 20), (20, 70), (90, 70)]:
        img[y:y + 30, x:x + 40] = 200
    run("goodFeaturesToTrack ties", lambda: pair(img, 0, 0.01, 3))


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    pyramids()
    corners()
    gftt()
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
