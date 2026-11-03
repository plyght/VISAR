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


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    pyramids()
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
