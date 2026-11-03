"""Differential tests: VISAR (through its C ABI) against OpenCV.

Every case runs the same operation in both libraries on the same inputs and
reports the largest absolute difference and the share of differing samples.
A case fails when it exceeds its tolerance. Run: zig build difftest
"""
import os
import sys

import cv2

# `--lib PATH` (passed by `zig build difftest/bench`) selects the library.
if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visar_ctypes as vs  # noqa: E402

rng = np.random.default_rng(1234)


def natural(h, w, c=3, dtype=np.uint8):
    """A photo-like test image: gradients, shapes, texture and noise."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 127 + 60 * np.sin(x / 37.0) * np.cos(y / 23.0) + 40 * (x / max(w, 1) - 0.5)
    planes = []
    for k in range(c):
        p = base + 30 * np.sin((x + y * (k + 1)) / 11.0) + rng.normal(0, 8, (h, w))
        planes.append(p)
    img = np.stack(planes, axis=-1) if c > 1 else planes[0]
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)
    cv2.rectangle(img, (w // 5, h // 5), (w // 2, h // 2), (255,) * c if c > 1 else 255, -1)
    cv2.circle(img, (2 * w // 3, 2 * h // 3), max(1, min(h, w) // 6), (0,) * c if c > 1 else 0, -1)
    if dtype == np.uint16:
        return img.astype(np.uint16) * 257
    if dtype == np.float32:
        return img.astype(np.float32) / 255.0
    if dtype == np.int16:
        return img.astype(np.int16) - 128
    return img


SIZES = [(480, 640), (37, 53), (1, 1), (7, 1), (1, 9)]
results = []


def compare(name, ours, ref, tol=0.0, pct=100.0):
    ref = np.asarray(ref)
    if ours.shape != ref.shape:
        results.append((name, "SHAPE", f"{ours.shape} vs {ref.shape}", False))
        return
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    mx = float(d.max()) if d.size else 0.0
    frac = 100.0 * float((d > (1e-5 if ours.dtype.kind == "f" else 0)).mean()) if d.size else 0.0
    ok = mx <= tol and frac <= pct
    results.append((name, f"{mx:.4g}", f"{frac:.2f}%", ok))


def run(name, fn, tol=0.0, pct=100.0):
    if os.environ.get("TRACE"):
        print(name, flush=True)
    try:
        ours, ref = fn()
        compare(name, ours, ref, tol, pct)
    except vs.VisarError as e:
        results.append((name, "ERROR", str(e), False))


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    for (h, w) in SIZES:
        tag = f"{w}x{h}"
        bgr = natural(h, w, 3)
        bgra = np.dstack([bgr, natural(h, w, 1)])
        gray = natural(h, w, 1)

        # --- color
        for code, src in [(cv2.COLOR_BGR2GRAY, bgr), (cv2.COLOR_RGB2GRAY, bgr), (cv2.COLOR_BGRA2GRAY, bgra),
                          (cv2.COLOR_BGR2RGB, bgr), (cv2.COLOR_BGR2BGRA, bgr), (cv2.COLOR_BGRA2BGR, bgra),
                          (cv2.COLOR_BGR2RGBA, bgr), (cv2.COLOR_GRAY2BGR, gray), (cv2.COLOR_GRAY2BGRA, gray),
                          (cv2.COLOR_BGR2HSV, bgr), (cv2.COLOR_RGB2HSV, bgr), (cv2.COLOR_BGR2YCrCb, bgr),
                          (cv2.COLOR_YCrCb2BGR, bgr), (cv2.COLOR_HSV2BGR, np.dstack([gray % 180, bgr[..., 1:]]))]:
            # OpenCV's HSV->BGR SIMD body and scalar tail round differently,
            # so row tails can differ by 1; VISAR follows the SIMD body.
            tol = 1 if code == cv2.COLOR_HSV2BGR else 0
            run(f"cvtColor {code} u8 {tag}", lambda: (vs.cvt_color(src, code), cv2.cvtColor(src, code)), tol)
        f = natural(h, w, 3, np.float32)
        run(f"cvtColor BGR2GRAY f32 {tag}", lambda: (vs.cvt_color(f, cv2.COLOR_BGR2GRAY), cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)), 1e-5)
        run(f"cvtColor BGR2HSV f32 {tag}", lambda: (vs.cvt_color(f, cv2.COLOR_BGR2HSV), cv2.cvtColor(f, cv2.COLOR_BGR2HSV)), 1e-3)
        u16 = natural(h, w, 3, np.uint16)
        run(f"cvtColor BGR2GRAY u16 {tag}", lambda: (vs.cvt_color(u16, cv2.COLOR_BGR2GRAY), cv2.cvtColor(u16, cv2.COLOR_BGR2GRAY)))

        # --- resize
        for (dw, dh) in [(max(1, w // 2), max(1, h // 2)), (w * 2, h * 2), (max(1, w * 3 // 4), max(1, h * 2 // 3)), (113, 71)]:
            for interp, nm, tol in [(cv2.INTER_NEAREST, "nearest", 0), (cv2.INTER_LINEAR, "linear", 1),
                                    (cv2.INTER_AREA, "area", 1), (cv2.INTER_CUBIC, "cubic", 1)]:
                run(f"resize {nm} u8 {tag}->{dw}x{dh}",
                    lambda: (vs.resize(bgr, (dw, dh), interp), cv2.resize(bgr, (dw, dh), interpolation=interp)), tol)
            run(f"resize linear f32 {tag}->{dw}x{dh}",
                lambda: (vs.resize(f, (dw, dh), 1), cv2.resize(f, (dw, dh), interpolation=cv2.INTER_LINEAR)), 1e-4)

        # --- filters
        for k, s in [(3, 0), (5, 0), (7, 0), (0, 1.5), (9, 2.0), (0, 3.0)]:
            # Known deviation: for kernels wider than ~15 taps OpenCV's fixed-point
            # accumulation order differs from VISAR's by up to 2 levels.
            run(f"GaussianBlur k{k} s{s} u8 {tag}",
                lambda: (vs.gaussian_blur(bgr, (k, k), s), cv2.GaussianBlur(bgr, (k, k), s)), 0 if s < 3 else 2)
            run(f"GaussianBlur k{k} s{s} f32 {tag}",
                lambda: (vs.gaussian_blur(f, (k, k), s), cv2.GaussianBlur(f, (k, k), s)), 1e-5)
        for border in [cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT, cv2.BORDER_WRAP]:
            if border == cv2.BORDER_WRAP:
                continue  # OpenCV GaussianBlur does not support WRAP
            run(f"GaussianBlur k5 border{border} u8 {tag}",
                lambda: (vs.gaussian_blur(gray, (5, 5), 0, border=border), cv2.GaussianBlur(gray, (5, 5), 0, borderType=border)))
        for k in [(3, 3), (5, 5), (7, 3), (15, 15)]:
            run(f"blur {k} u8 {tag}", lambda: (vs.box_filter(bgr, k), cv2.blur(bgr, k)), 1)
        for dx, dy, ks in [(1, 0, 3), (0, 1, 3), (1, 1, 3), (1, 0, 5), (2, 0, 5), (1, 0, -1)]:
            run(f"Sobel dx{dx} dy{dy} k{ks} u8->i16 {tag}",
                lambda: (vs.sobel(gray, np.int16, dx, dy, ks), cv2.Sobel(gray, cv2.CV_16S, dx, dy, ksize=ks)))
            run(f"Sobel dx{dx} dy{dy} k{ks} u8->f32 {tag}",
                lambda: (vs.sobel(gray, np.float32, dx, dy, ks), cv2.Sobel(gray, cv2.CV_32F, dx, dy, ksize=ks)), 1e-3)

        # --- threshold
        for kind in [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV, cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]:
            run(f"threshold {kind} u8 {tag}", lambda: (vs.threshold(bgr, 127.5, 200, kind)[1], cv2.threshold(bgr, 127.5, 200, kind)[1]))
            run(f"threshold {kind} f32 {tag}", lambda: (vs.threshold(f, 0.5, 1.0, kind)[1], cv2.threshold(f, 0.5, 1.0, kind)[1]))
        for auto in [cv2.THRESH_OTSU, cv2.THRESH_TRIANGLE]:
            def t():
                ta, a = vs.threshold(gray, 0, 255, cv2.THRESH_BINARY | auto)
                tb, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | auto)
                return np.array([ta]), np.array([tb])
            run(f"threshold auto{auto} value {tag}", t)

        # --- morphology
        for op in [cv2.MORPH_ERODE, cv2.MORPH_DILATE, cv2.MORPH_OPEN, cv2.MORPH_CLOSE, cv2.MORPH_GRADIENT, cv2.MORPH_TOPHAT, cv2.MORPH_BLACKHAT]:
            for shape in [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]:
                for ks in [(3, 3), (5, 7)]:
                    el = cv2.getStructuringElement(shape, ks)
                    run(f"morph op{op} shape{shape} {ks} u8 {tag}",
                        lambda: (vs.morphology(bgr, op, shape, ks), cv2.morphologyEx(bgr, op, el)))
        el = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        run(f"erode iter3 u8 {tag}", lambda: (vs.morphology(gray, 0, 0, (3, 3), 3), cv2.erode(gray, el, iterations=3)))

        # --- geometry
        for code in [0, 1, -1]:
            run(f"flip {code} {tag}", lambda: (vs.flip(bgr, code), cv2.flip(bgr, code)))
        for code in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            run(f"rotate {code} {tag}", lambda: (vs.rotate(bgr, code), cv2.rotate(bgr, code)))
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 30, 0.9)
        run(f"warpAffine nearest {tag}", lambda: (vs.warp_affine(bgr, m, (w, h), 0), cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_NEAREST)))
        run(f"warpAffine linear {tag}", lambda: (vs.warp_affine(bgr, m, (w, h), 1), cv2.warpAffine(bgr, m, (w, h), flags=cv2.INTER_LINEAR)))

        # --- histogram
        run(f"equalizeHist {tag}", lambda: (vs.equalize_hist(gray), cv2.equalizeHist(gray)))

        # --- core arithmetic
        other = natural(h, w, 3)[::-1, ::-1].copy()
        for op, ref in [("add", cv2.add), ("subtract", cv2.subtract), ("absdiff", cv2.absdiff), ("min", cv2.min),
                        ("max", cv2.max), ("and", cv2.bitwise_and), ("or", cv2.bitwise_or), ("xor", cv2.bitwise_xor),
                        ("multiply", cv2.multiply), ("divide", cv2.divide)]:
            run(f"{op} u8 {tag}", lambda: (vs.arith(op, bgr, other), ref(bgr, other)))
        run(f"addWeighted u8 {tag}", lambda: (vs.add_weighted(bgr, 0.3, other, 0.7, 5), cv2.addWeighted(bgr, 0.3, other, 0.7, 5)))
        run(f"convertTo u8->f32 {tag}", lambda: (vs.convert_to(bgr, np.float32, 1 / 255.0), bgr.astype(np.float32) * np.float32(1 / 255.0)), 1e-6)
        run(f"convertTo f32->u8 {tag}", lambda: (vs.convert_to(f, np.uint8, 255.0), (f * 255.0).round().clip(0, 255).astype(np.uint8)), 1)

    # --- codecs (decode files written by OpenCV; OpenCV decodes ours)
    img = natural(97, 131, 3)
    gray = natural(97, 131, 1)
    for ext, im in [(".png", img), (".png", gray), (".bmp", img), (".bmp", gray), (".ppm", img), (".pgm", gray),
                    (".png", natural(31, 17, 3, np.uint16))]:
        ok, buf = cv2.imencode(ext, im)
        run(f"imdecode {ext} {im.shape} {im.dtype}", lambda: (vs.imdecode(buf.tobytes()), cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)))
        enc = vs.imencode(ext, im)
        run(f"imencode {ext} {im.shape} {im.dtype}", lambda: (im, cv2.imdecode(np.frombuffer(enc, np.uint8), cv2.IMREAD_UNCHANGED)))
    for q in [50, 90]:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        run(f"imdecode .jpg q{q}", lambda: (vs.imdecode(buf.tobytes()), cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)), 1)

    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[3]]
    for name, mx, frac, ok in results:
        if not ok or os.environ.get("VERBOSE"):
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  max={mx:<8} differ={frac}")
    print(f"\n{len(results) - len(failed)}/{len(results)} cases match OpenCV within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
