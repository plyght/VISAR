"""Differential tests for objdetect: QR codes and ArUco markers, VISAR
(through its C ABI) against OpenCV. Run: zig build difftest

QR encoding is compared module for module with cv2.QRCodeEncoder. QR
detection runs on cv2-generated symbols rendered with perspective, blur
and noise: wherever cv2 decodes a symbol, VISAR must return the same
string (VISAR may also decode symbols cv2 misses; those are counted), and
the corners must agree within a few pixels. ArUco: generateImageMarker is
compared pixel for pixel; detectMarkers must find the same ids with
corners within 0.5 px (exactly equal without refinement), with the
default parameters, with CORNER_REFINE_SUBPIX and with inverted markers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing the core suite handles `--lib` and provides the helpers.
import test_core_vs_opencv as base  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import objdetect_ctypes as od  # noqa: E402
import visar_ctypes as vs  # noqa: E402

results = base.results


def record(name, ok, detail=""):
    results.append((name, "-" if ok else "MISMATCH", detail, bool(ok)))


# ------------------------------------------------------------------ QR

ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"


def cv_encode(text, version=0, level=0, mode=-1, structure_number=1):
    p = cv2.QRCodeEncoder_Params()
    p.version = version
    p.correction_level = level
    p.mode = mode
    p.structure_number = structure_number
    return cv2.QRCodeEncoder.create(p)


def random_text(rng, n, kind):
    if kind == "num":
        return "".join(rng.choice(list("0123456789"), n))
    if kind == "alnum":
        return "".join(rng.choice(list(ALNUM), n))
    if kind == "utf8":
        return "".join(rng.choice(list("abcdeé漢字ü€ 0123"), n))
    return "".join(chr(c) for c in rng.integers(32, 127, n))


def encoder_cases():
    rng = np.random.default_rng(1)
    n_ok = n = 0
    bad = []
    cases = []
    for length in [1, 3, 9, 17, 40, 100, 250, 600, 1200, 2000]:
        for kind in ["num", "alnum", "byte", "utf8"]:
            for level in range(4):
                cases.append((random_text(rng, length, kind), 0, level, -1))
    for v in [1, 2, 6, 7, 9, 10, 14, 20, 26, 27, 33, 40]:
        for level in range(4):
            cases += [("visar %d" % v, v, level, -1), ("ABC 123", v, level, 2), ("0123456789", v, level, 1),
                      ("bytes\x01", v, level, 4), ("eci text", v, level, 7)]
    for t, v, lv, m in cases:
        n += 1
        try:
            ref = cv_encode(t, v, lv, m).encode(t)
        except cv2.error:
            ref = None
        try:
            ours = od.qr_encode(t, v, lv, m)
        except vs.VisarError:
            ours = None
        if ref is None or ours is None:
            ok = (ref is None) == (ours is None)
        else:
            ok = ref.shape == ours.shape and not (ref != ours).any()
        n_ok += ok
        if not ok:
            bad.append(f"len {len(t)} v{v} L{lv} m{m}")
    record(f"QRCodeEncoder.encode ({n} symbols)", n_ok == n, "; ".join(bad[:4]))
    # Structured append.
    text = random_text(rng, 300, "byte")
    ref = cv_encode(text, mode=3, structure_number=3).encodeStructuredAppend(text)
    ours = od.qr_encode_structured_append(text, 3)
    ok = len(ref) == len(ours) and all(a.shape == b.shape and not (a != b).any() for a, b in zip(ref, ours))
    record("QRCodeEncoder.encodeStructuredAppend (3 symbols)", ok)


def render(symbol, module, rng, size=(480, 640), perspective=0.0, blur=0.0, noise=0.0, angle=0.0, center=None,
           bg=None):
    """Draws a cv2 symbol (with its quiet zone) on a background, then warps,
    blurs and adds noise. Returns the image."""
    h, w = size
    big = cv2.resize(symbol, None, fx=module, fy=module, interpolation=cv2.INTER_NEAREST)
    s = big.shape[0]
    img = bg.copy() if bg is not None else np.full((h, w), 200, np.uint8)
    cx, cy = center if center is not None else (w / 2, h / 2)
    src = np.float32([[0, 0], [s, 0], [s, s], [0, s]])
    a = np.deg2rad(angle)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (src - s / 2) @ rot.T
    dst += rng.uniform(-perspective, perspective, dst.shape) * s
    dst = (dst + [cx, cy]).astype(np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(big, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    mask = cv2.warpPerspective(np.full_like(big, 255), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    img = ((img.astype(np.float32) * (255 - mask) + warped.astype(np.float32) * mask) / 255)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = img + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def background(rng, h, w):
    g = base.natural(h, w, 1).astype(np.float32)
    return np.clip(g * 0.5 + 100, 0, 255).astype(np.uint8)


def qr_scenes(seed=0, count=60):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        kind = ["num", "alnum", "byte", "utf8"][i % 4]
        text = random_text(rng, int(rng.integers(1, 120)), kind)
        level = int(rng.integers(0, 4))
        sym = cv_encode(text, level=level).encode(text)
        sym = cv2.copyMakeBorder(sym, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
        size = (480, 640) if i % 3 else (720, 1280)
        module = float(rng.uniform(3, 7)) if sym.shape[0] < 60 else float(rng.uniform(2.5, 4))
        module = min(module, 0.8 * min(size) / sym.shape[0])
        img = render(sym, module, rng, size, perspective=float(rng.uniform(0, 0.12)), blur=float(rng.uniform(0, 1.2)),
                     noise=float(rng.uniform(0, 12)), angle=float(rng.uniform(-180, 180)),
                     bg=background(rng, *size) if i % 2 else None)
        out.append((f"qr {i} {kind} len {len(text)} L{level}", img, text))
    return out


def order_free_dist(a, b):
    """Largest corner distance after the best cyclic alignment."""
    best = 1e9
    for r in range(4):
        best = min(best, float(np.abs(np.roll(a, r, 0) - b).max()))
    return best


def qr_detect_cases():
    det = cv2.QRCodeDetector()
    agree = cv_only = ours_only = both_fail = 0
    corner_err = []
    bad = []
    for name, img, text in qr_scenes(0, 60):
        ref, pts, _ = det.detectAndDecode(img)
        ours, opts = od.qr_detect_and_decode(img)
        if ref:
            if ours == ref:
                agree += 1
                corner_err.append(order_free_dist(opts, pts.reshape(4, 2)))
            else:
                cv_only += 1
                bad.append(name)
        elif ours:
            ours_only += 1
            if ours != text:
                bad.append(name + " (wrong text)")
        else:
            both_fail += 1
    detail = f"same {agree}, VISAR only {ours_only}, cv2 only {cv_only}, neither {both_fail}"
    record("QRCodeDetector.detectAndDecode (perspective/blur/noise)", cv_only == 0 and not bad, detail + " " + "; ".join(bad[:3]))
    if corner_err:
        med = float(np.median(corner_err))
        record(f"QR corners vs cv2 (median {med:.2f} px, max {max(corner_err):.2f})", med < 2.0)


def qr_multi_cases():
    det = cv2.QRCodeDetector()
    rng = np.random.default_rng(5)
    n = ok = extra = 0
    bad = []
    for i in range(16):
        h, w = 720, 1280
        img = background(rng, h, w)
        texts = []
        k = int(rng.integers(2, 5))
        cells = [(w * (0.2 + 0.6 * (j % 2)), h * (0.27 + 0.46 * (j // 2))) for j in range(4)]
        for j in range(k):
            t = random_text(rng, int(rng.integers(5, 60)), ["byte", "alnum"][j % 2])
            sym = cv_encode(t, level=int(rng.integers(0, 4))).encode(t)
            sym = cv2.copyMakeBorder(sym, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
            module = min(5.0, 300 / sym.shape[0])
            img = render(sym, module, rng, (h, w), perspective=0.05, blur=0.6, noise=4, angle=float(rng.uniform(-40, 40)),
                         center=cells[j], bg=img)
            texts.append(t)
        n += 1
        r_ok, r_texts, _ = det.detectAndDecodeMulti(img)[:3]
        o_ok, o_texts, _ = od.qr_detect_and_decode_multi(img)
        ref = sorted(t for t in (r_texts if r_ok else ()) if t)
        ours = sorted(t for t in o_texts if t)
        if all(t in ours for t in ref):
            ok += 1
            extra += len(ours) - len(ref)
        else:
            bad.append(f"scene {i}: cv2 {len(ref)} VISAR {len(ours)}")
    record(f"QRCodeDetector.detectAndDecodeMulti ({n} scenes, VISAR decodes {extra} more)", ok == n, "; ".join(bad[:3]))


def qr_garbage_cases():
    rng = np.random.default_rng(9)
    ok = True
    for shape in [(480, 640), (30, 30), (1, 1), (21, 500), (480, 640, 3)]:
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        try:
            od.qr_detect_and_decode_multi(img)
            od.qr_detect_and_decode(img)
        except vs.VisarError:
            ok = False
    record("QR detector on noise and tiny images", ok)


# --------------------------------------------------------------- ArUco

DICTS = {
    "4X4_50": cv2.aruco.DICT_4X4_50, "4X4_1000": cv2.aruco.DICT_4X4_1000, "5X5_100": cv2.aruco.DICT_5X5_100,
    "6X6_250": cv2.aruco.DICT_6X6_250, "7X7_1000": cv2.aruco.DICT_7X7_1000, "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def adaptive_threshold_cases():
    for (h, w) in [(480, 640), (37, 53), (720, 1280)]:
        g = base.natural(h, w, 1)
        for method in (0, 1):
            for kind in (0, 1):
                for block, delta in [(3, 7), (13, 7), (23, 7), (83, 2), (5, -2.5)]:
                    ref = cv2.adaptiveThreshold(g, 255, method, kind, block, delta)
                    ours = od.adaptive_threshold(g, 255, method, kind, block, delta)
                    n = int((ref != ours).sum())
                    if n:
                        record(f"adaptiveThreshold {w}x{h} m{method} t{kind} b{block}", False, f"{n} px differ")
                        return
    record("adaptiveThreshold (MEAN_C/GAUSSIAN_C, BINARY/INV, 5 blocks, 3 sizes)", True)


def aruco_generate_cases():
    for name, d in DICTS.items():
        dic = cv2.aruco.getPredefinedDictionary(d)
        n = dic.bytesList.shape[0]
        ok = True
        for mid in sorted({0, 1, n // 2, n - 1}):
            for side, bb in [(50, 1), (77, 1), (120, 2), (9, 1)]:
                if side < dic.markerSize + 2 * bb:
                    continue
                ref = cv2.aruco.generateImageMarker(dic, mid, side, borderBits=bb)
                ours = od.aruco_generate(d, mid, side, bb)
                ok &= bool((ref == ours).all())
        record(f"aruco generateImageMarker {name}", ok)


def aruco_scene(d, seed, h=720, w=1280, count=20, perspective=0.08, blur=0.6, noise=4.0, invert=False):
    rng = np.random.default_rng(seed)
    dic = cv2.aruco.getPredefinedDictionary(d)
    img = np.full((h, w), 255, np.uint8) if seed % 2 else background(rng, h, w)
    n = dic.bytesList.shape[0]
    ids = rng.choice(n, count, replace=False)
    cols = int(np.ceil(np.sqrt(count * w / h)))
    rows = int(np.ceil(count / cols))
    cw, ch = w / cols, h / rows
    for k, mid in enumerate(ids):
        cx = (k % cols + 0.5) * cw + rng.uniform(-0.1, 0.1) * cw
        cy = (k // cols + 0.5) * ch + rng.uniform(-0.1, 0.1) * ch
        side = int(min(cw, ch) * rng.uniform(0.45, 0.65))
        m = cv2.aruco.generateImageMarker(dic, int(mid), side)
        if invert:
            m = 255 - m
        m = cv2.copyMakeBorder(m, side // 8, side // 8, side // 8, side // 8, cv2.BORDER_CONSTANT,
                               value=0 if invert else 255)
        img = render(m, 1.0, rng, (h, w), perspective=perspective, angle=float(rng.uniform(-180, 180)),
                     center=(cx, cy), bg=img)
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise > 0:
        img = np.clip(img + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def cv_aruco(img, d, **kw):
    p = cv2.aruco.DetectorParameters()
    for k, v in kw.items():
        setattr(p, k, v)
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d), p)
    corners, ids, rej = det.detectMarkers(img)
    return corners, ids, rej


def compare_markers(name, ref, ours, tol):
    rc, ri, _ = ref
    oc, oi = ours[:2]
    ri = [] if ri is None else list(ri.ravel())
    oi = [] if oi is None else list(oi.ravel())
    if ri != oi:
        record(name, False, f"ids differ: cv2 {len(ri)} VISAR {len(oi)}")
        return
    err = 0.0
    for a, b in zip(rc, oc):
        err = max(err, float(np.abs(a.reshape(4, 2) - b.reshape(4, 2)).max()))
    record(f"{name} ({len(ri)} markers, max corner diff {err:.3g} px)", err <= tol)


def aruco_detect_cases():
    for seed, (dname, d) in enumerate(DICTS.items()):
        for variant in ["plain", "sharp", "small"]:
            if variant == "plain":
                img = aruco_scene(d, seed)
            elif variant == "sharp":
                img = aruco_scene(d, seed + 100, perspective=0.03, blur=0, noise=0)
            else:
                img = aruco_scene(d, seed + 200, h=480, w=640, count=12)
            name = f"aruco detectMarkers {dname} {variant} {img.shape[1]}x{img.shape[0]}"
            compare_markers(name, cv_aruco(img, d), od.aruco_detect(img, d), 0.0)
            ref = cv_aruco(img, d, cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX)
            ours = od.aruco_detect(img, d, cornerRefinementMethod=1)
            compare_markers(name + " SUBPIX", ref, ours, 0.01)
    d = cv2.aruco.DICT_6X6_250
    img = aruco_scene(d, 300, invert=True)
    compare_markers("aruco detectMarkers inverted", cv_aruco(img, d, detectInvertedMarker=True),
                    od.aruco_detect(img, d, detectInvertedMarker=True), 0.0)
    img = aruco_scene(d, 301, count=6)
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    compare_markers("aruco detectMarkers BGR input", cv_aruco(bgr, d), od.aruco_detect(bgr, d), 0.0)
    # Rejected candidates.
    rc, ri, rr = cv_aruco(img, d)
    oc, oi, orr = od.aruco_detect(img, d, rejected=True)
    key = lambda c: tuple(np.round(c.ravel(), 3))  # noqa: E731
    record(f"aruco rejected candidates ({len(rr)})", sorted(map(key, rr)) == sorted(map(key, orr)))
    # Garbage.
    rng = np.random.default_rng(4)
    ok = True
    for shape in [(480, 640), (5, 5), (1, 1), (64, 3)]:
        noise = rng.integers(0, 256, shape, dtype=np.uint8)
        try:
            ref = cv_aruco(noise, d) if min(shape) > 1 else ((), None, ())
            ours = od.aruco_detect(noise, d)
            ok &= (0 if ref[1] is None else len(ref[1])) == (0 if ours[1] is None else len(ours[1]))
        except (vs.VisarError, cv2.error):
            ok = False
    record("aruco detectMarkers on noise and tiny images", ok)


def aruco_draw_cases():
    d = cv2.aruco.DICT_4X4_50
    img = aruco_scene(d, 7, count=5)
    corners, ids, _ = cv_aruco(img, d)
    for with_ids in (True, False):
        a = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        b = a.copy()
        cv2.aruco.drawDetectedMarkers(a, corners, ids if with_ids else None, (0, 255, 0))
        od.aruco_draw(b, corners, ids if with_ids else None, (0, 255, 0))
        diff = int((a != b).any(axis=2).sum())
        # The sides match; the corner square is not anti-aliased and the
        # "id=" labels use VISAR's bitmap font instead of Hershey.
        frac = 0.005 if with_ids else 0.001
        record(f"aruco drawDetectedMarkers ids={with_ids} ({diff} px differ)", diff <= a.shape[0] * a.shape[1] * frac)


def main():
    vs.set_num_threads(1)
    print(f"VISAR {vs.version()} vs OpenCV {cv2.__version__}")
    encoder_cases()
    qr_detect_cases()
    qr_multi_cases()
    qr_garbage_cases()
    adaptive_threshold_cases()
    aruco_generate_cases()
    aruco_detect_cases()
    aruco_draw_cases()
    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[3]]
    for name, _, detail, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} objdetect cases match OpenCV")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
