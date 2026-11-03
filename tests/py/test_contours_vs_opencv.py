"""Differential tests for contours and shape analysis: VISAR (C ABI) against
OpenCV. Contours must match exactly: the same contours, starting points,
point order, contour order and hierarchy. Run: zig build difftest
"""
import os
import sys

import cv2

if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visar_ctypes as vs  # noqa: E402

rng = np.random.default_rng(4321)
results = []

MODES = [(cv2.RETR_EXTERNAL, "EXT"), (cv2.RETR_LIST, "LIST"), (cv2.RETR_CCOMP, "CCOMP"), (cv2.RETR_TREE, "TREE")]
METHODS = [(cv2.CHAIN_APPROX_NONE, "NONE"), (cv2.CHAIN_APPROX_SIMPLE, "SIMPLE"),
           (cv2.CHAIN_APPROX_TC89_L1, "L1"), (cv2.CHAIN_APPROX_TC89_KCOS, "KCOS")]


def compare(name, ours, ref, tol=0.0, pct=100.0):
    ours = np.asarray(ours)
    ref = np.asarray(ref)
    if ours.shape != ref.shape:
        results.append((name, "SHAPE", f"{ours.shape} vs {ref.shape}", False))
        return
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    mx = float(d.max()) if d.size else 0.0
    frac = 100.0 * float((d > 0).mean()) if d.size else 0.0
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


def flatten(contours, hierarchy):
    """One array holding the count, every length, every point and the hierarchy."""
    n = len(contours)
    parts = [np.array([n])]
    parts.append(np.array([len(c) for c in contours], np.int64))
    parts += [np.asarray(c).reshape(-1) for c in contours]
    if hierarchy is not None:
        parts.append(np.asarray(hierarchy).reshape(-1))
    return np.concatenate([p.astype(np.int64) for p in parts])


def masks():
    """Random and structured binary masks."""
    out = []
    for h, w in [(1, 1), (1, 7), (7, 1), (2, 2), (3, 5)]:
        out.append((f"ones {w}x{h}", np.full((h, w), 255, np.uint8)))
        out.append((f"zeros {w}x{h}", np.zeros((h, w), np.uint8)))
        out.append((f"noise {w}x{h}", (rng.random((h, w)) > 0.5).astype(np.uint8)))
    for d in (0.1, 0.4, 0.5, 0.6, 0.9):
        for h, w in [(17, 23), (64, 64), (200, 300)]:
            m = (rng.random((h, w)) < d).astype(np.uint8) * rng.integers(1, 256, (h, w), dtype=np.uint8)
            out.append((f"noise d{d} {w}x{h}", m))
    # Single pixels, lines, checkerboard.
    m = np.zeros((40, 50), np.uint8)
    m[::3, ::4] = 1
    out.append(("pixels", m))
    m = np.zeros((40, 50), np.uint8)
    cv2.line(m, (0, 0), (49, 39), 255)
    cv2.line(m, (0, 39), (49, 0), 255)
    cv2.line(m, (25, 0), (25, 39), 255)
    cv2.line(m, (0, 5), (49, 5), 255)
    out.append(("lines", m))
    out.append(("checker", (np.indices((31, 33)).sum(0) % 2).astype(np.uint8)))
    # Nested rings (deep tree) and rings with holes and islands.
    m = np.zeros((420, 420), np.uint8)
    for r in range(205, 0, -3):
        cv2.circle(m, (210, 210), r, 255 if (r // 3) % 2 == 0 else 0, -1)
    out.append(("nested rings", m))
    m = np.zeros((300, 300), np.uint8)
    for r in range(148, 0, -2):
        cv2.rectangle(m, (150 - r, 150 - r), (150 + r, 150 + r), 255 if (r // 2) % 2 == 0 else 0, -1)
    out.append(("nested squares", m))
    # Random blobs, outlines and holes; many contours so the 7-bit labels wrap.
    for k, (h, w, n) in enumerate([(120, 160, 30), (480, 640, 400), (240, 320, 300)]):
        m = np.zeros((h, w), np.uint8)
        for _ in range(n):
            x, y = int(rng.integers(-10, w + 10)), int(rng.integers(-10, h + 10))
            r = int(rng.integers(1, max(2, min(h, w) // 8)))
            kind = rng.integers(0, 5)
            if kind == 0:
                cv2.circle(m, (x, y), r, 255, -1)
            elif kind == 1:
                cv2.circle(m, (x, y), r, 255, int(rng.integers(1, 4)))
            elif kind == 2:
                cv2.ellipse(m, (x, y), (r, max(1, r // 2)), float(rng.integers(0, 180)), 0, 360, 255, -1)
            elif kind == 3:
                cv2.rectangle(m, (x, y), (x + r, y + r // 2 + 1), 0, -1)
            else:
                cv2.rectangle(m, (x, y), (x + r, y + r), 255, 1)
        out.append((f"blobs{k} {w}x{h}", m))
    # Touching borders.
    m = np.zeros((30, 40), np.uint8)
    m[0, :] = 1
    m[:, 0] = 1
    m[10:20, 35:] = 1
    m[29, 10:30] = 1
    m[12:18, 37] = 0
    out.append(("borders", m))
    # Threshold of a smooth random field.
    f = cv2.GaussianBlur(rng.random((360, 480)).astype(np.float32), (0, 0), 6)
    out.append(("field", (f > np.median(f)).astype(np.uint8)))
    # Strided view.
    big = (rng.random((50, 70)) > 0.5).astype(np.uint8)
    out.append(("view", big[5:45, 7:60]))
    return out


def rel(ours, ref):
    """Scales both sides by the reference magnitude so `tol` is relative."""
    ours = np.asarray(ours, np.float64).ravel()
    ref = np.asarray(ref, np.float64).ravel()
    if ours.shape != ref.shape:
        return ours, ref
    s = np.maximum(np.abs(ref), 1.0)
    return ours / s, ref / s


def point_sets():
    """Contours from the masks plus random and degenerate point sets."""
    sets = []
    for nm, m in masks():
        if not nm.startswith(("blobs0", "nested squares", "borders", "field", "lines", "noise d0.4 64x64", "pixels")):
            continue
        for meth, an in ((cv2.CHAIN_APPROX_NONE, "none"), (cv2.CHAIN_APPROX_SIMPLE, "simple")):
            cs, _ = cv2.findContours(m, cv2.RETR_LIST, meth)
            step = max(1, len(cs) // 25)
            for i in range(0, len(cs), step):
                sets.append((f"{nm} {an} #{i}", cs[i]))
    for k in range(40):
        n = int(rng.integers(1, 60))
        sets.append((f"random int #{k}", rng.integers(-500, 500, (n, 1, 2)).astype(np.int32)))
        sets.append((f"random float #{k}", (rng.random((n, 1, 2)) * 200 - 50).astype(np.float32)))
    for k in range(10):
        t = np.sort(rng.random(int(rng.integers(5, 40)))) * 2 * np.pi
        e = np.stack([60 + 40 * np.cos(t), 50 + 20 * np.sin(t)], -1).reshape(-1, 1, 2)
        sets.append((f"ellipse float #{k}", e.astype(np.float32)))
        sets.append((f"ellipse int #{k}", np.round(e).astype(np.int32)))
    sets.append(("one point", np.array([[[3, 4]]], np.int32)))
    sets.append(("two points", np.array([[[3, 4]], [[10, -2]]], np.int32)))
    sets.append(("vertical", np.array([[[3, 4]], [[3, 9]]], np.int32)))
    sets.append(("collinear", np.array([[[i, 2 * i]] for i in range(10)], np.int32)))
    sets.append(("duplicates", np.array([[[1, 1]], [[1, 1]], [[5, 1]], [[5, 1]], [[5, 6]], [[1, 1]]], np.int32)))
    sets.append(("square", np.array([[[0, 0]], [[0, 10]], [[10, 10]], [[10, 0]]], np.int32)))
    sets.append(("square float", np.array([[[0.5, 0]], [[0, 10]], [[10, 10.25]], [[10, 0]]], np.float32)))
    return sets


def shape_tests():
    sets = point_sets()
    for nm, p in sets:
        run(f"moments {nm}", lambda: rel([*vs.moments(p).values()], [*cv2.moments(p).values()]), 1e-9)
        run(f"HuMoments {nm}", lambda: rel(vs.hu_moments(cv2.moments(p)), cv2.HuMoments(cv2.moments(p))), 1e-12)
        run(f"contourArea {nm}", lambda: (np.array([vs.contour_area(p), vs.contour_area(p, True)]),
                                          np.array([cv2.contourArea(p), cv2.contourArea(p, True)])))
        run(f"arcLength {nm}", lambda: (np.array([vs.arc_length(p, True), vs.arc_length(p, False)]),
                                        np.array([cv2.arcLength(p, True), cv2.arcLength(p, False)])))
        run(f"boundingRect {nm}", lambda: (np.array(vs.bounding_rect(p)), np.array(cv2.boundingRect(p))))
        run(f"isContourConvex {nm}", lambda: (np.array([vs.is_contour_convex(p)]), np.array([cv2.isContourConvex(p)])))
        for cw in (False, True):
            run(f"convexHull points cw{int(cw)} {nm}", lambda: (vs.convex_hull(p, cw), cv2.convexHull(p, clockwise=cw)))
            run(f"convexHull indices cw{int(cw)} {nm}",
                lambda: (vs.convex_hull(p, cw, False), cv2.convexHull(p, clockwise=cw, returnPoints=False)))
        if p.dtype == np.int32 and len(p) > 3:
            def cd():
                h = cv2.convexHull(p, returnPoints=False)
                try:
                    ref = cv2.convexityDefects(p, h)
                except cv2.error:
                    ref = "error"
                try:
                    ours = vs.convexity_defects(p, h)
                except vs.VisarError:
                    ours = "error"
                if isinstance(ref, str) or isinstance(ours, str):
                    return np.array([isinstance(ours, str)]), np.array([isinstance(ref, str)])
                if ref is None or ours is None:
                    return np.array([ours is None]), np.array([ref is None])
                return ours, ref
            run(f"convexityDefects {nm}", cd)
        def mar():
            (c, s, a) = vs.min_area_rect(p)
            (rc, rs, ra) = cv2.minAreaRect(p)
            return np.array([*c, *s, a]), np.array([*rc, *rs, ra])
        run(f"minAreaRect {nm}", mar)
        run(f"boxPoints {nm}", lambda: (vs.box_points(cv2.minAreaRect(p)), cv2.boxPoints(cv2.minAreaRect(p))))
        def mec():
            (c, r) = vs.min_enclosing_circle(p)
            (rc, rr) = cv2.minEnclosingCircle(p)
            return np.array([*c, r]), np.array([*rc, rr])
        run(f"minEnclosingCircle {nm}", mec)
        for eps in (0.5, 1.0, 3.0, 10.0):
            for closed in (True, False):
                run(f"approxPolyDP eps{eps} closed{int(closed)} {nm}",
                    lambda: (vs.approx_poly_dp(p, eps, closed), cv2.approxPolyDP(p, eps, closed)))
        q = p.reshape(-1, 2).astype(np.float64)
        lo, hi = q.min(0) - 3, q.max(0) + 3
        tests = [tuple(x) for x in p.reshape(-1, 2)[:5].astype(np.float32)]
        tests += [(float(rng.uniform(lo[0], hi[0])), float(rng.uniform(lo[1], hi[1]))) for _ in range(8)]
        tests += [(float(round(rng.uniform(lo[0], hi[0]))), float(round(rng.uniform(lo[1], hi[1])))) for _ in range(8)]
        for md in (False, True):
            run(f"pointPolygonTest md{int(md)} {nm}",
                lambda: (np.array([vs.point_polygon_test(p, t, md) for t in tests]),
                         np.array([cv2.pointPolygonTest(p, t, md) for t in tests])), 1e-12)
    for k in range(30):
        a = sets[int(rng.integers(0, len(sets)))][1]
        b = sets[int(rng.integers(0, len(sets)))][1]
        for meth in (1, 2, 3):
            run(f"matchShapes I{meth} #{k}", lambda: rel([vs.match_shapes(a, b, meth)], [cv2.matchShapes(a, b, meth, 0)]), 1e-9)
    # Raster moments.
    for nm, m in masks():
        if not nm.startswith(("blobs", "field", "noise d0.4", "ones", "pixels", "view")):
            continue
        # Non-binary 8-bit moments go through IPP on x86 OpenCV builds,
        # which sums in a different order (relative error ~1e-9).
        for binary in (False, True):
            run(f"moments raster u8 bin{int(binary)} {nm}",
                lambda: rel([*vs.moments(m, binary).values()], [*cv2.moments(m, binary).values()]), 1e-8)
        run(f"boundingRect raster {nm}", lambda: (np.array(vs.bounding_rect(m)), np.array(cv2.boundingRect(m))))
    img = rng.integers(0, 256, (123, 77)).astype(np.uint8)
    for dt in (np.uint16, np.int16, np.float32):
        a = (img.astype(np.float32) * (1.7 if dt == np.float32 else 1) - (100 if dt == np.int16 else 0)).astype(dt)
        if dt == np.uint16:
            a = a * 250
        for binary in (False, True):
            run(f"moments raster {np.dtype(dt).name} bin{int(binary)}",
                lambda: rel([*vs.moments(a, binary).values()], [*cv2.moments(a, binary).values()]), 1e-9)
    # fitEllipse: only well-posed sets. On degenerate ones (lines, L shapes,
    # duplicates, few-pixel blobs) the fit is ill-conditioned and OpenCV's
    # Jacobi SVD and VISAR's QR least squares legitimately diverge (OpenCV
    # also jitters such sets with its global RNG).
    for nm, p in sets:
        if len(p) < 6 or not nm.startswith(("ellipse", "random", "blobs0", "nested", "field")) or len(p) < 20 and nm[0] in "bnf":
            continue
        def fe():
            (c, s, a) = vs.fit_ellipse(p)
            (rc, rs, ra) = cv2.fitEllipse(p)
            return rel([*c, *s, a], [*rc, *rs, ra])
        run(f"fitEllipse {nm}", fe, 1e-4)


def draw_tests():
    """drawContours / fillPoly / polylines against OpenCV, pixel for pixel."""
    srcs = [m for nm, m in masks() if nm.startswith(("blobs", "nested", "borders", "field", "lines", "noise d0.4 64x64"))]
    for k, m in enumerate(srcs):
        h, w = m.shape
        cs, hier = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cs = list(cs)
        for th in (-1, 1, 2, 3, 6):
            for lt in (8, 4):
                for idx, lvl in ((-1, 0), (-1, 1), (-1, 2**31 - 1), (0, 0), (len(cs) // 2, 1)):
                    if idx >= len(cs):
                        continue
                    for cn in (1, 3):
                        def dc():
                            a = np.full((h, w, cn) if cn > 1 else (h, w), 17, np.uint8)
                            b = a.copy()
                            vs.draw_contours(a, cs, idx, (200, 100, 50), th, lt, hier, lvl, (1, -2))
                            cv2.drawContours(b, cs, idx, (200, 100, 50), th, lt, hier, lvl, (1, -2))
                            return a, b
                        if cn == 3 and (th not in (-1, 3) or lt != 8):
                            continue
                        run(f"drawContours #{k} th{th} lt{lt} idx{idx} lvl{lvl} cn{cn}", dc)
            def dn():
                a = np.zeros((h, w), np.uint8)
                b = a.copy()
                vs.draw_contours(a, cs, -1, 255, th)
                cv2.drawContours(b, cs, -1, 255, th)
                return a, b
            run(f"drawContours #{k} th{th} no hierarchy", dn)
    # Random polygons, partly outside the image.
    for k in range(60):
        h, w = 97, 131
        polys = [rng.integers(-60, 190, (int(rng.integers(1, 12)), 2)).astype(np.int32) for _ in range(int(rng.integers(1, 4)))]
        th = int(rng.choice([1, 2, 5, 13]))
        lt = int(rng.choice([4, 8]))
        def fp():
            a = np.zeros((h, w, 3), np.uint8)
            b = a.copy()
            vs.fill_poly(a, polys, (10, 20, 30), lt)
            cv2.fillPoly(b, polys, (10, 20, 30), lt)
            return a, b
        run(f"fillPoly random #{k}", fp)
        def pl():
            a = np.zeros((h, w), np.uint8)
            b = a.copy()
            closed = bool(k % 2)
            vs.polylines(a, polys, closed, 255, th, lt)
            cv2.polylines(b, polys, closed, 255, th, lt)
            return a, b
        run(f"polylines random #{k} th{th} lt{lt}", pl)


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    for nm, m in masks():
        for mode, mn in MODES:
            for method, an in METHODS:
                def fc():
                    c, h = vs.find_contours(m, mode, method)
                    rc, rh = cv2.findContours(m, mode, method)
                    return flatten(c, h), flatten(rc, rh)
                run(f"findContours {mn} {an} {nm}", fc)
        def fo():
            c, h = vs.find_contours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, (5, -3))
            rc, rh = cv2.findContours(m, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE, offset=(5, -3))
            return flatten(c, h), flatten(rc, rh)
        run(f"findContours offset {nm}", fo)

    draw_tests()
    shape_tests()

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
