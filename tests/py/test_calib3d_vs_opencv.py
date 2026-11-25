"""Differential tests for calib3d: homographies and affine estimation,
Rodrigues / projectPoints / undistortPoints, solvePnP(Ransac), undistort
and remap, cornerSubPix, findChessboardCorners and calibrateCamera, VISAR
(through its C ABI) against OpenCV. Run: zig build difftest

Clean (noise-free) data must agree to rounding. On noisy data the robust
estimators draw the same samples as OpenCV (same RNG and seed), so inlier
masks must agree (up to points within rounding of the threshold) and the
refined models must agree closely; poses and calibrations are compared
through their parameters and reprojection errors.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing the core suite handles `--lib` and provides the helpers.
import test_core_vs_opencv as base  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import visar_calib3d as vc  # noqa: E402
import visar_ctypes as vs  # noqa: E402

run = base.run
results = base.results
natural = base.natural
rng = np.random.default_rng(77)

K = np.array([[900.0, 0, 640.5], [0, 880.0, 359.2], [0, 0, 1]])
DISTS = {
    "none": np.zeros(5),
    "k1k2": np.array([-0.28, 0.09, 0, 0]),
    "full5": np.array([-0.28, 0.09, 0.001, -0.0005, -0.01]),
    "rational8": np.array([0.1, -0.2, 0.002, 0.001, 0.05, 0.01, 0.002, 0.003]),
}


def check(name, ok, detail=""):
    results.append((name, "ok" if ok else "FAIL", detail, bool(ok)))


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max() / max(1.0, np.abs(b).max()))


def homography_scene(n, outliers, noise, seed):
    r = np.random.default_rng(seed)
    H = np.array([[1.1, 0.05, 20], [-0.03, 0.95, 10], [2e-4, -1e-4, 1]])
    H[:2, :2] += r.normal(0, 0.05, (2, 2))
    src = r.uniform(0, [1280, 720], (n, 2)).astype(np.float32)
    dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    dst = (dst + r.normal(0, noise, dst.shape)).astype(np.float32)
    out = r.random(n) < outliers
    dst[out] = r.uniform(0, [1280, 720], (int(out.sum()), 2))
    return src, dst


def homographies():
    for seed in range(6):
        src, dst = homography_scene(60, 0, 0, seed)
        run(f"findHomography all clean {seed}", lambda: (vc.find_homography(src, dst, 0)[0] / 100, cv2.findHomography(src, dst, 0)[0] / 100), 1e-7)
        for n, frac, noise in ((500, 0.3, 0.5), (100, 0.4, 1.0), (40, 0.2, 0.3)):
            src, dst = homography_scene(n, frac, noise, 100 + seed)
            for method, nm in ((cv2.RANSAC, "RANSAC"), (cv2.LMEDS, "LMEDS")):
                hv, mv = vc.find_homography(src, dst, method)
                hc, mc = cv2.findHomography(src, dst, method)
                tag = f"findHomography {nm} n={n} out={frac} s={seed}"
                check(tag + " found", (hv is None) == (hc is None))
                if hv is None or hc is None:
                    continue
                # Masks: equal except points within rounding of the threshold.
                run(tag + " mask", lambda: (mv, mc), 1, 1.0)
                run(tag + " H", lambda: (hv / max(1.0, np.abs(hc).max()), hc / max(1.0, np.abs(hc).max())), 1e-4)
        src, dst = homography_scene(200, 0, 0.5, 200 + seed)
        run(f"findHomography all noisy {seed}", lambda: (vc.find_homography(src, dst, 0)[0], cv2.findHomography(src, dst, 0)[0]), 1e-5)
        s4, d4 = src[:4], dst[:4]
        run(f"getPerspectiveTransform {seed}", lambda: (vc.get_perspective_transform(s4, d4), cv2.getPerspectiveTransform(s4, d4)), 1e-9)
        run(f"getAffineTransform {seed}", lambda: (vc.get_affine_transform(s4[:3], d4[:3]), cv2.getAffineTransform(s4[:3], d4[:3])), 1e-9)
        Hm = cv2.getPerspectiveTransform(s4, d4)
        run(f"perspectiveTransform {seed}", lambda: (vc.perspective_transform(src, Hm), cv2.perspectiveTransform(src.reshape(-1, 1, 2), Hm).reshape(-1, 2)), 2e-3)


def affines():
    for seed in range(6):
        r = np.random.default_rng(300 + seed)
        n = 200
        src = r.uniform(0, [1280, 720], (n, 2)).astype(np.float32)
        a = r.uniform(-0.3, 0.3)
        sc = r.uniform(0.7, 1.3)
        A = np.array([[sc * np.cos(a), -sc * np.sin(a), 15], [sc * np.sin(a), sc * np.cos(a), -9]])
        for partial in (False, True):
            if not partial:
                A[0, 1] += 0.05
            dst = (src @ A[:, :2].T + A[:, 2]).astype(np.float32) + r.normal(0, 0.7, (n, 2)).astype(np.float32)
            out = r.random(n) < 0.3
            dst[out] = r.uniform(0, [1280, 720], (int(out.sum()), 2))
            f = cv2.estimateAffinePartial2D if partial else cv2.estimateAffine2D
            for method, nm in ((cv2.RANSAC, "RANSAC"), (cv2.LMEDS, "LMEDS")):
                av, iv = vc.estimate_affine_2d(src, dst, partial, method)
                ac, ic = f(src, dst, method=method)
                tag = f"estimateAffine{'Partial' if partial else ''}2D {nm} s={seed}"
                check(tag + " found", (av is None) == (ac is None))
                if av is None or ac is None:
                    continue
                run(tag + " inliers", lambda: (iv, ic), 1, 1.0)
                run(tag + " M", lambda: (av, ac), 1e-4)


def camera_models():
    for i in range(5):
        r = rng.normal(0, 1.2, 3)
        run(f"Rodrigues vec->mat {i}", lambda: (vc.rodrigues(r), cv2.Rodrigues(r)[0]), 1e-12)
        run(f"Rodrigues jacobian {i}", lambda: (vc.rodrigues(r, True)[1], cv2.Rodrigues(r)[1]), 1e-12)
        R = cv2.Rodrigues(r)[0]
        run(f"Rodrigues mat->vec {i}", lambda: (vc.rodrigues(R), cv2.Rodrigues(R)[0]), 1e-12)
    R180 = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))[0]
    run("Rodrigues mat->vec 180deg", lambda: (vc.rodrigues(R180), cv2.Rodrigues(R180)[0]), 1e-9)
    run("Rodrigues zero", lambda: (vc.rodrigues(np.zeros(3), True)[1], cv2.Rodrigues(np.zeros(3))[1]), 0)
    obj = rng.uniform(-1, 1, (200, 3))
    rv = np.array([0.3, -0.2, 0.15])
    tv = np.array([0.2, -0.1, 5.0])
    for nm, d in DISTS.items():
        pc, jc = cv2.projectPoints(obj, rv, tv, K, d)
        pv = vc.project_points(obj, rv, tv, K, d)
        run(f"projectPoints {nm}", lambda: (pv, pc.reshape(-1, 2)), 1e-9)
        if len(d) <= 5:
            _, jv = vc.project_points(obj, rv, tv, K, d, jacobian=True)
            run(f"projectPoints jacobian {nm}", lambda: (jv, jc[:, :jv.shape[1]]), 1e-7 * max(1, np.abs(jc).max()))
        run(f"undistortPoints {nm}", lambda: (vc.undistort_points(pv, K, d), cv2.undistortPoints(pv.reshape(-1, 1, 2), K, d).reshape(-1, 2)), 1e-12)
        Rr = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
        P = K.copy()
        P[0, 0] *= 0.9
        run(f"undistortPoints R P {nm}", lambda: (vc.undistort_points(pv, K, d, Rr, P), cv2.undistortPoints(pv.reshape(-1, 1, 2), K, d, R=Rr, P=P).reshape(-1, 2)), 1e-9)


def reproj(obj, img, r, t, d):
    return float(np.linalg.norm(cv2.projectPoints(obj, r, t, K, d)[0].reshape(-1, 2) - img, axis=1).mean())


def pnp():
    d = DISTS["full5"]
    for seed in range(5):
        r = np.random.default_rng(500 + seed)
        rv = r.normal(0, 0.4, 3)
        tv = np.array([r.uniform(-0.3, 0.3), r.uniform(-0.3, 0.3), r.uniform(4, 7)])
        for planar in (False, True):
            obj = r.uniform(-1, 1, (80, 3))
            if planar:
                obj[:, 2] = 0
            img = cv2.projectPoints(obj, rv, tv, K, d)[0].reshape(-1, 2)
            for noise in (0.0, 0.5):
                im = img + r.normal(0, noise, img.shape)
                for flags, nm in ((cv2.SOLVEPNP_ITERATIVE, "ITERATIVE"), (cv2.SOLVEPNP_EPNP, "EPNP")):
                    a = vc.solve_pnp(obj, im, K, d, 0 if flags == cv2.SOLVEPNP_ITERATIVE else 1)
                    _, r2, t2 = cv2.solvePnP(obj, im, K, d, flags=flags)
                    tag = f"solvePnP {nm} {'planar' if planar else '3d'} noise={noise} s={seed}"
                    tol = 1e-6 if noise == 0 else 1e-5
                    run(tag + " rvec", lambda: (a[0], r2), tol)
                    run(tag + " tvec", lambda: (a[1], t2), tol * 10)
            # RANSAC with 30% outliers
            im = img + r.normal(0, 0.5, img.shape)
            out = r.random(len(im)) < 0.3
            im[out] = r.uniform(0, [1280, 720], (int(out.sum()), 2))
            ok1, r1, t1, i1 = vc.solve_pnp_ransac(obj, im, K, d)
            ok2, r2, t2, i2 = cv2.solvePnPRansac(obj, im, K, d)
            tag = f"solvePnPRansac {'planar' if planar else '3d'} s={seed}"
            check(tag + " found", ok1 == ok2)
            if ok1 and ok2:
                s1, s2 = set(i1.ravel()), set(i2.ravel())
                check(tag + " inliers", len(s1 ^ s2) <= max(1, len(s2) // 100), f"{len(s1 ^ s2)} differ")
                run(tag + " rvec", lambda: (r1, r2), 1e-4)
                run(tag + " tvec", lambda: (t1, t2), 1e-3)


def undistortion():
    for (h, w) in ((720, 1280), (37, 53)):
        Kw = K.copy()
        Kw[0, 2], Kw[1, 2] = w / 2 - 0.5, h / 2 + 0.3
        Kw[0, 0] = Kw[1, 1] = 0.7 * w
        for nm, d in DISTS.items():
            for cn in (1, 3, 4):
                img = natural(h, w, cn) if cn != 4 else np.dstack([natural(h, w, 3), natural(h, w, 1)])
                run(f"undistort {nm} c{cn} {w}x{h}", lambda: (vc.undistort(img, Kw, d), cv2.undistort(img, Kw, d)), 0)
            nk = Kw.copy()
            nk[0, 0] *= 0.8
            img = natural(h, w, 3)
            run(f"undistort newK {nm} {w}x{h}", lambda: (vc.undistort(img, Kw, d, nk), cv2.undistort(img, Kw, d, None, nk)), 0)
            Rr = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
            mx, my = vc.init_undistort_rectify_map(Kw, d, Rr, nk, (w, h))
            cx, cy = cv2.initUndistortRectifyMap(Kw, d, Rr, nk, (w, h), cv2.CV_32FC1)
            run(f"initUndistortRectifyMap {nm} {w}x{h} x", lambda: (mx, cx), 1e-4 * max(w, 1))
            run(f"initUndistortRectifyMap {nm} {w}x{h} y", lambda: (my, cy), 1e-4 * max(h, 1))
            for cn in (1, 3):
                src = natural(h, w, cn)
                run(f"remap linear {nm} c{cn} {w}x{h}", lambda: (vc.remap(src, cx, cy), cv2.remap(src, cx, cy, cv2.INTER_LINEAR)), 0)


def board(cols, rows, sq=40, margin=60):
    w, h = cols * sq + 2 * margin, rows * sq + 2 * margin
    img = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[margin + r * sq: margin + (r + 1) * sq, margin + c * sq: margin + (c + 1) * sq] = 0
    return img


def board_scene(cols, rows, angle, persp=0.0, size=(1280, 720), scale=0.7, noise=0.0, blur=0.0, seed=0):
    """A chessboard of cols x rows squares rotated by `angle` degrees and
    warped by OpenCV, then blurred and noised."""
    b = board(cols, rows)
    h, w = b.shape
    W, H = size
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    a = np.deg2rad(angle)
    R = np.float32([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    dst = (src - [w / 2, h / 2]) * (min(W, H) * scale / max(w, h)) @ R.T + [W / 2, H / 2]
    dst[1] += [0, persp * 50]
    dst[2] += [persp * 30, 0]
    M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    img = cv2.warpPerspective(b, M, size, flags=cv2.INTER_LINEAR, borderValue=180)
    if blur:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if noise:
        r = np.random.default_rng(seed)
        img = np.clip(img + r.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def chessboards():
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # cornerSubPix from the same starting points (including near the border).
    img = board_scene(10, 7, 12.5, 0.3, noise=3, blur=0.8)
    ok, c = cv2.findChessboardCorners(img, (9, 6))
    c = c.reshape(-1, 1, 2)
    for win in ((11, 11), (5, 5), (3, 7)):
        start = c + rng.uniform(-1.5, 1.5, c.shape).astype(np.float32)
        run(f"cornerSubPix win={win}", lambda: (vc.corner_sub_pix(img, start, win), cv2.cornerSubPix(img, start.copy(), win, (-1, -1), crit).reshape(-1, 2)), 1e-4)
    run("cornerSubPix zero zone", lambda: (vc.corner_sub_pix(img, c, (7, 7), (2, 2)), cv2.cornerSubPix(img, c.copy(), (7, 7), (2, 2), crit).reshape(-1, 2)), 1e-4)
    edge = np.float32([[2.5, 3.5], [img.shape[1] - 2.2, 10.0], [100.0, img.shape[0] - 1.5]]).reshape(-1, 1, 2)
    run("cornerSubPix near border", lambda: (vc.corner_sub_pix(img, edge, (5, 5)), cv2.cornerSubPix(img, edge.copy(), (5, 5), (-1, -1), crit).reshape(-1, 2)), 1e-3)
    f = img.astype(np.float32)
    run("cornerSubPix f32", lambda: (vc.corner_sub_pix(f, c, (5, 5)), cv2.cornerSubPix(f, c.copy(), (5, 5), (-1, -1), crit).reshape(-1, 2)), 1e-3)

    found = agree = 0
    for pat in [(9, 6), (8, 6), (7, 7), (6, 9), (7, 5), (4, 3)]:
        for angle in range(0, 360, 30):
            for persp, noise, blur, scale in ((0.0, 0, 0, 0.7), (0.3, 4, 0.8, 0.5), (0.5, 8, 1.2, 0.85)):
                img = board_scene(pat[0] + 1, pat[1] + 1, angle + 0.5, persp, noise=noise, blur=blur, scale=scale)
                if (angle // 30) % 3 == 1:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                ok1, c1 = vc.find_chessboard_corners(img, pat)
                ok2, c2 = cv2.findChessboardCorners(img, pat)
                tag = f"findChessboardCorners {pat} a={angle} p={persp} n={noise}"
                check(tag + " found", ok1 == ok2)
                if not (ok1 and ok2):
                    continue
                found += 1
                c1, c2 = c1.reshape(-1, 2), c2.reshape(-1, 2)
                if pat[0] == pat[1] and np.abs(c1 - c2).max() > 3:
                    # Square boards are symmetric under quarter turns; which
                    # corner OpenCV starts from follows its quad traversal.
                    # VISAR starts at the top-most corner, which is OpenCV's
                    # choice in most views: accept any rotation of the grid.
                    g = c2.reshape(pat[1], pat[0], 2)
                    for k in range(1, 4):
                        rot = np.rot90(g, k).reshape(-1, 2)
                        if np.abs(c1 - rot).max() <= 3:
                            c2 = rot
                            break
                # Same order; the raw corners come from OpenCV's own 5x5
                # cornerSubPix pass, started from slightly different quad
                # vertices, so they agree to a couple of pixels...
                run(tag + " raw", lambda: (c1, c2), 3.0)
                # ...and to a few hundredths once refined as applications do.
                g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                r1 = vc.corner_sub_pix(g, c1, (11, 11))
                r2 = cv2.cornerSubPix(g, c2.reshape(-1, 1, 2).copy(), (11, 11), (-1, -1), crit).reshape(-1, 2)
                run(tag + " refined", lambda: (r1, r2), 0.05)
                agree += 1
    check(f"findChessboardCorners boards found ({found})", found > 150)
    blank = natural(480, 640, 1)
    check("findChessboardCorners no board", not vc.find_chessboard_corners(blank, (9, 6))[0])


def calibration():
    r = np.random.default_rng(900)
    Kt = np.array([[820.0, 0, 650], [0, 800, 355], [0, 0, 1]])
    d = np.array([-0.21, 0.06, 0.0012, -0.0007, -0.01])
    objp = np.zeros((54, 3))
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    for noise in (0.0, 0.2):
        objs, imgs = [], []
        for v in range(10):
            rv = r.normal(0, 0.3, 3)
            tv = np.array([r.uniform(-0.12, 0.0), r.uniform(-0.08, 0.0), r.uniform(0.35, 0.6)])
            p = cv2.projectPoints(objp, rv, tv, Kt, d)[0].reshape(-1, 2)
            objs.append(objp.astype(np.float32))
            imgs.append((p + r.normal(0, noise, p.shape)).astype(np.float32))
        for flags, nm in ((0, "default"), (cv2.CALIB_ZERO_TANGENT_DIST, "zero tangent"), (cv2.CALIB_FIX_K3, "fix k3"),
                          (cv2.CALIB_FIX_PRINCIPAL_POINT, "fix pp")):
            a = vc.calibrate_camera(objs, imgs, (1280, 720), flags=flags)
            b = cv2.calibrateCamera(objs, imgs, (1280, 720), None, None, flags=flags)
            tag = f"calibrateCamera {nm} noise={noise}"
            run(tag + " rms", lambda: (np.array([a[0]]), np.array([b[0]])), 1e-6)
            run(tag + " K", lambda: (a[1], b[1]), 1e-3)
            run(tag + " dist", lambda: (a[2], b[2].reshape(1, -1)[:, :5]), 1e-5)
            run(tag + " rvecs", lambda: (np.array(a[3]), np.array(b[3])), 1e-6)
            run(tag + " tvecs", lambda: (np.array(a[4]), np.array(b[4])), 1e-6)
        # With a guess and a fixed focal length.
        g = Kt * [[1.02], [0.98], [1]]
        a = vc.calibrate_camera(objs, imgs, (1280, 720), g, np.zeros(5), cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH)
        b = cv2.calibrateCamera(objs, imgs, (1280, 720), g.copy(), np.zeros(5), flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_FOCAL_LENGTH)
        run(f"calibrateCamera guess+fix focal noise={noise} rms", lambda: (np.array([a[0]]), np.array([b[0]])), 1e-6)
        run(f"calibrateCamera guess+fix focal noise={noise} K", lambda: (a[1], b[1]), 1e-3)


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    homographies()
    affines()
    camera_models()
    pnp()
    undistortion()
    chessboards()
    calibration()
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
