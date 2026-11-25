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
    yield from area_cases()
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
    pd_bgr = np.empty(((h + 1) // 2, (w + 1) // 2, 3), np.uint8)
    pd_gray = np.empty(((h + 1) // 2, (w + 1) // 2), np.uint8)
    pu_gray = np.empty((h * 2, w * 2), np.uint8)
    yield "pyrDown u8", lambda: vs.pyr_down(bgr, dst=pd_bgr), lambda: cv2.pyrDown(bgr, dst=pd_bgr)
    yield "pyrDown u8 gray", lambda: vs.pyr_down(gray, dst=pd_gray), lambda: cv2.pyrDown(gray, dst=pd_gray)
    yield "pyrUp u8 gray", lambda: vs.pyr_up(gray, dst=pu_gray), lambda: cv2.pyrUp(gray, dst=pu_gray)
    pd_f = np.empty(((h + 1) // 2, (w + 1) // 2, 3), np.float32)
    yield "pyrDown f32", lambda: vs.pyr_down(f, dst=pd_f), lambda: cv2.pyrDown(f, dst=pd_f)
    eig = np.empty((h, w), np.float32)
    yield "cornerMinEigenVal 3/3", lambda: vs.corner_min_eigen_val(gray, 3, 3, dst=eig), lambda: cv2.cornerMinEigenVal(gray, 3, dst=eig, ksize=3)
    yield "cornerHarris 2/3", lambda: vs.corner_harris(gray, 2, 3, 0.04, dst=eig), lambda: cv2.cornerHarris(gray, 2, 3, 0.04, dst=eig)
    yield "goodFeaturesToTrack 1000", lambda: vs.good_features_to_track(gray, 1000, 0.01, 10), lambda: cv2.goodFeaturesToTrack(gray, 1000, 0.01, 10)
    lk_a = cv2.GaussianBlur(gray, (3, 3), 0)
    lk_b = cv2.warpAffine(lk_a, np.float32([[1, 0, 3.4], [0, 1, -2.1]]), (w, h), borderMode=cv2.BORDER_REFLECT101)
    lk_pts = cv2.goodFeaturesToTrack(lk_a, 1000, 0.01, 8).reshape(-1, 2)
    yield f"calcOpticalFlowPyrLK {len(lk_pts)}pt", lambda: vs.calc_optical_flow_pyr_lk(lk_a, lk_b, lk_pts), lambda: cv2.calcOpticalFlowPyrLK(lk_a, lk_b, lk_pts.reshape(-1, 1, 2), None)
    ok, png = cv2.imencode(".png", bgr)
    pngb = png.tobytes()
    yield "imdecode PNG", lambda: vs.imdecode(pngb), lambda: cv2.imdecode(png, cv2.IMREAD_UNCHANGED)
    yield "imencode PNG", lambda: vs.imencode(".png", bgr), lambda: cv2.imencode(".png", bgr)
    ok, jpg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    jpgb = jpg.tobytes()
    yield "imdecode JPEG", lambda: vs.imdecode(jpgb), lambda: cv2.imdecode(jpg, cv2.IMREAD_UNCHANGED)
    yield "imencode JPEG q90", lambda: vs.imencode(".jpg", bgr, quality=90), lambda: cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    yield from contour_cases(h, w)
    yield from video_cases(h, w)
    yield from dnn_cases()
    yield from tracking_cases()
    yield from features2d_cases()
    yield from calib3d_cases()
    yield from photo_cases()


def photo_cases():
    """photo at fixed sizes (independent of --size): non-local means with
    OpenCV's defaults for 7/21 windows, inpainting a ~5% mask."""
    if not hasattr(vs.lib, "visar_inpaint"):
        return
    import photo_ctypes as ph
    from test_photo_vs_opencv import noisy, scratches
    for (w, h) in [(640, 480), (1280, 720)]:
        g, c = noisy(h, w, 1, seed=1), noisy(h, w, 3, seed=2)
        yield f"fastNlMeansDenoising h10 t7 s21 {w}x{h}", (lambda g=g: ph.fast_nl_means_denoising(g, 10, 7, 21)), \
            (lambda g=g: cv2.fastNlMeansDenoising(g, None, 10, 7, 21))
        yield f"fastNlMeansDenoisingColored h10 {w}x{h}", (lambda c=c: ph.fast_nl_means_denoising_colored(c, 10, 10)), \
            (lambda c=c: cv2.fastNlMeansDenoisingColored(c, None, 10, 10))
        m = scratches(h, w, seed=3)
        for flags, name in ((cv2.INPAINT_TELEA, "TELEA"), (cv2.INPAINT_NS, "NS")):
            for img, kind in ((c, "BGR"), (g, "gray")):
                yield f"inpaint {name} r3 5% {kind} {w}x{h}", (lambda img=img, f=flags, m=m: ph.inpaint(img, m, 3, f)), \
                    (lambda img=img, f=flags, m=m: cv2.inpaint(img, m, 3, f))


def feature_image(h, w, seed=0):
    """natural() with extra noise and random boxes: enough corners that ORB
    fills its feature budget."""
    rng = np.random.default_rng(seed)
    img = natural(h, w, 1).astype(np.float32) + rng.normal(0, 20, (h, w))
    for _ in range(h * w // 1500):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(3, 40)), y + int(rng.integers(3, 40))), float(rng.integers(0, 256)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def features2d_cases():
    """FAST, ORB and BFMatcher at fixed sizes (independent of --size)."""
    if not hasattr(vs.lib, "visar_orb_detect_and_compute"):
        return
    import features2d_ctypes as fd
    for (w, h) in [(640, 480), (1280, 720)]:
        g = feature_image(h, w)
        fast = cv2.FastFeatureDetector_create(20, True)
        yield f"FAST-9 t20 nms {w}x{h}", (lambda g=g: fd.fast(g, 20, True)), (lambda g=g, f=fast: f.detect(g))
        for n in (500, 1000):
            orb = cv2.ORB_create(n)
            yield f"ORB {n} detect+compute {w}x{h}", (lambda g=g, n=n: fd.orb(g, nfeatures=n)), (lambda g=g, o=orb: o.detectAndCompute(g, None))
    rng = np.random.default_rng(5)
    q = rng.integers(0, 256, (1000, 32), dtype=np.uint8)
    t = rng.integers(0, 256, (1000, 32), dtype=np.uint8)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    bfc = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    yield "BFMatcher Hamming match 1000x1000", (lambda: fd.bf_match(q, t)), (lambda: bf.match(q, t))
    yield "BFMatcher Hamming knn2 1000x1000", (lambda: fd.bf_knn_match(q, t, 2)), (lambda: bf.knnMatch(q, t, 2))
    yield "BFMatcher Hamming crossCheck 1000x1000", (lambda: fd.bf_match(q, t, cross_check=True)), (lambda: bfc.match(q, t))
    qf = rng.random((1000, 128), dtype=np.float32)
    tf = rng.random((1000, 128), dtype=np.float32)
    bfl = cv2.BFMatcher(cv2.NORM_L2)
    yield "BFMatcher L2 f32x128 knn2 1000x1000", (lambda: fd.bf_knn_match(qf, tf, 2, fd.NORM_L2)), (lambda: bfl.knnMatch(qf, tf, 2))


def calib3d_cases():
    """calib3d at fixed sizes (independent of --size)."""
    import visar_calib3d as vc
    from test_calib3d_vs_opencv import board_scene, homography_scene
    src, dst = homography_scene(500, 0.3, 0.5, 1)
    yield "findHomography RANSAC 500pt 30% out", lambda: vc.find_homography(src, dst, 8), lambda: cv2.findHomography(src, dst, cv2.RANSAC)
    yield "findHomography LMEDS 500pt 30% out", lambda: vc.find_homography(src, dst, 4), lambda: cv2.findHomography(src, dst, cv2.LMEDS)
    a_src, a_dst = homography_scene(500, 0.3, 0.5, 2)
    a_dst = (a_src @ np.float32([[0.9, -0.2], [0.2, 0.9]]).T + [5, -7]).astype(np.float32)
    yield "estimateAffine2D RANSAC 500pt", lambda: vc.estimate_affine_2d(a_src, a_dst), lambda: cv2.estimateAffine2D(a_src, a_dst)
    from test_calib3d_vs_opencv import two_views
    f1, f2, _ = two_views(300, 0.3, 0.3, 5)
    yield "findFundamentalMat RANSAC 300pt 30% out", lambda: vc.find_fundamental_mat(f1, f2, 8), lambda: cv2.findFundamentalMat(f1, f2, cv2.FM_RANSAC)
    r = np.random.default_rng(3)
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]])
    d = np.array([-0.28, 0.09, 0.001, -0.0005, -0.01])
    obj = r.uniform(-1, 1, (100, 3))
    img = cv2.projectPoints(obj, np.array([0.3, -0.2, 0.1]), np.array([0.1, -0.1, 5.0]), K, d)[0].reshape(-1, 2)
    img = img + r.normal(0, 0.5, img.shape)
    yield "solvePnP ITERATIVE 100pt", lambda: vc.solve_pnp(obj, img, K, d), lambda: cv2.solvePnP(obj, img, K, d)
    yield "solvePnP EPNP 100pt", lambda: vc.solve_pnp(obj, img, K, d, 1), lambda: cv2.solvePnP(obj, img, K, d, flags=cv2.SOLVEPNP_EPNP)
    img_o = img.copy()
    out = r.random(len(img)) < 0.3
    img_o[out] = r.uniform(0, [1280, 720], (int(out.sum()), 2))
    yield "solvePnPRansac 100pt 30% out", lambda: vc.solve_pnp_ransac(obj, img_o, K, d), lambda: cv2.solvePnPRansac(obj, img_o, K, d)
    yield "projectPoints 1000pt", lambda: vc.project_points(np.tile(obj, (10, 1)), [0.3, -0.2, 0.1], [0.1, -0.1, 5.0], K, d), \
        lambda: cv2.projectPoints(np.tile(obj, (10, 1)), np.array([0.3, -0.2, 0.1]), np.array([0.1, -0.1, 5.0]), K, d)
    board = board_scene(10, 7, 20.5, 0.3, noise=3, blur=0.8)
    yield "findChessboardCorners 9x6 1280x720", lambda: vc.find_chessboard_corners(board, (9, 6)), lambda: cv2.findChessboardCorners(board, (9, 6))
    ok, corners = cv2.findChessboardCorners(board, (9, 6))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    yield "cornerSubPix 54pt 11x11", lambda: vc.corner_sub_pix(board, corners, (11, 11)), lambda: cv2.cornerSubPix(board, corners.copy(), (11, 11), (-1, -1), crit)
    bgr = natural(720, 1280, 3)
    und = np.empty_like(bgr)
    yield "undistort 1280x720 BGR", lambda: vc.undistort(bgr, K, d, dst=und), lambda: cv2.undistort(bgr, K, d, dst=und)
    gray = natural(720, 1280, 1)
    und1 = np.empty_like(gray)
    yield "undistort 1280x720 gray", lambda: vc.undistort(gray, K, d, dst=und1), lambda: cv2.undistort(gray, K, d, dst=und1)
    mx, my = cv2.initUndistortRectifyMap(K, d, None, K, (1280, 720), cv2.CV_32FC1)
    yield "initUndistortRectifyMap 1280x720", lambda: vc.init_undistort_rectify_map(K, d, None, K, (1280, 720)), \
        lambda: cv2.initUndistortRectifyMap(K, d, None, K, (1280, 720), cv2.CV_32FC1)
    yield "remap linear 1280x720 BGR", lambda: vc.remap(bgr, mx, my, dst=und), lambda: cv2.remap(bgr, mx, my, cv2.INTER_LINEAR, dst=und)
    objp = np.zeros((54, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 0.025
    objs, imgs = [], []
    for _ in range(10):
        rv = r.normal(0, 0.3, 3)
        tv = np.array([r.uniform(-0.12, 0.0), r.uniform(-0.08, 0.0), r.uniform(0.35, 0.6)])
        p = cv2.projectPoints(objp, rv, tv, K, d)[0].reshape(-1, 2)
        objs.append(objp)
        imgs.append((p + r.normal(0, 0.2, p.shape)).astype(np.float32))
    yield "calibrateCamera 10 views 9x6", lambda: vc.calibrate_camera(objs, imgs, (1280, 720)), lambda: cv2.calibrateCamera(objs, imgs, (1280, 720), None, None)


def dnn_cases():
    """YuNet face detection, NanoDet and YOLOX networks (each skipped when
    VISAR_MODELS lacks its model)."""
    d = os.environ.get("VISAR_MODELS", "")
    for name, file, size in [("NanoDet", "object_detection_nanodet_2022nov.onnx", 416),
                             ("YOLOX-S", "object_detection_yolox_2022nov.onnx", 640)]:
        p = os.path.join(d, file)
        if not d or not os.path.isfile(p):
            continue
        img = cv2.resize(natural(480, 640, 3), (size, size), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(img.astype(np.float32))
        ours = vs.Net(open(p, "rb").read())
        ours.plan(blob.shape)
        theirs = cv2.dnn.readNet(p)
        names = theirs.getUnconnectedOutLayersNames()

        def run_cv(t=theirs, b=blob, n=names):
            t.setInput(b)
            return t.forward(n)
        yield f"{name} {size} net", (lambda o=ours, b=blob: o.forward(b)), run_cv
    path = next((os.path.join(d, n) for n in ("face_detection_yunet_2023mar.onnx", "y.onnx")
                 if d and os.path.isfile(os.path.join(d, n))), None)
    if path is None:
        return
    model = open(path, "rb").read()
    faces = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".jpg", ".png"))]
    src = cv2.imread(os.path.join(d, faces[0])) if faces else natural(480, 640, 3)
    for (w, h) in [(320, 240), (640, 480)]:
        img = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)
        ours = vs.FaceDetector(model, (w, h))
        theirs = cv2.FaceDetectorYN.create(path, "", (w, h))
        yield f"FaceDetectorYN {w}x{h}", (lambda o=ours, i=img: o.detect(i)), (lambda t=theirs, i=img: t.detect(i))
    # The 4K face pipeline's stages (bench/face4k.py has the full table):
    # a camera-native BGRA frame into the network input, the gray tracking
    # image, and the whole detection (network on the CPU).
    if not faces or not hasattr(vs.lib, "visar_face_detector_prepare"):
        return
    frame = cv2.cvtColor(cv2.resize(src, (3840, 2160), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2BGRA)
    bgr4k = np.empty((2160, 3840, 3), np.uint8)
    small = np.empty((360, 640, 3), np.uint8)
    gray = np.empty((360, 640), np.uint8)
    graybig = np.empty((2160, 3840), np.uint8)
    # One detector per library (bench/ab.py swaps `vs` between two builds).
    dets = {}

    def ours():
        d = dets.get(id(vs))
        if d is None:
            d = dets[id(vs)] = vs.FaceDetector(model, (640, 360), backend="cpu")
        return d
    theirs = cv2.FaceDetectorYN.create(path, "", (640, 360))

    def cv_prep():
        cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR, dst=bgr4k)
        cv2.resize(bgr4k, (640, 360), dst=small, interpolation=cv2.INTER_AREA)

    def cv_gray():
        cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY, dst=graybig)
        cv2.resize(graybig, (640, 360), dst=gray, interpolation=cv2.INTER_AREA)
    yield "face prep 4K BGRA->640x360", (lambda: ours().prepare(frame)), cv_prep
    yield "face gray 4K BGRA /6", (lambda: vs.gray_downsample(frame, 6, gray)), cv_gray
    def ours_prep_gray():
        # One pass where the library has it (bench/ab.py may run an older one).
        if hasattr(vs.lib, "visar_face_detector_prepare_gray"):
            ours().prepare_gray(frame, gray)
        else:
            ours().prepare(frame)
            vs.gray_downsample(frame, 6, gray)
    yield "face prep+gray 4K BGRA", ours_prep_gray, (lambda: (cv_prep(), cv_gray()))
    yield "face pipeline 4K->640x360", (lambda: ours().detect_frame(frame)), (lambda: (cv_prep(), theirs.detect(small)))


def blob_mask(h, w, n=None, seed=7):
    """A binary mask with many filled and ring-shaped blobs."""
    rng = np.random.default_rng(seed)
    m = np.zeros((h, w), np.uint8)
    for _ in range(n or max(1, h * w // 700)):
        x, y, r = int(rng.integers(0, w)), int(rng.integers(0, h)), int(rng.integers(2, max(3, min(h, w) // 40)))
        cv2.circle(m, (x, y), r, 255, -1 if rng.random() < 0.7 else 2)
    return m


def area_cases():
    """INTER_AREA shrinks at fixed sizes: fractional and integer factors,
    1, 3 and 4 channels."""
    for (sw, sh), (dw, dh), cns in [((640, 480), (416, 312), (1, 3, 4)), ((1920, 1080), (1280, 720), (3,)),
                                    ((1920, 1080), (640, 360), (3,)), ((1280, 720), (320, 180), (3,)),
                                    ((97, 61), (53, 37), (3,))]:
        for cn in cns:
            src = natural(sh, sw, cn)
            out = np.empty((dh, dw) if cn == 1 else (dh, dw, cn), np.uint8)
            yield (f"resize area {sw}x{sh}->{dw}x{dh} c{cn}", lambda src=src, out=out, dw=dw, dh=dh: vs.resize(src, (dw, dh), 3, out),
                   lambda src=src, out=out, dw=dw, dh=dh: cv2.resize(src, (dw, dh), dst=out, interpolation=cv2.INTER_AREA))


def contour_cases(h, w):
    mask = blob_mask(h, w)
    yield "findContours TREE simple", lambda: vs.find_contours(mask, 3, 2, raw=True), lambda: cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yield "findContours EXT none", lambda: vs.find_contours(mask, 0, 1, raw=True), lambda: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    yield "moments binary", lambda: vs.moments(mask, True), lambda: cv2.moments(mask, True)
    yield "connectedComponents stats", lambda: vs.connected_components_with_stats(mask, 8), lambda: cv2.connectedComponentsWithStats(mask, connectivity=8)
    big = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
    rng = np.random.default_rng(3)
    cloud = rng.integers(0, max(h, w), (20000, 1, 2)).astype(np.int32)
    yield "contour moments", lambda: vs.moments(big), lambda: cv2.moments(big)
    yield "convexHull 20k pts", lambda: vs.convex_hull(cloud), lambda: cv2.convexHull(cloud)
    yield "approxPolyDP", lambda: vs.approx_poly_dp(big, 2.0, True), lambda: cv2.approxPolyDP(big, 2.0, True)


def moving_scene(h, w, frames=48, objects=8, seed=11):
    """BGR frames of textured objects moving over a static textured
    background (the kind of input `visar track` sees)."""
    rng = np.random.default_rng(seed)
    bg = natural(h, w, 3)
    s = w / 960
    obj = []
    for _ in range(objects):
        ow, oh = int(rng.integers(40, 90) * s), int(rng.integers(30, 60) * s)
        tex = np.clip(rng.normal(60 if rng.random() < 0.5 else 200, 25, (oh, ow, 3)), 0, 255).astype(np.uint8)
        obj.append((tex, rng.uniform(0, w - ow), rng.uniform(0, h - oh), rng.uniform(-6, 6) * s, rng.uniform(-3, 3) * s))
    out = []
    for f in range(frames):
        img = bg.copy()
        for tex, x, y, vx, vy in obj:
            oh, ow = tex.shape[:2]
            # Bounce off the borders, so the sequence loops smoothly back and forth.
            px = int(abs((x + vx * f) % (2 * (w - ow)) - (w - ow))) if w > ow else 0
            py = int(abs((y + vy * f) % (2 * (h - oh)) - (h - oh))) if h > oh else 0
            img[py:py + oh, px:px + ow] = tex
        out.append(img)
    return out


def tracking_cases(h=720, w=1280):
    """The classic tracking stack at 720p: MOSSE, background subtraction and
    the full `visar track` detection pipeline, each against the closest
    OpenCV counterpart. Stateful: every call processes the next frame."""
    if not hasattr(vs.lib, "visar_mosse_create"):
        return
    frames = moving_scene(h, w)
    grays = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in frames]
    n = len(frames)

    def stepper(fn):
        i = [0]

        def step():
            fn(i[0] % n)
            i[0] += 1
        return step

    # MOSSE: one 80x60 textured target moving 4 px/frame (back and forth).
    if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerMOSSE_create"):
        rng = np.random.default_rng(5)
        tex = np.clip(rng.normal(128, 50, (60, 80)), 0, 255).astype(np.uint8)
        tex = cv2.GaussianBlur(tex, (5, 5), 0)
        base = cv2.GaussianBlur(rng.integers(70, 110, (h, w)).astype(np.uint8), (3, 3), 0)
        seq = []
        for f in range(2 * n):
            k = f if f < n else 2 * n - 1 - f
            img = base.copy()
            x, y = 300 + 4 * k, 200 + 2 * k
            img[y:y + 60, x:x + 80] = tex
            seq.append(img)
        box = (300, 200, 80, 60)
        ours, theirs = vs.Mosse(seq[0], box, scale=False), cv2.legacy.TrackerMOSSE_create()
        theirs.init(seq[0], box)
        j = [1, 1]

        def mo():
            ours.update(seq[j[0] % len(seq)])
            j[0] += 1

        def mt():
            theirs.update(seq[j[1] % len(seq)])
            j[1] += 1
        yield "MOSSE update 720p (1 scale)", mo, mt

    # Background subtraction: different algorithms, timed per frame. VISAR's
    # model is a running average with a selective update.
    mask = np.empty((h, w), np.uint8)
    bgv = [vs.Background(w, h, 4)]
    vis_bg = stepper(lambda i: bgv[0].apply(grays[i], mask, 28))
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=False)
    fg = np.empty((h, w), np.uint8)
    yield "motion mask (VISAR running average) vs MOG2 720p", vis_bg, stepper(lambda i: mog.apply(grays[i], fg))
    yield "motion mask (VISAR running average) vs KNN 720p", vis_bg, stepper(lambda i: knn.apply(grays[i], fg))
    acc = grays[0].astype(np.float32)
    bg8 = np.empty((h, w), np.uint8)
    diff = np.empty((h, w), np.uint8)

    def ocv_avg(src, out):
        cv2.convertScaleAbs(acc, dst=bg8)
        cv2.absdiff(src, bg8, dst=diff)
        cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY, dst=out)
        cv2.accumulateWeighted(src, acc, 1 / 16)
    yield "motion mask (VISAR running average) vs accumulateWeighted+absdiff 720p", vis_bg, stepper(lambda i: ocv_avg(grays[i], fg))

    # The full detection pipeline of `visar track` (demos/track_lib.zig
    # Detector) at 720p: gray, Gaussian 5x5, background, open 3x3, elliptical
    # dilate, connected components, boxes, tracker. OpenCV has no
    # multi-object tracker, so both sides feed the same VISAR tracker.
    sx = w / 960
    d = min(31, max(3, round(9 * sx)) | 1)
    min_area = int(250 * sx * sx)
    el3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eld = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))

    def pipeline(bgsub):
        gray = np.empty((h, w), np.uint8)
        blur = np.empty((h, w), np.uint8)
        m1 = np.empty((h, w), np.uint8)
        m2 = np.empty((h, w), np.uint8)
        trk = vs.Tracker(40 * sx)
        cv_acc = [None]
        pmog = cv2.createBackgroundSubtractorMOG2(detectShadows=False) if bgsub == "mog2" else None

        def ours(i):
            vs.cvt_color(frames[i], 6, gray)
            vs.gaussian_blur(gray, (5, 5), 0, dst=blur)
            bgsub.apply(blur, m1, 28)
            vs.morphology(m1, cv2.MORPH_OPEN, cv2.MORPH_RECT, (3, 3), 1, m2)
            vs.morphology(m2, cv2.MORPH_DILATE, cv2.MORPH_ELLIPSE, (d, d), 1, m1)
            _, _, st, _ = vs.connected_components_with_stats(m1, 8)
            trk.update(st[1:][st[1:, 4] >= min_area, :4].astype(np.float32))

        def theirs(i):
            cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY, dst=gray)
            cv2.GaussianBlur(gray, (5, 5), 0, dst=blur)
            if pmog is not None:
                pmog.apply(blur, m1)
            else:
                if cv_acc[0] is None:
                    cv_acc[0] = blur.astype(np.float32)
                cv2.convertScaleAbs(cv_acc[0], dst=bg8)
                cv2.absdiff(blur, bg8, dst=diff)
                cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY, dst=m1)
                cv2.accumulateWeighted(blur, cv_acc[0], 1 / 16)
            cv2.morphologyEx(m1, cv2.MORPH_OPEN, el3, dst=m2)
            cv2.dilate(m2, eld, dst=m1)
            _, _, st, _ = cv2.connectedComponentsWithStats(m1, connectivity=8)
            trk.update(st[1:][st[1:, 4] >= min_area, :4].astype(np.float32))
        return ours, theirs

    po, _ = pipeline(vs.Background(w, h, 4))
    _, pt = pipeline("avg")
    yield "track pipeline 720p vs OpenCV (accumulateWeighted)", stepper(po), stepper(pt)
    po, _ = pipeline(vs.Background(w, h, 4))
    _, pt = pipeline("mog2")
    yield "track pipeline 720p vs OpenCV (MOG2)", stepper(po), stepper(pt)


def video_cases(h, w, n=16):
    """Motion-JPEG AVI: write n frames, and read them all back (open, then
    every frame into a reused BGR buffer), against cv2.VideoWriter /
    cv2.VideoCapture with their default backend (FFmpeg) and OpenCV's own
    AVI/MJPEG backend."""
    if not hasattr(vs.lib, "visar_video_capture_open"):
        return
    import tempfile
    d = tempfile.mkdtemp(prefix="visar-bench-video-")
    frames = [natural(h, w, 3) for _ in range(2)]
    frames = [np.roll(frames[i % 2], 7 * i, axis=1) for i in range(n)]
    path_vs = os.path.join(d, "vs.avi")
    path_cv = os.path.join(d, "cv.avi")
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")
    backends = {b for b in cv2.videoio_registry.getBackends()}
    apis = [("FFmpeg", cv2.CAP_FFMPEG)] if cv2.CAP_FFMPEG in backends else [("default", cv2.CAP_ANY)]
    if cv2.CAP_OPENCV_MJPEG in backends:
        apis.append(("OpenCV MJPEG", cv2.CAP_OPENCV_MJPEG))

    def write_vs():
        wr = vs.VideoWriter(path_vs, vs.fourcc("MJPG"), 30, (w, h), quality=95)
        for f in frames:
            wr.write(f)
        wr.release()

    def writer_cv(api):
        def f():
            # FFmpeg takes no quality parameter (it encodes at its own default).
            params = [] if api == cv2.CAP_FFMPEG else [cv2.VIDEOWRITER_PROP_QUALITY, 95]
            wr = cv2.VideoWriter(path_cv, api, mjpg, 30, (w, h), params)
            assert wr.isOpened()
            for fr in frames:
                wr.write(fr)
            wr.release()
        return f

    buf_vs = np.empty((h, w, 3), np.uint8)
    buf_cv = np.empty((h, w, 3), np.uint8)

    def read_vs():
        cap = vs.VideoCapture(path_vs)
        k = 0
        while cap.read(buf_vs)[0]:
            k += 1
        cap.release()
        assert k == n

    def reader_cv(api):
        def f():
            cap = cv2.VideoCapture(path_vs, api)
            k = 0
            while cap.read(buf_cv)[0]:
                k += 1
            cap.release()
            assert k == n
        return f

    write_vs()
    for label, api in apis:
        yield f"VideoWriter MJPG {n}f vs {label}", write_vs, writer_cv(api)
    for label, api in apis:
        yield f"VideoCapture MJPG {n}f vs {label}", read_vs, reader_cv(api)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--md", help="write a Markdown table here")
    ap.add_argument("--filter", default="")
    ap.add_argument("--warn-below", type=float, default=0.0,
                    help="emit a GitHub Actions warning for cases slower than this speedup")
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
    if args.warn_below > 0:
        for name, v, o in rows:
            if o / v < args.warn_below:
                # Shows on the workflow run page, so a regression is visible
                # without reading the logs.
                print(f"::warning title=Slow vs OpenCV ({threads}T, {vs.lib.visar_build_info().decode()})::"
                      f"{name}: {o / v:.2f}x ({v * 1e3:.3f} ms vs {o * 1e3:.3f} ms)")
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
