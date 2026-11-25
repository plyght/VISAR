"""Differential tests for features2d: FAST, ORB and BFMatcher, VISAR (through
its C ABI) against OpenCV. Run: zig build difftest

Keypoints are compared field by field (FAST in OpenCV's row-major order; ORB
as sets, sorted by level and position, since OpenCV's order within a level
comes from std::nth_element), descriptors byte for byte, and matches by
train index and distance.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Importing the core suite handles `--lib` and provides the helpers.
import test_core_vs_opencv as base  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import features2d_ctypes as fd  # noqa: E402
import visar_ctypes as vs  # noqa: E402

natural = base.natural
run = base.run
results = base.results


def textured(h, w, seed=0):
    """natural() with noise and random boxes: many corners at every scale."""
    rng = np.random.default_rng(seed)
    img = natural(h, w, 1).astype(np.float32) + rng.normal(0, 20, (h, w))
    for _ in range(max(1, h * w // 1500)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(3, 40)), y + int(rng.integers(3, 40))), float(rng.integers(0, 256)), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def images():
    out = {"natural 640x480": natural(480, 640, 1), "textured 640x480": textured(480, 640),
           "textured 777x555": textured(555, 777, 3), "textured 1280x720": textured(720, 1280, 5)}
    d = os.environ.get("VISAR_MODELS", "")
    p = os.path.join(d, "largest_selfie.jpg")
    if d and os.path.isfile(p):
        out["selfie"] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return out


def kp_array(kps):
    return np.array([(k.pt[0], k.pt[1], k.size, k.angle, k.response, k.octave) for k in kps], np.float64).reshape(-1, 6)


def ours_array(k):
    return np.stack([k["x"], k["y"], k["size"], k["angle"], k["response"], k["octave"]], 1).astype(np.float64).reshape(-1, 6)


def fast_cases(imgs):
    small = {f"{w}x{h}": textured(h, w, w * h) for (h, w) in [(1, 1), (6, 9), (7, 7), (8, 40), (37, 53), (20, 97)]}
    for name, g in list(imgs.items())[:3] + list(small.items()):
        for t in (0, 10, 20, 60):
            for nms in (True, False):
                for ty in (0, 1, 2):
                    det = cv2.FastFeatureDetector_create(t, nms, ty)
                    run(f"FAST type{ty} t{t} nms{int(nms)} {name}",
                        lambda: (ours_array(fd.fast(g, t, nms, ty))[:, [0, 1, 4]], kp_array(det.detect(g))[:, [0, 1, 4]]))
    # A strided view (rows padded).
    g = imgs["textured 640x480"][:, 5:600]
    run("FAST strided view", lambda: (ours_array(fd.fast(g, 20))[:, [0, 1, 4]], kp_array(cv2.FastFeatureDetector_create(20).detect(np.ascontiguousarray(g)))[:, [0, 1, 4]]))


def orb_sorted(kp, d):
    order = np.lexsort((kp[:, 0], kp[:, 1], kp[:, 5]))
    return kp[order], (d[order] if d is not None and len(d) else np.zeros((0, 32), np.uint8))


def orb_case(name, img, mask=None, **kw):
    k1, d1 = cv2.ORB_create(**kw).detectAndCompute(img, mask)
    k2, d2 = fd.orb(img, mask, **kw)
    ref_k, ref_d = orb_sorted(kp_array(k1), d1)
    our_k, our_d = orb_sorted(ours_array(k2), d2)
    run(f"ORB keypoints {name}", lambda: (our_k, ref_k))
    run(f"ORB descriptors {name}", lambda: (our_d, ref_d))
    return k1, d1, k2, d2


def orb_cases(imgs):
    params = [{}, {"nfeatures": 1000}, {"nfeatures": 2000, "fastThreshold": 10}, {"scoreType": 1},
              {"WTA_K": 3}, {"WTA_K": 4}, {"patchSize": 21, "edgeThreshold": 21}, {"edgeThreshold": 10},
              {"firstLevel": 2, "nlevels": 5}, {"scaleFactor": 1.5, "nlevels": 4}, {"nlevels": 1},
              {"nfeatures": 0}]
    for name, g in imgs.items():
        for kw in params:
            tag = " ".join(f"{k}={v}" for k, v in kw.items()) or "defaults"
            orb_case(f"{tag} {name}", g, **kw)
    g = imgs["textured 640x480"]
    mask = np.zeros_like(g)
    cv2.circle(mask, (320, 240), 180, 255, -1)
    mask[:50] = 1  # nonzero values below 255 count as set
    orb_case("mask", g, mask)
    orb_case("mask firstLevel=1", g, mask, firstLevel=1)
    bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    bgr[..., 0] = np.roll(g, 7, 1)
    orb_case("BGR input", bgr)
    for (h, w) in [(3, 3), (31, 31), (64, 80), (100, 333)]:
        orb_case(f"small {w}x{h}", textured(h, w, w + h))


def matcher_cases(imgs):
    g1 = imgs["textured 640x480"]
    g2 = cv2.warpAffine(g1, cv2.getRotationMatrix2D((320, 240), 12, 0.9), (640, 480))
    _, d1 = cv2.ORB_create(1000).detectAndCompute(g1, None)
    _, d2 = cv2.ORB_create(1000).detectAndCompute(g2, None)
    rng = np.random.default_rng(9)
    sets = {"ORB": (d1, d2), "random 64B": (rng.integers(0, 256, (300, 64), dtype=np.uint8), rng.integers(0, 256, (500, 64), dtype=np.uint8)),
            "random 13B": (rng.integers(0, 256, (50, 13), dtype=np.uint8), rng.integers(0, 256, (70, 13), dtype=np.uint8)),
            "few train": (d1[:20], d2[:3])}

    def mt(ms):
        return np.array([(m.queryIdx, m.trainIdx, m.distance) for m in ms], np.float64).reshape(-1, 3)

    def ours(m):
        return np.stack([m["queryIdx"], m["trainIdx"], m["distance"]], 1).astype(np.float64).reshape(-1, 3)

    for name, (q, t) in sets.items():
        for norm, nn in ((cv2.NORM_HAMMING, "HAMMING"), (cv2.NORM_HAMMING2, "HAMMING2"), (cv2.NORM_L1, "L1"), (cv2.NORM_L2, "L2")):
            if norm != cv2.NORM_HAMMING and name != "ORB":
                continue
            tol = 1e-3 if norm == cv2.NORM_L2 else 0
            run(f"BF match {nn} {name}", lambda: (ours(fd.bf_match(q, t, norm)), mt(cv2.BFMatcher(norm).match(q, t))), tol)
            run(f"BF crossCheck {nn} {name}", lambda: (ours(fd.bf_match(q, t, norm, True)), mt(cv2.BFMatcher(norm, crossCheck=True).match(q, t))), tol)
            for k in (1, 2, 5):
                ref = cv2.BFMatcher(norm).knnMatch(q, t, k)
                out, counts = fd.bf_knn_match(q, t, k, norm)
                got = np.concatenate([ours(out[i][:counts[i]]) for i in range(len(q))])
                run(f"BF knnMatch k={k} {nn} {name}", lambda: (got, np.concatenate([mt(r) for r in ref])), tol)
                if k == 2 and norm == cv2.NORM_HAMMING:
                    good = [m[0] for m in ref if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
                    run(f"ratio test {name}", lambda: (ours(fd.ratio_test(out, counts, 0.75)), mt(good)))
    # Float descriptors: distances agree to rounding; indices must too
    # wherever the two best candidates are not within rounding of each other.
    qf = rng.random((200, 64), dtype=np.float32)
    tf = rng.random((300, 64), dtype=np.float32)
    for norm, nn in ((cv2.NORM_L2, "L2"), (cv2.NORM_L1, "L1"), (cv2.NORM_L2SQR, "L2SQR")):
        run(f"BF match f32 {nn}", lambda: (ours(fd.bf_match(qf, tf, norm)), mt(cv2.BFMatcher(norm).match(qf, tf))), 1e-3)
        ref = cv2.BFMatcher(norm).knnMatch(qf, tf, 3)
        out, counts = fd.bf_knn_match(qf, tf, 3, norm)
        run(f"BF knnMatch k=3 f32 {nn}", lambda: (ours(out.reshape(-1)), np.concatenate([mt(r) for r in ref])), 1e-3)
        run(f"BF crossCheck f32 {nn}", lambda: (ours(fd.bf_match(qf, tf, norm, True)), mt(cv2.BFMatcher(norm, crossCheck=True).match(qf, tf))), 1e-3)


def main():
    vs.set_num_threads(0)
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    imgs = images()
    fast_cases(imgs)
    orb_cases(imgs)
    matcher_cases(imgs)
    # Threads do not change results.
    g = imgs["textured 1280x720"]
    a = fd.orb(g, nfeatures=1500)
    vs.set_num_threads(1)
    b = fd.orb(g, nfeatures=1500)
    vs.set_num_threads(0)
    run("ORB 1 thread == N threads", lambda: (ours_array(a[0]), ours_array(b[0])))
    run("ORB descriptors 1 thread == N threads", lambda: (a[1], b[1]))
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
