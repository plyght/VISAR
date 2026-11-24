"""The Metal dnn backend against the CPU backend and OpenCV, plus timing.

Run on a Mac (CI: the macos-latest job of ci.yml):
    python3 tests/py/dnn_metal_check.py --lib zig-out/lib/libvisar.dylib

Fails (exit 1) when Metal is unavailable, when a raw YuNet head differs
from the CPU backend by more than 1e-4 relative to the head's magnitude,
or when FaceDetector results on Metal differ from the CPU backend or from
cv2.FaceDetectorYN. The timing table is information only (CI runners
expose a paravirtual GPU, not representative of real hardware).
Needs VISAR_MODELS (a directory holding face_detection_yunet_2023mar.onnx,
optionally face images).
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_core_vs_opencv as base  # noqa: E402  (handles --lib)
import test_dnn_vs_opencv as dt  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import visar_ctypes as vs  # noqa: E402

results = []


def record(name, ok, detail):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush=True)


def heads(net, blob):
    return net.forward(blob)


def bench(fn, n):
    fn()  # warm-up (first run pays for pipeline and page set-up)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e3


def main():
    vs.set_num_threads(0)
    if not vs.metal_available():
        print(f"FAIL: Metal backend unavailable: {vs.metal_device()}")
        return 1
    path = dt.find_model()
    if path is None:
        print("FAIL: YuNet model not found (set VISAR_MODELS)")
        return 1
    print(f"VISAR {vs.version()} on Metal device: {vs.metal_device()}")
    model = open(path, "rb").read()
    images = [("natural", base.natural(480, 640, 3))] + dt.face_images()

    for nm, img in images:
        for (w, h) in [(320, 240), (640, 480), (1280, 720), (97, 61)]:
            im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            tag = f"{nm} {w}x{h}"
            blob = cv2.dnn.blobFromImage(dt.pad32(im))
            cpu = vs.Net(model, backend="cpu")
            gpu = vs.Net(model, backend="metal")
            cpu.plan(blob.shape)
            gpu.plan(blob.shape)
            if gpu.backend != "metal" or cpu.backend != "cpu":
                record(f"backend {tag}", False, f"planned on {gpu.backend} / {cpu.backend}")
                continue
            want = heads(cpu, blob)
            got = heads(gpu, blob)
            worst = 0.0
            for name in dt.HEADS:
                a, b = got[name], want[name]
                if a.shape != b.shape or not np.all(np.isfinite(a)):
                    record(f"raw {name} {tag}", False, f"shape {a.shape} vs {b.shape} or non-finite")
                    continue
                err = float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max()))
                worst = max(worst, err)
                if err > 1e-4:
                    record(f"raw {name} {tag}", False, f"max rel {err:.3g} vs CPU")
            record(f"raw heads metal vs cpu {tag}", worst <= 1e-4, f"worst rel {worst:.3g}")
            for score, nms, top_k in [(0.9, 0.3, 5000), (0.5, 0.3, 5000), (0.3, 0.5, 20)]:
                dc = vs.FaceDetector(model, (w, h), score, nms, top_k, backend="cpu")
                dm = vs.FaceDetector(model, (w, h), score, nms, top_k, backend="metal")
                fc, fm = dc.detect(im), dm.detect(im)
                ref = cv2.FaceDetectorYN.create(path, "", (w, h), score, nms, top_k)
                fo = ref.detect(im)[1]
                fo = np.zeros((0, 15), np.float32) if fo is None else fo
                name = f"faces metal {tag} s{score} n{nms} k{top_k}"
                if dm.backend != "metal":
                    record(name, False, f"detector planned on {dm.backend}")
                    continue
                ok = True
                detail = f"{len(fm)} faces"
                for other, label in ((fc, "cpu"), (fo, "opencv")):
                    if other.shape != fm.shape:
                        ok = False
                        detail += f"; {len(other)} on {label}"
                    elif len(fm):
                        geo = float(np.abs(fm[:, :14] - other[:, :14]).max())
                        sc = float(np.abs(fm[:, 14] - other[:, 14]).max())
                        ok = ok and geo <= 2e-3 and sc <= 1e-4
                        detail += f"; vs {label} px {geo:.3g} score {sc:.3g}"
                record(name, ok, detail)

    # The auto backend: which one each size picks (information).
    for (w, h) in [(160, 120), (320, 240), (640, 480)]:
        d = vs.FaceDetector(model, (w, h), backend="auto")
        print(f"auto backend for {w}x{h}: {d.backend}")

    # Timing, information only.
    print("\nYuNet timing on this runner (median ms; paravirtual GPU, not representative):")
    print(f"{'size':>10} {'cpu net':>9} {'metal net':>10} {'cpu detect':>11} {'metal detect':>13} {'opencv detect':>14}")
    img = images[-1][1]
    for (w, h) in [(320, 240), (640, 480), (1280, 720)]:
        im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(dt.pad32(im))
        nets = {}
        for b in ("cpu", "metal"):
            nets[b] = vs.Net(model, backend=b)
            nets[b].plan(blob.shape)
        dets = {b: vs.FaceDetector(model, (w, h), backend=b) for b in ("cpu", "metal")}
        ref = cv2.FaceDetectorYN.create(path, "", (w, h))
        n = 20 if w <= 640 else 10
        row = [bench(lambda b=b: nets[b].forward(blob), n) for b in ("cpu", "metal")]
        row += [bench(lambda b=b: dets[b].detect(im), n) for b in ("cpu", "metal")]
        row.append(bench(lambda: ref.detect(im), n))
        print(f"{w}x{h:>4} " + " ".join(f"{t:>{k}.2f}" for t, k in zip(row, (9, 10, 11, 13, 14))))

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} Metal checks passed")
    if failed:
        print("FAILED:")
        for name, _, detail in failed:
            print(f"  {name}: {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
