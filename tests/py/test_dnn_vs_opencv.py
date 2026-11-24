"""Differential tests for visar.dnn: the YuNet face detector against
cv2.dnn and cv2.FaceDetectorYN. Run: zig build difftest

Models come from the OpenCV Model Zoo and are not in the repository: set
VISAR_MODELS to a directory holding face_detection_yunet_2023mar.onnx
(and optionally face images, *.jpg / *.png). Without the model the suite
reports a skip and passes.
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_core_vs_opencv as base  # noqa: E402  (handles --lib)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import visar_ctypes as vs  # noqa: E402

MODEL_NAMES = ["face_detection_yunet_2023mar.onnx", "y.onnx"]
HEADS = ["cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
         "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"]
results = []


def find_model():
    d = os.environ.get("VISAR_MODELS", "")
    for n in MODEL_NAMES:
        p = os.path.join(d, n)
        if d and os.path.isfile(p):
            return p
    return None


def face_images():
    d = os.environ.get("VISAR_MODELS", "")
    files = sorted(glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))) if d else []
    out = []
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if img is not None:
            out.append((os.path.basename(f), img))
    return out


def record(name, ok, detail):
    results.append((name, ok, detail))


def pad32(img):
    h, w = img.shape[:2]
    ph, pw = ((h - 1) // 32 + 1) * 32, ((w - 1) // 32 + 1) * 32
    return cv2.copyMakeBorder(img, 0, ph - h, 0, pw - w, cv2.BORDER_CONSTANT, value=0)


def raw_outputs(model_path, model_bytes, img, tag):
    """Network outputs on the same blob: relative to each head's magnitude."""
    blob = cv2.dnn.blobFromImage(pad32(img))
    ref = cv2.dnn.readNet(model_path)
    ref.setInput(blob)
    want = dict(zip(HEADS, ref.forward(HEADS)))
    net = vs.Net(model_bytes)
    net.plan(blob.shape)
    got = net.forward(blob)
    for name in HEADS:
        a, b = got[name], want[name].reshape(got[name].shape) if want[name].size == got[name].size else want[name]
        if a.shape != b.shape:
            record(f"raw {name} {tag}", False, f"shape {a.shape} vs {b.shape}")
            continue
        scale = max(1.0, float(np.abs(b).max()))
        err = float(np.abs(a - b).max()) / scale
        record(f"raw {name} {tag}", err <= 1e-4, f"max rel {err:.3g}")


def detections(model_path, model_bytes, img, tag, score=0.9, nms=0.3, top_k=5000):
    h, w = img.shape[:2]
    ref = cv2.FaceDetectorYN.create(model_path, "", (w, h), score, nms, top_k)
    want = ref.detect(img)[1]
    want = np.zeros((0, 15), np.float32) if want is None else want
    det = vs.FaceDetector(model_bytes, (w, h), score, nms, top_k)
    got = det.detect(img)
    name = f"faces {tag} s{score} n{nms} k{top_k}"
    if got.shape != want.shape:
        record(name, False, f"{len(got)} faces vs {len(want)}")
        return
    if len(got) == 0:
        record(name, True, "0 faces")
        return
    geo = float(np.abs(got[:, :14] - want[:, :14]).max())
    sc = float(np.abs(got[:, 14] - want[:, 14]).max())
    # Box and landmark coordinates are network outputs times the stride
    # (up to 32): a 1e-4 relative difference in the head is ~1e-3 px.
    record(name, geo <= 2e-3 and sc <= 1e-4, f"{len(got)} faces, max px {geo:.3g}, max score {sc:.3g}")


def main():
    vs.set_num_threads(0)
    path = find_model()
    if path is None:
        print("SKIP: YuNet model not found (set VISAR_MODELS to a directory holding face_detection_yunet_2023mar.onnx)")
        return 0
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}, model {path}")
    model = open(path, "rb").read()
    images = [("natural", base.natural(480, 640, 3))] + face_images()
    for nm, img in images:
        for (w, h) in [(320, 240), (640, 480), (300, 200), (1280, 720), (97, 61)]:
            im = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            tag = f"{nm} {w}x{h}"
            raw_outputs(path, model, im, tag)
            detections(path, model, im, tag)
            # Low thresholds: many candidates, so NMS and top_k decide.
            detections(path, model, im, tag, score=0.5, nms=0.3)
            detections(path, model, im, tag, score=0.3, nms=0.5, top_k=20)
    # setInputSize on a live detector.
    det = vs.FaceDetector(model, (320, 240))
    img = cv2.resize(images[-1][1], (640, 480))
    det.set_input_size((640, 480))
    ref = cv2.FaceDetectorYN.create(path, "", (640, 480))
    want = ref.detect(img)[1]
    got = det.detect(img)
    nw = 0 if want is None else len(want)
    record("setInputSize 320x240 -> 640x480", len(got) == nw and (nw == 0 or np.abs(got - want).max() < 2e-3), f"{len(got)} vs {nw}")

    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        if not ok or os.environ.get("VERBOSE"):
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} cases match OpenCV within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
