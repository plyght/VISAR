"""Differential tests for visar.dnn: the YuNet face detector against
cv2.dnn and cv2.FaceDetectorYN, and the NanoDet and YOLOX object detectors
against cv2.dnn and the OpenCV Model Zoo's Python pipelines run with cv2.
Run: zig build difftest

Models come from the OpenCV Model Zoo and are not in the repository: set
VISAR_MODELS to a directory holding face_detection_yunet_2023mar.onnx,
object_detection_nanodet_2022nov.onnx and object_detection_yolox_2022nov.onnx
(and optionally photos, *.jpg / *.png). Missing models are skipped; with
none the suite reports a skip and passes.
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
NANODET_NAMES = ["object_detection_nanodet_2022nov.onnx"]
YOLOX_NAMES = ["object_detection_yolox_2022nov.onnx"]
HEADS = ["cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32",
         "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"]
results = []


def find_model(names=MODEL_NAMES):
    d = os.environ.get("VISAR_MODELS", "")
    for n in names:
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


def yunet_cases(path):
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


# ----------------------------------------------------------- object detection
#
# The reference pipelines below are the OpenCV Model Zoo's
# models/object_detection_nanodet/{nanodet,demo}.py and
# models/object_detection_yolox/{yolox,demo}.py, transcribed (preprocessing,
# letterbox, decoding, NMS, unletterbox) with two changes: NanoDet's outputs
# are fetched in level order (cls8, box8, cls16, ...) instead of the order of
# getUnconnectedOutLayersNames(), which differs in OpenCV 5 and breaks the
# zoo code there; and the final unletterbox keeps fractional coordinates
# (the demos truncate to int for drawing), as boxes x, y, w, h.


def nanodet_letterbox(srcimg, target_size=(416, 416)):
    img = srcimg.copy()
    top, left, newh, neww = 0, 0, target_size[0], target_size[1]
    if img.shape[0] != img.shape[1]:
        hw_scale = img.shape[0] / img.shape[1]
        if hw_scale > 1:
            newh, neww = target_size[0], int(target_size[1] / hw_scale)
            img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
            left = int((target_size[1] - neww) * 0.5)
            img = cv2.copyMakeBorder(img, 0, 0, left, target_size[1] - neww - left, cv2.BORDER_CONSTANT, value=0)
        else:
            newh, neww = int(target_size[0] * hw_scale), target_size[1]
            img = cv2.resize(img, (neww, newh), interpolation=cv2.INTER_AREA)
            top = int((target_size[0] - newh) * 0.5)
            img = cv2.copyMakeBorder(img, top, target_size[0] - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    else:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img, [top, left, newh, neww]


def nanodet_unletterbox(bbox, original_image_shape, letterbox_scale):
    ret = bbox.copy()
    h, w = original_image_shape
    top, left, newh, neww = letterbox_scale
    if h == w:
        return ret * (h / newh)
    ratioh, ratiow = h / newh, w / neww
    ret[0] = max((ret[0] - left) * ratiow, 0)
    ret[1] = max((ret[1] - top) * ratioh, 0)
    ret[2] = min((ret[2] - left) * ratiow, w)
    ret[3] = min((ret[3] - top) * ratioh, h)
    return ret


def nanodet_level_names(net):
    """Output names paired per level, finest first: cls, box, cls, box, ..."""
    names = net.getUnconnectedOutLayersNames()
    shapes = dict(zip(names, net.forward(names)))
    cls = sorted([n for n in names if shapes[n].shape[-1] != 32], key=lambda n: -shapes[n].shape[1])
    box = sorted([n for n in names if shapes[n].shape[-1] == 32], key=lambda n: -shapes[n].shape[1])
    return [n for pair in zip(cls, box) for n in pair]


class NanoDetRef:
    def __init__(self, path, prob_threshold=0.35, iou_threshold=0.6):
        self.strides = (8, 16, 32, 64)
        self.image_shape = (416, 416)
        self.reg_max = 7
        self.prob_threshold = prob_threshold
        self.iou_threshold = iou_threshold
        self.project = np.arange(self.reg_max + 1)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.net = cv2.dnn.readNet(path)
        self.net.setInput(np.zeros((1, 3, 416, 416), np.float32))
        self.names = nanodet_level_names(self.net)
        self.anchors_mlvl = []
        for i in range(len(self.strides)):
            featmap_size = (int(self.image_shape[0] / self.strides[i]), int(self.image_shape[1] / self.strides[i]))
            stride = self.strides[i]
            feat_h, feat_w = featmap_size
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            xv, yv = np.meshgrid(shift_x, shift_y)
            xv = xv.flatten()
            yv = yv.flatten()
            cx = xv + 0.5 * (stride - 1)
            cy = yv + 0.5 * (stride - 1)
            self.anchors_mlvl.append(np.column_stack((cx, cy)))

    def pre_process(self, img):
        img = img.astype(np.float32)
        img = (img - self.mean) / self.std
        return cv2.dnn.blobFromImage(img)

    def infer(self, srcimg):
        blob = self.pre_process(srcimg)
        self.net.setInput(blob)
        outs = self.net.forward(self.names)
        return self.post_process(outs), blob

    def post_process(self, preds):
        cls_scores, bbox_preds = preds[::2], preds[1::2]
        bboxes_mlvl = []
        scores_mlvl = []
        for stride, cls_score, bbox_pred, anchors in zip(self.strides, cls_scores, bbox_preds, self.anchors_mlvl):
            if cls_score.ndim == 3:
                cls_score = cls_score.squeeze(axis=0)
            if bbox_pred.ndim == 3:
                bbox_pred = bbox_pred.squeeze(axis=0)
            x_exp = np.exp(bbox_pred.reshape(-1, self.reg_max + 1))
            x_sum = np.sum(x_exp, axis=1, keepdims=True)
            bbox_pred = x_exp / x_sum
            bbox_pred = np.dot(bbox_pred, self.project).reshape(-1, 4)
            bbox_pred *= stride
            nms_pre = 1000
            if nms_pre > 0 and cls_score.shape[0] > nms_pre:
                max_scores = cls_score.max(axis=1)
                topk_inds = max_scores.argsort()[::-1][0:nms_pre]
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cls_score = cls_score[topk_inds, :]
            points = anchors
            distance = bbox_pred
            max_shape = self.image_shape
            x1 = np.clip(points[:, 0] - distance[:, 0], 0, max_shape[1])
            y1 = np.clip(points[:, 1] - distance[:, 1], 0, max_shape[0])
            x2 = np.clip(points[:, 0] + distance[:, 2], 0, max_shape[1])
            y2 = np.clip(points[:, 1] + distance[:, 3], 0, max_shape[0])
            bboxes_mlvl.append(np.column_stack([x1, y1, x2, y2]))
            scores_mlvl.append(cls_score)
        bboxes_mlvl = np.concatenate(bboxes_mlvl, axis=0)
        scores_mlvl = np.concatenate(scores_mlvl, axis=0)
        bboxes_wh = bboxes_mlvl.copy()
        bboxes_wh[:, 2:4] = bboxes_wh[:, 2:4] - bboxes_wh[:, 0:2]
        classIds = np.argmax(scores_mlvl, axis=1)
        confidences = np.max(scores_mlvl, axis=1)
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.prob_threshold, self.iou_threshold)
        if len(indices) > 0:
            det_bboxes = bboxes_mlvl[indices]
            det_conf = confidences[indices]
            det_classid = classIds[indices]
            return np.concatenate([det_bboxes, det_conf.reshape(-1, 1), det_classid.reshape(-1, 1)], axis=1)
        return np.array([])

    def detect(self, image):
        """Demo pipeline: BGR image -> boxes (x, y, w, h), scores, class ids."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp, lb = nanodet_letterbox(rgb)
        preds, blob = self.infer(inp)
        boxes, scores, ids = [], [], []
        for pred in preds:
            b = nanodet_unletterbox(pred[:4], image.shape[:2], lb)
            boxes.append([b[0], b[1], b[2] - b[0], b[3] - b[1]])
            scores.append(pred[-2])
            ids.append(int(pred[-1]))
        return np.array(boxes, np.float64).reshape(-1, 4), np.array(scores, np.float64), np.array(ids, np.int32), blob


def yolox_letterbox(srcimg, target_size=(640, 640)):
    padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
    ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
    resized_img = cv2.resize(
        srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv2.INTER_LINEAR
    ).astype(np.float32)
    padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img
    return padded_img, ratio


class YoloXRef:
    def __init__(self, path, confThreshold=0.5, nmsThreshold=0.5):
        self.num_classes = 80
        self.net = cv2.dnn.readNet(path)
        self.input_size = (640, 640)
        self.strides = [8, 16, 32]
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.generateAnchors()

    def preprocess(self, img):
        blob = np.transpose(img, (2, 0, 1))
        return blob[np.newaxis, :, :, :]

    def infer(self, srcimg):
        input_blob = self.preprocess(srcimg)
        self.net.setInput(input_blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self.postprocess(outs[0]), input_blob

    def postprocess(self, outputs):
        dets = outputs[0]
        dets[:, :2] = (dets[:, :2] + self.grids) * self.expanded_strides
        dets[:, 2:4] = np.exp(dets[:, 2:4]) * self.expanded_strides
        boxes = dets[:, :4]
        boxes_xywh = np.ones_like(boxes)
        boxes_xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xywh[:, 2] = boxes[:, 2]
        boxes_xywh[:, 3] = boxes[:, 3]
        scores = dets[:, 4:5] * dets[:, 5:]
        max_scores = np.amax(scores, axis=1)
        max_scores_idx = np.argmax(scores, axis=1)
        keep = cv2.dnn.NMSBoxesBatched(boxes_xywh.tolist(), max_scores.tolist(), max_scores_idx.tolist(), self.confThreshold, self.nmsThreshold)
        candidates = np.concatenate([boxes_xywh, max_scores[:, None], max_scores_idx[:, None]], axis=1)
        if len(keep) == 0:
            return np.array([])
        return candidates[keep]

    def generateAnchors(self):
        self.grids = []
        self.expanded_strides = []
        hsizes = [self.input_size[0] // stride for stride in self.strides]
        wsizes = [self.input_size[1] // stride for stride in self.strides]
        for hsize, wsize, stride in zip(hsizes, wsizes, self.strides):
            xv, yv = np.meshgrid(np.arange(hsize), np.arange(wsize))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            self.grids.append(grid)
            shape = grid.shape[:2]
            self.expanded_strides.append(np.full((*shape, 1), stride))
        self.grids = np.concatenate(self.grids, 1)
        self.expanded_strides = np.concatenate(self.expanded_strides, 1)

    def detect(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inp, ratio = yolox_letterbox(rgb)
        preds, blob = self.infer(inp)
        if len(preds) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, np.int32), blob
        boxes = preds[:, :4] / ratio
        return boxes.astype(np.float64), preds[:, 4].astype(np.float64), preds[:, 5].astype(np.int32), blob


def object_images():
    imgs = [("natural", base.natural(480, 640, 3))] + face_images()
    # A scene with recognisable objects (the synthetic image has few).
    scene = base.natural(480, 640, 3)
    cv2.rectangle(scene, (100, 300), (260, 380), (40, 40, 200), -1)
    imgs.append(("scene", scene))
    return imgs


def detector_cases(arch, path):
    model = open(path, "rb").read()
    det = vs.ObjectDetector(model)
    record(f"{arch} arch detected", det.arch == arch, det.arch)
    defaults = (0.35, 0.6) if arch == "nanodet" else (0.5, 0.5)
    sizes = [None, (640, 480), (480, 640), (416, 416), (640, 640), (300, 200), (1280, 720), (97, 61)]
    ref_cache = {}
    total = 0
    for nm, img0 in object_images():
        for size in sizes:
            img = img0 if size is None else cv2.resize(img0, size, interpolation=cv2.INTER_AREA)
            h, w = img.shape[:2]
            tag = f"{arch} {nm} {w}x{h}"
            for score, nms in [defaults, (0.1, 0.5), (0.25, 0.3)]:
                key = (score, nms)
                if key not in ref_cache:
                    ref_cache[key] = NanoDetRef(path, score, nms) if arch == "nanodet" else YoloXRef(path, score, nms)
                ref = ref_cache[key]
                wb, ws, wi, blob = ref.detect(img)
                if score == defaults[0] and size in (None, (640, 480)):
                    # Raw network outputs on the zoo's blob.
                    net = vs.Net(model)
                    net.plan(blob.shape)
                    got = net.forward(blob)
                    names = ref.names if arch == "nanodet" else ["output"]
                    ref.net.setInput(blob)
                    want = dict(zip(names, ref.net.forward(names)))
                    errs = []
                    for k in names:
                        a, b = got[k], want[k].reshape(got[k].shape)
                        errs.append(float(np.abs(a - b).max()) / max(1.0, float(np.abs(b).max())))
                    record(f"raw {tag}", max(errs) <= 1e-4, f"max rel {max(errs):.3g}")
                det.set_thresholds(score, nms)
                gb, gs, gi = det.detect(img)
                total += len(gb)
                name = f"{tag} s{score} n{nms}"
                if len(gb) != len(wb) or not np.array_equal(gi, wi):
                    record(name, False, f"{len(gb)} dets vs {len(wb)}; classes {gi[:8]} vs {wi[:8]}")
                    continue
                if len(gb) == 0:
                    record(name, True, "0 detections")
                    continue
                # Boxes are network outputs scaled by the stride (up to 32)
                # and the letterbox ratio: a 1e-5 relative difference in a
                # head is a few thousandths of a pixel.
                px = float(np.abs(gb - wb).max())
                sc = float(np.abs(gs - ws).max())
                record(name, px <= 1e-2 and sc <= 1e-4, f"{len(gb)} dets, max px {px:.3g}, max score {sc:.3g}")
    record(f"{arch} found objects", total > 0, f"{total} detections over all cases")


def main():
    vs.set_num_threads(0)
    paths = {"yunet": find_model(MODEL_NAMES), "nanodet": find_model(NANODET_NAMES), "yolox": find_model(YOLOX_NAMES)}
    if not any(paths.values()):
        print("SKIP: no model found (set VISAR_MODELS to a directory holding the OpenCV Model Zoo files)")
        return 0
    print(f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}")
    for k, p in paths.items():
        print(f"  {k}: {p or 'not found, skipped'}")
    if paths["yunet"]:
        yunet_cases(paths["yunet"])
    for arch in ("nanodet", "yolox"):
        if paths[arch]:
            detector_cases(arch, paths[arch])

    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[1]]
    for name, ok, detail in results:
        if not ok or os.environ.get("VERBOSE"):
            print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print(f"\n{len(results) - len(failed)}/{len(results)} cases match OpenCV within tolerance")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
