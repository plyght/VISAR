"""The 4K face pipeline, VISAR against OpenCV, stage by stage.

    zig build face4k -Doptimize=ReleaseFast [-- --threads N] [--sizes 640x360,960x540] [--md FILE]
    python3 bench/face4k.py --lib LIB [--steady EXE] ...

Input: a camera-native 3840x2160 BGRA frame, the OpenCV Model Zoo selfie
(VISAR_MODELS/largest_selfie.jpg, 2048x1150, many faces) enlarged to 4K.
Steps: colour + downscale to the detector size, YuNet, boxes mapped back to
4K. VISAR does the first step in one fused pass straight into the network's
planar f32 input (FaceDetector.prepare); OpenCV runs cv2.cvtColor(BGRA2BGR) +
cv2.resize(INTER_AREA) + cv2.FaceDetectorYN.detect (which pads and builds the
blob itself) + the scaling. Every stage is timed alone, at 1 and N threads
(both libraries alike), as the median of repeated runs (minimum in brackets).

With --steady, the change-driven steady state of the face tracker
(bench/face4k_steady.zig) on the selfie moving slowly over a 4K canvas is
added: effective ms per frame of the whole per-frame work, and its
per-frame latency (p50/p95/p99/max over the frames, nearest rank; frames
with a whole-frame detection are the spikes). For comparison, the end to end
detection of every frame is also timed frame by frame (--frames runs, VISAR
and OpenCV alike), as a latency distribution.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "py"))
import visar_ctypes as vs  # noqa: E402

W4K, H4K = 3840, 2160


def timeit(f, min_time=0.5, min_runs=15, max_runs=400):
    f()
    ts = []
    start = time.perf_counter()
    while len(ts) < min_runs or (time.perf_counter() - start < min_time and len(ts) < max_runs):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3, min(ts) * 1e3


def rank(sorted_ts, p):
    """Nearest-rank percentile of sorted times."""
    return sorted_ts[max(1, -(-len(sorted_ts) * p // 100)) - 1]


def dist(ts):
    """{mean, p50, p95, p99, max} in ms of per-frame times in seconds."""
    ts = sorted(t * 1e3 for t in ts)
    return {"mean": sum(ts) / len(ts), "p50": rank(ts, 50), "p95": rank(ts, 95), "p99": rank(ts, 99), "max": ts[-1]}


def per_frame(f, frames):
    """Per-call times of `frames` calls of f (after one warm-up call)."""
    f()
    ts = []
    for _ in range(frames):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return dist(ts)


def fmt(t):
    return f"{t[0]:.2f} ({t[1]:.2f})" if t else "-"


def one_shot(model_path, model, frame, w, h, threads, frames):
    """Per-stage times {stage: (visar, opencv)} at `threads`, and the faces."""
    vs.set_num_threads(threads)
    cv2.setNumThreads(threads)
    sx, sy = W4K / w, H4K / h
    det = vs.FaceDetector(model, (w, h), backend="cpu")
    ref = cv2.FaceDetectorYN.create(model_path, "", (w, h))
    bgr4k = np.empty((H4K, W4K, 3), np.uint8)
    small = np.empty((h, w, 3), np.uint8)
    small4 = np.empty((h, w, 4), np.uint8)
    gray4k = np.empty((H4K, W4K), np.uint8)
    gray = np.empty((h, w), np.uint8)

    def cv_prep():
        cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR, dst=bgr4k)
        cv2.resize(bgr4k, (w, h), dst=small, interpolation=cv2.INTER_AREA)

    def cv_prep_gray():
        cv_prep()
        cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY, dst=gray4k)
        cv2.resize(gray4k, (w, h), dst=gray, interpolation=cv2.INTER_AREA)

    def cv_detect():
        f = ref.detect(small)[1]
        if f is not None:
            f[:, 0:14:2] *= sx
            f[:, 1:14:2] *= sy
        return f

    def cv_all():
        cv_prep()
        return cv_detect()

    def vs_unfused():
        vs.resize(frame, (w, h), 3, small4)
        vs.cvt_color(small4, 1, small)
        f = det.detect(small)
        f[:, 0:14:2] *= sx
        f[:, 1:14:2] *= sy
        return f

    def vs_detect():
        det.infer()
        return det.decode()

    r = {}
    det.prepare(frame)
    r["prep"] = (timeit(lambda: det.prepare(frame)), None)
    r["prep_gray"] = (timeit(lambda: det.prepare_gray(frame, gray)), timeit(cv_prep_gray))
    r["net"] = (timeit(det.infer), None)
    r["decode"] = (timeit(det.decode), None)
    r["detect"] = (timeit(vs_detect), None)
    r["all"] = (timeit(lambda: det.detect_frame(frame)), None)
    r["unfused"] = (timeit(vs_unfused), None)
    got = det.detect_frame(frame)
    cv_prep()
    r["prep"] = (r["prep"][0], timeit(cv_prep))
    r["detect"] = (r["detect"][0], timeit(cv_detect))
    r["all"] = (r["all"][0], timeit(cv_all))
    want = cv_all()
    want = np.zeros((0, 15), np.float32) if want is None else want
    r["latency"] = (per_frame(lambda: det.detect_frame(frame), frames), per_frame(cv_all, frames))
    if vs.metal_available():
        gpu = vs.FaceDetector(model, (w, h), backend="metal")
        gpu.detect_frame(frame)
        r["metal_net"] = (timeit(gpu.infer), None)
        r["metal_all"] = (timeit(lambda: gpu.detect_frame(frame)), None)
    return r, got, want


def steady(exe, threads, level, every, face_width, frames, extra=()):
    out = subprocess.run([exe, "--backend", "cpu", "--threads", str(threads), "--level", str(level), "--detect-every", str(every),
                          "--face-width", str(face_width), "--frames", str(frames), *extra],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def tail(r):
    return f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f} / {r['p99_ms']:.2f} / {r['max_ms']:.2f}"


def full(r):
    f = r["by_step"]["full"]
    return f"{f['mean_ms']:.2f} / {f['max_ms']:.2f}" if f["n"] else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="the N of the N-thread columns (0 = all cores)")
    ap.add_argument("--sizes", default="640x360,960x540", help="detector input sizes")
    ap.add_argument("--steady", help="visar-face4k-steady executable (adds the tracker's steady state)")
    ap.add_argument("--modes", default="0:1,2:1,2:4,2:4:whole",
                    help="steady state LEVEL:DETECT_EVERY pairs (:whole = the whole gray tracking image every frame, change check on every 4th row: the tracker before partial_gray)")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--md", help="write the tables here (Markdown)")
    args = ap.parse_args()
    d = os.environ.get("VISAR_MODELS", "")
    model_path = os.path.join(d, "face_detection_yunet_2023mar.onnx")
    img_path = os.path.join(d, "largest_selfie.jpg")
    if not d or not os.path.isfile(model_path) or not os.path.isfile(img_path):
        print("SKIP: set VISAR_MODELS to a directory with face_detection_yunet_2023mar.onnx and largest_selfie.jpg")
        return 0
    model = open(model_path, "rb").read()
    src = cv2.imread(img_path)
    frame = cv2.cvtColor(cv2.resize(src, (W4K, H4K), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2BGRA)
    n = args.threads or os.cpu_count()
    info = f"VISAR {vs.version()} ({vs.lib.visar_build_info().decode()}) vs OpenCV {cv2.__version__}, 1 and {n} threads"
    print(info)
    md = [f"{info}. Median ms of repeated runs (minimum in brackets).\n"]
    for size in args.sizes.split(","):
        w, h = map(int, size.split("x"))
        res = {}
        for t in (1, n):
            res[t], got, want = one_shot(model_path, model, frame, w, h, t, args.frames)
        ok = got.shape == want.shape and (len(got) == 0 or np.abs(got[:, :14] - want[:, :14]).max() <= 0.02)
        minw = got[:, 2].min() if len(got) else 0
        gpu_note = f" GPU rows: Metal on {vs.metal_device()} (not representative of real Apple GPUs on a VM)." if vs.metal_available() else ""
        head = (f"### 3840x2160 BGRA -> {w}x{h} YuNet -> 4K boxes\n\n{len(got)} faces (OpenCV {len(want)}, "
                f"{'same boxes' if ok else 'DIFFERENT'}), smallest {minw:.0f} px wide at 4K. Network on the CPU unless noted.{gpu_note}\n")
        rows = [("colour + downscale to the input", "prep", "fused into the f32 input", "cvtColor + resize AREA"),
                ("the same + the gray tracking image", "prep_gray", "one pass for both", "+ cvtColor GRAY + resize AREA"),
                ("network", "net", "", ""),
                ("decode + NMS + map to 4K", "decode", "", ""),
                ("detect (network + decode)", "detect", "", "FaceDetectorYN.detect + map"),
                ("**end to end**", "all", "", ""),
                ("end to end, VISAR as separate ops", "unfused", "resize + cvtColor + detect", "")]
        lines = [head, f"| Stage | VISAR 1T | OpenCV 1T | VISAR {n}T | OpenCV {n}T | VISAR | OpenCV |",
                 "| --- | ---: | ---: | ---: | ---: | --- | --- |"]
        if "metal_net" in res[1]:
            rows += [("network on the GPU (Metal)", "metal_net", "", ""), ("end to end, network on the GPU", "metal_all", "", "")]
        for label, key, vnote, cnote in rows:
            a, b = res[1][key], res[n][key]
            lines.append(f"| {label} | {fmt(a[0])} | {fmt(a[1])} | {fmt(b[0])} | {fmt(b[1])} | {vnote} | {cnote} |")
        for t in (1, n):
            v, c = res[t]["all"]
            lines.append(f"| fps end to end, {t}T | {1000 / v[0]:.0f} | {1000 / c[0]:.0f} | | | | |" if t == 1 else
                         f"| fps end to end, {t}T | | | {1000 / v[0]:.0f} | {1000 / c[0]:.0f} | | |")
        lines += ["", f"Per-frame latency of the end to end detection of every frame ({args.frames} frames one after another, "
                      "ms; nearest-rank percentiles; ratio = OpenCV / VISAR, above 1 = VISAR faster):\n",
                  "| Threads | | mean | p50 | p95 | p99 | max |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for t in (1, n):
            v, c = res[t]["latency"]
            keys = ("mean", "p50", "p95", "p99", "max")
            lines.append(f"| {t}T | VISAR | " + " | ".join(f"{v[k]:.2f}" for k in keys) + " |")
            lines.append(f"| {t}T | OpenCV | " + " | ".join(f"{c[k]:.2f}" for k in keys) + " |")
            lines.append(f"| {t}T | ratio | " + " | ".join(f"{c[k] / v[k]:.2f}x" for k in keys) + " |")
        text = "\n".join(lines) + "\n"
        print(text, flush=True)
        md.append(text)
    if args.steady:
        modes = [(int(m.split(":")[0]), int(m.split(":")[1]), *m.split(":")[2:]) for m in args.modes.split(",")]
        fw = int(args.sizes.split(",")[0].split("x")[0])
        lines = [f"### Face tracking steady state, 3840x2160, detector {fw} wide, selfie moving slowly (bench/face4k_steady.zig)\n",
                 "Per frame: gray tracking image + FaceTracker.step (MOSSE on every face, YuNet when scheduled). "
                 "Level 0 = whole-frame detection on schedule; level 2 = change-driven (postponed while still, "
                 "crops around tracked faces, whole frame every 4th detection). On frames that will probably run a whole-frame detection the gray image and the detector input come from one pass over the frame (FaceTracker.trackingImage); that pass counts as gray. "
                 "On the other frames only the parts of the gray image the tracker reads are computed (FaceOptions.partial_gray: MOSSE search windows, change-check rows, detection patches; bit-identical there), inside the tracking step; "
                 "'whole' rows compute all of it every frame (and check every 4th row for change), as before. Gray read = share of the frame read for the gray image. "
                 "Followed = frames with a track at IoU >= 0.5 to the face moved with the image; IoU and error = mean IoU and centre error (4K px) of the best track. "
                 "p50 / p95 / p99 / max: per-frame latency (each frame timed alone, nearest rank); 'full frames' = the frames with a whole-frame detection (mean / max), the spikes of the change-driven modes.\n",
                 f"| Level, detect every | 1T ms/frame (fps) | {n}T ms/frame (fps) | 1T p50 / p95 / p99 / max ms | {n}T p50 / p95 / p99 / max ms | 1T / {n}T full frames: mean / max ms | {n}T detector runs: full / crop rounds / postponed | 1T ms/frame: gray / tracking / detection | {n}T ms/frame: gray / tracking / detection | gray read | faces | followed | IoU, error |",
                 "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |"]
        for mode in modes:
            level, every = mode[0], mode[1]
            whole = len(mode) > 2 and mode[2] == "whole"
            extra = ("--full-gray", "--still-stride", "4") if whole else ()
            a = steady(args.steady, 1, level, every, fw, args.frames, extra)
            b = steady(args.steady, n, level, every, fw, args.frames, extra)
            lines.append(f"| {level}, {every}{', whole' if whole else ''} | {a['mean_ms']:.2f} ({a['fps']:.0f}) | {b['mean_ms']:.2f} ({b['fps']:.0f}) | "
                         f"{tail(a)} | {tail(b)} | {full(a)}; {full(b)} | {b['full_runs']} / {b['crop_rounds']} / {b['postponed']} of {b['frames']} "
                         f"(full {b['mean_full_ms']:.2f} ms, crops {b['mean_crop_ms']:.2f} ms) | "
                         f"{a['gray_ms']:.2f} / {a['track_ms']:.2f} / {a['detect_ms']:.2f} | {b['gray_ms']:.2f} / {b['track_ms']:.2f} / {b['detect_ms']:.2f} | {b['gray_read'] * 100:.0f}% | {b['faces']} | {b['followed']}/{b['truth_frames']} | {b['mean_iou']:.3f}, {b['mean_err_px']:.1f} |")
            print(lines[-1], flush=True)
        text = "\n".join(lines) + "\n"
        print(text)
        md.append(text)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
