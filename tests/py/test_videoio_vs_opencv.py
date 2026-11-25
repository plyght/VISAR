"""Differential tests for video files: VISAR (C ABI) against OpenCV.

  - Motion-JPEG AVIs written by VISAR open in cv2.VideoCapture (its default
    backend, and OpenCV's own AVI reader when present) with the right frame
    count, size and rate, and decode to VISAR's frames within JPEG
    tolerance; uncompressed AVIs exactly.
  - cv2.VideoWriter('MJPG') files (every backend that writes them) are read
    by VISAR with the same count and frames as cv2's own decode; seeking
    gives the frame cv2 gives.
  - .y4m written by VISAR reads back in cv2 (FFmpeg backend).
Run: zig build difftest
"""
import os
import sys
import tempfile

import cv2

if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visar_ctypes as vs  # noqa: E402

results = []


def check(name, ok, info=""):
    results.append((name, ok, info))


def scene(n, h, w, seed=3):
    """Moving shapes over a textured background: n BGR frames."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([127 + 60 * np.sin(x / 23.0 + k) * np.cos(y / 17.0 - k) for k in range(3)], -1)
    base = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    frames = []
    for t in range(n):
        f = base.copy()
        cv2.circle(f, (int(w * 0.2 + 3 * t) % w, h // 2), max(4, h // 8), (40, 200, 90), -1)
        cv2.rectangle(f, (w // 2, int(h * 0.1 + 2 * t) % h), (w // 2 + w // 6, int(h * 0.1 + 2 * t) % h + h // 6), (230, 40, 40), -1)
        cv2.putText(f, str(t), (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        frames.append(f)
    return frames


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def read_all_cv(path, api=cv2.CAP_ANY):
    cap = cv2.VideoCapture(path, api)
    if not cap.isOpened():
        return None, None
    props = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return props, out


def read_all_vs(path, channels=3):
    cap = vs.VideoCapture(path)
    props = (int(cap.get(7)), cap.width, cap.height, cap.get(5))
    out = []
    while True:
        ok, f = cap.read(channels=channels)
        if not ok:
            break
        out.append(f)
    return cap, props, out


def backends():
    """cv2 backends that may read/write AVI here (name, api id)."""
    names = {b: cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()}
    return names


def main():
    tmp = tempfile.mkdtemp(prefix="visar-videoio-")
    H, W, N, FPS = 120, 160, 24, 25.0
    src = scene(N, H, W)
    names = backends()
    print("cv2 video backends:", ", ".join(names.values()))
    mjpg = cv2.VideoWriter_fourcc(*"MJPG")

    # ---- VISAR writes, OpenCV reads
    for q in (95, 75):
        path = os.path.join(tmp, f"visar_q{q}.avi")
        w = vs.VideoWriter(path, vs.fourcc("MJPG"), FPS, (W, H), quality=q)
        for f in src:
            w.write(f)
        w.release()
        _, vprops, vframes = read_all_vs(path)
        check(f"VISAR MJPG q{q}: VISAR reads its own file", vprops == (N, W, H, FPS) and len(vframes) == N, str(vprops))
        apis = [("default", cv2.CAP_ANY)]
        if cv2.CAP_OPENCV_MJPEG in names:
            apis.append(("OPENCV_MJPEG", cv2.CAP_OPENCV_MJPEG))
        for label, api in apis:
            props, frames = read_all_cv(path, api)
            if props is None:
                check(f"VISAR MJPG q{q} -> cv2 {label}: opens", False)
                continue
            check(f"VISAR MJPG q{q} -> cv2 {label}: count/size/fps", props[0] == N and props[1:3] == (W, H) and abs(props[3] - FPS) < 1e-3 and len(frames) == N, f"{props}, {len(frames)} frames")
            if len(frames) == N:
                worst_src = min(psnr(a, b) for a, b in zip(frames, src))
                worst_dec = min(psnr(a, b) for a, b in zip(frames, vframes))
                mx = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(frames, vframes))
                # Against the source: JPEG loss. Against VISAR's decode of the
                # same file: decoder differences only (ffmpeg's IDCT and
                # chroma upsampling are not libjpeg's).
                # OpenCV's own JPEG at the same quality sets the bar.
                ref = min(psnr(cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR), f) for f in src)
                check(f"VISAR MJPG q{q} -> cv2 {label}: frames vs source", worst_src > ref - 1.5, f"min PSNR {worst_src:.1f} dB (cv2.imencode q{q}: {ref:.1f} dB)")
                exact = label == "OPENCV_MJPEG"
                check(f"VISAR MJPG q{q} -> cv2 {label}: same decode as VISAR", (mx <= 1) if exact else worst_dec > 36, f"min PSNR {worst_dec:.1f} dB, max diff {mx}")

    # Uncompressed AVI: exact.
    path = os.path.join(tmp, "visar_raw.avi")
    w = vs.VideoWriter(path, 0, FPS, (W, H))
    for f in src:
        w.write(f)
    w.release()
    props, frames = read_all_cv(path)
    ok = props is not None and props[0] == N and len(frames) == N and all(np.array_equal(a, b) for a, b in zip(frames, src))
    check("VISAR raw BGR AVI -> cv2: exact", ok, str(props))

    # .y4m.
    path = os.path.join(tmp, "visar.y4m")
    w = vs.VideoWriter(path, -1, FPS, (W, H))
    for f in src:
        w.write(f)
    w.release()
    _, vprops, vframes = read_all_vs(path)
    props, frames = read_all_cv(path, cv2.CAP_FFMPEG) if cv2.CAP_FFMPEG in names else (None, None)
    if props is not None:
        worst = min(psnr(a, b) for a, b in zip(frames, vframes)) if len(frames) == N else 0
        check("VISAR y4m -> cv2 FFMPEG: count and frames", len(frames) == N and props[1:3] == (W, H) and worst > 33, f"{props}, min PSNR {worst:.1f} dB vs VISAR's own read")
    check("VISAR y4m: reads back", vprops[:3] == (N, W, H) and min(psnr(a, b) for a, b in zip(vframes, src)) > 30, str(vprops))

    # ---- OpenCV writes, VISAR reads
    writers = [("default", cv2.CAP_ANY)]
    if cv2.CAP_OPENCV_MJPEG in names:
        writers.append(("OPENCV_MJPEG", cv2.CAP_OPENCV_MJPEG))
    if cv2.CAP_FFMPEG in names:
        writers.append(("FFMPEG", cv2.CAP_FFMPEG))
    for label, api in writers:
        path = os.path.join(tmp, f"cv2_{label}.avi")
        cw = cv2.VideoWriter(path, api, mjpg, FPS, (W, H))
        if not cw.isOpened():
            check(f"cv2 {label} MJPG writer opens", False)
            continue
        for f in src:
            cw.write(f)
        cw.release()
        cap, vprops, vframes = read_all_vs(path)
        check(f"cv2 {label} MJPG -> VISAR: count/size/fps", vprops[0] == N and vprops[1:3] == (W, H) and abs(vprops[3] - FPS) < 1e-3 and len(vframes) == N, str(vprops))
        # cv2's own decode of the file, through its own AVI reader when it
        # has one (libjpeg: VISAR matches it bit for bit) and through the
        # default backend.
        refs = []
        if cv2.CAP_OPENCV_MJPEG in names:
            refs.append(("OPENCV_MJPEG", cv2.CAP_OPENCV_MJPEG, True))
        refs.append(("default", cv2.CAP_ANY, False))
        for rlabel, rapi, exact in refs:
            props, frames = read_all_cv(path, rapi)
            if props is None or len(frames) != N:
                check(f"cv2 {label} MJPG -> VISAR vs cv2 {rlabel} decode", False, f"cv2 read {None if frames is None else len(frames)} frames")
                continue
            mx = max(int(np.abs(a.astype(int) - b.astype(int)).max()) for a, b in zip(frames, vframes))
            worst = min(psnr(a, b) for a, b in zip(frames, vframes))
            check(f"cv2 {label} MJPG -> VISAR vs cv2 {rlabel} decode", (mx <= 1) if exact else worst > 36, f"max diff {mx}, min PSNR {worst:.1f} dB")
            if rlabel == "default":
                # Seeking: frame k by CAP_PROP_POS_FRAMES matches cv2's frame k.
                bad = []
                for k in (17, 3, 0, N - 1):
                    cap.set(1, k)
                    ok, f = cap.read()
                    if not ok or psnr(f, frames[k]) < (99 if exact else 36):
                        bad.append(k)
                check(f"cv2 {label} MJPG -> VISAR seek", not bad, f"bad frames {bad}" if bad else "")
        # BGRA and gray reads of the same file.
        cap = vs.VideoCapture(path)
        ok4, f4 = cap.read(channels=4)
        check(f"cv2 {label} MJPG -> VISAR BGRA", ok4 and np.array_equal(f4[..., :3], vframes[0]) and (f4[..., 3] == 255).all())
        cap.release()

    # cv2 writes uncompressed I420 / raw: VISAR reads them.
    if cv2.CAP_FFMPEG in names:
        for code in ("I420", "IYUV"):
            path = os.path.join(tmp, f"cv2_{code}.avi")
            cw = cv2.VideoWriter(path, cv2.CAP_FFMPEG, cv2.VideoWriter_fourcc(*code), FPS, (W, H))
            if not cw.isOpened():
                continue
            for f in src:
                cw.write(f)
            cw.release()
            try:
                _, vprops, vframes = read_all_vs(path)
            except vs.VisarError as e:
                check(f"cv2 FFMPEG {code} -> VISAR", False, str(e))
                continue
            props, frames = read_all_cv(path, cv2.CAP_FFMPEG)
            worst = min(psnr(a, b) for a, b in zip(frames, vframes)) if len(frames) == len(vframes) == N else 0
            check(f"cv2 FFMPEG {code} -> VISAR vs cv2 decode", worst > 35, f"{vprops}, min PSNR {worst:.1f} dB")

    # Unsupported files fail cleanly with an explanation.
    path = os.path.join(tmp, "junk.avi")
    with open(path, "wb") as f:
        f.write(b"RIFF\x10\x00\x00\x00AVI LIST")
    try:
        vs.VideoCapture(path)
        check("truncated AVI fails cleanly", False)
    except vs.VisarError as e:
        check("truncated AVI fails cleanly", True, str(e))

    width = max(len(r[0]) for r in results)
    failed = [r for r in results if not r[1]]
    for name, ok, info in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {info}")
    print(f"\n{len(results) - len(failed)}/{len(results)} video cases pass")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
