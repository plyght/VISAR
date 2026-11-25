"""macOS: .mp4/.mov decoding through AVFoundation, checked against FFmpeg.

Usage: videoio_macos_check.py --lib libvisar.dylib CLIP REFERENCE.y4m [CLIP REFERENCE.y4m ...]

For every clip (H.264 / HEVC made by ffmpeg in CI), VISAR's AVFoundation
reader must give the same frame count as the reference (ffmpeg's own decode
of the clip, as .y4m, read by VISAR), the same size, frames close to the
reference (PSNR), and the right frame after seeking. cv2.VideoCapture (its
FFmpeg backend, software decoding) is timed on the same file for the read
speed ratio (above 1x = VISAR faster).
"""
import os
import sys
import time

if "--lib" in sys.argv:
    _i = sys.argv.index("--lib")
    os.environ["VISAR_LIB"] = sys.argv[_i + 1]
    del sys.argv[_i:_i + 2]
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visar_ctypes as vs  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def read_all(path):
    cap = vs.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return cap, frames


def main():
    args = sys.argv[1:]
    failed = []
    for clip, ref in zip(args[0::2], args[1::2]):
        cap, frames = read_all(clip)
        _, want = read_all(ref)
        name = os.path.basename(clip)
        fourcc = int(cap.get(6)).to_bytes(4, "little").decode(errors="replace")
        print(f"{name}: backend {cap.backend}, codec {fourcc}, {cap.width}x{cap.height}, {cap.get(5):.3f} fps, "
              f"FRAME_COUNT {cap.get(7):.0f}, read {len(frames)} frames; reference {len(want)}")
        if cap.backend != "AVFoundation":
            failed.append(f"{name}: backend {cap.backend}")
        if len(frames) != len(want) or int(cap.get(7)) != len(want):
            failed.append(f"{name}: {len(frames)} frames (count {cap.get(7)}), want {len(want)}")
        n = min(len(frames), len(want))
        if n:
            ps = [psnr(a, b) for a, b in zip(frames[:n], want[:n])]
            print(f"  PSNR vs ffmpeg's decode: min {min(ps):.1f} dB, mean {np.mean(ps):.1f} dB")
            if min(ps) < 30:
                failed.append(f"{name}: min PSNR {min(ps):.1f} dB")
        for k in (n // 2, n - 1, 3, 0):
            cap.set(1, k)
            ok, f = cap.read()
            p = psnr(f, want[k]) if ok else 0
            if p < 30:
                best = max(range(n), key=lambda j: psnr(f, want[j])) if ok else -1
                failed.append(f"{name}: seek to {k}: PSNR {p:.1f} (got frame {best}?)")
        print(f"  seek checks done (frames {n // 2}, {n - 1}, 3, 0)")
        cap.release()

        # Read speed: whole file, reused BGR buffer, best of 3.
        buf = np.empty((cap.height, cap.width, 3), np.uint8)

        def ours():
            c = vs.VideoCapture(clip)
            while c.read(buf)[0]:
                pass
            c.release()

        def theirs():
            c = cv2.VideoCapture(clip, cv2.CAP_FFMPEG)
            while c.read(buf)[0]:
                pass
            c.release()

        tv = min(timed(ours) for _ in range(3))
        line = f"  read {n} frames: VISAR (AVFoundation) {tv * 1e3:.1f} ms"
        if cv2 is not None:
            to = min(timed(theirs) for _ in range(3))
            line += f", cv2 FFmpeg {to * 1e3:.1f} ms, speedup {to / tv:.2f}x"
        print(line, flush=True)
    for f in failed:
        print("FAIL", f)
    print("all clips pass" if not failed else f"{len(failed)} failures")
    return 1 if failed else 0


def timed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


if __name__ == "__main__":
    sys.exit(main())
