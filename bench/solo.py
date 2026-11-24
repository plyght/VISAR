"""Times VISAR alone (no OpenCV calls in between, one library copy) at 1 and
N threads, to see how operations scale without other thread pools competing
for the cores.

    python3 bench/solo.py LIB [--filter a,b] [--threads N]
"""
import argparse
import importlib.util
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests", "py"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lib")
    ap.add_argument("--filter", default="")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--md")
    args = ap.parse_args()
    os.environ["VISAR_LIB"] = args.lib
    spec = importlib.util.spec_from_file_location("bench", os.path.join(here, "bench_vs_opencv.py"))
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["bench"]
    spec.loader.exec_module(m)
    sys.argv = saved
    import cv2
    cv2.setNumThreads(1)  # OpenCV is never called while timing; keep its pool idle
    n = args.threads or os.cpu_count()
    keys = [k.strip().lower() for k in args.filter.split(",") if k.strip()]
    rows = []
    for name, ours, _ in m.cases(1080, 1920):
        if keys and not any(k in name.lower() for k in keys):
            continue
        res = []
        for t in (1, n):
            m.vs.set_num_threads(t)
            ours()
            best = float("inf")
            t0 = time.perf_counter()
            k = 0
            while (time.perf_counter() - t0 < 1.0 or k < 10) and k < 500:
                s = time.perf_counter(); ours(); best = min(best, time.perf_counter() - s); k += 1
            res.append(best)
        rows.append((name, res[0], res[1]))
        print(f"  {name:<28} 1T {res[0] * 1e3:8.3f} ms   {n}T {res[1] * 1e3:8.3f} ms   scaling {res[0] / res[1]:5.2f}x", flush=True)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(f"VISAR alone, 1 vs {n} threads\n\n| Case | 1T (ms) | {n}T (ms) | Scaling |\n| --- | ---: | ---: | ---: |\n")
            for name, a, b in rows:
                fh.write(f"| {name} | {a * 1e3:.3f} | {b * 1e3:.3f} | {a / b:.2f}x |\n")


if __name__ == "__main__":
    main()
