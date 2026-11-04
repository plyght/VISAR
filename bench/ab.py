"""A/B two VISAR builds on the benchmark cases, timed interleaved.

    python3 bench/ab.py LIB_A LIB_B [--filter a,b] [--threads N] [--md FILE]

Both libraries run the same cases from bench_vs_opencv.py, alternating
call by call, so machine load affects them equally; the best time of each
is reported with B's speedup over A (> 1 means B is faster).
"""
import argparse
import importlib.util
import os
import sys
import time

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(here, "..", "tests", "py"))


def load(lib, tag, threads):
    os.environ["VISAR_LIB"] = lib
    sys.modules.pop("visar_ctypes", None)
    spec = importlib.util.spec_from_file_location("bench_" + tag, os.path.join(here, "bench_vs_opencv.py"))
    m = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["bench"]
    spec.loader.exec_module(m)
    sys.argv = saved
    m.vs.set_num_threads(threads)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--filter", default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--md")
    args = ap.parse_args()
    A, B = load(args.a, "a", args.threads), load(args.b, "b", args.threads)
    import cv2
    cv2.setNumThreads(args.threads)
    ca = {n: f for n, f, _ in A.cases(1080, 1920)}
    cb = {n: f for n, f, _ in B.cases(1080, 1920)}
    keys = [k.strip().lower() for k in args.filter.split(",") if k.strip()]
    rows = []
    for name, fa in ca.items():
        fb = cb.get(name)
        if fb is None or (keys and not any(k in name.lower() for k in keys)):
            continue
        ta = tb = float("inf")
        t0 = time.perf_counter()
        n = 0
        while (time.perf_counter() - t0 < 2.0 or n < 10) and n < 500:
            s = time.perf_counter(); fa(); ta = min(ta, time.perf_counter() - s)
            s = time.perf_counter(); fb(); tb = min(tb, time.perf_counter() - s)
            n += 1
        rows.append((name, ta, tb))
        print(f"  {name:<28} A {ta * 1e3:8.3f} ms   B {tb * 1e3:8.3f} ms   B vs A {ta / tb:5.2f}x", flush=True)
    if args.md:
        with open(args.md, "w") as fh:
            fh.write(f"A/B, {args.threads} thread(s)\n\n| Case | A (ms) | B (ms) | B vs A |\n| --- | ---: | ---: | ---: |\n")
            for name, ta, tb in rows:
                fh.write(f"| {name} | {ta * 1e3:.3f} | {tb * 1e3:.3f} | {ta / tb:.2f}x |\n")


if __name__ == "__main__":
    main()
