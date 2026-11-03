"""Runs every tests/py/test_*_vs_opencv.py against the library given by --lib
and fails if any of them fails. Used by `zig build difftest`."""
import glob
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
failed = []
for path in sorted(glob.glob(os.path.join(here, "test_*_vs_opencv.py"))):
    name = os.path.basename(path)
    print(f"== {name}", flush=True)
    if subprocess.call([sys.executable, path, *args]) != 0:
        failed.append(name)
if failed:
    print(f"\nFAILED: {', '.join(failed)}")
    sys.exit(1)
print("\nall differential suites passed")
