# VISAR

**Visual Imaging, Sensing, and Recognition**: a dependency-free computer-vision library in Zig, built as a full replacement for OpenCV.

![VISAR live tracking demo](https://raw.githubusercontent.com/plyght/VISAR/demo-assets/track.gif)

*Rendered on every push by a stock GitHub Actions runner ([demos workflow](.github/workflows/demos.yml)). The pipeline is gray → Gaussian 5×5 → background subtraction → morphology → connected components → multi-object tracker, entirely in VISAR. The job summary shows its frame rate, its accuracy against the synthetic scene's ground truth, and a full speed comparison against OpenCV on the same runner.*

```sh
curl -fsSL https://raw.githubusercontent.com/plyght/VISAR/main/install.sh | sh
```

- **No dependencies.** Only the Zig compiler; codecs, maths and threading are all in-tree. Cross-compile to any target from any host.
- **Matches OpenCV.** 741 differential cases against OpenCV 5.0 pass on every change (`zig build difftest`). Most are bit-exact; the few tolerances are documented in the test file.
- **Fast.** SIMD kernels on `@Vector`, a persistent thread pool, fixed-point paths that mirror OpenCV's arithmetic, and a pure-Zig PNG codec that decodes faster than OpenCV's.
- **Runs anywhere.** Linux, macOS, Windows, wasm32-wasi; aarch64, x86_64 and riscv64.
- **Usable from anything.** An idiomatic Zig API plus a stable C ABI (`include/visar.h`) as shared and static libraries.

Speed against OpenCV 5.0 is measured on every push on stock GitHub runners (x86-64 and Apple Silicon); see the [demos workflow](.github/workflows/demos.yml) job summaries. On Apple Silicon the median speedup across 32 kernels is about 2x with all cores.

See [PRD.md](PRD.md) for the full scope and roadmap. VISAR is early: the roadmap's M0 (foundations) is in place and M1 (core imaging) is in progress.

## What works today

| Area | Status |
| --- | --- |
| Core | `Image(T)` with strided views, `AnyImage`, saturating arithmetic, `convertTo`, `addWeighted`, reductions, split/merge, thread pool |
| imgproc | `cvtColor` (gray, BGR/RGB/A reorders, HSV, YCrCb), resize (nearest, linear, area, cubic), Gaussian/box/Sobel/Scharr/separable filters, threshold (incl. Otsu, triangle), `inRange`, morphology (all `morphologyEx` ops and element shapes), flip/rotate/transpose, `warpAffine`/`warpPerspective`, `copyMakeBorder`, `equalizeHist`, `integral`, `connectedComponentsWithStats`, `findContours` (all modes and methods, exact), shape analysis (moments, Hu, area, length, hulls, defects, minAreaRect, minEnclosingCircle, fitEllipse, approxPolyDP, pointPolygonTest, matchShapes), drawing (incl. OpenCV-exact `drawContours`, `fillPoly`, `polylines`) |
| Codecs | PNG (all colour types and depths, Adam7; custom inflate), JPEG (in progress), QOI, PNM/PAM, BMP (incl. RLE), animated GIF (encode) |
| Video | Background subtraction, multi-object tracker, MOSSE correlation tracker (follows a selected object, ~1600 updates/s per object) |
| Interfaces | Zig module, C ABI, `visar` CLI (clig.dev-style: JSON output, helpful errors, pipelines, bench, demos) |

## Build

Needs Zig 0.17.0.

```sh
zig build -Doptimize=ReleaseFast          # library, C ABI (zig-out/lib) and CLI (zig-out/bin/visar)
zig build test                            # unit tests
zig build difftest -Doptimize=ReleaseSafe # compare against OpenCV (needs python3, numpy, opencv-python-headless)
zig build bench -Doptimize=ReleaseFast -- --threads 1   # speed vs OpenCV
zig build demo-track -Doptimize=ReleaseFast -- --out track.gif
```

## Command line

```console
$ visar inspect photo.png photo.jpg
FILE       FORMAT     SIZE  CHANNELS  DEPTH     BYTES
─────────  ──────  ───────  ────────  ─────  ────────
photo.png  png     480x270  3 RGB     u8     33.2 KiB
photo.jpg  jpeg    480x270  3 RGB     u8     49.9 KiB

$ visar resize photo.png thumb.png 320x
  ✓ thumb.png  320×180 RGB 8-bit · PNG 72.9 KiB · 22.76 ms · area

$ visar pipeline scan.jpg mask.png -- gray , blur --sigma 1.5 , threshold --otsu , morph open
$ cat photo.png | visar gray - - --format png | visar inspect -
$ visar demo track --out track.gif          # the demo above, on your machine
$ visar bench --filter blur,resize          # time the kernels on this CPU
```

Commands: `info`, `inspect`, `convert`, `resize`, `rotate`, `flip`, `gray`, `blur`, `threshold`, `edges`, `equalize`, `morph`, `pipeline`, `bench`, `demo`. Every command takes `--json`, `--threads`, `-q` and `-v`. Errors come with a hint, and mistyped commands, options and file names get a "did you mean". An existing output file is never overwritten without `-f`. Run `visar help <command>` for examples.

## Use from Zig

```sh
zig fetch --save git+https://github.com/plyght/VISAR
```

```zig
// build.zig
exe.root_module.addImport("visar", b.dependency("visar", .{}).module("visar"));
```

```zig
const visar = @import("visar");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var pool = visar.Pool.init(gpa, init.io, null);
    defer pool.deinit();
    const exec = visar.Exec.on(&pool);

    var img = try visar.codecs.read(gpa, init.io, "in.png", .{ .mode = .color });
    defer img.deinit(gpa);
    const src = img.u8;

    var gray = try visar.Image(u8).init(gpa, src.width, src.height, 1);
    defer gray.deinit(gpa);
    try visar.imgproc.cvtColor(exec, src, gray, .rgb2gray);

    var blurred = try visar.Image(u8).initLike(gpa, gray);
    defer blurred.deinit(gpa);
    try visar.imgproc.gaussianBlur(exec, gpa, gray, blurred, .{ .sigma_x = 2 });

    try visar.codecs.write(gpa, init.io, "out.png", .{ .u8 = blurred }, .{});
}
```

Kernels never allocate behind your back: anything that needs scratch memory takes an allocator, and threading goes through an explicit `Exec` (`.serial` or a pool).

## Use from C

```c
#include <visar.h>

visar_image src, dst = { w, h, 1, VISAR_8U, w, out_pixels };
visar_imdecode(bytes, len, /*mode*/ 1, /*bgr*/ 1, &src);
visar_cvt_color(&src, &dst, /*COLOR_BGR2GRAY*/ 6);
visar_image_free(&src);
```

Constants use OpenCV's numbering, so porting code is mostly mechanical.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
