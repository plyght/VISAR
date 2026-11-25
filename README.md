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
| dnn | ONNX inference engine (protobuf reader, graph IR, fusion passes, liveness memory plan with in-place concatenation, blocked-layout SIMD CPU kernels with row-streamed conv chains and cache-blocked GEMM-class convolutions; a Metal GPU backend on macOS that runs the whole plan in one command buffer, loaded at run time with no SDK); `FaceDetector` runs YuNet with results matching `cv2.FaceDetectorYN` and about 3.5x faster on CPU; `ObjectDetector` runs the Model Zoo's NanoDet and YOLOX (COCO) with the zoo pipelines' detections, faster than `cv2.dnn` |
| Video | Background subtraction, multi-object tracker, MOSSE correlation tracker (follows a selected object, ~5000 updates/s per object with its three-scale search) |
| Video I/O | `VideoCapture` / `VideoWriter`: AVI (Motion-JPEG, uncompressed, I420/YV12/Y800; idx1 and OpenDML indexes, seeking by frame) and YUV4MPEG2 in pure Zig on every platform; .mp4/.mov (H.264, HEVC) through AVFoundation on macOS; cameras (AVFoundation, V4L2) |
| Interfaces | Zig module, C ABI, `visar` CLI (clig.dev-style: JSON output, helpful errors, pipelines, bench, demos) |

## Build

Needs Zig 0.17.0.

```sh
zig build -Doptimize=ReleaseFast          # library, C ABI (zig-out/lib) and CLI (zig-out/bin/visar)
zig build test                            # unit tests
zig build difftest -Doptimize=ReleaseSafe # compare against OpenCV (needs python3, numpy, opencv-python-headless)
zig build bench -Doptimize=ReleaseFast -- --threads 1   # speed vs OpenCV
zig build face4k -Doptimize=ReleaseFast         # the 4K face pipeline vs OpenCV, stage by stage (needs VISAR_MODELS)
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
$ visar faces group.jpg --out boxed.png     # YuNet faces: score, box and landmarks per line (or --json)
$ visar faces clip.avi --json               # tracked through a video (.avi, .y4m; .mp4/.mov on macOS)
$ visar convert clip.mp4 clip.avi           # video to Motion-JPEG AVI (or .y4m)
```

### Live tracking and face tracking

`visar track` runs a tracker on the camera (macOS, Linux) in a window: moving objects by default, objects you drag a box around (MOSSE locks), or faces:

```sh
curl -LO https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
visar track --faces face_detection_yunet_2023mar.onnx
```

Faces are detected with YuNet (`visar.dnn.FaceDetector`) and followed by MOSSE filters between detections, so the network runs on a fraction of the frames: every `--detect-every` frames (default 4), or at once when a face is lost. Scheduled detections are postponed while the picture does not change, and most of them run only on crops around the tracked faces (`--change-level`, PRD §8.6 "change-driven inference"); every fourth one still covers the whole frame to find new faces. Between whole-frame detections the gray tracking image is computed only where the tracker reads it (each filter's search window, the rows of the change check, the patches of detections), bit-identical there, so most frames read a small part of the camera frame (`--full-gray` computes all of it every frame). A whole-frame detection is spread over 4 frames (`--detect-slices N`; 1 = on one frame) so that no frame waits for all of the network, MOSSE carrying the faces meanwhile. The HUD and the `--json` summary show the faces, the detector's time per run, its rate, and how many frames it skipped or ran crop-only. `--source image:PATH` moves a still picture over a canvas, so the same pipeline can be tried and checked against a known motion anywhere:

```sh
visar track --faces face_detection_yunet_2023mar.onnx --source image:selfie.jpg --face-width 640 --headless --frames 600 --json
```

`--source video:PATH` plays a video file through the pipeline instead (at its frame rate in a window, as fast as it decodes with `--headless`), and `--record FILE.avi` (or `.y4m`) writes the annotated frames at full size instead of a GIF.

`visar faces` runs the same detector and tracker on files: an image (one detection; `--out` draws the faces), or a video (.avi, .y4m, .mp4/.mov on macOS, or a list of frame images; the tracker with the same `--detect-every` / `--change-level` options; `--out` writes an annotated GIF, or a full-size `.avi` / `.y4m`). The model comes from `--model FILE` or `VISAR_MODELS`:

```sh
VISAR_MODELS=. visar faces selfie.jpg --json --out boxed.png
VISAR_MODELS=. visar faces clip.mp4 --out tracked.avi        # macOS; elsewhere convert first:
ffmpeg -i clip.mp4 -c:v mjpeg -q:v 3 clip.avi && VISAR_MODELS=. visar faces clip.avi --out tracked.gif
```

### Video files

`visar.videoio.VideoCapture` and `VideoWriter` are OpenCV's VideoCapture/VideoWriter for files (also in the C ABI as `visar_video_capture_*` / `visar_video_writer_*`, with `cv2.CAP_PROP_*` numbers):

| Format | Read | Write | Where |
| --- | --- | --- | --- |
| AVI, Motion-JPEG (`MJPG`, OpenCV's portable writer format) | yes | yes (default for `.avi`; OpenDML past 1 GB) | everywhere, pure Zig |
| AVI, uncompressed BGR / I420 / YV12 / Y800 | yes | BGR (fourcc 0) | everywhere |
| YUV4MPEG2 (`.y4m`) | yes | yes (4:2:0) | everywhere |
| MP4 / QuickTime (H.264, HEVC, ...) | yes, hardware decoding | no | macOS (AVFoundation / VideoToolbox, loaded at run time); elsewhere an error that says how to convert with ffmpeg |

Frames are read into a caller-owned BGR, BGRA or gray image; the frame count, rate and FourCC come from the file, and seeking is by frame number (AVI through its `idx1` or OpenDML index, rebuilt by scanning when a file was never finished; YUV4MPEG2 by arithmetic; MP4 by time). Files are treated as untrusted: every chunk size is bounds-checked, and truncated or corrupted files (fuzzed in the tests) give an error or fewer frames, never a crash. `zig build difftest` checks both directions against OpenCV: VISAR's AVIs open in `cv2.VideoCapture` (FFmpeg and OpenCV's own reader) with the right count and frames, and `cv2.VideoWriter('MJPG')` files (both backends) read in VISAR bit-identical to OpenCV's own MJPEG decode. Speed against OpenCV 5.0 for 16 frames of 1280x720 Motion-JPEG (`bench/bench_vs_opencv.py --filter Video`; above 1x = VISAR faster):

| | x86_64, 4 cores: vs FFmpeg | vs OpenCV MJPEG | Apple M1 runner, 3 threads: vs FFmpeg | vs OpenCV MJPEG |
| --- | ---: | ---: | ---: | ---: |
| read (open + decode every frame) | 1.62x | 1.21x | 1.88x | 1.19x |
| write (encode q95 + mux) | 1.94x | 1.81x | 3.23x | 2.04x |

On macOS, .mp4/.mov frames come out of AVFoundation as 4:2:0 and are converted by VISAR, so they are bit-identical to FFmpeg's decode of the same file read through `.y4m` (checked in CI on H.264 and HEVC clips, including seeking). On the CI runner (a VM with no hardware decoder) AVFoundation reads at 0.4-0.7x of cv2's multi-threaded FFmpeg software decoding.

### 4K face detection: measured, and what 120 fps on an M3 needs

`zig build face4k -Doptimize=ReleaseFast` (needs `VISAR_MODELS` with the YuNet model and `largest_selfie.jpg`) times the whole pipeline on a camera-native 3840x2160 BGRA frame (the zoo selfie enlarged to 4K): colour + downscale straight into the network's planar f32 input in one pass (`FaceDetector.prepare`, bit-identical to `cvtColor` + `resize(INTER_AREA)` + `blobFromImage`), YuNet on the CPU, boxes mapped back to 4K, against `cv2.cvtColor` + `cv2.resize` + `cv2.FaceDetectorYN` at the same thread count. The detector runs at 640x360 (4K / 6): it finds faces from about 190 px wide at 4K (5% of the frame width; 960x540 finds them from about 124 px at roughly twice the network cost). Both libraries return the same boxes. Median ms (GitHub `macos-latest`, an M1-class VM; x86 is a 4-core Sapphire Rapids VM with AVX-512):

| 640x360, ms | macOS 1T | macOS 3T | x86 1T | x86 4T |
| --- | ---: | ---: | ---: | ---: |
| colour + downscale into the input | 1.73 | 0.63 | 1.27 | 0.64 |
| the same + the gray tracking image (one pass)¹ | 1.68 | 0.71 | 1.42 | 0.68 |
| YuNet (CPU; 206 M MACs) | 5.70 | 2.17 | 4.40 | 2.39 |
| decode + NMS + map to 4K | 0.01 | 0.01 | 0.01 | 0.01 |
| **VISAR end to end** | **7.86** | **2.80** | **5.73** | **3.08** |
| OpenCV end to end | 17.64 | 7.62 | 18.88 | 10.98 |
| VISAR change-driven steady state² | 0.96 | 0.55 | 0.90 | 0.58 |
| the same, whole gray image every frame (before) | 2.02 | 0.92 | 2.37 | 1.17 |

¹ `FaceDetector.prepareWithGray` / `FaceTracker.trackingImage`: the tracker's gray tracking image (`grayDownsample`, gray + /6 area shrink) from the same block sums as the detector input, both bit-identical to the separate passes; as two passes it costs 3.1-3.6 ms at 1T on the macOS runner (OpenCV, four calls: 11.5 ms). ² `visar track`'s face tracker at level 2, a detection scheduled every 4th frame, the selfie moving slowly over a 4K canvas (`bench/face4k_steady.zig`): effective ms per frame including the gray tracking image and MOSSE; whole-frame detections run on 38 of 600 frames, crop-only ones on 84, 112 are skipped as unchanged. On frames that will probably run a whole-frame detection the gray image comes from the same pass as the detector input; on the others only the parts the tracker reads are computed (`FaceOptions.partial_gray`): 17% of the frame is read for gray over the run instead of all of it every frame (1.3-1.7 ms at 1T before), with the same detector runs and the same tracks frame for frame (600/600 frames followed, mean IoU 0.922 to the moved face, 1.5 px centre error at 4K, in both). The change check now compares every 16th row of the tracking image (96 px apart at 4K, closer than the smallest face found), computed straight from the frame. Interleaved A/B on the runners, three rounds: macOS 1.91-2.35 -> 0.83-1.05 ms at 1T, 0.82-1.23 -> 0.46-0.64 ms at 3T; x86 2.38 -> 0.90 ms at 1T, 1.21 -> 0.58 ms at 4T; with a whole-frame detection on every 4th frame (level 0) 2.75-3.57 -> 1.95-2.36 ms (macOS 1T). The runners are noisy VMs: medians move by 20-50% between runs (the bench prints the minimum too).

**Per-frame latency.** The table above gives medians and means; a camera pipeline also cares about its slowest frames. `zig build face4k` now times every frame alone (nearest-rank percentiles over 600 frames). On the macOS runner (GitHub `macos-latest`, an M1-class VM with 3 cores, 1 and 3 threads, 640x360 detector), ranges over two runs of the face4k table for detection on every frame and medians (and ranges) over nine runs of `bench/face4k_steady.zig` for the tracker; both runs were on a noisier runner than the table above (end to end 7.5-12.5 ms median at 1T):

| ms per frame | mean | p50 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| VISAR end to end, every frame, 1T | 8.6-12.3 | 8.1-10.8 | 11.4-21.3 | 14.3-27.3 | 17-56 |
| OpenCV end to end, every frame, 1T | 23.1-27.5 | 21.5-26.3 | 32.0-38.5 | 39.1-45.7 | 55-84 |
| VISAR end to end, every frame, 3T | 5.9-7.1 | 5.5-6.9 | 11.7-13.2 | 16.0-18.0 | 21-29 |
| OpenCV end to end, every frame, 3T | 13.4-16.1 | 12.7-14.8 | 18.6-24.1 | 21.1-27.8 | 31-94 |
| VISAR change-driven steady state², detection on one frame (`--detect-slices 1`, before), 1T | 1.45 (1.15-1.81) | 0.56 (0.50-0.67) | 8.81 (8.25-10.14) | 13.6 (9.0-16.4) | 20.9 (10.3-30.7) |
| VISAR change-driven steady state², detection over 4 frames (default), 1T | 1.53 (1.30-1.84) | 0.77 (0.58-1.08) | **4.59** (3.38-5.85) | **6.9** (5.5-9.3) | **9.0** (8.0-18.2) |
| VISAR change-driven steady state², detection on one frame (before), 3T | 1.50 (1.40-1.75) | 0.96 (0.76-1.01) | 7.69 (6.57-8.31) | 9.5 (8.4-12.2) | 13.7 (10.4-21.5) |
| VISAR change-driven steady state², detection over 4 frames (default), 3T | 1.67 (1.06-2.52) | 1.15 (0.83-1.62) | **4.02** (2.25-7.00) | **7.1** (3.4-12.3) | 14.1 (5.7-27.5) |

Detecting on every frame, OpenCV takes 2.4-2.6x VISAR's time at the median on one thread, 1.8-2.8x at p95 and 1.7-2.7x at p99 (3T: 2.2-2.3x, 1.6-1.8x, 1.3-1.6x). The change-driven tracker's tail was its whole-frame detections: 38 of 600 frames (6%, so they set p95 and p99), about 11 ms each at 1T and 6-9 ms at 3T on the runner (the frames without a detection have a p99 of about 2 ms), so at 120 fps (8.3 ms) one thread missed the deadline on every whole-frame detection, about every 16th frame. A whole-frame detection is now spread over 4 frames (`FaceOptions.detect_slices`, `--detect-slices N`; at most `--detect-every`): the first writes the detector input from the frame (in the pass that makes the gray tracking image, as before), the next three each run a part of YuNet's CPU program of about equal estimated cost (`Net.splitRun`: conv MACs plus memory traffic per step; the last part smaller, as that frame also decodes and matches), while MOSSE carries the faces; the detections, of the first frame, are moved by the motion since then of the track they overlap there, then matched as before. Split, the network computes exactly what one run does; the faces just arrive 3 frames later, and new ones start tracks 3 frames late. The rows above are medians (and ranges) over ten interleaved runs on the macOS runner (two workflow runs of five rounds; `bench/face4k_steady.zig --slices N`, level 2, a detection scheduled every 4th frame): p95 drops about 2x at both thread counts and p99 to 6.9 / 7.1 ms at 1T / 3T, inside the 120 fps deadline; the frames of a spread detection average 3.4 ms at 1T (3.2 ms at 3T), and what is left in the tail is mostly crop-only detections (p99 6-8 ms on the runner) and the VM's noise. The detector runs the same 38 whole-frame and 84 crop-only times with 112 postponed; tracking quality is unchanged (600/600 frames followed, mean IoU 0.919 against 0.922, centre error 1.47 against 1.50 px). The mean moves by +0.1-0.2 ms (within the runs' ranges: the same work, in four pieces with colder caches), and p50 by +0.2 ms because 25% instead of 6% of the frames now carry detection work. Over 3 frames: p95 5.1 / 4.9 ms, p99 8.2 / 6.5 ms (1T / 3T); over 2: p95 7.4 / 6.9 ms, p99 13.3 / 8.8 ms. On the 4-core x86 VM (three rounds) p95 / p99 went 7.4 / 10.4 -> 3.3 / 4.3 ms at 1T and 3.4 / 6.7 -> 1.8 / 4.0 ms at 4T, the mean 1.25 -> 1.29 and 0.93 -> 0.88 ms. `--detect-slices 1` detects on one frame, with the same results as before frame for frame; `visar faces` (a file, no deadline) uses 1 unless asked. Other ways of cutting the spike were not taken: a detection thread does nothing at 1T and competes with the frame loop's threads otherwise, and detecting in horizontal bands misses faces larger than a band's overlap and pays for the overlaps.

Before the partial gray image the steady state was 2.72 / 1.42 ms on macOS (1T / 3T) and 3.15 / 1.42 ms on x86; before the round of CPU work before that the same table read, on the same runner class: YuNet 7.59 / 2.64 ms, end to end 9.97 / 3.52 ms, steady state 4.45 / 3.06 ms (1T / 3T). Interleaved A/B on the runner (`bench/face4k_steady.zig --profile`, every step timed alone): YuNet went from 6.4-6.9 to 5.6-6.0 ms at 1T, mostly because the conv kernels' epilogues now read their flags from a local copy (the output stores could alias the kernel struct, so LLVM reloaded the flags, slope and residual test at every stored vector: 1x1 convs with 16 input channels 24% faster, depthwise 3x3 10-15%) and because ReLU and no activation get their own one-op and zero-op epilogues; a whole-frame detection of the tracker no longer pays for a second pass over the 4K frame (colour + downscale + gray image 3.1 -> 1.6 ms).

So on the M1-class runner the end-to-end detection of every 4K frame is at 7.9 ms on one thread (127 fps median, 7.1 ms best) and fits it easily with 3 threads (357 fps); the change-driven tracker needs about 1 ms per frame on one thread. *Extrapolation, not measured:* an M3 performance core runs scalar and NEON code roughly 1.3-1.4x faster than an M1's and a base M3 has 4 of them, so the one-thread end-to-end time would be about 6 ms (inside 8.3 ms with some headroom) and under 2 ms on the performance cores; 960x540 (about 14 ms on one M1 thread) needs 2-3 cores. The network is still 75% of the one-thread time, at about 65% of the core's FMA peak (the two row-streamed chains of 3x3/1x1 convs at full resolution: 1.5 and 2.0 ms of it; the first 3x3 stride-2 conv runs at about 75% of peak, the depthwise 3x3s at about 65%, the 1x1s at 75-90%); once detections are change-driven and the gray tracking image is computed only around the faces, the tracker no longer reads most of the 33 MB frame on most frames: what is left per frame is the MOSSE filters (a few 64x64 FFTs per face) and the amortised detections, each whole-frame one spread over 4 frames. The Metal backend ran YuNet in 2-4 ms on the runners' paravirtual GPU, which says nothing about a real M3 GPU.

Commands: `info`, `inspect`, `convert` (images, and video to `.avi` / `.y4m`), `resize`, `rotate`, `flip`, `gray`, `blur`, `threshold`, `edges`, `equalize`, `morph`, `pipeline`, `bench`, `demo`, `track`. Every command takes `--json`, `--threads`, `-q` and `-v`. Errors come with a hint, and mistyped commands, options and file names get a "did you mean". An existing output file is never overwritten without `-f`. Run `visar help <command>` for examples.

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
