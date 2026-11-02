# VISAR — Product Requirements Document

**Visual Imaging, Sensing, and Recognition**
A complete, dependency-free replacement for OpenCV, written in Zig, that is faster than OpenCV and runs on everything from a microcontroller to a GPU cluster.

| | |
| --- | --- |
| Status | Draft v0.1 |
| Owner | plyght |
| Last updated | 2026-11-02 |
| Scope | Everything OpenCV + opencv_contrib does, plus imaging-pipeline, sensing, and training capabilities OpenCV does not have |

---

## 1. Summary

OpenCV is the default computer-vision library, but it is a 25-year-old C++ codebase with a large CMake build, many optional third-party dependencies (FFmpeg, GStreamer, GTK/Qt, protobuf, IPP, TBB, Eigen, libjpeg/png/tiff/webp/openexr, …), large binaries, uneven GPU support (OpenCL `UMat`, a separate CUDA module in contrib), weak WebAssembly and microcontroller support, and a DNN module that trails dedicated inference runtimes.

VISAR rebuilds the whole thing from first principles in Zig:

- **Complete.** VISAR covers every OpenCV main and contrib module, then goes further: a full camera ISP, depth and point clouds, IMU fusion, event, thermal, hyperspectral and medical imaging, and model training as well as inference.
- **Fast.** Median ≥1.5× OpenCV across a public benchmark suite, and no benchmarked operation slower than 0.95× OpenCV.
- **Small and self-contained.** Zero third-party code at build time. Users pay only for what they import.
- **Runs anywhere.** It targets servers, desktops, phones, browsers (WASM), and bare-metal MCUs, across CPU SIMD, every major GPU API, and NPUs.
- **Usable from any language.** It offers an idiomatic Zig API, a stable C ABI, first-class Python, and bindings for other languages. An optional `cv2`-compatible shim lets existing code migrate without a rewrite.

---

## 2. Problem statement

| Pain with OpenCV today | Consequence | VISAR answer |
| --- | --- | --- |
| Heavy CMake build with dozens of optional dependencies | Hard to cross-compile, reproduce, or vendor; builds differ by machine | `zig build` with no external deps. Cross-compile to any target from any host with one command |
| Large binaries and wheels (tens of MB) | Bloats containers, mobile apps, and serverless cold starts | Modular, compile-only-what-you-use. Hard size budgets (§10.3) |
| SIMD via universal intrinsics plus optional IPP. Many kernels are scalar or partly vectorized | Performance is uneven across operations and platforms | Every kernel is written once over Zig `@Vector`, specialized per ISA, and autotuned |
| GPU support is fragmented (OpenCL `UMat`, CUDA in contrib, partial Vulkan in DNN) | You get no single "run it on the GPU" story | One kernel source compiles to CPU, SPIR-V, PTX, AMDGCN, MSL, and WGSL. Same API on every device |
| Operations run one at a time, with a full memory round-trip between each | Pipelines are bandwidth-bound | Lazy op graphs with fusion and tiling, Halide-style, plus comptime pipelines for zero-overhead fixed graphs |
| Poor WASM and no real MCU support | Vision in browsers and on embedded devices needs other libraries | WASM and freestanding (no OS, no heap) are first-class tier-1 targets |
| DNN module trails ONNX Runtime and TensorRT, and has no training | Users bolt on a second framework | Full ONNX runtime, quantization, and an autograd training engine in the same library |
| Hidden global state, implicit allocation, exceptions | Hard to embed, hard to make real-time | Explicit allocators, no globals, error unions, no hidden allocations on hot paths |
| Image decoders are a long-running source of CVEs | Security exposure | Memory-safe Zig, safety-checked decoders, continuous fuzzing |

---

## 3. Goals and non-goals

### 3.1 Goals

1. **Functional superset of OpenCV 4.x/5.x plus opencv_contrib.** Every module has a VISAR equivalent (see the matrix in §7).
2. **Performance leadership.** VISAR beats OpenCV measurably on every tier-1 platform and is competitive with best-in-class specialist libraries.
3. **Zero build-time dependencies.** It is 100% Zig. Codecs, linear algebra, solvers, font rasterizing, and windowing are all implemented in-tree.
4. **Universal portability.** Tier-1 support spans server, desktop, mobile, browser, and bare metal (§9).
5. **Universal acceleration.** It supports CPU SIMD and threads, Vulkan, Metal, CUDA, ROCm, Level Zero, D3D12, WebGPU, and NPUs behind one device abstraction.
6. **Recognition end to end.** It handles ONNX import, inference, quantization, training and fine-tuning, and export.
7. **Sensing beyond cameras.** It covers RAW/ISP, depth, LiDAR, IMU, event, thermal, hyperspectral, and medical imaging.
8. **Easy migration.** A `cv2`-compatible Python shim and a `cv::`-style C++ header shim are provided.
9. **Correct by construction.** Every kernel is differentially tested against OpenCV and reference implementations, fuzzed, and given documented numeric tolerances.

### 3.2 Non-goals

- **Bug-for-bug OpenCV compatibility.** The shim matches documented behavior. Undocumented quirks are reproduced only when the quirk is load-bearing for common code.
- **Large-scale LLM training.** VISAR training covers vision models (classification, detection, segmentation, depth, keypoints, embeddings, small transformers). Multi-node, trillion-parameter training is out of scope.
- **A GUI toolkit.** VISAR ships `imshow`-level windows, overlays, trackbars, and a remote viewer, not a widget framework.
- **Reproducing OpenCV's Java/MATLAB/Julia generators.** Bindings are generated from VISAR's own API description (§12).

---

## 4. Users and personas

VISAR is for everyone who uses OpenCV today, plus the people OpenCV serves poorly.

| Persona | Needs | Primary surface |
| --- | --- | --- |
| **Zig developer** | Idiomatic API, comptime generics, explicit allocators, cross-compilation | Zig package |
| **Python / ML engineer** | `pip install`, NumPy/PyTorch zero-copy, cv2 ergonomics, fast data loading | `visar` and `visar.cv2` |
| **C/C++ developer** | A stable ABI, a single static lib, a header-only C++ wrapper | C ABI and `visar.hpp` |
| **Other languages** (Rust, Go, Swift, Kotlin/Java, C#, JS) | Native-feeling bindings without building C++ | Generated bindings over the C ABI |
| **Mobile developer** | Small binaries, camera access, NPU and GPU use | Swift and Kotlin packages |
| **Web developer** | In-browser vision with SIMD, threads, and WebGPU | npm package (WASM) |
| **Embedded / robotics engineer** | MCU builds with no heap, deterministic latency, sensor fusion, ROS 2 | Freestanding Zig and C ABI |
| **Imaging / camera engineer** | RAW pipelines, ISP tuning, color science | `visar.isp` |
| **Researcher** | Reproducibility, training, exotic sensors | All of the above |

---

## 5. Product principles

1. **No hidden costs.** No hidden allocations, threads, global state, or I/O. Anything that allocates takes an `Allocator`. Anything that spawns work takes an executor.
2. **Pay for what you use.** An app that only resizes a JPEG links only that code. Zig's lazy compilation makes this the default, and VISAR must never defeat it (no registries that pull in everything).
3. **One kernel, every device.** A kernel is written once and specialized by the compiler per ISA and backend. Hand-written per-backend code is allowed only where measured to win, and must be tested against the portable version.
4. **Fuse by default.** Pipelines are graphs, not sequences of full-image passes.
5. **Measured, not claimed.** Every performance claim is backed by the public benchmark suite and CI gates.
6. **Safe by default, fast on request.** Release builds keep bounds and overflow checks on untrusted input paths (decoders, parsers). Hot inner loops are proven or checked once up front.
7. **Stable where it matters.** The C ABI is the stability anchor. The Zig API can evolve with Zig until Zig 1.0.
8. **Every deviation is documented.** Wherever VISAR differs from OpenCV, the docs say so.

---

## 6. Definition of "zero dependencies"

Zero dependencies is a core promise, so it needs a precise definition:

- **Build time:** No third-party source, prebuilt binaries, headers, or code generators. Only the Zig compiler and its standard library. This covers BLAS/LAPACK, FFT, codecs, protobuf (for ONNX), font rasterizing, compression (zlib/deflate, zstd, LZ4, LZW, …), and the shader compilers or translators VISAR needs.
- **Run time:** VISAR may call **OS and driver interfaces** that are always present on a platform, via syscalls, dynamic loading (`dlopen`/`LoadLibrary`), or the Objective-C runtime. Examples are the Vulkan loader, `libcuda`, the Metal framework, V4L2 ioctls, VideoToolbox, MediaCodec, and Win32. These are behind feature flags and degrade gracefully when absent. VISAR never requires an SDK to be installed at build time. The Metal path must call the Objective-C runtime directly, not use a compiled shim that needs Apple's SDK headers.
- **Vendor NPU stacks** that are only available as proprietary user-space libraries (for example Qualcomm QNN or Rockchip RKNN) are loaded at run time as optional **device plugins**. VISAR builds and works fully without them.

---

## 7. Scope: module matrix

Priority: **P0** = needed for the first broadly useful release, **P1** = needed for 1.0 feature parity, **P2** = needed for 1.0 "beyond OpenCV" scope. Everything here is in scope for 1.0. Priority only orders the work.

### 7.1 OpenCV main modules

| OpenCV | VISAR module | Priority | Notes |
| --- | --- | --- | --- |
| core | `visar.core` | P0 | Tensor/Image types, arithmetic, reductions, LUTs, linear algebra, DFT/DCT, random, persistence (YAML/JSON/XML FileStorage-compatible), parallel runtime |
| imgproc | `visar.imgproc` | P0 | Filtering, geometric transforms, color conversions (all OpenCV codes), thresholding, morphology, histograms, contours, shape analysis, Hough, drawing, connected components, distance transform, watershed, grabCut, template matching |
| imgcodecs | `visar.codecs` | P0 | See §8.3 |
| videoio | `visar.videoio` | P1 | Capture and containers, see §8.4 |
| highgui | `visar.ui` | P1 | Windows on Wayland, X11, Win32, Cocoa, Android, iOS, and browser canvas; trackbars; mouse; plus a remote viewer that streams to a browser for headless boxes |
| video | `visar.video` | P1 | Optical flow (LK, Farneback, DIS), background subtraction (MOG2, KNN), Kalman filter, trackers |
| features2d | `visar.features` | P1 | FAST, ORB, SIFT, AKAZE/KAZE, BRISK, MSER, AGAST, GFTT, SimpleBlob; BF and FLANN-style matchers |
| calib3d | `visar.calib` and `visar.geom` | P1 | Pinhole, fisheye and omnidir models; calibration; PnP (all solvers); homography, fundamental and essential matrices; USAC/MAGSAC++ robust estimation; stereo rectification; triangulation; hand-eye |
| 3d (OpenCV 5) | `visar.three` | P1 | Point clouds, normals, ICP, RGB-D odometry, TSDF/volume fusion, meshing |
| objdetect | `visar.detect` | P1 | Haar/LBP cascades, HOG, QR encode/decode, 1D/2D barcodes, ArUco, ChArUco, AprilTag, DNN face detection |
| dnn | `visar.nn` | P1 | See §8.6 |
| ml | `visar.ml` | P1 | SVM, kNN, k-means, trees, random forests, boosting, EM/GMM, logistic regression, PCA/LDA, ANN_MLP |
| photo | `visar.photo` | P1 | Inpainting, non-local-means denoising, HDR merge and tone mapping, seamless cloning, decolor, stylization |
| stitching | `visar.stitch` | P1 | Panorama pipeline: features, bundle adjustment, warpers, seam finders, exposure compensation, blending |
| gapi | `visar.graph` | P0 | VISAR's graph is a core concept, not an add-on (§8.1) |
| flann | `visar.search` | P1 | KD-tree, k-means tree, LSH, HNSW, brute force with SIMD |

### 7.2 opencv_contrib modules

| Contrib | VISAR | Priority | Notes |
| --- | --- | --- | --- |
| xfeatures2d | `visar.features` | P1 | BRIEF, FREAK, DAISY, LATCH, LUCID, BEBLID, TEBLID, VGG, BoostDesc, StarDetector, HarrisLaplace, SURF |
| ximgproc | `visar.imgproc.x` | P1 | Guided filter, domain transform, fast global smoother, edge-aware filters, superpixels (SLIC, SEEDS, LSC), structured edges, fast line detector, thinning, ridge detection, edge drawing |
| xphoto | `visar.photo` | P1 | White balance, BM3D, oil painting, inpainting variants |
| optflow | `visar.video.flow` | P1 | TV-L1, RLOF, PCA flow, SimpleFlow, DeepFlow-class methods |
| tracking | `visar.video.track` | P1 | KCF, CSRT, MIL, MOSSE, MedianFlow, TLD, and DNN trackers (Nano, ViT-based, DaSiam-class) |
| bgsegm | `visar.video.bg` | P1 | GMG, CNT, LSBP, GSOC, MOG |
| stereo | `visar.stereo` | P1 | BM, SGBM, quasi-dense, plus GPU SGM |
| rgbd / sfm | `visar.three`, `visar.slam` | P2 | See §8.8 |
| surface_matching | `visar.three` | P2 | PPF 3D matching |
| face | `visar.face` | P1 | Facemark (LBF, AAM, Kazemi), recognizers |
| text | `visar.text` | P1 | Scene text detection and recognition, plus a full OCR pipeline |
| aruco / wechat_qrcode / mcc | `visar.detect` | P1 | Markers, robust QR, Macbeth chart detection |
| img_hash | `visar.hash` | P1 | aHash, pHash, marr-hildreth, radial variance, block mean, color moment |
| quality | `visar.quality` | P1 | PSNR, SSIM, MS-SSIM, GMSD, BRISQUE, plus LPIPS-class learned metrics through `nn` |
| saliency, line_descriptor, shape, fuzzy, intensity_transform, phase_unwrapping, structured_light, plot, superres, videostab, dnn_superres, alphamat, hfs, reg, bioinspired, ccalib, dpm, xobjdetect, rapid, ovis, viz, cvv, freetype, hdf | Various | P2 | Every module gets a home. `viz` and `ovis` become `visar.ui.3d`. `freetype` becomes an in-tree TrueType/OpenType rasterizer. `hdf` becomes an in-tree HDF5 reader. `cvv` becomes a debug-visualization mode in `visar.ui` |
| cudaarithm, cudaimgproc, cudafeatures2d, … (all `cuda*`) | — | — | Replaced by device-agnostic kernels. There are no separate GPU modules |

### 7.3 Beyond OpenCV

| Area | VISAR | Priority |
| --- | --- | --- |
| Camera ISP and RAW | `visar.isp` | P1 |
| Color science and management | `visar.color` | P1 |
| SLAM, VO/VIO, bundle adjustment, SfM | `visar.slam` | P2 |
| Sparse and dense solvers (Ceres/Eigen-class) | `visar.solve`, `visar.linalg` | P1 |
| Model training | `visar.nn.train` | P2 |
| Depth sensors and LiDAR | `visar.sense.depth`, `visar.sense.lidar` | P2 |
| IMU and multi-sensor fusion | `visar.sense.imu`, `visar.sense.sync` | P2 |
| Event cameras | `visar.sense.event` | P2 |
| Thermal | `visar.sense.thermal` | P2 |
| Hyperspectral and multispectral | `visar.sense.spectral` | P2 |
| Medical and volumetric imaging | `visar.sense.medical` | P2 |
| Data loading and augmentation for training | `visar.data` | P2 |

---

## 8. Functional requirements

### 8.1 Core (`visar.core`, `visar.graph`)

**Data model**
- `Tensor(T, rank)`: compile-time element type and rank for the Zig API. `AnyTensor`: runtime-typed, for bindings and the C ABI.
- Element types: u8, i8, u16, i16, u32, i32, u64, i64, f16, bf16, f32, f64, and bool. Also fp8 (E4M3/E5M2), int4/uint4 packed, and complex64/128.
- Arbitrary strides, zero-copy views and ROIs, broadcasting, negative strides (flips without copying).
- `Image` is a tensor plus metadata:
  - Pixel format: interleaved, planar, and semi-planar. All common YUV layouts (NV12, NV21, I420, YUY2, UYVY, P010, P016, Y210). Bayer patterns (RGGB, BGGR, GRBG, GBRG, Quad Bayer, X-Trans). Packed RAW (10/12/14-bit MIPI).
  - Color space, transfer function, primaries, and range (full or limited).
  - Orientation (EXIF), timestamp, and arbitrary user metadata.
- Device placement and memory kinds: host, pinned host, device, unified/shared, imported external memory (dma-buf, AHardwareBuffer, IOSurface, CUDA external memory, D3D12 shared handles).
- Interop: DLPack import and export, NumPy buffer protocol and array interface, `__cuda_array_interface__`, zero-copy to and from PyTorch, JAX, and CuPy.

**Memory**
- Every allocating API takes an `std.mem.Allocator`. VISAR provides arena, frame-pool (per-pipeline ring), and device allocators.
- A static mode: with shapes known at comptime, a pipeline uses only caller-provided buffers. This is required for MCUs.
- No allocation on the hot path of a prepared pipeline.

**Execution**
- `Executor` abstraction: inline (single thread), thread pool (work-stealing, NUMA-aware, QoS-aware on Apple and Android), and device queues.
- Pipelines and graphs:
  - **Eager mode:** call an operation, get a result. Same ergonomics as OpenCV.
  - **Lazy graph mode:** operations record into a graph. A scheduler fuses elementwise and stencil chains, tiles for cache and shared memory, chooses layouts, and places nodes on devices. The result is a prepared pipeline that is run repeatedly.
  - **Comptime pipelines:** when the graph and shapes are compile-time known, the whole pipeline is specialized and fused at compile time, with zero runtime scheduling overhead.
  - **Streaming:** async, multi-stage pipelines (capture, decode, process, encode or display) with back-pressure, zero-copy hand-off between stages, and per-frame timestamps.
- Autoscheduler plus per-device tuning cache. A TPE or Bayesian search over tile sizes, vector widths, and fusion choices, with tuned tables shipped for tier-1 hardware and optional on-device tuning.
- Determinism mode: bit-reproducible results across runs, and across thread counts on the same device and backend.

**Math**
- In-tree BLAS-like kernels (GEMM, GEMV, batched, low-precision with int8/fp8 dot products, AMX/SME where available).
- In-tree LAPACK-like dense routines (LU, QR, Cholesky, SVD, eigen, least squares).
- FFT and DCT for any size, real and complex, 1/2/3-D, GPU-capable.
- Random: counter-based PRNGs (Philox) for reproducible, parallel random numbers.

### 8.2 Image processing (`visar.imgproc`)

- Full parity with OpenCV imgproc, including every border mode, interpolation mode (nearest, linear, cubic, area, Lanczos4, plus exact and bit-exact variants), and every `cvtColor` code.
- Additional resamplers: Lanczos-n, Mitchell and other cubics, Magic Kernel Sharp, and anti-aliased downscaling matching Pillow, libvips, PyTorch, and TF `resize` semantics, selectable by flag (important for ML preprocessing parity).
- Separable and non-separable filters with automatic choice of direct, separable, recursive (Deriche, Young–van Vliet), or FFT implementation based on kernel size.
- Morphology with arbitrary structuring elements using van Herk/Gil-Werman for large rectangles.
- Contours: findContours (all modes), approxPolyDP, convex hull, convexity defects, moments, fitEllipse variants, minAreaRect, minEnclosingCircle and triangle.

### 8.3 Codecs (`visar.codecs`)

All codecs are pure Zig, fuzzed, and streaming-capable (decode a region or scanlines without decoding the whole file where the format allows).

| Format | Decode | Encode | Notes |
| --- | --- | --- | --- |
| JPEG (baseline, progressive, lossless, arithmetic, 12-bit) | P0 | P0 | Must match or beat libjpeg-turbo. GPU-assisted batch decode |
| PNG / APNG | P0 | P0 | Must match or beat libspng and fpng |
| BMP, PNM/PFM, TGA, ICO, QOI, GIF | P0 | P0 | |
| TIFF / BigTIFF (all common compressions, tiled, multi-page, float) | P1 | P1 | |
| WebP (lossy and lossless, animated) | P1 | P1 | |
| JPEG XL | P1 | P1 | |
| AVIF (needs AV1) | P1 | P1 | |
| HEIF / HEIC (needs HEVC) | P1 | P1 | |
| OpenEXR, Radiance HDR | P1 | P1 | |
| JPEG 2000 / HTJ2K | P1 | P1 | |
| DNG and camera RAW (major vendors) | P1 | DNG only | Feeds `visar.isp` |
| DICOM, NIfTI, FITS, ENVI, HDF5 | P2 | P2 | Medical, scientific, spectral |
| SVG (rasterize) | P2 | — | Basic profile |

Metadata: EXIF, XMP, ICC profiles, and IPTC are read and preserved on re-encode by default.

### 8.4 Video I/O (`visar.videoio`)

- **Software codecs in pure Zig:** MJPEG, H.264, HEVC, AV1, VP8, VP9, ProRes, FFV1 for decode, plus encoders for H.264, HEVC, AV1, and MJPEG at minimum. Decoders must be competitive with dav1d and FFmpeg's native decoders on the same core count.
- **Hardware codecs** through OS interfaces behind feature flags: VA-API, V4L2 M2M, NVDEC/NVENC (driver API), VideoToolbox, MediaCodec, Media Foundation/D3D12 Video, and Vulkan Video.
- **Containers:** MP4/MOV (fragmented too), MKV/WebM, AVI, MPEG-TS, raw elementary streams, Y4M.
- **Network:** RTSP/RTP, RTMP, HLS, SRT, and WebRTC receive and send, all in-tree, plus HTTP(S) streams with in-tree TLS.
- **Capture:** V4L2 (and the media controller), AVFoundation, Media Foundation, Android Camera2 (NDK), iOS, browser `getUserMedia`, UVC directly over USB where useful.
- **Industrial capture:** GenICam, GigE Vision, USB3 Vision, implemented in pure Zig.
- Frame-accurate seeking, timestamps, and hardware-trigger and multi-camera sync (§8.9).

### 8.5 Features, geometry, calibration, 3D

- Every OpenCV and contrib detector and descriptor (§7), GPU-capable, with batch APIs.
- Learned features via `nn` (SuperPoint/LightGlue/DISK-class) as first-class citizens of the same matcher API.
- USAC robust estimation framework (RANSAC variants, PROSAC, LO-RANSAC, MAGSAC++, GC-RANSAC).
- All calibration patterns (chessboard, circles, ChArUco, AprilGrid) and models (pinhole plus distortion, fisheye/Kannala-Brandt, omnidirectional/Mei, double sphere, unified). Multi-camera rig calibration. Camera–IMU and camera–LiDAR extrinsics. Rolling-shutter models.
- 3D: point-cloud types and I/O (PLY, PCD, LAS/LAZ, E57, OBJ, STL, glTF), voxel grids, normals, ICP (point-to-point, point-to-plane, generalized, colored), global registration (FPFH plus RANSAC), TSDF and voxel-block fusion, marching cubes, Poisson reconstruction, mesh simplification.

### 8.6 Neural networks (`visar.nn`)

**Inference**
- Pure-Zig ONNX importer (in-tree protobuf decoder) covering the full default domain at the latest opset, plus `com.microsoft` contrib ops used by common exported models.
- Additional weight and model formats: safetensors, GGUF, TFLite flatbuffers, and a native VISAR model format (memory-mappable, aligned, quantization-aware).
- Graph optimizer: constant folding, operator fusion (conv+bn+act, attention, layernorm, GELU, …), layout selection (NCHW, NHWC, blocked), memory planning with in-place reuse, dynamic shapes with shape specialization caching.
- Quantization: post-training INT8 and INT4 (per-tensor, per-channel, per-group), FP8, FP16/BF16 mixed precision, calibration toolkit, and weight-only quantization for transformer-style vision models.
- Runs on every backend in §9.2. Falls back per-operation from NPU to GPU to CPU when a device can't run an op.
- Model zoo with reference accuracy for each: classification, detection (YOLO-class, RT-DETR-class), instance and semantic segmentation (incl. SAM-class promptable segmentation), depth estimation, pose and keypoints, face detection and recognition, OCR, tracking, optical flow (RAFT-class), super-resolution, CLIP-class embeddings, and feature matching. Weights are fetched at run time, never bundled.

**Training (`visar.nn.train`)**
- Reverse-mode autograd over the same kernels used for inference, with backward kernels on every GPU backend.
- Modules: conv, linear, attention, norms, activations, pooling, embeddings, and common vision backbones (ResNet, ConvNeXt, ViT, …).
- Optimizers: SGD/momentum, Adam/AdamW, Lion, LAMB, Muon-class. LR schedules. Gradient clipping. EMA.
- Losses for classification, detection (box, GIoU, focal), segmentation (dice, Lovász), metric learning, and distillation.
- Mixed precision with loss scaling, activation checkpointing, gradient accumulation.
- Multi-GPU data parallel on one machine. Multi-node is a later extension.
- Quantization-aware training. Export to ONNX and the VISAR model format.
- `visar.data`: dataset readers (COCO, VOC, YOLO, ImageNet folder, WebDataset-style shards), GPU-side decode and augmentation (geometric, photometric, mosaic, mixup, cutmix, random erasing), deterministic sharded loaders. The goal is that data loading is never the training bottleneck.

### 8.7 Imaging pipeline and color (`visar.isp`, `visar.color`)

- RAW handling: black and white levels, defect pixel correction, lens shading, linearization.
- Demosaicing: bilinear, Malvar-He-Cutler, AHD, VNG, DCB, plus learned demosaicing via `nn`.
- White balance (gray world, white patch, learned), color correction matrices, DNG color pipeline (forward matrices, dual illuminant).
- Denoise (bilateral, NLM, BM3D, learned), sharpening, chromatic aberration correction, geometric lens correction.
- HDR: bracket merge (Debevec, Robertson, Mertens), burst alignment and fusion, local tone mapping.
- Color management: ICC v2 and v4, conversions among sRGB, Display P3, Adobe RGB, Rec.709, Rec.2020, Rec.2100 (PQ and HLG), ACES (AP0/AP1, ACEScg/cct), CIE XYZ, Lab, Luv, LCh, Oklab/Oklch, and HSV/HSL. Chromatic adaptation (Bradford, CAT16). Color difference metrics (ΔE76, ΔE2000).
- An ISP can be described as a graph and tuned. The same pipeline runs on CPU, GPU, or MCU.

### 8.8 SLAM, SfM, and solvers (`visar.slam`, `visar.solve`, `visar.linalg`)

- Nonlinear least squares (Ceres-class): Levenberg–Marquardt and dogleg, auto-differentiation via comptime dual numbers, robust losses, manifolds (SO3/SE3/Sim3 via Lie groups), sparse Schur complement, iterative and direct sparse solvers (sparse Cholesky, sparse QR, PCG).
- Factor graphs with incremental smoothing.
- Bundle adjustment, pose graphs, visual odometry (mono, stereo, RGB-D), visual-inertial odometry, place recognition (bag of words and learned global descriptors), loop closure, and incremental SfM. Multi-view stereo is P2+.

### 8.9 Sensing (`visar.sense`)

- **Depth:** RGB-D alignment and registration, depth filtering (bilateral, temporal, hole fill), depth-to-cloud conversion. Device support via UVC, vendor protocols where documented, and recorded formats.
- **LiDAR:** packet decoders for major spinning and solid-state LiDARs, PCAP replay, motion de-skew, ground segmentation, clustering, range-image projection.
- **IMU:** IMU preintegration, Allan variance calibration, AHRS (Madgwick, Mahony), ESKF/EKF fusion.
- **Time sync:** hardware-trigger and PTP-aware timestamps, cross-sensor clock alignment, multi-camera rigs.
- **Event cameras:** AEDAT and EVT 2.0/3.0 readers, event-stream tensors, accumulation and time surfaces, event-based optical flow, event+frame fusion.
- **Thermal:** radiometric decoding (temperature from raw counts), palettes, thermal/visible registration.
- **Hyperspectral:** ENVI cubes, band math, spectral indices (NDVI and so on), unmixing, dimensionality reduction.
- **Medical:** DICOM (incl. multi-frame and common transfer syntaxes), NIfTI, 3D volume processing (3D filtering, resampling, registration), windowing and leveling, multiplanar reconstruction.
- **Integration:** ROS 2 interop (message conversion and a native node, implementing the DDS/RTPS wire protocol in-tree as a P2 feature), MCAP and rosbag2 readers.

### 8.10 UI and visualization (`visar.ui`)

- `imshow`, `waitKey`, trackbars, mouse callbacks, and named windows across Wayland, X11, Win32, Cocoa, Android, iOS, and the browser. Protocols (Wayland, X11) are implemented directly with no libwayland or libX11.
- Drawing: anti-aliased primitives and text through the in-tree font rasterizer (with a bundled open font), Unicode shaping for common scripts.
- Remote viewer: `visar.ui.serve()` streams windows to a browser (WebSocket plus WebCodecs), ideal for servers, robots, and notebooks.
- Jupyter rich display.
- 3D viewer for point clouds, meshes, cameras, and trajectories.

---

## 9. Platform and accelerator support

### 9.1 Platform tiers

**Tier 1** means built, tested, and benchmarked in CI on every commit. **Tier 2** means built every commit and tested nightly or in emulation. **Tier 3** means it builds, with best-effort support.

| Platform | Architectures | Tier |
| --- | --- | --- |
| Linux (glibc and musl, static) | x86_64, aarch64 | 1 |
| macOS | aarch64, x86_64 | 1 |
| Windows | x86_64, aarch64 | 1 |
| iOS / iPadOS | aarch64 | 1 |
| Android | aarch64, x86_64 | 1 |
| Browser (WASM SIMD + threads), WASI | wasm32, wasm64 | 1 |
| Bare metal (no OS, no heap) | ARM Cortex-M4F/M7/M55 (Helium), RISC-V RV32, Xtensa/ESP32-S3 | 1 for Cortex-M7 and M55, 2 for others |
| Linux | riscv64 (RVV 1.0), armv7, ppc64le, s390x, loongarch64 | 2 |
| FreeBSD, OpenBSD, NetBSD, illumos | x86_64, aarch64 | 2 |
| Fuchsia, Haiku, others Zig supports | — | 3 |

### 9.2 Compute backends

| Backend | Mechanism | Tier |
| --- | --- | --- |
| CPU scalar + `@Vector` SIMD | SSE4.2, AVX2/FMA, AVX-512 (incl. VNNI, BF16, FP16), AMX, NEON, dotprod/i8mm, SVE/SVE2, SME/SME2, RVV 1.0, WASM SIMD128 (+relaxed), Helium/MVE | 1 |
| CPU threads | Work-stealing pool, NUMA-aware, big.LITTLE-aware | 1 |
| Vulkan compute | Loader opened at run time. Kernels compiled from Zig to SPIR-V | 1 |
| Metal | Objective-C runtime calls, no SDK headers. Kernels via in-tree SPIR-V → MSL translation, or direct AIR/MSL generation | 1 |
| CUDA | Driver API via `libcuda` at run time. Kernels compiled from Zig to PTX (nvptx64 target) | 1 |
| ROCm / HIP | HSA runtime at run time. Kernels compiled from Zig to AMDGCN | 2 |
| Intel Level Zero | Run-time loaded. SPIR-V kernels | 2 |
| D3D12 compute | COM via Zig. Kernels via SPIR-V → DXIL translation | 2 |
| WebGPU (browser) | JS glue. Kernels via in-tree SPIR-V → WGSL translation | 1 |
| OpenCL | Legacy and embedded GPUs, SPIR-V or generated OpenCL C | 3 |
| NPUs | Apple Neural Engine (via Core ML at run time), Qualcomm Hexagon, Intel NPU (Level Zero), MediaTek APU, Google Edge TPU, Rockchip RKNPU, Hailo, NVIDIA DLA, Arm Ethos-U (MCU) | 2, as device plugins |

**Multiversioned binaries.** A single VISAR binary contains kernels for several CPU feature levels (for example x86-64-v2, v3, and v4, or ARMv8.0, v8.2+dotprod, and SVE2) and picks the best one at load time. Optional `-Dcpu=native` builds are allowed for maximum performance.

---

## 10. Non-functional requirements

### 10.1 Performance

**Benchmark suite (`visar-bench`)**: public, versioned, reproducible. It covers more than 500 operations at representative sizes (VGA, 1080p, 4K, 8K; u8, u16, f32; 1/3/4 channels) plus end-to-end pipelines (decode → resize → normalize → infer; capture → undistort → detect → track; RAW → ISP → encode; stereo depth; panorama; SLAM sequences).

| Requirement | Target |
| --- | --- |
| Median speedup vs. latest OpenCV (built with all optimizations: IPP, its parallel backend, CUDA for GPU rows) on tier-1 CPUs | **≥ 1.5×** |
| Worst single operation vs. OpenCV | **≥ 0.95×**. A regression is a release blocker |
| GPU operations vs. OpenCV CUDA and NPP on NVIDIA | ≥ 1.5× median |
| JPEG and PNG codec throughput vs. libjpeg-turbo and libspng | ≥ 1.0×, target 1.3× |
| AV1 and HEVC software decode vs. dav1d and FFmpeg native | ≥ 0.9× (stretch 1.0×) on equal threads |
| DNN inference vs. ONNX Runtime (CPU EP) and TensorRT (NVIDIA) on the model zoo | ≥ 1.0× ORT median. ≥ 0.9× TensorRT median |
| Training throughput vs. PyTorch (eager and `torch.compile`) on reference vision models | ≥ 1.0× eager. ≥ 0.9× compiled |
| Pipeline fusion gain on multi-op image pipelines vs. op-by-op | ≥ 2× median |
| Cold start (load library, first `resize` call) | ≤ 5 ms desktop, ≤ 1 ms MCU |
| Per-call overhead on tiny images (eager, CPU) | ≤ 200 ns |

Benchmark hardware includes at minimum: recent Intel and AMD x86 desktop and server parts, Apple M-series, an AWS Graviton-class Arm server, flagship and midrange Android SoCs, an iPhone, Raspberry Pi 5, NVIDIA Jetson, a RISC-V RVV board, Cortex-M7 and M55 boards, ESP32-S3, NVIDIA consumer and datacenter GPUs, AMD RDNA and Instinct GPUs, Intel Arc, and Chrome, Safari, and Firefox. Results are published per release.

### 10.2 Accuracy and correctness

- Every kernel has a documented numeric contract: exact, bit-exact with OpenCV, or within N ULP or a stated absolute or relative tolerance.
- Differential testing against OpenCV (and specialist references such as libjpeg-turbo, NumPy/SciPy, PyTorch, and ONNX Runtime) on random and adversarial inputs, for all types, border modes, and shapes, including odd sizes, 1×1, unaligned, and non-contiguous inputs.
- Cross-backend consistency: CPU, GPU, and NPU results agree within the stated contract.
- Model zoo: published accuracy (mAP, mIoU, top-1, …) within 0.1 points of the reference implementation in FP32, and documented deltas for quantized variants.
- Codec conformance: official conformance suites (JPEG, PNG, AV1 Argon/libaom vectors, HEVC/H.264 JCT-VC streams, TIFF and DICOM samples).

### 10.3 Size and footprint

| Configuration | Budget (stripped, static) |
| --- | --- |
| Minimal: load PNG/JPEG, resize, save | ≤ 250 KB |
| core + imgproc + common codecs | ≤ 2 MB |
| Everything, all CPU backends, no model weights | ≤ 30 MB |
| Python wheel (per platform, everything) | ≤ 15 MB |
| WASM: core + imgproc + JPEG/PNG (gzip) | ≤ 300 KB |
| MCU profile: a fixed comptime pipeline (for example resize + color convert + threshold + contours) | ≤ 64 KB flash, ≤ caller-provided RAM, 0 heap |

### 10.4 Reliability, safety, and security

- Decoders, demuxers, and model and file parsers are fuzzed continuously (coverage-guided, plus structure-aware for ONNX and containers). New parsers ship only after 24+ CPU-hours of clean fuzzing.
- Untrusted-input code paths run with Zig safety checks in release builds. Limits on image dimensions, allocations, and decompression ratio (zip bombs) are configurable and enforced.
- No undefined behavior in safe build modes. Sanitizer-equivalent CI (Debug and ReleaseSafe test runs on all tier-1 targets).
- Security policy, private disclosure process, and CVE handling from day one.
- Real-time profile: bounded latency, no allocation, and no syscalls on the hot path of a prepared pipeline.

### 10.5 Developer experience

- `zig fetch --save` plus two lines in `build.zig` to use VISAR. Feature flags per module and backend.
- `pip install visar`, `npm install visar`, Swift Package Manager, Maven/Gradle (AAR), NuGet, crates.io, and Go modules.
- Docs: an API reference generated from source, task-oriented guides, an "OpenCV → VISAR" migration guide with a complete function mapping, and a browser playground (WASM) where every example runs live.
- `visar` CLI: `info` (devices, CPU features), `convert`, `bench`, `inspect` (images, videos, models), `tune` (run the autotuner and save the cache), `calibrate`, `serve` (remote viewer), `train`.
- Errors carry context: the operation, argument, expected versus actual, and a hint.

---

## 11. Architecture overview

```
┌───────────────────────────────────────────────────────────────────────┐
│ Bindings: Python · JS/WASM · Swift · Kotlin · Rust · Go · C# · C++    │
│ Compat shims: visar.cv2 (Python) · visar/cv.hpp (C++)                  │
├───────────────────────────────────────────────────────────────────────┤
│ Stable C ABI (visar.h) — opaque handles, versioned structs            │
├───────────────────────────────────────────────────────────────────────┤
│ Domain modules: imgproc · codecs · videoio · features · calib · 3d ·  │
│ detect · nn · nn.train · ml · photo · stitch · isp · color · slam ·   │
│ sense.* · ui · …                                                      │
├───────────────────────────────────────────────────────────────────────┤
│ Graph layer: eager ops ↔ lazy graphs ↔ comptime pipelines             │
│ fusion · tiling · layout · memory planning · device placement ·       │
│ autotuning cache                                                      │
├───────────────────────────────────────────────────────────────────────┤
│ Kernel layer: single-source Zig kernels (map, stencil, reduce, scan,  │
│ gather/scatter, GEMM/conv, sort, histogram, …)                        │
├───────────────────────────────────────────────────────────────────────┤
│ Codegen targets: CPU @Vector (multiversioned) · SPIR-V · PTX ·        │
│ AMDGCN · MSL/AIR · WGSL · DXIL — in-tree translators where needed     │
├───────────────────────────────────────────────────────────────────────┤
│ Runtime: allocators · executors · device queues · events · external   │
│ memory · OS interfaces (syscalls, dlopen, objc runtime)               │
└───────────────────────────────────────────────────────────────────────┘
```

**Key decisions**

1. **Single-source kernels.** Kernels are written against a small set of parallel primitives (map, stencil, reduce, scan, tiled GEMM, …), parameterized at comptime. The CPU path lowers to `@Vector` loops plus the thread pool. GPU paths compile the same Zig through Zig's SPIR-V, nvptx, and amdgcn targets. Where Zig has no direct target (MSL, WGSL, DXIL), VISAR ships its own SPIR-V translators in Zig. This is a major engineering item and a top risk (§15).
2. **Graph as a first-class citizen.** Eager calls are graphs of one node, so there is one code path. Fusion and tiling decisions come from cost models refined by autotuning.
3. **Comptime specialization** removes dispatch for fixed pipelines. This is what makes the MCU profile and the ≤200 ns call overhead possible.
4. **Runtime-loaded device APIs** keep the build dependency-free (§6).
5. **The C ABI is the stability boundary.** The Zig API follows Zig releases. The C ABI follows semver with versioned structs and no breaking changes within a major version.

**Example: Zig**

```zig
const v = @import("visar");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var img = try v.codecs.read(gpa, init.io, "in.jpg", .{});
    defer img.deinit(gpa);

    // Lazy graph: resize + color convert + blur fused into one pass, on the best device.
    var g = v.Graph.init(gpa);
    defer g.deinit();
    const out = g.input(img)
        .resize(.{ .w = 640, .h = 640 }, .area)
        .cvtColor(.bgr_to_rgb)
        .gaussianBlur(.{ .ksize = 5, .sigma = 1.2 });
    var result = try g.run(out, .{ .device = .auto });
    defer result.deinit(gpa);

    try v.codecs.write(init.io, "out.png", result, .{});
}
```

**Example: Python**

```python
import visar as v
img = v.imread("in.jpg")
out = img.resize((640, 640), interp="area").cvt("bgr2rgb").gaussian_blur(5, 1.2)
v.imwrite("out.png", out)

import visar.cv2 as cv2   # drop-in for existing code
cv2.imwrite("out.png", cv2.GaussianBlur(cv2.resize(cv2.imread("in.jpg"), (640, 640)), (5, 5), 1.2))
```

---

## 12. Bindings and compatibility

- **API description:** the public API is described once (Zig declarations plus annotations). Every binding, the C header, docs stubs, and the shims are generated from it, so no binding is maintained by hand.
- **C ABI (`visar.h`):** opaque handles, explicit allocators (or a default), error codes plus a thread-local error message, and versioned option structs.
- **Python:** built directly on the CPython stable ABI (abi3) from Zig, with no pybind or nanobind. One wheel per platform covers all supported Python versions. Free-threaded CPython is supported. NumPy and DLPack are zero-copy. Type stubs are included.
- **`visar.cv2` shim:** the goal is that existing OpenCV Python code runs unchanged. Success is measured by running OpenCV's own Python samples and tutorials, and a corpus of popular open-source projects that use `cv2`.
- **C++ shim (`visar/cv.hpp`):** a header-only `cv::Mat`-style wrapper over the C ABI covering the most-used API surface.
- **JS/WASM:** ESM package with TypeScript types. Works with `ImageData`, `VideoFrame`, `OffscreenCanvas`, and WebGPU buffers. Runs in Workers.
- **Swift, Kotlin, Rust, Go, C#:** idiomatic wrappers generated over the C ABI, published to each ecosystem's package manager.

---

## 13. Quality, testing, and CI

- Unit tests per kernel and type, differential tests against OpenCV and other references (§10.2), property-based tests, and golden-image tests with perceptual diff.
- Cross-compilation of every tier-1 and tier-2 target on every commit (Zig makes this cheap). Execution under QEMU for ISAs without hardware in CI (RVV, SVE2, SME, big-endian s390x).
- A hardware lab running the benchmark suite nightly. A performance regression above 3% on any tracked operation fails CI unless explicitly accepted.
- Fuzzing runs continuously, with a corpus checked in.
- Size budgets (§10.3) are checked in CI.
- The model zoo accuracy suite runs nightly on every backend.

---

## 14. Roadmap

Milestones are ordered by dependency, not by date. Each one has exit criteria. Work inside a milestone can proceed in parallel.

| Milestone | Contents | Exit criteria |
| --- | --- | --- |
| **M0 — Foundations** | Tensor/Image types, allocators, executors, thread pool, `@Vector` kernel primitives, multiversioning, C ABI skeleton, test and bench harness, differential testing against OpenCV, CI matrix | Cross-compiles to all tier-1 targets. Harness runs. 20 core ops beat OpenCV median 1.5× |
| **M1 — Core imaging** | imgproc parity, P0 codecs (JPEG, PNG, BMP, PNM, GIF, QOI, TGA), eager and lazy graph modes, fusion, Python package, `visar.cv2` shim v1, WASM build | imgproc and codecs parity tables complete. Perf gates green. Wheel ≤ 15 MB. Top-100 most-used cv2 functions work via shim |
| **M2 — GPUs** | Single-source kernel compiler to SPIR-V, PTX, MSL, and WGSL. Vulkan, Metal, CUDA, and WebGPU runtimes. Autotuner and tuning cache. External memory interop | All M1 ops run on all four GPU backends within contract. GPU ≥ 1.5× OpenCV CUDA median |
| **M3 — Geometry and vision** | features2d, calib3d, USAC, objdetect (incl. markers, QR, barcodes), video analysis (flow, background, trackers), stereo, ml, photo, stitching | Parity tables complete. OpenCV calibration and feature samples run via shim |
| **M4 — Video and UI** | Containers, software codecs (MJPEG, H.264, HEVC, AV1, VP8/9), hardware codec interfaces, capture on all OSes, network streams, `visar.ui` incl. remote viewer, remaining codecs (TIFF, WebP, JXL, AVIF, HEIF, EXR, J2K) | Codec conformance suites pass. Decode perf targets met |
| **M5 — Recognition** | ONNX importer, graph optimizer, quantization, all backends incl. NPU plugin interface, model zoo, DNN-based modules (text, face, trackers, super-resolution) | ORT and TensorRT perf targets. Zoo accuracy within tolerance |
| **M6 — Training** | Autograd, backward kernels on all GPU backends, optimizers, losses, mixed precision, multi-GPU data parallel, `visar.data` augmentation, QAT, ONNX export | Train reference models to published accuracy. PyTorch throughput targets |
| **M7 — 3D, SLAM, solvers** | `visar.linalg` sparse, `visar.solve` (LM, manifolds, autodiff), `visar.three`, `visar.slam` (BA, VO/VIO, loop closure, SfM) | Standard dataset benchmarks (e.g. EuRoC, TUM, KITTI) within published accuracy of reference systems |
| **M8 — Sensing and ISP** | `visar.isp`, `visar.color`, depth, LiDAR, IMU, sync, event, thermal, hyperspectral, medical, ROS 2 interop | Each sensing module has an end-to-end sample on real sensor data |
| **M9 — Everywhere** | MCU tier 1 (static pipelines, Helium, Ethos-U), mobile packages, remaining backends (ROCm, Level Zero, D3D12), long-tail contrib modules, all bindings | Every row of §7 is "parity". All size budgets met |
| **1.0** | API freeze for the C ABI. Docs and migration guide complete. Security review | All §10 targets green on tier 1. Shim passes the OpenCV samples corpus at ≥ 95% |

---

## 15. Risks and mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Scope is enormous (OpenCV is decades of work, plus codecs, DNN, training, SLAM) | Never ships | Strict dependency-ordered milestones. Each milestone is independently useful and released. The module matrix is the single source of truth for progress |
| Zig is pre-1.0. Language and std churn | Breakage, maintenance load | Pin a Zig version per VISAR release. CI against Zig master. The C ABI shields users. Keep std surface usage narrow |
| GPU codegen from Zig is immature (SPIR-V/PTX/AMDGCN backends, no MSL/WGSL/DXIL targets) | GPU story delayed | Start with a restricted kernel subset and grow it. Build the SPIR-V → MSL/WGSL translators early. Upstream Zig fixes. Keep a fallback of committed precompiled SPIR-V (as in ditch) while the in-tree path matures |
| Matching specialist libraries (libjpeg-turbo, dav1d, cuDNN-level conv, TensorRT) in pure Zig | Perf targets missed in some domains | Specialist-relative targets (§10.1) instead of blanket ones. Autotuning. Hand-specialized kernels allowed where measured |
| Patent-encumbered codecs and algorithms (H.264, HEVC; check SURF status) | Distribution risk for some users | Compile-time flags per codec. Royalty-free codecs (AV1, VP9, JPEG XL) on by default. Encumbered ones documented. Final decision is an open question |
| Video and image decoders are attack surface | Security incidents | Memory-safe Zig, safety checks on, continuous fuzzing, limits, security process (§10.4) |
| NPU vendor stacks are proprietary and fragmented | Uneven NPU coverage | Plugin interface. GPU/CPU fallback per op. Start with the most open stacks |
| Compatibility shim expectations ("it should behave exactly like OpenCV") | User frustration | Documented deviations. Shim conformance tests public. Quirk flags where needed |
| Accuracy drift between backends | Hard-to-debug differences | Numeric contracts, cross-backend CI, determinism mode |

---

## 16. Success metrics

- **Performance:** §10.1 targets green on every tier-1 platform, published per release.
- **Migration:** ≥ 95% of OpenCV's Python samples, and ≥ 80% of a corpus of the 500 most-starred open-source projects using `cv2`, run unchanged via `visar.cv2` with matching outputs.
- **Footprint:** every §10.3 budget met.
- **Correctness:** zero open accuracy-contract violations at release. Model zoo within tolerance.
- **Security:** zero unresolved critical or high CVEs. Continuous fuzzing coverage on every parser.
- **Adoption:** downloads across PyPI, npm, and the Zig package index. Projects migrating from OpenCV. External contributors per milestone.

---

## 17. Reusing work from ditch

`plyght/ditch` (same author, Zig 0.16) already proves out several pieces VISAR needs:

- **Runtime-loaded Vulkan backend** with committed, checksummed SPIR-V and GLSL sources (`src/vulkan/`). This is the bootstrap path for VISAR's Vulkan backend before Zig → SPIR-V kernels mature.
- **Metal backend** (`src/metal/`). VISAR should replace the SDK-dependent Objective-C shim with direct `objc_msgSend` calls to meet §6.
- **CPU kernel tuning:** target-derived vector width and register-tile shape (`src/tensor.zig`), build options to override them, and per-ISA release builds. VISAR generalizes this into multiversioned binaries.
- **Persistent fork-join thread pool** with futex sleep and reusable scratch (`tensor.zig: Pool`). A starting point for the VISAR executor, extended with work stealing and NUMA awareness.
- **Device selection and fallback** (`src/compute.zig`: `Device`, `select`, minimum-work thresholds). The same idea is needed for VISAR placement.
- **Quantized formats and loaders:** FP8, MXFP4, INT4 dequantization, safetensors, GGUF. Feeds `visar.nn` model loading.
- **TPE optimizer** used in ditch's search. It can drive VISAR's kernel autotuner.

ditch has no autograd or training loop, so `visar.nn.train` (M6) is new work.

---

## 18. Open questions

1. **License for VISAR.** Not decided yet. Needed before external contributions.
2. **Encumbered codecs and algorithms:** ship H.264/HEVC encoders and decoders on by default, off by default, or as a separate package?
3. **Shim fidelity:** which OpenCV quirks are worth reproducing (e.g. rounding in `resize` `INTER_LINEAR` for u8, `cvtColor` fixed-point paths)? Proposal: bit-exact mode for the top 50 most-used functions, contract mode elsewhere.
4. **Governance:** BDFL, core team, or foundation? RFC process for API changes?
5. **Native model format:** design it now, or adopt safetensors plus a graph sidecar?
6. **Python package name** availability (`visar` on PyPI and npm), and whether the shim is `visar.cv2` or a separate `visar-cv2` distribution.
7. **Minimum Zig version policy:** track the latest Zig release only, or support the last N?
8. **Model zoo weights:** which pretrained weights can be redistributed or fetched, and under what terms per model?

---

## Appendix A: Decisions log (from the initial Q&A)

| Topic | Decision |
| --- | --- |
| Target users | Everyone: Zig devs, C ABI users, Python/ML, embedded/edge |
| API strategy | Clean-slate idiomatic API plus a `cv2`/`cv::` compatibility shim |
| v1.0 scope | Everything: all OpenCV main and contrib modules plus beyond-OpenCV areas |
| Dependencies | Zero, pure Zig. OS and driver interfaces only at run time, behind feature flags |
| Acceleration | CPU SIMD and threads, Vulkan, Metal, CUDA, NPUs, WebGPU, and more |
| Platforms | Desktop and server, mobile, WASM, bare-metal MCU |
| Perf bar | ≥ 1.5× OpenCV median, no op worse than 0.95× |
| License | Not decided |
| Team and timeline | Not a constraint. The roadmap is dependency-ordered, not date-based |
| DNN | Own ONNX runtime plus training |
| Sensing scope | All: depth and point clouds, RAW/ISP, IMU fusion, exotic imaging |
| PRD extras | Whatever is necessary: architecture, roadmap, module matrix, risks, and open questions all included |

## Appendix B: Glossary

- **ISP:** Image Signal Processor, the pipeline that turns raw sensor data into a viewable image.
- **VIO:** Visual-Inertial Odometry.
- **TSDF:** Truncated Signed Distance Function, a volumetric representation used for 3D fusion.
- **USAC:** Universal Sample Consensus, OpenCV's unified robust-estimation framework.
- **ULP:** Unit in the Last Place, a measure of floating-point error.
- **DLPack:** a standard for zero-copy tensor exchange between frameworks.
- **Multiversioning:** shipping several CPU-feature-specific versions of a kernel in one binary and picking one at run time.
