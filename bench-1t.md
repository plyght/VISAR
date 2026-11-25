VISAR 0.1.0 vs OpenCV 5.0.0, 1920x1080, 1 thread(s), x86_64-linux (avx2)

| Operation | VISAR (ms) | OpenCV (ms) | Speedup |
| --- | ---: | ---: | ---: |
| cvtColor BGR2GRAY | 0.548 | 0.776 | 1.42x |
| cvtColor BGR2RGB | 0.231 | 0.221 | 0.96x |
| cvtColor BGR2BGRA | 0.234 | 0.323 | 1.38x |
| cvtColor BGR2HSV | 2.129 | 3.074 | 1.44x |
| cvtColor BGR2YCrCb | 1.269 | 1.696 | 1.34x |
| resize linear 1/2 | 0.363 | 0.630 | 1.74x |
| resize linear 2x | 4.209 | 9.556 | 2.27x |
| resize linear 0.75 | 1.350 | 2.115 | 1.57x |
| resize area 1/2 | 0.361 | 0.628 | 1.74x |
| resize area 640x480->416x312 c1 | 0.288 | 0.791 | 2.75x |
| resize area 640x480->416x312 c3 | 0.399 | 1.970 | 4.94x |
| resize area 640x480->416x312 c4 | 0.433 | 2.352 | 5.43x |
| resize area 1920x1080->1280x720 c3 | 2.333 | 9.234 | 3.96x |
| resize area 1920x1080->640x360 c3 | 0.963 | 3.192 | 3.31x |
| resize area 1280x720->320x180 c3 | 0.349 | 1.141 | 3.27x |
| resize area 97x61->53x37 c3 | 0.015 | 0.039 | 2.56x |
| resize nearest 2x | 1.265 | 8.701 | 6.88x |
| resize cubic 2x | 9.885 | 12.768 | 1.29x |
| GaussianBlur 3x3 | 0.639 | 1.182 | 1.85x |
| GaussianBlur 5x5 | 1.009 | 2.418 | 2.40x |
| GaussianBlur 7x7 | 3.977 | 4.121 | 1.04x |
| GaussianBlur 15x15 | 7.472 | 8.280 | 1.11x |
| GaussianBlur 31x31 | 14.764 | 19.624 | 1.33x |
| GaussianBlur s=3 | 9.563 | 10.995 | 1.15x |
| blur 3x3 | 1.262 | 1.690 | 1.34x |
| blur 15x15 | 2.644 | 3.162 | 1.20x |
| Sobel dx 3x3 ->i16 | 0.311 | 0.583 | 1.87x |
| threshold binary | 0.228 | 0.219 | 0.96x |
| threshold otsu | 0.825 | 0.970 | 1.18x |
| erode 3x3 rect | 0.382 | 0.483 | 1.26x |
| dilate 7x7 ellipse | 1.173 | 1.924 | 1.64x |
| flip horizontal | 0.265 | 0.251 | 0.95x |
| rotate 90cw | 0.903 | 0.927 | 1.03x |
| warpAffine linear | 5.945 | 11.382 | 1.91x |
| equalizeHist | 1.486 | 1.737 | 1.17x |
| add u8 | 0.343 | 0.331 | 0.97x |
| absdiff u8 | 0.344 | 0.331 | 0.97x |
| addWeighted u8 | 1.074 | 1.551 | 1.44x |
| GaussianBlur 5x5 f32 | 2.906 | 3.877 | 1.33x |
| pyrDown u8 | 0.632 | 1.803 | 2.85x |
| pyrDown u8 gray | 0.182 | 0.399 | 2.20x |
| pyrUp u8 gray | 0.881 | 3.409 | 3.87x |
| pyrDown f32 | 1.757 | 1.856 | 1.06x |
| cornerMinEigenVal 3/3 | 6.139 | 9.011 | 1.47x |
| cornerHarris 2/3 | 5.603 | 8.088 | 1.44x |
| goodFeaturesToTrack 1000 | 10.068 | 16.528 | 1.64x |
| calcOpticalFlowPyrLK 1000pt | 7.639 | 14.382 | 1.88x |
| imdecode PNG | 24.213 | 30.920 | 1.28x |
| imencode PNG | 31.612 | 60.660 | 1.92x |
| imdecode JPEG | 8.102 | 8.914 | 1.10x |
| imencode JPEG q90 | 7.290 | 6.903 | 0.95x |
| findContours TREE simple | 3.568 | 5.999 | 1.68x |
| findContours EXT none | 1.875 | 2.738 | 1.46x |
| moments binary | 0.791 | 3.026 | 3.82x |
| connectedComponents stats | 1.893 | 15.030 | 7.94x |
| contour moments | 0.088 | 0.215 | 2.45x |
| convexHull 20k pts | 0.254 | 1.574 | 6.21x |
| approxPolyDP | 0.702 | 1.638 | 2.33x |
| VideoWriter MJPG 16f vs FFmpeg | 147.797 | 336.928 | 2.28x |
| VideoWriter MJPG 16f vs OpenCV MJPEG | 147.934 | 532.408 | 3.60x |
| VideoCapture MJPG 16f vs FFmpeg | 153.619 | 204.658 | 1.33x |
| VideoCapture MJPG 16f vs OpenCV MJPEG | 153.951 | 189.843 | 1.23x |
| NanoDet 416 net | 15.657 | 24.750 | 1.58x |
| YOLOX-S 640 net | 344.297 | 375.028 | 1.09x |
| FaceDetectorYN 320x240 | 1.989 | 4.495 | 2.26x |
| FaceDetectorYN 640x480 | 7.403 | 16.274 | 2.20x |
| face prep 4K BGRA->640x360 | 2.137 | 10.674 | 5.00x |
| face gray 4K BGRA /6 | 1.947 | 6.343 | 3.26x |
| face prep+gray 4K BGRA | 2.373 | 17.174 | 7.24x |
| face pipeline 4K->640x360 | 7.954 | 23.997 | 3.02x |
| motion mask (VISAR running average) vs MOG2 720p | 0.293 | 11.835 | 40.37x |
| motion mask (VISAR running average) vs KNN 720p | 0.301 | 19.349 | 64.30x |
| motion mask (VISAR running average) vs accumulateWeighted+absdiff 720p | 0.258 | 0.250 | 0.97x |
| track pipeline 720p vs OpenCV (accumulateWeighted) | 1.593 | 9.515 | 5.97x |
| track pipeline 720p vs OpenCV (MOG2) | 1.610 | 20.903 | 12.98x |
| FAST-9 t20 nms 640x480 | 0.711 | 1.723 | 2.42x |
| ORB 500 detect+compute 640x480 | 3.720 | 6.458 | 1.74x |
| ORB 1000 detect+compute 640x480 | 4.109 | 7.208 | 1.75x |
| FAST-9 t20 nms 1280x720 | 2.182 | 5.727 | 2.62x |
| ORB 500 detect+compute 1280x720 | 10.201 | 16.998 | 1.67x |
| ORB 1000 detect+compute 1280x720 | 10.616 | 17.973 | 1.69x |
| BFMatcher Hamming match 1000x1000 | 1.601 | 8.445 | 5.28x |
| BFMatcher Hamming knn2 1000x1000 | 2.125 | 8.581 | 4.04x |
| BFMatcher Hamming crossCheck 1000x1000 | 3.169 | 16.803 | 5.30x |
| BFMatcher L2 f32x128 knn2 1000x1000 | 7.026 | 23.568 | 3.35x |
| findHomography RANSAC 500pt 30% out | 0.093 | 0.576 | 6.16x |
| findHomography LMEDS 500pt 30% out | 0.562 | 1.197 | 2.13x |
| estimateAffine2D RANSAC 500pt | 0.046 | 0.084 | 1.82x |
| findFundamentalMat RANSAC 300pt 30% out | 0.219 | 0.811 | 3.71x |
| solvePnP ITERATIVE 100pt | 0.082 | 0.163 | 1.99x |
| solvePnP EPNP 100pt | 0.068 | 0.063 | 0.93x |
| solvePnPRansac 100pt 30% out | 1.233 | 1.750 | 1.42x |
| projectPoints 1000pt | 0.048 | 0.182 | 3.83x |
| findChessboardCorners 9x6 1280x720 | 0.800 | 1.077 | 1.35x |
| cornerSubPix 54pt 11x11 | 0.146 | 0.330 | 2.26x |
| undistort 1280x720 BGR | 5.322 | 7.660 | 1.44x |
| undistort 1280x720 gray | 3.766 | 6.380 | 1.69x |
| initUndistortRectifyMap 1280x720 | 3.148 | 3.134 | 1.00x |
| remap linear 1280x720 BGR | 3.140 | 7.144 | 2.28x |
| calibrateCamera 10 views 9x6 | 3.204 | 12.896 | 4.03x |
| fastNlMeansDenoising h10 t7 s21 640x480 | 78.479 | 226.751 | 2.89x |
| fastNlMeansDenoisingColored h10 640x480 | 193.355 | 541.956 | 2.80x |
| inpaint TELEA r3 5% BGR 640x480 | 8.839 | 19.761 | 2.24x |
| inpaint TELEA r3 5% gray 640x480 | 6.843 | 11.896 | 1.74x |
| inpaint NS r3 5% BGR 640x480 | 7.456 | 15.102 | 2.03x |
| inpaint NS r3 5% gray 640x480 | 4.303 | 9.061 | 2.11x |
| edgePreservingFilter RECURS 640x480 | 13.679 | 44.349 | 3.24x |
| edgePreservingFilter NORMCONV 640x480 | 50.200 | 143.453 | 2.86x |
| stylization 640x480 | 51.144 | 145.501 | 2.84x |
| pencilSketch 640x480 | 17.589 | 48.691 | 2.77x |
| bilateralFilter d9 s75 BGR 640x480 | 14.322 | 26.148 | 1.83x |
| bilateralFilter d9 s75 gray 640x480 | 9.685 | 8.611 | 0.89x |
| fastNlMeansDenoising h10 t7 s21 1280x720 | 239.467 | 661.853 | 2.76x |
| fastNlMeansDenoisingColored h10 1280x720 | 571.874 | 1573.301 | 2.75x |
| inpaint TELEA r3 5% BGR 1280x720 | 25.400 | 55.607 | 2.19x |
| inpaint TELEA r3 5% gray 1280x720 | 19.231 | 32.396 | 1.68x |
| inpaint NS r3 5% BGR 1280x720 | 22.640 | 44.660 | 1.97x |
| inpaint NS r3 5% gray 1280x720 | 12.995 | 26.546 | 2.04x |
| edgePreservingFilter RECURS 1280x720 | 48.327 | 140.612 | 2.91x |
| edgePreservingFilter NORMCONV 1280x720 | 161.124 | 438.850 | 2.72x |
| stylization 1280x720 | 161.741 | 447.540 | 2.77x |
| pencilSketch 1280x720 | 56.211 | 141.708 | 2.52x |
| bilateralFilter d9 s75 BGR 1280x720 | 43.113 | 78.604 | 1.82x |
| bilateralFilter d9 s75 gray 1280x720 | 29.099 | 25.560 | 0.88x |
| QR detect+decode 640x480 | 0.824 | 13.622 | 16.53x |
| QR detect+decode 1280x720 | 1.909 | 47.351 | 24.80x |
| QR encode v10 | 0.452 | 29.208 | 64.64x |
| ArUco detect 20 markers 1280x720 | 8.261 | 26.157 | 3.17x |
| ArUco detect SUBPIX 1280x720 | 8.248 | 26.329 | 3.19x |

Median speedup **2.03x**, worst 0.88x.
