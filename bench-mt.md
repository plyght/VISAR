VISAR 0.1.0 vs OpenCV 5.0.0, 1920x1080, 4 thread(s), x86_64-linux (avx2)

| Operation | VISAR (ms) | OpenCV (ms) | Speedup |
| --- | ---: | ---: | ---: |
| cvtColor BGR2GRAY | 0.290 | 0.337 | 1.16x |
| cvtColor BGR2RGB | 0.301 | 0.097 | 0.32x |
| cvtColor BGR2BGRA | 0.253 | 0.183 | 0.73x |
| cvtColor BGR2HSV | 0.907 | 1.379 | 1.52x |
| cvtColor BGR2YCrCb | 0.581 | 0.645 | 1.11x |
| resize linear 1/2 | 0.201 | 0.337 | 1.68x |
| resize linear 2x | 2.143 | 5.018 | 2.34x |
| resize linear 0.75 | 0.714 | 1.116 | 1.56x |
| resize area 1/2 | 0.200 | 0.336 | 1.68x |
| resize area 640x480->416x312 c1 | 0.175 | 0.672 | 3.84x |
| resize area 640x480->416x312 c3 | 0.422 | 1.327 | 3.14x |
| resize area 640x480->416x312 c4 | 0.258 | 1.573 | 6.10x |
| resize area 1920x1080->1280x720 c3 | 1.145 | 4.941 | 4.32x |
| resize area 1920x1080->640x360 c3 | 0.527 | 1.729 | 3.28x |
| resize area 1280x720->320x180 c3 | 0.227 | 1.213 | 5.34x |
| resize area 97x61->53x37 c3 | 0.018 | 0.060 | 3.26x |
| resize nearest 2x | 0.539 | 4.773 | 8.85x |
| resize cubic 2x | 5.283 | 6.813 | 1.29x |
| GaussianBlur 3x3 | 0.392 | 0.670 | 1.71x |
| GaussianBlur 5x5 | 0.535 | 1.506 | 2.82x |
| GaussianBlur 7x7 | 1.895 | 2.267 | 1.20x |
| GaussianBlur 15x15 | 3.621 | 4.570 | 1.26x |
| GaussianBlur 31x31 | 7.215 | 11.323 | 1.57x |
| GaussianBlur s=3 | 4.548 | 6.247 | 1.37x |
| blur 3x3 | 0.662 | 1.716 | 2.59x |
| blur 15x15 | 1.288 | 3.188 | 2.48x |
| Sobel dx 3x3 ->i16 | 0.188 | 0.564 | 3.00x |
| threshold binary | 0.103 | 0.127 | 1.23x |
| threshold otsu | 0.391 | 1.107 | 2.83x |
| erode 3x3 rect | 0.263 | 0.419 | 1.59x |
| dilate 7x7 ellipse | 0.641 | 1.134 | 1.77x |
| flip horizontal | 0.141 | 0.351 | 2.49x |
| rotate 90cw | 0.451 | 0.929 | 2.06x |
| warpAffine linear | 2.389 | 4.717 | 1.97x |
| equalizeHist | 0.677 | 0.825 | 1.22x |
| add u8 | 0.145 | 0.246 | 1.70x |
| absdiff u8 | 0.145 | 0.249 | 1.72x |
| addWeighted u8 | 0.522 | 1.608 | 3.08x |
| GaussianBlur 5x5 f32 | 1.406 | 2.301 | 1.64x |
| pyrDown u8 | 0.328 | 1.062 | 3.23x |
| pyrDown u8 gray | 0.110 | 0.351 | 3.21x |
| pyrUp u8 gray | 0.442 | 3.465 | 7.84x |
| pyrDown f32 | 0.843 | 0.912 | 1.08x |
| cornerMinEigenVal 3/3 | 2.936 | 8.918 | 3.04x |
| cornerHarris 2/3 | 2.541 | 8.080 | 3.18x |
| goodFeaturesToTrack 1000 | 6.872 | 16.799 | 2.44x |
| calcOpticalFlowPyrLK 1000pt | 3.159 | 6.304 | 2.00x |
| imdecode PNG | 24.175 | 30.915 | 1.28x |
| imencode PNG | 31.682 | 60.437 | 1.91x |
| imdecode JPEG | 8.043 | 8.851 | 1.10x |
| imencode JPEG q90 | 7.280 | 6.907 | 0.95x |
| findContours TREE simple | 3.545 | 6.341 | 1.79x |
| findContours EXT none | 1.880 | 2.834 | 1.51x |
| moments binary | 0.422 | 3.097 | 7.34x |
| connectedComponents stats | 0.878 | 6.208 | 7.07x |
| contour moments | 0.088 | 0.216 | 2.45x |
| convexHull 20k pts | 0.254 | 1.571 | 6.18x |
| approxPolyDP | 0.701 | 1.638 | 2.34x |
| VideoWriter MJPG 16f vs FFmpeg | 147.472 | 336.322 | 2.28x |
| VideoWriter MJPG 16f vs OpenCV MJPEG | 149.213 | 283.121 | 1.90x |
| VideoCapture MJPG 16f vs FFmpeg | 153.896 | 208.142 | 1.35x |
| VideoCapture MJPG 16f vs OpenCV MJPEG | 154.076 | 191.674 | 1.24x |
| NanoDet 416 net | 9.738 | 14.789 | 1.52x |
| YOLOX-S 640 net | 171.160 | 188.982 | 1.10x |
| FaceDetectorYN 320x240 | 1.161 | 2.822 | 2.43x |
| FaceDetectorYN 640x480 | 4.157 | 9.413 | 2.26x |
| face prep 4K BGRA->640x360 | 1.203 | 5.267 | 4.38x |
| face gray 4K BGRA /6 | 0.896 | 2.908 | 3.24x |
| face prep+gray 4K BGRA | 1.313 | 8.267 | 6.30x |
| face pipeline 4K->640x360 | 4.497 | 13.503 | 3.00x |
| motion mask (VISAR running average) vs MOG2 720p | 0.452 | 6.626 | 14.65x |
| motion mask (VISAR running average) vs KNN 720p | 0.188 | 7.685 | 40.86x |
| motion mask (VISAR running average) vs accumulateWeighted+absdiff 720p | 0.148 | 0.326 | 2.21x |
| track pipeline 720p vs OpenCV (accumulateWeighted) | 0.954 | 5.347 | 5.60x |
| track pipeline 720p vs OpenCV (MOG2) | 0.988 | 11.420 | 11.56x |
| FAST-9 t20 nms 640x480 | 0.339 | 1.744 | 5.14x |
| ORB 500 detect+compute 640x480 | 1.958 | 6.745 | 3.45x |
| ORB 1000 detect+compute 640x480 | 2.102 | 7.467 | 3.55x |
| FAST-9 t20 nms 1280x720 | 0.921 | 5.684 | 6.17x |
| ORB 500 detect+compute 1280x720 | 5.433 | 16.786 | 3.09x |
| ORB 1000 detect+compute 1280x720 | 5.628 | 17.618 | 3.13x |
| BFMatcher Hamming match 1000x1000 | 0.803 | 4.358 | 5.43x |
| BFMatcher Hamming knn2 1000x1000 | 0.995 | 4.540 | 4.56x |
| BFMatcher Hamming crossCheck 1000x1000 | 1.585 | 8.510 | 5.37x |
| BFMatcher L2 f32x128 knn2 1000x1000 | 3.415 | 9.139 | 2.68x |
| findHomography RANSAC 500pt 30% out | 0.093 | 0.574 | 6.17x |
| findHomography LMEDS 500pt 30% out | 0.561 | 1.199 | 2.14x |
| estimateAffine2D RANSAC 500pt | 0.046 | 0.084 | 1.82x |
| findFundamentalMat RANSAC 300pt 30% out | 0.218 | 0.803 | 3.68x |
| solvePnP ITERATIVE 100pt | 0.083 | 0.163 | 1.97x |
| solvePnP EPNP 100pt | 0.067 | 0.063 | 0.93x |
| solvePnPRansac 100pt 30% out | 1.233 | 1.749 | 1.42x |
| projectPoints 1000pt | 0.048 | 0.181 | 3.81x |
| findChessboardCorners 9x6 1280x720 | 0.655 | 1.445 | 2.21x |
| cornerSubPix 54pt 11x11 | 0.147 | 0.329 | 2.24x |
| undistort 1280x720 BGR | 2.134 | 9.005 | 4.22x |
| undistort 1280x720 gray | 1.570 | 7.570 | 4.82x |
| initUndistortRectifyMap 1280x720 | 1.353 | 1.324 | 0.98x |
| remap linear 1280x720 BGR | 1.167 | 2.949 | 2.53x |
| calibrateCamera 10 views 9x6 | 3.149 | 10.236 | 3.25x |
| fastNlMeansDenoising h10 t7 s21 640x480 | 40.878 | 114.487 | 2.80x |
| fastNlMeansDenoisingColored h10 640x480 | 98.860 | 272.356 | 2.75x |
| inpaint TELEA r3 5% BGR 640x480 | 8.912 | 19.828 | 2.22x |
| inpaint TELEA r3 5% gray 640x480 | 6.770 | 11.900 | 1.76x |
| inpaint NS r3 5% BGR 640x480 | 7.398 | 15.093 | 2.04x |
| inpaint NS r3 5% gray 640x480 | 4.236 | 9.042 | 2.13x |
| edgePreservingFilter RECURS 640x480 | 13.618 | 43.190 | 3.17x |
| edgePreservingFilter NORMCONV 640x480 | 49.607 | 140.659 | 2.84x |
| stylization 640x480 | 50.511 | 146.479 | 2.90x |
| pencilSketch 640x480 | 17.624 | 46.205 | 2.62x |
| bilateralFilter d9 s75 BGR 640x480 | 7.023 | 12.446 | 1.77x |
| bilateralFilter d9 s75 gray 640x480 | 4.798 | 4.098 | 0.85x |
| fastNlMeansDenoising h10 t7 s21 1280x720 | 120.988 | 322.032 | 2.66x |
| fastNlMeansDenoisingColored h10 1280x720 | 287.429 | 788.384 | 2.74x |
| inpaint TELEA r3 5% BGR 1280x720 | 24.743 | 54.579 | 2.21x |
| inpaint TELEA r3 5% gray 1280x720 | 18.699 | 32.077 | 1.72x |
| inpaint NS r3 5% BGR 1280x720 | 22.074 | 44.189 | 2.00x |
| inpaint NS r3 5% gray 1280x720 | 12.656 | 26.391 | 2.09x |
| edgePreservingFilter RECURS 1280x720 | 41.925 | 137.084 | 3.27x |
| edgePreservingFilter NORMCONV 1280x720 | 155.790 | 448.179 | 2.88x |
| stylization 1280x720 | 155.946 | 460.190 | 2.95x |
| pencilSketch 1280x720 | 55.822 | 139.828 | 2.50x |
| bilateralFilter d9 s75 BGR 1280x720 | 20.806 | 37.305 | 1.79x |
| bilateralFilter d9 s75 gray 1280x720 | 14.203 | 11.870 | 0.84x |
| QR detect+decode 640x480 | 0.807 | 14.684 | 18.20x |
| QR detect+decode 1280x720 | 1.995 | 47.201 | 23.66x |
| QR encode v10 | 0.439 | 29.217 | 66.53x |
| ArUco detect 20 markers 1280x720 | 8.174 | 13.728 | 1.68x |
| ArUco detect SUBPIX 1280x720 | 8.378 | 13.865 | 1.65x |

Median speedup **2.44x**, worst 0.32x.
