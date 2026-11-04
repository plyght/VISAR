VISAR 0.1.0 vs OpenCV 5.0.0, 1920x1080, 1 thread(s), x86_64-linux (avx2)

| Operation | VISAR (ms) | OpenCV (ms) | Speedup |
| --- | ---: | ---: | ---: |
| cvtColor BGR2GRAY | 0.551 | 0.766 | 1.39x |
| cvtColor BGR2RGB | 0.293 | 0.176 | 0.60x |
| cvtColor BGR2BGRA | 0.348 | 0.324 | 0.93x |
| cvtColor BGR2HSV | 2.123 | 3.091 | 1.46x |
| cvtColor BGR2YCrCb | 1.263 | 1.693 | 1.34x |
| resize linear 1/2 | 0.362 | 0.627 | 1.73x |
| resize linear 2x | 4.113 | 9.533 | 2.32x |
| resize linear 0.75 | 1.344 | 2.117 | 1.58x |
| resize area 1/2 | 0.364 | 0.629 | 1.73x |
| resize nearest 2x | 1.059 | 8.710 | 8.22x |
| resize cubic 2x | 9.983 | 12.775 | 1.28x |
| GaussianBlur 5x5 | 0.958 | 2.716 | 2.83x |
| GaussianBlur s=3 | 9.193 | 10.983 | 1.19x |
| blur 3x3 | 1.268 | 1.697 | 1.34x |
| blur 15x15 | 2.649 | 3.191 | 1.20x |
| Sobel dx 3x3 ->i16 | 0.319 | 0.589 | 1.85x |
| threshold binary | 0.166 | 0.157 | 0.95x |
| threshold otsu | 0.813 | 0.974 | 1.20x |
| erode 3x3 rect | 0.410 | 0.507 | 1.24x |
| dilate 7x7 ellipse | 1.207 | 1.934 | 1.60x |
| flip horizontal | 0.266 | 0.251 | 0.94x |
| rotate 90cw | 0.891 | 0.926 | 1.04x |
| warpAffine linear | 5.640 | 11.368 | 2.02x |
| equalizeHist | 1.490 | 1.732 | 1.16x |
| add u8 | 0.246 | 0.238 | 0.97x |
| absdiff u8 | 0.317 | 0.304 | 0.96x |
| addWeighted u8 | 1.066 | 1.552 | 1.46x |
| GaussianBlur 5x5 f32 | 2.929 | 3.951 | 1.35x |
| pyrDown u8 | 0.634 | 1.805 | 2.85x |
| pyrDown u8 gray | 0.180 | 0.401 | 2.23x |
| pyrUp u8 gray | 0.877 | 3.406 | 3.88x |
| pyrDown f32 | 2.494 | 1.769 | 0.71x |
| cornerMinEigenVal 3/3 | 6.191 | 9.179 | 1.48x |
| cornerHarris 2/3 | 5.633 | 8.459 | 1.50x |
| goodFeaturesToTrack 1000 | 10.072 | 16.629 | 1.65x |
| calcOpticalFlowPyrLK 1000pt | 8.439 | 13.094 | 1.55x |
| imdecode PNG | 24.418 | 30.676 | 1.26x |
| imencode PNG | 31.924 | 61.445 | 1.92x |
| imdecode JPEG | 8.165 | 8.891 | 1.09x |
| imencode JPEG q90 | 7.331 | 6.899 | 0.94x |
| findContours TREE simple | 3.550 | 5.942 | 1.67x |
| findContours EXT none | 1.884 | 2.726 | 1.45x |
| moments binary | 0.789 | 3.054 | 3.87x |
| contour moments | 0.088 | 0.215 | 2.46x |
| convexHull 20k pts | 0.256 | 1.576 | 6.16x |
| approxPolyDP | 0.702 | 1.640 | 2.33x |

Median speedup **1.46x**, worst 0.60x.
