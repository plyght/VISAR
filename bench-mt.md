VISAR 0.1.0 vs OpenCV 5.0.0, 1920x1080, 4 thread(s), x86_64-linux (avx2)

| Operation | VISAR (ms) | OpenCV (ms) | Speedup |
| --- | ---: | ---: | ---: |
| cvtColor BGR2GRAY | 0.291 | 0.355 | 1.22x |
| cvtColor BGR2RGB | 0.160 | 0.131 | 0.82x |
| cvtColor BGR2BGRA | 0.189 | 0.291 | 1.54x |
| cvtColor BGR2HSV | 0.903 | 1.418 | 1.57x |
| cvtColor BGR2YCrCb | 0.582 | 0.723 | 1.24x |
| resize linear 1/2 | 0.200 | 0.359 | 1.79x |
| resize linear 2x | 2.039 | 5.150 | 2.53x |
| resize linear 0.75 | 0.709 | 1.117 | 1.58x |
| resize area 1/2 | 0.202 | 0.304 | 1.51x |
| resize nearest 2x | 0.425 | 4.668 | 10.97x |
| resize cubic 2x | 5.086 | 6.818 | 1.34x |
| GaussianBlur 5x5 | 0.524 | 1.362 | 2.60x |
| GaussianBlur s=3 | 4.543 | 5.801 | 1.28x |
| blur 3x3 | 0.659 | 1.723 | 2.61x |
| blur 15x15 | 1.240 | 3.221 | 2.60x |
| Sobel dx 3x3 ->i16 | 0.414 | 0.467 | 1.13x |
| threshold binary | 0.361 | 0.087 | 0.24x |
| threshold otsu | 0.385 | 0.965 | 2.51x |
| erode 3x3 rect | 0.261 | 0.415 | 1.59x |
| dilate 7x7 ellipse | 0.640 | 1.191 | 1.86x |
| flip horizontal | 0.141 | 0.347 | 2.46x |
| rotate 90cw | 0.456 | 1.004 | 2.20x |
| warpAffine linear | 2.383 | 4.722 | 1.98x |
| equalizeHist | 0.675 | 0.942 | 1.39x |
| add u8 | 0.149 | 0.237 | 1.59x |
| absdiff u8 | 0.149 | 0.238 | 1.60x |
| addWeighted u8 | 0.534 | 1.615 | 3.02x |
| GaussianBlur 5x5 f32 | 1.604 | 2.724 | 1.70x |
| pyrDown u8 | 0.330 | 0.919 | 2.78x |
| pyrDown u8 gray | 0.108 | 0.206 | 1.90x |
| pyrUp u8 gray | 0.419 | 3.451 | 8.24x |
| pyrDown f32 | 1.274 | 0.941 | 0.74x |
| cornerMinEigenVal 3/3 | 2.939 | 9.483 | 3.23x |
| cornerHarris 2/3 | 2.566 | 8.785 | 3.42x |
| goodFeaturesToTrack 1000 | 6.959 | 17.643 | 2.54x |
| calcOpticalFlowPyrLK 1000pt | 3.754 | 6.338 | 1.69x |
| imdecode PNG | 24.427 | 30.790 | 1.26x |
| imencode PNG | 31.861 | 61.507 | 1.93x |
| imdecode JPEG | 8.116 | 8.883 | 1.09x |
| imencode JPEG q90 | 7.316 | 6.986 | 0.95x |
| findContours TREE simple | 3.548 | 6.272 | 1.77x |
| findContours EXT none | 1.939 | 3.014 | 1.55x |
| moments binary | 0.417 | 3.066 | 7.36x |
| contour moments | 0.088 | 0.216 | 2.46x |
| convexHull 20k pts | 0.256 | 1.579 | 6.17x |
| approxPolyDP | 0.704 | 1.638 | 2.33x |

Median speedup **1.78x**, worst 0.24x.
