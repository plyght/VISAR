### VISAR live tracking demo

| Metric | Value |
| --- | --- |
| Target | `x86_64-linux (avx2)` |
| Threads | 4 |
| Resolution | 960x540 |
| Frames | 150 |
| Pipeline time per frame (mean / p50 / p95) | 0.86 / 0.67 / 2.41 ms |
| Throughput | **1157 FPS** |
| Detection recall vs ground truth | 95.5% |
| Identity switches | 9 |
| GIF | `track.gif` (3300 KB) |

Pipeline: gray, Gaussian 5x5, background subtraction, open 3x3, dilate 9x9 ellipse, connected components, tracker.
