/*
 * VISAR: Visual Imaging, Sensing, and Recognition. Stable C ABI.
 * SPDX-License-Identifier: AGPL-3.0-or-later
 *
 * Images are described by visar_image over caller-owned memory (except
 * results of visar_imdecode, which VISAR owns until visar_image_free).
 * Functions return VISAR_OK (0) or a negative visar_status; the last error's
 * name is available from visar_last_error() (thread-local).
 * Constants (depths, color codes, interpolation, border, threshold and
 * morphology types) use OpenCV's numbering.
 */
#ifndef VISAR_H
#define VISAR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum visar_status {
    VISAR_OK = 0,
    VISAR_INVALID_ARGUMENT = -1,
    VISAR_SIZE_MISMATCH = -2,
    VISAR_UNSUPPORTED = -3,
    VISAR_OUT_OF_MEMORY = -4,
    VISAR_INVALID_DATA = -5,
    VISAR_INTERNAL = -6,
};

enum visar_depth { VISAR_8U = 0, VISAR_8S = 1, VISAR_16U = 2, VISAR_16S = 3, VISAR_32S = 4, VISAR_32F = 5, VISAR_64F = 6, VISAR_16F = 7 };

typedef struct visar_image {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t depth;   /* enum visar_depth */
    size_t stride;    /* bytes between row starts */
    void *data;
} visar_image;

const char *visar_version(void);
const char *visar_build_info(void);
const char *visar_last_error(void);
void visar_set_num_threads(int n); /* 0 = one per CPU, 1 = serial */
int visar_get_num_threads(void);

/* imgproc */
int visar_cvt_color(const visar_image *src, const visar_image *dst, int code);
/* BT.601 gray of a BGR/BGRA 8U image shrunk by the integer factor k (1-16) with box averaging, in one
   pass; dst is (width / k) x (height / k), 1 channel. */
int visar_gray_downsample(const visar_image *src, const visar_image *dst, int k);
int visar_resize(const visar_image *src, const visar_image *dst, int interpolation);
int visar_gaussian_blur(const visar_image *src, const visar_image *dst, int kw, int kh, double sigma_x, double sigma_y, int border);
int visar_box_filter(const visar_image *src, const visar_image *dst, int kw, int kh, int normalize, int border);
int visar_sobel(const visar_image *src, const visar_image *dst, int dx, int dy, int ksize, double scale, int border);
/* pyramids: dst is ((w+1)/2, (h+1)/2) for pyr_down, (2w, 2h) for pyr_up (border must be 4, BORDER_DEFAULT) */
int visar_pyr_down(const visar_image *src, const visar_image *dst, int border);
int visar_pyr_up(const visar_image *src, const visar_image *dst, int border);
/* corners: src VISAR_8U or VISAR_32F with one channel, dst VISAR_32F; ksize 1, 3, 5, 7 or -1 (Scharr) */
int visar_corner_min_eigen_val(const visar_image *src, const visar_image *dst, int block_size, int ksize, int border);
int visar_corner_harris(const visar_image *src, const visar_image *dst, int block_size, int ksize, double k, int border);
/* writes up to cap (x, y) pairs to corners and the number found to *count; mask may be NULL */
int visar_good_features_to_track(const visar_image *src, float *corners, size_t cap, size_t *count, int max_corners,
                                 double quality_level, double min_distance, const visar_image *mask, int block_size,
                                 int gradient_size, int use_harris, double k);
int visar_threshold(const visar_image *src, const visar_image *dst, double thresh, double maxval, int type, double *used);
/* adaptiveThreshold (8U, one channel): method 0 MEAN_C, 1 GAUSSIAN_C; type 0 BINARY, 1 BINARY_INV. */
int visar_adaptive_threshold(const visar_image *src, const visar_image *dst, double maxval, int method, int type,
                             int block_size, double delta);
int visar_morphology(const visar_image *src, const visar_image *dst, int op, int shape, int kw, int kh, int iterations);
int visar_flip(const visar_image *src, const visar_image *dst, int code);
int visar_rotate(const visar_image *src, const visar_image *dst, int code);
int visar_warp_affine(const visar_image *src, const visar_image *dst, const double m[6], int interpolation, int border, double value);
int visar_equalize_hist(const visar_image *src, const visar_image *dst);
/* labels: VISAR_32S image; stats: max rows of left, top, width, height, area; centroids: max rows of cx, cy */
int visar_connected_components(const visar_image *src, const visar_image *labels, int connectivity, int32_t *stats, double *centroids, size_t max, size_t *count);

/* contours: results are VISAR-owned buffers, released with visar_contours_free.
 * Contour i is points[2*starts[i] .. 2*starts[i+1]) as (x, y) pairs;
 * hierarchy[4*i .. 4*i+4) is OpenCV's [next, previous, first child, parent]. */
typedef struct visar_contours {
    size_t count;       /* number of contours */
    size_t total;       /* number of points over all contours */
    int32_t *points;    /* total (x, y) pairs */
    size_t *starts;     /* count + 1 offsets into points (in points, not ints) */
    int32_t *hierarchy; /* count * 4 */
} visar_contours;
/* mode: RETR_EXTERNAL 0, RETR_LIST 1, RETR_CCOMP 2, RETR_TREE 3;
 * method: CHAIN_APPROX_NONE 1, SIMPLE 2, TC89_L1 3, TC89_KCOS 4. src: 8-bit, 1 channel, nonzero = foreground */
int visar_find_contours(const visar_image *src, int mode, int method, int offset_x, int offset_y, visar_contours *out);
void visar_contours_free(visar_contours *contours);

/* drawing (8-bit images; matches OpenCV pixel for pixel). Polygons use the
 * visar_contours layout: points (x, y pairs) and starts (count + 1 entries).
 * color: per-channel values (like cv::Scalar); line_type: 8 (LINE_8) or 4 (LINE_4). */
int visar_draw_contours(const visar_image *img, const int32_t *points, const size_t *starts, size_t count,
                        const int32_t *hierarchy /* count * 4 or NULL */, int contour_idx /* -1 = all */,
                        const double color[4], int thickness /* < 0 fills */, int line_type, int max_level,
                        int offset_x, int offset_y);
int visar_fill_poly(const visar_image *img, const int32_t *points, const size_t *starts, size_t count,
                    const double color[4], int line_type, int offset_x, int offset_y);
int visar_polylines(const visar_image *img, const int32_t *points, const size_t *starts, size_t count, int closed,
                    const double color[4], int thickness, int line_type);

/* shape analysis. Point sets are n (x, y) pairs of int32 (depth VISAR_32S)
 * or float (VISAR_32F), like OpenCV's CV_32SC2 / CV_32FC2. */
typedef struct visar_moments {
    double m00, m10, m01, m20, m11, m02, m30, m21, m12, m03;
    double mu20, mu11, mu02, mu30, mu21, mu12, mu03;
    double nu20, nu11, nu02, nu30, nu21, nu12, nu03;
} visar_moments_t;
typedef struct visar_rotated_rect { float cx, cy, width, height, angle; } visar_rotated_rect;
int visar_moments(const visar_image *img /* 1 channel: 8U, 16U, 16S, 32F */, int binary, visar_moments_t *out);
int visar_contour_moments(const void *pts, size_t n, int depth, visar_moments_t *out);
void visar_hu_moments(const visar_moments_t *m, double hu[7]);
int visar_contour_area(const void *pts, size_t n, int depth, int oriented, double *out);
int visar_arc_length(const void *pts, size_t n, int depth, int closed, double *out);
int visar_bounding_rect(const void *pts, size_t n, int depth, int32_t rect[4] /* x, y, w, h */);
int visar_mask_bounding_rect(const visar_image *mask /* 8U, 1 channel */, int32_t rect[4]);
/* indices: room for n; receives hull indices into pts in OpenCV's order */
int visar_convex_hull(const void *pts, size_t n, int depth, int clockwise, int return_points, int32_t *indices, size_t *count);
/* pts: int32; hull: indices from visar_convex_hull(return_points = 0); out: room for hull_len * 4 */
int visar_convexity_defects(const int32_t *pts, size_t n, const int32_t *hull, size_t hull_len, int32_t *out, size_t *count);
int visar_is_contour_convex(const void *pts, size_t n, int depth, int *out);
int visar_min_area_rect(const void *pts, size_t n, int depth, visar_rotated_rect *out);
void visar_box_points(const visar_rotated_rect *box, float out[8]);
int visar_min_enclosing_circle(const void *pts, size_t n, int depth, float out[3] /* cx, cy, radius */);
int visar_fit_ellipse(const void *pts, size_t n, int depth, visar_rotated_rect *out);
/* out: room for n points of the input depth */
int visar_approx_poly_dp(const void *pts, size_t n, int depth, double epsilon, int closed, void *out, size_t *count);
int visar_point_polygon_test(const void *pts, size_t n, int depth, float x, float y, int measure_dist, double *out);
int visar_match_shapes(const void *a, size_t na, int depth_a, const void *b, size_t nb, int depth_b, int method, double *out);

/* features2d: keypoints and matches use cv::KeyPoint / cv::DMatch layouts. Arrays returned by VISAR are
   released with visar_keypoints_free / visar_matches_free; ORB descriptors (count * 32 bytes) with visar_free.
   Keypoints match cv2 exactly (FAST in OpenCV's order; ORB per pyramid level, within a level in FAST's
   row-major order where OpenCV leaves std::nth_element's order). */
typedef struct visar_keypoint {
    float x, y, size, angle, response;
    int32_t octave, class_id;
} visar_keypoint;
typedef struct visar_dmatch {
    int32_t query_idx, train_idx, img_idx;
    float distance;
} visar_dmatch;
/* FAST corners of a one-channel VISAR_8U image; type: 0 TYPE_5_8, 1 TYPE_7_12, 2 TYPE_9_16 */
int visar_fast(const visar_image *src, int threshold, int nonmax_suppression, int type, visar_keypoint **out, size_t *count);
void visar_keypoints_free(visar_keypoint *kps, size_t count);
/* cv2.ORB_create parameters (visar_orb_default_params fills the defaults); score_type 0 HARRIS, 1 FAST */
typedef struct visar_orb_params {
    int nfeatures;
    float scale_factor;
    int nlevels, edge_threshold, first_level, wta_k, score_type, patch_size, fast_threshold;
} visar_orb_params;
void visar_orb_default_params(visar_orb_params *p);
/* ORB detectAndCompute: src VISAR_8U gray, BGR or BGRA; mask (VISAR_8U, may be NULL); params NULL = defaults;
   descriptors NULL = keypoints only */
int visar_orb_detect_and_compute(const visar_image *src, const visar_image *mask, const visar_orb_params *params,
                                 visar_keypoint **kps, size_t *count, uint8_t **descriptors);
/* BFMatcher: descriptors are images with one row per descriptor, VISAR_8U (any norm) or VISAR_32F (L1, L2,
   L2SQR). norm: NORM_L1 2, NORM_L2 4, NORM_L2SQR 5, NORM_HAMMING 6, NORM_HAMMING2 7 */
int visar_bf_match(const visar_image *query, const visar_image *train, int norm, int cross_check, visar_dmatch **out,
                   size_t *count);
void visar_matches_free(visar_dmatch *m, size_t count);
/* out: room for query rows * k matches (row q's at q * k); counts: query rows (matches found per query) */
int visar_bf_knn_match(const visar_image *query, const visar_image *train, int k, int norm, visar_dmatch *out,
                       uint32_t *counts);
/* Lowe's ratio test over visar_bf_knn_match output (k >= 2); out: room for nq matches */
int visar_ratio_test(const visar_dmatch *knn, const uint32_t *counts, size_t nq, int k, float ratio, visar_dmatch *out,
                     size_t *count);
/* drawing: out is VISAR_8U BGR (img's size; for matches max(h1, h2) x (w1 + w2)); color NULL = one colour per
   keypoint/match; flags: 1 DRAW_OVER_OUTIMG, 2 NOT_DRAW_SINGLE_POINTS, 4 DRAW_RICH_KEYPOINTS */
int visar_draw_keypoints(const visar_image *img, const visar_keypoint *kps, size_t count, const visar_image *out,
                         const double color[4], int flags);
int visar_draw_matches(const visar_image *img1, const visar_keypoint *kp1, size_t n1, const visar_image *img2,
                       const visar_keypoint *kp2, size_t n2, const visar_dmatch *matches, size_t nm, const visar_image *out,
                       const double match_color[4], const double single_color[4], int flags);
/* video: pyramidal Lucas-Kanade on VISAR_8U images; points are n (x, y) float pairs; err may be NULL.
   criteria_type: 1 COUNT | 2 EPS; flags: 4 USE_INITIAL_FLOW | 8 LK_GET_MIN_EIGENVALS */
int visar_calc_optical_flow_pyr_lk(const visar_image *prev, const visar_image *next, const float *prev_pts, float *next_pts,
                                   uint8_t *status, float *err, size_t n, int win_w, int win_h, int max_level,
                                   int criteria_type, int max_count, double epsilon, int flags, double min_eig_threshold);
/* video: MOSSE correlation tracker on VISAR_8U gray frames. box: x, y, w, h; scale != 0 searches three
   scales per frame. psr and ok may be NULL; ok is 0 when the target is lost (the filter is not updated). */
typedef struct visar_mosse visar_mosse;
int visar_mosse_create(const visar_image *frame, const float box[4], int scale, visar_mosse **out);
int visar_mosse_update(visar_mosse *t, const visar_image *frame, float box[4], float *psr, int *ok);
void visar_mosse_free(visar_mosse *t);
/* video: running-average background model (learning rate 1 / 2^shift); mask: VISAR_8U, frame size */
typedef struct visar_background visar_background;
int visar_background_create(uint32_t width, uint32_t height, int shift, visar_background **out);
int visar_background_apply(visar_background *b, const visar_image *frame, const visar_image *mask, int threshold);
void visar_background_free(visar_background *b);
/* video: multi-object tracker; boxes: n (x, y, w, h) floats; tracks (may be NULL) receives the live track count */
typedef struct visar_tracker visar_tracker;
int visar_tracker_create(float gate, visar_tracker **out);
int visar_tracker_update(visar_tracker *t, const float *boxes, size_t n, size_t *tracks);
void visar_tracker_free(visar_tracker *t);

/* core: op 0 add, 1 subtract, 2 absdiff, 3 multiply, 4 divide, 5 min, 6 max, 7 and, 8 or, 9 xor */
int visar_arith(int op, const visar_image *a, const visar_image *b, const visar_image *dst);
int visar_add_weighted(const visar_image *a, double alpha, const visar_image *b, double beta, double gamma, const visar_image *dst);
int visar_convert_to(const visar_image *src, const visar_image *dst, double alpha, double beta);

/* codecs: mode -1 unchanged, 0 gray, 1 color; bgr != 0 for BGR channel order */
int visar_imdecode(const uint8_t *bytes, size_t len, int mode, int bgr, visar_image *out);
void visar_image_free(visar_image *img);
int visar_imencode(const char *ext, const visar_image *img, int bgr, int quality, uint8_t **out, size_t *out_len);
void visar_free(uint8_t *p, size_t len);

/* videoio: video files, like cv2.VideoCapture / cv2.VideoWriter. Reading: .avi (Motion-JPEG,
   uncompressed RGB, I420/YV12, Y800) and .y4m everywhere; .mp4/.mov (H.264, HEVC, ...) on macOS
   through AVFoundation (elsewhere VISAR_UNSUPPORTED, and visar_last_error() says how to convert the
   file with ffmpeg). read() fills a caller-owned 8U image of the video's size: 3 channels BGR,
   4 BGRA or 1 gray; *got is 0 at the end. get/set take cv2.CAP_PROP_* numbers: 0 POS_MSEC,
   1 POS_FRAMES (set = seek), 3 FRAME_WIDTH, 4 FRAME_HEIGHT, 5 FPS, 6 FOURCC, 7 FRAME_COUNT
   (-1 when unknown). */
typedef struct visar_video_capture visar_video_capture;
int visar_video_capture_open(const char *path, visar_video_capture **out);
void visar_video_capture_free(visar_video_capture *c);
int visar_video_capture_read(visar_video_capture *c, const visar_image *dst, int *got);
double visar_video_capture_get(visar_video_capture *c, int prop);
int visar_video_capture_set(visar_video_capture *c, int prop, double value);
const char *visar_video_capture_backend(visar_video_capture *c); /* "AVI", "YUV4MPEG2", "AVFoundation" */
/* Writing: .avi (fourcc 'MJPG' as cv2.VideoWriter_fourcc builds it: Motion-JPEG, OpenDML past 1 GB;
   0: uncompressed BGR) or .y4m ('I420'); fourcc -1 picks the extension's default. quality 1-100
   for Motion-JPEG (<= 0: 95). Frames are 8U BGR, BGRA or gray of the given size. close() finishes
   the file and frees the writer, also when it fails. */
typedef struct visar_video_writer visar_video_writer;
int visar_video_writer_open(const char *path, int fourcc, double fps, int width, int height, int quality,
                            visar_video_writer **out);
int visar_video_writer_write(visar_video_writer *w, const visar_image *frame);
int visar_video_writer_close(visar_video_writer *w);

/* dnn: ONNX inference (f32). Model bytes are copied. A net is planned for one input shape;
   the input buffer (plain NCHW) and outputs live inside the net and stay valid until the next plan. */
typedef struct visar_net visar_net;
int visar_dnn_net_load(const void *bytes, size_t len, visar_net **out);
void visar_dnn_net_free(visar_net *net);
int visar_dnn_net_plan(visar_net *net, const size_t *shape, size_t rank);
float *visar_dnn_net_input(visar_net *net, size_t *count);
int visar_dnn_net_run(visar_net *net);
size_t visar_dnn_net_output_count(visar_net *net);
const char *visar_dnn_net_output_name(visar_net *net, size_t i);
int visar_dnn_net_output(visar_net *net, size_t i, size_t shape[6], size_t *rank, const float **data);
/* Backends: 0 auto (default: Metal on macOS for large enough models whose ops all have GPU kernels,
   else CPU), 1 CPU, 2 Metal (planning fails without a Metal device or for unsupported ops).
   set_backend applies at the next plan; net_backend reports the current plan's (1 CPU, 2 Metal,
   0 before planning). */
int visar_dnn_metal_available(void);
const char *visar_dnn_metal_device(void); /* device name, or why Metal is unavailable */
int visar_dnn_net_set_backend(visar_net *net, int backend);
int visar_dnn_net_backend(visar_net *net);

/* YuNet face detection, matching cv2.FaceDetectorYN. detect() takes a BGR VISAR_8U image of the
   configured size and writes up to cap rows of 15 floats: x, y, w, h, right eye, left eye, nose tip,
   right and left mouth corners (x, y each), score. *count receives the number of faces. */
typedef struct visar_face_detector visar_face_detector;
int visar_face_detector_create(const void *model, size_t len, int width, int height, float score_threshold,
                               float nms_threshold, int top_k, visar_face_detector **out);
void visar_face_detector_free(visar_face_detector *d);
int visar_face_detector_set_input_size(visar_face_detector *d, int width, int height);
int visar_face_detector_set_backend(visar_face_detector *d, int backend); /* re-plans */
int visar_face_detector_backend(visar_face_detector *d);
void visar_face_detector_set_thresholds(visar_face_detector *d, float score_threshold, float nms_threshold, int top_k);
int visar_face_detector_detect(visar_face_detector *d, const visar_image *img, float *out, size_t cap, size_t *count);
/* detect() for a BGR or BGRA frame of any size: it is shrunk to the detector size with INTER_AREA
   (integer factors such as 3840x2160 -> 640x360 in one fused pass into the network input) and the
   results are in frame pixels. prepare/infer/decode are its three stages, for timing them. */
int visar_face_detector_detect_frame(visar_face_detector *d, const visar_image *img, float *out, size_t cap,
                                     size_t *count);
int visar_face_detector_prepare(visar_face_detector *d, const visar_image *img);
/* prepare() plus visar_gray_downsample(img, gray, img width / gray width): when the frame shrinks
   by one integer factor k > 1 on both axes and gray is the detector size, both come from one pass
   over the frame (the same results as the two calls). */
int visar_face_detector_prepare_gray(visar_face_detector *d, const visar_image *img, const visar_image *gray);
int visar_face_detector_infer(visar_face_detector *d);
int visar_face_detector_decode(visar_face_detector *d, float *out, size_t cap, size_t *count);

/* Object detection with the OpenCV Model Zoo's NanoDet and YOLOX models (COCO classes), matching
   the zoo's Python pipelines. arch: -1 detect from the model, 0 NanoDet, 1 YOLOX; a negative
   threshold selects the zoo demo's default. detect() takes a BGR VISAR_8U image of any size and
   writes up to cap detections in decreasing score order; *count receives the number found. */
typedef struct visar_object_detector visar_object_detector;
typedef struct {
    float box[4]; /* x, y, width, height in image pixels */
    float score;
    int class_id; /* COCO class index, see visar_coco_class_name */
} visar_detection;
int visar_object_detector_create(const void *model, size_t len, int arch, float score_threshold, float nms_threshold,
                                 visar_object_detector **out);
void visar_object_detector_free(visar_object_detector *d);
int visar_object_detector_arch(visar_object_detector *d);
void visar_object_detector_set_thresholds(visar_object_detector *d, float score_threshold, float nms_threshold);
int visar_object_detector_detect(visar_object_detector *d, const visar_image *img, visar_detection *out, size_t cap,
                                 size_t *count);
const char *visar_coco_class_name(int i);

/* calib3d. Points are interleaved floats (x, y) or doubles ((x, y) /
   (x, y, z)) as noted; matrices are row-major doubles; methods and flags use
   OpenCV's numbering. Robust estimators draw OpenCV's RANSAC samples (same
   RNG and seed), so results are deterministic. *found = 0 means no model
   (OpenCV's empty result). */
int visar_find_homography(const float *src, const float *dst, size_t n, int method /* 0, 4 LMEDS, 8 RANSAC */,
                          double ransac_threshold, int max_iters, double confidence, uint8_t *mask /* n or NULL */,
                          double h[9], int *found);
int visar_get_perspective_transform(const float src[8], const float dst[8], double m[9]);
int visar_get_affine_transform(const float src[6], const float dst[6], double m[6]);
void visar_perspective_transform(const float *src, float *dst, size_t n, const double m[9]);
/* findFundamentalMat: method 1 FM_7POINT, 2 FM_8POINT, 4 FM_LMEDS, 8 FM_RANSAC; f receives up to 3 matrices
   (*count of them; several only for FM_7POINT with 7 points). */
int visar_find_fundamental_mat(const float *p1, const float *p2, size_t n, int method, double ransac_threshold,
                               double confidence, int max_iters, uint8_t *mask, double f[27], int *count);
/* solveCubic: real roots of c[0] x^3 + c[1] x^2 + c[2] x + c[3]; returns their number (-1: any x). */
int visar_solve_cubic(const double c[4], double roots[3]);
int visar_estimate_affine_2d(const float *src, const float *dst, size_t n, int partial, int method, double ransac_threshold,
                             int max_iters, double confidence, int refine_iters, uint8_t *inliers, double m[6], int *found);
/* Rodrigues: n = 3 (vector -> matrix, optional 3x9 jacobian dR/dr) or n = 9 (matrix -> vector, jac NULL). */
int visar_rodrigues(const double *src, size_t n, double *dst, double jac[27]);
/* dist: nd = 0, 4, 5 or 8 coefficients (k1 k2 p1 p2 [k3 [k4 k5 k6]]). jac (2n x (10 + nd), nd <= 5) may be NULL. */
int visar_project_points(const double *obj, size_t n, const double rvec[3], const double tvec[3], const double k[9],
                         const double *dist, size_t nd, double *out, double *jac);
int visar_undistort_points(const double *src, size_t n, double *dst, const double k[9], const double *dist, size_t nd,
                           const double *r /* 3x3 or NULL */, const double *p /* 3x3 or NULL */);
/* flags: 0 SOLVEPNP_ITERATIVE, 1 SOLVEPNP_EPNP; use_guess seeds ITERATIVE with rvec/tvec. */
int visar_solve_pnp(const double *obj, const double *img, size_t n, const double k[9], const double *dist, size_t nd,
                    int flags, int use_guess, double rvec[3], double tvec[3]);
int visar_solve_pnp_ransac(const double *obj, const double *img, size_t n, const double k[9], const double *dist, size_t nd,
                           int iterations, double reprojection_error, double confidence, int flags, double rvec[3],
                           double tvec[3], uint8_t *inliers /* n or NULL */, int *found);
/* undistort: 8U, 1-4 channels, dst the size of src; new_k may be NULL. */
int visar_undistort(const visar_image *src, const visar_image *dst, const double k[9], const double *dist, size_t nd,
                    const double *new_k);
/* CV_32FC1 maps (VISAR_32F, one channel, the destination size). */
int visar_init_undistort_rectify_map(const double k[9], const double *dist, size_t nd, const double *r, const double *new_k,
                                     const visar_image *mapx, const visar_image *mapy);
/* remap: 8U 1-4 channels through 32F maps, INTER_LINEAR, BORDER_CONSTANT with `value`. */
int visar_remap(const visar_image *src, const visar_image *dst, const visar_image *mapx, const visar_image *mapy,
                double value);
/* cornerSubPix on a 1-channel 8U or 32F image; criteria_type 1 COUNT, 2 EPS, 3 both; zero zone -1 = none. */
int visar_corner_sub_pix(const visar_image *img, float *corners, size_t n, int win_w, int win_h, int zero_w, int zero_h,
                         int criteria_type, int max_count, double epsilon);
/* findChessboardCorners (8U gray, BGR or BGRA): pattern_w * pattern_h (x, y) floats in OpenCV's order. */
int visar_find_chessboard_corners(const visar_image *img, int pattern_w, int pattern_h, float *corners, int flags,
                                  int *found);
/* calibrateCamera over planar views: counts[i] points per view, points back to back; dist has 5 coefficients.
   Flags: USE_INTRINSIC_GUESS 1, FIX_PRINCIPAL_POINT 4, ZERO_TANGENT_DIST 8, FIX_FOCAL_LENGTH 16, FIX_K1..K3 32/64/128. */
int visar_calibrate_camera(const double *obj, const double *img, const int *counts, size_t nviews, int width, int height,
                           double k[9], double dist[5], int flags, double *rvecs, double *tvecs, double *rms);

/* objdetect: QR codes (QRCodeEncoder / QRCodeDetector) and ArUco markers. Enumerations use OpenCV's
   numbering. visar_qr_encode returns a size x size 8U image (0 dark, 255 light, 2-module quiet zone),
   bit-identical to cv2.QRCodeEncoder; free it with visar_free(out, size * size). level: 0 L, 1 M, 2 Q, 3 H;
   mode: -1 auto, 1 numeric, 2 alphanumeric, 4 byte, 7 ECI (UTF-8), 8 kanji; version 0 = smallest. */
int visar_qr_encode(const char *text, size_t len, int version, int level, int mode, uint8_t **out, uint32_t *size);
/* n (2..16) symbols back to back in out (free with visar_free(out, sum of sizes[i]^2)). */
int visar_qr_encode_structured_append(const char *text, size_t len, int version, int level, int n, uint8_t **out,
                                      uint32_t sizes[16], uint32_t *count);
typedef struct visar_qr_code {
    float corners[8]; /* top-left, top-right, bottom-right, bottom-left of the symbol (x, y) */
    uint32_t offset, length; /* the decoded text is text[offset .. offset + length] */
    int32_t decoded, version;
} visar_qr_code;
/* detect (decode = 0), detectAndDecode (multi = 0) or detectAndDecodeMulti (multi = 1) on an 8U gray, BGR or
   BGRA image; *count = 0 when nothing is found. Free with visar_qr_free. */
int visar_qr_detect(const visar_image *src, int multi, int decode, visar_qr_code **codes, size_t *count, char **text,
                    size_t *text_len);
void visar_qr_free(visar_qr_code *codes, size_t count, char *text, size_t text_len);
/* cv::aruco::DetectorParameters; corner_refinement_method 0 NONE or 1 SUBPIX. */
typedef struct visar_aruco_params {
    int adaptive_thresh_win_size_min, adaptive_thresh_win_size_max, adaptive_thresh_win_size_step;
    double adaptive_thresh_constant, min_marker_perimeter_rate, max_marker_perimeter_rate,
        polygonal_approx_accuracy_rate, min_corner_distance_rate;
    int min_distance_to_border;
    double min_marker_distance_rate;
    float min_group_distance;
    int corner_refinement_method, corner_refinement_win_size;
    float relative_corner_refinement_win_size;
    int corner_refinement_max_iterations;
    double corner_refinement_min_accuracy;
    int marker_border_bits, perspective_remove_pixel_per_cell;
    double perspective_remove_ignored_margin_per_cell, max_erroneous_bits_in_border_rate, min_otsu_std_dev,
        error_correction_rate;
    float valid_bit_id_threshold;
    int detect_inverted_marker;
} visar_aruco_params;
void visar_aruco_default_params(visar_aruco_params *p);
/* dict: cv::aruco::PredefinedDictionaryType (DICT_4X4_50 = 0 ... DICT_ARUCO_MIP_36h12 = 21). */
int visar_aruco_generate(int dict, int id, int side, int border_bits, const visar_image *dst);
/* detectMarkers: corners (count * 8 floats), ids (count int32) and, when `rejected` is not NULL, rejected
   candidates (nrejected * 8 floats); free each with visar_free(ptr, bytes). params may be NULL. */
int visar_aruco_detect(const visar_image *src, int dict, const visar_aruco_params *params, uint8_t **corners,
                       uint8_t **ids, size_t *count, uint8_t **rejected, size_t *nrejected);
/* drawDetectedMarkers on 8U gray or BGR (in place); ids may be NULL; color is (B, G, R). */
int visar_aruco_draw(const visar_image *img, const float *corners, const int32_t *ids, size_t n, const double color[3]);

#ifdef __cplusplus
}
#endif
#endif /* VISAR_H */
