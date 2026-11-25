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
} visar_moments;
typedef struct visar_rotated_rect { float cx, cy, width, height, angle; } visar_rotated_rect;
int visar_moments(const visar_image *img /* 1 channel: 8U, 16U, 16S, 32F */, int binary, visar_moments *out);
int visar_contour_moments(const void *pts, size_t n, int depth, visar_moments *out);
void visar_hu_moments(const visar_moments *m, double hu[7]);
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

#ifdef __cplusplus
}
#endif
#endif /* VISAR_H */
