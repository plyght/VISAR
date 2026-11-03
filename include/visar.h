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

/* core: op 0 add, 1 subtract, 2 absdiff, 3 multiply, 4 divide, 5 min, 6 max, 7 and, 8 or, 9 xor */
int visar_arith(int op, const visar_image *a, const visar_image *b, const visar_image *dst);
int visar_add_weighted(const visar_image *a, double alpha, const visar_image *b, double beta, double gamma, const visar_image *dst);
int visar_convert_to(const visar_image *src, const visar_image *dst, double alpha, double beta);

/* codecs: mode -1 unchanged, 0 gray, 1 color; bgr != 0 for BGR channel order */
int visar_imdecode(const uint8_t *bytes, size_t len, int mode, int bgr, visar_image *out);
void visar_image_free(visar_image *img);
int visar_imencode(const char *ext, const visar_image *img, int bgr, int quality, uint8_t **out, size_t *out_len);
void visar_free(uint8_t *p, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* VISAR_H */
