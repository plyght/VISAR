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
