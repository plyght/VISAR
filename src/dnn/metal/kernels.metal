// VISAR dnn: Metal compute kernels (f32).
//
// Compiled at run time with newLibraryWithSource:options:error: (see
// device.zig); the parameter structs are generated from the Zig extern
// structs in encode.zig and prepended, so their layouts match by
// construction. Every kernel takes the same three bindings:
//
//   buffer(0)  A  the plan's arena: every intermediate tensor, at the
//                 element offsets the memory plan assigned
//   buffer(1)  K  constants: packed weights, biases, per-channel vectors,
//                 resize maps (uint bits), constant operands
//   buffer(2)  p  this dispatch's parameters (setBytes)
//
// Blocked tensors are NCHW with channels in blocks of four innermost
// ([N][C/4][H][W][4], one float4 per pixel and block); padding channels of
// the last block are kept zero, as the CPU backend does. Each thread is
// independent (no threadgroup memory), and the host rounds the grid up to
// whole threadgroups, so every kernel bounds-checks its position.

// ------------------------------------------------------------- helpers

static inline device const float *srcp(device float *A, device const float *K, uint k, uint off) {
    return k != 0 ? K + off : (device const float *)(A) + off;
}

// Activation codes (encode.zig actCode): 0 none, 1 relu, 2 leaky relu (a),
// 3 clip [a, b], 4 sigmoid, 5 hard sigmoid (a x + b), 6 hard swish,
// 7 tanh, 8 silu.
static inline float4 act4(float4 v, uint k, float a, float b) {
    switch (k) {
    case 1:
        return max(v, float4(0.0f));
    case 2:
        return select(v * a, v, v >= float4(0.0f));
    case 3:
        return min(max(v, float4(a)), float4(b));
    case 4:
        return float4(1.0f) / (float4(1.0f) + exp(-v));
    case 5:
        return min(max(v * a + b, float4(0.0f)), float4(1.0f));
    case 6:
        return v * min(max(v / 6.0f + 0.5f, float4(0.0f)), float4(1.0f));
    case 7:
        return tanh(v);
    case 8:
        return v / (float4(1.0f) + exp(-v));
    default:
        return v;
    }
}

static inline float act1(float v, uint k, float a, float b) {
    return act4(float4(v), k, a, b).x;
}

// Channel of element i of a blocked tensor with hw pixels per plane and
// nb channel blocks per image.
static inline uint blk_chan(uint i, uint hw, uint nb) {
    return ((i >> 2) / hw % nb) * 4 + (i & 3);
}

// Binary op codes (encode.zig binCode): add, sub, mul, div, max, min, pow.
static inline float powz(float x, float y) {
    if (y == 0.0f) return 1.0f;
    if (x < 0.0f && y == floor(y)) {
        float r = pow(-x, y);
        return fmod(y, 2.0f) != 0.0f ? -r : r;
    }
    return pow(x, y);
}

static inline float binop(uint op, float a, float b) {
    switch (op) {
    case 0:
        return a + b;
    case 1:
        return a - b;
    case 2:
        return a * b;
    case 3:
        return a / b;
    case 4:
        return max(a, b);
    case 5:
        return min(a, b);
    default:
        return powz(a, b);
    }
}

// ---------------------------------------------------------- convolution

// Bias is already in v. Residual before or after the activation, then the
// padding lanes of the last block zeroed. idx4: float4 index of the pixel
// in the output (and residual, same shape and layout).
static inline void conv_store(device float *A, constant ConvP &p, uint idx4, uint cob, float4 v) {
    if (p.has_res != 0) {
        float4 r = ((device const float4 *)(A + p.res))[idx4];
        if (p.res_after == 0) v += r;
        v = act4(v, p.act, p.a0, p.a1);
        if (p.res_after != 0) v += r;
    } else {
        v = act4(v, p.act, p.a0, p.a1);
    }
    if (cob * 4 + 4 > p.cout) {
        uint4 lane = uint4(0, 1, 2, 3) + uint4(cob * 4);
        v = select(v, float4(0.0f), lane >= uint4(p.cout));
    }
    ((device float4 *)(A + p.out))[idx4] = v;
}

// 1x1, stride 1, no padding, one group, blocked in and out: a GEMM over
// input channel blocks. Weights: one float4x4 per (output block, input
// block), column j = input lane j's weights into the four output lanes.
// Each thread: four consecutive pixels of one output block.
// Grid: (ceil(H*W / 4), coutb, N).
kernel void k_conv_pw(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                    constant ConvP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint hw = p.h * p.iw;
    uint p0 = gid.x * 4;
    uint cob = gid.y;
    uint n = gid.z;
    if (p0 >= hw || cob >= p.coutb || n >= p.n) return;
    uint last = hw - 1;
    uint q1 = min(p0 + 1, last);
    uint q2 = min(p0 + 2, last);
    uint q3 = min(p0 + 3, last);
    device const float4 *x = (device const float4 *)(A + p.x) + n * p.cinb * hw;
    device const float4x4 *w = (device const float4x4 *)(K + p.wt) + cob * p.cinb;
    float4 b = ((device const float4 *)(K + p.bias))[cob];
    float4 s0 = b;
    float4 s1 = b;
    float4 s2 = b;
    float4 s3 = b;
    for (uint cb = 0; cb < p.cinb; cb++) {
        float4x4 m = w[cb];
        device const float4 *xp = x + cb * hw;
        s0 += m * xp[p0];
        s1 += m * xp[q1];
        s2 += m * xp[q2];
        s3 += m * xp[q3];
    }
    uint ob = (n * p.coutb + cob) * hw;
    conv_store(A, p, ob + p0, cob, s0);
    if (p0 + 1 < hw) conv_store(A, p, ob + p0 + 1, cob, s1);
    if (p0 + 2 < hw) conv_store(A, p, ob + p0 + 2, cob, s2);
    if (p0 + 3 < hw) conv_store(A, p, ob + p0 + 3, cob, s3);
}

// Depthwise (one group per channel), blocked: each lane is one channel.
// Weights: [blocks][kh * kw] float4. Grid: (OW, OH, N * blocks).
kernel void k_conv_dw(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                    constant ConvP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.n * p.coutb) return;
    uint cb = z % p.coutb;
    device const float4 *x = (device const float4 *)(A + p.x) + z * p.h * p.iw;
    device const float4 *w = (device const float4 *)(K + p.wt) + cb * p.kh * p.kw;
    float4 s = ((device const float4 *)(K + p.bias))[cb];
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    for (uint ky = 0; ky < p.kh; ky++) {
        int iy = iy0 + int(ky * p.dh);
        if (iy < 0 || iy >= int(p.h)) continue;
        device const float4 *row = x + uint(iy) * p.iw;
        for (uint kx = 0; kx < p.kw; kx++) {
            int ix = ix0 + int(kx * p.dw);
            if (ix < 0 || ix >= int(p.iw)) continue;
            s += row[uint(ix)] * w[ky * p.kw + kx];
        }
    }
    conv_store(A, p, (z * p.oh + oy) * p.ow + ox, cb, s);
}

// One group, any kernel, blocked input. Weights: [coutb][cinb][kh][kw]
// float4x4. Grid: (OW, OH, N * coutb).
kernel void k_conv_dir_b(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                       constant ConvP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.n * p.coutb) return;
    uint n = z / p.coutb;
    uint cob = z % p.coutb;
    device const float4 *x = (device const float4 *)(A + p.x) + n * p.cinb * p.h * p.iw;
    device const float4x4 *w = (device const float4x4 *)(K + p.wt) + cob * p.cinb * p.kh * p.kw;
    float4 s = ((device const float4 *)(K + p.bias))[cob];
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    for (uint cb = 0; cb < p.cinb; cb++) {
        device const float4 *plane = x + cb * p.h * p.iw;
        device const float4x4 *wc = w + cb * p.kh * p.kw;
        for (uint ky = 0; ky < p.kh; ky++) {
            int iy = iy0 + int(ky * p.dh);
            if (iy < 0 || iy >= int(p.h)) continue;
            device const float4 *row = plane + uint(iy) * p.iw;
            for (uint kx = 0; kx < p.kw; kx++) {
                int ix = ix0 + int(kx * p.dw);
                if (ix < 0 || ix >= int(p.iw)) continue;
                s += wc[ky * p.kw + kx] * row[uint(ix)];
            }
        }
    }
    conv_store(A, p, (z * p.oh + oy) * p.ow + ox, cob, s);
}

// One group, any kernel, plain NCHW input (a network's first layer).
// Weights: [coutb][cin][kh][kw] float4 (output lanes). Grid: (OW, OH,
// N * coutb).
kernel void k_conv_dir_p(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                       constant ConvP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.n * p.coutb) return;
    uint n = z / p.coutb;
    uint cob = z % p.coutb;
    device const float *x = (device const float *)(A + p.x) + n * p.cin * p.h * p.iw;
    device const float4 *w = (device const float4 *)(K + p.wt) + cob * p.cin * p.kh * p.kw;
    float4 s = ((device const float4 *)(K + p.bias))[cob];
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    for (uint ci = 0; ci < p.cin; ci++) {
        device const float *plane = x + ci * p.h * p.iw;
        device const float4 *wc = w + ci * p.kh * p.kw;
        for (uint ky = 0; ky < p.kh; ky++) {
            int iy = iy0 + int(ky * p.dh);
            if (iy < 0 || iy >= int(p.h)) continue;
            device const float *row = plane + uint(iy) * p.iw;
            for (uint kx = 0; kx < p.kw; kx++) {
                int ix = ix0 + int(kx * p.dw);
                if (ix < 0 || ix >= int(p.iw)) continue;
                s += row[uint(ix)] * wc[ky * p.kw + kx];
            }
        }
    }
    conv_store(A, p, (z * p.oh + oy) * p.ow + ox, cob, s);
}

// Grouped convolutions that are neither depthwise nor one group: plain
// loops per output lane. Input plain (x_plain) or blocked; weights raw
// OIHW. Grid: (OW, OH, N * coutb).
kernel void k_conv_ref(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                     constant ConvP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.n * p.coutb) return;
    uint n = z / p.coutb;
    uint cob = z % p.coutb;
    uint cpg = p.cin / p.group;
    uint opg = p.cout / p.group;
    device const float *x = (device const float *)(A + p.x);
    device const float *w = K + p.wt;
    int iy0 = int(oy * p.sh) - int(p.pt);
    int ix0 = int(ox * p.sw) - int(p.pl);
    float r[4];
    for (uint l = 0; l < 4; l++) {
        uint co = cob * 4 + l;
        float s = 0.0f;
        if (co < p.cout) {
            uint grp = co / opg;
            for (uint cl = 0; cl < cpg; cl++) {
                uint ci = grp * cpg + cl;
                for (uint ky = 0; ky < p.kh; ky++) {
                    int iy = iy0 + int(ky * p.dh);
                    if (iy < 0 || iy >= int(p.h)) continue;
                    for (uint kx = 0; kx < p.kw; kx++) {
                        int ix = ix0 + int(kx * p.dw);
                        if (ix < 0 || ix >= int(p.iw)) continue;
                        uint xi = p.x_plain != 0
                            ? ((n * p.cin + ci) * p.h + uint(iy)) * p.iw + uint(ix)
                            : (((n * p.cinb + ci / 4) * p.h + uint(iy)) * p.iw + uint(ix)) * 4 + (ci & 3);
                        s += x[xi] * w[((co * cpg + cl) * p.kh + ky) * p.kw + kx];
                    }
                }
            }
        }
        r[l] = s;
    }
    float4 v = float4(r[0], r[1], r[2], r[3]) + ((device const float4 *)(K + p.bias))[cob];
    conv_store(A, p, (z * p.oh + oy) * p.ow + ox, cob, v);
}

// -------------------------------------------------------------- pooling

// Max or average pooling, blocked. Grid: (OW, OH, N * blocks).
kernel void k_pool(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                 constant PoolP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.planes) return;
    device const float4 *x = (device const float4 *)(A + p.in) + z * p.ih * p.iw;
    float4 acc = float4(p.init);
    uint cnt = 0;
    uint cnt_pad = 0;
    for (uint ky = 0; ky < p.kh; ky++) {
        int iy = int(oy * p.sh + ky * p.dh) - int(p.pt);
        bool in_pad_y = iy < int(p.ih + p.pb);
        for (uint kx = 0; kx < p.kw; kx++) {
            int ix = int(ox * p.sw + kx * p.dw) - int(p.pl);
            if (in_pad_y && ix < int(p.iw + p.pr)) cnt_pad++;
            if (iy < 0 || iy >= int(p.ih) || ix < 0 || ix >= int(p.iw)) continue;
            float4 v = x[uint(iy) * p.iw + uint(ix)];
            acc = p.is_max != 0 ? max(acc, v) : acc + v;
            cnt++;
        }
    }
    if (p.is_max == 0) {
        acc /= float(max(1u, p.incl_pad != 0 ? cnt_pad : cnt));
    } else if (cnt == 0) {
        acc = float4(0.0f);
    }
    ((device float4 *)(A + p.out))[(z * p.oh + oy) * p.ow + ox] = acc;
}

// Global average pooling, blocked: one thread per plane. Grid: (planes).
kernel void k_gap(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                constant GapP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint z = gid.x;
    if (z >= p.planes) return;
    device const float4 *x = (device const float4 *)(A + p.in) + z * p.hw;
    float4 acc = float4(0.0f);
    for (uint i = 0; i < p.hw; i++) acc += x[i];
    ((device float4 *)(A + p.out))[z] = acc * (1.0f / float(p.hw));
}

// Nearest or bilinear resize, blocked, with source maps computed on the
// host (as uint bits in K). Grid: (OW, OH, N * blocks).
kernel void k_resize(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                   constant ResizeP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint ox = gid.x;
    uint oy = gid.y;
    uint z = gid.z;
    if (ox >= p.ow || oy >= p.oh || z >= p.planes) return;
    device const float4 *x = (device const float4 *)(A + p.in) + z * p.ih * p.iw;
    device const uint *ym = (device const uint *)(K + p.ymap);
    device const uint *xm = (device const uint *)(K + p.xmap);
    device float4 *o = (device float4 *)(A + p.out) + (z * p.oh + oy) * p.ow + ox;
    if (p.linear == 0) {
        *o = x[ym[oy] * p.iw + xm[ox]];
        return;
    }
    uint y0 = ym[oy];
    uint y1 = min(y0 + 1, p.ih - 1);
    uint x0 = xm[ox];
    uint x1 = min(x0 + 1, p.iw - 1);
    float fy = K[p.yw + oy];
    float fx = K[p.xw + ox];
    device const float4 *r0 = x + y0 * p.iw;
    device const float4 *r1 = x + y1 * p.iw;
    float4 a = r0[x0] + (r0[x1] - r0[x0]) * fx;
    float4 b = r1[x0] + (r1[x1] - r1[x0]) * fx;
    *o = a + (b - a) * fy;
}

// ---------------------------------------------------------- element-wise

// Grid for all of these: (count).
kernel void k_act(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                constant ActP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    float v = act1(A[p.in + i], p.act, p.a0, p.a1);
    if (p.blocked != 0 && blk_chan(i, p.hw, p.nb) >= p.c) v = 0.0f;
    A[p.out + i] = v;
}

// y = x * scale[c] + shift[c] (scale and shift zero-padded to whole blocks).
kernel void k_affine(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                   constant AffineP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint ch = p.blocked != 0 ? blk_chan(i, p.inner, p.nb) : (i / p.inner) % p.c;
    A[p.out + i] = A[p.in + i] * K[p.scale + ch] + K[p.shift + ch];
}

// Same shapes and layout (operands may be constants).
kernel void k_bin_same(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                     constant BinP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    float v = binop(p.op, srcp(A, K, p.ak, p.a)[i], srcp(A, K, p.bk, p.b)[i]);
    if (p.blocked != 0 && blk_chan(i, p.hw, p.nb) >= p.c) v = 0.0f;
    A[p.out + i] = v;
}

// Blocked tensor (a, in A) with a per-channel constant vector (b, in K);
// const_first swaps the operands.
kernel void k_bin_pc(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                   constant BinP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint ch = blk_chan(i, p.hw, p.nb);
    float x = A[p.a + i];
    float c = K[p.b + ch];
    float v = p.const_first != 0 ? binop(p.op, c, x) : binop(p.op, x, c);
    if (ch >= p.c) v = 0.0f;
    A[p.out + i] = v;
}

// Plain numpy broadcasting: per output axis, the element stride of each
// operand (0 where it broadcasts).
kernel void k_bin_bc(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                   constant BinP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint rem = i;
    uint ia = 0;
    uint ib = 0;
    for (int d = int(p.rank) - 1; d >= 0; d--) {
        uint idx = rem % p.od[d];
        rem /= p.od[d];
        ia += idx * p.sa[d];
        ib += idx * p.sb[d];
    }
    A[p.out + i] = binop(p.op, srcp(A, K, p.ak, p.a)[ia], srcp(A, K, p.bk, p.b)[ib]);
}

// --------------------------------------------------------------- layouts

// Plain NCHW -> blocked. Grid: (N * nb * H * W * 4).
kernel void k_to_blocked(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                       constant LayoutP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint lane = i & 3;
    uint q = i >> 2;
    uint x = q % p.w;
    q /= p.w;
    uint y = q % p.h;
    q /= p.h;
    uint blk = q % p.nb;
    uint n = q / p.nb;
    uint ch = blk * 4 + lane;
    A[p.out + i] = ch < p.c ? srcp(A, K, p.ik, p.in)[((n * p.c + ch) * p.h + y) * p.w + x] : 0.0f;
}

// Blocked -> plain NCHW. Grid: (N * C * H * W).
kernel void k_to_plain(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                     constant LayoutP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint x = i % p.w;
    uint q = i / p.w;
    uint y = q % p.h;
    q /= p.h;
    uint ch = q % p.c;
    uint n = q / p.c;
    A[p.out + i] = srcp(A, K, p.ik, p.in)[(((n * p.nb + ch / 4) * p.h + y) * p.w + x) * 4 + (ch & 3)];
}

kernel void k_copy(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                 constant LayoutP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    A[p.out + i] = srcp(A, K, p.ik, p.in)[i];
}

// out[i * orow + at + j] = in[i * len + j]: concatenation, one dispatch per
// input. Grid: (len, outer).
kernel void k_copy2d(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                   constant Copy2dP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint j = gid.x;
    uint i = gid.y;
    if (j >= p.len || i >= p.outer) return;
    A[p.out + i * p.orow + p.at + j] = srcp(A, K, p.ik, p.in)[i * p.len + j];
}

// Plain transpose: od = output dims, is = input stride per output axis.
kernel void k_transpose(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                      constant TransposeP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint rem = i;
    uint src = 0;
    for (int d = int(p.rank) - 1; d >= 0; d--) {
        uint idx = rem % p.od[d];
        rem /= p.od[d];
        src += idx * p.is[d];
    }
    A[p.out + i] = srcp(A, K, p.ik, p.in)[src];
}

// Channels [c0, c0 + cn) of a blocked tensor, activated, to plain NHWC
// (detection-head outputs) or NCHW. Grid: (count).
kernel void k_extract(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                    constant ExtractP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.x;
    if (i >= p.count) return;
    uint x;
    uint y;
    uint c;
    uint n;
    if (p.nhwc != 0) {
        c = i % p.cn;
        uint q = i / p.cn;
        x = q % p.w;
        q /= p.w;
        y = q % p.h;
        n = q / p.h;
    } else {
        x = i % p.w;
        uint q = i / p.w;
        y = q % p.h;
        q /= p.h;
        c = q % p.cn;
        n = q / p.cn;
    }
    uint ch = p.c0 + c;
    float v = A[p.in + (((n * p.nb + ch / 4) * p.h + y) * p.w + x) * 4 + (ch & 3)];
    A[p.out + i] = act1(v, p.act, p.a0, p.a1);
}

// Softmax along one axis of a plain tensor. Grid: (inner, outer).
kernel void k_softmax(device float *A [[buffer(0)]], device const float *K [[buffer(1)]],
                    constant SoftmaxP &p [[buffer(2)]], uint3 gid [[thread_position_in_grid]]) {
    uint b = gid.x;
    uint a = gid.y;
    if (b >= p.inner || a >= p.outer) return;
    uint at = a * p.d * p.inner + b;
    device const float *x = (device const float *)(A + p.in) + at;
    device float *o = A + p.out + at;
    float m = x[0];
    for (uint i = 1; i < p.d; i++) m = max(m, x[i * p.inner]);
    float sum = 0.0f;
    for (uint i = 0; i < p.d; i++) {
        float e = exp(x[i * p.inner] - m);
        o[i * p.inner] = e;
        sum += e;
    }
    for (uint i = 0; i < p.d; i++) o[i * p.inner] /= sum;
}
