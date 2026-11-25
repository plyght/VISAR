//! Small dense linear algebra for calib3d: OpenCV's Jacobi eigen solver
//! (cv::eigen for symmetric matrices), its one-sided Jacobi SVD
//! (cv::SVD::compute), LU with partial pivoting (cv::solve DECOMP_LU),
//! Cholesky, pseudo-inverse solves and two Levenberg-Marquardt drivers
//! (OpenCV's LMSolver and CvLevMarq).
//!
//! All matrices are row-major `f64` slices. The eigen, SVD and LU routines
//! follow OpenCV's loops operation by operation, so the decompositions the
//! estimators build on round like OpenCV's.

const std = @import("std");

const dbl_eps = std.math.floatEps(f64);

/// OpenCV's MWC random number generator (cv::RNG), so RANSAC draws the
/// same samples as OpenCV for the same seed.
pub const Rng = struct {
    state: u64,

    pub fn init(seed: u64) Rng {
        return .{ .state = if (seed != 0) seed else 0xffffffff };
    }

    pub fn next(r: *Rng) u32 {
        r.state = @as(u64, @as(u32, @truncate(r.state))) *% 4164903690 +% (r.state >> 32);
        return @truncate(r.state);
    }

    /// Uniform integer in [a, b).
    pub fn uniform(r: *Rng, a: u32, b: u32) u32 {
        if (a == b) return a;
        return r.next() % (b - a) + a;
    }
};

/// Eigenvalues (descending) and eigenvectors (rows of `v`) of the
/// symmetric n x n matrix `a` (destroyed), by OpenCV's Jacobi rotations.
pub fn eigenSym(a: []f64, n: usize, w: []f64, v: ?[]f64) void {
    std.debug.assert(n <= 64);
    if (v) |vv| {
        @memset(vv[0 .. n * n], 0);
        for (0..n) |i| vv[i * n + i] = 1;
    }
    var ind_r: [64]usize = undefined;
    var ind_c: [64]usize = undefined;
    for (0..n) |k| {
        w[k] = a[(n + 1) * k];
        if (k < n - 1) {
            var m = k + 1;
            var mv = @abs(a[n * k + m]);
            for (k + 2..n) |i| {
                const val = @abs(a[n * k + i]);
                if (mv < val) {
                    mv = val;
                    m = i;
                }
            }
            ind_r[k] = m;
        }
        if (k > 0) {
            var m: usize = 0;
            var mv = @abs(a[k]);
            for (1..k) |i| {
                const val = @abs(a[n * i + k]);
                if (mv < val) {
                    mv = val;
                    m = i;
                }
            }
            ind_c[k] = m;
        }
    }
    if (n > 1) {
        const max_iters = n * n * 30;
        for (0..max_iters) |_| {
            var k: usize = 0;
            var mv = @abs(a[ind_r[0]]);
            for (1..n - 1) |i| {
                const val = @abs(a[n * i + ind_r[i]]);
                if (mv < val) {
                    mv = val;
                    k = i;
                }
            }
            var l = ind_r[k];
            for (1..n) |i| {
                const val = @abs(a[n * ind_c[i] + i]);
                if (mv < val) {
                    mv = val;
                    k = ind_c[i];
                    l = i;
                }
            }
            const p = a[n * k + l];
            // OpenCV stops only when the largest off-diagonal entry is below
            // DBL_EPSILON in absolute terms, which a matrix with large
            // entries never reaches (it then spins through all n²·30
            // rotations of rounding noise). Also stop once that entry is
            // negligible against the two diagonal entries it couples: the
            // remaining rotations move the eigenvectors by less than an ulp.
            if (@abs(p) <= dbl_eps or @abs(p) <= dbl_eps * 1e-2 * @max(@abs(w[k]), @abs(w[l]))) break;
            const y = (w[l] - w[k]) * 0.5;
            var t = @abs(y) + std.math.hypot(p, y);
            var s = std.math.hypot(p, t);
            const c = t / s;
            s = p / s;
            t = (p / t) * p;
            if (y < 0) {
                s = -s;
                t = -t;
            }
            a[n * k + l] = 0;
            w[k] -= t;
            w[l] += t;
            for (0..k) |i| rot(&a[n * i + k], &a[n * i + l], c, s);
            for (k + 1..l) |i| rot(&a[n * k + i], &a[n * i + l], c, s);
            for (l + 1..n) |i| rot(&a[n * k + i], &a[n * l + i], c, s);
            if (v) |vv| for (0..n) |i| rot(&vv[n * k + i], &vv[n * l + i], c, s);
            for ([2]usize{ k, l }) |idx| {
                if (idx < n - 1) {
                    var m = idx + 1;
                    var mv2 = @abs(a[n * idx + m]);
                    for (idx + 2..n) |i| {
                        const val = @abs(a[n * idx + i]);
                        if (mv2 < val) {
                            mv2 = val;
                            m = i;
                        }
                    }
                    ind_r[idx] = m;
                }
                if (idx > 0) {
                    var m: usize = 0;
                    var mv2 = @abs(a[idx]);
                    for (1..idx) |i| {
                        const val = @abs(a[n * i + idx]);
                        if (mv2 < val) {
                            mv2 = val;
                            m = i;
                        }
                    }
                    ind_c[idx] = m;
                }
            }
        }
    }
    // Sort descending.
    for (0..n -| 1) |k| {
        var m = k;
        for (k + 1..n) |i| {
            if (w[m] < w[i]) m = i;
        }
        if (k != m) {
            std.mem.swap(f64, &w[m], &w[k]);
            if (v) |vv| for (0..n) |i| std.mem.swap(f64, &vv[n * m + i], &vv[n * k + i]);
        }
    }
}

inline fn rot(v0: *f64, v1: *f64, c: f64, s: f64) void {
    const a0 = v0.*;
    const b0 = v1.*;
    v0.* = a0 * c - b0 * s;
    v1.* = a0 * s + b0 * c;
}

/// OpenCV's JacobiSVDImpl_: `at` holds the n x m transpose of an m x n
/// matrix (m >= n) and is overwritten by the first `n1` left singular
/// vectors (as rows); `w` receives the n singular values (descending) and
/// `vt` (n x n) the right singular vectors as rows.
pub fn jacobiSvd(at: []f64, m: usize, n: usize, n1: usize, w: []f64, vt: ?[]f64) void {
    const eps = dbl_eps * 10;
    const minval = std.math.floatMin(f64);
    const max_iter = @max(m, 30);
    var wb: [64]f64 = undefined;
    std.debug.assert(n <= 64);
    for (0..n) |i| {
        var sd: f64 = 0;
        for (0..m) |k| sd += at[i * m + k] * at[i * m + k];
        wb[i] = sd;
        if (vt) |v| {
            @memset(v[i * n ..][0..n], 0);
            v[i * n + i] = 1;
        }
    }
    for (0..max_iter) |_| {
        var changed = false;
        for (0..n -| 1) |i| for (i + 1..n) |j| {
            const ai = at[i * m ..][0..m];
            const aj = at[j * m ..][0..m];
            var a = wb[i];
            var b = wb[j];
            var p: f64 = 0;
            for (0..m) |k| p += ai[k] * aj[k];
            if (@abs(p) <= eps * @sqrt(a * b)) continue;
            p *= 2;
            const beta = a - b;
            const gamma = std.math.hypot(p, beta);
            var c: f64 = undefined;
            var s: f64 = undefined;
            if (beta < 0) {
                const delta = (gamma - beta) * 0.5;
                s = @sqrt(delta / gamma);
                c = p / (gamma * s * 2);
            } else {
                c = @sqrt((gamma + beta) / (gamma * 2));
                s = p / (gamma * c * 2);
            }
            a = 0;
            b = 0;
            for (0..m) |k| {
                const t0 = c * ai[k] + s * aj[k];
                const t1 = -s * ai[k] + c * aj[k];
                ai[k] = t0;
                aj[k] = t1;
                a += t0 * t0;
                b += t1 * t1;
            }
            wb[i] = a;
            wb[j] = b;
            changed = true;
            if (vt) |v| {
                const vi = v[i * n ..][0..n];
                const vj = v[j * n ..][0..n];
                for (0..n) |k| {
                    const t0 = c * vi[k] + s * vj[k];
                    const t1 = -s * vi[k] + c * vj[k];
                    vi[k] = t0;
                    vj[k] = t1;
                }
            }
        };
        if (!changed) break;
    }
    for (0..n) |i| {
        var sd: f64 = 0;
        for (0..m) |k| sd += at[i * m + k] * at[i * m + k];
        wb[i] = @sqrt(sd);
    }
    for (0..n -| 1) |i| {
        var j = i;
        for (i + 1..n) |k| {
            if (wb[j] < wb[k]) j = k;
        }
        if (i != j) {
            std.mem.swap(f64, &wb[i], &wb[j]);
            if (vt) |v| {
                for (0..m) |k| std.mem.swap(f64, &at[i * m + k], &at[j * m + k]);
                for (0..n) |k| std.mem.swap(f64, &v[i * n + k], &v[j * n + k]);
            }
        }
    }
    @memcpy(w[0..n], wb[0..n]);
    if (vt == null) return;
    var rng = Rng.init(0x12345678);
    for (0..n1) |i| {
        var sd: f64 = if (i < n) wb[i] else 0;
        var ii: usize = 0;
        while (ii < 100 and sd <= minval) : (ii += 1) {
            const val0 = 1.0 / @as(f64, @floatFromInt(m));
            for (0..m) |k| at[i * m + k] = if (rng.next() & 256 != 0) val0 else -val0;
            for (0..2) |_| {
                for (0..i) |j| {
                    sd = 0;
                    for (0..m) |k| sd += at[i * m + k] * at[j * m + k];
                    var asum: f64 = 0;
                    for (0..m) |k| {
                        const t = at[i * m + k] - sd * at[j * m + k];
                        at[i * m + k] = t;
                        asum += @abs(t);
                    }
                    asum = if (asum > eps * 100) 1 / asum else 0;
                    for (0..m) |k| at[i * m + k] *= asum;
                }
            }
            sd = 0;
            for (0..m) |k| sd += at[i * m + k] * at[i * m + k];
            sd = @sqrt(sd);
        }
        const s = if (sd > minval) 1 / sd else 0;
        for (0..m) |k| at[i * m + k] *= s;
    }
}

/// SVD of a fixed-size M x N matrix as cv::SVD::compute (thin U).
pub fn Svd(comptime M: usize, comptime N: usize) type {
    const m: usize = @max(M, N);
    const n: usize = @min(M, N);
    return struct {
        w: [n]f64,
        /// M x n (left singular vectors as columns).
        u: [M][n]f64,
        /// n x N (right singular vectors as rows).
        vt: [n][N]f64,

        pub fn compute(a: [M][N]f64) @This() {
            var at: [n * m]f64 = undefined;
            var v: [n * n]f64 = undefined;
            var r: @This() = undefined;
            const transposed = M < N;
            if (!transposed) {
                for (0..M) |i| for (0..N) |j| {
                    at[j * m + i] = a[i][j];
                };
            } else {
                for (0..M) |i| for (0..N) |j| {
                    at[i * m + j] = a[i][j];
                };
            }
            jacobiSvd(&at, m, n, n, &r.w, &v);
            if (!transposed) {
                for (0..M) |i| for (0..n) |j| {
                    r.u[i][j] = at[j * m + i];
                };
                for (0..n) |i| for (0..N) |j| {
                    r.vt[i][j] = v[i * n + j];
                };
            } else {
                for (0..M) |i| for (0..n) |j| {
                    r.u[i][j] = v[j * n + i];
                };
                for (0..n) |i| for (0..N) |j| {
                    r.vt[i][j] = at[i * m + j];
                };
            }
            return r;
        }
    };
}

/// Solves A x = b (A is n x n, b n x k; both overwritten, x in b) by
/// OpenCV's LU with partial pivoting. Returns false for a singular A.
pub fn solveLU(a: []f64, n: usize, b: []f64, k: usize) bool {
    const eps = dbl_eps * 100;
    for (0..n) |i| {
        var p = i;
        for (i + 1..n) |j| {
            if (@abs(a[j * n + i]) > @abs(a[p * n + i])) p = j;
        }
        if (@abs(a[p * n + i]) < eps) return false;
        if (p != i) {
            for (i..n) |j| std.mem.swap(f64, &a[i * n + j], &a[p * n + j]);
            for (0..k) |j| std.mem.swap(f64, &b[i * k + j], &b[p * k + j]);
        }
        const d = -1 / a[i * n + i];
        for (i + 1..n) |j| {
            const alpha = a[j * n + i] * d;
            for (i + 1..n) |c| a[j * n + c] += alpha * a[i * n + c];
            for (0..k) |c| b[j * k + c] += alpha * b[i * k + c];
        }
    }
    var i = n;
    while (i > 0) {
        i -= 1;
        for (0..k) |j| {
            var s = b[i * k + j];
            for (i + 1..n) |c| s -= a[i * n + c] * b[c * k + j];
            b[i * k + j] = s / a[i * n + i];
        }
    }
    return true;
}

/// Cholesky solve of the symmetric positive definite n x n system A x = b
/// (A overwritten; x in b). Returns false when A is not positive definite.
pub fn solveCholesky(a: []f64, n: usize, b: []f64) bool {
    for (0..n) |j| {
        var d = a[j * n + j];
        for (0..j) |k| d -= a[j * n + k] * a[j * n + k];
        if (!(d > 0)) return false;
        const l = @sqrt(d);
        a[j * n + j] = l;
        const il = 1 / l;
        for (j + 1..n) |i| {
            var s = a[i * n + j];
            for (0..j) |k| s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s * il;
        }
    }
    for (0..n) |i| {
        var s = b[i];
        for (0..i) |k| s -= a[i * n + k] * b[k];
        b[i] = s / a[i * n + i];
    }
    var i = n;
    while (i > 0) {
        i -= 1;
        var s = b[i];
        for (i + 1..n) |k| s -= a[k * n + i] * b[k];
        b[i] = s / a[i * n + i];
    }
    return true;
}

/// Pseudo-inverse solve of a symmetric n x n system through its eigen
/// decomposition (cv::solve DECOMP_EIG): eigenvalues below
/// 2 eps * sum|w| are dropped. `a` is destroyed; x in `x`.
pub fn solveSymEig(a: []f64, n: usize, b: []const f64, x: []f64) void {
    var w: [64]f64 = undefined;
    var v: [64 * 64]f64 = undefined;
    eigenSym(a, n, w[0..n], v[0 .. n * n]);
    var thr: f64 = 0;
    for (w[0..n]) |e| thr += @abs(e);
    thr *= dbl_eps * 2;
    @memset(x[0..n], 0);
    for (0..n) |i| {
        if (@abs(w[i]) <= thr) continue;
        var d: f64 = 0;
        for (0..n) |k| d += v[i * n + k] * b[k];
        d /= w[i];
        for (0..n) |k| x[k] += d * v[i * n + k];
    }
}

/// Inverse of a 3x3 matrix as cv::Matx33d::inv (DECOMP_LU takes the
/// cofactor formula for 3x3).
pub fn inv3(a: [9]f64) ?[9]f64 {
    const det = a[0] * (a[4] * a[8] - a[7] * a[5]) - a[1] * (a[3] * a[8] - a[6] * a[5]) + a[2] * (a[3] * a[7] - a[6] * a[4]);
    if (det == 0) return null;
    const d = 1 / det;
    return .{
        (a[4] * a[8] - a[5] * a[7]) * d, (a[2] * a[7] - a[1] * a[8]) * d, (a[1] * a[5] - a[2] * a[4]) * d,
        (a[5] * a[6] - a[3] * a[8]) * d, (a[0] * a[8] - a[2] * a[6]) * d, (a[2] * a[3] - a[0] * a[5]) * d,
        (a[3] * a[7] - a[4] * a[6]) * d, (a[1] * a[6] - a[0] * a[7]) * d, (a[0] * a[4] - a[1] * a[3]) * d,
    };
}

pub fn det3(a: [9]f64) f64 {
    return a[0] * (a[4] * a[8] - a[7] * a[5]) - a[1] * (a[3] * a[8] - a[6] * a[5]) + a[2] * (a[3] * a[7] - a[6] * a[4]);
}

pub fn mul3(a: [9]f64, b: [9]f64) [9]f64 {
    var r: [9]f64 = undefined;
    for (0..3) |i| for (0..3) |j| {
        r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    };
    return r;
}

pub fn transpose3(a: [9]f64) [9]f64 {
    return .{ a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8] };
}

// ------------------------------------------------------------ LM solvers

/// OpenCV's LMSolver (calib3d levmarq.cpp, used by findHomography and
/// estimateAffine2D refinement). `cb.normal(x, A, v)` fills JᵀJ (np x np)
/// and Jᵀr and returns the residual's (sum of squares, max |r|);
/// `cb.cost(x)` returns the same pair without derivatives. Returns the
/// iteration count. np <= 16.
pub fn lmSolve(np: usize, x0: []f64, max_iters: usize, cb: anytype) usize {
    const eps = std.math.floatEps(f32);
    var x: [16]f64 = undefined;
    var xd: [16]f64 = undefined;
    var a: [256]f64 = undefined;
    var ap: [256]f64 = undefined;
    var v: [16]f64 = undefined;
    var d: [16]f64 = undefined;
    var diag: [16]f64 = undefined;
    @memcpy(x[0..np], x0[0..np]);
    var res = cb.normal(x[0..np], a[0 .. np * np], v[0..np]);
    var s = res[0];
    for (0..np) |i| diag[i] = a[i * np + i];
    const rlo = 0.25;
    const rhi = 0.75;
    var lambda: f64 = 1;
    var lc: f64 = 0.75;
    var iter: usize = 0;
    while (true) {
        @memcpy(ap[0 .. np * np], a[0 .. np * np]);
        for (0..np) |i| ap[i * np + i] += lambda * diag[i];
        // OpenCV solves with DECOMP_EIG; the damped normal matrix is
        // positive definite in practice, where Cholesky gives the same step
        // far faster. Fall back to the eigen pseudo-inverse otherwise.
        var ch: [256]f64 = undefined;
        @memcpy(ch[0 .. np * np], ap[0 .. np * np]);
        @memcpy(d[0..np], v[0..np]);
        if (!solveCholesky(ch[0 .. np * np], np, d[0..np])) solveSymEig(ap[0 .. np * np], np, v[0..np], d[0..np]);
        for (0..np) |i| xd[i] = x[i] - d[i];
        const sd = cb.cost(xd[0..np])[0];
        // dS = d . (2 v - A d)
        var ds: f64 = 0;
        for (0..np) |i| {
            var ad: f64 = 0;
            for (0..np) |k| ad += a[i * np + k] * d[k];
            ds += d[i] * (2 * v[i] - ad);
        }
        const r = (s - sd) / (if (@abs(ds) > dbl_eps) ds else 1);
        if (r > rhi) {
            lambda *= 0.5;
            if (lambda < lc) lambda = 0;
        } else if (r < rlo) {
            var t: f64 = 0;
            for (0..np) |i| t += d[i] * v[i];
            var nu = (sd - s) / (if (@abs(t) > dbl_eps) t else 1) + 2;
            nu = @min(@max(nu, 2), 10);
            if (lambda == 0) {
                // lambda = lc = 1 / max |diag(A^-1)|
                var inv_diag: f64 = dbl_eps;
                var tmp: [256]f64 = undefined;
                var w: [16]f64 = undefined;
                var vv: [256]f64 = undefined;
                @memcpy(tmp[0 .. np * np], a[0 .. np * np]);
                eigenSym(tmp[0 .. np * np], np, w[0..np], vv[0 .. np * np]);
                var thr: f64 = 0;
                for (w[0..np]) |e| thr += @abs(e);
                thr *= dbl_eps * 2;
                for (0..np) |i| {
                    var dd: f64 = 0;
                    for (0..np) |k| {
                        if (@abs(w[k]) > thr) dd += vv[k * np + i] * vv[k * np + i] / w[k];
                    }
                    inv_diag = @max(inv_diag, @abs(dd));
                }
                lambda = 1 / inv_diag;
                lc = lambda;
                nu *= 0.5;
            }
            lambda *= nu;
        }
        if (sd < s) {
            s = sd;
            @memcpy(x[0..np], xd[0..np]);
            res = cb.normal(x[0..np], a[0 .. np * np], v[0..np]);
        }
        iter += 1;
        var dinf: f64 = 0;
        for (d[0..np]) |e| dinf = @max(dinf, @abs(e));
        const proceed = iter < max_iters and dinf >= eps and res[1] >= eps;
        if (!proceed) break;
    }
    @memcpy(x0[0..np], x[0..np]);
    return iter;
}

/// OpenCV's CvLevMarq (solvePnP ITERATIVE, calibrateCamera): JᵀJ damped by
/// (1 + 10^λ) on the diagonal, λ stepped by decades, stopping after
/// `max_iters` or when the relative parameter change drops below `eps`.
/// `cb.normal(x, A, v)` fills JᵀJ and Jᵀr and returns Σr²; `cb.cost(x)`
/// returns Σr². `work` needs 3 np² + 4 np floats. `mask` (optional) fixes
/// parameters whose entry is false.
pub fn levMarq(np: usize, x: []f64, max_iters: usize, eps: f64, work: []f64, mask: ?[]const bool, cb: anytype) void {
    const a = work[0 .. np * np];
    const ap = work[np * np ..][0 .. np * np];
    const v = work[2 * np * np ..][0..np];
    const d = work[2 * np * np + np ..][0..np];
    const prev = work[2 * np * np + 2 * np ..][0..np];
    const vv = work[2 * np * np + 3 * np ..][0..np];
    const sub = work[2 * np * np + 4 * np ..][0 .. np * np];
    var lambda_lg10: f64 = -3;
    var err = cb.normal(x, a, v);
    var iters: usize = 0;
    while (true) {
        @memcpy(prev, x);
        // Solve (A + λ diag A) d = v on the free parameters.
        const lambda = std.math.pow(f64, 10, lambda_lg10);
        var nf: usize = 0;
        var idx: [512]usize = undefined;
        for (0..np) |i| {
            if (mask == null or mask.?[i]) {
                idx[nf] = i;
                nf += 1;
            }
        }
        for (0..nf) |i| {
            for (0..nf) |j| ap[i * nf + j] = a[idx[i] * np + idx[j]];
            ap[i * nf + i] *= 1 + lambda;
            vv[i] = v[idx[i]];
        }
        @memcpy(sub[0 .. nf * nf], ap[0 .. nf * nf]);
        if (!solveCholesky(ap[0 .. nf * nf], nf, vv[0..nf])) {
            if (nf <= 64) {
                var tmp: [64]f64 = undefined;
                for (0..nf) |i| tmp[i] = v[idx[i]];
                solveSymEig(sub[0 .. nf * nf], nf, tmp[0..nf], vv[0..nf]);
            } else @memset(vv[0..nf], 0);
        }
        @memset(d, 0);
        for (0..nf) |i| d[idx[i]] = vv[i];
        for (0..np) |i| x[i] = prev[i] - d[i];
        const e2 = cb.cost(x);
        if (e2 > err) {
            // Reject: raise λ and retry from the previous parameters.
            lambda_lg10 = @min(lambda_lg10 + 1, 16);
            @memcpy(x, prev);
            iters += 1;
            if (iters >= max_iters) break;
            continue;
        }
        lambda_lg10 = @max(lambda_lg10 - 1, -16);
        iters += 1;
        var dn: f64 = 0;
        var xn: f64 = 0;
        for (0..np) |i| {
            dn += (x[i] - prev[i]) * (x[i] - prev[i]);
            xn += x[i] * x[i];
        }
        const change = @sqrt(dn) / (@sqrt(xn) + dbl_eps);
        if (iters >= max_iters or change < eps) break;
        err = cb.normal(x, a, v);
    }
}

test "eigenSym and svd" {
    var a = [_]f64{ 4, 1, 2, 1, 3, 0, 2, 0, 5 };
    const orig = a;
    var w: [3]f64 = undefined;
    var v: [9]f64 = undefined;
    eigenSym(&a, 3, &w, &v);
    for (0..3) |k| for (0..3) |i| {
        var s: f64 = 0;
        for (0..3) |j| s += orig[i * 3 + j] * v[k * 3 + j];
        try std.testing.expectApproxEqAbs(w[k] * v[k * 3 + i], s, 1e-12);
    };
    try std.testing.expect(w[0] >= w[1] and w[1] >= w[2]);
    const m = [3][2]f64{ .{ 1, 2 }, .{ 3, 4 }, .{ 5, 6 } };
    const r = Svd(3, 2).compute(m);
    for (0..3) |i| for (0..2) |j| {
        var s: f64 = 0;
        for (0..2) |k| s += r.u[i][k] * r.w[k] * r.vt[k][j];
        try std.testing.expectApproxEqAbs(m[i][j], s, 1e-12);
    };
    var lu = [_]f64{ 2, 1, 1, 3 };
    var b = [_]f64{ 3, 5 };
    try std.testing.expect(solveLU(&lu, 2, &b, 1));
    try std.testing.expectApproxEqAbs(@as(f64, 0.8), b[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.4), b[1], 1e-12);
}
