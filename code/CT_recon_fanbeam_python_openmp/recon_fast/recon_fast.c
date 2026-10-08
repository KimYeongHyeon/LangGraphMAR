// recon_fast: drop-in C extension for the LangGraph-MAR fan-beam projectors (opt-in; the original recon.c is untouched).
//
// Same operator and the same float32 arithmetic as recon.c; only the execution order of independent work changes.
//   FPd  forward projection. Bit-identical to the original FP for every thread count (FP has no reduction).
//   BPd  backward projection. The original BP is `omp parallel for reduction(+:img[:nx*ny])` with a static schedule, so
//        thread t accumulates one contiguous chunk of rays into a private float32 image and the copies are added in
//        thread-arrival order (non-deterministic). BPd reproduces the same P chunks (P = the original team size) bit for
//        bit and adds them in a fixed order 0..P-1, so its output is one member of the original's run-to-run result set
//        and does not depend on how many threads run it.
//   BPparts  the P unfolded partial images, for the equivalence check in code/scripts/verify_fastproj.py.
// mode bits for FPd/BPd: 1 = clip each ray to the iy range that can pass the original bounds test (skipped samples add
// nothing), 2 = compute coordinates in a separate vectorizable loop, 4 (BPd only) = also weights/offsets in the vector
// loop. Used by utils/fastproj.py: FPd mode 3, BPd mode 5.
// Build with -O3 -mavx2 -mno-fma -ffp-contract=off (see setup_fast.py): FMA contraction would change the rounding.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

static long n_fp = 0, n_bp = 0;

/* ---------- deterministic BP that reproduces the original reduction exactly ----------
 * Original: `omp parallel for reduction(+:img[:n])`, static schedule over ii = 0..nu*nview-1 with T threads ->
 * thread t accumulates its contiguous chunk (libgomp/GCC static formula) into a zeroed private copy in ii, iy order,
 * then copies are added into img (=0) one after another under GOMP_atomic in completion order (nondeterministic).
 * Here: P chunks with the same formula (P = the original team size, independent of how many threads run them),
 * each chunk accumulated in the same order into its own buffer, then folded in the fixed order 0..P-1
 * -> bit-identical to the original run whose merge order happened to be 0..P-1 (one member of its run-to-run set).
 * mode bit 1 (clip): per ray, iterate only the iy range where the sample can pass the original bounds test
 *   (rx, ry are affine in iy; range widened by 2 samples, exact original test kept) -> skipped samples add nothing.
 *   Rays with sino==0 are skipped (adds +-0 only, value-preserving).
 * mode bit 4 (split2): also weights/offsets/valid flags in the vector loop, only the 4 adds scalar.
 * mode bit 2 (split): coordinates of a ray's samples computed in a separate loop (vectorizable, same IEEE ops),
 *   then scattered in iy order. prof (optional, 4 doubles): alloc+zero, ray loop, fold, samples visited. */
static void chunk_of(long n, int P, int c, long* s, long* e) {
    long q = n / P, t = n % P;
    if (c < t) { t = 0; q++; }
    *s = q * c + t; *e = *s + q;
}

static void iy_range(double t, float sinval, float cosval, float dso, int nx, int ny, float dx, float dy, int* lo, int* hi) {
    // rx(iy) = (posx c + posy s)/dx + cxn, ry(iy) = (-posx s + posy c)/dy + cyn, posx = t (posy + dso), posy = (iy - cyn) dy
    const double c = cosval, s = sinval, cxn = (nx - 1.0) / 2.0, cyn = (ny - 1.0) / 2.0;
    const double kx = (t * c + s) * dy / dx, bx = t * dso * c / dx + cxn - kx * cyn;      // rx = bx + kx*iy
    const double ky = (c - t * s), by = -t * dso * s / dy + cyn - ky * cyn;               // ry = by + ky*iy
    double l = -1e30, h = 1e30;
    const double kk[2] = {kx, ky}, bb[2] = {bx, by}, up[2] = {nx - 1.0, ny - 1.0};
    for (int d = 0; d < 2; d++) {
        if (fabs(kk[d]) < 1e-12) {
            if (bb[d] <= -1e-3 || bb[d] >= up[d] + 1e-3) { *lo = 1; *hi = 0; return; }
            continue;
        }
        double a = (0.0 - bb[d]) / kk[d], b = (up[d] - bb[d]) / kk[d];
        if (a > b) { double tmp = a; a = b; b = tmp; }
        if (a > l) l = a;
        if (b < h) h = b;
    }
    if (l > h) { *lo = 1; *hi = 0; return; }
    int L = (int) floor(l) - 2, H = (int) ceil(h) + 2;
    *lo = L < 0 ? 0 : L;
    *hi = H > ny - 1 ? ny - 1 : H;
}

static void bp_det_kernel(const float* sino, const double* deg, int nview, float dso, int nx, int ny, float dx, float dy,
                          int nu, float da, float off_a, float* img, int P, int mode, double* prof) {
    const int clip = mode & 1, split = mode & 2, split2 = mode & 4;
    const size_t npix = (size_t) nx * ny;
    const long n = (long) nu * nview;
    double t0 = omp_get_wtime();
    float* sv = malloc(nview * sizeof(float));
    float* cv = malloc(nview * sizeof(float));
    double* ta = malloc(nu * sizeof(double));
    float* dist = malloc(nu * sizeof(float));
    for (int v = 0; v < nview; v++) {
        sv[v] = (float) sin(deg[v] / 180.0f * 3.141592f);
        cv[v] = (float) cos(deg[v] / 180.0f * 3.141592f);
    }
    for (int iu = 0; iu < nu; iu++) {
        float a = ((iu - (nu - 1.0f) / 2.0f) - off_a) * da;   // == original (ii%nu) expression
        ta[iu] = tan(a);
        dist[iu] = dy / cos(a);
    }
    float* part = calloc((size_t) P * npix, sizeof(float));
    double t1 = omp_get_wtime();
    long visited = 0;
    int c;
    #pragma omp parallel for schedule(dynamic, 1) reduction(+:visited)
    for (c = 0; c < P; c++) {
        float* pb = part + (size_t) c * npix;
        float* RX = malloc(ny * sizeof(float));
        float* RY = malloc(ny * sizeof(float));
        float *W0 = malloc(ny * sizeof(float)), *W1 = malloc(ny * sizeof(float)), *W2 = malloc(ny * sizeof(float)), *W3 = malloc(ny * sizeof(float));
        int *OK = malloc(ny * sizeof(int)), *OF = malloc(ny * sizeof(int));
        long s, e;
        chunk_of(n, P, c, &s, &e);
        for (long ii = s; ii < e; ii++) {
            const int iview = (int) (ii / nu), iu = (int) (ii % nu);
            const float sinval = sv[iview], cosval = cv[iview];
            const float val = sino[ii] * dist[iu];
            int lo = 0, hi = ny - 1;
            if (clip) {
                if (sino[ii] == 0.0f) continue;
                iy_range(ta[iu], sinval, cosval, dso, nx, ny, dx, dy, &lo, &hi);
            }
            visited += hi - lo + 1 > 0 ? hi - lo + 1 : 0;
            if (split2) {        // weights and offsets also in the vector loop; only the 4 adds stay scalar (same values, same order)
                const double t = ta[iu];
                for (int iy = lo; iy <= hi; iy++) {
                    float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                    float posx = t * (posy + dso);
                    float x = posx * cosval + posy * sinval;
                    float y = -posx * sinval + posy * cosval;
                    float rx = (x) / dx + (nx - 1.0f) / 2.0f;
                    float ry = (y) / dy + (ny - 1.0f) / 2.0f;
                    int ix = (int) rx, iyy = (int) ry;
                    float wx = rx - ix, wy = ry - iyy;
                    OK[iy] = (rx > 0) & (rx < nx - 1) & (ry > 0) & (ry < ny - 1);
                    OF[iy] = iyy * nx + ix;
                    W0[iy] = (1.0f - wx) * (1.0f - wy) * val;
                    W1[iy] = (wx) * (1.0f - wy) * val;
                    W2[iy] = (1.0f - wx) * (wy) * val;
                    W3[iy] = (wx) * (wy) * val;
                }
                for (int iy = lo; iy <= hi; iy++) {
                    if (OK[iy]) {
                        float* q = pb + OF[iy];
                        q[0] += W0[iy]; q[1] += W1[iy]; q[nx] += W2[iy]; q[nx + 1] += W3[iy];
                    }
                }
            } else if (split) {
                const double t = ta[iu];
                for (int iy = lo; iy <= hi; iy++) {       // vectorizable: elementwise IEEE ops identical to the scalar path
                    float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                    float posx = t * (posy + dso);
                    float x = posx * cosval + posy * sinval;
                    float y = -posx * sinval + posy * cosval;
                    RX[iy] = (x) / dx + (nx - 1.0f) / 2.0f;
                    RY[iy] = (y) / dy + (ny - 1.0f) / 2.0f;
                }
                for (int iy = lo; iy <= hi; iy++) {
                    const float rx = RX[iy], ry = RY[iy];
                    if (rx > 0 && rx < nx - 1 && ry > 0 && ry < ny - 1) {
                        float wx = rx - ((int) rx), wy = ry - ((int) ry);
                        float* q = pb + ((int) ry) * nx + ((int) rx);
                        q[0] += (1.0f - wx) * (1.0f - wy) * val;
                        q[1] += (wx) * (1.0f - wy) * val;
                        q[nx] += (1.0f - wx) * (wy) * val;
                        q[nx + 1] += (wx) * (wy) * val;
                    }
                }
            } else {
                for (int iy = lo; iy <= hi; iy++) {
                    float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                    float posx = ta[iu] * (posy + dso);
                    float x = posx * cosval + posy * sinval;
                    float y = -posx * sinval + posy * cosval;
                    float rx = (x) / dx + (nx - 1.0f) / 2.0f;
                    float ry = (y) / dy + (ny - 1.0f) / 2.0f;
                    if (rx > 0 && rx < nx - 1 && ry > 0 && ry < ny - 1) {
                        float wx = rx - ((int) rx);
                        float wy = ry - ((int) ry);
                        pb[((int) ry) * nx + ((int) rx)] += (1.0f - wx) * (1.0f - wy) * val;
                        pb[((int) ry) * nx + ((int) rx + 1)] += (wx) * (1.0f - wy) * val;
                        pb[((int) ry + 1) * nx + ((int) rx)] += (1.0f - wx) * (wy) * val;
                        pb[((int) ry + 1) * nx + ((int) rx + 1)] += (wx) * (wy) * val;
                    }
                }
            }
        }
        free(RX); free(RY); free(W0); free(W1); free(W2); free(W3); free(OK); free(OF);
    }
    double t2 = omp_get_wtime();
    size_t j;
    #pragma omp parallel for schedule(static)
    for (j = 0; j < npix; j++) {      // fixed fold order 0..P-1 (original: 0 + p_first + p_second ...; 0 + p == p)
        float acc = part[j];
        for (int k = 1; k < P; k++) acc += part[(size_t) k * npix + j];
        img[j] = acc;
    }
    double t3 = omp_get_wtime();
    if (prof) { prof[0] = t1 - t0; prof[1] = t2 - t1; prof[2] = t3 - t2; prof[3] = (double) visited; }
    free(part); free(sv); free(cv); free(ta); free(dist);
}

// FP with the same clip/split options. Per ray the sum runs over iy ascending exactly as the original; skipped samples add nothing.
static void fp_det_kernel(const float* obj, const double* deg, int nview, float dso, int nx, int ny, float dx, float dy,
                          int nu, float da, float off_a, float* sino, int mode) {
    const int clip = mode & 1, split = mode & 2;
    float* sv = malloc(nview * sizeof(float));
    float* cv = malloc(nview * sizeof(float));
    double* ta = malloc(nu * sizeof(double));
    float* dist = malloc(nu * sizeof(float));
    for (int v = 0; v < nview; v++) {
        sv[v] = (float) sin(deg[v] / 180.0f * 3.141592f);
        cv[v] = (float) cos(deg[v] / 180.0f * 3.141592f);
    }
    for (int iu = 0; iu < nu; iu++) {
        float a = ((iu - (nu - 1.0f) / 2.0f) - off_a) * da;
        ta[iu] = tan(a);
        dist[iu] = dy / cos(a);
    }
    #pragma omp parallel
    {
        float *W0 = malloc(ny * sizeof(float)), *W1 = malloc(ny * sizeof(float)), *W2 = malloc(ny * sizeof(float)), *W3 = malloc(ny * sizeof(float));
        int *OK = malloc(ny * sizeof(int)), *OF = malloc(ny * sizeof(int));
        long ii;
        #pragma omp for schedule(static)
        for (ii = 0; ii < (long) nu * nview; ii++) {
            const int iview = (int) (ii / nu), iu = (int) (ii % nu);
            const float sinval = sv[iview], cosval = cv[iview], dst = dist[iu];
            float temp = 0.0f;
            int lo = 0, hi = ny - 1;
            if (clip) iy_range(ta[iu], sinval, cosval, dso, nx, ny, dx, dy, &lo, &hi);
            if (split) {
                const double t = ta[iu];
                for (int iy = lo; iy <= hi; iy++) {
                    float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                    float posx = t * (posy + dso);
                    float rx = (posx * cosval + posy * sinval) / dx + (nx - 1.0f) / 2.0f;
                    float ry = (-posx * sinval + posy * cosval) / dy + (ny - 1.0f) / 2.0f;
                    int ix = (int) rx, iyy = (int) ry;
                    float wx = rx - ix, wy = ry - iyy;
                    OK[iy] = (rx > 0) & (rx < nx - 1) & (ry > 0) & (ry < ny - 1);
                    OF[iy] = iyy * nx + ix;
                    W0[iy] = (1.0f - wx) * (1.0f - wy); W1[iy] = (wx) * (1.0f - wy); W2[iy] = (1.0f - wx) * (wy); W3[iy] = (wx) * (wy);
                }
                for (int iy = lo; iy <= hi; iy++) {
                    if (OK[iy]) {
                        const float* q = obj + OF[iy];
                        temp += (W0[iy] * q[0] + W1[iy] * q[1] + W2[iy] * q[nx] + W3[iy] * q[nx + 1]) * dst;
                    }
                }
            } else {
                for (int iy = lo; iy <= hi; iy++) {
                    float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                    float posx = ta[iu] * (posy + dso);
                    float rx = (posx * cosval + posy * sinval) / dx + (nx - 1.0f) / 2.0f;
                    float ry = (-posx * sinval + posy * cosval) / dy + (ny - 1.0f) / 2.0f;
                    if (rx > 0 && rx < nx - 1 && ry > 0 && ry < ny - 1) {
                        float wx = rx - ((int) rx);
                        float wy = ry - ((int) ry);
                        temp += ((1.0f - wx) * (1.0f - wy) * obj[((int) ry) * nx + ((int) rx)] + (wx) * (1.0f - wy) * obj[((int) ry) * nx + ((int) rx + 1)] +
                                 (1.0f - wx) * (wy) * obj[((int) ry + 1) * nx + ((int) rx)] + (wx) * (wy) * obj[((int) ry + 1) * nx + ((int) rx + 1)]) * dst;
                    }
                }
            }
            sino[ii] = temp;
        }
        free(W0); free(W1); free(W2); free(W3); free(OK); free(OF);
    }
    free(sv); free(cv); free(ta); free(dist);
}

// partial buffers only (no fold) — for the permutation identity test against the original reduction
static void bp_parts_kernel(const float* sino, const double* deg, int nview, float dso, int nx, int ny, float dx, float dy,
                            int nu, float da, float off_a, float* parts, int P) {
    const size_t npix = (size_t) nx * ny;
    const long n = (long) nu * nview;
    for (int c = 0; c < P; c++) {
        long s, e;
        chunk_of(n, P, c, &s, &e);
        // original BP loop body verbatim, restricted to chunk c, into its own zeroed buffer
        float* pb = parts + (size_t) c * npix;
        for (long ii = s; ii < e; ii++) {
            const int iview = (int) (ii / nu);
            float sinval = (float) sin(deg[iview] / 180.0f * 3.141592f), cosval = (float) cos(deg[iview] / 180.0f * 3.141592f);
            float a = (((ii % nu) - (nu - 1.0f) / 2.0f) - off_a) * da;
            float dist = dy / cos(a);
            for (int iy = 0; iy < ny; iy++) {
                float posy = (iy - (ny - 1.0f) / 2.0f) * dy;
                float posx = tan(a) * (posy + dso);
                float x = posx * cosval + posy * sinval;
                float y = -posx * sinval + posy * cosval;
                float rx = (x) / dx + (nx - 1.0f) / 2.0f;
                float ry = (y) / dy + (ny - 1.0f) / 2.0f;
                if (rx > 0 && rx < nx - 1 && ry > 0 && ry < ny - 1) {
                    float wx = rx - ((int) rx);
                    float wy = ry - ((int) ry);
                    float val = sino[ii] * dist;
                    pb[((int) ry) * nx + ((int) rx)] += (1.0f - wx) * (1.0f - wy) * val;
                    pb[((int) ry) * nx + ((int) rx + 1)] += (wx) * (1.0f - wy) * val;
                    pb[((int) ry + 1) * nx + ((int) rx)] += (1.0f - wx) * (wy) * val;
                    pb[((int) ry + 1) * nx + ((int) rx + 1)] += (wx) * (wy) * val;
                }
            }
        }
    }
}

/* ---------- Python glue ---------- */
// ndarray glue: any array-like -> contiguous float32 / float64 (copy only when needed; float64->float32 rounds to nearest = original (float) cast); needs >= n elements (original reads the first n)
static PyArrayObject* as_arr(PyObject* o, int typ, npy_intp n, const char* nm) {
    PyArrayObject* a = (PyArrayObject*) PyArray_FROMANY(o, typ, 0, 0, NPY_ARRAY_IN_ARRAY | NPY_ARRAY_FORCECAST);
    if (!a) return NULL;
    if (PyArray_SIZE(a) < n) { PyErr_Format(PyExc_ValueError, "%s has %zd elements, need >= %zd", nm, (Py_ssize_t) PyArray_SIZE(a), (Py_ssize_t) n); Py_DECREF(a); return NULL; }
    return a;
}

// BPd(sino, deg, nview, dsd, dso, nx, ny, dx, dy, nu, da, off_a, P, mode) -> float32 image (nx*ny, flat)
static PyObject* BPd(PyObject* self, PyObject* args) {
    PyObject *x, *deg; int nview, nx, ny, nu, P, mode; float dsd, dso, dx, dy, da, off_a;
    if (!PyArg_ParseTuple(args, "OOiffiiffiffii", &x, &deg, &nview, &dsd, &dso, &nx, &ny, &dx, &dy, &nu, &da, &off_a, &P, &mode)) return NULL;
    if (P < 1) { PyErr_SetString(PyExc_ValueError, "P >= 1"); return NULL; }
    PyArrayObject* xa = as_arr(x, NPY_FLOAT32, (npy_intp) nu * nview, "sino"); if (!xa) return NULL;
    PyArrayObject* dg = as_arr(deg, NPY_FLOAT64, nview, "deg"); if (!dg) { Py_DECREF(xa); return NULL; }
    npy_intp nout = (npy_intp) nx * ny;
    PyArrayObject* out = (PyArrayObject*) PyArray_ZEROS(1, &nout, NPY_FLOAT32, 0);
    Py_BEGIN_ALLOW_THREADS
    bp_det_kernel((const float*) PyArray_DATA(xa), (const double*) PyArray_DATA(dg), nview, dso, nx, ny, dx, dy, nu, da, off_a,
                  (float*) PyArray_DATA(out), P, mode, NULL);
    Py_END_ALLOW_THREADS
    n_bp++;
    Py_DECREF(xa); Py_DECREF(dg);
    return (PyObject*) out;
}

static PyObject* FPd(PyObject* self, PyObject* args) {     // FPd(img, deg, ..., off_a, mode) -> float32 sinogram
    PyObject *x, *deg; int nview, nx, ny, nu, mode; float dsd, dso, dx, dy, da, off_a;
    if (!PyArg_ParseTuple(args, "OOiffiiffiffi", &x, &deg, &nview, &dsd, &dso, &nx, &ny, &dx, &dy, &nu, &da, &off_a, &mode)) return NULL;
    PyArrayObject* xa = as_arr(x, NPY_FLOAT32, (npy_intp) nx * ny, "img"); if (!xa) return NULL;
    PyArrayObject* dg = as_arr(deg, NPY_FLOAT64, nview, "deg"); if (!dg) { Py_DECREF(xa); return NULL; }
    npy_intp nout = (npy_intp) nu * nview;
    PyArrayObject* out = (PyArrayObject*) PyArray_ZEROS(1, &nout, NPY_FLOAT32, 0);
    Py_BEGIN_ALLOW_THREADS
    fp_det_kernel((const float*) PyArray_DATA(xa), (const double*) PyArray_DATA(dg), nview, dso, nx, ny, dx, dy, nu, da, off_a, (float*) PyArray_DATA(out), mode);
    Py_END_ALLOW_THREADS
    n_fp++;
    Py_DECREF(xa); Py_DECREF(dg);
    return (PyObject*) out;
}

static PyObject* BPparts(PyObject* self, PyObject* args) {     // (P, nx*ny) partial buffers of the original reduction
    PyObject *x, *deg; int nview, nx, ny, nu, P; float dsd, dso, dx, dy, da, off_a;
    if (!PyArg_ParseTuple(args, "OOiffiiffiffi", &x, &deg, &nview, &dsd, &dso, &nx, &ny, &dx, &dy, &nu, &da, &off_a, &P)) return NULL;
    PyArrayObject* xa = as_arr(x, NPY_FLOAT32, (npy_intp) nu * nview, "sino"); if (!xa) return NULL;
    PyArrayObject* dg = as_arr(deg, NPY_FLOAT64, nview, "deg"); if (!dg) { Py_DECREF(xa); return NULL; }
    npy_intp dims[2] = {P, (npy_intp) nx * ny};
    PyArrayObject* out = (PyArrayObject*) PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    Py_BEGIN_ALLOW_THREADS
    bp_parts_kernel((const float*) PyArray_DATA(xa), (const double*) PyArray_DATA(dg), nview, dso, nx, ny, dx, dy, nu, da, off_a, (float*) PyArray_DATA(out), P);
    Py_END_ALLOW_THREADS
    Py_DECREF(xa); Py_DECREF(dg);
    return (PyObject*) out;
}

static PyObject* ncalls(PyObject* self, PyObject* noargs) {
    return Py_BuildValue("{s:l,s:l}", "FP", n_fp, "BP", n_bp);
}

static PyMethodDef methods[] = {
    {"FPd", FPd, METH_VARARGS, "Forward projection, bit-identical to the original FP (mode bits: 1 clip, 2 split)."},
    {"BPd", BPd, METH_VARARGS, "Backward projection reproducing the original reduction in P chunks with a fixed fold order (mode bits: 1 clip, 2 split, 4 split2)."},
    {"BPparts", BPparts, METH_VARARGS, "The P unfolded partial images of the original reduction."},
    {"ncalls", ncalls, METH_NOARGS, "Call counters (identity proof)."},
    {NULL, NULL, 0, NULL}};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "recon_fast", NULL, -1, methods};

PyMODINIT_FUNC PyInit_recon_fast(void) {
    import_array();
    return PyModule_Create(&module);
}
