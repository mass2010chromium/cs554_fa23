#pragma once

#include "utils.h"

#include <math.h>

#define vptr motion_dtype*
#ifdef VO_RESTRICT
#define vptr_r motion_dtype* restrict
#else
#define vptr_r motion_dtype*
#endif


/**
 * Return a+c*b where a and b are vectors.
 */
PY_FUNC(PyObject* vectorops_madd(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __vo_madd(vptr dest, const vptr a, const vptr b, motion_dtype c, int len) {
    switch(len) {
#ifdef CASE_3_6
        case 9:
            dest[8] = a[8] + c*b[8];
        case 8:
            dest[7] = a[7] + c*b[7];
        case 7:
            dest[6] = a[6] + c*b[6];
        case 6:
            dest[5] = a[5] + c*b[5];
        case 5:
            dest[4] = a[4] + c*b[4];
        case 4:
            dest[3] = a[3] + c*b[3];
        case 3:
            dest[2] = a[2] + c*b[2];
        case 2:
            dest[1] = a[1] + c*b[1];
        case 1:
            dest[0] = a[0] + c*b[0];
            return;
#endif
        default:
            for (int i = 0; i < len; ++i) {
                dest[i] = a[i] + c*b[i];
            }
    }
}

#ifdef CASE_3_6 
#define __VO_BINOP_DECL(name, op) \
PY_FUNC(PyObject* vectorops_ ## name (PyObject* self, PyObject* const* args, Py_ssize_t nargs)); \
 \
static inline void __vo_ ## name (vptr dest, const vptr a, motion_dtype b, int len) { \
    switch(len) { \
        case 9: \
            dest[8] = op(a[8], b); \
        case 8: \
            dest[7] = op(a[7], b); \
        case 7: \
            dest[6] = op(a[6], b); \
        case 6: \
            dest[5] = op(a[5], b); \
        case 5: \
            dest[4] = op(a[4], b); \
        case 4: \
            dest[3] = op(a[3], b); \
        case 3: \
            dest[2] = op(a[2], b); \
        case 2: \
            dest[1] = op(a[1], b); \
        case 1: \
            dest[0] = op(a[0], b); \
            return; \
        default: \
            for (int i = 0; i < len; ++i) { \
                dest[i] = op(a[i], b); \
            } \
    } \
} \
 \
static inline void __vo_ ## name ## v (vptr dest, const vptr a, const vptr b, int len) { \
    switch(len) { \
        case 9: \
            dest[8] = op(a[8], b[8]); \
        case 8: \
            dest[7] = op(a[7], b[7]); \
        case 7: \
            dest[6] = op(a[6], b[6]); \
        case 6: \
            dest[5] = op(a[5], b[5]); \
        case 5: \
            dest[4] = op(a[4], b[4]); \
        case 4: \
            dest[3] = op(a[3], b[3]); \
        case 3: \
            dest[2] = op(a[2], b[2]); \
        case 2: \
            dest[1] = op(a[1], b[1]); \
        case 1: \
            dest[0] = op(a[0], b[0]); \
            return; \
        default: \
            for (int i = 0; i < len; ++i) { \
                dest[i] = op(a[i], b[i]); \
            } \
    } \
}

#else
#define __VO_BINOP_DECL(name, op) \
PY_FUNC(PyObject* vectorops_ ## name (PyObject* self, PyObject* const* args, Py_ssize_t nargs)); \
 \
static inline void __vo_ ## name (vptr dest, const vptr a, motion_dtype b, int len) { \
    for (int i = 0; i < len; ++i) { \
        dest[i] = op(a[i], b); \
    } \
} \
 \
static inline void __vo_ ## name ## v (vptr dest, const vptr a, const vptr b, int len) { \
    for (int i = 0; i < len; ++i) { \
        dest[i] = op(a[i], b[i]); \
    } \
}

#endif

/**
 * Add a vector b to a, or add a scalar
 */
#define __VO_ADD(a, b) (a) + (b)
__VO_BINOP_DECL(add, __VO_ADD)

/**
 * Subtract a vector b from a, or subtract a scalar
 */
#define __VO_MINUS(a, b) (a) - (b)
__VO_BINOP_DECL(sub, __VO_MINUS)

/**
 * Multiply a vector either elementwise with another vector, or with a scalar.
 */
#define __VO_TIMES(a, b) (a) * (b)
__VO_BINOP_DECL(mul, __VO_TIMES)

/**
 * Elementwise division with another vector, or with a scalar.
 */
#define __VO_DIVIDE(a, b) (a) / (b)
__VO_BINOP_DECL(div, __VO_DIVIDE)

/**
 * Elementwise max
 */
// Lol not safe but its just for my usage anyway
#define __VO_MAX(a, b) ((a) > (b) ? (a) : (b))
__VO_BINOP_DECL(maximum, __VO_MAX)

/**
 * Elementwise min
 */
// Lol not safe but its just for my usage anyway
#define __VO_MIN(a, b) ((a) < (b) ? (a) : (b))
__VO_BINOP_DECL(minimum, __VO_MIN)

/**
 * Dot product.
 */
PY_FUNC(PyObject* vectorops_dot(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_dot(const vptr a, const vptr b, int len) {
    motion_dtype res = 0;
    switch(len) {
#ifdef CASE_3_6
        case 6:
            res += a[5] * b[5];
            res += a[4] * b[4];
            res += a[3] * b[3];
        case 3:
            res += a[2] * b[2];
            res += a[1] * b[1];
            res += a[0] * b[0];
            return res;
#endif
        default:
            for (int i = 0; i < len; ++i) {
                res += a[i] * b[i];
            }
            return res;
    }
}

/**
 * Returns the norm of a, squared.
 */
PY_FUNC(PyObject* vectorops_normSquared(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_normSquared(const vptr a, int len) {
    return __vo_dot(a, a, len);
}

/**
 * L2 norm
 */
PY_FUNC(PyObject* vectorops_norm(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_norm(const vptr a, int len) {
    return sqrt(__vo_normSquared(a, len));
}

/**
 * Returns the unit vector in the direction a.  If the norm of
 * a is less than epsilon, a is left unchanged.
 */
PY_FUNC(PyObject* vectorops_unit(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __vo_unit(vptr dest, const vptr a, motion_dtype eps, int len) {
    motion_dtype norm = __vo_norm(a, len);
    if (norm > eps) {
        motion_dtype norm_recip = 1 / norm;
        for (int i = 0; i < len; ++i) {
            dest[i] = a[i] * norm_recip;
        }
    }
    else {
        for (int i = 0; i < len; ++i) {
            dest[i] = a[i];
        }
    }
}

/**
 * L1 norm
 */
PY_FUNC(PyObject* vectorops_norm_L1(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_norm_L1(const vptr_r a, int len) {
    motion_dtype res = 0;
    switch(len) {
#ifdef CASE_3_6
        case 9:
            res += fabs(a[8]);
        case 8:
            res += fabs(a[7]);
        case 7:
            res += fabs(a[6]);
        case 6:
            res += fabs(a[5]);
        case 5:
            res += fabs(a[4]);
        case 4:
            res += fabs(a[3]);
        case 3:
            res += fabs(a[2]);
        case 2:
            res += fabs(a[1]);
        case 1:
            res += fabs(a[0]);
            return res;
#endif
        default:
            for (int i = 0; i < len; ++i) {
                res += fabs(a[i]);
            }
            return res;
    }
}

/**
 * L-infinity norm
 */
PY_FUNC(PyObject* vectorops_norm_Linf(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_norm_Linf(const vptr_r a, int len) {
    motion_dtype max = 0;
    motion_dtype tmp;
    // No case BS... if statements are magic
    for (int i = 0; i < len; ++i) {
        if ((tmp = fabs(a[i])) > max) max = tmp;
    }
    return max;
}

/**
 * Squared L2 distance
 */
PY_FUNC(PyObject* vectorops_distanceSquared(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_distanceSquared(const vptr_r a, const vptr_r b, int len) {
    motion_dtype res = 0;
    switch(len) {
#ifdef CASE_3_6
        case 9:
            res += (a[8] - b[8]) * (a[8] - b[8]);
        case 8:
            res += (a[7] - b[7]) * (a[7] - b[7]);
        case 7:
            res += (a[6] - b[6]) * (a[6] - b[6]);
        case 6:
            res += (a[5] - b[5]) * (a[5] - b[5]);
        case 5:
            res += (a[4] - b[4]) * (a[4] - b[4]);
        case 4:
            res += (a[3] - b[3]) * (a[3] - b[3]);
        case 3:
            res += (a[2] - b[2]) * (a[2] - b[2]);
        case 2:
            res += (a[1] - b[1]) * (a[1] - b[1]);
        case 1:
            res += (a[0] - b[0]) * (a[0] - b[0]);
            return res;
#endif
        default:
            for (int i = 0; i < len; ++i) {
                res += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return res;
    }
}

/**
 * L2 distance
 */
PY_FUNC(PyObject* vectorops_distance(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_distance(const vptr_r a, const vptr_r b, int len) {
    return sqrt(__vo_distanceSquared(a, b, len));
}

/**
 * Cross product between a 3-vector or a 2-vector
 */
PY_FUNC(PyObject* vectorops_cross(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline motion_dtype __vo_cross2(const vptr_r a, const vptr_r b) {
    return a[0] * b[1] - a[1] * b[0];
}

static inline void __vo_cross3(vptr_r dest, vptr_r a, vptr_r b) {
    dest[0] = a[1] * b[2] - a[2] * b[1];
    dest[1] = a[2] * b[0] - a[0] * b[2];
    dest[2] = a[0] * b[1] - a[1] * b[0];
}

/**
 * Linear interpolation between a and b
 */
PY_FUNC(PyObject* vectorops_interpolate(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __vo_interpolate(vptr dest, const vptr a, const vptr b, motion_dtype u, int n) {
    __vo_subv(dest, b, a, n);
    __vo_madd(dest, a, dest, u, n);
}
