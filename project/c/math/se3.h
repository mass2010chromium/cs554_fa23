#pragma once

#include <math.h>
#include "utils.h"
#include "vectorops.h"
#include "so3.h"

#define tptr motion_dtype*
#ifdef SE3_RESTRICT
#define tptr_r motion_dtype* restrict
#else
#define tptr_r motion_dtype*
#endif

PY_FUNC(
#ifndef MOTION_DEBUG
static inline
#endif
PyObject* return_se3(const tptr_r data) {
    PyObject* rot = vector_to_list(data, 9);
#ifdef MOTION_DEBUG
    if (rot == NULL) {
        return NULL;
    }
#endif
    PyObject* trans = vector_to_list(data+9, 3);
#ifdef MOTION_DEBUG
    if (trans == NULL) {
        Py_DECREF(rot);
        return NULL;
    }
#endif
    PyObject* ret = Py_BuildValue("(NN)", rot, trans);
#ifdef MOTION_DEBUG
    if (ret == NULL) {
        Py_DECREF(trans);
        Py_DECREF(rot);
        return NULL;
    }
#endif
    return ret;
}
)

PY_FUNC(
#ifndef MOTION_DEBUG
static inline
#endif
int parse_se3 (tptr_r dest, PyObject* vec) {
#ifdef MOTION_DEBUG
    Py_ssize_t n = PyObject_Length(vec);
#ifdef SE3_STRICT
    if (n != 2)
#else
    if (n < 2)
#endif
    {
        PyErr_SetString(PyExc_ValueError, "Transform expected (R, T)");
        return 1;
    }
#endif
    PyObject* it = PyObject_GetIter(vec);
#ifdef MOTION_DEBUG
    if (it == NULL) {
        return 1;
    }
#endif

    PyObject* curr = PyIter_Next(it);
#ifdef MOTION_DEBUG
    if (curr == NULL) {
        return 1;
    }
#endif
    if (parse_rotation(dest, curr)) {
        Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.
        Py_DECREF(it);
        return 1;
    }
    Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.

    curr = PyIter_Next(it);
#ifdef MOTION_DEBUG
    if (curr == NULL) {
        return 1;
    }
#endif
    if (parse_vec3(dest+9, curr)) {
        Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.
        Py_DECREF(it);
        return 1;
    }
    Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.
    Py_DECREF(it);
    return 0;
}
)

const motion_dtype SE3_ID[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};

/**
 * Returns the identity transformation.
 */
PY_FUNC(PyObject* se3_identity(PyObject* self, PyObject* args));

/**
 * Returns the inverse of the transformation.
 */
PY_FUNC(PyObject* se3_inv(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __se3_inv(tptr_r ret, const tptr_r transform) {
    __so3_inv(ret, transform);
    __so3_apply(ret+9, ret, transform+9);
    __vo_mul(ret+9, ret+9, -1, 3);
}

/**
 * Applies the transform T to the given point
 */
PY_FUNC(PyObject* se3_apply(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __se3_apply(vptr_r ret, const tptr_r transform, const vptr_r point) {
    __so3_apply(ret, transform, point);
    __vo_addv(ret, ret, transform+9, 3);
}

/**
 * Applies only the rotation part of T
 */
PY_FUNC(PyObject* se3_apply_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns the 3x3 rotation matrix corresponding to T's rotation
 */
PY_FUNC(PyObject* se3_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns a transformation T corresponding to the 3x3 rotation matrix mat
 */
PY_FUNC(PyObject* se3_from_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns the translation vector corresponding to T's translation
 */
PY_FUNC(PyObject* se3_translation(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns a transformation T that translates points by t
 */
PY_FUNC(PyObject* se3_from_translation(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns the 4x4 homogeneous transform corresponding to T
 */
PY_FUNC(PyObject* se3_homogeneous(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Returns a T corresponding to the 4x4 homogeneous transform mat
 */
PY_FUNC(PyObject* se3_from_homogeneous(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

/**
 * Composes two transformations.
 */
PY_FUNC(PyObject* se3_mul(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __se3_mul(tptr_r ret, const tptr_r t1, const tptr_r t2) {
    __so3_mul(ret, t1, t2);
    __so3_apply(ret+9, t1, t2+9);
    __vo_addv(ret+9, ret+9, t1+9, 3);
}

/**
 * Returns a distance metric between the two transformations. The rotation distance is weighted by Rweight and the translation distance is weighted by tweight
 */
PY_FUNC(PyObject* se3_distance(PyObject* self, PyObject* args, PyObject* kwargs));

static inline motion_dtype __se3_distance(const tptr_r t1, const tptr_r t2,
                             motion_dtype rweight, motion_dtype tweight) {
    return __so3_distance(t1, t2) * rweight
         + __vo_distance(t1+9, t2+9, 3) * tweight;
}

/**
 * Returns a 6D "difference vector" that describes how far T1 is from T2. More precisely, this is the Lie derivative (w,v).
 */
PY_FUNC(PyObject* se3_error(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __se3_error(vptr_r ret, const tptr_r t1, const tptr_r t2) {
    __so3_error(ret, t1, t2);
    __vo_subv(ret+3, t1+9, t2+9, 3);
}

/**
 * Interpolate linearly between the two transformations T1 and T2.
 */
PY_FUNC(PyObject* se3_interpolate(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

static inline void __se3_interpolate(tptr_r ret, const tptr_r t1, const tptr_r t2, motion_dtype u) {
    __so3_interpolate(ret, t1, t2, u);
    __vo_interpolate(ret+9, t1+9, t2+9, u, 3);
}

typedef struct {
    motion_dtype rot[9];
    motion_dtype axis[3];
    motion_dtype angle;
    motion_dtype t1[3];
    motion_dtype dt[3];
} se3_interpolator_t;

static inline void __se3_interpolator_init(se3_interpolator_t* ret, const tptr_r t1, const tptr_r t2) {
    __so3_interpolator_init((so3_interpolator_t*) ret, t1, t2);
    memcpy(ret->t1, t1+9, sizeof(motion_dtype)*3);
    __vo_subv(ret->dt, t2+9, t1+9, 3);
}

/**
 * Returns a function of one parameter u that interpolates linearly
 * between the two transformations T1 and T2.
 * 
 * After f(u) is constructed, calling f(u) is about 2x faster than calling interpolate(T1,T2,u).
 */
PY_FUNC(PyObject* se3_interpolator(PyObject* self, PyObject* const* args, Py_ssize_t nargs));
