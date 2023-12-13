#pragma once

//#define CASE_3_6

#ifndef motion_dtype
#define motion_dtype double
#endif

#ifdef PYTHON
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define PY_FUNC(x) x
#else
#define PY_FUNC(x)
#endif

#include <math.h>
#include <stdlib.h>

#define print_vector(vec, n) {\
    motion_dtype* v = (vec);\
    printf("[ ");\
    for (size_t r = 0; r < (n); ++r) {\
        printf("%f ", v[r]);\
    }\
    printf("]\n");\
}

#ifdef MOTION_DEBUG
#include <assert.h>
#define _assert assert
#else
#define _assert(x) ((void) (x))
#endif

PY_FUNC(
/**
 * Create a new list from a C array.
 * Return value: new reference
 */
#ifndef MOTION_DEBUG
static inline
#endif
PyObject* vector_to_list(const motion_dtype* vec, Py_ssize_t size) {
    PyObject* ret = PyList_New(size);
#ifdef MOTION_DEBUG
    if (ret == NULL) {
        return NULL;
    }
#endif
    for (int i = 0; i < size; ++i) {
        PyObject* item = PyFloat_FromDouble(vec[i]);
#ifdef MOTION_DEBUG
        if (item == NULL) {
            Py_DECREF(ret);
            return NULL;
        }
#endif
        PyList_SET_ITEM(ret, i, item);
    }
    return ret;
}

/**
 * Parse a list into a buffer.
 * WILL FAIL if the object is malicious and
 * passes a object with spoofed len() lol
 */
#ifndef MOTION_DEBUG
static inline
#endif
int list_to_vector(PyObject* list, motion_dtype* vec) {
    PyObject* it = PyObject_GetIter(list);
#ifdef MOTION_DEBUG
    if (it == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed to get iterator");
        return 1;
    }
#endif

    PyObject* curr;
    motion_dtype* head = vec;
    while ((curr = PyIter_Next(it))) {
#ifdef MOTION_DEBUG
        if (curr == NULL) {
            return 1;
        }
        if (!(PyFloat_Check(curr) || PyLong_Check(curr))) {
            Py_DECREF(curr);
            Py_DECREF(it);
            PyErr_SetString(PyExc_TypeError, "expected float array");
            return 1;
        }
#endif
        *(head++) = PyFloat_AsDouble(curr); // Works for long too hmm
        Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.
    }
    Py_DECREF(it);
    return 0;
}

/**
 * Returns NULL on failure, or the iterator handle on success.
 */
#ifndef MOTION_DEBUG
static inline
#endif
PyObject* list_to_vector_n(PyObject* list, motion_dtype* vec, int n) {
    PyObject* it = PyObject_GetIter(list);
#ifdef MOTION_DEBUG
    if (it == NULL) {
        return NULL;
    }
#endif

    PyObject* curr;
    motion_dtype* head = vec;
    for (int i = 0; i < n; ++i) {
        curr = PyIter_Next(it);
#ifdef MOTION_DEBUG
        if (curr == NULL) {
            return NULL;
        }
        if (!(PyFloat_Check(curr) || PyLong_Check(curr))) {
            Py_DECREF(curr);
            Py_DECREF(it);
            PyErr_SetString(PyExc_TypeError, "expected float array");
            return NULL;
        }
#endif
        *(head++) = PyFloat_AsDouble(curr); // Works for long too hmm
        Py_DECREF(curr);    // Can't trigger gc cause container still exists. I think.
    }
    return it;
}

)   // PY_FUNC


#define rptr motion_dtype*
#ifdef SO3_RESTRICT
#define rptr_r motion_dtype* restrict
#else
#define rptr_r motion_dtype*
#endif

#ifdef PYTHON
#ifdef MOTION_DEBUG
#ifdef SO3_STRICT
#define PARSE_VEC_FIX(name, len) \
int parse_ ## name (rptr_r dest, PyObject* vec) { \
    Py_ssize_t n = PyObject_Length(vec); \
    if (n < 0) { \
        PyErr_SetString(PyExc_TypeError, "object has no length"); \
        return 1; \
    } \
    if (n != (len)) { \
        PyErr_SetString(PyExc_ValueError, #name " expected " #len " element vector"); \
        return 1; \
    } \
    PyObject* it = list_to_vector_n(vec, dest, (len)); \
    if (it == NULL) return 1; \
    Py_DECREF(it); \
    return 0; \
}
#else
#define PARSE_VEC_FIX(name, len) \
static inline int parse_ ## name (rptr_r dest, PyObject* vec) { \
    Py_ssize_t n = PyObject_Length(vec); \
    if (n < 0) { \
        PyErr_SetString(PyExc_TypeError, "object has no length"); \
        return 1; \
    } \
    if (n > (len)) { \
        PyErr_SetString(PyExc_ValueError, #name " expected " #len " element vector"); \
        return 1; \
    } \
    return list_to_vector(vec, dest); \
}
#endif
#else
#define PARSE_VEC_FIX(name, len) \
static inline int parse_ ## name (rptr_r dest, PyObject* vec) { \
    return list_to_vector(vec, dest); \
}
#endif
#else
#define PARSE_VEC_FIX(name, len)
#endif // PYTHON

PARSE_VEC_FIX(rotation, 9);
PARSE_VEC_FIX(vec3, 3);
PARSE_VEC_FIX(quat, 4);


/**
 * Signum, written to match Klampt
 */
static inline int signum(motion_dtype x) {
    if (x > 0) return 1;
    if (x < 0) return -1;
    return 0;
}

/**
 * Clamp to +1 and -1 for trig
 */
static inline motion_dtype clamp_unity(motion_dtype x) {
    if (x > 1.0) return 1.0;
    if (x < -1.0) return -1.0;
    return x;
}

/**
 * Gaussian distribution stolen from online.
 * Uses the Box-Muller method to generate values from
 * a gaussian distribution with mean 0 and stdev 1.
 * Source: https://stackoverflow.com/a/10645091
 * Optionally returns a second value through the pointer.
 */
static inline motion_dtype sampleNormal(motion_dtype * b) {
    motion_dtype u = ((motion_dtype) rand() / (RAND_MAX)) * 2 - 1;
    motion_dtype v = ((motion_dtype) rand() / (RAND_MAX)) * 2 - 1;
    motion_dtype r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal(b);
    motion_dtype c = sqrt(-2 * log(r) / r);
    if (b) *b = v * c;
    return u * c;
}
