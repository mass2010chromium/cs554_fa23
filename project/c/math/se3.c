#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "utils.h"

#include "se3.h"

#include <math.h>
#include "vectorops.h"
#include "so3.h"

PyDoc_STRVAR(se3_identity_doc, "Returns the identity transformation");
PyDoc_STRVAR(se3_inv_doc, "Returns the inverse of the transformation.");
PyDoc_STRVAR(se3_apply_doc, "Applies the transform T to the given point");
PyDoc_STRVAR(se3_apply_rotation_doc, "Applies only the rotation part of T");
PyDoc_STRVAR(se3_rotation_doc, "Returns the 3x3 rotation matrix corresponding to T's rotation");
PyDoc_STRVAR(se3_from_rotation_doc, "Returns a transformation T corresponding to the 3x3 rotation matrix mat");
PyDoc_STRVAR(se3_translation_doc, "Returns the translation vector corresponding to T's translation");
PyDoc_STRVAR(se3_from_translation_doc, "Returns a transformation T that translates points by t");
PyDoc_STRVAR(se3_homogeneous_doc, "Returns the 4x4 homogeneous transform corresponding to T");
PyDoc_STRVAR(se3_from_homogeneous_doc, "Returns a T corresponding to the 4x4 homogeneous transform mat");
PyDoc_STRVAR(se3_mul_doc, "Composes two transformations.");
PyDoc_STRVAR(se3_distance_doc, "Returns a distance metric between the two transformations. The rotation distance is weighted by Rweight and the translation distance is weighted by tweight");
PyDoc_STRVAR(se3_error_doc, "Returns a 6D 'difference vector' that describes how far T1 is from T2. More precisely, this is the Lie derivative (w,v).");
PyDoc_STRVAR(se3_interpolate_doc, "Interpolate linearly between the two transformations T1 and T2.'difference vector' that describes how far T1 is from T2. More precisely, this is the Lie derivative (w,v).");
PyDoc_STRVAR(se3_interpolator_doc, "Returns a function of one parameter u that interpolates linearly between the two transformations T1 and T2. After f(u) is constructed, calling f(u) is about 2x faster than calling interpolate(T1,T2,u).'difference vector' that describes how far T1 is from T2. More precisely, this is the Lie derivative (w,v).");

static PyMethodDef se3Methods[] = {
    {"identity", se3_identity, METH_NOARGS, se3_identity_doc},
    {"inv", (PyCFunction) se3_inv, METH_FASTCALL, se3_inv_doc},
    {"apply", (PyCFunction) se3_apply, METH_FASTCALL, se3_apply_doc},
    {"apply_rotation", (PyCFunction) se3_apply_rotation, METH_FASTCALL, se3_apply_rotation_doc},
    {"rotation", (PyCFunction) se3_rotation, METH_FASTCALL, se3_rotation_doc},
    {"from_rotation", (PyCFunction) se3_from_rotation, METH_FASTCALL, se3_from_rotation_doc},
    {"translation", (PyCFunction) se3_translation, METH_FASTCALL, se3_translation_doc},
    {"from_translation", (PyCFunction) se3_from_translation, METH_FASTCALL, se3_from_translation_doc},
    {"homogeneous", (PyCFunction) se3_homogeneous, METH_FASTCALL, se3_homogeneous_doc},
    {"from_homogeneous", (PyCFunction) se3_from_homogeneous, METH_FASTCALL, se3_from_homogeneous_doc},
    {"mul", (PyCFunction) se3_mul, METH_FASTCALL, se3_mul_doc},
    {"distance", (PyCFunction) se3_distance, METH_VARARGS | METH_KEYWORDS, se3_distance_doc},
    {"error", (PyCFunction) se3_error, METH_FASTCALL, se3_error_doc},
    {"interpolate", (PyCFunction) se3_interpolate, METH_FASTCALL, se3_interpolate_doc},
    {"interpolator", (PyCFunction) se3_interpolator, METH_FASTCALL, se3_interpolator_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef se3Module = {
    PyModuleDef_HEAD_INIT,
    "motion.se3",
    NULL,   // Documentation
    -1,     /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    se3Methods
};

PyMODINIT_FUNC PyInit_se3() {
    return PyModule_Create(&se3Module);
}

/**
 * Returns the identity transformation.
 */
PyObject* se3_identity(PyObject* self, PyObject* args) {
    return return_se3(SE3_ID);
}

/**
 * Returns the inverse of the transformation.
 */
PyObject* se3_inv(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double data[12];
    double ret[12];
    if (parse_se3(data, args[0])) {
        return NULL;
    }
    __se3_inv(ret, data);
    return return_se3(ret);
}

/**
 * Applies the transform T to the given point
 */
PyObject* se3_apply(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double transform[12];
    double point[3];
    double ret[3];
    if (parse_se3(transform, args[0])) {
        return NULL;
    }
    if (parse_vec3(point, args[1])) {
        return NULL;
    }
    __se3_apply(ret, transform, point);
    return vector_to_list(ret, 3);
}

/**
 * Applies only the rotation part of T
 */
PyObject* se3_apply_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double transform[12];
    double point[3];
    double ret[3];
    if (parse_se3(transform, args[0])) {
        return NULL;
    }
    if (parse_vec3(point, args[1])) {
        return NULL;
    }
    __so3_apply(ret, transform, point);
    return vector_to_list(ret, 3);
}

/**
 * Returns the 3x3 rotation matrix corresponding to T's rotation
 */
PyObject* se3_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double transform[12];
    if (parse_se3(transform, args[0])) {
        return NULL;
    }
    return __so3_matrix(transform);
}

/**
 * Returns a transformation T corresponding to the 3x3 rotation matrix mat
 */
PyObject* se3_from_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double transform[12];
    if (__so3_from_matrix(transform, args[0])) {
        return NULL;
    }
    transform[9] = 0;
    transform[10] = 0;
    transform[11] = 0;
    return return_se3(transform);
}

/**
 * Returns the translation vector corresponding to T's translation
 */
PyObject* se3_translation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double transform[12];
    if (parse_se3(transform, args[0])) {
        return NULL;
    }
    return vector_to_list(transform+9, 3);
}

/**
 * Returns a transformation T that translates points by t
 */
PyObject* se3_from_translation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double transform[12] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    if (parse_vec3(transform+9, args[0])) {
        return NULL;
    }
    return return_se3(transform);
}

/**
 * Returns the 4x4 homogeneous transform corresponding to T
 */
PyObject* se3_homogeneous(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double transform[12];
    if (parse_se3(transform, args[0])) {
        return NULL;
    }
    double scratch[10];
    __so3_inv(scratch, transform);
    double tmp = scratch[3];
    scratch[3] = transform[9];
    PyObject* l1 = vector_to_list(scratch, 4);
#ifdef MOTION_DEBUG
    if (l1 == NULL) {
        return NULL;
    }
#endif
    scratch[3] = tmp;
    tmp = scratch[6];
    scratch[6] = transform[10];
    PyObject* l2 = vector_to_list(scratch+3, 4);
#ifdef MOTION_DEBUG
    if (l2 == NULL) {
        Py_DECREF(l1);
        return NULL;
    }
#endif
    scratch[6] = tmp;
    scratch[9] = transform[11];
    PyObject* l3 = vector_to_list(scratch+6, 4);
#ifdef MOTION_DEBUG
    if (l3 == NULL) {
        Py_DECREF(l1);
        Py_DECREF(l2);
        return NULL;
    }
#endif
    double scratch2[4] = {0, 0, 0, 1};
    PyObject* l4 = vector_to_list(scratch2, 4);
#ifdef MOTION_DEBUG
    if (l4 == NULL) {
        Py_DECREF(l1);
        Py_DECREF(l2);
        Py_DECREF(l3);
        return NULL;
    }
#endif
    PyObject* result = Py_BuildValue("[NNNN]", l1, l2, l3, l4);
#ifdef MOTION_DEBUG
    if (result == NULL) {
        Py_DECREF(l1);
        Py_DECREF(l2);
        Py_DECREF(l3);
        Py_DECREF(l4);
        return NULL;
    }
#endif
    return result;
}

/**
 * Returns a T corresponding to the 4x4 homogeneous transform mat
 */
PyObject* se3_from_homogeneous(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    PyObject* rot = args[0];
#ifdef MOTION_DEBUG
    Py_ssize_t n;
    n = PyObject_Length(rot);
#ifdef SO3_STRICT
    if (n != 4)
#else
    if (n < 4)
#endif
    {
        PyErr_SetString(PyExc_ValueError, "Homogenous Transform is 4x4 matrix");
        return NULL;
    }
#endif

    PyObject* it = PyObject_GetIter(rot);
#ifdef MOTION_DEBUG
    if (it == NULL) {
        return NULL;
    }
#endif

    double buf[9];
    double output[12];
    double* buf_head = buf;
    double* vec_head = output+9;
    PyObject* curr;
    while ((curr = PyIter_Next(it))) {
#ifdef MOTION_DEBUG
        if (curr == NULL) {
            Py_DECREF(it);
            return NULL;
        }
        n = PyObject_Length(rot);
#ifdef SO3_STRICT
        if (n != 4)
#else
        if (n < 4)
#endif
        {
            PyErr_SetString(PyExc_ValueError, "Homogenous Transform is 4x4 matrix");
            Py_DECREF(curr);
            Py_DECREF(it);
            return NULL;
        }
#endif
        PyObject* it2 = list_to_vector_n(curr, buf_head, 3);
#ifdef MOTION_DEBUG
        if (it == NULL) {
            Py_DECREF(curr);
            Py_DECREF(it);
            return NULL;
        }
#endif
        PyObject* last = PyIter_Next(it2);
#ifdef MOTION_DEBUG
        if (last == NULL) {
            Py_DECREF(it2);
            Py_DECREF(curr);
            Py_DECREF(it);
            return NULL;
        }
        if (!(PyFloat_Check(last) || PyLong_Check(last))) {
            Py_DECREF(last);
            Py_DECREF(it2);
            Py_DECREF(curr);
            Py_DECREF(it);
            PyErr_SetString(PyExc_TypeError, "expected float array");
            return NULL;
        }
#endif
        buf_head += 3;
        *(vec_head++) = PyFloat_AsDouble(last);
        Py_DECREF(last);
        Py_DECREF(it2);
        Py_DECREF(curr);
    }
    Py_DECREF(it);
    __so3_inv(output, buf);
    return return_se3(output);
}

/**
 * Composes two transformations.
 */
PyObject* se3_mul(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double t1[12];
    double t2[12];
    double ret[12];
    if (parse_se3(t1, args[0])) {
        return NULL;
    }
    if (parse_se3(t2, args[1])) {
        return NULL;
    }
    __se3_mul(ret, t1, t2);
    return return_se3(ret);
}

/**
 * Returns a distance metric between the two transformations. The rotation distance is weighted by Rweight and the translation distance is weighted by tweight
 */
PyObject* se3_distance(PyObject* self, PyObject* args, PyObject* kwargs) {
    static char* kwlist[] = {"Rweight", "tweight"};
    PyObject* _t1;
    PyObject* _t2;
    double rweight = 1.0;
    double tweight = 1.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|dd", kwlist, 
                                     &_t1, &_t2, &rweight, &tweight)) {
        return NULL;
    }
    double t1[12];
    double t2[12];
    if (parse_se3(t1, _t1)) {
        return NULL;
    }
    if (parse_se3(t2, _t2)) {
        return NULL;
    }
    return PyFloat_FromDouble(__se3_distance(t1, t2, rweight, tweight));
}

/**
 * Returns a 6D "difference vector" that describes how far T1 is from T2. More precisely, this is the Lie derivative (w,v).
 */
PyObject* se3_error(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double t1[12];
    double t2[12];
    double ret[6];
    if (parse_se3(t1, args[0])) {
        return NULL;
    }
    if (parse_se3(t2, args[1])) {
        return NULL;
    }
    __se3_error(ret, t1, t2);
    return vector_to_list(ret, 6);
}

/**
 * Interpolate linearly between the two transformations T1 and T2.
 */
PyObject* se3_interpolate(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 3) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 3)");
        return NULL;
    }
    if (!(PyFloat_Check(args[2]) || PyLong_Check(args[2]))) {
        PyErr_SetString(PyExc_TypeError, "Expected number for args[2]");
        return NULL;
    }
#endif
    double u = PyFloat_AsDouble(args[2]);

    double t1[12];
    double t2[12];
    double ret[12];
    if (parse_se3(t1, args[0])) {
        return NULL;
    }
    if (parse_se3(t2, args[1])) {
        return NULL;
    }
    __se3_interpolate(ret, t1, t2, u);
    return return_se3(ret);
}

/**
 * "lambda" function for interpolator.
 * First argument: Bytes object with the raw data
 */
static PyObject* __se3_interpolate_f(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
    if (!(PyFloat_Check(args[0]) || PyLong_Check(args[0]))) {
        PyErr_SetString(PyExc_TypeError, "Expected number for args[0]");
        return NULL;
    }
#endif
    double u = PyFloat_AsDouble(args[0]);
    se3_interpolator_t* data = (se3_interpolator_t*) PyBytes_AS_STRING(self);
    double tmp[9];
    double result[12];
    __so3_rotation(tmp, data->axis, data->angle*u);
    __so3_mul(result, data->rot, tmp);
    __vo_madd(result+9, data->t1, data->dt, u, 3);
    return return_se3(result);
}

static struct PyMethodDef
__se3_interpolator_cb = {.ml_name  = "f@interpolator",
                         .ml_meth  = (PyCFunction) __se3_interpolate_f,
                         .ml_flags = METH_FASTCALL,
                         .ml_doc   = NULL};

/**
 * Returns a function of one parameter u that interpolates linearly between the two transformations T1 and T2. After f(u) is constructed, calling f(u) is about 2x faster than calling interpolate(T1,T2,u).
 */
PyObject* se3_interpolator(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double t1[12];
    double t2[12];
    se3_interpolator_t result;
    if (parse_se3(t1, args[0])) {
        return NULL;
    }
    if (parse_se3(t2, args[1])) {
        return NULL;
    }
    __se3_interpolator_init(&result, t1, t2);

    PyObject* byte_data = PyBytes_FromStringAndSize(NULL, sizeof(result));
    if (byte_data == NULL) {
        return NULL;
    }
    char* save_data = PyBytes_AS_STRING(byte_data);
    memcpy(save_data, &result, sizeof(result));

    PyObject* ret = PyCFunction_New(&__se3_interpolator_cb, byte_data);
    Py_DECREF(byte_data);
    return ret;
}
