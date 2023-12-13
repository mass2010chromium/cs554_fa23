#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "utils.h"

#include "so3.h"

#include <math.h>
#include "vectorops.h"

PyDoc_STRVAR(so3_identity_doc, "Returns the identity rotation");
PyDoc_STRVAR(so3_inv_doc, "Inverts the rotation");
PyDoc_STRVAR(so3_apply_doc, "Applies the rotation to a point");
PyDoc_STRVAR(so3_matrix_doc, "Returns the 3x3 rotation matrix corresponding to R");
PyDoc_STRVAR(so3_from_matrix_doc, "Returns an R corresponding to the 3x3 rotation matrix mat");
PyDoc_STRVAR(so3_mul_doc, "Multiplies two rotations.");
PyDoc_STRVAR(so3_trace_doc, "Computes the trace of the rotation matrix.");
PyDoc_STRVAR(so3_angle_doc, "Returns absolute deviation of R from identity");
PyDoc_STRVAR(so3_rpy_doc, "Converts a rotation matrix to a roll,pitch,yaw angle triple. The result is given in radians.");
PyDoc_STRVAR(so3_from_rpy_doc, "Converts from roll,pitch,yaw angle triple to a rotation matrix.  The triple is given in radians.  The x axis is 'roll', y is 'pitch', and z is 'yaw'.");
PyDoc_STRVAR(so3_rotation_vector_doc, "Returns the rotation vector w (exponential map) representation of R such that e^[w] = R.  Equivalent to axis-angle representation with w/||w||=axis, ||w||=angle.");
PyDoc_STRVAR(so3_axis_angle_doc, "Returns the (axis,angle) pair representing R");
PyDoc_STRVAR(so3_from_axis_angle_doc, "Converts an axis-angle representation (axis,angle) to a 3D rotation matrix.");
PyDoc_STRVAR(so3_from_rotation_vector_doc, "Converts a rotation vector representation w to a 3D rotation matrix.");
PyDoc_STRVAR(so3_from_quaternion_doc, "Given a unit quaternion (w,x,y,z), produce the corresponding rotation matrix.");
PyDoc_STRVAR(so3_quaternion_doc, "Given a Klamp't rotation representation, produces the corresponding unit quaternion (w,x,y,z).");
PyDoc_STRVAR(so3_distance_doc, "Returns the absolute angle one would need to rotate in order to get from R1 to R2");
PyDoc_STRVAR(so3_error_doc, "Returns a 3D 'difference vector' that describes how far R1 is from R2. More precisely, this is the (local) Lie derivative, which is the rotation vector representation of R1*R2^T. Fun fact: this is related to the derivative of interpolate(R2,R1,u) at u=0 by d/du interpolate(R2,R1,0) = mul(error(R1,R2),R2).");
PyDoc_STRVAR(so3_cross_product_doc, "Returns the cross product matrix associated with w. The matrix [w]R is the derivative of the matrix R as it rotates about the axis w/||w|| with angular velocity ||w||. ");
PyDoc_STRVAR(so3_diag_doc, "Returns the diagonal of the 3x3 matrix reprsenting the so3 element R.");
PyDoc_STRVAR(so3_deskew_doc, "If R is a (flattened) cross-product matrix of the 3-vector w, this will return w.  Otherwise, it will return a representation w of (R-R^T)/2 (off diagonals of R) such that (R-R^T)/2 = cross_product(w). ");
PyDoc_STRVAR(so3_rotation_doc, "Given a unit axis and an angle in radians, returns the rotation matrix.");
PyDoc_STRVAR(so3_canonical_doc, "Given a unit vector v, finds R that defines a basis [x,y,z] such that x = v and y and z are orthogonal");
PyDoc_STRVAR(so3_vector_rotation_doc, "Finds the minimal-angle matrix that rotates v1 to v2.  v1 and v2 are assumed to be nonzero");
PyDoc_STRVAR(so3_interpolate_doc, "Interpolate linearly between the two rotations R1 and R2. ");
PyDoc_STRVAR(so3_interpolator_doc, "Returns a function of one parameter u that interpolates linearly between the two rotations R1 and R2. After f(u) is constructed, calling f(u) is about 2x faster than calling interpolate(R1,R2,u).");
PyDoc_STRVAR(so3_det_doc, "Returns the determinant of the 3x3 matrix R");
PyDoc_STRVAR(so3_is_rotation_doc, "Returns true if R is a rotation matrix, i.e. is orthogonal to the given tolerance and has + determinant");
PyDoc_STRVAR(so3_sample_doc, "Returns a uniformly distributed rotation matrix.");

static PyMethodDef so3Methods[] = {
    {"identity", so3_identity, METH_NOARGS, so3_identity_doc},
    {"inv", so3_inv, METH_VARARGS, so3_inv_doc},
    {"apply", so3_apply, METH_VARARGS, so3_apply_doc},
    {"matrix", (PyCFunction) so3_matrix, METH_FASTCALL, so3_matrix_doc},
    {"from_matrix", (PyCFunction) so3_from_matrix, METH_FASTCALL, so3_from_matrix_doc},
    {"mul", so3_mul, METH_VARARGS, so3_mul_doc},
    {"trace", so3_trace, METH_VARARGS, so3_trace_doc},
    {"angle", so3_angle, METH_VARARGS, so3_angle_doc},
    {"rpy", so3_rpy, METH_VARARGS, so3_rpy_doc},
    {"from_rpy", so3_from_rpy, METH_VARARGS, so3_from_rpy_doc},
    {"rotation_vector", so3_rotation_vector, METH_VARARGS, so3_rotation_vector_doc},
    {"moment", so3_rotation_vector, METH_VARARGS, so3_rotation_vector_doc},
    {"axis_angle", so3_axis_angle, METH_VARARGS, so3_axis_angle_doc},
    {"from_axis_angle", so3_from_axis_angle, METH_VARARGS, so3_from_axis_angle_doc},
    {"from_rotation_vector", so3_from_rotation_vector, METH_VARARGS, so3_from_rotation_vector_doc},
    {"from_moment", so3_from_rotation_vector, METH_VARARGS, so3_from_rotation_vector_doc},
    {"from_quaternion", so3_from_quaternion, METH_VARARGS, so3_from_quaternion_doc},
    {"quaternion", so3_quaternion, METH_VARARGS, so3_quaternion_doc},
    {"distance", so3_distance, METH_VARARGS, so3_distance_doc},
    {"error", so3_error, METH_VARARGS, so3_error_doc},
    {"cross_product", so3_cross_product, METH_VARARGS, so3_cross_product_doc},
    {"diag", so3_diag, METH_VARARGS, so3_diag_doc},
    {"deskew", so3_deskew, METH_VARARGS, so3_deskew_doc},
    {"rotation", so3_rotation, METH_VARARGS, so3_rotation_doc},
    {"canonical", so3_canonical, METH_VARARGS, so3_canonical_doc},
    {"vector_rotation", so3_vector_rotation, METH_VARARGS, so3_vector_rotation_doc},
    {"interpolate", (PyCFunction) so3_interpolate, METH_FASTCALL, so3_interpolate_doc},
    {"interpolator", (PyCFunction) so3_interpolator, METH_FASTCALL, so3_interpolator_doc},
    {"det", (PyCFunction) so3_det, METH_FASTCALL, so3_det_doc},
    {"is_rotation", (PyCFunction) so3_is_rotation, METH_FASTCALL, so3_is_rotation_doc},
    {"sample", (PyCFunction) so3_sample, METH_NOARGS, so3_sample_doc},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef so3Module = {
    PyModuleDef_HEAD_INIT,
    "motion.so3",
    NULL,   // Documentation
    -1,     /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    so3Methods
};

PyMODINIT_FUNC PyInit_so3() {
    return PyModule_Create(&so3Module);
}

/**
 * Returns the identity rotation
 */
PyObject* so3_identity(PyObject* self, PyObject* args) {
    return vector_to_list(SO3_ID, 9);
}

/**
 * Inverts the rotation
 */
PyObject* so3_inv(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "inv", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double buffer2[9];
    __so3_inv(buffer2, buffer);
    return vector_to_list(buffer2, 9);
}

/**
 * Applies the rotation to a point
 */
PyObject* so3_apply(PyObject* self, PyObject* args) {
    PyObject* rot;
    PyObject* vec;
    if (!PyArg_UnpackTuple(args, "apply", 2, 2, &rot, &vec)) {
        return NULL;
    }
    double r_buf[9];
    double v_buf[3];
    if (parse_rotation(r_buf, rot)) {
        return NULL;
    }
    if (parse_vec3(v_buf, vec)) {
        return NULL;
    }
    double result[3];
    __so3_apply(result, r_buf, v_buf);
    return Py_BuildValue("(ddd)", result[0], result[1], result[2]);
}

/**
 * Returns the 3x3 rotation matrix corresponding to R
 */
PyObject* so3_matrix(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    PyObject* rot = args[0];
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    return __so3_matrix(buffer);
}

/**
 * Returns an R corresponding to the 3x3 rotation matrix mat
 */
PyObject* so3_from_matrix(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    double ret[9];
    if (__so3_from_matrix(ret, args[0])) {
        return NULL;
    }
    return vector_to_list(ret, 9);
}

/**
 * Multiplies two rotations.
 */
PyObject* so3_mul(PyObject* self, PyObject* args) {
    PyObject* r1;
    PyObject* r2;
    if (!PyArg_UnpackTuple(args, "mul", 2, 2, &r1, &r2)) {
        return NULL;
    }
    double r1_buf[9];
    double r2_buf[9];
    double result_buf[9];
    if (parse_rotation(r1_buf, r1)) {
        return NULL;
    }
    if (parse_rotation(r2_buf, r2)) {
        return NULL;
    }
    __so3_mul(result_buf, r1_buf, r2_buf);
    return vector_to_list(result_buf, 9);
}

/**
 * Computes the trace of the rotation matrix.
 */
PyObject* so3_trace(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "inv", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    return PyFloat_FromDouble(__so3_trace(buffer));
}

/**
 * Returns absolute deviation of R from identity
 */
PyObject* so3_angle(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "inv", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    return PyFloat_FromDouble(__so3_angle(buffer));
}

/**
 * Converts a rotation matrix to a roll,pitch,yaw angle triple. The result is given in radians.
 */
PyObject* so3_rpy(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "inv", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double dest[3];
    __so3_rpy(dest, buffer);
    return Py_BuildValue("(ddd)", dest[0], dest[1], dest[2]);
}

/**
 * Converts from roll,pitch,yaw angle triple to a rotation matrix.  The triple is given in radians.  The x axis is 'roll', y is 'pitch', and z is 'yaw'.
 */
PyObject* so3_from_rpy(PyObject* self, PyObject* args) {
    PyObject* rpy;
    if (!PyArg_UnpackTuple(args, "from_rpy", 1, 1, &rpy)) {
        return NULL;
    }
    double rpy_buf[3];
    if (parse_vec3(rpy_buf, rpy)) {
        return NULL;
    }
    double dest[9];
    __so3_from_rpy(dest, rpy_buf);
    return vector_to_list(dest, 9);
}

/**
 * Returns the rotation vector w (exponential map) representation of R such that e^[w] = R.  Equivalent to axis-angle representation with w/||w||=axis, ||w||=angle.
 */
PyObject* so3_rotation_vector(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "diag", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double result[3];
    __so3_rotation_vector(result, buffer);
    return vector_to_list(result, 3);
}

/**
 * Returns the (axis,angle) pair representing R
 */
PyObject* so3_axis_angle(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "diag", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double result[3];
    __so3_rotation_vector(result, buffer);
    double len = __vo_norm(result, 3);
    __vo_unit(result, result, 1e-5, 3);
    PyObject* ret_axis = vector_to_list(result, 3);
#ifdef MOTION_DEBUG
    if (ret_axis == NULL) {
        return NULL;
    }
#endif
    PyObject* ret = Py_BuildValue("(Od)", ret_axis, len);
#ifdef MOTION_DEBUG
    if (ret == NULL) {
        Py_DECREF(ret_axis);
        return NULL;
    }
#endif
    return ret;
}

/**
 * Converts an axis-angle representation (axis,angle) to a 3D rotation matrix.
 */
PyObject* so3_from_axis_angle(PyObject* self, PyObject* args) {
    PyObject* axis;
    double angle;
    if (!PyArg_ParseTuple(args, "(Od)", &axis, &angle)) {
        return NULL;
    }
    double _axis[3];
    if (parse_vec3(_axis, axis)) {
        return NULL;
    }
    double rot[9];
    __so3_rotation(rot, _axis, angle);
    return vector_to_list(rot, 9);
}

/**
 * Converts a rotation vector representation w to a 3D rotation matrix.
 */
PyObject* so3_from_rotation_vector(PyObject* self, PyObject* args) {
    PyObject* vec;
    if (!PyArg_UnpackTuple(args, "from_rotation_vector", 1, 1, &vec)) {
        return NULL;
    }
    double _vec[3];
    if (parse_vec3(_vec, vec)) {
        return NULL;
    }
    double length = __vo_norm(_vec, 3);
    if (length < 1e-7) return vector_to_list(SO3_ID, 9);

    double rot[9];
    __vo_mul(_vec, _vec, 1/length, 3);
    __so3_rotation(rot, _vec, length);
    return vector_to_list(rot, 9);
}

/**
 * Given a unit quaternion (w,x,y,z), produce the corresponding rotation matrix.
 */
PyObject* so3_from_quaternion(PyObject* self, PyObject* args) {
    PyObject* quat;
    if (!PyArg_UnpackTuple(args, "from_quaternion", 1, 1, &quat)) {
        return NULL;
    }
    double _quat[4];
    if (parse_quat(_quat, quat)) {
        return NULL;
    }
    double rot[9];
    __so3_from_quaternion(rot, _quat);
    return vector_to_list(rot, 9);
}

/**
 * Given a Klamp't rotation representation, produces the corresponding unit quaternion (w,x,y,z).
 */
PyObject* so3_quaternion(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "quaternion", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double result[4];
    if (__so3_quaternion(result, buffer)) {
        PyErr_SetString(PyExc_ValueError, "Could not solve for quaternion... Invalid rotation matrix?");
        return NULL;
    }
    return vector_to_list(result, 4);
}

/**
 * Returns the absolute angle one would need to rotate in order to get from R1 to R2
 */
PyObject* so3_distance(PyObject* self, PyObject* args) {
    PyObject* r1;
    PyObject* r2;
    if (!PyArg_UnpackTuple(args, "error", 2, 2, &r1, &r2)) {
        return NULL;
    }
    double r1_buf[9];
    double r2_buf[9];
    if (parse_rotation(r1_buf, r1)) {
        return NULL;
    }
    if (parse_rotation(r2_buf, r2)) {
        return NULL;
    }
    return PyFloat_FromDouble(__so3_distance(r1_buf, r2_buf));
}

/**
 * Returns a 3D 'difference vector' that describes how far R1 is from R2.
 * More precisely, this is the (local) Lie derivative,
 * which is the rotation vector representation of R1*R2^T.
 *
 * Fun fact: this is related to the derivative of interpolate(R2,R1,u) at u=0
 * by d/du interpolate(R2,R1,0) = mul(error(R1,R2),R2).
 */
PyObject* so3_error(PyObject* self, PyObject* args) {
    PyObject* r1;
    PyObject* r2;
    if (!PyArg_UnpackTuple(args, "error", 2, 2, &r1, &r2)) {
        return NULL;
    }
    double r1_buf[9];
    double r2_buf[9];
    double ret[3];
    if (parse_rotation(r1_buf, r1)) {
        return NULL;
    }
    if (parse_rotation(r2_buf, r2)) {
        return NULL;
    }
    __so3_error(ret, r1_buf, r2_buf);
    return vector_to_list(ret, 3);
}

/**
 * Returns the cross product matrix associated with w.
 * The matrix [w]R is the derivative of the matrix R
 * as it rotates about the axis w/||w|| with angular velocity ||w||. 
 */
PyObject* so3_cross_product(PyObject* self, PyObject* args) {
    PyObject* vec;
    if (!PyArg_UnpackTuple(args, "cross_product", 1, 1, &vec)) {
        return NULL;
    }
    double buffer[3];
    if (parse_vec3(buffer, vec)) {
        return NULL;
    }
    double buffer2[9];
    __so3_cross_product(buffer2, buffer);
    return vector_to_list(buffer2, 9);
}

/**
 * Returns the diagonal of the 3x3 matrix reprsenting the so3 element R.
 */
PyObject* so3_diag(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "diag", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double result[3];
    __so3_diag(result, buffer);
    return vector_to_list(result, 3);
}

/**
 * If R is a (flattened) cross-product matrix of the 3-vector w, this will return w.
 * Otherwise, it will return a representation w of (R-R^T)/2
 * (off diagonals of R) such that (R-R^T)/2 = cross_product(w). 
 */
PyObject* so3_deskew(PyObject* self, PyObject* args) {
    PyObject* rot;
    if (!PyArg_UnpackTuple(args, "deskew", 1, 1, &rot)) {
        return NULL;
    }
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    double result[3];
    __so3_deskew(result, buffer);
    return vector_to_list(result, 3);
}

/**
 * Given a unit axis and an angle in radians, returns the rotation matrix.
 */
PyObject* so3_rotation(PyObject* self, PyObject* args) {
    PyObject* axis;
    double angle;
    if (!PyArg_ParseTuple(args, "Od", &axis, &angle)) {
        return NULL;
    }
    double _axis[3];
    double rot[9];
    if (parse_vec3(_axis, axis)) {
        return NULL;
    }
    __so3_rotation(rot, _axis, angle);
    return vector_to_list(rot, 9);
}

/**
 * Given a unit vector v, finds R that defines a basis [x,y,z] such that x = v and y and z are orthogonal
 */
PyObject* so3_canonical(PyObject* self, PyObject* args) {
    PyObject* vec;
    if (!PyArg_UnpackTuple(args, "cross_product", 1, 1, &vec)) {
        return NULL;
    }
    double buffer[3];
    if (parse_vec3(buffer, vec)) {
        return NULL;
    }
    double normsq = __vo_normSquared(buffer, 3);
    if (fabs(normsq - 1.0) > 1e-4) {
        PyErr_SetString(PyExc_ValueError, "Nonunit vector supplied to canonical()");
        return NULL;
    }
    double buffer2[9];
    __so3_canonical(buffer2, buffer);
    return vector_to_list(buffer2, 9);
}

/**
 * Finds the minimal-angle matrix that rotates v1 to v2.  v1 and v2 are assumed to be nonzero
 */
PyObject* so3_vector_rotation(PyObject* self, PyObject* args) {
    PyObject* v1;
    PyObject* v2;
    if (!PyArg_UnpackTuple(args, "error", 2, 2, &v1, &v2)) {
        return NULL;
    }
    double v1_buf[3];
    double v2_buf[3];
    if (parse_vec3(v1_buf, v1)) {
        return NULL;
    }
    if (parse_vec3(v2_buf, v2)) {
        return NULL;
    }
    double output[9];
    __so3_vector_rotation(output, v1_buf, v2_buf);
    return vector_to_list(output, 9);
}

/**
 * Interpolate linearly between the two rotations R1 and R2. 
 */
PyObject* so3_interpolate(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
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
    PyObject* r1 = args[0];
    PyObject* r2 = args[1];
    double c = PyFloat_AsDouble(args[2]);
    double r1_buf[9];
    double r2_buf[9];
    double result_buf[9];
    if (parse_rotation(r1_buf, r1)) {
        return NULL;
    }
    if (parse_rotation(r2_buf, r2)) {
        return NULL;
    }
    __so3_interpolate(result_buf, r1_buf, r2_buf, c);
    return vector_to_list(result_buf, 9);
}

/**
 * "lambda" function for interpolator.
 * First argument: Bytes object with the raw data
 */
static PyObject* __so3_interpolate_f(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
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
    so3_interpolator_t* data = (so3_interpolator_t*) PyBytes_AS_STRING(self);
    double tmp[9];
    double result[9];
    __so3_rotation(tmp, data->axis, data->angle*u);
    __so3_mul(result, data->rot, tmp);
    return vector_to_list(result, 9);
}

static struct PyMethodDef
__so3_interpolator_cb = {.ml_name  = "f@interpolator",
                         .ml_meth  = (PyCFunction) __so3_interpolate_f,
                         .ml_flags = METH_FASTCALL,
                         .ml_doc   = NULL};

/**
 * Returns a function of one parameter u that interpolates linearly between the two rotations R1 and R2. After f(u) is constructed, calling f(u) is about 2x faster than calling interpolate(R1,R2,u).
 */
PyObject* so3_interpolator(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 2)");
        return NULL;
    }
#endif
    double r1[9];
    double r2[9];
    so3_interpolator_t result;
    if (parse_rotation(r1, args[0])) {
        return NULL;
    }
    if (parse_rotation(r2, args[1])) {
        return NULL;
    }
    __so3_interpolator_init(&result, r1, r2);

    PyObject* byte_data = PyBytes_FromStringAndSize(NULL, sizeof(result));
    if (byte_data == NULL) {
        return NULL;
    }
    char* save_data = PyBytes_AS_STRING(byte_data);
    memcpy(save_data, &result, sizeof(result));

    PyObject* ret = PyCFunction_New(&__so3_interpolator_cb, byte_data);
    Py_DECREF(byte_data);
    return ret;
}

/**
 * Returns the determinant of the 3x3 matrix R
 */
PyObject* so3_det(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1)");
        return NULL;
    }
#endif
    PyObject* rot = args[0];
    double buffer[9];
    if (parse_rotation(buffer, rot)) {
        return NULL;
    }
    return PyFloat_FromDouble(__so3_det(buffer));
}

/**
 * Returns true if R is a rotation matrix, i.e. is orthogonal to the given tolerance and has + determinant
 */
PyObject* so3_is_rotation(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs <= 0 || nargs > 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong Number of arguments (expected 1-2)");
        return NULL;
    }
#endif
    PyObject* a = args[0];
    double eps;
    if (nargs == 1) {
        eps = 1e-5;
    }
    else {
#ifdef MOTION_DEBUG
        if (!(PyFloat_Check(args[1]) || PyLong_Check(args[1]))) {
            PyErr_SetString(PyExc_TypeError, "Expected number for args[1]");
            return NULL;
        }
#endif
        eps = PyFloat_AsDouble(args[1]);
    }
    double buffer[9];
    if (parse_rotation(buffer, a)) {
        return NULL;
    }
    return PyBool_FromLong(__so3_is_rotation(buffer, eps));
}

/**
 * Returns a uniformly distributed rotation matrix.
 */
PyObject* so3_sample(PyObject* self, PyObject* args) {
    // Note: This is going to differ subtly from python rng
    double buf[9];
    __so3_sample(buf);
    return vector_to_list(buf, 9);
}
