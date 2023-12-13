#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "types.h"
#include "math/utils.h"
#include "interface.h"
#include "problem.h"
#include "matrix_math.h"

#include <stdio.h>
#include <stdlib.h>

PyDoc_STRVAR(Py_linsolve_doc, "solve by cg for an invertible A.");
PyDoc_STRVAR(Py_init_solver_doc, "allocate space for the solver.");
PyDoc_STRVAR(Py_set_lin_term_doc, "add a linear term to the optimization objective.");
PyDoc_STRVAR(Py_add_quad_term_doc, "add a quad form term to the optimization objective. Use cholesky factor L^T to specify.");
PyDoc_STRVAR(Py_finalize_solver_doc, "turn solver internals into compressed form (no more adding things)");

static PyMethodDef socpMethods[] = {
    {"linsolve", (PyCFunction) Py_linsolve, METH_FASTCALL, Py_linsolve_doc},
    {"init_solver", (PyCFunction) Py_init_solver, METH_NOARGS, Py_init_solver_doc},
    {"set_lin_term", (PyCFunction) Py_set_lin_term, METH_FASTCALL, Py_set_lin_term_doc},
    {"add_quad_term", (PyCFunction) Py_add_quad_term, METH_FASTCALL, Py_add_quad_term_doc},
    {"finalize_solver", (PyCFunction) Py_finalize_solver, METH_NOARGS, Py_finalize_solver_doc},
/*
    {"sub", (PyCFunction) vectorops_sub, METH_FASTCALL, vectorops_sub_doc},
    {"mul", (PyCFunction) vectorops_mul, METH_FASTCALL, vectorops_mul_doc},
    {"div", (PyCFunction) vectorops_div, METH_FASTCALL, vectorops_div_doc},
    {"maximum", (PyCFunction) vectorops_maximum, METH_FASTCALL, vectorops_maximum_doc},
    {"minimum", (PyCFunction) vectorops_minimum, METH_FASTCALL, vectorops_minimum_doc},
    {"dot", (PyCFunction) vectorops_dot, METH_FASTCALL, vectorops_dot_doc},
    {"normSquared", (PyCFunction) vectorops_normSquared, METH_FASTCALL, vectorops_normSquared_doc},
    {"norm", (PyCFunction) vectorops_norm, METH_FASTCALL, vectorops_norm_doc},
    {"unit", (PyCFunction) vectorops_unit, METH_FASTCALL, vectorops_unit_doc},
    {"norm_L2", (PyCFunction) vectorops_norm, METH_FASTCALL, vectorops_norm_doc},
    {"norm_L1", (PyCFunction) vectorops_norm_L1, METH_FASTCALL, vectorops_norm_L1_doc},
    {"norm_Linf", (PyCFunction) vectorops_norm_Linf, METH_FASTCALL, vectorops_norm_Linf_doc},
    {"distanceSquared", (PyCFunction) vectorops_distanceSquared, METH_FASTCALL, vectorops_distanceSquared_doc},
    {"distance", (PyCFunction) vectorops_distance, METH_FASTCALL, vectorops_distance_doc},
    {"cross", (PyCFunction) vectorops_cross, METH_FASTCALL, vectorops_cross_doc},
    {"interpolate", (PyCFunction) vectorops_interpolate, METH_FASTCALL, vectorops_interpolate_doc},
 */
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef socpModule = {
    PyModuleDef_HEAD_INIT,
    "socplib.socp",
    NULL,   // Documentation
    -1,     /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    socpMethods
};

cg_scratch space;

PyMODINIT_FUNC PyInit_socp() {
    memset(&space, 0, sizeof(space));
    return PyModule_Create(&socpModule);
}

int parse_py_coo(Vector/*<matrix_entry*>*/* dest, PyObject* coo) {
    Py_ssize_t n = PyObject_Length(coo);
#ifdef MOTION_DEBUG
    if (n < 0) {
        PyErr_SetString(PyExc_TypeError, "object has no length");
        return 1;
    }
#endif
    PyObject* it = PyObject_GetIter(coo);
#ifdef MOTION_DEBUG
    if (it == NULL) {
        return 1;
    }
#endif

    Vector_clear_free(dest, n);

    numeric buf[3];
    PyObject* curr;
    while ((curr = PyIter_Next(it))) {
        if (parse_vec3(buf, curr)) {
            Py_DECREF(curr);
            Py_DECREF(it);
            return 1;
        }
        Vector_push(dest, make_matrix_entry(buf[0], buf[1], buf[2]));
        Py_DECREF(curr);
    }
    Py_DECREF(it);
    return 0;
}

PyObject* Py_linsolve(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs < 2) {
        PyErr_SetString(PyExc_TypeError, "Not enough arguments (expected 2+)");
    }
#endif
    PyObject* first = args[0];

    Vector coo_mat;
    inplace_make_Vector(&coo_mat, 1);
    if (parse_py_coo(&coo_mat, first)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse coo matrix A");
        return NULL;
    }
    PyObject* second = args[1];
    Py_ssize_t n = PyObject_Length(second);

    numeric* b = malloc(n*sizeof(numeric));
    if (list_to_vector(second, b)) {
        return NULL;
    }

    csr_mat* A = csr_from_entries(n, n, &coo_mat);
    Vector_clear_free(&coo_mat, 1);

    numeric* res = malloc(n*sizeof(numeric));
    if (nargs >= 3) {
        PyObject* third = args[2];
        list_to_vector_n(third, res, n);
    }
    else {
        memset(res, 0, sizeof(numeric)*n);
    }
    size_t niter = linsolve_cg(res, &space, (matrix_like*) A, spmv, transpose_spmv, b, 1e-10);
    printf("C cg solve: %ld iterations\n", niter);

    PyObject* ret = vector_to_list(res, n);
    free(b);
    free(res);
    csr_dealloc(A);
    free(A);
    return ret;
}

solver_build solver_build_struct;
solver_data problem;

PyObject* Py_init_solver(PyObject* self, PyObject* args) {
    solver_build_init(&solver_build_struct);
    Py_RETURN_NONE;
}

PyObject* Py_set_lin_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 1) {
        PyErr_SetString(PyExc_TypeError, "Wrong number of arguments (expected 1)");
    }
#endif
    PyObject* first = args[0];
    Py_ssize_t n = PyObject_Length(first);
    numeric* c_data = malloc(n*sizeof(numeric));
    if (list_to_vector(first, c_data)) {
        return NULL;
    }

    solver_build_set_lin_term(&solver_build_struct, c_data, n);
    Py_RETURN_NONE;
}

PyObject* Py_add_quad_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs) {
#ifdef MOTION_DEBUG
    if (nargs != 2) {
        PyErr_SetString(PyExc_TypeError, "Wrong number of arguments (expected 2)");
    }
    if (!(PyFloat_Check(args[1]) || PyLong_Check(args[1]))) {
        PyErr_SetString(PyExc_TypeError, "Expected number for args[1]");
        return NULL;
    }
#endif
    PyObject* first = args[0];
    double n = PyFloat_AsDouble(args[1]);

    Vector coo_mat;
    inplace_make_Vector(&coo_mat, 1);
    if (parse_py_coo(&coo_mat, first)) {
        PyErr_SetString(PyExc_ValueError, "Failed to parse coo matrix A");
        return NULL;
    }

    solver_build_add_quad_term(&solver_build_struct, &coo_mat, n);
    Py_RETURN_NONE;
}

PyObject* Py_finalize_solver(PyObject* self, PyObject* args) {
    solver_build_freeze(&problem, &solver_build_struct);
    print_csr_entries(problem.A);
    print_vector(problem.b, problem.A->n_rows);
    print_vector(problem.c, problem.A->n_cols);
    Py_RETURN_NONE;
}
