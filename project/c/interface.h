#pragma once
#include "math/utils.h"

PyObject* Py_linsolve(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_init_solver(PyObject* self, PyObject* args);

PyObject* Py_set_lin_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_add_linear_constraint(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_add_quad_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_add_l2_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_finalize_solver(PyObject* self, PyObject* args);

PyObject* Py_socp_solve(PyObject* self, PyObject* args);

PyObject* Py_dealloc_solver(PyObject* self, PyObject* args);
