#pragma once
#include "math/utils.h"

PY_FUNC(PyObject* Py_linsolve(PyObject* self, PyObject* const* args, Py_ssize_t nargs));

PyObject* Py_init_solver(PyObject* self, PyObject* args);

PyObject* Py_set_lin_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_add_quad_term(PyObject* self, PyObject* const* args, Py_ssize_t nargs);

PyObject* Py_finalize_solver(PyObject* self, PyObject* args);
