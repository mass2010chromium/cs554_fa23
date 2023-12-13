#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "ur5controller.h"

static PyMethodDef UR5Methods[] = {
    {"test", ur5_test, METH_VARARGS, "test lol"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef UR5Module = {
    PyModuleDef_HEAD_INIT,
    "ur5controller",
    NULL,   // Documentation
    -1,     /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    UR5Methods
};

PyMODINIT_FUNC PyInit_ur5controller() {
    return PyModule_Create(&UR5Module);
}

PyObject* ur5_test(PyObject* self, PyObject* args) {
    const char* msg;
    if (!PyArg_ParseTuple(args, "s", &msg)) {
        // Exception occured in parse.
        return NULL;
    }
    printf("%s\n", msg);

    Py_INCREF(Py_None);
    return Py_None;
}

