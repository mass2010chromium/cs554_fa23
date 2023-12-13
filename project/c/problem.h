#include "types.h"
#include "matrix_math.h"
#include "structures/Vector.h"

typedef struct solver_data {
    csr_mat* A; // m x n. m = num vars.
    union {
        numeric* h; // length: m+n. has allocated mem
        numeric* c; // length: m, pointer into h
    };
    numeric* b; // length: n, pointer into
    size_t unconstrained_dim;
    size_t* cone_dims;  // List of second-order cone dimensions.
    size_t num_cones;

    // Scratch space.
    cg_scratch cg_space;
    numeric* M_inv_h;
    numeric* M_inv_w;
    numeric* uv_scratch;
    numeric* K;
    numeric* u;
    numeric* v;
    numeric* u_tilde;
} solver_data;

typedef struct solver_build {
    Vector/* <matrix_entry*> */ A_data;
    Vector/* <numeric> */ b_data;
    Vector/* <numeric> */ c_data;
    size_t unconstrained_dim;
    Vector/* <numeric*> */ cones;
} solver_build;

void solver_build_init(solver_build* res);

void solver_build_set_lin_term(solver_build* self, numeric* c_data, size_t n);

// Consumes L_transpose! Steals all its elements and destroys it (but does not free it).
void solver_build_add_quad_term(solver_build* self, Vector/*<matrix_entry*>*/* L_transpose, size_t num_vars);

// res should be uninitialized (just space).
void solver_build_freeze(solver_data* res, solver_build* input);

void dump_solver(solver_data* problem);

size_t solve(numeric* res, solver_data* problem);
