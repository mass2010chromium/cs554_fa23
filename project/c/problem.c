#include "problem.h"
#include <stdlib.h>
#include <string.h>
#include "structures/Vector.h"
#include "math/utils.h"
#include "math/vectorops.h"
#include "matrix_math.h"
#include "cone.h"

void solver_build_init(solver_build* res) {
    inplace_make_Vector(&res->A_data, 8);
    inplace_make_Vector(&res->b_data, 8);
    inplace_make_Vector(&res->c_data, 8);
    res->unconstrained_dim = 0;
    inplace_make_Vector(&res->cones, 8);
}

#include "cone_constructions.c"

// res should be uninitialized (just space).
void solver_build_freeze(solver_data* res, solver_build* input) {
    res->unconstrained_dim = input->unconstrained_dim;
    size_t n_rows = input->b_data.size;
    size_t n_cols = input->c_data.size;
    res->A = csr_from_entries(n_rows, n_cols, &input->A_data);

    res->h = malloc((n_rows + n_cols) * sizeof(numeric));
    // res->c = res->c;
    res->b = res->h + n_cols;
    for (size_t i = 0; i < n_rows; ++i) {
        res->b[i] = Vector_getRaw(&input->b_data, i, numeric);
    }
    for (size_t i = 0; i < n_cols; ++i) {
        res->c[i] = Vector_getRaw(&input->c_data, i, numeric);
    }

    size_t n_cones = input->cones.size;
    res->num_cones = n_cones;
    res->cone_dims = malloc(n_cones * sizeof(size_t));
    for (size_t i = 0; i < n_cones; ++i) {
        res->cone_dims[i] = (size_t) input->cones.elements[i];
    }
    res->cg_space.r = malloc(n_cols * sizeof(numeric));
    res->cg_space.b = malloc(n_cols * sizeof(numeric));
    res->cg_space.scratch1 = malloc(n_cols * sizeof(numeric));
    res->cg_space.scratch2 = malloc(n_cols * sizeof(numeric));
    res->cg_space.p = malloc(n_cols * sizeof(numeric));

    res->M_inv_h = malloc((n_rows+n_cols) * sizeof(numeric));
    memset(res->M_inv_h, 0, (n_rows+n_cols) * sizeof(numeric));
    res->M_inv_w = malloc((n_rows+n_cols) * sizeof(numeric));
    memset(res->M_inv_w, 0, (n_rows+n_cols) * sizeof(numeric));
    res->uv_scratch = malloc((n_rows+n_cols) * sizeof(numeric));

    res->K = malloc((n_rows+n_cols) * sizeof(numeric));
    res->u = malloc((n_rows+n_cols) * sizeof(numeric));       // excluding tau
    res->v = malloc((n_rows+n_cols) * sizeof(numeric));       // excluding tau
    res->u_tilde = malloc((n_rows+n_cols) * sizeof(numeric)); // excluding tau
}

// We are interested in solving this system
// |  I A^T | | z_x | + | w_x |
// | -A  I  | | z_y |   | w_y |
//
// this cg solves
//   z_x = (I + A^TA)^-1 b
//
// where
//   b = (w_x - A^Tw_y)
//
// (do that part yourself!)
size_t linsolve_cg_custom(numeric* res, cg_scratch* space, csr_mat* A, numeric* b, numeric eps) {
    //size_t m = A->n_rows;
    size_t n = A->n_cols;

    numeric* r = space->r;
    numeric* scratch1 = space->scratch1;
    numeric* scratch2 = space->scratch2;
    numeric* p = space->p;

    //r = b - A @ x
    spmv(scratch1, A, res);
    transpose_spmv(scratch2, A, scratch1);
    __vo_addv(scratch2, scratch2, res, n);
    __vo_subv(r, b, scratch2, n);
    if (__vo_norm(r, n) < eps) {
        return 0;
    }

    // lol no copy operation. it'll probably be optimized out as a +0 macro
    __vo_add(p, r, 0, n);
    numeric r_dot = __vo_dot(r, r, n);
    size_t n_iters = 0;
    for (;; ++n_iters) {
        // alpha = <r, r> / <p, p>_A
        spmv(scratch1, A, p);
        transpose_spmv(scratch2, A, scratch1);
        __vo_addv(scratch2, scratch2, p, n);
        numeric denom = __vo_dot(scratch2, p, n);
        numeric alpha = r_dot / denom;

        // x = x + alpha*p
        __vo_mul(scratch1, p, alpha, n);
        __vo_addv(res, res, scratch1, n);

        //r = r - alpha*p
        __vo_mul(scratch2, scratch2, alpha, n);
        __vo_subv(r, r, scratch2, n);
        if (__vo_norm(r, n) < eps) {
            return n_iters;
        }

        numeric r_dot2 = __vo_dot(r, r, n);
        numeric beta = r_dot2 / r_dot;
        r_dot = r_dot2;

        // p = r + beta*p
        __vo_mul(p, p, beta, n);
        __vo_addv(p, p, r, n);
    }
}

// solve for (M^-1) . rhs.
// result must be initialized (either previous, or just zeros is fine.
static size_t solve_M_inverse(numeric* result, solver_data* problem, numeric* rhs, numeric tolerance) {
    size_t m = problem->A->n_rows;
    size_t n = problem->A->n_cols;

    transpose_spmv(problem->cg_space.b, problem->A, rhs + n);
    __vo_subv(problem->cg_space.b, rhs, problem->cg_space.b, n);

    // NOTE: this will only solve the for z_x (the top part of M_inv_h).
    size_t niter = linsolve_cg_custom(result, &problem->cg_space, problem->A, problem->cg_space.b, tolerance);

    // Fill the bottom part. (set z_y = Az_x + w_y)
    numeric* M_inv_x_b = result + n;
    spmv(M_inv_x_b, problem->A, result);
    __vo_addv(M_inv_x_b, M_inv_x_b, rhs + n, m);
    return niter;
}

size_t solve(numeric* res, solver_data* problem) {
    size_t n_iters = 0;
    size_t m = problem->A->n_rows;
    size_t n = problem->A->n_cols;

    // PRESOLVE: solve for (M^-1 h) as documented.
    size_t presolve_niter = solve_M_inverse(problem->M_inv_h, problem, problem->b, 1e-5);

    numeric h_M_inv_h = __vo_dot(problem->h, problem->M_inv_h, m + n);
    numeric alpha = 1.0 / (1.0 + __vo_dot(problem->h, problem->M_inv_h, m+n));

    memset(problem->u, 0, (m + n) * sizeof(numeric));
    memset(problem->v, 0, (m + n) * sizeof(numeric));
    numeric u_t = 1;
    numeric v_t = 1;
    numeric u_tilde_t;

    // K = alpha(M_inv_h h^T M_inv_h)
    __vo_mul(problem->K, problem->M_inv_h, alpha * h_M_inv_h, m + n);

    numeric tolerance = 1e-5;
    for (;; ++n_iters) {
        // Step 1: projection onto the affine subspace.
        //
        // ~u = (M^{-1} - (M^{-1}h h^T M^{-1})/(1 + h^T M^{-1} h)) (w - w_t.h)
        //    = M_inv_w - w_t.M_inv_h - (M_inv_h h^T M_inv_w - w_t.M_inv_h h^T M_inv_h) / (1 + h^T M_inv_h)
        //    = M_inv_w - alpha(M_inv_h h^T M_inv_w) + alpha(M_inv_h h^T M_inv_h) - w_t.M_inv_h 
        //    = M_inv_w - alpha(M_inv_h h^T M_inv_w) + K - w_t.M_inv_h 
        //
        // Precomputed:
        //       alpha = 1 / (1 + h^T M_inv_h)
        //           K = alpha(M_inv_h h^T M_inv_h)
        size_t w_t = u_t + v_t;
        __vo_addv(problem->uv_scratch, problem->u, problem->v, m + n);
        solve_M_inverse(problem->M_inv_w, problem, problem->uv_scratch, tolerance);
        numeric h_M_inv_w = __vo_dot(problem->h, problem->M_inv_w, m + n);
        __vo_madd(problem->u_tilde, problem->M_inv_w, problem->M_inv_h, -h_M_inv_w * alpha, m + n);
        __vo_addv(problem->u_tilde, problem->u_tilde, problem->K, m + n);
        __vo_madd(problem->u_tilde, problem->u_tilde, problem->M_inv_h, -w_t, m + n);
        u_tilde_t = w_t + __vo_dot(problem->M_inv_w, problem->h, m + n);


        // Step 2: project onto the cone.
        //
        // u = proj(~u - v)
        __vo_subv(problem->u, problem->u_tilde, problem->v, m + n);
        numeric* proj_start = problem->u + problem->unconstrained_dim;
        project_to_socs(proj_start, problem->cone_dims, problem->num_cones);
        // complementary vars stay positive.
        u_t = u_tilde_t - v_t;
        if (u_t < 0) { u_t = 0; }


        // Step 3: update v.
        __vo_subv(problem->v, problem->v, problem->u_tilde, m+n);
        __vo_addv(problem->v, problem->u, problem->u, m+n);
        v_t = v_t - u_tilde_t + u_t;

        tolerance = tolerance * 0.9;
    }
    return n_iters;
}
