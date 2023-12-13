#include "matrix_math.h"
#include "types.h"
#include "math/vectorops.h"
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

matrix_entry* make_matrix_entry(size_t r, size_t c, numeric v) {
    matrix_entry* entry = malloc(sizeof(matrix_entry));
    entry->r = r;
    entry->c = c;
    entry->v = v;
    return entry;
}
matrix_entry* copy_matrix_entry(matrix_entry* m) {
    return make_matrix_entry(m->r, m->c, m->v);
}

csr_mat* csr_alloc(size_t n_rows, size_t n_cols, size_t nnz) {
    csr_mat* ret = malloc(sizeof(csr_mat));
    ret->n_rows = n_rows;
    ret->n_cols = n_cols;
    ret->row_index = malloc((n_rows+1) * sizeof(size_t));
    ret->col_index = malloc(nnz * sizeof(size_t));
    ret->values = malloc(nnz * sizeof(numeric));
    return ret;
}
void csr_dealloc(csr_mat* m) {
    free(m->row_index);
    free(m->col_index);
    free(m->values);
    m->row_index = NULL;
    m->col_index = NULL;
    m->values = NULL;
}

size_t csr_nnz(csr_mat* m) {
    return m->row_index[m->n_rows];
}

int _matrix_entry_cmp(void* _a, void* _b) {
    matrix_entry* a = _a;
    matrix_entry* b = _b;
    if (a->r > b->r) { return 1; }
    if (a->r < b->r) { return -1; }
    if (a->c > b->c) { return 1; }
    if (a->c < b->c) { return -1; }
    if (a->v > b->v) { return 1; }
    if (a->v < b->v) { return -1; }
    return 0;
}

csr_mat* csr_from_entries(size_t n_rows, size_t n_cols, Vector/*<matrix_entry*>*/* items) {
    Vector_sort(items, _matrix_entry_cmp);
    size_t nnz = 0;
    size_t cur_col = -1;
    size_t cur_row = 0;
    for (size_t i = 0; i < items->size; ++i) {
        matrix_entry* entry = (matrix_entry*) items->elements[i];
        if (entry->c != cur_col || entry->r != cur_row) {
            cur_col = entry->c;
            cur_row = entry->r;
            nnz += 1;
        }
    }
    csr_mat* ret = csr_alloc(n_rows, n_cols, nnz);
    for (size_t i = 0; i < nnz; ++i) {
        ret->values[i] = 0;
    }
    
    // Yes its overflowing. No I don't care.
    size_t v_idx = -1;
    cur_col = -1;
    cur_row = 0;
    ret->row_index[0] = 0;
    size_t* cptr = ret->col_index - 1;

    for (size_t i = 0; i < items->size; ++i) {
        matrix_entry* entry = (matrix_entry*) items->elements[i];
        if (entry->c != cur_col || entry->r != cur_row) {
            ++v_idx;
            ++cptr;
            cur_col = entry->c;
            *cptr = cur_col;
        }
        ret->values[v_idx] += entry->v;
        while (cur_row < entry->r) {
            ++cur_row;
            ret->row_index[cur_row] = v_idx;
        }
    }
    ret->row_index[cur_row+1] = v_idx+1;
    return ret;
}

void print_csr_entries(csr_mat* mat) {
    printf("(%ld x %ld) sparse matrix:\n", mat->n_rows, mat->n_cols);
    for (size_t r = 0; r < mat->n_rows; ++r) {
        for (size_t ci = mat->row_index[r]; ci < mat->row_index[r+1]; ++ci) {
            printf("    (%ld, %ld): %f\n", r, mat->col_index[ci], mat->values[ci]);
        }
    }
}

// Out should point to an initialized, empty vector.
void csr_to_coo(Vector/*<matrix_entry*>*/* out, csr_mat* m) {
    for (size_t r = 0; r < m->n_rows; ++r) {
        for (size_t ci = m->row_index[r]; ci < m->row_index[r+1]; ++ci) {
            size_t c = m->col_index[ci];
            numeric v = m->values[ci];
            Vector_push(out, make_matrix_entry(r, c, v));
        }
    }
}


// Math operations.
void spmv(numeric* res, csr_mat* m, numeric* v) {
    for (size_t r = 0; r < m->n_rows; ++r) {
        numeric sum = 0;
        for (size_t ci = m->row_index[r]; ci < m->row_index[r+1]; ++ci) {
            size_t c = m->col_index[ci];
            sum += m->values[ci] * v[c];
        }
        res[r] = sum;
    }
}

// the access pattern is also pretty bad. Check if its worth precomputing A^T.
void transpose_spmv(numeric* res, csr_mat* m, numeric* v) {
    memset(res, 0, sizeof(numeric)*m->n_cols);
    for (size_t r = 0; r < m->n_rows; ++r) {
        for (size_t ci = m->row_index[r]; ci < m->row_index[r+1]; ++ci) {
            size_t c = m->col_index[ci];
            res[c] += m->values[ci] * v[r];
        }
    }
}

// see: https://en.wikipedia.org/wiki/Conjugate_gradient_method
// A is m x n.
size_t linsolve_cg(numeric* res, cg_scratch* space, matrix_like* A,
                                                    void(*A_matmul)(numeric*, matrix_like*, numeric*),
                                                    void(*AT_matmul)(numeric*, matrix_like*, numeric*),
                                                    numeric* b, numeric eps) {
    //size_t m = A->n_rows;
    size_t n = A->n_cols;
    size_t n_iters = 0;

    numeric* r;
    numeric* ATb;
    numeric* scratch1;
    numeric* scratch2;
    numeric* p;
    if (space == NULL) {
        r = malloc(sizeof(numeric)*n);
        ATb = malloc(sizeof(numeric)*n);
        scratch1 = malloc(sizeof(numeric)*n);
        scratch2 = malloc(sizeof(numeric)*n);
        p = malloc(sizeof(numeric)*n);
    }
    else {
        r = realloc(space->r, sizeof(numeric)*n);
        ATb = realloc(space->ATb, sizeof(numeric)*n);
        scratch1 = realloc(space->scratch1, sizeof(numeric)*n);
        scratch2 = realloc(space->scratch2, sizeof(numeric)*n);
        p = realloc(space->p, sizeof(numeric)*n);
        space->r = r;
        space->ATb = ATb;
        space->scratch1 = scratch1;
        space->scratch2 = scratch2;
        space->p = p;
    }
    AT_matmul(ATb, A, b);

    //r = b - A @ x
    A_matmul(r, A, res);
    AT_matmul(scratch1, A, r);
    __vo_subv(r, ATb, scratch1, n);
    if (__vo_norm(r, n) < eps) {
        goto csr_linsolve_ret;
    }

    // lol no copy operation. it'll probably be optimized out as a +0 macro
    __vo_add(p, r, 0, n);
    numeric r_dot = __vo_dot(r, r, n);
    for (;; ++n_iters) {
        // alpha = <r, r> / <p, p>_A
        A_matmul(scratch1, A, p);
        AT_matmul(scratch2, A, scratch1);
        numeric alpha = r_dot / __vo_dot(scratch1, scratch1, n);

        // x = x + alpha*p
        __vo_mul(scratch1, p, alpha, n);
        __vo_addv(res, res, scratch1, n);

        //r = r - alpha*p
        __vo_mul(scratch2, scratch2, alpha, n);
        __vo_subv(r, r, scratch2, n);
        if (__vo_norm(r, n) < eps) {
            goto csr_linsolve_ret;
        }

        numeric r_dot2 = __vo_dot(r, r, n);
        numeric beta = r_dot2 / r_dot;
        r_dot = r_dot2;

        // p = r + beta*p
        __vo_mul(p, p, beta, n);
        __vo_addv(p, p, r, n);
    }

csr_linsolve_ret:
    if (space == NULL) {
        free(r);
        free(ATb);
        free(scratch1);
        free(scratch2);
        free(p);
    }
    return n_iters;
}
