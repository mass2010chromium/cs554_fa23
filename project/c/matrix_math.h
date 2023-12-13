#pragma once

#include "types.h"
#include "structures/Vector.h"

typedef struct matrix_like {
    size_t n_rows;
    size_t n_cols;
} matrix_like;

typedef struct csr_mat {
    size_t n_rows;
    size_t n_cols;
    size_t* row_index;
    size_t* col_index;
    numeric* values;
} csr_mat;

typedef struct matrix_entry {
    size_t r;
    size_t c;
    numeric v;
} matrix_entry;

matrix_entry* make_matrix_entry(size_t r, size_t c, numeric v);
matrix_entry* copy_matrix_entry(matrix_entry* m);

csr_mat* csr_alloc(size_t n_rows, size_t n_cols, size_t nnz);
void csr_dealloc(csr_mat* m);

size_t csr_nnz(csr_mat* m);

csr_mat* csr_from_entries(size_t n_rows, size_t n_cols, Vector/*<matrix_entry*>*/* items);
void print_csr_entries(csr_mat* mat);

// Out should point to an initialized, empty vector.
void csr_to_coo(Vector/*<matrix_entry*>*/* out, csr_mat* m);

// Math operations.
void spmv(numeric* res, csr_mat* m, numeric* v);
void transpose_spmv(numeric* res, csr_mat* m, numeric* v);

typedef struct cg_scratch {
    numeric* r;
    union {
        numeric* ATb;
        numeric* b;
    };
    numeric* scratch1;
    numeric* scratch2;
    numeric* p;
} cg_scratch;
// res should be initial guess, or zeroed out.
size_t linsolve_cg(numeric* res, cg_scratch* space, matrix_like* A,
                                                    void(*A_matmul)(numeric*, matrix_like*, numeric*),
                                                    void(*AT_matmul)(numeric*, matrix_like*, numeric*),
                                                    numeric* b, numeric eps);
