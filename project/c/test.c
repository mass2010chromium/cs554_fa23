#include "matrix_math.h"
#include "structures/Vector.h"

#include <stdio.h>
#include <stdlib.h>

#define n_entries 25

int main() {
    const size_t n_rows = 5;
    const size_t n_cols = 5;
    matrix_entry entries[n_entries] = {
{0, 0, 0.5488135039273248},
{0, 1, 0.7151893663724195},
{0, 2, 0.6027633760716439},
{0, 3, 0.5448831829968969},
{0, 4, 0.4236547993389047},
{1, 0, 0.6458941130666561},
{1, 1, 0.4375872112626925},
{1, 2, 0.8917730007820798},
{1, 3, 0.9636627605010293},
{1, 4, 0.3834415188257777},
{2, 0, 0.7917250380826646},
{2, 1, 0.5288949197529045},
{2, 2, 0.5680445610939323},
{2, 3, 0.925596638292661},
{2, 4, 0.07103605819788694},
{3, 0, 0.08712929970154071},
{3, 1, 0.02021839744032572},
{3, 2, 0.832619845547938},
{3, 3, 0.7781567509498505},
{3, 4, 0.8700121482468192},
{4, 0, 0.978618342232764},
{4, 1, 0.7991585642167236},
{4, 2, 0.46147936225293185},
{4, 3, 0.7805291762864555},
{4, 4, 0.11827442586893322},
    };

    
    Vector entries_vec;
    inplace_make_Vector(&entries_vec, n_entries);
    for (size_t i = 0; i < n_entries; ++i) {
        Vector_push(&entries_vec, &entries[i]);
    }

    csr_mat* csr = csr_from_entries(n_rows, n_cols, &entries_vec);

    for (size_t r = 0; r < n_rows; ++r) {
        for (size_t ci = csr->row_index[r]; ci < csr->row_index[r+1]; ++ci) {
            printf("(%ld, %ld): %f\n", r, csr->col_index[ci], csr->values[ci]);
        }
    }
    printf("------\n");

    numeric b[5] = {0.63992102, 0.14335329, 0.94466892, 0.52184832, 0.41466194};
    numeric x[5];

    linsolve_cg(x, NULL, csr, spmv, transpose_spmv, b, 1e-10);
    printf("[ ");
    for (size_t r = 0; r < n_rows; ++r) {
        printf("%f ", x[r]);
    }
    printf("]\n");

    csr_dealloc(csr);
    free(csr);
    Vector_destroy(&entries_vec);

    return 0;
}
