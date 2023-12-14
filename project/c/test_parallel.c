#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include "types.h"
#include "parallel.h"


typedef struct outer_prod_dat {
    int ret;
    numeric* left;
    size_t left_N;
    numeric* right;
    size_t right_N;
    numeric* out_buf;
} outer_prod_dat;

int outer_prod(job_spec* arg) {
    outer_prod_dat* data = (outer_prod_dat*) arg;
    numeric* res = data->out_buf;
    size_t left_N = data->left_N;
    size_t right_N = data->right_N;
    numeric* left = data->left;
    numeric* right = data->right;
    for (size_t i = 0; i < left_N; ++i) {
        for (size_t j = 0; j < right_N; ++j) {
            res[i*right_N+j] = left[i] * right[j];
        }
    }
    return 0;
}

#include "cone.h"

typedef struct cone_proj_dat {
    int ret;
    numeric* x;
    size_t* cone_dims;
    size_t n_cones;
} cone_proj_dat;

#define PROJ_ITERS 1

int cone_proj(job_spec* arg) {
    cone_proj_dat* data = (cone_proj_dat*) arg;
    for (size_t i = 0; i < PROJ_ITERS; ++i) {
        project_to_socs(data->x, data->cone_dims, data->n_cones);
    }
    return 0;
}

// note: this distribution is not great. But I don't care!
numeric rand_float() {
    return rand() / (numeric) RAND_MAX;
}

void run(N, nthreads) {
    printf("N=%ld\n", N);
    numeric* buf = calloc(N, sizeof(numeric));
    numeric* res1 = calloc(N, sizeof(numeric));
    numeric* res2 = calloc(N, sizeof(numeric));
    size_t* cones = malloc(N * sizeof(size_t) / 4);

    for (size_t i = 0; i < N; ++i) {
        buf[i] = 2*rand_float() - 1;
    }
    size_t n_cones = N / 4;
    for (size_t i = 0; i < n_cones; ++i) {
        cones[i] = 4;
    }

    struct timespec start, end;
    uint64_t dt;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    memcpy(res1, buf, N*sizeof(numeric));
    for (size_t i = 0; i < PROJ_ITERS; ++i) {
        project_to_socs(res1, cones, n_cones);
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    dt = (end.tv_sec - start.tv_sec)*1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("serial time taken: \n\t%ld nanoseconds\n", dt);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    dt = (end.tv_sec - start.tv_sec)*1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("serial time taken: \n\t%ld nanoseconds\n", dt);

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    worker_batch* workers = make_worker_batch(nthreads);

    size_t cone_offset = 0;
    numeric* buf_ptr = res2;
    size_t blocksize = ceil(n_cones / (numeric)nthreads);
    cone_proj_dat job_data[nthreads];
    for (size_t i = 0; i < nthreads; ++i) {
        size_t end_n = cone_offset + blocksize;
        if (end_n > n_cones) { end_n = n_cones; }
        size_t this_blocksize = end_n - cone_offset;

        size_t total_cone_size = 0;
        for (size_t j = 0; j < this_blocksize; ++j) {
            total_cone_size += cones[cone_offset+j];
        }

        job_data[i].x = buf_ptr;
        job_data[i].cone_dims = cones + cone_offset;
        job_data[i].n_cones = this_blocksize;

        cone_offset += blocksize;
        buf_ptr += total_cone_size;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    memcpy(res2, buf, N*sizeof(numeric));

    send_job_homogenous(workers, cone_proj, (job_spec*) job_data, sizeof(cone_proj_dat));
    batch_wait(workers);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    dt = (end.tv_sec - start.tv_sec)*1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("parallel time taken (%ld threads): \n\t%ld nanoseconds\n", nthreads, dt);

    if (memcmp(res1, res2, N*sizeof(numeric))) {
        printf("output arrays differ!\n");
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    memcpy(res1, buf, N*sizeof(numeric));
    for (size_t i = 0; i < PROJ_ITERS; ++i) {
        project_to_socs(res1, cones, n_cones);
    }
    memcpy(res2, buf, N*sizeof(numeric));

    send_job_homogenous(workers, cone_proj, (job_spec*) job_data, sizeof(cone_proj_dat));
    batch_wait(workers);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    dt = (end.tv_sec - start.tv_sec)*1000000000 + (end.tv_nsec - start.tv_nsec);
    printf("parallel time taken (%ld threads): \n\t%ld nanoseconds\n", nthreads, dt);

    destroy_worker_batch(workers);
    free(workers);

}

// NOTE: parallelization experiment results
// It's only worth parallelizing computations on the order of 10^6.
// Outer product (1000x1000): parallelize with 4 threads
// cone projection (1024000 x 4): parallelize with 4 threads.
//      I don't think this actually happens so we just don't parallelize.
int main(int argc, char** argv) {
    srand(0);

    size_t N, nthreads;
    if (argc == 1) {
        size_t min_sz = 1024;
        nthreads = 2;
        for (size_t i = 0; i < 4; ++i) {
            N = min_sz;
            for (size_t j = 0; j < 6; ++j) {
                run(N, nthreads);
                N = N * 10;
            }
            nthreads = nthreads * 2;
        }
        return 0;
    }

    N = 1024;
    if (argc > 1) {
        N = strtoll(argv[1], NULL, 10);
    }
    nthreads = 2;
    if (argc > 2) {
        nthreads = strtoll(argv[2], NULL, 10);
    }
    run(N, nthreads);
    return 0;
}
