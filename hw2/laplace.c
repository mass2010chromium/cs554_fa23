#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "make_message.c"

void laplace_iterate(float* buf, int n, float left, float right) {
    float tmp2, tmp = left;
    for (int i = 0; i < n-1; ++i) {
        tmp2 = (tmp + buf[i+1]) / 2;
        tmp = buf[i];
        buf[i] = tmp2;
    }
    buf[n-1] = (tmp + right)/2;
}

int main ( int argc , char ** argv ) {
    int k , p , me , left , right , count = 1, tag = 1, nit = 3000000, n_local, n_total=30000;
    float ul, ur, u, u0 = 1.0 , alpha = 1.0 , beta = 2.0;
    MPI_Status status;

    MPI_Init (&argc , &argv);
    double t0 = MPI_Wtime();
    MPI_Comm_size (MPI_COMM_WORLD , &p);
    MPI_Comm_rank (MPI_COMM_WORLD , & me);
    left = me-1;
    right = me+1;

    n_local = n_total / p;
    if (me == 0) {
        ul = alpha;
    }
    if (me == p-1) {
        ur = beta;
        n_local += n_total % p;
    }
    float* buf = malloc(sizeof(float)*n_local);
    for (int i = 0; i < n_local; ++i) {
        buf[i] = u0;
    }
    printf("%d %d\n", me, n_local);

    for (k = 1; k <= nit ; k ++) {
        
        if (me > 0) {
            // Send my value to left neighbor.
            MPI_Request req;
            MPI_Isend(&buf[0], count, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // so safe. we care so much about this request
        }
        if (me < p-1) {
            // Send my value to right neighbor.
            MPI_Request req;
            MPI_Isend(&buf[n_local-1], count, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req); // so safe. we care so much about this request
        }
        if (me > 0) {
            // Pick up value from left neighbor.
            MPI_Recv(&ul, count, MPI_FLOAT, left, tag, MPI_COMM_WORLD, &status);
        }
        if (me < p-1) {
            // Pick up value from right neighbor.
            MPI_Recv(&ur, count, MPI_FLOAT, right, tag, MPI_COMM_WORLD, &status);
        }
        laplace_iterate(buf, n_local, ul, ur);
    }
    double t1 = MPI_Wtime();

    printf("%d completed. writing output...\n", me);

    char* outfile_name = make_message("%d.out", me);
    FILE* outfile = fopen(outfile_name, "w");
    free(outfile_name);
    fprintf(outfile, "%f\n", t1-t0);
    for (int i = 0; i < n_local; ++i) {
        fprintf(outfile, "%f\n", buf[i]);
    }

    fclose(outfile);
    free(buf);
    MPI_Finalize ();
}
