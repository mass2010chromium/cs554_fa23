nnodes?=2
ntasks?=4

run: laplace
	srun --partition=secondary --nodes=$(nnodes) --ntasks=$(ntasks) --mpi=pmi2 ./laplace

laplace: laplace.c
	mpicc laplace.c -std=c99 -o laplace

.PHONY: clean
clean:
	rm -f *.out
	rm -f laplace
