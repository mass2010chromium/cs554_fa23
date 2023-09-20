nnodes?=2
ntasks?=4

run: a.out
	srun --partition=secondary --nodes=$(nnodes) --ntasks=$(ntasks) --mpi=pmi2 ./a.out

a.out:
	mpicc laplace.c -std=c99

.PHONY: clean
clean:
	rm -f *.out
