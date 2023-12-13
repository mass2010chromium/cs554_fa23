.PHONY: test
test:
	gcc test.c matrix_math.c cone.c problem.c structures/Vector.c -lm -ggdb -o test

test_parallel:
	gcc test_parallel.c parallel.c cone.c -lpthread -lm -O3
