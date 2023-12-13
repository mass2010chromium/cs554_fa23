import numpy as np
import numpy.linalg as la

np.random.seed(0)
A = np.random.rand(1000, 1000)

A_coo = []
for r in range(A.shape[0]):
    for c in range(A.shape[1]):
        if np.random.rand() < 10 / A.shape[0]:
            A_coo.append([r, c, A[r, c]])
        else:
            A[r, c] = 0
        #print(f"{{{r}, {c}, {A[r, c]}}},")
print(len(A_coo))
print(la.matrix_rank(A))


b = np.random.rand(A.shape[0])
#print(b)
import time

t1 = time.time()
x1 = np.linalg.solve(A, b)
t2 = time.time()


def cg(A, b, eps=1e-10, maxiter=None):
    if maxiter is None:
        maxiter = len(b)*10
    x = np.zeros_like(b)
    r = b - A @ x
    if la.norm(r) < eps:
        return x

    p = r
    r_prod = np.dot(r, r)
    #print(r)
    for i in range(maxiter):
        #print(i, r_prod)
        alpha = r_prod / np.dot(p, A @ p)
        x = x + alpha * p
        r = r - alpha*(A @ p)
        if la.norm(r) < eps:
            print(f"cg converged in {i} iterations")
            return x
        r_prod2 = np.dot(r, r)
        beta = r_prod2 / r_prod
        r_prod = r_prod2
        
        p = r + beta * p
    print("cg didn't converge...")
    return x

t3 = time.time()
x2 = cg(A.T @ A, A.T @ b)
t4 = time.time()

import socplib.socp

t5 = time.time()
x3 = socplib.socp.linsolve(A_coo, b)
t6 = time.time()

#print(x1 - x2)
print(f"numpy solve: {t2 - t1}; conjugate gradient solve: {t4 - t3}; residual={la.norm(x1-x2)}");
print(f"C conjugate gradient solve: {t6 - t5}; residual={la.norm(x1-x3)}");

t7 = time.time()
x4 = socplib.socp.linsolve(A_coo, b, x1)
t8 = time.time()
print(f"C conjugate gradient re-solve: {t8 - t7}; residual={la.norm(x1-x4)}");
