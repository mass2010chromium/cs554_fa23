import numpy as np
import scipy as sp
import scipy.linalg

class LassoSolver:
    
    """
    Solve the problem:

    min_b 1/2|y - Xb|_2^2 + lamda|b|_1
    """
    def __init__(self, X, y, lam, rho):
        self.y = y
        self.X = X
        self.lam = lam
        self.rho = rho

        m = X.T @ X
        m += rho * np.eye(m.shape[0])
        self.L = np.linalg.cholesky(m)

        self.b1 = np.zeros(X.shape[1])
        self.b2 = np.zeros_like(self.b1)
        self.w = np.zeros_like(self.b1)

    def iterate(self):
        rhs = (self.X.T @ self.y) + self.rho * (self.b2 - self.w)
        self.b1 = sp.linalg.cho_solve((self.L, True), rhs)

        threshold_val = self.lam / self.rho
        pre_threshold = self.b1 + self.w
        center_mask = (pre_threshold >= -threshold_val) * (pre_threshold <= threshold_val)
        pre_threshold[center_mask] = 0
        pre_threshold[pre_threshold > threshold_val] -= threshold_val
        pre_threshold[pre_threshold < -threshold_val] += threshold_val
        self.b2 = pre_threshold

        self.w += self.b1 - self.b2

    def loss(self):
        v = self.y - self.X @ self.b1
        return 0.5*np.dot(v, v) + self.lam * np.linalg.norm(v, 1)

    def violation(self):
        return np.linalg.norm(self.b1 - self.b2, 1)
 
if __name__ == "__main__":
    import time
    np.random.seed(0)

    n1 = 1200
    n2 = 1200
    y = np.random.random(n1)
    X = np.random.random((n1, n2))
    rho = 1
    lam = 0.01

    eps = 1e-5

    solver = LassoSolver(X, y, lam, rho)
    t1 = time.time()
    iters = 0
    while True:
        solver.iterate()
        iters += 1
        if solver.violation() < eps:
            break
    t2 = time.time()
    print(f"time: {t2 - t1}, nit: {iters}, loss: {solver.loss()}, violation: {solver.violation()}")
    input()

