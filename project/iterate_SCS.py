import numpy as np
import numpy.linalg as la

#x1, x2
#min 4x1^2 + 12x1x2 + 10x2^2 + 4x1 + 8x2
#
# x^X A x + c^T x
#
# A = [[4, 6], [6, 10]]
# c = [4, 8]

#A = np.array([
#    [4, 6, 0, -1, 0],
#    [0, 2, 0, 0, -1],
#    [0, 0, 1, 0, 0]
#])

A = np.array([
    [0, 0, -1],
    [-4, -6, 0],
    [0, -2, 0],
    [0, 0, -1]
])

b = np.array([0, 0, 0, -2])
c = np.array([4, 8, 1])
h = np.hstack([c, b])

M = np.eye(7)
M[:3, 3:] = A.T
M[3:, :3] = -A
print(h)

print(c - A.T @ b)
M_inv = la.inv(M)

alpha = 1 / (1 + np.dot(M_inv @ h, h))
print("alpha: ", alpha)
K = alpha * M_inv @ h * np.dot(M_inv @ h, h)
print("K", K)
# [-0.23140496  0.33057851 -0.33333333  0.33333333 -1.05785124 -0.66115702 -1.66666667]
iteration_matrix = M_inv - (M_inv @ np.outer(h, h) @ M_inv) * alpha

I_Q = np.eye(8)
I_Q[:3, 3:7] = A.T
I_Q[3:7, :3] = -A
I_Q[:-1, -1] = h
I_Q[-1, :-1] = -h
I_Q_inv = la.inv(I_Q)

u = np.array([0, 0, 0, 0, 0, 0, 0, 1])
v = np.array([0, 0, 0, 0, 0, 0, 0, 1])
u_tilde = np.empty(8)

M_inv_h = M_inv @ h
for i in range(10):

    # Affine projection.
    w = u + v
    print("rhs", w)
    M_inv_w = M_inv @ w[:-1]
    v1 = M_inv_w - w[-1]*M_inv_h - alpha*(M_inv_h * np.dot(h, M_inv_w)) + w[-1]*K 
    print(M_inv_w)
    print(np.dot(h, M_inv_w))
    #v2 = iteration_matrix @ (w[:-1] - w[-1]*h)

    u_tilde[:-1] = v1
    u_tilde[-1] = w[-1] + np.dot(h, u_tilde[:-1])
    #u_tilde = I_Q_inv @ w
    
    # Cone projection.
    u = u_tilde - v
    x_len = la.norm(u[4:7])
    z = u[3]
    if x_len <= -z:
        u[3:7] = 0
    elif x_len > z:
        a = (z + x_len)/2
        u[3] = a
        u[4:7] *= a / x_len
    if u[-1] < 0:
        u[-1] = 0

    # v update.
    v = v - u_tilde + u

    print(f"iteration {i}")
    #print(u)
    #print(u_tilde)
    print("\t", u[:2] / u[-1])

