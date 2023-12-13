import numpy as np
import socplib.socp

L_transpose = np.array([[2, 3], [0, 1]])

LT_coo = []
for r in range(L_transpose.shape[0]):
    for c in range(L_transpose.shape[1]):
        if L_transpose[r, c]:
            LT_coo.append([r, c, L_transpose[r, c]])

c = [4, 8]
print(len(L_transpose))

socplib.socp.init_solver()
socplib.socp.set_lin_term(c)
socplib.socp.add_quad_term(LT_coo, len(L_transpose))
socplib.socp.finalize_solver()
print(socplib.socp.socp_solve())
socplib.socp.dealloc_solver()
