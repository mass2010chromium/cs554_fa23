import numpy as np

n = 5
m = 300
p = 15

#v = np.array(range(n+1)) + 1
v = np.random.random(n+1)

assert len(v) == n+1

A = np.zeros((m, m))
for r in range(n):
    A[r, :r+1] = v[-(r+1):]
for r in range(n, m):
    A[r, r-n:r+1] = v

#print(A)

def make_P(m, n, p):
    assert m//p == m/p

    P = np.zeros((m, m))
    row = 0
    col = 0

    # Build first section.
    for i in range(p):
        # build a mini identity matrix for V_i \ W_i.
        for j in range(m//p - n):
            P[row, col] = 1
            row += 1
            col += 1

        # Offset the target column by |W_i|.
        col += n

    # Build the second section.
    col = 0
    for i in range(p):
        # Offset the target column by |V_i \ W_i|.
        col += (m//p - n)

        # build a mini identity matrix for W_i.
        for j in range(n):
            P[row, col] = 1
            row += 1
            col += 1

    return P

P = make_P(m, n, p)

import matplotlib.pyplot as plt
from PIL import Image
def save_nonzero_image(mat, fn):
    im = Image.fromarray(np.array(mat == 0, dtype=np.uint8) * 255)
    im.save(fn)

blocked = P @ A @ P.T
save_nonzero_image(blocked, "B.png")

B11 = blocked[:m - n*p, :m - n*p]
B12 = blocked[:m - n*p, m - n*p:]
B21 = blocked[m - n*p:, :m - n*p]
B22 = blocked[m - n*p:, m - n*p:]
save_nonzero_image(B12.T, "B12_transpose.png")
save_nonzero_image(np.linalg.inv(B11), "B11_inv.png")
plt.imshow(np.linalg.inv(B11) != 0)
plt.show()

#schur_complement_part = B21 @ np.linalg.inv(B11) @ B12
schur_complement_part = np.linalg.inv(B11) @ B12
save_nonzero_image(schur_complement_part, "B11_inv_B12.png")
save_nonzero_image(schur_complement_part.T, "B11_inv_B12_T.png")
save_nonzero_image(B21 @ schur_complement_part, "schur_complement_nnz.png")
plt.imshow(schur_complement_part != 0)
plt.show()
