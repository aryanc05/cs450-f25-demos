# import numpy as np
# def Gram_Schimidit_factorization(A):
#     # We are given a matrix A here, we will find Q and R
#     Q = np.zeros_like(A)
#     for i in range(len(A[0])):
#         q = A[:, i]
#         for j in range(i):
#             q = q - np.inner(Q[:, j],  A[:, i]) * Q[:, j]
#         Q[:, i] = q / np.linalg.norm(q)
#     return Q

# A = np.array([[2, 1 + np.sqrt(2)], [2 * np.sqrt(2),np.sqrt(2)], [2, 1 - np.sqrt(2)]])
# Q = Gram_Schimidit_factorization(A)
# R = Q.T@A
# print(Q @ R - A)

import numpy as np
import numpy.linalg as la
# We want to make a QR decomposition of a given matrix using reflections.
def householder_qr(A):
    m, n = A.shape
    R = np.array(A, dtype = float)
    Q = np.eye(m)
    for i in range(n):
        a = R[i:, i]
        # We only take the rows of the column that we want to zero out over here for this question
        e = np.zeros_like(a)
        e[0] = 1
        sign = 1.0 if a[0] >= 0 else -1.0
        v = a + sign * la.norm(a) * e
        # v = a - la.norm(a) * e
        H = np.eye(m - i) - 2 * np.outer(v, v) / (v @ v)
        R[i:, i:] = H @ R[i:, i:]
        # We need to multiply this by the other over here 
        H_full = np.eye(m)
        H_full[i:, i:] = H
        Q = Q @ H_full.T
    return Q ,R



























    # R = A.copy().astype(float)
    # Q = np.eye(m)
    # # We initially have Q as np.eye(m)
    # for i in range(n):
    #     a = R[i:, i]
    #     e1 = np.zeros(len(a))
    #     e1[0] = 1
    #     sign = np.copysign(1.0, a[0])
    #     v = a + sign * la.norm(a) * e1
    #     # Our identity keeps getting smaller and smaller here 
    #     H = np.eye(m - i) - 2 * np.outer(v, v) / (v @ v)
    #     R[i:, i:] = H @ R[i:, i:]
    #     H_full = np.eye(m)
    #     H_full[i:, i:] = H
    #     Q = Q @ H_full
    # return Q, R

A = np.array([[1, 1, 2], [1, -1, 3], [1, 1, 4], [1, -1, 1]])
Q, R = householder_qr(A)
print(R)
print(Q @ R - A)


# Lets make given rotation over here:
# def given_rotation(A):
#     R = np.array(A, dtype = float)
#     Q = np.eye(len(A))
#     for col in range(len(A[0])):
#         for row in range(len(A) - 1, col, -1):
#             # We need to make the matrix over here for the current operation that we want here
#             val = (R[row][col] ** 2 + R[col][col] ** 2) ** 0.5
#             sin_ = R[row][col] / val
#             cos_ = R[col][col] / val
#             G = np.eye(len(A))
#             G[row][col] = - sin_
#             G[row][row] = cos_
#             G[col][col] = cos_
#             G[col][row] = sin_
#             Q = Q @ G.T
#             R = G @ R
#     return Q, R

# Make sure this is right with the professor over here 
# A = np.array([[1, 1, 2], [1, -1, 3], [1, 1, 4], [2, 1, 3], [5, 5, 5]])
# Q, R = given_rotation(A)
# print(Q @ R - A)
    
# A = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 1]])
# b = np.array([3, 6, 9, 12]).T
# x0 = np.linalg.pinv(A) @ b
# print(x0)

