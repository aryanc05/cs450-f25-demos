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

# A = np.array([[1, 1, 2], [1, -1, 3], [1, 1, 4], [1, -1, 1]])
# Q, R = householder_qr(A)
# print(R)
# print(Q @ R - A)


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


# Let us make orothogonal iteration 

# A = np.random.rand(4, 4)
# x = A.copy()
# val = np.eye(len(A))
# for _ in range(5000):
#     Q, R = np.linalg.qr(x)
#     x =  A @ Q
#     val = val @ Q
# print(np.linalg.eig(A)[0])

# How do we get the eigenvectors over here 

# A = np.random.rand(4, 4)
# A = (A + A.T)/2
# x = A.copy()
# for _ in range(500):
#     Q, R = np.linalg.qr(x)
#     x = A @ Q

# print(Q.T @ A @ Q) # schur
# print('----------')
# print(np.linalg.eig(A)[0]) #eigenvalues
# print('----------')
# print(Q) #eigenvectors
# print('----------')
# print(np.linalg.eig(A)[1]) #eigenvectors
# A = np.random.rand(4, 4)
# x = A.copy()
# val = np.eye(len(A))
# for _ in range(5000):
#     Q, R = np.linalg.qr(x - np.eye(len(A)) * x[len(x) - 1][len(x) - 1])
#     x = R @ Q + np.eye(len(A)) * x[len(A) - 1][len(A) - 1]# unshifted QR iteration
#     val = val @ Q
# print(val)
# eigvecs = np.linalg.eig(A)[1]
# print(np.linalg.eig(np.array([[2.37423790e-001,  2.53776679e-001], [-4.95652999e-001,  5.26626072e-001]]))[0])

A = np.array([[1, 15, -6, 0], [3, 7, 3, 12], [4, -7, -3, 0], [0, -28, 15, 3]])
A = A + A.T
def householder_vec(A,i):
    # ADD CODE HERE
    # Remember to choose the sign of the scaled unit vector the
    # same as the corresponding matrix entry, to avoid cancellation.
    # We need to find the householder vec over here
    col = A[i + 1:, i]
    e = np.zeros(len(col))
    e[0] = 1
    v= col + np.sign(A[i + 1][i]) * e * la.norm(col)
    v_final = np.zeros(len(A))
    v_final[len(v_final) - len(v):] = v
    return v_final

def householder_mat(v):
    H = np.eye(v.shape[0]) - 2*np.outer(v,v)/np.inner(v,v)
    return H



v0 = householder_vec(A,0)
H = A.copy()
for i in range(len(A) - 2):
    v = householder_vec(H,i)
    Q = householder_mat(v)
    # print(Q)
    H = Q.T @ H @ Q


# print(H)
# We need to remove the diagnol entries over here

# Once we have H, we will need to use Given 

# We need to find QR of a matrix here which is R
def given_rotation(A):
    R = np.array(A, dtype = float)
    Q = np.eye(len(A))
    for col in range(len(A[0]) - 1):
        row = col + 1
        # We need to make the matrix over here for the current operation that we want here
        val = (R[row][col] ** 2 + R[col][col] ** 2) ** 0.5
        sin_ = R[row][col] / val
        cos_ = R[col][col] / val
        G = np.eye(len(A))
        G[row][col] = - sin_
        G[row][row] = cos_
        G[col][col] = cos_
        G[col][row] = sin_
        Q = Q @ G.T
        R = G @ R
    return Q, R
Q, R = given_rotation(H)
print(R)

A2 = H.copy()
qriter = 5000
for k in range(qriter):
    Q, R = given_rotation(A2)
    A2 = R @ Q

print(A2)

print(np.linalg.eig(A)[0])