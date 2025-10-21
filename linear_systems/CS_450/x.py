import numpy as np
import sympy as sp
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

# import numpy as np
# import numpy.linalg as la
# # We want to make a QR decomposition of a given matrix using reflections.
# def householder_qr(A):
#     m, n = A.shape
#     R = np.array(A, dtype = float)
#     Q = np.eye(m)
#     for i in range(n):
#         a = R[i:, i]
#         # We only take the rows of the column that we want to zero out over here for this question
#         e = np.zeros_like(a)
#         e[0] = 1
#         sign = 1.0 if a[0] >= 0 else -1.0
#         v = a + sign * la.norm(a) * e
#         # v = a - la.norm(a) * e
#         H = np.eye(m - i) - 2 * np.outer(v, v) / (v @ v)
#         R[i:, i:] = H @ R[i:, i:]
#         # We need to multiply this by the other over here 
#         H_full = np.eye(m)
#         H_full[i:, i:] = H
#         Q = Q @ H_full.T
#     return Q ,R



























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

# A = np.array([[1, 15, -6, 0], [3, 7, 3, 12], [4, -7, -3, 0], [0, -28, 15, 3]])
# A = A + A.T
# def householder_vec(A,i):
#     # ADD CODE HERE
#     # Remember to choose the sign of the scaled unit vector the
#     # same as the corresponding matrix entry, to avoid cancellation.
#     # We need to find the householder vec over here
#     col = A[i + 1:, i]
#     e = np.zeros(len(col))
#     e[0] = 1
#     v= col + np.sign(A[i + 1][i]) * e * la.norm(col)
#     v_final = np.zeros(len(A))
#     v_final[len(v_final) - len(v):] = v
#     return v_final

# def householder_mat(v):
#     H = np.eye(v.shape[0]) - 2*np.outer(v,v)/np.inner(v,v)
#     return H



# v0 = householder_vec(A,0)
# H = A.copy()
# for i in range(len(A) - 2):
#     v = householder_vec(H,i)
#     Q = householder_mat(v)
#     # print(Q)
#     H = Q.T @ H @ Q


# print(H)
# We need to remove the diagnol entries over here

# Once we have H, we will need to use Given 

# # We need to find QR of a matrix here which is R
# def given_rotation(A):
#     R = np.array(A, dtype = float)
#     Q = np.eye(len(A))
#     for col in range(len(A[0]) - 1):
#         row = col + 1
#         # We need to make the matrix over here for the current operation that we want here
#         val = (R[row][col] ** 2 + R[col][col] ** 2) ** 0.5
#         sin_ = R[row][col] / val
#         cos_ = R[col][col] / val
#         G = np.eye(len(A))
#         G[row][col] = - sin_
#         G[row][row] = cos_
#         G[col][col] = cos_
#         G[col][row] = sin_
#         Q = Q @ G.T
#         R = G @ R
#     return Q, R
# Q, R = given_rotation(H)
# print(R)

# A2 = H.copy()
# qriter = 5000
# for k in range(qriter):
#     Q, R = given_rotation(A2)
#     A2 = R @ Q

# print(A2)

# print(np.linalg.eig(A)[0])

# We will code out golden search method for this question over here 
# def golden_search(f, a, b, tol=1e-10):
#     gr = (np.sqrt(5) - 1) / 2  # golden ratio constant
#     # We need only one function computation in each step over here 
#     x1 = a + (1 - gr) * (b - a)
#     x2 = a + gr * (b - a)
#     f1, f2 = f(x1), f(x2)

#     while np.abs(b - a) > tol:
#         if f1 > f2:
#             a = x1
#             x1 = x2
#             f1 = f2
#             x2 = a + gr * (b - a)
#             f2 = f(x2)
#         else:
#             b = x2
#             x2 = x1
#             f2 = f1
#             x1 = b - gr * (b - a)
#             f1 = f(x1)

#     return (a + b) / 2  # best estimate of minimum

# Newton method we have properly already implemented here 


# def f_double_prime(x):
#     return 46
# def f_prime(x):
#     return 46 * x - 4
# x0 = 1

# x_prev = x0
# count = 0
# while (True):
    
#     x_prev = x0
#     x0 = x0 - (f_prime(x0)) / (f_double_prime(x0))
#     if (np.abs(x0 - x_prev) < 10 ** (-10)):
#         break
#     count += 1
# print(count)

# Now let us move on to look at the method of the steepest descent over here


# Steepest descent, already coded it up in the HW


# def golden_section_search(fun, a, b):
#     ratio = (np.sqrt(5)-1) / 2
#     x1 = a + (1- ratio) * (b - a)
#     x2 = a + ratio * (b - a)
#     while (np.abs(a - b) >= 10 ** (-5)):
#         if (fun(x1) > fun(x2)):
#             a = x1
#             x1 = a + (1- ratio) * (b - a)
#             x2 = a + ratio * (b - a)
#         else:
#             b = x2
#             x1 = a + (1- ratio) * (b - a)
#             x2 = a + ratio * (b - a)
#     return a

# x0 = [0, 0]
# # start with an initial guess of `x0`

# def f(x):
#     return x[0] ** 2 - x[0] + x[1] + np.exp(5)
# def gradf(x):
#     return np.array([2 * x[0] - 1, 1])
# xk = x0
# for i in range(1):
#     grad_f_curr  = -gradf(xk)
#     def func_to_minimize(alpha):
#         return f(xk + alpha * grad_f_curr)

#     # find the best line search parameter `alpha` by minimizing `func_to_minimize`.
#     alpha = golden_section_search(func_to_minimize, -1, 1)
#     xk = xk + alpha * grad_f_curr
# print(xk)

# # We need to use (x + y) ** 3
# x = sp.var('x')
# y = sp.var('y')
# f = x ** 0.5 - x * y + y ** 2
# H = sp.hessian(f, (x, y))
# eigs = np.linalg.eig(np.array([[4, -1], [-1, 0]]))
# print(eigs[0])


# def bfgs(x0, errors=None, xhistory=None):
#     x = x0.copy()
#     B = np.eye(2)
#     C = np.eye(2)
    
#     for k in range(100):

#         s = -C @ df(x)

#         def f1d(alpha):
#             return f(x + alpha*s)
#         alpha = sopt.golden(f1d)
#         alpha = 0.01
#         xnew = x + alpha * s
        
#         y = df(xnew) - df(x)
        
#         Bnew = B + (1/np.dot(y, s))*np.outer(y, y) - (1/np.dot(B@s, s))*np.outer(B@s, B@s)
        
#         u = s - C @ y
#         Cnew = C + (1/np.dot(s,y))*np.outer(u, s) + (1/np.dot(s,y))*np.outer(s, u) - (np.dot(y,u)/np.dot(s,y)**2)*np.outer(s,s)
        
#         B = Bnew
#         x = xnew
#         C = Cnew
#     return x
# # This is superlinear convergence over here for this here 

def jacobian(x):
    return np.array([1, np.exp(x)]).T
def residual(x):
    