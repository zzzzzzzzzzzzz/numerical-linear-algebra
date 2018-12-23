# encoding: utf-8
# pset2.py

import numpy as np
from scipy import sparse
# don't forget import packages, e.g. scipy
# but make sure you didn't put unnecessary stuff in here

# INPUT : diag_broadcast - list of diagonals value to broadcast,length equal to 3 or 5; n - integer, band matrix shape.
# OUTPUT : L - 2D np.ndarray, L.shape[0] depends on bandwidth, L.shape[1] = n-1, do not store main diagonal, where all ones;                  add zeros to the right side of rows to handle with changing length of diagonals.
#          U - 2D np.ndarray, U.shape[0] = n, U.shape[1] depends on bandwidth;
#              add zeros to the bottom of columns to handle with changing length of diagonals.
def band_lu(diag_broadcast, n): # 5 pts
    assert (len(diag_broadcast) == 3) or (len(diag_broadcast) == 5), "Only shape 3 or 5 diags"
    L_row = np.zeros(n-1)
    U_col = np.zeros(n)
    if len(diag_broadcast) == 3:
        a = diag_broadcast[0]
        b = diag_broadcast[1]
        c = diag_broadcast[2]
        
        L = np.full((1, n-1), float(a))
        U = np.column_stack((U_col, U_col))
        
        for i in range(n-1):
            U[i][1] = c
        
        for i in range(n):
            U[i][0] = b
        
        for i in range(1, n):
            L[0][i-1] = L[0][i-1] / U[i-1][0]
            U[i][0] = U[i][0] - L[0][i-1] * U[i-1][1]
        
        return L, U
            
    if len(diag_broadcast) == 5:
        a = np.float(diag_broadcast[0])
        b = np.float(diag_broadcast[1])
        c = np.float(diag_broadcast[2])
        d = np.float(diag_broadcast[3])
        e = np.float(diag_broadcast[4])
        
        L = np.full((2, n-1), 0.0)
        U = np.column_stack((U_col, U_col, U_col))
        
        U[0][0] = c
        U[0][1] = d
        U[0][2] = e
        U[1][0] = c - d * b/c
        U[1][1] = d - e * b/c
        U[1][2] = e
        
        L[0][0] = b/c
        
        for i in range(2, n):
            L[1][i-2] = a/U[i-2][0]
            L[0][i-1] = (b - L[1][i-2]*U[i-2][1])/U[i-1][0]
            U[i][0] = c - L[1][i-2]*U[i-2][2] - L[0][i-1]*U[i-1][1]
            if i < n-1:
                U[i][1] = d - L[0][i-1]*U[i-1][2]
            if i < n-2:
                U[i][2] = e
        
        return L, U


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def gram_schmidt_qr(A): # 5 pts
    assert A.shape[0] >= A.shape[1], 'm is not >= n'
    new_vectors = np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[1]):
        new_vectors[:,i] = A[:,i]
        for j in range(i):
            new_vectors[:,i] -= (np.dot(A[:,i], new_vectors[:,j])/np.dot(new_vectors[:,j],new_vectors[:,j])) * new_vectors[:,j]
    
    # normalization
    for i in range(new_vectors.shape[1]):
        new_vectors[:,i] /= np.linalg.norm(new_vectors[:,i])
    Q = new_vectors
    R = np.zeros((A.shape[1],A.shape[1]))
    for i in range(A.shape[1]):
        for j in range(i, A.shape[1]):
            R[i,j] = np.dot(A[:,j], new_vectors[:,i])
    return Q, R

# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A = QR
def modified_gram_schmidt_qr(A): # 5 pts
    assert A.shape[0] >= A.shape[1], 'm is not >= n'
    new_vectors = np.zeros((A.shape[0],A.shape[1]))
    for i in range(A.shape[1]):
        new_vectors[:,i] = A[:,i]
        if i > 0:
            new_vectors[:,i] -= (np.dot(A[:,i], new_vectors[:,0])/np.dot(new_vectors[:,0],new_vectors[:,0])) * new_vectors[:,0]
        
        for j in range(1,i):
            new_vectors[:,i] -= (np.dot(new_vectors[:,i], new_vectors[:,j])/np.dot(new_vectors[:,j],new_vectors[:,j])) * new_vectors[:,j]
        
    # normalization
    for i in range(new_vectors.shape[1]):
        new_vectors[:,i] /= np.linalg.norm(new_vectors[:,i])
    Q = new_vectors
    R = np.zeros((A.shape[1],A.shape[1]))
    for i in range(A.shape[1]):
        for j in range(i, A.shape[1]):
            R[i,j] = np.dot(A[:,j], new_vectors[:,i])
    return Q, R


# INPUT : rectangular matrix A
# OUTPUT: matrices Q - orthogonal and R - upper triangular such that A=QR
def householder_qr(A): # 7 pts
    E = np.eye(A.shape[0])
    R = A
    Q = E
    for i in range(np.min((A.shape[0], A.shape[1]))-1):
        x = A[i:,:][:,i]
        E = np.eye(x.shape[0])
        e1 = E[:,0]
        u = (x-np.linalg.norm(x)*e1)/np.sqrt(2*(np.linalg.norm(x)**2 - np.linalg.norm(x)*x[0]))
        H = E - 2*np.dot(u.reshape(u.shape[0],1),u.reshape(1,u.shape[0]))
        H_big = np.eye(A.shape[0])
        H_big[i:,i:] = H
        H = H_big
        R[:,i] = np.dot(H, R[:,i])
        Q = np.dot(H, Q)
        
    Q = Q.transpose()
    return Q, R


# INPUT:  G - np.ndarray
# OUTPUT: A - np.ndarray (of size G.shape)
def pagerank_matrix(G, return_dense=True): # 5 pts
    G_sparse = sparse.csr_matrix(G, dtype='float')
    sh = G_sparse.get_shape()
    sums = G_sparse.sum(axis=1)
    for i in range(sh[0]):
        for j in G_sparse.getrow(i).indices:
            G_sparse[i,j] = 1/sums[i,0]
    G_sparse = G_sparse.transpose()
    if return_dense:
        return G_sparse.todense()
    else:
        return G_sparse


# INPUT:  A - np.ndarray (2D), x0 - np.ndarray (1D), num_iter - integer (positive) 
# OUTPUT: x - np.ndarray (of size x0), l - float, res - np.ndarray (of size num_iter + 1 [include initial guess])
def power_method(A, x0, num_iter): # 5 pts
    l = np.random.rand()
    x = x0

    res = np.zeros(num_iter+1)
    res[0] = np.linalg.norm(np.dot(A,x) - l*x)
    for i in range(num_iter):
        new_x = np.dot(A,x)
        
        norm = np.linalg.norm(new_x)
        l = np.dot(new_x, x)
        
        x = new_x / norm
        
        res[i+1] = np.linalg.norm(np.dot(A,x) - l*x)

    return x, l, res


# INPUT:  A - np.ndarray (2D), d - float (from 0.0 to 1.0), x - np.ndarray (1D, size of A.shape[0/1])
# OUTPUT: y - np.ndarray (1D, size of x)
def pagerank_matvec(A, d, x): # 2 pts
    A_sparse = sparse.csr_matrix(d*A)
    sh = A_sparse.get_shape()
    x_sparse = sparse.csr_matrix(x.reshape((x.shape[0], 1)))
    second_part_value = ((1-d)/sh[0])*np.sum(x)
    second_part_vector = np.full(sh[0], second_part_value)
    second_part_vector_sparse = sparse.csr_matrix(second_part_vector.reshape((second_part_vector.shape[0], 1)))
    return A_sparse.dot(x_sparse) + second_part_vector_sparse 

def return_words():
    # insert the (word, cosine_similarity) tuples
    # for the words 'numerical', 'linear', 'algebra' words from the notebook
    # into the corresponding lists below
    # words_and_cossim = [('word1', 'cossim1'), ...]
    
    numerical_words_and_cossim = [
         ('computation', 0.547),
         ('mathematical', 0.532),
         ('calculations', 0.500),
         ('polynomial', 0.485),
         ('calculation', 0.473),
         ('practical', 0.460),
         ('statistical', 0.456),
         ('symbolic', 0.455),
         ('geometric', 0.441),
         ('simplest', 0.438)
    ]
    linear_words_and_cossim = [
         ('differential', 0.759),
         ('equations', 0.724),
         ('equation', 0.682),
         ('continuous', 0.674),
         ('multiplication', 0.674),
         ('integral', 0.672),
         ('algebraic', 0.667),
         ('vector', 0.654),
         ('algebra', 0.630),
         ('inverse', 0.622)
    ]
    algebra_words_and_cossim = [
         ('geometry', 0.795),
         ('calculus', 0.730),
         ('algebraic', 0.716),
         ('differential', 0.687),
         ('equations', 0.665),
         ('equation', 0.648),
         ('theorem', 0.647),
         ('topology', 0.634),
         ('linear', 0.630),
         ('integral', 0.618)
    ]
    
    return numerical_words_and_cossim, linear_words_and_cossim, algebra_words_and_cossim