import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.array(A)
    i, j = A.shape
    X = np.zeros(shape=(j, i))
    for idx in range(j):
        X[idx] = A[:, idx]
    return X
