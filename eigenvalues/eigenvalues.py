import numpy as np
from typing import Union

def calculate_eigenvalues(matrix):
    """
    Calculate eigenvalues of a square matrix.
    """
    # Write code here
    if not all(isinstance(row, list) for row in matrix):
        return None
    
    row_lengths = [len(row) for row in matrix]
    if len(set(row_lengths)) > 1:
        return None
    
    matrix = np.array(matrix, dtype=float)   

    if (matrix.ndim != 2) or (matrix.shape[0] != matrix.shape[1]):
        return None        
    else:        
        eigvals = np.linalg.eigvals(matrix)        
        lexsort = np.lexsort((eigvals.imag, eigvals.real))
        return eigvals[lexsort]
        