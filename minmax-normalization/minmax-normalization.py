import numpy as np

def minmax_scale(X, axis=0, eps=1e-12):
    """
    Scale X to [0,1]. If 2D and axis=0 (default), scale per column.
    Return np.ndarray (float).
    """
    # Write code here
    X = np.array(X)
    dim = X.ndim
    if dim == 2:
        return (X - np.min(X, axis=axis, keepdims=True)) / np.maximum(eps, np.max(X, axis=axis, keepdims=True) - np.min(X, axis=axis, keepdims=True))
    else:      
        return (X - np.min(X, axis=axis)) / np.maximum(eps, np.max(X, axis=axis) - np.min(X, axis=axis))        
        