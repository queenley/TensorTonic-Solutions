import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    # Write code here
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_norms = np.linalg.norm(a)
    b_norms = np.linalg.norm(b)
    denominator = np.dot(a_norms, b_norms)
    if denominator:
        return np.dot(a, b) / denominator
    return 0.0