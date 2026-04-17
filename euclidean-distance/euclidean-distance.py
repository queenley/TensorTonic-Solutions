import numpy as np

def euclidean_distance(x, y):
    """
    Compute the Euclidean (L2) distance between vectors x and y.
    Must return a float.
    """
    # Write code here
    x = np.array(x)
    y = np.array(y)

    euclidean_distance = np.sqrt(np.sum((x - y) ** 2)).item()
    return euclidean_distance
    