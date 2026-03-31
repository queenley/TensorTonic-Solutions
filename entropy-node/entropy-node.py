import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here    
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(np.array(y), return_counts=True)
    probs = counts / len(y)
    h = -probs * np.log2(probs)
    return np.sum(h)