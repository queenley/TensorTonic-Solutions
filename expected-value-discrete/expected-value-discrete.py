import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    is_valid_prob = np.allclose(1, sum(p))
    if is_valid_prob:
        x = np.array(x)
        p = np.array(p)
        return np.dot(x, p)
    else:
        raise ValueError("Probabilities should sum to 1")
        
