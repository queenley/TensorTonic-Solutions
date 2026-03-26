import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    rand = rng.random(np.array(x).shape) if rng else np.random.random(np.array(x).shape)
    
    mask = rand >= p    
    scale = 1 / (1 - p)    
    
    pattern = mask * scale    
    output = x * pattern
    
    return (output, pattern)