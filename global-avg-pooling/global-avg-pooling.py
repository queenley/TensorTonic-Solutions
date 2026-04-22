import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    # Write code here    
    x = np.array(x)
    if x.ndim not in (3, 4):
        raise ValueError(f"Input should be 3D (C,H,W) or 4D (N,C,H,W), "
            f"received {x.ndim}D")
        
    gap = x.mean(axis=(-2, -1))
    return gap
    
        
    