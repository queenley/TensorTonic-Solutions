import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here    
    p = np.array(points)
    T = np.array(T)
    single = p.ndim == 1
    p = np.atleast_2d(p)
    n = p.shape[0]
    homogeneous = np.hstack([p, np.ones((n, 1))])
    transform = T @ homogeneous.T
    output = transform.T[:, :3]
    if single:
        return output[0].tolist()
    return output.tolist()
    