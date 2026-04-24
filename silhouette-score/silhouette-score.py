import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = X.shape[0]
    
    # Step 1: Pairwise Euclidean distance matrix (n, n)
    # Use identity: ||x-y||² = ||x||² + ||y||² - 2·x·y
    sq_norms = np.sum(X**2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X @ X.T)
    dist = np.sqrt(np.maximum(dist_sq, 0))  # clip negatives from float errors
    
    # Step 2: Compute sum of distances from each point to each cluster
    unique_clusters, own_idx = np.unique(labels, return_inverse=True)
    n_clusters = len(unique_clusters)
    
    # One-hot encode labels: shape (n, n_clusters)
    # membership[i, k] = 1 if point i belongs to cluster k, 0 otherwise
    membership = np.zeros((n, n_clusters))
    membership[np.arange(n), own_idx] = 1
    
    # Sum of distances: dist (n, n) @ membership (n, n_clusters)
    # → cluster_dist_sum (n, n_clusters)
    # cluster_dist_sum[i, k] = total distance from i to all points in cluster k
    cluster_dist_sum = dist @ membership
    
    # Size of each cluster
    cluster_sizes = membership.sum(axis=0)  # (n_clusters,)
    
    # Step 3a: Compute a(i) — average distance within same cluster (exclude self)
    own_sum = cluster_dist_sum[np.arange(n), own_idx]  # (n,)
    own_size = cluster_sizes[own_idx]                   # (n,)
    
    # For clusters of size 1, set a = 0 to avoid division by zero
    a = np.where(own_size > 1, own_sum / np.maximum(own_size - 1, 1), 0)
    
    # Step 3b: Compute b(i) — min average distance to other clusters
    # avg_to_clusters[i, k] = average distance from i to cluster k
    avg_to_clusters = cluster_dist_sum / cluster_sizes  # broadcasting
    
    # Mask own cluster of each point to exclude it from min
    mask_own = np.zeros((n, n_clusters), dtype=bool)
    mask_own[np.arange(n), own_idx] = True
    avg_to_clusters[mask_own] = np.inf
    
    b = avg_to_clusters.min(axis=1)  # (n,)
    
    # Step 4: Silhouette score
    # s(i) = 0 when the cluster contains only one point (by convention)
    denominator = np.maximum(a, b)
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.where(
            (own_size > 1) & (denominator > 0),
            (b - a) / denominator,
            0.0
        )
    
    return float(s.mean())