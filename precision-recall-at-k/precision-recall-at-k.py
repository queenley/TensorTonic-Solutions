import numpy as np
def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    top_recommended = np.array(recommended[:k])   
    relevant = np.array(relevant)
    count = len(np.intersect1d(top_recommended, relevant))
    return [count / k, count / len(relevant)]