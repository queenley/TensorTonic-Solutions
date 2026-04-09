import numpy as np

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True)    
    return 1 - sum((counts / len(y)) ** 2)
    
    
def decision_tree_split(X, y):
    """
    Find the best feature and threshold to split the data.
    """
    # Write code here
    y = np.array(y)    
    X = np.array(X)
    n_samples, n_features = X.shape

    gini_s = gini_impurity(y)

    best_ig = -1
    best_feature = None
    best_threshold = None
    
    for feat_idx in range(n_features):
        unique_vals = np.unique(X[:, feat_idx])        
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2

        for thres in thresholds:
            left_mask = X[:, feat_idx] <= thres
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)

            gini_split = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
            ig = gini_s - gini_split

            condition1 = (ig > best_ig)
            condition2 = ((ig == best_ig) and (feat_idx < best_feature))
            condition3 = ((ig == best_ig) and (feat_idx == best_feature) and (thres < best_threshold))
            if (condition1 or condition2 or condition3):
                best_ig = ig
                best_threshold = thres
                best_feature = feat_idx
            
    return (best_feature, best_threshold)     
        