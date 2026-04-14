import numpy as np
def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    TP = (y_pred == y_true).sum()
    FP = len(y_true) - TP
    FN = len(y_pred) - TP

    return (2 * TP) / (2 * TP + FP + FN)