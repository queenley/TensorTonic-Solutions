import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_selected = y_pred[np.arange(len(y_true)), y_true]    
    losses = -np.log(y_selected)    
    cross_entropy_loss = np.mean(losses)

    return cross_entropy_loss