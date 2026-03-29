import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here    
    T = np.array(transitions)
    V = np.array(values)
    R = np.array(rewards)
    
    S = T @ V    
    Q = R + gamma * S    
    output = np.max(Q, axis=1)
    return output.tolist()
    