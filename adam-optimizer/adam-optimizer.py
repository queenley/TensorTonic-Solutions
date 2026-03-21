import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    m = np.array(m)
    v = np.array(v)
    grad = np.array(grad)
    
    m_new = beta1 * m + (1 - beta1) * grad
    v_new = beta2 * v + (1 - beta2) * grad ** 2
    # m_new = np.array([beta1 * _m for _m in m]) + np.array([(1 - beta1) * _g for _g in grad])
    # v_new = np.array([beta2 * _v for _v in v]) + np.array([(1 - beta2) * _g ** 2 for _g in grad])

    mt = m_new / (1 - beta1 ** t)
    vt = v_new / (1 - beta2 ** t)

    param_new = param - lr * (mt / (np.sqrt(vt) + eps))

    return param_new, m_new, v_new