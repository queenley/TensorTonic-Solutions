import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    pe = np.zeros(shape=(seq_len, d_model))
        
    n_even = (d_model + 1) // 2
    n_odd = d_model // 2
    i_even = np.arange(n_even)[np.newaxis, :]
    i_odd = np.arange(n_odd)[np.newaxis, :]

    pos = np.arange(seq_len)[:, np.newaxis]

    pe[:, 0::2] = np.sin(pos / (base ** (2 * i_even / d_model)))
    pe[:, 1::2] = np.cos(pos / (base ** (2 * i_odd / d_model)))

    return pe