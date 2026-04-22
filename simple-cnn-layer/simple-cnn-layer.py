import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here
    x = np.asarray(x, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)    
    
    N, C_in, H, _ = x.shape
    C_out, _, KH, KW = W.shape

    # Create sliding windows on: (H, W)
    # windows shape: (N, C_in, H_out, W_out, KH, KW)
    windows = np.lib.stride_tricks.sliding_window_view(x, (KH, KW), axis=(-2, -1))
    
    # Einsum naming:
    # n = batch index
    # c = input channel
    # i = H_out index
    # j = W_out index
    # u = kernel row (KH)
    # v = kernel col (KW)
    # o = output channel
    #
    # windows: 'ncijuv' (N, C_in, H_out, W_out, KH, KW)
    # W:       'ocuv'   (C_out, C_in, KH, KW)
    # Result:  'noij'   (N, C_out, H_out, W_out)
    # Sum over: c, u, v (xuất hiện ở cả 2 input nhưng không trong output)
    
    y = np.einsum('ncijuv,ocuv->noij', windows, W)
    
    # Add bias with broadcasting
    y += b[np.newaxis, :, np.newaxis, np.newaxis]
    
    return y