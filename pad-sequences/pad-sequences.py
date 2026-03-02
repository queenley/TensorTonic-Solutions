import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.empty(shape=(0, 0))
    
    if not max_len:
        max_len = max(len(seq) for seq in seqs)

    padded_result = np.full((len(seqs), max_len), pad_value)
    for id, seq in enumerate(seqs):        
        padded_result[id, :min(len(seq), max_len)] = seq[:max_len]

    return padded_result
    

    