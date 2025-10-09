import numpy as np
from typing import List


def pad_sequence(seq: List[int], max_len: int, pad_id: int, bos_id: int, eos_id: int) -> List[int]:
    """
    Add <s> (BOS) and </s> (EOS) tokens, and pad or truncate the sequence
    to make its length equal to max_len.

    - If the sequence is longer than max_len → truncate it.
    - If shorter → pad with the pad token at the end.

    Returns
    -------
    List[int]
        A list of token IDs with a fixed length of `max_len`.
    """
    if len(seq) >= max_len:
        seq_pad = [bos_id] + seq[:max_len - 2] + [eos_id]
    else:
        seq_pad = [bos_id] + seq + [pad_id] * (max_len - len(seq) - 2) + [eos_id]
    
    return seq_pad


def shift_right(batch_ids: np.ndarray, pad_id: int) -> np.ndarray:
    """
    Shift the input sequence one position to the right to create labels
    for next-token prediction.

    Parameters
    ----------
    batch_ids : np.ndarray
        Input batch of token IDs with shape (batch_size, seq_len).
    pad_id : int
        ID used for padding tokens.

    Returns
    -------
    np.ndarray
        A shifted version of `batch_ids` where each sequence is moved one step
        to the right and the first position is filled with the pad token.
    """
    shifted = np.full_like(batch_ids, pad_id)
    shifted[:, 1:] = batch_ids[:, :-1]
    
    return shifted
