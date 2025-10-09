from tokenizers import ByteLevelBPETokenizer
from pathlib import Path
from typing import List


class TokenizerWrapper:
    """
    A convenient wrapper for loading a ByteLevelBPE tokenizer
    and performing text encoding/decoding.

    Parameters
    ----------
    vocab_path : str
        Path to the vocab.json file.
    merges_path : str
        Path to the merges.txt file.
    """
    def __init__(self, vocab_path: str, merges_path: str):
        self.tokenizer = ByteLevelBPETokenizer(vocab_path, merges_path)

    def encode(self, text: str) -> List[int]:
        """Encode a text string into a list of token IDs."""
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token IDs back into text."""
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        """Return the ID of a special token such as <pad>, <s>, or </s>."""
        return self.tokenizer.token_to_id(token)
