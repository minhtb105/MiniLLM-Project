import json
import numpy as np
from pathlib import Path
from typing import List
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.utils import pad_sequence, shift_right
from src.data.tokenizer_utils import TokenizerWrapper


class TextDataset:
    """
    Prepare text data from multiple sources (e.g., Wikipedia, news, GitHub...).
    Automatically encodes, pads, and batches text sequences for training.

    Parameters
    ----------
    files : list[str]
        List of paths to .jsonl files containing text data.
    tokenizer : TokenizerWrapper
        A preloaded tokenizer instance.
    max_len : int, optional (default=512)
        Maximum sequence length.
    batch_size : int, optional (default=32)
        Number of samples per batch.
    """
    def __init__(self, files: List[str], tokenizer: TokenizerWrapper,
                 max_len: int = 512, batch_size: int = 32):
        self.files = files
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.batch_size = batch_size

        self.pad_id = tokenizer.token_to_id("<pad>")
        self.bos_id = tokenizer.token_to_id("<s>")
        self.eos_id = tokenizer.token_to_id("</s>")

        self.padded_corpus = self._build_dataset()

    def _build_dataset(self) -> List[List[int]]:
        """Read text data from files, encode each sequence, and apply padding."""
        encoded_corpus = []

        for file in self.files:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    text = obj.get("text", "")
                    token_ids = self.tokenizer.encode(text)
                    padded = pad_sequence(token_ids, self.max_len, self.pad_id, self.bos_id, self.eos_id)
                    encoded_corpus.append(padded)

        return encoded_corpus

    def get_batches(self) -> List[np.ndarray]:
        """Split the encoded corpus into fixed-size batches."""
        n = len(self.padded_corpus)
        return [np.array(self.padded_corpus[i:i+self.batch_size])
                for i in range(0, n, self.batch_size)]


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    print(BASE_DIR)
    tokenizer = TokenizerWrapper(
        str(BASE_DIR / "data/processed/tokenizer/vocab.json"),
        str(BASE_DIR / "data/processed/tokenizer/merges.txt")
    )

    files = [
        str(BASE_DIR / "data/raw/wiki/wiki.jsonl"),
        str(BASE_DIR / "data/raw/news/news.jsonl"),
        str(BASE_DIR / "data/raw/github_repos/corpus.jsonl")
    ]

    dataset = TextDataset(files, tokenizer, max_len=512, batch_size=32)
    batches = dataset.get_batches()

    batch_ids = batches[0]
    label_ids = shift_right(batch_ids, dataset.pad_id)

    print("Batch shape:", batch_ids.shape)      # (32, 512)
    print("Shifted shape:", label_ids.shape)    # (32, 512)
