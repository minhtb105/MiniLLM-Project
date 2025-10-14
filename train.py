import numpy as np
from src.model.gpt import GPTModel
from src.data.dataset_builders import TextDataset
from src.data.tokenizer_utils import TokenizerWrapper
from src.data.utils import shift_right
from src.core.optim import Adam


def train_gpt(
    model: GPTModel,
    dataset: TextDataset,
    epochs: int = 3,
    lr: float = 1e-4,
):
    optimizer = Adam([p for _, p in model.parameters()], lr=lr)
    batches = dataset.get_batches()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch_ids in batches:
            # Forward
            logits = model(batch_ids)
            # Shift labels
            labels = shift_right(batch_ids, dataset.pad_id)
            # Compute loss (cross-entropy)
            logits_flat = logits.data.reshape(-1, logits.data.shape[-1])
            labels_flat = labels.reshape(-1)
            # Simple cross-entropy
            probs = np.exp(logits_flat - np.max(logits_flat, axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            loss = -np.log(probs[np.arange(len(labels_flat)), labels_flat] + 1e-12).mean()
            # Backward
            model.zero_grad()
            model.backward(loss)
            print("Loss:", loss)
            optimizer.step() 


if __name__ == "__main__":
    # Load tokenizer and dataset
    tokenizer = TokenizerWrapper(
        "data/processed/tokenizer/vocab.json",
        "data/processed/tokenizer/merges.txt"
    )
    files = [
        "data/raw/wiki/wiki.jsonl",
        "data/raw/news/news.jsonl",
        "data/raw/github_repos/corpus.jsonl"
    ]
    dataset = TextDataset(files, tokenizer, max_len=512, batch_size=32)
    model = GPTModel(
        vocab_size=tokenizer.tokenizer.get_vocab_size(),
        max_seq_len=512,
        d_model=256,
        n_layers=4,
        n_heads=4,
        mlp_ratio=4,
        dropout=0.1,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.token_to_id("</s>")
    )
    train_gpt(model, dataset)
    