# MiniLLM-Project

A minimal, educational implementation of GPT-family language models in pure Python and NumPy. This project covers data collection, preprocessing, tokenization, model building, training, and inference for autoregressive language modeling.

## Features

- **Data Collection**: Scripts for crawling and collecting data from Wikipedia, news, and GitHub repositories (`src/data_collection/`).
- **Preprocessing & Tokenization**: Utilities for cleaning, chunking, and tokenizing text using ByteLevel BPE (`src/data/`).
- **Custom Deep Learning Framework**: Lightweight tensor, autograd, and module system inspired by PyTorch (`src/core/`).
- **Model Components**: Modular implementation of Embedding, Positional Embedding, MultiHeadAttention, FeedForward, LayerNorm/RMSNorm, Dropout, and Linear layers (`src/model/layers.py`).
- **GPT-family Models**: Flexible GPT and CausalLM classes with attention cache for efficient generation (`src/model/gpt.py`).
- **Training & Evaluation**: Example training script (`train.py`) and Jupyter notebooks for data EDA and baseline experiments (`notebooks/`).
- **Utilities**: Random seed control, parameter management, and optimizer (`src/utils/`, `src/core/optim.py`).

## Folder Structure

```
MiniLLM-Project/
│
├── src/
│   ├── core/              # Minimal deep learning framework (tensor, autograd, module, optim)
│   ├── data/              # Dataset builders, tokenization, preprocessing utilities
│   ├── data_collection/   # Web crawlers for wiki, news, github
│   ├── model/             # Model layers and GPT-family implementations
│   ├── utils/             # General utilities (random seed, etc.)
│   └── train.py           # Training script for GPT-family models
│
├── notebooks/             # Jupyter notebooks for EDA and baseline experiments
│
└── README.md
```

## Quick Start

1. **Install dependencies**  
   - Python 3.8+
   - `numpy`
   - `tokenizers` (for ByteLevelBPETokenizer)

2. **Collect and preprocess data**
   ```bash
   python src/data_collection/craw_wiki.py
   python src/data_collection/craw_news.py
   python src/data_collection/craw_github.py
   # Preprocess and chunk data as needed
   ```

3. **Train tokenizer**
   ```python
   from src.data.tokenizer_utils import TokenizerWrapper
   tokenizer = TokenizerWrapper.train_from_files([...])
   ```

4. **Build dataset**
   ```python
   from src.data.dataset_builders import TextDataset
   dataset = TextDataset([...], tokenizer, max_len=512, batch_size=32)
   ```

5. **Train model**
   ```bash
   python src/train.py
   ```

## Notebooks

- `notebooks/data_eda.ipynb`: Data exploration and visualization.
- `notebooks/training_baseline.ipynb`: Baseline training and gradient checks.

## Custom Framework Highlights

- **Tensor & Autograd**: Minimal implementation of tensor operations and automatic differentiation.
- **Module System**: PyTorch-like module registration and parameter management.
- **Attention Cache**: Efficient autoregressive generation with key/value caching.

## License

MIT License

## Authors

- [Minhtb105]

## Acknowledgements

Inspired by PyTorch, HuggingFace Transformers, and the "Attention is All You Need" paper.