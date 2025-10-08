import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.tensor import Tensor
from src.core.module import Module
from src.utils.random_utils import set_seed


class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        W = Tensor(np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim), requires_grad=True)
        self.add_parameter("weight", W)

        if bias:
            b = Tensor(np.zeros(out_dim), requires_grad=True)
            self.add_parameter("bias", b)
        else:
            self._parameters["bias"] = None
            
    def __call__(self, x: Tensor):
        y = x.matmul(self._parameters["weight"])
        b = self._parameters["bias"]
        if b is not None:
            y = y + b
            
        return y
        
        
class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.add_parameter("gamma", Tensor(np.ones(dim), requires_grad=True))
        self.add_parameter("beta", Tensor(np.zeros(dim), requires_grad=True))

    def __call__(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        norm = (x - mean) / (var + self.eps) ** 0.5
        
        return self._parameters["gamma"] * norm + self._parameters["beta"]    
    
    
class MLP(Module):
    def __init__(self, layer_dims):
        """
        layer_dims: List of integers, e.g. [input_dim, hidden1, ..., output_dim]
        """
        super().__init__()
        assert len(layer_dims) >= 2, "MLP needs at least input and output dims"
        
        self.layers = []
        
        for i in range(len(layer_dims) - 1):
            linear = Linear(layer_dims[i], layer_dims[i + 1])
            self.add_module(f"fc{i + 1}", linear)
            self.layers.append(linear)
            
    def __call__(self, x: Tensor):
        for layer in self.layers:
           x = layer(x)
           
        return x 


class Embedding(Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.add_parameter("weight", 
            Tensor(np.random.randn(vocab_size, d_model) * 0.02, 
            requires_grad=True))
        
    def __call__(self, token_ids: list[int]):
        return self._parameters["weight"][token_ids]
    
    
class PositionalEmbedding(Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.add_parameter("pos_emb",
            Tensor(np.random.randn(max_len, d_model) * 0.02, 
            requires_grad=True))
        
    def __call__(self, x: Tensor):
        L = x.shape[1]
        return x + self._parameters["pos_emb"][:L]


# ---------------- RMSNorm (LLaMA/Falcon) ----------------
class RMSNorm(Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.add_parameter("weight", Tensor(np.ones(dim), requires_grad=True))

    def __call__(self, x: Tensor):
        norm = x / (x.pow(2).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        return self._parameters["weight"] * norm


class Dropout(Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, x: Tensor, training=True):
        if not training or self.p == 0.0:
            return x
        
        mask = (np.random.rand(*x.shape) >= self.p).astype(np.float32) / (1.0 - self.p)
        return x * mask


class MultiHeadAttention(Module):
    def __init__(self, d_model, n_heads, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.head_dim = d_model // n_heads

        self.Wq = Linear(d_model, d_model, bias=False)
        self.Wk = Linear(d_model, d_model, bias=False)
        self.Wv = Linear(d_model, d_model, bias=False)
        self.Wo = Linear(d_model, d_model, bias=False)
        
        self.dropout = Dropout(0.1)

    def __call__(self, x: Tensor):
        batch, seq_len, d_model = x.shape
        n_heads = self.n_heads
        head_dim = self.head_dim

        # Project and reshape (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_model)
        Q = self.Wq(x).reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        K = self.Wk(x).reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        V = self.Wv(x).reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        # Attention score: (batch, n_heads, seq_len, seq_len)
        scores = (Q.matmul(K.transpose(0,1,3,2))) / np.sqrt(Hd)

        if self.causal:
            mask = np.triu(np.ones((L, L)), k=1) * -1e9
            scores = scores + Tensor(mask[np.newaxis, np.newaxis, :, :])

        attn = scores.softmax(axis=-1)
        attn = self.dropout(attn)

        # Weighted sum: (batch, n_heads, seq_len, head_dim)
        out = attn.matmul(V)

        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        
        return self.Wo(out)

def FeedForward(Module):
    def __init__(self, dim, hidden_dim, activation="relu", dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        self.fc1 = Linear(dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, dim)
        self.dropout = Dropout(dropout)
        self.activation = activation
        
    def __call__(self, x: Tensor):
        x = self.fc1(x)
        if self.activation == "relu":
            x = x.relu()
        elif self.activation == "tanh":
            x = x.tanh()
            
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    set_seed(42)
    mlp = MLP([2, 10, 10, 8])
    for name, param in mlp.parameters():
        print(name)
    