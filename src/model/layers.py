import numpy as np
import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.core.tensor import Tensor
from src.core.module import Module
from src.utils.random_utils import set_seed
from src.core.init import (
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    normal_,
    uniform_,
    zeros_,
)



# ==============================================================
# Utility: Unified parameter initialization
# ==============================================================
def _init_param(tensor: Tensor, init: str, gain: float = 1.0):
    """
    Initialize a Tensor using the specified initialization method.
    
    Parameters
    ----------
    tensor : Tensor
        The tensor to initialize (its `.data` will be replaced in-place).
    init : str
        One of ['xavier_uniform', 'xavier_normal', 
                'kaiming_uniform', 'kaiming_normal',
                'uniform', 'normal', 'none']
    gain : float, optional
        Scaling factor applied to variance (useful for activations like ReLU).
    """
    shape = tensor.data.shape

    if init == "xavier_uniform":
        tensor.data = xavier_uniform_(shape)
    elif init == "xavier_normal":
        tensor.data = xavier_normal_(shape)
    elif init == "kaiming_uniform":
        tensor.data = kaiming_uniform_(shape)
    elif init == "kaiming_normal":
        tensor.data = kaiming_normal_(shape)
    elif init == "uniform":
        tensor.data = uniform_(shape)
    elif init == "normal":
        tensor.data = normal_(shape)
    elif init == "ones":
        tensor.data = np.ones(shape, dtype=np.float32)
    elif init == "zeros":
        tensor.data = np.zeros(shape, dtype=np.float32)
    elif init == "none":
        pass
    else:
        raise ValueError(f"Unknown init method: {init}")
    return tensor


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, init: str = "kaiming_uniform"):
        """
        Fully connected linear layer: y = xW + b
        
        Parameters
        ----------
        in_dim : int
            Number of input features
        out_dim : int
            Number of output features
        bias : bool
            Whether to include bias term
        init : str
            Initialization method ('xavier_uniform', 'kaiming_normal', etc.)
        """
        super().__init__()
        W = Tensor(np.empty((in_dim, out_dim)), requires_grad=True)
        self.add_parameter("weight", _init_param(W, init))

        if bias:
            b = Tensor(np.zeros(out_dim), requires_grad=True)
            self.add_parameter("bias", b)
        else:
            self._parameters["bias"] = None

    def __call__(self, x: Tensor):
        y = x.matmul(self._parameters["weight"])
        b = self._parameters["bias"]
        
        return y + b if b is not None else y
        
        
class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5, init: str = "default"):
        """
        Layer Normalization layer.

        Parameters
        ----------
        dim : int
            Number of features per token.
        eps : float
            Small constant added for numerical stability.
        init : str
            Initialization type for gamma/beta. 
            If "default", uses gamma=ones, beta=zeros (PyTorch style).
        """
        super().__init__()
        self.eps = eps

        gamma = Tensor(np.empty(dim), requires_grad=True)
        beta = Tensor(np.empty(dim), requires_grad=True)

        if init == "default":
            _init_param(gamma, "ones")
            _init_param(beta, "zeros")
        else:
            _init_param(gamma, init)
            _init_param(beta, init)

        self.add_parameter("gamma", gamma)
        self.add_parameter("beta", beta)

    def __call__(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        norm = (x - mean) / (var + self.eps).sqrt()
        
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
    def __init__(self, vocab_size: int, d_model: int, init: str = "xavier_normal"):
        """
        Word embedding layer mapping token IDs → vectors.
        """
        super().__init__()
        W = Tensor(np.empty((vocab_size, d_model)), requires_grad=True)
        self.add_parameter("weight", _init_param(W, init))

    def __call__(self, token_ids: list[int]):
        return self._parameters["weight"][token_ids]
    
    
class PositionalEmbedding(Module):
    def __init__(self, max_len: int, d_model: int, init: str = "normal"):
        """
        Learnable positional embeddings added to token embeddings.
        """
        super().__init__()
        P = Tensor(np.empty((max_len, d_model)), requires_grad=True)
        self.add_parameter("pos_emb", _init_param(P, init))

    def __call__(self, x: Tensor):
        L = x.shape[1]
        
        return x + self._parameters["pos_emb"][:L]


# ---------------- RMSNorm (LLaMA/Falcon) ----------------
class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-8, init: str = "default"):
        """
        Root Mean Square Normalization (used in LLaMA/Falcon models).
        
        Parameters
        ----------
        dim : int
            Hidden dimension.
        eps : float
            Stability constant.
        init : str
            Initialization method for the weight. Default = ones.
        """
        super().__init__()
        self.eps = eps

        weight = Tensor(np.empty(dim), requires_grad=True)
        if init == "default":
            _init_param(weight, "ones")
        else:
            _init_param(weight, init)

        self.add_parameter("weight", weight)

    def __call__(self, x: Tensor):
        rms = (x.pow(2).mean(axis=-1, keepdims=True) + self.eps).sqrt()
        
        return self._parameters["weight"] * (x / rms)


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

    def __call__(self, x: Tensor, past_key_value=None, use_cache=False):
        batch, seq_len, d_model = x.shape
        n_heads = self.n_heads
        head_dim = self.head_dim

        # Project and reshape (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_model)
        Q = self.Wq(x).data.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        K = self.Wk(x).data.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
        V = self.Wv(x).data.reshape(batch, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        # If past_key_value is provided, concatenate
        if past_key_value is not None:
            past_k, past_v = past_key_value
            K = Tensor(np.concatenate([past_k.data, K], axis=2))
            V = Tensor(np.concatenate([past_v.data, V], axis=2))

        # Save new cache if use_cache
        new_past = (K, V) if use_cache else None

        # Attention score: (batch, n_heads, seq_len, seq_len)
        attn_scores = (Q.matmul(K.transpose(0,1,3,2))) / np.sqrt(head_dim)

        if self.causal:
            mask = np.triu(np.ones((L, L)), k=1) * -1e9
            attn_scores = attn_scores + Tensor(mask[np.newaxis, np.newaxis, :, :])

        attn_probs = attn_scores.softmax(axis=-1)
        attn_probs = self.dropout(attn_probs)

        # Weighted sum: (batch, n_heads, seq_len, head_dim)
        attn_out = attn_probs.matmul(V)

        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, L, D)

        if use_cache:
            return self.Wo(attn_out), new_past
        
        return self.Wo(attn_out)

class FeedForward(Module):
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
    
    print("\n---- MLP ----")
    mlp = MLP([2, 10, 10, 8])
    for name, param in mlp.parameters():
        print(name)
    
    print("---- Linear ----")
    linear = Linear(4, 6, init="kaiming_uniform")
    for name, param in linear.parameters():
        print(name, param.data.shape)

    print("\n---- Embedding ----")
    emb = Embedding(vocab_size=20, d_model=8, init="xavier_normal")
    for name, param in emb.parameters():
        print(name, param.data.shape)

    print("\n---- PositionalEmbedding ----")
    pos_emb = PositionalEmbedding(max_len=16, d_model=8, init="uniform")
    for name, param in pos_emb.parameters():
        print(name, param.data.shape)
        
    print("\n---- LayerNorm ----")
    ln = LayerNorm(8)
    for n, p in ln.parameters():
        print("LayerNorm:", n, p.data.shape, p.data.mean())

    print("\n---- RMSNorm ----")
    rms = RMSNorm(8)
    for n, p in rms.parameters():
        print("RMSNorm:", n, p.data.shape, p.data.mean())    
    