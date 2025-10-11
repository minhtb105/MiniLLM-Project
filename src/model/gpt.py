import numpy as np
from typing import List, Optional
from src.core.module import Module
from src.core.tensor import Tensor
from src.model.layers import (
    Embedding, PositionalEmbedding,
    LayerNorm, RMSNorm,
    MultiHeadAttention, FeedForward, Dropout, Linear
)
from src.core.init import xavier_normal_, normal_
from src.core.optim import Adam


class TransformerBlock(Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.1, attn_dropout: float = 0.1,
                 norm_type: str = "layernorm"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        if norm_type == "rmsnorm":
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)
        else:
            self.ln1 = LayerNorm(d_model)
            self.ln2 = LayerNorm(d_model)
            
        self.attn = MultiHeadAttention(d_model, n_heads, causal=True)
        self.ffn = FeedForward(d_model, d_model * mlp_ratio, activation="gelu", dropout=dropout)
        self.dropout = Dropout(dropout)

    def __call__(self, x: Tensor, past_key_value=None, use_cache=False):
        # x: (B, L, D)
        # Attention block (pre-LN)
        x_norm = self.ln1(x)
        attn_out = self.attn(x_norm)  # shape (B, L, D)
        x = x + self.dropout(attn_out)

        # Feed-forward block
        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)
        
        return x
    
class CausalLM(Module):
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    def forward(self, input_ids: np.ndarray, past_key_values=None, use_cache=False) -> Tensor:
        raise NotImplementedError
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: int = 4,
    ) -> List[int]:
        """
        Generate tokens using Top-p Beam Sampling + Temperature + Repetition Penalty.
        Uses cache (past_key_values) for efficient autoregressive generation.
        """
        beams = [(input_ids, 0.0, None)]  # list of (token_ids, score, past_key_values)
        eos = getattr(self, "eos_token_id", None)
        finished = []
        
        for _ in range(max_new_tokens):
            new_beams = []
            for seq, log_prob, past_key_value in beams:
                # Get logits for the last token
                input_token = seq[-1:].reshape(1, 1)
                logits, new_past = self.forward(input_token, 
                                                past_key_values=past_key_value, 
                                                use_cache=True)
                logits = logits.data[0, -1]  # (vocab_size,)  # get numpy array
                
                # Apply repetition penalty
                for t in set(seq.tolist()):
                    if logits[t] < 0:
                        logits[t] *= repetition_penalty
                    else:
                        logits[t] /= repetition_penalty
                    
                # Apply temperature
                logits = logits / max(temperature, 1e-8)
                
                # Convert logits to probabilities
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                
                # Top-p filtering
                sorted_idx = np.argsort(probs)[::-1]
                sorted_probs = probs[sorted_idx]
                cumulative_probs = np.cumsum(sorted_probs)
                cutoff = np.where(cumulative_probs > top_p)[0]
                if len(cutoff) > 0:
                    sorted_idx = sorted_idx[:cutoff[0] + 1]
                    sorted_probs = sorted_probs[:cutoff[0] + 1]

                sorted_probs = sorted_probs / sorted_probs.sum()    
                
                # Sample next tokens from top-p filtered distribution
                next_token_ids = np.random.choice(sorted_idx, p=sorted_probs)
                next_log_prob = np.log(probs[next_token_ids] + 1e-12)
                
                # update beam sequences
                new_seq = np.concatenate([seq, [next_token_ids]])
                new_beams.append((new_seq, log_prob + next_log_prob, new_past))

            # Select top beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)
            beams = new_beams[:num_beams]
            
            # early stop
            if eos is not None and all(seq[-1] == eos for seq, _ in beams):
                break
            
            # Check for finished sequences (e.g., if EOS token is generated)
            for seq, _ in beams:
                if seq[-1] == eos:
                    finished.append(seq)
                    
            if len(finished) >= num_beams:
                break

        if finished:
            best_seq = sorted(finished, key=lambda x: len(x))[0]
        else:
            best_seq = beams[0][0]

        return best_seq.tolist()
    
    
class GPTModel(CausalLM):
    def __init__(self,
                 vocab_size: int,
                 max_seq_len: int,
                 d_model: int = 768,
                 n_layers: int = 6,
                 n_heads: int = 6,
                 mlp_ratio: int = 4,
                 dropout: float = 0.1,
                 tie_word_embeddings: bool = True,
                 norm_type: str = "layernorm",
                 eos_token_id: Optional[int] = None):
        """
        Minimal GPT-style model.

        - vocab_size: size of vocabulary
        - max_seq_len: maximum context length (positional embeddings)
        - d_model: model hidden dim
        - n_layers: number of transformer blocks
        - n_heads: number of attention heads
        - tie_word_embeddings: whether to tie token embedding and lm_head
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.n_layers = n_layers
        self.eos_token_id = eos_token_id
    
        # Embedding layers
        self.token_emb = Embedding(vocab_size, d_model, init="xavier_normal")
        
        self.pos_emb = PositionalEmbedding(max_seq_len, d_model)
      
        # Transformer blocks
        self.blocks: List[TransformerBlock] = []
        for i in range(n_layers):
            blk = TransformerBlock(d_model, n_heads, mlp_ratio=mlp_ratio, dropout=dropout, norm_type=norm_type)
            self.blocks.append(blk)
            
        # Final normalization (pre-head)
        if norm_type == "rmsnorm":
            self.ln_f = RMSNorm(d_model)
        else:
            self.ln_f = LayerNorm(d_model)
            
        # Output projection (lm head)
        # If tying embeddings, we use token_emb.weight.T as projection weight at forward-time.
        self.lm_head = Linear(d_model, vocab_size, bias=False, init="normal")

        self.tie_word_embeddings = tie_word_embeddings
        if tie_word_embeddings:
            # We will tie weights at forward time.
            pass

    def forward(self, input_ids: np.ndarray, past_key_values=None, use_cache=False) -> Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            past_key_values: list of (key, value) for each layer
            use_cache: bool -> return updated cache
        Returns:
            logits: (batch, seq_len, vocab_size)
            past_key_values (n_layers list of tuples) if use_cache=True
        """
        # Convert input ids to embeddings via model's Embedding (which returns Tensor)
        x = self.token_emb(input_ids) + self.pos_emb(input_ids)  # expects token_ids list/ndarray -> returns Tensor (B,L,D)
        new_past = []
        
        for i, blk in enumerate(self.blocks):
            past = past_key_values[i] if past_key_values is not None else None
            x, layer_past = blk(x, past=past, use_cache=use_cache) if use_cache else (blk(x), None)
            if use_cache:
                new_past.append(layer_past)

        x = self.ln_f(x)  # (B, L, D)

        # LM head: project to vocab logits
        if self.tie_word_embeddings:
            # tied weights: logits = x @ token_emb.weight.T
            W = self.token_emb._parameters["weight"]  # Tensor(vocab_size, d_model)
            logits = x.matmul(W.transpose(1,0))  # x (B,L,D) @ (D, V) -> (B,L,V)
        else:
            logits = self.lm_head(x)  # Linear layer returns Tensor

        return logits
    
    def __call__(self, input_ids: np.ndarray) -> Tensor:
        return self.forward(input_ids)
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        num_beams: int = 4,
    ) -> List[int]:
        """
        Generate tokens using Top-p Beam Sampling + Temperature + Repetition Penalty.
        """
        return super().generate(input_ids, max_new_tokens, temperature, top_p, repetition_penalty, num_beams)
    