import math

import torch
import torch.nn as nn
from einops import einsum, rearrange

from .building_blocks import Linear, RMSNorm, RotaryPositionalEmbedding
from .utils import softmax


class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None, device=None, dtype=None):
        super().__init__()
        if not d_ff:
            d_ff = int((d_model * 8 / 3) // 64) * 64
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim (batch,seq_len,d_model)
        """
        o_1 = self.silu(self.w1(x))
        o_3 = self.w3(x)
        o_final = self.w2(o_1 * o_3)
        return o_final




def Attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor | None = None):
    """
    Scaled dot-product attention
    Input:
        queries: torch.Tensor (batch_size, ..., seq_len_q, d_k)
        keys: torch.Tensor (batch_size, ..., seq_len_k, d_k)
        values: torch.Tensor (batch_size, ..., seq_len_k, d_v)
        mask: torch.Tensor (seq_len_q, seq_len_k), True for positions to keep.
    """
    d_k = queries.shape[-1]
    q_dot_k_scaled = einsum(
        queries, keys, "... seq_len_q d_k,  ... seq_len_k d_k -> ... seq_len_q seq_len_k"
    ) / math.sqrt(d_k)
    
    # --- 修正开始 ---
    attention_scores = q_dot_k_scaled
    if mask is not None:
        
        attention_scores = attention_scores.masked_fill(mask == False, float("-inf"))
    
    softmax_q_dot_k_scaled = softmax(attention_scores, dim=-1)

    output = einsum(softmax_q_dot_k_scaled, values, "... seq_len_q seq_len_k , ... seq_len_k d_v -> ... seq_len_q d_v")
    return output

    

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention
    """

    def __init__(
        self, d_model: int, num_heads: int, positional_embedding_layer: nn.Module | None = None, device=None, dtype=None
    ):
        super().__init__()
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.q_proj = Linear(
            in_features=self.d_model, out_features=self.d_k * self.num_heads, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            in_features=self.d_model, out_features=self.d_k * self.num_heads, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            in_features=self.d_model, out_features=self.d_v * self.num_heads, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            in_features=self.d_model, out_features=self.d_v * self.num_heads, device=device, dtype=dtype
        )
        self.positional_embedding_layer = positional_embedding_layer

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x dim: (batch_size, seq_len, d_model)
        positional_embedding_layer: nn.Module that applies Rotary Positional Embedding
        token_positions: torch.Tensor
        """
        ## compute the key, value, and query projections
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q_heads = rearrange(
            Q, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k
        )
        K_heads = rearrange(
            K, "... seq_len (num_heads d_k) -> ... num_heads seq_len d_k", num_heads=self.num_heads, d_k=self.d_k
        )
        V_heads = rearrange(
            V, "... seq_len (num_heads d_v) -> ... num_heads seq_len d_v", num_heads=self.num_heads, d_v=self.d_v
        )
        ## apply rotary positional embedding to query and key vectors
        if self.positional_embedding_layer is not None:
            Q_heads = self.positional_embedding_layer(x=Q_heads, token_positions=token_positions)
            K_heads = self.positional_embedding_layer(x=K_heads, token_positions=token_positions)
        ## create causal mask
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]
        causal_mask = ~torch.triu(torch.ones((seq_len_q, seq_len_k), dtype=torch.bool), diagonal=1)
        ## apply attention function
        attention_heads = Attention(
            queries=Q_heads, keys=K_heads, values=V_heads, mask=causal_mask
        )  # (... num_heads, seq_len, d_v)
        attention = rearrange(attention_heads, "... num_heads seq_len d_v -> ... seq_len (num_heads d_v)")
        ## final output
        output = self.output_proj(attention)
        return output


class GroupQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        nums_key_value_head: int,
        positional_embedding_layer: nn.Module | None = None,
        device=None,
        dtype=None,
    ):
        """
        Causal group-query self-attention
        """
        super().__init__()
        assert d_model % num_heads == 0
        assert num_heads % nums_key_value_head == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.nums_key_value_head = nums_key_value_head
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(
            in_features=self.d_model, out_features=self.head_dim * self.num_heads, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            in_features=self.d_model, out_features=self.head_dim * self.nums_key_value_head, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            in_features=self.d_model, out_features=self.head_dim * self.nums_key_value_head, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            in_features=self.d_model, out_features=self.head_dim * self.num_heads, device=device, dtype=dtype
        )
        self.positional_embedding_layer = positional_embedding_layer

    def forward(self, X: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """
        x dim: (batch_size, seq_len, d_model)
        positional_embedding_layer: nn.Module that applies Rotary Positional Embedding
        token_positions: torch.Tensor
        """
        Q = self.q_proj(X)
        K = self.k_proj(X)
        V = self.v_proj(X)

        Q_heads = rearrange(
            Q,
            "... seq_len (num_heads head_dim) -> ... num_heads seq_len head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        K_heads = rearrange(
            K,
            "... seq_len (nums_key_value_head head_dim) -> ... nums_key_value_head seq_len head_dim",
            nums_key_value_head=self.nums_key_value_head,
            head_dim=self.head_dim,
        )
        V_heads = rearrange(
            V,
            "... seq_len (nums_key_value_head head_dim) -> ... nums_key_value_head seq_len head_dim",
            nums_key_value_head=self.nums_key_value_head,
            head_dim=self.head_dim,
        )

        if self.positional_embedding_layer is not None:
            Q_heads = self.positional_embedding_layer(x=Q_heads, token_positions=token_positions)
            K_heads = self.positional_embedding_layer(x=K_heads, token_positions=token_positions)

        K_heads = K_heads.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=-3)
        V_heads = V_heads.repeat_interleave(self.num_heads // self.nums_key_value_head, dim=-3)

        ## create causal mask
        seq_len_q = Q.shape[-2]
        seq_len_k = K.shape[-2]
        causal_mask = ~torch.triu(torch.ones((seq_len_q, seq_len_k), dtype=torch.bool, device=X.device), diagonal=1)

        ## apply attention function
        attention_heads = Attention(
            queries=Q_heads, keys=K_heads, values=V_heads, mask=causal_mask
        )  # (... num_heads, seq_len, d_v)
        attention = rearrange(attention_heads, "... num_heads seq_len head_dim -> ... seq_len (num_heads head_dim)")

        ## final output
        output = self.output_proj(attention)
        return output


class Transformer_Block(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, device=None, dtype=None
    ):
        super().__init__()
        """
        rms norm
        """
        self.ln1 = RMSNorm(hidden_size=d_model)
        self.ln2 = RMSNorm(hidden_size=d_model)
        """
        ROPE
        """
        self.rope = RotaryPositionalEmbedding(
            theta=theta, hidden_size=d_model // num_heads, max_seq_len=max_seq_len, device=device
        )
        """
        attention layer
        """
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, positional_embedding_layer=self.rope, device=device, dtype=dtype
        )
        """
        point-wise layer
        """
        self.ffn = positionwise_feedforward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        X shape (batch,seq_len,d_model)
        """
        seq_len = X.shape[-2]
        token_position = rearrange(torch.arange(seq_len), "seq_len -> 1 1 seq_len ")
        output_1 = self.attn(self.ln1(X), token_position)
        output_2 = output_1 + X
        output = output_2 + self.ffn(self.ln2(output_2))
        return output
