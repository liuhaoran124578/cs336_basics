import logging

import torch
import torch.nn as nn
from einops import einsum

logger = logging.getLogger(__name__)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        std = (2 / (self.in_features + self.out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0, a=-3 * std, b=3 * std)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return einsum(X, self.weight, "... d_in, d_out d_in -> ... d_out")

    def extra_repr(self):
        return f"d_out={self.weight.shape[0]}, d_in={self.weight.shape[1]}"


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

    def extra_repr(self):
        return f"vocab_size={self.weight.shape[0]}, d={self.weight.shape[1]}"


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"hidden_size={self.weight.shape[0]}, eps={self.eps}"


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, hidden_size, max_seq_len, device=None):
        super().__init__()

        indices = torch.arange(0, hidden_size, 2, device=device)
        inv_freq = 1.0 / (theta ** (indices / hidden_size))

        t = torch.arange(max_seq_len, device=device, dtype=inv_freq.dtype)

        freqs = torch.outer(t, inv_freq)
        emb = freqs.repeat_interleave(2, dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, token_positions):
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        x_rotated = torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)

        x_out = x * cos + x_rotated * sin
        return x_out.to(x.dtype)
