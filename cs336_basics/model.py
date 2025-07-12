import json
import logging
import os

import torch
import torch.nn as nn

from .building_blocks import Embedding, Linear, RMSNorm
from .layers import Transformer_Block
from .utils import softmax

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransformerLM(nn.Module):
    """
    Trainable Parameters compute
    1 token_embedding : vocab_size * d_model
    2 Transformer Block (num_layers = 48)
    2.1 RMSNorm : 2 * d_model
    2.2 MHA : 4 * d_model * d_model
    2.3 FFN(SWiGLU) : 3 * d_ff * d_model
    3 final_RMSNorm : d_model
    4 lm_head : d_model * vocab_size

    single-precision floating( 4 bytes)

    1KB = 2^10 Bytes
    1MB = 2^10 KB
    1GB = 2^10 MB
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                Transformer_Block(d_model, num_heads, d_ff, context_length, theta, device, dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(hidden_size=d_model)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size)
        # report number of parameters
        logger.info(f"number of non-embedding parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the lm_head parameters get subtracted.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.lm_head.weight.numel()

        return n_params

    def forward(self, X):
        """
        compute FLOPS
        1 Transformer Block flops compute (num_layers = 48)

        1.1 MHA
        1.1.1 Q,K,V Projection : 3 * 2 *  context_length * d_model * d_model
        1.1.2 score compute(Q @ K.T) :  2 * context_length * d_model * context_length
        1.1.3 logit compute(score @ V) : 2 * context_length * d_model * d_model

        1.2 SWiGLU
        1.2.1 W1,W3 Projection: 2 * (2 * context_length * d_model * d_ff)
        1.2.2 W2 Projection : 2 * context_length * d_ff * d_model

        2 lm_head
        output: 2 * context_length * d_model * vocab_size
        """
        hidden_states = self.token_embeddings(X)
        for layer in self.layers:
            hidden_states = layer(hidden_states)  # Shape remains (b, s, d_model)
        hidden_states = self.ln_final(hidden_states)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
    ):
        """
        Args:
            x: LongTensor of shape `(1, sequence_length,)` or `(sequence_length, )`.
                Input IDs to condition on when generating.
            max_new_tokens: int
                Maximum number of tokens to generate.
            temperature: float
                Temperature to use during generation.
            top_k: int
                If provided, only sample from the `top_k` vocab items (by probability).
            eos_token_id: int
                If provided, stop generation when we generate this ID.

        Returns: A LongTensor of shape (max_new_tokens,) with the generated model output.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_sequence_length = x.size(-1)
        for _ in range(max_new_tokens):
            # Take the last `context_length` tokens if the input is
            # beyond the model's context length
            x = x[:, -self.context_length :] if x.size(1) > self.context_length else x
            # Get the logits from the model
            logits = self.forward(x)
            # Take the logits for the next token
            next_token_logits = logits[:, -1]
            # apply temperature scaling
            temperature_scaled_next_token_logits = next_token_logits / temperature
            # If top-k is provided, take the tokens with the highest score
            if top_k:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, temperature_scaled_next_token_logits.size(-1)),
                )
                # Get the score of the kth item that we kept---items with lower scores should be masked.
                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill(topk_mask, float("-inf"))
            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            # End generation if we see the EOS token ID
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break
            x = torch.cat((x, next_token_id), dim=-1)
        new_token_ids = x[:, original_sequence_length:]
        return new_token_ids

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path) as f:
            config = json.load(f)
        model = cls(**config)
        weights_path = os.path.join(pretrained_model_path, "model.pt")
        state_dict = torch.load(weights_path)

        # Remove _orig_mod. prefix that comes from serializing a compiled model
        unwanted_prefix = "_orig_mod."
        for k, _ in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        return model
