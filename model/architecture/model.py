# --- Fix Python path so we can run this script directly ---
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------------

import json
from typing import Optional

import torch
from torch import nn

from .layers import GPTConfig, TransformerBlock, RMSNorm


class GPTModel(nn.Module):
    """
    A small GPT-style decoder-only transformer.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size
        )

        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.ln_f = RMSNorm(config.hidden_size)

        # LM head (tied weights is also common; we can do that later if desired)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @staticmethod
    def from_config_file(path: str) -> "GPTModel":
        with open(path, "r") as f:
            cfg_dict = json.load(f)
        cfg = GPTConfig(**cfg_dict)
        return GPTModel(cfg)

    def _build_causal_mask(self, seq_len: int, device):
        """
        Returns a causal mask of shape (1, 1, seq_len, seq_len)
        with -inf above the diagonal, 0 on/under it.
        """
        mask = torch.full(
            (seq_len, seq_len),
            float("-inf"),
            device=device,
        )
        mask = torch.triu(mask, diagonal=1)
        # shape: (1, 1, seq_len, seq_len) so it broadcasts over batch & heads
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        """
        input_ids: (batch, seq_len)
        attention_mask: optional, (batch, seq_len) where 1 = valid, 0 = pad.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}"
            )

        x = self.embed_tokens(input_ids)  # (b, s, h)
        x = self.drop(x)

        # Base causal mask
        causal_mask = self._build_causal_mask(seq_len, device=device)

        # Optionally combine with padding mask
        if attention_mask is not None:
            # attention_mask: (b, s) → (b, 1, 1, s)
            pad_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            # where pad_mask is True, we add -inf
            causal_mask = causal_mask.expand(bsz, -1, -1, -1)
            causal_mask = causal_mask.masked_fill(pad_mask, float("-inf"))
        else:
            causal_mask = causal_mask  # (1, 1, s, s)

        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (b, s, vocab_size)
        return logits
