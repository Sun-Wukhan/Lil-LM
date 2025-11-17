from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from .rotary import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class GPTConfig:
    vocab_size: int
    n_layers: int
    n_heads: int
    hidden_size: int
    intermediate_size: int
    max_seq_len: int
    rotary_pct: float = 1.0
    rotary_emb_base: int = 10000
    dropout: float = 0.1
    bias: bool = False


class RMSNorm(nn.Module):
    """
    RMSNorm from PaLM / LLaMA style models.
    Slightly cheaper and often better behaved than LayerNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.hidden_size // config.n_heads
        assert (
            config.hidden_size % config.n_heads == 0
        ), "hidden_size must be divisible by n_heads"

        self.qkv_proj = nn.Linear(
            config.hidden_size, 3 * config.hidden_size, bias=config.bias
        )
        self.out_proj = nn.Linear(
            config.hidden_size, config.hidden_size, bias=config.bias
        )

        rotary_dim = int(self.head_dim * config.rotary_pct)
        self.rotary_dim = rotary_dim
        self.rotary_emb = RotaryEmbedding(
            rotary_dim, base=config.rotary_emb_base
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, hidden_size)
        attn_mask: (batch, 1, seq_len, seq_len) with -inf for masked positions
        """
        bsz, seq_len, _ = x.size()

        qkv = self.qkv_proj(x)  # (batch, seq_len, 3 * hidden)
        qkv = qkv.view(
            bsz, seq_len, 3, self.n_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        # q, k, v: (batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary to first rotary_dim dims
        if self.rotary_dim > 0:
            cos, sin = self.rotary_emb(seq_len, device=x.device)
            q_rot, k_rot = apply_rotary_pos_emb(
                q[..., : self.rotary_dim],
                k[..., : self.rotary_dim],
                cos,
                sin,
            )
            q = torch.cat([q_rot, q[..., self.rotary_dim :]], dim=-1)
            k = torch.cat([k_rot, k[..., self.rotary_dim :]], dim=-1)

        # scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (b, h, s, s)
        attn_scores = attn_scores / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)  # (b, h, s, d)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        )
        out = self.out_proj(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=config.bias
        )
        self.fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=config.bias
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = RMSNorm(config.hidden_size)
        self.ln2 = RMSNorm(config.hidden_size)
        self.attn = MultiHeadAttention(config)
        self.mlp = FeedForward(config)

    def forward(self,
                x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        # MLP with residual
        x = x + self.mlp(self.ln2(x))
        return x
