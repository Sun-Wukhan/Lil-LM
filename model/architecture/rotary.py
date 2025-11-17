import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """
    Rotary positional embeddings (RoPE) for attention.
    Shapes are aligned to (batch, heads, seq_len, head_dim)
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base

        # frequencies: dim/2 values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device=None):
        """
        Returns cos and sin with shape:
        (1, 1, seq_len, dim)
        which correctly broadcasts to q,k of shape:
        (batch, heads, seq_len, head_dim)
        """
        device = device if device is not None else self.inv_freq.device

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        # freqs shape: (seq_len, dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)

        # duplicate to get full dim
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)

        # reshape to (1, 1, seq_len, dim) for correct broadcasting
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]

        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Given last dimension D, treat as (D/2 complex numbers).
    Perform a 90-degree complex rotation.
    """
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor,
                         k: torch.Tensor,
                         cos: torch.Tensor,
                         sin: torch.Tensor):
    """
    Apply RoPE. Inputs:
    q, k: (batch, heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim)
    """
    seq_len = q.shape[2]

    cos = cos[:, :, :seq_len, :]
    sin = sin[:, :, :seq_len, :]

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot
