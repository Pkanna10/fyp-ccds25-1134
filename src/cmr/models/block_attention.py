"""Block-causal attention for training passes (``cmr-stage-b --block-attention``).

Computes causal attention in query blocks, each scored only against the keys up to its own
end, so the block of the score matrix that is entirely masked is never computed (0.75 n^2
instead of n^2 scores with two blocks). PyTorch's SDPA training path on the Apple-silicon GPU
builds the full matrix and then masks it (report Section 3.8.2, Figure 3.5). Exact up to
rounding. Applies only to unpadded causal self-attention with gradients on; everything else
uses SDPA.
"""

from __future__ import annotations

import itertools
from typing import Any

import torch

_ORIGINAL: Any = None  # transformers' sdpa_attention_forward while the override is on
_MASKS: dict[tuple[int, int, int, str], torch.Tensor] = {}
N_BLOCKS = 2


def _mask(lo: int, hi: int, groups: int, device: torch.device) -> torch.Tensor:
    """Causal mask (groups * (hi - lo), hi): row (g, p), for query p in [lo, hi), allows keys <= p.

    Rows are ordered by query head ``g`` within a key/value group, then by position ``p``.
    """
    if torch.compiler.is_compiling():  # dict mutation inside a graph forces recompiles
        pos = torch.arange(lo, hi, device=device).repeat(groups)
        return torch.arange(hi, device=device)[None, :] <= pos[:, None]
    key = (lo, hi, groups, str(device))
    if key not in _MASKS:
        pos = torch.arange(lo, hi, device=device).repeat(groups)
        _MASKS[key] = torch.arange(hi, device=device)[None, :] <= pos[:, None]
    return _MASKS[key]


def block_causal_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scaling: float,
    n_blocks: int = N_BLOCKS,
) -> torch.Tensor:
    """Causal (grouped-query) attention, (B, H, L, D) x (B, Hkv, L, D) -> (B, H, L, D).

    Query head h uses key/value head h // (H // Hkv), as ``enable_gqa`` does; the query heads
    that share a key/value head are stacked into one matrix multiplication per query block.
    """
    b, h, n, d = query.shape
    hkv = key.shape[1]
    g = h // hkv
    cuts = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    outs = []
    for lo, hi in itertools.pairwise(cuts):
        if hi == lo:
            continue
        qg = query[:, :, lo:hi].reshape(b, hkv, g * (hi - lo), d)  # rows ordered (g, position)
        s = torch.matmul(qg, key[:, :, :hi].transpose(-1, -2)) * scaling
        s = s.masked_fill(~_mask(lo, hi, g, query.device), float("-inf"))
        p = torch.softmax(s, dim=-1, dtype=torch.promote_types(s.dtype, torch.float32))
        p = p.to(query.dtype)  # softmax in at least fp32, as transformers' eager path
        outs.append(torch.matmul(p, value[:, :, :hi]).reshape(b, h, hi - lo, d))
    return torch.cat(outs, dim=2)


def _attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """transformers attention-interface function: block-causal when it applies, else SDPA."""
    causal = is_causal if is_causal is not None else getattr(module, "is_causal", False)
    if (
        torch.is_grad_enabled()
        and causal
        and attention_mask is None
        and dropout == 0.0
        and query.shape[2] == key.shape[2] > 1
        and query.shape[1] % key.shape[1] == 0
        and kwargs.get("position_bias") is None
    ):
        scale = scaling if scaling is not None else query.shape[-1] ** -0.5
        out = block_causal_attention(query, key, value, scale)
        return out.transpose(1, 2).contiguous(), None
    result: tuple[torch.Tensor, None] = _ORIGINAL(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        **kwargs,
    )
    return result


def enable() -> None:
    """Route transformers' ``sdpa`` attention through :func:`_attention` (idempotent)."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    _ORIGINAL = sdpa_attention_forward
    AttentionInterface.register("sdpa", _attention)


def disable() -> None:
    """Restore transformers' own ``sdpa`` attention."""
    global _ORIGINAL
    if _ORIGINAL is None:
        return
    from transformers import AttentionInterface

    AttentionInterface.register("sdpa", _ORIGINAL)
    _ORIGINAL = None


def enabled() -> bool:
    """True while the block-causal override is installed."""
    return _ORIGINAL is not None
