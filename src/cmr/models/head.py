"""Apply the LM head in fp32 at the scored positions.

A bf16 head rounds logits coarsely and differently for different GEMM shapes, so letter
log-probs depend on the code path. Applying the head to the final hidden state of the
scored rows in fp32 removes that noise. The frozen weight's fp32 copy is cached on the head.
Only allow-listed architectures with logits exactly ``lm_head(last_hidden_state)`` and a
plain ``nn.Linear`` head qualify; all others keep the model's own head.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported lazily so the CLI constants stay cheap
    import torch
    from torch import nn

# class names whose logits are exactly lm_head(last_hidden_state)
PLAIN_HEAD_ARCHS = frozenset(
    {
        # text CausalLMs
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "Qwen2ForCausalLM",
        "Qwen3ForCausalLM",
    }
)
_CACHE_ATTR = "_cmr_fp32_weight"  # (weight identity key, fp32 copy) on the head module


def plain_head_base(model: Any) -> Any | None:
    """The PEFT-unwrapped model if its architecture is in :data:`PLAIN_HEAD_ARCHS`, else None."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if type(base).__name__ not in PLAIN_HEAD_ARCHS or not hasattr(base, "lm_head"):
        return None
    return base


def fp32_capable(head: Any) -> bool:
    """True for an exact ``nn.Linear`` with a floating weight (no quantised subclasses)."""
    import torch
    from torch import nn

    return type(head) is nn.Linear and torch.is_floating_point(head.weight)


def fp32_head(model: Any) -> nn.Linear | None:
    """The model's LM head if fp32 scoring applies to it, else ``None``."""
    base = plain_head_base(model)
    if base is None or not fp32_capable(base.lm_head):
        return None
    head: nn.Linear = base.lm_head
    return head


def _fp32_weight(head: nn.Linear) -> torch.Tensor:
    """fp32 copy of a frozen head weight, cached until the weight changes."""
    import torch

    w = head.weight
    key = (w.data_ptr(), w._version, w.dtype, str(w.device), tuple(w.shape))

    cached = head.__dict__.get(_CACHE_ATTR)
    if cached is None or cached[0] != key:
        # not an inference tensor, so a later training pass can save it for backward
        with torch.inference_mode(False), torch.no_grad():
            cached = (key, w.detach().float())
        head.__dict__[_CACHE_ATTR] = cached  # plain attribute: not a buffer, not saved
    copy: torch.Tensor = cached[1]
    return copy


def clear_fp32_cache(model: Any) -> None:
    """Drop the cached fp32 head copy of ``model`` (if any).

    Call after writes the cache key cannot see (``.data`` writes, PEFT merges).
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    head = getattr(base, "lm_head", None)
    if head is not None:
        head.__dict__.pop(_CACHE_ATTR, None)


def fp32_head_logits(hidden: torch.Tensor, head: nn.Linear) -> torch.Tensor:
    """fp32 logits ``(..., V)`` from final hidden states ``(..., H)``; differentiable in ``hidden``.

    ``hidden`` is the decoder output after its final norm (``last_hidden_state``).

    A trainable head weight is cast per call (so gradients reach it); a frozen one uses
    the cached fp32 copy.
    """
    import torch
    from torch import nn

    h = hidden.float()
    w, b = head.weight, head.bias
    bias = None if b is None else b.float()

    if w.dtype == torch.float32:  # fp32 model: exactly the head's own F.linear
        return nn.functional.linear(h, w, bias)
    if w.requires_grad and torch.is_grad_enabled():
        return nn.functional.linear(h, w.float(), bias)
    return nn.functional.linear(h, _fp32_weight(head), bias)


def head_dtype_info(requested_fp32: bool, applied: bool) -> dict[str, str]:
    """Run-provenance entry: the requested and the effective LM-head dtype.

    >>> head_dtype_info(True, False)
    {'head_dtype_requested': 'fp32', 'head_dtype': 'model'}
    """
    return {
        "head_dtype_requested": "fp32" if requested_fp32 else "model",
        "head_dtype": "fp32" if (requested_fp32 and applied) else "model",
    }


# shared --head-dtype CLI option
HEAD_DTYPE_CHOICES = ("fp32", "model")
HEAD_DTYPE_HELP = (
    "LM head at the scored position: fp32 (default; letter log-probs that do not depend on the "
    "code path) or model (the LLM's own dtype, as in a plain Hugging Face forward pass)"
)


def parse_head_dtype(value: str) -> bool:
    """CLI ``--head-dtype {fp32,model}`` -> ``head_fp32``; ``ValueError`` otherwise.

    >>> parse_head_dtype("fp32"), parse_head_dtype("model")
    (True, False)
    """
    if value not in HEAD_DTYPE_CHOICES:
        raise ValueError(f"--head-dtype must be fp32 or model, got {value!r}")
    return value == "fp32"
