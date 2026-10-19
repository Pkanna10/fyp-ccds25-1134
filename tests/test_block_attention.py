"""Block-causal training attention (cmr.models.block_attention; Section 3.8.2, Figure 3.5).

It must give the same values and gradients as PyTorch's causal attention, and leave the bridge
loss and gradients unchanged when switched on.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cmr.models import block_attention as ba  # noqa: E402
from tests.test_bridge import img, tiny_model  # noqa: E402


@pytest.mark.parametrize(("heads", "kv_heads", "length"), [(4, 2, 9), (4, 4, 7), (6, 2, 16)])
def test_matches_sdpa_values_and_gradients(heads: int, kv_heads: int, length: int) -> None:
    torch.manual_seed(0)
    q = torch.randn(2, heads, length, 8, dtype=torch.float64, requires_grad=True)
    k = torch.randn(2, kv_heads, length, 8, dtype=torch.float64, requires_grad=True)
    v = torch.randn(2, kv_heads, length, 8, dtype=torch.float64, requires_grad=True)
    g = heads // kv_heads
    ref = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(g, 1), v.repeat_interleave(g, 1), is_causal=True, scale=0.3
    )
    out = ba.block_causal_attention(q, k, v, 0.3)
    assert torch.allclose(out, ref, atol=1e-12)
    cot = torch.randn_like(ref)
    gr = torch.autograd.grad(ref, (q, k, v), cot)
    go = torch.autograd.grad(out, (q, k, v), cot)
    assert all(torch.allclose(a, b, atol=1e-12) for a, b in zip(go, gr, strict=True))


def test_bridge_loss_and_gradients_unchanged_and_eval_untouched() -> None:
    m = tiny_model()
    texts = ["Question: what colour?\nA. red\nB. blue", "Question: how many?\nA. one\nB. two"]
    images, targets = [img(0), img(1)], ["A", "B"]

    def run() -> tuple[torch.Tensor, list[torch.Tensor]]:
        params = list(m.projector.parameters())
        for p in params:
            p.grad = None
        loss = m.loss_parts(texts[:1], images[:1], targets[:1]).all
        loss.sum().backward()
        return loss.detach(), [p.grad.clone() for p in params]

    base_loss, base_grads = run()
    try:
        ba.enable()
        assert ba.enabled()
        loss, grads = run()
    finally:
        ba.disable()
    assert not ba.enabled()
    assert torch.allclose(loss, base_loss, rtol=1e-5, atol=1e-6)
    for a, b in zip(grads, base_grads, strict=True):  # fp32: rounding-level differences only
        assert ((a - b).norm() / b.norm()).item() < 1e-4
