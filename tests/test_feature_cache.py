"""Feature cache for the frozen SigLIP encoder (FrozenBridgeVLM.feature_cache_dir; Table 4.14).

Saved encoder outputs must give bit-identical losses, gradients and scores, and the encoder
then runs once per image. Uses the tiny random bridge of test_bridge.py (CPU, no downloads).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cmr.models.base import Query  # noqa: E402
from tests.test_bridge import img, tiny_model  # noqa: E402

TEXTS = ["Question: what colour?\nOptions:\nA. red\nB. blue\nAnswer with the letter.",
         "Question: how many?\nOptions:\nA. one\nB. two\nAnswer with the letter."]  # fmt: skip
TARGETS = ["A", "B"]


def keyed(i: int, split: str = "train") -> Any:
    im = img(i)
    im.info["cmr_key"] = f"{split}/{i}"
    return im


def loss_and_grads(m: Any, images: list[Any]) -> tuple[Any, list[Any]]:
    params = list(m.projector.parameters())
    for p in params:
        p.grad = None
    loss = m.loss_parts(TEXTS, images, TARGETS).all
    loss.sum().backward()
    return loss.detach(), [p.grad.clone() for p in params]


def test_cache_gives_bitwise_identical_loss_gradients_and_scores(tmp_path: Path) -> None:
    m = tiny_model()
    images = [keyed(0), keyed(1)]
    base_loss, base_grads = loss_and_grads(m, images)
    base_score = m.letter_logprobs(Query(TEXTS[0], images[0]), 2)

    m.feature_cache_dir = tmp_path
    for _ in range(2):  # first pass writes (misses), second reads (hits)
        loss, grads = loss_and_grads(m, images)
        assert torch.equal(loss, base_loss)
        assert all(torch.equal(a, b) for a, b in zip(grads, base_grads, strict=True))
        m.clear_cache()
        assert m.letter_logprobs(Query(TEXTS[0], images[0]), 2) == base_score
    assert len(list(tmp_path.rglob("*.pt"))) == 2


def test_cache_skips_unkeyed_images_and_other_splits(tmp_path: Path) -> None:
    m = tiny_model()
    m.feature_cache_dir = tmp_path
    m.feature_cache_splits = frozenset({"val"})
    m.loss_parts(TEXTS, [img(0), keyed(1, "train")], TARGETS)  # unkeyed, then a train image
    assert list(tmp_path.rglob("*.pt")) == []
    m.loss_parts(TEXTS, [keyed(2, "val"), keyed(3, "val")], TARGETS)
    assert len(list(tmp_path.rglob("*.pt"))) == 2


def test_cache_files_are_keyed_by_tower_dtype(tmp_path: Path) -> None:
    m = tiny_model()
    m.feature_cache_dir = tmp_path
    fp32_path = m._feature_path(keyed(0))
    m.vision.to(torch.bfloat16)  # the tower now computes in bf16: a different cache file
    bf16_path = m._feature_path(keyed(0))
    assert fp32_path is not None and bf16_path is not None and fp32_path != bf16_path
    assert fp32_path.name == bf16_path.name == "train_0.pt"
