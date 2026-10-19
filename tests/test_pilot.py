"""Laptop speed-ups used only in pilot runs (Table 4.14, Section 3.8.2; cmr.train.pilot).

Padding training sequences to a multiple of 32, passing precomputed vision-encoder features,
CPU-GPU batch splitting (``cpu_share``, "co-training" in the code) and torch.compile of the
decoder layers must leave losses and gradients unchanged. Tiny float32 CPU models from
tests/test_bridge.py. The co-trainer here is a second CPU copy (in real use the main model is
on the GPU), so co-training must reproduce the plain loop up to summation order.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cmr.models.bridge import _right_pad  # noqa: E402
from cmr.train.bridge_sft import TrainConfig, config_hash, train_bridge  # noqa: E402
from cmr.train.pilot import CompiledLayers, CpuCoTrainer  # noqa: E402
from cmr.train.targets import TargetFormat, answer_char_span  # noqa: E402
from tests.test_bridge import img, loader, make_items, tiny_model  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread() -> Any:
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_right_pad_to_multiple() -> None:
    rows = [(torch.ones(5, 2), torch.arange(5)), (torch.ones(3, 2), torch.arange(3))]
    e, lab, mask = _right_pad(rows, 4)
    assert e.shape == (2, 8, 2) and lab.shape == (2, 8)
    assert mask.tolist() == [[1] * 8, [1, 1, 1, 0, 0, 1, 1, 1]]
    assert (lab[:, 5:] == -100).all() and (e[:, 5:] == 0).all()
    e1, _, m1 = _right_pad(rows)  # default: no extra padding
    assert e1.shape == (2, 5, 2) and m1.tolist() == [[1] * 5, [1, 1, 1, 0, 0]]


def _parts_and_grads(m: Any, **kw: Any) -> tuple[list[Any], list[Any]]:
    for p in m.projector.parameters():
        p.grad = None
    target = "B\nBecause: it is blue."
    parts = m.loss_parts(
        ["Which colour?\nA. red\nB. blue"], [img(2)], [target],
        [answer_char_span(target, TargetFormat.ANSWER_THEN_RATIONALE)], **kw,
    )  # fmt: skip
    (parts.answer + 0.3 * parts.rest).sum().backward()
    vals = [parts.all.detach(), parts.answer.detach(), parts.rest.detach()]
    return vals, [p.grad.clone() for p in m.projector.parameters()]


def test_padding_and_tower_features_leave_loss_and_gradients_unchanged() -> None:
    m = tiny_model()
    base_v, base_g = _parts_and_grads(m)
    m.pad_multiple = 32
    pad_v, pad_g = _parts_and_grads(m)
    m.pad_multiple = 1
    feat_v, feat_g = _parts_and_grads(m, tower_feats=m.tower_features([img(2)]))
    for vals, grads in ((pad_v, pad_g), (feat_v, feat_g)):
        assert all(
            torch.allclose(a, b, rtol=1e-5, atol=1e-6) for a, b in zip(vals, base_v, strict=True)
        )
        for a, b in zip(grads, base_g, strict=True):
            assert ((a - b).norm() / b.norm()).item() < 1e-5


@pytest.mark.parametrize("target", [TargetFormat.ANSWER_ONLY, TargetFormat.ANSWER_THEN_RATIONALE])
def test_cpu_cotraining_reproduces_the_plain_loop(tmp_path: Path, target: TargetFormat) -> None:
    items = make_items(8)
    common = dict(
        target=target, epochs=2, batch_size=1, grad_accum=4, lr=1e-2, eval_every=100,
        option_shuffle=True, rationale_weight=0.3, select="last",
    )  # fmt: skip
    plain = tiny_model(seed=1)
    r1 = train_bridge(plain, items, loader, TrainConfig(**common), tmp_path / "plain")
    pilot = tiny_model(seed=1)
    copy = tiny_model(seed=1)
    r2 = train_bridge(
        pilot, items, loader, TrainConfig(**common, cpu_share=2), tmp_path / "pilot",
        cpu_model=copy, cpu_threads=1,
    )  # fmt: skip
    assert len(r1["history"]) == len(r2["history"]) == 4
    for a, b in zip(r1["history"], r2["history"], strict=True):
        assert a["ce"] == pytest.approx(b["ce"], rel=1e-5)
        assert {"sec_step", "sec_gpu", "sec_cpu"} <= set(b)
    for a, b in zip(plain.projector.parameters(), pilot.projector.parameters(), strict=True):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)
    # the CPU copy ends with the main model's weights
    for a, b in zip(pilot.projector.parameters(), copy.projector.parameters(), strict=True):
        assert torch.equal(a.detach(), b.detach())


def test_pilot_settings_change_the_hash_only_when_on() -> None:
    m, items = tiny_model(), make_items(4)

    def h(cfg: TrainConfig) -> str:
        return config_hash(cfg, m, items, items)

    assert h(TrainConfig()) == h(TrainConfig(cpu_share=0, compile=False))
    assert h(TrainConfig(cpu_share=2)) != h(TrainConfig())
    assert h(TrainConfig(compile=True)) != h(TrainConfig())


def test_pilot_refuses_bad_settings(tmp_path: Path) -> None:
    m, items = tiny_model(), make_items(4)
    with pytest.raises(ValueError, match="needs cpu_model"):
        train_bridge(m, items, loader, TrainConfig(grad_accum=4, cpu_share=2), tmp_path)
    with pytest.raises(ValueError, match="cpu_share must be"):
        train_bridge(
            m, items, loader, TrainConfig(grad_accum=2, cpu_share=2), tmp_path, cpu_model=m
        )
    with pytest.raises(ValueError, match="margin arm"):
        train_bridge(m, items, loader, TrainConfig(compile=True, margin_lambda=0.5), tmp_path)
    with pytest.raises(ValueError, match="on the CPU"):
        CpuCoTrainer(SimpleNamespace(device=torch.device("meta")), 1)  # type: ignore[arg-type]


def test_compiled_layers_switch_on_and_off() -> None:
    m = tiny_model()
    ctx = CompiledLayers(m, backend="eager")  # the eager backend: fast, same arithmetic
    assert all(layer._compiled_call_impl is None for layer in ctx.layers)
    base, _ = _parts_and_grads(m)
    with ctx.active():
        assert all(layer._compiled_call_impl is not None for layer in ctx.layers)
        comp, _ = _parts_and_grads(m)
    assert all(layer._compiled_call_impl is None for layer in ctx.layers)
    assert all(torch.allclose(a, b, rtol=1e-6) for a, b in zip(comp, base, strict=True))


def test_auto_share_balances_measured_times() -> None:
    co = CpuCoTrainer(SimpleNamespace(device=torch.device("cpu")), 1, share=5, auto=True)  # type: ignore[arg-type]
    for _ in range(3):  # compile warm-up steps: ignored
        co.observe(10.0, 11, 1.0, 5, 16)
    for _ in range(10):  # GPU 0.165 s, CPU 0.6 s per example -> r = 3.6 -> 16 / 4.6 -> 3
        co.observe(0.165 * 11, 11, 0.6 * 5, 5, 16)
    assert co.share == 3
    fixed = CpuCoTrainer(SimpleNamespace(device=torch.device("cpu")), 1, share=5)  # type: ignore[arg-type]
    for _ in range(20):
        fixed.observe(0.165 * 11, 11, 0.6 * 5, 5, 16)
    assert fixed.share == 5


def test_auto_share_training_runs(tmp_path: Path) -> None:
    items = make_items(8)
    cfg = TrainConfig(epochs=1, batch_size=1, grad_accum=4, lr=1e-2, eval_every=100, cpu_share=-1)
    res = train_bridge(
        tiny_model(seed=1), items, loader, cfg, tmp_path, cpu_model=tiny_model(seed=1)
    )
    assert all(1 <= row["cpu_share"] <= 3 for row in res["history"])
