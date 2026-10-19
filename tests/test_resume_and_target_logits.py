"""Exact mid-run resume of train_bridge, and LM-head logits computed only where needed.

Uses the tiny CPU models from tests/test_bridge.py:
  * resume: K steps without interruption give the same weights, AdamW state and log as
    k steps, a crash and a resume to K; changed settings or code versions are refused
  * loss_parts: logits computed only at the target positions give the same loss and gradients
    as the full [batch, length, vocabulary] logits; likewise last-position-only logits for
    scoring and generation
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

from cmr.data.aokvqa import AOKVQAItem  # noqa: E402
from cmr.models import head as head_mod  # noqa: E402
from cmr.models.base import Query  # noqa: E402
from cmr.models.bridge import FrozenBridgeVLM  # noqa: E402
from cmr.train.bridge_sft import (  # noqa: E402
    RESUME_FILE,
    TrainConfig,
    _answer_logprob,
    train_bridge,
)
from cmr.train.generate import greedy_generate, rationale_then_letter  # noqa: E402
from cmr.train.targets import (  # noqa: E402
    LETTER_FORMATS,
    TargetFormat,
    answer_char_span,
    prompt_for,
    target_for,
)
from tests.test_bridge import (  # noqa: E402
    img,
    loader,
    make_items,
    tiny_checkpoints,  # noqa: F401  (fixture)
    tiny_model,
)


@pytest.fixture(autouse=True)
def one_thread() -> Any:
    """Run tiny models on one thread (faster on a busy CPU)."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


# ---------------------------------------------------------------------- resume


class Crash(RuntimeError):
    pass


def crashing_loader(after: int) -> Any:
    """The image loader dies on its ``after``-th call (a kill in the middle of a step)."""
    calls = {"n": 0}

    def load(it: AOKVQAItem) -> Image.Image:
        calls["n"] += 1
        if calls["n"] > after:
            raise Crash("killed")
        return loader(it)

    return load


RESUME_CFG = TrainConfig(
    target=TargetFormat.ANSWER_THEN_RATIONALE,  # draws rationales: target RNG matters
    epochs=2, batch_size=1, grad_accum=2, lr=3e-2, eval_every=2, margin_lambda=0.5,
    select="dev_logp", dev_teacher_forced=False,
)  # fmt: skip  # 5 items x 2 epochs / 2 per step -> 5 steps; the order wraps mid-step


def _log(out: Path) -> str:
    return (out / "train_log.jsonl").read_text()


def _opt_tensors(out: Path) -> dict[str, Any]:
    return dict(torch.load(out / RESUME_FILE, weights_only=True))


def test_resume_is_bit_identical_to_an_uninterrupted_run(tmp_path: Path) -> None:
    items, dev = make_items(5), make_items(2)
    straight = tiny_model(seed=1)
    res_a = train_bridge(straight, items, loader, RESUME_CFG, tmp_path / "a", dev, 2)
    assert res_a["steps"] == 5 and res_a["resumed_from"] is None

    # run b is killed in micro-step 1 of step 4 (2 images per micro-step: real + swapped,
    # for the image-margin loss; 2 dev images at the step-2 eval): checkpoint at step 2,
    # step 3 logged
    out_b = tmp_path / "b"
    with pytest.raises(Crash):
        train_bridge(tiny_model(seed=1), items, crashing_loader(3 * 4 + 2 + 1), RESUME_CFG,
                     out_b, dev, 2)  # fmt: skip
    assert torch.load(out_b / RESUME_FILE, weights_only=True)["step"] == 2
    assert len(_log(out_b).splitlines()) == 3  # step 3 must be dropped and redone
    resumed = tiny_model(seed=1)  # a fresh process: initial weights again
    with pytest.raises(FileExistsError, match="--resume"):  # never resumes silently
        train_bridge(tiny_model(seed=1), items, loader, RESUME_CFG, out_b, dev, 2)
    res_b = train_bridge(resumed, items, loader, RESUME_CFG, out_b, dev, 2, "if_present")
    assert res_b["resumed_from"] == 2

    for (name, p), q in zip(
        straight.trainable_state().items(), resumed.trainable_state().values(), strict=True
    ):
        assert torch.equal(p, q), name
    a, b = _opt_tensors(tmp_path / "a"), _opt_tensors(out_b)
    assert a["step"] == b["step"] == 5
    assert all(torch.equal(a["trainable"][k], b["trainable"][k]) for k in a["trainable"])
    sa, sb = a["optimizer"]["state"], b["optimizer"]["state"]
    assert sa.keys() == sb.keys()
    for k in sa:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            assert torch.equal(sa[k][field], sb[k][field]), (k, field)
    assert a["optimizer"]["param_groups"] == b["optimizer"]["param_groups"]
    assert a["scheduler"] == b["scheduler"]
    assert a["order"] == b["order"] and a["rng_target"] == b["rng_target"]
    assert _log(tmp_path / "a") == _log(out_b)  # no duplicated or missing step
    steps = [json.loads(line)["step"] for line in _log(out_b).splitlines()]
    assert steps == list(range(1, 6))
    assert res_a["history"] == res_b["history"]
    assert res_a["best_dev_acc"] == res_b["best_dev_acc"]
    for f in ("projector.pt", "trainable.pt"):  # the dev_logp-selected checkpoint
        sa_, sb_ = (torch.load(d / f, weights_only=True) for d in (tmp_path / "a", out_b))
        assert all(torch.equal(sa_[k], sb_[k]) for k in sa_)

    # finished run: re-running is a no-op that returns the logged history
    again = train_bridge(tiny_model(seed=1), items, loader, RESUME_CFG, out_b, dev, 2, "require")
    assert again["resumed_from"] == 5 and again["history"] == res_a["history"]
    assert _log(out_b) == _log(tmp_path / "a")


def test_resume_refuses_a_changed_config_and_honours_flags(tmp_path: Path) -> None:
    items = make_items(4)
    cfg = TrainConfig(epochs=1, batch_size=2, grad_accum=1, lr=1e-2, eval_every=100)
    train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path, checkpoint_every=1)
    with pytest.raises(FileExistsError, match=r"resume\.pt exists"):  # default: refuse
        train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path)
    for mode in ("if_present", "require"):
        with pytest.raises(ValueError, match="different configuration"):
            train_bridge(
                tiny_model(seed=2), items, loader, TrainConfig(**{**cfg.__dict__, "lr": 2e-2}),
                tmp_path, resume=mode,
            )  # fmt: skip
    with pytest.raises(ValueError, match="different configuration"):  # other training data
        train_bridge(tiny_model(seed=2), make_items(5), loader, cfg, tmp_path, resume="require")
    with pytest.raises(ValueError, match="different configuration"):  # other dev set
        train_bridge(
            tiny_model(seed=2), items, loader, cfg, tmp_path, make_items(2), resume="require"
        )
    with pytest.raises(FileNotFoundError):
        train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path / "new", resume="require")
    with pytest.raises(ValueError, match="unknown resume mode"):
        train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path / "x", resume="yes")  # type: ignore[arg-type]
    # "restart" starts over: same log as a fresh folder, stale state replaced
    fresh = train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path, resume="restart")
    assert fresh["resumed_from"] is None
    assert [json.loads(r)["step"] for r in _log(tmp_path).splitlines()] == [1, 2]
    assert not list(tmp_path.glob(".*.tmp"))  # atomic writes leave no temporary files


def test_resume_across_code_versions_needs_explicit_resume(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    items = make_items(4)
    cfg = TrainConfig(epochs=1, batch_size=2, grad_accum=1, lr=1e-2, eval_every=100)
    kw: dict[str, Any] = {"checkpoint_every": 1}
    train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path, code_version="aaa", **kw)
    assert torch.load(tmp_path / RESUME_FILE, weights_only=True)["code_version"] == "aaa"
    same = train_bridge(
        tiny_model(seed=2), items, loader, cfg, tmp_path, resume="if_present", code_version="aaa"
    )
    assert same["resumed_from"] == 2 and same["saved_code_version"] == "aaa"
    with pytest.raises(ValueError, match="code aaa; this run is code bbb"):
        train_bridge(
            tiny_model(seed=2), items, loader, cfg, tmp_path, resume="if_present",
            code_version="bbb",
        )  # fmt: skip
    res = train_bridge(
        tiny_model(seed=2), items, loader, cfg, tmp_path, resume="require", code_version="bbb"
    )
    assert res["resumed_from"] == 2 and res["saved_code_version"] == "aaa"
    assert "WARNING" in capsys.readouterr().out


def test_preflight_resume_refuses_like_train_bridge_without_side_effects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """preflight_resume raises the same refusals as train_bridge and writes nothing."""
    from cmr.train.bridge_sft import preflight_resume

    items, dev = make_items(4), make_items(2)
    cfg = TrainConfig(epochs=1, batch_size=2, grad_accum=1, lr=1e-2, eval_every=100)
    model = tiny_model(seed=2)
    new = tmp_path / "new"
    assert preflight_resume(model, items, cfg, new, dev, "refuse") == 0  # nothing yet
    assert preflight_resume(model, items, cfg, new, dev, "if_present") == 0
    with pytest.raises(FileNotFoundError, match="--resume"):
        preflight_resume(model, items, cfg, new, dev, "require")
    with pytest.raises(ValueError, match="unknown resume mode"):
        preflight_resume(model, items, cfg, new, dev, "yes")  # type: ignore[arg-type]
    assert not new.exists()  # no side effect

    out = tmp_path / "run"
    train_bridge(tiny_model(seed=2), items, loader, cfg, out, dev, 1, code_version="aaa")
    before = {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()}
    capsys.readouterr()
    with pytest.raises(FileExistsError, match="--resume"):
        preflight_resume(model, items, cfg, out, dev, "refuse", "aaa")
    with pytest.raises(ValueError, match="code aaa; this run is code bbb"):
        preflight_resume(model, items, cfg, out, dev, "if_present", "bbb")
    with pytest.raises(ValueError, match="different configuration"):
        preflight_resume(model, make_items(5), cfg, out, dev, "if_present", "aaa")
    assert preflight_resume(model, items, cfg, out, dev, "if_present", "aaa") == 2
    assert preflight_resume(model, items, cfg, out, dev, "require", "bbb") == 2  # accepted
    assert preflight_resume(model, items, cfg, out, dev, "restart", "bbb") == 0
    assert "WARNING" not in capsys.readouterr().out  # train_bridge itself warns once
    assert {p.name: p.read_bytes() for p in out.iterdir() if p.is_file()} == before


def test_selected_checkpoint_after_last_resume_state_does_not_survive_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A best saved after the last resume.pt does not survive a kill and lower-scoring replay.

    The replay leaves the selected step-4 state on disk, not the killed segment's step 6.
    """
    import cmr.train.bridge_sft as sft

    def fake_dev(scores: list[float]) -> Any:
        calls = {"n": 0}

        def dev_metrics(*args: Any, **kwargs: Any) -> dict[str, float]:
            calls["n"] += 1
            return {"dev_acc": scores[calls["n"] - 1], "dev_answer_nll": 1.0}

        return dev_metrics

    cfg = TrainConfig(
        target=TargetFormat.ANSWER_ONLY, epochs=4, batch_size=2, grad_accum=1, lr=3e-2,
        eval_every=2, select="dev_acc", dev_teacher_forced=False,
    )  # fmt: skip  # 6 items x 4 epochs / 2 -> 12 steps; evals at 2, 4, 6, ...
    items, dev = make_items(6), make_items(2)
    # first run: new bests at steps 2, 4, 6; killed right after the step-6 best is saved
    monkeypatch.setattr(sft, "dev_metrics", fake_dev([0.25, 0.5, 0.75]))
    real_save, saves = sft._save, {"n": 0}

    def save_then_die(st: dict[str, Any], out: Path) -> None:
        real_save(st, out)
        saves["n"] += 1
        if saves["n"] == 3:
            raise Crash("killed")

    monkeypatch.setattr(sft, "_save", save_then_die)
    with pytest.raises(Crash):
        train_bridge(tiny_model(seed=1), items, loader, cfg, tmp_path, dev, 4)
    monkeypatch.setattr(sft, "_save", real_save)
    saved = torch.load(tmp_path / RESUME_FILE, weights_only=True)
    assert saved["step"] == 6 and saved["pending_row"]["step"] == 6  # pre-dev state
    step4 = saved["best_state"]  # the selected state as of the saved step
    killed = torch.load(tmp_path / "trainable.pt", weights_only=True)  # step 6
    assert not all(torch.equal(killed[k], step4[k]) for k in step4)
    # replay: step 6 now ties step 4 (strict > keeps step 4), nothing later is better
    monkeypatch.setattr(sft, "dev_metrics", fake_dev([0.5] + [0.0] * 4))
    res = train_bridge(tiny_model(seed=1), items, loader, cfg, tmp_path, dev, 4, "require")
    assert [r["step"] for r in res["history"] if r.get("selected")] == [2, 4]
    on_disk = torch.load(tmp_path / "trainable.pt", weights_only=True)
    proj = torch.load(tmp_path / "projector.pt", weights_only=True)
    assert on_disk.keys() == step4.keys()
    assert all(torch.equal(on_disk[k], step4[k]) for k in step4)
    assert all(torch.equal(proj[k.removeprefix("projector.")], v) for k, v in step4.items())
    assert res["best_dev_acc"] == 0.5


def test_resume_without_a_selection_yet_removes_stale_checkpoint_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cmr.train.bridge_sft as sft

    cfg = TrainConfig(epochs=2, batch_size=2, grad_accum=1, lr=1e-2, eval_every=100)
    items = make_items(4)  # 4 steps, no dev items -> only the final step is selected
    train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path, checkpoint_every=2)
    state = torch.load(tmp_path / RESUME_FILE, weights_only=True)
    state["step"], state["best_state"] = 2, None  # as if killed between step 2 and the end
    torch.save(state, tmp_path / RESUME_FILE)
    assert (tmp_path / "trainable.pt").exists()  # the stale file of the "killed" segment
    seen: list[bool] = []
    real_save = sft._save

    def spy(st: dict[str, Any], out: Path) -> None:
        seen.append((out / "trainable.pt").exists() or (out / "projector.pt").exists())
        real_save(st, out)

    monkeypatch.setattr(sft, "_save", spy)
    # resume the step-2 state (its log rows are the first two of the finished run)
    res = train_bridge(tiny_model(seed=2), items, loader, cfg, tmp_path, checkpoint_every=2,
                       resume="require")  # fmt: skip
    assert res["resumed_from"] == 2
    assert seen == [False]  # removed on resume, written again only by the final step


def test_config_hash_covers_lora_hyperparameters() -> None:
    pytest.importorskip("peft")
    from cmr.train.bridge_sft import _json_default, config_hash

    cfg, items = TrainConfig(), make_items(4)
    hashes = []
    for alpha in (4, 64, 4):
        m = tiny_model(seed=0)
        m.enable_lora(r=2, alpha=alpha, target_modules=("q_proj", "v_proj"))
        hashes.append(config_hash(cfg, m, items))
    assert hashes[0] != hashes[1] and hashes[0] == hashes[2]
    assert _json_default({"v_proj", "q_proj"}) == ["q_proj", "v_proj"]
    assert _json_default(frozenset({"b", "a"})) == ["a", "b"]


@pytest.mark.slow
def test_config_hash_is_stable_across_hash_seeds() -> None:
    """config_hash ignores the PYTHONHASHSEED-dependent order of LoRA target_modules."""
    import os
    import subprocess
    import sys

    code = (
        "import torch; torch.set_num_threads(1)\n"
        "from tests.test_bridge import tiny_model, make_items\n"
        "from cmr.train.bridge_sft import TrainConfig, config_hash\n"
        "m = tiny_model(seed=0)\n"
        "m.enable_lora(r=2, alpha=4, target_modules=('q_proj', 'v_proj', 'k_proj', 'o_proj'))\n"
        "print(list(m.llm.peft_config['default'].target_modules))\n"
        "print(config_hash(TrainConfig(), m, make_items(4)))\n"
    )
    root = Path(__file__).resolve().parents[1]
    outs = []
    for seed in ("1", "2", "3"):
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}  # fmt: skip
        res = subprocess.run(
            [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
        )
        outs.append(res.stdout.splitlines()[-2:])
    assert len({order for order, _ in outs}) > 1, outs  # the set order really differs
    assert len({h for _, h in outs}) == 1, outs


# ---------------------------------------------------------------------- target logits


def _batch(n_fmt: int | None = None) -> tuple[list[str], list[Any], list[str], list[Any]]:
    """Every target format in one right-padded batch, plus a no-image row."""
    it = make_items(3)[2]
    rng = random.Random(0)
    fmts = list(LETTER_FORMATS)[:n_fmt]  # every target that contains an answer letter
    targets = [target_for(it, f, rng) for f in fmts]
    prompts = [prompt_for(it, f) for f in fmts]
    images: list[Any] = [img(k) for k in range(len(fmts))]
    spans = [answer_char_span(t, f) for t, f in zip(targets, fmts, strict=True)]
    prompts.append("No image here?")
    images.append(None)
    targets.append("B because it is.")
    spans.append((0, 1))
    return prompts, images, targets, spans


def _parts_and_grads(
    m: FrozenBridgeVLM, fast: bool, weight: float = 0.5
) -> tuple[list[Any], list[Any]]:
    m.target_logits_only = fast
    m.projector.zero_grad(set_to_none=True)
    parts = m.loss_parts(*_batch())
    assert parts.answer is not None and parts.rest is not None
    (parts.all.mean() + parts.answer.mean() + weight * parts.rest.mean()).backward()
    grads = [p.grad.detach().clone() for p in m.projector.parameters()]  # type: ignore[union-attr]
    return [parts.all.detach(), parts.answer.detach(), parts.rest.detach()], grads


def _max_rel(a: Any, b: Any) -> float:
    return float(((a - b).abs() / b.abs().clamp_min(1e-12)).max())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_target_position_logits_match_full_logits(dtype: str) -> None:
    m = tiny_model(seed=3)
    if dtype == "bfloat16":  # as from_pretrained: frozen modules bf16, projector fp32
        m.llm.to(torch.bfloat16)
        m.vision.to(torch.bfloat16)
    assert m._decoder_and_head() is not None  # the fast path is really taken
    full, g_full = _parts_and_grads(m, fast=False)
    fast, g_fast = _parts_and_grads(m, fast=True)
    loss_rel, grad_tol = (1e-6, 1e-5) if dtype == "float32" else (1e-2, 5e-2)
    for x, y in zip(fast, full, strict=True):
        assert _max_rel(x, y) <= loss_rel
    for x, y in zip(g_fast, g_full, strict=True):
        assert float((x - y).abs().max()) <= grad_tol * float(y.abs().max())


def test_target_logits_with_lora_and_fallback_for_other_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("peft")
    m = tiny_model(seed=4)
    m.enable_lora(r=2, alpha=4, target_modules=("q_proj", "v_proj"))
    assert m._decoder_and_head() is not None  # PEFT wrapper is unwrapped
    full, _ = _parts_and_grads(m, fast=False)
    fast, _ = _parts_and_grads(m, fast=True)
    for x, y in zip(fast, full, strict=True):
        assert _max_rel(x, y) <= 1e-6
    lora = [p for n, p in m.named_parameters() if "lora_" in n and p.requires_grad]
    assert lora and all(p.grad is not None for p in lora)  # LoRA still trains
    monkeypatch.setattr(head_mod, "PLAIN_HEAD_ARCHS", frozenset())
    assert m._decoder_and_head() is None  # unknown head -> full logits, same numbers
    other, _ = _parts_and_grads(m, fast=True)
    for x, y in zip(other, full, strict=True):
        assert torch.equal(x, y)


def test_last_position_logits_in_margin_arm_and_generation() -> None:
    """Last-position-only logits leave the image-margin loss term (Table 4.15), greedy
    generation and R→A scoring unchanged."""
    m = tiny_model(seed=5)
    it = make_items(2)[1]
    fmt = TargetFormat.RATIONALE_THEN_ANSWER
    prompt = prompt_for(it, fmt)
    prefix = target_for(it, fmt, random.Random(0)).rsplit("Answer:", 1)[0] + "Answer:"
    q = Query(prompt, img(1))
    out = {}
    for fast in (True, False):
        m.score_last_logits_only = fast
        lp = _answer_logprob(m, [prompt, prompt], [img(1), img(2)], [prefix, ""], ["B", "C"])
        out[fast] = (lp.detach(), greedy_generate(m, q, max_new_tokens=5),
                     rationale_then_letter(m, q, 4, max_new_tokens=5))  # fmt: skip
    assert torch.allclose(out[True][0], out[False][0], rtol=1e-6, atol=1e-6)
    assert out[True][1] == out[False][1]
    lp_t, lp_f = out[True][2][0], out[False][2][0]
    assert lp_t == pytest.approx(lp_f, rel=1e-6, abs=1e-6)


@pytest.mark.slow
def test_stage_b_cli_refuses_rerun_and_records_resume(
    tiny_checkpoints: Path,  # noqa: F811  (the fixture imported above)
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cmr.runners import stage_b
    from tests.test_bridge import write_split

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 16)
    write_split(aokvqa, coco, "val", 4)
    out = tmp_path / "run"
    argv = [
        "--bridge", str(tiny_checkpoints / "spec.json"), "--target", "answer_only",
        "--aokvqa-dir", str(aokvqa), "--coco-dir", str(coco), "--epochs", "1",
        "--batch-size", "4", "--grad-accum", "1", "--eval-every", "100", "--select", "last",
        "--conditions", "real", "--checkpoint-every", "1", "--out", str(out),
    ]  # fmt: skip
    stage_b.main(argv)
    args_path = out / "run_args.json"
    first = json.loads(args_path.read_text())
    assert first["resumed_from_step"] is None and first["resume_saved_commit"] is None
    (seg,) = first["manifest"]["segments"]
    assert seg["start_step"] == 0 and seg["first_record_index"] == 0
    assert seg["git_commit"] == first["manifest"]["git_commit"]
    written = args_path.read_bytes()
    with pytest.raises(FileExistsError, match="--resume"):
        stage_b.main(argv)  # a rerun into the same folder is never silent
    assert args_path.read_bytes() == written  # refused before the rewrite
    real = stage_b.run_manifest

    def other_commit(*a: Any, **kw: Any) -> dict[str, Any]:
        return {**real(*a, **kw), "git_commit": "f" * 40, "started_utc": "LATER"}

    monkeypatch.setattr(stage_b, "run_manifest", other_commit)
    with pytest.raises(ValueError, match="Refusing to auto-resume across code versions"):
        stage_b.main([*argv, "--resume-if-present"])
    assert args_path.read_bytes() == written  # the code-version refusal too
    monkeypatch.setattr(stage_b, "run_manifest", real)
    stage_b.main([*argv, "--resume-if-present"])  # as the launchers run it: same commit -> go on
    again = json.loads(args_path.read_text())
    steps = [json.loads(r)["step"] for r in (out / "train_log.jsonl").read_text().splitlines()]
    assert again["resumed_from_step"] == steps[-1] == len(steps)
    assert again["resume_saved_commit"] == again["manifest"]["git_commit"]
    # the first manifest is kept; the restart is a second segment
    assert {k: v for k, v in again["manifest"].items() if k != "segments"} == {
        k: v for k, v in first["manifest"].items() if k != "segments"
    }
    n_val = len((out / "eval_val" / "predictions.jsonl").read_text().splitlines())
    assert [(s["start_step"], s["first_record_index"]) for s in again["manifest"]["segments"]] == [
        (0, 0),
        (len(steps), n_val),
    ]
