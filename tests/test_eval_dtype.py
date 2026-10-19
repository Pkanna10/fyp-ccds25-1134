"""Training in bf16 and evaluating in fp32, on tiny random models.

Every training run in the report used ``--dtype bfloat16 --eval-dtype float32``: the frozen
encoder and LLM run in bf16 while training and in fp32 for dev and val scoring (Section 3.4).
Checked:
(a) after bf16 training, dev and val are scored in fp32, the training model is freed before
    the evaluation model is built, caches that depend on the dtype (prefix KV cache, fp32
    copy of the output layer) are rebuilt, and the val predictions equal
    exactly those of a separate fp32 evaluation (cmr-stage-a) of the trained bridge
(b) --eval-only with --dtype bfloat16 --eval-dtype float32 == a pure fp32 eval-only run
(c) resuming refuses to mix predictions scored with the models in different dtypes
"""

from __future__ import annotations

import copy
import gc
import json
import weakref
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.slow
pytest.importorskip("transformers")

from cmr.models import head as head_mod  # noqa: E402
from cmr.models.base import Query  # noqa: E402
from cmr.models.bridge import FrozenBridgeVLM  # noqa: E402
from tests.test_bridge import img, tiny_checkpoints, tiny_model, write_split  # noqa: E402, F401

TEXT = "Pick one:\nA. x\nB. y\nC. z\nD. w"


def tower_dtypes(m: FrozenBridgeVLM) -> set[Any]:
    return {p.dtype for t in (m.vision, m.llm) for p in t.parameters()}


def frozen_state(m: FrozenBridgeVLM) -> dict[str, Any]:
    return {k: v.clone() for t in ("vision", "llm") for k, v in getattr(m, t).state_dict().items()}


def bf16_model() -> FrozenBridgeVLM:
    m = tiny_model(seed=3)
    m.vision.to(torch.bfloat16)
    m.llm.to(torch.bfloat16)
    return m


def test_frozen_dtype_scope_is_exact_and_clears_dtype_caches() -> None:
    m = bf16_model()
    ref = copy.deepcopy(m)  # the same weights upcast the ordinary way
    ref.vision.float()
    ref.llm.float()
    before = frozen_state(m)
    buffers = {id(b) for b in m.llm.buffers()}
    queries = [Query(TEXT, img(1)), Query(TEXT + " ", img(1)), Query(TEXT)]
    m.letter_logprobs(queries[0], 4)  # bf16 prefix KV cache + fp32 head copy
    head = m.llm.lm_head
    assert m._prefix_cache is not None and head_mod._CACHE_ATTR in head.__dict__
    with m.frozen_dtype(torch.float32):
        assert tower_dtypes(m) == {torch.float32}
        assert next(m.projector.parameters()).dtype == torch.float32
        assert m._prefix_cache is None and head_mod._CACHE_ATTR not in head.__dict__
        assert m.describe()["dtype"] == "float32"
        inside = [m.letter_logprobs(q, 4) for q in queries]  # cached, cached, uncached
    assert inside == [ref.letter_logprobs(q, 4) for q in queries]  # exactly fp32
    # back to the training weights, bit for bit; caches of the fp32 scope dropped
    assert tower_dtypes(m) == {torch.bfloat16} and m.describe()["dtype"] == "bfloat16"
    after = frozen_state(m)
    assert all(
        torch.equal(before[k], after[k]) and before[k].dtype == after[k].dtype for k in before
    )
    assert {id(b) for b in m.llm.buffers()} == buffers  # originals restored
    assert m._prefix_cache is None and head_mod._CACHE_ATTR not in head.__dict__
    with m.frozen_dtype(None), m.frozen_dtype(torch.bfloat16):  # no-ops
        assert tower_dtypes(m) == {torch.bfloat16}


def test_dev_dtype_enters_the_resume_hash_only_when_it_differs() -> None:
    from cmr.train.bridge_sft import TrainConfig, config_hash

    m, cfg = bf16_model(), TrainConfig()
    base = config_hash(cfg, m, [])
    assert config_hash(cfg, m, [], dev_dtype=torch.bfloat16) == base
    assert config_hash(cfg, m, [], dev_dtype=torch.float32) != base


def run_b(ckpt: Path, data: Path, out: Path, *extra: str) -> None:
    from cmr.runners import stage_b

    stage_b.main(
        [
            "--bridge", str(ckpt / "spec.json"), "--aokvqa-dir", str(data / "aokvqa"),
            "--coco-dir", str(data / "coco"), "--threads", "1", "--out", str(out), *extra,
        ]
    )  # fmt: skip


@pytest.fixture
def data(tmp_path: Path) -> Path:
    (tmp_path / "aokvqa").mkdir()
    write_split(tmp_path / "aokvqa", tmp_path / "coco", "train", 40)
    write_split(tmp_path / "aokvqa", tmp_path / "coco", "val", 4)
    return tmp_path


def test_bf16_training_is_evaluated_in_fp32(
    tiny_checkpoints: Path,  # noqa: F811
    data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cmr.runners import stage_a, stage_b
    from cmr.train import bridge_sft

    seen: dict[str, list[Any]] = {"train": [], "dev": [], "val": []}
    trained: list[weakref.ref[FrozenBridgeVLM]] = []

    def spy(name: str, fn: Any, record: Any) -> Any:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            seen[name].append(record(*args, **kwargs))
            return fn(*args, **kwargs)

        return wrapped

    def train_bridge(model: FrozenBridgeVLM, *args: Any, **kwargs: Any) -> Any:
        trained.append(weakref.ref(model))
        return real_train(model, *args, **kwargs)

    def at_val(items: Any, adapter: Any, *args: Any) -> dict[str, Any]:
        m = adapter.model
        gc.collect()
        return {
            "dtypes": tower_dtypes(m),
            "projector": next(m.projector.parameters()).dtype,
            "is_training_model": m is trained[0](),
            "training_model_alive": trained[0]() is not None,
            "kv": m._prefix_cache,
            "head_copy": head_mod._CACHE_ATTR in m.llm.lm_head.__dict__,
        }

    real_train = stage_b.train_bridge
    monkeypatch.setattr(stage_b, "train_bridge", train_bridge)
    monkeypatch.setattr(
        FrozenBridgeVLM,
        "loss_parts",
        spy("train", FrozenBridgeVLM.loss_parts, lambda m, *a, **k: tower_dtypes(m)),
    )
    monkeypatch.setattr(
        bridge_sft,
        "dev_metrics",
        spy(
            "dev",
            bridge_sft.dev_metrics,
            lambda m, *a, **k: (tower_dtypes(m), m._prefix_cache),
        ),
    )
    monkeypatch.setattr(stage_a, "run_stage_a", spy("val", stage_a.run_stage_a, at_val))
    out = data / "b"
    conds = "real,none,swapped,blank"
    run_b(tiny_checkpoints, data, out, "--target", "answer_only", "--dtype", "bfloat16",
          "--eval-dtype", "float32", "--batch-size", "4", "--grad-accum", "1",
          "--train-limit", "12", "--eval-every", "2", "--select", "last", "--rotations", "2",
          "--conditions", conds)  # fmt: skip

    bf16, fp32 = {torch.bfloat16}, {torch.float32}
    # loss passes: steps 1, 2 (bf16), dev teacher-forced CE (fp32), step 3, dev again
    assert seen["train"] == [bf16, bf16, fp32, bf16, fp32]
    assert len(seen["dev"]) == 2  # step 2 (mid-run: training continued in bf16) and 3
    assert all(d == fp32 and kv is None for d, kv in seen["dev"])
    (val,) = seen["val"]
    assert val["dtypes"] == fp32 and val["projector"] == torch.float32
    assert not val["is_training_model"] and not val["training_model_alive"]  # one copy
    assert val["kv"] is None and not val["head_copy"]  # nothing left from bf16
    run_args = json.loads((out / "run_args.json").read_text())
    assert (run_args["train_dtype"], run_args["eval_dtype"]) == ("bfloat16", "float32")
    mi = run_args["model_input"]
    assert (mi["dtype"], mi["train_dtype"], mi["eval_dtype"]) == ("float32", "bfloat16", "float32")
    log = [json.loads(r) for r in (out / "train_log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "dev_acc" in r] == [2, 3]

    # the val predictions == a separate fp32 cmr-stage-a run of the trained projector, exactly
    monkeypatch.undo()
    ref = data / "a"
    stage_a.main(
        [
            "--model", f"bridge:{tiny_checkpoints / 'spec.json'}", "--projector",
            str(out / "projector.pt"), "--dtype", "float32", "--threads", "1",
            "--aokvqa-dir", str(data / "aokvqa"), "--coco-dir", str(data / "coco"),
            "--conditions", conds, "--rotations", "2", "--out", str(ref),
        ]
    )  # fmt: skip
    got = stage_a.load_records(out / "eval_val" / "predictions.jsonl")
    want = stage_a.load_records(ref / "predictions.jsonl")
    assert got == want and len(want) == 32


def test_eval_only_in_bf16_mode_equals_pure_fp32(
    tiny_checkpoints: Path,  # noqa: F811
    data: Path,
) -> None:
    args = ["--eval-only", "--target", "answer_then_rationale", "--rotations", "2",
            "--conditions", "real,none,swapped,blank"]  # fmt: skip
    run_b(tiny_checkpoints, data, data / "mixed", *args, "--dtype", "bfloat16",
          "--eval-dtype", "float32")  # fmt: skip
    run_b(tiny_checkpoints, data, data / "fp32", *args, "--dtype", "float32")
    mixed = (data / "mixed" / "eval_val" / "predictions.jsonl").read_text()
    assert mixed == (data / "fp32" / "eval_val" / "predictions.jsonl").read_text()
    assert len(mixed.splitlines()) == 4 * 4 * 2
    run_args = json.loads((data / "mixed" / "run_args.json").read_text())
    assert run_args["train_dtype"] is None and run_args["model_input"]["dtype"] == "float32"


def test_resume_guard_refuses_a_model_dtype_change(
    tiny_checkpoints: Path,  # noqa: F811
    data: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cmr.runners.stage_a import check_resume_scoring, scoring_settings

    out = data / "b"
    args = ["--eval-only", "--conditions", "none", "--val-limit", "2"]
    run_b(tiny_checkpoints, data, out, *args, "--dtype", "bfloat16", "--eval-dtype", "bfloat16")
    preds, run_args = out / "eval_val" / "predictions.jsonl", out / "run_args.json"
    before = (preds.read_text(), run_args.read_text())
    with pytest.raises(SystemExit, match="would mix scoring paths") as err:
        run_b(tiny_checkpoints, data, out, *args, "--dtype", "bfloat16", "--eval-dtype", "float32")
    assert "'model_dtype': 'bfloat16'" in str(err.value) and "'model_dtype': 'float32'" in str(
        err.value
    )
    assert (preds.read_text(), run_args.read_text()) == before  # untouched
    same = ["--dtype", "bfloat16", "--eval-dtype", "bfloat16", "--val-limit", "3"]
    run_b(tiny_checkpoints, data, out, *args, *same)  # same dtype: resumes
    assert len(preds.read_text().splitlines()) == 3
    with pytest.raises(SystemExit) as narrow:
        run_b(tiny_checkpoints, data, out, *args, "--dtype", "float32", "--eval-dtype", "bfloat16")
    assert narrow.value.code == 2 and "narrower than --dtype" in capsys.readouterr().err

    # older run records (cmr-stage-a config.json, cmr-stage-b run_args.json) store the dtype
    # only as the CLI --dtype
    new = {"model_input": {"head_dtype": "fp32", "prefix_cache": True, "dtype": "float32"}}
    old_input = {"head_dtype": "fp32", "prefix_cache": True}
    stage_a_old = {"model_input": old_input, "meta": {"dtype": "float32"}}  # Stage A config.json
    stage_b_old = {"model_input": old_input, "dtype": "bfloat16"}  # run_args.json
    assert scoring_settings(stage_a_old) == scoring_settings(new)
    assert scoring_settings(stage_b_old)["model_dtype"] == "bfloat16"
    check_resume_scoring(preds, stage_a_old, new, out)  # an old Stage A run still resumes
    with pytest.raises(SystemExit, match="would mix scoring paths"):
        check_resume_scoring(preds, stage_b_old, new, out)
