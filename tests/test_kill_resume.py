"""A run killed at any point and re-run with --resume-if-present matches an uninterrupted run.

Cloud jobs can be pre-empted and resumed (five were; Appendix C). Kills are raised from a
chosen call (training forward pass, dev or val scoring, a file write); the re-run uses a fresh
model, as a new process would. Covers train_bridge, cmr-stage-b, repair of a torn last line
of a JSONL file, atomic file writes and a real SIGKILL of an evaluation subprocess
(cmr.runners.stage_a).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from cmr.fsutil import read_jsonl, write_text_atomic  # noqa: E402
from cmr.models.bridge import FrozenBridgeVLM  # noqa: E402
from cmr.train import bridge_sft  # noqa: E402
from cmr.train.bridge_sft import RESUME_FILE, TrainConfig, train_bridge  # noqa: E402
from cmr.train.targets import TargetFormat  # noqa: E402
from tests.test_bridge import (  # noqa: E402
    loader,
    make_items,
    tiny_checkpoints,  # noqa: F401  (fixture)
    tiny_model,
    write_split,
)


class Kill(BaseException):
    """A SIGKILL stand-in (BaseException: no ``except Exception`` handler can swallow it)."""


@pytest.fixture(autouse=True)
def one_thread() -> Any:
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


class CallCounter:
    """Wrap ``cls.name``: count calls, raise :class:`Kill` on call number ``kill_at``."""

    def __init__(self, mp: pytest.MonkeyPatch, cls: Any, name: str) -> None:
        self.n, self.kill_at = 0, 0
        real = getattr(cls, name)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self.n += 1
            if self.n == self.kill_at:
                raise Kill(f"{name} call {self.n}")
            return real(*args, **kwargs)

        mp.setattr(cls, name, wrapped)

    def arm(self, kill_at: int = 0) -> None:
        self.n, self.kill_at = 0, kill_at


# ------------------------------------------------------------------ JSONL + atomic


def test_unterminated_last_line_is_dropped_and_cut_off_on_resume(tmp_path: Path) -> None:
    from cmr.runners.stage_a import load_records

    p = tmp_path / "predictions.jsonl"
    good = [{"question_id": "q1", "v": 1.5}, {"question_id": "q2", "v": -2.25}]
    body = "".join(json.dumps(r) + "\n" for r in good)
    for tail in ('{"question_id": "q3", "v": 0.', '{"question_id": "q3"', "   ", "[1, 2]", '"x"'):
        p.write_text(body + tail)
        assert load_records(p) == good  # read-only: dropped, file untouched
        assert p.read_text() == body + tail
        assert load_records(p, repair=True) == good  # resume: cut off
        assert p.read_text() == body
    p.write_text(body + "{not json\n" + json.dumps(good[0]) + "\n")
    with pytest.raises(ValueError, match=r":3: corrupt record"):
        load_records(p)
    assert load_records(tmp_path / "missing.jsonl") == []
    # a multi-byte character cut in half is part of the dropped fragment
    p.write_bytes(body.encode() + '{"question_id": "é'.encode()[:-1])
    assert read_jsonl(p, repair=True) == good and p.read_text() == body


def test_whole_unterminated_last_record_is_kept_and_terminated_on_resume(
    tmp_path: Path,
) -> None:
    """A whole last record missing only its newline is kept, and terminated on resume."""
    from cmr.runners.stage_a import load_records

    p = tmp_path / "predictions.jsonl"
    good = [{"question_id": "q1", "v": 1.5}, {"question_id": "q2", "v": "é"}]
    body = json.dumps(good[0]) + "\n" + json.dumps(good[1], ensure_ascii=False)
    for tail in ("", "  "):  # trailing blanks after the object are not a torn record
        p.write_text(body + tail)
        assert load_records(p) == good  # read-only: kept, file untouched
        assert p.read_text() == body + tail
        assert load_records(p, repair=True) == good  # resume: terminated
        assert p.read_text() == body + tail + "\n"
        with p.open("a") as f:
            f.write(json.dumps({"question_id": "q3"}) + "\n")
        assert [r["question_id"] for r in load_records(p)] == ["q1", "q2", "q3"]
    p.write_text(json.dumps(good[0]))  # a one-record file
    assert read_jsonl(p, repair=True) == [good[0]] and p.read_text().endswith("}\n")


_SIGKILL_CHILD = """
import os, signal, sys
from pathlib import Path
from cmr.data.aokvqa import load_aokvqa
from cmr.eval.conditions import ImageCondition
from cmr.models.fake import PriorOnlyModel
from cmr.runners.stage_a import RunSpec, run_stage_a

aokvqa, out, kill_at, pad = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])

class Dies(PriorOnlyModel):
    n = 0
    last_extras = {"pad": "x" * pad}

    def letter_logprobs(self, query, n_options):
        Dies.n += 1
        if Dies.n == kill_at:  # a real SIGKILL: no unwinding, buffers are NOT flushed
            os.kill(os.getpid(), signal.SIGKILL)
        return super().letter_logprobs(query, n_options)

items = [it for it in load_aokvqa(aokvqa, "val") if it.has_labels]
run_stage_a(items, Dies(), None, RunSpec(conditions=(ImageCondition.NONE,), rotations=4), out)
"""


def test_real_sigkill_mid_buffer_loses_at_most_the_current_item(
    tmp_path: Path, aokvqa_dir: Path
) -> None:
    """A real SIGKILL keeps every earlier item, loses at most the current one, resumes exactly."""
    import os
    import signal
    import subprocess
    import sys

    import cmr
    from cmr.data.aokvqa import load_aokvqa
    from cmr.eval.conditions import ImageCondition
    from cmr.models.fake import PriorOnlyModel
    from cmr.runners.stage_a import RunSpec, run_stage_a

    per_item = 4  # one condition x one context x 4 rotations; flushed once per item
    spec = RunSpec(conditions=(ImageCondition.NONE,), rotations=per_item)
    items = [it for it in load_aokvqa(aokvqa_dir, "val") if it.has_labels]
    child = tmp_path / "child.py"
    child.write_text(_SIGKILL_CHILD)
    src = str(Path(cmr.__file__).resolve().parents[1])
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")])}
    partial_item_seen = lost_seen = 0

    class Padded(PriorOnlyModel):
        """The child's model without the kill; counts its scoring calls."""

        def __init__(self, pad: int) -> None:
            self.last_extras = {"pad": "x" * pad}
            self.calls = 0

        def letter_logprobs(self, query: Any, n_options: int) -> list[float]:
            self.calls += 1
            return super().letter_logprobs(query, n_options)

    # pad 0: a whole item fits the 8 KiB write buffer; pad 3000/5000: the buffer overflows
    # inside an item, so part of the current item is on disk at the kill
    for kill_at, pad in ((1, 0), (11, 0), (11, 3000), (18, 3000), (23, 5000), (30, 5000)):
        ref_dir, out = tmp_path / f"ref_{kill_at}_{pad}", tmp_path / f"killed_{kill_at}_{pad}"
        ref = run_stage_a(items, Padded(pad), None, spec, ref_dir)
        want = (ref_dir / "predictions.jsonl").read_bytes()
        proc = subprocess.run(
            [sys.executable, str(child), str(aokvqa_dir), str(out), str(kill_at), str(pad)],
            env=env,
            capture_output=True,
            timeout=120,
        )
        assert proc.returncode == -signal.SIGKILL, proc.stderr.decode()
        data = (out / "predictions.jsonl").read_bytes()
        assert want.startswith(data)  # whole records of the uninterrupted run (+ a prefix)
        complete = data[: data.rfind(b"\n") + 1].count(b"\n")
        scored = kill_at - 1  # records computed before the kill
        assert scored - scored % per_item <= complete <= scored  # every earlier item kept
        assert scored - complete < per_item  # lost: at most the current item's records
        partial_item_seen += complete > scored - scored % per_item
        lost_seen += complete < scored
        # read-only: a torn tail is dropped (a whole record missing its newline is kept)
        on_disk = len(read_jsonl(out / "predictions.jsonl"))
        assert complete <= on_disk <= complete + 1
        model = Padded(pad)
        assert run_stage_a(items, model, None, spec, out) == ref
        assert model.calls == len(ref) - on_disk  # nothing scored twice
        assert (out / "predictions.jsonl").read_bytes() == want
    assert partial_item_seen and lost_seen  # a kill really cut an item mid-buffer


def test_atomic_write_keeps_the_old_file_when_killed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cmr.fsutil as fsutil

    target = tmp_path / "run_args.json"
    write_text_atomic(target, '{"old": 1}')

    def die(*args: Any) -> None:
        raise Kill("killed before the rename")

    monkeypatch.setattr(fsutil.os, "replace", die)
    with pytest.raises(Kill):
        write_text_atomic(target, '{"new": 2, "padding": "' + "x" * 10_000 + '"}')
    assert json.loads(target.read_text()) == {"old": 1}


def test_stage_a_val_resume_after_a_torn_line_equals_an_uninterrupted_run(
    tmp_path: Path, items: list[Any]
) -> None:
    from cmr.eval.conditions import ImageCondition
    from cmr.models.fake import PriorOnlyModel
    from cmr.runners.stage_a import RunSpec, run_stage_a
    from tests.conftest import colour_image

    spec = RunSpec(conditions=(ImageCondition.REAL, ImageCondition.NONE), rotations=2)

    def load(it: Any) -> Any:
        return colour_image(it.image_id)

    ref = run_stage_a(items, PriorOnlyModel(), load, spec, tmp_path / "ref")

    class Dies(PriorOnlyModel):
        n = 0

        def letter_logprobs(self, query: Any, n_options: int) -> list[float]:
            Dies.n += 1
            if Dies.n == 30:
                raise Kill("killed")
            return super().letter_logprobs(query, n_options)

    out = tmp_path / "killed"
    with pytest.raises(Kill):
        run_stage_a(items, Dies(), load, spec, out)
    pred = out / "predictions.jsonl"
    n_before = len(pred.read_text().splitlines())
    assert 0 < n_before < len(ref)
    with pred.open("a") as f:  # the kill also tore the next record mid-write
        f.write('{"question_id": "q0')
    calls = {"n": 0}

    class Counting(PriorOnlyModel):
        def letter_logprobs(self, query: Any, n_options: int) -> list[float]:
            calls["n"] += 1
            return super().letter_logprobs(query, n_options)

    res = run_stage_a(items, Counting(), load, spec, out)
    assert calls["n"] == len(ref) - n_before  # nothing scored twice
    assert pred.read_text() == (tmp_path / "ref" / "predictions.jsonl").read_text()
    assert res == ref
    pred.write_text(pred.read_text() + pred.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate record keys"):
        run_stage_a(items, Counting(), load, spec, out)


# ------------------------------------------------------------------ train_bridge


CFG = TrainConfig(
    target=TargetFormat.ANSWER_ONLY, epochs=2, batch_size=2, grad_accum=1, lr=3e-2,
    eval_every=3, select="last",
)  # fmt: skip  # 6 items x 2 epochs / 2 -> 6 steps; dev at steps 3 and 6 (final)


def _outputs(out: Path) -> dict[str, Any]:
    state = torch.load(out / RESUME_FILE, weights_only=True)
    return {
        "log": (out / "train_log.jsonl").read_text(),
        "trainable": torch.load(out / "trainable.pt", weights_only=True),
        "projector": torch.load(out / "projector.pt", weights_only=True),
        "resume_trainable": state["trainable"],
        "history": state["history"],
        "step": state["step"],
        "pending": state.get("pending_row"),
    }


def _same(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert a["log"] == b["log"] and a["history"] == b["history"]
    assert a["step"] == b["step"] and a["pending"] is b["pending"] is None
    for k in ("trainable", "projector", "resume_trainable"):
        assert a[k].keys() == b[k].keys()
        assert all(torch.equal(a[k][n], b[k][n]) for n in a[k]), k


@pytest.mark.parametrize("fmt", [TargetFormat.ANSWER_ONLY, TargetFormat.RATIONALE_THEN_ANSWER])
@pytest.mark.parametrize("phase", ["training", "intermediate_dev", "final_dev"])
def test_train_bridge_killed_anywhere_resumes_to_identical_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, fmt: TargetFormat
) -> None:
    from cmr.train import generate

    cfg = TrainConfig(**{**CFG.__dict__, "target": fmt})
    items, dev = make_items(6), make_items(3)
    ref = train_bridge(tiny_model(seed=1), items, loader, cfg, tmp_path / "ref", dev, 2)
    want = _outputs(tmp_path / "ref")
    assert [r["step"] for r in ref["history"] if "dev_acc" in r] == [3, 6]

    # dev scoring = one call per dev item; training = one loss_parts per step (+1 per
    # teacher-forced dev batch: 3 dev items fit one batch)
    score_name = "rationale_then_letter_logprobs" if fmt.value.startswith("rationale") else None
    if score_name:
        scorer = CallCounter(monkeypatch, generate, score_name)
    else:
        scorer = CallCounter(monkeypatch, FrozenBridgeVLM, "letter_logprobs")
    fwd = CallCounter(monkeypatch, FrozenBridgeVLM, "loss_parts")
    out = tmp_path / "killed"
    if phase == "training":  # in step 5: resume.pt holds step 4 (checkpoint every 2)
        fwd.arm(kill_at=5 + 1)  # 4 steps + the step-3 dev batch, then step 5
    else:  # in the 2nd dev item of the step-3 / step-6 dev pass
        scorer.arm(kill_at=2 if phase == "intermediate_dev" else 3 + 2)
    with pytest.raises(Kill):
        train_bridge(tiny_model(seed=1), items, loader, cfg, out, dev, 2)
    saved = torch.load(out / RESUME_FILE, weights_only=True)
    if phase == "training":
        assert saved["step"] == 4 and saved["pending_row"] is None
    else:
        step = 3 if phase == "intermediate_dev" else 6
        # the training state of that step is on disk before its dev pass
        assert saved["step"] == step and saved["pending_row"]["step"] == step
        assert "dev_acc" not in saved["pending_row"]
        cache = list((out / bridge_sft.DEV_CACHE_DIR).iterdir())
        assert len(cache) == 1 and len(cache[0].read_text().splitlines()) == 1  # 1 item done
    if phase == "final_dev":  # select last: the final weights were kept BEFORE the dev
        on_disk = torch.load(out / "trainable.pt", weights_only=True)
        assert all(torch.equal(on_disk[k], want["trainable"][k]) for k in on_disk)
        assert saved["best_state"] is not None

    scorer.arm()
    fwd.arm()
    res = train_bridge(tiny_model(seed=1), items, loader, cfg, out, dev, 2, "if_present")
    assert res["history"] == ref["history"] and res["best_dev_acc"] == ref["best_dev_acc"]
    _same(_outputs(out), want)
    assert not list((out / bridge_sft.DEV_CACHE_DIR).iterdir())  # caches are removed
    if phase == "final_dev":  # only the 2 dev items left were scored; no training step
        assert scorer.n == 2 and fwd.n == 1  # the teacher-forced dev batch
    if phase == "intermediate_dev":  # 2 items left at step 3 + 3 at step 6
        assert scorer.n == 2 + 3

    # a re-run of the finished run trains nothing and evaluates nothing
    scorer.arm()
    fwd.arm()
    again = train_bridge(tiny_model(seed=1), items, loader, cfg, out, dev, 2, "if_present")
    assert again["resumed_from"] == 6 and again["history"] == ref["history"]
    assert scorer.n == fwd.n == 0
    _same(_outputs(out), want)


def test_kill_while_writing_resume_pt_replays_from_the_previous_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cmr.fsutil as fsutil

    items, dev = make_items(6), make_items(3)
    train_bridge(tiny_model(seed=1), items, loader, CFG, tmp_path / "ref", dev, 2)
    real, writes = fsutil.os.replace, {"n": 0}

    def replace(src: Any, dst: Any) -> None:  # the 4th resume.pt write dies mid-write
        if Path(dst).name == RESUME_FILE:
            writes["n"] += 1
            if writes["n"] == 4:
                raise Kill("killed while writing resume.pt")
        real(src, dst)

    monkeypatch.setattr(fsutil.os, "replace", replace)
    out = tmp_path / "killed"
    with pytest.raises(Kill):
        train_bridge(tiny_model(seed=1), items, loader, CFG, out, dev, 2)
    monkeypatch.setattr(fsutil.os, "replace", real)
    # writes: step 2, step 3 pre-dev, step 3 post-dev, [step 4 killed]
    assert torch.load(out / RESUME_FILE, weights_only=True)["step"] == 3
    train_bridge(tiny_model(seed=1), items, loader, CFG, out, dev, 2, "if_present")
    _same(_outputs(out), _outputs(tmp_path / "ref"))


def test_dev_cache_of_another_state_is_never_used(tmp_path: Path) -> None:
    items, dev = make_items(6), make_items(3)
    out = tmp_path / "run"
    train_bridge(tiny_model(seed=1), items, loader, CFG, out, dev, 2)
    want = _outputs(out)
    # a stale cache (e.g. from a run that was restarted) with a wrong value
    chash = torch.load(out / RESUME_FILE, weights_only=True)["config_hash"]
    stale = bridge_sft._dev_cache_path(out, chash, 3)
    stale.write_text(json.dumps({"kind": "lp", "i": 0, "qid": dev[0].question_id,
                                 "v": [0.0, -50.0, -50.0, -50.0]}) + "\n")  # fmt: skip
    train_bridge(tiny_model(seed=1), items, loader, CFG, out, dev, 2, "restart")
    _same(_outputs(out), want)


# ------------------------------------------------------------------ cmr-stage-b CLI


def _run_b(ckpt: Path, data: Path, out: Path, *extra: str) -> None:
    from cmr.runners import stage_b

    stage_b.main(
        [
            "--bridge", str(ckpt / "spec.json"), "--aokvqa-dir", str(data / "aokvqa"),
            "--coco-dir", str(data / "coco"), "--threads", "1", "--out", str(out), *extra,
        ]
    )  # fmt: skip


TRAIN_ARGS = [
    "--target", "answer_only", "--dtype", "bfloat16", "--eval-dtype", "float32",
    "--select", "last", "--eval-every", "1000000", "--train-limit", "12", "--epochs", "1",
    "--batch-size", "2", "--grad-accum", "1", "--checkpoint-every", "4",
    "--conditions", "real,swapped", "--rotations", "2", "--resume-if-present",
]  # fmt: skip  # 6 steps, checkpoints at 4 and 6, one dev pass (the final one)


@pytest.fixture
def stage_b_data(tmp_path: Path) -> Path:
    (tmp_path / "aokvqa").mkdir()
    write_split(tmp_path / "aokvqa", tmp_path / "coco", "train", 80)
    write_split(tmp_path / "aokvqa", tmp_path / "coco", "val", 4)
    return tmp_path


def _n_dev(data: Path) -> int:
    from cmr.data.aokvqa import dev_split, load_aokvqa

    return len(dev_split(load_aokvqa(data / "aokvqa", "train"), dev_fraction=0.06, seed=0)[1])


def _final(out: Path) -> dict[str, Any]:
    log, weights = out / "train_log.jsonl", out / "trainable.pt"
    return {
        "summary": json.loads((out / "eval_val" / "summary.json").read_text()),
        "summary_md": (out / "eval_val" / "summary.md").read_text(),
        "preds": (out / "eval_val" / "predictions.jsonl").read_text().splitlines(),
        "train_log": log.read_text() if log.exists() else None,
        "trainable": torch.load(weights, weights_only=True) if weights.exists() else None,
    }


def _assert_identical(got: dict[str, Any], want: dict[str, Any]) -> None:
    assert got["summary"] == want["summary"] and got["summary_md"] == want["summary_md"]
    assert len(set(got["preds"])) == len(got["preds"])  # no duplicates
    assert set(got["preds"]) == set(want["preds"]) and got["preds"] == want["preds"]
    assert got["train_log"] == want["train_log"]
    if want["trainable"] is not None:
        assert all(
            torch.equal(got["trainable"][k], want["trainable"][k]) for k in want["trainable"]
        )


@pytest.mark.slow
@pytest.mark.parametrize("phase", ["training", "final_dev", "val", "val_torn_line"])
def test_stage_b_killed_in_any_phase_reruns_to_identical_outputs(
    tiny_checkpoints: Path,  # noqa: F811
    stage_b_data: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    data, n_dev, n_val = stage_b_data, _n_dev(stage_b_data), 4 * 2 * 2
    assert 2 <= n_dev <= 8  # the teacher-forced dev pass is one batch
    _run_b(tiny_checkpoints, data, data / "ref", *TRAIN_ARGS)
    want = _final(data / "ref")
    assert len(want["preds"]) == n_val

    built: list[torch.dtype] = []
    real_build = FrozenBridgeVLM.from_pretrained.__func__  # type: ignore[attr-defined]

    def from_pretrained(cls: Any, spec: Any, device: str = "cpu", dtype: Any = None) -> Any:
        built.append(dtype)
        return real_build(cls, spec, device, dtype)

    monkeypatch.setattr(FrozenBridgeVLM, "from_pretrained", classmethod(from_pretrained))
    scorer = CallCounter(monkeypatch, FrozenBridgeVLM, "letter_logprobs")
    fwd = CallCounter(monkeypatch, FrozenBridgeVLM, "loss_parts")
    out = data / "killed"
    if phase == "training":
        fwd.arm(kill_at=6)  # step 6 (resume.pt at step 4)
    elif phase == "final_dev":
        scorer.arm(kill_at=2)  # 2nd dev item of the final dev pass
    else:
        scorer.arm(kill_at=n_dev + 5)  # 5th val score
    with pytest.raises(Kill):
        _run_b(tiny_checkpoints, data, out, *TRAIN_ARGS)
    assert built == [torch.bfloat16] + ([torch.float32] if phase.startswith("val") else [])
    saved = torch.load(out / RESUME_FILE, weights_only=True)
    if phase == "final_dev":  # final weights + training state saved before the dev pass
        assert saved["step"] == 6 and saved["pending_row"]["step"] == 6
        on_disk = torch.load(out / "trainable.pt", weights_only=True)
        assert all(torch.equal(on_disk[k], want["trainable"][k]) for k in on_disk)
    preds = out / "eval_val" / "predictions.jsonl"
    if phase == "val_torn_line":
        with preds.open("a") as f:
            f.write('{"question_id": "val1", "condition": "swa')

    built.clear()
    scorer.arm()
    fwd.arm()
    _run_b(tiny_checkpoints, data, out, *TRAIN_ARGS)
    _assert_identical(_final(out), want)
    if phase == "training":
        assert built == [torch.bfloat16, torch.float32]
        assert fwd.n == 2 + 1  # steps 5-6 + the teacher-forced dev batch
    if phase == "final_dev":  # the dev item already scored is not scored again
        assert built == [torch.bfloat16, torch.float32]
        assert scorer.n == (n_dev - 1) + n_val and fwd.n == 1
    if phase.startswith("val"):  # finished training: straight to the fp32 model, no dev
        assert built == [torch.float32] and fwd.n == 0
        assert scorer.n == n_val - 4  # 4 val records were written before the kill
        args = json.loads((out / "run_args.json").read_text())
        assert args["resumed_from_step"] == 6 and args["model_input"]["dtype"] == "float32"

    # once done, a re-run re-scores nothing and rewrites the same summary
    built.clear()
    scorer.arm()
    fwd.arm()
    _run_b(tiny_checkpoints, data, out, *TRAIN_ARGS)
    assert built == [torch.float32] and scorer.n == fwd.n == 0
    _assert_identical(_final(out), want)


@pytest.mark.slow
def test_stage_b_eval_only_killed_mid_val_reruns_to_identical_outputs(
    tiny_checkpoints: Path,  # noqa: F811
    stage_b_data: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ["--eval-only", "--target", "answer_only", "--dtype", "bfloat16",
            "--eval-dtype", "float32", "--conditions", "real,swapped,blank",
            "--rotations", "2"]  # fmt: skip
    data = stage_b_data
    _run_b(tiny_checkpoints, data, data / "ref", *args)
    want = _final(data / "ref")
    scorer = CallCounter(monkeypatch, FrozenBridgeVLM, "letter_logprobs")
    scorer.arm(kill_at=7)
    with pytest.raises(Kill):
        _run_b(tiny_checkpoints, data, data / "killed", *args)
    preds = data / "killed" / "eval_val" / "predictions.jsonl"
    with preds.open("a") as f:
        f.write('{"question_id": "val')
    scorer.arm()
    _run_b(tiny_checkpoints, data, data / "killed", *args)
    assert scorer.n == 4 * 3 * 2 - 6
    _assert_identical(_final(data / "killed"), want)
