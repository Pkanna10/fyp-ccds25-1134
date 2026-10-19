"""The direct_answer training target (DA, Section 4.11): a free-text answer instead of a letter.

Model-free: the target (most frequent direct answer, ties to the first given, else the
correct option's text) and the prompt (identical to the DA evaluation prompt). With torch
(tiny models from tests/test_bridge.py): settings that need an answer letter are refused,
items come in the same order as for the letter targets, CPU-GPU batch splitting
(``cpu_share``) gives the same result, and (slow) a tiny cmr-stage-b run trains and then
evaluates by generating answers.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.conditions import ContextSource, ImageCondition
from cmr.eval.direct_answer import SHORT_ANSWER_INSTRUCTION, run_generation, short_answer_prompt
from cmr.fsutil import read_jsonl
from cmr.models.base import Query
from cmr.train.targets import (
    INSTRUCTIONS,
    LETTER_FORMATS,
    TargetFormat,
    answer_char_span,
    direct_answer_target,
    prompt_for,
    target_for,
)
from tests.conftest import colour_image

try:  # the torch tests below need the tiny models of tests/test_bridge.py
    import torch
    import transformers  # noqa: F401
except ImportError:
    HAVE_TORCH = False
else:
    HAVE_TORCH = True
    from tests.test_bridge import (  # noqa: F401  (tiny_checkpoints: a fixture)
        loader,
        make_items,
        tiny_checkpoints,
        tiny_model,
        write_split,
    )
needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="needs torch and transformers")

DA = TargetFormat.DIRECT_ANSWER

# "rain" and "raining" both have 4 votes and "rain" is given first, so the target is "rain".
ANSWERS = ("rain", "raining", "raining", "rain", "weather", "rain", "raining", "wet", "raining")
ANSWERS = (*ANSWERS, "rain")  # 10 answers: rain 4, raining 4, weather 1, wet 1
UMBRELLA = AOKVQAItem(
    question_id="ex1-umbrella",
    image_id=1,
    question="Why is the man holding the umbrella?",
    choices=("it is raining", "to block the sun", "he is selling it", "it is a costume"),
    correct_choice_idx=0,
    direct_answers=ANSWERS,
    difficult_direct_answer=False,
    rationales=("The pavement is wet and people open umbrellas to stay dry in the rain.",),
    split="val",
)
UMBRELLA_PROMPT = (
    "Question: Why is the man holding the umbrella?\n"
    "Answer the question using a single word or phrase."
)


# ----------------------------------------------------------------------------- model-free


def test_target_is_the_most_frequent_direct_answer() -> None:
    rng = random.Random(1)  # the trainer's rationale stream for seed 0
    before = rng.getstate()
    assert target_for(UMBRELLA, DA, rng) == "rain"  # 4-4 tie: "rain" was given first
    assert rng.getstate() == before  # deterministic: no draw from the rationale stream
    assert direct_answer_target(UMBRELLA) == "rain"
    # the tie goes the other way when "raining" is given first
    flipped = replace(UMBRELLA, direct_answers=(*UMBRELLA.direct_answers[1:], "rain"))
    assert flipped.direct_answers[0] == "raining" and target_for(flipped, DA, rng) == "raining"
    # a clear majority wins wherever it appears; surrounding whitespace is ignored
    majority = replace(UMBRELLA, direct_answers=("wet", " sun", "sun ", "sun", "wet"))
    assert target_for(majority, DA, rng) == "sun"
    # no direct answers: the correct option's text; neither: refused
    assert target_for(replace(UMBRELLA, direct_answers=()), DA, rng) == "it is raining"
    with pytest.raises(ValueError, match="no direct answers and no label"):
        target_for(replace(UMBRELLA, direct_answers=(), correct_choice_idx=None), DA, rng)
    # the letter targets are unchanged
    assert target_for(UMBRELLA, TargetFormat.ANSWER_ONLY, rng) == "A"


def test_training_prompt_is_the_da_evaluation_prompt(tmp_path: Path) -> None:
    """Training (prompt_for) and evaluation (run_generation) send the same text."""
    assert prompt_for(UMBRELLA, DA) == UMBRELLA_PROMPT
    assert prompt_for(UMBRELLA, DA) == short_answer_prompt(UMBRELLA.question)
    assert INSTRUCTIONS[DA] == SHORT_ANSWER_INSTRUCTION
    assert "Options:" not in UMBRELLA_PROMPT
    assert not any(c in UMBRELLA_PROMPT for c in UMBRELLA.choices)  # no options shown

    class Recorder:
        """A generator that records the prompts the DA evaluation sends."""

        name = "recorder"

        def __init__(self) -> None:
            self.prompts: list[str] = []

        def generate(self, query: Query, max_new_tokens: int) -> str:
            self.prompts.append(query.text)
            return "rain"

    items = [UMBRELLA, replace(UMBRELLA, question_id="ex2", image_id=2, question="Who?")]
    conditions = (ImageCondition.REAL, ImageCondition.SWAPPED, ImageCondition.NONE)
    gen = Recorder()
    recs = run_generation(
        items, gen, lambda it: colour_image(it.image_id), tmp_path, conditions=conditions,
        contexts=(ContextSource.NONE,),
    )  # fmt: skip
    # every item under every image condition: exactly the training prompt
    assert gen.prompts == [prompt_for(it, DA) for it in items for _ in conditions]
    assert {r["score"] for r in recs} == {1.0}  # "rain": 4 annotators -> min(1, 4/3)


def test_direct_answer_has_no_letter() -> None:
    assert DA not in LETTER_FORMATS
    assert LETTER_FORMATS == (
        TargetFormat.ANSWER_ONLY,
        TargetFormat.ANSWER_THEN_RATIONALE,
        TargetFormat.RATIONALE_THEN_ANSWER,
        TargetFormat.ANSWER_THEN_MISMATCHED_RATIONALE,
    )
    assert TargetFormat("direct_answer") is DA
    with pytest.raises(ValueError, match="no option letter"):
        answer_char_span("rain", DA)


# ----------------------------------------------------------------------------- with torch


def colour_items(n: int = 8) -> list[AOKVQAItem]:
    """tests/test_bridge.make_items with direct answers that depend on the image."""
    return [
        replace(it, direct_answers=(f"colour {it.image_id % 4}",) * 3 + ("other",))
        for it in make_items(n)
    ]


def recording_loader(seen: list[str]) -> Any:
    """tests/test_bridge.loader that also records which item each image is for."""

    def load(it: AOKVQAItem) -> Any:
        seen.append(it.question_id)
        return loader(it)

    return load


@needs_torch
def test_train_config_refuses_what_needs_a_letter() -> None:
    from cmr.train.bridge_sft import TrainConfig

    TrainConfig(target=DA, select="last")  # the supported recipe: default loss, final step
    for kw, what in (
        ({"rationale_weight": 0.3}, "rationale_weight"),
        ({"rationale_weight": 0.0}, "rationale_weight"),
        ({"margin_lambda": 0.5}, "margin_lambda"),
        ({"option_shuffle": True}, "option_shuffle"),
        ({"select": "dev_acc"}, "select"),
        ({"select": "dev_logp"}, "select"),
    ):
        args: dict[str, Any] = {"target": DA, "select": "last", **kw}
        with pytest.raises(ValueError, match=f"direct_answer does not support {what}"):
            TrainConfig(**args)
    # the letter targets accept these settings
    TrainConfig(target=TargetFormat.ANSWER_ONLY, option_shuffle=True, rationale_weight=0.3)


@needs_torch
def test_train_bridge_and_dev_metrics_refuse_dev_scoring(tmp_path: Path) -> None:
    from cmr.train.bridge_sft import TrainConfig, dev_metrics, train_bridge

    m, items = tiny_model(), colour_items(4)
    cfg = TrainConfig(target=DA, select="last", batch_size=2, grad_accum=1)
    with pytest.raises(ValueError, match="no letter-scored dev check"):
        train_bridge(m, items, loader, cfg, tmp_path, dev_items=items[:2])
    assert not (tmp_path / "train_config.json").exists()  # refused before writing anything
    with pytest.raises(ValueError, match="no letter-scored dev metrics"):
        dev_metrics(m, items, loader, DA)


@needs_torch
def test_same_items_order_and_rng_streams_as_the_letter_arms(tmp_path: Path) -> None:
    from cmr.train.bridge_sft import TrainConfig, train_bridge

    items = colour_items(8)
    common: dict[str, Any] = dict(
        epochs=2, batch_size=2, grad_accum=2, lr=1e-2, eval_every=100, seed=3, select="last"
    )
    seen: dict[str, list[str]] = {"ao": [], "da": []}
    for name, fmt in (("ao", TargetFormat.ANSWER_ONLY), ("da", DA)):
        train_bridge(
            tiny_model(seed=1), items, recording_loader(seen[name]),
            TrainConfig(target=fmt, **common), tmp_path / name,
        )  # fmt: skip
    assert len(seen["da"]) == 16 and seen["da"] == seen["ao"]  # same items, same order
    states = {
        n: torch.load(tmp_path / n / "resume.pt", map_location="cpu", weights_only=True)
        for n in seen
    }

    def next_draw(state: Any) -> float:
        r = random.Random()
        r.setstate(state)
        return r.random()

    assert next_draw(states["da"]["rng_order"]) == next_draw(states["ao"]["rng_order"])
    # neither target draws from the rationale stream: it is still at random.Random(seed + 1)
    assert next_draw(states["da"]["rng_target"]) == random.Random(3 + 1).random()
    rows = read_jsonl(tmp_path / "da" / "train_log.jsonl")
    assert len(rows) == 4 and rows[-1]["selected"]
    for row in rows:  # the default per-token mean loss; no letter split, no dev metrics
        assert row["ce"] == row["ce_all"] and math.isfinite(row["ce"])
        assert not {"ce_answer", "ce_rest", "dev_acc"} & set(row)
    cfg = json.loads((tmp_path / "da" / "train_config.json").read_text())
    assert cfg["target"] == "direct_answer" and cfg["select"] == "last"
    assert (tmp_path / "da" / "projector.pt").exists()


@needs_torch
def test_direct_answer_training_reduces_the_loss(tmp_path: Path) -> None:
    from cmr.train.bridge_sft import TrainConfig, train_bridge

    cfg = TrainConfig(
        target=DA, epochs=12, batch_size=4, grad_accum=1, lr=3e-2, eval_every=10, select="last"
    )
    hist = train_bridge(tiny_model(seed=1), colour_items(8), loader, cfg, tmp_path)["history"]
    assert hist[-1]["ce"] < hist[0]["ce"]


@needs_torch
def test_cpu_cotraining_reproduces_the_plain_loop_for_direct_answer(tmp_path: Path) -> None:
    from cmr.train.bridge_sft import TrainConfig, train_bridge

    n = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        items = colour_items(8)
        common: dict[str, Any] = dict(
            target=DA, epochs=2, batch_size=1, grad_accum=4, lr=1e-2, eval_every=100,
            select="last",
        )  # fmt: skip
        plain = tiny_model(seed=1)
        r1 = train_bridge(plain, items, loader, TrainConfig(**common), tmp_path / "plain")
        # cpu_share=2: 2 of the 4 examples of each step run on the CPU copy (cmr.train.pilot)
        pilot, copy = tiny_model(seed=1), tiny_model(seed=1)
        r2 = train_bridge(
            pilot, items, loader, TrainConfig(**common, cpu_share=2), tmp_path / "pilot",
            cpu_model=copy, cpu_threads=1,
        )  # fmt: skip
    finally:
        torch.set_num_threads(n)
    assert len(r1["history"]) == len(r2["history"]) == 4
    for a, b in zip(r1["history"], r2["history"], strict=True):
        assert a["ce"] == pytest.approx(b["ce"], rel=1e-5)
        assert {"sec_step", "sec_gpu", "sec_cpu"} <= set(b)
        assert "ce_answer" not in a and "ce_answer" not in b
    for a, b in zip(plain.projector.parameters(), pilot.projector.parameters(), strict=True):
        assert torch.allclose(a, b, rtol=1e-5, atol=1e-6)


@needs_torch
@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rotations", "4"], "--rotations must be 1"),
        (["--prior-run", "runs/none"], "--prior-run"),
        (["--option-shuffle"], "option_shuffle"),
        (["--rationale-weight", "0.3"], "rationale_weight"),
        (["--margin-lambda", "0.1"], "margin_lambda"),
        (["--select", "dev_acc"], "select other than 'last'"),
    ],
)
def test_stage_b_cli_refuses_letter_settings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    from cmr.runners import stage_b

    args = [
        "--bridge", "missing.json", "--target", "direct_answer", "--aokvqa-dir", "missing",
        "--coco-dir", "missing", "--select", "last", "--out", str(tmp_path / "run"), *extra,
    ]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        stage_b.main(args)
    assert exc.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "run").exists()  # refused before any data, model or output


@needs_torch
@pytest.mark.slow
def test_stage_b_cli_trains_and_evaluates_direct_answer(
    tiny_checkpoints: Path,
    tmp_path: Path,
) -> None:
    from cmr.runners import stage_b

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 40)
    write_split(aokvqa, coco, "val", 6)
    out = tmp_path / "run"
    args = [
        "--bridge", str(tiny_checkpoints / "spec.json"), "--target", "direct_answer",
        "--aokvqa-dir", str(aokvqa), "--coco-dir", str(coco), "--batch-size", "4",
        "--grad-accum", "1", "--select", "last", "--conditions", "real,swapped",
        "--out", str(out),
    ]  # fmt: skip
    stage_b.main(args)
    assert (out / "projector.pt").exists()
    rows = read_jsonl(out / "train_log.jsonl")
    assert rows and all("dev_acc" not in r and r["ce"] == r["ce_all"] for r in rows)
    # evaluation: the cmr-transfer aokvqa-da generation path, not letter scoring
    summary = json.loads((out / "eval_val" / "summary.json").read_text())
    assert summary["metric"] == "A-OKVQA DA" and summary["n_questions"] == 6
    assert {(r["condition"], r["context"]) for r in summary["rows"]} == {
        ("real", "none"),
        ("swapped", "none"),
    }
    assert (out / "eval_val" / "summary.md").read_text().startswith("# A-OKVQA DA")
    recs = read_jsonl(out / "eval_val" / "predictions.jsonl")
    assert len(recs) == 12 and all({"raw", "answer", "score", "score_norm"} <= set(r) for r in recs)
    assert "rotation" not in recs[0]  # generation records, not letter-scoring records
    # a re-run of the finished job trains nothing and generates nothing new
    stage_b.main([*args, "--resume-if-present"])
    assert len(read_jsonl(out / "eval_val" / "predictions.jsonl")) == 12
    run_args = json.loads((out / "run_args.json").read_text())
    assert run_args["target"] == "direct_answer" and len(run_args["manifest"]["segments"]) == 2
