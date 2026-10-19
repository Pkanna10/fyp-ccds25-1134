"""A-OKVQA direct-answer (DA) evaluation (cmr.eval.direct_answer; Table 4.16).

Covers the prompt, answer extraction, the official score min(1, n/3), the generation runner
and its summary. Two fake generators give known extremes, as in test_stage_a_e2e.py:
    image-blind generator  -> every image-reliance number must be exactly zero
    image-using generator  -> its DA score drops to zero when the image is removed, swapped
                              or blank
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.conditions import ContextSource, ImageCondition
from cmr.eval.direct_answer import (
    SHORT_ANSWER_INSTRUCTION,
    BridgeGenerator,
    TextGenerator,
    as_generator,
    da_items,
    da_scorer,
    extract_answer,
    generation_markdown,
    run_generation,
    short_answer_prompt,
    summarize_generation,
)
from cmr.models.base import Query
from cmr.models.fake import image_key
from tests.conftest import colour_image

ALL = (ImageCondition.REAL, ImageCondition.NONE, ImageCondition.SWAPPED, ImageCondition.BLANK)


def loader(it: AOKVQAItem) -> Image.Image:
    return colour_image(it.image_id)


class BlindGenerator:
    """Answers from the text alone: right on every third question, else a fixed guess."""

    name = "fake-blind-gen"

    def __init__(self, items: list[AOKVQAItem]) -> None:
        self.by_q = {it.question: it for it in items}

    def generate(self, query: Query, max_new_tokens: int) -> str:
        for q, it in self.by_q.items():
            if q in query.text:
                n = int(it.question_id[1:])
                return f"{it.direct_answers[0]}." if n % 3 == 0 else "wrong guess"
        return "unknown"


class LookupGenerator:
    """Knows the answer only when shown the question's own image."""

    name = "fake-lookup-gen"

    def __init__(self, items: list[AOKVQAItem]) -> None:
        self.answers = {
            (it.question, image_key(colour_image(it.image_id))): it.direct_answers[0]
            for it in items
        }
        self.calls = 0

    def generate(self, query: Query, max_new_tokens: int) -> str:
        self.calls += 1
        for (q, key), ans in self.answers.items():
            if q in query.text and image_key(query.image) == key:
                return f"Answer: {ans.upper()}.\nBecause it is."
        return "no idea"


def rows(summary: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(r["condition"], r["context"]): r for r in summary["rows"]}


# ----------------------------------------------------------------------------- units


def test_short_answer_prompt_layout() -> None:
    assert short_answer_prompt(" What? ") == f"Question: What?\n{SHORT_ANSWER_INSTRUCTION}"
    with_ctx = short_answer_prompt("What?", "A dog runs.")
    assert with_ctx.startswith("Context: A dog runs.\n\nQuestion: What?")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Dog.", "dog"),
        ("  \nA red car\nsecond line", "a red car"),
        ("Answer: Skiing.", "skiing"),
        ("The answer is: two", "two"),
        ('"Blue"', "blue"),
        ("", ""),
    ],
)
def test_extract_answer(raw: str, expected: str) -> None:
    assert extract_answer(raw) == expected


def test_da_scorer_official_vs_normalised(items: list[AOKVQAItem]) -> None:
    it = items[1]  # direct_answers: 6 x "use-1-1", 4 x "other"
    assert da_scorer("use-1-1", it) == {"score": 1.0, "score_norm": 1.0}
    assert da_scorer("Other.", it)["score"] == 1.0  # 4 annotators -> min(1, 4/3)
    dog = AOKVQAItem("x", 1, "q", (), None, ("dog", "dog", "cat"), False, (), "val")
    s = da_scorer("A dog", dog)
    assert s["score"] == 0.0  # official: string equality, "a dog" != "dog"
    assert s["score_norm"] == pytest.approx(2 / 3)  # article stripped -> 2 matches


def test_da_items_drops_difficult(items: list[AOKVQAItem]) -> None:
    kept = da_items(items)
    assert all(not it.difficult_direct_answer for it in kept)
    assert len(kept) == sum(not it.difficult_direct_answer for it in items) < len(items)


def test_as_generator_dispatch() -> None:
    blind = BlindGenerator([])
    assert as_generator(blind) is blind
    assert isinstance(blind, TextGenerator)
    fake_bridge = type("FrozenBridgeVLM", (), {"name": "b"})()
    assert isinstance(as_generator(fake_bridge), BridgeGenerator)
    with pytest.raises(TypeError):
        as_generator(object())


# ----------------------------------------------------------------------------- end to end


def test_image_blind_generator_has_zero_reliance(items: list[AOKVQAItem], tmp_path: Path) -> None:
    da = da_items(items)
    recs = run_generation(da, BlindGenerator(da), loader, tmp_path, conditions=ALL)
    summary = summarize_generation(recs)
    r = rows(summary)
    ref = r[("real", "none")]
    assert 0.0 < ref["score"] < 1.0
    for cond in ("none", "swapped", "blank"):
        assert r[(cond, "none")]["delta_vs_ref"] == 0.0
        assert r[(cond, "none")]["flip_rate"] == 0.0
    assert summary["arbitration"]["blind_agreement"] == 1.0
    assert summary["arbitration"]["score_on_prior_wrong"] == 0.0


def test_image_using_generator_shows_reliance(items: list[AOKVQAItem], tmp_path: Path) -> None:
    da = da_items(items)
    model = LookupGenerator(da)
    recs = run_generation(da, model, loader, tmp_path, conditions=ALL)
    summary = summarize_generation(recs)
    r = rows(summary)
    assert r[("real", "none")]["score"] == 1.0  # extraction undoes "Answer: X." + case
    for cond in ("none", "swapped", "blank"):
        row = r[(cond, "none")]
        assert row["score"] == 0.0
        assert row["delta_vs_ref"] == 1.0 and row["delta_ci"][0] == 1.0
        assert row["flip_rate"] == 1.0
    arb = summary["arbitration"]
    assert arb["n_prior_wrong"] == len(da) and arb["score_on_prior_wrong"] == 1.0
    md = generation_markdown(summary)
    assert "| real | none |" in md and "Arbitration" in md

    # Resumable: a second run makes no new model calls and writes no new lines.
    n_calls = model.calls
    again = run_generation(da, model, loader, tmp_path, conditions=ALL)
    assert model.calls == n_calls and len(again) == len(recs)
    lines = (tmp_path / "predictions.jsonl").read_text().splitlines()
    assert len(lines) == len(da) * len(ALL)
    rec = json.loads(lines[0])
    assert {"raw", "answer", "score", "score_norm"} <= rec.keys()
    # a kill tore the last record: it is cut off and generated again (fsutil.read_jsonl)
    pred = tmp_path / "predictions.jsonl"
    full = pred.read_text()
    pred.write_text(full[: full.rindex("\n", 0, len(full) - 1) + 1] + lines[-1][:20])
    again = run_generation(da, model, loader, tmp_path, conditions=ALL)
    assert model.calls == n_calls + 1 and pred.read_text() == full


def test_rationale_context_reaches_prompt(items: list[AOKVQAItem], tmp_path: Path) -> None:
    seen: list[str] = []

    class Spy:
        name = "spy"

        def generate(self, query: Query, max_new_tokens: int) -> str:
            seen.append(query.text)
            return "x"

    run_generation(
        items[:2],
        Spy(),
        None,
        tmp_path,
        conditions=(ImageCondition.NONE,),
        contexts=(ContextSource.NONE, ContextSource.RATIONALE),
    )
    assert sum(t.startswith("Context: Object") for t in seen) == 2


def test_image_conditions_need_loader(items: list[AOKVQAItem], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_generation(items, BlindGenerator(items), None, tmp_path)


def test_cli_aokvqa_da(
    aokvqa_dir: Path, tmp_path: Path, items: list[AOKVQAItem], monkeypatch: pytest.MonkeyPatch
) -> None:
    import cmr.runners.stage_a as stage_a
    from cmr.runners.transfer import main

    monkeypatch.setattr(stage_a, "build_model", lambda *a, **k: LookupGenerator(items))
    out = tmp_path / "run"
    main(
        [
            "--benchmark", "aokvqa-da", "--model", "fake", "--out", str(out),
            "--aokvqa-dir", str(aokvqa_dir), "--coco-dir", str(aokvqa_dir.parent / "coco"),
            "--conditions", "real,none", "--limit", "10",
        ]
    )  # fmt: skip
    summary = json.loads((out / "summary.json").read_text())
    assert summary["n_questions"] == 10
    r = rows(summary)
    # JPEG round-trip changes pixels, so the lookup misses: plumbing, not accuracy, here.
    assert set(r) == {("real", "none"), ("none", "none")}
    assert (out / "summary.md").read_text().startswith("# A-OKVQA DA")
    manifest = json.loads((out / "config.json").read_text())["manifest"]
    assert list(manifest["data_sha256"]) == [str(aokvqa_dir / "aokvqa_v1p0_val.json")]


def test_da_masked_rationale_hides_direct_answers_too() -> None:
    from cmr.data.aokvqa import AOKVQAItem
    from cmr.eval.conditions import ContextSource
    from cmr.eval.direct_answer import da_context

    it = AOKVQAItem(
        question_id="q", image_id=1, question="What is it used for?",
        choices=("surfing", "cooking", "sleeping", "reading"), correct_choice_idx=0,
        direct_answers=("surf",) * 6 + ("ride waves",) * 4, difficult_direct_answer=False,
        rationales=("People surf on it.", "It is used to ride waves."), split="val",
    )  # fmt: skip
    text = da_context(it, ContextSource.RATIONALE_MASKED, None) or ""
    assert "surf" not in text.lower() and "ride waves" not in text.lower()
    assert "People ___ on it." in text
