"""MMStar transfer (cmr.eval.transfer, cmr-transfer; Table 4.16) on synthetic rows.

Row shapes follow the Hugging Face Hub dataset; no network.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cmr.eval.conditions import ImageCondition
from cmr.eval.prompts import mc_prompt, rotate
from cmr.eval.transfer import (
    TransferSet,
    chance_accuracy,
    hashed_subset,
    image_bytes,
    mmstar_set,
    parse_mmstar_question,
    per_category_accuracy,
)
from cmr.models.fake import LookupModel, PriorOnlyModel, image_key
from cmr.runners.stage_a import RunSpec, run_stage_a
from cmr.runners.transfer import main, mc_extras, run_mc, summarize_mc

C = ImageCondition


def png(v: int, size: tuple[int, int] = (16, 16)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (v % 256, (3 * v) % 256, (7 * v) % 256)).save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- parsing

INLINE = "What is shown?\nOptions: A: a cat, a dog, B: a bed, C: a bus, D: a car"
PAREN = (
    "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, "
    "at the end.\nQuestion: Is it max?\nChoices:\n(A) yes\n(B) no\n(C) nan\n(D) nan"
)
BARE = (
    "Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, "
    "at the end.\nQuestion: What percentage?\n(A) 50\n(B) 25\n(C) 60\n(D) 75"
)


def test_parse_mmstar_layouts() -> None:
    assert parse_mmstar_question(INLINE) == (
        "What is shown?",
        ["a cat, a dog", "a bed", "a bus", "a car"],  # commas inside an option survive
    )
    assert parse_mmstar_question(PAREN) == ("Is it max?", ["yes", "no"])  # nan padding gone
    assert parse_mmstar_question(BARE) == ("What percentage?", ["50", "25", "60", "75"])
    with pytest.raises(ValueError):
        parse_mmstar_question("no options at all")


def mmstar_rows(n: int = 12) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        q = f"Question {i}?\nOptions: A: o{i}a, B: o{i}b, C: o{i}c, D: o{i}d"
        if i % 4 == 3:
            q = f"Question {i}?\nOptions: A: o{i}a, B: o{i}b"
        out.append(
            {
                "index": i,
                "question": q,
                "answer": "AB"[i % 2],
                "category": ["coarse perception", "math"][i % 2],
                "l2_category": "x",
                "image": {"bytes": png(i), "path": None},
            }
        )
    out.append({**out[0], "index": 99, "answer": "Z"})  # invalid answer -> skipped
    return out


def test_mmstar_set_items_and_images() -> None:
    s = mmstar_set(mmstar_rows())
    assert len(s.items) == 12 and s.skipped == {"bad_answer_or_options": 1}
    it = s.items[3]
    assert it.question == "Question 3?" and it.choices == ("o3a", "o3b")
    assert it.correct_choice_idx == 1 and s.category[it.question_id] == "math"
    assert s.load_image(it).getpixel((0, 0)) == (3, 9, 21)
    assert len({i.image_id for i in s.items}) == 12
    assert chance_accuracy(s.items) == pytest.approx((9 * 0.25 + 3 * 0.5) / 12)


def test_image_bytes() -> None:
    a = Image.new("RGB", (4, 6))
    assert image_bytes(None) is None
    assert Image.open(io.BytesIO(image_bytes(a) or b"")).size == (4, 6)
    assert image_bytes(b"abc") == b"abc"
    with pytest.raises(TypeError):
        image_bytes(3)


# ----------------------------------------------------------------------------- subsets


def test_hashed_subset_deterministic_and_order_free() -> None:
    rows = [{"id": i} for i in range(200)]
    a = hashed_subset(rows, 20, lambda r: str(r["id"]))
    shuffled = rows[:]
    random.Random(1).shuffle(shuffled)
    b = hashed_subset(shuffled, 20, lambda r: str(r["id"]))
    assert len(a) == 20 and {r["id"] for r in a} == {r["id"] for r in b}
    assert {r["id"] for r in a} != {r["id"] for r in rows[:20]}  # not just "first N"
    assert hashed_subset(rows, None, lambda r: str(r["id"])) == rows
    assert {r["id"] for r in hashed_subset(rows, 20, lambda r: str(r["id"]), seed=1)} != {
        r["id"] for r in a
    }


# ----------------------------------------------------------------------------- MC end to end


def _lookup(tset: TransferSet, rotations: int) -> LookupModel:
    answers = {}
    for it in tset.items:
        key = image_key(tset.load_image(it))
        for k in range(rotations):
            shown, original = rotate(it.choices, k)
            answers[(mc_prompt(it.question, shown), key)] = original.index(
                it.correct_choice_idx or 0
            )
    return LookupModel(answers, confidence=8.0)


def test_mmstar_through_stage_a_scorer(tmp_path: Path) -> None:
    s = mmstar_set(mmstar_rows())
    conds = (C.REAL, C.NONE, C.SWAPPED, C.BLANK)
    summary = run_mc(s, _lookup(s, 2), tmp_path, conds, 2, 0, {})
    rows = {r["condition"]: r for r in summary["rows"]}
    assert rows["real"]["acc"] == 1.0 and rows["real"]["circ_acc"] == 1.0
    assert rows["none"]["delta_acc_vs_ref"] > 0 and rows["swapped"]["vig_nats"] > 0
    arb = summary["arbitration"]  # ragged option counts (2 and 4) handled
    assert arb["acc_on_prior_wrong"] == 1.0
    assert set(arb["debiased_acc"]) == {"0", "0.25", "0.5", "0.75", "1"}
    cats = summary["transfer"]["per_category"]["real"]
    assert set(cats) == {"coarse perception", "math"} and cats["math"]["acc"] == 1.0
    md = (tmp_path / "summary.md").read_text()
    assert md.startswith("# mmstar") and "| math |" in md and "Chance accuracy" in md


def test_summarize_mc_matches_reliance_when_not_ragged(tmp_path: Path) -> None:
    from cmr.analysis.reliance import summarize

    rows = [r for r in mmstar_rows() if r["index"] % 4 != 3 and r["index"] != 99]
    s = mmstar_set(rows)
    spec = RunSpec(conditions=(C.REAL, C.NONE))
    recs = run_stage_a(s.items, PriorOnlyModel(), s.load_image, spec, tmp_path)
    assert summarize_mc(recs) == summarize(recs)
    assert set(mc_extras(recs, s)) == {"benchmark", "chance_acc", "skipped", "per_category"}
    assert per_category_accuracy(recs, s.category)["real"]["math"]["n"] == 3  # 1, 5, 9


# ----------------------------------------------------------------------------- CLI


def test_cli_mmstar(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import cmr.runners.stage_a as stage_a
    import cmr.runners.transfer as transfer

    star = mmstar_set(mmstar_rows())
    monkeypatch.setattr(transfer, "load_benchmark", lambda name, **kw: star)
    monkeypatch.setattr(stage_a, "build_model", lambda *a, **k: PriorOnlyModel())
    written: list[str] = []  # every whole-file output goes through write_text_atomic
    real_write = transfer.write_text_atomic

    def write_text_atomic(path: Path, text: str) -> None:
        written.append(f"{path.parent.name}/{path.name}")
        real_write(path, text)

    monkeypatch.setattr(transfer, "write_text_atomic", write_text_atomic)
    main(
        ["--benchmark", "mmstar", "--model", "x", "--out", str(tmp_path / "m"), "--rotations", "2"]
    )
    summary = json.loads((tmp_path / "m" / "summary.json").read_text())
    assert summary["transfer"]["benchmark"] == "mmstar" and summary["n_questions"] == 12
    assert json.loads((tmp_path / "m" / "config.json").read_text())["benchmark"] == "mmstar"
    assert written == [f"m/{f}" for f in ("config.json", "summary.json", "summary.md")]
    assert not list(tmp_path.glob("*/.*.tmp"))


def test_subset_indices_matches_hashed_subset() -> None:
    from cmr.eval.transfer import subset_indices

    keys = [f"k{i}" for i in range(50)]
    rows = [{"k": k} for k in keys]
    picked = {r["k"] for r in hashed_subset(rows, 7, lambda r: r["k"], seed=3)}
    idx = subset_indices(keys, 7, seed=3)
    assert idx == sorted(idx) and {keys[i] for i in idx} == picked
    assert subset_indices(keys, None) == list(range(50))
