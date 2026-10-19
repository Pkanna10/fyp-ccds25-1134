"""Shared fixtures: a small synthetic A-OKVQA-format dataset with real (generated) images."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem, item_from_record

# the runners pin torch to --threads (default $CMR_THREADS or 4) when main() runs: keep
# CLI tests single-threaded unless the caller chose otherwise
os.environ.setdefault("CMR_THREADS", "1")


def make_records(n_questions: int = 24, n_images: int = 12) -> list[dict[str, object]]:
    recs = []
    for i in range(n_questions):
        recs.append(
            {
                "question_id": f"q{i:03d}",
                "image_id": 1000 + (i % n_images),
                "question": f"What is object number {i} used for?",
                "choices": [f"use-{i}-{k}" for k in range(4)],
                "correct_choice_idx": i % 4,
                "direct_answers": [f"use-{i}-{i % 4}"] * 6 + ["other"] * 4,
                "difficult_direct_answer": i % 7 == 0,
                "rationales": [f"Object {i} is shown being used.", "It is common sense."],
                "split": "val",
            }
        )
    return recs


@pytest.fixture
def records() -> list[dict[str, object]]:
    return make_records()


@pytest.fixture
def items(records: list[dict[str, object]]) -> list[AOKVQAItem]:
    return [item_from_record(r) for r in records]


def colour_image(image_id: int) -> Image.Image:
    """A distinct solid-colour image per image_id (so swaps change the pixels)."""
    v = (image_id * 37) % 256
    return Image.new("RGB", (32, 32), (v, 255 - v, (v * 3) % 256))


@pytest.fixture
def aokvqa_dir(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    d = tmp_path / "aokvqa"
    d.mkdir()
    (d / "aokvqa_v1p0_val.json").write_text(json.dumps(records))
    coco = tmp_path / "coco" / "val2017"
    coco.mkdir(parents=True)
    for r in records:
        colour_image(int(r["image_id"])).save(coco / f"{int(r['image_id']):012d}.jpg")  # type: ignore[call-overload]
    return d
