"""A-OKVQA loading and the image-grouped dev split.

Schema of the official ``aokvqa_v1p0_{split}.json`` (allenai/aokvqa):

    question_id, image_id, question, choices[4], correct_choice_idx,
    direct_answers[10], difficult_direct_answer, rationales[3], split

The test split has no public answers, so ``val`` is the held-out test set and settings are
chosen on a dev split held out from ``train`` (:func:`dev_split`; report Section 3.2).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class AOKVQAItem:
    """One A-OKVQA question (fields as in the official JSON).

    ``correct_choice_idx`` is ``None`` on the unlabelled test split; ``image_id`` is a COCO
    2017 id for A-OKVQA items.
    """

    question_id: str
    image_id: int
    question: str
    choices: tuple[str, ...]
    correct_choice_idx: int | None
    direct_answers: tuple[str, ...]
    difficult_direct_answer: bool
    rationales: tuple[str, ...]
    split: str

    @property
    def has_labels(self) -> bool:
        """True when the multiple-choice answer is known (train and val splits)."""
        return self.correct_choice_idx is not None

    @property
    def correct_choice(self) -> str:
        """The text of the correct option; ``ValueError`` on an unlabelled item.

        >>> item_from_record({"question_id": "q1", "image_id": 7, "question": "Q?",
        ...                   "choices": ["a", "b"], "correct_choice_idx": 1}).correct_choice
        'b'
        """
        if self.correct_choice_idx is None:
            raise ValueError(f"{self.question_id}: no label (test split?)")
        return self.choices[self.correct_choice_idx]


def item_from_record(rec: Mapping[str, Any], split: str | None = None) -> AOKVQAItem:
    """Build an item from one official JSON record (label fields may be missing).

    ``split`` is used only when the record has no ``split`` field.

    >>> it = item_from_record({"question_id": "q1", "image_id": "7", "question": "Q?",
    ...                        "choices": ["a", "b"]}, split="test")
    >>> it.image_id, it.has_labels, it.split
    (7, False, 'test')
    """
    return AOKVQAItem(
        question_id=str(rec["question_id"]),
        image_id=int(rec["image_id"]),
        question=str(rec["question"]),
        choices=tuple(str(c) for c in rec["choices"]),
        correct_choice_idx=(
            int(rec["correct_choice_idx"]) if rec.get("correct_choice_idx") is not None else None
        ),
        direct_answers=tuple(str(a) for a in rec.get("direct_answers") or ()),
        difficult_direct_answer=bool(rec.get("difficult_direct_answer", False)),
        rationales=tuple(str(r) for r in rec.get("rationales") or ()),
        split=str(rec.get("split") or split or "unknown"),
    )


def load_aokvqa(aokvqa_dir: str | Path, split: str, version: str = "v1p0") -> list[AOKVQAItem]:
    """Load ``{aokvqa_dir}/aokvqa_{version}_{split}.json`` into items.

    Raises:
        ValueError: ``split`` is not train, val or test.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    path = Path(aokvqa_dir) / f"aokvqa_{version}_{split}.json"
    with path.open() as f:
        records = json.load(f)
    return [item_from_record(r, split) for r in records]


def coco_image_path(coco_dir: str | Path, split: str, image_id: int) -> Path:
    """Path of a COCO 2017 image (A-OKVQA images are COCO 2017 images).

    >>> str(coco_image_path("coco", "val", 42))
    'coco/val2017/000000000042.jpg'
    """
    return Path(coco_dir) / f"{split}2017" / f"{image_id:012d}.jpg"


def _unit_hash(key: str, seed: int) -> float:
    """First 8 bytes of ``sha256(f"{seed}:{key}")`` as a fraction in [0, 1).

    >>> round(_unit_hash("1", 0), 6)
    0.933888
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


DEV_FRACTION = 0.06  # image-grouped dev share used by every run


def dev_split(
    items: Sequence[AOKVQAItem], dev_fraction: float = DEV_FRACTION, seed: int = 0
) -> tuple[list[AOKVQAItem], list[AOKVQAItem]]:
    """Split ``items`` into (train, dev), grouped by image, each in input order.

    Questions share images, so a question-level split would leak dev images into training.
    An image goes to dev when ``_unit_hash(str(image_id), seed) < dev_fraction``.
    """
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be in (0, 1)")
    train: list[AOKVQAItem] = []
    dev: list[AOKVQAItem] = []
    for it in items:
        (dev if _unit_hash(str(it.image_id), seed) < dev_fraction else train).append(it)
    return train, dev
