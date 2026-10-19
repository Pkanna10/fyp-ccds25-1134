"""A-OKVQA loading (cmr.data.aokvqa): the val file, unlabelled test-split records, COCO
image paths, and the dev split held out from train (grouped by image, deterministic)."""

from pathlib import Path

import pytest

from cmr.data.aokvqa import (
    AOKVQAItem,
    coco_image_path,
    dev_split,
    item_from_record,
    load_aokvqa,
)


def test_load_val(aokvqa_dir: Path) -> None:
    items = load_aokvqa(aokvqa_dir, "val")
    assert len(items) == 24
    assert items[0].correct_choice == items[0].choices[items[0].correct_choice_idx or 0]


def test_test_split_records_without_labels() -> None:
    it = item_from_record(
        {"question_id": "x", "image_id": 1, "question": "q?", "choices": ["a", "b", "c", "d"]},
        "test",
    )
    assert not it.has_labels
    with pytest.raises(ValueError):
        _ = it.correct_choice


def test_bad_split_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_aokvqa(tmp_path, "dev")


def test_coco_path_format() -> None:
    assert coco_image_path("/c", "val", 42) == Path("/c/val2017/000000000042.jpg")


def test_dev_split_is_image_grouped_deterministic_and_complete(items: list[AOKVQAItem]) -> None:
    train, dev = dev_split(items, dev_fraction=0.3, seed=1)
    assert len(train) + len(dev) == len(items)
    assert {it.image_id for it in train}.isdisjoint({it.image_id for it in dev})
    _, dev2 = dev_split(list(reversed(items)), dev_fraction=0.3, seed=1)
    assert {it.question_id for it in dev} == {it.question_id for it in dev2}


@pytest.mark.parametrize("frac", [0.0, 1.0, -0.1])
def test_dev_split_rejects_bad_fraction(items: list[AOKVQAItem], frac: float) -> None:
    with pytest.raises(ValueError):
        dev_split(items, dev_fraction=frac)
