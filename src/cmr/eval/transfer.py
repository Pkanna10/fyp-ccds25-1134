"""Convert the MMStar benchmark (report Section 4.11) into AOKVQAItems.

The items are then scored by the unchanged A-OKVQA multiple-choice scorer
(:func:`cmr.runners.stage_a.run_stage_a`). String image keys map to dense int ids
(``TransferSet.image_keys`` maps back). :func:`mmstar_set` works on plain row dicts.
"""

from __future__ import annotations

import hashlib
import heapq
import io
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from cmr.data.aokvqa import AOKVQAItem
from cmr.eval.prompts import LETTERS

Row = Mapping[str, Any]

HF_SOURCES: dict[str, dict[str, Any]] = {
    "mmstar": {"repo": "Lin-Chen/MMStar", "config": "val", "split": "val"},
}
MC_BENCHMARKS = tuple(HF_SOURCES)  # scored by letter log-probs with run_stage_a


@dataclass
class TransferSet:
    """One converted benchmark: items as AOKVQAItems plus their encoded images."""

    benchmark: str
    items: list[AOKVQAItem]
    images: dict[int, bytes]  # image_id -> encoded image
    image_keys: dict[int, str]  # image_id -> original image key
    category: dict[str, str] = field(default_factory=dict)  # question_id -> category
    extra: dict[str, dict[str, Any]] = field(default_factory=dict)  # question_id -> extras
    skipped: dict[str, int] = field(default_factory=dict)  # reason -> count

    def load_image(self, item: AOKVQAItem) -> Image.Image:
        """Decode ``item``'s image to RGB (the ``image_loader`` passed to the runners)."""
        return Image.open(io.BytesIO(self.images[item.image_id])).convert("RGB")


def image_bytes(value: Any) -> bytes | None:
    """Encoded bytes of an HF image cell (dict, PIL image re-encoded as PNG, or bytes), or None.

    >>> image_bytes(b"abc"), image_bytes(None), image_bytes({"bytes": b"xy", "path": None})
    (b'abc', None, b'xy')
    """
    if value is None:
        return None
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if isinstance(value, Mapping):
        if value.get("bytes"):
            return bytes(value["bytes"])
        if value.get("path"):
            return Path(value["path"]).read_bytes()
        return None
    if isinstance(value, Image.Image):
        buf = io.BytesIO()
        value.convert("RGB").save(buf, format="PNG")
        return buf.getvalue()
    raise TypeError(f"unsupported image cell {type(value).__name__}")


# ----------------------------------------------------------------------------- subsets


def hash_rank(key: str, seed: int = 0) -> int:
    """Deterministic rank of ``key``: the first 8 bytes of sha256("seed:key")."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")


def hashed_subset(
    rows: Iterable[Row], limit: int | None, key: Callable[[Row], str], seed: int = 0
) -> list[Row]:
    """Keep the ``limit`` rows with the smallest hash(key), in original order (None keeps all).

    Independent of file order (HF files are often sorted by category); one pass, O(limit)
    memory.

    >>> rows = [{"id": k} for k in "abcdef"]
    >>> hashed_subset(rows, 3, lambda r: r["id"])
    [{'id': 'a'}, {'id': 'd'}, {'id': 'f'}]
    """
    if limit is None:
        return list(rows)
    heap: list[tuple[int, int, Row]] = []  # max-heap via negated rank
    for i, row in enumerate(rows):
        entry = (-hash_rank(key(row), seed), i, row)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [row for _, _, row in sorted(heap, key=lambda e: e[1])]


# ----------------------------------------------------------------------------- builder


class _Builder:
    """Accumulates one :class:`TransferSet`: dense image ids, items and skip counts."""

    def __init__(self, benchmark: str) -> None:
        self.set = TransferSet(benchmark, [], {}, {})
        self._ids: dict[str, int] = {}

    def image_id(self, key: str, blob: bytes | None) -> int:
        """Dense int id of image ``key``; stores ``blob`` once."""
        if key not in self._ids:
            self._ids[key] = len(self._ids)
            self.set.image_keys[self._ids[key]] = key
        iid = self._ids[key]
        if blob and iid not in self.set.images:
            self.set.images[iid] = blob
        return iid

    def skip(self, reason: str) -> None:
        """Count one dropped row under ``reason``."""
        self.set.skipped[reason] = self.set.skipped.get(reason, 0) + 1

    def add(
        self,
        qid: str,
        image_key: str,
        blob: bytes | None,
        question: str,
        choices: Sequence[str],
        answer_idx: int,
        category: str = "",
        **extra: Any,
    ) -> None:
        """Append one multiple-choice item, with its category and any extra fields."""
        item = AOKVQAItem(
            question_id=qid,
            image_id=self.image_id(image_key, blob),
            question=question,
            choices=tuple(choices),
            correct_choice_idx=answer_idx,
            direct_answers=(),
            difficult_direct_answer=False,
            rationales=(),
            split=self.set.benchmark,
        )
        self.set.items.append(item)
        if category:
            self.set.category[qid] = category
        if extra:
            self.set.extra[qid] = dict(extra)


def _letter_idx(letter: str, n: int) -> int | None:
    """Index of answer letter ``letter`` among the first ``n`` letters, else None."""
    letter = letter.strip().upper()
    if len(letter) != 1 or letter not in LETTERS[:n]:
        return None
    return LETTERS.index(letter)


# ----------------------------------------------------------------------------- MMStar

_MMSTAR_HINT = re.compile(
    r"^Hint: Please answer the question and provide the correct option[^\n]*\n"
)
_PAREN_OPT = re.compile(r"(?:^|\n)\(([A-J])\)[ \t]*")
_NAN = {"nan", "none", ""}


def _split_inline_options(text: str) -> list[str] | None:
    """``A: x, B: y, C: z`` -> [x, y, z], scanning letters in order (commas in x are kept)."""
    text = text.strip()
    if not text.startswith("A:"):
        return None
    starts = [0]
    pos = 0
    for letter in LETTERS[1:]:
        m = re.compile(rf",\s*{letter}:\s").search(text, pos)
        if m is None:
            break
        starts.append(m.start())
        pos = m.end()
    out = []
    for k, s in enumerate(starts):
        end = starts[k + 1] if k + 1 < len(starts) else len(text)
        chunk = text[s:end].lstrip(",").strip()
        out.append(chunk.split(":", 1)[1].strip())
    return out


def parse_mmstar_question(text: str) -> tuple[str, list[str]]:
    """Split an MMStar question into (stem, options); ValueError if < 2 options are found.

    Handles inline ``Options: A: x, B: y`` and ``(A) x`` line layouts, drops the generic
    "Hint:" line (our template has its own instruction) and trailing "nan" options.

    >>> parse_mmstar_question("What is shown?\\nOptions: A: a cat, B: a dog, C: nan")
    ('What is shown?', ['a cat', 'a dog'])
    """
    t = _MMSTAR_HINT.sub("", text.strip())

    # Inline "Options: A: x, B: y" list.
    options: list[str] | None = None
    if "Options:" in t:
        stem, _, opts = t.rpartition("Options:")
        options = _split_inline_options(opts)

    # "(A) x" lines; only marks that run A, B, C... in order count.
    if options is None:
        marks = list(_PAREN_OPT.finditer(t))
        expected = [m for k, m in enumerate(marks) if m.group(1) == LETTERS[k]]
        if len(expected) < 2:
            raise ValueError(f"cannot parse MMStar options: {text[:120]!r}")
        stem = t[: expected[0].start()]
        options = [
            t[m.end() : (expected[k + 1].start() if k + 1 < len(expected) else len(t))].strip()
            for k, m in enumerate(expected)
        ]
        stem = re.sub(r"\n?Choices:\s*$", "", stem.rstrip())

    # Drop "nan" padding options and the "Question:" prefix.
    while options and options[-1].strip().lower() in _NAN:
        options.pop()
    stem = re.sub(r"^Question:\s*", "", stem.strip())
    return stem.strip(), options


def mmstar_set(rows: Iterable[Row]) -> TransferSet:
    """Convert MMStar rows, parsing options with :func:`parse_mmstar_question`; skips bad rows."""
    b = _Builder("mmstar")
    for row in rows:
        try:
            stem, options = parse_mmstar_question(str(row["question"]))
        except ValueError:
            b.skip("unparseable_options")
            continue
        idx = _letter_idx(str(row["answer"]), len(options))
        if idx is None or len(options) > len(LETTERS):
            b.skip("bad_answer_or_options")
            continue
        qid = f"mmstar-{row['index']}"
        b.add(
            qid,
            qid,
            image_bytes(row.get("image")),
            stem,
            options,
            idx,
            category=str(row.get("category", "")),
            l2_category=str(row.get("l2_category", "")),
        )
    return b.set


# ----------------------------------------------------------------------------- HF loading


def hf_dataset(repo: str, config: str, split: str, image_columns: Sequence[str] = ()) -> Any:
    """Load an HF dataset (cached), casting ``image_columns`` to undecoded image cells."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(repo, config, split=split)
    for col in image_columns:
        ds = ds.cast_column(col, HFImage(decode=False))
    return ds


def subset_indices(keys: Sequence[str], limit: int | None, seed: int = 0) -> list[int]:
    """Row indices of :func:`hashed_subset` over ``keys``, in original order.

    >>> subset_indices(list("abcdef"), 3)
    [0, 3, 5]
    """
    rows = [{"i": i, "k": k} for i, k in enumerate(keys)]
    return [int(r["i"]) for r in hashed_subset(rows, limit, lambda r: str(r["k"]), seed)]


def load_benchmark(benchmark: str, limit: int | None = None, seed: int = 0) -> TransferSet:
    """Download and convert one benchmark; ``limit`` keeps a hashed subset by item id.

    Only the selected rows' images are read. Raises KeyError for a name not in ``HF_SOURCES``.
    """
    src = HF_SOURCES[benchmark]
    ds = hf_dataset(src["repo"], src["config"], src["split"], ["image"])
    idx = subset_indices([str(v) for v in ds["index"]], limit, seed)
    return mmstar_set(ds.select(idx))


# ----------------------------------------------------------------------------- extras


def _rot0(records: Sequence[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """(condition, question_id) -> rotation-0 record, context 'none' only."""
    return {
        (r["condition"], r["question_id"]): r
        for r in records
        if int(r["rotation"]) == 0 and r.get("context", "none") == "none"
    }


def per_category_accuracy(
    records: Sequence[dict[str, Any]], category: Mapping[str, str]
) -> dict[str, dict[str, dict[str, float]]]:
    """condition -> category -> {n, acc} (rotation 0, no context)."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (cond, qid), r in _rot0(records).items():
        acc[cond][category.get(qid, "?")].append(float(r["pred_idx"] == r["correct_idx"]))
    return {
        c: {k: {"n": float(len(v)), "acc": float(np.mean(v))} for k, v in sorted(cats.items())}
        for c, cats in sorted(acc.items())
    }


def chance_accuracy(items: Sequence[AOKVQAItem]) -> float:
    """Expected accuracy of uniform guessing (not every MMStar item has four options)."""
    return float(np.mean([1.0 / len(it.choices) for it in items])) if items else 0.0


def extras_markdown(extras: Mapping[str, Any]) -> str:
    """Markdown for whichever extras are present: chance accuracy, skips, per-category."""
    lines: list[str] = []
    if "chance_acc" in extras:
        lines.append(f"Chance accuracy (uniform guess): {100 * extras['chance_acc']:.1f}%\n")
    if extras.get("skipped"):
        lines.append(f"Skipped items: {json.dumps(extras['skipped'])}\n")
    cat = extras.get("per_category")
    if cat:
        conds = list(cat)
        names = sorted({k for c in conds for k in cat[c]})
        lines.append("| category | n | " + " | ".join(f"Acc {c} %" for c in conds) + " |")
        lines.append("|---|---|" + "---|" * len(conds))
        for k in names:
            n = next(int(cat[c][k]["n"]) for c in conds if k in cat[c])
            accs = [f"{100 * cat[c][k]['acc']:.1f}" if k in cat[c] else "-" for c in conds]
            lines.append(f"| {k} | {n} | " + " | ".join(accs) + " |")
        lines.append("")
    return "\n".join(lines)
