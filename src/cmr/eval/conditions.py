"""Input conditions: what the frozen LLM is shown besides the question.

Two axes crossed in every evaluation run (:func:`cmr.runners.stage_a.run_stage_a`): the image
slot (:class:`ImageCondition`) and the extra text context (:class:`ContextSource`). This
module only names the conditions and builds the image swap; it loads no model or images.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


class ImageCondition(str, Enum):
    """What goes into the image slot; the value is the record's ``condition`` field.

    >>> ImageCondition("swapped") is ImageCondition.SWAPPED
    True
    """

    REAL = "real"  # the question's own image
    NONE = "none"  # no image: language prior only
    SWAPPED = "swapped"  # another question's image: same format, wrong evidence
    BLANK = "blank"  # uniform grey image


class ContextSource(str, Enum):
    """Extra text before the question; the value is the record's ``context`` field.

    >>> ContextSource.RATIONALE_MASKED.value
    'rationale_masked'
    """

    NONE = "none"
    RATIONALE = "rationale"  # gold rationales; often name the answer
    RATIONALE_MASKED = "rationale_masked"  # options and direct answers masked (mask_options)
    CAPTION = "caption"  # gold COCO captions; leaks flagged by ``caption_leak``


def swap_assignment(image_ids: Sequence[int], seed: int = 0, max_iter: int = 100_000) -> list[int]:
    """Deterministic permutation ``perm`` with ``image_ids[perm[i]] != image_ids[i]`` for all i.

    Question i is shown the image of question ``perm[i]``; each image is used once, so the
    image distribution matches REAL.

    Raises:
        ValueError: fewer than two distinct images, or one image owns over half the questions.
        RuntimeError: no valid assignment within ``max_iter`` repair steps.

    >>> swap_assignment([10, 20, 30])
    [2, 0, 1]
    """
    ids = np.asarray(image_ids)
    n = ids.size
    if n < 2 or np.unique(ids).size < 2:
        raise ValueError("need at least two distinct images to swap")
    if np.max(np.unique(ids, return_counts=True)[1]) > n / 2:
        raise ValueError("one image owns more than half the questions; no valid swap exists")

    # Start from a random permutation, then repair clashes by pairwise swaps.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    for _ in range(max_iter):
        bad = np.flatnonzero(ids[perm] == ids)
        if bad.size == 0:
            return [int(p) for p in perm]
        i = int(bad[0])
        j = int(rng.integers(n))
        if ids[perm[j]] != ids[i] and ids[perm[i]] != ids[j]:
            perm[i], perm[j] = perm[j], perm[i]
    raise RuntimeError("could not find a valid swap assignment")
