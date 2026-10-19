"""Deterministic fake models for tests and smoke runs.

PriorOnlyModel ignores the image (image-reliance gap 0); LookupModel knows the answer only
with the right image (large image-reliance gap).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image

from cmr.models.base import Query


def image_key(image: Image.Image | None) -> str:
    """First 16 hex digits of the SHA-1 of an image's pixel bytes (``"<none>"`` for None).

    >>> image_key(None)
    '<none>'
    >>> image_key(Image.new("RGB", (2, 2))) == image_key(Image.new("RGB", (2, 2)))
    True
    """
    if image is None:
        return "<none>"
    return hashlib.sha1(image.tobytes()).hexdigest()[:16]


def _text_logits(text: str, n: int) -> np.ndarray:
    """``n`` pseudo-random logits seeded by the SHA-1 of ``text``."""
    seed = int.from_bytes(hashlib.sha1(text.encode()).digest()[:4], "big")
    return np.random.default_rng(seed).normal(size=n)


def _log_softmax(x: np.ndarray) -> list[float]:
    """Numerically stable log-softmax of a 1-D array, as Python floats.

    >>> [round(v, 4) for v in _log_softmax(np.array([0.0, 0.0]))]
    [-0.6931, -0.6931]
    """
    m = float(np.max(x))
    lse = m + math.log(float(np.sum(np.exp(x - m))))
    return [float(v - lse) for v in x]


class PriorOnlyModel:
    """Letter log-probs that depend on the prompt text alone; the image is ignored.

    >>> m = PriorOnlyModel()
    >>> with_image = Query("Q?", Image.new("RGB", (2, 2)))
    >>> m.letter_logprobs(Query("Q?"), 4) == m.letter_logprobs(with_image, 4)
    True
    """

    name = "fake-prior-only"

    def letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """Log-probs from logits seeded by ``query.text``; the image is unused."""
        return _log_softmax(_text_logits(query.text, n_options))


class LookupModel:
    """Prior scores, plus ``confidence`` on ``answers[(text, image_key(image))]`` if present.

    Args:
        answers: ``{(prompt text, image_key(image)): correct letter index}``.
    """

    name = "fake-lookup"

    def __init__(self, answers: dict[tuple[str, str], int], confidence: float = 4.0) -> None:
        self.answers = answers
        self.confidence = confidence

    def letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """Prior logits for ``query.text``, plus ``confidence`` on the looked-up letter."""
        logits = _text_logits(query.text, n_options)
        target = self.answers.get((query.text, image_key(query.image)))
        if target is not None:
            logits[target] += self.confidence
        return _log_softmax(logits)
