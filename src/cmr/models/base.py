"""The model interface: next-token log-probs of the option letters for a query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class Query:
    """One scoring request: the rendered prompt (no chat template) and an optional image."""

    text: str
    image: Image.Image | None = None


@runtime_checkable
class LetterScorer(Protocol):
    """Anything that scores option letters for a :class:`Query`; ``name`` labels runs."""

    name: str

    def letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """Next-token log-probs of letters A, B, ... (full vocabulary, not renormalised)."""
        ...
