"""Greedy generation for the bridge model, and R->A letter scoring (report Section 3.5).

R->A scoring generates the rationale greedily until "Answer:", the end of the turn or 64
tokens; keeps the text before "Answer:", appends a newline and "Answer:" (also when the
model did not write it) and scores the option letters at the next position.
"""

from __future__ import annotations

import torch

from cmr.eval.prompts import letter_token_ids
from cmr.models.base import Query
from cmr.models.bridge import FrozenBridgeVLM
from cmr.train.targets import ANSWER_CUE


@torch.no_grad()
def greedy_generate(
    model: FrozenBridgeVLM, query: Query, prefix: str = "", max_new_tokens: int = 64,
    stop: str | None = ANSWER_CUE,
) -> str:  # fmt: skip
    """Greedy continuation of ``query`` after the unscored assistant text ``prefix``.

    Stops at EOS, after ``max_new_tokens`` tokens, or as soon as the decoded text contains
    ``stop`` (the returned text then includes it).
    """
    with model.scoring_mode():
        return _greedy_generate(model, query, prefix, max_new_tokens, stop)


def _greedy_generate(
    model: FrozenBridgeVLM, query: Query, prefix: str, max_new_tokens: int, stop: str | None
) -> str:
    visual = model._visual_batch([query.image])[0]
    text = model.format_prompt(query.text, visual is not None)
    embeds, _ = model.build_inputs(text, visual, assistant_prefix=prefix or None)
    embeds = embeds.unsqueeze(0)
    out_ids: list[int] = []
    past = None
    step_in = embeds
    emb = model.llm.get_input_embeddings()
    for _ in range(max_new_tokens):
        logits, past = model.next_token_logits(step_in, past_key_values=past, use_cache=True)
        nxt = int(logits.argmax())
        if nxt == model.tokenizer.eos_token_id:
            break
        out_ids.append(nxt)
        decoded = model.tokenizer.decode(out_ids, skip_special_tokens=True)
        if stop and stop in decoded:
            return str(decoded)
        step_in = emb(torch.tensor([[nxt]], device=model.device))
    return str(model.tokenizer.decode(out_ids, skip_special_tokens=True))


@torch.no_grad()
def rationale_then_letter(
    model: FrozenBridgeVLM, query: Query, n_options: int, max_new_tokens: int = 64
) -> tuple[list[float], str, bool]:
    """(letter log-probs, generated rationale, whether the model emitted the cue itself)."""
    with model.scoring_mode():
        return _rationale_then_letter(model, query, n_options, max_new_tokens)


def _rationale_then_letter(
    model: FrozenBridgeVLM, query: Query, n_options: int, max_new_tokens: int
) -> tuple[list[float], str, bool]:
    generated = greedy_generate(model, query, max_new_tokens=max_new_tokens)
    cue_found = ANSWER_CUE in generated
    head = generated.split(ANSWER_CUE, 1)[0].rstrip()
    # no trailing space: training saw " B" as one token after "Answer:"
    prefix = f"{head}\n{ANSWER_CUE}"
    visual = model._visual_batch([query.image])[0]
    text = model.format_prompt(query.text, visual is not None)
    embeds, _ = model.build_inputs(text, visual, assistant_prefix=prefix)
    logits = model.next_token_logits(embeds.unsqueeze(0))[0]
    logp = torch.log_softmax(logits, -1)
    lps = [float(torch.logsumexp(logp[g], 0)) for g in letter_token_ids(model.tokenizer, n_options)]
    return lps, head, cue_found


def rationale_then_letter_logprobs(
    model: FrozenBridgeVLM, query: Query, n_options: int, max_new_tokens: int = 64
) -> list[float]:
    return rationale_then_letter(model, query, n_options, max_new_tokens)[0]
