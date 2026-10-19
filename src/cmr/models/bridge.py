"""Frozen vision encoder -> trainable bridge (projector) -> frozen LLM.

The projector's visual tokens replace the ``<image>`` marker in the LLM's input
embeddings; only the projector (and LoRA, if enabled) is trained. LLaVA
``mm_projector.bin`` weights load directly: the configs start from the LLaVA-OneVision
stage-1 projector (theta_0 in the report).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from PIL import Image
from torch import nn

from cmr.eval.prompts import letter_token_ids
from cmr.models import head as head_mod
from cmr.models.base import Query

IMAGE_MARKER = "<image>"


def build_projector(kind: Literal["mlp2x_gelu"], d_in: int, d_out: int) -> nn.Module:
    """A randomly initialised projector ``(..., d_in) -> (..., d_out)``.

    ``"mlp2x_gelu"`` is the Linear-GELU-Linear projector of LLaVA-1.5 and LLaVA-OneVision;
    its layer indices (0 and 2) match the keys of their ``mm_projector.bin``.
    """
    if kind == "mlp2x_gelu":
        return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))
    raise ValueError(f"unknown projector {kind!r}")


def resolve_path(path: str) -> str:
    """``hf://<org>/<repo>/<file>`` -> local cached file; other paths unchanged."""
    if not path.startswith("hf://"):
        return path
    from huggingface_hub import hf_hub_download

    org, repo, filename = path[len("hf://") :].split("/", 2)
    return str(hf_hub_download(f"{org}/{repo}", filename))


def load_llava_projector(projector: nn.Module, path: str) -> None:
    """Load a LLaVA ``mm_projector.bin`` (local or ``hf://``) strictly into ``projector``."""
    raw = torch.load(resolve_path(path), map_location="cpu", weights_only=True)
    state = {k.split("mm_projector.", 1)[1]: v for k, v in raw.items() if "mm_projector." in k}
    if not state:
        raise ValueError(f"no mm_projector.* keys in {path}")
    projector.load_state_dict(state, strict=True)


@dataclass(frozen=True)
class LossParts:
    """Per-example (B,) mean NLL over all target tokens, answer-letter tokens and the rest.

    ``answer`` and ``rest`` are the CE(letter) and CE(rest) of the balanced loss; they are
    ``None`` when no answer spans were given.
    """

    all: torch.Tensor
    answer: torch.Tensor | None
    rest: torch.Tensor | None


@dataclass(frozen=True)
class BridgeSpec:
    """Everything needed to rebuild the model from public checkpoints (``configs/*.json``)."""

    vision_id: str
    llm_id: str
    projector_kind: Literal["mlp2x_gelu"] = "mlp2x_gelu"
    projector_path: str | None = None
    select_layer: int = -2  # LLaVA convention: penultimate layer
    drop_cls: bool = True  # CLIP has a CLS token, SigLIP does not
    pool: int = 1  # k x k average pooling of the patch grid
    prompt_style: Literal["chat_template"] = "chat_template"  # the LLM's own chat template
    pad_to_square: bool = True  # LLaVA-1.5 "pad" aspect handling; configs/*.json turn it off

    @classmethod
    def from_json(cls, path: str) -> BridgeSpec:
        """Read a spec from JSON; keys starting with ``_`` are comments and are ignored."""
        import json
        from pathlib import Path

        raw = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in raw.items() if not k.startswith("_")})


class FrozenBridgeVLM(nn.Module):
    """Frozen vision tower -> trainable projector -> frozen LLM; a :class:`LetterScorer`.

    Usually built with :meth:`from_pretrained`. The constructor freezes ``vision`` and
    ``llm``. The boolean switches set in ``__init__`` are plain attributes.
    """

    vision: Any  # HF vision tower (frozen)
    llm: Any  # HF causal LM (frozen)
    projector: nn.Module  # the only trainable part

    def __init__(
        self,
        vision: nn.Module,
        projector: nn.Module,
        llm: nn.Module,
        tokenizer: Any,
        image_processor: Any | None,
        spec: BridgeSpec,
        name: str = "frozen-bridge",
    ) -> None:
        super().__init__()
        self.vision, self.projector, self.llm = vision, projector, llm
        self.tokenizer, self.image_processor, self.spec, self.name = (
            tokenizer,
            image_processor,
            spec,
            name,
        )
        # Scoring prefix cache: the KV states of [prompt text before the image + image tokens]
        # are reused across queries on the same image (e.g. its four option orders). Stored as
        # (image object, prefix text, KV cache, prefix length, trainable-weight versions).
        # Fused AdamW does not bump _version, so loss_parts also clears the cache every step.
        self._prefix_cache: tuple[Image.Image, str, Any, int, tuple[int, ...]] | None = None
        self.use_prefix_cache = True  # False = one full forward per query
        # scoring: ask the LLM for last-position logits only (logits_to_keep=1); used only when
        # the fp32 head is off or does not apply
        self.score_last_logits_only = True
        # training: apply the LM head only at scored positions (exact); False = full logits
        self.target_logits_only = True
        # LM head in fp32 at the scored rows (see cmr.models.head)
        self.head_fp32 = True
        # per-image frozen-tower feature cache, keyed by PIL info["cmr_key"] ("<split>/<image
        # id>", set by the cmr-stage-b image loader); None = off
        self.feature_cache_dir: Path | None = None
        self.feature_cache_splits: frozenset[str] = frozenset({"train", "val"})
        self._vision_load_dtype = next(vision.parameters()).dtype
        # right-pad training rows to a multiple of this (1 = off) so torch.compile sees few
        # shapes; padding follows every real token, so causal attention never sees it
        self.pad_multiple = 1

        for module in (self.vision, self.llm):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_pretrained(
        cls, spec: BridgeSpec, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> FrozenBridgeVLM:
        """Load the parts named in ``spec`` from the HF hub and build the model.

        Args:
            dtype: dtype of the frozen towers (the projector stays fp32).
        """
        from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer

        # keep only the vision half of a dual encoder (e.g. SigLIP)
        vision_full = AutoModel.from_pretrained(spec.vision_id, torch_dtype=dtype)
        vision = getattr(vision_full, "vision_model", vision_full)

        llm = AutoModelForCausalLM.from_pretrained(spec.llm_id, torch_dtype=dtype)
        tok = AutoTokenizer.from_pretrained(spec.llm_id, use_fast=True)
        proc = AutoImageProcessor.from_pretrained(spec.vision_id)  # type: ignore[no-untyped-call]

        d_v = int(vision.config.hidden_size)
        embed: Any = llm.get_input_embeddings()
        d_llm = int(embed.weight.shape[1])
        # fp32 master weights: a bf16 projector under AdamW silently drops small updates
        projector = build_projector(spec.projector_kind, d_v, d_llm)
        if spec.projector_path:
            load_llava_projector(projector, spec.projector_path)
        model = cls(vision, projector, llm, tok, proc, spec, name=f"bridge:{spec.llm_id}")
        return model.to(device)

    @property
    def device(self) -> torch.device:
        """The projector's device (the whole model moves together)."""
        return next(self.projector.parameters()).device

    # ------------------------------------------------------------------ vision side
    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """PIL images -> ``pixel_values`` (B, 3, H, W), padded to square if configured."""
        if self.image_processor is None:
            raise ValueError("no image_processor configured")
        imgs = [_pad_to_square(im) if self.spec.pad_to_square else im for im in images]
        out = self.image_processor(images=[im.convert("RGB") for im in imgs], return_tensors="pt")
        return out["pixel_values"]  # type: ignore[no-any-return]

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (B,3,H,W) -> visual tokens (B, N, D_llm); the tower runs without grad."""
        return self._project(self._tower(pixel_values))

    def _tower(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Frozen vision tower: the selected hidden layer (CLS dropped if set), before pooling."""
        with torch.no_grad():
            out = self.vision(
                pixel_values=pixel_values.to(self.device, self._vision_dtype()),
                output_hidden_states=True,
            )
            feats = out.hidden_states[self.spec.select_layer]
            if self.spec.drop_cls:
                feats = feats[:, 1:]
        return feats  # type: ignore[no-any-return]

    def _project(self, feats: torch.Tensor) -> torch.Tensor:
        """Pool the patch grid (no grad), then the trainable projector in its own dtype."""
        with torch.no_grad():
            feats = _pool_grid(feats, self.spec.pool)
        return self.projector(feats.to(next(self.projector.parameters()).dtype))  # type: ignore[no-any-return]

    def _feature_path(self, image: Image.Image) -> Path | None:
        """Cache file for ``image``'s tower output, or None (cache off / unkeyed image)."""
        key = image.info.get("cmr_key") if self.feature_cache_dir is not None else None
        if not isinstance(key, str) or key.split("/", 1)[0] not in self.feature_cache_splits:
            return None
        assert self.feature_cache_dir is not None
        tag = (
            f"{self.spec.vision_id.replace('/', '--')}_L{self.spec.select_layer}"
            f"_cls{int(self.spec.drop_cls)}_pad{int(self.spec.pad_to_square)}"
            f"_{dtype_name(self._vision_dtype())}_from_{dtype_name(self._vision_load_dtype)}"
        )
        return self.feature_cache_dir / tag / f"{key.replace('/', '_')}.pt"

    def _cached_tower(self, images: list[Image.Image]) -> torch.Tensor:
        """Tower features (B, N, D_v) through the feature cache; misses are written atomically."""
        paths = [self._feature_path(im) for im in images]
        miss = [i for i, path in enumerate(paths) if path is None or not path.exists()]
        rows: list[torch.Tensor | None] = [None] * len(images)
        if miss:
            feats = self._tower(self.preprocess([images[i] for i in miss]))
            for i, f in zip(miss, feats, strict=True):
                rows[i] = f
                path = paths[i]
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(f".tmp{os.getpid()}")
                    torch.save(f.detach().to("cpu").clone(), tmp)
                    tmp.replace(path)
        for i, path in enumerate(paths):
            if rows[i] is None:
                assert path is not None
                rows[i] = torch.load(path, map_location=self.device, weights_only=True)
        return torch.stack([r for r in rows if r is not None])

    def tower_features(self, images: list[Image.Image]) -> torch.Tensor:
        """Frozen-tower outputs (B, N, D_v) before pooling, through the cache when it is on."""
        if self.feature_cache_dir is not None:
            return self._cached_tower(images)
        return self._tower(self.preprocess(images))

    def _vision_dtype(self) -> torch.dtype:
        """dtype of the vision tower's weights (pixel values are cast to it)."""
        dtype: torch.dtype = next(self.vision.parameters()).dtype
        return dtype

    # ------------------------------------------------------------------ text side
    def format_prompt(self, text: str, has_image: bool) -> str:
        """The full prompt string, with ``<image>`` before the question when ``has_image``."""
        user = f"{IMAGE_MARKER}\n{text}" if has_image else text
        return str(
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
            )
        )

    def _embed(self, ids: list[int]) -> torch.Tensor:
        """Token ids (L,) -> the frozen LLM's input embeddings (L, D_llm)."""
        emb = self.llm.get_input_embeddings()
        return emb(torch.tensor(ids, device=self.device, dtype=torch.long))  # type: ignore[no-any-return]

    def build_inputs(
        self,
        prompt: str,
        visual: torch.Tensor | None,
        target: str | None = None,
        assistant_prefix: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(inputs_embeds (L, D), labels (L,)) for one example.

        Labels are -100 (ignored by the loss) everywhere except on the target tokens.

        Args:
            visual: visual tokens (N, D_llm) replacing the ``<image>`` marker, or ``None``.
            target: training target (EOS appended), or ``None`` when scoring.
            assistant_prefix: unscored text after the prompt.
        """
        parts = prompt.split(IMAGE_MARKER)
        if (visual is None) != (len(parts) == 1) or len(parts) > 2:
            raise ValueError("prompt must contain exactly one image marker iff visual is given")

        pieces: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        for i, part in enumerate(parts):
            ids = self.tokenizer.encode(part, add_special_tokens=(i == 0))
            if ids:
                pieces.append(self._embed(ids))
                labels.append(torch.full((len(ids),), -100, dtype=torch.long))
            if i == 0 and visual is not None:
                pieces.append(visual)
                labels.append(torch.full((visual.shape[0],), -100, dtype=torch.long))

        if assistant_prefix:  # tokenised exactly as the start of a training target
            a_ids = self.tokenizer.encode(assistant_prefix, add_special_tokens=False)
            pieces.append(self._embed(a_ids))
            labels.append(torch.full((len(a_ids),), -100, dtype=torch.long))
        if target is not None:
            t_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                t_ids = [*t_ids, int(self.tokenizer.eos_token_id)]
            pieces.append(self._embed(t_ids))
            labels.append(torch.tensor(t_ids, dtype=torch.long))

        embeds = torch.cat([p.to(pieces[0].dtype) for p in pieces], dim=0)
        return embeds, torch.cat(labels).to(self.device)

    # ------------------------------------------------------------------ training
    def loss(
        self,
        texts: list[str],
        images: list[Image.Image | None],
        targets: list[str],
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean over examples of the per-example mean target NLL (optionally weighted).

        Args:
            images: ``None`` for no image.
        """
        per_example = self.loss_parts(texts, images, targets).all
        if weights is not None:
            return (per_example * weights.to(per_example)).sum() / weights.sum()
        return per_example.mean()

    def loss_parts(
        self,
        texts: list[str],
        images: list[Image.Image | None],
        targets: list[str],
        answer_spans: list[tuple[int, int]] | None = None,
        tower_feats: torch.Tensor | None = None,
    ) -> LossParts:
        """Per-example mean target NLL, also split into answer-letter tokens and the rest.

        Args:
            answer_spans: character span of the answer letter in each target
                (:func:`cmr.train.targets.answer_char_span`).
            tower_feats: precomputed frozen-tower outputs (B, N, D_v); ``images`` then
                only gives the batch length.
        """
        self.clear_cache()  # weights may change after this call

        if tower_feats is not None:
            visual: list[torch.Tensor | None] = [*self._project(tower_feats.to(self.device))]
        else:
            visual = self._visual_batch(images)
        rows = [
            self.build_inputs(self.format_prompt(t, v is not None), v, tgt)
            for t, v, tgt in zip(texts, visual, targets, strict=True)
        ]
        embeds, labels, mask = _right_pad(rows, self.pad_multiple)
        gold = labels[:, 1:]
        nll = self._target_nll(embeds, mask, gold)
        valid = (gold != -100).float()
        all_mean = (nll * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        if answer_spans is None:
            return LossParts(all_mean, None, None)

        is_ans = torch.zeros_like(valid)
        for i, (tgt, span) in enumerate(zip(targets, answer_spans, strict=True)):
            flags = self._answer_token_flags(tgt, span)
            pos = (gold[i] != -100).nonzero().flatten()
            if len(flags) != len(pos):
                raise RuntimeError("target tokenisation mismatch while locating the answer")
            is_ans[i, pos] = torch.tensor(flags, dtype=is_ans.dtype, device=is_ans.device)

        rest = valid - is_ans
        n_ans = is_ans.sum(1)
        if bool((n_ans == 0).any()):
            raise ValueError("an answer span covers no target token")
        ans_mean = (nll * is_ans).sum(1) / n_ans
        rest_mean = (nll * rest).sum(1) / rest.sum(1).clamp_min(1.0)
        return LossParts(all_mean, ans_mean, rest_mean)

    def _target_nll(
        self, embeds: torch.Tensor, mask: torch.Tensor, gold: torch.Tensor
    ) -> torch.Tensor:
        """Token NLL (B, T-1) of ``gold`` (labels shifted by one); 0 where gold is -100.

        Fast path: apply the LM head only to the hidden states at scored positions.
        """
        parts = self._decoder_and_head() if self.target_logits_only else None

        if parts is None:  # full (B, T, V) logits
            out = self.llm(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
            logits = out.logits[:, :-1]
            return nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), gold.reshape(-1), reduction="none"
            ).view(gold.shape)

        decoder, head = parts
        # no KV cache in a training pass (memory)
        hidden = decoder(
            inputs_embeds=embeds, attention_mask=mask, use_cache=False
        ).last_hidden_state
        scored = gold != -100
        rows = hidden[:, :-1][scored]  # (K, D), K = number of scored tokens
        logits = head_mod.fp32_head_logits(rows, head) if self.head_fp32_active else head(rows)
        nll_scored = nn.functional.cross_entropy(logits.float(), gold[scored], reduction="none")

        nll = torch.zeros(gold.shape, dtype=nll_scored.dtype, device=nll_scored.device)
        nll[scored] = nll_scored  # index_put is differentiable
        return nll

    def _decoder_and_head(self) -> tuple[Any, Any] | None:
        """(decoder, lm_head) for allow-listed architectures (PEFT unwrapped), else None."""
        base = head_mod.plain_head_base(self.llm)
        if base is None:
            return None
        return base.model, base.lm_head

    def _fp32_parts(self) -> tuple[Any, nn.Linear] | None:
        """(decoder, lm_head) when ``head_fp32`` is on and applies to this LLM, else None."""
        if not self.head_fp32:
            return None
        parts = self._decoder_and_head()
        if parts is None or not head_mod.fp32_capable(parts[1]):
            return None
        return parts[0], parts[1]

    @property
    def head_fp32_active(self) -> bool:
        """``head_fp32`` is on and applies to this LLM."""
        return self._fp32_parts() is not None

    def describe(self) -> dict[str, Any]:
        """Scoring settings for run provenance: head dtype, prefix cache, frozen LLM dtype."""
        info: dict[str, Any] = head_mod.head_dtype_info(self.head_fp32, self.head_fp32_active)
        return {**info, "prefix_cache": self.use_prefix_cache, "dtype": self.frozen_dtype_name}

    @property
    def frozen_dtype_name(self) -> str:
        """dtype of the frozen LLM's weights, e.g. ``"float32"``."""
        return dtype_name(next(self.llm.parameters()).dtype)

    def next_token_logits(self, embeds: torch.Tensor, **kwargs: Any) -> tuple[torch.Tensor, Any]:
        """fp32 next-token logits (V,) at the last position of ``embeds`` (1, L, D), and the KV.

        ``kwargs`` go to the forward (``past_key_values``, ``use_cache``).
        """
        parts = self._fp32_parts()
        if parts is not None:
            decoder, head = parts
            out = decoder(inputs_embeds=embeds, **kwargs)
            last = out.last_hidden_state[0, -1:]
            return head_mod.fp32_head_logits(last, head)[0], out.past_key_values
        keep: dict[str, int] = {"logits_to_keep": 1} if self.score_last_logits_only else {}
        out = self.llm(inputs_embeds=embeds, **kwargs, **keep)
        return out.logits[0, -1].float(), out.past_key_values

    def _answer_token_flags(self, target: str, span: tuple[int, int]) -> list[float]:
        """1.0 for target tokens whose characters overlap ``span`` (EOS appended as 0.0)."""
        enc = self.tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != self.tokenizer.encode(target, add_special_tokens=False):
            raise RuntimeError("offset encoding differs from the training encoding")
        flags = [float(a < span[1] and b > span[0]) for a, b in enc["offset_mapping"]]
        if self.tokenizer.eos_token_id is not None:
            flags.append(0.0)
        return flags

    def _visual_batch(self, images: list[Image.Image | None]) -> list[torch.Tensor | None]:
        """Visual tokens (N, D_llm) per image; ``None`` (no image) stays None."""
        present = [im for im in images if im is not None]
        if present and self.feature_cache_dir is not None:
            feats = iter(self._project(self._cached_tower(present)))
        else:
            feats = iter(self.encode_images(self.preprocess(present))) if present else iter(())
        out: list[torch.Tensor | None] = []
        for im in images:
            out.append(None if im is None else next(feats))
        return out

    # ------------------------------------------------------------------ scoring
    @contextmanager
    def scoring_mode(self) -> Iterator[None]:
        """Eval mode for every submodule inside the block; each module's flag is restored."""
        flags = [(m, m.training) for m in self.modules()]
        self.eval()
        try:
            yield
        finally:
            for module, was_training in flags:
                module.training = was_training

    def clear_dtype_caches(self) -> None:
        """Drop the prefix KV cache and the fp32 head copy."""
        self.clear_cache()
        head_mod.clear_fp32_cache(self.llm)

    @contextmanager
    def frozen_dtype(self, dtype: torch.dtype | None) -> Iterator[None]:
        """Upcast the frozen towers in place to the wider ``dtype`` inside the block.

        The casts are exact both ways, so training resumes on identical weights. Trainable
        tensors are untouched; dtype-dependent caches are dropped on entry and exit. The
        upcast weights are the (e.g. bf16-rounded) training weights, so results can differ
        slightly from a model loaded in ``dtype`` from the checkpoint.
        """
        if dtype is None or not any(_widens(t.dtype, dtype) for t in self._frozen_tensors()):
            yield
            return

        params: list[tuple[nn.Parameter, torch.dtype]] = []
        buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
        self.clear_dtype_caches()
        try:
            for tower in (self.vision, self.llm):
                for mod in tower.modules():
                    for p in mod._parameters.values():
                        if p is not None and not p.requires_grad and _widens(p.dtype, dtype):
                            params.append((p, p.dtype))  # tied weights: cast once
                            p.data = p.data.to(dtype)
                    for name, b in mod._buffers.items():
                        if b is not None and _widens(b.dtype, dtype):
                            buffers.append((mod, name, b))  # small: keep the original
                            mod._buffers[name] = b.to(dtype)
            yield
        finally:
            for p, old in params:
                p.data = p.data.to(old)
            for mod, name, b in buffers:
                mod._buffers[name] = b
            self.clear_dtype_caches()

    def _frozen_tensors(self) -> Iterator[torch.Tensor]:
        """Frozen parameters and all buffers of the vision tower and the LLM."""
        for tower in (self.vision, self.llm):
            yield from (p for p in tower.parameters() if not p.requires_grad)
            yield from tower.buffers()

    def _weights_version(self) -> tuple[int, ...]:
        """In-place versions of the trainable tensors."""
        return tuple(p._version for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """Letter log-probs in :meth:`scoring_mode`, reusing the cached image-prefix KV.

        Exact: attention is causal, so the prefix states do not depend on the text after it.
        """
        with self.scoring_mode():
            return self._letter_logprobs(query, n_options)

    def _letter_logprobs(self, query: Query, n_options: int) -> list[float]:
        """:meth:`letter_logprobs` without the scoring-mode wrapper."""
        if query.image is None or not self.use_prefix_cache:
            visual = self._visual_batch([query.image])[0]
            prompt = self.format_prompt(query.text, visual is not None)
            embeds, _ = self.build_inputs(prompt, visual)
            logits = self.next_token_logits(embeds.unsqueeze(0))[0]
        else:
            pre, post = self.format_prompt(query.text, True).split(IMAGE_MARKER)
            cache = self._prefix_cache
            version = self._weights_version()

            # miss: new image object, new header or changed weights
            if (
                cache is None
                or cache[0] is not query.image
                or cache[1] != pre
                or cache[4] != version
            ):
                visual = self._visual_batch([query.image])[0]
                assert visual is not None
                pre_ids = self.tokenizer.encode(pre, add_special_tokens=True)
                prefix = torch.cat([self._embed(pre_ids), visual.to(self._embed(pre_ids).dtype)])
                parts = self._fp32_parts()
                if parts is not None:  # only the KV states are needed
                    pre_out = parts[0](inputs_embeds=prefix.unsqueeze(0), use_cache=True)
                else:
                    keep = {"logits_to_keep": 1} if self.score_last_logits_only else {}
                    pre_out = self.llm(inputs_embeds=prefix.unsqueeze(0), use_cache=True, **keep)
                cache = (query.image, pre, pre_out.past_key_values, prefix.shape[0], version)
                self._prefix_cache = cache

            # trim the previous query's suffix off the cached KV
            kv, prefix_len = cache[2], cache[3]
            extra = kv.get_seq_length() - prefix_len
            if extra > 0:
                kv.crop(-extra)  # DynamicCache.crop: a negative length drops that many tokens
            post_ids = self.tokenizer.encode(post, add_special_tokens=False)
            logits = self.next_token_logits(
                self._embed(post_ids).unsqueeze(0), past_key_values=kv, use_cache=True
            )[0]

        # letter log-prob = log-sum-exp over its token ids (with and without a space)
        logp = torch.log_softmax(logits, dim=-1)
        groups = letter_token_ids(self.tokenizer, n_options)
        return [float(torch.logsumexp(logp[g], dim=0)) for g in groups]

    def clear_cache(self) -> None:
        """Forget cached prefix states (call whenever trainable weights may change)."""
        self._prefix_cache = None

    def trainable_parameters(self) -> int:
        """Number of trainable weights."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def trainable_state(self) -> dict[str, torch.Tensor]:
        """The trainable parameters only (projector, plus LoRA if enabled), on CPU."""
        return {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        """Load a :meth:`trainable_state` checkpoint; clears the prefix KV cache."""
        missing = set(state) - {n for n, _ in self.named_parameters()}
        if missing:
            raise KeyError(f"unknown parameters in checkpoint: {sorted(missing)[:3]}")
        self.load_state_dict(state, strict=False)
        self.clear_cache()  # computed under the old weights

    def enable_lora(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
        target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    ) -> None:
        """Wrap the LLM with LoRA adapters (base weights stay frozen)."""
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(target_modules)
        )
        self.llm = get_peft_model(self.llm, cfg)
        self.clear_cache()
        head_mod.clear_fp32_cache(self.llm)


# ---------------------------------------------------------------------- helpers


def dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``."""
    return str(dtype).removeprefix("torch.")


def _widens(src: torch.dtype, dst: torch.dtype) -> bool:
    """``src`` floating and exactly representable in a different ``dst`` (bf16 -> fp32)."""
    return src.is_floating_point and src != dst and torch.promote_types(src, dst) == dst


def _pad_to_square(img: Image.Image, fill: tuple[int, int, int] = (122, 116, 104)) -> Image.Image:
    """LLaVA-1.5 'pad' mode: centre the image on a square canvas filled with CLIP's mean colour."""
    w, h = img.size
    if w == h:
        return img
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
    return canvas


def _pool_grid(feats: torch.Tensor, k: int) -> torch.Tensor:
    """Average-pool an (B, N, D) square patch grid by k x k (k=1: no-op)."""
    if k == 1:
        return feats
    b, n, d = feats.shape
    side = round(n**0.5)
    if side * side != n:
        raise ValueError(f"cannot pool non-square grid of {n} tokens")
    grid = feats.transpose(1, 2).reshape(b, d, side, side)
    pooled = nn.functional.avg_pool2d(grid, kernel_size=k, stride=k, ceil_mode=True)
    return pooled.flatten(2).transpose(1, 2)


def _right_pad(
    rows: list[tuple[torch.Tensor, torch.Tensor]],
    multiple: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack (embeds, labels) rows into a right-padded batch: embeds, labels, attention mask.

    The length is rounded up to a multiple of ``multiple``. The columns beyond the longest
    row get mask 1: they follow every real token, so causal attention never lets a real token
    see them, and a batch of one keeps an all-ones mask (the unmasked causal attention path).
    """
    real = max(e.shape[0] for e, _ in rows)
    length = -(-real // multiple) * multiple
    d = rows[0][0].shape[1]
    dev, dt = rows[0][0].device, rows[0][0].dtype
    embeds = torch.zeros(len(rows), length, d, device=dev, dtype=dt)
    labels = torch.full((len(rows), length), -100, dtype=torch.long, device=dev)
    mask = torch.zeros(len(rows), length, dtype=torch.long, device=dev)
    for i, (e, lab) in enumerate(rows):
        embeds[i, : e.shape[0]] = e
        labels[i, : lab.shape[0]] = lab
        mask[i, : e.shape[0]] = 1
    mask[:, real:] = 1  # after every real token and never scored
    return embeds, labels, mask
