"""The bridge model (FrozenBridgeVLM) and its training loop, on tiny random models (no downloads).

Covers: only the bridge is trainable; image tokens are inserted at the image marker; the loss
covers the target only; letter scoring, generation, the three training targets, LoRA,
checkpoint selection, and the cmr-stage-a / cmr-stage-b command lines end to end. The tiny
model uses the real Hugging Face classes (CLIP vision tower, Llama decoder, byte-level
tokenizer) with a few thousand parameters, so the code paths of the real 0.5B and 7B systems
run on a CPU in seconds. Other test modules import tiny_model, img, loader, make_items,
write_split and the tiny_checkpoints fixture from here.
"""

from __future__ import annotations

import itertools
import math
import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.slow
transformers = pytest.importorskip("transformers")

from PIL import Image  # noqa: E402

from cmr.data.aokvqa import AOKVQAItem  # noqa: E402
from cmr.models.base import Query  # noqa: E402
from cmr.models.bridge import (  # noqa: E402
    IMAGE_MARKER,
    BridgeSpec,
    FrozenBridgeVLM,
    _pool_grid,
    build_projector,
    load_llava_projector,
)
from cmr.train.bridge_sft import TrainConfig, train_bridge  # noqa: E402
from cmr.train.generate import greedy_generate, rationale_then_letter_logprobs  # noqa: E402
from cmr.train.targets import LETTER_FORMATS, TargetFormat, prompt_for, target_for  # noqa: E402
from tests.conftest import colour_image  # noqa: E402

N_PATCHES = 16  # 32px image, 8px patches -> 4x4 grid


def tiny_tokenizer() -> transformers.PreTrainedTokenizerFast:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    vocab = {t: i for i, t in enumerate(specials + alphabet)}
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    tok.chat_template = (
        "{% for m in messages %}U:{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}A:{% endif %}"
    )
    return tok


def tiny_model(seed: int = 0) -> FrozenBridgeVLM:
    torch.manual_seed(seed)
    vcfg = transformers.CLIPVisionConfig(
        hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2,
        image_size=32, patch_size=8,
    )  # fmt: skip
    vision = transformers.CLIPVisionModel(vcfg)
    tok = tiny_tokenizer()
    lcfg = transformers.LlamaConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=4096,
        initializer_range=0.2,
    )  # fmt: skip
    # init std 0.2, not HF's 0.02: at 0.02 a random LLM gives ~uniform logits for any input
    llm = transformers.LlamaForCausalLM(lcfg)
    proc = transformers.CLIPImageProcessor(
        size={"shortest_edge": 32}, crop_size={"height": 32, "width": 32}
    )
    spec = BridgeSpec(vision_id="tiny", llm_id="tiny", projector_kind="mlp2x_gelu")
    return FrozenBridgeVLM(vision, build_projector("mlp2x_gelu", 16, 32), llm, tok, proc, spec)


@pytest.fixture(scope="module")
def model() -> FrozenBridgeVLM:
    return tiny_model()


def img(i: int) -> Image.Image:
    return colour_image(i).resize((40, 30))  # non-square: exercises pad-to-square


def test_only_projector_is_trainable(model: FrozenBridgeVLM) -> None:
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(n.startswith("projector.") for n in trainable)
    assert model.trainable_parameters() == sum(p.numel() for p in model.projector.parameters())


def test_visual_tokens_are_spliced_at_marker(model: FrozenBridgeVLM) -> None:
    vis = model.encode_images(model.preprocess([img(1)]))
    assert vis.shape == (1, N_PATCHES, 32)  # CLS dropped
    prompt = model.format_prompt("Q?", has_image=True)
    assert IMAGE_MARKER in prompt
    embeds, labels = model.build_inputs(prompt, vis[0], target="B")
    n_text = len(model.tokenizer.encode(prompt.replace(IMAGE_MARKER, ""), add_special_tokens=True))
    assert embeds.shape[0] == n_text + N_PATCHES + 2  # + "B" + eos
    assert (labels != -100).sum() == 2
    with pytest.raises(ValueError):
        model.build_inputs(prompt, None)


def test_gradients_reach_only_the_projector() -> None:
    m = tiny_model()
    loss = m.loss(["What colour?"], [img(3)], ["B"])
    loss.backward()
    assert all(p.grad is not None for p in m.projector.parameters())
    assert all(p.grad is None for p in m.llm.parameters())
    assert all(p.grad is None for p in m.vision.parameters())


def test_loss_masks_prompt_and_supports_no_image_rows(model: FrozenBridgeVLM) -> None:
    loss = model.loss(["Q one?", "Q two, longer prompt?"], [img(1), None], ["A", "C because"])
    assert torch.isfinite(loss)
    weighted = model.loss(
        ["Q?", "Q?"], [img(1), img(2)], ["A", "B"], weights=torch.tensor([1.0, 0.0])
    )
    single = model.loss(["Q?"], [img(1)], ["A"])
    assert weighted.item() == pytest.approx(single.item(), rel=1e-4)


def test_letter_logprobs_are_valid(model: FrozenBridgeVLM) -> None:
    for image in (img(1), None):
        lp = model.letter_logprobs(Query("Pick:\nA. x\nB. y\nC. z\nD. w", image), 4)
        assert len(lp) == 4 and all(math.isfinite(v) and v <= 0 for v in lp)


def test_projector_loads_llava_checkpoint(tmp_path: Path) -> None:
    src = build_projector("mlp2x_gelu", 16, 32)
    state = {f"model.mm_projector.{k}": v for k, v in src.state_dict().items()}
    torch.save(state, tmp_path / "mm_projector.bin")
    dst = build_projector("mlp2x_gelu", 16, 32)
    load_llava_projector(dst, str(tmp_path / "mm_projector.bin"))
    for a, b in zip(src.state_dict().values(), dst.state_dict().values(), strict=True):
        assert torch.equal(a, b)
    torch.save({"other": torch.zeros(1)}, tmp_path / "bad.bin")
    with pytest.raises(ValueError):
        load_llava_projector(dst, str(tmp_path / "bad.bin"))


def test_pool_grid() -> None:
    x = torch.arange(2 * 16 * 3, dtype=torch.float32).view(2, 16, 3)
    assert _pool_grid(x, 1) is x
    assert _pool_grid(x, 2).shape == (2, 4, 3)
    with pytest.raises(ValueError):
        _pool_grid(torch.zeros(1, 15, 3), 2)


def make_items(n: int = 8) -> list[AOKVQAItem]:
    return [
        AOKVQAItem(
            question_id=f"t{i}",
            image_id=i,
            question="Which?",
            choices=("w", "x", "y", "z"),
            correct_choice_idx=i % 4,
            direct_answers=(),
            difficult_direct_answer=False,
            rationales=(f"r{i}a", f"r{i}b"),
            split="train",
        )
        for i in range(n)
    ]


def test_targets_formats() -> None:
    it = make_items(3)[2]
    rng = random.Random(0)
    assert target_for(it, TargetFormat.ANSWER_ONLY, rng) == "C"
    assert target_for(it, TargetFormat.ANSWER_THEN_RATIONALE, rng).startswith("C\nBecause: r2")
    assert target_for(it, TargetFormat.RATIONALE_THEN_ANSWER, rng).endswith("Answer: C")
    assert prompt_for(it, TargetFormat.ANSWER_ONLY) != prompt_for(
        it, TargetFormat.RATIONALE_THEN_ANSWER
    )


def loader(it: AOKVQAItem) -> Image.Image:
    # The answer is a function of the image (image_id % 4), so only vision can solve it.
    return colour_image(10 + 40 * it.image_id)


@pytest.mark.parametrize("margin", [0.0, 0.5])  # 0.5: with the image-margin loss (Table 4.15)
def test_training_reduces_loss_and_saves_projector(tmp_path: Path, margin: float) -> None:
    m = tiny_model(seed=1)
    before = [p.detach().clone() for p in m.llm.parameters()]
    items = make_items(8)
    cfg = TrainConfig(
        target=TargetFormat.ANSWER_ONLY, epochs=12, batch_size=4, grad_accum=1, lr=3e-2,
        eval_every=10, margin_lambda=margin,
    )  # fmt: skip
    res = train_bridge(m, items, loader, cfg, tmp_path, dev_items=items[:4])
    hist = res["history"]
    assert hist[-1]["ce"] < 0.7 * hist[0]["ce"]
    assert (tmp_path / "projector.pt").exists() and (tmp_path / "train_log.jsonl").exists()
    assert all(torch.equal(a, b) for a, b in zip(before, m.llm.parameters(), strict=True))
    if margin:
        assert any(h["margin"] > 0 for h in hist)


def test_generation_paths(model: FrozenBridgeVLM) -> None:
    q = Query(prompt_for(make_items(1)[0], TargetFormat.RATIONALE_THEN_ANSWER), img(2))
    text = greedy_generate(model, q, max_new_tokens=5)
    assert isinstance(text, str)
    lp = rationale_then_letter_logprobs(model, q, 4, max_new_tokens=5)
    assert len(lp) == 4 and all(math.isfinite(v) for v in lp)


# --------------------------------------------------------------------------- end-to-end CLI


@pytest.fixture(scope="module")
def tiny_checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Save the tiny model as real HF checkpoints + a BridgeSpec JSON (no network)."""
    root = tmp_path_factory.mktemp("ckpt")
    m = tiny_model(seed=2)
    m.vision.save_pretrained(root / "vision")
    m.image_processor.save_pretrained(root / "vision")
    m.llm.save_pretrained(root / "llm")
    m.tokenizer.save_pretrained(root / "llm")
    projector = {f"model.mm_projector.{k}": v for k, v in m.projector.state_dict().items()}
    torch.save(projector, root / "mm_projector.bin")
    spec = {
        "_comment": "tiny test stack",
        "vision_id": str(root / "vision"),
        "llm_id": str(root / "llm"),
        "projector_path": str(root / "mm_projector.bin"),
        "prompt_style": "chat_template",
    }
    (root / "spec.json").write_text(__import__("json").dumps(spec))
    return root


def write_split(aokvqa: Path, coco: Path, split: str, n: int) -> None:
    import json

    recs = []
    (coco / f"{split}2017").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        image_id = (0 if split == "train" else 500) + i
        loader_img = colour_image(10 + 40 * image_id)
        loader_img.save(coco / f"{split}2017" / f"{image_id:012d}.jpg")
        recs.append(
            {
                "question_id": f"{split}{i}",
                "image_id": image_id,
                "question": "Which?",
                "choices": ["w", "x", "y", "z"],
                "correct_choice_idx": image_id % 4,
                "direct_answers": ["w"] * 10,
                "difficult_direct_answer": False,
                "rationales": [f"because {i}"],
                "split": split,
            }
        )
    (aokvqa / f"aokvqa_v1p0_{split}.json").write_text(json.dumps(recs))


@pytest.mark.parametrize(  # the last target: A→R with another question's rationale (Table 4.15)
    "target", ["answer_then_rationale", "rationale_then_answer", "answer_then_mismatched_rationale"]
)
def test_stage_b_cli_end_to_end(tiny_checkpoints: Path, tmp_path: Path, target: str) -> None:
    from cmr.runners import stage_b

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 40)
    write_split(aokvqa, coco, "val", 8)
    out = tmp_path / "run"
    stage_b.main(
        [
            "--bridge", str(tiny_checkpoints / "spec.json"), "--target", target,
            "--aokvqa-dir", str(aokvqa), "--coco-dir", str(coco), "--epochs", "1",
            "--batch-size", "4", "--grad-accum", "1", "--eval-every", "5",
            "--margin-lambda", "0.1", "--conditions", "real,none,swapped,blank",
            "--out", str(out),
        ]
    )  # fmt: skip
    assert (out / "projector.pt").exists()
    summary = (out / "eval_val" / "summary.md").read_text()
    for cond in ("real", "none", "swapped", "blank"):
        assert f"| {cond} |" in summary
    run_args = __import__("json").loads((out / "run_args.json").read_text())
    assert run_args["head_dtype"] == "fp32" and run_args["model_input"]["head_dtype"] == "fp32"


def test_stage_a_cli_with_bridge_model(tiny_checkpoints: Path, tmp_path: Path) -> None:
    import json

    from cmr.runners import stage_a

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "val", 6)
    stage_a.main(
        [
            "--model", f"bridge:{tiny_checkpoints / 'spec.json'}", "--aokvqa-dir", str(aokvqa),
            "--coco-dir", str(coco), "--conditions", "real,swapped", "--out", str(tmp_path / "b"),
        ]
    )  # fmt: skip
    assert "| swapped |" in (tmp_path / "b" / "summary.md").read_text()
    cfg = json.loads((tmp_path / "b" / "config.json").read_text())
    assert cfg["model_input"]["head_dtype"] == "fp32"  # default, recorded
    stage_a.main(
        [
            "--model", f"bridge:{tiny_checkpoints / 'spec.json'}", "--aokvqa-dir", str(aokvqa),
            "--conditions", "none", "--contexts", "none,rationale", "--out", str(tmp_path / "t"),
            "--head-dtype", "model",
        ]
    )  # fmt: skip
    assert "rationale" in (tmp_path / "t" / "summary.md").read_text()
    cfg = json.loads((tmp_path / "t" / "config.json").read_text())
    assert cfg["meta"]["head_dtype"] == "model" and cfg["model_input"]["head_dtype"] == "model"


# "stage C": LoRA adapters on the LLM, trained together with the bridge (Table 4.15)
def test_stage_c_lora_trains_adapters_but_not_base_llm(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    m = tiny_model(seed=3)
    base = {n: p.detach().clone() for n, p in m.llm.named_parameters()}
    proj_only = m.trainable_parameters()
    m.enable_lora(r=4, alpha=8)
    names = [n for n, p in m.named_parameters() if p.requires_grad]
    assert any("lora_" in n for n in names) and m.trainable_parameters() > proj_only
    items = make_items(8)
    cfg = TrainConfig(epochs=2, batch_size=4, grad_accum=1, lr=1e-2, eval_every=100)
    train_bridge(m, items, loader, cfg, tmp_path)
    for n, p in m.llm.named_parameters():  # peft renames base weights with a prefix
        if "lora_" in n:
            continue
        key = n.replace("base_model.model.", "").replace(".base_layer", "")
        assert torch.equal(p, base[key]), n
    state = torch.load(tmp_path / "trainable.pt")
    assert any("lora_" in k for k in state)
    fresh = tiny_model(seed=3)
    fresh.enable_lora(r=4, alpha=8)
    fresh.load_trainable_state(state)
    with pytest.raises(KeyError):
        fresh.load_trainable_state({"nope": torch.zeros(1)})


def test_stage_c_cli(tiny_checkpoints: Path, tmp_path: Path) -> None:
    pytest.importorskip("peft")
    from cmr.runners import stage_b

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 24)
    write_split(aokvqa, coco, "val", 6)
    stage_b.main(
        [
            "--bridge", str(tiny_checkpoints / "spec.json"), "--target", "answer_only",
            "--aokvqa-dir", str(aokvqa), "--coco-dir", str(coco), "--batch-size", "4",
            "--grad-accum", "1", "--lora-r", "4", "--conditions", "real,none",
            "--out", str(tmp_path / "c"),
        ]
    )  # fmt: skip
    assert (tmp_path / "c" / "eval_val" / "summary.md").exists()


def test_rationale_first_scoring_context_matches_training(model: FrozenBridgeVLM) -> None:
    """The scored position must see exactly the tokens it saw in training (no stray space)."""
    from cmr.train.bridge_sft import _prefix_before_answer

    target = "Because: it is red.\nAnswer: B"
    prefix = _prefix_before_answer(target, TargetFormat.RATIONALE_THEN_ANSWER)
    vis = model.encode_images(model.preprocess([img(1)]))[0]
    prompt = model.format_prompt("Q?", has_image=True)
    train_embeds, labels = model.build_inputs(prompt, vis, target=target)
    eval_embeds, _ = model.build_inputs(prompt, vis, assistant_prefix=prefix)
    n = eval_embeds.shape[0]
    assert torch.equal(train_embeds[:n], eval_embeds)  # identical context up to the letter
    letter_pos = int((labels != -100).nonzero()[0]) + len(
        model.tokenizer.encode(prefix, add_special_tokens=False)
    )
    assert letter_pos == n  # the next training label is the answer letter itself


def test_projector_stays_fp32_trainable_under_bf16_llm() -> None:
    m = tiny_model()
    m.llm.to(torch.bfloat16)
    loss = m.loss(["Q?"], [img(1)], ["B"])
    loss.backward()
    assert all(p.dtype == torch.float32 and p.grad is not None for p in m.projector.parameters())


def test_rationale_then_answer_training_logs_the_balanced_loss_parts(tmp_path: Path) -> None:
    m = tiny_model(seed=4)
    cfg = TrainConfig(
        target=TargetFormat.RATIONALE_THEN_ANSWER, epochs=2, batch_size=4, grad_accum=1,
        lr=1e-2, eval_every=2, rationale_weight=0.5,
    )  # fmt: skip
    res = train_bridge(m, make_items(8), loader, cfg, tmp_path, dev_items=make_items(4))
    assert res["best_dev_acc"] >= 0.0 and (tmp_path / "projector.pt").exists()
    row = res["history"][0]
    assert row["ce"] == pytest.approx(row["ce_answer"] + 0.5 * row["ce_rest"], rel=1e-4)
    evals = [h for h in res["history"] if "dev_acc" in h]
    assert evals and all(h["dev_answer_ce_tf"] > 0 and h["dev_answer_nll"] > 0 for h in evals)


def test_prefix_kv_cache_is_exact_and_invalidated() -> None:
    m = tiny_model(seed=4)
    m.projector.eval()
    image = img(5)
    prompts = [f"Q{k}?\nA. a\nB. b\nC. c\nD. d" for k in range(3)]

    @torch.no_grad()
    def uncached(text: str) -> list[float]:
        embeds, _ = m.build_inputs(m.format_prompt(text, True), m._visual_batch([image])[0])
        logp = torch.log_softmax(m.llm(inputs_embeds=embeds.unsqueeze(0)).logits[0, -1], -1)
        from cmr.eval.prompts import letter_token_ids

        return [float(torch.logsumexp(logp[g], 0)) for g in letter_token_ids(m.tokenizer, 4)]

    for text in prompts * 2:  # repeated use of one cached prefix, in mixed order
        assert m.letter_logprobs(Query(text, image), 4) == pytest.approx(uncached(text), abs=1e-5)
    assert m._prefix_cache is not None and m._prefix_cache[0] is image
    with torch.no_grad():  # weights change -> stale cache must not be used
        for p in m.projector.parameters():
            p.add_(0.5)
    m.clear_cache()
    assert m.letter_logprobs(Query(prompts[0], image), 4) == pytest.approx(
        uncached(prompts[0]), abs=1e-5
    )
    m.loss(["Q?"], [image], ["A"])
    assert m._prefix_cache is None  # a training step always clears it


def test_letter_logprobs_match_an_explicit_forward_pass() -> None:
    from cmr.eval.prompts import letter_token_ids

    m = tiny_model(seed=5)
    m.projector.eval()
    image = img(3)
    with torch.no_grad():
        vis = m.encode_images(m.preprocess([image]))[0]
    q = Query("Pick:\nA. x\nB. y\nC. z\nD. w", image)

    @torch.no_grad()
    def explicit(text: str) -> list[float]:
        embeds, _ = m.build_inputs(m.format_prompt(text, True), vis)
        logp = torch.log_softmax(m.llm(inputs_embeds=embeds.unsqueeze(0)).logits[0, -1], -1)
        return [float(torch.logsumexp(logp[g], 0)) for g in letter_token_ids(m.tokenizer, 4)]

    for _ in range(2):  # second call goes through the prefix KV cache
        assert m.letter_logprobs(q, 4) == pytest.approx(explicit(q.text), abs=1e-5)
    assert isinstance(greedy_generate(m, q, max_new_tokens=3), str)


def test_loss_parts_split_answer_letter_from_rationale(model: FrozenBridgeVLM) -> None:
    from cmr.train.targets import answer_char_span

    it = make_items(3)[2]
    rng = random.Random(0)
    fmts = list(LETTER_FORMATS)  # the targets that contain an answer letter
    targets = [target_for(it, f, rng) for f in fmts]
    prompts = [prompt_for(it, f) for f in fmts]
    images = [img(1)] * len(fmts)
    spans = [answer_char_span(t, f) for t, f in zip(targets, fmts, strict=True)]
    with torch.no_grad():
        parts = model.loss_parts(prompts, images, targets, spans)
        legacy = model.loss(prompts, images, targets)
    assert parts.answer is not None and parts.rest is not None
    assert parts.all.mean().item() == pytest.approx(legacy.item(), rel=1e-6)
    for k, (tgt, span) in enumerate(zip(targets, spans, strict=True)):
        flags = model._answer_token_flags(tgt, span)
        ids = model.tokenizer.encode(tgt, add_special_tokens=False)
        assert sum(flags) == 1  # byte-level tiny tokenizer: the letter is one token
        assert model.tokenizer.decode([ids[flags.index(1.0)]]) == "C"
        # answer CE = -log p(letter token | everything before it), computed by hand
        vis = model.encode_images(model.preprocess([img(1)]))[0]
        embeds, labels = model.build_inputs(model.format_prompt(prompts[k], True), vis, tgt)
        pos = int((labels != -100).nonzero()[0]) + flags.index(1.0)
        with torch.no_grad():
            logits = model.llm(inputs_embeds=embeds.unsqueeze(0)).logits[0, pos - 1].float()
        ce = -torch.log_softmax(logits, -1)[labels[pos]]
        assert parts.answer[k].item() == pytest.approx(ce.item(), rel=1e-4)


def test_last_position_logits_only_does_not_change_scores() -> None:
    m = tiny_model(seed=6)
    m.projector.eval()
    text = "Pick:\nA. x\nB. y\nC. z\nD. w"
    for image in (None, img(3)):  # full pass, and cached prefix + suffix passes
        fast = m.letter_logprobs(Query(text, image), 4)
        m.score_last_logits_only = False
        m.clear_cache()
        full = m.letter_logprobs(Query(text, image), 4)
        m.score_last_logits_only = True
        m.clear_cache()
        assert fast == pytest.approx(full, abs=1e-5)


@pytest.mark.parametrize("select", ["last", "dev_logp"])
def test_checkpoint_rules(tmp_path: Path, select: str) -> None:
    m = tiny_model(seed=7)
    cfg = TrainConfig(
        epochs=2, batch_size=4, grad_accum=1, lr=1e-2, eval_every=1, select=select,  # type: ignore[arg-type]
        dev_teacher_forced=False,
    )  # fmt: skip
    res = train_bridge(m, make_items(8), loader, cfg, tmp_path, dev_items=make_items(4))
    hist = res["history"]
    chosen = [h["step"] for h in hist if h.get("selected")]
    assert res["select"] == select and chosen
    if select == "last":
        assert chosen == [hist[-1]["step"]]
        saved = torch.load(tmp_path / "projector.pt")
        assert all(torch.equal(v, m.projector.state_dict()[k]) for k, v in saved.items())
    else:  # each newly selected checkpoint has a strictly better dev log p(gold)
        nll = [h["dev_answer_nll"] for h in hist if h.get("selected")]
        assert all(a > b for a, b in itertools.pairwise(nll))
        assert min(h["dev_answer_nll"] for h in hist) == nll[-1]


# --eval-only scores the untrained bridge θ₀, as for Table 4.1
def test_stage_b_eval_only_scores_theta0_without_training(
    tiny_checkpoints: Path, tmp_path: Path
) -> None:
    from cmr.runners import stage_b

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 8)
    write_split(aokvqa, coco, "val", 4)
    out = tmp_path / "theta0"
    stage_b.main(
        [
            "--bridge", str(tiny_checkpoints / "spec.json"), "--target", "answer_then_rationale",
            "--aokvqa-dir", str(aokvqa), "--coco-dir", str(coco), "--eval-only",
            "--conditions", "real,swapped", "--rotations", "2", "--out", str(out),
        ]
    )  # fmt: skip
    assert not (out / "projector.pt").exists() and not (out / "train_log.jsonl").exists()
    assert "| swapped |" in (out / "eval_val" / "summary.md").read_text()
