"""The LLM's output layer (LM head) applied in fp32 at the scored positions, on tiny models.

During bf16 training the output layer runs in fp32 for the loss (Section 3.4);
experiments/analysis/head_precision.py measures the effect on real models.
(a) fp32 model: the fp32-head path equals the model's own head.
(b) bf16 model: bridge scores are identical for logits_to_keep=1 vs full and cached vs
    uncached, and the fp32-head training loss is closer to a float64 head than the bf16 one.
(c) tied input/output embeddings.
(d) architectures outside the allow-list (and non-plain heads) keep the model's head.
Also: scoring with the prefix KV cache, and resuming refuses to mix predictions made with
different scoring settings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from cmr.models import head as head_mod  # noqa: E402
from cmr.models.base import Query  # noqa: E402
from cmr.models.bridge import FrozenBridgeVLM  # noqa: E402
from tests.test_bridge import (  # noqa: E402, F401
    N_PATCHES,
    img,
    tiny_checkpoints,
    tiny_model,
    tiny_tokenizer,
)

TEXT = "Pick one:\nA. x\nB. y\nC. z\nD. w"


def tiny_lm(kind: str, dtype: Any = None, tied: bool = False, seed: int = 0) -> Any:
    torch.manual_seed(seed)
    vocab = len(tiny_tokenizer())
    if kind == "gpt2":
        cfg: Any = transformers.GPT2Config(
            vocab_size=vocab, n_embd=32, n_layer=2, n_head=4, initializer_range=0.2
        )
        model = transformers.GPT2LMHeadModel(cfg)
    else:
        config_cls, model_cls = {
            "llama": (transformers.LlamaConfig, transformers.LlamaForCausalLM),
            "qwen2": (transformers.Qwen2Config, transformers.Qwen2ForCausalLM),
        }[kind]
        cfg = config_cls(
            vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, initializer_range=0.2,
            tie_word_embeddings=tied, max_position_embeddings=512,
        )  # fmt: skip
        model = model_cls(cfg)
    return model.to(dtype or torch.float32).eval()


def bridge_with_llm(llm: Any, seed: int = 0) -> FrozenBridgeVLM:
    m = tiny_model(seed=seed)
    m.llm = llm
    for p in llm.parameters():
        p.requires_grad_(False)
    m.projector.eval()  # enables the prefix KV cache for image queries
    return m


def scores(m: FrozenBridgeVLM, image: Any, **attrs: Any) -> list[float]:
    for k, v in attrs.items():
        setattr(m, k, v)
    m.clear_cache()
    first = m.letter_logprobs(Query(TEXT, image), 4)
    if image is not None:  # second call: suffix pass over the cached prefix
        assert m.letter_logprobs(Query(TEXT, image), 4) == first
    return first


def max_diff(a: list[float], b: list[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=True))


# ------------------------------------------------------------------------------- (a)


def test_fp32_head_equals_model_head_on_fp32_model() -> None:
    m = tiny_model(seed=7)
    m.projector.eval()
    assert m.head_fp32_active and m.describe()["head_dtype"] == "fp32"
    for image in (None, img(2)):
        new = scores(m, image, head_fp32=True)
        old = scores(m, image, head_fp32=False)
        assert max_diff(new, old) <= 1e-6
    # training loss: target-position head in fp32 == the head's own F.linear in fp32
    m.projector.train()
    args = (["Q?", "R?"], [img(1), None], ["B because x.", "C"])
    m.head_fp32 = True
    new_loss = m.loss_parts(*args).all
    m.head_fp32 = False
    old_loss = m.loss_parts(*args).all
    assert torch.allclose(new_loss, old_loss, rtol=1e-6, atol=0)


# ------------------------------------------------------------------------------- (b)


def test_bf16_bridge_scores_do_not_depend_on_logits_to_keep() -> None:
    m = bridge_with_llm(tiny_lm("llama", torch.bfloat16, seed=3), seed=3)
    m.vision.to(torch.bfloat16)
    assert m.head_fp32_active
    for image in (None, img(4)):  # one full pass; cached prefix + suffix pass
        fast = scores(m, image, score_last_logits_only=True)
        full = scores(m, image, score_last_logits_only=False)
        assert fast == full  # bit-identical floats


def test_bf16_target_nll_uses_fp32_head() -> None:
    m = bridge_with_llm(tiny_lm("llama", torch.bfloat16, seed=4), seed=4)
    torch.manual_seed(0)
    embeds = (torch.randn(2, 9, 32) * 0.5).to(torch.bfloat16)
    mask = torch.ones(2, 9, dtype=torch.long)
    mask[1, 7:] = 0
    gold = torch.randint(0, m.llm.config.vocab_size, (2, 8))
    gold[:, :3] = -100
    gold[1, 6:] = -100
    with torch.no_grad():
        hidden = m.llm.model(inputs_embeds=embeds, attention_mask=mask).last_hidden_state
        logits = hidden[:, :-1].double() @ m.llm.lm_head.weight.double().T
        exact = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), gold.reshape(-1), reduction="none"
        ).view(gold.shape)
        nll_fp32 = m._target_nll(embeds, mask, gold)
        m.head_fp32 = False
        nll_bf16 = m._target_nll(embeds, mask, gold)
    scored = gold != -100
    err_fp32 = float((nll_fp32.double() - exact)[scored].abs().max())
    err_bf16 = float((nll_bf16.double() - exact)[scored].abs().max())
    assert err_fp32 <= 1e-5 and err_bf16 > 10 * err_fp32
    assert bool((nll_fp32[~scored] == 0).all())


# ------------------------------------------------------------------------------- (c)


def test_tied_embeddings() -> None:
    llm = tiny_lm("qwen2", torch.bfloat16, tied=True, seed=6)
    assert llm.lm_head.weight is llm.model.embed_tokens.weight
    m = bridge_with_llm(llm, seed=6)
    m.vision.to(torch.bfloat16)
    assert m.head_fp32_active
    for image in (None, img(5)):
        assert scores(m, image, score_last_logits_only=True) == scores(
            m, image, score_last_logits_only=False
        )


# ------------------------------------------------------------------------------- (d)


def test_unknown_architecture_falls_back_to_model_head() -> None:
    gpt2 = tiny_lm("gpt2", torch.bfloat16, seed=7)
    assert head_mod.plain_head_base(gpt2) is None and head_mod.fp32_head(gpt2) is None


def test_bridge_falls_back_for_unlisted_llm_and_non_plain_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    m = bridge_with_llm(tiny_lm("llama", torch.bfloat16, seed=8), seed=8)
    m.vision.to(torch.bfloat16)
    off = [scores(m, image, head_fp32=False) for image in (None, img(6))]
    m.head_fp32 = True
    assert m.head_fp32_active
    monkeypatch.setattr(head_mod, "PLAIN_HEAD_ARCHS", frozenset())
    assert not m.head_fp32_active
    assert m.describe() == {
        "head_dtype_requested": "fp32",
        "head_dtype": "model",
        "prefix_cache": True,
        "dtype": "bfloat16",
    }
    assert [scores(m, image) for image in (None, img(6))] == off
    monkeypatch.undo()

    class QuantLinear(torch.nn.Linear):  # e.g. bitsandbytes Linear4bit subclasses Linear
        pass

    assert head_mod.fp32_capable(m.llm.lm_head)
    assert not head_mod.fp32_capable(QuantLinear(4, 4))
    m.llm.lm_head.__class__ = QuantLinear
    assert not m.head_fp32_active and m._decoder_and_head() is not None
    assert [scores(m, image) for image in (None, img(6))] == off


def test_config_hash_keys_fp32_head_only_when_it_changes_arithmetic() -> None:
    from cmr.train.bridge_sft import TrainConfig, config_hash

    cfg = TrainConfig()

    def hashes(m: FrozenBridgeVLM) -> tuple[str, str]:
        m.head_fp32 = True
        on = config_hash(cfg, m, [])
        m.head_fp32 = False
        return on, config_hash(cfg, m, [])

    on, off = hashes(tiny_model(seed=9))  # fp32 LLM: no-op, resume.pt files stay valid
    assert on == off
    on, off = hashes(bridge_with_llm(tiny_lm("llama", torch.bfloat16), seed=9))
    assert on != off


def test_fp32_weight_copy_is_cached_and_invalidated() -> None:
    head = torch.nn.Linear(8, 50, bias=True).to(torch.bfloat16).requires_grad_(False)
    h = torch.randn(3, 8)
    with torch.inference_mode():  # as in the HF adapters
        first = head_mod.fp32_head_logits(h, head)
    copy = head.__dict__[head_mod._CACHE_ATTR][1]
    assert not copy.is_inference() and copy.dtype == torch.float32
    exact = h.double() @ head.weight.double().T + head.bias.double()
    assert torch.allclose(first.double(), exact, atol=1e-5)
    hg = h.clone().requires_grad_(True)  # training: the cached copy is saved for backward
    head_mod.fp32_head_logits(hg, head).sum().backward()
    assert head.__dict__[head_mod._CACHE_ATTR][1] is copy and hg.grad is not None
    with torch.no_grad():
        head.weight.mul_(2)  # in-place update (e.g. load_state_dict) -> fresh copy
    again = head_mod.fp32_head_logits(h, head)
    assert head.__dict__[head_mod._CACHE_ATTR][1] is not copy
    assert torch.allclose(again - head.bias.float(), 2 * (first - head.bias.float()), atol=1e-5)
    assert head_mod._CACHE_ATTR not in head.state_dict()


# ----------------------------------------------------------------------- scoring mode


def decoder_lengths(m: FrozenBridgeVLM) -> tuple[list[int], Any]:
    """Spy: sequence length of every decoder call (prefix pass vs cached suffix pass)."""
    lengths: list[int] = []

    def spy(_mod: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        lengths.append(int(kwargs["inputs_embeds"].shape[1]))

    return lengths, m.llm.model.register_forward_pre_hook(spy, with_kwargs=True)


@torch.no_grad()
def full_pass(m: FrozenBridgeVLM, text: str, image: Any) -> list[float]:
    """Reference: one uncached forward of the whole prompt, the model's own fp32 head."""
    from cmr.eval.prompts import letter_token_ids

    with m.scoring_mode():
        embeds, _ = m.build_inputs(m.format_prompt(text, True), m._visual_batch([image])[0])
        logits = m.llm(inputs_embeds=embeds.unsqueeze(0)).logits[0, -1]
    logp = torch.log_softmax(logits, -1)
    return [float(torch.logsumexp(logp[g], 0)) for g in letter_token_ids(m.tokenizer, 4)]


def test_scoring_in_train_mode_uses_prefix_cache_and_restores_modes() -> None:
    m = tiny_model(seed=10)
    m.train()  # as the training loop has it
    flags = {name: mod.training for name, mod in m.named_modules()}
    image = img(7)
    prompts = [f"Q{k}?\nA. a\nB. b\nC. c\nD. d" for k in range(3)]
    lengths, handle = decoder_lengths(m)
    try:
        got = [m.letter_logprobs(Query(t, image), 4) for t in prompts]
    finally:
        handle.remove()
    assert {name: mod.training for name, mod in m.named_modules()} == flags  # restored
    # one prefix pass ([header + visual tokens]), then one suffix-only pass per prompt
    assert m._prefix_cache is not None
    prefix_len = m._prefix_cache[3]
    with torch.no_grad():
        full_lens = [
            m.build_inputs(m.format_prompt(t, True), m._visual_batch([image])[0])[0].shape[0]
            for t in prompts
        ]
    assert lengths[0] == prefix_len > N_PATCHES
    assert lengths[1:] == [n - prefix_len for n in full_lens]
    for text, lp in zip(prompts, got, strict=True):
        assert max_diff(lp, full_pass(m, text, image)) <= 1e-6  # fp32: KV reuse is exact
    m.use_prefix_cache = False  # A/B switch: one full pass per query
    lengths, handle = decoder_lengths(m)
    try:
        uncached = m.letter_logprobs(Query(prompts[0], image), 4)
    finally:
        handle.remove()
    assert lengths == full_lens[:1] and max_diff(uncached, got[0]) <= 1e-6


def test_prefix_cache_is_invalidated_by_an_optimizer_step() -> None:
    m = tiny_model(seed=11)
    m.train()
    image, text = img(8), TEXT
    before = m.letter_logprobs(Query(text, image), 4)
    opt = torch.optim.SGD(m.projector.parameters(), lr=0.5)
    for p in m.projector.parameters():
        p.grad = torch.ones_like(p)
    opt.step()  # in place, no clear_cache(): the weight versions must catch it
    after = m.letter_logprobs(Query(text, image), 4)
    assert max_diff(after, full_pass(m, text, image)) <= 1e-5
    assert max_diff(after, before) > 1e-3


def test_dev_metrics_scores_in_eval_mode_and_restores_train_mode() -> None:
    from cmr.train.bridge_sft import dev_metrics
    from cmr.train.targets import TargetFormat
    from tests.test_bridge import loader, make_items

    m = tiny_model(seed=12)
    m.projector.train()  # as in the training loop
    m.llm.train()  # a train-mode LLM (e.g. LoRA dropout active) must be scored in eval mode
    flags = {name: mod.training for name, mod in m.named_modules()}
    seen: list[bool] = []
    orig = m._letter_logprobs

    def spy(query: Query, n: int) -> list[float]:
        seen.append(m.projector.training or m.llm.training)
        return orig(query, n)

    m._letter_logprobs = spy  # type: ignore[method-assign]
    out = dev_metrics(m, make_items(2), loader, TargetFormat.ANSWER_ONLY)
    assert seen == [False, False] and "dev_answer_ce_tf" in out
    assert {name: mod.training for name, mod in m.named_modules()} == flags  # restored
    assert m.projector.training and m.llm.training and not m.vision.training


def test_fp32_weight_copy_is_shared_by_all_scoring_passes() -> None:
    from cmr.train.generate import greedy_generate, rationale_then_letter

    m = bridge_with_llm(tiny_lm("llama", torch.bfloat16, seed=13), seed=13)
    m.vision.to(torch.bfloat16)
    head = m.llm.lm_head
    assert head_mod._CACHE_ATTR not in head.__dict__  # lazy: nothing before scoring
    m.letter_logprobs(Query(TEXT, img(9)), 4)  # prefix (no head) + cached suffix pass
    copy = head.__dict__[head_mod._CACHE_ATTR][1]
    m.letter_logprobs(Query(TEXT), 4)  # uncached full pass
    greedy_generate(m, Query(TEXT, img(9)), max_new_tokens=2)
    rationale_then_letter(m, Query(TEXT, img(9)), 4, max_new_tokens=2)
    assert head.__dict__[head_mod._CACHE_ATTR][1] is copy  # one copy, reused everywhere
    fp32_copies = [t for t in head.__dict__.values() if isinstance(t, tuple)]
    assert len(fp32_copies) == 1


# ------------------------------------------------------------------------ resume guard


def test_stage_a_resume_refuses_a_different_scoring_path(
    tiny_checkpoints: Path,  # noqa: F811  (the fixture imported above)
    tmp_path: Path,
) -> None:
    import json

    from cmr.runners import stage_a
    from tests.test_bridge import write_split

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "val", 4)
    out = tmp_path / "run"
    base = ["--model", f"bridge:{tiny_checkpoints / 'spec.json'}", "--aokvqa-dir", str(aokvqa),
            "--conditions", "none", "--threads", "1", "--out", str(out)]  # fmt: skip
    preds, cfg_path = out / "predictions.jsonl", out / "config.json"

    def n_records() -> int:
        return len(preds.read_text().splitlines())

    stage_a.main([*base, "--limit", "2", "--head-dtype", "model"])
    assert n_records() == 2
    cfg_before = cfg_path.read_text()
    for change in (["--head-dtype", "fp32"], ["--head-dtype", "model", "--threads", "2"]):
        with pytest.raises(SystemExit, match="would mix scoring paths") as err:
            stage_a.main([*base, "--limit", "4", *change])
        assert err.value.code != 0  # a message, not 0
        assert n_records() == 2 and cfg_path.read_text() == cfg_before  # untouched
    torch.set_num_threads(1)
    stage_a.main([*base, "--limit", "4", "--head-dtype", "model"])  # same settings: resumes
    assert n_records() == 4
    # a run whose settings were not recorded (model_input null) cannot be resumed
    legacy = json.loads(cfg_path.read_text())
    legacy["model_input"] = None
    cfg_path.write_text(json.dumps(legacy))
    with pytest.raises(SystemExit, match="not recorded"):
        stage_a.main([*base, "--limit", "4", "--head-dtype", "model"])
    assert n_records() == 4


def test_resume_guard_ignores_settings_that_do_not_apply(tmp_path: Path) -> None:
    from cmr.runners.stage_a import check_resume_scoring, scoring_settings

    preds = tmp_path / "predictions.jsonl"
    hf = {"model_input": {"head_dtype": "fp32"}, "manifest": {"compute": {"torch_num_threads": 4}}}
    check_resume_scoring(preds, None, hf, tmp_path)  # no records yet: nothing to mix
    preds.write_text('{"x": 1}\n')
    check_resume_scoring(preds, hf, hf, tmp_path)  # no prefix cache on either side
    assert scoring_settings(hf)["prefix_cache"] == scoring_settings(None)["prefix_cache"]
    with pytest.raises(SystemExit):
        check_resume_scoring(preds, None, hf, tmp_path)


def test_stage_b_eval_resume_refuses_a_different_scoring_path(
    tiny_checkpoints: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    import json

    from cmr.runners import stage_b
    from tests.test_bridge import write_split

    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "train", 8)
    write_split(aokvqa, coco, "val", 2)
    out = tmp_path / "b"
    (out / "eval_val").mkdir(parents=True)
    preds = out / "eval_val" / "predictions.jsonl"
    preds.write_text('{"question_id": "v0"}\n')
    old = {"model_input": {"head_dtype": "model", "prefix_cache": True},
           "manifest": {"compute": {"torch_num_threads": 1}}}  # fmt: skip
    (out / "run_args.json").write_text(json.dumps(old))
    argv = ["--bridge", str(tiny_checkpoints / "spec.json"), "--aokvqa-dir", str(aokvqa),
            "--coco-dir", str(coco), "--eval-only", "--conditions", "none", "--threads", "1",
            "--out", str(out)]  # fmt: skip
    with pytest.raises(SystemExit, match="would mix scoring paths"):
        stage_b.main([*argv, "--head-dtype", "fp32"])
    assert preds.read_text() == '{"question_id": "v0"}\n'
    assert json.loads((out / "run_args.json").read_text()) == old  # not overwritten


def test_head_dtype_cli_values() -> None:
    assert head_mod.parse_head_dtype("fp32") is True
    assert head_mod.parse_head_dtype("model") is False
    with pytest.raises(ValueError, match="head-dtype"):
        head_mod.parse_head_dtype("bf16")


@pytest.mark.slow
def test_head_precision_audit_end_to_end_on_tiny_checkpoints(
    tiny_checkpoints: Path,  # noqa: F811  (the fixture imported above)
    tmp_path: Path,
) -> None:
    import importlib.util
    import json

    from tests.test_bridge import write_split

    path = Path(__file__).resolve().parents[1] / "experiments" / "analysis" / "head_precision.py"
    spec = importlib.util.spec_from_file_location("audit_head_precision", path)
    assert spec and spec.loader
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    aokvqa, coco = tmp_path / "aokvqa", tmp_path / "coco"
    aokvqa.mkdir()
    write_split(aokvqa, coco, "val", 3)
    out = tmp_path / "audit"
    common = ["--bridge", str(tiny_checkpoints / "spec.json"), "--aokvqa-dir", str(aokvqa)]
    common += ["--coco-dir", str(coco), "--limit", "2", "--rotations", "2", "--threads", "1"]
    txt = tmp_path / "head_precision_output.txt"
    summary = audit.main([*common, "--out", str(out), "--output-txt", str(txt)])
    assert summary["describe"]["cached/fp32_head"]["head_dtype"] == "fp32"
    assert summary["describe"]["uncached/fp32_head"]["prefix_cache"] is False
    assert summary["describe"]["cached/bf16_keep1"]["head_dtype"] == "model"
    assert summary["compute"]["torch_num_threads"] == 1
    res = summary["results"]
    pairs = [f"{a} vs {b}" for a, b in audit.PAIRS]
    assert set(res) == {f"[{m}] {p}" for m in ("cached", "uncached") for p in pairs} | {
        f"{v}: cached vs uncached" for v in audit.VARIANTS
    }
    assert all(r["n"] == 4 for pair in res.values() for r in pair.values())
    assert json.loads((out / "summary.json").read_text())["results"] == res
    saved = txt.read_text()
    assert saved.startswith("# run ") and "threads 1" in saved and "| [cached] " in saved
    with pytest.raises(SystemExit, match="not empty"):
        audit.main([*common, "--out", str(out), "--output-txt", ""])
    uncached = audit.main([*common, "--cache-modes", "uncached", "--variants",
                           "fp32_head,fp32_ref", "--out", str(tmp_path / "u"),
                           "--output-txt", ""])  # fmt: skip
    assert list(uncached["results"]) == ["[uncached] fp32_head vs fp32_ref"]
    with pytest.raises(SystemExit, match="cache-modes"):
        audit.main([*common, "--cache-modes", "warm", "--dry-run"])
