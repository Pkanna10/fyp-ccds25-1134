"""First set of laptop speed benchmarks on the Apple GPU (MPS): Table 4.14 and Section 4.9.

Run from the repo root with the GPU otherwise idle:
    PYTHONPATH=src \\
      python experiments/laptop/mac_bench.py --out results/runs/laptop/mac_bench
Writes <out>/bench.json (the recorded one is results/runs/laptop/mac_bench/bench.json). Uses
the first 64 train questions with their COCO images and the untrained 0.5B bridge only.

Measurements ("ms/item" is per training example, median after a warm-up; report use in [ ]):
  P0  baseline training micro-step (batch 1, SigLIP encoder in the step)
  P1  the same step with cached SigLIP features, estimated as P0 minus the encoder alone (P1a)
      [feature cache: the encoder is 25% of a training step]
  P2  micro-batch 2 and 4 [599 / 588 vs 609 ms per example, 8-11 GB; not adopted]
  P3  shared-prefix step: image + question once, the 4 option orders as a batch of suffixes
      on its KV cache; gradient checked against 4 separate sequences
  P4  SigLIP throughput at batch 1 / 8
  P5  evaluation: cold fp32 scoring pass, the fp32 encoder alone, 4 orders one by one
      [feature cache: the encoder is 48% of a cold evaluation pass]
  P6  matmul and attention speed in fp32 / bf16 / fp16 at the model's shapes [the bf16 GEMM
      rate, 5.47 TFLOP/s, sets the FLOP-ideal of mac_bench2.py B1]
Also records MPS memory after each measurement and swap use before and after.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmr.data.aokvqa import coco_image_path, load_aokvqa  # noqa: E402
from cmr.models import head as head_mod  # noqa: E402
from cmr.models.bridge import BridgeSpec, FrozenBridgeVLM  # noqa: E402
from cmr.train.targets import (  # noqa: E402
    TargetFormat,
    answer_char_span,
    prompt_for,
    target_for,
)

DEV = "mps"
SPLIT_AT = "Options:\n"  # prefix = header + image + question; suffix = options onward
RESULTS: dict[str, object] = {}


def sync() -> None:
    torch.mps.synchronize()


def mem() -> dict[str, float]:
    return {
        "alloc_gb": round(torch.mps.current_allocated_memory() / 2**30, 2),
        "driver_gb": round(torch.mps.driver_allocated_memory() / 2**30, 2),
    }


def swap() -> str:
    try:
        return subprocess.run(
            ["sysctl", "vm.swapusage"], capture_output=True, text=True
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def timed(fn, n: int, warm: int = 2) -> list[float]:  # noqa: ANN001
    for _ in range(warm):
        fn()
    sync()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        out.append(time.perf_counter() - t)
    return out


def report(name: str, secs: list[float], per: int = 1, note: str = "") -> float:
    ms = 1000 * statistics.median(secs) / per
    RESULTS[name] = {"ms_per_item": round(ms, 1), "n": len(secs), **mem(), "note": note}
    print(f"{name:34s} {ms:8.1f} ms/item   {mem()}  {note}", flush=True)
    return ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="configs/ov_qwen2_05b_frozen.json")
    ap.add_argument("--aokvqa-dir", default="data/raw/aokvqa")
    ap.add_argument("--coco-dir", default="data/raw/coco")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="results/runs/laptop/mac_bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    RESULTS["swap_before"] = swap()
    print("swap before:", RESULTS["swap_before"])

    # ---- P6 (first, model-free): GEMM and attention speed by dtype; is bf16 native here?
    T, d, f = 815, 896, 4864
    for dt in (torch.float32, torch.bfloat16, torch.float16):
        a = torch.randn(T, d, device=DEV, dtype=dt)
        w1 = torch.randn(d, f, device=DEV, dtype=dt)
        w2 = torch.randn(f, d, device=DEV, dtype=dt)
        flops = 2 * T * d * f * 2 * 20
        secs = timed(lambda: [(a @ w1) @ w2 for _ in range(20)], 5)
        tf = flops / statistics.median(secs) / 1e12
        q = torch.randn(1, 14, T, 64, device=DEV, dtype=dt, requires_grad=True)
        att = timed(
            lambda: (
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
                .sum()
                .backward()
            ),
            5,
        )
        name = str(dt).split(".")[-1]
        RESULTS[f"P6 {name}"] = {
            "mlp_gemm_tflops": round(tf, 2),
            "attn_fwd_bwd_ms": round(1000 * statistics.median(att), 2),
        }
        print(
            f"P6 {name:9s} MLP GEMMs {tf:5.2f} TFLOP/s   attention fwd+bwd "
            f"{1000 * statistics.median(att):6.2f} ms",
            flush=True,
        )
        del a, w1, w2, q
    torch.mps.empty_cache()

    torch.manual_seed(0)
    spec = BridgeSpec.from_json(args.bridge)
    t0 = time.time()
    model = FrozenBridgeVLM.from_pretrained(spec, device=DEV, dtype=torch.bfloat16)
    print(f"model loaded in {time.time() - t0:.0f}s  {mem()}")
    items = load_aokvqa(args.aokvqa_dir, "train")[:64]
    from PIL import Image

    imgs = [
        Image.open(coco_image_path(args.coco_dir, "train", it.image_id)).convert("RGB")
        for it in items
    ]
    fmt = TargetFormat.ANSWER_ONLY
    rng = random.Random(0)
    prompts = [prompt_for(it, fmt) for it in items]
    targets = [target_for(it, fmt, rng) for it in items]
    spans = [answer_char_span(t, fmt) for t in targets]
    params = [p for p in model.parameters() if p.requires_grad]
    n = args.steps
    k = iter(range(10**9))

    # ---- P0 baseline micro-step
    def step0() -> None:
        i = next(k) % len(items)
        parts = model.loss_parts([prompts[i]], [imgs[i]], [targets[i]], [spans[i]])
        (parts.all.mean() / 16).backward()

    base = report("P0 baseline step (batch 1)", timed(step0, n))

    # ---- P1 tower precomputed: time the tower alone, the step minus the tower
    with torch.no_grad():
        px = [model.preprocess([im]) for im in imgs[: n + 2]]
    vis = model.vision

    def tower() -> None:
        i = next(k) % len(px)
        with torch.no_grad():
            vis(pixel_values=px[i].to(DEV, torch.bfloat16), output_hidden_states=True)

    t_tower = report("P1a SigLIP tower alone (batch 1)", timed(tower, n))
    RESULTS["P1 projected step with cached features"] = {
        "ms_per_item": round(base - t_tower, 1),
        "note": f"= P0 - P1a; tower share {100 * t_tower / base:.0f}%",
    }
    print(
        f"P1  projected step, cached features   {base - t_tower:8.1f} ms/item  "
        f"(tower = {100 * t_tower / base:.0f}% of the step)"
    )

    # ---- P2 micro-batch 2 and 4
    for b in (2, 4):

        def stepb(b: int = b) -> None:
            i = (next(k) * b) % (len(items) - b)
            parts = model.loss_parts(
                prompts[i : i + b], imgs[i : i + b], targets[i : i + b], spans[i : i + b]
            )
            (parts.all.mean() / (16 / b)).backward()

        try:
            report(f"P2 micro-batch {b}", timed(stepb, max(4, n // b)), per=b)
        except RuntimeError as e:  # out of memory etc.
            print(f"P2 micro-batch {b}: failed ({str(e)[:80]})")
            RESULTS[f"P2 micro-batch {b}"] = {"error": str(e)[:200]}
        torch.mps.empty_cache()

    # ---- P3 shared-prefix multi-order step, with a gradient check
    decoder, head = model._decoder_and_head()
    fp32 = model.head_fp32_active
    tok = model.tokenizer

    def orders(it):  # noqa: ANN001, ANN202
        """4 cyclic option orders of one item: (item copy, target, span) each."""
        from dataclasses import replace

        out_ = []
        for r in range(4):
            ch = list(it.choices[r:] + it.choices[:r])
            gold = (it.correct_choice_idx - r) % 4
            it2 = replace(it, choices=tuple(ch), correct_choice_idx=gold)
            tgt = target_for(it2, fmt, random.Random(0))
            out_.append((it2, tgt, answer_char_span(tgt, fmt)))
        return out_

    def pieces(it2, tgt):  # noqa: ANN001, ANN202
        """(prefix text before the image marker, prefix text after it, suffix ids, labels)."""
        full = model.format_prompt(prompt_for(it2, fmt), True)
        pre, post = full.split("<image>")
        cut = post.index(SPLIT_AT) + len(SPLIT_AT)
        q_text, s_text = post[:cut], post[cut:]
        s_ids = tok.encode(s_text, add_special_tokens=False)
        t_ids = tok.encode(tgt, add_special_tokens=False) + [int(tok.eos_token_id)]
        return pre, q_text, s_ids, t_ids

    def nll_rows(hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """mean target NLL per row; hidden (B, T, D), labels (B, T) with -100 unscored."""
        gold = labels[:, 1:]
        sc = gold != -100
        rows = hidden[:, :-1][sc]
        logits = head_mod.fp32_head_logits(rows, head) if fp32 else head(rows)
        nll = torch.nn.functional.cross_entropy(logits.float(), gold[sc], reduction="none")
        per = torch.zeros(gold.shape, device=nll.device, dtype=nll.dtype)
        per[sc] = nll
        return per.sum(1) / sc.sum(1)

    def emb(ids: list[int]) -> torch.Tensor:
        return model._embed(ids)

    def separate(it) -> torch.Tensor:  # noqa: ANN001
        """4 full sequences, one forward each (what training with shuffling does per order)."""
        v = model.encode_images(model.preprocess([imgs[items.index(it)]]))[0]
        losses = []
        for it2, tgt, _ in orders(it):
            pre, q, s_ids, t_ids = pieces(it2, tgt)
            e = torch.cat(
                [
                    emb(tok.encode(pre)),
                    v.to(torch.bfloat16),
                    emb(tok.encode(q, add_special_tokens=False)),
                    emb(s_ids),
                    emb(t_ids),
                ]
            ).unsqueeze(0)
            lab = torch.full((1, e.shape[1]), -100, device=DEV, dtype=torch.long)
            lab[0, -len(t_ids) :] = torch.tensor(t_ids, device=DEV, dtype=torch.long)
            h = decoder(inputs_embeds=e, use_cache=False).last_hidden_state
            losses.append(nll_rows(h, lab)[0])
        return torch.stack(losses).mean()

    def shared(it) -> torch.Tensor:  # noqa: ANN001
        """image + question once (KV cached, with grad), the 4 orders as one suffix batch."""
        v = model.encode_images(model.preprocess([imgs[items.index(it)]]))[0]
        os_ = orders(it)
        pre, q, _, _ = pieces(*os_[0][:2])
        pe = torch.cat(
            [
                emb(tok.encode(pre)),
                v.to(torch.bfloat16),
                emb(tok.encode(q, add_special_tokens=False)),
            ]
        ).unsqueeze(0)
        po = decoder(inputs_embeds=pe, use_cache=True)
        cache = po.past_key_values
        cache.batch_repeat_interleave(4)
        sfx, labs = [], []
        for it2, tgt, _ in os_:
            _, _, s_ids, t_ids = pieces(it2, tgt)
            sfx.append(torch.cat([emb(s_ids), emb(t_ids)]))
            lab = torch.full((len(s_ids) + len(t_ids),), -100, device=DEV, dtype=torch.long)
            lab[-len(t_ids) :] = torch.tensor(t_ids, device=DEV, dtype=torch.long)
            labs.append(lab)
        L = max(x.shape[0] for x in sfx)
        if any(x.shape[0] != L for x in sfx):
            raise RuntimeError("suffix lengths differ across orders")
        se = torch.stack(sfx)
        P = pe.shape[1]
        mask = torch.ones(4, P + L, device=DEV, dtype=torch.long)
        h = decoder(
            inputs_embeds=se, past_key_values=cache, attention_mask=mask, use_cache=True
        ).last_hidden_state
        # labels are shifted inside nll_rows: prepend the prefix's last hidden state
        hp = po.last_hidden_state[:, -1:].expand(4, 1, -1)
        lab0 = torch.full((4, 1), -100, device=DEV, dtype=torch.long)
        return nll_rows(torch.cat([hp, h], 1), torch.cat([lab0, torch.stack(labs)], 1)).mean()

    def grads(fn, it) -> torch.Tensor:  # noqa: ANN001
        for p in params:
            p.grad = None
        fn(it).backward()
        return torch.cat([p.grad.flatten().float() for p in params])

    it0 = items[0]
    try:
        ga, gb = grads(separate, it0), grads(shared, it0)
        cos = torch.nn.functional.cosine_similarity(ga, gb, dim=0).item()
        rel = ((ga - gb).norm() / ga.norm()).item()
        RESULTS["P3 gradient check"] = {"cosine": cos, "rel_l2_diff": rel}
        print(f"P3  gradient check: cosine {cos:.6f}, relative L2 difference {rel:.2e}")
        j = iter(range(10**9))
        t_sep = report(
            "P3a 4 orders, separate sequences",
            timed(lambda: separate(items[next(j) % 32]).backward(), max(3, n // 4)),
            per=4,
        )
        t_sh = report(
            "P3b 4 orders, shared prefix (NEW)",
            timed(lambda: shared(items[next(j) % 32]).backward(), max(3, n // 4)),
            per=4,
        )
        print(f"P3  speed-up per order: {t_sep / t_sh:.2f}x")
    except Exception as e:  # noqa: BLE001
        print(f"P3 failed: {type(e).__name__}: {str(e)[:200]}")
        RESULTS["P3"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    for p in params:
        p.grad = None
    torch.mps.empty_cache()

    # ---- P4 SigLIP throughput for a cache build
    for b in (1, 8):
        pxb = model.preprocess(imgs[:b]).to(DEV, torch.bfloat16)
        report(
            f"P4 SigLIP batch {b}",
            timed(lambda pxb=pxb: vis(pixel_values=pxb, output_hidden_states=True) and None, 4),
            per=b,
        )

    # ---- P5 evaluation (fp32 scoring, the precision the runs use)
    with model.frozen_dtype(torch.float32), torch.no_grad(), model.scoring_mode():
        from cmr.models.base import Query

        q = [Query(prompts[i], imgs[i]) for i in range(16)]
        jj = iter(range(10**9))

        def cold() -> None:
            model.clear_cache()
            model.letter_logprobs(q[next(jj) % 16], 4)

        t_cold = report("P5a eval cold pass fp32", timed(cold, 8))
        pxf = [model.preprocess([im]) for im in imgs[:10]]

        def tower32() -> None:
            vis(pixel_values=pxf[next(jj) % 10].to(DEV, torch.float32), output_hidden_states=True)

        t_t32 = report("P5b fp32 tower alone", timed(tower32, 8))
        print(
            f"P5  projected cold pass with cached fp32 features: {t_cold - t_t32:.0f} ms "
            f"(tower {100 * t_t32 / t_cold:.0f}% of a cold pass)"
        )

        def warm4() -> None:
            i = next(jj) % 16
            model.clear_cache()
            for r in range(4):
                it = items[i]
                ch = list(it.choices[r:] + it.choices[:r])
                from cmr.eval.prompts import mc_prompt

                model.letter_logprobs(Query(mc_prompt(it.question, ch), imgs[i]), 4)

        report(
            "P5c 4 orders scored one by one",
            timed(warm4, 6),
            per=4,
            note="1 cold + 3 cached passes per item",
        )
    RESULTS["memory_end"] = mem()
    RESULTS["swap_after"] = swap()
    print("swap after:", RESULTS["swap_after"])
    (out / "bench.json").write_text(json.dumps(RESULTS, indent=2))
    print(f"wrote {out / 'bench.json'}")


if __name__ == "__main__":
    main()
