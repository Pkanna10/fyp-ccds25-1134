"""Second set of laptop speed benchmarks (MPS): per-operation step times and exact alternatives.

Report Table 4.14, Section 4.9 and Figure 4.10 (left). Run from the repo root with the GPU
otherwise idle:
    PYTHONPATH=src PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \\
      python experiments/laptop/mac_bench2.py --out results/runs/laptop/mac_bench
Writes <out>/bench2.json (the recorded one is results/runs/laptop/mac_bench/bench2.json).
B1-B5 use random inputs, B6-B7 the first 8 train questions and their COCO images; untrained
0.5B bridge only. Report use in [ ]:

B1  op-by-op time of one Qwen2-0.5B decoder layer (fwd + bwd, bf16, T=830) and of the whole
    decoder, against the FLOP-ideal at the measured GEMM rate [attention block 8.6 of 19.2 ms]
B2  attention variants, fwd + bwd, each checked against the baseline in fp32:
      base      torch SDPA, is_causal, enable_gqa
      f32       the same in fp32
      grouped   the 7 query heads that share a KV head stacked into one matmul
      block2/4  causal blocks: each query block only meets the keys it can see
                [block2 = the adopted block-causal attention, 6.71 -> 4.67 ms; block4 rejected]
      fused     fused no-grad MPS kernel forward, hand-written FlashAttention-2 backward
B3  MLP: gate and up projections as two GEMMs vs one concatenated GEMM [rejected: slower]
B4  torch.compile of one decoder layer (fixed length)
B5  CPU: BLAS in use, fp32 layer time, GPU decoder step time with the CPU busy (whether
    CPU-GPU batch splitting can help)
B6  shared-prefix multi-order gradient (mac_bench.py P3) re-checked in fp32
B7  evaluation: 4 option orders as one batch on the shared prefix vs one by one (fp32)
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmr.data.aokvqa import coco_image_path, load_aokvqa  # noqa: E402
from cmr.eval.prompts import mc_prompt  # noqa: E402
from cmr.models import head as head_mod  # noqa: E402
from cmr.models.base import Query  # noqa: E402
from cmr.models.bridge import BridgeSpec, FrozenBridgeVLM  # noqa: E402
from cmr.eval.prompts import letter_token_ids  # noqa: E402

DEV = "mps"
R: dict[str, object] = {}
T, D, NH, NKV, HD = 830, 896, 14, 2, 64
G = NH // NKV
SCALE = HD**-0.5


def sync() -> None:
    torch.mps.synchronize()


def timed(fn, n: int = 6, warm: int = 2) -> float:  # noqa: ANN001
    for _ in range(warm):
        fn()
    sync()
    xs = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync()
        xs.append(time.perf_counter() - t)
    return 1000 * statistics.median(xs)


def say(key: str, ms: float | None = None, **kw: object) -> None:
    R[key] = {"ms": None if ms is None else round(ms, 2), **kw}
    extra = "  ".join(f"{k}={v}" for k, v in kw.items())
    print(f"{key:44s} {'' if ms is None else f'{ms:8.2f} ms'}  {extra}", flush=True)


def fwd_bwd(f, *xs):  # noqa: ANN001, ANN202
    """Closure: forward f(*xs), backward a fixed random cotangent."""
    xs = [x.detach().requires_grad_(True) for x in xs]
    with torch.no_grad():
        y0 = f(*xs)
        y0 = y0[0] if isinstance(y0, tuple) else y0
    gy = torch.randn_like(y0)

    def run() -> None:
        for x in xs:
            x.grad = None
        y = f(*xs)
        y = y[0] if isinstance(y, tuple) else y
        y.backward(gy)

    return run, xs, gy


def grads_of(f, xs, gy):  # noqa: ANN001, ANN202
    xs = [x.detach().requires_grad_(True) for x in xs]
    y = f(*xs)
    y = y[0] if isinstance(y, tuple) else y
    y.backward(gy)
    return [y.detach()] + [x.grad.detach() for x in xs]


def rel(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(((x.float() - y.float()).norm() / y.float().norm()).item() for x, y in zip(a, b))


# ----------------------------------------------------------------------------- attention
_masks: dict[tuple, torch.Tensor] = {}


def causal_mask(q_lo: int, q_hi: int, k_hi: int, rows_per_pos: int = 1) -> torch.Tensor:
    """bool (rows, k_hi): True where query position p (stacked rows_per_pos times) may see key j."""
    key = (q_lo, q_hi, k_hi, rows_per_pos)
    if key not in _masks:
        pos = torch.arange(q_lo, q_hi, device=DEV).repeat(rows_per_pos)
        _masks[key] = torch.arange(k_hi, device=DEV)[None, :] <= pos[:, None]
    return _masks[key]


def att_base(q, k, v):  # noqa: ANN001, ANN201
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True, scale=SCALE)


def att_f32(q, k, v):  # noqa: ANN001, ANN201
    return att_base(q.float(), k.float(), v.float()).to(q.dtype)


def att_grouped(q, k, v):  # noqa: ANN001, ANN201
    B, _, L, _ = q.shape
    qg = q.view(B, NKV, G * L, HD)  # heads h = kv*G + g, rows ordered (g, position)
    s = (qg @ k.transpose(-1, -2)) * SCALE
    s = s.masked_fill(~causal_mask(0, L, L, G), float("-inf"))
    p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
    return (p @ v).view(B, NH, L, HD)


def att_blocks(nb: int):  # noqa: ANN201
    def f(q, k, v):  # noqa: ANN001, ANN202
        B, _, L, _ = q.shape
        cuts = [round(i * L / nb) for i in range(nb + 1)]
        outs = []
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            qg = q[:, :, lo:hi].reshape(B, NKV, G * (hi - lo), HD)
            s = (qg @ k[:, :, :hi].transpose(-1, -2)) * SCALE
            s = s.masked_fill(~causal_mask(lo, hi, hi, G), float("-inf"))
            p = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)
            outs.append((p @ v[:, :, :hi]).view(B, NH, hi - lo, HD))
        return torch.cat(outs, dim=2)

    return f


class FusedFwdAttention(torch.autograd.Function):
    """Forward: the fused no-grad MPS SDPA kernel. Backward: exact recompute (FA-2 algebra)."""

    @staticmethod
    def forward(ctx, q, k, v):  # noqa: ANN001, ANN205
        with torch.no_grad():
            o = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=True, scale=SCALE
            )
        ctx.save_for_backward(q, k, v, o)
        return o

    @staticmethod
    def backward(ctx, do):  # noqa: ANN001, ANN205
        q, k, v, o = ctx.saved_tensors
        B, _, L, _ = q.shape
        qg, og, dog = (t.reshape(B, NKV, G * L, HD) for t in (q, o, do.contiguous()))
        s = (qg @ k.transpose(-1, -2)) * SCALE
        s = s.masked_fill(~causal_mask(0, L, L, G), float("-inf"))
        p = torch.softmax(s, dim=-1, dtype=torch.float32)
        dv = p.to(q.dtype).transpose(-1, -2) @ dog  # the G stacked heads sum into their KV head
        dp = (dog @ v.transpose(-1, -2)).float()
        dsum = (dog.float() * og.float()).sum(-1, keepdim=True)
        ds = (p * (dp - dsum) * SCALE).to(q.dtype)
        dq = (ds @ k).view(B, NH, L, HD)
        dk = ds.transpose(-1, -2) @ qg
        return dq, dk, dv


def att_fused(q, k, v):  # noqa: ANN001, ANN201
    return FusedFwdAttention.apply(q, k, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="configs/ov_qwen2_05b_frozen.json")
    ap.add_argument("--aokvqa-dir", default="data/raw/aokvqa")
    ap.add_argument("--coco-dir", default="data/raw/coco")
    ap.add_argument("--out", default="results/runs/laptop/mac_bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    # ---------------------------------------------------------------- B2 attention variants
    print("B2 attention, fwd+bwd per layer (bf16), and max relative error vs base in fp32")
    q = torch.randn(1, NH, T, HD, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, NKV, T, HD, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, NKV, T, HD, device=DEV, dtype=torch.bfloat16)
    variants = {
        "base": att_base,
        "f32": att_f32,
        "grouped": att_grouped,
        "block2": att_blocks(2),
        "block4": att_blocks(4),
        "fused": att_fused,
    }
    q32, k32, v32 = q.float(), k.float(), v.float()
    _, _, gy32 = fwd_bwd(att_base, q32, k32, v32)
    ref = grads_of(att_base, [q32, k32, v32], gy32)
    for name, f in variants.items():
        try:
            run, _, _ = fwd_bwd(f, q, k, v)
            ms = timed(run, 8)
            err = rel(grads_of(f, [q32, k32, v32], gy32), ref) if name != "f32" else 0.0
            say(f"B2 attention {name}", ms, x24=round(24 * ms), fp32_rel_err=f"{err:.1e}")
        except Exception as e:  # noqa: BLE001
            say(f"B2 attention {name}", None, error=f"{type(e).__name__}: {str(e)[:120]}")
    del q, k, v, q32, k32, v32
    torch.mps.empty_cache()

    # ---------------------------------------------------------------- model
    spec = BridgeSpec.from_json(args.bridge)
    model = FrozenBridgeVLM.from_pretrained(spec, device=DEV, dtype=torch.bfloat16)
    dec = model.llm.model
    layer = dec.layers[3]
    ids = torch.randint(0, 150000, (1, T), device=DEV)
    with torch.no_grad():
        h = dec.embed_tokens(ids) * 4.0
    pos = torch.arange(T, device=DEV)[None]
    pe = dec.rotary_emb(h, pos)

    # ---------------------------------------------------------------- B1 op-by-op
    print("B1 one decoder layer, fwd+bwd (bf16, T=830); x24 = per training step")
    sa = layer.self_attn
    parts = {
        "whole layer": lambda x: layer(x, position_embeddings=pe),
        "input RMSNorm": layer.input_layernorm,
        "attention block (q,k,v,rope,sdpa,o)": lambda x: sa(
            hidden_states=x, position_embeddings=pe, attention_mask=None
        )[0],
        "  q_proj": sa.q_proj,
        "  k_proj": sa.k_proj,
        "  v_proj": sa.v_proj,
        "MLP (gate, up, silu*mul, down)": layer.mlp,
        "  gate_proj": layer.mlp.gate_proj,
        # down_proj takes a 4,864-wide input: h repeated to that width
        "  down_proj": lambda x: layer.mlp.down_proj(x.repeat(1, 1, 6)[..., :4864]),
    }
    total = None
    for name, f in parts.items():
        run, _, _ = fwd_bwd(f, h)
        ms = timed(run, 8)
        say(f"B1 {name}", ms, x24=round(24 * ms))
        if name == "whole layer":
            total = ms
    # 27.22e9 = forward FLOPs of one layer at T = 830 (projections, MLP, full attention score
    # matrix); the factor adds the backward pass (input gradients only, ~1.09 x the forward)
    flop_layer = 27.22e9 * (1 + 713 / 653)
    say(
        "B1 FLOP-ideal per layer at 5.47 TFLOP/s",
        1000 * flop_layer / 5.47e12,
        efficiency=f"{100 * (1000 * flop_layer / 5.47e12) / total:.0f}% of ideal" if total else "",
    )

    def decoder_step() -> None:
        x = h.detach().requires_grad_(True)
        y = dec(inputs_embeds=x, use_cache=False).last_hidden_state
        y[:, -2:].float().sum().backward()

    say("B1 full decoder 24 layers fwd+bwd", timed(decoder_step, 5))

    # ---------------------------------------------------------------- B3 MLP fusion
    mlp = layer.mlp
    wgu = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)

    def mlp_fused(x, wgu=wgu):  # noqa: ANN001, ANN202
        g, u = F.linear(x, wgu).chunk(2, dim=-1)
        return mlp.down_proj(F.silu(g) * u)

    run, _, gy = fwd_bwd(mlp, h)
    say("B3 MLP separate gate/up", timed(run, 8))
    run, _, _ = fwd_bwd(mlp_fused, h)
    ms = timed(run, 8)
    with model.frozen_dtype(torch.float32):
        wgu32 = torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0)
        h32, gy32 = h.float(), gy.float()
        e = rel(grads_of(lambda x: mlp_fused(x, wgu32), [h32], gy32), grads_of(mlp, [h32], gy32))
    say("B3 MLP fused gate+up GEMM", ms, fp32_rel_err=f"{e:.1e}")

    # ---------------------------------------------------------------- B4 torch.compile
    try:
        t0 = time.time()
        cl = torch.compile(lambda x: layer(x, position_embeddings=pe), dynamic=False)
        run, _, _ = fwd_bwd(cl, h)
        run()
        sync()
        say("B4 torch.compile layer fwd+bwd", timed(run, 8), compile_s=round(time.time() - t0))
    except Exception as e:  # noqa: BLE001
        say("B4 torch.compile layer", None, error=f"{type(e).__name__}: {str(e)[:160]}")
    torch.mps.empty_cache()

    # ---------------------------------------------------------------- B5 CPU co-execution
    cfg = torch.__config__.show()
    blas = [ln.strip() for ln in cfg.splitlines() if "BLAS" in ln or "LAPACK" in ln]
    say("B5 CPU BLAS", None, blas="; ".join(blas)[:160], threads=torch.get_num_threads())
    lcpu = copy.deepcopy(layer).float().cpu()
    hc = h.detach().float().cpu()
    pec = tuple(t.float().cpu() for t in pe)
    run_c, _, _ = fwd_bwd(lambda x: lcpu(x, position_embeddings=pec), hc)
    t0 = time.perf_counter()
    for _ in range(3):
        run_c()
    cpu_ms = 1000 * (time.perf_counter() - t0) / 3
    say("B5 CPU fp32 layer fwd+bwd", cpu_ms, x24=round(24 * cpu_ms))
    solo = timed(decoder_step, 5)
    stop = threading.Event()
    n_cpu = [0]

    def cpu_loop() -> None:
        while not stop.is_set():
            run_c()
            n_cpu[0] += 1

    th = threading.Thread(target=cpu_loop)
    th.start()
    t0 = time.perf_counter()
    busy = timed(decoder_step, 5, warm=1)
    span = time.perf_counter() - t0
    stop.set()
    th.join()
    say(
        "B5 GPU decoder step, CPU idle vs busy",
        None,
        solo_ms=round(solo),
        busy_ms=round(busy),
        cpu_layers_done=n_cpu[0],
        cpu_ms_per_layer_busy=round(1000 * span / max(1, n_cpu[0])),
    )
    del lcpu
    torch.mps.empty_cache()

    # ---------------------------------------------------------------- B6 / B7 need real data
    items = load_aokvqa(args.aokvqa_dir, "train")[:8]
    from PIL import Image

    imgs = [
        Image.open(coco_image_path(args.coco_dir, "train", it.image_id)).convert("RGB")
        for it in items
    ]
    tok = model.tokenizer

    def rotated(it, r):  # noqa: ANN001, ANN202
        return list(it.choices[r:] + it.choices[:r])

    # B7 evaluation: 4 orders batched on one prefix (fp32, as the runs score)
    with model.frozen_dtype(torch.float32), torch.no_grad(), model.scoring_mode():
        decoder, head = model._decoder_and_head()
        fp32 = model.head_fp32_active
        groups = letter_token_ids(tok, 4)

        def seq(i: int) -> list[list[float]]:
            model.clear_cache()
            return [
                model.letter_logprobs(
                    Query(mc_prompt(items[i].question, rotated(items[i], r)), imgs[i]), 4
                )
                for r in range(4)
            ]

        def batched(i: int) -> list[list[float]]:
            it = items[i]
            pres, posts = zip(
                *(
                    model.format_prompt(mc_prompt(it.question, rotated(it, r)), True).split(
                        "<image>"
                    )
                    for r in range(4)
                )
            )
            vis = model._visual_batch([imgs[i]])[0]
            emb = model._embed(tok.encode(pres[0], add_special_tokens=True))
            prefix = torch.cat([emb, vis.to(emb.dtype)])[None]
            kv = decoder(inputs_embeds=prefix, use_cache=True).past_key_values
            kv.batch_repeat_interleave(4)
            pids = [tok.encode(p, add_special_tokens=False) for p in posts]
            lmax = max(map(len, pids))
            e = torch.zeros(4, lmax, emb.shape[-1], dtype=emb.dtype, device=DEV)
            mask = torch.ones(4, prefix.shape[1] + lmax, dtype=torch.long, device=DEV)
            for r, p in enumerate(pids):
                e[r, : len(p)] = model._embed(p)
                mask[r, prefix.shape[1] + len(p) :] = 0
            hs = decoder(
                inputs_embeds=e, past_key_values=kv, attention_mask=mask, use_cache=True
            ).last_hidden_state
            rows = hs[
                torch.arange(4, device=DEV), torch.tensor([len(p) - 1 for p in pids], device=DEV)
            ]
            logits = head_mod.fp32_head_logits(rows, head) if fp32 else head(rows).float()
            lp = torch.log_softmax(logits, -1)
            return [[float(torch.logsumexp(lp[r][g], 0)) for g in groups] for r in range(4)]

        diffs = [
            max(abs(a - b) for ra, rb in zip(seq(i), batched(i)) for a, b in zip(ra, rb))
            for i in range(6)
        ]
        jj = iter(range(10**6))
        say("B7 eval 4 orders one by one (per item)", timed(lambda: seq(next(jj) % 8), 6))
        say(
            "B7 eval 4 orders batched (per item)",
            timed(lambda: batched(next(jj) % 8), 6),
            max_abs_logp_diff=f"{max(diffs):.1e}",
        )
    torch.mps.empty_cache()

    # B6 shared-prefix multi-order gradient, fp32
    params = [p for p in model.projector.parameters()]
    with model.frozen_dtype(torch.float32):
        decoder, head = model._decoder_and_head()
        fp32 = model.head_fp32_active
        it, im = items[0], imgs[0]

        def pieces(r: int):  # noqa: ANN202
            full = model.format_prompt(mc_prompt(it.question, rotated(it, r)), True)
            pre, post = full.split("<image>")
            cut = post.index("Options:\n") + len("Options:\n")
            gold = (it.correct_choice_idx - r) % 4
            t_ids = tok.encode("ABCD"[gold], add_special_tokens=False) + [int(tok.eos_token_id)]
            return pre, post[:cut], tok.encode(post[cut:], add_special_tokens=False), t_ids

        def nll(hs: torch.Tensor, n_t: int) -> torch.Tensor:
            rows = hs[0, -n_t - 1 : -1]
            logits = head_mod.fp32_head_logits(rows, head) if fp32 else head(rows).float()
            return F.cross_entropy(logits, gold_t[:n_t], reduction="mean")

        def separate() -> torch.Tensor:
            vis = model._visual_batch([im])[0]
            loss = 0
            for r in range(4):
                pre, qt, s_ids, t_ids = pieces(r)
                emb = model._embed(tok.encode(pre))
                x = torch.cat(
                    [
                        emb,
                        vis.to(emb.dtype),
                        model._embed(tok.encode(qt, add_special_tokens=False)),
                        model._embed(s_ids + t_ids),
                    ]
                )[None]
                hs = decoder(inputs_embeds=x, use_cache=False).last_hidden_state
                global gold_t
                gold_t = torch.tensor(t_ids, device=DEV)
                loss = loss + nll(hs, len(t_ids)) / 4
            return loss

        def shared() -> torch.Tensor:
            vis = model._visual_batch([im])[0]
            pre, qt, _, _ = pieces(0)
            emb = model._embed(tok.encode(pre))
            x = torch.cat(
                [emb, vis.to(emb.dtype), model._embed(tok.encode(qt, add_special_tokens=False))]
            )[None]
            kv = decoder(inputs_embeds=x, use_cache=True).past_key_values
            kv.batch_repeat_interleave(4)
            suf = [pieces(r) for r in range(4)]
            e = torch.stack([model._embed(s + t) for _, _, s, t in suf])
            hs = decoder(
                inputs_embeds=e,
                past_key_values=kv,
                use_cache=True,
                attention_mask=torch.ones(4, x.shape[1] + e.shape[1], device=DEV, dtype=torch.long),
            ).last_hidden_state
            loss = 0
            global gold_t
            for r, (_, _, _, t_ids) in enumerate(suf):
                gold_t = torch.tensor(t_ids, device=DEV)
                loss = loss + nll(hs[r : r + 1], len(t_ids)) / 4
            return loss

        def g(fn) -> torch.Tensor:  # noqa: ANN001
            for p in params:
                p.grad = None
            fn().backward()
            return torch.cat([p.grad.flatten() for p in params])

        try:
            ga, gb = g(separate), g(shared)
            say(
                "B6 shared-prefix gradient vs separate, fp32",
                None,
                rel_l2=f"{((ga - gb).norm() / ga.norm()).item():.1e}",
                cosine=f"{F.cosine_similarity(ga, gb, dim=0).item():.6f}",
            )
        except Exception as e:  # noqa: BLE001
            say("B6 shared-prefix fp32", None, error=f"{type(e).__name__}: {str(e)[:160]}")

    (out / "bench2.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"wrote {out / 'bench2.json'}")


gold_t: torch.Tensor


if __name__ == "__main__":
    main()
