"""Third set of laptop speed benchmarks (MPS): torch.compile and CPU co-training, with checks.

Report Table 4.14 (torch.compile row: step 493 -> 401 ms, 1.23x, identical fp32 gradients) and
Section 3.8.2 (CPU co-training = CPU-GPU batch splitting). Run from the repo root with the GPU
otherwise idle:
    PYTHONPATH=src \\
    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.5 \\
      python experiments/laptop/mac_bench3.py --out results/runs/laptop/mac_bench
Writes <out>/bench3.json (the recorded one is results/runs/laptop/mac_bench/bench3.json).
Random inputs and the untrained 0.5B bridge only.

C1  torch.compile of each decoder layer (block attention on) over a spread of sequence
    lengths: compile cost per new length, step time vs eager, fp32 gradient difference vs eager
C2  CPU co-training: full CPU decoder fwd + bwd (fp32, bf16) at 4 / 6 / 8 threads, GPU step
    time while the CPU works, process memory, and the best GPU/CPU split of 16 micro-steps
"""

from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from cmr.models import block_attention  # noqa: E402
from cmr.models.bridge import BridgeSpec, FrozenBridgeVLM  # noqa: E402

DEV = "mps"
R: dict[str, object] = {}
LENGTHS = [812, 821, 830, 838, 846, 857]  # spread of real AO prompt+target lengths


def sync() -> None:
    torch.mps.synchronize()


def med_ms(fn, n: int = 5, warm: int = 1) -> float:  # noqa: ANN001
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


def say(key: str, **kw: object) -> None:
    R[key] = kw
    print(f"{key:46s} " + "  ".join(f"{k}={v}" for k, v in kw.items()), flush=True)


def rss_gb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)  # macOS: bytes


def step_fn(dec, x: torch.Tensor):  # noqa: ANN001, ANN201
    """One decoder forward + backward from the last two positions (like an AO loss)."""

    def run() -> torch.Tensor:
        xi = x.detach().requires_grad_(True)
        y = dec(inputs_embeds=xi, use_cache=False).last_hidden_state
        y[:, -2:].float().sum().backward()
        return xi.grad

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge", default="configs/ov_qwen2_05b_frozen.json")
    ap.add_argument("--out", default="results/runs/laptop/mac_bench")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    block_attention.enable()  # as in the laptop training runs (--block-attention)

    spec = BridgeSpec.from_json(args.bridge)
    model = FrozenBridgeVLM.from_pretrained(spec, device=DEV, dtype=torch.bfloat16)
    dec = model.llm.model
    xs = {
        n: dec.embed_tokens(torch.randint(0, 150000, (1, n), device=DEV)).detach() * 4
        for n in LENGTHS
    }

    # ------------------------------------------------------------------ C1 torch.compile
    eager = {n: med_ms(step_fn(dec, xs[n]), 3) for n in LENGTHS[:3]}
    say("C1 eager step (block attention)", ms={n: round(v) for n, v in eager.items()})
    layers = list(dec.layers)
    try:
        for layer in layers:
            layer.compile(dynamic=True)
        firsts = {}
        for n in LENGTHS:
            t = time.perf_counter()
            step_fn(dec, xs[n])()
            sync()
            firsts[n] = round(time.perf_counter() - t, 1)
        say("C1 compiled: first call per new length (s)", s=firsts)
        comp = {n: med_ms(step_fn(dec, xs[n]), 3) for n in LENGTHS[:3]}
        say(
            "C1 compiled step",
            ms={n: round(v) for n, v in comp.items()},
            speedup=round(statistics.mean(eager.values()) / statistics.mean(comp.values()), 2),
        )
        # exactness: fp32 compiled vs fp32 eager, same input
        with model.frozen_dtype(torch.float32):
            x32 = xs[830].float()
            g_c = step_fn(dec, x32)()
            for layer in layers:  # back to eager: the original forward
                layer._compiled_call_impl = None
            g_e = step_fn(dec, x32)()
        err = ((g_c - g_e).norm() / g_e.norm()).item()
        say("C1 compiled vs eager input-gradient, fp32", rel_l2=f"{err:.1e}")
    except Exception as e:  # noqa: BLE001
        say("C1 torch.compile", error=f"{type(e).__name__}: {str(e)[:200]}")
    for layer in layers:
        layer._compiled_call_impl = None
    torch._dynamo.reset()
    torch.mps.empty_cache()

    # ------------------------------------------------------------------ C2 CPU co-training
    from transformers import AutoModelForCausalLM

    gpu_solo = med_ms(step_fn(dec, xs[830]), 4)
    say("C2 GPU step solo", ms=round(gpu_solo))
    res = {}
    for dt in (torch.float32, torch.bfloat16):
        cpu_llm = AutoModelForCausalLM.from_pretrained(spec.llm_id, dtype=dt)
        cdec = cpu_llm.model
        for p in cpu_llm.parameters():
            p.requires_grad_(False)
        xc = xs[830].to("cpu", dt)
        for th in (4, 6, 8):
            torch.set_num_threads(th)
            run_c = step_fn(cdec, xc)
            run_c()
            t = time.perf_counter()
            run_c()
            cpu_ms = 1000 * (time.perf_counter() - t)
            # GPU step time while a CPU thread runs CPU steps back to back
            stop, done = threading.Event(), [0]

            def loop(run_c=run_c, stop=stop, done=done) -> None:  # noqa: ANN001
                while not stop.is_set():
                    run_c()
                    done[0] += 1

            th_ = threading.Thread(target=loop)
            th_.start()
            t0 = time.perf_counter()
            busy = med_ms(step_fn(dec, xs[830]), 4, warm=0)
            span = time.perf_counter() - t0
            stop.set()
            th_.join()
            cpu_busy_ms = 1000 * span / max(1, done[0])
            # best fixed split of 16 micro-steps: k on CPU, 16-k on GPU, run in parallel
            best = min(range(0, 9), key=lambda k: max((16 - k) * busy, k * cpu_busy_ms))
            step_new = max((16 - best) * busy, best * cpu_busy_ms) / 1000
            key = f"C2 CPU {str(dt).split('.')[-1]} {th} threads"
            res[key] = dict(
                cpu_ms=round(cpu_ms),
                gpu_busy_ms=round(busy),
                cpu_ms_while_gpu=round(cpu_busy_ms),
                best_cpu_share=f"{best}/16",
                speedup=round(16 * gpu_solo / 1000 / step_new, 2),
                rss_gb=rss_gb(),
            )
            say(key, **res[key])
        del cpu_llm, cdec, xc
    torch.set_num_threads(8)

    (out / "bench3.json").write_text(json.dumps(R, indent=2, default=str))
    print(f"wrote {out / 'bench3.json'}")


if __name__ == "__main__":
    main()
