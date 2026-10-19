# Results

| Path | Contents |
|---|---|
| `run_metrics.json` | per-seed accuracies of every cloud run (`scripts/collect_runs.py`) |
| `tables/` | report tables as CSV (`scripts/make_report_tables.py`) |
| `analysis/` | outputs of `experiments/analysis/*.py`, of `fuse_siglip.py`, `order_average.py` and `adaptive_explanation.py`, and the laptop run times |
| `checkpoints.csv` | released bridges: run, LLM, target, data, seed, accuracy, SHA-256 |
| `runs/` | one folder per run: arguments (`run_args.json` or `config.json`) and `summary.{json,md}` |

Predictions (`predictions.jsonl`), training logs and bridges (`projector.pt`) are release assets;
`scripts/download_release.sh` unpacks them into the same run folders. The laptop run records'
`git_commit` fields refer to the development repository this one was cleaned from; the cloud jobs
ran from a packaged copy of the code, so their `git_commit` is null, and the configuration-hash
check on resume compared code versions only through that null field. Model revisions for the
cloud runs are recorded per phase in `runs/cloud/_status_phase*.json` (the `revision` fields in
`run_args.json` are null); `experiments/cloud/fyp_modal.py` now enforces them.

In `runs/cloud/*/modal_job.json`, `gpu` is empty and `max_mem_gb` is 0 for every job, because
both were read in the launcher process rather than the training process; the GPU type is in
`gpu_name`. These files record only each job's final attempt.

Two laptop folders hold arguments only, from runs stopped before evaluation and not used:
`runs/laptop/B_rationale_then_answer_n4000_s2` (CPU) and `runs/laptop/mac_AB/AO_shuf_full_s0`
(an option-shuffled run on all 16,024 questions, stopped after training). The evaluation-only
folders in `runs/laptop/mac_final/` name the bridge they evaluate in the first line of
`summary.md` with its laptop path `runs_modal/<run>/projector.pt`, which is
`runs/cloud/<run>/projector.pt` here.

`analysis/analyze_phase1.txt`, `analyze_phase2_hp.txt` and `analyze_bal.txt` are the outputs of
the three `experiments/cloud/analyze_*.py` scripts (Tables 4.2 and 4.4–4.6, including the dev
accuracies of Table 4.5). Each script draws its own bootstrap samples, so the same contrast can
differ by 0.1 pp between scripts (compare `analysis/bootstrap_check.txt`).

## Cloud runs (`runs/cloud/`)

`G05` = Qwen2-0.5B, `G7` = Qwen2-7B; `n4000` = 4k training questions, `full` = all 16,024;
targets `answer_only` (AO), `answer_then_rationale` (A→R), `rationale_then_answer` (R→A); `_s<n>` = seed.

| Runs | Report |
|---|---|
| `G*_theta0_*` | Table 4.1 (untrained bridge θ₀) |
| `G*_B_*` | Tables 4.2–4.4, 4.8–4.10, F.1 |
| `G05_HP_*` | Table 4.5 (hyperparameters) |
| `G*_BAL_*` | Table 4.6 (balanced loss) |
| `G7_A1_instrument` | Table 4.7, 7B (text-only check) |
| `G05_gate64` | Section 4.1 (GPU vs CPU check) |
| `_status_phase*.json` | job list, model revisions, GPU time (Section 4.9) |

## CPU and laptop runs (`runs/laptop/`)

The CPU runs in this folder ran on an Intel Xeon machine (`cpu_model` in their `run_args.json` or
`config.json`): `pilot_a1v2_instrument`, `pilot_a2v2_bridge_init`, `B0_*_cpu` and seeds 0–2 of
`B_*_n4000` (R→A: seeds 0–1 completed, seed 2 stopped; see above). Every other run here ran on
the Apple M4 Pro laptop.

| Runs | Report |
|---|---|
| `pilot_a1v2_instrument` | Table 4.7, 0.5B; defines the prior-wrong subset P (Xeon CPU) |
| `text_only_baselines/` | text-only baselines (Section 4.5) |
| `pilot_a2v2_bridge_init`, `B0_*_cpu` | θ₀ on the Xeon CPU (Section 4.1) |
| `B_*_n4000_s*` | first 0.5B runs, Table 4.2: seeds 0–2 on the Xeon CPU, seeds 3–4 on the laptop |
| `mac_extra/` | laptop baselines, transfer bridges, Table 4.15 |
| `mac_AB/` | Tables 4.11, 4.12; direct-answer bridges (Table 4.16) |
| `mac_final/` | Tables 4.8 (A→R), 4.9, 4.16 (7B MMStar and direct answers) |
| `mac_transfer/` | Table 4.16 (0.5B) |
| `mac_bench/` | Table 4.14 (`bench.json` P0–P6: feature cache, micro-batches, shared prefix; `bench2.json` B1–B7: per-layer times, attention variants, fused GEMM; `bench3.json` C1–C2: torch.compile and CPU co-training) |
| `audit_head/` | fp32 vs bf16 evaluation check (Section 3.4) |
| `siglip_val/` (release asset) | SigLIP scores for Table 4.10 |
