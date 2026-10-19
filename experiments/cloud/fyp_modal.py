"""Modal jobs for the cloud study: every run in results/runs/cloud (Tables 4.1-4.7 and F.1).

Each job runs cmr.runners.stage_a (evaluation only) or cmr.runners.stage_b (train the bridge,
then evaluate on val) on a Modal GPU, with the same training arguments as the earlier CPU and
laptop runs except --device cuda. The jobs form the phases of report Table 3.5 (job counts and
GPU-hours in Table C.1):

  phase "1" (39 jobs)
    0.5B on L4:   G05_gate64, the GPU check of Section 4.1 (theta_0 on 64 val questions,
                  compared with the CPU run results/runs/laptop/pilot_a2v2_bridge_init);
                  theta_0 under each target's instruction (Table 4.1);
                  {AO, A->R, R->A} x {4k, full 16k} x seeds 0-4               -> 34 jobs
    7B on L40S:   G7_A1_instrument, the text-only check of Table 4.7 (no image, four text
                  contexts); theta_0 (AO, A->R); AO and A->R, 4k, seed 0       ->  5 jobs
  phase "2" (14 jobs): the rest of the 7B grid
  phase "hp" (46 jobs): 0.5B hyperparameter study (Table 4.5), 4k items, seeds 0-1, one
            setting changed at a time: learning rate, batch 64, 4 epochs, and the
            balanced-loss weight lambda (A->R and R->A only)
  phase "bal" (10 jobs): balanced loss with lambda = 0.3 (Table 4.6), 0.5B full data
            (seeds 0-2) and 7B 4k (seeds 0-1)

Run names: G05 / G7 = 0.5B / 7B LLM; B = main runs, HP = hyperparameter study, BAL = balanced
loss; n4000 / full = 4k / all 16,024 training questions; rw<x> = lambda; s<n> = seed.

Inputs (launch.sh uploads them to the Modal volume "fyp-data"): data/raw/aokvqa, data/raw/coco
and results/runs/laptop/pilot_a1v2_instrument, the text-only run of the frozen 0.5B LLM that
defines the prior-wrong subset P (its predictions.jsonl is a release asset).

Launch with experiments/cloud/launch.sh; fetch results from the Modal volume "fyp-runs" with
    modal volume get fyp-runs / results/runs/cloud/
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "fyp-ccds25-1134"
REPO_REMOTE = "/repo"

# Library versions of every cloud run (torch, transformers, peft and numpy are also recorded in
# each run_args.json manifest).
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.19.0",
        "accelerate==1.15.0",
        "peft==0.21.2",
        "huggingface_hub==2.2.0",
        "sentencepiece==0.2.2",
        "protobuf==7.36.2",
        "numpy==2.5.3",
        "pillow==12.3.0",
        "safetensors",
        "scipy",
    )
    .env({"PYTHONPATH": f"{REPO_REMOTE}/src", "HF_HOME": "/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir("src", f"{REPO_REMOTE}/src")
    .add_local_dir("configs", f"{REPO_REMOTE}/configs")
)

app = modal.App(APP_NAME, image=image)
data_vol = modal.Volume.from_name("fyp-data", create_if_missing=True)  # aokvqa, coco, prior run
hf_vol = modal.Volume.from_name("fyp-hf-cache", create_if_missing=True)  # model weights
runs_vol = modal.Volume.from_name("fyp-runs", create_if_missing=True)  # outputs
VOLS = {"/data": data_vol, "/hf": hf_vol, "/runs": runs_vol}

CFG_05 = f"{REPO_REMOTE}/configs/ov_qwen2_05b_frozen.json"
CFG_7 = f"{REPO_REMOTE}/configs/ov_qwen2_7b_frozen.json"
AOK = "/data/aokvqa"
COCO = "/data/coco"
PRIOR_05 = "/data/prior/pilot_a1v2_instrument"  # text-only 0.5B run that defines P (0.5B only)
CAPTIONS = f"{COCO}/annotations/captions_val2017.json"

# Hugging Face revisions used by every cloud run (recorded in results/runs/cloud/_status_phase*).
PINS = {
    "google/siglip-so400m-patch14-384": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
    "Qwen/Qwen2-0.5B-Instruct": "c540970f9e29518b1d8f06ab8b24cba66ad77b6d",
    "lmms-lab/llava-onevision-projectors": "a325adc5b0eae860a2d156b47b793dc727fd394e",
    "Qwen/Qwen2-7B-Instruct": "f2826a00ceef68f0f2b946d945ecc0477ce4450c",
}

# Every training run keeps its last checkpoint and checks dev only at the end of training
# (--eval-every larger than any run); effective batch 16 = 1 x 16 accumulation steps.
TRAIN_COMMON = [
    "--select",
    "last",
    "--eval-every",
    "1000000",
    "--batch-size",
    "1",
    "--grad-accum",
    "16",
    "--resume-if-present",
]
PREC = ["--device", "cuda", "--dtype", "bfloat16", "--eval-dtype", "float32", "--threads", "4"]
TARGETS = ["answer_only", "answer_then_rationale", "rationale_then_answer"]


# ----------------------------------------------------------------------------- jobs
def _b(name: str, cfg: str, target: str, *, prior: str | None, extra: list[str]) -> dict:
    """A stage_b job: train (unless --eval-only), then score val with real and swapped images.

    ``prior``: the run that defines the prior-wrong subset P (0.5B jobs only).
    """
    # R->A is scored in the original option order only (as on CPU and the laptop)
    rot = "1" if target == "rationale_then_answer" else "4"
    cmd = [
        "python",
        "-m",
        "cmr.runners.stage_b",
        "--bridge",
        cfg,
        "--aokvqa-dir",
        AOK,
        "--coco-dir",
        COCO,
        "--conditions",
        "real,swapped",
        "--rotations",
        rot,
        "--target",
        target,
        *PREC,
        *extra,
        "--out",
        f"/runs/{name}",
    ]
    if prior:
        cmd[cmd.index("--conditions") : cmd.index("--conditions")] = ["--prior-run", prior]
    return {"name": name, "cmd": cmd}


def jobs_05b() -> list[dict]:
    """The 34 0.5B jobs of phase 1 (L4)."""
    j: list[dict] = []
    # GPU check (Section 4.1): theta_0 on the first 64 val questions, compared with the same
    # records of the CPU run results/runs/laptop/pilot_a2v2_bridge_init by
    # experiments/analysis/report_numbers.py (section parity)
    j.append(
        {
            "name": "G05_gate64",
            "cmd": [
                "python",
                "-m",
                "cmr.runners.stage_a",
                "--model",
                f"bridge:{CFG_05}",
                "--device",
                "cuda",
                "--dtype",
                "float32",
                "--threads",
                "4",
                "--aokvqa-dir",
                AOK,
                "--coco-dir",
                COCO,
                "--split",
                "val",
                "--conditions",
                "real,none,blank",
                "--contexts",
                "none",
                "--rotations",
                "4",
                "--seed",
                "0",
                "--limit",
                "64",
                "--out",
                "/runs/G05_gate64",
            ],
        }
    )
    for t in TARGETS:  # theta_0 under each instruction, on this GPU
        j.append(_b(f"G05_theta0_{t}", CFG_05, t, prior=PRIOR_05, extra=["--eval-only"]))
    for dose, lim in (("n4000", ["--train-limit", "4000"]), ("full", [])):
        ck = ["--checkpoint-every", "10" if dose == "n4000" else "25"]
        for seed in range(5):
            for t in TARGETS:
                # end-of-training dev check: 500 questions, 100 for R->A (generation is slow)
                dev = ["--dev-size", "100"] if t == "rationale_then_answer" else []
                j.append(
                    _b(
                        f"G05_B_{t}_{dose}_s{seed}",
                        CFG_05,
                        t,
                        prior=PRIOR_05,
                        extra=["--seed", str(seed), *lim, *TRAIN_COMMON, *ck, *dev],
                    )
                )
    return j


def a1_7b() -> dict:
    """Text-only check of the frozen 7B LLM (Table 4.7): no image, four text contexts."""
    return {
        "name": "G7_A1_instrument",
        "cmd": [
            "python",
            "-m",
            "cmr.runners.stage_a",
            "--model",
            f"bridge:{CFG_7}",
            "--device",
            "cuda",
            "--dtype",
            "float32",
            "--threads",
            "4",
            "--aokvqa-dir",
            AOK,
            "--split",
            "val",
            "--conditions",
            "none",
            "--contexts",
            "none,caption,rationale_masked,rationale",
            "--captions",
            CAPTIONS,
            "--rotations",
            "4",
            "--seed",
            "0",
            "--out",
            "/runs/G7_A1_instrument",
        ],
    }


def jobs_7b(phase: str) -> list[dict]:
    """7B jobs (L40S): the 5 of phase "1", or the 14 others for phase "2"."""
    dev = ["--dev-size", "100"]
    first = [
        a1_7b(),
        _b("G7_theta0_answer_only", CFG_7, "answer_only", prior=None, extra=["--eval-only"]),
        _b(
            "G7_theta0_answer_then_rationale",
            CFG_7,
            "answer_then_rationale",
            prior=None,
            extra=["--eval-only"],
        ),
        *[
            _b(
                f"G7_B_{t}_n4000_s0",
                CFG_7,
                t,
                prior=None,
                extra=[
                    "--seed",
                    "0",
                    "--train-limit",
                    "4000",
                    *TRAIN_COMMON,
                    "--checkpoint-every",
                    "10",
                    *dev,
                ],
            )
            for t in ("answer_only", "answer_then_rationale")
        ],
    ]
    if phase == "1":
        return first
    rest = [
        _b(
            "G7_theta0_rationale_then_answer",
            CFG_7,
            "rationale_then_answer",
            prior=None,
            extra=["--eval-only"],
        )
    ]
    for seed in (1, 2):
        for t in ("answer_only", "answer_then_rationale"):
            rest.append(
                _b(
                    f"G7_B_{t}_n4000_s{seed}",
                    CFG_7,
                    t,
                    prior=None,
                    extra=[
                        "--seed",
                        str(seed),
                        "--train-limit",
                        "4000",
                        *TRAIN_COMMON,
                        "--checkpoint-every",
                        "10",
                        *dev,
                    ],
                )
            )
    for seed in (0, 1, 2):
        rest.append(
            _b(
                f"G7_B_rationale_then_answer_n4000_s{seed}",
                CFG_7,
                "rationale_then_answer",
                prior=None,
                extra=[
                    "--seed",
                    str(seed),
                    "--train-limit",
                    "4000",
                    *TRAIN_COMMON,
                    "--checkpoint-every",
                    "10",
                    *dev,
                ],
            )
        )
        for t in ("answer_only", "answer_then_rationale"):
            rest.append(
                _b(
                    f"G7_B_{t}_full_s{seed}",
                    CFG_7,
                    t,
                    prior=None,
                    extra=["--seed", str(seed), *TRAIN_COMMON, "--checkpoint-every", "25", *dev],
                )
            )
    return rest


def jobs_bal() -> tuple[list[dict], list[dict]]:
    """Balanced-loss jobs (Table 4.6) for A->R and R->A, as (0.5B jobs, 7B jobs).

    lambda = 0.3 (--rationale-weight), the best lambda > 0 on dev in the hyperparameter study
    (Table 4.5). 0.5B on all 16k items (seeds 0-2) and 7B on 4k items (seeds 0-1).
    """
    lam = ["--rationale-weight", "0.3"]
    small, large = [], []
    for seed in (0, 1, 2):
        for t in ("answer_then_rationale", "rationale_then_answer"):
            dev = ["--dev-size", "100"] if t == "rationale_then_answer" else []
            small.append(
                _b(
                    f"G05_BAL_{t}_full_rw0.3_s{seed}",
                    CFG_05,
                    t,
                    prior=PRIOR_05,
                    extra=[
                        "--seed",
                        str(seed),
                        *TRAIN_COMMON,
                        "--checkpoint-every",
                        "25",
                        *dev,
                        *lam,
                    ],
                )
            )
    for seed in (0, 1):
        for t in ("answer_then_rationale", "rationale_then_answer"):
            large.append(
                _b(
                    f"G7_BAL_{t}_n4000_rw0.3_s{seed}",
                    CFG_7,
                    t,
                    prior=None,
                    extra=[
                        "--seed",
                        str(seed),
                        "--train-limit",
                        "4000",
                        *TRAIN_COMMON,
                        "--checkpoint-every",
                        "10",
                        "--dev-size",
                        "100",
                        *lam,
                    ],
                )
            )
    return small, large


def jobs_hp() -> list[dict]:
    """Hyperparameter study (Table 4.5): 0.5B, 4k items, seeds 0-1, one setting changed at a time.

    The setting of each target is chosen on the dev split, not on val.
    """
    j: list[dict] = []
    lim = ["--train-limit", "4000"]

    def add(tag: str, t: str, seed: int, extra: list[str]) -> None:
        dev = ["--dev-size", "100"] if t == "rationale_then_answer" else []
        j.append(
            _b(
                f"G05_HP_{t}_{tag}_s{seed}",
                CFG_05,
                t,
                prior=PRIOR_05,
                extra=[
                    "--seed",
                    str(seed),
                    *lim,
                    *TRAIN_COMMON,
                    "--checkpoint-every",
                    "25",
                    *dev,
                    *extra,
                ],
            )
        )

    for seed in (0, 1):
        for t in TARGETS:
            for lr in ("3e-5", "3e-4", "1e-3"):  # the default 1e-4 = the phase-1 4k runs
                add(f"lr{lr}", t, seed, ["--lr", lr])
            add("ep4", t, seed, ["--epochs", "4"])  # 1,000 steps, about one pass over 16k
            # effective batch 64 (1 x 64) at 4x the learning rate (linear scaling)
            add("bs64", t, seed, ["--lr", "4e-4"])
        for t in ("answer_then_rationale", "rationale_then_answer"):
            for rw in ("0", "0.1", "0.3", "1.0"):  # lambda: weight of CE(rest of the target)
                add(f"rw{rw}", t, seed, ["--rationale-weight", rw])
    # bs64 needs grad-accum 64 instead of 16
    for job in j:
        if "_bs64_" in job["name"]:
            c = job["cmd"]
            c[c.index("--grad-accum") + 1] = "64"
    return j


# ----------------------------------------------------------------------------- runner
def _done(out: Path) -> bool:
    """A job is finished once its summary.md exists (stage_a: <out>/, stage_b: <out>/eval_val/)."""
    return (out / "summary.md").exists() or (out / "eval_val" / "summary.md").exists()


def _run(job: dict) -> dict:
    """Run one job in its container; log to <out>/queue.log and write <out>/modal_job.json.

    The recorded modal_job.json files also hold "gpu" (empty) and "max_mem_gb" (0) fields,
    written by the launcher version that ran the study (see results/README.md).
    """
    out = Path("/runs") / job["name"]
    if _done(out):
        return {"name": job["name"], "status": "already_done"}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(out / "queue.log", "a") as log:
        log.write(
            f"=== {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} modal "
            f"{os.environ.get('MODAL_TASK_ID', '')} {' '.join(job['cmd'])}\n"
        )
        log.flush()
        env = {**os.environ, "HF_HUB_OFFLINE": "1"}
        rc = subprocess.run(
            job["cmd"], cwd=REPO_REMOTE, env=env, stdout=log, stderr=subprocess.STDOUT
        ).returncode
    info = {
        "name": job["name"],
        "status": "ok" if rc == 0 else f"exit_{rc}",
        "seconds": round(time.time() - t0),
    }
    try:
        import torch

        info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    (out / "modal_job.json").write_text(json.dumps(info, indent=2))
    runs_vol.commit()  # volumes also commit in the background; this is the final one
    if rc != 0:
        raise RuntimeError(
            f"{job['name']} failed with exit {rc}; see /runs/{job['name']}/queue.log"
        )
    return info


@app.function(
    gpu="L4",
    cpu=4,
    memory=16384,
    volumes=VOLS,
    timeout=6 * 3600,
    retries=modal.Retries(max_retries=4, initial_delay=10.0),
    single_use_containers=True,
)
def run_small(job: dict) -> dict:
    return _run(job)


@app.function(
    gpu="L40S",
    cpu=4,
    memory=49152,
    volumes=VOLS,
    timeout=20 * 3600,
    retries=modal.Retries(max_retries=4, initial_delay=10.0),
    single_use_containers=True,
)
def run_large(job: dict) -> dict:
    return _run(job)


@app.function(cpu=4, memory=16384, volumes=VOLS, timeout=3 * 3600)
def prefetch(models: list[str]) -> dict:
    """Download the pinned model snapshots once into the shared HF cache."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    got = {}
    for repo in models:
        allow = ["0.5b/*", "7b/*", "*.json"] if repo.endswith("onevision-projectors") else None
        path = snapshot_download(repo, revision=PINS[repo], allow_patterns=allow)
        sha = Path(path).name
        if sha != PINS[repo]:
            raise RuntimeError(f"{repo}: got snapshot {sha}, pinned {PINS[repo]}")
        # The jobs load offline by name, which resolves refs/main: point it at the pin.
        ref = Path(HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(sha)
        got[repo] = sha
        print(f"{repo}: {sha}", flush=True)
    hf_vol.commit()
    return got


# nonpreemptible: a preempted orchestrator restarts and would re-spawn the jobs.
@app.function(cpu=0.5, memory=2048, volumes=VOLS, timeout=24 * 3600, nonpreemptible=True)
def orchestrate(phase: str = "1", token: str = "") -> dict:
    """Run one phase from the cloud (the local machine can sleep).

    Downloads the pinned models, spawns every job of the phase and keeps
    /runs/_status_phase<phase>.json up to date (results/runs/cloud/_status_phase*.json).
    """
    data_vol.reload()
    need = [p for p in (AOK, COCO, PRIOR_05) if not Path(p).exists()]
    if need:
        raise RuntimeError(
            f"missing data on volume fyp-data: {need} (experiments/cloud/launch.sh uploads them)"
        )
    models = [
        "google/siglip-so400m-patch14-384",
        "lmms-lab/llava-onevision-projectors",
        "Qwen/Qwen2-7B-Instruct",
    ]
    if phase in ("1", "hp", "bal"):
        models.insert(1, "Qwen/Qwen2-0.5B-Instruct")
    pins = prefetch.remote(models)
    small = jobs_05b() if phase == "1" else jobs_hp() if phase == "hp" else []
    large = jobs_7b(phase) if phase in ("1", "2") else []
    if phase == "bal":
        small, large = jobs_bal()
    # If this container is ever restarted, it gets the same arguments (and so the same
    # launch token): re-attach to the calls already spawned instead of spawning them again.
    calls_dict = modal.Dict.from_name("fyp-calls", create_if_missing=True)
    input_id = token

    def _call(fn, job):  # noqa: ANN001, ANN202
        key = f"{input_id}:{job['name']}"
        cid = calls_dict.get(key) if input_id else None
        if cid:
            return modal.FunctionCall.from_id(cid)
        c = fn.spawn(job)
        if input_id:
            calls_dict[key] = c.object_id
        return c

    calls = [_call(run_small, j) for j in small] + [_call(run_large, j) for j in large]
    results = []
    for c, j in zip(calls, small + large):
        try:
            results.append(c.get())
        except Exception as e:  # noqa: BLE001
            results.append({"name": j["name"], "status": f"failed: {e}"})
        runs_vol.reload()
        status = {
            "phase": phase,
            "pins": pins,
            "jobs": results,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        Path(f"/runs/_status_phase{phase}.json").write_text(json.dumps(status, indent=2))
        runs_vol.commit()
    return {
        "phase": phase,
        "n_jobs": len(results),
        "ok": sum(r.get("status") in ("ok", "already_done") for r in results),
    }


@app.local_entrypoint()
def main(phase: str = "1", dry_run: bool = False) -> None:
    """Local entry point (launch.sh): count the phase's jobs and start orchestrate() for it."""
    small = jobs_05b() if phase == "1" else jobs_hp() if phase == "hp" else []
    large = jobs_7b(phase) if phase in ("1", "2") else []
    if phase == "bal":
        small, large = jobs_bal()
    print(f"phase {phase}: {len(small)} jobs on L4 (0.5B), {len(large)} jobs on L40S (7B)")
    if dry_run:
        for j in small + large:
            print(j["name"], " ".join(j["cmd"]))
        return
    token = f"{phase}-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}"
    print(orchestrate.spawn(phase, token).get())  # spawn().get(): recommended for long jobs
