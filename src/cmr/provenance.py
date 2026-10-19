"""Run manifest: the code, packages, data, weights, threads and CPU behind a run.

Stored under ``"manifest"`` in a run's config file (``config.json`` or ``run_args.json``).
Every field degrades to ``None`` rather than failing: a manifest must never stop a run.
Also supplies thread pinning and the runners' ``--threads`` default.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import platform
import subprocess
from collections.abc import Iterable
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

# packages whose installed versions go into the manifest
PACKAGES = ("torch", "transformers", "peft", "numpy")
THREADS_ENV = "CMR_THREADS"  # default for the runners' --threads
DEFAULT_THREADS = 4
_COMPONENTS = ("model", "llm", "vision")  # attributes holding HF modules in cmr adapters


def _git(*args: str) -> str | None:
    """Output of ``git args`` in the package's own checkout (the code that ran), or None."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_state() -> tuple[str | None, bool | None]:
    """(HEAD sha, tracked files modified?); (None, None) outside a git checkout.

    ``dirty`` ignores untracked files and is None when ``git status`` fails.
    """
    sha = _git("rev-parse", "HEAD")
    if not sha:
        return None, None
    status = _git("status", "--porcelain", "--untracked-files=no")
    return sha, None if status is None else bool(status)


def _version(package: str) -> str | None:
    """Installed version of ``package`` (e.g. ``"2.4.1"``), or None if not installed."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def sha256_file(path: str | Path) -> str | None:
    """Hex sha256 of a file, streamed in 1 MiB blocks; None if it does not exist.

    >>> import tempfile
    >>> p = Path(tempfile.mkdtemp()) / "x.txt"
    >>> _ = p.write_bytes(b"abc")
    >>> sha256_file(p)[:16]
    'ba7816bf8f01cfea'
    >>> sha256_file(p.with_name("missing.txt")) is None
    True
    """
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weights_file(path: str) -> str:
    """Local file behind a weights path (``hf://`` resolved via the HF cache), else ``path``."""
    try:
        from cmr.models.bridge import resolve_path

        return resolve_path(path)
    except Exception:  # offline, no torch, bad path: the manifest must never stop a run
        return path


def hf_models(model: Any, _depth: int = 0) -> dict[str, dict[str, str | None]]:
    """``{attribute: {"id", "revision"}}`` for every HF module inside a cmr model.

    Looks one wrapper deep (e.g. cmr.runners.stage_b.FormatAdapter -> FrozenBridgeVLM ->
    .llm / .vision).
    ``revision`` is the hub commit hash, None for local paths.
    """
    out: dict[str, dict[str, str | None]] = {}
    for attr in _COMPONENTS:
        sub = getattr(model, attr, None)
        if sub is None or sub is model:
            continue
        cfg = getattr(sub, "config", None)
        if cfg is not None and getattr(cfg, "_name_or_path", None):
            out[attr] = {
                "id": str(cfg._name_or_path),
                "revision": getattr(cfg, "_commit_hash", None),
            }
        elif _depth == 0:
            out.update(hf_models(sub, _depth + 1))
    return out


def default_threads() -> int:
    """``$CMR_THREADS`` if set, else :data:`DEFAULT_THREADS` (the runners' --threads default).

    Raises:
        ValueError: ``$CMR_THREADS`` is not an integer or is < 1.
    """
    raw = os.environ.get(THREADS_ENV, "").strip()
    if not raw:
        return DEFAULT_THREADS
    try:
        n = int(raw)
    except ValueError:
        raise ValueError(f"{THREADS_ENV}={raw!r} is not an integer") from None
    if n < 1:
        raise ValueError(f"{THREADS_ENV} must be >= 1, got {n}")
    return n


def pin_threads(n: int) -> None:
    """``torch.set_num_threads(n)`` and one inter-op thread, for reproducible numerics.

    GEMM reductions are split across threads, so the thread count changes the last bits.
    Call before any model work. An inter-op count already fixed is left as is.

    Raises:
        ValueError: ``n`` < 1.
    """
    if n < 1:
        raise ValueError(f"--threads must be >= 1, got {n}")
    try:
        import torch
    except ImportError:  # torch-free runs (--model fake-prior) have nothing to pin
        return
    torch.set_num_threads(n)
    if torch.get_num_interop_threads() != 1:
        # RuntimeError: already set, or inter-op work has started in this process
        with contextlib.suppress(RuntimeError):
            torch.set_num_interop_threads(1)


def cpu_info(cpuinfo: str | Path = "/proc/cpuinfo") -> dict[str, Any]:
    """CPU model, sorted feature flags and count, from the first block of ``/proc/cpuinfo``.

    Falls back to ``platform.processor()`` / None when the file or a field is missing.

        >>> import tempfile
        >>> p = Path(tempfile.mkdtemp()) / "cpuinfo"
        >>> _ = p.write_text("model name : Toy CPU\\nflags : sse2 avx2\\n\\nmodel name : B\\n")
        >>> info = cpu_info(p)
        >>> info["cpu_model"], info["cpu_flags"]
        ('Toy CPU', 'avx2 sse2')
    """
    fields: dict[str, str] = {}
    try:
        with open(cpuinfo) as f:
            for line in f:
                if not line.strip():
                    if fields:
                        break  # end of the first processor block
                    continue
                key, _, value = line.partition(":")
                fields.setdefault(key.strip(), value.strip())
    except OSError:
        pass
    flags = fields.get("flags") or fields.get("Features")
    return {
        "cpu_model": fields.get("model name") or platform.processor() or None,
        "cpu_flags": " ".join(sorted(flags.split())) if flags else None,
        "cpu_count": os.cpu_count(),
    }


def compute_info() -> dict[str, Any]:
    """Threads torch uses, the CPU ISA it dispatches to, thread env vars and :func:`cpu_info`."""
    info: dict[str, Any] = {
        "torch_num_threads": None,
        "torch_interop_threads": None,
        "torch_cpu_capability": None,
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        THREADS_ENV.lower() + "_env": os.environ.get(THREADS_ENV),
    }
    try:
        import torch

        info["torch_num_threads"] = torch.get_num_threads()
        info["torch_interop_threads"] = torch.get_num_interop_threads()
        info["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    except Exception:  # no torch / old torch: the manifest must never stop a run
        pass
    return {**info, **cpu_info()}


def run_manifest(data_paths: Iterable[str | Path | None], model: Any = None) -> dict[str, Any]:
    """Git state, package versions, data hashes, HF revisions, compute info, start time.

    Args:
        data_paths: files the run read; None entries are skipped.
        model: the loaded cmr model, for :func:`hf_models` and the initial projector hash.
    """
    sha, dirty = git_state()
    models: dict[str, dict[str, str | None]] = hf_models(model) if model is not None else {}
    # the bridge's initial projector: hash the (cached) file
    spec = getattr(model, "spec", None) or getattr(getattr(model, "model", None), "spec", None)
    if spec is not None and getattr(spec, "projector_path", None):
        path = str(spec.projector_path)
        models["projector"] = {
            "id": path,
            "revision": None,
            "sha256": sha256_file(weights_file(path)),
        }
    return {
        "git_commit": sha,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "packages": {p: _version(p) for p in PACKAGES},
        "data_sha256": {str(p): sha256_file(p) for p in data_paths if p is not None},
        "hf_models": models,
        "compute": compute_info(),
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# manifest fields copied into each segment (one segment per start of a run)
SEGMENT_FIELDS = ("git_commit", "git_dirty", "started_utc")


def _segment(manifest: dict[str, Any], position: dict[str, Any]) -> dict[str, Any]:
    """One ``manifest["segments"]`` entry: :data:`SEGMENT_FIELDS`, threads, ``position``."""
    compute = manifest.get("compute")
    threads = compute.get("torch_num_threads") if isinstance(compute, dict) else None
    return {
        **{k: manifest.get(k) for k in SEGMENT_FIELDS},
        "torch_num_threads": threads,
        **position,
    }


def continue_manifest(
    old_config: dict[str, Any] | None, manifest: dict[str, Any], **position: Any
) -> dict[str, Any]:
    """The manifest a (re)started run writes: the first manifest plus one segment per start.

    Args:
        old_config: the run's config before this start (None or no manifest: a fresh run).
        manifest: this start's :func:`run_manifest`.
        position: where this start picks up its work, e.g. ``start_step``.

    Example (a run started at commit ``abc``, killed, and restarted at ``def``):

        >>> m = {"git_commit": "abc", "git_dirty": False, "started_utc": "T1",
        ...      "compute": {"torch_num_threads": 4}}
        >>> first = continue_manifest(None, m, start_step=0)
        >>> sorted(first["segments"][0].items())  # doctest: +NORMALIZE_WHITESPACE
        [('git_commit', 'abc'), ('git_dirty', False), ('start_step', 0),
         ('started_utc', 'T1'), ('torch_num_threads', 4)]
        >>> later = {**m, "git_commit": "def", "started_utc": "T2"}
        >>> again = continue_manifest({"manifest": first}, later, start_step=120)
        >>> again["git_commit"], [s["git_commit"] for s in again["segments"]]
        ('abc', ['abc', 'def'])
    """
    seg = _segment(manifest, position)
    old = old_config.get("manifest") if isinstance(old_config, dict) else None
    if not isinstance(old, dict):
        return {**manifest, "segments": [seg]}
    segments = old.get("segments")
    if not isinstance(segments, list):
        segments = [_segment(old, dict.fromkeys(position))]
    return {**old, "segments": [*segments, seg]}
