"""Run manifest (cmr.provenance), stored in every run's run_args.json or config.json.

It records the git commit, package versions, SHA-256 of the data files, model ids and
revisions, thread counts, and one segment per (re)start of a resumed run; it must still be
written outside a git checkout.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cmr import provenance
from cmr.provenance import hf_models, run_manifest, sha256_file


def test_manifest_records_git_versions_data_and_time(tmp_path: Path) -> None:
    data = tmp_path / "d.json"
    data.write_text('{"x": 1}')
    m = run_manifest([data, None, tmp_path / "missing.json"])
    assert m["data_sha256"] == {
        str(data): hashlib.sha256(data.read_bytes()).hexdigest(),
        str(tmp_path / "missing.json"): None,
    }
    assert set(m["packages"]) == {"torch", "transformers", "peft", "numpy"}
    assert m["packages"]["numpy"] and m["python"].count(".") == 2
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", m["started_utc"])
    pkg = Path(provenance.__file__).parent
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=pkg, capture_output=True, text=True)
    if head.returncode == 0:  # running from a checkout (CI, dev); an sdist has no git
        assert m["git_commit"] == head.stdout.strip() and isinstance(m["git_dirty"], bool)


def test_manifest_degrades_outside_git(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(provenance.subprocess, "run", no_git)
    m = run_manifest([])
    assert m["git_commit"] is None and m["git_dirty"] is None
    assert sha256_file(tmp_path) is None  # a directory is not a data file


def test_hf_models_finds_ids_and_revisions_through_wrappers() -> None:
    def hf(name: str, rev: str | None) -> SimpleNamespace:
        return SimpleNamespace(config=SimpleNamespace(_name_or_path=name, _commit_hash=rev))

    bridge = SimpleNamespace(llm=hf("org/llm", "abc"), vision=hf("org/vit", None))
    adapter = SimpleNamespace(model=bridge)  # stage_b.FormatAdapter -> FrozenBridgeVLM
    assert hf_models(adapter) == {
        "llm": {"id": "org/llm", "revision": "abc"},
        "vision": {"id": "org/vit", "revision": None},
    }
    assert hf_models(SimpleNamespace(model=hf("org/vlm", "def"))) == {
        "model": {"id": "org/vlm", "revision": "def"}
    }
    assert run_manifest([], object())["hf_models"] == {}


def test_manifest_hashes_the_bridge_init_projector(tmp_path: Path) -> None:
    weights = tmp_path / "mm_projector.bin"
    weights.write_bytes(b"\x00\x01weights")
    spec = SimpleNamespace(projector_path=str(weights))
    expected = {
        "id": str(weights),
        "revision": None,
        "sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    }
    assert run_manifest([], SimpleNamespace(spec=spec))["hf_models"]["projector"] == expected
    adapter = SimpleNamespace(model=SimpleNamespace(spec=spec))  # stage_b.FormatAdapter
    assert run_manifest([], adapter)["hf_models"]["projector"] == expected
    no_init = SimpleNamespace(spec=SimpleNamespace(projector_path=None))
    assert "projector" not in run_manifest([], no_init)["hf_models"]


# ----------------------------------------------------------------------------- threads / CPU


def test_default_threads_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(provenance.THREADS_ENV, raising=False)
    assert provenance.default_threads() == provenance.DEFAULT_THREADS == 4
    monkeypatch.setenv(provenance.THREADS_ENV, "3")
    assert provenance.default_threads() == 3
    for bad in ("x", "0"):
        monkeypatch.setenv(provenance.THREADS_ENV, bad)
        with pytest.raises(ValueError, match=provenance.THREADS_ENV):
            provenance.default_threads()


def test_pin_threads_sets_intra_and_interop() -> None:
    torch = pytest.importorskip("torch")
    before = torch.get_num_threads()
    try:
        provenance.pin_threads(2)
        assert torch.get_num_threads() == 2
        info = provenance.compute_info()
        assert info["torch_num_threads"] == 2 and info["torch_interop_threads"] >= 1
        assert info["torch_cpu_capability"]
    finally:
        provenance.pin_threads(before)
    with pytest.raises(ValueError, match="threads"):
        provenance.pin_threads(0)


def test_cpu_info_parses_first_processor_block(tmp_path: Path) -> None:
    f = tmp_path / "cpuinfo"
    f.write_text(
        "processor\t: 0\nmodel name\t: Test CPU @ 2GHz\nflags\t\t: sse2 avx2 amx_bf16\n\n"
        "processor\t: 1\nmodel name\t: Other\nflags\t\t: sse2\n"
    )
    info = provenance.cpu_info(f)
    assert info["cpu_model"] == "Test CPU @ 2GHz"
    assert info["cpu_flags"] == "amx_bf16 avx2 sse2"
    missing = provenance.cpu_info(tmp_path / "none")
    assert missing["cpu_flags"] is None and missing["cpu_count"]
    assert set(run_manifest([])["compute"]) >= {"torch_num_threads", "cpu_model", "cpu_flags"}


def _manifest(commit: str | None, started: str, threads: int = 4) -> dict[str, object]:
    return {
        "git_commit": commit,
        "git_dirty": False,
        "packages": {"torch": commit},
        "compute": {"torch_num_threads": threads},
        "started_utc": started,
    }


def test_continue_manifest_keeps_the_first_start_and_appends_segments() -> None:
    """A resume never overwrites git_commit / started_utc."""
    from cmr.provenance import continue_manifest

    first = continue_manifest(None, _manifest("aaa", "t1"), first_record_index=0)
    assert first == {
        **_manifest("aaa", "t1"),
        "segments": [{"git_commit": "aaa", "git_dirty": False, "started_utc": "t1",
                      "torch_num_threads": 4, "first_record_index": 0}],
    }  # fmt: skip
    second = continue_manifest(
        {"manifest": first}, _manifest("bbb", "t2", threads=2), first_record_index=96
    )
    assert {k: v for k, v in second.items() if k != "segments"} == _manifest("aaa", "t1")
    assert second["segments"] == [
        first["segments"][0],
        {"git_commit": "bbb", "git_dirty": False, "started_utc": "t2",
         "torch_num_threads": 2, "first_record_index": 96},
    ]  # fmt: skip
    for fresh in (None, {}, {"manifest": None}, {"manifest": "garbage"}):  # no old manifest
        assert continue_manifest(fresh, _manifest("c", "t"))["git_commit"] == "c"


def test_continue_manifest_backfills_a_config_written_before_segments() -> None:
    from cmr.provenance import continue_manifest

    legacy = {"manifest": _manifest("old", "t0", threads=1)}  # no "segments" key
    m = continue_manifest(legacy, _manifest("new", "t9"), start_step=4, first_record_index=0)
    assert m["git_commit"] == "old" and m["started_utc"] == "t0"
    assert m["segments"] == [
        {"git_commit": "old", "git_dirty": False, "started_utc": "t0",
         "torch_num_threads": 1, "start_step": None, "first_record_index": None},
        {"git_commit": "new", "git_dirty": False, "started_utc": "t9",
         "torch_num_threads": 4, "start_step": 4, "first_record_index": 0},
    ]  # fmt: skip


def test_resume_guard_reads_the_last_segments_thread_count() -> None:
    from cmr.runners.stage_a import scoring_settings

    cfg = {"manifest": {**_manifest("a", "t", threads=4), "segments": [
        {"torch_num_threads": 4}, {"torch_num_threads": 2}]}}  # fmt: skip
    assert scoring_settings(cfg)["torch_num_threads"] == 2
    cfg["manifest"]["segments"] = []  # type: ignore[index]
    assert scoring_settings(cfg)["torch_num_threads"] == 4
