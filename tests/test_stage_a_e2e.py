"""End-to-end tests of the evaluation pipeline (cmr.runners.stage_a, cmr.analysis.reliance).

Fake models with known behaviour give known extremes, so the tests check the metrics and not
just the plumbing:
    image-blind model    -> every image-reliance number must be exactly zero
    image-using model    -> accuracy collapses when the image is removed, swapped or blank
Also covers resume, text contexts (rationale, masked rationale, captions; Table 4.7), the
prior-wrong subset P (Section 3.5) and the run manifest.
"""

from pathlib import Path

import pytest
from PIL import Image

from cmr.analysis.reliance import summarize, to_markdown
from cmr.data.aokvqa import AOKVQAItem, load_aokvqa
from cmr.eval.conditions import ContextSource, ImageCondition
from cmr.eval.prompts import mc_prompt, rotate
from cmr.models.fake import LookupModel, PriorOnlyModel, image_key
from cmr.runners.stage_a import RunSpec, load_records, main, run_stage_a
from tests.conftest import colour_image

ALL = (ImageCondition.REAL, ImageCondition.NONE, ImageCondition.SWAPPED, ImageCondition.BLANK)


def loader(it: AOKVQAItem) -> Image.Image:
    return colour_image(it.image_id)


def rows_by_condition(summary: dict) -> dict:
    return {(r["condition"], r["context"]): r for r in summary["rows"]}


def test_image_blind_model_has_zero_visual_reliance(
    items: list[AOKVQAItem], tmp_path: Path
) -> None:
    recs = run_stage_a(items, PriorOnlyModel(), loader, RunSpec(conditions=ALL), tmp_path)
    rows = rows_by_condition(summarize(recs))
    for cond in ("none", "swapped", "blank"):
        r = rows[(cond, "none")]
        assert r["delta_acc_vs_ref"] == 0.0
        assert r["vig_nats"] == pytest.approx(0.0)
        assert r["flip_rate"] == 0.0
        assert r["mcnemar_p"] == 1.0


def _lookup_for(items: list[AOKVQAItem], rotations: int) -> LookupModel:
    answers = {}
    for it in items:
        for k in range(rotations):
            shown, original = rotate(it.choices, k)
            pos = original.index(it.correct_choice_idx or 0)
            answers[(mc_prompt(it.question, shown), image_key(colour_image(it.image_id)))] = pos
    return LookupModel(answers, confidence=8.0)


def test_image_using_model_shows_large_reliance(items: list[AOKVQAItem], tmp_path: Path) -> None:
    model = _lookup_for(items, rotations=4)
    spec = RunSpec(conditions=ALL, rotations=4)
    summary = summarize(run_stage_a(items, model, loader, spec, tmp_path))
    rows = rows_by_condition(summary)
    ref = rows[("real", "none")]
    assert ref["acc"] == 1.0 and ref["circ_acc"] == 1.0 and ref["perm_acc"] == 1.0
    for cond in ("none", "swapped", "blank"):
        r = rows[(cond, "none")]
        assert r["delta_acc_vs_ref"] > 0.4
        assert r["vig_nats"] > 1.0
    # "arbitration": real-image predictions compared with the no-image prediction (the prior)
    arb = summary["arbitration"]
    assert arb["acc_on_prior_wrong"] == 1.0  # the image rescues every prior error
    assert arb["blind_agreement"] < 0.6
    assert "Arbitration" in to_markdown(summary)


def test_resume_does_not_duplicate(items: list[AOKVQAItem], tmp_path: Path) -> None:
    spec = RunSpec(conditions=(ImageCondition.NONE,), contexts=(ContextSource.NONE,))
    run_stage_a(items[:10], PriorOnlyModel(), None, spec, tmp_path)
    run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)
    recs = load_records(tmp_path / "predictions.jsonl")
    assert len(recs) == len(items)
    assert len({r["question_id"] for r in recs}) == len(items)


def test_rationale_context_condition_runs_without_images(
    items: list[AOKVQAItem], tmp_path: Path
) -> None:
    spec = RunSpec(
        conditions=(ImageCondition.NONE,),
        contexts=(ContextSource.NONE, ContextSource.RATIONALE),
    )
    recs = run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)
    assert {r["context"] for r in recs} == {"none", "rationale"}


def test_image_conditions_require_loader(items: list[AOKVQAItem], tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_stage_a(items, PriorOnlyModel(), None, RunSpec(), tmp_path)


def test_caption_context_requires_captions(items: list[AOKVQAItem], tmp_path: Path) -> None:
    spec = RunSpec(conditions=(ImageCondition.NONE,), contexts=(ContextSource.CAPTION,))
    with pytest.raises(ValueError):
        run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)


def test_cli_end_to_end(aokvqa_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "run"
    main(
        [
            "--model", "fake-prior",
            "--aokvqa-dir", str(aokvqa_dir),
            "--coco-dir", str(aokvqa_dir.parent / "coco"),
            "--conditions", "real,none,swapped",
            "--rotations", "2",
            "--threads", "2",
            "--out", str(out),
        ]
    )  # fmt: skip
    from cmr.provenance import pin_threads

    torch_mod = pytest.importorskip("torch")
    threads_seen = torch_mod.get_num_threads()
    pin_threads(1)  # back to the test default
    assert threads_seen == 2
    assert (out / "summary.md").exists() and (out / "config.json").exists()
    n_expected = len(load_aokvqa(aokvqa_dir, "val")) * 3 * 2  # items x conditions x rotations
    assert n_expected == len(load_records(out / "predictions.jsonl"))
    import hashlib
    import json

    manifest = json.loads((out / "config.json").read_text())["manifest"]
    val = aokvqa_dir / "aokvqa_v1p0_val.json"
    assert manifest["data_sha256"] == {str(val): hashlib.sha256(val.read_bytes()).hexdigest()}
    assert {"git_commit", "git_dirty", "packages", "hf_models", "started_utc"} <= set(manifest)
    assert manifest["compute"]["torch_num_threads"] == 2  # pinned before the manifest
    assert manifest["compute"]["torch_interop_threads"] is not None


def test_cli_restart_keeps_first_manifest_and_appends_segments(
    aokvqa_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each (re)start appends a manifest segment; the first manifest is never overwritten."""
    import json

    from cmr.runners import stage_a

    out = tmp_path / "run"
    argv = ["--model", "fake-prior", "--aokvqa-dir", str(aokvqa_dir), "--conditions", "none",
            "--rotations", "2", "--threads", "1", "--out", str(out)]  # fmt: skip
    real = stage_a.run_manifest
    starts = iter([("c1" * 20, False, "T1"), ("c2" * 20, True, "T2"), ("c3" * 20, False, "T3")])

    def manifest(*args: object, **kwargs: object) -> dict[str, object]:
        commit, dirty, started = next(starts)
        return {**real(*args, **kwargs), "git_commit": commit, "git_dirty": dirty,
                "started_utc": started}  # fmt: skip

    monkeypatch.setattr(stage_a, "run_manifest", manifest)
    stage_a.main([*argv, "--limit", "5"])
    cfg_path = out / "config.json"
    first = json.loads(cfg_path.read_text())["manifest"]
    stage_a.main([*argv, "--limit", "8"])  # a restart at another commit: 3 more items
    stage_a.main([*argv, "--limit", "8"])  # a re-run of the finished run
    m = json.loads(cfg_path.read_text())["manifest"]
    assert {k: v for k, v in m.items() if k != "segments"} == {
        k: v for k, v in first.items() if k != "segments"
    }
    assert m["git_commit"] == "c1" * 20 and m["started_utc"] == "T1"
    assert [(s["git_commit"][:2], s["git_dirty"], s["started_utc"], s["first_record_index"],
             s["torch_num_threads"]) for s in m["segments"]] == [
        ("c1", False, "T1", 0, 1), ("c2", True, "T2", 10, 1), ("c3", False, "T3", 16, 1)
    ]  # fmt: skip
    assert len(load_records(out / "predictions.jsonl")) == 16
    # a config written before segments existed: its manifest becomes the first segment
    legacy = json.loads(cfg_path.read_text())
    legacy["manifest"] = {k: v for k, v in first.items() if k != "segments"}
    cfg_path.write_text(json.dumps(legacy))
    starts = iter([("c4" * 20, False, "T4")])
    stage_a.main([*argv, "--limit", "8"])
    m = json.loads(cfg_path.read_text())["manifest"]
    assert m["git_commit"] == "c1" * 20
    assert [(s["git_commit"][:2], s["first_record_index"]) for s in m["segments"]] == [
        ("c1", None),
        ("c4", 16),
    ]


def test_masked_rationale_context_and_leak_flag(items: list[AOKVQAItem], tmp_path: Path) -> None:
    spec = RunSpec(
        conditions=(ImageCondition.NONE,),
        contexts=(ContextSource.RATIONALE, ContextSource.RATIONALE_MASKED),
    )
    recs = run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)
    assert {r["context"] for r in recs} == {"rationale", "rationale_masked"}
    assert all(isinstance(r["rationale_leak"], bool) for r in recs)
    assert all(isinstance(r["rationale_leak_stem"], bool) for r in recs)
    masked = [r for r in recs if r["context"] == "rationale_masked"]
    assert all(r["context_leak"] is False and r["mask_version"] == 2 for r in masked)


def test_caption_leak_flag_and_leak_free_row(items: list[AOKVQAItem], tmp_path: Path) -> None:
    captions = {
        it.image_id: [f"A photo showing {it.choices[it.correct_choice_idx or 0]}."]
        if it.image_id % 2
        else ["A photo."]
        for it in items
    }
    spec = RunSpec(conditions=(ImageCondition.NONE,), contexts=(ContextSource.CAPTION,))
    recs = run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path, captions=captions)
    assert {r["caption_leak"] for r in recs} == {True, False}
    assert all(r["caption_leak"] == (r["question_id"] in {
        it.question_id for it in items if it.image_id % 2
    }) for r in recs)  # fmt: skip
    row = rows_by_condition(summarize(recs))[("none", "caption")]
    assert row["n_leak_free"] == sum(1 for it in items if it.image_id % 2 == 0)
    assert "Leak-free subset" in to_markdown(summarize(recs))


def test_tie_credit_and_ragged_arbitration() -> None:
    import numpy as np

    from cmr.analysis.reliance import arbitration_analysis, tie_credit

    assert tie_credit(np.zeros(4), 2) == 0.25  # image-blind model at alpha=1: chance, not "A"
    assert tie_credit(np.array([0.0, 1.0, 1.0]), 1) == 0.5
    assert tie_credit(np.array([3.0, 1.0]), 1) == 0.0

    def rec(lps: list[float], gold: int) -> dict:
        return {0: {"option_logprobs": lps, "correct_idx": gold, "pred_idx": int(np.argmax(lps))}}

    real = {"a": rec([-0.1, -3.0], 0), "b": rec([-2.0, -0.2, -3.0, -4.0], 1)}
    none = {"a": rec([-3.0, -0.1], 0), "b": rec([-2.0, -0.2, -3.0, -4.0], 1)}
    out = arbitration_analysis(real, none, ["a", "b"])  # 2- and 4-option questions mixed
    assert out["n_prior_wrong"] == 1 and out["acc_on_prior_wrong"] == 1.0
    assert out["debiased_acc"]["1"] == pytest.approx((1.0 + 0.25) / 2)


# --------------------------------------------------------------------------- prior-wrong / debias


def _rec(q: str, cond: str, rot: int, option_lp: list[float], gold: int, ctx: str = "none") -> dict:
    import numpy as np

    lp = list(np.log(np.asarray(option_lp) / np.sum(option_lp)))
    return {
        "question_id": q, "condition": cond, "context": ctx, "rotation": rot,
        "option_logprobs": lp, "pred_idx": int(np.argmax(lp)), "correct_idx": gold,
    }  # fmt: skip


def test_prior_wrong_contrasts_use_rotation_averaged_prior_and_controls() -> None:
    # q0: prior right at rotation 0 only (letter bias); rotation-averaged it is wrong.
    # q1: prior wrong everywhere; real right, swapped wrong.  q2: prior right everywhere.
    recs = [
        _rec("q0", "none", 0, [0.7, 0.1, 0.1, 0.1], 0),
        _rec("q0", "none", 1, [0.1, 0.1, 0.1, 0.7], 0),  # mass moved to option 3 = letter A
        _rec("q1", "none", 0, [0.1, 0.7, 0.1, 0.1], 0),
        _rec("q1", "none", 1, [0.1, 0.7, 0.1, 0.1], 0),
        _rec("q2", "none", 0, [0.7, 0.1, 0.1, 0.1], 0),
        _rec("q2", "none", 1, [0.7, 0.1, 0.1, 0.1], 0),
    ]
    for cond, right in (("real", {"q0", "q1", "q2"}), ("swapped", {"q0", "q2"}), ("blank", {"q2"})):
        for q in ("q0", "q1", "q2"):
            for k in (0, 1):
                recs.append(
                    _rec(
                        q, cond, k, [0.7, 0.1, 0.1, 0.1] if q in right else [0.1, 0.7, 0.1, 0.1], 0
                    )
                )
    s = summarize(recs)
    assert s["prior"]["source"] == "own" and s["prior"]["n_prior_wrong"] == 2  # q0, q1
    assert s["arbitration"]["n_prior_wrong"] == 1  # rotation-0 definition
    real = rows_by_condition(s)[("real", "none")]
    assert real["acc_prior_wrong"] == 1.0
    assert real["acc_prior_wrong_real_minus_swapped"] == pytest.approx(0.5)
    assert real["acc_prior_wrong_real_minus_blank"] == pytest.approx(1.0)
    assert rows_by_condition(s)[("swapped", "none")]["acc_prior_wrong"] == 0.5
    assert s["prior"]["blind_agreement_bound"] >= s["prior"]["blind_agreement"]
    md = to_markdown(s)
    assert "Prior-wrong subset P" in md and "swapped: +50.0" in md

    # a fixed external prior overrides the run's own NONE group
    ext = [_rec(q, "none", 0, [0.7, 0.1, 0.1, 0.1], 0) for q in ("q0", "q1", "q2")]
    s2 = summarize(recs, prior_records=ext)
    assert s2["prior"]["source"] == "external" and s2["prior"]["n_prior_wrong"] == 0
    assert rows_by_condition(s2)[("real", "none")]["acc_prior_wrong"] is None
    with pytest.raises(ValueError):
        summarize(recs, prior_records=[_rec("zz", "none", 0, [0.5, 0.5], 0)])
    with pytest.raises(ValueError):
        summarize(recs, prior_records=[_rec("q0", "real", 0, [0.5, 0.5], 0)])


def test_rotation_averaged_debias_cancels_letter_bias_and_reports_blank_prior() -> None:
    # Pure letter-A bias in both terms and no content: at rotation 0 the debiased score
    # still depends on letters; averaged over all 4 rotations every option ties -> 1/4.
    recs = []
    for q, gold in (("a", 0), ("b", 2)):
        for k in range(4):
            biased = [0.1] * 4
            biased[(0 + k) % 4] = 0.7  # letter A shows original option k
            for cond in ("real", "none", "blank"):
                recs.append(_rec(q, cond, k, biased, gold))
    arb = summarize(recs)["arbitration"]
    assert arb["debias_rotations"] == 4
    assert arb["debiased_acc_rotavg"]["1"] == pytest.approx(0.25)
    assert arb["debiased_tie_rate_rotavg"] == 1.0
    assert arb["debiased_acc_blank_prior"]["1"] == pytest.approx(0.25)
    assert arb["debiased_acc_rotavg"]["0"] == pytest.approx(0.25)  # rotation-averaged raw
    assert "debiased_acc" in arb and "debiased_tie_rate_rot0" in arb  # rotation-0 fields kept


def test_image_using_model_has_visual_part_on_prior_wrong(
    items: list[AOKVQAItem], tmp_path: Path
) -> None:
    spec = RunSpec(conditions=ALL, rotations=4)
    s = summarize(run_stage_a(items, _lookup_for(items, 4), loader, spec, tmp_path / "u"))
    real = rows_by_condition(s)[("real", "none")]
    assert real["acc_prior_wrong"] == 1.0
    assert real["acc_prior_wrong_real_minus_swapped"] > 0.5
    assert real["acc_prior_wrong_real_minus_blank"] > 0.5
    blind = summarize(run_stage_a(items, PriorOnlyModel(), loader, spec, tmp_path / "b"))
    if blind["prior"]["n_prior_wrong"]:
        assert rows_by_condition(blind)[("real", "none")]["acc_prior_wrong_real_minus_swapped"] == 0


def test_resume_refuses_mixed_mask_versions(items: list[AOKVQAItem], tmp_path: Path) -> None:
    import json

    spec = RunSpec(conditions=(ImageCondition.NONE,), contexts=(ContextSource.RATIONALE_MASKED,))
    run_stage_a(items[:3], PriorOnlyModel(), None, spec, tmp_path)
    run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)  # same version: resumes fine
    path = tmp_path / "predictions.jsonl"
    recs = load_records(path)
    del recs[0]["mask_version"]  # a mask-v1 record
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    with pytest.raises(ValueError, match="mask v1"):
        run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path)


# "Instrument" runs score the frozen LLM without an image, with or without a text context
# (Table 4.7; results/runs/laptop/pilot_a1v2_instrument, which also defines P).
def test_instrument_only_run_gets_prior_block_without_blind_agreement(
    items: list[AOKVQAItem], tmp_path: Path
) -> None:
    spec = RunSpec(conditions=(ImageCondition.NONE,), contexts=(ContextSource.NONE,), rotations=2)
    s = summarize(run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path))
    assert s["prior"]["source"] == "own" and "blind_agreement" not in s["prior"]
    assert "arbitration" not in s


def test_instrument_run_contexts_get_deltas_against_the_no_context_prior(
    items: list[AOKVQAItem], tmp_path: Path
) -> None:
    # no (real, none) group -> (none, none) is the reference
    spec = RunSpec(
        conditions=(ImageCondition.NONE,),
        contexts=(ContextSource.NONE, ContextSource.RATIONALE),
        rotations=2,
    )
    s = summarize(run_stage_a(items, PriorOnlyModel(), None, spec, tmp_path))
    assert s["reference"] == ["none", "none"]
    r = rows_by_condition(s)
    ref, ctx = r[("none", "none")], r[("none", "rationale")]
    assert "delta_acc_vs_ref" not in ref and "vig_nats" not in ref
    assert ctx["delta_acc_vs_ref"] == pytest.approx(ref["acc"] - ctx["acc"])
    assert ctx["vig_nats"] == pytest.approx(ref["mean_logp_correct"] - ctx["mean_logp_correct"])
    assert ctx["vig_nats"] != 0.0  # the context changes the fake model's scores
    assert "arbitration" not in s and "blind_agreement" not in s["prior"]
    assert "| ref |" in to_markdown(s)
