"""
Regression gate on the recorded evaluation results.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_eval_regression.py

Reads the committed JSON artefacts rather than calling the model, so it is free,
deterministic, offline, and safe for CI. That means it does not detect drift in
the live model -- it detects someone changing the prompt, the retrieval settings,
or the eval set and committing worse numbers. Regenerating the artefacts is the
deliberate, quota-spending act:

    python -m medbot.eval.refusal_trials --trials 3 --out-prefix sprint4_
    python -m medbot.eval.run_eval --variant cot

Thresholds are set at the values actually recorded in Sprint 4, not at
aspirational ones. A test that passes with room to spare is not a gate.
"""

import glob
import json
import os

import pytest

from medbot.eval.dataset import EVAL_QUESTIONS
from medbot.eval.refusal_trials import (
    OVERANSWER_QUESTIONS,
    REFUSAL_QUESTIONS,
    is_refusal,
)

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "medbot", "eval")

# Recorded in Sprint 3 and unchanged by Sprint 4's prompt work. Retrieval is
# identical across arms by construction (the variants differ only in the prompt),
# so a drop here means the index or the retriever changed, not the prompt.
MIN_PRECISION_AT_K = 0.83

# The dataset the standing headline rests on: 24 questions x 2 arms x 5 trials,
# scored by the corrected `is_refusal`, baseline 7/24 vs cot 0/24, p = 0.0094.
#
# These gates read `sprint4_t5_` and not `sprint4_`. The 3-trial `sprint4_` run is
# the SUPERSEDED one — re-scored it is 3/24 vs 0/24, p = 0.2340, not significant —
# and every gate in this file used to point at it, so the artefact behind the
# result being claimed had no regression gate at all while the retired one had
# four. Found by the 2026-07-27 audit. The 3-trial file is still covered by the
# label-drift test below, which globs every trial file rather than naming two.
REFUSAL_TRIALS = "sprint4_t5_refusal_trials.json"
TRIALS_PER_CELL = 5


def _load(name):
    path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not generated yet; run the harness to create it")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _refusal_counts(trials, variant):
    """-> {question: (n_refused, n_trials)} for one arm."""
    out = {}
    for question, by_variant in trials.items():
        attempts = by_variant.get(variant, [])
        out[question] = (sum(1 for a in attempts if a["refusal"]), len(attempts))
    return out


# --- the headline result ---------------------------------------------------

def test_cot_never_falsely_refuses():
    """
    Sprint 3's headline, re-gated on Sprint 4's 24-question suite.

    Every question here has a dedicated encyclopedia entry in the indexed corpus,
    so a refusal is always a bug -- there is no correct refusal in this file.
    """
    trials = _load(REFUSAL_TRIALS)
    counts = _refusal_counts(trials, "cot")
    offenders = {q: c for q, (c, _) in counts.items() if c}
    assert not offenders, f"cot arm refused questions the corpus answers: {offenders}"


def test_cot_still_beats_baseline_on_refusals():
    """Guards the direction of the effect, not just cot's absolute score."""
    trials = _load(REFUSAL_TRIALS)
    base = sum(c for c, _ in _refusal_counts(trials, "baseline").values())
    cot = sum(c for c, _ in _refusal_counts(trials, "cot").values())
    assert cot < base, f"cot ({cot}) no longer refuses less than baseline ({base})"


# --- the guard against the opposite failure --------------------------------

def test_cot_still_refuses_out_of_corpus_questions():
    """
    The failure mode the CoT change pushes toward: the cheap way to never refuse
    is to answer everything, which on medical questions means inventing.

    Every question in this file is one the corpus does NOT cover -- verified per
    question with `medbot.eval.verify_coverage`, not inferred from its topic --
    so here a refusal is the CORRECT answer and an answer is a hallucination.

    Runs on whatever trials were recorded, including a partial run: incomplete
    data can still catch a real over-answer, it just cannot establish absence.
    Completeness is the separate test below.
    """
    trials = _load("sprint4_overanswer_trials.json")
    counts = _refusal_counts(trials, "cot")
    answered = {q: (n - c) for q, (c, n) in counts.items() if c < n and n > 0}
    assert not answered, (
        f"cot answered out-of-corpus questions instead of refusing: {answered}. "
        "Check the answers by hand -- either the model invented content, or the "
        "question is not actually out-of-corpus (see audit F8)."
    )


def test_out_of_corpus_gate_is_fully_armed():
    """
    Separate from the assertion above on purpose: that one runs on whatever was
    recorded, this one asserts that everything was recorded.

    The run originally died on the free tier's 500 requests/day at question 2 of
    10, so this skipped, which kept the suite green while making the gap visible.
    It was completed on 2026-07-27 (60/60) — and the skip then became the defect:
    with the data complete, `missing` is empty and the body asserted **nothing**,
    so the test named "fully armed" could no longer fail. Truncating the guard
    data would have re-armed the skip rather than failing the build. Found by the
    2026-07-27 audit; it now asserts completeness and the refusals themselves.
    """
    trials = _load("sprint4_overanswer_trials.json")
    missing = [
        q for q in OVERANSWER_QUESTIONS
        if not trials.get(q, {}).get("cot") or not trials.get(q, {}).get("baseline")
    ]
    assert not missing, (
        f"out-of-corpus gate not armed: {len(OVERANSWER_QUESTIONS) - len(missing)}"
        f"/{len(OVERANSWER_QUESTIONS)} questions have trials. Missing: {missing}. "
        "Re-run: python -m medbot.eval.refusal_trials --trials 3 --suite overanswer "
        "--out-prefix sprint4_ --resume"
    )

    leaked = {}
    for question in OVERANSWER_QUESTIONS:
        for variant in ("baseline", "cot"):
            attempts = trials[question][variant]
            answered = [i for i, a in enumerate(attempts) if not a["refusal"]]
            if answered:
                leaked[f"{question} [{variant}]"] = answered
    assert not leaked, f"invented answers on out-of-corpus questions: {leaked}"


# --- retrieval -------------------------------------------------------------

def test_retrieval_precision_has_not_dropped():
    results = _load("results_cot.json")
    actual = results["summary"]["mean_precision_at_k"]
    assert actual >= MIN_PRECISION_AT_K, (
        f"Precision@4 fell to {actual:.3f} from {MIN_PRECISION_AT_K}"
    )


def test_retrieval_is_identical_across_arms():
    """
    The A/B is only a prompt comparison if retrieval is held fixed. If these ever
    diverge, every groundedness delta in results_sprint3.md is confounded.
    """
    base = {r["question"]: r["precision_at_k"] for r in _load("results_baseline.json")["results"]}
    cot = {r["question"]: r["precision_at_k"] for r in _load("results_cot.json")["results"]}
    assert base == cot, "per-question Precision@4 differs between arms"


# --- suite integrity -------------------------------------------------------

def test_refusal_suite_covers_the_whole_eval_set():
    """The suite is the eval set; hand-picking a subset is what audit F1 corrected."""
    assert set(REFUSAL_QUESTIONS) == {q["question"] for q in EVAL_QUESTIONS}


def test_the_two_suites_do_not_overlap():
    """A question cannot both require an answer and require a refusal."""
    assert not set(REFUSAL_QUESTIONS) & set(OVERANSWER_QUESTIONS)


def test_out_of_corpus_suite_is_large_enough():
    """Audit F7: the guard rested on 2 questions, which cannot establish absence."""
    assert len(OVERANSWER_QUESTIONS) >= 8


def test_recorded_refusal_trials_cover_the_whole_suite():
    """
    A silently truncated run would otherwise make the headline gate vacuous. The
    trial count is asserted too: the headline is a rate per question, so a cell
    holding fewer trials than the run claims is a different measurement.
    """
    trials = _load(REFUSAL_TRIALS)
    assert set(trials) == set(REFUSAL_QUESTIONS), "refusal trials do not match the suite"
    for question, by_variant in trials.items():
        for variant in ("baseline", "cot"):
            attempts = by_variant.get(variant)
            assert attempts, f"{question} has no {variant} trials"
            assert len(attempts) == TRIALS_PER_CELL, (
                f"{question} [{variant}] has {len(attempts)} trials, not {TRIALS_PER_CELL}"
            )


def test_refusal_labels_match_the_heuristic():
    """
    The stored `refusal` booleans must still be what `is_refusal` says about the
    stored text. If either reader in the gate is edited, this fails and the
    hand-validation behind those labels has to be redone rather than silently
    inherited — `medbot.eval.relabel` is the tool for applying it.

    Globs every trial file rather than naming two. It named `sprint4_refusal` and
    `sprint4_overanswer` until the 2026-07-27 audit, which left 8 of the 10 trial
    files unguarded, including both files the current headline and the ablation
    conclusion rest on.
    """
    paths = sorted(glob.glob(os.path.join(EVAL_DIR, "*trials.json")))
    assert len(paths) >= 8, f"expected the recorded trial files, found {len(paths)}"
    for path in paths:
        name = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for question, by_variant in data.items():
            for variant, attempts in by_variant.items():
                for i, a in enumerate(attempts):
                    assert a["refusal"] == is_refusal(a["text"]), (
                        f"{name}: {question} [{variant}#{i}] label disagrees with is_refusal"
                    )


if __name__ == "__main__":
    raise SystemExit(
        "This module uses pytest.skip. Run:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_eval_regression.py"
    )
