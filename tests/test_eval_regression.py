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

from medbot.eval.dataset import EVAL_QUESTIONS, EVAL_QUESTIONS_V1
from medbot.eval.refusal_trials import (
    OVERANSWER_QUESTIONS,
    REFUSAL_QUESTIONS,
    is_refusal,
)

# The 24 questions every pre-2026-08-03 trial file covers. Those files are
# COMPLETE records of the suite as it stood, so their coverage gates are pinned
# here rather than to REFUSAL_QUESTIONS — which grew to 46 on 2026-08-03 and would
# otherwise mark a finished historical dataset as truncated, i.e. fail loudly for
# a reason that is not a defect.
FROZEN_V1_QUESTIONS = [c["question"] for c in EVAL_QUESTIONS_V1]

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

# The four-arm ablation, which carries the cost decision rather than the headline:
# baseline / instruction-only / cot / no-examples on the same 24 questions at 5
# trials, plus the 10-question out-of-corpus guard at 3.
#
# `no-examples` was added to both files on 2026-08-03. Until then these two files
# had NO gate of any kind -- the 2026-07-27 audit found every gate pointing at the
# retired 3-trial dataset and fixed the headline ones, but the ablation files were
# left uncovered, and the exemplar decision rests on them.
ABLATION_REFUSAL = "ablation_t5_refusal_trials.json"
ABLATION_OVERANSWER = "ablation_t5_overanswer_trials.json"
ABLATION_ARMS = ("baseline", "instruction-only", "cot", "no-examples")
ABLATION_GUARD_TRIALS = 3

# The expanded suite: 46 questions x 5 trials, recorded 2026-08-03 (330 calls) by
# resuming the 24 already-recorded questions and measuring only the 22 new ones.
#
# Three arms, not four. `instruction-only` is deliberately not extended: §8 settled
# what it was there to answer, and it is the arm that substitutes a neighbouring
# exemplar's question, so 110 calls to carry it forward would buy nothing. It stays
# at 24 in the ablation file, which is why THAT file keeps its own four-arm gate
# against the frozen list.
EXPANDED_REFUSAL = "expanded_t5_refusal_trials.json"
EXPANDED_ARMS = ("baseline", "cot", "no-examples")


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


# --- the standing headline, at n=46 ----------------------------------------

def test_neither_shipped_candidate_falsely_refuses_at_n46():
    """
    The 2026-08-03 expansion headline: baseline 18/46 questions refusing, cot and
    no-examples 0/46 each, Fisher p = 0.0000.

    Both arms are gated, not just the shipped one, because `cot` is the documented
    fallback if this decision is revisited — a fallback nobody is measuring is not
    a fallback. Every question here has a dedicated encyclopedia entry (screened by
    verify_entry for the 22 new ones), so a refusal is always a bug.
    """
    trials = _load(EXPANDED_REFUSAL)
    for arm in ("cot", "no-examples"):
        offenders = {q: c for q, (c, _) in _refusal_counts(trials, arm).items() if c}
        assert not offenders, f"{arm} refused questions the corpus answers: {offenders}"


def test_the_expansion_did_not_weaken_the_baseline_gap():
    """
    Direction and magnitude, not just the zero above. The expansion exists because
    at n=24 the result was fragile: dropping the three questions that refused on
    exactly one trial took p from 0.0094 to 0.1092, i.e. not significant. At n=46
    the same subtraction leaves p = 0.0002.

    So this pins the property that made the expansion worth 330 calls — that the
    effect survives dropping every one-trial refuser — rather than pinning a
    p-value, which would fail on a harmless re-measure.
    """
    trials = _load(EXPANDED_REFUSAL)
    counts = _refusal_counts(trials, "baseline")
    refusing = {q: c for q, (c, _) in counts.items() if c}
    assert len(refusing) >= 12, (
        f"baseline now refuses only {len(refusing)}/46 questions, was 18. The "
        "comparison arm got better, so the recorded gap no longer describes this data"
    )
    robust = {q: c for q, c in refusing.items() if c >= 2}
    assert len(robust) >= 12, (
        f"only {len(robust)} baseline refusals survive dropping one-trial cases; "
        "the n=46 result has decayed to the fragile n=24 shape"
    )


# --- the same result on the frozen 24, kept as recorded --------------------

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
    assert set(trials) == set(FROZEN_V1_QUESTIONS), "refusal trials do not match the suite"
    for question, by_variant in trials.items():
        for variant in ("baseline", "cot"):
            attempts = by_variant.get(variant)
            assert attempts, f"{question} has no {variant} trials"
            assert len(attempts) == TRIALS_PER_CELL, (
                f"{question} [{variant}] has {len(attempts)} trials, not {TRIALS_PER_CELL}"
            )


# --- the no-examples arm, and the ablation files generally ------------------

def test_no_examples_never_falsely_refuses():
    """
    The result that makes the cheap prompt viable: 0/120 trials on the 24
    questions the corpus answers, matching `cot` exactly at ~1/14th the tokens.

    Gated on the ablation file rather than the headline one because that is where
    the arm was measured. Same standard as `test_cot_never_falsely_refuses`: every
    question here has a dedicated encyclopedia entry, so there is no correct
    refusal in this file.
    """
    trials = _load(ABLATION_REFUSAL)
    counts = _refusal_counts(trials, "no-examples")
    offenders = {q: c for q, (c, _) in counts.items() if c}
    assert not offenders, (
        f"no-examples refused questions the corpus answers: {offenders}. "
        "The shipped prompt decision rests on this being zero."
    )


def test_no_examples_refuses_out_of_corpus_questions():
    """
    The other half, and the one that could sink the arm: a prompt that never
    refuses scores a perfect zero above by inventing medical answers.

    `no-examples` phrases these refusals as "there is no information/mention of
    ...", which is the exact wording that defeated `is_refusal`'s marker list on
    the `instruction-only` arm in 2026-07-27's ablation (results_sprint4.md 8).
    The union gate handles it, and this test is what keeps that true.
    """
    trials = _load(ABLATION_OVERANSWER)
    counts = _refusal_counts(trials, "no-examples")
    answered = {q: (n - c) for q, (c, n) in counts.items() if c < n and n > 0}
    assert not answered, (
        f"no-examples invented answers on out-of-corpus questions: {answered}. "
        "Read the stored text by hand before adjusting anything -- either the "
        "model invented content, or the question is not out-of-corpus (audit F8)."
    )


def test_ablation_arms_are_completely_recorded():
    """
    A truncated arm would make both gates above vacuous, and truncation is a live
    risk here rather than a hypothetical: `refusal_trials.run()` rebuilds its
    output dict from scratch and checkpoints after every variant, so a run that
    dies partway writes back a file containing only the questions it reached --
    silently dropping the other arms' recorded cells for every question it did
    not. That is how 72 already-paid-for cells could vanish on a quota failure.
    """
    trials = _load(ABLATION_REFUSAL)
    assert set(trials) == set(FROZEN_V1_QUESTIONS), (
        "ablation refusal trials no longer match the frozen 24-question set"
    )
    for question, by_variant in trials.items():
        for arm in ABLATION_ARMS:
            attempts = by_variant.get(arm)
            assert attempts, f"{question} has no {arm} trials"
            assert len(attempts) == TRIALS_PER_CELL, (
                f"{question} [{arm}] has {len(attempts)} trials, not {TRIALS_PER_CELL}"
            )

    guard = _load(ABLATION_OVERANSWER)
    assert set(guard) == set(OVERANSWER_QUESTIONS), (
        "ablation guard no longer matches the out-of-corpus suite"
    )
    for question, by_variant in guard.items():
        for arm in ABLATION_ARMS:
            attempts = by_variant.get(arm)
            assert attempts, f"{question} has no {arm} guard trials"
            assert len(attempts) == ABLATION_GUARD_TRIALS, (
                f"{question} [{arm}] has {len(attempts)} guard trials, "
                f"not {ABLATION_GUARD_TRIALS}"
            )


def test_expanded_suite_is_completely_recorded():
    """
    The expanded file is the one the live headline is computed from, so a partial
    run must fail rather than quietly report a statistic over whichever questions
    happened to finish. That is the F3/§9 failure shape: a gate that goes green on
    incomplete data is not a gate.

    Checks the question SET, not just the count — 46 rows containing a duplicate
    and a missing question would otherwise pass.
    """
    trials = _load(EXPANDED_REFUSAL)
    assert set(trials) == set(REFUSAL_QUESTIONS), (
        f"expanded trials cover {len(trials)} questions, suite has "
        f"{len(REFUSAL_QUESTIONS)}. Missing: "
        f"{sorted(set(REFUSAL_QUESTIONS) - set(trials))}"
    )
    for question, by_variant in trials.items():
        for arm in EXPANDED_ARMS:
            attempts = by_variant.get(arm)
            assert attempts, f"{question} has no {arm} trials"
            assert len(attempts) == TRIALS_PER_CELL, (
                f"{question} [{arm}] has {len(attempts)} trials, not {TRIALS_PER_CELL}"
            )


def test_the_shipped_default_has_been_measured_and_does_not_falsely_refuse():
    """
    "Unmeasured prompts do not ship" was enforced by asserting the default equalled
    the string "cot". That detects a change but not the mistake: it would have gone
    green on any rename, and red on a perfectly well-measured replacement.

    So the gate is now the property. Whatever `DEFAULT_PROMPT_VARIANT` names must:

      1. have recorded claim-level results covering the whole eval set, and
      2. never falsely refuse a question the corpus answers.

    (2) is what stops a *measured but bad* prompt shipping — `baseline` has full
    recorded results and refuses 7/24, so requirement (1) alone would wave it
    through. Both current qualifiers are `cot` and `no-examples`.
    """
    from medbot.prompt import DEFAULT_PROMPT_VARIANT as shipped

    claims_path = os.path.join(EVAL_DIR, f"results_{shipped}_claims.json")
    assert os.path.exists(claims_path), (
        f"the app ships {shipped!r} but there are no recorded claim-level results "
        f"for it ({os.path.basename(claims_path)}). Measure it before shipping it: "
        f"python -m medbot.eval.run_eval --variant {shipped} && "
        f"python -m medbot.eval.rejudge --variants {shipped}"
    )
    with open(claims_path, encoding="utf-8") as f:
        scored = [r for r in json.load(f)["results"] if r.get("claim_score") is not None]
    # Against the FROZEN 24, not the expanded 46, and this is a deliberate
    # weakening recorded rather than hidden: the 2026-08-03 expansion re-measured
    # refusals over all 46 but not groundedness, because run_eval has no
    # question-subset filter and no resume, so re-scoring would have cost 138 calls
    # to re-derive 24 values already recorded. Claim-level coverage of the 22 new
    # questions is OPEN WORK -- see results_sprint4.md §12. Tighten this to
    # EVAL_QUESTIONS once that run happens.
    assert len(scored) >= len(EVAL_QUESTIONS_V1), (
        f"{shipped} has {len(scored)} scored answers, fewer than the "
        f"{len(EVAL_QUESTIONS_V1)} recorded — a partial measurement is not a measurement"
    )

    # Refusals ARE gated over the full 46: it is the safety-relevant half, and the
    # expansion measured it.
    trials = _load(EXPANDED_REFUSAL)
    counts = _refusal_counts(trials, shipped)
    assert any(n for _, n in counts.values()), (
        f"no recorded refusal trials for the shipped variant {shipped!r} in "
        f"{ABLATION_REFUSAL}; it cannot be shown not to refuse"
    )
    offenders = {q: c for q, (c, _) in counts.items() if c}
    assert not offenders, (
        f"the app ships {shipped!r}, which falsely refuses {offenders}"
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
