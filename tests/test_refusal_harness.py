"""
Checks on the refusal-trial harness itself, with the model mocked out.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_refusal_harness.py

Two things are pinned here.

`is_refusal` is the measurement instrument for every refusal number in
`results_sprint3.md` and `results_sprint4.md`. Audit F9 hand-validated it against
60 stored trial texts; these tests pin the *rules* that validation established,
so that editing the marker list fails here rather than silently re-scoring two
sprints of results.

`run(..., resume=True)` is what lets a quota-exhausted run be finished. Sprint 4's
out-of-corpus measurement died at question 2 of 10 against the 500/day free tier,
and without resume the retry restarts at question 1 -- re-spending quota on cells
already recorded, and never reaching question 3 if the quota runs out in the same
place twice. The suite sat stuck in that loop for a day, so the escape is tested.

Everything here is offline: `conftest.py`'s socket guard fails the test if any of
these mocks miss their target. The patch targets are `medbot.eval.refusal_trials`
and not the defining modules, because `refusal_trials` does `from ... import ...`
and has already bound those names into its own namespace -- the exact mistake that
put the socket guard in conftest in the first place.
"""

import json
from unittest import mock

import pytest

from medbot.eval import refusal_trials as rt


# --- is_refusal: the measurement instrument -------------------------------

def test_refusal_markers_are_detected():
    assert rt.is_refusal("I don't know the answer based on the provided context.")
    assert rt.is_refusal("The context does not contain information about this.")
    assert rt.is_refusal("I cannot answer that from the given text.")


def test_empty_answer_counts_as_a_refusal():
    """A failed call must not be scored as a successful answer."""
    assert rt.is_refusal("")
    assert rt.is_refusal(None)


def test_a_trailing_caveat_is_not_a_refusal():
    """
    The rule audit F9 validated, and the one that keeps the CoT arm honest: the
    CoT prompt deliberately asks for partial answers that name what the context
    leaves out, so a closing caveat is expected and is not a declined question.
    """
    answer = (
        "Bursitis is inflammation of a bursa, the fluid-filled sac that cushions "
        "a joint. The context describes it flaring up after repetitive motion and "
        "settling with rest and ice. " + "Padding to push the caveat well past the "
        "detection window. " * 4 +
        "It does not mention the full list of risk factors."
    )
    assert len(answer) > rt.REFUSAL_WINDOW_CHARS
    assert not rt.is_refusal(answer)


def test_the_partial_answer_from_audit_f9_is_not_a_refusal():
    """
    The closest call in the F9 hand-validation: it opens by declining to *define*
    the term, then supplies causes and symptoms anyway. "not a refusal" is the
    correct label.

    Now genuinely verbatim -- 603 chars, of which 380 survive the absence strip.
    It previously claimed to be verbatim while quoting a shortened paraphrase
    whose surviving content was only 97 chars, under MIN_SUBSTANCE_CHARS. That
    passed for the wrong reason: the narrow marker gate never opened on "does not
    explicitly define", so the substance rule was never consulted. Widening the
    gate on 2026-07-27 made the fixture fail, correctly -- the paraphrase really
    is too thin to count as an answer. The instrument was right and the shortened
    quote was wrong, which is only visible against the stored trial.
    """
    answer = (
        "The provided context does not explicitly define what bursitis is, so I "
        "cannot give a formal definition based on it. However, the text notes that "
        "it can flare up for no known reason or be caused by repeated physical "
        "activity, trauma, rheumatoid arthritis, gout, and acute or chronic "
        "infection. Common symptoms include pain and tenderness, limited and "
        "painful movement, and—if the affected joint is close to the skin like "
        "the shoulder, knee, elbow, or Achilles tendon—swelling, redness, and "
        "warmth. Please consult a qualified healthcare provider for a proper "
        "definition and information regarding bursitis."
    )
    assert len(rt.delivered_content(answer)) > rt.MIN_SUBSTANCE_CHARS
    assert not rt.is_refusal(answer)


def test_cannot_give_and_unable_to_are_deliberately_not_markers():
    """
    Documented in refusal_trials.py: adding these would turn the partial answer
    above into a false positive. Pinned so the reasoning is not re-litigated by
    someone extending the marker list.
    """
    assert "cannot give" not in rt.REFUSAL_MARKERS
    assert "unable to" not in rt.REFUSAL_MARKERS


def test_detection_is_limited_to_the_opening_window():
    """
    The window rule, isolated from the substance rule that now follows it.

    The tail here is kept under MIN_SUBSTANCE_CHARS deliberately. This test used
    to pad with a full window's worth of filler, which the substance rule (added
    2026-07-27) correctly reads as an answer that happens to open with a hedge --
    so the padding, not the window, would decide the outcome and the test would
    no longer be testing what it names.
    """
    lead = "x" * rt.REFUSAL_WINDOW_CHARS
    assert not rt.is_refusal(lead + " I don't know.")

    short_tail = "x" * (rt.MIN_SUBSTANCE_CHARS - 20)
    assert rt.is_refusal("I don't know. " + short_tail)


# --- the substance rule (added 2026-07-27 after the 5-trial re-run) --------

def test_few_shot_contamination_is_not_scored_as_a_refusal():
    """
    The pattern the 5-trial run exposed: the model declines a question from its
    own exemplar, then answers the real one. Verbatim shape of the stored
    baseline trial for "How should burns be treated?", which the window rule
    alone scored as a false refusal.

    It is still a defect -- it is a contamination defect, and counting it as a
    declined question attributes it to the wrong bug and inflates the baseline
    arm the CoT result is measured against.
    """
    answer = (
        "I don't know the answer to your question about bee stings based on the "
        "provided context. However, regarding how burns should be treated, the "
        "information outlines several methods depending on the type of burn. "
        "Minor burns should be cooled with running water and covered with a "
        "sterile dressing, while deep or extensive burns need urgent medical care."
    )
    assert not rt.is_refusal(answer)


def test_a_leading_hedge_before_a_full_answer_is_not_a_refusal():
    """
    Trailing caveats were already handled by the window; this is the same thing
    at the front, and it was the single CoT hit in the 5-trial run.
    """
    answer = (
        'The context does not provide a single, straightforward list labeled '
        '"symptoms of alcoholism," but it describes numerous health problems '
        "and withdrawal effects associated with the condition. Mild withdrawal "
        "symptoms include nausea, achiness, diarrhea, difficulty sleeping, "
        "sweatiness, anxiety, and trembling. More severe effects can include "
        "hallucinations, seizures, confusion, fever and a fast heart rate."
    )
    assert not rt.is_refusal(answer)


def test_a_refusal_that_lists_what_the_corpus_covers_instead_is_still_a_refusal():
    """
    The reason length alone cannot be the test, and the case that makes the
    out-of-corpus guard work. This runs well past any length threshold while
    delivering nothing about the question asked -- naming adjacent entries is
    not answering.
    """
    answer = (
        "I don't know based on the provided context. The context contains "
        "information on treating acne, bedsores, and atopic dermatitis, but it "
        "does not mention psoriasis or how it is treated."
    )
    assert rt.is_refusal(answer)


def test_the_question_term_appearing_in_an_absence_clause_does_not_flip_it():
    """
    Pins the fix that was tried and rejected during calibration. Matching the
    question's own terms looks reasonable until you notice that refusals end
    "...does not mention psoriasis" -- so a term check flips precisely the
    out-of-corpus refusals the hallucination guard depends on.
    """
    assert rt.is_refusal(
        "I do not know based on the provided context. The text discusses what "
        "causes diabetes and how it is diagnosed and tested, but it does not "
        "list the symptoms of diabetes anywhere in the material provided."
    )


def test_absence_phrases_tolerate_inserted_adverbs():
    """
    Both of these were mislabelled by fixed-string matching during calibration:
    the adverb and the object break "does not define" / "does not support".
    """
    assert rt.ABSENCE_RE.search("the text does not explicitly define bursitis")
    assert rt.ABSENCE_RE.search("the context does not support an answer to your question")


def test_delivered_content_drops_only_the_absence_sentences():
    kept = rt.delivered_content(
        "I don't know based on the provided context. "
        "Bursitis is inflammation of a bursa. "
        "The text does not mention risk factors."
    )
    assert kept == "Bursitis is inflammation of a bursa."


def test_the_out_of_corpus_guard_survives_the_substance_rule():
    """
    The direction this instrument must never fail in. A rule that lets invented
    answers pass as refusals is worse than the false-positive bug it fixes, so
    the shipped guard data is re-scored here on every run: all 60 trials of
    Sprint 4's out-of-corpus suite must still be refusals.
    """
    import json
    import os

    path = os.path.join(rt.RESULTS_DIR, "sprint4_overanswer_trials.json")
    data = json.load(open(path, encoding="utf-8"))
    scored = [(q, arm, i, t["text"])
              for q, arms in data.items()
              for arm, trials in arms.items()
              for i, t in enumerate(trials, 1)]
    assert len(scored) == 60, "the guard dataset changed; re-check this pin"

    leaked = [(q, arm, i) for q, arm, i, text in scored if not rt.is_refusal(text)]
    assert not leaked, f"substance rule let out-of-corpus answers through: {leaked}"


# --- the candidate gate (widened 2026-07-27, third-arm phrasing gap) ------

def test_the_third_arm_refusal_phrasing_is_detected():
    """
    Verbatim from the instruction-only ablation run. The marker list was
    calibrated on baseline and cot trials, which say "I don't know" or "does not
    mention"; this arm says "there is no mention", which matched nothing, so the
    gate stayed shut and six flat refusals were scored as invented ANSWERS on
    out-of-corpus questions -- a false alarm in the worst possible direction.
    """
    assert rt.is_refusal(
        "Based on the provided context, there is no mention of the symptoms of "
        "shingles. Therefore, the context does not support an answer to this question."
    )
    assert rt.is_refusal(
        "Based on the provided context, there is no mention of mumps or its "
        "symptoms. Therefore, the context does not support an answer to this question."
    )
    assert rt.is_refusal(
        "Based on the provided context, there is no mention of the symptoms of "
        "schizophrenia. (The context does mention psychotic symptoms and "
        "hallucinations in the context of extreme mania related to bipolar "
        "disorder, but it does not address schizophrenia.)"
    )


def test_the_gate_is_the_union_of_the_marker_list_and_the_absence_regex():
    """
    Why the fix is a union and not a replacement: neither reader contains the
    other. Pinned so that "ABSENCE_RE covers everything, drop the list" is not
    tried -- it would reopen the marker-only hole.
    """
    marker_only = "The context cannot answer that."
    regex_only = "There is no mention of measles here."

    assert any(m in marker_only.lower() for m in rt.REFUSAL_MARKERS)
    assert not rt.ABSENCE_RE.search(marker_only)

    assert not any(m in regex_only.lower() for m in rt.REFUSAL_MARKERS)
    assert rt.ABSENCE_RE.search(regex_only)

    assert rt.opens_with_absence_language(marker_only)
    assert rt.opens_with_absence_language(regex_only)


def test_the_wider_gate_still_needs_the_substance_test():
    """
    A wider gate can only add *candidates*. This answer opens with the phrasing
    the gate now catches and then answers anyway, so it must stay an answer --
    otherwise widening the gate would trade one false-positive bug for another.
    """
    answer = (
        "There is no mention of a single labelled list of bursitis risk factors. "
        "Bursitis is inflammation of a bursa, the fluid-filled sac cushioning a "
        "joint, and the context attributes it to repeated physical activity, "
        "trauma, rheumatoid arthritis, gout, and acute or chronic infection."
    )
    assert rt.opens_with_absence_language(answer)
    assert not rt.is_refusal(answer)


def test_the_instruction_only_guard_data_is_fully_armed():
    """
    The ablation arm gets the same treatment as the shipped one: if the exemplars
    are ever dropped on the strength of this ablation, instruction-only becomes
    the shipped prompt, so its out-of-corpus behaviour is load-bearing evidence
    and is re-scored on every test run.
    """
    import json
    import os

    path = os.path.join(rt.RESULTS_DIR, "ablation_t5_overanswer_trials.json")
    data = json.load(open(path, encoding="utf-8"))
    trials = [(q, i, t["text"])
              for q, arms in data.items()
              for arm, ts in arms.items() if arm == "instruction-only"
              for i, t in enumerate(ts, 1)]
    assert len(trials) == 30, "the instruction-only guard dataset changed; re-check this pin"

    leaked = [(q, i) for q, i, text in trials if not rt.is_refusal(text)]
    assert not leaked, f"instruction-only invented answers out of corpus: {leaked}"


# --- resume: finishing a quota-killed run ---------------------------------

@pytest.fixture
def harness(tmp_path, monkeypatch):
    """
    `run` with the vector store, the model and the chain replaced by fakes.

    Returns a counter of how many model calls were made, which is the whole point
    of resume -- the assertion is about quota spent, not just about output shape.
    """
    calls = []

    def fake_run_query(chain, question, variant=None):
        calls.append((question, variant))
        return {"result": f"An answer to {question} from {variant}."}

    monkeypatch.setattr(rt, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(rt, "create_vector_database", lambda: object())
    monkeypatch.setattr(rt, "initialize_model", lambda: object())
    monkeypatch.setattr(rt, "create_query_chain", lambda m, v, q, variant=None: object())
    # call_with_backoff is imported into refusal_trials' namespace from run_eval;
    # patching run_eval would miss it. It also sleeps 5s per call, which a unit
    # test must not do.
    monkeypatch.setattr(rt, "call_with_backoff", lambda fn, *a, **kw: fn(*a, **kw))
    monkeypatch.setattr(rt, "run_query", fake_run_query)
    # Stubbed for the same reason as the rest: it makes a real one-token call to
    # check the daily quota before the run commits. `preflight` is exercised
    # directly below instead.
    monkeypatch.setattr(rt, "preflight", lambda model: None)
    return tmp_path, calls


def _write_trials(tmp_path, name, payload):
    (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")


def test_resume_skips_cells_that_already_have_enough_trials(harness):
    tmp_path, calls = harness
    _write_trials(tmp_path, "t.json", {
        "Q1": {"baseline": [{"refusal": True, "text": "I don't know."}] * 3,
               "cot": [{"refusal": True, "text": "I don't know."}] * 3},
    })

    out = rt.run(3, ["baseline", "cot"], ["Q1", "Q2"], "t.json", resume=True)

    assert [q for q, _ in calls] == ["Q2"] * 6, "Q1 was re-measured despite being complete"
    assert len(out["Q1"]["baseline"]) == 3
    assert len(out["Q2"]["cot"]) == 3


def test_without_resume_everything_is_remeasured(harness):
    """The default must stay a clean-slate run; resume is opt-in."""
    tmp_path, calls = harness
    _write_trials(tmp_path, "t.json", {
        "Q1": {"baseline": [{"refusal": True, "text": "I don't know."}] * 3},
    })

    rt.run(3, ["baseline"], ["Q1", "Q2"], "t.json", resume=False)

    assert sorted(q for q, _ in calls) == ["Q1", "Q1", "Q1", "Q2", "Q2", "Q2"]


def test_resume_does_not_discard_the_cells_it_skips(harness):
    """
    The failure this guards against is the one that made resume necessary: the
    checkpoint overwrites the whole file, so a resumed run that dropped skipped
    cells from `out` would erase the data it just avoided re-measuring.
    """
    tmp_path, _ = harness
    _write_trials(tmp_path, "t.json", {
        "Q1": {"baseline": [{"refusal": True, "text": "I don't know."}] * 3,
               "cot": [{"refusal": True, "text": "I don't know."}] * 3},
    })

    rt.run(3, ["baseline", "cot"], ["Q1", "Q2"], "t.json", resume=True)

    on_disk = json.loads((tmp_path / "t.json").read_text(encoding="utf-8"))
    assert set(on_disk) == {"Q1", "Q2"}
    assert len(on_disk["Q1"]["baseline"]) == 3, "skipped cell was lost from the checkpoint"


def test_a_partially_filled_cell_is_rerun_not_topped_up(harness):
    """
    A cell is the unit the rate is computed over. Topping one up would blend
    trials from two runs inside a single measurement; re-running the cell costs
    at most `trials` calls and keeps each cell internally consistent.
    """
    tmp_path, calls = harness
    _write_trials(tmp_path, "t.json", {
        "Q1": {"baseline": [{"refusal": True, "text": "I don't know."}]},
    })

    out = rt.run(3, ["baseline"], ["Q1"], "t.json", resume=True)

    assert len(calls) == 3, "partial cell should be re-run in full"
    assert len(out["Q1"]["baseline"]) == 3
    assert all("An answer to" in a["text"] for a in out["Q1"]["baseline"]), \
        "the stale partial trial survived into the cell"


def test_resume_with_no_existing_file_is_a_normal_run(harness):
    tmp_path, calls = harness
    out = rt.run(2, ["cot"], ["Q1"], "missing.json", resume=True)
    assert len(calls) == 2
    assert len(out["Q1"]["cot"]) == 2


# --- preflight: fail fast on a dead daily quota ---------------------------

def _resource_exhausted(message):
    from google.api_core.exceptions import ResourceExhausted
    return ResourceExhausted(message)


def test_preflight_passes_when_the_model_answers():
    from unittest.mock import Mock
    from medbot.eval.run_eval import preflight

    assert preflight(Mock()) is None


def test_preflight_reports_the_daily_cap_and_says_when_it_resets():
    """
    The message has to name the Pacific rollover. Assuming local midnight is why
    the Sprint 4 out-of-corpus run was retried at 01:00 IST against a quota that
    had not reset, and why it stalled instead of failing.
    """
    from unittest.mock import Mock
    from medbot.eval.run_eval import preflight

    model = Mock()
    # Copied from the real 429 seen on 2026-07-26.
    model.invoke.side_effect = _resource_exhausted(
        "429 Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 500 "
        'violations { quota_id: "GenerateRequestsPerDayPerProjectPerModel-FreeTier" }'
    )
    blocked = preflight(model)
    assert blocked
    assert "Pacific" in blocked
    assert "500" in blocked


def test_preflight_lets_a_per_minute_limit_through_to_the_backoff():
    """
    The two 429s need different handling: RPM recovers in 60s and the existing
    backoff deals with it, so treating it as fatal would abort runs that would
    have succeeded. Only the daily cap cannot be waited out.
    """
    from unittest.mock import Mock
    from medbot.eval.run_eval import preflight

    model = Mock()
    # Note the metric name is the SAME as the daily one -- only the quota id
    # differs. A substring match on the metric would abort here, killing a run
    # that a 65s backoff would have carried through.
    model.invoke.side_effect = _resource_exhausted(
        "429 Quota exceeded for metric: "
        "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 15 "
        'violations { quota_id: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier" }'
    )
    assert preflight(model) is None


def test_run_aborts_before_spending_calls_when_preflight_fails(harness, monkeypatch):
    """The whole point: nothing is run, so no quota is spent on a doomed run."""
    tmp_path, calls = harness
    monkeypatch.setattr(rt, "preflight", lambda model: "Daily free-tier quota is exhausted.")

    with pytest.raises(SystemExit, match="Preflight failed"):
        rt.run(3, ["cot"], ["Q1"], "t.json")

    assert not calls, "spent model calls despite a failed preflight"


def test_labels_are_recomputed_from_the_text(harness):
    """`refusal` must always be `is_refusal(text)`; test_eval_regression relies on it."""
    tmp_path, _ = harness
    out = rt.run(1, ["cot"], ["Q1"], "t.json", resume=False)
    for attempts in out["Q1"].values():
        for a in attempts:
            assert a["refusal"] == rt.is_refusal(a["text"])


if __name__ == "__main__":
    raise SystemExit(
        "This module uses pytest fixtures. Run:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_refusal_harness.py"
    )
