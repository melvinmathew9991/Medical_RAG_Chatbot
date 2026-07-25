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
    The closest call in the F9 hand-validation, kept verbatim: it opens by
    declining to *define* the term, then supplies causes and symptoms anyway.
    "not a refusal" is the correct label.
    """
    answer = (
        "The provided context does not explicitly define what bursitis is, so I "
        "cannot give a formal definition based on it. However, the text notes that "
        "it can flare up after repetitive motion, and that rest and ice help."
    )
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
    lead = "x" * rt.REFUSAL_WINDOW_CHARS
    assert not rt.is_refusal(lead + " I don't know.")
    assert rt.is_refusal("I don't know. " + lead)


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
