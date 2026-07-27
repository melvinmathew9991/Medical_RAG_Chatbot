"""
Checks on `medbot.eval.refusal_stats`.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_refusal_stats.py

This module computes the headline number of Sprint 4 -- Fisher exact p = 0.0219
for the refusal fix -- with a hand-rolled hypergeometric sum, because scipy is not
a project dependency. Its docstring claimed validation against known values but
nothing enforced it, so a silent edit to the two-sided rule would have quietly
restated every significance claim in results_sprint4.md. These pin it.

The second half pins the missing-data rule. On the partially-complete
out-of-corpus data, `report` used to count a question with zero recorded trials
as "never refused" -- scoring a question the model was never asked as though it
had over-answered, and feeding that phantom cell into the 2x2. Given that Sprint 4
exists largely because two suites were measuring the wrong questions, a stats
tool that invents observations is worth a regression test.
"""

import pytest

from medbot.eval.refusal_stats import analyse, fisher_exact_two_sided, report


# --- the test itself -------------------------------------------------------

def test_tea_tasting_matches_the_textbook_value():
    """Fisher's own lady-tasting-tea table, [[3, 1], [1, 3]] -> 0.4857."""
    assert fisher_exact_two_sided(3, 1, 1, 3) == pytest.approx(0.4857, abs=5e-5)


def test_sprint3_four_question_table_is_not_significant():
    """
    The audit's correction to Sprint 3: 2 of 4 questions fixed is p=0.43, not the
    p=0.00044 the draft got by treating 5 repeats of 4 questions as 20 samples.
    """
    assert fisher_exact_two_sided(2, 2, 0, 4) == pytest.approx(0.4286, abs=5e-5)


def test_sprint4_headline_table_is_significant():
    """The Sprint 4 headline: [[6, 18], [0, 24]] -> p = 0.0219."""
    p = fisher_exact_two_sided(6, 18, 0, 24)
    assert p == pytest.approx(0.0219, abs=5e-5)
    assert p < 0.05


def test_no_difference_gives_p_of_one():
    assert fisher_exact_two_sided(2, 2, 2, 2) == pytest.approx(1.0)


def test_complete_separation_on_a_large_table_is_small():
    p = fisher_exact_two_sided(20, 0, 0, 20)
    assert p < 1e-9


def test_the_test_is_symmetric_under_row_swap():
    """Two-sided p must not depend on which arm is written first."""
    assert fisher_exact_two_sided(6, 18, 0, 24) == pytest.approx(
        fisher_exact_two_sided(0, 24, 6, 18)
    )


def test_p_never_exceeds_one():
    """The 1e-9 float slack in the tail sum could otherwise push p over 1.0."""
    for table in [(1, 0, 0, 1), (0, 1, 1, 0), (5, 5, 5, 5), (1, 1, 1, 1)]:
        assert fisher_exact_two_sided(*table) <= 1.0


# --- missing data must not be imputed as zero ------------------------------

def _cell(n_refused, n_total):
    return [{"refusal": i < n_refused, "text": "x"} for i in range(n_total)]


def test_a_question_missing_an_arm_is_excluded_and_named():
    trials = {
        "measured": {"baseline": _cell(3, 3), "cot": _cell(0, 3)},
        "half measured": {"baseline": _cell(3, 3), "cot": []},
    }
    text = report(analyse(trials))

    assert "INCOMPLETE" in text
    assert "half measured" in text
    assert "[[1, 0], [1, 0]]" not in text or "1/1" in text
    # The denominator is the measured questions, not every question in the file.
    assert "questions ever refusing 1/1" in text


def test_an_unmeasured_arm_is_not_reported_as_never_refusing():
    """
    The specific false claim: baseline refused every trial and cot "refused on
    none", for a question cot was never asked.
    """
    trials = {"never asked in cot": {"baseline": _cell(3, 3), "cot": []}}
    text = report(analyse(trials))
    assert "never asked in cot" in text
    assert "refused on EVERY trial" not in text, (
        "claimed within-question consistency using an arm that has no trials"
    )


def test_fully_measured_data_still_reports_a_p_value():
    trials = {
        f"q{i}": {"baseline": _cell(3 if i < 6 else 0, 3), "cot": _cell(0, 3)}
        for i in range(24)
    }
    text = report(analyse(trials))
    assert "INCOMPLETE" not in text
    assert "[[6, 18], [0, 24]]" in text
    assert "p = 0.0219" in text
    assert "-- significant at 0.05" in text


def test_the_guard_suite_is_reported_in_the_direction_that_can_see_a_leak():
    """
    The two suites point in opposite directions, and reporting the guard in the
    false-refusal unit hides a partial leak. Ten out-of-corpus questions; one arm
    answers four of them at least once, which is invented medical content. In the
    "ever refused" unit that arm still scores 9/10 and p = 1.0000 -- a clean pass.
    Counting questions that ever ANSWERED puts the four failures in the 2x2.
    """
    trials = {
        f"q{i}": {"cot": _cell(3, 3), "leaky": _cell(3 if i >= 4 else 2, 3)}
        for i in range(10)
    }
    variants = ("leaky", "cot")

    default_unit = report(analyse(trials, variants), variants)
    assert "questions ever refusing 10/10" in default_unit
    assert "(unit: ever-refused): [[10, 0], [10, 0]]" in default_unit
    assert "p = 1.0000" in default_unit

    guard_unit = report(analyse(trials, variants), variants, unit="ever-answered")
    assert "ever answering 4/10  <- tested" in guard_unit
    assert "(unit: ever-answered): [[4, 6], [0, 10]]" in guard_unit
    assert "p = 0.0867" in guard_unit
    # Both counts stay visible whichever unit is tested, so the familiar line does
    # not silently change meaning between the two reports.
    assert "questions ever refusing 10/10" in guard_unit


def test_an_unknown_unit_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        report(analyse({"q": {"baseline": _cell(1, 3), "cot": _cell(0, 3)}}),
               unit="ever-hedged")


def test_trial_totals_still_count_every_recorded_trial():
    """
    Excluding a question from the 2x2 must not hide the calls that were spent on
    it -- the trial line is the record of what the quota bought.
    """
    trials = {
        "measured": {"baseline": _cell(1, 3), "cot": _cell(0, 3)},
        "half measured": {"baseline": _cell(3, 3), "cot": []},
    }
    text = report(analyse(trials))
    assert "trials 4/6" in text, text


if __name__ == "__main__":
    raise SystemExit(
        "Run with pytest:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_refusal_stats.py"
    )
