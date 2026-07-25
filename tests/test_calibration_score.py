"""
Checks on `medbot.eval.calibration_score`.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_calibration_score.py

All synthetic. The real `calibration_sheet.md` is deliberately left unlabelled:
the exercise measures whether the judge favours its own arm, and the judge is
this model, so labels produced here would not be independent evidence. What can
be tested without labelling is that the arithmetic and the parsing are right --
so that when a human does fill the sheet in, the number that comes out is
trustworthy.

The headline the scorer computes is DIFFERENTIAL bias, cot minus baseline. A
judge that is wrong by the same amount in both arms leaves the Sprint 3 claim
(0.841 -> 0.997) standing, because that claim is a difference and a constant
bias cancels. Only a bias that differs between arms can eat the delta.
"""

import json

import pytest

from medbot.eval.calibration_score import parse_sheet, report, score


def _sheet(rows):
    """Build a sheet in the format calibration_sample.py emits."""
    out = ["# Groundedness judge — blinded calibration sheet", "", "---", ""]
    for i, (supported, claims) in enumerate(rows, 1):
        out += [
            f"## {i}. A question?",
            "",
            "**CONTEXT**",
            "",
            "> **[0]** some retrieved text",
            "",
            "**ANSWER**",
            "",
            "```",
            "an answer",
            "```",
            "",
            f"`LABEL: supported={supported} / claims={claims}   comment: `",
            "",
            "---",
            "",
        ]
    return "\n".join(out)


def _key(entries):
    return [
        {"item": i, "question": "A question?", "variant": variant,
         "judge_score": judge, "judge_claims": 4, "judge_supported": 4,
         "judge_rationale": ""}
        for i, (variant, judge) in enumerate(entries, 1)
    ]


def _run(tmp_path, rows, entries):
    path = tmp_path / "sheet.md"
    path.write_text(_sheet(rows), encoding="utf-8")
    return score(parse_sheet(str(path)), _key(entries))


# --- parsing ---------------------------------------------------------------

def test_unfilled_labels_are_excluded_not_read_as_zero(tmp_path):
    """The shipped sheet is all `__`; scoring those as 0/0 would invent labels."""
    scored, refusals, unlabelled, malformed = _run(
        tmp_path, [("__", "__"), ("__", "__")], [("cot", 1.0), ("baseline", 1.0)]
    )
    assert unlabelled == [1, 2]
    assert not scored and not refusals and not malformed


def test_a_partially_filled_sheet_scores_only_what_is_filled(tmp_path):
    scored, _, unlabelled, _ = _run(
        tmp_path, [(2, 4), ("__", "__")], [("cot", 1.0), ("baseline", 1.0)]
    )
    assert [r["item"] for r in scored] == [1]
    assert unlabelled == [2]


def test_supported_greater_than_claims_is_rejected(tmp_path):
    """A slip of the pen, not a 125%-grounded answer."""
    scored, _, _, malformed = _run(tmp_path, [(5, 4)], [("cot", 1.0)])
    assert not scored
    assert malformed and "item 1" in malformed[0]


def test_non_numeric_label_fails_loudly(tmp_path):
    path = tmp_path / "sheet.md"
    path.write_text(_sheet([("two", 4)]), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a whole number"):
        parse_sheet(str(path))


# --- the refusal carve-out -------------------------------------------------

def test_refusals_are_held_out_rather_than_scored_as_zero(tmp_path):
    """
    0/0 is undefined, not 0.0. Averaging refusals in as zeros would drag the
    baseline arm down and manufacture exactly the differential bias this tool
    exists to detect -- and refusals are a different bug, measured elsewhere.
    """
    scored, refusals, _, _ = _run(
        tmp_path, [(0, 0), (4, 4)], [("baseline", 0.0), ("cot", 1.0)]
    )
    assert [r["item"] for r in refusals] == [1]
    assert [r["item"] for r in scored] == [2]
    assert all("human" not in r for r in refusals)


# --- the arithmetic that matters -------------------------------------------

def test_human_score_is_the_ratio_and_delta_is_judge_minus_human(tmp_path):
    scored, _, _, _ = _run(tmp_path, [(3, 4)], [("cot", 1.0)])
    assert scored[0]["human"] == pytest.approx(0.75)
    assert scored[0]["delta"] == pytest.approx(0.25)


def test_a_bias_equal_in_both_arms_cancels(tmp_path):
    """
    The central point. The judge is +0.25 optimistic everywhere, which is a real
    inaccuracy -- but it does not touch a claim expressed as a difference.
    """
    scored, refusals, unlabelled, malformed = _run(
        tmp_path,
        [(3, 4), (3, 4)],
        [("baseline", 1.0), ("cot", 1.0)],
    )
    text = report(scored, refusals, unlabelled, malformed)
    assert "DIFFERENTIAL BIAS (cot - baseline): +0.000" in text
    assert "survives the check" in text


def test_a_bias_only_in_the_cot_arm_is_flagged_as_large(tmp_path):
    """The failure mode: the judge rewards the hedging the CoT prompt produces."""
    scored, refusals, unlabelled, malformed = _run(
        tmp_path,
        [(4, 4), (2, 4)],          # baseline judged accurately, cot inflated
        [("baseline", 1.0), ("cot", 1.0)],
    )
    text = report(scored, refusals, unlabelled, malformed)
    assert "DIFFERENTIAL BIAS (cot - baseline): +0.500" in text
    assert "not safe as stated" in text


def test_a_small_differential_is_reported_as_survivable(tmp_path):
    scored, refusals, unlabelled, malformed = _run(
        tmp_path,
        [(4, 4), (39, 40)],        # cot inflated by only 0.025
        [("baseline", 1.0), ("cot", 1.0)],
    )
    text = report(scored, refusals, unlabelled, malformed)
    assert "+0.025" in text
    assert "survives the check" in text


def test_large_disagreements_are_listed_for_rereading(tmp_path):
    scored, refusals, unlabelled, malformed = _run(
        tmp_path, [(1, 4), (4, 4)], [("cot", 1.0), ("baseline", 1.0)]
    )
    text = report(scored, refusals, unlabelled, malformed)
    assert "|delta| >= 0.25" in text
    assert "item   1" in text


def test_report_is_honest_when_there_is_nothing_to_score(tmp_path):
    scored, refusals, unlabelled, malformed = _run(
        tmp_path, [("__", "__")], [("cot", 1.0)]
    )
    text = report(scored, refusals, unlabelled, malformed)
    assert "Nothing scoreable yet" in text
    assert "DIFFERENTIAL BIAS" not in text, "claimed a result from zero labels"


def test_the_real_sheet_and_key_line_up(tmp_path):
    """
    Guards the join itself: the shipped sheet must have one labelled slot per key
    entry, or a filled-in sheet would silently score against the wrong answers.
    """
    import os
    from medbot.eval.calibration_score import RESULTS_DIR

    sheet = os.path.join(RESULTS_DIR, "calibration_sheet.md")
    key_path = os.path.join(RESULTS_DIR, "calibration_key.json")
    if not (os.path.exists(sheet) and os.path.exists(key_path)):
        pytest.skip("calibration artefacts not generated")

    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)
    assert set(parse_sheet(sheet)) == {k["item"] for k in key}


if __name__ == "__main__":
    raise SystemExit(
        "Run with pytest:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_calibration_score.py"
    )
