"""
Question-level significance test on the refusal trial data.

    .venv-gemini/Scripts/python.exe -m medbot.eval.refusal_stats --prefix sprint4_

Reads the stored trials, so it costs nothing to re-run.

Why a question-level test rather than counting trials: Sprint 3's draft reported
"10/20 vs 0/20 false refusals" and computed p=0.00044 from it. Those 20 trials
were 5 repeats of each of 4 questions -- repeated measures on 4 units, not 20
independent observations. Treating repeats as independent inflates n fivefold
and shrinks the p-value accordingly. Audit finding F1 corrected the claim to
Fisher exact p=0.43 on the 4 questions, which is not significant.

So this script deliberately collapses each question to a single binary outcome
("did this question ever falsely refuse in this arm?") before testing. The
within-question repetition is reported separately, as a consistency measure --
it is real evidence that the effect is not sampling noise, but it is evidence
about reliability, not about how many questions the fix generalises to.

Fisher's exact test is implemented here rather than imported: scipy is not a
project dependency, and pulling it in for one hypergeometric sum is not worth
the install on this machine.
"""

import argparse
import json
import os
from math import comb

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def fisher_exact_two_sided(a, b, c, d):
    """
    Two-sided Fisher exact p for the 2x2 table [[a, b], [c, d]].

    Sums the hypergeometric probability of every table with the same margins that
    is no more likely than the observed one -- the standard two-sided convention.
    """
    n = a + b + c + d
    row1, col1 = a + b, a + c

    def prob(x):
        return (comb(row1, x) * comb(n - row1, col1 - x)) / comb(n, col1)

    lo = max(0, col1 - (n - row1))
    hi = min(col1, row1)
    observed = prob(a)
    # 1e-9 slack: tables that are equally likely in exact arithmetic can differ in
    # the last bits after floating-point division, and dropping them understates p.
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= observed * (1 + 1e-9)))


def load(prefix, name):
    path = os.path.join(RESULTS_DIR, f"{prefix}{name}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyse(trials, variants=("baseline", "cot")):
    per_question = {}
    for question, by_variant in trials.items():
        row = {}
        for v in variants:
            attempts = by_variant.get(v, [])
            refused = sum(1 for a in attempts if a["refusal"])
            row[v] = (refused, len(attempts))
        per_question[question] = row
    return per_question


# Which outcome is the failure being counted. The two suites point in opposite
# directions and the tool cannot tell them apart from the data:
#
#   ever-refused  -- the false-refusal suite. The corpus answers every question,
#                    so a refusal is the bug.
#   ever-answered -- the out-of-corpus guard. The corpus answers none of them, so
#                    an answer is invented content, and refusing is correct.
#
# Added 2026-07-27, because reporting the guard in the default unit is actively
# misleading. On the instruction-only ablation arm it printed "9/10 questions ever
# refusing, p = 1.0000" -- which reads as a clean pass -- while four of the ten
# questions had produced answers on at least one trial. "Ever refused" is nearly
# always true for an arm that mostly behaves, so it cannot see a partial leak;
# "ever answered" counts the failures directly. (Those six trials turned out to be
# an `is_refusal` gate gap rather than real leaks, but the reporting hole is real
# either way, and the next leak may not be an artefact.)
UNITS = ("ever-refused", "ever-answered")


def _events(row, variant, unit):
    """How many trials of `variant` count as failures for this question."""
    refused, total = row[variant]
    return refused if unit == "ever-refused" else total - refused


def report(per_question, variants=("baseline", "cot"), label="False refusals",
           unit="ever-refused"):
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    verb = "refused" if unit == "ever-refused" else "answered"
    verb_ing = "refusing" if unit == "ever-refused" else "answering"
    lines = [f"\n{label}", "=" * len(label), ""]
    width = max(len(q) for q in per_question)
    lines.append(f"{'question':<{width}}  " + "  ".join(f"{v:>14}" for v in variants))
    for q, row in sorted(per_question.items()):
        cells = "  ".join(f"{row[v][0]:>6}/{row[v][1]:<7}" for v in variants)
        lines.append(f"{q:<{width}}  {cells}")

    # A question is only usable for the comparison if EVERY arm actually has
    # trials for it. An unmeasured cell is not a zero: on the partially-complete
    # Sprint 4 overanswer data, counting missing cot trials as "never refused"
    # scored a question the model was never asked as an over-answer, and fed that
    # phantom into the 2x2. Missing data is reported, never imputed.
    measured = {q: row for q, row in per_question.items()
                if all(row[v][1] > 0 for v in variants)}
    unmeasured = [q for q in per_question if q not in measured]

    lines.append("")
    trial_totals, question_totals = {}, {}
    for v in variants:
        refused = sum(row[v][0] for row in per_question.values())
        total = sum(row[v][1] for row in per_question.values())
        trial_totals[v] = (refused, total)
        ever = sum(1 for row in measured.values() if _events(row, v, unit) > 0)
        question_totals[v] = (ever, len(measured))
        # Both counts are printed whatever the unit: the one being tested, and the
        # other one, so that a reader who came for the familiar "ever refusing"
        # line still finds it and can see which of the two the 2x2 is built on.
        other_unit = "ever-answered" if unit == "ever-refused" else "ever-refused"
        other = sum(1 for row in measured.values() if _events(row, v, other_unit) > 0)
        n_refusing, n_answering = ((ever, other) if unit == "ever-refused"
                                   else (other, ever))
        tested = "  <- tested" if unit == "ever-answered" else ""
        lines.append(f"  {v:<16} trials {refused}/{total}   "
                     f"questions ever refusing {n_refusing}/{len(measured)}"
                     f"   ever answering {n_answering}/{len(measured)}{tested}")

    if unmeasured:
        lines += [
            "",
            f"  INCOMPLETE: {len(unmeasured)} of {len(per_question)} questions lack trials "
            f"in at least one arm and are excluded below:",
        ]
        for q in sorted(unmeasured):
            have = ", ".join(f"{v}={per_question[q][v][1]}" for v in variants)
            lines.append(f"    - {q}  ({have})")

    if len(variants) == 2 and measured:
        v1, v2 = variants
        a, n1 = question_totals[v1]
        c, n2 = question_totals[v2]
        b, d = n1 - a, n2 - c
        p = fisher_exact_two_sided(a, b, c, d)
        lines += [
            "",
            f"  Question-level 2x2 (unit: {unit}): [[{a}, {b}], [{c}, {d}]]",
            f"  Fisher exact (two-sided) p = {p:.4f}"
            f"{'  -- significant at 0.05' if p < 0.05 else '  -- NOT significant at 0.05'}",
        ]

        # Consistency, reported separately and explicitly not as a second p-value.
        # Both arms must have trials: "refused on none" is only a fact about an
        # arm that was actually run.
        consistent = [
            q for q, row in measured.items()
            if _events(row, v1, unit) == row[v1][1] and _events(row, v2, unit) == 0
        ]
        if consistent:
            lines += [
                "",
                f"  Questions where {v1} {verb} on EVERY trial and {v2} on none: {len(consistent)}",
                "  (within-question consistency -- shows the effect is not sampling noise,",
                "   but says nothing about how many questions it generalises to)",
            ]
            for q in sorted(consistent):
                lines.append(f"    - {q}")

        # How much of the significance rests on single observations.
        #
        # A question counts toward `a` in the 2x2 if it refused even once, so a
        # question that refused 1 of 5 trials carries the same weight as one that
        # refused 5 of 5 -- but it is one coin-flip from not counting at all. On
        # the corrected 5-trial data, three of the seven refusers are 1/5; drop
        # them and the table is [[4, 20], [0, 24]] and p = 0.109, not significant.
        # That fragility belongs next to the p-value, not only in the prose of
        # results_sprint4.md.
        # Why the advice below says "more questions" and not "more trials".
        #
        # This file used to close the fragility block with "More trials per
        # question is the fix, not a larger claim." That advice was taken on
        # 2026-07-27: the refusal suite was re-run at 5 trials instead of 3, 240
        # calls against a 500/day quota. It did not work, and could not have:
        #
        #   - the raw p-value moved 0.0219 -> 0.0226;
        #   - the warning still fired, on 3 of 7 refusers instead of 2 of 6;
        #   - and the membership CHURNED. "How should burns be treated?" refused
        #     1/3 at three trials and 0/5 at five. Autism and bedsores, both clean
        #     at three trials, appeared at 1/5.
        #
        # That churn is the point. These are low-rate stochastic refusals, so
        # adding trials does not resolve the existing 1-of-N questions into
        # confident refusers -- it surfaces new ones that were previously missed,
        # and each arrives at exactly one hit. The count of marginal questions is
        # roughly stable in proportion however many trials are run, so this check
        # keeps failing no matter how much quota is spent on it.
        #
        # The binding constraint is the number of QUESTIONS (24), which is what
        # sets how much a single question can swing the 2x2. At the observed rates
        # a 48-question set would put p near 0.0004 -- a projection, not a
        # measurement, but the right order of magnitude and the right axis.
        #
        # The 5-trial run was still worth it, for an unrelated reason: it produced
        # the data that exposed the `is_refusal` contamination bug. See
        # results_sprint4.md section 7.
        marginal = sorted(
            q for q, row in measured.items()
            if _events(row, v1, unit) == 1 < row[v1][1]
        )
        if marginal and p < 0.05:
            a_min = a - len(marginal)
            p_min = fisher_exact_two_sided(a_min, n1 - a_min, c, n2 - c)
            lines += [
                "",
                f"  FRAGILITY: {len(marginal)} of the {a} {verb_ing} questions did so on "
                f"exactly ONE trial:",
            ]
            for q in marginal:
                lines.append(f"    - {q}  ({row_hits(measured[q], v1)})")
            lines += [
                f"  Drop those and the table is [[{a_min}, {n1 - a_min}], [{c}, {n2 - c}]], "
                f"p = {p_min:.4f}"
                f"{' -- still significant' if p_min < 0.05 else ' -- NOT significant'}.",
                "  MORE QUESTIONS is the fix. More trials per question is NOT: this was",
                "  tested on 2026-07-27 and going 3 -> 5 trials left the warning firing,",
                "  because extra trials detect MORE low-rate refusers at exactly one.",
                "  See the note in the source before spending quota on a 5th trial.",
            ]
    return "\n".join(lines)


def row_hits(row, variant):
    refused, total = row[variant]
    return f"{refused}/{total}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="sprint4_")
    parser.add_argument("--variants", default="baseline,cot")
    args = parser.parse_args()
    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())

    out = []
    refusal = load(args.prefix, "refusal_trials.json")
    out.append(report(analyse(refusal, variants), variants,
                      "False refusals on questions the corpus DOES answer (lower is better)"))

    path = os.path.join(RESULTS_DIR, f"{args.prefix}overanswer_trials.json")
    if os.path.exists(path):
        over = load(args.prefix, "overanswer_trials.json")
        # unit="ever-answered": here the answer is the failure, so the 2x2 has to
        # count questions that ever answered. Counting "ever refused" on this
        # suite reads a partial leak as a pass -- see the note above UNITS.
        out.append(report(analyse(over, variants), variants,
                          "Invented answers on out-of-corpus questions (lower is better - "
                          "the corpus has no entry for any of these)",
                          unit="ever-answered"))

    text = "\n".join(out)
    print(text)
    return text


if __name__ == "__main__":
    main()
