"""
Score a filled-in calibration sheet against the judge (audit F6, analysis half).

    .venv-gemini/Scripts/python.exe -m medbot.eval.calibration_score

Reads `calibration_sheet.md` (your labels) and `calibration_key.json` (the hidden
arm and judge score), and answers the question the exercise exists for.

`calibration_sample.py` builds the sheet; nothing consumed it until this existed,
so a finished labelling session produced 25 hand-written ratios and no result.

WHAT THIS TESTS, and what it does not
-------------------------------------
The worry is not "is the judge accurate" -- an inaccurate judge that is *equally*
inaccurate in both arms leaves the Sprint 3 comparison intact, because the claim
is a delta (0.841 -> 0.997) and a constant bias cancels. The worry is that the
judge inflates the *cot* arm specifically: CoT answers hedge, name their gaps and
bullet their structure, and a self-grading judge may read that as support.

So the headline here is DIFFERENTIAL bias -- (judge - human) in cot minus
(judge - human) in baseline. That is what would eat the delta. Per-arm bias is
reported too, but on its own it is much less alarming than it looks.

This cannot be run by the model that produced the answers, which is the same
model being audited. Labels have to be yours.
"""

import argparse
import json
import os
import re
import statistics

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

ITEM_RE = re.compile(r"^## (\d+)\.\s*(.+?)\s*$", re.M)
# Tolerant of the blanks in the unfilled sheet, of `=` spacing, and of the
# surrounding backticks the sheet wraps the line in.
LABEL_RE = re.compile(
    r"LABEL:\s*supported\s*=\s*(\S+?)\s*/\s*claims\s*=\s*(\S+?)\s*(?:comment:\s*(.*?))?\s*`?\s*$",
    re.M,
)
UNFILLED = {"__", "_", "", "?", "--"}


def parse_sheet(path):
    """-> {item_number: {"supported": int|None, "claims": int|None, "comment": str}}"""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    items = list(ITEM_RE.finditer(text))
    if not items:
        raise SystemExit(f"No '## N. question' headers found in {path}")

    out = {}
    for i, match in enumerate(items):
        number = int(match.group(1))
        block = text[match.end(): items[i + 1].start() if i + 1 < len(items) else len(text)]
        label = LABEL_RE.search(block)
        if not label:
            out[number] = {"supported": None, "claims": None, "comment": ""}
            continue
        supported, claims, comment = label.group(1), label.group(2), (label.group(3) or "")
        out[number] = {
            "supported": None if supported in UNFILLED else _int(supported, number, "supported"),
            "claims": None if claims in UNFILLED else _int(claims, number, "claims"),
            "comment": comment.strip().rstrip("`").strip(),
        }
    return out


def _int(raw, number, field):
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"Item {number}: {field}={raw!r} is not a whole number.")


def score(labels, key):
    """Join labels to the key, splitting refusals (0 claims) from scoreable answers."""
    by_item = {k["item"]: k for k in key}
    scored, refusals, unlabelled, malformed = [], [], [], []

    for number, label in sorted(labels.items()):
        entry = by_item.get(number)
        if entry is None:
            malformed.append(f"item {number} is not in the key")
            continue
        if label["supported"] is None or label["claims"] is None:
            unlabelled.append(number)
            continue
        if label["supported"] > label["claims"]:
            malformed.append(
                f"item {number}: supported={label['supported']} > claims={label['claims']}"
            )
            continue

        row = {
            "item": number,
            "variant": entry["variant"],
            "judge": entry["judge_score"],
            "judge_claims": entry["judge_claims"],
            "human_claims": label["claims"],
            "comment": label["comment"],
        }
        # A refusal makes no claims, so supported/claims is 0/0 -- undefined, not
        # zero. The sheet asks for 0/0 on those deliberately: they are the
        # false-refusal bug, a different failure from poor groundedness, and
        # averaging them in as 0.0 would conflate the two.
        if label["claims"] == 0:
            refusals.append(row)
        else:
            row["human"] = label["supported"] / label["claims"]
            row["delta"] = row["judge"] - row["human"]
            scored.append(row)

    return scored, refusals, unlabelled, malformed


def report(scored, refusals, unlabelled, malformed):
    lines = ["", "Judge calibration vs human labels", "=" * 33, ""]

    if malformed:
        lines += ["  MALFORMED, excluded:"] + [f"    - {m}" for m in malformed] + [""]
    if unlabelled:
        lines += [f"  Unlabelled, excluded: {len(unlabelled)} items "
                  f"({', '.join(str(n) for n in unlabelled)})", ""]

    if not scored:
        lines += ["  Nothing scoreable yet. Fill in the LABEL lines in "
                  "calibration_sheet.md and re-run.", ""]
        return "\n".join(lines)

    lines += [f"  {len(scored)} scoreable answers "
              f"({len(refusals)} refusals held out, see below)", ""]
    lines.append(f"  {'item':>4}  {'arm':<9} {'judge':>6} {'human':>6} {'delta':>7}")
    for row in sorted(scored, key=lambda r: -abs(r["delta"])):
        lines.append(f"  {row['item']:>4}  {row['variant']:<9} {row['judge']:>6.2f} "
                     f"{row['human']:>6.2f} {row['delta']:>+7.2f}")

    lines.append("")
    per_arm = {}
    for variant in ("baseline", "cot"):
        rows = [r for r in scored if r["variant"] == variant]
        if not rows:
            continue
        deltas = [r["delta"] for r in rows]
        per_arm[variant] = statistics.fmean(deltas)
        lines.append(
            f"  {variant:<9} n={len(rows):<3} judge {statistics.fmean(r['judge'] for r in rows):.3f}"
            f"   human {statistics.fmean(r['human'] for r in rows):.3f}"
            f"   bias {per_arm[variant]:+.3f}"
        )

    lines += ["", f"  Mean absolute error: "
                  f"{statistics.fmean(abs(r['delta']) for r in scored):.3f}"]

    if len(per_arm) == 2:
        differential = per_arm["cot"] - per_arm["baseline"]
        lines += [
            "",
            f"  DIFFERENTIAL BIAS (cot - baseline): {differential:+.3f}",
            "  This is the number that matters. A judge biased equally in both arms",
            "  leaves the 0.841 -> 0.997 delta standing, because the claim is a",
            "  difference and a constant bias cancels.",
        ]
        measured_delta = 0.997 - 0.841
        if abs(differential) >= measured_delta / 2:
            lines.append(f"  -> LARGE relative to the measured delta of "
                         f"{measured_delta:.3f}. The Sprint 3 groundedness claim "
                         f"is not safe as stated.")
        elif abs(differential) >= measured_delta / 5:
            lines.append(f"  -> Material next to the {measured_delta:.3f} delta. "
                         f"Quote the delta with this caveat attached.")
        else:
            lines.append(f"  -> Small next to the {measured_delta:.3f} delta. The "
                         f"comparison survives the check.")

    if refusals:
        lines += ["", f"  Refusals held out ({len(refusals)}), labelled 0 claims:"]
        for row in refusals:
            note = f" -- {row['comment']}" if row["comment"] else ""
            lines.append(f"    item {row['item']:>3} ({row['variant']}, judge "
                         f"{row['judge']:.2f}){note}")
        lines.append("  These are the false-refusal bug, not a groundedness failure;")
        lines.append("  the refusal suite measures them, not this sheet.")

    disagreements = [r for r in scored if abs(r["delta"]) >= 0.25]
    if disagreements:
        lines += ["", f"  Worth reading by hand -- |delta| >= 0.25 ({len(disagreements)}):"]
        for row in sorted(disagreements, key=lambda r: -abs(r["delta"])):
            note = f" -- {row['comment']}" if row["comment"] else ""
            lines.append(f"    item {row['item']:>3} ({row['variant']}) judge "
                         f"{row['judge']:.2f} vs human {row['human']:.2f}{note}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", default=os.path.join(RESULTS_DIR, "calibration_sheet.md"))
    parser.add_argument("--key", default=os.path.join(RESULTS_DIR, "calibration_key.json"))
    args = parser.parse_args()

    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)

    text = report(*score(parse_sheet(args.sheet), key))
    print(text)
    return text


if __name__ == "__main__":
    main()
