"""
Re-derive the stored `refusal` labels from the stored answer text.

    python -m medbot.eval.relabel            # dry run, prints every change
    python -m medbot.eval.relabel --write    # rewrite the trial files

Why this exists. `refusal` is a *derived* field: the answer text is the
measurement, `is_refusal(text)` is the instrument, and the boolean is what the
instrument said at the time the trial ran. When the instrument is corrected, the
stored booleans are stale, and `test_refusal_labels_match_the_heuristic` fails
by design -- its docstring requires the F9 hand-validation to be redone rather
than silently inherited. This is the tool that applies the redone validation, so
that "fix the instrument" and "re-score the data with it" cannot drift apart.

It costs no quota: every answer is already on disk. Nothing here calls a model.

The text is never modified, only the derived label, so any run of this is
reversible by re-running it under a different version of `is_refusal`.
"""

import argparse
import glob
import json
import os

from medbot.eval.refusal_trials import is_refusal

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))


def relabel_file(path, write=False):
    """Returns (changes, total) where changes is a list of flipped trials."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    changes, total = [], 0
    for question, by_variant in data.items():
        for variant, attempts in by_variant.items():
            for i, attempt in enumerate(attempts):
                total += 1
                was = attempt.get("refusal")
                now = is_refusal(attempt.get("text", ""))
                if was != now:
                    changes.append((question, variant, i + 1, was, now))
                    if write:
                        attempt["refusal"] = now

    if write and changes:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    return changes, total


def question_counts(path):
    """Questions that ever refuse, per arm -- the unit the Fisher test uses."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for _question, by_variant in data.items():
        for variant, attempts in by_variant.items():
            ever = any(is_refusal(a.get("text", "")) for a in attempts)
            out.setdefault(variant, [0, 0])
            out[variant][0] += ever
            out[variant][1] += 1
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="Rewrite the files. Without this, nothing is modified.")
    parser.add_argument("--glob", default="*trials.json",
                        help="Which trial files to re-score.")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, args.glob)))
    grand_changes = grand_total = 0

    for path in paths:
        changes, total = relabel_file(path, write=args.write)
        grand_changes += len(changes)
        grand_total += total
        if not changes:
            continue
        print(f"\n{os.path.basename(path)}  ({len(changes)} of {total} labels change)")
        for question, variant, trial, was, now in changes:
            print(f"  {variant:<16} {question[:44]:<46} t{trial}  {was} -> {now}")
        counts = question_counts(path)
        summary = "  ".join(f"{v}: {c[0]}/{c[1]} questions ever refusing"
                            for v, c in sorted(counts.items()))
        print(f"  after: {summary}")

    verb = "rewritten" if args.write else "would change (dry run)"
    print(f"\n{grand_changes} of {grand_total} labels {verb} across {len(paths)} files")
    if not args.write and grand_changes:
        print("Re-run with --write to apply.")


if __name__ == "__main__":
    main()
