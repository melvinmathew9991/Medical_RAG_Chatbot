"""
Build a blinded sample for hand-calibrating the groundedness judge (audit F6).

    .venv-gemini/Scripts/python.exe -m medbot.eval.calibration_sample

Writes `calibration_sheet.md` (to label) and `calibration_key.json` (the answer
key -- do not read the key until the sheet is filled in).

Why: the claim-level judge is the same model that wrote the answers, has never
been checked against a human, and demonstrably grades things it should not --
the only sub-1.0 answer left in the shipped CoT arm scores 15/16 because the
model wrote "seizers" for "seizures" and the judge counted the misspelling as an
unsupported claim. Every number in results_sprint3.md inherits whatever bias it
has.

Three things are blinded, each for a reason:
  - which arm produced the answer, since the whole question is whether the judge
    favours the treatment arm;
  - the judge's own score, so it cannot anchor the labeller;
  - the order, so `baseline` and `cot` answers to the same question are not
    adjacent and cannot be read as a pair.

Markdown formatting is NOT stripped. It was tempting, since bolding and bullets
are a visible arm cue, but rewriting the text would change what is being judged
and a human labeller can be told to ignore layout. The cue is disclosed in the
sheet instead.

Sampling is stratified rather than random: every answer the judge did not score
1.00 is included, because those are the cases where its discrimination is doing
work and where a disagreement would matter most, plus a fixed-seed sample of
perfect scores to catch the opposite error -- a wrong answer waved through.
"""

import json
import os
import random

from medbot.data_processing import create_vector_database

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 20260725
N_PERFECT_PER_ARM = 9


def load(variant):
    path = os.path.join(RESULTS_DIR, f"results_{variant}_claims.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["results"]


def retrieved_context(vectordb, question, k=4):
    """
    Re-retrieve the context the answer was graded against.

    Not stored in the results files, so it is regenerated here. Safe because
    retrieval is deterministic against a fixed index and the same default k=4 the
    app and harness use -- but it is a regeneration, not a recording, and if the
    index is ever rebuilt this will no longer reproduce the original grading
    context.
    """
    docs = vectordb.as_retriever(search_kwargs={"k": k}).invoke(question)
    return [" ".join(d.page_content.split()) for d in docs]


def build():
    rows = []
    for variant in ("baseline", "cot"):
        for r in load(variant):
            rows.append({
                "variant": variant,
                "question": r["question"],
                "answer": r["answer"],
                "judge_score": r["claim_score"],
                "judge_claims": r["claims"],
                "judge_supported": r["supported"],
                "judge_rationale": r["claim_rationale"],
            })

    imperfect = [r for r in rows if r["judge_score"] < 1.0]
    perfect = [r for r in rows if r["judge_score"] >= 1.0]

    rng = random.Random(SEED)
    sampled_perfect = []
    for variant in ("baseline", "cot"):
        pool = [r for r in perfect if r["variant"] == variant]
        sampled_perfect += rng.sample(pool, min(N_PERFECT_PER_ARM, len(pool)))

    sample = imperfect + sampled_perfect
    rng.shuffle(sample)
    return sample


def write(sample, vectordb):
    sheet = os.path.join(RESULTS_DIR, "calibration_sheet.md")
    key_path = os.path.join(RESULTS_DIR, "calibration_key.json")

    lines = [
        "# Groundedness judge — blinded calibration sheet",
        "",
        f"{len(sample)} answers, arm and judge score hidden, order shuffled (seed {SEED}).",
        "",
        "## How to label",
        "",
        "For each item, read the CONTEXT, then the ANSWER, and count:",
        "",
        "- **claims** — distinct factual assertions the answer makes.",
        "- **supported** — how many of those the context actually supports.",
        "",
        "Write both numbers in the `LABEL:` line. Judge *support by the context only* —",
        "not whether the claim is true in general medicine, and not whether the answer is",
        "well written. A refusal (\"I don't know\") makes zero claims: write `0/0` and note",
        "in the comment whether the context did in fact answer the question, since that is",
        "the false-refusal bug rather than a groundedness failure.",
        "",
        "Ignore formatting. Some answers are bulleted and bolded and some are plain prose;",
        "that is an unavoidable cue to which prompt produced them, and it is exactly the",
        "bias this exercise is trying to measure, so please do not let it sway the count.",
        "",
        "Do not open `calibration_key.json` until this sheet is filled in.",
        "",
        "---",
        "",
    ]

    key = []
    for i, r in enumerate(sample, 1):
        context = retrieved_context(vectordb, r["question"])
        lines += [
            f"## {i}. {r['question']}",
            "",
            "**CONTEXT**",
            "",
        ]
        for j, chunk in enumerate(context):
            lines.append(f"> **[{j}]** {chunk}")
            lines.append(">")
        lines += [
            "",
            "**ANSWER**",
            "",
            "```",
            r["answer"],
            "```",
            "",
            "`LABEL: supported=__ / claims=__   comment: ____`",
            "",
            "---",
            "",
        ]
        key.append({
            "item": i,
            "question": r["question"],
            "variant": r["variant"],
            "judge_score": r["judge_score"],
            "judge_claims": r["judge_claims"],
            "judge_supported": r["judge_supported"],
            "judge_rationale": r["judge_rationale"],
        })

    with open(sheet, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2)

    print(f"wrote {sheet}")
    print(f"wrote {key_path}")
    n_imperfect = sum(1 for k in key if k["judge_score"] < 1.0)
    print(f"{len(sample)} items: {n_imperfect} the judge scored below 1.00, "
          f"{len(sample) - n_imperfect} it scored perfect")
    for variant in ("baseline", "cot"):
        print(f"  {variant}: {sum(1 for k in key if k['variant'] == variant)}")


if __name__ == "__main__":
    vectordb = create_vector_database()
    if vectordb is None:
        raise SystemExit("No vector index; run medbot.data_processing first.")
    write(build(), vectordb)
