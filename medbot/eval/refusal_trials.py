"""
Repeated-trial check on the questions Sprint 2 found the model falsely refusing.

Run with:
    .venv-gemini/Scripts/python.exe -m medbot.eval.refusal_trials --trials 5

Why this exists separately from run_eval: Sprint 2 established that these four
questions refuse *intermittently* (bedsores refused in the harness run, then 0/6
times in the temperature experiment). A single A/B pass therefore cannot tell a
real prompt improvement from run-to-run variance. This runs each question N
times per prompt variant so the comparison is a rate, not a coin flip.

For all four questions the retrieved context does contain the answer - verified
by hand in Sprint 2 - so here a refusal is always a failure, never correct
behaviour. (Calibration in the other direction, refusing when the context really
doesn't support an answer, is covered by the anorexia exemplar in
medbot/prompt.py, not by this script.)
"""

import argparse
import json
import os

from medbot.data_processing import create_vector_database
from medbot.model_handler import initialize_model
from medbot.query_handler import create_query_chain, run_query
from medbot.eval.run_eval import call_with_backoff

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# The four zero-groundedness cases from medbot/eval/results.md. The corpus
# answers all of these, so every refusal here is a failure.
REFUSAL_QUESTIONS = [
    "What causes bedsores?",
    "What are the symptoms of breast cancer?",
    "What is bursitis?",
    "What is an abscess?",
]

# The opposite check, and the one that keeps this sprint honest. Sprint 3 pushes
# the model to answer more readily, and the cheap way to score 0/20 above is a
# prompt that never refuses anything - which would swap a refusal bug for a
# hallucination bug on exactly the medical questions where making something up is
# worst. The indexed corpus covers encyclopedia entries A-B only (see
# medbot/eval/dataset.py), so it has no entry for any of these. Refusing is the
# correct answer, and a substantive reply means the model invented it.
#
# "What causes a stroke?" was in this list on the first Sprint 3 run and has been
# removed, so the reason is recorded here rather than lost: it is not actually
# out-of-corpus. Stroke causation is covered inside the *A* entries for embolism
# and atherosclerosis, and both prompt variants correctly answered from them
# (cot 3/3, baseline 2/3, all citing embolus and carotid blockage). It failed the
# inclusion criterion for this list, not the model. The wider lesson, which
# corrects the rule of thumb in medbot/eval/dataset.py: "the corpus is A-B only"
# does not imply a C-Z topic is unanswerable, because some C-Z topics are
# discussed inside A-B entries. Verify coverage per question, don't infer it from
# the first letter.
OVERANSWER_QUESTIONS = [
    "What are the symptoms of diabetes?",
    "How is psoriasis treated?",
]

REFUSAL_MARKERS = [
    "i don't know",
    "i do not know",
    "does not contain",
    "doesn't contain",
    "does not provide",
    "doesn't provide",
    "does not mention",
    "doesn't mention",
    "cannot answer",
    "can't answer",
    "no information",
]

# A refusal marker only counts when it appears near the start of the answer.
# The CoT prompt deliberately asks for partial answers that name what the context
# leaves out, so trailing caveats like "it does not provide a full list of risks"
# are expected and must not be scored as refusals - only a marker in the opening
# sentence means the model declined the question outright.
#
# Validated 2026-07-25 (audit finding F9): all 60 stored trial texts, 34 of them
# distinct, were hand-labelled and compared against this heuristic. Zero
# disagreements. The closest call was a CoT answer opening "The provided context
# does not explicitly define what bursitis is, so I cannot give a formal
# definition based on it. However, the text notes that it can flare up..." - which
# then supplies causes and symptoms, so "not a refusal" is the correct label and
# the heuristic got it right.
#
# That case is also why "cannot give" and "unable to" are deliberately NOT in the
# marker list: they appear in genuine partial answers, and adding them would
# convert a correct label into a false positive. The residual gap is a true
# refusal phrased with an unlisted verb, which would be scored as an answer and
# therefore flatter the CoT arm; re-run this validation if the marker list or the
# prompt wording changes.
REFUSAL_WINDOW_CHARS = 200


def is_refusal(answer):
    if not answer:
        return True
    head = answer[:REFUSAL_WINDOW_CHARS].lower()
    return any(marker in head for marker in REFUSAL_MARKERS)


def run(trials, variants, questions, out_filename):
    vectordb = create_vector_database()
    model = initialize_model()
    if vectordb is None or model is None:
        raise RuntimeError("Need a working vectordb and model; check GOOGLE_API_KEY and vectorstore/.")

    out = {}
    for question in questions:
        out[question] = {}
        for variant in variants:
            attempts = []
            for trial in range(1, trials + 1):
                chain = create_query_chain(model, vectordb, question, variant=variant)
                if chain is None:
                    raise RuntimeError(f"chain creation failed for variant {variant!r}")
                response = call_with_backoff(run_query, chain, question, variant=variant)
                answer = response.get("result", "")
                refused = is_refusal(answer)
                attempts.append({"refusal": refused, "text": answer})
                print(f"  {variant} | {question[:40]:<40} trial {trial}/{trials} "
                      f"-> {'REFUSED' if refused else 'answered'}")
            out[question][variant] = attempts

            # Checkpoint after each variant so a late rate-limit failure doesn't
            # discard calls already spent against the daily quota.
            with open(os.path.join(RESULTS_DIR, out_filename), "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
    return out


def summarize(out, variants, trials, label):
    lines = ["", f"{label} ({trials} trials per question):", ""]
    header = "| Question | " + " | ".join(variants) + " |"
    lines.append(header)
    lines.append("|---" * (len(variants) + 1) + "|")
    totals = {v: 0 for v in variants}
    for question, by_variant in out.items():
        cells = []
        for v in variants:
            n = sum(1 for a in by_variant.get(v, []) if a["refusal"])
            totals[v] += n
            cells.append(f"{n}/{trials}")
        lines.append(f"| {question} | " + " | ".join(cells) + " |")
    overall = [f"{totals[v]}/{len(out) * trials}" for v in variants]
    lines.append("| **overall** | " + " | ".join(overall) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5, help="Trials per question per variant.")
    parser.add_argument("--variants", default="baseline,cot",
                        help="Comma-separated prompt variants to compare. Ablation "
                             "arms: instruction-only, examples-only.")
    parser.add_argument("--out-prefix", default="",
                        help="Prefix for output filenames, so an ablation run does "
                             "not overwrite the main A/B trial data.")
    parser.add_argument("--suite", choices=["refusal", "overanswer", "both"], default="both",
                        help="'refusal': questions the corpus answers (refusing is a bug). "
                             "'overanswer': questions it does not (refusing is correct).")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.suite in ("refusal", "both"):
        results = run(args.trials, variants, REFUSAL_QUESTIONS,
                      f"{args.out_prefix}refusal_trials.json")
        print(summarize(results, variants, args.trials,
                        "False refusals on questions the corpus DOES answer (lower is better)"))

    if args.suite in ("overanswer", "both"):
        results = run(args.trials, variants, OVERANSWER_QUESTIONS,
                      f"{args.out_prefix}overanswer_trials.json")
        print(summarize(results, variants, args.trials,
                        "Refusals on out-of-corpus questions (higher is better - "
                        "an answer here is invented)"))

    print(f"\nRaw trials written to {RESULTS_DIR}")
