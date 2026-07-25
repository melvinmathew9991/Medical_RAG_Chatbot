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
from medbot.eval.dataset import EVAL_QUESTIONS
from medbot.eval.run_eval import call_with_backoff, preflight

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# The whole evaluation set. Every question in it was chosen in Sprint 2 because a
# dedicated encyclopedia entry for that exact topic exists in the indexed volume,
# verified by reading the retrieved chunks (see medbot/eval/dataset.py). So the
# corpus answers all of them and every refusal here is a failure, which is the
# only property this suite needs.
#
# Sprint 4 (audit finding F1) widened this from 4 questions to all 24. The old
# list was the four zero-groundedness cases named in Sprint 2's results.md, and
# 4 units is far too few to say anything at the question level -- the Sprint 3
# headline, 2 of 4 questions fixed, was Fisher exact p=0.43.
#
# Widening also corrected the selection itself. Re-mining the stored Sprint 3
# answers with `is_refusal` (the criterion the audit suggested: high Precision@4
# but low groundedness) found FOUR baseline refusals, and only two of them were
# in the old list:
#
#   question                          P@4   baseline claim   in old list
#   What are the symptoms of breast cancer?  1.00   0.00      yes
#   What is bursitis?                        0.75   0.00      yes
#   What causes bladder cancer?              1.00   0.00      NO
#   What is atherosclerosis?                 1.00   0.50      NO
#
# "What causes bladder cancer?" answered "I don't know, as the provided context
# does not state the exact cause of bladder cancer" on a perfect Precision@4 --
# a textbook instance of the exact bug this suite exists to measure, sitting
# outside the suite for two sprints. Meanwhile "What causes bedsores?" and "What
# is an abscess?", both in the old list, did not refuse at all in that run.
# Hand-picking from a prose summary is how that happens; the eval set is the
# honest denominator.
REFUSAL_QUESTIONS = [q["question"] for q in EVAL_QUESTIONS]

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
#
# Sprint 4 (audit finding F7) took this from 2 questions to 10, and every
# candidate was checked with `python -m medbot.eval.verify_coverage` -- local
# retrieval only, no quota -- instead of being reasoned about from its first
# letter. That immediately caught the F8 mistake still live in this very list:
#
#   "What are the symptoms of diabetes?" IS COVERED and has been removed. The
#   top-4 chunks for it include the *blood sugar tests* entry (a B entry),
#   which explains that "a person with diabetes mellitus either does not make
#   enough insulin, or makes insulin that does not work properly... blood sugar
#   that remains high, a condition called hyperglycemia". A model answering
#   from that is reading the corpus, not inventing. Scoring its answer as a
#   hallucination would have been wrong, and it was half of the 2-question
#   guard the Sprint 3 result leaned on.
#
# Also rejected for contamination, with the term appearing in retrieved text:
# Parkinson's, tuberculosis, kidney stones, malaria, migraine, lupus, epilepsy,
# varicose veins, rabies, hemorrhoids, tonsillitis, warts, vertigo, tinnitus.
# Not even considered: gout and rheumatoid arthritis (named in the bursitis
# entry), osteoporosis and anorexia (chain-of-thought exemplars in prompt.py).
# The corpus cross-references far more C-Z topics than "A-B only" suggests.
OVERANSWER_QUESTIONS = [
    "How is psoriasis treated?",
    "What are the symptoms of schizophrenia?",
    "What are the symptoms of shingles?",
    "How is glaucoma treated?",
    "What are the symptoms of measles?",
    "How is scabies treated?",
    "How is ringworm treated?",
    "What are the symptoms of mumps?",
    "How is sciatica treated?",
    "What causes plantar fasciitis?",
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


def load_existing(out_filename):
    """Stored trials for this output file, or {} if there are none."""
    path = os.path.join(RESULTS_DIR, out_filename)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def run(trials, variants, questions, out_filename, resume=False):
    """
    Run `trials` repeats of every question in every variant, checkpointing as it goes.

    With `resume=True`, any (question, variant) cell that already has at least
    `trials` recorded attempts is skipped instead of re-measured.

    Why resume exists: this suite costs 60-144 calls against a 500/day free-tier
    quota, and the Sprint 4 overanswer run died of quota exhaustion at question 2
    of 10. Without resume the retry restarts at question 1 and re-spends quota on
    cells that are already recorded -- so if the quota runs out at the same place
    twice, the run can never reach question 3. The suite was stuck in exactly that
    loop before this was added.

    The trade resume makes: cells recorded by different runs are combined, so if
    the prompt changed in between, the arms are no longer comparable. The caller
    warns about this. Use a fresh --out-prefix, not --resume, after a prompt edit.
    """
    existing = load_existing(out_filename) if resume else {}
    if existing:
        recorded = sum(1 for q in existing for v in existing[q] if len(existing[q][v]) >= trials)
        print(f"resuming: {recorded} (question, variant) cells already have "
              f">={trials} trials and will be skipped", flush=True)

    # Model first, then the quota check, then the index. Loading the FAISS index
    # and the fastembed model takes ~25s and is pure waste if the quota is dead,
    # so the cheap fatal check goes ahead of the expensive setup: this now fails
    # in under two seconds instead of after the index load.
    model = initialize_model()
    if model is None:
        raise RuntimeError("Need a working model; check GOOGLE_API_KEY.")

    # Before committing 54-240 calls. Costs one request; saves the run from
    # spending several minutes in backoff only to die with nothing recorded.
    blocked = preflight(model)
    if blocked:
        raise SystemExit(f"\nPreflight failed, nothing was run.\n\n{blocked}\n")

    vectordb = create_vector_database()
    if vectordb is None:
        raise RuntimeError("Need a working vectordb; check vectorstore/.")

    out = {}
    for question in questions:
        out[question] = dict(existing.get(question, {})) if existing else {}
        for variant in variants:
            done = out[question].get(variant, [])
            if len(done) >= trials:
                print(f"  {variant} | {question[:40]:<40} "
                      f"skipped, {len(done)} trials already recorded", flush=True)
                continue

            # A partially-filled cell is re-run from scratch rather than topped
            # up. The cell -- one question in one arm -- is the unit the rate is
            # computed over, so mixing trials from two runs inside one cell would
            # silently blend two prompt versions if the prompt changed between
            # them. Whole cells recorded under different runs are the same risk
            # in principle, which is what --resume's warning is for; within a
            # cell it costs at most `trials` calls to avoid it entirely.
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
                      f"-> {'REFUSED' if refused else 'answered'}", flush=True)
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
    parser.add_argument("--resume", action="store_true",
                        help="Keep (question, variant) cells that already have enough "
                             "trials recorded and only measure the missing ones. Use "
                             "after a run dies on the 500/day quota. Do NOT use after "
                             "editing the prompt -- it would mix prompt versions.")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    if args.resume:
        print("--resume: combining cells across runs. Only valid if the prompt, the "
              "index and the model are unchanged since the recorded trials.\n", flush=True)

    if args.suite in ("refusal", "both"):
        results = run(args.trials, variants, REFUSAL_QUESTIONS,
                      f"{args.out_prefix}refusal_trials.json", resume=args.resume)
        print(summarize(results, variants, args.trials,
                        "False refusals on questions the corpus DOES answer (lower is better)"))

    if args.suite in ("overanswer", "both"):
        results = run(args.trials, variants, OVERANSWER_QUESTIONS,
                      f"{args.out_prefix}overanswer_trials.json", resume=args.resume)
        print(summarize(results, variants, args.trials,
                        "Refusals on out-of-corpus questions (higher is better - "
                        "an answer here is invented)"))

    print(f"\nRaw trials written to {RESULTS_DIR}")
