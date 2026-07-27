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
import re

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

# Second condition, added 2026-07-27 after the 5-trial re-run (see below). A
# marker in the opening window makes an answer a refusal *candidate*; it only
# scores as a refusal if the answer then fails to deliver any substance.
#
# Why this was needed. The 5-trial run surfaced two patterns the 3-trial run had
# too few samples to show, both scored as refusals by the window rule alone and
# both wrong:
#
#   1. Few-shot contamination. The model declines a question from its own
#      selected exemplar, then answers the real one:
#        "I don't know the answer to your question about BEE STINGS based on the
#         provided context. However, regarding how burns should be treated, ..."
#      (2,462 chars of correct burn treatment followed.) Also seen as osteoporosis
#      standing in for atherosclerosis, and nosebleeds for bursitis. This is a
#      real defect -- the Sprint 3 audit already recorded it -- but it is a
#      *contamination* defect, not a declined question, and counting it as a
#      false refusal attributes it to the wrong bug.
#   2. A leading hedge before a full answer:
#        "The context does not provide a single, straightforward list labeled
#         'symptoms of alcoholism,' but it describes numerous health problems..."
#      followed by 1,189 chars of symptoms. The note on REFUSAL_WINDOW_CHARS
#      anticipated *trailing* caveats; this is the same thing at the front.
#
# What makes this hard, and why the two obvious fixes are wrong:
#
#   - Length alone does not work. Genuine refusals frequently run 200-400 chars
#     because they helpfully enumerate what the corpus covers *instead*:
#     "I don't know based on the provided context. The context contains
#      information on treating acne, bedsores, and atopic dermatitis, but it does
#      not mention psoriasis." That is a refusal, and a long one.
#   - Matching the question's own terms is worse than useless: the refusal above
#     ends "...does not mention PSORIASIS", so a term check flips exactly the
#     out-of-corpus refusals that the hallucination guard depends on. Tried and
#     rejected -- it turned 60/60 into a much weaker guard, which is the one
#     direction this instrument must never fail in.
#
# So the test is: strip every sentence that asserts absence, and see whether any
# content survives. A refusal is all absence; a contaminated or hedged answer has
# substance left over. Validated below against all 572 stored trials.
ABSENCE_RE = re.compile(
    r"(?:do(?:es)?|did)\s*n(?:o|')t\s+(?:\w+\s+){0,2}"
    r"(?:contain|provide|mention|list|state|define|specify|describe|explain|"
    r"include|say|detail|support|address|cover|give|offer|discuss)"
    r"|i\s+(?:do\s*n(?:o|')t|cannot|can'?t|am\s+unable\s+to)\s+"
    r"(?:know|answer|find|determine|provide|give)"
    r"|no\s+(?:information|mention|details|specific\s+information)"
    r"|not\s+(?:mentioned|found|stated|listed|specified|covered|discussed)"
    r"|outside\s+the\s+provided|beyond\s+the\s+provided"
    r"|consult\s+(?:a|your|with)"
    r"|i\s*'?\s*a?m\s+sorry",
    re.IGNORECASE,
)
# The {0,2} gap is not decoration: "does not EXPLICITLY define" and "does not
# support AN ANSWER TO" both appeared in the stored trials and were mislabelled
# by fixed-string matching during calibration.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?:])\s+|\n+")

# Chosen from the calibration, not by taste. Across the 138 marker-flagged
# trials, surviving content is 0 chars for 119 of them (flat refusals) and 190+
# for every contaminated or hedged answer. The only values in between are three
# identical 113-char cases, so the threshold sits in the 113-190 gap.
#
# Those three are the one genuinely debatable label in the set, and they are
# recorded here rather than smoothed over. All three are "What causes bladder
# cancer?" under baseline:
#
#   "I don't know, as the provided context does not state the exact cause of
#    bladder cancer. However, the text notes that the exact cause is not known,
#    though smoking is considered the greatest risk factor."
#
# Scored a refusal at this threshold. The argument the other way is real: the
# corpus itself says the cause is unknown, so declining to give one is arguably
# correct, and the answer does supply the risk factor. Lowering the threshold to
# 110 would flip all three and take baseline from 7/24 questions to 6/24. It is
# left as a refusal because the answer opens by declining the question that was
# asked, and because Sprint 4 (audit F1) singled this question out as the
# textbook false refusal -- flipping it silently here would quietly erase that
# finding. Revisit deliberately if at all, not by nudging the constant.
MIN_SUBSTANCE_CHARS = 120


def delivered_content(answer):
    """The part of `answer` that is not asserting the absence of an answer."""
    kept = []
    for sentence in SENTENCE_SPLIT.split(answer):
        sentence = sentence.strip(" \t-*•()")
        if sentence and not ABSENCE_RE.search(sentence):
            kept.append(sentence)
    return " ".join(kept)


def opens_with_absence_language(text):
    """
    Whether `text` opens by asserting that an answer is not available -- the
    candidate gate the substance rule then arbitrates.

    Both readers are consulted, and the reason is a defect found on 2026-07-27
    while re-running the Sprint 3 ablation. REFUSAL_MARKERS was calibrated
    against stored baseline and cot trials only, because those were the only two
    arms with data. The instruction-only arm phrases the same refusal
    differently:

        "Based on the provided context, there is no mention of the symptoms of
         shingles. Therefore, the context does not support an answer to this
         question."

    "no mention" is not in the list ("no information" and "does not mention"
    are), so the gate never opened, the substance rule was never reached, and
    six flat refusals across four out-of-corpus questions were scored as
    ANSWERS -- i.e. as invented content, in the one direction the hallucination
    guard must never fail in. It read as instruction-only inventing answers to
    mumps, measles, shingles and schizophrenia when it had refused all six times
    correctly.

    The general lesson, which is the same shape as audit F1: a marker list
    calibrated on two arms does not transfer to a third. A prompt change changes
    the model's refusal wording, so any instrument keyed to wording has to be
    re-validated per arm.

    ABSENCE_RE already covered this phrasing, since it was written against
    sentences rather than openings. It is not a superset of the marker list
    though -- "cannot answer" without a leading "I" matches a marker and not the
    regex -- so the fix is the union, not a replacement. The union is strictly
    broader than the old gate, so it can only add candidates, and every added
    candidate still has to survive the substance test to score as a refusal.
    That bounds the damage a wider gate can do: verified across all 1022 stored
    trials, it changes exactly the six labels above.
    """
    head = text[:REFUSAL_WINDOW_CHARS]
    lowered = head.lower()
    return (any(marker in lowered for marker in REFUSAL_MARKERS)
            or bool(ABSENCE_RE.search(head)))


def is_refusal(answer):
    if not answer or not answer.strip():
        return True
    text = answer.strip()
    if not opens_with_absence_language(text):
        return False
    return len(delivered_content(text)) < MIN_SUBSTANCE_CHARS


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
