"""
MEDBOT Sprint 2 evaluation harness.

Run with: .venv-gemini/Scripts/python.exe -m medbot.eval.run_eval

For each question in dataset.EVAL_QUESTIONS: retrieves top-K chunks and scores
Precision@K, then runs the real answer-generation chain and scores groundedness
via an LLM-judge.

Writes `results_<variant>.json` and `results_<variant>.md` next to this file --
always suffixed, resolving an omitted `--variant` to the app default rather than
writing the unsuffixed names, which would overwrite Sprint 2's recorded run (see
`results_sprint4.md` §6). This docstring said "results.json and results.md" until
the 2026-07-27 audit; that is what the code did before that fix, not after.
"""

import argparse
import json
import os
import statistics
import time

from google.api_core.exceptions import ResourceExhausted

from medbot.data_processing import create_vector_database
from medbot.eval.dataset import EVAL_QUESTIONS
from medbot.eval.groundedness import judge_groundedness
from medbot.eval.retrieval_metrics import precision_at_k
from medbot.model_handler import initialize_model
from medbot.prompt import DEFAULT_PROMPT_VARIANT, PROMPT_VARIANTS
from medbot.query_handler import create_query_chain, run_query

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_K = 4
# gemini-flash-lite-latest's free tier is capped at 15 requests/minute (separate
# from the 500/day cap) - space calls out and back off hard on 429s rather than
# relying on langchain's built-in retry, which gives up well before a 60s cooldown.
SECONDS_BETWEEN_CALLS = 5
RATE_LIMIT_BACKOFF_SECONDS = 65
MAX_RATE_LIMIT_RETRIES = 3


def call_with_backoff(fn, *args, **kwargs):
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 1):
        try:
            time.sleep(SECONDS_BETWEEN_CALLS)
            return fn(*args, **kwargs)
        except ResourceExhausted:
            if attempt == MAX_RATE_LIMIT_RETRIES:
                raise
            # flush: without it this line sits in the stdout buffer whenever the
            # run is piped, so a quota-dead run is indistinguishable from a slow
            # one -- 12 minutes of empty console before the first checkpoint.
            print(f"  rate limited, backing off {RATE_LIMIT_BACKOFF_SECONDS}s "
                  f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})", flush=True)
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)


# Telling the two 429s apart. The important difference is that backing off cannot
# clear the daily cap: RPM recovers in 60s, RPD only at the rollover, so retrying
# a day-capped run just burns MAX_RATE_LIMIT_RETRIES x 65s per call and dies anyway.
#
# The metric name alone does NOT discriminate: Google reports both against
# `generate_content_free_tier_requests`, and the per-minute quota id merely
# suffixes it -- so a substring match on the metric treats an RPM blip as fatal
# and aborts runs that would have finished. The quota *id* is what differs
# (GenerateRequestsPerDayPerProjectPerModel-FreeTier vs ...PerMinute...), with
# the observed daily limit as a fallback signal.
PER_MINUTE_SIGNALS = ("perminute", "per_minute", "requests_per_minute")
DAILY_QUOTA_SIGNALS = ("perday", "per_day", "limit: 500")


def preflight(model):
    """
    One minimal call, to fail a quota-dead run in seconds instead of after the
    first 3x65s backoff cycle. Returns None if the model answers.

    Costs 1 request against the daily 500. Worth it before committing 54-240.

    Distinguishes the two 429s: a per-minute limit is transient and the caller
    should just proceed into the normal backoff path, but the daily cap cannot be
    waited out within a run. The rollover is midnight US Pacific, NOT local
    midnight -- assuming otherwise is why the Sprint 4 out-of-corpus run was
    retried at 01:00 IST and failed against a quota that had not reset.
    """
    try:
        model.invoke("ok")
        return None
    except ResourceExhausted as exc:
        message = str(exc)
        low = message.lower()
        # Checked first: the per-minute id contains the daily metric name as a
        # substring, so testing for the daily signal first would swallow it.
        if any(signal in low for signal in PER_MINUTE_SIGNALS):
            return None  # transient, the normal backoff path handles it
        if any(signal in low for signal in DAILY_QUOTA_SIGNALS):
            return (
                "Daily free-tier quota (500 requests) is exhausted. Backing off "
                "will not clear it -- the counter rolls over at midnight US "
                "Pacific (~12:30 IST), not local midnight.\n"
                f"  {message.splitlines()[0]}"
            )
        # An unrecognised 429. Prefer letting the run try and back off over
        # aborting on a message format that may simply have changed.
        return None


def result_paths(variant):
    """
    Always write to a variant-suffixed filename, resolving None to whichever
    variant actually ran.

    Sprint 2's results.json/results.md are a recorded historical run and nothing
    here should overwrite them. An earlier version of this function wrote the
    unsuffixed names when `variant` was None, which was a trap: the app default
    is now "cot", so a plain `python -m medbot.eval.run_eval` would have silently
    replaced Sprint 2's baseline record with CoT numbers under a filename that
    still claimed to be the baseline.
    """
    variant = variant or DEFAULT_PROMPT_VARIANT
    return (os.path.join(RESULTS_DIR, f"results_{variant}.json"),
            os.path.join(RESULTS_DIR, f"results_{variant}.md"))


def run(variant=None):
    vectordb = create_vector_database()
    model = initialize_model()
    if vectordb is None or model is None:
        raise RuntimeError(
            "Eval harness needs a working vectordb and model; check GOOGLE_API_KEY and vectorstore/."
        )

    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})
    results = []

    for i, case in enumerate(EVAL_QUESTIONS, start=1):
        question = case["question"]
        expected_keywords = case["expected_keywords"]
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")

        retrieved_docs = retriever.invoke(question)
        retrieved_texts = [d.page_content for d in retrieved_docs]
        precision = precision_at_k(retrieved_texts, expected_keywords)

        chain = create_query_chain(model, vectordb, question, variant=variant)
        if chain is None:
            results.append({
                "question": question,
                "precision_at_k": precision,
                "groundedness": None,
                "groundedness_rationale": "chain creation failed",
                "answer": None,
            })
            continue

        # Goes through run_query, not chain.invoke, so the answer scored here is
        # the trace-stripped one the user would actually see.
        response = call_with_backoff(run_query, chain, question, variant=variant)
        answer = response.get("result", "")
        context = "\n\n".join(retrieved_texts)
        judged = call_with_backoff(judge_groundedness, model, question, context, answer)

        results.append({
            "question": question,
            "precision_at_k": precision,
            "groundedness": judged["score"],
            "groundedness_rationale": judged["rationale"],
            "answer": answer,
        })

        # Checkpoint after every question so a late-run rate-limit failure
        # doesn't discard already-completed (and already Gemini-billed) work.
        with open(result_paths(variant)[0], "w", encoding="utf-8") as f:
            json.dump({"summary": summarize(results), "results": results}, f, indent=2)

    return results


def summarize(results):
    precisions = [r["precision_at_k"] for r in results if r.get("precision_at_k") is not None]
    groundedness_scores = [r["groundedness"] for r in results if r.get("groundedness") is not None]
    return {
        "n_questions": len(results),
        "mean_precision_at_k": statistics.mean(precisions) if precisions else None,
        "mean_groundedness": statistics.mean(groundedness_scores) if groundedness_scores else None,
        "questions_below_0.5_precision": [
            r["question"] for r in results
            if r.get("precision_at_k") is not None and r["precision_at_k"] < 0.5
        ],
        "questions_below_0.5_groundedness": [
            r["question"] for r in results
            if r.get("groundedness") is not None and r["groundedness"] < 0.5
        ],
    }


def write_report(results, summary, variant=None):
    json_path, md_path = result_paths(variant)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    precision_line = (
        f"- **Mean Precision@{TOP_K}**: {summary['mean_precision_at_k']:.2f}"
        if summary["mean_precision_at_k"] is not None else "- Mean Precision@K: n/a"
    )
    groundedness_line = (
        f"- **Mean groundedness (binary LLM-judge, 0-1)**: {summary['mean_groundedness']:.2f}"
        if summary["mean_groundedness"] is not None else "- Mean groundedness: n/a"
    )
    # Sprint 3 audit finding F2: this judge returns only 0.0 or 1.0 in practice, so
    # the figure above reads as a quality score but is really a pass/fail rate. Say
    # so in the generated report itself - a reader who opens only this file should
    # not walk away quoting it as a quality benchmark.
    judge_caveat = (
        "> **Read the groundedness figure with care.** This judge returned only 0.0 or 1.0 on "
        "every answer it graded, so it behaves as pass/fail and cannot see partial degradation. "
        "For discriminating scores use the claim-level judge "
        "(`medbot.eval.rejudge` -> `results_<variant>_claims.json`), and see "
        "`results_sprint3.md` for the interpretation."
    )

    lines = [
        "# MEDBOT Evaluation Results"
        + (f" - `{variant}` prompt" if variant else " - Sprint 2"),
        "",
        f"Ran {summary['n_questions']} questions through the live RAG pipeline "
        f"(retrieval top-{TOP_K}, Gemini `gemini-flash-lite-latest` generation"
        + (f", `{variant}` prompt variant" if variant else "") + ").",
        "",
        precision_line,
        groundedness_line,
        "",
        judge_caveat,
        "",
        "**Methodology and limitations:** Precision@K uses keyword-containment against "
        "manually verified expected phrases (see `dataset.py`), not embedding similarity "
        "or human judgment — stricter than a human grader on paraphrased-but-relevant "
        "chunks. Groundedness is scored by the same Gemini model that generated the "
        "answer (disclosed self-grading bias) rather than an independent judge model. "
        "Both are intentionally scoped v1 approaches.",
        "",
        "## Weak spots",
        "",
        f"- Questions with Precision@{TOP_K} < 0.5: "
        f"{', '.join(summary['questions_below_0.5_precision']) or 'none'}",
        f"- Questions with groundedness < 0.5: "
        f"{', '.join(summary['questions_below_0.5_groundedness']) or 'none'}",
        "",
        "## Per-question results",
        "",
        "| Question | Precision@K | Groundedness | Rationale |",
        "|---|---|---|---|",
    ]
    for r in results:
        p = f"{r['precision_at_k']:.2f}" if r.get("precision_at_k") is not None else "n/a"
        g = f"{r['groundedness']:.2f}" if r.get("groundedness") is not None else "n/a"
        rationale = (r.get("groundedness_rationale") or "").replace("|", "/")
        lines.append(f"| {r['question']} | {p} | {g} | {rationale} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=sorted(PROMPT_VARIANTS), default=None,
        help=f"Prompt variant to evaluate. Omit to use the app default "
             f"({DEFAULT_PROMPT_VARIANT}). Output is always written to "
             f"results_<variant>.json/.md.",
    )
    args = parser.parse_args()

    start = time.time()
    results = run(variant=args.variant)
    summary = summarize(results)
    json_path, md_path = write_report(results, summary, variant=args.variant)
    print(f"\nDone in {time.time() - start:.0f}s. Report: {md_path}")
    print(json.dumps(summary, indent=2))
