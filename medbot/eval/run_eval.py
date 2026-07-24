"""
MEDBOT Sprint 2 evaluation harness.

Run with: .venv-gemini/Scripts/python.exe -m medbot.eval.run_eval

For each question in dataset.EVAL_QUESTIONS: retrieves top-K chunks and scores
Precision@K, then runs the real answer-generation chain and scores groundedness
via an LLM-judge. Writes results.json and results.md next to this file.
"""

import json
import os
import statistics
import time

from google.api_core.exceptions import ResourceExhausted

from medbot.data_processing import create_vector_database
from medbot.model_handler import initialize_model
from medbot.query_handler import create_query_chain
from medbot.eval.dataset import EVAL_QUESTIONS
from medbot.eval.retrieval_metrics import precision_at_k
from medbot.eval.groundedness import judge_groundedness

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
            print(f"  rate limited, backing off {RATE_LIMIT_BACKOFF_SECONDS}s "
                  f"(attempt {attempt}/{MAX_RATE_LIMIT_RETRIES})")
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)


def run():
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

        chain = create_query_chain(model, vectordb, question)
        if chain is None:
            results.append({
                "question": question,
                "precision_at_k": precision,
                "groundedness": None,
                "groundedness_rationale": "chain creation failed",
                "answer": None,
            })
            continue

        response = call_with_backoff(chain.invoke, {"query": question})
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
        with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as f:
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


def write_report(results, summary):
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    precision_line = (
        f"- **Mean Precision@{TOP_K}**: {summary['mean_precision_at_k']:.2f}"
        if summary["mean_precision_at_k"] is not None else "- Mean Precision@K: n/a"
    )
    groundedness_line = (
        f"- **Mean groundedness (LLM-judge, 0-1)**: {summary['mean_groundedness']:.2f}"
        if summary["mean_groundedness"] is not None else "- Mean groundedness: n/a"
    )

    lines = [
        "# MEDBOT Sprint 2 Evaluation Results",
        "",
        f"Ran {summary['n_questions']} questions through the live RAG pipeline "
        f"(retrieval top-{TOP_K}, Gemini `gemini-flash-lite-latest` generation).",
        "",
        precision_line,
        groundedness_line,
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

    md_path = os.path.join(RESULTS_DIR, "results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


if __name__ == "__main__":
    start = time.time()
    results = run()
    summary = summarize(results)
    json_path, md_path = write_report(results, summary)
    print(f"\nDone in {time.time() - start:.0f}s. Report: {md_path}")
    print(json.dumps(summary, indent=2))
