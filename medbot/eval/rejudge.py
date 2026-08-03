"""
Re-score already-generated answers with the claim-level groundedness judge.

    .venv-gemini/Scripts/python.exe -m medbot.eval.rejudge --variants baseline,cot

Sprint 3's audit found the original 0-100 judge collapsing to exactly 0.0 or 1.0
on every answer (finding F2), which makes "mean groundedness 0.92 -> 1.00"
uninformative about anything except how many answers were wholly unsupported.
This re-grades the stored answers in results_<variant>.json with
judge_groundedness_claims and writes results_<variant>_claims.json.

It re-judges rather than re-runs: the answers already exist, so regenerating them
would spend twice the quota and introduce fresh sampling variation into a
comparison that is supposed to hold the answers fixed.
"""

import argparse
import json
import os
import statistics

from medbot.data_processing import create_vector_database
from medbot.eval.groundedness import judge_groundedness_claims
from medbot.eval.run_eval import RESULTS_DIR, TOP_K, call_with_backoff, preflight
from medbot.model_handler import initialize_model


def rejudge(variant, retriever, model):
    src = os.path.join(RESULTS_DIR, f"results_{variant}.json")
    with open(src, encoding="utf-8") as f:
        stored = json.load(f)

    out = []
    for i, row in enumerate(stored["results"], start=1):
        question, answer = row["question"], row.get("answer")
        print(f"[{i}/{len(stored['results'])}] {variant}: {question}", flush=True)
        if not answer:
            out.append({**row, "claim_score": None, "claims": None, "supported": None,
                        "claim_rationale": "no stored answer"})
            continue

        # Re-retrieve rather than storing context in results.json: retrieval is
        # local, deterministic and free, and was verified identical across arms.
        context = "\n\n".join(d.page_content for d in retriever.invoke(question))
        judged = call_with_backoff(judge_groundedness_claims, model, question, context, answer)
        out.append({**row,
                    "claim_score": judged["score"],
                    "claims": judged["claims"],
                    "supported": judged["supported"],
                    "claim_rationale": judged["rationale"]})

        with open(os.path.join(RESULTS_DIR, f"results_{variant}_claims.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"variant": variant, "results": out}, f, indent=2)
    return out


def summarize(variant, rows):
    scores = [r["claim_score"] for r in rows if r.get("claim_score") is not None]
    distinct = sorted({round(s, 3) for s in scores})
    return {
        "variant": variant,
        "n_scored": len(scores),
        "mean_claim_groundedness": statistics.mean(scores) if scores else None,
        "distinct_values": distinct,
        "is_still_binary": set(distinct) <= {0.0, 1.0},
        "below_1.0": [r["question"] for r in rows
                      if r.get("claim_score") is not None and r["claim_score"] < 1.0],
    }


def main(variants):
    """
    Set up, preflight, and re-judge every variant in `variants`.

    A function rather than the body of `if __name__ == "__main__"` so the quota
    preflight below can be tested. It lived in the __main__ block until
    2026-08-03, which made it unreachable from pytest and therefore removable
    without any test noticing -- the same shape as the gates §9 found that could
    not fail.
    """
    # Model, then quota, then index -- see the note in run_eval.run. One judge
    # call per stored answer, so a full re-judge of two arms is ~48 requests
    # against the 500/day cap and is worth one preflight call.
    model = initialize_model()
    if model is None:
        raise RuntimeError("Need a working model; check GOOGLE_API_KEY.")

    blocked = preflight(model)
    if blocked:
        raise SystemExit(f"\nPreflight failed, nothing was re-judged.\n\n{blocked}\n")

    vectordb = create_vector_database()
    if vectordb is None:
        raise RuntimeError("Need a working vectordb; check vectorstore/.")
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    return [summarize(v, rejudge(v, retriever, model)) for v in variants]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", default="baseline,cot")
    args = parser.parse_args()

    summaries = main([v.strip() for v in args.variants.split(",") if v.strip()])
    print()
    print(json.dumps(summaries, indent=2))
