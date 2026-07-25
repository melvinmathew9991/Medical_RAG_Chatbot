"""
LLM-as-judge faithfulness/groundedness scoring.

Disclosed limitation: the judge is the same Gemini model (gemini-flash-lite-latest)
that generated the answer being graded, not an independent model — this is a
known form of self-grading bias, kept here rather than hidden, because a second
model/provider isn't available under the project's free-tier constraint. A truly
independent judge is future work, not this v1.
"""

import re

JUDGE_PROMPT_TEMPLATE = (
    "You are grading whether a generated answer is grounded in the provided context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Generated answer: {answer}\n\n"
    "Score how well the generated answer's claims are supported by the context above, "
    "on a 0-100 scale, where 100 means every claim is directly supported by the context "
    "and 0 means the answer is entirely unsupported or hallucinated. Ignore the general "
    "medical disclaimer sentence, if present, when scoring.\n\n"
    "Respond in exactly this format:\n"
    "SCORE: <integer 0-100>\n"
    "RATIONALE: <one sentence>"
)


CLAIM_JUDGE_PROMPT_TEMPLATE = (
    "You are auditing whether a generated answer is supported by a source context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Generated answer: {answer}\n\n"
    "Break the generated answer into its distinct factual claims. Treat a statement "
    "that the context does not contain the answer as itself a claim, and mark it "
    "UNSUPPORTED if the context does in fact address the question. Ignore the general "
    "medical disclaimer sentence, and ignore advice to consult a doctor - neither is a "
    "factual claim about the question.\n\n"
    "For each claim, decide whether the context directly supports it.\n\n"
    "Respond in exactly this format:\n"
    "CLAIMS: <total number of claims>\n"
    "SUPPORTED: <number of those claims the context directly supports>\n"
    "RATIONALE: <one sentence, naming any unsupported claim>"
)


def judge_groundedness_claims(model, question, context, answer):
    """
    Claim-level groundedness: supported claims / total claims.

    Sprint 3's audit (finding F2) found the 0-100 rubric in judge_groundedness
    collapsing to exactly 0.0 or 1.0 across all 48 graded answers - it behaves as
    pass/fail, so it cannot see partial quality loss, which is precisely what a
    verbosity-increasing prompt change risks. Counting claims forces the judge to
    produce genuine intermediate values.

    Returns score=None when the answer contains no factual claims to audit, rather
    than a misleading 0.0 or 1.0.
    """
    prompt = CLAIM_JUDGE_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    try:
        response = model.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return {"score": None, "claims": None, "supported": None,
                "rationale": f"judge call failed: {e}"}

    claims_match = re.search(r"CLAIMS:\s*(\d+)", text)
    supported_match = re.search(r"SUPPORTED:\s*(\d+)", text)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text)

    claims = int(claims_match.group(1)) if claims_match else None
    supported = int(supported_match.group(1)) if supported_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()

    if claims is None or supported is None:
        score = None
    elif claims == 0:
        score = None
        rationale = f"no factual claims to audit; {rationale}"
    else:
        # Clamp: the judge occasionally reports more supported than total.
        score = min(supported, claims) / claims

    return {"score": score, "claims": claims, "supported": supported, "rationale": rationale}


def judge_groundedness(model, question, context, answer):
    prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    try:
        response = model.invoke(prompt)
        text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return {"score": None, "rationale": f"judge call failed: {e}"}

    score_match = re.search(r"SCORE:\s*(\d+)", text)
    rationale_match = re.search(r"RATIONALE:\s*(.+)", text)
    score = int(score_match.group(1)) / 100 if score_match else None
    rationale = rationale_match.group(1).strip() if rationale_match else text.strip()
    return {"score": score, "rationale": rationale}
