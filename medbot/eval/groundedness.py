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
