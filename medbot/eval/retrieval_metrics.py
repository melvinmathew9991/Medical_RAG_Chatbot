"""
Precision@K for MEDBOT's retriever.

Relevance is judged by case-insensitive substring match against a question's
`expected_keywords` (see dataset.py) rather than an embedding-based judge —
simple and transparent, at the cost of being stricter than a human grader
would be about paraphrased-but-relevant chunks.
"""


def is_relevant(chunk_text, expected_keywords):
    text = chunk_text.lower()
    return any(kw.lower() in text for kw in expected_keywords)


def precision_at_k(chunks, expected_keywords):
    """Fraction of the given (already top-K) retrieved chunks judged relevant."""
    if not chunks:
        return 0.0
    relevant = sum(1 for c in chunks if is_relevant(c, expected_keywords))
    return relevant / len(chunks)
