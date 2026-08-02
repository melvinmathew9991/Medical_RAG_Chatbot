"""
Pins the out-of-corpus guard's question selection against real retrieval.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_out_of_corpus_selection.py

Offline but not free in time: it loads the committed FAISS index and the local
fastembed model (~25s cold). No network and no API quota, so it stays in the
default run -- `conftest.py`'s socket guard is active here and will fail the test
if anything reaches out.

What it protects. The out-of-corpus suite only means anything if the corpus really
does not cover those questions: a refusal is scored CORRECT there, so a question
the corpus quietly does cover turns a correct answer into a recorded failure and
weakens the hallucination guard to nothing. That is not hypothetical -- it has
happened twice. Audit F8 caught "What causes a stroke?" (covered inside the A
entries for embolism and atherosclerosis) and Sprint 4 caught "What are the
symptoms of diabetes?" (covered by the *blood sugar tests* B entry), which was
half of the entire 2-question guard the Sprint 3 result rested on.

`verify_coverage.py` made that check repeatable. This makes it automatic, so a
rebuilt index or a changed retriever fails a test rather than silently degrading
the guard between sprints.

The index is loaded directly rather than through `create_vector_database()`,
which falls back to *building* the index when it is absent or incomplete --
1225 chunks of embedding is not something a test should ever start.
"""

import os

import pytest

from medbot.config import PERSIST_DIR
from medbot.eval.refusal_trials import OVERANSWER_QUESTIONS
from medbot.eval.verify_coverage import CANDIDATES, check


@pytest.fixture(scope="module")
def vectordb():
    index_path = os.path.join(PERSIST_DIR, "index.faiss")
    if not os.path.exists(index_path):
        # Ergonomics, not a gate: skipping keeps a contributor mid-rebuild from
        # reading a wall of failures. Absence itself is caught as a *failure* by
        # tests/test_vector_index.py, so this cannot make the suite green on
        # nothing.
        pytest.skip(f"no FAISS index at {index_path}")

    # Built here rather than imported: `create_vector_database` constructs its
    # embeddings inside the function, and importing the module gives no handle on
    # them. Model weights are read from the local fastembed cache; if they are
    # missing, fastembed would fetch them and conftest's socket guard will fail
    # the test by name rather than letting a "unit" test download 130MB.
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    from medbot.config import LOCAL_EMBEDDING_MODEL

    embeddings = FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)


def test_every_shipped_question_is_still_absent_from_the_corpus(vectordb):
    """
    The gate. If retrieval starts surfacing the topic for any of these, the
    question is no longer out-of-corpus and scoring a refusal as correct is wrong.
    """
    covered = {}
    for question in OVERANSWER_QUESTIONS:
        _, terms, hits = check(vectordb, question)
        if hits:
            covered[question] = sorted({t for _, found in hits for t in found})

    assert not covered, (
        f"out-of-corpus questions whose topic now appears in retrieved text: {covered}. "
        "Either the index changed, or the question was never out-of-corpus -- see "
        "audit F8 and re-screen with `python -m medbot.eval.verify_coverage`."
    )


@pytest.mark.parametrize("question", [
    "What are the symptoms of diabetes?",
    "What causes Parkinson's disease?",
    "What causes vertigo?",
])
def test_rejected_candidates_are_still_correctly_rejected(vectordb, question):
    """
    The other direction, and the one that keeps the screen honest: a screen that
    accepts everything would pass the test above trivially. Diabetes is included
    by name because it is the specific question that slipped through into the
    shipped guard.
    """
    _, _, hits = check(vectordb, question)
    assert hits, (
        f"{question!r} no longer looks covered, so the coverage screen is not "
        "discriminating -- it would now accept a question it correctly rejected."
    )


def test_the_shipped_suite_is_a_subset_of_the_screened_candidates():
    """
    Selection has to be auditable: every shipped question must have gone through
    the screen. `CANDIDATES` held only 12 of the 25 actually screened until this
    was noticed, which made the sprint's own count ("14 of 26") uncheckable.
    """
    missing = sorted(set(OVERANSWER_QUESTIONS) - set(CANDIDATES))
    assert not missing, f"shipped without appearing in the screened list: {missing}"


def test_the_rejections_are_kept_as_evidence():
    """
    The rejected candidates are what shows the kept 10 were not cherry-picked, so
    the list must stay meaningfully larger than the suite.
    """
    rejected = set(CANDIDATES) - set(OVERANSWER_QUESTIONS)
    assert len(rejected) >= 12, (
        f"only {len(rejected)} rejected candidates recorded; deleting rejections "
        "loses the evidence that the selection was a screen and not a search"
    )


if __name__ == "__main__":
    raise SystemExit(
        "This module uses pytest fixtures. Run:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_out_of_corpus_selection.py"
    )
