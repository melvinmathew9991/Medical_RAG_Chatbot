"""
Pins the screened question-set expansion against real retrieval.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_expansion_selection.py

Offline but not free in time: loads the committed FAISS index and the local
fastembed model. No network, no API quota — `conftest.py`'s socket guard is active.

Sibling of `test_out_of_corpus_selection.py`, and the mirror of it. That file pins
questions the corpus must NOT cover; this one pins questions it MUST cover, with
keywords that must be groundable in the chunks retrieval actually returns. Both
directions are checked, because a screen that accepts everything would pass the
first test in either file trivially.

Why it matters here specifically: an eval question whose entry is not really in the
corpus turns every correct refusal into a recorded bug, and its Precision@4 into a
measurement of corpus coverage rather than retrieval quality. That error has been
made twice in this project already (audit F8's stroke question, Sprint 4's diabetes
question), both times in the guard; this is the same error waiting in the eval set.
"""

import os
import statistics

import pytest

from medbot.config import PERSIST_DIR
from medbot.eval.dataset import EVAL_QUESTIONS, EVAL_QUESTIONS_V1, EXPANSION_QUESTIONS
from medbot.eval.refusal_trials import OVERANSWER_QUESTIONS
from medbot.eval.retrieval_metrics import precision_at_k
from medbot.eval.verify_entry import CANDIDATES, check_entry

# Recorded when the set was screened. Not aspirational: the point of a pinned
# threshold is that it fails if retrieval degrades, so it sits just under the
# measured value (0.8523).
MIN_MEAN_PRECISION = 0.84

# Auto-rejected by `verify_entry`, kept as the discrimination check. If the screen
# ever starts accepting these, it is no longer screening. "What causes back pain?"
# is here because it is the clearest failure: the top chunks were the bursitis and
# arthritis entries, matched on the word "pain".
REJECTED_BY_THE_SCREEN = [
    "What causes back pain?",
    "What causes blood clots?",
    "What is Barrett's esophagus?",
    "What is a blood sugar test?",
    "What are the symptoms of a bladder infection?",
]


@pytest.fixture(scope="module")
def vectordb():
    index_path = os.path.join(PERSIST_DIR, "index.faiss")
    if not os.path.exists(index_path):
        # Ergonomics, not a gate: skipping keeps a contributor mid-rebuild from
        # reading twelve failures. Absence itself is caught as a *failure* by
        # tests/test_vector_index.py, so this cannot make the suite green on
        # nothing.
        pytest.skip(f"no FAISS index at {index_path}")

    # Loaded directly, not via create_vector_database(), which BUILDS the index
    # when it is absent — 1225 chunks of embedding is not something a test starts.
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    from medbot.config import LOCAL_EMBEDDING_MODEL

    embeddings = FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)


def test_every_expansion_question_still_retrieves_its_entry(vectordb):
    """The gate: a rebuilt index or a changed retriever fails here, loudly."""
    failed = {}
    for case in EXPANSION_QUESTIONS:
        _, term, with_term, entry_chunks, ok = check_entry(vectordb, case["question"])
        if not ok:
            failed[case["question"]] = {
                "term": term, "chunks_with_term": with_term, "entry_shaped": entry_chunks,
            }
    assert not failed, (
        f"screened questions that no longer retrieve their entry: {failed}. "
        "Re-screen with `python -m medbot.eval.verify_entry` before trusting any "
        "Precision@4 or refusal number measured on them."
    )


def test_the_screen_still_rejects_what_it_rejected(vectordb):
    """
    The other direction. A screen that has degraded into accepting everything would
    pass the test above without discriminating at all.
    """
    wrongly_accepted = [
        q for q in REJECTED_BY_THE_SCREEN if check_entry(vectordb, q)[4]
    ]
    assert not wrongly_accepted, (
        f"the entry screen now accepts questions it correctly rejected: "
        f"{wrongly_accepted}. It is no longer discriminating."
    )


@pytest.fixture(scope="module")
def retrieved(vectordb):
    """
    One retrieval pass, shared by every test below that needs chunk text.

    Retrieving per test doubled the offline suite's runtime (7s -> 14s) for four
    identical queries. The suite being free and fast is what Sprint 5's CI depends
    on, so the cheap fixture is worth it.
    """
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    return {case["question"]: [d.page_content for d in retriever.invoke(case["question"])]
            for case in EXPANSION_QUESTIONS}


def test_every_keyword_is_grounded_in_retrieved_text(retrieved):
    """
    `expected_keywords` are the relevance judgement itself, so a keyword that
    appears in no retrieved chunk silently drags Precision@4 down and looks like a
    retrieval failure. Two traps here in practice: the corpus PDF uses curly
    apostrophes (Bell’s), and its text runs words together across line breaks.
    """
    ungrounded = {}
    for case in EXPANSION_QUESTIONS:
        chunks = [c.lower() for c in retrieved[case["question"]]]
        missing = [kw for kw in case["expected_keywords"]
                   if not any(kw.lower() in c for c in chunks)]
        if missing:
            ungrounded[case["question"]] = missing
    assert not ungrounded, f"keywords that match no retrieved chunk: {ungrounded}"


def test_expansion_precision_matches_what_was_recorded(retrieved):
    scores = {
        case["question"]: precision_at_k(retrieved[case["question"]],
                                         case["expected_keywords"])
        for case in EXPANSION_QUESTIONS
    }
    mean = statistics.fmean(scores.values())
    assert mean >= MIN_MEAN_PRECISION, (
        f"mean Precision@4 fell to {mean:.4f} from the recorded 0.8523: {scores}"
    )
    # No question should be scoring near zero: that means the keywords or the
    # question are wrong, not that retrieval is hard.
    weakest = {q: p for q, p in scores.items() if p < 0.5}
    assert not weakest, f"questions retrieving almost nothing relevant: {weakest}"


def test_the_expansion_set_does_not_overlap_anything(vectordb=None):
    """
    A question cannot be in the eval set twice, and cannot be both corpus-answerable
    and out-of-corpus. Also excludes the CoT exemplars' own topics, so the eval set
    stays held out from the prompt that is being evaluated.
    """
    from medbot.prompt import COT_EXAMPLES, lazy_loader

    new = [c["question"] for c in EXPANSION_QUESTIONS]
    assert len(new) == len(set(new)), "duplicate questions inside EXPANSION_QUESTIONS"
    # V1, not EVAL_QUESTIONS: since the 2026-08-03 merge the expansion IS part of
    # EVAL_QUESTIONS, so comparing against it would compare the list with itself
    # and pass vacuously. The property still worth asserting is that the two
    # halves are disjoint.
    assert not set(new) & {c["question"] for c in EVAL_QUESTIONS_V1}
    assert not set(new) & set(OVERANSWER_QUESTIONS)
    assert not set(new) & {ex["question"] for ex in COT_EXAMPLES}
    assert not set(new) & {ex["question"] for ex in lazy_loader.load_medical_examples()}


def test_every_expansion_question_was_actually_screened():
    """Selection has to be auditable — the F7 lesson, applied to this list."""
    unscreened = sorted({c["question"] for c in EXPANSION_QUESTIONS} - set(CANDIDATES))
    assert not unscreened, f"in the set without passing the screen: {unscreened}"


def test_the_rejections_are_kept_as_evidence():
    """What shows the kept 22 were screened rather than searched for."""
    rejected = set(CANDIDATES) - {c["question"] for c in EXPANSION_QUESTIONS}
    assert len(rejected) >= 10, (
        f"only {len(rejected)} rejected candidates recorded; deleting rejections "
        "loses the evidence that this was a screen"
    )


def test_the_expansion_is_merged_and_the_suite_is_46():
    """
    Replaces `test_the_expansion_is_not_merged_into_the_eval_set_yet`, which held
    the two-step sequencing: merge only in the same change as the calls that
    re-measure the suite. That happened on 2026-08-03 (330 calls), so the pin now
    points the other way — the merge must not be silently reverted, which would
    quietly shrink the denominator behind every refusal statistic.
    """
    assert len(EVAL_QUESTIONS_V1) == 24
    assert len(EXPANSION_QUESTIONS) == 22
    assert len(EVAL_QUESTIONS) == 46, (
        f"eval set is {len(EVAL_QUESTIONS)} questions, not 46: the expansion "
        "merge was reverted, and the recorded trials now over-cover the suite"
    )
    merged = [c["question"] for c in EVAL_QUESTIONS]
    assert merged[:24] == [c["question"] for c in EVAL_QUESTIONS_V1], (
        "the original 24 must stay first and in order — the pre-merge trial files "
        "are keyed by question text, and reordering breaks nothing loudly"
    )


if __name__ == "__main__":
    raise SystemExit(
        "This module uses pytest fixtures. Run:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_expansion_selection.py"
    )
