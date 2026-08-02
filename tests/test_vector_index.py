"""
The committed FAISS index is present and intact.

Every other test that touches retrieval loads the index through a module-scoped
fixture that calls `pytest.skip()` when `index.faiss` is missing. That skip is
good ergonomics -- a contributor mid-rebuild should not get twelve confusing
failures -- but on its own it is a hole: a missing index makes CI *green* while
silently testing nothing. That is the same shape as the Sprint 4 audit finding
where `test_out_of_corpus_gate_is_fully_armed` was written to skip and then fell
through asserting nothing.

So these tests exist to fail. Nothing here skips, and nothing here needs the
embedding model or the network -- the index is read directly with `faiss`, which
keeps this fast and makes it independent of the fastembed cache being warm.

Two incidents in this repo's history are what the assertions are pinned to:

  * `core.autocrlf=true` with no `.gitattributes` corrupted `index.faiss` on a
    git rename. `faiss.read_index` raises on a mangled file, so any recurrence
    fails here first, naming the file, instead of surfacing as a confusing
    retrieval error.
  * The docs once claimed a complete 1223/1223 rebuild while only 900 of 1225
    chunks were actually embedded on disk. A truncated index is therefore a
    tested condition, not a trusted one.
"""

import os

import faiss

from medbot.config import PERSIST_DIR

# 1225 chunks at 384 dimensions (BAAI/bge-small-en-v1.5). Both are pinned rather
# than derived: deriving the count from the corpus would re-implement the
# chunker and agree with it by construction, and deriving the dimensionality
# would mean loading the embedding model this module deliberately avoids.
# If a re-chunk or a model change makes these wrong, that is a deliberate act and
# updating the numbers here is part of it -- see MIGRATION_STATUS.md.
EXPECTED_CHUNKS = 1225
EXPECTED_DIMENSIONS = 384

INDEX_PATH = os.path.join(PERSIST_DIR, "index.faiss")
DOCSTORE_PATH = os.path.join(PERSIST_DIR, "index.pkl")


def _read_index():
    """
    Read the index, failing with a readable message when the file is simply
    absent. Without this, `faiss.read_index` raises a swig-level RuntimeError
    from inside site-packages, which buries the one fact that matters.
    """
    assert os.path.isfile(INDEX_PATH), (
        f"no FAISS index at {INDEX_PATH} -- see test_the_index_files_are_present"
    )
    return faiss.read_index(INDEX_PATH)


def test_the_index_files_are_present():
    """
    Deliberately not a skip.

    If this fails in CI, the index did not survive checkout -- check that
    `.gitattributes` still marks `*.faiss` and `*.pkl` as binary.
    """
    assert os.path.isfile(INDEX_PATH), (
        f"no FAISS index at {INDEX_PATH}. Every retrieval test skips without it, "
        "so this must fail rather than let the suite pass on nothing."
    )
    assert os.path.isfile(DOCSTORE_PATH), (
        f"no docstore at {DOCSTORE_PATH}. index.faiss holds the vectors; the "
        "chunk text lives here, and retrieval needs both."
    )


def test_the_index_holds_every_chunk():
    """A partially-embedded index retrieves plausible-looking wrong answers."""
    index = _read_index()
    assert index.ntotal == EXPECTED_CHUNKS, (
        f"index holds {index.ntotal} vectors, expected {EXPECTED_CHUNKS}. A short "
        "index is the failure this project has already had once: retrieval keeps "
        "working, it just cannot find what was never embedded."
    )


def test_the_index_matches_the_embedding_model():
    """
    Dimensionality is the cheapest way to catch an index built by a different
    model -- vectors from a swapped `LOCAL_EMBEDDING_MODEL` are not comparable
    to these, and the failure is otherwise silent and quality-shaped.
    """
    index = _read_index()
    assert index.d == EXPECTED_DIMENSIONS, (
        f"index is {index.d}-dimensional, expected {EXPECTED_DIMENSIONS}. The "
        "index and LOCAL_EMBEDDING_MODEL have diverged; one of them was changed "
        "without the other being rebuilt."
    )
