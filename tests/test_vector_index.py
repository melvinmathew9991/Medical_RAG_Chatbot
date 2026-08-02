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
import pickle

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


def _read_docstore():
    """
    Return the persisted chunks in FAISS row order.

    Unpickles `index.pkl` directly rather than going through
    `FAISS.load_local`, which would construct an embeddings object and, on a
    cold cache, download the model -- exactly what this module avoids.
    """
    assert os.path.isfile(DOCSTORE_PATH), (
        f"no docstore at {DOCSTORE_PATH} -- see test_the_index_files_are_present"
    )
    with open(DOCSTORE_PATH, "rb") as fh:
        docstore, index_to_id = pickle.load(fh)
    return [docstore.search(index_to_id[i]) for i in range(len(index_to_id))]


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


def test_every_chunk_can_be_cited():
    """
    Sprint 6's whole premise. `process_documents` returned bare strings until
    then, so all 1225 chunks sat in the docstore with `metadata == {}` and no
    answer could say where it came from.

    The failure mode this guards is silent: a rebuild that drops metadata
    retrieves exactly as well as one that keeps it, and the only symptom is a
    missing citation block that `format_sources` renders as nothing at all,
    by design, so an old index does not crash.
    """
    docs = _read_docstore()
    missing = [i for i, d in enumerate(docs) if not d.metadata.get("source")]
    assert not missing, (
        f"{len(missing)} of {len(docs)} chunks have no source metadata "
        f"(rows {missing[:5]}...). The index predates Sprint 6, or a rebuild "
        "dropped metadata -- see medbot/data_processing.py::_with_citation_metadata."
    )


def test_no_chunk_carries_a_filesystem_path():
    """
    `index.pkl` is a committed artefact, so a full path in it publishes whoever
    built it. Not hypothetical: the original prototype shipped a hardcoded
    `E:/brototype/Langchain/...` in config.py, and removing it was part of the
    2026-07-06 restructure. The loaders hand back absolute paths, so this stays
    correct only for as long as something reduces them to a basename.
    """
    offenders = sorted(
        {d.metadata["source"] for d in _read_docstore()
         if any(c in d.metadata.get("source", "") for c in ("/", "\\", ":"))}
    )
    assert not offenders, f"source metadata contains filesystem paths: {offenders}"


def test_chunk_index_matches_faiss_row_order():
    """
    `chunk_index` is only useful for debugging retrieval if it means what it
    says. It is assigned from split order and FAISS rows are appended in that
    same order, so any divergence means the index was assembled out of order --
    which would also mean the row a similarity search returns is not the chunk
    the metadata describes.
    """
    docs = _read_docstore()
    wrong = [(i, d.metadata.get("chunk_index")) for i, d in enumerate(docs)
             if d.metadata.get("chunk_index") != i]
    assert not wrong, f"chunk_index diverges from row order at {wrong[:5]}"


def test_pdf_chunks_carry_a_usable_page_number():
    """
    `format_sources` renders `page + 1`, so a non-int page silently drops the
    citation to a bare filename and a negative one would cite page 0.
    """
    docs = _read_docstore()
    pdf_docs = [d for d in docs if d.metadata["source"].lower().endswith(".pdf")]
    assert pdf_docs, "no PDF-sourced chunks; the corpus is a single PDF"
    bad = [(d.metadata["chunk_index"], d.metadata.get("page")) for d in pdf_docs
           if not isinstance(d.metadata.get("page"), int) or d.metadata["page"] < 0]
    assert not bad, f"chunks with an unusable page number: {bad[:5]}"


def test_the_docstore_and_the_vectors_agree_on_length():
    """
    The two files are written together but read separately. A mismatch means a
    lookup by FAISS row can miss or point at the wrong chunk, and retrieval
    keeps working while citing the wrong page.
    """
    assert len(_read_docstore()) == _read_index().ntotal


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
