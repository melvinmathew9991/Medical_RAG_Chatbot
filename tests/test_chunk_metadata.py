"""
Checks on the two helpers that decide what metadata reaches `index.pkl`.

Runnable with pytest, or standalone:

    .venv-gemini/Scripts/python.exe -m tests.test_chunk_metadata

`tests/test_vector_index.py` asserts the *committed* index has good metadata,
which is the check that matters most -- but it can only fail after a rebuild has
already been done and shipped. These are the same properties at the point they
are decided, so a mistake surfaces before ~15 minutes of CPU rather than after.

No index, no embedding model, no network.
"""

from types import SimpleNamespace

from medbot.data_processing import _lacks_citation_metadata, _with_citation_metadata


def _chunk(**metadata):
    return SimpleNamespace(page_content="text", metadata=metadata)


def test_windows_path_is_reduced_to_a_filename():
    """
    The reason this helper exists. `index.pkl` is committed, so an absolute path
    here would publish the builder's directory layout to the repo -- the same
    class of mistake as the prototype's hardcoded `E:/brototype/...`.
    """
    chunk = _with_citation_metadata(
        _chunk(source=r"D:\Medical_RAG_Chatbot\Medical_RAG_Chatbot\data\gale.pdf"), 0
    )
    assert chunk.metadata["source"] == "gale.pdf"


def test_posix_path_is_reduced_to_a_filename():
    """CI runs on windows-latest, but the devcontainer image is Debian."""
    chunk = _with_citation_metadata(_chunk(source="/home/x/repo/data/gale.pdf"), 0)
    assert chunk.metadata["source"] == "gale.pdf"


def test_a_bare_filename_is_left_alone():
    chunk = _with_citation_metadata(_chunk(source="gale.pdf"), 0)
    assert chunk.metadata["source"] == "gale.pdf"


def test_chunk_index_is_the_position():
    """It has to equal the FAISS row, which is assigned by enumeration order."""
    chunks = [_with_citation_metadata(_chunk(source="a.pdf"), i) for i in range(3)]
    assert [c.metadata["chunk_index"] for c in chunks] == [0, 1, 2]


def test_loader_metadata_survives():
    """`page` is what the citation renders; losing it silently drops page numbers."""
    chunk = _with_citation_metadata(_chunk(source="a.pdf", page=41, extra="kept"), 7)
    assert chunk.metadata == {
        "source": "a.pdf",
        "page": 41,
        "extra": "kept",
        "chunk_index": 7,
    }


def test_missing_source_is_not_invented():
    """Better a chunk with no citation than one attributed to the wrong file."""
    chunk = _with_citation_metadata(_chunk(page=3), 1)
    assert "source" not in chunk.metadata
    assert chunk.metadata["chunk_index"] == 1


def test_the_callers_dict_is_not_mutated():
    """
    The splitter can hand the same metadata dict to several chunks. Writing
    `chunk_index` in place would leave every one of them holding the last value
    written -- and `chunk_index` would then disagree with the FAISS row, which
    is precisely what `test_chunk_index_matches_faiss_row_order` exists to catch,
    but only after a full rebuild.
    """
    shared = {"source": "a.pdf", "page": 2}
    first = _with_citation_metadata(SimpleNamespace(page_content="a", metadata=shared), 0)
    second = _with_citation_metadata(SimpleNamespace(page_content="b", metadata=shared), 1)
    assert shared == {"source": "a.pdf", "page": 2}, "helper mutated the caller's dict"
    assert first.metadata["chunk_index"] == 0
    assert second.metadata["chunk_index"] == 1


def _fake_db(ntotal, doc):
    return SimpleNamespace(
        index=SimpleNamespace(ntotal=ntotal),
        index_to_docstore_id={0: "id-0"},
        docstore=SimpleNamespace(search=lambda _id: doc),
    )


def test_a_pre_sprint6_index_is_detected_as_stale():
    """1225 chunks, every one with `metadata == {}` -- the shape actually on disk."""
    assert _lacks_citation_metadata(_fake_db(1225, _chunk())) is True


def test_a_current_index_is_not_stale():
    """A false positive here costs a needless ~15-minute rebuild on every startup."""
    db = _fake_db(1225, _chunk(source="a.pdf", page=0, chunk_index=0))
    assert _lacks_citation_metadata(db) is False


def test_staleness_is_keyed_on_chunk_index_not_source():
    """
    A `.txt` chunk legitimately has no `source`-bearing page metadata but is
    still current. Keying on `source` would rebuild a perfectly good index.
    """
    db = _fake_db(1225, _chunk(chunk_index=0))
    assert _lacks_citation_metadata(db) is False


def test_an_empty_index_is_not_stale():
    """
    Empty is not stale, it is unbuilt. Reporting it stale would set vectordb to
    None and discard a partial build the resume logic could have continued --
    the same defect Sprint 4 fixed in the trial runner.
    """
    assert _lacks_citation_metadata(_fake_db(0, None)) is False


def test_a_missing_docstore_id_is_treated_as_stale():
    """InMemoryDocstore.search returns an error *string*, which has no .metadata."""
    db = _fake_db(1225, "ID id-0 not found.")
    assert _lacks_citation_metadata(db) is True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
