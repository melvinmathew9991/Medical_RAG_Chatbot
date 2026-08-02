"""
Checks on `format_sources`, the renderer for the "Retrieved from" block.

Runnable with pytest, or standalone:

    .venv-gemini/Scripts/python.exe -m tests.test_format_sources

Pure function, no network, no index. Its output goes to the user verbatim as
markdown under a medical answer, which is what makes the small things here worth
pinning:

  * PyPDFLoader numbers pages from 0 and every PDF reader numbers them from 1.
    An off-by-one in a citation is not a cosmetic bug -- it points a user
    checking the claim at the wrong page, and the failure is invisible unless
    someone opens the PDF.
  * `chunk_size=3000` with `chunk_overlap=300` regularly puts two retrieved
    chunks on one page. Listing that page twice reads as two independent
    corroborating sources.
  * An index built before Sprint 6 has empty metadata on every chunk. Those must
    produce no citation block rather than an exception, because the answer above
    the block is still perfectly good.
"""

from types import SimpleNamespace

from medbot.query_handler import format_sources

PDF = "71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf"


def _doc(source=PDF, page=0, **extra):
    """A stand-in for a langchain Document; only .metadata is read."""
    metadata = {}
    if source is not None:
        metadata["source"] = source
    if page is not None:
        metadata["page"] = page
    metadata.update(extra)
    return SimpleNamespace(metadata=metadata)


def test_returns_none_for_no_documents():
    """None, not "" -- the caller uses it to decide whether to render the block."""
    assert format_sources(None) is None
    assert format_sources([]) is None


def test_returns_none_when_metadata_is_absent():
    """
    The pre-Sprint-6 index shape: 1225 chunks, every one with `metadata == {}`.
    Someone running against an old local `vectorstore/` must still get answers.
    """
    assert format_sources([SimpleNamespace(metadata={}) for _ in range(4)]) is None
    assert format_sources([SimpleNamespace(metadata=None)]) is None


def test_page_numbering_is_one_based_for_the_reader():
    """The loader's page 0 is the PDF reader's page 1. This is the whole point."""
    assert format_sources([_doc(page=0)]) == f"- {PDF}, p. 1"
    assert format_sources([_doc(page=123)]) == f"- {PDF}, p. 124"


def test_source_without_a_page_renders_bare():
    """.txt documents carry a source but no page; no trailing 'p. None'."""
    assert format_sources([_doc(page=None)]) == f"- {PDF}"


def test_non_integer_page_is_not_arithmetic():
    """A string page must not raise, and must not be concatenated as '12' + 1."""
    assert format_sources([_doc(page="12")]) == f"- {PDF}"


def test_repeated_page_is_listed_once():
    """Two chunks off one page are one source, not two corroborating ones."""
    out = format_sources([_doc(page=40), _doc(page=40), _doc(page=40)])
    assert out == f"- {PDF}, p. 41"


def test_distinct_pages_are_all_kept():
    out = format_sources([_doc(page=40), _doc(page=41)])
    assert out == f"- {PDF}, p. 41\n- {PDF}, p. 42"


def test_retrieval_order_is_preserved():
    """
    The list is ranked by similarity, and the top chunk is the one most likely to
    be what the answer leaned on. Sorting for tidiness would discard that.
    """
    out = format_sources([_doc(page=99), _doc(page=2), _doc(page=50)])
    assert out == f"- {PDF}, p. 100\n- {PDF}, p. 3\n- {PDF}, p. 51"


def test_dedupe_keeps_the_first_occurrence_position():
    out = format_sources([_doc(page=9), _doc(page=1), _doc(page=9)])
    assert out == f"- {PDF}, p. 10\n- {PDF}, p. 2"


def test_different_sources_on_the_same_page_are_distinct():
    out = format_sources([_doc(source="a.pdf", page=0), _doc(source="b.pdf", page=0)])
    assert out == "- a.pdf, p. 1\n- b.pdf, p. 1"


def test_chunks_without_a_source_are_skipped_not_fatal():
    """A partially-migrated index must degrade to the citations it does have."""
    out = format_sources([SimpleNamespace(metadata={"chunk_index": 7}), _doc(page=3)])
    assert out == f"- {PDF}, p. 4"


def test_max_sources_caps_the_list():
    docs = [_doc(page=p) for p in range(10)]
    assert format_sources(docs, max_sources=2) == f"- {PDF}, p. 1\n- {PDF}, p. 2"


def test_default_cap_matches_the_retriever_k():
    """
    The retriever returns 4 chunks by default, so the default cap only bites if
    that changes. If a future k > 4 should show more, change both together.
    """
    docs = [_doc(page=p) for p in range(10)]
    assert format_sources(docs).count("\n- ") == 3  # 4 lines
    assert "p. 4" in format_sources(docs) and "p. 5" not in format_sources(docs)


def test_cap_applies_after_deduplication():
    """
    Four chunks off two pages must yield two citations, not two lines consumed by
    duplicates. Capping first would silently drop the second real page.
    """
    docs = [_doc(page=1), _doc(page=1), _doc(page=1), _doc(page=8)]
    assert format_sources(docs, max_sources=2) == f"- {PDF}, p. 2\n- {PDF}, p. 9"


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
