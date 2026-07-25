"""
Checks on `format_external_results`, the renderer for the external-sources block.

Runnable with pytest, or standalone:

    .venv-gemini/Scripts/python.exe -m tests.test_format_external_results

Pure function, no network, and every one of its outputs is shown to the user
verbatim as markdown -- which is what makes the `<span>` stripping load-bearing.
Wikipedia's search API returns snippets with `<span class="searchmatch">` around
the matched terms; streamlit renders markdown with HTML passed through, so an
unstripped snippet shows up styled by Wikipedia's own CSS classes, or with raw
tags visible where those classes do not exist.
"""

from types import SimpleNamespace

from medbot.query_handler import format_external_results


def _pubmed(*titles):
    """PubMedLoader yields Documents; only .metadata['Title'] is read."""
    return [SimpleNamespace(metadata={"Title": t} if t is not None else {}) for t in titles]


def test_returns_none_when_every_source_is_empty():
    """None, not "" -- the caller uses it to decide whether to render the block at all."""
    assert format_external_results({}) is None
    assert format_external_results({"pubmed": [], "wikipedia": [], "serpapi": []}) is None


def test_returns_none_when_sources_are_present_but_none_valued():
    """search_external_sources returns {} on failure, but a partial dict is possible."""
    assert format_external_results({"pubmed": None, "wikipedia": None, "serpapi": None}) is None


def test_pubmed_titles_are_rendered_as_a_bullet_list():
    out = format_external_results({"pubmed": _pubmed("Bursitis review", "Shoulder pain")})
    assert out == "**PubMed**\n- Bursitis review\n- Shoulder pain"


def test_pubmed_missing_title_falls_back_to_untitled():
    out = format_external_results({"pubmed": _pubmed(None)})
    assert out == "**PubMed**\n- Untitled"


def test_wikipedia_span_tags_are_stripped():
    """The reason this function exists rather than joining snippets inline."""
    snippet = 'Bursitis is <span class="searchmatch">inflammation</span> of a bursa.'
    out = format_external_results({"wikipedia": [snippet]})
    assert out == "**Wikipedia**\n- Bursitis is inflammation of a bursa."
    assert "<span" not in out and "</span>" not in out


def test_wikipedia_strips_multiple_and_bare_span_tags():
    snippet = "<span>a</span> and <span class='x'>b</span>"
    out = format_external_results({"wikipedia": [snippet]})
    assert out == "**Wikipedia**\n- a and b"


def test_wikipedia_leaves_other_markup_alone():
    """Only span is stripped; anything else is Wikipedia's problem, not this function's."""
    out = format_external_results({"wikipedia": ["a <b>bold</b> claim"]})
    assert out == "**Wikipedia**\n- a <b>bold</b> claim"


def test_serpapi_renders_markdown_links():
    results = [{"title": "Bursitis", "link": "https://example.org/b"}]
    out = format_external_results({"serpapi": results})
    assert out == "**Google (via SerpAPI)**\n- [Bursitis](https://example.org/b)"


def test_serpapi_missing_fields_fall_back():
    out = format_external_results({"serpapi": [{}]})
    assert out == "**Google (via SerpAPI)**\n- [Untitled]()"


def test_max_per_source_caps_each_section_independently():
    out = format_external_results(
        {
            "pubmed": _pubmed(*[f"p{i}" for i in range(10)]),
            "wikipedia": [f"w{i}" for i in range(10)],
        },
        max_per_source=2,
    )
    assert out.count("\n- ") == 4, out
    assert "p0" in out and "p1" in out and "p2" not in out
    assert "w0" in out and "w1" in out and "w2" not in out


def test_default_cap_is_five():
    out = format_external_results({"wikipedia": [f"w{i}" for i in range(10)]})
    assert out.count("\n- ") == 5
    assert "w4" in out and "w5" not in out


def test_only_populated_sources_get_headers():
    """An empty PubMed result must not leave a dangling '**PubMed**' heading."""
    out = format_external_results({"pubmed": [], "wikipedia": ["snippet"], "serpapi": []})
    assert out == "**Wikipedia**\n- snippet"
    assert "PubMed" not in out and "SerpAPI" not in out


def test_sections_are_blank_line_separated_and_ordered():
    out = format_external_results(
        {
            "pubmed": _pubmed("p"),
            "wikipedia": ["w"],
            "serpapi": [{"title": "s", "link": "https://e.org"}],
        }
    )
    assert out == (
        "**PubMed**\n- p\n\n"
        "**Wikipedia**\n- w\n\n"
        "**Google (via SerpAPI)**\n- [s](https://e.org)"
    )


def test_unknown_keys_are_ignored():
    assert format_external_results({"arxiv": ["ignored"]}) is None


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
