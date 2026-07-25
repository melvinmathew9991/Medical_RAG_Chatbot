"""
Checks on `medbot.external_search`, with the network mocked out.

Runnable with pytest, or standalone:

    .venv-gemini/Scripts/python.exe -m tests.test_external_search

These three functions are the app's only outbound calls, and each one is written
to fail soft -- a dead source should degrade the answer, not break the page. The
contract worth pinning is therefore mostly negative: on a malformed payload or a
raising client, return `[]` rather than propagating.

`search_serpapi` used to catch only `RequestException` while the other two caught
`Exception`, which cost the user all three sources whenever the serpapi client
raised one of its own types. Sprint 4 widened it; the two tests that pin the fix
are `test_serpapi_catches_non_request_exceptions_too` and
`test_a_serpapi_failure_no_longer_costs_the_other_two_sources`.

The `lru_cache` decorators on `cached_search_pubmed` and `fetch_wikipedia_data`
make these functions stateful across calls, so every test that touches them
clears the cache first. Without that, test order changes the results.
"""

from unittest import mock

from requests.exceptions import RequestException

import medbot.external_search as es


def _clear_caches():
    es.cached_search_pubmed.cache_clear()
    es.fetch_wikipedia_data.cache_clear()


# --- Wikipedia ------------------------------------------------------------

WIKI_PAYLOAD = {
    "query": {
        "search": [
            {"snippet": "Bursitis is <span class=\"searchmatch\">inflammation</span> of a bursa."},
            {"snippet": "Treatment includes rest and ice."},
        ]
    }
}


def test_wikipedia_extracts_snippets_in_order():
    _clear_caches()
    with mock.patch.object(es, "fetch_wikipedia_data", return_value=WIKI_PAYLOAD):
        assert es.search_wikipedia("bursitis") == [
            "Bursitis is <span class=\"searchmatch\">inflammation</span> of a bursa.",
            "Treatment includes rest and ice.",
        ]


def test_wikipedia_returns_empty_on_malformed_payload():
    """Wikipedia answers 200 with an `error` body for some queries -- no 'query' key."""
    _clear_caches()
    with mock.patch.object(es, "fetch_wikipedia_data", return_value={"error": "bad"}):
        assert es.search_wikipedia("bursitis") == []


def test_wikipedia_returns_empty_when_the_fetch_raises():
    _clear_caches()
    with mock.patch.object(es, "fetch_wikipedia_data", side_effect=RequestException("down")):
        assert es.search_wikipedia("bursitis") == []


def test_wikipedia_returns_empty_on_no_results():
    _clear_caches()
    with mock.patch.object(es, "fetch_wikipedia_data", return_value={"query": {"search": []}}):
        assert es.search_wikipedia("zzzz") == []


# --- SerpAPI --------------------------------------------------------------

def _serpapi_returning(payload):
    """Patch the GoogleSearch class so .get_dict() yields `payload`."""
    client = mock.Mock()
    client.get_dict.return_value = payload
    return mock.patch.object(es, "GoogleSearch", return_value=client)


def test_serpapi_returns_organic_results():
    results = [{"title": "Bursitis", "link": "https://example.org/bursitis"}]
    with _serpapi_returning({"organic_results": results}):
        assert es.search_serpapi("bursitis") == results


def test_serpapi_returns_empty_when_organic_results_absent():
    """SerpAPI omits the key entirely on a quota error, rather than sending an empty list."""
    with _serpapi_returning({"error": "quota exceeded"}):
        assert es.search_serpapi("bursitis") == []


def test_serpapi_returns_empty_on_request_exception():
    with mock.patch.object(es, "GoogleSearch", side_effect=RequestException("timeout")):
        assert es.search_serpapi("bursitis") == []


def test_serpapi_catches_non_request_exceptions_too():
    """
    Regression test for the Sprint 4 bug fix.

    This caught only `RequestException` until Sprint 4, so anything else -- an
    auth or quota error raised as the serpapi client's own type, a KeyError --
    escaped the function. See the test below for what that cost.
    """
    with mock.patch.object(es, "GoogleSearch", side_effect=ValueError("bad api key")):
        assert es.search_serpapi("bursitis") == []


def test_a_serpapi_failure_no_longer_costs_the_other_two_sources():
    """
    The user-visible reason the fix above matters.

    `search_external_sources` wraps all three calls in one blanket `except` that
    returns {}, so a single escaping exception discarded PubMed and Wikipedia as
    well -- a bad SerpAPI key silently emptied the whole external-sources block.
    Now that each source swallows its own failures, the other two survive.

    Patched on `medbot.query_handler`, not on `medbot.external_search`:
    query_handler does `from medbot.external_search import search_pubmed, ...`,
    which binds the functions into its own namespace at import time, so patching
    the source module does not intercept them -- it just lets the real network
    calls through. That mistake made an earlier version of this test hit the live
    APIs while appearing to pass; see tests/conftest.py.
    """
    import medbot.query_handler as qh

    _clear_caches()
    with mock.patch.object(qh, "search_pubmed", return_value=["doc"]), \
         mock.patch.object(qh, "search_wikipedia", return_value=["snippet"]), \
         mock.patch.object(qh, "search_serpapi", side_effect=ValueError("bad api key")):
        assert qh.search_external_sources("bursitis") == {}, (
            "search_external_sources still collapses on an escaping exception; "
            "the fix belongs in the individual source function"
        )

    # And with the real (now-fixed) search_serpapi in the chain, the other two survive.
    with mock.patch.object(qh, "search_pubmed", return_value=["doc"]), \
         mock.patch.object(qh, "search_wikipedia", return_value=["snippet"]), \
         mock.patch.object(es, "GoogleSearch", side_effect=ValueError("bad api key")):
        assert qh.search_external_sources("bursitis") == {
            "pubmed": ["doc"],
            "wikipedia": ["snippet"],
            "serpapi": [],
        }


def test_external_sources_are_collected_when_all_three_succeed():
    import medbot.query_handler as qh

    _clear_caches()
    with mock.patch.object(qh, "search_pubmed", return_value=["doc"]), \
         mock.patch.object(qh, "search_wikipedia", return_value=["snippet"]), \
         mock.patch.object(qh, "search_serpapi", return_value=[{"title": "t"}]):
        assert qh.search_external_sources("bursitis") == {
            "pubmed": ["doc"],
            "wikipedia": ["snippet"],
            "serpapi": [{"title": "t"}],
        }


# --- PubMed ---------------------------------------------------------------

def test_pubmed_returns_loaded_documents():
    _clear_caches()
    loader = mock.Mock()
    loader.load.return_value = ["doc1", "doc2"]
    with mock.patch.object(es, "PubMedLoader", return_value=loader):
        assert es.search_pubmed("bursitis") == ["doc1", "doc2"]


def test_pubmed_returns_empty_when_the_loader_raises():
    """PubMed's endpoint 500s and rate-limits often enough that this is the common path."""
    _clear_caches()
    with mock.patch.object(es, "PubMedLoader", side_effect=Exception("E-utilities 429")):
        assert es.search_pubmed("bursitis") == []


def test_pubmed_result_is_cached_per_query():
    """
    The lru_cache is load-bearing: search_external_sources is called on every
    turn, and PubMed is the slowest of the three sources.
    """
    _clear_caches()
    loader = mock.Mock()
    loader.load.return_value = ["doc"]
    with mock.patch.object(es, "PubMedLoader", return_value=loader) as ctor:
        es.search_pubmed("bursitis")
        es.search_pubmed("bursitis")
        assert ctor.call_count == 1, "repeated query should hit the cache"
        es.search_pubmed("gout")
        assert ctor.call_count == 2, "a different query should miss the cache"


def test_pubmed_caches_the_empty_failure_result():
    """
    Characterisation, not endorsement: a transient PubMed outage is cached for the
    rest of the process, so a query that failed once returns [] until restart even
    after the endpoint recovers.
    """
    _clear_caches()
    with mock.patch.object(es, "PubMedLoader", side_effect=Exception("429")):
        assert es.search_pubmed("bursitis") == []

    loader = mock.Mock()
    loader.load.return_value = ["doc"]
    with mock.patch.object(es, "PubMedLoader", return_value=loader) as ctor:
        assert es.search_pubmed("bursitis") == []
        assert ctor.call_count == 0, "recovered endpoint is never retried"


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
