"""
Shared pytest configuration.

The one thing here earns its place: a network guard. While writing
`test_external_search.py` a mock was pointed at the wrong module -- patching
`medbot.external_search.search_serpapi` when `query_handler` had already bound
the name into its own namespace via `from ... import ...`. The test still
passed its own assertions on a later run, but it was reaching PubMed, Wikipedia
and SerpAPI for real: ~13 seconds, and results that depended on the internet.

That failure is silent by construction -- a mock that misses looks exactly like
a mock that works, only slower. So the default run refuses outbound sockets and
names the test that tried. Tests that genuinely need the network must say so
with `@pytest.mark.live`, which pytest.ini deselects by default.

Implemented as `pytest_runtest_setup`/`teardown` hooks rather than the autouse
fixture it was in Sprint 4. That was not a style change. An autouse fixture is
function-scoped, and pytest instantiates higher-scoped fixtures *first*, so
anything set up in a `scope="module"` or `scope="session"` fixture ran before
the guard existed. `test_expansion_selection.py` and
`test_out_of_corpus_selection.py` build their FAISS retriever in exactly such a
fixture, and both files carry a comment asserting the guard would stop an
uncached fastembed model from downloading. It would not: on a cold cache they
silently pulled ~130MB from HuggingFace and passed. Found in Sprint 5 while
proving whether CI needed a cache-warming step -- it does, and now that is true
for the documented reason. The hooks run before any fixture of any scope, and
`tests/test_network_guard.py` pins that.
"""

import socket

import pytest

_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_local(address):
    """Allow loopback: streamlit's AppTest and similar run a server in-process."""
    if not isinstance(address, tuple) or not address:
        return False
    host = address[0]
    return host in ("127.0.0.1", "::1", "localhost", "0.0.0.0", "")


def _install(test_id):
    """
    Refuse outbound sockets, naming the test that tried.

    Raises rather than returning a canned response on purpose: a test that wanted
    the network and silently got nothing is the same debugging problem in a new
    place. The message says which test and which host so the fix is obvious.
    """

    def guard(self, address, *args, **kwargs):
        if _is_local(address):
            return _real_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"{test_id} tried to reach {address!r}. Offline tests must mock the "
            "network -- check the patch target is the module that *calls* the "
            "function, not the one that defines it. If the call is intentional, "
            "mark the test @pytest.mark.live. If this names huggingface.co, the "
            "local fastembed model cache is cold -- see README, 'Running the tests'."
        )

    def guard_ex(self, address, *args, **kwargs):
        if _is_local(address):
            return _real_connect_ex(self, address, *args, **kwargs)
        raise RuntimeError(f"{test_id} tried to reach {address!r} via connect_ex.")

    socket.socket.connect = guard
    socket.socket.connect_ex = guard_ex


def _restore():
    socket.socket.connect = _real_connect
    socket.socket.connect_ex = _real_connect_ex


def guard_is_installed():
    """Exposed for `test_network_guard.py`, which has no other way to see this."""
    return socket.socket.connect is not _real_connect


# tryfirst: this hook must run before `_pytest.runner`'s implementation, which is
# what actually instantiates the item's fixtures. That ordering is the whole fix.
@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    if item.get_closest_marker("live"):
        _restore()
        return
    _install(item.nodeid)


# trylast, symmetrically: module- and session-scoped fixtures are torn down by
# the default teardown implementation, and those finalizers stay guarded too.
@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item, nextitem):
    _restore()
