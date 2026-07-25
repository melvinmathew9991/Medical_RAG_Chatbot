"""
Shared pytest fixtures.

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


@pytest.fixture(autouse=True)
def no_network(request):
    """
    Fail any non-`live` test that opens an outbound socket.

    Raises rather than returning a canned response on purpose: a test that wanted
    the network and silently got nothing is the same debugging problem in a new
    place. The message says which test and which host so the fix is obvious.
    """
    if request.node.get_closest_marker("live"):
        yield
        return

    test_id = request.node.nodeid

    def guard(self, address, *args, **kwargs):
        if _is_local(address):
            return _real_connect(self, address, *args, **kwargs)
        raise RuntimeError(
            f"{test_id} tried to reach {address!r}. Offline tests must mock the "
            "network -- check the patch target is the module that *calls* the "
            "function, not the one that defines it. If the call is intentional, "
            "mark the test @pytest.mark.live."
        )

    def guard_ex(self, address, *args, **kwargs):
        if _is_local(address):
            return _real_connect_ex(self, address, *args, **kwargs)
        raise RuntimeError(f"{test_id} tried to reach {address!r} via connect_ex.")

    socket.socket.connect = guard
    socket.socket.connect_ex = guard_ex
    try:
        yield
    finally:
        socket.socket.connect = _real_connect
        socket.socket.connect_ex = _real_connect_ex
