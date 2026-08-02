"""
Tests for the network guard in `conftest.py`.

Sprint 4 built the guard and probe-tested it by hand -- but with a plain
function-scoped test, which is the one case that always worked. The case that
did not was fixture setup at module or session scope, and it went unnoticed for
three sprints because nothing here committed the probe. So the guard now has
tests, and the first of them is the one that would have caught it.

Every address used below is 192.0.2.0/24 (RFC 5737 TEST-NET-1), reserved for
documentation and never routable. If the guard ever regresses, these tests fail
on a timeout rather than generating real traffic.
"""

import socket

import pytest

from tests.conftest import guard_is_installed

BLACKHOLE = ("192.0.2.1", 80)


def _attempt(address=BLACKHOLE):
    """Returns the RuntimeError the guard raised, or a string saying what escaped."""
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        sock.connect(address)
        return "connected -- the guard was not installed"
    except RuntimeError as exc:
        return exc
    except OSError:
        return "reached the real socket layer -- the guard was not installed"
    finally:
        sock.close()


@pytest.fixture(scope="module")
def attempt_from_module_scoped_fixture():
    """
    The regression case.

    Module-scoped fixtures are instantiated before function-scoped autouse
    fixtures, so while the guard lived in one it was not yet installed here.
    That is how `test_expansion_selection.py` and
    `test_out_of_corpus_selection.py` downloaded a 130MB model on a cold cache
    while their own comments said the guard would prevent exactly that.
    """
    return _attempt()


def test_module_scoped_fixture_setup_is_guarded(attempt_from_module_scoped_fixture):
    outcome = attempt_from_module_scoped_fixture
    assert isinstance(outcome, RuntimeError), (
        f"a module-scoped fixture reached the network: {outcome}. The guard must be "
        "installed before fixtures of any scope -- see the hooks in conftest.py."
    )
    assert "192.0.2.1" in str(outcome), "the message must name the host that was refused"


def test_the_guard_is_active_during_an_ordinary_test():
    assert guard_is_installed()


def test_connect_is_blocked_and_names_the_test():
    outcome = _attempt()
    assert isinstance(outcome, RuntimeError)
    assert "test_connect_is_blocked_and_names_the_test" in str(outcome), (
        "the error must identify the offending test; a bare 'connection refused' "
        "is the debugging problem this guard exists to remove"
    )


def test_connect_ex_is_blocked_too():
    """`connect_ex` returns an errno instead of raising, so it needs its own patch."""
    sock = socket.socket()
    sock.settimeout(0.25)
    try:
        with pytest.raises(RuntimeError, match="connect_ex"):
            sock.connect_ex(BLACKHOLE)
    finally:
        sock.close()


def test_loopback_is_still_allowed():
    """streamlit's AppTest runs a server in-process; blocking loopback breaks it."""
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.socket()
    client.settimeout(1)
    try:
        client.connect(server.getsockname())  # must not raise
    finally:
        client.close()
        server.close()


@pytest.mark.live
def test_live_marked_tests_are_exempt():
    """
    Asserts the exemption without using it -- this test opens no socket, so it
    is safe to collect anywhere, and it still fails if `live` stops exempting.
    """
    assert not guard_is_installed()
