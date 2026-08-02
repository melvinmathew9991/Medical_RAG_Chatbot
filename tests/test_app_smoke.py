"""
End-to-end smoke test of the real streamlit app.

    .venv-gemini/Scripts/python.exe -m pytest tests/test_app_smoke.py -m live

Marked `live`: it loads the FAISS index, calls Gemini, and hits the external
search APIs, so pytest.ini deselects it from the default run and Sprint 5's CI
(which has no API key) will skip it.

This exists because the same AppTest script has been written from scratch and
thrown away in two consecutive sprints -- once to confirm the Gemini migration
worked, once as audit check A4 in Sprint 3. Both times the interesting thing was
not "does it start" but "does what the harness measures match what a user sees".
The eval harness calls `run_query` directly; only this test proves `app.py` wires
that same path to the screen.

Deliberately not asserted: the *content* of the answer. That is the eval
harness's job, it costs quota to judge, and a smoke test that fails when the
model rephrases something is a test nobody keeps.
"""

import os

import pytest

from medbot.config import PERSIST_DIR

pytestmark = pytest.mark.live

# The Sprint 2 question that refused 6/6 at both temperatures, and the specific
# case the Sprint 3 CoT prompt was built to fix. If the shipped app regresses on
# anything, this is the question it regresses on first.
BURSITIS = "What is bursitis?"

REASONING_LEAKS = ("Reasoning:", "**Answer:**", "Answer:")


def _requirements_met():
    if not os.getenv("GOOGLE_API_KEY"):
        return "GOOGLE_API_KEY not set"
    if not os.path.isdir(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
        return f"no vector index at {PERSIST_DIR} -- run data_processing first"
    return None


@pytest.fixture(scope="module")
def answered_app():
    """Run the app once and ask one question; module-scoped to spend one call, not four."""
    skip_reason = _requirements_met()
    if skip_reason:
        pytest.skip(skip_reason)

    from streamlit.testing.v1 import AppTest

    # Generous: cold start loads FAISS and the local fastembed model before the
    # first token, and the external-source calls are serial.
    app = AppTest.from_file("app.py", default_timeout=180)
    app.run()
    assert not app.exception, f"app raised on startup: {app.exception}"

    app.chat_input[0].set_value(BURSITIS).run()
    assert not app.exception, f"app raised while answering: {app.exception}"
    return app


def _assistant_markdown(app):
    """The assistant turn is the last markdown block the app rendered."""
    blocks = [m.value for m in app.markdown]
    assert blocks, "app rendered no markdown at all"
    return blocks[-1]


def _answer_body(app):
    """
    The model's answer with the appended blocks removed.

    Splits on the first `---` rather than on a named heading. The named-heading
    version silently stopped working in Sprint 6, when the "Retrieved from"
    block was inserted *above* "Related external sources": the split still
    succeeded, so no test failed, but the citation block was quietly being fed
    to the refusal check and the reasoning-leak check as if it were answer text.
    """
    return _assistant_markdown(app).split("\n---\n")[0]


def test_app_starts_and_answers(answered_app):
    answer = _assistant_markdown(answered_app)
    assert answer.strip(), "assistant turn was empty"
    assert BURSITIS not in answer, "assistant echoed the question instead of answering"


def test_no_reasoning_trace_reaches_the_user(answered_app):
    """
    The CoT prompt makes the model emit 'Reasoning: ... Answer: ...'. `run_query`
    strips it. This asserts the app actually routes through `run_query` -- a call
    to `chain.invoke` added directly to app.py would leak the trace and pass every
    other test in the suite.
    """
    # The appended blocks are split off: external-source article titles
    # legitimately contain these words.
    body = _answer_body(answered_app)
    for marker in REASONING_LEAKS:
        assert marker not in body, f"reasoning trace leaked {marker!r} into the answer:\n{body}"


def test_answer_is_not_a_refusal(answered_app):
    """
    Sprint 3's headline result, checked against the shipped app rather than the
    harness. Uses the same marker list as `medbot/eval/refusal_trials.py`, and the
    same first-200-character window -- a closing sentence that names a gap is a
    partial answer, not a refusal.
    """
    from medbot.eval.refusal_trials import is_refusal

    body = _answer_body(answered_app)
    assert not is_refusal(body), f"app refused the bursitis question:\n{body[:400]}"


def test_the_answer_cites_the_corpus(answered_app):
    """
    Sprint 6's citation path, end to end through the shipped app.

    `format_sources` is unit-tested against hand-built metadata, and the index
    is checked for metadata directly -- but neither proves `app.py` passes the
    chain's `source_documents` to the renderer. Dropping that argument breaks
    citations everywhere while every other test in the suite still passes.

    Asserted on the *shape* (a heading and a page-numbered PDF line), not on
    which page: which chunks retrieval returns is the eval harness's business
    and would make this fail on an unrelated index change.
    """
    answer = _assistant_markdown(answered_app)
    assert "**Retrieved from**" in answer, (
        "no citation block in the answer -- check app.py passes "
        "response['source_documents'] to format_sources"
    )
    citations = answer.split("**Retrieved from**")[1].split("\n---\n")[0]
    assert ".pdf, p. " in citations, (
        f"citation block has no page-numbered source line:\n{citations}"
    )


def test_disclaimer_is_shown(answered_app):
    """Medical-advice disclaimer is the one piece of UI copy that must not be dropped."""
    captions = " ".join(c.value for c in answered_app.caption)
    assert "not a substitute for professional medical advice" in captions


if __name__ == "__main__":
    raise SystemExit(
        "This module needs pytest fixtures. Run:\n"
        "    .venv-gemini/Scripts/python.exe -m pytest tests/test_app_smoke.py -m live"
    )
