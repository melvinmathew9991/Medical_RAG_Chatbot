"""
Checks on the prompt variants and the reasoning-trace stripper.

Runnable today without pytest, which is not yet a project dependency:

    .venv-gemini/Scripts/python.exe -m tests.test_prompt_variants

Written as `test_*` functions so Sprint 4 can point pytest at this directory
without rewriting anything.

Why this file exists at all, given testing is Sprint 4's remit: `strip_reasoning`
had a real bug (it left the trailing `**` of "**Answer:**" in user-facing text),
and the validity of Sprint 3's whole A/B rests on the `cot` prompt still being the
string that was measured. Both are cheap to pin down and expensive to get wrong.
"""

import hashlib

from medbot.prompt import (
    COT_DISCLAIMER,
    CONTEXT_JUDGEMENT_GUIDANCE,
    INSTRUCTION_ONLY_DISCLAIMER,
    PROMPT_VARIANTS,
    build_context_prompt,
    emits_reasoning,
)
from medbot.query_handler import strip_reasoning

# sha256 (first 16 hex chars) of the rendered `cot` prompt as evaluated in the
# Sprint 3 A/B run (medbot/eval/results_cot.json). If this fails, the shipped
# prompt has drifted from the one the recorded numbers describe. That is not
# automatically a bug - but the fix is to re-run the A/B and update this hash and
# the results together, never to update the hash alone.
#
# History:
#   17942003d241f2de - initial Sprint 3 A/B
#   50d24c4858744560 - post-audit fix round: added the "do not join two separate
#                      statements into a cause-and-effect chain" clause (the
#                      bedsores over-linking regression) and trimmed the bulimia
#                      symptom list in the anorexia exemplar (audit F5). cot arm
#                      re-run in full against this string.
COT_PROMPT_SHA256 = "50d24c4858744560"


def _render(variant):
    return build_context_prompt("What is bursitis?", variant=variant).format(
        context="CTX", question="Q"
    )


def test_guidance_text_is_shared_verbatim_across_arms():
    """The ablation only separates the two factors if this wording is identical."""
    assert CONTEXT_JUDGEMENT_GUIDANCE in COT_DISCLAIMER
    assert CONTEXT_JUDGEMENT_GUIDANCE in INSTRUCTION_ONLY_DISCLAIMER


def test_cot_prompt_matches_the_string_that_was_evaluated():
    digest = hashlib.sha256(_render("cot").encode()).hexdigest()[:16]
    assert digest == COT_PROMPT_SHA256, (
        f"cot prompt changed (got {digest}, expected {COT_PROMPT_SHA256}). "
        "Re-run the A/B and update the results before updating this hash."
    )


def test_every_variant_renders_with_both_placeholders():
    for variant in PROMPT_VARIANTS:
        rendered = _render(variant)
        assert "CTX" in rendered, f"{variant} dropped the context placeholder"
        assert "Q" in rendered, f"{variant} dropped the question placeholder"
        assert "{" not in rendered, f"{variant} left an unescaped brace"


def test_emits_reasoning_matches_how_the_prompt_ends():
    """A prompt ending on 'Reasoning:' must have its trace stripped, and vice versa."""
    for variant in PROMPT_VARIANTS:
        ends_on_reasoning = _render(variant).rstrip().endswith("Reasoning:")
        assert ends_on_reasoning == emits_reasoning(variant), (
            f"{variant}: prompt ending and emits_reasoning disagree"
        )


def test_unknown_variant_raises():
    """Silently defaulting would let a typo compare a variant against itself."""
    try:
        build_context_prompt("q", variant="does-not-exist")
    except ValueError:
        return
    raise AssertionError("unknown variant should raise ValueError")


def test_strip_reasoning():
    cases = [
        # The bug this file exists for: bolded label left "**" in the answer.
        ("Reasoning: nothing here.\n**Answer:** I don't know.", "I don't know."),
        # "answer:" inside the reasoning prose must not be treated as the marker.
        ("Reasoning: the context does support an answer: it defines X.\nAnswer: X is a thing.",
         "X is a thing."),
        # Baseline responses have no marker and must pass through untouched.
        ("A plain baseline answer with no marker.", "A plain baseline answer with no marker."),
        ("", ""),
        (None, None),
        ("Reasoning: r\nAnswer:\nMulti\nline answer.", "Multi\nline answer."),
    ]
    for raw, expected in cases:
        assert strip_reasoning(raw) == expected, f"strip_reasoning({raw!r})"


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
