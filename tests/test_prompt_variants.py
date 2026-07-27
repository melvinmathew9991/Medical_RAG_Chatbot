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
    """
    The exact string the A/B was run on. `COT_PROMPT_SHA256` is a hash of this, so
    the fill values are part of the pin and must not be changed casually -- use
    `_render_with_sentinels` for anything that needs to *find* the placeholders.
    """
    return build_context_prompt("What is bursitis?", variant=variant).format(
        context="CTX", question="Q"
    )


# Distinctive fill values for the placeholder test. `assert "Q" in rendered` was
# vacuous: every template contains the literal word "Question:", so it held whether
# or not the {question} placeholder survived. Found by the 2026-07-27 audit. These
# cannot occur in prompt prose, so their absence really does mean a dropped
# placeholder. Kept separate from `_render` so the sha256 pin above still describes
# the string that was measured.
CONTEXT_SENTINEL = "CTX_SENTINEL_7f3a"
QUESTION_SENTINEL = "Q_SENTINEL_91c4"


def _render_with_sentinels(variant):
    return build_context_prompt("What is bursitis?", variant=variant).format(
        context=CONTEXT_SENTINEL, question=QUESTION_SENTINEL
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
        rendered = _render_with_sentinels(variant)
        assert CONTEXT_SENTINEL in rendered, f"{variant} dropped the context placeholder"
        assert QUESTION_SENTINEL in rendered, f"{variant} dropped the question placeholder"
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


def test_the_no_examples_arm_contains_no_example_at_all():
    """
    The property the arm exists for, asserted rather than assumed: with no example
    in the prompt there is no other question for the model to answer, so the
    few-shot contamination seen on 5/5 bursitis trials under `instruction-only`
    (results_sprint4.md §8) is impossible by construction.

    Checked against every legacy example's question text, not just a sample, so
    adding a 28th example to `MEDICAL_EXAMPLES` cannot quietly leak into this arm.
    """
    from medbot.prompt import lazy_loader

    rendered = _render_with_sentinels("no-examples")
    for example in lazy_loader.load_medical_examples():
        assert example["question"] not in rendered, (
            f"no-examples arm contains the example question {example['question']!r}"
        )
    assert "nosebleed" not in rendered.lower()
    assert "Question:" in rendered  # the user's own question is still labelled
    assert CONTEXT_JUDGEMENT_GUIDANCE in rendered


def test_the_no_examples_arm_is_the_cheapest():
    """
    Its whole case is cost: it must stay far below the exemplar arms, or there is
    no reason to prefer it over `cot`. Measured on the template, before context.
    """
    sizes = {v: len(_render_with_sentinels(v)) for v in PROMPT_VARIANTS}
    assert sizes["no-examples"] < sizes["instruction-only"] < sizes["cot"]
    assert sizes["no-examples"] * 5 < sizes["cot"], (
        f"no-examples ({sizes['no-examples']} chars) is not decisively cheaper than "
        f"cot ({sizes['cot']} chars); the arm's only argument is token cost"
    )


def test_no_examples_is_not_the_shipped_default():
    """
    It is unmeasured. Shipping an unmeasured prompt is the mistake Sprint 3 exists
    to have avoided, so this fails if the default moves before the numbers do.
    """
    from medbot.prompt import DEFAULT_PROMPT_VARIANT

    assert DEFAULT_PROMPT_VARIANT == "cot"


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
