# Sprint 4 — testing foundation, and a much larger evidence base

Sprint 4 was scoped as "add pytest". It became mostly a measurement sprint, because
the first thing the tests did was find that two of the three question sets the Sprint 3
result rested on were selecting the wrong questions.

**Headline: the Sprint 3 refusal result now reaches significance.** On the full
24-question eval set the baseline falsely refuses 6 questions and the CoT arm refuses
0 — **Fisher exact p = 0.0219**, against p = 0.43 on Sprint 3's hand-picked 4.

---

## 1. Test suite

103 tests: **99 offline** (~6s, no network, no quota) and **4 live**.

| File | n | What it protects |
|---|---|---|
| `test_prompt_variants.py` | 6 | `strip_reasoning`; sha256 pin of the rendered CoT prompt (Sprint 3) |
| `test_config.py` | 9 | The `or`-fallback at `config.py:16`; LangSmith tracing gating |
| `test_external_search.py` | 14 | All three sources: happy path, malformed payload, raising client, `lru_cache` |
| `test_format_external_results.py` | 14 | `<span>` stripping, per-source caps, `None`-when-empty |
| `test_eval_regression.py` | 11 | The gates below |
| `test_refusal_harness.py` | 16 | `is_refusal`'s rules; `--resume`; quota preflight (§6) |
| `test_refusal_stats.py` | 11 | The Fisher implementation; missing-data handling (§6) |
| `test_calibration_score.py` | 12 | The judge-vs-human arithmetic (§6) |
| `test_out_of_corpus_selection.py` | 6 | The guard's questions are still absent from the corpus (§6) |
| `test_app_smoke.py` | 4 (live) | The real streamlit app end to end |

The suite runs in ~6s, down from ~33s: `data_processing.py` used to load the FAISS
index at *import* time, so every test file that imported anything two hops away paid
a 25s index load (§6, A8).

*(An earlier draft of this file said "57 tests: 53 offline". That was wrong twice
over: `test_format_external_results.py` has 14 tests and not 13, and the count
omitted the one skipping test. The figure was 58/54 before this sprint's audit
round added 29 more.)*

`pytest.ini` sets `addopts = -m "not live"`, so the default run is offline, free and
deterministic. Sprint 5's CI has no API key and wants exactly that.

### The network guard

`tests/conftest.py` fails any non-`live` test that opens an outbound socket.

It exists because of a mistake made writing these tests. A mock was pointed at
`medbot.external_search.search_serpapi`, but `query_handler` had already bound that
name into its own namespace via `from ... import ...`. The patch missed entirely and
the test reached PubMed, Wikipedia and SerpAPI for real — while passing its own
assertions. **A mock that misses looks exactly like a mock that works, only slower.**
That is not a mistake worth relying on discipline to avoid, so it is now enforced:
the guard names the offending test and the host it tried to reach.

### Bug found and fixed: a bad SerpAPI key silently emptied all three sources

`search_serpapi` caught only `RequestException` while the other two caught `Exception`.
The serpapi client raises its own types for auth and quota errors, so those escaped
into `search_external_sources`, whose blanket `except` discarded **all three** sources
and returned `{}`. A user with a bad SerpAPI key lost their PubMed and Wikipedia
results too, with no error shown.

Fixed by widening the `except` to match its siblings. Regression test asserts the other
two survive a SerpAPI failure.

---

## 2. The refusal suite was measuring the wrong questions (audit F1)

The old suite was 4 questions taken from Sprint 2's prose summary. Re-mining the stored
Sprint 3 answers with the criterion the audit itself suggested — high Precision@4 but
low groundedness — found **four** baseline refusals, and **only two were in the suite**:

| Question | P@4 | baseline claim score | in old suite |
|---|---|---|---|
| What are the symptoms of breast cancer? | 1.00 | 0.00 | yes |
| What is bursitis? | 0.75 | 0.00 | yes |
| **What causes bladder cancer?** | **1.00** | **0.00** | **no** |
| **What is atherosclerosis?** | **1.00** | **0.50** | **no** |

`"What causes bladder cancer?"` answered *"I don't know, as the provided context does
not state the exact cause of bladder cancer"* on a **perfect Precision@4** — a textbook
instance of the exact bug the suite exists to measure, sitting outside it for two
sprints. Meanwhile `"What causes bedsores?"` and `"What is an abscess?"`, both in the
old suite, did not refuse at all in that run.

The suite is now **the whole 24-question eval set**. Every question in it was verified
corpus-answerable in Sprint 2 by reading retrieved chunks, so a refusal is always a bug;
no hand-picking is involved, and the denominator is honest.

### Result — 24 questions × 2 arms × 3 trials = 144 calls

| | baseline | cot |
|---|---|---|
| Trials refused | 13/72 | **0/72** |
| **Questions ever refusing** | **6/24** | **0/24** |

Question-level 2×2 `[[6, 18], [0, 24]]` → **Fisher exact two-sided p = 0.0219**, significant at 0.05.

Baseline refusers: breast cancer (3/3), atherosclerosis (3/3), bursitis (3/3), bladder
cancer (2/3), bed-wetting (1/3), burns (1/3). The last three are intermittent, which is
why a rate over repeated trials is the right measurement and a single pass is not.

**Read this at the question level, not the trial level.** 13/72 vs 0/72 is the same
mistake Sprint 3's draft made — those 72 are 3 repeats of 24 questions, repeated
measures on 24 units. The trial repetition is evidence the effect is not sampling
noise; it is not evidence about how many questions the fix generalises to. The
question-level test is the claim.

Fisher's exact test is implemented in `refusal_stats.py` rather than imported, since
scipy is not a project dependency. Validated against the audit's 0.43 for the Sprint 3
table and the classic tea-tasting 0.4857.

---

## 3. The out-of-corpus guard included a question the corpus covers (audit F7, F8)

`"What are the symptoms of diabetes?"` was **half of the 2-question guard** — and it is
not out-of-corpus. Its top-4 chunks include the *blood sugar tests* entry (a **B** entry)
explaining that "a person with diabetes mellitus either does not make enough insulin, or
makes insulin that does not work properly… blood sugar that remains high, a condition
called hyperglycemia."

A model answering from that is reading the corpus, not inventing. This is precisely the
error audit F8 identified in the stroke question — still live in the shipped guard, one
sprint after the lesson was written down.

The fix is a tool, not a resolution to be more careful: **`medbot/eval/verify_coverage.py`**
retrieves the top chunks for a candidate and reports whether the topic appears, using
local retrieval only, so it costs no quota and can be re-run freely. It **rejected 15 of
25** candidates — diabetes, Parkinson's, tuberculosis, kidney stones, malaria, migraine,
lupus, epilepsy, varicose veins, rabies, hemorrhoids, tonsillitis, warts, vertigo,
tinnitus. The corpus cross-references far more C–Z topics than "A–B only" suggests.

*(This count was "14 of 26" in the first draft — wrong in both figures, and unverifiable
at the time because `CANDIDATES` in `verify_coverage.py` held only 12 of the 25 questions
actually screened. The full screened list is now in that file, rejections included, and
re-running it reproduces 10 absent / 15 contaminated exactly. See §6.)*

The guard is now 10 verified questions. Not considered at all: gout and rheumatoid
arthritis (named in the bursitis entry), osteoporosis and anorexia (CoT exemplars in
`prompt.py`).

### This measurement is INCOMPLETE

The run exhausted the Gemini free tier's 500 requests/day at question 2 of 10.

| Question | baseline | cot |
|---|---|---|
| How is psoriasis treated? | 3/3 refused | 3/3 refused |
| What are the symptoms of schizophrenia? | 3/3 refused | 1/1 refused |
| *(8 remaining)* | — | — |

No over-answering in what was recorded, but **1 of 10 questions is not a hallucination
guard**. `test_out_of_corpus_gate_is_fully_armed` skips with the list of missing
questions rather than passing quietly. To arm it:

```
python -m medbot.eval.refusal_trials --trials 3 --suite overanswer --out-prefix sprint4_ --resume
```

~54 calls with `--resume` (the two recorded cells are kept), ~60 without.

**Still not armed as of 2026-07-26.** The retry was attempted and the free tier was
still exhausted: `generate_content_free_tier_requests, limit: 500` on a bare one-token
call. The daily counter rolls over at **midnight US Pacific**, which is ~12:30 IST, not
at local midnight — so "run it tomorrow morning" was never going to work, and that is
the actual reason this has now slipped twice rather than any problem with the harness.
Run it after ~12:30 IST.

That attempt did expose two real defects, both fixed in §6: the run was not resumable
(so a second quota failure at the same question would have left it stuck forever), and
`refusal_stats.py` was scoring the *unmeasured* cot cell for schizophrenia as a
non-refusal — reporting 1/2 questions ever refusing, and computing a Fisher p, from a
question the model had never been asked.

---

## 4. Judge calibration (audit F6) — partial, and it needs a human

The judge is uncalibrated, self-grading, and demonstrably grades things it should not
(it counts the misspelling "seizers" as an unsupported claim). One component of the
worry can be tested without human labels, and was:

| | baseline | cot | delta |
|---|---|---|---|
| Mean answer length | 781 chars | 995 chars | +27.5% |
| Mean claims extracted | 7.71 | 9.54 | +23.8% |
| **Chars per extracted claim** | **101** | **104** | ~equal |

The usual mechanism — a longer answer inflating the score — is **ruled out**. The score
is a ratio, and the denominator scales with length at a near-constant rate across arms.

What this **cannot** rule out is the judge rewarding hedging phrasing of the kind the CoT
prompt produces. The CoT arm scores ~0.997 with almost no variance, so there is nothing
left to correlate against. Within-arm, `corr(length, score)` is +0.54 for baseline —
but that is driven by refusals being both short and zero-scored, not by a length reward.

**`calibration_sheet.md` is ready to label**: 25 answers, arm hidden, judge score hidden,
order shuffled (seed 20260725), retrieved context included. Stratified — every answer the
judge scored below 1.00, plus a fixed-seed sample of perfect scores to catch the opposite
error. `calibration_key.json` holds the answer key; do not open it first.

Markdown formatting was deliberately **not** stripped: rewriting the text would change
what is being judged. The formatting is an arm cue, and the sheet says so.

This needs a human labeller and cannot be closed by me — I am the same model being
audited, so my labels would not be independent evidence.

---

## 5. Where the evidence stands

**Stronger than Sprint 3:**
- Refusal fix now significant at the question level on a 6× larger suite (p = 0.0219),
  and the Fisher implementation behind that number is now itself tested (§6).
- The suite is the whole eval set, so selection bias in the question list is gone.
- Out-of-corpus candidates are now verified by retrieval, not by reasoning about topic —
  and re-verified automatically on every test run, not once by hand.
- 83 offline tests, and the network guard makes a whole class of fake test impossible.

**Still open:**
- **The hallucination guard is not armed** — 1 of 10 questions measured. Quota, and the
  quota window is Pacific. Re-run with `--resume` after ~12:30 IST.
- **The judge is still uncalibrated.** Length bias ruled out; hedging bias untested.
- Three trials per question, not five. Enough for a rate given the observed
  within-question consistency, but fewer than Sprint 3 used.
- `test_eval_regression.py` reads committed artefacts, so it catches someone committing
  worse numbers — not live model drift. That is a deliberate trade for a free, offline,
  deterministic CI gate.

---

## 6. Audit round (2026-07-26)

Re-checking Sprint 4 against its own artefacts. Every headline number was recomputed
from the raw JSON rather than read from the prose. **The headline survives unchanged**:
6/24 vs 0/24, Fisher exact p = 0.0219, verified by re-running `refusal_stats`. So do the
calibration figures (781/995 chars, 7.71/9.54 claims, 101/104 chars per claim), the claim
scores (0.841 / 0.997), Precision@4 = 0.8333, and the 25-item calibration sheet.

Four defects found, all fixed.

**A1 — `refusal_stats.py` imputed missing measurements as zeros.** On the incomplete
out-of-corpus data it reported cot "questions ever refusing **1/2**", counting the
schizophrenia cell — which has *no cot trials at all* — as a question the model answered
rather than refused. It then fed that phantom into the 2×2 and printed a p-value, and
listed schizophrenia as a question where "baseline refused every trial and cot none",
for an arm that was never run. A sprint whose entire theme is *we were measuring the
wrong questions* should not ship a stats tool that invents observations. Questions
missing any arm are now excluded from the comparison and reported under `INCOMPLETE`,
with the trial totals still showing every call the quota actually bought.

**A2 — the Fisher implementation had no test.** It produces the sprint's headline number
and is hand-rolled (scipy is not a dependency); its docstring claimed validation against
the tea-tasting table and the Sprint 3 value, but nothing enforced it, so an edit to the
two-sided rule would have silently restated every significance claim in this file.
`test_refusal_stats.py` pins tea-tasting = 0.4857, Sprint 3 = 0.4286, Sprint 4 = 0.0219,
plus row-swap symmetry and the p ≤ 1 bound that the 1e-9 float slack could otherwise
break.

**A3 — the trial runner could not resume, which is why this sprint kept stalling.**
`run()` began from an empty dict and the checkpoint overwrote the file, so a quota-killed
run restarted at question 1 and re-spent quota on cells already recorded. If the quota
ran out at the same place twice, question 3 was unreachable — the suite was in exactly
that loop. `--resume` keeps any cell with enough trials and measures only the gaps. A
partially-filled cell is re-run rather than topped up, since the cell is the unit the
rate is computed over and blending two runs inside one cell could blend two prompt
versions. Twelve tests cover it, including that a resumed run does not drop the skipped
cells from the checkpoint it overwrites.

**A4 — the coverage screen was not reproducible.** `CANDIDATES` held 12 questions while
the sprint had screened 25, so `--candidates` could not reproduce the selection and the
recorded count ("14 of 26") was both wrong and uncheckable. The true figures are **25
screened, 15 rejected, 10 kept**. The complete list now lives in `verify_coverage.py`
with rejections retained — they are the evidence the kept 10 were a screen and not a
search — and `test_out_of_corpus_selection.py` re-runs the screen against the live index
on every test run. It asserts both directions: the 10 shipped questions are still absent,
and questions the screen rejected are still detected as covered, so a screen that
degraded into accepting everything fails rather than passing trivially.

Also corrected: the test counts in §1 (57/53 → the real 58/54 before this round), and a
stale docstring in `test_external_search.py` still describing the SerpAPI bug as an
unfixed asymmetry after Sprint 4 had fixed it.

### Second pass — unblocking the work that quota does not gate

**A5 — the calibration exercise had no analysis step.** `calibration_sample.py` writes
the sheet and the key; **nothing read the key back**. A finished labelling session would
have produced 25 hand-written ratios and no result. `calibration_score.py` closes it,
and reports the statistic the exercise actually turns on: **differential** bias, cot
minus baseline. A judge that is wrong by the same margin in both arms leaves the
0.841 → 0.997 claim standing, because that claim is a difference and a constant bias
cancels; only a bias that *differs* between arms can eat the delta. Refusals (0 claims)
are held out rather than averaged in as 0.0 — they are the false-refusal bug, and
scoring them as zero groundedness would drag the baseline arm down and manufacture
exactly the differential bias the tool is looking for. Tested on synthetic labels only:
the real sheet stays unlabelled, because I am the model under audit.

**A6 — a dead run was indistinguishable from a slow one.** The backoff notice at
`run_eval.py:45` was unflushed, so a piped run printed nothing while it burned
3 × 65s per call against a quota that could not recover. Added `flush=True`, and a
`preflight()` that spends one call to check the daily cap before committing 54–240.
It distinguishes the two 429s: the per-minute cap is transient and must fall through to
the normal backoff, while the daily cap cannot be waited out. That distinction is not
cosmetic — Google reports **both** against the same metric name, so the obvious
substring check treats an RPM blip as fatal and kills runs that would have finished.
The message names the Pacific rollover, since assuming local midnight is what wasted
today. Fail time: **32s → 6.5s**, nothing spent.

**A7 — the fragility was only in the prose.** `refusal_stats` now prints, next to the
p-value, how many of the refusing questions rest on a single trial, and recomputes the
table without them: 2 of the 6 are 1/3, and dropping them gives **p = 0.1092, not
significant**. The headline stands as measured, but the reader now sees what it rests on
without having to find it in a paragraph.

**A8 — `data_processing.py` loaded the FAISS index at import time.** A module-level
`vectordb = create_vector_database()` that **nothing consumed** — `app.py` and all five
eval scripts import the function and call it themselves, so each paid the load twice,
and the test suite paid it merely for importing a module two hops away. Slowness was the
mild half: the call *builds* the index when it is absent, so importing this module on a
machine without `vectorstore/` would start embedding 1225 chunks — from `--help`, or
from `pytest --collect-only`. That is a live trap for Sprint 5's CI. Removed. Offline
suite **33s → 5.6s**.

**Not fixed, and not fixable by me:** the judge calibration (§4) still needs a human
labeller — the tooling is now complete on both sides of it, but the labels have to be
yours. And the out-of-corpus measurement still needs quota that does not exist until the
Pacific rollover.
