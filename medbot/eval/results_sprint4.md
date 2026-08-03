# Sprint 4 — testing foundation, and a much larger evidence base

Sprint 4 was scoped as "add pytest". It became mostly a measurement sprint, because
the first thing the tests did was find that two of the three question sets the Sprint 3
result rested on were selecting the wrong questions.

**Headline: the Sprint 3 refusal result reaches significance.** On the full 24-question
eval set the baseline falsely refuses **7** questions and the CoT arm refuses **0** —
**Fisher exact p = 0.0094**, against p = 0.43 on Sprint 3's hand-picked 4.

> Those are the numbers from the 5-trial re-run under the corrected `is_refusal`
> (§7, 2026-07-27). This file originally headlined **6 vs 0, p = 0.0219** from a 3-trial
> run; that run rested on an instrument that scored few-shot contamination as refusal,
> and re-scored it is **3 vs 0, p = 0.2340 — not significant**. The superseded sections
> are marked in place rather than rewritten, so the correction stays checkable.

**The out-of-corpus hallucination guard is armed** (§3): 10 questions × 2 arms × 3 trials,
**60/60 refused**, zero invented answers, unchanged by the instrument correction.

---

## 1. Test suite

127 tests: **123 offline** (~10s, no network, no quota) and **4 live**.

| File | n | What it protects |
|---|---|---|
| `test_prompt_variants.py` | 9 | `strip_reasoning`; sha256 pin of the rendered CoT prompt (Sprint 3); the `no-examples` arm (§10) |
| `test_config.py` | 9 | The `or`-fallback at `config.py:16`; LangSmith tracing gating |
| `test_external_search.py` | 14 | All three sources: happy path, malformed payload, raising client, `lru_cache` |
| `test_format_external_results.py` | 14 | `<span>` stripping, per-source caps, `None`-when-empty |
| `test_eval_regression.py` | 11 | The gates below |
| `test_refusal_harness.py` | 27 | `is_refusal`'s rules and both gate readers; `--resume`; quota preflight (§6–§8) |
| `test_refusal_stats.py` | 13 | The Fisher implementation; missing-data handling; the reporting unit (§6, §8) |
| `test_calibration_score.py` | 12 | The judge-vs-human arithmetic (§6) |
| `test_out_of_corpus_selection.py` | 6 | The guard's questions are still absent from the corpus (§6) |
| `test_expansion_selection.py` | 8 | The screened expansion questions still retrieve their entries, both directions (§10) |
| `test_app_smoke.py` | 4 (live) | The real streamlit app end to end |

*(This table row-summed to 99 while the headline said 106, because the third audit
round's 7 new tests were never added to the `test_refusal_harness.py` row. Both
figures are now recomputed from `pytest --collect-only` whenever they change, and
they agree: 123.)*

The suite runs in ~10s, down from ~33s: `data_processing.py` used to load the FAISS
index at *import* time, so every test file that imported anything two hops away paid
a 25s index load (§6, A8). It was ~6s until §10 added index-backed selection tests;
those share a single retrieval pass through a module fixture, which is what keeps the
figure at 10s rather than 14s.

*(An earlier draft of this file said "57 tests: 53 offline". That was wrong twice
over: `test_format_external_results.py` has 14 tests and not 13, and the count
omitted the one skipping test. The figure was 58/54 before this sprint's audit
round added 29 more, 106/4 after §7, and 112/4 after §8 and §9.)*

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

> ⚠️ **SUPERSEDED 2026-07-27 — do not quote these numbers.** Three of the six baseline
> "refusals" below were `is_refusal` miscounting few-shot contamination as a declined
> question. Re-scored, this run is **3/24 vs 0/24, p = 0.2340 — not significant.** The
> result that stands is the 5-trial re-run in §7. The table is kept as recorded so the
> correction can be checked against it.

| | baseline | cot |
|---|---|---|
| Trials refused | 13/72 | **0/72** |
| **Questions ever refusing** | **6/24** | **0/24** |

Question-level 2×2 `[[6, 18], [0, 24]]` → **Fisher exact two-sided p = 0.0219**, significant at 0.05.

Baseline refusers: breast cancer (3/3), atherosclerosis (3/3), bursitis (3/3), bladder
cancer (2/3), bed-wetting (1/3), burns (1/3). The last three are intermittent, which is
why a rate over repeated trials is the right measurement and a single pass is not.
*(Of these, atherosclerosis 3/3, bed-wetting 1/3 and burns 1/3 were contamination, not
refusal — see §7.)*

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

### This measurement is COMPLETE (closed 2026-07-27)

The first run exhausted the Gemini free tier's 500 requests/day at question 2 of 10, and
a retry on 2026-07-26 was still quota-blocked: `generate_content_free_tier_requests,
limit: 500` on a bare one-token call. The daily counter rolls over at **midnight US
Pacific**, which is ~12:30 IST, not at local midnight — so every "run it tomorrow
morning" attempt before ~12:30 IST was spending against the *previous* Pacific day and
had no chance of succeeding. That timezone assumption, not any problem with the harness,
is why this slipped twice.

Finished at 12:37 IST on 2026-07-27 with `--resume` (51 calls; the 3 recorded cells were
kept), one rollover after the last failed attempt.

**All 10 questions × 2 arms × 3 trials refused. 60/60, zero invented answers.**

| | baseline | cot |
|---|---|---|
| trials refused | 30/30 | 30/30 |
| questions ever refusing | 10/10 | 10/10 |

Fisher exact between arms p = 1.0000, which is the *desired* outcome here: no difference
means CoT drove false refusals to zero without buying it by over-answering on topics the
corpus does not cover. `test_out_of_corpus_gate_is_fully_armed` no longer skips.

The guard was re-scored under the corrected `is_refusal` (§7) and is **unchanged at
60/60** — the one direction that instrument must never fail in, and it is now asserted on
every test run rather than trusted.

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
- Refusal fix significant at the question level on a 6× larger suite (**p = 0.0094** on the
  corrected 5-trial data, §7), and the Fisher implementation behind that number is now
  itself tested (§6).
- The suite is the whole eval set, so selection bias in the question list is gone.
- Out-of-corpus candidates are now verified by retrieval, not by reasoning about topic —
  and re-verified automatically on every test run, not once by hand.
- The hallucination guard is armed: 60/60 across all 10 questions (§3), 90/90 once the
  ablation arm is included (§8) — and the gate asserting it can now actually fail (§9).
- 112 offline tests, and the network guard makes a whole class of fake test impossible —
  verified by probe, not assumed (§9).
- The measurement instrument itself is now tested against the patterns that defeated it
  twice, and re-scoring is a tool rather than a manual pass (§7, §8).
- Every documented number in this file has been re-derived from the stored answer text
  under all three historical instruments, and reconciles exactly (§9).

**Still open:**
- **The judge is still uncalibrated.** Length bias ruled out; hedging bias untested.
- **Question-set expansion is the real constraint** — 24 questions, not trials per
  question. The fragility warning survives 3 → 5 trials and will survive more (§7).
- ~~**The Sprint 3 ablation needs re-running** under the corrected instrument before any
  decision about whether the CoT exemplars earn their tokens (§7)~~ — **done, §8.** All 24
  questions: instruction-only vs cot **p = 1.0000** on refusals, the real cost is **~2,770
  tokens per query**, and the exemplars stay on non-refusal grounds. The arm to run next is
  the instruction with *no* examples at all.
- ~~`refusal_stats.py` still advises more trials per question~~ — **fixed 2026-07-27.** The
  fragility block now says more *questions* is the fix, and the source carries the evidence
  for why more trials cannot be, so the next person does not re-spend the 240 calls.
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
Pacific rollover. *(That last one was closed on 2026-07-27 — see §3.)*

---

## 7. Third audit round — the instrument was wrong (2026-07-27)

`refusal_stats` prints a fragility warning on its own headline: two of the six refusing
questions (bed-wetting, burns) refused on exactly one trial of three, and dropping them
takes p from 0.0219 to 0.1092. Its closing advice is *"More trials per question is the
fix, not a larger claim."* That advice was taken and the suite re-run at 5 trials —
24 × 2 × 5 = 240 calls, fresh `--out-prefix sprint4_t5_`, deliberately **not** `--resume`
since every existing cell held 3 trials and mixing cell sizes would corrupt the rate.

The extra trials did not settle the statistics. They exposed a defect in `is_refusal` —
the instrument behind **every refusal number in Sprints 2, 3 and 4**.

### What it was counting

`is_refusal` scored any refusal marker in the opening 200 characters, regardless of what
came after. Two patterns defeat that, both invisible at 3 trials:

**1. Few-shot contamination.** The model declines a question drawn from its own selected
exemplar, then answers the real one:

> *"I don't know the answer to your question about **bee stings** based on the provided
> context. However, regarding how burns should be treated, …"* — followed by 2,462
> characters of correct burn treatment.

Also seen as osteoporosis standing in for atherosclerosis (5/5 trials) and nosebleeds for
bursitis. This is a real defect and the Sprint 3 audit already recorded it — but it is a
*contamination* defect. Scoring it as a declined question files it under the wrong bug
and inflates the baseline arm that the entire CoT result is measured against.

**2. A leading hedge before a full answer.**

> *"The context does not provide a single, straightforward list labeled 'symptoms of
> alcoholism,' but it describes numerous health problems…"* — followed by 1,189
> characters of symptoms.

The note on `REFUSAL_WINDOW_CHARS` anticipated *trailing* caveats. This is the same thing
at the front, and it was the single CoT hit in the 5-trial run.

### The fix, and two fixes that were rejected

A marker now makes an answer a refusal *candidate*; it scores as a refusal only if
nothing survives stripping the sentences that assert absence (`delivered_content` +
`ABSENCE_RE`). Calibrated against all **572 stored trials**: 16 labels change, every one
inspected by hand.

Both obvious alternatives are wrong, and both are pinned as tests so they are not
re-attempted:

- **Length alone.** Genuine refusals routinely run 200–400 characters because they
  helpfully enumerate what the corpus covers instead: *"I don't know based on the
  provided context. The context contains information on treating acne, bedsores, and
  atopic dermatitis, but it does not mention psoriasis."* That is a refusal, and a long
  one.
- **Matching the question's own terms.** Actively dangerous. The refusal above ends
  "…does not mention **psoriasis**", so a term check flips precisely the out-of-corpus
  refusals the hallucination guard depends on. Tried during calibration; it degraded the
  60/60 guard, which is the one direction this instrument must never fail in.

The threshold (`MIN_SUBSTANCE_CHARS = 120`) sits in an observed gap: 119 of the 138
marker-flagged trials leave 0 characters, every contaminated or hedged answer leaves
190+, and the only values between are three identical 113-character cases. Those three
are `"What causes bladder cancer?"` under baseline and are the one genuinely debatable
label in the set; the reasoning for leaving them as refusals is recorded at the constant
itself, along with what moving it would do.

`medbot/eval/relabel.py` re-derives the stored booleans from the stored text (dry-run by
default, no quota). It exists because `test_refusal_labels_match_the_heuristic` fails by
design when the instrument changes — so the fix and the re-scoring cannot drift apart.
Answer text is never touched, making any re-scoring reversible.

### What it costs

| | as recorded | re-scored |
|---|---|---|
| **3 trials** (§2, the shipped headline) | `[[6,18],[0,24]]` p = 0.0219 | `[[3,21],[0,24]]` **p = 0.2340 — not significant** |
| **5 trials** (this run) | `[[8,16],[1,23]]` p = 0.0226 | `[[7,17],[0,24]]` **p = 0.0094** |
| Out-of-corpus guard | 60/60 | **60/60 unchanged** |
| Sprint 3 ablation | instruction-only 5, examples-only 1 | **instruction-only 0, examples-only 1** |

**Sprint 4's own headline does not survive its own corrected instrument.** The result
that stands is the 5-trial run: baseline **7/24** questions, cot **0/24**, 20/120 vs
0/120 trials, **p = 0.0094**. CoT returns to a clean zero.

The 5-trial run was worth running, but not for the reason it was launched. It moved the
raw p-value by 0.0007. Its value was generating the data that exposed the instrument bug,
and being the only run that survives correction.

### The advice in `refusal_stats.py` is wrong

Going 3 → 5 trials left the fragility warning firing on 3 of 7 questions, and the
membership churned: burns (1/3) did not reproduce at **0/5**, while autism and bedsores
appeared fresh at 1/5. That churn is the signature of low-rate stochastic refusal — more
trials detect *more* questions at exactly one, so the "drop the 1-trial questions"
sensitivity check keeps failing however much quota is spent on it.

The binding constraint is **24 questions**, not trials per question. At the observed
rates, doubling the question set would put p near 0.0004 (a projection, not a
measurement). Question-set expansion is the fix; the advice line should be corrected
before it costs another 240 calls.

### Consequence for Sprint 3 — acted on in §8

The re-scored ablation reads instruction-only **0**, examples-only **1** — inverting the
recorded conclusion that bursitis needs the exemplars and that only the combination
reaches zero. On this evidence the instruction rewrite alone suffices and the exemplars'
~2,770 tokens per query (measured in §8; recorded as ~2,400 before that) are unjustified.
**Not acted on here:** 4 questions is far too small a base to strip the exemplars on, and
the ablation should be re-run under the corrected instrument first. → **§8 does exactly
that on all 24 questions, and the refusal parity holds (p = 1.0000). The exemplars still
stay, for two reasons that are not refusal-shaped.**

---

## 8. Sprint 3 ablation, re-run under the corrected instrument (2026-07-27)

§7 left one decision open: the CoT exemplars cost tokens on every query, and the re-scored
4-question ablation said the instruction rewrite alone reached zero refusals. That was too
small a base to strip them on — 4 hand-picked questions is the same selection mistake as
audit F1 — so the arm was re-measured on the honest denominator.

**Design.** `instruction-only`, all 24 eval questions, 5 trials, 120 calls, written into a
copy of `sprint4_t5_refusal_trials.json` (`--out-prefix ablation_t5_ --resume`) so all three
arms sit at 5 trials on the same 24 questions with the prompt, index and model unchanged.
Then the 10-question out-of-corpus guard at 3 trials, 30 calls, into a copy of
`sprint4_overanswer_trials.json` for the same reason.

`examples-only` was deliberately **not** re-run. It answers "do the exemplars work without
the instruction", which is not a decision anyone faces — nobody would ship exemplars under
the old disclaimer — and it would have cost another 120 calls out of a 500/day quota.

**What the exemplars actually cost.** Rendered template, `"What is bursitis?"`, before any
retrieved context:

| variant | template | est. tokens |
|---|---|---|
| baseline | 567 chars | ~141 |
| instruction-only | 1,064 chars | ~266 |
| examples-only | 11,122 chars | ~2,780 |
| **cot** (shipped) | **12,151 chars** | **~3,037** |

So the exemplars cost **~2,770 tokens per query**, not the ~2,400 recorded in Sprint 3 and
in `MIGRATION_STATUS.md`. Corrected in both.

### Result — 24 questions × 5 trials, corrected instrument

| arm | questions ever refusing | trials | vs cot | vs baseline |
|---|---|---|---|---|
| baseline | **7/24** | 20/120 | p = 0.0094 | — |
| instruction-only | **1/24** | 1/120 | **p = 1.0000** | p = 0.0479 |
| cot (shipped) | **0/24** | 0/120 | — | p = 0.0094 |

Out-of-corpus guard, 10 questions × 3 trials: **all three arms refuse 30/30**, 0/10 questions
ever answering, p = 1.0000 between every pair. 90/90 overall.

**The exemplars buy no measurable refusal improvement over the instruction rewrite alone.**
instruction-only's single refusing question is bursitis at 1/5 trials, and that trial is
itself contamination — the model declined a question about *nosebleeds* and never reached
bursitis, so nothing was delivered and the substance rule correctly calls it a refusal. The
same question is contaminated in the other 4 trials too, and there it does deliver bursitis
content afterwards.

### The instrument was wrong again, and the third arm is what found it

The guard run first reported instruction-only **answering** 4 of 10 out-of-corpus questions —
invented medical content, the one direction this suite must never fail in. Reading the stored
text rather than the booleans showed all six such trials were flat refusals:

> *"Based on the provided context, there is no mention of the symptoms of shingles.
> Therefore, the context does not support an answer to this question."*

`"no mention"` is not in `REFUSAL_MARKERS` — `"no information"` and `"does not mention"` are.
The candidate gate never opened, the substance rule was never consulted, and six correct
refusals were scored as inventions.

**Why it surfaced only now: `REFUSAL_MARKERS` was calibrated against stored baseline and cot
trials, because those were the only arms with data.** A prompt change changes the model's
refusal *wording*, so an instrument keyed to wording does not transfer to a new arm. Same
shape as F1 — the measurement was validated on a sample that excluded the case that breaks it.

- **Fix:** the gate is now `REFUSAL_MARKERS ∪ ABSENCE_RE` over the opening window
  (`opens_with_absence_language`). A union rather than a replacement, because neither reader
  contains the other: `"cannot answer"` without a leading "I" matches a marker but not the
  regex. The union is strictly broader than the old gate, so it can only add *candidates*, and
  each still has to fail the substance test to score as a refusal.
- **Blast radius, measured before changing anything:** all 1,022 stored trials across 10 files
  were scanned for refusals the old gate missed. Exactly **6**, all in the new instruction-only
  guard data. Every baseline and cot label is unchanged, so **the shipped headline (7/24 vs
  0/24, p = 0.0094) and the 60/60 guard survive untouched** — this defect never touched them,
  because neither arm phrases refusals this way.
- Re-scored with `relabel.py --write`; the guard is now **90/90 across three arms**.
- Pinned: `test_the_third_arm_refusal_phrasing_is_detected` (the three verbatim texts),
  `test_the_gate_is_the_union_of_the_marker_list_and_the_absence_regex` (so "ABSENCE_RE covers
  everything, drop the list" is not tried), `test_the_wider_gate_still_needs_the_substance_test`,
  and `test_the_instruction_only_guard_data_is_fully_armed` — the ablation arm gets the same
  standing pin as the shipped one, because if the exemplars are ever dropped, instruction-only
  *becomes* the shipped prompt.

### A fixture that had been passing for the wrong reason

Widening the gate broke `test_the_partial_answer_from_audit_f9_is_not_a_refusal`, and the
fixture was at fault, not the instrument. Its docstring said "kept verbatim" while quoting a
shortened paraphrase: 97 characters of surviving content, below `MIN_SUBSTANCE_CHARS`. Under
the narrow gate that never mattered, because "does not explicitly define" matched no marker and
the substance rule was never reached. The real stored answer is 603 characters with 380
surviving. Replaced with the actual text and pinned with an explicit
`len(delivered_content(answer)) > MIN_SUBSTANCE_CHARS` assertion. A truncated fixture is a test
that agrees with you for a reason you did not intend.

### `refusal_stats` was reporting the guard in the direction that cannot see a leak

While those six trials were still believed to be leaks, the guard report read:

```
instruction-only  trials 24/30   questions ever refusing 9/10
Fisher exact (two-sided) p = 1.0000  -- NOT significant at 0.05
```

Which reads as a pass. Four of the ten questions had produced answers. "Ever refused" is nearly
always true of an arm that mostly behaves, so on this suite it cannot see a partial leak — the
failure unit is *ever answered*.

`report()` now takes `unit=` (`ever-refused` | `ever-answered`), the guard suite is reported
with `ever-answered`, the 2×2 line names the unit it counted, and **both** counts print in
either mode so the familiar line does not silently change meaning. The consistency and
fragility blocks follow the same unit.
`test_the_guard_suite_is_reported_in_the_direction_that_can_see_a_leak` builds the exact shape
— 10 questions, one arm answering 4 of them at least once — and pins 9/10 & p=1.0000 in the old
unit against 4/10 & p=0.0867 in the new one. The six trials turned out to be an artefact; the
reporting hole was real regardless, and the next leak may not be.

### Contamination is a property of the legacy example format, not of "having examples"

Counting marker-flagged trials that *do* deliver substance — the population the corrected
instrument separates out — across 24 questions × 5 trials:

| arm | flagged but answered | question |
|---|---|---|
| baseline | 5 | atherosclerosis (5/5, osteoporosis substituted) |
| instruction-only | 4 | bursitis (5/5 including the refusal, nosebleeds substituted) |
| cot | 1 | alcoholism — a leading hedge, not a substituted question |

Both contaminated arms are the two that use the **legacy semantic 1-shot selector**
(`SemanticSimilarityExampleSelector`, k=1, over the 27 generic Q→A examples in `prompt.py`);
neither CoT arm shows it. "What is bursitis?" retrieves the *nosebleed* example, and the model
answers the example's question. So the ablation's real 2×2 is {old, new} instruction ×
{legacy semantic 1-shot, fixed CoT exemplars}, and dropping the CoT exemplars does not mean
"no examples" — it means **returning to the format that contaminates**.

### Recommendation: do not strip the exemplars yet, and the reason is specific

On refusals the exemplars are unjustified: p = 1.0000 against instruction-only, on all 24
questions, both arms clean on the guard. Two things stand in the way of acting on that, and
neither of them is "more caution":

1. **instruction-only's groundedness has never been measured.** The 0.841 → 0.997 claim-level
   result is baseline vs cot. Stripping the exemplars on refusal parity alone would trade a
   measured axis for an unmeasured one. Cost to close: 24 answers + judge, ~48 calls.
2. **instruction-only contaminates** on 5/5 bursitis trials, which is user-visible and which
   the CoT arm does not do.

There is also a **fifth arm nobody has measured**: the new instruction with **no examples at
all** (~150 tokens, and contamination-proof by construction, since there is no other question
in the prompt to answer). If it holds refusals near zero it dominates every arm here on cost.
That is the arm to run next, not `examples-only`.

**Open, unchanged:** blinded human calibration of the groundedness judge (F6). The sheet, key
and scoring loop are complete and verified end-to-end against a synthetic fill; the labels have
to come from a human, since the judge is the model being audited.

---

## 9. Fourth audit round (2026-07-27) — the gates, not the numbers

The three previous rounds each corrected a *number*. This one recomputed every number
and found them all correct, then audited the thing that had never been audited: whether
the tests protecting them can fail.

**Method, chosen from §6's mistake.** §6 "recomputed from the raw JSON" and missed a
broken instrument, because it recomputed statistics from the stored `refusal` booleans —
which the instrument itself had produced. So here every refusal figure is re-derived from
the stored **answer text** under all three historical instruments, reimplemented
side-by-side:

| | gate | substance rule |
|---|---|---|
| **v0** (as shipped in `e869c33`) | marker in `head[:200]` | none |
| **v1** (first correction, §7) | marker in `head[:200]` | yes |
| **v2** (current, §8) | marker ∪ `ABSENCE_RE` | yes |

### Every documented number reconciles

| claim | documented | recomputed |
|---|---|---|
| 3-trial run, as recorded (v0) | 6/24 vs 0/24, 13/72 vs 0/72, p = 0.0219 | ✅ identical |
| 3-trial run, re-scored (v2) | 3/24 vs 0/24, p = 0.2340 | ✅ identical |
| 5-trial run, as recorded (v0) | `[[8,16],[1,23]]`, p = 0.0226 | ✅ identical |
| **5-trial run, corrected (v2) — the headline** | **7/24 vs 0/24, 20/120, p = 0.0094** | ✅ **identical** |
| Fragility: drop the three 1-of-5 refusers | p = 0.1092 | ✅ identical |
| Ablation (§8): instruction-only | 1/24, 1/120, p = 1.0000 vs cot, 0.0479 vs baseline | ✅ identical |
| Sprint 3 ablation file, v0 → v2 | instruction-only 5 → 0, examples-only 1 → 1 | ✅ identical |
| Out-of-corpus guard | 60/60 (`sprint4_`), 90/90 (with instruction-only) | ✅ identical |
| Claim-level groundedness | 0.841 → 0.997 | ✅ 0.841, 0.997 (n=24 each) |
| Precision@4, both arms | 0.8333 | ✅ identical, and per-question equal across arms |
| Judge length bias | 781 / 995 chars, 7.71 / 9.54 claims, 101 / 104 per claim | ✅ identical |
| Coverage screen | 25 screened, 15 rejected, 10 kept | ✅ re-ran: 10 absent, 15 contaminated |
| Calibration sheet | 25 items, 15 baseline / 10 cot | ✅ identical, 7 scored below 1.00 |

Three checks that go beyond recomputation:

- **Blind-spot scan.** Every one of the **1,022** stored trials was checked for the class
  the gate structurally cannot see: an answer scored *answered* whose content is thin once
  absence-asserting sentences are stripped. **0 suspects.** This is the scan that would
  have caught §8's defect before it was found by accident.
- **Inverse invariant.** Trials scored *refusal* while delivering substance: **0**, as the
  substance rule requires by construction.
- **Label drift.** Stored booleans vs the current instrument across all 10 trial files:
  **0 of 1,022** disagree.

### The socket guard was tested rather than believed

`conftest.py`'s network guard is the reason a whole class of fake test is impossible, and
nothing had ever verified it fires. A throwaway probe test that opens a real socket was
run and deleted: both `connect` and `connect_ex` were blocked, each naming the offending
test and host. It works.

### Findings: five gates that could not fail, and one environment trap

**A1 — every headline gate was pointing at the retired dataset.** `test_eval_regression.py`
read `sprint4_refusal_trials.json` — the **3-trial** run this file marks *"do not quote
these numbers"* — in all four of its refusal gates. The 5-trial run the standing headline
comes from, and both ablation files, had **no gate at all**. Re-pointed at
`sprint4_t5_refusal_trials.json` via a named constant, with the trial count per cell now
asserted too (a rate over 2 trials is not the measurement the headline claims).

**A2 — the label-drift test covered 2 of 10 trial files.** It named `sprint4_refusal` and
`sprint4_overanswer`, so drift in the eight others — including both files behind the
current headline and the ablation conclusion — would not have failed anything. Now globs
`*trials.json` and asserts it found at least 8.

**A3 — `test_out_of_corpus_gate_is_fully_armed` asserted nothing.** It was written to
`pytest.skip` while the guard data was incomplete. Once §3 completed the data, `missing`
became empty and the body fell through with **no assertion**: a test named "fully armed"
that could not fail, and that would have gone back to *skipping* — not failing — if the
guard data were truncated. It now asserts completeness and that all 60 trials refused.

**A4 — `python -m medbot.eval.refusal_stats` printed the superseded result.** The default
`--prefix sprint4_` reported **p = 0.2340, NOT significant** from the 3-trial data, while
the documented headline is p = 0.0094 from `sprint4_t5_`. Anyone re-checking this sprint
the obvious way — no arguments — would have concluded the result had evaporated. Default
is now `sprint4_t5_`, every report prints the filename it came from, and the guard's
fallback to `sprint4_overanswer_trials.json` is printed rather than silent.

**A5 — a vacuous assertion.** `test_every_variant_renders_with_both_placeholders` did
`assert "Q" in rendered` to check the `{question}` placeholder survived. Every template
contains the literal word "Question:", so the assertion held whether or not the
placeholder did. Now filled with sentinels that cannot occur in prompt prose, kept
separate from `_render` so the sha256 pin still hashes the string that was measured.
*(Same family as §8's truncated fixture: a test that agrees with you for a reason you did
not intend.)*

**A6 — 38 stale `.pyc` files made every traceback cite a path that does not exist.** The
repo moved from `D:\Medbot\` to `D:\Medical_RAG_Chatbot\`, and the bytecode caches came
with it. Source mtime and size still match, so Python reuses them, and `co_filename` is
baked in at compile time — so pytest failures pointed at
`D:\Medbot\Medical_RAG_Chatbot\tests\conftest.py:51`. The executed code was correct; only
the reported location was fiction, which is a debugging tax paid at exactly the wrong
moment. Caches cleared; they regenerate with correct paths. Worth knowing before Sprint 5
wires CI, where a phantom path in a failure log is much more expensive.

**A7 (documentation) — the test-count table row-summed to 99 while the headline said 106.**
The `test_refusal_harness.py` row was never updated when §7 added 7 tests. Both are now
recomputed from `pytest --collect-only`; §1 also had `run_eval.py`'s docstring still
claiming it writes `results.json`/`results.md`, which is what it did *before* the §6 fix.

### The new gates were mutation-tested, not assumed

Every fix above is a claim that a test now fails when something is wrong, which is the
same kind of claim A3 got wrong. So each was verified by breaking the artefact and
watching the gate fail, then reverting:

| mutation | expected | result |
|---|---|---|
| flip one out-of-corpus cot trial to an answer | guard fails | ✅ 2 tests failed |
| truncate one cot cell from 5 trials to 2 | completeness fails | ✅ failed |
| flip a stored label in `ablation_t5_` (a file the old test ignored) | drift fails | ✅ failed |

All 112 offline tests pass after the reverts, in ~6s.

### What this round did not do

- It did not re-measure anything against the live model: no quota was spent, by design.
  Every figure here comes from committed artefacts.
- `preflight`'s "fails in 6.5s instead of 32s" (§6) is not verifiable offline; it needs a
  quota-dead key to reproduce.
- The judge is still uncalibrated (F6). Nothing in this round changes that, and nothing
  in it can: the sheet needs a human labeller.

---

## 10. Prepared, not yet measured (2026-07-27)

Two pieces of work that cost **no quota** and that the next measuring run needs. Both are
deliberately inert: the code and the selection are committed, the numbers are not, and
nothing shipped changes until they are run.

### The `no-examples` prompt arm

> **MEASURED AND SHIPPED 2026-08-03 — see §11.** 0/24 refusals, guard 30/30,
> Precision@4 0.8333, claim-level 0.9917. Parity with `cot` at 1/14th the tokens.
> `DEFAULT_PROMPT_VARIANT` moved to `no-examples` after these numbers existed, so the
> third test named below — "the shipped default is still `cot`" — has been replaced by a
> gate on the *property* rather than the name: whatever ships must have recorded
> claim-level results and must not falsely refuse.

§8 concluded that the CoT exemplars buy no measurable refusal improvement over the
instruction rewrite (p = 1.0000 against `instruction-only` on all 24 questions) but kept
them anyway, because `instruction-only` substitutes a neighbouring example's question on
5/5 bursitis trials — it answers about *nosebleeds*. That contamination belongs to the
legacy semantic 1-shot format, which both non-CoT arms use.

`no-examples` removes the mechanism instead of the symptom: the new instruction with no
example in the prompt, so there is no other question available to answer.

| variant | template | est. tokens |
|---|---|---|
| baseline | 567 chars | ~141 |
| **no-examples** | **865 chars** | **~216** |
| instruction-only | 1,064 chars | ~266 |
| examples-only | 11,122 chars | ~2,780 |
| cot (shipped) | 12,151 chars | ~3,037 |

**14× cheaper than the shipped prompt.** If it holds refusals near zero and the
out-of-corpus guard at 10/10, it dominates every arm measured so far on cost — and it is
the arm to run before `examples-only`, which answers a question nobody is asking.

Pinned by three tests: that no legacy example's question text can appear in it (checked
against all 27, so adding a 28th cannot leak in), that it stays decisively cheaper than
`cot`, and that `DEFAULT_PROMPT_VARIANT` is still `cot` — shipping an unmeasured prompt is
the mistake Sprint 3 exists to have avoided.

Naming trap, documented rather than fixed: **`instruction-only` is a misnomer.** It is the
new instruction *plus* baseline's legacy 1-shot example, not the instruction alone.
Renaming it now would orphan every recorded result filed under that name.

### 22 screened expansion questions

> **MERGED AND MEASURED 2026-08-03 — see §12.** The eval set is now 46 questions.
> baseline 18/46 vs cot 0/46 vs no-examples 0/46, p = 0.0000, and the fragility warning
> that qualified every refusal number since Sprint 4 no longer fires: dropping all six
> one-trial refusers leaves p = 0.0002.

The binding constraint on the refusal headline is the number of questions (§7). Screening
and keyword grounding are free; only the trials cost quota. So the selection is done now:

- **34 candidates screened** by the new `medbot/eval/verify_entry.py` — local retrieval,
  no quota, re-runnable.
- **10 auto-rejected, 2 rejected by hand, 22 kept.**
- `expected_keywords` read out of actually-retrieved chunks. **Every keyword appears in at
  least one retrieved chunk**, and **mean Precision@4 = 0.8523** (1.00 ×12, 0.75 ×7,
  0.50 ×3) against **0.8333** for the current 24 — comparable difficulty, and a *higher*
  floor, since the existing set has two questions at 0.25.

**`verify_entry.py` exists because `verify_coverage.py`'s rule cannot be reused here.**
That rule asks "is this topic absent?", so one loose hit is a useful rejection. Inverted to
*accept* eval questions it accepted 35 of 36 candidates, including:

| candidate | what retrieval actually returned |
|---|---|
| "What causes back pain?" | the **bursitis** and arthritis entries — matched on "pain" |
| "What causes bad breath?" | the anoxia/hypoxia entry — matched on "breath" |
| "What is Barrett's esophagus?" | **acetaminophen** |
| "What is a biopsy?" | *breast biopsy* and *bone biopsy* — two entries, no general one |

The stricter rule requires the distinctive term in ≥2 of the top-4 chunks **and** one chunk
that looks like the entry itself (an encyclopedia "Definition" heading, or a copular
sentence about the term). An eval question that fails this turns every correct refusal into
a recorded bug and its Precision@4 into a measurement of corpus coverage — audit F8's error,
pointing the other way.

Two of its own rules were wrong on first run and were fixed by looking at the rejections,
not the acceptances: a `\W{0,40}` gap could not span "Atrial fibrillation **and flutter**
Definition", and an enumerated list of permitted copulas rejected "the major symptom of
Bell's palsy **is one** side of the face". Both were false negatives on real entries.

`test_expansion_selection.py` (8 tests) re-checks all of it against the live index every
run, in both directions — the 22 must still retrieve their entries, and five of the
rejected candidates must still be rejected, or the screen has stopped discriminating.

**Held back on purpose:** `EXPANSION_QUESTIONS` is a separate list, not appended to
`EVAL_QUESTIONS`. Merging redefines `REFUSAL_QUESTIONS` from 24 to 46, which invalidates
every recorded trial file at a stroke. The merge belongs in the same change as the ~336
calls that re-measure the suite, and a test pins that sequencing until then.

---

## 11. The `no-examples` arm, measured (2026-08-03)

§10 prepared this arm and left it inert. This closes it. 225 calls: the 24-question
refusal suite at 5 trials (120), the 10-question out-of-corpus guard at 3 (30),
`run_eval` (48), and the claim-level rejudge (24), plus preflights.

Run against the recorded arms rather than alongside a fresh re-run of them, which is only
legitimate because the other three arms' prompts are byte-identical to when they were
measured: the sole change to `medbot/prompt.py` since `07e215d` is two import lines
reordered by ruff, and the rendered-CoT sha256 pin passes. Sprint 6 changed `index.pkl`
but proved `index.faiss` bit-identical, and `format_sources` is called by `app.py` rather
than inside `run_query`, so no citation text can reach the scored answer.

### Result

| arm | template | est. tokens | refusals | guard | P@4 | claim-level |
|---|---|---|---|---|---|---|
| baseline | 567 ch | ~141 | 7/24 | 10/10 | 0.8333 | 0.8413 |
| instruction-only | 1,064 ch | ~266 | 1/24 | 10/10 | 0.8333 | not measured |
| **no-examples** | **865 ch** | **~216** | **0/24** | **10/10** | **0.8333** | **0.9917** |
| cot (shipped) | 12,151 ch | ~3,037 | 0/24 | 10/10 | 0.8333 | 0.9974 |

- **baseline vs no-examples: 7/24 vs 0/24, Fisher exact p = 0.0094** — the shipped
  headline reproduced by a prompt costing **1/14th** as much.
- **cot vs no-examples: 0/24 vs 0/24, p = 1.0000.** instruction-only vs no-examples:
  1/24 vs 0/24, p = 1.0000.
- **Guard: 30/30 trials, 0 invented answers**, taking all four arms to 120/120. The
  refusal-suite zero is not bought by over-answering.
- **Precision@4 identical at 0.8333**, as it must be — retrieval is untouched by the
  prompt. The sanity check the harness exists to provide.

**The same fragility caveat as the shipped result applies, and is not weakened by
reproducing it.** 3 of baseline's 7 refusing questions do so on exactly one trial; drop
them and p = 0.1092. The binding constraint remains 24 questions (§7).

> **RESOLVED the same day — §12.** The question set was doubled to 46. baseline refuses
> 18/46, both candidate arms 0/46, p = 0.0000, and dropping all six one-trial refusers
> still leaves p = 0.0002. This caveat no longer applies to the standing result; it is
> kept here because it correctly described the evidence at the time §11 was written.

### What the parity result does and does not say

`p = 1.0000` against `cot` is a **null result on 24 questions**, and a null result cannot
tell equivalence from an underpowered test. §8 made exactly this point about
`instruction-only` and kept the exemplars anyway. That reasoning is not repeated here for
a specific reason: §8's stated ground for keeping them was that `instruction-only`
substitutes a neighbouring exemplar's question on 5/5 bursitis trials. **That mechanism is
absent here by construction, and the construction was verified rather than assumed.**

Head-to-head on bursitis — the question that refused 6/6 in Sprint 2 and 5/5 under
baseline:

- `baseline`: 5/5 flat refusals, 62-181 chars.
- `instruction-only`: 5/5 open by declining a question about **nosebleeds**, drawn from
  its own selected exemplar, before addressing bursitis. §8 reproduced exactly.
- `no-examples`: 5/5 answer about bursitis, 851-1,070 chars. The hedge names the **asked**
  topic ("does not explicitly define what bursitis is ... but describes").
- `cot`: 5/5 answer, 3 with no hedge at all.

So the exemplars' remaining justification was contamination-avoidance, and a 216-token
prompt achieves it by removing the mechanism instead of out-performing it.

### The instrument was re-validated on this arm, not trusted

A wording-keyed instrument has now failed to transfer to a new arm twice (§7, §8), and
this is a new arm. The stored booleans are not evidence for it; the text is.

- **0 label/instrument disagreements** across all 150 trials.
- **Only 5 of 120** refusal-suite trials opened the candidate gate — all five the bursitis
  "does not explicitly define" hedge, 573-712 chars surviving `delivered_content` against
  a 120 threshold. The case the `REFUSAL_WINDOW_CHARS` note already documents as correctly
  *not* a refusal.
- **The shortest answer in the arm is 293 chars** (median 1,026), so no short flat refusal
  is hiding behind wording the marker list cannot see — the failure mode that would have
  flattered this arm.
- **On the guard, the dangerous direction is the opposite** — an invented answer
  mislabelled as a refusal would make the guard look armed when it is not. Maximum
  surviving content across all 30 guard trials is **5 characters**. These are flat
  refusals that name what the corpus covers instead.
- Those guard refusals are phrased *"there is no information/mention of ..."* — the exact
  wording that defeated `REFUSAL_MARKERS` on `instruction-only` in §8. The union gate
  handles it. §8's fix is now validated on an arm it was not calibrated against, which is
  the test it previously failed.

**A contamination detector was written, rejected, and rewritten.** The first version
flagged `no-examples` on 27 trials — impossible by construction — because it matched
incidental word overlap (an allergic-rhinitis answer containing "allergic" and "reaction")
rather than a declined foreign question. Restricted to gate-open trials, where
contamination is actually visible, `no-examples` names a foreign topic **zero** times. The
discarded version is recorded because it is the same error shape as F8: a screen that
accepts everything is not a screen.

### Verbosity: the obvious way a ratio metric could be gamed

Claim-level groundedness is supported claims / total claims, so a terse arm can score well
by saying less. It is not what happened. Mean answer length over the 120 refusal-suite
trials: baseline 752, instruction-only 944, cot 1,053, **no-examples 1,267** — the longest
of the four. It says more and remains grounded.

### The single sub-1.0 answer is a judge artefact, like cot's

`What is bulimia nervosa?` scored 0.8 (4 of 5 claims supported). The unsupported "claim" is
the model's own trailing meta-caveat about what the context does not cover — a statement
about the context, not a medical assertion — and all four substantive claims about bulimia
are supported. Structurally the same as cot's one sub-1.0 case, where the judge scored the
misspelling "seizers" as an unsupported claim.

So the 0.9974 vs 0.9917 gap is one meta-caveat, not a difference in medical grounding, and
both arms are effectively 24/24. **This is further evidence for F6 — blinded human
calibration of the judge — which remains open and which I still cannot close, being the
model under audit.**

### Gates added

The ablation files carried the exemplar decision and had **no gate at all**: §9 found every
gate pointing at the retired dataset and fixed the headline ones, leaving these two
uncovered. Now gated in `tests/test_eval_regression.py`, and each mutation-tested — flip a
refusal-suite trial to REFUSED, flip a guard trial to an invented answer, truncate a cell
5 to 2, drop an arm from a guard question — every mutation failed exactly the intended test
and the data restored bit-identical afterwards.

### Shipped, on Melvin's call, after the numbers existed

`DEFAULT_PROMPT_VARIANT` moved from `cot` to `no-examples` on 2026-08-03. The ordering is
the point: measured first, shipped second. Sprint 3 exists to have avoided the reverse.

The decision was made on the cost difference, not on out-performing `cot` — nothing here
shows it is better, and parity rests on a null result at n=24. What it shows is
equivalence on every axis measured plus a 14× token saving, and that the one reason §8
gave for keeping the exemplars no longer applies.

**The test that enforced "unmeasured prompts do not ship" was rewritten rather than
flipped.** It asserted `DEFAULT_PROMPT_VARIANT == "cot"`, which detects a change but not
the mistake — it would pass on any rename and fail on a well-measured replacement. It now
asserts the property: whatever the app serves must (1) have recorded claim-level results
covering the whole eval set, and (2) never falsely refuse a question the corpus answers.
(2) is load-bearing on its own — `baseline` has complete recorded results and refuses
7/24, so (1) alone would wave it through. Mutation-tested against `baseline` (fails on the
refusals), `examples-only` (fails as unmeasured, with the command to measure it), and
`cot` (passes, still a legitimate alternative).

The residual risk, stated plainly: the arms are equivalent *as measured*, and the measuring
instrument is 24 questions. If the expansion to 46 shows `cot` ahead, this decision should
be revisited — the recorded artefacts for both arms make that a re-read, not a re-run.

### Two harness defects found while running this, not fixed here

- **`refusal_trials.run()` can destroy recorded data on a mid-run failure.** It builds its
  output dict from scratch and checkpoints after every variant, so a run that dies partway
  writes back a file containing only the questions it reached — silently dropping the other
  arms' cells for every question it did not. This run put 72 already-paid-for cells at
  risk; they were backed up manually first. `--resume` reads the file it can truncate.
- **`run_eval` has no `preflight()` and its per-question progress `print` lacks
  `flush=True`** — both fixed in `refusal_trials` by §6's second pass, neither carried
  across. A quota-dead `run_eval` still looks exactly like a slow one, which is the failure
  mode §6 spent a session diagnosing.

---

## 12. The question set doubled, and the fragility warning is gone (2026-08-03)

The 22 candidates screened in §10 were merged into `EVAL_QUESTIONS`, taking the eval set
from 24 to 46, together with the 330 calls that re-measured the refusal suite over all of
it. Merging without that run is what the (now replaced) sequencing test prevented.

**Why this and not more trials.** §7 established the binding constraint was the number of
questions: going 3 → 5 trials cost 240 calls and moved the raw p-value by 0.0007, while
leaving the fragility warning firing. §11 then hit the same wall from the other direction —
`cot` vs `no-examples` parity was a null result at n=24 that could not distinguish
equivalence from an underpowered test, and the shipped default was changed on it.

### Result — 46 questions × 3 arms × 5 trials

| comparison | questions ever refusing | Fisher exact p |
|---|---|---|
| baseline vs **no-examples** | 18/46 vs **0/46** | **0.0000** |
| baseline vs **cot** | 18/46 vs **0/46** | **0.0000** |
| cot vs **no-examples** | 0/46 vs **0/46** | 1.0000 |

Trial-level: baseline **55/230**, cot **0/230**, no-examples **0/230**.

**The fragility warning is resolved, and that is the headline.** It has qualified every
refusal number since Sprint 4. At n=24, dropping the three questions that refused on
exactly one trial took p from 0.0094 to **0.1092 — not significant**. At n=46, six
questions refuse once; dropping all six leaves **p = 0.0002, still significant**. The
result no longer depends on its weakest observations. §7 predicted exactly this, and the
prediction is now tested rather than asserted.

**The n=24 caveat attached to §11's ship decision is substantially discharged.** Doubling
the question set found more than twice as many baseline refusals (7 → 18) and still zero in
either candidate arm, across 230 trials each. The parity is now a null on twice the
evidence. It remains a null — no experiment shows `no-examples` is *better* than `cot` —
but the specific worry, that n=24 was too small to see a real difference, has been tested
and did not materialise.

### The new questions are harder, which is why they were worth buying

Four of the 22 refuse **5/5** under baseline: bad breath, bone marrow transplant, barium
enema, alopecia. Read side by side, these are textbook instances of the bug this project
exists to measure — the same retrieved context, one arm declining it in 54 characters:

- `baseline`: *"I don't know the answer based on the provided context."* (54 ch)
- `no-examples`: 717 characters correctly describing stem-cell extraction from a healthy
  donor and transfer to a recipient.

The original 24 had a baseline refusal rate of 7/24 (29%); the 22 new ones run 11/22 (50%).
Not because they are unfair — every one was screened by `verify_entry` against the live
index, and the set's recorded Precision@4 (0.8523) is *higher* than the original's
(0.8333) — but because the screen selected entries that exist without selecting for the
paraphrase-friendliness the original set happened to have.

### The instrument was re-validated on all 22 new questions

220 trials with zero refusals across two arms, on questions the instrument had never seen,
is precisely what a third `is_refusal` failure would look like (§7, §8). So the text was
read, not the booleans:

- **0 label/instrument disagreements** across all 330 new trials.
- Candidate gate opened **40/110 baseline, 6/110 no-examples, 0/110 cot**.
- The short tail is where a missed refusal would hide, so it was inspected directly.
  `cot`'s shortest new answer is 74 characters — *"Byssinosis is caused by inhaling
  particles of cotton, flax, hemp, or jute"* — which is correct and complete; the
  encyclopedia entry is simply terse. `no-examples` has five identical 110-character
  arteriography answers, same story.
- The 360 seeded trials are byte-identical to the `ablation_t5_` file they came from.

### The frozen 24 stay addressable, and the pre-merge files are not "truncated"

`EVAL_QUESTIONS_V1` holds the original 24. Every trial file recorded before this merge
covers exactly those questions and always will — they are complete records of the suite as
it stood, not partial records of the new one. Their coverage gates are pinned to
`EVAL_QUESTIONS_V1` rather than to `REFUSAL_QUESTIONS`, so growing the eval set does not
retroactively fail a finished dataset. The merged list keeps the original 24 first and in
order, pinned by a test, because trial files are keyed by question text.

### Three arms, not four

`instruction-only` was not extended. §8 settled the question it existed to answer, and it
is the arm that substitutes a neighbouring exemplar's question, so 110 calls to carry it
forward buys nothing. It stays at 24 questions in `ablation_t5_refusal_trials.json`, which
keeps its own four-arm gate against the frozen list.

### What this run did NOT measure, stated rather than implied

- **Claim-level groundedness over the 22 new questions.** `run_eval` iterates all of
  `EVAL_QUESTIONS` with no subset filter and no resume, so extending it costs **138 calls**
  (92 + 46) to re-derive 24 values already recorded. The shipped-default gate therefore
  asserts claim coverage against the frozen 24 and says so in a comment. **This is the open
  item.** Until it runs, "claim-level 0.9917" describes the original 24 only.
- **The out-of-corpus guard.** Deliberately not re-run: it is a separate 10-question suite
  that does not change when the eval set grows, and all four arms are recorded at 30/30
  (120/120 combined). `refusal_stats` says so explicitly when reporting the expanded
  prefix, rather than silently falling back.

### Gates

`test_neither_shipped_candidate_falsely_refuses_at_n46` covers both `cot` and
`no-examples` — a documented fallback nobody measures is not a fallback.
`test_the_expansion_did_not_weaken_the_baseline_gap` pins the *property* the 330 calls
bought (the effect survives dropping every one-trial refuser) rather than a p-value, which
would fail on a harmless re-measure. Both mutation-tested, along with the completeness gate:
flip a `no-examples` trial to refused, flatten baseline to never refusing, decay every
baseline refusal to exactly one trial, truncate a cell — each failed exactly the intended
test, and the data was restored bit-identical.
