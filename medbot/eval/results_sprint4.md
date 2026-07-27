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

110 tests: **106 offline** (~6s, no network, no quota) and **4 live**.

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
- The hallucination guard is armed: 60/60 across all 10 questions (§3).
- 106 offline tests, and the network guard makes a whole class of fake test impossible.
- The measurement instrument itself is now tested against the two patterns that defeated
  it, and re-scoring is a tool rather than a manual pass (§7).

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
