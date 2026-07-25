# Sprint 3 — end-to-end audit

Post-implementation audit of the chain-of-thought prompting change, covering the code, the
experimental method, and the validity of the claims made in `results_sprint3.md`. Written
adversarially against my own work: the goal was to find reasons the reported result might be
wrong or overstated, not to confirm it.

**Overall assessment:** the change is safe to ship and the behaviour improvement on the two
target questions is real and reproducible. The *statistical* framing in the first draft of
`results_sprint3.md` was overstated and has been corrected. The headline result is not
attributable to chain-of-thought reasoning specifically, because three variables changed at once.

Scope audited: `medbot/prompt.py`, `medbot/query_handler.py`, `medbot/model_handler.py`, `app.py`,
`medbot/eval/run_eval.py`, `medbot/eval/refusal_trials.py`, and the four result artefacts.

---

## A. What was verified as correct

These were checked, not assumed.

| # | Check | Method | Result |
|---|---|---|---|
| A1 | Retrieval genuinely unchanged between arms | Compared Precision@4 per question across both result files | Identical on all 24. Confirms the harness isolates the prompt, and that the 0.83/0.83 match is a real invariance rather than a copy-paste artefact |
| A2 | No orphaned call sites bypassing trace-stripping | Grepped every use of `create_query_chain`, `chain.invoke`, `run_query`, `strip_reasoning` | All three consumers (app, `run_eval`, `refusal_trials`) route through `run_query`. Only internal call to `chain.invoke` is inside `run_query` itself |
| A3 | No reasoning trace leaks to users | Scanned all 24 CoT answers for `Reasoning:` / "The question asks" openers | 0/24 leaks |
| A4 | Fix holds in the real app, not just the harness | `AppTest` against the live `app.py`, asking the 6/6-refusing bursitis question | Answers correctly; no leak; external sources still render |
| A5 | Exemplar questions held out of the eval set | Token overlap of all 6 exemplar questions against all 24 eval questions | Clean, with one caveat — see F5 |
| A6 | Trials are not a single deterministic response repeated | Counted distinct answer texts per question per variant | 5/5 distinct in 7 of 8 cells. Wording varies at temperature 0; the refusal *outcome* is what stayed constant |
| A7 | Strip logic handles real model output shapes | Unit cases incl. `**Answer:**`, no-marker passthrough, empty, multiline | Caught a genuine bug pre-deployment (`**` leaking into user text); fixed and covered |

---

## B. Findings

Severity: **High** = affects whether a stated conclusion is true. **Medium** = real defect or risk,
does not invalidate the result. **Low** = worth fixing, no current impact.

### F1 — Statistical claim was overstated (High) — *corrected*

The draft reported "10/20 vs 0/20 false refusals". Those 20 trials are 5 repeats of each of 4
questions, so they are repeated measures on 4 units, not 20 independent observations. Treating them
as independent gives p=0.00044; the correct question-level test gives **Fisher exact p=0.43, not
significant**.

The honest statement has two halves, and both matter:
- Within a question the effect is complete and repeatable (5/5 → 0/5, with varying wording).
- Across questions, n=4 with 2 changed is far too small for a population-level claim.

*Status:* `results_sprint3.md` corrected to report both levels and to state the non-significance
explicitly. **Recommendation:** expand the refusal suite to 15–20 questions before making any
general claim about refusal rates. A useful source of candidates: questions where Precision@4 is
high (retrieval succeeded) but groundedness is low — that pattern is close to a definition of the
false-refusal bug and could be mined automatically from `results_*.json`.

### F2 — The groundedness judge is effectively binary (High) — *fixed*

**Update:** `judge_groundedness_claims` (supported claims / total claims) was added and the stored
answers re-graded via `rejudge.py`. It discriminates: baseline now spans {0.0, 0.5, 0.75, 0.94, 1.0}
instead of {0.0, 1.0}. Claim-level means are **baseline 0.841, cot 0.990** (0.997 after the fix
round), versus 0.917/1.000 under the binary judge — so the old judge was overstating baseline quality by ~7.6 points, and Sprint 2's
0.83 headline was coarser than it claimed.

It immediately earned its keep, in both directions:

- It revised four baseline answers down from a perfect 1.00, including one that exposed **few-shot
  contamination in the baseline prompt**: the answer to "What is atherosclerosis?" opens "I don't
  know the answer to how osteoporosis is diagnosed…" because the semantic selector had picked the
  osteoporosis exemplar and the model bled its question into the answer. The binary judge scored
  that 1.00 and praised it.
- It found **a genuine regression in the shipped CoT arm** that the binary judge structurally could
  not see: "What causes bedsores?" falls 1.00 → 0.75 on an over-linked causal chain from moisture to
  bedsore development. So the honest per-question tally was 6 improved / 1 regressed, not the
  2 improved / 0 regressed the binary judge reported. That regression was then fixed (see the fix
  round in `results_sprint3.md`), bringing it to 6 improved / 0 regressed.

Residual: the claim-level judge is still the same model, and still unvalidated against human labels.
See the note in `results_sprint3.md` on its harsh bladder-cancer call.

---

*Original finding as filed:*

Across 48 graded answers the judge returned only 0.0 or 1.0 — never an intermediate value — despite
a 0–100 rubric. Consequences:

- "0.92 → 1.00" means precisely "two fewer zeros out of 24", not a broad quality gain.
- "0 regressions" is much weaker evidence than it sounds: the metric can only detect an answer
  flipping to *wholly* unsupported. Partial degradation — a CoT answer that is right but padded, or
  that over-hedges — is invisible to it.

This matters more in Sprint 3 than it did in Sprint 2, because the change under test increases
answer length by 32%, which is exactly the kind of quality drift this metric cannot see.

**Recommendation:** make the judge discriminate before relying on it for anything finer than
pass/fail. Cheapest first step is claim-level scoring — have the judge enumerate the answer's
distinct claims and mark each supported/unsupported, then score the ratio. That produces genuine
intermediate values and is a within-budget change to `groundedness.py`. Also worth adding a second
axis the current rubric ignores entirely: *completeness* (did the answer use what the context
offered), since a terse refusal-adjacent answer and a full one can both be perfectly "grounded".

### F3 — The intervention is confounded three ways (High) — *resolved by ablation*

**Update:** the ablation below was run. Result, refusals over 5 trials on the 4-question suite:
baseline **10/20**, instruction-only **5/20**, examples-only **1/20**, cot **0/20**.

The confound is now resolved, and it resolved *against* the hypothesis stated in this finding.
Breast cancer is fixed by either factor alone, but bursitis — the case that refused 6/6 in Sprint 2
— is completely unmoved by the instruction rewrite (5/5, identical to baseline) and needs the worked
exemplars. So the exemplars are not redundant with the wording, the two are not substitutes, and
only the combination reaches zero. The ~2,400 tokens per query buy the hard case rather than
duplicating a cheaper fix.

`results_sprint3.md` now carries the full table. The original finding is preserved below as written,
since the reasoning that motivated the ablation is worth keeping.

---

*Original finding as filed:*

The "CoT arm" changed three things simultaneously:

1. **Instruction text** — `COT_DISCLAIMER` adds "judge on meaning, not matching phrasing" and
   "partial coverage still counts", directly targeting the bug.
2. **Example format** — question → context → reasoning → answer, instead of question → answer.
3. **Example selection** — 6 fixed exemplars, instead of 1 semantically selected from 27.

Any one of these could account for the entire effect. In particular, hypothesis (1) is cheap and
plausible: simply telling the model that paraphrased support counts might fix breast cancer and
bursitis with no reasoning step at all. The sprint therefore demonstrates *that the new prompt is
better*, not *that chain-of-thought is why* — and the sprint was framed as a chain-of-thought
upgrade.

**Recommendation (now carried out):** run two ablations against the same 4-question refusal suite,
~40 calls and around 10 minutes each. This is the highest-value follow-up in this list:

| Arm | Examples | Instruction | Isolates |
|---|---|---|---|
| `baseline` | 1 semantic Q→A | old | (control) |
| `instruction-only` | 1 semantic Q→A | new COT_DISCLAIMER | whether wording alone fixes it |
| `examples-only` | 6 fixed CoT | old disclaimer | whether the exemplars carry it |
| `cot` | 6 fixed CoT | new | (shipped) |

If `instruction-only` recovers most of the gain, the six exemplars and their ~2,400 extra tokens per
query are not paying for themselves and should be trimmed. *(It recovered half — and none of the
hard case. The exemplars stay.)*

### F4 — `run_eval` could silently overwrite Sprint 2's recorded results (Medium) — *fixed*

`result_paths(None)` returned the unsuffixed `results.json` / `results.md`. Since the app default is
now `cot`, a plain `python -m medbot.eval.run_eval` would have written CoT numbers into the file that
documents the Sprint 2 baseline — destroying the historical record under a filename still claiming to
be it. Nobody had run it that way yet, so no data was lost.

*Status:* fixed. `result_paths` now resolves `None` to the active variant and always suffixes, so the
Sprint 2 artefacts cannot be reached by the harness at all.

### F5 — Minor exemplar/eval content overlap (Medium)

No exemplar *question* appears in the eval set. But the anorexia exemplar's context block contains
bulimia's warning signs, and the eval set asks "What is bulimia nervosa?". So the prompt does carry
some content relevant to a scored question, and "fully held out" overstates it.

Observed impact appears to be none — both arms scored 1.00 on that question, so it did not create the
delta.

**Decision: not fixed in this sprint, deliberately.** Editing that exemplar changes the `cot` prompt
string, which would invalidate every number in `results_cot.json` and the ablation, and require
re-running roughly 90 calls to restore them — to correct an overlap with no measured effect. Trading
a real, recorded result for a cosmetic fix is the wrong way round. The overlap is documented in
`results_sprint3.md`, and `tests/test_prompt_variants.py` now pins the prompt hash so the swap cannot
happen accidentally without the re-run being noticed.

**For the next dataset revision:** either swap that exemplar's context for a non-overlapping
retrieval or drop bulimia from the eval set, and re-run the A/B in the same change. Standing rule
going forward: new exemplars get checked against `dataset.py` for *content* overlap, not just
question overlap.

### F6 — Possible judge bias toward the treatment arm (Medium)

CoT answers are 32% longer and contain explicit hedges of the exact form the judge is asked to
assess ("the context does not provide a formal definition, but it describes..."). A self-grading
judge plausibly rewards that. The direction of this bias favours the conclusion I drew, which is
the worst direction for it to run.

**Recommendation:** cannot be resolved with the same model as judge. Two options within constraints:
grade a blinded sample by hand (24 answers is tractable in one sitting) to calibrate the automated
judge, or shuffle arm order and strip formatting cues before judging. Longer term this is the
strongest argument for an independent judge model.

### F7 — The over-answering guard is thin (Medium)

After removing the stroke question (F8), the guard rests on 2 questions × 3 trials per arm. Both
arms refused 6/6, so there is no evidence of a hallucination regression — but 2 questions is not
enough to *establish* its absence, and this is the guard protecting against the failure mode the
sprint's own incentives push toward.

**Recommendation:** raise to 8–10 verified out-of-corpus questions, each confirmed absent by
inspecting retrieved chunks rather than assumed absent from its first letter (the mistake in F8).
This should be a permanent regression gate before any future prompt change ships, and belongs in
Sprint 4's pytest work.

### F8 — A test question was mis-specified, and it invalidated a Sprint 2 heuristic (Low, but informative)

"What causes a stroke?" was included as out-of-corpus on the assumption that an A–B corpus cannot
cover an S topic. Wrong: stroke causation is covered inside the **A** entries for embolism and
atherosclerosis. Both arms answered correctly from real context (cot 3/3, baseline 2/3).

Two consequences worth keeping:
- The question was removed for failing its inclusion criterion, not for its result. Its data remains
  in `overanswer_trials.json` and the reason is recorded in `refusal_trials.py`.
- **It corrects Sprint 2's documented conclusion.** "Corpus is A–B only, so C–Z questions retrieve
  badly" is too strong; some C–Z topics are discussed inside A–B entries. Coverage must be verified
  per question. This affects how any future retrieval complaint should be triaged.

### F9 — Refusal detection was an unvalidated heuristic (Low) — *validated*

`is_refusal` matches marker phrases in the first 200 characters. The window is deliberate — it stops
a correct partial answer that names a gap in its closing sentence from being scored as a refusal —
but the threshold was chosen by judgement and had never been checked against human labels.

**Done:** all 60 stored trial texts (34 distinct) were hand-labelled and compared against the
heuristic. **Zero disagreements.** The closest call was a CoT answer opening *"The provided context
does not explicitly define what bursitis is, so I cannot give a formal definition based on it.
However, the text notes that it can flare up..."* which then supplies causes and symptoms — correctly
labelled not-a-refusal.

That case is also why "cannot give" and "unable to" were deliberately left out of the marker list:
they occur inside genuine partial answers, so adding them would turn a correct label into a false
positive. Residual gap, now recorded in `refusal_trials.py`: a true refusal phrased with an unlisted
verb would be scored as an answer, and that error would flatter the CoT arm. Re-validate if the
marker list or the prompt wording changes.

### F10 — Prompt cost roughly 6× (Low)

The CoT prompt is ~3,050 tokens versus a few hundred, on every query, plus 32% longer outputs. Fine
against the 500 requests/day free tier, but it consumes the daily eval budget faster and adds
latency to every user turn. If F3's ablation shows the instruction change carries most of the
benefit, most of this cost is removable.

### F11 — No automated tests (Low) — *partly addressed*

**Done:** `tests/test_prompt_variants.py` now covers the two things whose failure would be
expensive and silent: the `strip_reasoning` cases (including the `**Answer:**` bug that was found
during this sprint), and a **pinned sha256 of the rendered `cot` prompt**, so the shipped prompt
cannot drift away from the string the recorded numbers describe without the check failing. It also
asserts that every variant renders both placeholders with no unescaped braces, that
`emits_reasoning` agrees with how each prompt ends, and that the guidance text is shared verbatim
across arms — the property the ablation's validity depends on.

pytest is not yet a project dependency (that is Sprint 4), so the file is written as `test_*`
functions with a plain-python runner: `python -m tests.test_prompt_variants` works today, and
pointing pytest at the directory will work in Sprint 4 with no rewrite. What remains for Sprint 4:
the eval-regression test asserting CoT refusals stay at 0/20, and unit tests for config and
external-search.

---

*Original finding as filed:*

Everything here was verified by one-off scripts, two of which lived in a scratch directory and are
now gone. `refusal_trials.py` was committed specifically to avoid repeating Sprint 2's mistake of
keeping the data but not the code that produced it — but `strip_reasoning`, the highest-risk pure
function in the change, still has no committed test despite a real bug having been found in it.

**Recommendation:** this is Sprint 4's remit and shouldn't expand Sprint 3's scope, but the
`strip_reasoning` cases from the smoke script are worth porting into the first pytest file, along
with a regression test asserting CoT refusals stay at 0/20.

---

## C. Status of the recommendations

| | Recommendation | Status |
|---|---|---|
| F3 | Ablations to attribute the improvement | **Done** — exemplars vindicated; instruction alone does not fix the hard case |
| F2 | Claim-level groundedness judge | **Done** — discriminates, and found a regression the binary judge could not |
| F4 | Stop `run_eval` clobbering Sprint 2's results | **Done** — output always variant-suffixed |
| F9 | Validate the refusal heuristic | **Done** — 60 texts hand-labelled, zero disagreements |
| F11 | Pin the prompt and cover `strip_reasoning` | **Done** — `tests/test_prompt_variants.py`, incl. sha256 prompt pin |
| F5 | Exemplar/eval content overlap | **Reduced** — bulimia list trimmed to one clause in the fix round; not eliminable without losing the exemplar's lesson |
| — | Bedsores over-linking regression | **Fixed** — anti-over-linking clause; bedsores 0.75 → 1.00, no new regression |
| F10 | Prompt token cost | **Accepted** — F3 showed the tokens buy the hard case |
| F7 | Out-of-corpus guard too thin | **Partly closed** — now 10/10 at 5 trials, up from 6/6 at 3; still only 2 questions |
| F1/F7 | Expand question sets | **Open** — the main remaining weakness |
| F6 | Blinded human calibration of the judge | **Open** |

Remaining order of work: F1/F7 suite expansion first (it is what turns this from two anecdotes into
a regression gate), then F6.

F6 gained a concrete motivating example in the fix round: the only sub-1.0 answer left in the CoT arm
scores 15/16 because the model wrote "seizers" for "seizures" and the judge counted the misspelling
as an unsupported claim. The judge is grading orthography as if it were factual support, which a
human calibration pass would have caught.

## D. Bottom line

Ship it, with the numbers stated correctly.

What is solid: the refusal fix is real, reproducible, verified in the live app, and now *attributed*
— the ablation shows the worked exemplars do work the instruction rewrite cannot, so the sprint's
stated thesis holds rather than being an artefact of three bundled changes. The failure mode the
change could have introduced was tested for rather than assumed away.

The one regression this audit found was fixed rather than filed: the bedsores over-linking is gone
(0.75 → 1.00) and the re-run produced no new regression.

What must not be overstated:

- **Quote the claim-level numbers, 0.84 → 1.00, not the binary 0.92 → 1.00.** The binary judge was
  overstating baseline quality and cannot see partial degradation. The two happen to agree on the
  post-fix CoT arm; they disagree sharply on the baseline, which is the point.
- **It is still 24 questions and a 2-question refusal delta**, Fisher p=0.43 at the question level.
  Large, clean, reproducible effect; small sample. Both halves are true and neither should be
  dropped when this gets summarised.
- **The judge itself is uncalibrated.** It counts a misspelling as an unsupported claim. Treat
  individual scores as directional.
