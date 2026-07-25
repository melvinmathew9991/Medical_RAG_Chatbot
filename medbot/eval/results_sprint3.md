# MEDBOT Sprint 3 — Chain-of-thought prompting, A/B results

Sprint 2 found the model falsely refusing 4 of 24 questions ("I don't know the answer based on
the provided context") when the retrieved context plainly contained the answer, and showed that
temperature alone would not fix it. Sprint 3 rewrote the few-shot prompt into a
question → context → reasoning → answer form and measured whether that helped.

Hand-written summary; everything it cites is generated. Artefact inventory, since there are both
pre-fix and post-fix runs on disk:

| File | What it is |
|---|---|
| `results_baseline.{json,md}` | Baseline arm, full 24-question eval. Still current — the baseline prompt was never changed |
| `results_cot.{json,md}` | CoT arm, full eval. **Post-fix** (re-run after the fix round) |
| `results_{baseline,cot}_claims.json` | Claim-level re-scoring of the above, via `rejudge.py` |
| `refusal_trials.json` / `overanswer_trials.json` | Pre-fix, baseline vs cot, 5 and 3 trials |
| `fixed_refusal_trials.json` / `fixed_overanswer_trials.json` | **Post-fix**, cot only, 5 trials each |
| `ablation_refusal_trials.json` | Four-arm ablation. Pre-fix prompt — see the caveat at the end of the fix-round section |
| `results.{json,md}` | Sprint 2's original run. Historical; nothing here overwrites it |

## Headline

| | Baseline prompt | CoT prompt |
|---|---|---|
| Mean Precision@4 | 0.83 | 0.83 |
| Mean groundedness (binary judge) | 0.92 | 1.00 |
| **Mean groundedness (claim-level judge)** | **0.84** | **0.99** |
| False refusals, single pass | 2 / 24 | 0 / 24 |
| Questions refusing at all (of 4 retested) | 2 / 4 | 0 / 4 |
| Trials refusing (5 per question) | 10 / 20 | 0 / 20 |

Per-question groundedness under the binary judge: 2 improved, 0 regressed, 22 unchanged. Under the
claim-level judge, which can see partial degradation: 6 improved, 1 regressed, 17 unchanged — and
after the fix round below, **6 improved, 0 regressed**.

## Claim-level groundedness (audit finding F2)

The original judge returned only 0.0 or 1.0 on all 48 answers despite a 0-100 rubric — it was a
pass/fail check wearing a continuous scale, and could not detect an answer getting *partly* worse.
`judge_groundedness_claims` replaces the single score with supported-claims / total-claims. Stored
answers were re-graded rather than regenerated (`rejudge.py`), holding the answers fixed so only the
measurement changes.

| | Binary judge | Claim-level judge |
|---|---|---|
| Baseline mean | 0.917 | **0.841** |
| CoT mean | 1.000 | **0.997** |
| Distinct values, baseline | {0.0, 1.0} | {0.0, 0.5, 0.75, 0.94, 1.0} |

(CoT figures are the post-fix re-run — see "Fix round" below. The pre-fix CoT arm scored 0.990.)

It works: five distinct values instead of two, and it revised four baseline answers *down* from a
perfect 1.00 (atherosclerosis 0.50, bladder cancer 0.00, angina 0.75, autism 0.94). **Sprint 2's
0.83 headline was measuring something coarser than it claimed, and the binary judge was overstating
baseline quality by roughly 7.6 points.**

Two things it caught that matter:

- **Few-shot contamination in the baseline prompt.** The baseline answer to "What is
  atherosclerosis?" opens *"I don't know the answer to how osteoporosis is diagnosed based on the
  provided context."* The semantic selector had chosen the osteoporosis exemplar and the model bled
  that exemplar's *question* into its answer. The binary judge scored this 1.00 with an approving
  rationale. This is an independent argument for the CoT arm's fixed example set, and a live defect
  in the baseline prompt that nothing in Sprint 2 could see.
- **A real regression in the CoT arm.** "What causes bedsores?" dropped 1.00 → 0.75 (3/4 claims).
  The CoT answer linked moisture to skin infection to bedsore development, but the context supports
  only part of that chain — a small inferential overreach, exactly the kind of drift a verbosity-
  increasing prompt risks and a binary judge cannot register. **Since fixed — see below.**

## Fix round (post-audit)

The regression above was fixed rather than merely recorded. Two prompt changes, then the whole CoT
arm was re-run against the new string (the baseline arm was untouched, so it did not need re-running):

1. **Anti-over-linking clause** added to `COT_DISCLAIMER`: *"State each supported fact on its own
   terms: do not join two separate statements from the context into a cause-and-effect chain unless
   the context itself asserts that link."*
2. **Anorexia exemplar context trimmed** (audit F5) — the bulimia symptom list cut from six items to
   a one-clause characterisation, reducing content overlap with the eval question "What is bulimia
   nervosa?".

| Metric | Pre-fix CoT | Post-fix CoT |
|---|---|---|
| Claim-level groundedness | 0.990 | **0.997** |
| Answers at 1.00 | 23/24 | 23/24 |
| Bedsores (the regression) | 0.75 (3/4 claims) | **1.00 (18/18 claims)** |
| False refusals | 0/20 | 0/20 |
| Out-of-corpus refusals (guard) | 6/6 @3 trials | **10/10 @5 trials** |
| Precision@4 | 0.83 | 0.83 |

The fix worked and cost nothing measurable elsewhere: no new regression, refusals still zero, the
over-answering guard still perfect on a larger trial count. Bedsores went from 4 claims to 18 — the
clause made the model enumerate risk factors separately instead of welding them into a causal chain,
which is exactly the intended behaviour.

**The one remaining sub-1.0 answer is a judge-calibration artefact, not a grounding failure.** "What
are the symptoms of a brain tumor?" scores 15/16 because the model wrote *"seizers"* instead of
*"seizures"* (confirmed: the answer contains zero instances of the correct spelling) and the judge
counted the misspelling as an unsupported claim. That is a real but trivial output defect, and a
claim-level judge should be assessing factual support rather than orthography. It is concrete
evidence for why F6 — human calibration of the judge — is still worth doing.

**Caveat on the ablation:** the four-arm ablation was run against the *pre-fix* prompt string. Its
conclusion is about the design (instruction wording vs worked exemplars) and is unaffected by a
clause added to the disclaimer afterwards, but the exact strings it compared are the pre-fix ones.
Re-running it was not judged worth ~40 further calls against a daily quota to re-confirm an
attribution the fix does not bear on.

Worth noting against the verbosity concern: mean claims per answer barely moved (7.7 → 8.2), so the
+32% character count is mostly elaboration and hedging rather than a flood of new assertions.

**Read the trial row carefully.** The 20 trials are 5 repeats each of 4 questions, so they are
repeated measures, not 20 independent samples — quoting "10/20 vs 0/20, p<0.001" would be
pseudo-replication. The defensible reading has two parts:

- *Within* a question the effect is total and repeatable: breast cancer and bursitis refused on
  5 of 5 baseline trials and 0 of 5 CoT trials, with the wording varying between trials (the model
  is not fully deterministic even at temperature 0), so this is not one lucky sample.
- *Across* questions the sample is tiny: n=4, of which 2 changed. Fisher's exact at the question
  level gives **p=0.43 — not significant.** Two questions moved from always-refusing to
  never-refusing, and nothing regressed; that is a real, reproducible behaviour change on those
  questions, not a demonstrated population-level improvement.

See `sprint3_audit.md` for the full audit, including the confound that stops this being attributable
to chain-of-thought reasoning specifically.

Precision@4 is identical in both arms, which is the expected result and a useful sanity check —
Sprint 3 changed only the prompt, not retrieval, so any movement there would have indicated a
measurement bug rather than an improvement.

## Attribution: which part of the change did the work

The first draft of this document could not say *why* the CoT arm was better, because it changed
three things at once (audit finding F3). A four-arm ablation on the refusal suite, 5 trials each,
separates them. All four arms share the same retrieval, the same temperature, and — where the
guidance wording is present at all — the same `CONTEXT_JUDGEMENT_GUIDANCE` string, so only one
factor moves at a time.

| Question | baseline | instruction-only | examples-only | cot |
|---|---|---|---|---|
| What causes bedsores? | 0/5 | 0/5 | 0/5 | 0/5 |
| What are the symptoms of breast cancer? | 5/5 | **0/5** | **0/5** | **0/5** |
| What is bursitis? | 5/5 | 5/5 | 1/5 | **0/5** |
| What is an abscess? | 0/5 | 0/5 | 0/5 | 0/5 |
| **Total refusals** | **10/20** | 5/20 | 1/20 | **0/20** |

- `instruction-only` = new guidance wording, but baseline's single Q→A example and no reasoning step.
- `examples-only` = the six CoT exemplars under the *original* disclaimer — reasoning demonstrated
  rather than instructed.

Two distinct failure modes, not one:

- **Breast cancer is the easy case.** Either change fixes it alone. Simply telling the model that
  paraphrased support counts is enough.
- **Bursitis is the hard case, and the instruction rewrite does nothing for it** — still 5/5
  refusals, identical to baseline. Only the worked exemplars move it (1/5), and only the full
  combination eliminates it (0/5).

This answers the question the audit raised, and answers it against the audit's own hypothesis: the
six exemplars are not redundant with the instruction wording, and are not merely paying for a result
that cheaper prompting would have delivered. The ~2,400 extra tokens per query buy the hard case.
The instruction and the exemplars are also not substitutes — each fixes something the other does
not, and only together do they reach 0/20.

Caveat carried over from F1: this is still 4 questions. The attribution is clean, but the sample is
small, and "bursitis is hard, breast cancer is easy" is a claim about two questions.

## How the A/B was run

Both arms were run fresh, in the same session, at `temperature=0`, against the same 24-question
set — the baseline arm was **not** compared against Sprint 2's recorded 0.83/0.83 numbers. Those
were taken at `temperature=0.1`, so comparing against them would have confounded the prompt change
with the temperature change. Re-running the baseline isolates the prompt as the only variable.

Reproduce with:

```
.venv-gemini/Scripts/python.exe -m medbot.eval.run_eval --variant baseline
.venv-gemini/Scripts/python.exe -m medbot.eval.run_eval --variant cot
.venv-gemini/Scripts/python.exe -m medbot.eval.refusal_trials --trials 5
# ablation arms (writes ablation_*.json so the main trial data is not overwritten)
.venv-gemini/Scripts/python.exe -m medbot.eval.refusal_trials --trials 5 --suite refusal \
    --variants instruction-only,examples-only --out-prefix ablation_
# claim-level re-scoring of the stored answers
.venv-gemini/Scripts/python.exe -m medbot.eval.rejudge --variants baseline,cot
```

## Splitting the credit between temperature and prompt

Sprint 2's four refusing questions were bedsores, breast cancer, bursitis, and abscess. The two
changes fixed different ones, and the honest split is:

- **Temperature 0.1 → 0 fixed two**: bedsores and abscess. Both answer correctly under the
  *baseline* prompt at temperature 0 (0/5 refusals each), so the CoT prompt gets no credit here.
- **The CoT prompt fixed the other two**: breast cancer and bursitis refused **5/5 under the
  baseline prompt at temperature 0** — deterministic, not sampling noise — and 0/5 under CoT.

The 5-trial design was the point of the exercise. Sprint 2 had flagged that these questions refuse
intermittently (bedsores refused in the harness run, then 0/6 in the temperature experiment), so a
single A/B pass could not have distinguished a real improvement from variance. At 5/5 vs 0/5 on two
questions, it can.

Qualitatively, bursitis is the clearest win. It refused in all 6 trials at both temperatures in
Sprint 2. Under CoT it now answers:

> While the context does not provide a formal definition, it describes bursitis as a condition that
> can flare up for no known reason or be caused by repeated physical activity, trauma, rheumatoid
> arthritis, gout, and acute or chronic infection...

That is the trained partial-coverage behaviour working as intended — answering the part the context
covers while naming what it lacks, instead of refusing the whole question.

## Guarding against the obvious way to cheat this metric

Driving false refusals to zero is trivially achievable with a prompt that never refuses, which would
trade a refusal bug for a hallucination bug — worse, on medical questions. Two things guard against
that:

1. **A negative exemplar in the prompt.** One of the six CoT examples asks for anorexia nervosa
   symptoms, where retrieval genuinely returns *bulimia's* symptoms plus anorexia risk factors. The
   worked reasoning notices the mismatch and declines. It teaches discrimination, not compliance.
2. **An out-of-corpus trial suite** (`--suite overanswer`), on questions the corpus does not cover,
   where refusing is the correct answer. Both variants refused **6/6** (diabetes 3/3, psoriasis 3/3).
   No over-answering regression.

## Correction to a Sprint 2 conclusion

The over-answering suite originally included "What causes a stroke?", on the assumption that an A–B
corpus cannot cover an S topic. That assumption was wrong. Stroke causation is covered inside the
**A** entries for embolism and atherosclerosis, and both variants answered from them correctly
(cot 3/3, baseline 2/3, all citing embolus and carotid artery blockage). The question was removed
because it failed its own inclusion criterion, not because of the result — its numbers are recorded
here and its raw trials remain in `overanswer_trials.json`.

The correction matters beyond this test: Sprint 2 recorded "the corpus is A–B only, so C–Z questions
will retrieve badly." That is too strong. Some C–Z topics *are* discussed inside A–B entries.
Coverage should be verified per question, not inferred from the first letter.

## Limitations

- **The judge is still self-grading.** Groundedness is scored by the same Gemini model that produced
  the answer. A mean of 1.00 should be read as "the judge found no unsupported claims", not as
  "the answers are perfect". This bias is unchanged from Sprint 2 and an independent judge model
  remains out of reach under the project's free-tier constraint.
- **The original judge was effectively binary** — fixed, see the claim-level section above. The
  binary numbers are retained in the headline table only for continuity with Sprint 2; the
  claim-level figures are the ones to quote.
- **The claim-level judge is itself unvalidated against human labels.** It is stricter, and it
  found real defects, but its bladder-cancer call (baseline 1.00 → 0.00) is arguably harsh: the
  answer's substance — that the exact cause is unknown — does match the context, and it was
  penalised for framing that as "the context does not state" it. Defensible, since the answer also
  dropped the smoking risk factor the context supplies, but a human grader might score it above
  zero. Treat individual claim-level scores as directional, not authoritative.
- **CoT answers are 32% longer** (781 → 1034 mean characters). A self-grading judge may reward the
  extra explicit hedging ("the context does not provide a formal definition, but...") that the CoT
  prompt asks for. This remains an unmeasured possible bias in favour of the treatment arm, though
  the near-flat claims-per-answer count (7.7 → 8.2) argues the extra length is not extra assertions.
- **Small n.** 24 questions, 5 trials on 4 of them, 3 trials on 2 out-of-corpus questions. Enough to
  call a 5/5 → 0/5 change real; not enough for a confident estimate of the overall refusal rate.
- **Refusal detection is a heuristic** — marker phrases in the first 200 characters of the answer
  (`refusal_trials.py`). The window exists so that a CoT answer which correctly names a gap in its
  closing sentence is not miscounted as a refusal. It has not been validated against human labels.
- **The eval set is unchanged from Sprint 2**, so it still tests A–B topics only, and the two
  questions with Precision@4 of 0.25 (alcoholism, acetaminophen) are still weak — those are
  retrieval problems, untouched by this sprint, and remain Sprint 6's business.
- **The CoT prompt is ~3,050 tokens** versus a few hundred for the baseline, a real cost and latency
  increase on every query. It fits the free tier comfortably, but it is not free.
- **The CoT exemplars are held out from the eval set, with one caveat.** None of the six exemplar
  *questions* appears in `dataset.py`. The anorexia exemplar's context still mentions bulimia, and
  the eval set asks "What is bulimia nervosa?", so **the overlap is reduced but not eliminated** —
  it is inherent to an exemplar whose whole lesson is that bulimia's symptoms get retrieved for an
  anorexia question. Evidence that it does not matter: the baseline arm, which never sees this
  exemplar, also scored 1.00 on that question.
- **Three changes are bundled into the "CoT" arm** — the instruction text, the example format, and
  the example-selection strategy all changed at once, so the improvement cannot be attributed to
  chain-of-thought reasoning specifically. This is the most significant methodological weakness;
  `sprint3_audit.md` sets out the ablations that would separate them.
