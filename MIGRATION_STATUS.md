# MEDBOT — Where We Started, Where We Are, Where We're Going

Single source of truth for project status. Supersedes earlier drafts of this file, which contained
claims later disproved by actually running the app (noted inline below where relevant).

---

## 1. Where we started

**Original prototype (2024-05-23 → 2024-06-15, 26 commits):** a bootcamp/learning-style build —
loose root-level scripts, OpenAI (`ChatOpenAI` + `text-embedding-ada-002`), a Chroma vector store
migrated to FAISS mid-development after repeated SQLite version conflicts (8 commits fighting it),
and a hardcoded personal path (`E:/brototype/Langchain/Ollama/test_chatbot`) left in `config.py`.
Few-shot prompting and external search (PubMed/Wikipedia/SerpAPI) were built but, per that era's
code, not connected to the answer the user actually saw. No tests, no CI, no medical disclaimer.

**2026-07-06 restructure:** scripts reorganized into a proper `medbot/` package, the hardcoded path
replaced with a portable repo-relative config, dead code isolated into `medbot/legacy/`. Still
running on OpenAI at this point — needed a paid key the user doesn't have.

**This session's starting point (2026-07-23):** the user has no paid OpenAI access and no
local-LLM-capable hardware (see laptop spec in memory), so the chat backend had to move to a free
tier. On opening the repo, the actual state was:
- A large restructure + partial Gemini migration sitting **entirely uncommitted**.
- The on-disk FAISS index **corrupted** (`core.autocrlf=true` with no `.gitattributes` had mangled
  the binary file on a git rename/checkout).
- Docs (this file, in an earlier draft) claiming the index rebuild was complete at 1223/1223 chunks —
  actually only **900/1225** were embedded on disk.
- External search (PubMed/Wikipedia/SerpAPI) and few-shot prompting *looked* disconnected per an
  earlier project report — this turned out to be stale; see §2.

---

## 2. What we have now (Sprint 0 — verified, not just claimed)

Everything below was confirmed by reading the actual current code and by running the app
end-to-end via `streamlit.testing.v1.AppTest` (a real question, a real Gemini call, real retrieval,
all three external sources returning) — not carried over from older docs.

- **Chat**: Google Gemini, `gemini-flash-lite-latest` (500 RPD free tier vs. 20 RPD for full Flash).
- **Embeddings**: local `fastembed` (`BAAI/bge-small-en-v1.5`, CPU/ONNX) — no API key, no quota.
  Gemini's free embedding quota (1000/day, charged per chunk) could never cover this corpus.
- **Vector index**: genuinely complete now — **1225/1225 chunks**, rebuilt this session (the
  remaining 325 were embedded live during testing), verified loading from disk on repeat runs.
- **Few-shot prompting + medical disclaimer** — confirmed wired into the live `RetrievalQA` chain
  (`medbot/prompt.py`'s `build_context_prompt`, called from `medbot/query_handler.py`). An earlier
  project report claimed this was unused; that was wrong as of the current code.
- **External source corroboration** — PubMed, Wikipedia, and Google (via SerpAPI) all confirmed
  wired into the visible answer (`format_external_results` in `app.py`), not discarded.
- **Bugs found and fixed this session**:
  - PubMed: TLS interception on this network was rejecting requests — fixed with `pip-system-certs`.
  - Wikipedia: 403 (missing `User-Agent`, per Wikimedia's robot policy) and a strict Content-Type
    check on the JSON response — both fixed in `medbot/external_search.py`.
  - `.env` / `.env.example` reconciled to the vars `medbot/config.py` actually reads (dropped a dead
    `OPENAI_API_KEY` entry, fixed a misnamed `GEMINI_EMBEDDING_MODEL` → `LOCAL_EMBEDDING_MODEL`).
  - Stale vectorstore files re-staged (they were committed mid-rebuild, before completion).
- **Git**: the whole migration + restructure + fixes committed as `9f28cb2`, pushed on branch
  `openai-to-gemini-migration`, PR open now.

**Sprint 1 (environment & repo hygiene) — also done (2026-07-25, commit `50363f4`):** exact
dependency pins in `requirements.txt`, README reconciled to the Gemini/fastembed setup,
`PROJECT_REPORT.md` marked with a historical-snapshot banner, three stale backup vectorstore
folders deleted, `.venv-gemini` kept as-is.

**Sprint 2 (evaluation harness) — also done (2026-07-25):** `medbot/eval/` — a 24-question test
set grounded in the real corpus (`dataset.py`; every question was chosen because a dedicated
encyclopedia entry for it actually exists — see the coverage note below), a Precision@K
retrieval metric (keyword-containment against manually verified phrases), and an LLM-judge
groundedness score (same Gemini model grading its own answer, disclosed self-grading bias).
Run via `python -m medbot.eval.run_eval`; full numbers and per-question results in
`medbot/eval/results.md`.
- **Mean Precision@4: 0.83, mean groundedness: 0.83** across 24 questions.
- **Corpus coverage fact discovered while building the test set:** the indexed PDF (Gale
  Encyclopedia of Medicine, Vol. 1, 2nd ed.) covers entries alphabetically from roughly
  "Abdominal ultrasound" to "Byssinosis" — **A-B only**. Asking about any C-Z condition (diabetes,
  psoriasis, stroke, etc.) will retrieve weak or irrelevant chunks not because retrieval is
  broken, but because the corpus doesn't have a dedicated entry for it. Worth knowing before
  trusting any future retrieval-quality complaint about a non-A/B topic.
- **Real bug-shaped finding, audited end-to-end:** 4 of the 24 questions scored 0 groundedness
  because the model falsely refused to answer ("I don't know based on the provided context")
  even though the retrieved context clearly contained the answer. A follow-up audit confirmed
  this is real, not a harness artifact: reproduced live through the actual Streamlit app via
  `AppTest`; a temperature=0 vs 0.1 experiment (24 calls) showed the refusal rate drops from
  67% to 42% at temperature=0 but doesn't go away — one question ("bursitis") refused in all 6
  trials at both temperatures, so this isn't pure sampling noise. Direct motivation for Sprint
  3's CoT rewrite (temperature=0 alone won't fix it). Full writeup, including a precision audit
  of the two weakest retrieval cases, in `medbot/eval/results.md`.

**Sprint 3 (chain-of-thought prompting) — also done (2026-07-25):** the false-refusal bug Sprint 2
found is fixed and measured. `medbot/prompt.py` gained a `cot` prompt variant (the default
from 2026-07-25 until **2026-08-03, when `no-examples` replaced it** — see the entry at the
end of this section; `cot` remains supported and its recorded numbers all stand):
six worked examples in question → context → reasoning → answer form, each built from chunks
actually retrieved from this corpus, on A–B topics deliberately held out of the eval set. The
reasoning trace is stripped before the user sees it (`strip_reasoning`/`run_query` in
`medbot/query_handler.py`, used by both the app and the harness, so what's measured is what's
shown). `temperature` dropped 0.1 → 0. Full write-up in `medbot/eval/results_sprint3.md`.
- **A/B, both arms re-run fresh at temperature 0** so the prompt is the only variable (comparing
  against Sprint 2's temp-0.1 numbers would have confounded the two changes): claim-level mean
  groundedness **0.84 → 1.00**, false refusals **2/24 → 0/24**, per-question **6 improved,
  0 regressed** (after the fix round below). Precision@4 identical at 0.83 in both arms — expected,
  since retrieval was untouched, and a useful sanity check that the harness measures what it claims.
- **Attributed, not just observed** — ⚠️ **but the attribution was WITHDRAWN on 2026-07-27; see the
  `is_refusal` correction under Sprint 4.** A four-arm ablation separated the instruction rewrite
  from the worked exemplars (refusals over 20 trials): baseline 10, instruction-only 5,
  examples-only 1, cot 0. The conclusion drawn was that **bursitis is untouched by the instruction
  rewrite (5/5, same as baseline)** and needs the exemplars, that the two are not substitutes, and
  that only the combination reaches zero — so the CoT exemplars earn their ~2,400 tokens per query.
  **All five instruction-only "refusals" were few-shot contamination**, not refusals: the model
  declined a question about *nosebleeds* (from its own selected exemplar) and then answered
  "What is bursitis?" correctly in the same reply. Re-scored, the ablation reads **instruction-only
  0, examples-only 1** — inverting the conclusion. On this evidence the instruction rewrite alone
  reaches zero and the exemplars alone do not, so **the exemplars' tokens are not justified by the
  refusal result.** Not acted on at the time: 4 questions is far too small a base to strip the
  exemplars on. **Re-run on all 24 questions the same day** — the refusal parity holds
  (instruction-only vs cot p = 1.0000), the true token cost is **~2,770 per query**, and the
  exemplars still stay, on grounds that are not refusal-shaped. See the ablation re-run entry
  under Sprint 4.
- **Credit splits between the two changes.** Temperature 0 alone fixed bedsores and abscess. The
  CoT prompt fixed the other two: breast cancer and bursitis refused **5/5 under the baseline
  prompt at temperature 0** — deterministic, not noise — and 0/5 under CoT. Bursitis had refused
  6/6 at both temperatures in Sprint 2.
- **Guarded against the cheap win.** Driving refusals to zero is trivial with a prompt that never
  refuses, which would swap a refusal bug for a hallucination bug. Two defences: one of the six
  exemplars is a genuine "context doesn't support it" case (asking for anorexia symptoms retrieves
  *bulimia's*), and a new out-of-corpus trial suite where refusing is correct — both variants
  refused 6/6, no over-answering regression.
- **Corrects a Sprint 2 claim:** "the corpus is A–B only, so C–Z questions retrieve badly" is too
  strong. Stroke causation turned out to be covered inside the *A* entries for embolism and
  atherosclerosis, and both variants answered it correctly from them. Coverage has to be checked
  per question, not inferred from the first letter.
- Verified end-to-end through the real Streamlit app via `AppTest`, not just the harness: bursitis
  answers correctly, no reasoning trace leaks to the UI, external sources still render.
- **Audited afterwards** (`medbot/eval/sprint3_audit.md`) — the audit corrected two overstatements
  in the first draft of the results and found one real defect, all now fixed:
  - The "10/20 vs 0/20" framing was pseudo-replication: those are 5 repeats of 4 questions, not 20
    independent samples. At the question level, Fisher exact gives **p=0.43 — not significant**. The
    within-question effect is total and repeatable; the across-question sample is simply too small.
  - The old groundedness judge only ever returned 0.0 or 1.0. Replaced with a claim-level judge
    (supported claims / total claims, `judge_groundedness_claims` + `rejudge.py`), which
    discriminates properly and revised **baseline down from 0.917 to 0.841** — Sprint 2's 0.83
    headline was coarser than it claimed.
  - The stricter judge found two things the binary one could not: **few-shot contamination in the
    baseline prompt** (the atherosclerosis answer opens by declining a question about *osteoporosis*,
    bled in from its own selected exemplar, and the old judge scored that 1.00), and **a genuine
    regression in the shipped CoT arm** (bedsores 1.00 → 0.75, an over-linked causal chain). The
    honest tally was 6 improved / 1 regressed — the regression was then fixed, see the fix round below.
  - Fixed defect: `run_eval` with no `--variant` would have silently overwritten Sprint 2's recorded
    `results.json`/`results.md` with CoT numbers. Output is now always variant-suffixed.
  - `tests/test_prompt_variants.py` added: covers `strip_reasoning` and **pins a sha256 of the
    rendered CoT prompt**, so the shipped prompt cannot drift from the string the recorded numbers
    describe without the check failing. Runs standalone; pytest-collectable in Sprint 4.
  - **Fix round: the regression the audit found was fixed, not just filed.** An anti-over-linking
    clause was added to the CoT instruction ("do not join two separate statements from the context
    into a cause-and-effect chain unless the context asserts that link") and the anorexia exemplar's
    bulimia content was trimmed. The CoT arm was then re-run in full against the new prompt:
    bedsores **0.75 → 1.00**, claim-level mean **0.990 → 0.997**, refusals still 0/20, out-of-corpus
    guard strengthened to 10/10 at 5 trials, Precision@4 unchanged. No new regression.
  - The single remaining sub-1.0 answer is a **judge artefact**: the model wrote "seizers" for
    "seizures" and the claim judge scored the misspelling as an unsupported claim. Real but trivial
    output defect; the judge is grading spelling as if it were factual support. Concrete reason F6
    (human calibration of the judge) is still worth doing.
  - **Quote the claim-level 0.84 → 1.00, not the binary 0.92 → 1.00**, and keep the sample caveat:
    Fisher p=0.43 at the question level. Large clean effect, small sample — both halves are true.
  - **Still open after this sprint:** question-set expansion (24 questions, a 2-question refusal
    delta) and blinded human calibration of the judge. Both are measurement-confidence work, not
    defects in the shipped behaviour. *(Sprint 4 closed the first: the refusal result is
    significant on all 24 questions — at p=0.0094 after the 2026-07-27 instrument correction,
    p=0.0219 as originally measured. See below.)*

**Sprint 4 (testing foundation) — mostly done (2026-07-25):** scoped as "add pytest", became
mostly a measurement sprint, because the tests immediately found that two of the three question
sets Sprint 3's result rested on were selecting the wrong questions. Full write-up in
`medbot/eval/results_sprint4.md`.

  - **110 tests: 106 offline (~6s, no network, no quota) + 4 live.** `pytest.ini` sets
    `addopts = -m "not live"`, so the default run is free and deterministic — which is what
    Sprint 5's CI needs, having no API key. (The sprint first reported "57 tests: 53
    offline"; the real figure before the audit round was 58/54 — see results_sprint4.md §6.
    The third audit round added 7 more covering the `is_refusal` correction.)
  - **`tests/conftest.py` fails any non-`live` test that opens an outbound socket.** Written
    after a mock in this very sprint was pointed at the defining module instead of the calling
    one (`from ... import ...` had already bound the name), missed entirely, and let the test
    hit PubMed/Wikipedia/SerpAPI for real *while passing*. A mock that misses looks exactly like
    a mock that works, only slower — so it is enforced rather than left to discipline.
  - **Bug found and fixed:** `search_serpapi` caught only `RequestException` while its two
    siblings caught `Exception`, so an auth/quota error from the serpapi client escaped into
    `search_external_sources`, whose blanket `except` then discarded **all three** sources. A bad
    SerpAPI key silently cost the user their PubMed and Wikipedia results too.
  - **The refusal suite was measuring the wrong questions (audit F1).** Re-mining the stored
    Sprint 3 answers found FOUR baseline refusals, only two of them in the hand-picked 4-question
    suite. `"What causes bladder cancer?"` answered "I don't know, as the provided context does
    not state the exact cause" on a **perfect Precision@4** and had been outside the suite for two
    sprints. The suite is now the whole 24-question eval set.
  - **Headline: the refusal result now reaches significance.** 24 questions × 2 arms × 3 trials:
    baseline refuses **6/24** questions, cot **0/24**. **Fisher exact p = 0.0219**, versus p=0.43
    on Sprint 3's 4 questions. Still to be read at the question level — 13/72 vs 0/72 trials is
    the same repeated-measures mistake the Sprint 3 draft made.
    - ⚠️ **SUPERSEDED 2026-07-27.** Half of those six baseline refusals were the measurement
      instrument miscounting few-shot contamination. Re-scored, this run is **3/24 vs 0/24,
      p = 0.2340 — not significant.** The result that stands is the 5-trial re-run below. See
      "The instrument was wrong" for the full account.
  - **The out-of-corpus guard contained a question the corpus covers (audit F7/F8).** "What are
    the symptoms of diabetes?" was half the 2-question guard; its top chunks include the *blood
    sugar tests* entry (a B entry) explaining insulin and hyperglycemia. Exactly the F8 stroke
    mistake, still live one sprint after the lesson was recorded. Fixed with a tool rather than
    a resolution to be careful: `medbot/eval/verify_coverage.py` checks candidates against real
    retrieval, costs no quota, and rejected 14 of 26 candidates.
  - **The hallucination guard is armed — CLOSED 2026-07-27.** The original run hit the free
    tier's 500 requests/day at question 2 of 10, and a retry on 2026-07-26 was still
    quota-blocked, because the daily counter rolls over at **midnight US Pacific (~12:30 IST)**,
    not local midnight — so every "run it tomorrow morning" attempt before ~12:30 IST was
    spending against the *previous* Pacific day and had no chance of succeeding. Completed at
    12:37 IST on 2026-07-27, one rollover later, via `--resume` (51 calls; 3 cells already held
    trials). **All 10 questions × 2 arms × 3 trials refused: 60/60, zero invented answers.**
    Both arms 10/10 questions, Fisher p = 1.0000 between them — which is the desired result
    here: CoT drove false refusals down without buying it by over-answering.
    `test_out_of_corpus_gate_is_fully_armed` no longer skips. The guard survived the
    `is_refusal` correction below unchanged, still 60/60, which is the one direction that
    instrument must never fail in.
  - **Judge calibration (F6) is partial and needs a human.** Length bias is *ruled out*: answers
    are +27.5% longer in the cot arm but the judge extracts +23.8% more claims — ~101 vs 104
    chars per claim, and the score is a ratio. Hedging bias remains untested, since the cot arm
    has almost no score variance left to correlate against. `medbot/eval/calibration_sheet.md`
    is ready to hand-label: 25 answers, arm and score hidden, order shuffled, context included.
    I cannot close this one — I am the model being audited, so my labels are not independent.

**Sprint 4 audit round (2026-07-26):** every headline number recomputed from the raw JSON
rather than read from the prose. The headline appeared to **survive unchanged** — 6/24 vs 0/24,
Fisher exact p=0.0219 — as did the calibration figures (781/995 chars, 7.71/9.54 claims), the
claim scores (0.841 → 0.997), Precision@4 (0.8333) and the 25-item calibration sheet. Four
defects found and fixed; full write-up in `medbot/eval/results_sprint4.md` §6.

  ⚠️ **The refusal headline did not actually survive, and the method of this audit is why it
  looked like it did.** "Recompute from the raw JSON" recomputed the *statistics* from the
  stored `refusal` booleans — but those booleans are themselves derived, by `is_refusal`, and
  that function was the thing that was wrong. Re-deriving a number from data the same broken
  instrument produced cannot detect the instrument. What caught it a day later was reading the
  stored answer **text**, which is the actual raw measurement. Recompute from the rawest thing
  available, not merely from something rawer than prose.
  - **`refusal_stats.py` counted unmeasured cells as zeros.** On the incomplete out-of-corpus
    data it reported cot "1/2 questions ever refusing" and computed a Fisher p that included
    the schizophrenia question — which has no cot trials at all — scoring a question the model
    was never asked as one it answered. Missing arms are now excluded and reported as
    `INCOMPLETE`. A sprint about measuring the wrong questions should not ship a stats tool
    that invents observations.
  - **The hand-rolled Fisher exact had no test** despite producing the sprint's headline; its
    docstring claimed validation that nothing enforced. Now pinned against tea-tasting
    (0.4857), Sprint 3 (0.4286) and Sprint 4 (0.0219), plus symmetry and the p ≤ 1 bound.
  - **The trial runner could not resume** — `run()` started from an empty dict and the
    checkpoint overwrote the file, so a quota-killed run restarted at question 1 and could
    never reach question 3 if the quota died twice in the same place. That is the actual
    mechanism behind this sprint stalling, not bad luck. `--resume` added, 12 tests.
  - **The coverage screen was not reproducible:** `CANDIDATES` held 12 of the 25 questions
    actually screened, so `--candidates` could not reproduce the selection and the recorded
    "14 of 26" was wrong and uncheckable. True figures **25 screened, 15 rejected, 10 kept**;
    the full list including rejections now lives in the file, and
    `test_out_of_corpus_selection.py` re-runs the screen against the live index every test
    run, asserting both that the shipped 10 are still absent and that rejected candidates are
    still detected — so a screen degrading into accepting everything fails loudly.

  Second pass, on the work quota does *not* gate:
  - **The calibration exercise had no analysis step** — `calibration_key.json` was written by
    `calibration_sample.py` and read by nothing, so labelling all 25 answers would have produced
    no result. `calibration_score.py` closes the loop and reports **differential** bias (cot −
    baseline), which is the statistic that matters: a judge wrong by the same margin in both
    arms leaves the 0.841 → 0.997 claim intact, since a constant bias cancels out of a
    difference. Refusals are held out, not scored as 0.0.
  - **A quota-dead run looked exactly like a slow one** — the backoff notice was unflushed and
    there was no preflight, so tonight's attempt sat silent for 12 minutes. Added `flush=True`
    and a one-call `preflight()` that tells the daily cap from the per-minute cap (Google
    reports both under the same metric name, so the obvious substring check would kill
    recoverable runs). Fails in **6.5s instead of 32s**, spending nothing.
  - **`data_processing.py` loaded the FAISS index at import time** via a module-level
    `vectordb = create_vector_database()` that nothing consumed — `app.py` and all five eval
    scripts call the function themselves. Every consumer paid the load twice and the test suite
    paid it for nothing. Worse, the call *builds* the index when absent, so importing the module
    on a machine without `vectorstore/` would start embedding 1225 chunks — from `--help`, or
    from `pytest --collect-only`. A live trap for Sprint 5's CI. Removed; offline suite
    **33s → 5.6s**.

**Sprint 4 second audit round — the instrument was wrong (2026-07-27):** the fragility warning
`refusal_stats` prints on its own headline was taken at face value and the refusal suite re-run at
5 trials per cell instead of 3 (240 calls). The extra trials did not settle the statistics — they
exposed a defect in `is_refusal`, the measurement instrument behind every refusal number in
Sprints 2–4. Full write-up in `medbot/eval/results_sprint4.md`.

  - **`is_refusal` counted two things that are not refusals.** It scored any refusal marker in the
    opening 200 characters, regardless of what followed:
    - **Few-shot contamination.** The model declines a question drawn from its own selected
      exemplar, then answers the real one: *"I don't know the answer to your question about **bee
      stings**… However, regarding how burns should be treated,"* followed by 2,462 characters of
      correct burn treatment. Also osteoporosis standing in for atherosclerosis, and nosebleeds
      for bursitis. A real defect — the Sprint 3 audit already logged it — but a *contamination*
      defect, and scoring it as a declined question attributes it to the wrong bug while
      inflating the baseline arm the whole CoT result is measured against.
    - **A leading hedge before a full answer.** *"The context does not provide a single,
      straightforward list labeled 'symptoms of alcoholism,' but it describes numerous health
      problems…"* — then 1,189 characters of symptoms. The existing note anticipated *trailing*
      caveats; this is the same thing at the front.
  - **Fixed by a substance rule:** a marker makes an answer a refusal *candidate*, and it only
    scores as a refusal if nothing survives stripping the sentences that assert absence
    (`delivered_content` + `ABSENCE_RE`). Calibrated against all **572 stored trials**; 16 labels
    change and every one was inspected by hand.
  - **Two plausible fixes were tried and rejected**, both pinned as tests so they are not
    re-attempted. *Length alone* fails because genuine refusals routinely run 200–400 characters
    while helpfully enumerating what the corpus covers instead. *Matching the question's own
    terms* is actively dangerous: refusals end "…does not mention **psoriasis**", so a term check
    flips precisely the out-of-corpus refusals the hallucination guard depends on.
  - **`medbot/eval/relabel.py`** re-derives the stored `refusal` booleans from the stored text
    (dry-run by default, no quota). It exists because `test_refusal_labels_match_the_heuristic`
    fails by design when the instrument changes — the fix and the re-scoring cannot drift apart.
    Answer text is never modified, so any re-scoring is reversible.
  - **Sprint 4's shipped headline does not survive its own corrected instrument.** The 3-trial
    run goes from 6/24 vs 0/24, p=0.0219, to **3/24 vs 0/24, p = 0.2340 — not significant.**
  - **The result that stands** is the 5-trial run under the corrected instrument: baseline
    **7/24** questions, cot **0/24**, 20/120 vs 0/120 trials, **Fisher exact p = 0.0094**. CoT is
    back to a clean zero — its single apparent refusal was the alcoholism false positive.
  - **The 5-trial run was worth it, but not for the stated reason.** It was launched to settle the
    fragility of two 1-of-3 questions; it moved the raw p-value only 0.0219 → 0.0226. Its actual
    value was generating the data that exposed the instrument bug, and being the only run that
    survives correction.
  - **More trials per question is NOT the fix — and the advice that said so was `refusal_stats`'s
    own output, now corrected.** Going 3 → 5 left the fragility warning firing on 3 of 7 questions
    (p=0.1092 if they are dropped), and the membership churned: burns (1/3) did not reproduce at
    0/5 while autism and bedsores appeared fresh at 1/5. That churn is the signature of low-rate
    stochastic refusal — extra trials do not resolve the existing 1-of-N questions into confident
    refusers, they surface new ones that were previously missed, each arriving at exactly one hit.
    So the sensitivity check keeps failing however much quota is spent on it. The binding
    constraint is **24 questions**, not trials per question. The fragility block now says so, and
    the source carries the evidence, so the next person does not re-spend the 240 calls.
  - **Still open:** blinded human calibration of the groundedness judge (F6) is still not
    something I can close, for the same reason as before. *(The ablation re-run, the other item
    listed here, was done the same day — see below.)*

**Sprint 3 ablation re-run under the corrected instrument (2026-07-27):** the open decision was
whether the CoT exemplars earn their per-query tokens. Re-measured on the honest denominator —
`instruction-only`, all 24 eval questions, 5 trials (120 calls), plus the 10-question
out-of-corpus guard at 3 trials (30 calls), all three arms now at 5 trials on the same
questions with prompt, index and model unchanged. Full write-up in `medbot/eval/results_sprint4.md` §8.

  - **The exemplars cost ~2,770 tokens per query, not ~2,400** — measured from the rendered
    templates (cot 12,151 chars vs instruction-only 1,064). The older figure understated it.
  - **Refusals: baseline 7/24 questions, instruction-only 1/24, cot 0/24.** baseline vs cot
    p = 0.0094 (the shipped headline, reproduced); baseline vs instruction-only p = 0.0479;
    **instruction-only vs cot p = 1.0000.** Out-of-corpus guard: all three arms 30/30, so
    **90/90**. On refusals the exemplars buy nothing measurable over the instruction rewrite.
  - **A third `is_refusal` defect, found by the third arm.** The guard first read as
    instruction-only inventing answers to 4 of 10 out-of-corpus questions. All six such trials
    were flat refusals phrased *"there is no mention of…"* — `"no mention"` was not in
    `REFUSAL_MARKERS`, so the candidate gate never opened. **The marker list had been calibrated
    on baseline and cot trials only, because those were the only arms with data; a prompt change
    changes refusal wording, so a wording-keyed instrument does not transfer to a new arm.**
    Fixed by making the gate the union of the marker list and `ABSENCE_RE` (neither contains the
    other). Blast radius measured across all **1,022** stored trials before changing anything:
    exactly **6 labels**, all in the new instruction-only guard data — **every baseline and cot
    label, the p = 0.0094 headline and the 60/60 guard are untouched.**
  - **`refusal_stats` was reporting the guard in the unit that cannot see a leak** — "questions
    ever refusing" reads 9/10 and p = 1.0000 for an arm that answered 4 of 10 out-of-corpus
    questions. `report()` now takes `unit=ever-refused|ever-answered`, the guard uses
    `ever-answered`, and both counts always print. The six trials were an artefact; the reporting
    hole was real.
  - **A fixture had been passing for the wrong reason:** the F9 "kept verbatim" partial answer
    was a shortened paraphrase (97 chars surviving, under the threshold) that only passed because
    the narrow gate never reached the substance rule. Replaced with the real 603-char stored text.
  - **Contamination belongs to the legacy example format, not to "having examples."** The two
    arms using the legacy semantic 1-shot selector both substitute a neighbouring example's
    question (baseline: osteoporosis for atherosclerosis 5/5; instruction-only: nosebleeds for
    bursitis 5/5); neither CoT arm does. So dropping the CoT exemplars means returning to the
    format that contaminates, not to "no examples".
  - **Decision: the exemplars stay for now**, and not out of caution — instruction-only's
    claim-level groundedness has never been measured (the 0.841 → 0.997 result is baseline vs
    cot, ~48 calls to close), and instruction-only contaminates on 5/5 bursitis trials. The arm
    worth running next is the one nobody has measured: **new instruction with no examples at
    all** (~150 tokens, contamination-proof by construction) — not `examples-only`.
  - Offline suite now **112 tests** (was 106): the gate fix, the union rule, the wider gate's
    interaction with the substance test, the instruction-only guard data, and the reporting unit
    are all pinned.

**Sprint 4 fourth audit round (2026-07-27) — the gates, not the numbers.** The three earlier
rounds each corrected a number. This one recomputed every number from the stored answer text
under all three historical instruments (as-shipped, first correction, current) — **every
documented figure reconciles exactly**, including the superseded ones — and then audited what
had never been audited: whether the tests protecting them can fail. Full write-up in
`medbot/eval/results_sprint4.md` §9. Six findings, all fixed:

  - **Every headline gate was reading the retired dataset.** `test_eval_regression.py` gated on
    the 3-trial `sprint4_` run — the one the file marks "do not quote" — while the 5-trial run
    the standing p = 0.0094 comes from, and both ablation files, had **no gate at all**.
  - **The label-drift test covered 2 of 10 trial files**, excluding both files the current
    headline and the ablation conclusion rest on. Now globs them all.
  - **`test_out_of_corpus_gate_is_fully_armed` asserted nothing.** Written to skip while the
    guard data was incomplete; once completed, the body fell through with no assertion — a test
    named "fully armed" that could not fail, and that would have gone back to *skipping* rather
    than failing if the data were truncated.
  - **`python -m medbot.eval.refusal_stats` with no arguments printed p = 0.2340, NOT
    significant** — the superseded run — so anyone re-checking the sprint the obvious way would
    have concluded the result had evaporated. Default is now the run that stands, and every
    report prints the file it came from.
  - **A vacuous assertion:** `assert "Q" in rendered` cannot detect a dropped `{question}`
    placeholder, because every template contains the word "Question:".
  - **38 stale `.pyc` files** from the repo's old `D:\Medbot\` location made every pytest
    traceback cite a path that does not exist (`co_filename` is baked in at compile time, and
    source mtime/size still matched). Executed code was correct; the reported location was
    fiction. Cleared — worth knowing before Sprint 5 wires CI.
  - **Each fix was mutation-tested**: flip an out-of-corpus trial to an answer, truncate a cell
    from 5 trials to 2, flip a stored label in a previously-unguarded file — every gate failed
    as intended, then the mutations were reverted. The socket guard was probe-tested too:
    `connect` and `connect_ex` both blocked, naming the test.
  - No quota spent: every figure came from committed artefacts.

**Prepared for the next measuring run (2026-07-27, no quota spent).** Both items are inert on
purpose — code and selection committed, numbers not, nothing shipped changed. `results_sprint4.md` §10.

  - **A `no-examples` prompt arm.** The new instruction with no exemplar at all: **865 chars,
    ~216 tokens — 14× cheaper than the shipped `cot` prompt** (~3,037). §8 found the exemplars
    buy no measurable refusal improvement over `instruction-only` (p = 1.0000) but kept them
    because `instruction-only` substitutes a neighbouring example's question 5/5 on bursitis.
    This arm removes the *mechanism*: with no example in the prompt there is no other question
    to answer, so contamination is impossible by construction. It is the arm to run before
    `examples-only`. Pinned by tests that no legacy example text can leak into it, that it stays
    decisively cheaper than `cot`, and that the shipped default is still `cot` —
    unmeasured prompts do not ship. *(Naming trap documented, not fixed: `instruction-only` is a
    misnomer — it keeps baseline's legacy 1-shot example. Renaming would orphan recorded results.)*
  - **22 screened question-set expansion candidates**, via a new `medbot/eval/verify_entry.py`
    (local retrieval, no quota). 34 screened, 10 auto-rejected, 2 rejected by hand; every
    `expected_keywords` phrase verified present in real retrieved chunks; **mean Precision@4
    0.8523** vs 0.8333 for the current 24, with a higher floor. Held in a separate
    `EXPANSION_QUESTIONS` list — merging it redefines `REFUSAL_QUESTIONS` from 24 to 46 and
    invalidates every recorded trial file, so the merge belongs with the ~336 calls that
    re-measure the suite. A test pins that sequencing.
  - **Why a second screening tool:** `verify_coverage`'s rule proves *absence* for the guard, so
    one loose hit is a useful rejection. Inverted to accept eval questions it passed 35 of 36 —
    including "What causes back pain?" (top chunks: the **bursitis** entry, matched on "pain")
    and "What is Barrett's esophagus?" (top chunk: **acetaminophen**). Accepting those would turn
    every correct refusal into a recorded bug: audit F8's error pointing the other way.
  - Offline suite now **123 tests, ~10s**.

**Sprint 5 (CI/CD) — done (2026-08-02, branch `sprint-5-ci-cd`).** GitHub Actions on every push
and PR, ruff lint, pre-commit, and dev dependencies split out of `requirements.txt`. No quota
spent, no eval number recomputed, and no prompt text altered — every source diff in this sprint
is import ordering, whitespace, or an unused name, verified by reading the diff of
`medbot/prompt.py` specifically before trusting the autofixer.

  - **The sprint's real finding is that Sprint 4's socket guard did not guard.** It was an
    autouse fixture, and autouse fixtures are function-scoped: pytest instantiates
    higher-scoped fixtures *first*, so anything set up in a `scope="module"` fixture ran before
    the guard existed. `test_expansion_selection.py` and `test_out_of_corpus_selection.py` build
    their FAISS retriever in exactly such a fixture, and **both carry a comment stating the
    guard would stop an uncached fastembed model from downloading.** It would not. Measured, not
    reasoned: pointed `FASTEMBED_CACHE_PATH` at an empty directory and ran them — 8 passed in
    23s having silently pulled ~130MB from HuggingFace. Rewritten as `pytest_runtest_setup` /
    `pytest_runtest_teardown` hooks (`tryfirst` / `trylast`), which run before fixtures of any
    scope. The same cold-cache run now fails with 8 socket-guard errors naming the host.
  - **It survived three sprints because the probe that "verified" it was itself the easy case.**
    Sprint 4's fourth audit round probe-tested `connect` and `connect_ex` and recorded them as
    passing — with a plain function-scoped test, the one arrangement that always worked, and
    the probe was never committed. `tests/test_network_guard.py` now commits it, and its first
    test is the module-scoped-fixture case that would have caught this.
  - **This is what makes CI's cache-warming step load-bearing rather than decorative.** The
    workflow downloads the model in its own step, outside pytest, because the guard now makes
    an in-test download a hard failure. Written before the guard was fixed, that step was
    justified by a guarantee that did not hold; it is true now.
  - **Lint: 41 findings, 29 autofixed, 4 fixed by hand**, of which one was a real defect class —
    `zip(batch, vectors)` in `data_processing.py` truncates silently on a length mismatch, which
    is precisely the shape of "index written with chunks missing but no error" that had this
    store at 900/1225 while the docs claimed complete. Now `strict=True`.
  - **`ruff format` is deliberately not used, and E501 is off.** Prompt templates are pinned by
    content hash and the recorded numbers only compare while the rendered prompt is
    byte-identical, so a reformatter re-wrapping an implicitly-concatenated string can
    invalidate a measurement with no test failing at the time. `ruff.toml` records this, plus
    the two rule families trialled and dropped (SIM fired on
    `assert cfg.DATA_DIR == os.path.join(...)` as a Yoda condition — a rule that cries wolf gets
    the whole linter ignored).
  - **pre-commit's first run tried to rewrite recorded evidence.** `trailing-whitespace` wanted
    12 lines of `medbot/eval/calibration_sheet.md`, where the trailing spaces are part of model
    answers pasted verbatim for the human groundedness labelling. Excluded from the two hooks
    that *rewrite* files and deliberately not from the ones that only read, so `check-json` still
    validates the trial artefacts.
  - **A green build could still have tested nothing, and now cannot.** Both retrieval test
    modules `pytest.skip()` when `index.faiss` is absent, so a checkout that lost the index
    would have gone green on 8 skips — the same shape as audit finding F3. `tests/
    test_vector_index.py` asserts presence, **1225 vectors** and **384 dimensions**, reading the
    index directly with `faiss` so it needs neither the embedding model nor a warm cache, and
    it never skips. Mutation-tested three ways: absent → 3 failures; truncated to 900 → only
    the count test fails; rebuilt at 128 dimensions → only the dimensionality test fails.
    The skips stay for ergonomics, now backstopped rather than load-bearing.
  - Offline suite now **131 tests, ~7s warm** (5 guard tests, 3 index tests); live suite 4 → 5.
  - **CI runs on `windows-latest`, not ubuntu.** The first draft used ubuntu-latest out of
    convention; Melvin's call, and the better one — Windows is the only platform this project is
    developed or run on, and a green Linux run would not have said anything about it. The repo is
    public, so runner minutes are free and cost did not favour either. Known gap, recorded rather
    than papered over: `.devcontainer/` pins a Debian bullseye image, so the Codespaces path is
    not covered by CI.
  - **The whole pipeline was mirrored locally, not assumed.** A clean Python 3.11 venv,
    `pip install -r requirements-dev.txt` from scratch (every pin resolves — numpy 1.26.4,
    faiss-cpu 1.14.3, fastembed 0.8.0), then lint → warm → pytest against an empty
    `FASTEMBED_CACHE_PATH`: **128 passed in 11.15s**. Same OS family as the runner, so nothing
    about the install is left for the first CI run to discover. (Had CI stayed on ubuntu this
    would have been unverifiable here — there is no Docker and no WSL distro on this machine.)

**Sprint 6 (retrieval quality) — source citations done (2026-08-02, branch `sprint-6-retrieval-quality`).**
The sprint has two halves and only the first is done. Citations and chunk metadata are shipped;
the **relevance threshold is deliberately not started**, because it changes what retrieval returns
and therefore changes answers, so it cannot be shipped without re-measuring the eval suite — which
costs quota. This half costs none and invalidates no recorded number.

  - **The metadata was being thrown away one line after it was created.** `process_documents`
    ended in `[c.page_content for c in chunks]`, discarding the `source` and `page` the loaders
    had already attached. All 1225 chunks sat in the docstore with `metadata == {}` — verified on
    disk, not inferred — so no answer could say where it came from. `RetrievalQA` was also built
    without `return_source_documents`, so the retrieved chunks were discarded a second time inside
    the chain.
  - **The rebuild was proved identical rather than assumed identical.** The claim the whole sprint
    rests on is that adding metadata cannot affect retrieval, since the embedded text is unchanged.
    That was checked by rebuilding into a scratch directory and comparing against the shipped index
    row by row before overwriting anything: **1225/1225 bit-identical rows, max absolute difference
    0.000e+00, zero text mismatches.** Independently corroborated by git, which does not list
    `vectorstore/index.faiss` in the diff at all — only `index.pkl` changed (2,861,119 → 2,942,744
    bytes). So **Precision@4 0.8333, the p = 0.0094 refusal headline and the 0.841 → 0.997
    claim-level scores all stand unre-measured**, and no eval trial file is invalidated.
  - **`page` is 0-based, checked against the PDF instead of against convention.** `format_sources`
    renders `page + 1`, and an off-by-one in a citation is a bad failure — it points a user
    checking a medical claim at the wrong page, and nothing surfaces it unless someone opens the
    file. The decisive evidence is the last chunk: the PDF has **637 pages and the highest stored
    `page` is 636**. Two probed chunks also matched at exactly their recorded page and nowhere
    else; three matched on adjacent pages too, because the probe strings ("Treatment", "TheGALE")
    are not unique — weak probes, not counter-evidence.
  - **The block is labelled "Retrieved from", not "Sources".** These are the four chunks the model
    was *given*, which is a weaker claim than the chunks it *used*. Calling them sources would let
    a passage the answer contradicts read as a citation supporting it — the wrong direction to be
    wrong in for a medical tool. Citations are deduplicated on (source, page) because
    `chunk_size=3000`/`overlap=300` routinely puts two retrieved chunks on one page, and listing it
    twice reads as two corroborating sources.
  - **`source` is stored as a bare filename, and a test enforces it.** The loaders return absolute
    paths and `index.pkl` is a committed artefact, so the unnormalised version would have published
    this machine's directory layout into the repo — the same class of mistake as the prototype's
    hardcoded `E:/brototype/Langchain/...`.
  - **An index built before this sprint is detected and rebuilt on load.** It holds correct vectors
    with empty metadata, so it retrieves perfectly and the only symptom is a missing citation block
    that `format_sources` renders as nothing by design. The resume check would have called such an
    index complete forever. Keyed on `chunk_index` rather than `source`, since a `.txt` chunk
    legitimately has no source-bearing page metadata.
  - **A live test had quietly stopped testing what it named.** `test_app_smoke` split the answer on
    `"\n---\n**Related external sources**"` to isolate the model's text; the new citation block is
    inserted *above* that heading, so the split kept succeeding while feeding the citation block to
    the refusal check and the reasoning-leak check as if it were answer text. No test would have
    failed. Now splits on the first `---`.
  - **Red before green:** the three new index-metadata tests were written first and confirmed
    failing against the shipped pre-Sprint-6 index (`KeyError: 'source'`), then passing after the
    swap — so they are known to be capable of failing, not merely observed passing.
  - Offline suite **131 → 162 tests, ~13s** (14 for `format_sources`, 12 for the metadata helpers,
    5 for the index artefact); live suite 5 → 6, the new one asserting the shipped app actually
    passes `source_documents` to the renderer — the unit tests and the index tests both pass if
    `app.py` drops that argument. **Live suite run and green, 6/6 in 61s (one Gemini call):** the
    real app loaded the rebuilt index from disk, answered bursitis without refusing, leaked no
    reasoning trace, and rendered a page-numbered citation. Asserted on the block's shape, not on
    which page — which chunks retrieval returns is the eval harness's business, and pinning it
    here would fail on an unrelated index change.
  - **Not done, and next:** the relevance threshold and chunk-metadata *filtering*. Both change
    retrieval, so both need the eval suite re-run against them.

**The `no-examples` prompt arm — measured (2026-08-03, branch `no-examples-prompt-arm`).**
The §10 item prepared on 2026-07-27 and left inert. 225 calls; nothing shipped changed.
Full write-up in `medbot/eval/results_sprint4.md` §11.

  - **A 216-token prompt matches the 3,037-token shipped one on every measured axis.**
    Refusals **0/24 questions, 0/120 trials** — identical to `cot`, and **p = 0.0094
    against baseline's 7/24**, reproducing the standing headline at **1/14th the prompt
    cost**. Out-of-corpus guard **30/30, zero invented answers**, taking all four arms to
    120/120. Precision@4 **0.8333**, identical to every other arm, as it must be —
    retrieval is untouched by the prompt, and that is the sanity check the harness exists
    to provide. Claim-level groundedness **0.9917** vs baseline 0.8413 and cot 0.9974.
  - **The exemplars' last justification is gone.** §8 found them buying no measurable
    refusal improvement over `instruction-only` (p = 1.0000) but kept them because
    `instruction-only` answers a question about *nosebleeds* on 5/5 bursitis trials,
    contaminated from its own selected exemplar. Reproduced here exactly — and
    `no-examples` answers about bursitis 5/5, at 851–1,070 chars, because with no example
    in the prompt there is no other question to answer. It removes the mechanism rather
    than out-performing the symptom.
  - **The parity claim is stated at its real strength, not more.** `p = 1.0000` against
    `cot` is a null result at n=24 and cannot distinguish equivalence from an underpowered
    test — the same caveat §8 applied to `instruction-only`. What changed is not the
    statistics but the mechanism. The fragility warning on the baseline comparison also
    still holds: 3 of baseline's 7 refusers do so on exactly one trial, and dropping them
    gives p = 0.1092.
  - **The instrument was re-validated on the new arm rather than trusted.** `is_refusal`
    is wording-keyed and has failed to transfer to a new arm twice, so a clean zero from
    it is exactly what a third failure would look like. Every one of the 150 trials was
    read: 0 label disagreements, only 5 of 120 opened the candidate gate (all the bursitis
    "does not explicitly define" hedge, 573–712 chars surviving a 120 threshold), and the
    shortest answer in the arm is 293 chars against a 1,026 median — so no flat refusal is
    hiding behind unlisted wording. On the guard the risk runs the other way, and maximum
    surviving content across all 30 trials is **5 characters**. Those guard refusals are
    phrased *"there is no information/mention of…"* — the exact wording that defeated the
    marker list in §8 — so this also validates §8's union-gate fix on an arm it was never
    calibrated against.
  - **A contamination detector was written, rejected and rewritten.** The first version
    flagged `no-examples` 27 times, which is impossible by construction; it was matching
    incidental word overlap, not a declined foreign question. Restricted to gate-open
    trials it reads zero. Recorded because it is F8's error shape: a screen that accepts
    everything is not a screen.
  - **Not scoring well by saying less** — the obvious way a supported/total ratio can be
    gamed. `no-examples` writes the **longest** answers of the four arms (mean 1,267 chars
    vs cot's 1,053). Its single sub-1.0 score is a judge artefact of the same class as
    cot's "seizers" case: the judge scored the model's meta-caveat *about the context* as
    an unsupported medical claim, while all four substantive claims were supported. More
    evidence for F6, which is still open and still needs a human.
  - **The two ablation files had no regression gate at all** — §9 fixed the headline gates
    and left these, though the exemplar decision rests on them. Now gated, and each gate
    mutation-tested (flip a refusal, flip a guard trial to an invented answer, truncate a
    cell, drop an arm) with the data restored bit-identical after.
  - **Shipped the same day, after the numbers existed.** `DEFAULT_PROMPT_VARIANT` moved
    from `cot` to `no-examples` — measured first, shipped second, which is the ordering
    Sprint 3 exists to have avoided reversing. The call was made on cost, not on
    out-performing `cot`: nothing here shows it is better, and the parity is a null result
    at n=24. What it shows is equivalence on every measured axis plus a 14× token saving,
    with §8's one remaining reason to keep the exemplars no longer applying. If the
    expansion to 46 questions puts `cot` ahead, this is a decision to revisit — both arms'
    artefacts are recorded, so that would be a re-read rather than a re-run.
  - **The gate that enforced "unmeasured prompts do not ship" was rewritten, not
    flipped.** It asserted `DEFAULT_PROMPT_VARIANT == "cot"` — which detects a change but
    not the mistake, passing on any rename and failing on a well-measured replacement. It
    now asserts the property: whatever ships must have recorded claim-level results over
    the whole eval set **and** must never falsely refuse. The second half is load-bearing
    by itself, since `baseline` is fully measured and refuses 7/24. Mutation-tested
    against `baseline`, `examples-only` and `cot`.
  - **Two harness defects found in passing, recorded not fixed:** `refusal_trials.run()`
    rebuilds its output dict from scratch and checkpoints per variant, so a mid-run failure
    writes back only the questions it reached and silently drops every other arm's cells
    for the rest — 72 paid-for cells were at risk here and were backed up by hand first.
    And `run_eval` still has no `preflight()` and an unflushed progress `print`, both of
    which §6 fixed in `refusal_trials` only.

---

## 3. What we're going to do (Sprints 2–9)

Reordered 2026-07-25 to pull evaluation forward: you can't judge whether the CoT prompting
change (new Sprint 3) actually helps without a groundedness/retrieval metric (new Sprint 2)
existing first, and the eval harness doubles as a safety-relevant baseline that Sprint 8
(refuse/redirect logic) should be validated against rather than built on an unmeasured RAG
pipeline. Reordering ahead of pytest/CI (now Sprints 4–5) is judged low-risk for a solo project
where one person holds full context — those sprints existed to protect a team's shared state,
which doesn't apply the same way here, and nothing about doing eval/CoT first makes Sprints 4–5
harder later.

| Sprint | Focus | Why it's here |
|---|---|---|
| **2** | Evaluation harness | Build the measurement tool everything after this depends on: Precision@K retrieval metric (reused from the AML fraud project) against a 20–30 question test set with known-correct source passages, plus a faithfulness/groundedness check (NLI-based or documented manual rubric) on generated answers. Report numbers honestly, weak spots included. |
| **3** ✅ | Chain-of-thought few-shot upgrade | Done — see §2. Implemented as six new held-out CoT exemplars used as a fixed set, rather than rewriting 5–8 of the existing 27 in place: with the selector at `k=1` only one example reaches the prompt, so rewriting a minority of 27 would have left ~70% of queries seeing no CoT demonstration at all, and would have diluted the A/B. |
| **4** ✅ | Automated testing foundation | Scoped as "add pytest"; became mostly a measurement sprint once the tests found that both the refusal suite and the out-of-corpus guard were selecting the wrong questions. 106 offline tests + 4 live, a socket guard that makes silently-fake mocks impossible, and three audit rounds. The out-of-corpus guard is closed at 60/60 across all 10 questions. The third round found that `is_refusal` itself — the instrument behind every refusal number in Sprints 2–4 — was scoring few-shot contamination as refusal; fixing it withdrew the sprint's own p=0.0219 headline and replaced it with **7/24 vs 0/24, p=0.0094** from a 5-trial re-run. |
| **5** ✅ | CI/CD pipeline | Done — see §2. GitHub Actions on push/PR, `windows-latest` (lint + the 131 offline tests, no secrets, no quota), ruff, pre-commit, dev deps split out. Scoped as plumbing; the sprint's actual result was discovering that Sprint 4's socket guard never covered module-scoped fixture setup, so two test modules had been downloading a 130MB model from the internet while their own comments said they could not. |
| **6** ◑ | Retrieval quality improvements | Half done — see §2. Chunk metadata and source citations shipped: the chunker was discarding the loaders' `source`/`page` one line after they were attached, so all 1225 chunks had empty metadata and no answer could cite anything. The index was rebuilt and proved bit-identical row by row before being swapped in, so no recorded eval number moved. The **relevance threshold is deliberately still open** — it changes what retrieval returns, so it needs the eval suite re-run against it, which costs quota. |
| **7** | Observability & UX polish | Replace `print()` logging with structured logging, finish wiring the now-live LangSmith key, implement real token streaming. |
| **8** | Safety & hardening | Add refuse/redirect logic for diagnosis/dosing/emergency questions beyond the general disclaimer — validated against Sprint 2's groundedness baseline, not an unmeasured one. |
| **9** (stretch) | Persistence & scale | Only if scope grows past a single-user local tool: persistent chat history, wider corpus, optional auth. |

**Working agreement:** sprints are scope units, not calendar locks — pace is whatever availability
allows.

---

*Historical detail on the Gemini/embeddings pivot itself (quota math, exact error messages, the
autocrlf corruption incident) is preserved below for anyone debugging a recurrence.*

## Appendix: Gemini embedding quota — root cause

Google's free-tier daily quota (`EmbedContentRequestsPerDayPerProjectPerModel-FreeTier`, limit
1000/day) counts **one unit per individual text embedded**, not one unit per `batch_embed_contents`
API call — confirmed by reading `langchain_google_genai/embeddings.py::embed_documents`. This
corpus (1225 chunks) could never finish in a single day via Gemini embeddings no matter how it was
batched or retried, which is why every earlier attempt stalled partway through. The fix that stuck
was dropping Gemini embeddings entirely in favor of local `fastembed` — no daily cap at all.

## Appendix: environment notes

- Python 3.11 is required on this machine — 3.13 has no `numpy<2` wheel needed by the pinned
  LangChain 0.2.x stack. Venv lives at `.venv-gemini` (repo root) — see Sprint 1 re: renaming.
- `core.autocrlf=true` with no `.gitattributes` previously corrupted the binary FAISS index on a git
  rename/checkout. Fixed by adding a repo-root `.gitattributes` marking `*.faiss`, `*.pkl`, `*.pdf`
  as binary — keep any future binary file types covered by it too.
