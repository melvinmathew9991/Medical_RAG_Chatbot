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
found is fixed and measured. `medbot/prompt.py` gained a `cot` prompt variant (now the default):
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
- **Attributed, not just observed.** A four-arm ablation separates the instruction rewrite from the
  worked exemplars (refusals over 20 trials): baseline 10, instruction-only 5, examples-only 1,
  cot 0. Breast cancer is fixed by either change alone; **bursitis is untouched by the instruction
  rewrite (5/5, same as baseline)** and needs the exemplars. The two are not substitutes and only
  the combination reaches zero — so the CoT exemplars earn their ~2,400 tokens per query.
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
    defects in the shipped behaviour. *(Sprint 4 closed the first: the refusal result is now
    significant at p=0.0219 on all 24 questions. See below.)*

**Sprint 4 (testing foundation) — mostly done (2026-07-25):** scoped as "add pytest", became
mostly a measurement sprint, because the tests immediately found that two of the three question
sets Sprint 3's result rested on were selecting the wrong questions. Full write-up in
`medbot/eval/results_sprint4.md`.

  - **103 tests: 99 offline (~6s, no network, no quota) + 4 live.** `pytest.ini` sets
    `addopts = -m "not live"`, so the default run is free and deterministic — which is what
    Sprint 5's CI needs, having no API key. (The sprint first reported "57 tests: 53
    offline"; the real figure before the audit round was 58/54 — see results_sprint4.md §6.)
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
  - **The out-of-corpus guard contained a question the corpus covers (audit F7/F8).** "What are
    the symptoms of diabetes?" was half the 2-question guard; its top chunks include the *blood
    sugar tests* entry (a B entry) explaining insulin and hyperglycemia. Exactly the F8 stroke
    mistake, still live one sprint after the lesson was recorded. Fixed with a tool rather than
    a resolution to be careful: `medbot/eval/verify_coverage.py` checks candidates against real
    retrieval, costs no quota, and rejected 14 of 26 candidates.
  - **NOT FINISHED — the hallucination guard is not armed.** The run hit the free tier's 500
    requests/day at question 2 of 10. No over-answering in what was recorded, but 1 of 10
    questions does not establish absence. `test_out_of_corpus_gate_is_fully_armed` skips with the
    missing questions listed rather than passing quietly. **Run this first, before any further
    prompt work:** `python -m medbot.eval.refusal_trials --trials 3 --suite overanswer
    --out-prefix sprint4_ --resume` (~54 calls).
    - Retried 2026-07-26 and **still quota-blocked**: `generate_content_free_tier_requests,
      limit: 500` on a bare one-token call. The daily counter rolls over at **midnight US
      Pacific (~12:30 IST)**, not local midnight — which is why "run it tomorrow morning"
      failed. Run it after ~12:30 IST.
  - **Judge calibration (F6) is partial and needs a human.** Length bias is *ruled out*: answers
    are +27.5% longer in the cot arm but the judge extracts +23.8% more claims — ~101 vs 104
    chars per claim, and the score is a ratio. Hedging bias remains untested, since the cot arm
    has almost no score variance left to correlate against. `medbot/eval/calibration_sheet.md`
    is ready to hand-label: 25 answers, arm and score hidden, order shuffled, context included.
    I cannot close this one — I am the model being audited, so my labels are not independent.

**Sprint 4 audit round (2026-07-26):** every headline number recomputed from the raw JSON
rather than read from the prose. The headline **survives unchanged** — 6/24 vs 0/24, Fisher
exact p=0.0219 — as do the calibration figures (781/995 chars, 7.71/9.54 claims), the claim
scores (0.841 → 0.997), Precision@4 (0.8333) and the 25-item calibration sheet. Four defects
found and fixed; full write-up in `medbot/eval/results_sprint4.md` §6.
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
| **4** ⚠️ | Automated testing foundation | Scoped as "add pytest"; became mostly a measurement sprint once the tests found that both the refusal suite and the out-of-corpus guard were selecting the wrong questions. 83 offline tests + 4 live, a socket guard that makes silently-fake mocks impossible, the refusal result re-measured on all 24 questions (p=0.0219), and an audit round that recomputed every headline from raw data and fixed four defects. **One piece still unfinished:** the out-of-corpus guard has 1 of 10 questions measured, blocked on the 500/day free-tier quota, which rolls over at midnight US Pacific (~12:30 IST). |
| **5** | CI/CD pipeline | Tests nobody runs automatically aren't a safety net — GitHub Actions on push/PR, lint + pre-commit. |
| **6** | Retrieval quality improvements | Use Sprint 2's findings to act on them: chunk metadata, a relevance threshold, source citations. (Building the harness itself moved to Sprint 2 — this is now about using it.) |
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
