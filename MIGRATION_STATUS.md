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
| **3** | Chain-of-thought few-shot upgrade | Rewrite 5–8 of the 27 few-shot examples as Question → Reasoning (what does the question ask, what does retrieved context say, does it actually support an answer) → Answer. Strip the reasoning trace before it reaches the user-facing answer. A/B against the original few-shot-only prompt using Sprint 2's harness — don't ship on vibes. |
| **4** | Automated testing foundation | Zero tests exist today. Add `pytest`, unit tests for config/external-search, formalize the `AppTest` smoke script, and a regression test that fails if Sprint 2/3's eval numbers drop. |
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
