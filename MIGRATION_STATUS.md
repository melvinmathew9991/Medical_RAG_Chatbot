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

---

## 3. What we're going to do (Sprints 1–7)

Full backlog with effort/priority tags lives in the delivery-plan artifact from this session; summary:

| Sprint | Focus | Why it's next |
|---|---|---|
| **1** | Environment & repo hygiene | Close loose ends: decide `.venv-gemini`'s fate, delete confirmed-safe backup folders, reconcile this file and `PROJECT_REPORT.md` with reality, pin exact dependency versions. |
| **2** | Automated testing foundation | Zero tests exist today — every check this session was manual. Add `pytest`, unit tests for config/external-search, formalize the `AppTest` smoke script. |
| **3** | CI/CD pipeline | Tests nobody runs automatically aren't a safety net — GitHub Actions on push/PR, lint + pre-commit. |
| **4** | Retrieval quality & evaluation | Chain returns top-k chunks unconditionally, no citations. Add chunk metadata, a relevance threshold, source citations, and a real eval harness (RAGAS/LLM-judge). |
| **5** | Observability & UX polish | Replace `print()` logging with structured logging, finish wiring the now-live LangSmith key, implement real token streaming. |
| **6** | Safety & hardening | Add refuse/redirect logic for diagnosis/dosing/emergency questions beyond the general disclaimer. |
| **7** (stretch) | Persistence & scale | Only if scope grows past a single-user local tool: persistent chat history, wider corpus, optional auth. |

**Working agreement:** sprints are scope units, not calendar locks — pace is whatever availability
allows. Order matters: 1–3 exist so 4–6 aren't built on an unverifiable foundation.

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
