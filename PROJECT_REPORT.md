# MEDBOT — Medical Retrieval-Augmented Generation Chatbot: Project Report

**Repository:** https://github.com/melvinmathew9991/Medical_RAG_Chatbot
**Stack:** Python · LangChain · OpenAI · FAISS · Streamlit
**Status:** Working prototype (restructured into standard layout)
**Original active development:** 2024-05-23 → 2024-06-15 (26 commits)
**Structural refactor:** 2026-07-06

---

## 1. Overview & Business Problem

MEDBOT is a single-user, browser-based medical question-answering assistant. It exists to shorten the distance between "I have a health question" and "a grounded, cited-in-spirit answer," without requiring the user to manually search a medical reference, PubMed, or the open web separately.

The underlying business problem it addresses: general-purpose chat models answer confidently but ungrounded, which is especially risky in a medical context where a plausible-sounding but wrong answer has real consequences. MEDBOT's core idea is to constrain the model's answer to a trusted medical reference text via retrieval-augmented generation (RAG), while giving the user a path to corroborate that answer against live external sources (PubMed literature, Wikipedia, general web search) in the same turn.

Judging by a hardcoded path originally left in `config.py` (`PERSIST_DIR = "E:/brototype/Langchain/Ollama/test_chatbot"`), this project was built as a learning/portfolio exercise — most plausibly during a bootcamp — rather than as a production clinical system. This report evaluates it as a well-executed applied-AI prototype, not as a certified medical product.

---

## 2. Use Case & Objectives

**Primary use case:** a patient, student, or caregiver types a plain-language medical question into a chat interface and receives (a) an answer grounded in a medical encyclopedia, and (b) supplementary results pulled live from PubMed, Wikipedia, and Google.

**Objectives:**
- Stand up a working RAG pipeline over a real medical reference text rather than relying on the LLM's parametric knowledge alone.
- Demonstrate multi-source retrieval: one local/offline vector index plus three live external APIs.
- Use few-shot prompting with semantic example selection to steer answer tone and format toward concise, clinically-styled responses.
- Package the whole thing behind a minimal conversational UI (Streamlit chat) with session-scoped history.
- Make the environment reproducible via a Dev Container so the app can be opened directly in GitHub Codespaces.

---

## 3. Proposed Solution

A single-process Streamlit application that, on startup, builds (or loads) a FAISS vector index from a medical PDF, then on every user turn runs two things: a LangChain `RetrievalQA` chain (local knowledge, grounded) and a fan-out to three external search sources (fresh, uncurated). The few-shot prompt template — with examples chosen by semantic similarity to the current question — is the intended mechanism for keeping answers in a consistent medical-FAQ voice, though it is not currently wired into the live query path (see §13).

---

## 4. Project Architecture

The system is a modular monolith — one Streamlit process, no separate backend service. Logic lives in an installable-shaped `medbot` package instead of loose root-level scripts:

| Layer | Module | Responsibility |
|---|---|---|
| UI / orchestration | `app.py` | Streamlit chat UI, session history, wires every other module together |
| Configuration | `medbot/config.py` | Loads `.env`, exposes API keys, and resolves `DATA_DIR`/`PERSIST_DIR` relative to the repo root |
| Ingestion / indexing | `medbot/data_processing.py` | Loads PDFs/text from `data/`, chunks them, embeds with OpenAI, builds/loads the FAISS index |
| Model | `medbot/model_handler.py` | Constructs and caches the `ChatOpenAI` LLM instance |
| Retrieval + QA chain | `medbot/query_handler.py` | Builds the `RetrievalQA` chain over the FAISS retriever; fans out to external search |
| External search | `medbot/external_search.py` | PubMed (threaded + LRU-cached), Wikipedia (async via aiohttp, LRU-cached), Google via SerpAPI |
| Prompting | `medbot/prompt.py` | Few-shot template with a semantic-similarity example selector over a hand-written medical Q&A set |
| Legacy / experimental | `medbot/legacy/retriever.py` | Standalone FAISS retriever over `speech.txt`; isolated into its own subpackage, still not imported by `app.py` |

Data flow is linear and synchronous: `app.py` calls into the model and vector-store modules at import time, then calls `query_handler` per user turn, which calls `external_search`. There is no message queue, no database, and no persistence layer beyond the FAISS index files and the ephemeral Streamlit session state.

---

## 5. Folder Structure

```
Medical_RAG_Chatbot/
├── .devcontainer/
│   └── devcontainer.json        # runs `streamlit run app.py`
├── .vscode/
│   └── settings.json
├── data/
│   └── 71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf
├── medbot/
│   ├── __init__.py
│   ├── config.py                 # PERSIST_DIR/DATA_DIR relative to repo root, env-overridable
│   ├── data_processing.py
│   ├── external_search.py
│   ├── model_handler.py
│   ├── prompt.py
│   ├── query_handler.py
│   └── legacy/
│       ├── __init__.py
│       ├── retriever.py          # path to speech.txt relative to this file, not CWD
│       └── speech.txt
├── vectorstore/
│   ├── index.faiss
│   └── index.pkl
├── .env.example                  # documents required env vars
├── .gitignore                    # ignores __pycache__/, *.pyc
├── app.py                        # Streamlit entrypoint; imports from `medbot.*`
├── README.md                     # setup instructions, layout, notes
├── PROJECT_REPORT.md             # this file
└── requirements.txt
```

**Prior layout, for reference** (before the 2026-07-06 restructure): all logic sat as loose scripts at the repo root (`main.py`, `config.py`, `data_processing.py`, `external_search.py`, `model_handler.py`, `prompt.py`, `query_handler.py`, `retriever.py`, `speech.txt`), the source PDF lived in `docs/`, the FAISS index files (`index.faiss`, `index.pkl`) sat at the repo root, and `__pycache__/*.pyc` files were accidentally committed to git.

**What changed and why:**
- All application logic moved from loose root-level scripts into a `medbot` package, with every `from config import X`-style import rewritten to `from medbot.config import X`.
- `docs/` → `data/` (the actual convention for source/input data, reserving "docs" for documentation).
- `index.faiss` / `index.pkl` → `vectorstore/` (the generated artifact gets its own clearly-labeled directory instead of sitting loose at the repo root).
- The dead `retriever.py` + its unrelated `speech.txt` sample were moved into `medbot/legacy/`, making the "this is not part of the live app" status structural rather than something you have to notice by reading the code.
- `main.py` → `app.py` (the Streamlit-idiomatic entrypoint name); `.devcontainer/devcontainer.json`'s `postAttachCommand` and `openFiles` were updated to match.
- Added `README.md` and `.env.example`, which the project lacked entirely (the devcontainer config even referenced a `README.md` that never existed).
- `__pycache__/*.pyc` files were found accidentally committed to git; removed and added to `.gitignore`.

---

## 6. End-to-End Workflow

1. **Startup.** `app.py` imports trigger `initialize_model()` (constructs a cached `ChatOpenAI`, temperature 0.1, streaming enabled) and `create_vector_database()`, from `medbot.model_handler` / `medbot.data_processing`.
2. **Index build or load.** If `vectorstore/index.faiss` doesn't exist, `medbot/data_processing.py` loads every PDF/`.txt` under `data/` (via the `DATA_DIR` constant) concurrently, concatenates their text, splits it with `RecursiveCharacterTextSplitter` (chunk size 3000, overlap 300), embeds with `text-embedding-ada-002`, and writes a FAISS index to `vectorstore/`. If it already exists — which it does in this repo — the index is deserialized directly (`allow_dangerous_deserialization=True`).
3. **Chat UI render.** Streamlit renders prior turns from `st.session_state.history` and shows a `st.chat_input` box.
4. **User submits a question.** The question is appended to history and echoed in the chat.
5. **Grounded answer.** `medbot.query_handler.create_query_chain` wraps the FAISS index as a retriever and builds a `RetrievalQA` chain with the cached `ChatOpenAI` model; `.invoke({"query": ...})` returns `{"result": ...}`.
6. **External corroboration (computed, not shown).** In parallel, `medbot.query_handler.search_external_sources` queries PubMed, Wikipedia, and Google. The combined dict is folded into a local `response_content` string in `app.py` that is never rendered (see §13).
7. **Response rendered.** Only `response['result']` is appended to session history and displayed in the chat bubble.

---

## 7. Technologies Used

| Category | Technology | Role |
|---|---|---|
| UI | Streamlit 1.35 | Chat interface, session state, spinner |
| Orchestration | LangChain 0.2.x (core, community, openai) | Document loaders, text splitting, `RetrievalQA`, few-shot prompting, example selection |
| LLM | OpenAI `ChatOpenAI` (via `langchain_openai`) | Answer generation, temperature 0.1, streaming enabled |
| Embeddings | OpenAI `text-embedding-ada-002` | Document and example embeddings for similarity search |
| Vector store | FAISS (`faiss-cpu`) | Nearest-neighbor retrieval over encyclopedia chunks; final choice after an earlier Chroma-based approach |
| Document ingestion | `PyPDFLoader`, `TextLoader`, `DirectoryLoader`, `pypdf` | Load PDF/text sources from `data/` |
| External literature search | `PubMedLoader` (langchain_community) | Live query against PubMed |
| External encyclopedic search | Wikipedia MediaWiki API + `aiohttp` | Async live Wikipedia snippet search |
| External web search | SerpAPI (`google-search-results`) | Live Google organic results |
| Concurrency | `concurrent.futures.ThreadPoolExecutor`, `asyncio`/`aiohttp` | Parallel document loading and async HTTP for Wikipedia |
| Caching | `functools.lru_cache`, a hand-rolled memoization decorator | Avoid recomputation of PubMed/Wikipedia queries and model init |
| Config & secrets | `python-dotenv` | Loads `OPENAI_API_KEY`, `SERPAPI_API_KEY`, `LANGCHAIN_API_KEY` (+ optional `DATA_DIR`/`PERSIST_DIR` overrides) from `.env` |
| Observability | LangSmith (`LANGCHAIN_TRACING_V2`) | Tracing hook enabled in config, not otherwise instrumented |
| Environment | Dev Container (`mcr.microsoft.com/devcontainers/python:3.11`) | Reproducible Codespaces environment, auto-runs `streamlit run app.py` |
| Version control | Git / GitHub | 26 commits pre-restructure, single `master` branch |

*Historical note:* `medbot/model_handler.py` still imports `langchain_community.llms.Ollama` (unused) — a remnant of an earlier local-LLM approach abandoned in favor of the OpenAI API.

---

## 8. Implementation Details

**Ingestion & chunking.** `create_vector_database()` loads every PDF/text file under `DATA_DIR` (`data/`) in parallel threads, joins all page content per file type into one long string per type, then runs `RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)` over each concatenated blob before combining the two chunk lists. Concatenating first and splitting second means chunk boundaries are not aligned to page or document boundaries.

**Configuration.** `medbot/config.py` derives `BASE_DIR` from `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (the repo root, computed from the package's own location) and sets `DATA_DIR`/`PERSIST_DIR` relative to it, each overridable via an environment variable. This replaces the original hardcoded absolute path and is the one behavioral fix bundled with the restructuring (folder moves alone would have left the app unable to find its index under the new `vectorstore/` location).

**Retrieval.** `medbot.query_handler.create_query_chain` calls `vectordb.as_retriever()` with default search parameters and feeds it into `RetrievalQA.from_chain_type`'s default "stuff" combination method — all retrieved chunks are concatenated directly into the prompt context.

**Prompting.** `medbot/prompt.py`'s `LazyLoader` builds a 27-example few-shot template with a semantic example selector (`SemanticSimilarityExampleSelector`, k=1, backed by its own small FAISS index). Still not invoked from `app.py`'s live query path (§13).

**External search.** Three independently-implemented integrations: PubMed via a synchronous loader wrapped in an LRU cache; Wikipedia via a hand-rolled async `aiohttp` call also LRU-cached; SerpAPI via a synchronous `GoogleSearch` call. Each is individually try/excepted, logging to stdout and degrading to an empty list/dict on failure.

**Legacy isolation.** `medbot/legacy/retriever.py`'s hardcoded `TextLoader("./speech.txt")` (a CWD-relative path that only worked if the app happened to be launched from the repo root) was fixed to resolve relative to the module's own file location, so the legacy module remains internally correct even though it is still unused.

---

## 9. Methodology

Development followed an iterative, exploratory pattern typical of a learning project, visible directly in the commit history — 26 commits over roughly 3.5 weeks:

| Date | Milestone |
|---|---|
| 2024-05-23 | first commit — initial scaffold |
| 2024-05-24 | second commit |
| 2024-06-03 | prompt updation — few-shot prompting introduced |
| 2024-06-04 | error handling added across modules |
| 2024-06-05 | wikipedia search integration |
| 2024-06-06 | hallucination — mitigation work on grounding |
| 2024-06-08 | optimization pass |
| 2024-06-10 → 06-11 | chroma → 8 commits fighting sqlite3 version incompatibility → migrated to FAISS |
| 2024-06-11 | pypdf loader added |
| 2024-06-13 | OpenAI integration finalized, requirements cleaned up |
| 2024-06-13 → 06-15 | Dev Container added for Codespaces reproducibility |
| 2026-07-06 | restructured into standard package layout |

The original methodology was **trial-and-integrate**: bring in one capability at a time, validate manually, commit once it works — no branching strategy, no PR review, no TDD. The restructuring session followed a verification-first approach instead: move files with `git mv` (preserving history), fix every import, then validate the fix statically before calling it done (§13).

---

## 10. Results

The repository represents a working, runnable prototype, organized as a standard `app.py` + `medbot` package layout instead of loose scripts. Given an `OPENAI_API_KEY` (and optionally `SERPAPI_API_KEY`), `streamlit run app.py` loads the pre-built FAISS index from `vectorstore/` over the encyclopedia in `data/` and answers free-text medical questions through a `RetrievalQA` chain. The restructuring is a pure reorganization plus one portability fix (relative `DATA_DIR`/`PERSIST_DIR`) — it does not change the app's functional behavior, including the two disconnected features noted in §13.

---

## 11. Evaluation Metrics

No formal evaluation harness exists in the repository — no test files, no evaluation scripts, no logged accuracy/quality metrics. The only quantitative signal captured anywhere is wall-clock timing printed to stdout at four points:

| Instrumented step | Where | What's measured |
|---|---|---|
| Vector DB build/load | `medbot.data_processing.create_vector_database` | Seconds to build or deserialize the FAISS index |
| Model initialization | `medbot.model_handler.initialize_model` | Seconds to construct the `ChatOpenAI` client |
| PubMed search | `medbot.external_search.threaded_search_pubmed` | Seconds per PubMed query |
| Wikipedia search | `medbot.external_search.search_wikipedia` | Seconds per Wikipedia query |
| SerpAPI search | `medbot.external_search.search_serpapi` | Seconds per Google query |

There is no retrieval-quality metric (precision@k, recall, MRR), no generation-quality metric (faithfulness/groundedness, answer relevancy), and no user-facing feedback loop.

---

## 12. Result Analysis

The RAG core is sound and idiomatic LangChain — load, split, embed, index, retrieve, stuff, generate — and this reasoning is unaffected by *where* the files live. The two secondary features diverge from their apparent design intent:

- **Few-shot prompting** is fully built (27 examples, semantic selector, template) but structurally disconnected from the live query path, so it has no effect on answers as shipped.
- **External corroboration** is fully built and functionally correct in isolation, but its output is discarded before reaching the user.

Retrieval quality is bounded by a single 3000-character chunk size with no metadata (page numbers, section titles) attached, and default (unranked, unfiltered) FAISS similarity search with no relevance threshold — the chain always returns its top-k chunks even when none are relevant to an out-of-scope question.

---

## 13. Auditing & Validation of Outcomes

**Verification method for this restructuring:** rather than running the full Streamlit app (which needs a paid OpenAI key), validation was done in two zero-cost passes:
1. A cross-file static check parsed every changed file's AST and confirmed all 11 `medbot.*` imports resolve to real names in their target modules — **passed**.
2. An attempt to install `requirements.txt` into a throwaway venv for a fuller mocked-network smoke test was blocked by pinned dependency versions incompatible with the available Python (see finding below) and was abandoned in favor of the static check, keeping verification zero-network and zero-cost.

**Findings carried over, unresolved (require behavior changes, not just restructuring):**

- **Dead result path.** `external_results`/`response_content` in `app.py` are computed from all three external sources but never rendered — only `response['result']` reaches the UI.
- **Unused prompting layer.** `app.py`'s `format_prompt()` wraps the few-shot template but is never called from `process_user_input`.
- **Silent failure handling.** Broad `try/except Exception` blocks across `medbot/data_processing.py`, `medbot/query_handler.py`, `medbot/external_search.py` degrade silently to `None`/`[]`/`{}` with only a stdout print — a misconfigured API key or network failure produces no message in the Streamlit UI itself.
- **Deserialization trust.** `FAISS.load_local(..., allow_dangerous_deserialization=True)` unpickles the committed `index.pkl` without restriction — safe given its trusted provenance, worth flagging as a pattern.
- **No medical safety framing.** No disclaimer language anywhere in the prompt or UI stating responses are informational, not a substitute for professional medical advice.

**Findings resolved by this restructuring:**

- **Non-portable configuration — fixed.** `PERSIST_DIR`/`DATA_DIR` are now computed relative to the repo root and overridable via env vars, instead of a developer's personal `E:/brototype/...` path.
- **Dead module ambiguity — improved.** `retriever.py` and `speech.txt` are now structurally isolated under `medbot/legacy/`, with its internal path bug (CWD-relative `TextLoader` path) also fixed, and the README documents it explicitly as not part of the live app.
- **Accidentally committed bytecode — fixed.** `__pycache__/*.pyc` files were tracked in git; removed and added to `.gitignore`.

**New finding — stale/unused dependencies in `requirements.txt`:**
`langchain_chroma==0.1.1`, `chromadb==0.5.0`, and `chroma-hnswlib==0.7.3` are still pinned even though the code fully migrated to FAISS (per the `chroma to faiss` commit) and nothing imports Chroma anymore. This was confirmed directly: attempting to install `requirements.txt` for a smoke test failed immediately because `langchain_chroma==0.1.1` has no build for the available Python version. Separately, `faiss-cpu==1.8.0` also has no prebuilt wheel for Python 3.13, and installing without that pin pulled a `numpy` version that tried to compile from source and failed for lack of a C compiler — meaning `requirements.txt`'s pins are now stale against current Python releases, independent of the Chroma issue.

---

## 14. Challenges Faced

*(Original development)*
- **Vector store instability.** Eight consecutive commits fighting the well-known Chroma/SQLite version-incompatibility issue before migrating to FAISS entirely.
- **Hallucination control.** A dedicated commit shows deliberate grounding work.
- **Bootcamp-style hardcoding.** The absolute `PERSIST_DIR` path (fixed in this restructuring).
- **Multi-source integration complexity.** Reconciling three different concurrency models into one `search_external_sources` call.

*(This restructuring session)*
- **Verifying without execution.** With no API key and no intention to spend one just to check file moves, the main challenge was finding a verification method with real signal (catches actual broken imports) but zero cost — landing on AST-based static resolution after a real dependency install attempt failed on Python-version-incompatible pins.
- **Dependency staleness surfaced by the install attempt.** Confirmed `requirements.txt` itself is now a source of friction independent of code correctness.

---

## 15. Limitations

- Knowledge base is a single PDF (volume 1 only of a multi-volume encyclopedia) — coverage is necessarily partial.
- External search results are computed but not surfaced in the UI.
- Few-shot prompt template is built but not wired into the live query path.
- No automated tests, no CI, no evaluation harness of any kind.
- No retrieved-source citation is shown alongside answers.
- No medical disclaimer or safety guardrail language anywhere in the prompt or UI.
- Chat history is held only in `st.session_state` — lost on refresh; no database-backed persistence, no auth.
- Streaming is enabled on the LLM client but never consumed via a streaming callback or `st.write_stream`.
- Errors are logged to stdout only, not surfaced in the Streamlit UI.
- `requirements.txt` pins versions (`faiss-cpu==1.8.0`, `langchain_chroma==0.1.1`, etc.) that are no longer installable on current Python releases without either downgrading Python or re-pinning.

---

## 16. Future Improvements

- Render the already-computed PubMed/Wikipedia/SerpAPI results in the UI.
- Wire the few-shot template into `create_query_chain`/`process_user_input`.
- Add source attribution (retrieved chunk/page) alongside each answer.
- Add an automated evaluation harness (retrieval precision/recall, groundedness/faithfulness — e.g. RAGAS or an LLM-judge).
- Add an explicit medical-disclaimer system prompt and refuse/redirect logic for diagnosis/dosing requests.
- Consume the model's existing `streaming=True` setting via a callback handler or `st.write_stream`.
- Expand the corpus beyond volume 1 of a single encyclopedia; attach richer chunk metadata at split time.
- Move chat history to persistent storage.
- Replace print-based timing with structured logging; finish wiring the already-configured LangSmith tracing.
- Clean up `requirements.txt` — drop the unused `langchain_chroma`/`chromadb`/`chroma-hnswlib` entries, and re-pin `faiss-cpu`/`numpy`/the LangChain stack to versions with prebuilt wheels for a current Python release (or pin a supported Python version explicitly, e.g. via `.python-version`).
- Consider adding a `pyproject.toml` if the `medbot` package ever needs to be installed (`pip install -e .`) rather than relied upon via repo-root-relative imports.

---

## Condensed Summaries

### Five-bullet summary
- MEDBOT is a Streamlit chatbot that answers medical questions using retrieval-augmented generation over the Gale Encyclopedia of Medicine (vol. 1), via a LangChain `RetrievalQA` chain and OpenAI's chat and embedding models, organized as a standard `app.py` + `medbot` package layout.
- Documents are chunked, embedded, and indexed with FAISS (in `vectorstore/`) — the project migrated to FAISS mid-development after hitting SQLite version conflicts with an earlier Chroma-based store, though unused Chroma packages still linger in `requirements.txt`.
- It also implements live external lookups against PubMed, Wikipedia, and Google (via SerpAPI), plus a semantic few-shot prompt template for answer styling — both fully built but not connected to the answer the user actually sees.
- A restructuring pass moved all logic into a proper package, isolated dead code into `medbot/legacy/`, fixed a hardcoded personal file path into a portable relative config, and was verified with a zero-cost static import check after a real dependency install hit Python-version incompatibilities.
- The project has no automated tests or evaluation metrics; it's a working single-user prototype (built during a learning/bootcamp context) rather than a production or clinically-validated system.

### Three-bullet summary
- A RAG-based medical Q&A chatbot (Streamlit + LangChain + OpenAI + FAISS) grounded in a medical encyclopedia, with live PubMed/Wikipedia/Google search built in alongside it, organized into a standard package layout (`app.py` + `medbot/`).
- Functionally working end-to-end, but two features — external-source display and few-shot prompting — remain implemented in code without being connected to the live user-facing flow.
- No test suite or evaluation metrics exist; the restructuring was validated via a static, zero-network import check rather than a full run, since real execution needs a paid API key and the pinned dependencies no longer install cleanly on a current Python.

### One-line description
MEDBOT is a Streamlit + LangChain medical Q&A chatbot combining FAISS-based retrieval-augmented generation over a medical encyclopedia with live PubMed/Wikipedia/Google search, organized as a standard `app.py` + `medbot` package layout with portable configuration.
