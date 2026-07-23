# Gemini Migration — Status & Resume Guide

**Why:** the user has no OpenAI key and can't run local LLMs (laptop can't handle it), so chat/generation
is being swapped from OpenAI to Google Gemini's free tier (Google AI Studio key, no credit card).
Embeddings ended up pivoting to a **local** model instead (see below) — the laptop handles that fine,
it's only full LLM inference that's out of reach. This file is the single place to check "where did we
leave off."

Last updated: 2026-07-23 (index rebuild completed).

## Current state: DONE — index rebuilt successfully with local embeddings, no quota involved

The Gemini embedding free-tier quota (1000 requests/day, consumed per individual text — see historical
note below) could never fit this corpus. Rather than keep shrinking chunk size to squeeze under the cap,
the fix that stuck was **dropping Gemini for embeddings entirely** and switching to local `fastembed`
(CPU, ONNX, `BAAI/bge-small-en-v1.5`, via `langchain_community.embeddings.FastEmbedEmbeddings`) — no API
key, no quota, no rate limit. Gemini (`gemini-flash-lite-latest`) is still used for chat/generation.

Separately, the on-disk index also got **corrupted** at one point: `core.autocrlf=true` with no
`.gitattributes` mangled the binary `vectorstore/index.faiss` during a git rename/checkout (`faiss.read_index`
failed with "Index type not recognized"). Fixed by adding a repo-root `.gitattributes` marking `*.faiss`,
`*.pkl`, `*.pdf` as binary. The corrupted copy is preserved at `vectorstore-corrupted-backup/` — safe to
delete once the app has been used successfully a few times.

**Rebuild result (2026-07-23, ~12:20 local):** all **1223/1223 chunks** embedded locally in ~887 seconds
(~15 min, one-time CPU cost, no quota wall hit). Verified with `similarity_search("what is diabetes", k=2)`
— returned correct, relevant passages. `chunk_size` is back to the original `3000`/`300` overlap (no longer
needs to be inflated to dodge a quota, since local embeddings have none).

**Root cause, now understood (was mis-diagnosed earlier):** Google's free-tier daily quota
(`EmbedContentRequestsPerDayPerProjectPerModel-FreeTier`, limit 1000/day) counts **one unit per
individual text embedded**, not one unit per `batch_embed_contents` API call. Confirmed by reading
`langchain_google_genai/embeddings.py::embed_documents` — it groups texts into `BatchEmbedContentsRequest`
calls, but each text inside still consumes one quota unit server-side. This means the *original*
`chunk_size=3000` setup (1223 chunks) could **never** complete in one day — it needed more requests than
the entire daily allowance, regardless of retries or batching. That's why every previous attempt stalled
partway through.

Confirmed via the AI Studio dashboard (2026-07-23, ~11:24 local / ~05:54 Pacific): `Gemini Embedding 1` —
RPM 102/100, TPM 51.82K/30K, **RPD 1K/1K (fully capped)**. All three limits were exceeded, not just RPD.
**Do not attempt any more embedding calls until this resets** — check the same dashboard
(https://aistudio.google.com/apikey → rate limits) before retrying. Reset timing is unconfirmed — may not
be a simple Pacific-midnight calendar reset (it was still capped this many hours after that point), possibly
a rolling 24h window instead. Wait and re-check the dashboard rather than guessing.

## Fix applied today: chunk_size 3000 → 6000 (medbot/data_processing.py)

`chunk_overlap` scaled proportionally (300 → 600). This roughly halves total chunks to ~600-650, which
**does** fit under the 1000/day cap in a single sitting (unlike the old 1223), assuming no leftover quota
usage that day. Trade-off accepted: coarser retrieved context per chunk (user decision, 2026-07-23).

Because chunk boundaries shifted, the old partial index (20/1223 chunks under the old scheme) was
**deleted** (`vectorstore/index.faiss`, `vectorstore/index.pkl`) rather than resumed — old and new chunk
lists don't line up, so keeping it would have silently corrupted the resume-by-count logic in
`create_vector_database()`. Loss was trivial (20 chunks). Next rebuild starts clean at 0/~600-650.

The code is still checkpointed (saves after every batch), so once quota clears, a rebuild attempt that
gets partway through and hits the wall again will not lose progress — reruns resume from
`vectordb.index.ntotal`.

### Remaining steps (index rebuild itself is done):

1. ~~Rebuild the index~~ — done, 1223/1223 chunks, retrieval spot-checked and working.
2. Run the actual app end-to-end: `.venv-gemini/Scripts/python.exe -m streamlit run app.py` (or activate
   `.venv-gemini` first) and try a real question through the UI, not just direct `similarity_search`.
3. Decide on cleanup: delete `vectorstore-corrupted-backup/` and `vectorstore-openai-backup/` once the
   app has been confirmed working a few times.
4. Commit the (currently entirely uncommitted) restructure + migration — see "What's NOT done yet" below.

## What's done

- **Provider swap** — `medbot/model_handler.py`, `medbot/data_processing.py`, `medbot/prompt.py` all
  now use `langchain_google_genai` (`ChatGoogleGenerativeAI`, `GoogleGenerativeAIEmbeddings`) instead
  of OpenAI equivalents.
- **Config fixes** (`medbot/config.py`):
  - Switched `OPENAI_API_KEY` → `GOOGLE_API_KEY`.
  - Fixed a crash: `os.environ["X"] = os.getenv("X")` used to raise `TypeError` if a var was fully unset.
  - Fixed a second, related bug: `.env` declares optional vars as blank lines, which `python-dotenv` sets
    to `""` — and `os.getenv(key, default)`'s default only applies when the var is *absent*, not empty.
    Every optional override (`DATA_DIR`, `PERSIST_DIR`, `GEMINI_CHAT_MODEL`, `GEMINI_EMBEDDING_MODEL`)
    was silently resolving to `""` until this was changed to `os.getenv(key) or default`.
- **Model choice** (also in `config.py`, both overridable via `.env`):
  - `GEMINI_CHAT_MODEL` = `gemini-flash-lite-latest` — chosen over `gemini-flash-latest` because, per
    the AI Studio quota dashboard, "Flash Lite" models get **15 RPM / 500 RPD** vs. full "Flash" models'
    **5 RPM / 20 RPD**. Much more usable for actual day-to-day chatting.
  - `GEMINI_EMBEDDING_MODEL` = `models/gemini-embedding-001` — `text-embedding-004` (the old obvious
    choice) no longer exists on this account; confirmed via `genai.list_models()`.
  - Both accounts are **new-user-restricted** from older models — `gemini-2.0-flash` and
    `gemini-2.5-flash` both returned errors (`limit: 0` / "no longer available to new users") when tested.
- **Resumable index rebuild** (`medbot/data_processing.py`) — rewritten to save the FAISS index to disk
  after *every* batch (not just at the end), and to resume from `vectordb.index.ntotal` on the next run
  instead of restarting from scratch. Retry logic distinguishes per-minute throttling (worth a short
  backoff) from daily-quota errors (bail immediately, no point backing off).
- **Environment fix (unrelated to Gemini, blocking regardless)** — `requirements.txt`'s pinned
  `langchain==0.2.3` requires `numpy<2`, which has no prebuilt wheel for Python 3.13 (the only Python on
  this machine) and no C compiler is installed to build from source. Fixed by installing **Python 3.11**
  via `winget install --id Python.Python.3.11 --scope user`, and creating a fresh venv at
  **`.venv-gemini`** (repo root) using that interpreter. Also had to relax `langchain_core==0.2.5` →
  `langchain_core==0.2.43` and `langsmith==0.1.75` → `langsmith>=0.1.112,<0.2.0` to satisfy
  `langchain-google-genai==1.0.10`'s dependency range. All of `requirements.txt` now installs cleanly
  in that venv.
- **`.env`** created locally (gitignored, not committed) with the user's real `GOOGLE_API_KEY`.
- **`.gitignore`** updated: `.venv-gemini` wasn't covered by the old `venv` entry (different name) — now
  covered by a `.venv*` pattern; also added `vectorstore-openai-backup/`.
- **Old OpenAI-embedded FAISS index** moved (not deleted) to `vectorstore-openai-backup/` — safe fallback,
  not committed, not needed once the Gemini index is complete.
- Separately from the Gemini migration, earlier in this session: wired the few-shot prompt and external
  search results into the live `app.py` path, added a medical disclaimer, fixed silent-failure UI
  handling, and cleaned up `requirements.txt` (dropped unused Chroma packages, unused `Ollama` import).

## What's NOT done yet

- **Full end-to-end app test** — index rebuild is done and retrieval spot-checked directly, but
  `streamlit run app.py` itself hasn't been run yet.
- **`.venv-gemini`'s long-term status** — currently a throwaway test env. Needs a decision: keep it as
  the project's permanent environment (maybe rename it), or the user manages Python environments
  differently.
- **Backup cleanup** — `vectorstore-openai-backup/` and `vectorstore-corrupted-backup/` (new, from the
  autocrlf corruption incident) can both be deleted once the app is confirmed working, or kept as
  fallbacks — undecided.
- **Git** — a large restructure from earlier in the session (loose scripts → `medbot/` package, plus all
  the fixes and this whole migration) is sitting **entirely uncommitted**. Needs a decision on how to
  split it into commits now that everything is verified working. Also uncommitted: the new
  `.gitattributes` (binary markers for `*.faiss`/`*.pkl`/`*.pdf`).
- **Streaming** — `ChatGoogleGenerativeAI` doesn't take the old `streaming=True` constructor kwarg the
  way `ChatOpenAI` did; it was dropped rather than guessed at. Not consumed by the UI anyway (pre-existing
  gap). Would need an LCEL-style chain to actually stream tokens into Streamlit.
- **Tests / CI / eval harness** — never existed, not addressed.

## Key facts to remember

- Total chunks: **1223** (from `data/71763-gale-encyclopedia-of-medicine.-vol.-1.-2nd-ed.pdf`,
  `RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)`), all embedded and persisted.
- Embeddings run **locally** via `fastembed` (`BAAI/bge-small-en-v1.5`, CPU/ONNX) — no API key, no quota,
  no rate limit. Took ~887 seconds (~15 min) one-time to embed all 1223 chunks on this machine.
- Chat quota: `gemini-flash-lite-latest` → 15 RPM / **500 RPD** — this is the only remaining Gemini quota
  concern in the app (embeddings no longer touch Gemini at all).
- Python 3.11 is required for this project on this machine — Python 3.13 cannot install the pinned
  LangChain 0.2.x stack. Use `.venv-gemini\Scripts\python.exe` (or activate it) for everything.
- `core.autocrlf=true` + missing `.gitattributes` previously corrupted the binary FAISS index on a git
  rename/checkout — now fixed with a repo-root `.gitattributes`. Keep binary vectorstore/PDF files
  covered by it going forward.
