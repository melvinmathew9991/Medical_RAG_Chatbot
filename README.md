# MEDBOT — Medical RAG Chatbot

A Streamlit chat assistant that answers medical questions using retrieval-augmented generation (RAG) over the Gale Encyclopedia of Medicine, combined with live PubMed, Wikipedia, and Google search.

## Project layout

```
Medical_RAG_Chatbot/
├── app.py                  # Streamlit entrypoint
├── medbot/                 # application package
│   ├── config.py           # env vars, API keys, data/vectorstore paths
│   ├── data_processing.py  # document loading, chunking, FAISS index build/load
│   ├── external_search.py  # PubMed / Wikipedia / SerpAPI search
│   ├── model_handler.py    # ChatGoogleGenerativeAI (Gemini) initialization + caching
│   ├── prompt.py           # few-shot prompt template + example selector
│   ├── query_handler.py    # RetrievalQA chain + external search orchestration
│   ├── eval/               # evaluation harness, results, and audits
│   └── legacy/             # experimental code not wired into app.py
│       ├── retriever.py
│       └── speech.txt
├── tests/                  # pytest suite (see "Tests" below)
├── data/                   # source documents ingested into the vector index
├── vectorstore/            # committed FAISS index (index.faiss, index.pkl)
├── .github/workflows/ci.yml  # lint + offline test suite on every push and PR
├── .devcontainer/          # GitHub Codespaces / VS Code dev container
├── .env.example            # required environment variables
├── .pre-commit-config.yaml
├── ruff.toml
├── pytest.ini
├── requirements.txt        # runtime
└── requirements-dev.txt    # runtime + pytest, ruff, pre-commit
```

## Setup

Requires **Python 3.11** — the pinned LangChain 0.2.x stack needs `numpy<2`, which has no
prebuilt wheel for Python 3.13.

1. Create a Python 3.11 virtual environment (e.g. `.venv-gemini`) and activate it.
2. `pip install -r requirements.txt` — or `pip install -r requirements-dev.txt` to get the
   test and lint tooling as well.
3. Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` (required — free key from
   [Google AI Studio](https://aistudio.google.com/apikey), no credit card) and `SERPAPI_API_KEY`
   (optional, needed for Google search corroboration).
4. `streamlit run app.py`

Chat/generation uses Google Gemini (`gemini-flash-lite-latest` by default, overridable via
`GEMINI_CHAT_MODEL` — chosen for its 500 requests/day free-tier quota vs. 20/day for full Flash).
Embeddings run **locally** via `fastembed` (`BAAI/bge-small-en-v1.5`, CPU/ONNX) — no API key, no
quota, since Gemini's free embedding quota can't cover this corpus.

On first run, `medbot/data_processing.py` builds a FAISS index from every PDF/text file in `data/`
and saves it to `vectorstore/`. On subsequent runs it loads the existing index instead of
rebuilding it.

## Tests

```
pytest              # 131 offline tests, ~7s warm, no network and no API quota
pytest -m live      # 5 end-to-end tests against the real app (needs GOOGLE_API_KEY)
pytest -m ""        # everything
```

The default run is offline by design (`addopts = -m "not live"` in `pytest.ini`), so it is
free, deterministic, and safe for CI. Anything that reaches the network or spends Gemini
free-tier quota must be marked `@pytest.mark.live`.

`tests/conftest.py` enforces that: a non-`live` test that opens an outbound socket fails with
the offending test and host named. This exists because a mock pointed at the wrong module
once let a test hit PubMed, Wikipedia and SerpAPI for real *while still passing* — a mock that
misses looks exactly like a mock that works, only slower. The guard is installed by
`pytest_runtest_setup`, not an autouse fixture, so that it covers module- and session-scoped
fixture setup too; `tests/test_network_guard.py` pins that. See "Continuous integration".

**First run needs the embedding model on disk.** Two test modules load the FAISS index through
`fastembed`, and the network guard will refuse to let them download it. If a run fails with a
socket-guard error naming `huggingface.co`, warm the cache once:

```
python -c "from langchain_community.embeddings import FastEmbedEmbeddings; from medbot.config import LOCAL_EMBEDDING_MODEL; FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL).embed_query('warm the cache')"
```

That is a ~130MB download, once. It lands in the system temp directory unless
`FASTEMBED_CACHE_PATH` points elsewhere (CI sets it so the model can be cached between runs).

Note `tests/test_eval_regression.py` reads the committed JSON artefacts in `medbot/eval/`
rather than calling the model. It catches someone committing worse numbers; it does not
detect live model drift. Regenerating those artefacts is a deliberate, quota-spending act —
see `medbot/eval/results_sprint4.md`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and every pull request: install, `ruff check .`,
then the offline pytest suite on Python 3.11 / `windows-latest`. It needs no secrets and spends
no Gemini quota — no `GOOGLE_API_KEY` is set, so `live` tests stay deselected. If a test ever
starts needing the network or a key, CI fails rather than quietly billing a quota it does not own.

Windows, because that is where this project is developed and run; the repo is public, so runner
minutes are free and there was no cost reason to prefer Linux. One gap worth naming:
`.devcontainer/` pins a Debian image, so the Codespaces path is not covered by CI.

The workflow warms the fastembed model cache in its own step, before pytest. That step is
load-bearing, not an optimisation: the network guard makes an in-test download a hard failure,
which is the whole point — the alternative is a "unit" test silently pulling 130MB.

Linting is `ruff check` only; `ruff format` is deliberately not used. Prompt templates in
`medbot/prompt.py` are pinned by content hash, and every recorded eval number is only
comparable while the rendered prompt is byte-identical, so a reformatter that re-wraps a string
literal can silently invalidate a measurement. `ruff.toml` documents the rule selection and the
two rules that were trialled and dropped.

Optional locally, same checks minus the test suite:

```
pre-commit install      # then hooks run on every commit
pre-commit run --all-files
```

## Evaluation

`medbot/eval/` holds the harness and every recorded result. Start with the sprint write-ups:

- `results_sprint4.md` — current state of the evidence, and what it does not support
- `results_sprint3.md` + `sprint3_audit.md` — the chain-of-thought A/B and its adversarial audit
- `results.md` — the original Sprint 2 baseline

Two tools worth knowing about, both free to run since they use only local retrieval:

- `python -m medbot.eval.verify_coverage --candidates` — checks whether the corpus actually
  covers a topic, by reading what retrieval returns. Use this before adding any question to
  the out-of-corpus suite; assuming coverage from a topic's name has produced two wrong
  questions so far. The full screened list, rejections included, lives in that file.
- `python -m medbot.eval.refusal_stats --prefix sprint4_` — question-level significance on the
  stored refusal trials. Questions missing trials in either arm are excluded and reported,
  never counted as zeros; also prints how much of the result rests on single trials.
- `python -m medbot.eval.calibration_score` — scores a filled-in `calibration_sheet.md`
  against the judge. Reports differential bias (cot − baseline), which is what would
  undermine the groundedness comparison; a bias equal in both arms cancels out of it.
- `python -m medbot.eval.relabel` — re-derives the stored `refusal` booleans from the stored
  answer text (dry run; `--write` to apply). Run it after any change to `is_refusal`: the
  booleans are derived data, and `test_refusal_labels_match_the_heuristic` fails on purpose
  until the recorded trials are re-scored with the instrument that now ships.

Note `medbot/data_processing.py` does **not** load the index on import — call
`create_vector_database()` explicitly. Importing it used to build the index when absent,
which made a bare `--help` or `pytest --collect-only` capable of starting a 1225-chunk embed.

Trial runs that spend quota take `--resume`, which keeps cells already recorded and measures
only the gaps — the free tier is 500 requests/day and it rolls over at midnight **US
Pacific**, not local midnight:

```
python -m medbot.eval.refusal_trials --trials 3 --suite overanswer --out-prefix sprint4_ --resume
```

## Notes

- `medbot/legacy/` holds an earlier, standalone retriever experiment that is not imported by `app.py` — kept for reference, not part of the live app.
- `DATA_DIR`, `PERSIST_DIR`, `GEMINI_CHAT_MODEL`, and `LOCAL_EMBEDDING_MODEL` can be overridden via environment variables — see `.env.example`.
- See `MIGRATION_STATUS.md` for the full history of the OpenAI → Gemini migration and current project status/roadmap.
