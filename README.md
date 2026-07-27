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
├── .devcontainer/          # GitHub Codespaces / VS Code dev container
├── .env.example            # required environment variables
├── pytest.ini
└── requirements.txt
```

## Setup

Requires **Python 3.11** — the pinned LangChain 0.2.x stack needs `numpy<2`, which has no
prebuilt wheel for Python 3.13.

1. Create a Python 3.11 virtual environment (e.g. `.venv-gemini`) and activate it.
2. `pip install -r requirements.txt`
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
pytest              # 112 offline tests, ~6s, no network and no API quota
pytest -m live      # 4 end-to-end tests against the real app (needs GOOGLE_API_KEY)
pytest -m ""        # everything
```

The default run is offline by design (`addopts = -m "not live"` in `pytest.ini`), so it is
free, deterministic, and safe for CI. Anything that reaches the network or spends Gemini
free-tier quota must be marked `@pytest.mark.live`.

`tests/conftest.py` enforces that: a non-`live` test that opens an outbound socket fails with
the offending test and host named. This exists because a mock pointed at the wrong module
once let a test hit PubMed, Wikipedia and SerpAPI for real *while still passing* — a mock that
misses looks exactly like a mock that works, only slower.

Note `tests/test_eval_regression.py` reads the committed JSON artefacts in `medbot/eval/`
rather than calling the model. It catches someone committing worse numbers; it does not
detect live model drift. Regenerating those artefacts is a deliberate, quota-spending act —
see `medbot/eval/results_sprint4.md`.

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
