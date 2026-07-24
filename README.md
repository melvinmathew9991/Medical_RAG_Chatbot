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
│   └── legacy/             # experimental code not wired into app.py
│       ├── retriever.py
│       └── speech.txt
├── data/                   # source documents ingested into the vector index
├── vectorstore/            # committed FAISS index (index.faiss, index.pkl)
├── .devcontainer/          # GitHub Codespaces / VS Code dev container
├── .env.example            # required environment variables
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

## Notes

- `medbot/legacy/` holds an earlier, standalone retriever experiment that is not imported by `app.py` — kept for reference, not part of the live app.
- `DATA_DIR`, `PERSIST_DIR`, `GEMINI_CHAT_MODEL`, and `LOCAL_EMBEDDING_MODEL` can be overridden via environment variables — see `.env.example`.
- See `MIGRATION_STATUS.md` for the full history of the OpenAI → Gemini migration and current project status/roadmap.
