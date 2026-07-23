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
│   ├── model_handler.py    # ChatOpenAI initialization + caching
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

1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` (required) and `SERPAPI_API_KEY` (optional, needed for Google search).
3. `streamlit run app.py`

On first run, `medbot/data_processing.py` builds a FAISS index from every PDF/text file in `data/` and saves it to `vectorstore/`. On subsequent runs it loads the existing index instead of rebuilding it.

## Notes

- `medbot/legacy/` holds an earlier, standalone retriever experiment that is not imported by `app.py` — kept for reference, not part of the live app.
- `DATA_DIR` and `PERSIST_DIR` can be overridden via environment variables if you want the corpus or index stored outside the repo.
