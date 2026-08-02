import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `.env` declares these as blank lines for discoverability; python-dotenv sets
# blank vars to "", and os.getenv's default only applies when a var is fully
# absent — so `or` (not the getenv default arg) is required to fall back correctly.
DATA_DIR = os.getenv("DATA_DIR") or os.path.join(BASE_DIR, "data")
PERSIST_DIR = os.getenv("PERSIST_DIR") or os.path.join(BASE_DIR, "vectorstore")

# "-lite" gets a far more generous free-tier quota (500 RPD) than the full
# Flash model (20 RPD) as of this writing — see https://ai.google.dev/gemini-api/docs/models
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL") or "gemini-flash-lite-latest"
# Embeddings run locally via fastembed (CPU, ONNX, no API/quota) instead of the
# Gemini embedding API — see MIGRATION_STATUS.md for why (free-tier RPD quota
# made the Gemini embedding path unworkable for this corpus size).
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL") or "BAAI/bge-small-en-v1.5"
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
