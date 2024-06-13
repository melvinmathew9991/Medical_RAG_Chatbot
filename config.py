import os
from dotenv import load_dotenv 

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "E:/brototype/Langchain/Ollama/test_chatbot"
OPENAI_EMBEDDING = os.getenv("OPENAI_EMBEDDING", "text-embedding-ada-002")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
