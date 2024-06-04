import os
from dotenv import load_dotenv 
load_dotenv()

# Set up environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Model type selection
MODEL_TYPE = os.environ.get("MODEL_TYPE", "ollama")

# OLLAMA model name
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "phi3")

# OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Persist directory path for the vector database
PERSIST_DIR = "E:/brototype/Langchain/Ollama/chatbot"

# OpenAI embedding engine name (if using OpenAI)
OPENAI_EMBEDDING = os.environ.get("OPENAI_EMBEDDING", "text-embedding-ada-002")

#SerAPI key
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")