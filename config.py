import os
from dotenv import load_dotenv 
load_dotenv()

# Model type selection
MODEL_TYPE = os.environ.get("MODEL_TYPE", "ollama")


OLLAMA_MODEL_NAME = "phi3"
#OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "phi3")


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Persist directory path for the vector database
PERSIST_DIR = "E:/brototype/Langchain/Ollama/chatbot"

# OpenAI embedding engine name (if using OpenAI)
OPENAI_EMBEDDING = os.environ.get("OPENAI_EMBEDDING", "text-embedding-ada-002")

#SerAPI key
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")