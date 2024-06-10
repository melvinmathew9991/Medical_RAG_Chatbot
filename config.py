import os
from dotenv import load_dotenv 

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key from the environment variable
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set Langchain tracing to version 2
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Set the Langchain API key from the environment variable
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Get the model type from the environment variable or default to "ollama"
MODEL_TYPE = os.environ.get("MODEL_TYPE", "ollama")

# Get the OLLAMA model name from the environment variable or default to "phi3"
OLLAMA_MODEL_NAME = os.environ.get("OLLAMA_MODEL_NAME", "phi3")

# Get the OpenAI API key from the environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Set the directory path for persisting the vector database
PERSIST_DIR = "E:/brototype/Langchain/Ollama/chatbot"

# Get the OpenAI embedding engine name from the environment variable or default to "text-embedding-ada-002"
OPENAI_EMBEDDING = os.environ.get("OPENAI_EMBEDDING", "text-embedding-ada-002")

# Get the SERPAPI API key from the environment variable
SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")
