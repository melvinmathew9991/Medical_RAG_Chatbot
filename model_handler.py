from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_openai import OpenAI
import os
from config import MODEL_TYPE, OPENAI_API_KEY

def initialize_model(model_type):
    """Initializes the language model based on the chosen type."""
    if model_type == "ollama":
        model_name = os.environ.get("OLLAMA_MODEL_NAME", "phi3")
        model = Ollama(
            model=model_name,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
        )
    elif model_type == "openai":
        model = ChatOpenAI(
            temperature=0.1,
            convert_system_message_to_human=True
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

