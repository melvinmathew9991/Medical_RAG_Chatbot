import os
import functools
import time

from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

# Define a cache decorator to cache responses
def cache(func):
    cached_results = {}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cached_results:
            cached_results[key] = func(*args, **kwargs)
        return cached_results[key]

    return wrapper

@cache
def initialize_model(model_type):
    try:
        start_time = time.time()  # Start timing model initialization

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
        
        end_time = time.time()  # End timing model initialization
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Time taken for {model_type} model initialization:", elapsed_time, "seconds")

        return model
    except ValueError as e:
        print("Error occurred during model initialization:", e)
        return None
