import os
import traceback
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from config import PERSIST_DIR
import time
import concurrent.futures  # Import concurrent.futures for multi-threading

def create_vector_database():
    try:
        start_time = time.time()
        
        model_kwargs = {}
        embeddings = OpenAIEmbeddings(**model_kwargs)

        if not os.path.exists(PERSIST_DIR):
            pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
            text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)

            pdf_documents = pdf_loader.load()
            text_documents = text_loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)

            pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
            text_context = "\n\n".join(str(p.page_content) for p in text_documents)

            pdfs = splitter.split_text(pdf_context)
            texts = splitter.split_text(text_context)

            data = pdfs + texts

            print("Data Processing Complete")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                vectordb = executor.submit(
                    Chroma.from_texts, data, embeddings, persist_directory=PERSIST_DIR
                ).result()
                vectordb.persist()

            print("Vector DB Creating Complete\n")
        else:
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        print("Vector DB Loaded\n")
        
        end_time = time.time()
        print("Time taken for create_vector_database():", end_time - start_time, "seconds")
        
        return vectordb
    except Exception as e:
        print("Error occurred during vector database creation:")
        print(traceback.format_exc())
        return None

vectordb = create_vector_database()
