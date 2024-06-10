import os
import traceback
import time
import concurrent.futures
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config import PERSIST_DIR

def load_documents(loader, documents):
    """
    Load documents using the specified loader.
    """
    return loader.load(documents)

def process_documents(pdf_documents, text_documents):
    """
    Process documents by splitting them into chunks.
    """
    # Create text splitter with recommended chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    
    # Assuming pdf_documents and text_documents are lists of document contents
    pdf_context = "\n\n".join(str(p.page_content) for p in pdf_documents)
    text_context = "\n\n".join(str(p.page_content) for p in text_documents)

    # Split the documents into chunks
    pdf_chunks = splitter.split_text(pdf_context)
    text_chunks = splitter.split_text(text_context)

    # Combine pdfs and texts chunks
    data = pdf_chunks + text_chunks
    
    return data

def create_vector_database():
    """
    Create or load a vector database from the processed documents.
    """
    try:
        start_time = time.time()
        
        model_kwargs = {}
        embeddings = OpenAIEmbeddings(**model_kwargs)

        # Check if the persistence directory exists
        if not os.path.exists(PERSIST_DIR):
            # If it doesn't exist, load documents and process them
            pdf_loader = DirectoryLoader("./docs/", glob="./*.pdf", loader_cls=PyPDFLoader)
            text_loader = DirectoryLoader("./docs/", glob="./*.txt", loader_cls=TextLoader)

            # Load documents concurrently using threads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                pdf_documents_future = executor.submit(load_documents, pdf_loader, None)
                text_documents_future = executor.submit(load_documents, text_loader, None)

                pdf_documents = pdf_documents_future.result()
                text_documents = text_documents_future.result()

            # Process the documents into chunks
            data = process_documents(pdf_documents, text_documents)

            print("Data Processing Complete")

            # Create a vector database and persist it
            with concurrent.futures.ThreadPoolExecutor() as executor:
                vectordb = executor.submit(
                    Chroma.from_texts, data, embeddings, persist_directory=PERSIST_DIR
                ).result()
                vectordb.persist()

            print("Vector DB Creating Complete\n")
        else:
            # If the persistence directory exists, load the vector database
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

        print("Vector DB Loaded\n")
        
        end_time = time.time()
        print("Time taken for create_vector_database():", end_time - start_time, "seconds")
        
        return vectordb
    except Exception as e:
        # Handle any exceptions that occur during vector database creation
        print("Error occurred during vector database creation:")
        print(traceback.format_exc())
        return None

# Example usage
vectordb = create_vector_database()
