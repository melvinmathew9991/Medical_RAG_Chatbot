import os
import traceback
import time
import concurrent.futures
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from config import PERSIST_DIR

def load_documents(loader):
    """
    Load documents using the specified loader.
    """
    return loader.load()

def process_documents(pdf_documents, text_documents):
    """
    Process documents by splitting them into chunks.
    """
    # Create text splitter with recommended chunk size and overlap
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    
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
        index_path = os.path.join(PERSIST_DIR, "index.faiss")

        if not os.path.exists(index_path):
            pdf_loader = DirectoryLoader("./docs/", glob="*.pdf", loader_cls=PyPDFLoader)
            text_loader = DirectoryLoader("./docs/", glob="*.txt", loader_cls=TextLoader)

            # Load documents concurrently using threads
            with concurrent.futures.ThreadPoolExecutor() as executor:
                pdf_documents_future = executor.submit(load_documents, pdf_loader)
                text_documents_future = executor.submit(load_documents, text_loader)

                pdf_documents = pdf_documents_future.result()
                text_documents = text_documents_future.result()

            data = process_documents(pdf_documents, text_documents)

            print("Data Processing Complete")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                vectordb = executor.submit(
                    FAISS.from_texts, data, embeddings
                ).result()
                vectordb.save_local(PERSIST_DIR)

            print("Vector DB Creating Complete\n")
        else:
            vectordb = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

        print("Vector DB Loaded\n")
        
        end_time = time.time()
        print("Time taken for create_vector_database():", end_time - start_time, "seconds")
        
        return vectordb
    except Exception as e:
        print("Error occurred during vector database creation:")
        print(traceback.format_exc())
        return None


vectordb = create_vector_database()
