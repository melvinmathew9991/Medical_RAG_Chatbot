from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from config import PERSIST_DIR, OPENAI_EMBEDDING

def create_vector_database():
    """Creates the vector database if it doesn't exist, otherwise loads it."""
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

        vectordb = Chroma.from_texts(data, embeddings, persist_directory=PERSIST_DIR)
        vectordb.persist()

        print("Vector DB Creating Complete\n")
    else:
        vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    print("Vector DB Loaded\n")
    return vectordb

# Export vectordb for use in other modules
vectordb = create_vector_database()
