import concurrent.futures
import os
import time
import traceback

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS

from medbot.config import DATA_DIR, LOCAL_EMBEDDING_MODEL, PERSIST_DIR


def load_documents(loader):
    """
    Load documents using the specified loader.
    """
    return loader.load()

def process_documents(pdf_documents, text_documents):
    """
    Process documents by splitting them into chunks.

    Splits each source document (e.g. each PDF page) independently rather than
    concatenating the whole corpus into one string first — otherwise chunk
    boundaries land at arbitrary character offsets that can span unrelated
    encyclopedia entries, which badly hurts retrieval relevance.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_documents(pdf_documents + text_documents)
    return [c.page_content for c in chunks]

def create_vector_database():
    """
    Create, resume, or load a vector database from the processed documents.

    Chunking is deterministic (same source files + splitter config), so on
    every run we recompute the full chunk list and compare it against how
    many vectors are already persisted (`vectordb.index.ntotal`) to figure
    out where to resume — the index is saved after every batch, so an
    interrupted run (e.g. process killed) doesn't throw away progress.
    Embeddings run locally via fastembed (CPU, no API/quota, no rate limits).
    """
    try:
        start_time = time.time()

        embeddings = FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
        index_path = os.path.join(PERSIST_DIR, "index.faiss")

        pdf_loader = DirectoryLoader(DATA_DIR, glob="*.pdf", loader_cls=PyPDFLoader)
        text_loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)

        # Load documents concurrently using threads
        with concurrent.futures.ThreadPoolExecutor() as executor:
            pdf_documents_future = executor.submit(load_documents, pdf_loader)
            text_documents_future = executor.submit(load_documents, text_loader)

            pdf_documents = pdf_documents_future.result()
            text_documents = text_documents_future.result()

        data = process_documents(pdf_documents, text_documents)
        print(f"Data Processing Complete: {len(data)} chunks")

        vectordb = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True) if os.path.exists(index_path) else None
        already_embedded = vectordb.index.ntotal if vectordb is not None else 0

        if already_embedded >= len(data):
            print(f"Vector DB already complete ({already_embedded}/{len(data)} chunks), loaded from disk\n")
        else:
            remaining = data[already_embedded:]
            print(f"Resuming: {already_embedded}/{len(data)} chunks already embedded, {len(remaining)} remaining")

            batch_size = 100
            total_batches = (len(remaining) + batch_size - 1) // batch_size
            for batch_num, i in enumerate(range(0, len(remaining), batch_size), start=1):
                batch = remaining[i:i + batch_size]
                try:
                    vectors = embeddings.embed_documents(batch)
                except Exception as e:
                    print(f"Stopping: batch {batch_num}/{total_batches} failed ({e}).")
                    print(f"Progress saved: {already_embedded + i}/{len(data)} chunks. Re-run later to resume.")
                    break

                # strict=True: if the embedder ever returns fewer vectors than
                # texts, the default zip truncates silently and the index is
                # written with chunks missing but no error -- the exact failure
                # shape that had this store sitting at 900/1225 while the docs
                # claimed it was complete. Better to raise here.
                if vectordb is None:
                    vectordb = FAISS.from_embeddings(list(zip(batch, vectors, strict=True)), embeddings)
                else:
                    vectordb.add_embeddings(list(zip(batch, vectors, strict=True)))
                vectordb.save_local(PERSIST_DIR)
                print(f"Embedded batch {batch_num}/{total_batches} ({vectordb.index.ntotal}/{len(data)} total)")

        print("Vector DB Loaded\n")

        end_time = time.time()
        print("Time taken for create_vector_database():", end_time - start_time, "seconds")

        return vectordb
    except Exception:
        print("Error occurred during vector database creation:")
        print(traceback.format_exc())
        return None


# NOTE: there was a module-level `vectordb = create_vector_database()` here.
# Nothing consumed it -- app.py and all five eval scripts import the *function*
# and call it themselves -- so every one of them paid the index load twice, and
# the whole test suite paid it just for importing a module.
#
# Worse than slow: the call builds the index when it is absent or incomplete, so
# importing this module on a machine without `vectorstore/` would start embedding
# 1225 chunks. `pytest --collect-only`, `--help`, or any unit test that touches
# something two imports away would trigger a multi-minute build as a side effect.
# Import should not do work. Call `create_vector_database()` explicitly.
