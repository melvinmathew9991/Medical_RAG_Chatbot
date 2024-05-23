from langchain.chains import RetrievalQA
from data_processing import vectordb, create_vector_database 

def create_query_chain(model, vectordb):
    """Creates a RAG (Retriever-Augmented Generator) pipeline for the chatbot."""
    retriever = vectordb.as_retriever()  # Use the as_retriever method of Chroma
    document_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        llm=model
    )
    return document_chain




