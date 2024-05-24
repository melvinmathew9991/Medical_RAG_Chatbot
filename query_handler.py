from langchain.chains import RetrievalQA
from data_processing import vectordb, create_vector_database 
from external_search import search_pubmed, search_wikipedia, search_serpapi

def create_query_chain(model, vectordb):
    """Creates a RAG (Retriever-Augmented Generator) pipeline for the chatbot."""
    retriever = vectordb.as_retriever()  # Use the as_retriever method of Chroma
    document_chain = RetrievalQA.from_chain_type(
        retriever=retriever,
        llm=model
    )
    return document_chain

def search_external_sources(query):
    pubmed_results = search_pubmed(query)
    #arxiv_results = search_arxiv(query)
    wikipedia_results = search_wikipedia(query)
    serpapi_results = search_serpapi(query)
    
    results = {
        "pubmed": pubmed_results,
        "wikipedia": wikipedia_results,
        "serpapi": serpapi_results
    }
    
    return results





