import traceback  # Importing traceback for exception handling
from langchain.chains import RetrievalQA  
from external_search import search_pubmed, search_wikipedia, search_serpapi
def create_query_chain(model, vectordb):
    """
    Create a query chain for retrieving documents based on the given model and vector database.

    Args:
        model: The language model used for query processing.
        vectordb: The vector database used for document retrieval.

    Returns:
        langchain.chains.RetrievalQA: The query chain for document retrieval, or None if creation fails.
    """
    try:
        retriever = vectordb.as_retriever()  # Convert the vector database to a retriever
        document_chain = RetrievalQA.from_chain_type(  # Create a RetrievalQA instance
            retriever=retriever,
            llm=model
        )
        return document_chain  # Return the created query chain
    except Exception as e:
        # Handle any exceptions that occur during query chain creation
        print("Error occurred during query chain creation:")  
        print(traceback.format_exc())  # Print the traceback for debugging
        return None  # Return None if creation fails

def search_external_sources(query):
    """
    Search external sources (PubMed, Wikipedia, SERPAPI) for the given query.

    Args:
        query (str): The query string to search external sources.

    Returns:
        dict: A dictionary containing search results from PubMed, Wikipedia, and SERPAPI.
    """
    try:
        # Perform searches on external sources
        pubmed_results = search_pubmed(query)
        wikipedia_results = search_wikipedia(query)
        serpapi_results = search_serpapi(query)
        
        # Aggregate and return the search results
        results = {
            "pubmed": pubmed_results,
            "wikipedia": wikipedia_results,
            "serpapi": serpapi_results
        }
        return results  
    except Exception as e:
        # Handle any exceptions that occur during external source search
        print("Error occurred during external source search:")  
        print(traceback.format_exc())  # Print the traceback for debugging
        return {}  
