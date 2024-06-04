import traceback
from langchain.chains import RetrievalQA 
from external_search import search_pubmed, search_wikipedia, search_serpapi

def create_query_chain(model, vectordb):
    try:
        retriever = vectordb.as_retriever()  
        document_chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            llm=model
        )
        return document_chain
    except Exception as e:
        print("Error occurred during query chain creation:")
        print(traceback.format_exc())
        return None

def search_external_sources(query):
    try:
        pubmed_results = search_pubmed(query)
        wikipedia_results = search_wikipedia(query)
        serpapi_results = search_serpapi(query)
        
        results = {
            "pubmed": pubmed_results,
            "wikipedia": wikipedia_results,
            "serpapi": serpapi_results
        }
        
        return results
    except Exception as e:
        print("Error occurred during external source search:")
        print(traceback.format_exc())
        return {}
