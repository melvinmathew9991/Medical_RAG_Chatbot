from langchain_community.document_loaders import PubMedLoader
from langchain_community.document_loaders import WikipediaLoader
from serpapi import GoogleSearch
from config import SERPAPI_API_KEY

def search_pubmed(query):
    loader = PubMedLoader(query = query)
    documents = loader.load()
    return documents

def search_wikipedia(query):
    loader = WikipediaLoader(query = query)
    documents = loader.load()
    return documents

def search_serpapi(query):
    params = {
        "q" : query,
        "api_key" : SERPAPI_API_KEY,
        "engine" : "google"
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()
    
    # Check if 'organic_results' key exists in the results
    if 'organic_results' in results:
        return results['organic_results']
    else:
        return []  # Return an empty list if 'organic_results' is not found


