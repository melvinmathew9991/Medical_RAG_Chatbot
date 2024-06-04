from langchain_community.document_loaders import PubMedLoader
from langchain_community.document_loaders import WikipediaLoader
from serpapi import GoogleSearch
from requests.exceptions import RequestException
from config import SERPAPI_API_KEY

def search_pubmed(query):
    try:
        loader = PubMedLoader(query=query)
        documents = loader.load()
        return documents
    except Exception as e:
        print("Error occurred during PubMed search:", e)
        return []

def search_wikipedia(query):
    try:
        loader = WikipediaLoader(query=query)
        documents = loader.load()
        return documents
    except Exception as e:
        print("Error occurred during Wikipedia search:", e)
        return []

def search_serpapi(query):
    try:
        params = {
            "q" : query,
            "api_key" : SERPAPI_API_KEY,
            "engine" : "google"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if 'organic_results' in results:
            return results['organic_results']
        else:
            return []  
    except RequestException as e:
        print("Error occurred during SERPAPI request:", e)
        return []
