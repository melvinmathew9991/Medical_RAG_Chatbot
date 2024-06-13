from langchain_community.document_loaders import PubMedLoader
from serpapi import GoogleSearch
from requests.exceptions import RequestException
from config import SERPAPI_API_KEY
import time
import asyncio
import aiohttp  
from functools import lru_cache

# Implement a threaded version of PubMed search
def threaded_search_pubmed(query):
    try:
        start_time = time.time()
        loader = PubMedLoader(query=query)
        documents = loader.load()
        end_time = time.time()
        print("Time taken for PubMed search:", end_time - start_time, "seconds")
        return documents
    except Exception as e:
        print("Error occurred during PubMed search:", e)
        return []

# Utilize caching for PubMed search results using LRU cache
@lru_cache(maxsize=32)
def cached_search_pubmed(query):
    return threaded_search_pubmed(query)

# Update the function to use the cached version
def search_pubmed(query):
    return cached_search_pubmed(query)

# Use a decorator for caching
@lru_cache(maxsize=32)
def fetch_wikipedia_data(query):
    # Define asynchronous function to fetch Wikipedia data
    async def fetch(session, url):
        async with session.get(url) as response:
            return await response.json()

    async def fetch_all_data():
        # Construct Wikipedia API URL with query
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={query}&format=json"
        # Initialize aiohttp session
        async with aiohttp.ClientSession() as session:
            # Fetch data
            return await fetch(session, url)

    # Ensure the event loop runs in the correct context
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    data = loop.run_until_complete(fetch_all_data())
    loop.close()
    return data

def search_wikipedia(query):
    try:
        start_time = time.time()
        # Fetch Wikipedia data
        data = fetch_wikipedia_data(query)
        # Extract snippets from data
        documents = [item['snippet'] for item in data['query']['search']]
        end_time = time.time()
        print("Time taken for Wikipedia search:", end_time - start_time, "seconds")
        return documents
    except Exception as e: 
        print("Error occurred during Wikipedia search:", e)
        return []

def search_serpapi(query):
    try:
        start_time = time.time()
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "engine": "google"
        }
        # Perform Google search using SERPAPI
        search = GoogleSearch(params)
        results = search.get_dict()
        end_time = time.time()
        print("Time taken for SERPAPI request:", end_time - start_time, "seconds")
        
        # Extract organic results if available
        if 'organic_results' in results:
            return results['organic_results']
        else:
            return []  
    except RequestException as e:
        print("Error occurred during SERPAPI request:", e)
        return []
