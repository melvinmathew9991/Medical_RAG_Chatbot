import re
import traceback
from langchain.chains import RetrievalQA
from medbot.external_search import search_pubmed, search_wikipedia, search_serpapi
from medbot.prompt import build_context_prompt

def create_query_chain(model, vectordb, question):
    try:
        retriever = vectordb.as_retriever()
        prompt = build_context_prompt(question)
        document_chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            llm=model,
            chain_type_kwargs={"prompt": prompt},
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

def format_external_results(results, max_per_source=5):
    """Render pubmed/wikipedia/serpapi results as a markdown block, or None if empty."""
    sections = []

    pubmed = results.get("pubmed") or []
    if pubmed:
        lines = [
            f"- {doc.metadata.get('Title', 'Untitled')}"
            for doc in pubmed[:max_per_source]
        ]
        sections.append("**PubMed**\n" + "\n".join(lines))

    wikipedia = results.get("wikipedia") or []
    if wikipedia:
        lines = [
            f"- {re.sub('</?span[^>]*>', '', snippet)}"
            for snippet in wikipedia[:max_per_source]
        ]
        sections.append("**Wikipedia**\n" + "\n".join(lines))

    serpapi = results.get("serpapi") or []
    if serpapi:
        lines = [
            f"- [{r.get('title', 'Untitled')}]({r.get('link', '')})"
            for r in serpapi[:max_per_source]
        ]
        sections.append("**Google (via SerpAPI)**\n" + "\n".join(lines))

    if not sections:
        return None
    return "\n\n".join(sections)

