import re
import traceback
from langchain.chains import RetrievalQA
from medbot.external_search import search_pubmed, search_wikipedia, search_serpapi
from medbot.prompt import build_context_prompt, emits_reasoning

# The chain-of-thought prompt makes the model emit "Reasoning: ... Answer: ...".
# Only the answer is for the user. Anchored to the start of a line because the
# reasoning text itself often contains the phrase "support an answer:" mid-
# sentence, which an unanchored match would split on.
# The trailing \** matters: the model often bolds the label as "**Answer:**", and
# without it the closing asterisks survive into the text shown to the user.
ANSWER_MARKER_RE = re.compile(
    r"^[ \t]*\**[ \t]*answer[ \t]*\**[ \t]*:[ \t]*\**",
    re.IGNORECASE | re.MULTILINE,
)


def strip_reasoning(text):
    """
    Return just the answer portion of a chain-of-thought response.

    Falls back to the full text when no "Answer:" marker is present. That is the
    right failure mode here: the baseline prompt produces no marker at all and
    must pass through untouched, and on a malformed CoT response showing
    something verbose beats showing the user nothing.
    """
    if not text:
        return text
    matches = list(ANSWER_MARKER_RE.finditer(text))
    if not matches:
        return text.strip()
    return text[matches[-1].end():].strip()


def create_query_chain(model, vectordb, question, variant=None):
    try:
        retriever = vectordb.as_retriever()
        prompt = build_context_prompt(question, variant=variant)
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


def run_query(chain, question, variant=None):
    """
    Invoke `chain` and return its response with the reasoning trace removed.

    Both the app and the eval harness go through here, so what gets measured is
    what the user actually sees.
    """
    response = chain.invoke({"query": question})
    if emits_reasoning(variant):
        response = dict(response)
        response["result"] = strip_reasoning(response.get("result", ""))
    return response

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

