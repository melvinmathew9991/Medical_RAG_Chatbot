import re
import traceback

from langchain.chains import RetrievalQA

from medbot.external_search import search_pubmed, search_serpapi, search_wikipedia
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
            # Additive: the response gains a "source_documents" key and every
            # existing consumer reads only "result". Without this the chunks the
            # answer was built from are discarded inside the chain, so the app
            # cannot cite them and a wrong answer cannot be traced to what it
            # was given.
            return_source_documents=True,
        )
        return document_chain
    except Exception:
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

def format_sources(source_documents, max_sources=4):
    """
    Render retrieved chunks as a deduplicated source list, or None if empty.

    Says where the retrieved *context* came from, which is not the same claim as
    where the answer came from -- the model is handed four chunks and may lean on
    one. The app labels this "Retrieved from" for that reason. Overstating it as
    "Sources" would let a chunk that the answer contradicts read as a citation
    supporting it, which is the wrong direction to be wrong in for a medical tool.

    Deduplicates on (source, page) while keeping retrieval order, because
    `chunk_size=3000` with `chunk_overlap=300` routinely puts two neighbouring
    chunks on the same PDF page, and listing that page twice suggests two pieces
    of corroborating evidence where there is one.

    Returns None rather than an empty string when there is nothing citable,
    matching `format_external_results` so callers can use one truthiness check.
    """
    seen = []
    for doc in source_documents or []:
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source")
        if not source:
            # Pre-Sprint-6 indexes carry no metadata at all. Skipping keeps a
            # stale index citation-free rather than crashing the answer.
            continue
        key = (source, metadata.get("page"))
        if key not in seen:
            seen.append(key)

    if not seen:
        return None

    lines = []
    for source, page in seen[:max_sources]:
        # PyPDFLoader numbers pages from 0; readers and PDF viewers number from 1.
        label = f"{source}, p. {page + 1}" if isinstance(page, int) else source
        lines.append(f"- {label}")
    return "\n".join(lines)


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
    except Exception:
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
