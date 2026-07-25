"""
Check whether the indexed corpus actually covers a topic, by reading what
retrieval returns for it.

    .venv-gemini/Scripts/python.exe -m medbot.eval.verify_coverage --candidates
    .venv-gemini/Scripts/python.exe -m medbot.eval.verify_coverage "What causes gout?"

Uses only the local FAISS index and the local fastembed model, so it costs no
API quota and can be re-run freely.

Why this exists: Sprint 3's out-of-corpus guard included "What causes a stroke?"
on the reasoning that an A-B encyclopedia cannot cover an S topic. That was
wrong -- stroke causation is discussed inside the A entries for embolism and
atherosclerosis, both variants answered it correctly from real context, and the
question had to be thrown out (audit F8). The lesson recorded there was "verify
coverage per question, don't infer it from the first letter". This is that
verification, made repeatable instead of manual.

The output is evidence for a human decision, not the decision itself. A topic
term appearing in a retrieved chunk can be a passing mention rather than real
coverage, so the retrieved text is printed for reading.
"""

import argparse
import re

from medbot.data_processing import create_vector_database

# Candidates for the out-of-corpus guard (audit F7 wants 8-10, up from 2).
# Each must be a topic the corpus does NOT cover, so that refusing is the
# CORRECT answer and a substantive reply means the model invented it.
#
# Chosen to avoid two specific contamination traps:
#   - "gout" and "rheumatoid arthritis" appear in the bursitis entry's keywords,
#   - "osteoporosis" and "anorexia" are chain-of-thought exemplars in prompt.py,
# so none of those are candidates however C-Z they look.
#
# This is the COMPLETE list that was screened, kept and rejected together, so
# `--candidates` reproduces the Sprint 4 selection rather than a sample of it.
# An earlier version held only the first 12 while the sprint had actually
# screened 25 -- which made the selection unauditable and put a wrong count
# ("14 of 26") in results_sprint4.md. Add rejected candidates here, do not
# delete them: the rejections are the evidence that the kept 10 were not
# cherry-picked.
#
# Verdicts as of the 1225-chunk index: 10 absent (kept), 15 contaminated
# (rejected). test_out_of_corpus_selection.py pins that, so a change to the
# index or the retriever that silently makes a shipped question answerable
# fails a test instead of quietly weakening the guard.
CANDIDATES = [
    # --- kept: retrieval shows no trace of the topic ---
    "How is psoriasis treated?",
    "What are the symptoms of schizophrenia?",
    "What are the symptoms of shingles?",
    "How is glaucoma treated?",
    "What are the symptoms of measles?",
    "How is scabies treated?",
    "How is ringworm treated?",
    "What are the symptoms of mumps?",
    "How is sciatica treated?",
    "What causes plantar fasciitis?",
    # --- rejected: the topic term appears in retrieved text ---
    # Diabetes is the important one, and the reason this tool exists: it was
    # HALF of the 2-question guard the Sprint 3 result leaned on. Its top-4
    # chunks include the *blood sugar tests* entry (a B entry) explaining
    # insulin and hyperglycemia, so a model answering it is reading the corpus,
    # not inventing -- audit F8's stroke mistake, repeated.
    "What are the symptoms of diabetes?",
    "What causes Parkinson's disease?",
    "How is tuberculosis treated?",
    "What causes kidney stones?",
    "How is malaria transmitted?",
    "What causes migraine headaches?",
    "What are the symptoms of lupus?",
    "How is epilepsy treated?",
    "What causes varicose veins?",
    "How is rabies treated?",
    "What causes hemorrhoids?",
    "How is tonsillitis treated?",
    "How are warts treated?",
    "What causes vertigo?",
    "What causes tinnitus?",
]

# The word whose presence in retrieved text would mean the corpus does discuss
# the topic. Derived from the question rather than hand-written, then overridden
# where the useful term is not the longest word.
TOPIC_OVERRIDES = {
    "What causes Parkinson's disease?": ["parkinson"],
    "What causes migraine headaches?": ["migraine"],
    "What causes kidney stones?": ["kidney stone", "renal calcul"],
    "How is malaria transmitted?": ["malaria"],
}

STOPWORDS = {
    "what", "how", "is", "are", "the", "of", "a", "an", "causes", "cause",
    "symptoms", "treated", "transmitted", "does", "do", "his", "her", "their",
}


def topic_terms(question):
    if question in TOPIC_OVERRIDES:
        return TOPIC_OVERRIDES[question]
    words = re.findall(r"[a-z']+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 3]


def check(vectordb, question, k=4, show_chars=220):
    docs = vectordb.as_retriever(search_kwargs={"k": k}).invoke(question)
    terms = topic_terms(question)
    hits = []
    for i, doc in enumerate(docs):
        text = doc.page_content
        found = [t for t in terms if t in text.lower()]
        if found:
            hits.append((i, found))
    return docs, terms, hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="Questions to check.")
    parser.add_argument("--candidates", action="store_true",
                        help="Check the built-in out-of-corpus candidate list.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--quiet", action="store_true",
                        help="Verdict lines only, no retrieved text.")
    parser.add_argument("--chars", type=int, default=220,
                        help="Characters of each retrieved chunk to print.")
    args = parser.parse_args()

    questions = list(args.questions)
    if args.candidates or not questions:
        questions += CANDIDATES

    vectordb = create_vector_database()
    if vectordb is None:
        raise SystemExit("No vector index; run medbot.data_processing first.")

    clean, contaminated = [], []
    for question in questions:
        docs, terms, hits = check(vectordb, question, k=args.k)
        verdict = "COVERED?" if hits else "absent"
        (contaminated if hits else clean).append(question)
        print(f"\n{'='*78}\n{verdict:>9}  {question}\n          terms={terms}")
        if hits:
            for i, found in hits:
                print(f"          -> chunk {i} contains {found}")
        if not args.quiet:
            for i, doc in enumerate(docs):
                snippet = " ".join(doc.page_content.split())[:args.chars]
                print(f"          [{i}] {snippet}")

    print(f"\n{'='*78}")
    print(f"absent (usable as out-of-corpus): {len(clean)}")
    for q in clean:
        print(f"  - {q}")
    print(f"\nterm appears in retrieved text (READ the chunks before using): {len(contaminated)}")
    for q in contaminated:
        print(f"  - {q}")


if __name__ == "__main__":
    main()
