"""
Screen candidate EVAL questions: does the corpus hold a dedicated entry for this
topic, and does retrieval actually return it?

    .venv-gemini/Scripts/python.exe -m medbot.eval.verify_entry            # the candidate list
    .venv-gemini/Scripts/python.exe -m medbot.eval.verify_entry "What is astigmatism?"

Local retrieval only — no API key, no quota, re-runnable freely.

This is `verify_coverage.py`'s mirror image and it needs a stricter rule, which is
the whole reason it exists as a separate module. `verify_coverage` asks "is this
topic ABSENT?" for the out-of-corpus guard, so a single loose hit is a useful
rejection: if the term appears anywhere, the question is not safely out-of-corpus.
Reusing that same loose rule to *accept* eval questions is invalid, and measurably
so — screening 36 candidates with it accepted 35, including:

  - "What causes back pain?"  -> top chunks were the BURSITIS and arthritis
    entries. Terms were ['back', 'pain'], and "pain" appears in most of a medical
    encyclopedia.
  - "What causes bad breath?" -> matched on "breath" via the anoxia/hypoxia entry.
  - "What is a biopsy?"       -> matched breast biopsy and bone biopsy, two
    different entries, neither a general "biopsy" entry.
  - "What is Barrett's esophagus?" -> top chunk was ACETAMINOPHEN.

An eval question built on any of those would score Precision@4 against chunks that
never discuss the topic, and every refusal on it would be *correct* behaviour
recorded as a bug — the same class of error as audit F8, pointing the other way.

The rule here, therefore:

  1. the DISTINCTIVE term (the longest content word, or an override) must appear
     in at least MIN_CHUNKS_WITH_TERM of the top k chunks, not merely one; and
  2. at least one of those chunks must look like the entry itself rather than a
     passing mention — an encyclopedia "Definition" heading, or a copular
     sentence about the term ("Byssinosis is a chronic...").

Both conditions come from the observed failures above: generic single-word matches
fail (1), and cross-reference mentions in a neighbouring entry fail (2).
"""

import argparse
import os
import re

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

MIN_CHUNKS_WITH_TERM = 2

# Candidates screened for the Sprint 4 -> Sprint 5 question-set expansion. The
# rejected ones stay in the list on purpose: they are the evidence that the screen
# is a screen and not a search, which is the F7/F8 lesson recorded in
# verify_coverage.py, and `test_expansion_selection.py` re-checks both directions.
CANDIDATES = [
    # --- A entries
    "What are the symptoms of Alzheimer's disease?",
    "What causes anemia?",
    "What is angioplasty?",
    "What are the symptoms of ankylosing spondylitis?",
    "What is an arrhythmia?",
    "What causes asbestosis?",
    "What is astigmatism?",
    "What are the symptoms of atrial fibrillation?",
    "What is an audiometry test?",
    "What causes anosmia?",
    "What is aphasia?",
    "What causes atopic dermatitis?",
    "What is arteriography?",
    "What causes alopecia?",
    "What is amblyopia?",
    "What are the symptoms of anthrax?",
    "What is an antibody test?",
    # --- B entries
    "What causes back pain?",
    "What are the symptoms of Bell's palsy?",
    "What is a barium enema?",
    "What causes bronchiectasis?",
    "What is a bone marrow transplant?",
    "What causes byssinosis?",
    "What are the symptoms of a bladder infection?",
    "What is blepharitis?",
    "What causes blood clots?",
    "What is a biopsy?",
    "What is bradycardia?",
    "What causes bad breath?",
    "What is a blood sugar test?",
    "What are the symptoms of berylliosis?",
    "What is balloon valvuloplasty?",
    "What causes bruises?",
    "What is Barrett's esophagus?",
]

# Where the longest word is not the distinctive one, or where the entry's own
# spelling differs from the question's.
TERM_OVERRIDES = {
    "What are the symptoms of Alzheimer's disease?": "alzheimer",
    "What are the symptoms of ankylosing spondylitis?": "ankylosing spondylitis",
    "What are the symptoms of atrial fibrillation?": "atrial fibrillation",
    "What is an audiometry test?": "audiometry",
    "What is an antibody test?": "antibody test",
    "What are the symptoms of Bell's palsy?": "bell",
    "What is a bone marrow transplant?": "bone marrow transplant",
    "What are the symptoms of a bladder infection?": "bladder infection",
    "What is a barium enema?": "barium enema",
    "What is a blood sugar test?": "blood sugar test",
    "What is balloon valvuloplasty?": "valvuloplasty",
    "What is Barrett's esophagus?": "barrett",
    "What causes back pain?": "back pain",
    "What causes bad breath?": "bad breath",
    "What causes blood clots?": "blood clot",
}

STOPWORDS = {
    "what", "how", "why", "is", "are", "the", "of", "a", "an", "and", "in",
    "causes", "cause", "symptoms", "treated", "performed", "used", "does", "do",
    "test", "children", "infants",
}


def distinctive_term(question):
    """The one term whose presence means this entry, not a neighbouring one."""
    if question in TERM_OVERRIDES:
        return TERM_OVERRIDES[question]
    words = re.findall(r"[a-z']+", question.lower())
    content = [w for w in words if w not in STOPWORDS and len(w) > 3]
    return max(content, key=len) if content else ""


def looks_like_the_entry(text, term):
    """
    Whether `text` reads like the encyclopedia entry for `term` rather than a
    passing mention of it.

    Two accepted shapes, both observed in this corpus: the "Definition" heading
    that follows an entry title, and a copular sentence introducing the term.
    """
    low = " ".join(text.split()).lower()
    head = term.split()[-1]
    # `.{0,60}` and not `\W{0,60}`: entry titles carry words between the head term
    # and the heading -- "Atrial fibrillation AND FLUTTER Definition" -- and a
    # non-word-character gap cannot cross them. That bug alone rejected the atrial
    # fibrillation entry while its chunk began with the entry title verbatim.
    if re.search(rf"{re.escape(head)}.{{0,60}}definition", low):
        return True
    # Any copular sentence about the term. An earlier version enumerated the
    # permitted continuations ("is a", "is an", "is the result"...) and so rejected
    # "the major symptom of Bell's palsy IS ONE side of the face...". Generic terms
    # are already excluded by MIN_CHUNKS_WITH_TERM plus hand review of the accepted
    # list, so the narrow list bought nothing and cost real entries.
    return bool(re.search(rf"{re.escape(term)}\W{{0,20}}\s(?:is|are)\s", low))


def check_entry(vectordb, question, k=4):
    """-> (docs, term, chunks_with_term, entry_chunks, accepted)"""
    docs = vectordb.as_retriever(search_kwargs={"k": k}).invoke(question)
    term = distinctive_term(question)
    with_term, entry_chunks = [], []
    for i, doc in enumerate(docs):
        low = " ".join(doc.page_content.split()).lower()
        if term in low:
            with_term.append(i)
            if looks_like_the_entry(doc.page_content, term):
                entry_chunks.append(i)
    accepted = len(with_term) >= MIN_CHUNKS_WITH_TERM and bool(entry_chunks)
    return docs, term, with_term, entry_chunks, accepted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="*", help="Questions to screen.")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--window", type=int, default=0,
                        help="Print this many chars either side of the term, for "
                             "grounding expected_keywords in real retrieved text.")
    args = parser.parse_args()

    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.vectorstores import FAISS

    from medbot.config import LOCAL_EMBEDDING_MODEL, PERSIST_DIR

    embeddings = FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL)
    db = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)

    questions = args.questions or CANDIDATES
    accepted, rejected = [], []

    for question in questions:
        docs, term, with_term, entry_chunks, ok = check_entry(db, question, k=args.k)
        (accepted if ok else rejected).append(question)
        print(f"\n{'ACCEPT' if ok else 'reject'}  {question}")
        print(f"   term {term!r}: in chunks {with_term}, entry-shaped {entry_chunks}")
        if args.window:
            for i in with_term:
                text = " ".join(docs[i].page_content.split())
                at = text.lower().find(term)
                lo = max(0, at - args.window)
                print(f"   [{i}] …{text[lo:at + len(term) + args.window]}…")

    print(f"\n{'=' * 70}")
    print(f"screened {len(questions)}, accepted {len(accepted)}, rejected {len(rejected)}")
    for question in rejected:
        print(f"  reject: {question}")


if __name__ == "__main__":
    main()
