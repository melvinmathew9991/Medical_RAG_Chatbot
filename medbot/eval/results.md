# MEDBOT Sprint 2 Evaluation Results

Ran 24 questions through the live RAG pipeline (retrieval top-4, Gemini `gemini-flash-lite-latest` generation).

- **Mean Precision@4**: 0.83
- **Mean groundedness (LLM-judge, 0-1)**: 0.83

**Methodology and limitations:** Precision@K uses keyword-containment against manually verified expected phrases (see `dataset.py`), not embedding similarity or human judgment — stricter than a human grader on paraphrased-but-relevant chunks. Groundedness is scored by the same Gemini model that generated the answer (disclosed self-grading bias) rather than an independent judge model. Both are intentionally scoped v1 approaches.

## Weak spots

- Questions with Precision@4 < 0.5: What are the symptoms of alcoholism?, What are the side effects of acetaminophen?
- Questions with groundedness < 0.5: What causes bedsores?, What are the symptoms of breast cancer?, What is bursitis?, What is an abscess?

**Finding on the 4 zero-groundedness cases, and follow-up audit (2026-07-25):** all four failed
the same way — the model answered "I don't know the answer based on the provided context" even
though the retrieved context did contain the answer (verified by hand for "What is an abscess?").
Follow-up audit, run after the initial harness pass:

1. **Reproduced live in the real app, not just the harness.** Running the actual `app.py` via
   `streamlit.testing.v1.AppTest` and asking "What is an abscess?" through the real UI produced
   the same false refusal. Confirms this is a production behavior, not an eval-script artifact.
   Rest of the app worked correctly in this same run: no startup exceptions, PubMed/Wikipedia/
   SerpAPI all returned results, and the Wikipedia fallback in the "Related external sources"
   section usefully surfaced a correct abscess definition even while the main RAG answer refused
   — an accidental but real resilience property of the current UX.
2. **Temperature=0 helps, but doesn't fix it.** Ran each of the 4 questions 3x at the app's
   default `temperature=0.1` and 3x at `temperature=0`, same formatted prompt each time (24 calls
   total). Refusal rate dropped from 8/12 (67%) at 0.1 to 5/12 (42%) at 0. But "bedsores" went
   from refusing in the original run to 0/6 refusals here, and **"bursitis" refused in all 6
   trials regardless of temperature** — i.e. temperature is not the only variable, and at least
   one case is refusing for structural/prompt reasons that temperature alone won't touch. Raw
   trial data: `medbot/eval/temperature_experiment.json`.
3. **Conclusion:** dropping the default `temperature` to `0` (`medbot/model_handler.py`) is a
   cheap, low-risk partial mitigation worth making regardless, but Sprint 3's chain-of-thought
   rewrite remains necessary — it targets the actual mechanism (the model pattern-matching on
   the "say you don't know" disclaimer instead of checking whether the context supports an
   answer), not just the sampling noise on top of it.

**Precision audit on the 2 low-Precision@4 cases:** read the actual retrieved chunks by hand.
- *"What are the side effects of acetaminophen?"* (0.25) — genuine retrieval miss, not a
  keyword-set artifact: chunk 0 is correctly about acetaminophen, but chunks 1-3 are side-effect
  lists for completely different drugs (an ACE inhibitor, several beta blockers, an antihelminthic).
  The retriever appears to be matching on the surface structure of "drug + side effects" bullet
  lists rather than the specific drug named in the question — a real embedding-quality limitation
  worth remembering for Sprint 6 (retrieval quality improvements).
- *"What are the symptoms of alcoholism?"* (0.25) — mixed cause: chunk 2 (bibliography/references)
  is a genuine irrelevant miss, chunk 0 (treatment protocol) is topically adjacent but not
  symptoms, and chunk 3 (alcoholic myopathy symptoms) is arguably relevant to a human grader but
  didn't match this dataset's chosen `expected_keywords` — a real instance of the disclosed
  "stricter than a human grader" limitation, not a pure retrieval failure. A human-graded
  Precision@4 here would likely be closer to 0.5, not 0.25.

## Per-question results

| Question | Precision@K | Groundedness | Rationale |
|---|---|---|---|
| What causes acne? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the provided text regarding the causes and mechanisms of acne. |
| What are the symptoms of alcoholism? | 0.25 | 1.00 | Every symptom, withdrawal effect, and physical sign listed in the generated answer is directly supported by the provided context. |
| What causes allergic rhinitis? | 1.00 | 1.00 | Every detail in the generated answer regarding the definition, types of triggers, and underlying immune mechanisms of allergic rhinitis is directly and accurately supported by the provided text. |
| What are the symptoms of anaphylaxis? | 1.00 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned in the provided context under the symptoms and description sections for anaphylaxis. |
| What are the symptoms of appendicitis? | 1.00 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned and directly supported by the provided text. |
| What triggers an asthma attack? | 1.00 | 1.00 | Every trigger listed in the generated answer is explicitly mentioned in the provided text as a cause or trigger for asthma attacks. |
| What is atherosclerosis? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the definitions and descriptions provided in the context. |
| What are the symptoms of autism? | 0.75 | 1.00 | Every symptom listed in the generated answer is directly extracted and supported by the provided text about autism. |
| How is bed-wetting treated in children? | 0.75 | 1.00 | Every treatment method, technique, and specific detail listed in the generated answer is directly and accurately supported by the provided text. |
| What causes bedsores? | 0.75 | 0.00 | The generated answer falsely claims ignorance, whereas the context explicitly explains that bedsores are caused by constant pressure pinching blood vessels, friction, moisture, and various risk factors. |
| What are the symptoms of bipolar disorder? | 0.75 | 1.00 | Every symptom and characteristic mentioned in the generated answer is explicitly and accurately derived from the provided context about bipolar disorder. |
| What causes bladder cancer? | 1.00 | 1.00 | The generated answer is fully supported by the text, which explicitly states that the exact cause of bladder cancer is not known and that smoking is the greatest risk factor. |
| What are the symptoms of botulism? | 1.00 | 1.00 | Every detail provided in the generated answer regarding the symptoms of botulism across its various types and progression is directly supported by the text. |
| What are the symptoms of a brain tumor? | 0.75 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned in the provided text as a potential sign of a brain tumor. |
| What are the symptoms of breast cancer? | 1.00 | 0.00 | The generated answer incorrectly claims the context does not contain the answer, whereas the context explicitly lists multiple symptoms of breast cancer (such as breast lumps, skin dimpling, nipple retraction, and discharge). |
| What are the symptoms of bronchitis? | 1.00 | 1.00 | Every symptom listed in the generated answer for both acute and chronic bronchitis is directly and accurately supported by the provided text. |
| What is bulimia nervosa? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by information found in the provided context about bulimia nervosa. |
| What causes a bunion? | 0.75 | 1.00 | Every claim in the generated answer regarding the causes of bunions is directly supported by the provided text. |
| How should burns be treated? | 0.75 | 1.00 | Every detail in the generated answer regarding the treatment of thermal, chemical, electrical, and alternative burn remedies is directly supported by the provided text. |
| What is bursitis? | 0.75 | 0.00 | The generated answer incorrectly claims that the information is not in the context, whereas the context actually discusses bursitis in detail (causes, symptoms, diagnosis, treatment). |
| What is an abscess? | 1.00 | 0.00 | The generated answer incorrectly states that the text does not contain the definition of an abscess, whereas the provided text explicitly describes abscesses and their characteristics throughout. |
| What are the side effects of acetaminophen? | 0.25 | 1.00 | Every listed side effect of acetaminophen in the generated answer is explicitly and accurately mentioned in the provided context. |
| What causes amnesia? | 1.00 | 1.00 | Every specific root cause of amnesia listed in the generated answer is explicitly mentioned and supported by the provided context. |
| What are the symptoms of angina? | 1.00 | 1.00 | All the symptoms listed in the generated answer are explicitly mentioned in the provided context regarding angina. |
