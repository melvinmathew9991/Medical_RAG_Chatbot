# MEDBOT Evaluation Results - `baseline` prompt

Ran 24 questions through the live RAG pipeline (retrieval top-4, Gemini `gemini-flash-lite-latest` generation, `baseline` prompt variant).

- **Mean Precision@4**: 0.83
- **Mean groundedness (binary LLM-judge, 0-1)**: 0.92

> **Read the groundedness figure with care.** This judge returned only 0.0 or 1.0 on every answer it graded, so it behaves as pass/fail and cannot see partial degradation. For discriminating scores use the claim-level judge (`medbot.eval.rejudge` -> `results_<variant>_claims.json`), and see `results_sprint3.md` for the interpretation.

**Methodology and limitations:** Precision@K uses keyword-containment against manually verified expected phrases (see `dataset.py`), not embedding similarity or human judgment — stricter than a human grader on paraphrased-but-relevant chunks. Groundedness is scored by the same Gemini model that generated the answer (disclosed self-grading bias) rather than an independent judge model. Both are intentionally scoped v1 approaches.

## Weak spots

- Questions with Precision@4 < 0.5: What are the symptoms of alcoholism?, What are the side effects of acetaminophen?
- Questions with groundedness < 0.5: What are the symptoms of breast cancer?, What is bursitis?

## Per-question results

| Question | Precision@K | Groundedness | Rationale |
|---|---|---|---|
| What causes acne? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the provided text about the causes and mechanisms of acne. |
| What are the symptoms of alcoholism? | 0.25 | 1.00 | Every symptom, withdrawal effect, physical sign, and related condition listed in the generated answer is directly supported by the provided text. |
| What causes allergic rhinitis? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by information found in the provided context about allergic rhinitis, its causes, and its mechanisms. |
| What are the symptoms of anaphylaxis? | 1.00 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned in the provided context under the symptoms sections. |
| What are the symptoms of appendicitis? | 1.00 | 1.00 | Every single symptom listed in the generated answer is directly mentioned and fully supported by the provided text. |
| What triggers an asthma attack? | 1.00 | 1.00 | Every single trigger listed in the generated answer is explicitly mentioned and supported by the provided context. |
| What is atherosclerosis? | 1.00 | 1.00 | The generated answer accurately defines atherosclerosis using details directly sourced from the provided text, while appropriately noting that the unrelated question about osteoporosis cannot be answered from the text. |
| What are the symptoms of autism? | 0.75 | 1.00 | Every symptom and detail included in the generated answer is directly mentioned and fully supported by the provided text. |
| How is bed-wetting treated in children? | 0.75 | 1.00 | Every treatment method, description, and specific detail listed in the generated answer is directly and accurately supported by the provided context. |
| What causes bedsores? | 0.75 | 1.00 | Every claim in the generated answer is directly and accurately supported by the provided context regarding the causes of bedsores. |
| What are the symptoms of bipolar disorder? | 0.75 | 1.00 | Every symptom and detail included in the generated answer is directly taken from and supported by the provided text about bipolar disorder. |
| What causes bladder cancer? | 1.00 | 1.00 | The generated answer correctly states that the exact cause of bladder cancer is not known, which is explicitly supported by the text ("Although the exact cause of bladder cancer is not known"). |
| What are the symptoms of botulism? | 1.00 | 1.00 | Every detail in the generated answer regarding the symptoms of food-borne, infant, and wound botulism is directly supported by the provided text. |
| What are the symptoms of a brain tumor? | 0.75 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned in the provided text as a potential sign of a brain tumor. |
| What are the symptoms of breast cancer? | 1.00 | 0.00 | The generated answer incorrectly claims the context does not contain the answer, whereas the context explicitly lists multiple symptoms of breast cancer. |
| What are the symptoms of bronchitis? | 1.00 | 1.00 | Every symptom listed in the generated answer for both acute and chronic bronchitis is directly supported by the provided text. |
| What is bulimia nervosa? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by information found in the provided context about bulimia nervosa. |
| What causes a bunion? | 0.75 | 1.00 | Every cause and contributing factor listed in the generated answer is directly supported by the text under the "Causes and symptoms" and "Definition" sections for bunions. |
| How should burns be treated? | 0.75 | 1.00 | Every detail in the generated answer regarding the treatment of thermal, chemical, and electrical burns is directly extracted and supported by the provided text. |
| What is bursitis? | 0.75 | 0.00 | The generated answer incorrectly claims the information is missing, whereas the context provides a detailed explanation of bursitis (causes, symptoms, diagnosis, and treatment). |
| What is an abscess? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the provided text describing the definition, types, composition, and locations of abscesses. |
| What are the side effects of acetaminophen? | 0.25 | 1.00 | Every claim in the generated answer regarding the side effects and overdose effects of acetaminophen is directly and accurately supported by the provided context. |
| What causes amnesia? | 1.00 | 1.00 | Every cause of amnesia listed in the generated answer is explicitly mentioned in the provided text. |
| What are the symptoms of angina? | 1.00 | 1.00 | Every symptom listed in the generated answer is directly found and supported by the provided context. |
