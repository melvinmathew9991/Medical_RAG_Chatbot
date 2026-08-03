# MEDBOT Evaluation Results - `no-examples` prompt

Ran 24 questions through the live RAG pipeline (retrieval top-4, Gemini `gemini-flash-lite-latest` generation, `no-examples` prompt variant).

- **Mean Precision@4**: 0.83
- **Mean groundedness (binary LLM-judge, 0-1)**: 1.00

> **Read the groundedness figure with care.** This judge returned only 0.0 or 1.0 on every answer it graded, so it behaves as pass/fail and cannot see partial degradation. For discriminating scores use the claim-level judge (`medbot.eval.rejudge` -> `results_<variant>_claims.json`), and see `results_sprint3.md` for the interpretation.

**Methodology and limitations:** Precision@K uses keyword-containment against manually verified expected phrases (see `dataset.py`), not embedding similarity or human judgment — stricter than a human grader on paraphrased-but-relevant chunks. Groundedness is scored by the same Gemini model that generated the answer (disclosed self-grading bias) rather than an independent judge model. Both are intentionally scoped v1 approaches.

## Weak spots

- Questions with Precision@4 < 0.5: What are the symptoms of alcoholism?, What are the side effects of acetaminophen?
- Questions with groundedness < 0.5: none

## Per-question results

| Question | Precision@K | Groundedness | Rationale |
|---|---|---|---|
| What causes acne? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the provided text regarding the causes, contributing factors, and triggers of acne. |
| What are the symptoms of alcoholism? | 0.25 | 1.00 | Every health problem, symptom, and personal effect listed in the generated answer is explicitly mentioned in the provided context under the description of alcoholism. |
| What causes allergic rhinitis? | 1.00 | 1.00 | Every claim in the generated answer regarding the causes and triggers of allergic rhinitis is directly and accurately supported by the provided text. |
| What are the symptoms of anaphylaxis? | 1.00 | 1.00 | Every symptom listed in the generated answer is explicitly mentioned and supported by the provided context. |
| What are the symptoms of appendicitis? | 1.00 | 1.00 | Every detail included in the generated answer regarding the symptoms of appendicitis is directly extracted and fully supported by the provided text. |
| What triggers an asthma attack? | 1.00 | 1.00 | Every single trigger and category listed in the generated answer is explicitly mentioned and supported by the provided context. |
| What is atherosclerosis? | 1.00 | 1.00 | Every claim in the generated answer is directly extracted and supported by the provided text about atherosclerosis and its key terms. |
| What are the symptoms of autism? | 0.75 | 1.00 | Every symptom and detail included in the generated answer is explicitly mentioned and directly supported by the provided text about autism. |
| How is bed-wetting treated in children? | 0.75 | 1.00 | Every treatment method, description, and detail included in the generated answer is directly extracted and fully supported by the provided context. |
| What causes bedsores? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the causes and symptoms section of the provided context. |
| What are the symptoms of bipolar disorder? | 0.75 | 1.00 | Every symptom and detail included in the generated answer for bipolar disorder's various episodes, states, and age groups is directly supported by the provided text. |
| What causes bladder cancer? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the provided context regarding the causes and risk factors of bladder cancer. |
| What are the symptoms of botulism? | 1.00 | 1.00 | Every detail provided in the generated answer regarding the symptoms of food-borne, infant, and wound botulism is directly extracted and supported by the text. |
| What are the symptoms of a brain tumor? | 0.75 | 1.00 | Every symptom listed in the generated answer is directly extracted and supported by the provided context. |
| What are the symptoms of breast cancer? | 1.00 | 1.00 | Every claim in the generated answer regarding the signs, findings, and trouble symptoms is directly supported by the provided context. |
| What are the symptoms of bronchitis? | 1.00 | 1.00 | Every symptom listed in the generated answer for both acute and chronic bronchitis is directly and accurately supported by the provided text. |
| What is bulimia nervosa? | 0.75 | 1.00 | Every claim made in the generated answer is directly extracted from and supported by the provided text. |
| What causes a bunion? | 0.75 | 1.00 | Every point listed in the generated answer regarding the causes of bunions is directly derived from and supported by the provided text. |
| How should burns be treated? | 0.75 | 1.00 | Every detail provided in the generated answer regarding the treatment of burns is directly and accurately supported by the provided text. |
| What is bursitis? | 0.75 | 1.00 | Every claim made in the generated answer regarding the symptoms and causes of bursitis is directly supported by the provided text. |
| What is an abscess? | 1.00 | 1.00 | Every detail provided in the generated answer regarding the definition, composition, and types of abscesses is directly and accurately supported by the text. |
| What are the side effects of acetaminophen? | 0.25 | 1.00 | Every listed side effect of acetaminophen in the generated answer is directly extracted and supported by the provided text. |
| What causes amnesia? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the provided text regarding the root causes of amnesia. |
| What are the symptoms of angina? | 1.00 | 1.00 | Every claim in the generated answer regarding the symptoms of angina is directly supported by the provided text. |
