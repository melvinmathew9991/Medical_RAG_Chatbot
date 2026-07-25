# MEDBOT Evaluation Results - `cot` prompt

Ran 24 questions through the live RAG pipeline (retrieval top-4, Gemini `gemini-flash-lite-latest` generation, `cot` prompt variant).

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
| What causes acne? | 0.75 | 1.00 | Every single point in the generated answer is directly extracted from the provided text under the causes and risk factors section of acne. |
| What are the symptoms of alcoholism? | 0.25 | 1.00 | Every symptom, physical sign, withdrawal effect, and behavioral problem mentioned in the generated answer is directly sourced and supported by the provided text. |
| What causes allergic rhinitis? | 1.00 | 1.00 | Every claim in the generated answer regarding the causes of allergic rhinitis, seasonal triggers, and perennial triggers is directly supported by details explicitly stated in the provided context. |
| What are the symptoms of anaphylaxis? | 1.00 | 1.00 | Every symptom and detail included in the generated answer is directly supported by the provided text. |
| What are the symptoms of appendicitis? | 1.00 | 1.00 | Every detail provided in the generated answer regarding the symptoms of appendicitis and signs of rupture is directly and accurately supported by the provided text. |
| What triggers an asthma attack? | 1.00 | 1.00 | Every single trigger listed in the generated answer is explicitly mentioned and supported by the provided context. |
| What is atherosclerosis? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by information found in the provided context. |
| What are the symptoms of autism? | 0.75 | 1.00 | Every symptom and detail included in the generated answer is directly supported by the provided text regarding autism. |
| How is bed-wetting treated in children? | 0.75 | 1.00 | Every treatment method listed in the generated answer—including bladder training, motivational therapy, drugs, surgery, psychotherapy, diet therapy, and alternative treatments—is directly and accurately supported by the provided text. |
| What causes bedsores? | 0.75 | 1.00 | Every single cause and risk factor listed in the generated answer is directly extracted from the provided text without any unsupported claims. |
| What are the symptoms of bipolar disorder? | 0.75 | 1.00 | Every claim in the generated answer regarding the symptoms of bipolar disorder for both adults and younger patients is directly supported by the provided text. |
| What causes bladder cancer? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by information found in the provided context. |
| What are the symptoms of botulism? | 1.00 | 1.00 | Every detail provided in the generated answer regarding the symptoms of food-borne, infant, and wound botulism is directly supported by the text. |
| What are the symptoms of a brain tumor? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the information provided in the text. |
| What are the symptoms of breast cancer? | 1.00 | 1.00 | Every symptom and sign listed in the generated answer is explicitly mentioned and supported by the provided text. |
| What are the symptoms of bronchitis? | 1.00 | 1.00 | Every detail regarding the symptoms of both acute and chronic bronchitis in the generated answer is directly and accurately supported by the provided text. |
| What is bulimia nervosa? | 0.75 | 1.00 | Every claim in the generated answer is directly supported by the details provided in the context about bulimia nervosa. |
| What causes a bunion? | 0.75 | 1.00 | Every claim in the generated answer regarding the causes and contributing factors of bunions is directly supported by the provided text. |
| How should burns be treated? | 0.75 | 1.00 | Every detail in the generated answer regarding the treatment of thermal, chemical, and electrical burns, as well as alternative treatments, is directly derived from and supported by the provided text. |
| What is bursitis? | 0.75 | 1.00 | Every detail provided in the generated answer regarding the causes, symptoms, and nature of bursitis is directly extracted and supported by the text. |
| What is an abscess? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the provided text regarding the definition, formation, and types of abscesses. |
| What are the side effects of acetaminophen? | 0.25 | 1.00 | Every detail provided in the generated answer regarding the side effects, allergic reactions, and overdoses of acetaminophen is directly and accurately supported by the provided text. |
| What causes amnesia? | 1.00 | 1.00 | Every claim in the generated answer is directly and accurately supported by the provided context regarding the causes of amnesia. |
| What are the symptoms of angina? | 1.00 | 1.00 | Every claim in the generated answer is directly supported by the provided text regarding the symptoms of angina. |
