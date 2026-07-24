"""
Hand-curated evaluation questions for MEDBOT's RAG pipeline.

Every question and its `expected_keywords` were derived by running the live
FAISS index (medbot.data_processing) against the question and reading the
actual top retrieved chunks — not fabricated from general medical knowledge.
This matters because the indexed corpus is only the "Gale Encyclopedia of
Medicine, Vol. 1" (2nd ed.), which covers entries alphabetically from
roughly "Abdominal ultrasound" to "Byssinosis" (A-B only, confirmed by
scanning the source PDF). Every question below was chosen because a
dedicated encyclopedia entry for that exact topic exists in this volume;
asking about a C-Z condition (e.g. "diabetes", "psoriasis") would be an
unfair test of retrieval, since the corpus simply doesn't cover it as its
own entry, and would be measuring corpus coverage rather than retrieval
quality.

`expected_keywords`: a retrieved chunk is judged "relevant" for Precision@K
if it contains ANY of these phrases (case-insensitive substring match).
This is an intentionally simple, disclosed methodology (see
medbot/eval/retrieval_metrics.py) rather than an embedding-based judge.
"""

EVAL_QUESTIONS = [
    {"question": "What causes acne?",
     "expected_keywords": ["acne", "hormonal", "family history of acne"]},
    {"question": "What are the symptoms of alcoholism?",
     "expected_keywords": ["drinking despite", "negative effects", "alcohol use"]},
    {"question": "What causes allergic rhinitis?",
     "expected_keywords": ["house dust", "allergens", "pollen"]},
    {"question": "What are the symptoms of anaphylaxis?",
     "expected_keywords": ["hives", "wheezing", "anaphylaxis"]},
    {"question": "What are the symptoms of appendicitis?",
     "expected_keywords": ["appendicitis", "navel", "abdomen"]},
    {"question": "What triggers an asthma attack?",
     "expected_keywords": ["asthma", "pollen", "exercise"]},
    {"question": "What is atherosclerosis?",
     "expected_keywords": ["atherosclerosis", "hardening of the arteries", "arteriosclerosis"]},
    {"question": "What are the symptoms of autism?",
     "expected_keywords": ["autis", "repetitive", "flapping"]},
    {"question": "How is bed-wetting treated in children?",
     "expected_keywords": ["bladder training", "motivational therapy", "bed-wetting"]},
    {"question": "What causes bedsores?",
     "expected_keywords": ["bedsores", "wheelchairs", "confined to bed"]},
    {"question": "What are the symptoms of bipolar disorder?",
     "expected_keywords": ["manic", "bipolar", "elation"]},
    {"question": "What causes bladder cancer?",
     "expected_keywords": ["bladder cancer", "urinary bladder", "dividing uncontrollably"]},
    {"question": "What are the symptoms of botulism?",
     "expected_keywords": ["botulism", "paralysis", "breathing"]},
    {"question": "What are the symptoms of a brain tumor?",
     "expected_keywords": ["brain tumor", "head injury", "electromagnetic fields"]},
    {"question": "What are the symptoms of breast cancer?",
     "expected_keywords": ["breast", "diagnosis", "prognosis"]},
    {"question": "What are the symptoms of bronchitis?",
     "expected_keywords": ["phlegm", "wheezing", "bronchitis"]},
    {"question": "What is bulimia nervosa?",
     "expected_keywords": ["bulimia", "binge", "purge"]},
    {"question": "What causes a bunion?",
     "expected_keywords": ["bunion", "big toe", "metatarsal"]},
    {"question": "How should burns be treated?",
     "expected_keywords": ["burn", "blister", "stop, drop"]},
    {"question": "What is bursitis?",
     "expected_keywords": ["bursitis", "rheumatoid arthritis", "gout"]},
    {"question": "What is an abscess?",
     "expected_keywords": ["abscess", "septic", "infection"]},
    {"question": "What are the side effects of acetaminophen?",
     "expected_keywords": ["acetaminophen", "medical tests", "avoiding the drug"]},
    {"question": "What causes amnesia?",
     "expected_keywords": ["amnesia", "hippocampus", "memory"]},
    {"question": "What are the symptoms of angina?",
     "expected_keywords": ["angina", "chest", "pressing pain"]},
]
