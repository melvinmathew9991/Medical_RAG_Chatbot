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

# The original 24, frozen under this name on 2026-08-03 when EXPANSION_QUESTIONS
# was merged in. `EVAL_QUESTIONS` is now these plus the expansion (see the bottom
# of this file) and is what everything measures against.
#
# This list is kept addressable because the trial files recorded before the merge
# -- sprint4_t5_* and ablation_t5_* -- cover exactly these 24 questions and always
# will. Their regression gates are pinned to this list rather than to
# EVAL_QUESTIONS, so growing the eval set does not retroactively mark a complete
# historical dataset as truncated. Do not add to it; add to EXPANSION_QUESTIONS.
EVAL_QUESTIONS_V1 = [
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


# ---------------------------------------------------------------------------
# Question-set expansion, screened 2026-07-27 — MERGED INTO THE EVAL SET
# 2026-08-03, together with the 330 calls that re-measured the refusal suite
# over all 46 questions. Kept as a named list so the provenance of each half
# stays legible and so EVAL_QUESTIONS_V1 remains the frozen pre-merge set.
# ---------------------------------------------------------------------------
#
# Why this exists. Sprint 4 established that the binding constraint on the
# refusal result is the number of QUESTIONS, not trials per question: at 24
# questions the significance test survives (p = 0.0094) but three of the seven
# refusing questions refuse on exactly one trial of five, and dropping those puts
# p at 0.1092. Going 3 -> 5 trials cost 240 calls and moved the raw p-value by
# 0.0007 (see results_sprint4.md §7). More questions is the fix.
#
# Why it was held separate until 2026-08-03. Merging redefines REFUSAL_QUESTIONS
# from 24 to 46 questions, which invalidates every recorded trial file at a stroke
# — they cover 24 — so the merge had to be the *second* step, taken together with
# the calls that re-measure the suite. Screening and keyword grounding cost
# nothing, so they were done and committed first and the numbers followed.
#
# What that cost in the end: 330 calls (22 new questions x 5 trials x 3 arms),
# resumed onto the recorded 24 rather than re-running them, which is only valid
# because the prompts, index and model were unchanged. The pre-merge files stay
# frozen at 24 and are gated against EVAL_QUESTIONS_V1.
#
# How these were selected — the same discipline as the list above, with a tool:
#   - 34 candidates screened by `python -m medbot.eval.verify_entry`, which
#     requires the distinctive term in >= 2 of the top-4 chunks AND at least one
#     chunk that looks like the entry itself rather than a cross-reference.
#   - 10 auto-rejected. Notably "What causes back pain?" (top chunks were the
#     BURSITIS entry — "pain" appears throughout a medical encyclopedia) and
#     "What is Barrett's esophagus?" (top chunk was ACETAMINOPHEN). The loose rule
#     used for the out-of-corpus guard accepted 35 of 36; that rule is right for
#     proving ABSENCE and wrong for proving presence.
#   - 2 rejected by hand after reading the retrieved text: "What is a biopsy?" and
#     "What is an antibody test?". Both pass the screen, but the corpus has no
#     general entry for either — it has *breast biopsy* / *bone biopsy* and
#     *antimyocardial* / *antinuclear antibody test*. The question would be scored
#     against whichever specific entry retrieval happened to surface.
#   - Topics used by the CoT exemplars in medbot/prompt.py (aortic aneurysm,
#     anxiety disorders, bronchoscopy, anemia blood tests, arthroscopy, anorexia
#     nervosa) are excluded, so the eval stays held out from the prompt.
#
# `expected_keywords` were read out of the actually-retrieved chunks, never from
# general medical knowledge. Verified 2026-07-27 against the live index: **every
# keyword appears in at least one retrieved chunk for its question**, and
# Precision@4 is 1.00 x12, 0.75 x7, 0.50 x3 — **mean 0.8523**, against 0.8333 for
# the current 24 (whose distribution is 1.00 x12, 0.75 x10, 0.25 x2, so this set
# is comparable in difficulty and its floor is higher, not lower).
# `tests/test_expansion_selection.py` re-checks all of that on every test run.
EXPANSION_QUESTIONS = [
    {"question": "What causes anemia?",
     "expected_keywords": ["anemia", "folic acid", "pernicious"]},
    {"question": "What is angioplasty?",
     "expected_keywords": ["angioplasty", "widen vessels narrowed", "stenoses"]},
    {"question": "What is an arrhythmia?",
     "expected_keywords": ["arrhythmia", "heartbeat pattern", "abnormality in the heart"]},
    {"question": "What causes asbestosis?",
     "expected_keywords": ["asbestosis", "occupational exposure", "asbestos fiber"]},
    {"question": "What is astigmatism?",
     "expected_keywords": ["astigmatism", "cornea", "blurred image"]},
    {"question": "What are the symptoms of atrial fibrillation?",
     "expected_keywords": ["atrial fibrillation", "out of sync", "shortness of breath"]},
    {"question": "What is an audiometry test?",
     "expected_keywords": ["audiometry", "sound frequencies", "audiogram"]},
    {"question": "What is aphasia?",
     "expected_keywords": ["aphasia", "ability to communicate", "written words"]},
    {"question": "What causes atopic dermatitis?",
     "expected_keywords": ["atopic dermatitis", "itchy", "eczema"]},
    {"question": "What is arteriography?",
     "expected_keywords": ["arteriography", "angiography", "blood vessels visible"]},
    {"question": "What causes alopecia?",
     "expected_keywords": ["alopecia", "hair loss", "thyroid"]},
    {"question": "What is amblyopia?",
     "expected_keywords": ["amblyopia", "decrease in vision", "one or both eyes"]},
    {"question": "What are the symptoms of anthrax?",
     "expected_keywords": ["anthrax", "inhalation", "spores"]},
    # "palsy" and not "bell's palsy": the PDF uses a curly apostrophe (Bell’s), so
    # an ASCII-quoted keyword matches nothing. Two more such traps are avoided the
    # same way across this list.
    {"question": "What are the symptoms of Bell's palsy?",
     "expected_keywords": ["palsy", "facial weakness", "hsv infection"]},
    {"question": "What is a barium enema?",
     "expected_keywords": ["barium enema", "screening", "colorectal cancer"]},
    {"question": "What causes bronchiectasis?",
     "expected_keywords": ["bronchiectasis", "bronchial tubes", "obstructed"]},
    {"question": "What is a bone marrow transplant?",
     "expected_keywords": ["bone marrow transplant", "stem cells", "sponge-like tissue"]},
    {"question": "What causes byssinosis?",
     "expected_keywords": ["byssinosis", "brown lung", "textile worker"]},
    {"question": "What causes bad breath?",
     "expected_keywords": ["bad breath", "halitosis", "unpleasant odor"]},
    {"question": "What are the symptoms of berylliosis?",
     "expected_keywords": ["berylliosis", "beryllium", "lung inflammation"]},
    {"question": "What is balloon valvuloplasty?",
     "expected_keywords": ["valvuloplasty", "heart valve", "stretched open"]},
    {"question": "What causes bruises?",
     "expected_keywords": ["bruises", "ecchymoses", "purpura senilis"]},
]

# The eval set everything measures against: the original 24 plus the 22 screened
# expansion questions, merged 2026-08-03 in the same change as the 330 calls that
# re-measured the refusal suite over all 46. Merging without that run is what the
# now-deleted test_the_expansion_is_not_merged_into_the_eval_set_yet prevented,
# because it silently redefines REFUSAL_QUESTIONS and leaves every recorded trial
# file covering only half the suite.
#
# Why expand at all: the refusal headline was limited by the number of QUESTIONS,
# not by trials per question. results_sprint4.md §7 established that going 3 -> 5
# trials does not resolve the fragility warning -- extra trials surface new
# one-in-N refusers rather than confirming the existing ones -- and §11 hit the
# same wall from the other side, where cot-vs-no-examples parity was a null result
# at n=24 that could not distinguish equivalence from an underpowered test.
EVAL_QUESTIONS = EVAL_QUESTIONS_V1 + EXPANSION_QUESTIONS
