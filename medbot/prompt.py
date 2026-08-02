from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.prompts import PromptTemplate

from medbot.config import LOCAL_EMBEDDING_MODEL

DISCLAIMER = (
    "You are a medical assistant. Answer the user's question using only the "
    "context provided below. If the answer isn't in the context, say you don't "
    "know rather than guessing at a diagnosis or treatment. This information is "
    "educational only and is not a substitute for professional medical advice, "
    "diagnosis, or treatment."
)

# Sprint 3: the chain-of-thought variant's system instruction. Differs from
# DISCLAIMER in one way that matters: it tells the model to actually check the
# context before concluding it can't answer. Sprint 2's audit found the model
# refusing on 4/24 questions whose answer was plainly in the retrieved context
# (medbot/eval/results.md) - the "say you don't know" clause above was being
# pattern-matched instead of applied. The safety intent is unchanged; only the
# order of operations is made explicit.
COT_DISCLAIMER = (
    "You are a medical assistant. Answer the user's question using only the "
    "context provided below.\n\n"
    "Work in two steps.\n\n"
    "First, under 'Reasoning:', state what the question is asking, then quote or "
    "paraphrase the specific parts of the context that bear on it, and decide "
    "whether they support an answer. The context is drawn from a medical "
    "encyclopedia and frequently answers the question in different words than the "
    "question uses - judge it on meaning, not on matching phrasing. If the context "
    "covers only part of the question, that still counts: answer the part it covers "
    "and say which part it does not. Only conclude that the context does not support "
    "an answer if, after actually checking it, nothing there addresses the question. "
    "Do not treat information about a different condition as if it were about the one "
    "asked about.\n\n"
    "Second, under 'Answer:', give the answer to the user based on that reasoning. "
    "The user never sees the reasoning, so write the answer so it stands on its own. "
    "State each supported fact on its own terms: do not join two separate statements "
    "from the context into a cause-and-effect chain unless the context itself asserts "
    "that link. "
    "Do not diagnose, and do not recommend a treatment the context does not state. "
    "This information is educational only and is not a substitute for professional "
    "medical advice, diagnosis, or treatment."
)

# The substantive guidance inside COT_DISCLAIMER, isolated so the ablation arms can
# reuse it verbatim. Sprint 3's audit (medbot/eval/sprint3_audit.md, finding F3)
# flagged that the shipped "cot" arm changed three things at once - this wording,
# the example format, and the example-selection strategy - so the improvement could
# not be attributed to chain-of-thought reasoning specifically. Holding this text
# identical across arms is what makes the two factors separable.
#
# COT_DISCLAIMER above deliberately still spells this out inline rather than
# interpolating the constant: it was measured byte-for-byte in the A/B run, and
# refactoring the string it was measured with would silently invalidate that run.
# test_prompt_variants.py asserts the two copies stay in sync.
CONTEXT_JUDGEMENT_GUIDANCE = (
    "The context is drawn from a medical "
    "encyclopedia and frequently answers the question in different words than the "
    "question uses - judge it on meaning, not on matching phrasing. If the context "
    "covers only part of the question, that still counts: answer the part it covers "
    "and say which part it does not. Only conclude that the context does not support "
    "an answer if, after actually checking it, nothing there addresses the question. "
    "Do not treat information about a different condition as if it were about the one "
    "asked about."
)

# Ablation arm: the new guidance, but no reasoning step and no CoT exemplars.
# Answers whether simply telling the model that paraphrased and partial support
# count is enough on its own - in which case the six exemplars and their ~2,400
# tokens per query are not earning their place.
INSTRUCTION_ONLY_DISCLAIMER = (
    "You are a medical assistant. Answer the user's question using only the "
    "context provided below. " + CONTEXT_JUDGEMENT_GUIDANCE + " "
    "Do not diagnose, and do not recommend a treatment the context does not state. "
    "This information is educational only and is not a substitute for professional "
    "medical advice, diagnosis, or treatment."
)

EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}"
)

COT_EXAMPLE_PROMPT = PromptTemplate(
    input_variables=["question", "context", "reasoning", "answer"],
    template="Question: {question}\nContext: {context}\nReasoning: {reasoning}\nAnswer: {answer}"
)

# Chain-of-thought exemplars. Every `context` below is condensed from chunks
# actually retrieved from this corpus (Gale Encyclopedia of Medicine Vol. 1) by
# running the live FAISS index - same discipline as medbot/eval/dataset.py, so
# these demonstrate the real retrieval shape rather than an idealized one.
#
# Topics are deliberately chosen from A-B entries that are NOT in the eval set,
# so the eval remains a held-out test rather than something the prompt has seen.
#
# The mix is intentional: four cases where the context supports an answer, one
# where it only partly does, and one where it genuinely does not. Sprint 3 is
# fixing false refusals, and the cheap way to score well on that would be a
# prompt that never refuses - which would trade a refusal bug for a
# hallucination bug. The anorexia example exists to keep refusal calibrated:
# retrieval returns bulimia's symptoms for that question, and the right
# behaviour is to notice they belong to a different disorder.
COT_EXAMPLES = [
    {
        "question": "What is an aortic aneurysm?",
        "context": (
            "Aortic aneurysm. Definition: An aneurysm is an abnormal bulging or swelling of a "
            "portion of a blood vessel. The aorta, which can develop these abnormal bulges, is "
            "the large blood vessel that carries oxygen-rich blood away from the heart to the "
            "rest of the body. Because the aorta is an artery, its walls are made up of three "
            "layers; a thin inner layer, a muscular middle layer that gives the vessel its "
            "flexibility under pressure, and a fiber-like outer layer that gives the vessel "
            "strength. Aortic aneurysms are also caused by a breakdown of the muscular middle "
            "layer of the artery wall, by high blood pressure, by direct injury to the chest, "
            "and although rare, by bacteria that can infect the aorta."
        ),
        "reasoning": (
            "The question asks for a definition of an aortic aneurysm. The context opens with "
            "exactly that: an aneurysm is 'an abnormal bulging or swelling of a portion of a "
            "blood vessel', and it identifies the aorta as the large vessel carrying "
            "oxygen-rich blood away from the heart. It goes on to describe the three-layered "
            "artery wall and to list causes. The context directly supports a definition, so I "
            "should give one."
        ),
        "answer": (
            "An aortic aneurysm is an abnormal bulging or swelling of a portion of the aorta, "
            "the large blood vessel that carries oxygen-rich blood away from the heart to the "
            "rest of the body. The aortic wall has three layers: a thin inner layer, a muscular "
            "middle layer that lets the vessel flex under the pressure of blood filling it, and "
            "a fibrous outer layer that gives it strength. Aneurysms can develop from a "
            "breakdown of that muscular middle layer, from high blood pressure, from direct "
            "injury to the chest, or, rarely, from bacteria infecting the aorta."
        ),
    },
    {
        "question": "What causes anxiety disorders?",
        "context": (
            "Anxiety may be produced by general social factors, and may produce physical, "
            "cognitive, emotional, or behavioral symptoms. The patient's ethnic or cultural "
            "background may also influence his or her vulnerability to certain forms of "
            "anxiety. Genetic factors that lead to biochemical abnormalities may also play a "
            "role. Anxiety in children may be caused by suffering from abuse, as well as by the "
            "factors that cause anxiety in adults. Anxiety may also be caused by environmental "
            "or occupational factors. People who must live or work around sudden or loud "
            "noises, bright or flashing lights, chemical vapors, or similar nuisances, which "
            "they cannot avoid or control, may develop heightened anxiety levels."
        ),
        "reasoning": (
            "The question asks about causes. The context names several directly: general social "
            "factors, ethnic or cultural background influencing vulnerability, genetic factors "
            "leading to biochemical abnormalities, abuse in children, and environmental or "
            "occupational factors such as uncontrollable noise, flashing lights, or chemical "
            "vapors. That is an explicit list of causes, so the context supports an answer."
        ),
        "answer": (
            "Several factors are described as contributing to anxiety. General social factors "
            "can produce it, and a person's ethnic or cultural background may influence how "
            "vulnerable they are to particular forms of anxiety. Genetic factors that lead to "
            "biochemical abnormalities may also play a role. In children, anxiety may be caused "
            "by abuse in addition to the factors that cause it in adults. Environmental and "
            "occupational conditions matter too: people who live or work around sudden or loud "
            "noises, bright or flashing lights, chemical vapors, or similar nuisances they "
            "cannot avoid or control may develop heightened anxiety levels."
        ),
    },
    {
        "question": "Why is a bronchoscopy performed?",
        "context": (
            "To allow examination of a suspected tumor, obstruction, secretion, bleeding, or "
            "foreign body in the airways; to determine the cause of a persistent cough, "
            "wheezing, or a cough that includes blood in the sputum; to evaluate the "
            "effectiveness of lung cancer treatments. There are two types of bronchoscopes, a "
            "rigid tube and a fiberoptic tube. Because of its flexibility, the fiberoptic tube "
            "is usually preferred. However, if the purpose of the procedure is to remove a "
            "foreign body caught in the windpipe or lungs of a child, the more rigid tube must "
            "be used because of its larger size."
        ),
        "reasoning": (
            "The question asks why the procedure is done. The context lists its purposes "
            "explicitly - examining a suspected tumor, obstruction, secretion, bleeding, or "
            "foreign body in the airways; finding the cause of a persistent cough, wheezing, or "
            "a cough with blood in the sputum; and evaluating how well lung cancer treatments "
            "are working. It also notes removal of a foreign body as a purpose when discussing "
            "which scope to use. The context supports a full answer."
        ),
        "answer": (
            "A bronchoscopy is performed to examine a suspected tumor, obstruction, secretion, "
            "bleeding, or foreign body in the airways; to determine the cause of a persistent "
            "cough, wheezing, or a cough that brings up blood in the sputum; and to evaluate "
            "how effective lung cancer treatments have been. It is also used to remove a "
            "foreign body caught in the windpipe or lungs - in a child, that particular use "
            "calls for the larger rigid scope rather than the usually preferred flexible "
            "fiberoptic one."
        ),
    },
    {
        "question": "What blood tests are used to check for anemia?",
        "context": (
            "Personal and family health history may suggest the presence of certain types of "
            "anemia. Laboratory tests that measure the percentage of red blood cells or the "
            "amount of hemoglobin in the blood are used to confirm diagnosis and determine "
            "which type of anemia is responsible for a patient's symptoms. X rays and "
            "examinations of bone marrow may be used to identify the source of bleeding. "
            "Anemia due to nutritional deficiencies can usually be treated at home with iron "
            "supplements or self administered injections of vitamin B12."
        ),
        "reasoning": (
            "The question asks specifically about blood tests. The context never uses the phrase "
            "'blood test', but it describes 'laboratory tests that measure the percentage of red "
            "blood cells or the amount of hemoglobin in the blood' - that is the same thing in "
            "different words, and it states what those tests are used for. Judging on meaning "
            "rather than matching phrasing, the context supports an answer."
        ),
        "answer": (
            "Anemia is checked for with laboratory tests that measure the percentage of red "
            "blood cells or the amount of hemoglobin in the blood. These are used both to "
            "confirm the diagnosis and to determine which type of anemia is causing the "
            "symptoms. Personal and family health history may point toward certain types of "
            "anemia beforehand, and X rays or examination of the bone marrow may be used "
            "separately to locate a source of bleeding."
        ),
    },
    {
        "question": "What is arthroscopy and what are its risks?",
        "context": (
            "Arthroscopy is used primarily by doctors who specialize in treating disorders of "
            "the bones and related structures (orthopedics) to help diagnose joint problems. "
            "This procedure is most commonly used to diagnose knee and shoulder problems, "
            "although the elbow, hip, wrist, and ankle may also be examined with an "
            "arthroscope. The arthroscope is an instrument used to look directly into the "
            "joint. It contains magnifying lenses and glass-coated fibers that send "
            "concentrated light into the joint. A camera attached to the arthroscope allows the "
            "surgeon to see a clear image of the joint. Because of the anesthesia, oxygen, "
            "suction, and monitoring equipment must be available. The patient's cardiac status "
            "should always be monitored in the event that any cardiac abnormalities arise "
            "during the arthroscopy."
        ),
        "reasoning": (
            "The question has two parts. On what arthroscopy is, the context is complete: an "
            "orthopedic procedure for diagnosing joint problems, most often knee and shoulder, "
            "using a lens-and-fiber instrument with a camera to look inside the joint. On "
            "risks, the context is thinner - it says only that anesthesia requires oxygen, "
            "suction and monitoring equipment to be on hand and that cardiac status must be "
            "monitored for abnormalities. It gives no list of complications such as infection "
            "or their rates. Partial coverage still deserves an answer, so I should describe "
            "the procedure fully, report the precautions the context does give, and say plainly "
            "that a full account of risks is not there."
        ),
        "answer": (
            "Arthroscopy is a procedure used mainly by orthopedic doctors to diagnose joint "
            "problems, most commonly in the knee and shoulder, though the elbow, hip, wrist, "
            "and ankle can also be examined. It uses an arthroscope, an instrument with "
            "magnifying lenses and glass-coated fibers that direct concentrated light into the "
            "joint; an attached camera gives the surgeon a clear image of the structures "
            "inside.\n\nOn risks, the available information is limited. It notes that because "
            "anesthesia is used, oxygen, suction, and monitoring equipment must be available, "
            "and that the patient's cardiac status should be monitored throughout in case "
            "abnormalities arise. It does not provide a complete list of possible "
            "complications, so ask a healthcare provider for a full discussion of the risks in "
            "your case."
        ),
    },
    {
        "question": "What are the symptoms of anorexia nervosa?",
        "context": (
            "According to the American Anorexia/Bulimia Association, Inc., warning signs of "
            "bulimia include uncontrolled binge eating followed by purging, and preoccupation "
            "with body weight. Groups at "
            "risk include dancers, fashion models, professional athletes, and actresses. Women "
            "whose biological mothers or sisters have the disorder appear to be at increased "
            "risk. A number of theories have been advanced to explain the psychological aspects "
            "of the disorder. Anorexia nervosa has been interpreted as a rejection of female "
            "sexual maturity, a reaction to sexual abuse or assault, or an overemphasis on "
            "control, autonomy, and independence."
        ),
        "reasoning": (
            "The question asks for the symptoms of anorexia nervosa. The first part of the "
            "context does describe symptoms, but they are labelled as warning signs of "
            "bulimia - a different eating disorder - so they cannot be reported as anorexia's. "
            "The second part is about anorexia, but it covers who is at risk (dancers, models, "
            "athletes, relatives of sufferers) and psychological theories of why the disorder "
            "develops. Those are risk factors and causes, not symptoms. Nothing here lists the "
            "symptoms of anorexia nervosa itself, so I should say so rather than pass off "
            "bulimia's symptoms as the answer."
        ),
        "answer": (
            "I don't know based on the provided context. The context lists warning signs of "
            "bulimia, which is a different eating disorder, and for anorexia nervosa it covers "
            "only risk factors and psychological theories about why the disorder develops - not "
            "its symptoms. I'd rather tell you that than present bulimia's symptoms as though "
            "they were anorexia's. Please consult a qualified healthcare provider for the "
            "symptoms of anorexia nervosa."
        ),
    },
]

class LazyLoader:
    def __init__(self):
        self.medical_examples = None
        self.example_selector = None

    def load_medical_examples(self):
        if not self.medical_examples:
            self.medical_examples = [
    {"question": "What are the symptoms of a common cold?", "answer": "The common cold can cause a runny or stuffy nose, sore throat, cough, and congestion. Some people may also experience fever, headache, muscle aches, or fatigue."},
    {"question": "What is the difference between a migraine and a tension headache?", "answer": "Migraines are typically much more severe than tension headaches and can cause throbbing pain on one side of the head, along with nausea, vomiting, and sensitivity to light and sound. Tension headaches often feel like a tight band around the head and don't usually cause other symptoms."},
    {"question": "What are some home remedies for a sore throat?", "answer": "Gargling with warm salt water, sucking on lozenges, and drinking plenty of fluids can help soothe a sore throat. You can also try over-the-counter pain relievers like ibuprofen or acetaminophen."},
    {"question": "What are the symptoms of COVID-19?", "answer": "COVID-19 symptoms can range from mild to severe and may include fever or chills, cough, shortness of breath or difficulty breathing, fatigue, muscle or body aches, headache, new loss of taste or smell, sore throat, congestion or runny nose, nausea or vomiting, and diarrhea."},
    {"question": "How is diabetes diagnosed?", "answer": "Diabetes is diagnosed through blood tests that measure blood sugar levels. Common tests include fasting blood sugar test, hemoglobin A1c test, and oral glucose tolerance test."},
    {"question": "What are the signs of a heart attack?", "answer": "Signs of a heart attack include chest pain or discomfort, shortness of breath, nausea or vomiting, pain or discomfort in the jaw, neck, or back, pain or discomfort in one or both arms, and cold sweats."},
    {"question": "What causes acne?", "answer": "Acne is caused by a combination of factors including excess oil production, clogged hair follicles, bacteria, and hormonal changes."},
    {"question": "How can you prevent the flu?", "answer": "Flu prevention methods include getting an annual flu vaccine, practicing good hand hygiene by washing hands frequently, avoiding close contact with sick individuals, and staying home when you are sick."},
    {"question": "What are the symptoms of asthma?", "answer": "Common symptoms of asthma include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early in the morning."},
    {"question": "What are the risk factors for developing high blood pressure?", "answer": "Risk factors for high blood pressure include being overweight or obese, lack of physical activity, excessive salt intake, smoking, excessive alcohol consumption, stress, and family history of high blood pressure."},
    {"question": "How is strep throat treated?", "answer": "Strep throat is typically treated with antibiotics prescribed by a healthcare provider. It's important to complete the full course of antibiotics as directed."},
    {"question": "What are the symptoms of a urinary tract infection (UTI)?", "answer": "Symptoms of a UTI may include a strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, cloudy or reddish urine, and pelvic pain in women."},
    {"question": "What is the recommended age for getting a colonoscopy?", "answer": "Screening colonoscopies are recommended starting at age 45 for average-risk individuals to screen for colorectal cancer. Individuals with higher risk factors may need to start screening earlier."},
    {"question": "How is arthritis managed?", "answer": "Arthritis management includes lifestyle changes such as regular exercise, maintaining a healthy weight, and using hot or cold therapy. Medications and physical therapy may also be recommended."},
    {"question": "What are the symptoms of depression?", "answer": "Symptoms of depression can include persistent feelings of sadness, loss of interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of worthlessness or guilt, difficulty concentrating, and thoughts of death or suicide."},
    {"question": "What are the benefits of regular exercise?", "answer": "Regular exercise can improve cardiovascular health, strengthen muscles and bones, help with weight management, reduce stress and anxiety, and boost overall mood and energy levels."},
    {"question": "What is the treatment for a bee sting?", "answer": "Treatment for a bee sting may involve removing the stinger if present, washing the affected area with soap and water, applying a cold compress, taking over-the-counter pain relievers, and using antihistamines or topical corticosteroids for allergic reactions."},
    {"question": "What are the symptoms of a concussion?", "answer": "Symptoms of a concussion can include headache, confusion, dizziness, nausea or vomiting, blurred vision, sensitivity to light or noise, memory problems, and changes in mood or behavior."},
    {"question": "How is osteoporosis diagnosed?", "answer": "Osteoporosis is diagnosed through bone density testing, such as a dual-energy X-ray absorptiometry (DXA) scan, which measures bone mineral density and assesses fracture risk."},
    {"question": "What are the symptoms of an allergic reaction?", "answer": "Symptoms of an allergic reaction can range from mild to severe and may include hives, itching, rash, swelling (face, lips, tongue, or throat), difficulty breathing, wheezing, chest tightness, and anaphylaxis."},
    {"question": "What are the treatment options for hypertension?", "answer": "Treatment options for hypertension (high blood pressure) include lifestyle modifications (such as dietary changes, regular exercise, weight loss, and stress reduction) and medications prescribed by a healthcare provider."},
    {"question": "Why do people get nosebleeds?", "answer": "Nosebleeds can be caused by dry air, frequent nose picking, trauma to the nose, allergies, or underlying medical conditions like high blood pressure."},
    {"question": "Why is it important to get enough sleep?", "answer": "Adequate sleep is crucial for overall health and well-being. It supports brain function, emotional well-being, physical health, and helps the body recover and repair itself."},
    {"question": "How does stress affect the body?", "answer": "Stress can affect the body in numerous ways, including increased heart rate, elevated blood pressure, digestive issues, weakened immune system, and can exacerbate existing health conditions."},
    {"question": "How can you manage allergies?", "answer": "Managing allergies involves avoiding triggers when possible, using medications like antihistamines or nasal sprays, and possibly undergoing allergen immunotherapy (allergy shots) for severe allergies."},
    {"question": "When should you see a doctor for a cough?", "answer": "You should see a doctor for a cough if it persists for more than a few weeks, is accompanied by fever, shortness of breath, chest pain, or if you cough up blood."},
    {"question": "When is the best time to take medication for high blood pressure?", "answer": "The timing of blood pressure medication can vary depending on the type of medication. It's important to follow your healthcare provider's instructions regarding when and how to take your medication."},
]

        return self.medical_examples

    def get_example_selector(self):
        if not self.example_selector:
            self.example_selector = SemanticSimilarityExampleSelector.from_examples(
                examples=self.load_medical_examples(),
                embeddings=FastEmbedEmbeddings(model_name=LOCAL_EMBEDDING_MODEL),
                vectorstore_cls=FAISS,
                k=1,
            )
        return self.example_selector

lazy_loader = LazyLoader()

# Which prompt the app ships with. "cot" is Sprint 3's default; "baseline" is the
# Sprint 2 prompt, kept runnable so the two can be A/B'd on the same harness
# rather than compared against numbers from an older run.
DEFAULT_PROMPT_VARIANT = "cot"


def _escape_braces(text):
    """Protect literal braces from PromptTemplate's {context}/{question} formatting."""
    return text.replace("{", "{{").replace("}", "}}")


def build_baseline_prompt(question):
    """
    Sprint 2's prompt: disclaimer + one semantically-selected question/answer
    example + placeholders for retrieved context.
    """
    examples = lazy_loader.get_example_selector().select_examples({"question": question})
    examples_block = "\n".join(EXAMPLE_PROMPT.format(**example) for example in examples)

    template = (
        f"{DISCLAIMER}\n\n"
        f"{_escape_braces(examples_block)}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def build_cot_prompt():
    """
    Sprint 3's prompt: all COT_EXAMPLES, each showing question -> context ->
    reasoning -> answer, then placeholders for retrieved context. Ends on
    "Reasoning:" so the model reasons before answering; the trace is stripped
    downstream by medbot.query_handler.strip_reasoning.

    Unlike the baseline this uses a fixed example set rather than semantic
    selection. Two reasons: the reasoning procedure is what these examples
    teach, and it does not vary by topic; and holding the examples constant
    keeps one more source of run-to-run variance out of the A/B, which matters
    given how noisy Sprint 2's refusals turned out to be.
    """
    examples_block = "\n\n".join(COT_EXAMPLE_PROMPT.format(**ex) for ex in COT_EXAMPLES)

    template = (
        f"{COT_DISCLAIMER}\n\n"
        "Here are worked examples of the two-step format:\n\n"
        f"{_escape_braces(examples_block)}\n\n"
        "Now answer this one in the same format.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Reasoning:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def build_instruction_only_prompt(question):
    """
    Ablation arm: baseline's single semantically-selected question/answer example,
    but with the new context-judgement guidance. No reasoning step.
    """
    examples = lazy_loader.get_example_selector().select_examples({"question": question})
    examples_block = "\n".join(EXAMPLE_PROMPT.format(**example) for example in examples)

    template = (
        f"{INSTRUCTION_ONLY_DISCLAIMER}\n\n"
        f"{_escape_braces(examples_block)}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def build_no_examples_prompt():
    """
    The arm §8 of results_sprint4.md identified as missing, and the cheapest one:
    the new guidance with NO examples at all.

    Why it matters. `instruction-only` is a misnomer inherited from Sprint 3 -- it
    is the new instruction plus baseline's legacy semantically-selected 1-shot
    example, not the instruction alone. The 2026-07-27 ablation found that arm at
    refusal parity with `cot` (1/24 vs 0/24 questions, Fisher p = 1.0000) while
    costing ~2,770 fewer tokens per query, so the exemplars are unjustified on
    refusals. What stopped them being dropped was that `instruction-only`
    substitutes a neighbouring example's question on 5/5 bursitis trials -- it
    answers about *nosebleeds*, drawn from its own selected exemplar. That defect
    belongs to the legacy 1-shot format, and both arms that use it show it.

    This arm removes the mechanism rather than the symptom: with no example in the
    prompt there is no other question for the model to answer, so contamination is
    impossible by construction rather than by measurement. At ~150 tokens it is
    also the cheapest arm in the set.

    Unmeasured until quota allows. Nothing ships on it yet; the app default stays
    `cot` (DEFAULT_PROMPT_VARIANT) until the refusal suite, the out-of-corpus guard
    and the claim-level judge have all been run against this variant.
    """
    template = (
        f"{INSTRUCTION_ONLY_DISCLAIMER}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


def build_examples_only_prompt():
    """
    Ablation arm: the six CoT exemplars, but under the ORIGINAL disclaimer - the
    reasoning procedure is demonstrated rather than instructed. Isolates how much
    the worked examples carry on their own.
    """
    examples_block = "\n\n".join(COT_EXAMPLE_PROMPT.format(**ex) for ex in COT_EXAMPLES)

    template = (
        f"{DISCLAIMER}\n\n"
        "Here are worked examples of the format to follow:\n\n"
        f"{_escape_braces(examples_block)}\n\n"
        "Now answer this one in the same format.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Reasoning:"
    )
    return PromptTemplate(input_variables=["context", "question"], template=template)


# The 2x2 behind the arms: {old, new} instruction x {Q->A, CoT} examples.
# `emits_reasoning` tells medbot.query_handler whether a response needs its
# reasoning trace stripped - it is a property of the prompt, so it belongs here
# rather than being re-derived by every caller.
# `needs_question` records whether the arm selects examples per question (the
# semantic-selector arms) or uses a fixed set (the CoT arms). Declared explicitly
# rather than inferred by catching TypeError, which would swallow genuine
# TypeErrors raised from inside a builder.
#
# "instruction-only" is a misnomer kept for continuity with the recorded Sprint 3
# and Sprint 4 numbers: it is the new instruction with baseline's legacy 1-shot
# example, NOT the instruction on its own. "no-examples" is the arm that actually
# has no examples. Renaming the older one now would orphan every recorded result
# filed under it, so the trap is documented instead.
PROMPT_VARIANTS = {
    "baseline":         {"builder": build_baseline_prompt,
                         "needs_question": True,  "emits_reasoning": False},
    "instruction-only": {"builder": build_instruction_only_prompt,
                         "needs_question": True,  "emits_reasoning": False},
    "no-examples":      {"builder": build_no_examples_prompt,
                         "needs_question": False, "emits_reasoning": False},
    "examples-only":    {"builder": build_examples_only_prompt,
                         "needs_question": False, "emits_reasoning": True},
    "cot":              {"builder": build_cot_prompt,
                         "needs_question": False, "emits_reasoning": True},
}


def _variant_spec(variant):
    variant = variant or DEFAULT_PROMPT_VARIANT
    try:
        return PROMPT_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"Unknown prompt variant {variant!r}; expected one of "
            f"{sorted(PROMPT_VARIANTS)}."
        ) from None


def emits_reasoning(variant=None):
    """Whether `variant`'s responses carry a reasoning trace needing removal."""
    return _variant_spec(variant)["emits_reasoning"]


def build_context_prompt(question, variant=None):
    """
    Build a RetrievalQA prompt for `question`.

    variant: one of PROMPT_VARIANTS; defaults to DEFAULT_PROMPT_VARIANT ("cot").
    An unknown variant raises rather than falling back - silently defaulting
    would let a typo in the eval harness quietly compare a variant against itself
    and report a null result as a finding.
    """
    spec = _variant_spec(variant)
    return spec["builder"](question) if spec["needs_question"] else spec["builder"]()
