from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

medical_examples = [
    {    
        "question": "What are the symptoms of a common cold?",
        "answer": "The common cold can cause a runny or stuffy nose, sore throat, cough, and congestion. Some people may also experience fever, headache, muscle aches, or fatigue."
    },
    {
        "question": "What is the difference between a migraine and a tension headache?",
        "answer": "Migraines are typically much more severe than tension headaches and can cause throbbing pain on one side of the head, along with nausea, vomiting, and sensitivity to light and sound. Tension headaches often feel like a tight band around the head and don't usually cause other symptoms."
    },
    {
        "question": "What are some home remedies for a sore throat?",
        "answer": "Gargling with warm salt water, sucking on lozenges, and drinking plenty of fluids can help soothe a sore throat. You can also try over-the-counter pain relievers like ibuprofen or acetaminophen."
    },
    {
        "question": "What are the symptoms of COVID-19?",
        "answer": "COVID-19 symptoms can range from mild to severe and may include fever or chills, cough, shortness of breath or difficulty breathing, fatigue, muscle or body aches, headache, new loss of taste or smell, sore throat, congestion or runny nose, nausea or vomiting, and diarrhea."
    },
    {
        "question": "How is diabetes diagnosed?",
        "answer": "Diabetes is diagnosed through blood tests that measure blood sugar levels. Common tests include fasting blood sugar test, hemoglobin A1c test, and oral glucose tolerance test."
    },
    {
        "question": "What are the signs of a heart attack?",
        "answer": "Signs of a heart attack include chest pain or discomfort, shortness of breath, nausea or vomiting, pain or discomfort in the jaw, neck, or back, pain or discomfort in one or both arms, and cold sweats."
    },
    {
        "question": "What causes acne?",
        "answer": "Acne is caused by a combination of factors including excess oil production, clogged hair follicles, bacteria, and hormonal changes."
    },
    {
        "question": "How can you prevent the flu?",
        "answer": "Flu prevention methods include getting an annual flu vaccine, practicing good hand hygiene by washing hands frequently, avoiding close contact with sick individuals, and staying home when you are sick."
    },
    {
        "question": "What are the symptoms of asthma?",
        "answer": "Common symptoms of asthma include wheezing, shortness of breath, chest tightness, and coughing, especially at night or early in the morning."
    },
    {
        "question": "What are the risk factors for developing high blood pressure?",
        "answer": "Risk factors for high blood pressure include being overweight or obese, lack of physical activity, excessive salt intake, smoking, excessive alcohol consumption, stress, and family history of high blood pressure."
    },
    {
        "question": "How is strep throat treated?",
        "answer": "Strep throat is typically treated with antibiotics prescribed by a healthcare provider. It's important to complete the full course of antibiotics as directed."
    },
    {
        "question": "What are the symptoms of a urinary tract infection (UTI)?",
        "answer": "Symptoms of a UTI may include a strong, persistent urge to urinate, a burning sensation when urinating, passing frequent, small amounts of urine, cloudy or reddish urine, and pelvic pain in women."
    },
    {
        "question": "What is the recommended age for getting a colonoscopy?",
        "answer": "Screening colonoscopies are recommended starting at age 45 for average-risk individuals to screen for colorectal cancer. Individuals with higher risk factors may need to start screening earlier."
    },
    {
        "question": "How is arthritis managed?",
        "answer": "Arthritis management includes lifestyle changes such as regular exercise, maintaining a healthy weight, and using hot or cold therapy. Medications and physical therapy may also be recommended."
    },
    {
        "question": "What are the symptoms of depression?",
        "answer": "Symptoms of depression can include persistent feelings of sadness, loss of interest or pleasure in activities, changes in appetite or weight, sleep disturbances, fatigue, feelings of worthlessness or guilt, difficulty concentrating, and thoughts of death or suicide."
    },
    {
        "question": "What are the benefits of regular exercise?",
        "answer": "Regular exercise can improve cardiovascular health, strengthen muscles and bones, help with weight management, reduce stress and anxiety, and boost overall mood and energy levels."
    },
    {
        "question": "What is the treatment for a bee sting?",
        "answer": "Treatment for a bee sting may involve removing the stinger if present, washing the affected area with soap and water, applying a cold compress, taking over-the-counter pain relievers, and using antihistamines or topical corticosteroids for allergic reactions."
    },
    {
        "question": "What are the symptoms of a concussion?",
        "answer": "Symptoms of a concussion can include headache, confusion, dizziness, nausea or vomiting, blurred vision, sensitivity to light or noise, memory problems, and changes in mood or behavior."
    },
    {
        "question": "How is osteoporosis diagnosed?",
        "answer": "Osteoporosis is diagnosed through bone density testing, such as a dual-energy X-ray absorptiometry (DXA) scan, which measures bone mineral density and assesses fracture risk."
    },
    {
        "question": "What are the symptoms of an allergic reaction?",
        "answer": "Symptoms of an allergic reaction can range from mild to severe and may include hives, itching, rash, swelling (face, lips, tongue, or throat), difficulty breathing, wheezing, chest tightness, and anaphylaxis."
    },
    {
        "question": "What are the treatment options for hypertension?",
        "answer": "Treatment options for hypertension (high blood pressure) include lifestyle modifications (such as dietary changes, regular exercise, weight loss, and stress reduction) and medications prescribed by a healthcare provider."
    }
]

# Starting prompt for your medical chatbot
starting_prompt = "Welcome to our medical chatbot! How can I assist you today?"

# Create a PromptTemplate for formatting questions and answers
prompt_template = PromptTemplate(
    input_variables = ["question", "answer"],
    template="""
    Question : {question}
    Answer : {answer}
    """
)

# Create the SemanticSimilarityExampleSelector
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples = medical_examples,
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),  
    vectorstore_cls = Chroma,  
    k = 1,  
)

# Create a FewShotPromptTemplate using the ExampleSelector
few_shot_template = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt = prompt_template,
    prefix = "Answer according to the question asked by the user",
    suffix = "Question : {question}\nAnswer:",
    input_variables = ["question"],
    example_separator = "\n"
)

# Example usage function
def format_prompt(question):
    return few_shot_template.format(question = question)

