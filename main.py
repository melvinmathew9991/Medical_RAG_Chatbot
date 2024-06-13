import streamlit as st
from model_handler import initialize_model
from data_processing import create_vector_database
from query_handler import create_query_chain, search_external_sources
from prompt import few_shot_template

# Add a header for your chatbot
st.header("MEDBOT - Your Medical Chat Assistant")

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Load model based on configuration
model = initialize_model()

# Create or load the vector database
vectordb = create_vector_database()

# Use the format_prompt function to generate prompts based on user input
def format_prompt(question):
    return few_shot_template.format(question=question)

def process_user_input(user_query):
    query_chain = create_query_chain(model, vectordb)
    response = query_chain.invoke({"query": user_query})
    return response

def process_external_queries(user_query):
    return search_external_sources(user_query)

# Display conversation history
for msg in st.session_state.history:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# User input field and processing
prompt = st.chat_input("Say something")
if prompt:
    st.session_state.history.append({
        'role': 'user',
        'content': prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner('Thinking'):
        response = process_user_input(prompt)
        external_results = process_external_queries(prompt)
        response_content = f"{response['result']}\n\nExternal Search Results:\n{external_results}"

    st.session_state.history.append({
        'role': 'Assistant',
        'content': response['result']
    })

    with st.chat_message("Assistant"):
        st.markdown(response['result'])
