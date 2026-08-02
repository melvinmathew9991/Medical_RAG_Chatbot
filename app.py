import streamlit as st

from medbot.data_processing import create_vector_database
from medbot.model_handler import initialize_model
from medbot.query_handler import (
    create_query_chain,
    format_external_results,
    format_sources,
    run_query,
    search_external_sources,
)

# Add a header for your chatbot
st.header("MEDBOT - Your Medical Chat Assistant")
st.caption(
    "MEDBOT provides general medical information for educational purposes only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider with questions about a medical condition."
)

# Initialize session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Load model based on configuration
model = initialize_model()

# Create or load the vector database
vectordb = create_vector_database()

if model is None or vectordb is None:
    st.error(
        "MEDBOT couldn't start: the language model or knowledge base failed to load. "
        "Check that GOOGLE_API_KEY is set correctly and see the terminal logs for details."
    )
    st.stop()

def process_user_input(user_query):
    query_chain = create_query_chain(model, vectordb, user_query)
    if query_chain is None:
        return {"result": "Sorry, something went wrong setting up the retrieval chain. Please try again."}
    # run_query strips the chain-of-thought reasoning trace so only the answer is shown.
    return run_query(query_chain, user_query)

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
        external_summary = format_external_results(external_results)

    answer = response['result']

    # "Retrieved from", not "Sources": these are the chunks the model was given,
    # which is a weaker claim than the chunks it used. See format_sources.
    citations = format_sources(response.get('source_documents'))
    if citations:
        answer = (
            f"{answer}\n\n---\n**Retrieved from** *(the corpus passages given to the "
            f"model; page numbers are PDF pages)*\n\n{citations}"
        )

    if external_summary:
        answer = f"{answer}\n\n---\n**Related external sources**\n\n{external_summary}"

    st.session_state.history.append({
        'role': 'Assistant',
        'content': answer
    })

    with st.chat_message("Assistant"):
        st.markdown(answer)
