import streamlit as st
from config import PINECONE_NAMESPACE
from rag import list_namespaces, perform_rag

GREETING = "Hi! I'm here to answer your questions about the codebase. How can I assist you today?"

# Set the Streamlit app configuration
st.set_page_config(page_title="Codebase Chatbot", layout="wide")


@st.cache_data(ttl=60)
def get_codebases():
    # Cached briefly so the sidebar doesn't call Pinecone on every rerun
    return list_namespaces()


# Sidebar: choose which indexed codebase to chat about
with st.sidebar:
    st.header("Codebase")
    codebases = get_codebases()
    if codebases:
        default = codebases.index(PINECONE_NAMESPACE) if PINECONE_NAMESPACE in codebases else 0
        namespace = st.selectbox("Indexed codebases", codebases, index=default)
    else:
        namespace = None
        st.info("No codebases indexed yet.")
    st.caption("Add your own with:")
    st.code("python ingest.py https://github.com/owner/repo", language="bash")
    if st.button("Refresh list"):
        get_codebases.clear()
        st.rerun()

# Title and Description
st.title("Codebase Chatbot")
st.write(
    """
    Welcome to your codebase chatbot! Engage in a conversation about your codebase,
    and receive detailed answers based on the embedded context.
    """
)

# Start a fresh conversation whenever a different codebase is selected
if "messages" not in st.session_state or st.session_state.namespace != namespace:
    st.session_state.namespace = namespace
    st.session_state.messages = [{"role": "assistant", "content": GREETING}]

# Display previous messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # `role` can be "user" or "assistant"
        st.markdown(message["content"])  # Render the message content

# Input box for user query
if user_input := st.chat_input("Ask your question about the codebase...", disabled=namespace is None):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)  # Display user's message

    # Process the query with the RAG function
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            try:
                # Call perform_rag to generate a response
                response = perform_rag(user_input, namespace)
                st.markdown(response)  # Display assistant's response
                # Add assistant's response to session state
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
