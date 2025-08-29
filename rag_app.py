# app.py
import streamlit as st
from workflow.rag import load_pdf_file, text_split, build_rag_pipeline

st.set_page_config(page_title="RemoteLock RAG Chatbot", layout="wide")
st.title("🔑 RemoteLock RAG Chatbot")

# Load data + pipeline once
if "rag_chain" not in st.session_state:
    extracted_data = load_pdf_file("Data/")   # Load PDFs
    text_chunks = text_split(extracted_data)  # Split into chunks
    st.session_state.rag_chain = build_rag_pipeline(text_chunks)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if query := st.chat_input("Ask me anything about RemoteLock..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Run RAG pipeline
    result = st.session_state.rag_chain.invoke({"input": query})
    answer = result["answer"]

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
