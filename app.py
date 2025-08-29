# app.py
import streamlit as st
import asyncio
from workflow.ec_workflow import ExistingCustomerWorkflow
from workflow.nc_workflow import NewCustomerWorkflow
from workflow.rag import load_pdf_file, text_split, build_rag_pipeline
from src.memory import get_memory
from src import config

# ---------------- Setup ---------------- #
# Load RAG Knowledge Base
extracted_data = load_pdf_file("Data/")
text_chunks = text_split(extracted_data)
rag_chain = build_rag_pipeline(text_chunks)

# Shared memory
memory = get_memory()

# Initialize workflows
# Initialize workflows once and persist in session_state
if "ec_workflow" not in st.session_state:
    st.session_state.ec_workflow = ExistingCustomerWorkflow()

if "nc_workflow" not in st.session_state:
    st.session_state.nc_workflow = NewCustomerWorkflow()


# ---------------- Streamlit UI ---------------- #
st.set_page_config(page_title="RemoteLock Assistant", page_icon="🔒")
st.title("🔒 RemoteLock Support Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "bot", "content": "Hi! How can I help you today?"}]
    st.session_state.stage = "intro"

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input box
user_input = st.chat_input("Type your message...")
if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    async def process_input(user_input):
        stage = st.session_state.stage
        response = None
        result = None  # <-- define result upfront

        # ---------------- Conversation Flow ---------------- #
        if stage == "intro":
            response = "Are you an existing customer or a new customer? Or do you just need general information?"
            st.session_state.stage = "choose_flow"

        elif stage == "choose_flow":
            if "new" in user_input.lower():
                st.session_state.stage = "new_customer"
                result = await st.session_state.nc_workflow.process(user_input)

                response = result["answer"] if isinstance(result, dict) else result

            elif "exist" in user_input.lower():
                st.session_state.stage = "existing_customer"
                result = await st.session_state.ec_workflow.process(user_input)

                response = result["answer"] if isinstance(result, dict) else result

            elif "info" in user_input.lower() or "general" in user_input.lower():
                rag_answer = rag_chain.run(user_input)
                response = f"Here’s what I found:\n\n{rag_answer}"
                st.session_state.stage = "intro"  # reset after answering

            else:
                response = "Please type 'new', 'existing', or 'general information'."

        elif stage == "new_customer":
            result = await st.session_state.nc_workflow.process(user_input)
            response = result["answer"] if isinstance(result, dict) else result

        elif stage == "existing_customer":
            result = await st.session_state.ec_workflow.process(user_input)
            response = result["answer"] if isinstance(result, dict) else result


        else:
            response = "Sorry, I didn’t understand. Please say new, existing, or general info."

        return response


    # Run async workflow
    response_text = asyncio.run(process_input(user_input))

    # Show bot response
    with st.chat_message("bot"):
        st.markdown(response_text)
    st.session_state.messages.append({"role": "bot", "content": response_text})
