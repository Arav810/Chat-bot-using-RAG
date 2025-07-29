import streamlit as st
import requests
import os

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG_pdf_Q & A", layout="wide")
st.title("PDF-Based Question Answering App using RAG")

# Initialize storage for multiple file-based chat histories
if "file_chat_histories" not in st.session_state:
    st.session_state.file_chat_histories = {}

# Upload multiple files
st.sidebar.header("Upload PDF / DOCX / TXT / CSV / Excel files")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more files",
    type=["pdf", "docx", "txt", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

# Upload files to backend
if uploaded_files:
    for uploaded_file in uploaded_files:
        with st.spinner(f"Uploading and processing {uploaded_file.name}..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(f"{BACKEND_URL}/upload", files=files)
            if response.status_code == 200:
                st.sidebar.success(f"{uploaded_file.name} processed successfully!")
                # Set current_file to first uploaded file if not set
                if "current_file" not in st.session_state:
                    st.session_state["current_file"] = uploaded_file.name
                # Initialize chat history if not already done
                if uploaded_file.name not in st.session_state.file_chat_histories:
                    st.session_state.file_chat_histories[uploaded_file.name] = []
            else:
                st.sidebar.error(f"Failed to upload {uploaded_file.name}")

# File selection dropdown
if st.session_state.file_chat_histories:
    selected_file = st.sidebar.selectbox(
        "Select file to chat with",
        list(st.session_state.file_chat_histories.keys()),
        index=list(st.session_state.file_chat_histories.keys()).index(st.session_state["current_file"])
        if "current_file" in st.session_state else 0
    )
    st.session_state["current_file"] = selected_file
    file_name = selected_file

    # Load chat history from backend if not already loaded
    if file_name not in st.session_state.file_chat_histories or not st.session_state.file_chat_histories[file_name]:
        try:
            res = requests.get(f"{BACKEND_URL}/history/{file_name}")
            if res.status_code == 200:
                history_data = res.json().get("history", [])
                chat_list = [(entry["query"], entry["answer"]) for entry in history_data]
                st.session_state.file_chat_histories[file_name] = chat_list
            else:
                st.session_state.file_chat_histories[file_name] = []
        except Exception as e:
            st.error(f"Error fetching chat history: {e}")
            st.session_state.file_chat_histories[file_name] = []

    # Set current chat history for display and update
    st.session_state.chat_history = st.session_state.file_chat_histories[file_name]

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat[0])
        with st.chat_message("assistant"):
            st.markdown(chat[1])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Getting answer from LLM..."):
            data = {
                "query": prompt,
                "file_name": file_name
            }
            res = requests.post(f"{BACKEND_URL}/query", data=data)

        if res.status_code == 200:
            answer = res.json().get("answer", "")
        else:
            answer = "Error getting response from backend."

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Update chat history in session
        st.session_state.chat_history.append((prompt, answer))
        st.session_state.file_chat_histories[file_name] = st.session_state.chat_history

else:
    st.info("Please upload a file to start chatting.")