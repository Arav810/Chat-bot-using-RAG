import streamlit as st
import requests

st.set_page_config(page_title="Chatbot", page_icon="🤖")
st.title("💬 Chatbot with RAG")

BACKEND_URL = "http://127.0.0.1:8000"

uploaded_files = st.file_uploader("Upload a File:-",type=["pdf","txt","docx","csv", "xlsx"],accept_multiple_files=True)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) selected.")

    with st.spinner("Processing your files..."):

        files_payload=[]
        for file in uploaded_files:
            files_payload.append(("files",(file.name,file,file.type)))
        try:
            response = requests.post(f"{BACKEND_URL}/upload-doc",files= files_payload)
            if response.status_code == 200:
              st.success("file processed successfully")
            else:
              st.error(f"failed to process file:{response.text}")
        except Exception as e:
            st.error(f"Error uploading files: {e}")


if "loaded_history" not in st.session_state:
    try:
        response = requests.get(f"{BACKEND_URL}/history")
        if response.status_code == 200:
            history_data = response.json().get("history", [])
            st.session_state.chat_history = [(item["user"], item["bot"]) for item in history_data]
        else:
            st.session_state.chat_history = []
    except:
        st.session_state.chat_history = []
    st.session_state.loaded_history = True

user_input = st.text_input("Ask your question:")

if user_input:
    
    st.session_state.chat_history.append(("You", user_input))

    try:
        response = requests.post(f"{BACKEND_URL}/chat", json={"query": user_input})
        bot_reply = response.json().get("answer", "Error: No response")
    except Exception as e:
        bot_reply = f"⚠️ Error contacting backend: {e}"

    st.session_state.chat_history.append(("Bot", bot_reply))

for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"🧑‍💻 **You:** {msg}")
    else:
        st.markdown(f"🤖 **Bot:** {msg}")
