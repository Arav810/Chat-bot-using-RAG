from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from utils import chunk_text, load_document
from rag_pipeline import process_query
from chat_history import save_chat_to_db, get_chat_history
import os
import nltk

os.environ["NLTK_DATA"] = "C:/Users/jeena/nltk_data"
nltk.data.find("tokenizers/punkt")

app = FastAPI()

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        os.makedirs("data", exist_ok=True)

        content = await file.read()
        file_path = os.path.join("data", file.filename)
        with open(file_path, "wb") as f:
            f.write(content)

        # Use the universal loader from utils.py
        documents = load_document(file_path)
        chunk_text(documents, file.filename)

        return {"message": f"File {file.filename} processed successfully"}

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def ask_question(query: str = Form(...), file_name: str = Form(...)):
    print(f"Received query: {query} for file: {file_name}")
    
    try:
        answer = process_query(query, file_name)
        print(f"Answer generated: {answer}")
        save_chat_to_db(file_name, query, answer)
        return {"answer": answer}
    
    except Exception as e:
        print(f"[ERROR] Failed to answer query: {e}")
        return {"answer": f"Error: {str(e)}"}

@app.get("/history/{file_name}")
async def get_history(file_name: str):
    history = get_chat_history(file_name)
    return {"history": history}