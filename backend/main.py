from fastapi import FastAPI, UploadFile, File, HTTPException
from .schemas import ChatResponse, ChatRequest
from .rag_chain import run_rag_pipeline, process_document, qdrant_client
import redis
import json
import os
from typing import List
from dotenv import load_dotenv
import asyncio
from qdrant_client.http import models as rest

load_dotenv()

app = FastAPI()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

try:
    redis_client = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    redis_client.ping() 
except redis.ConnectionError:
    raise RuntimeError("Redis server is not running or unreachable.")

CHAT_HISTORY_KEY = "chat_history"

@app.get("/")
def root():
    return {"message": "RAG backend is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        user_question = request.query

       
        answer = await asyncio.to_thread(run_rag_pipeline, user_question)

        
        chat_entry = {"user": user_question, "bot": answer}
        redis_client.rpush(CHAT_HISTORY_KEY, json.dumps(chat_entry))

        return ChatResponse(answer=answer)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in /chat: {str(e)}")

@app.post("/upload-doc")
async def upload_document(files: List[UploadFile] = File(...)):
    processed_files = []
    try:

        qdrant_client.delete(
        collection_name="chatbot-knowledge",
        points_selector=rest.Filter(must=[])  # No filter = delete all points
        )

        for file in files:
            await asyncio.to_thread(process_document, file)
            processed_files.append(file.filename)

        return {"message": f"Files processed successfully: {', '.join(processed_files)}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")

@app.get("/history")
def get_chat_history():
    try:
        history = redis_client.lrange(CHAT_HISTORY_KEY, 0, -1)
        formatted_history = [json.loads(item) for item in history]
        return {"history": formatted_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")