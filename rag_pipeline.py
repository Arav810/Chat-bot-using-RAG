import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import sanitize_class_name
import httpx
import weaviate
from weaviate import Client

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")

# Initialize embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Weaviate client
client = Client(WEAVIATE_URL)

# Query processing
def process_query(query, file_name, k=3):
    query_embedding = model.encode(query).tolist()
    sanitized_class_name = sanitize_class_name(file_name)

    try:
        # Query Weaviate directly
        response = client.query.get(
            class_name=sanitized_class_name,
            properties=["text"]
        ).with_near_vector({
            "vector": query_embedding
        }).with_limit(k).with_additional(["certainty", "distance"]).do()

        print("[DEBUG] Weaviate raw response:")
        print(response)

        # FIX: Case-insensitive lookup of class name in response
        get_data = response.get("data", {}).get("Get", {})
        matched_key = next((key for key in get_data.keys() if key.lower() == sanitized_class_name.lower()), None)
        results = get_data.get(matched_key, []) if matched_key else []

        if not results:
            print("[INFO] No results returned from Weaviate.")
            return "No relevant chunks found."

        print(f"[INFO] Retrieved {len(results)} chunks from Weaviate.")

        for i, chunk in enumerate(results):
            text = chunk.get("text", "")
            certainty = chunk.get("_additional", {}).get("certainty")
            distance = chunk.get("_additional", {}).get("distance")
            print(f"\n[DEBUG] Chunk {i+1}:")
            print(f"Certainty: {certainty}, Distance: {distance}")
            print(f"Text Preview: {text[:300]}...")

        # Filter relevant chunks
        relevant_chunks = [
            obj["text"]
            for obj in results
            if "text" in obj
        ]
        if not relevant_chunks:
            print("[INFO] No relevant 'text' field found in chunks.")
            return "No relevant text found in Weaviate results."

    except Exception as e:
        print("[ERROR] Exception during Weaviate query:", e)
        return f"Error querying Weaviate: {e}"

    # UPDATED PROMPT
    context = "\n\n".join(relevant_chunks)
    prompt = f"""You are a data analysis assistant. Use the context below to answer the question accurately.
If the answer is not present in the context, reply with 'The information is not available in the document.'

Context:
{context}

Question: {query}

Answer:"""

    # DEBUG: Print final prompt
    print("[DEBUG] Final Prompt sent to LLM:")
    print(prompt)

    return call_llm(prompt)

# LLM call
def call_llm(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}"
    }

    json_data = {
        "model": GROQ_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 300
    }

    try:
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=json_data)
        print("[DEBUG] Groq Raw Response:", response.text)
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("[ERROR] Groq API failed:", e)
        return f"Error calling LLM: {e}"