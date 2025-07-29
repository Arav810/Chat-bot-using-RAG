import os
import re
from dotenv import load_dotenv
from typing import List
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain_core.documents import Document
from weaviate import Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Weaviate

from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)

# Load environment variables
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def sanitize_class_name(file_name: str) -> str:
    base = os.path.splitext(file_name)[0]
    base = re.sub(r'\W+', '_', base)
    if not base[0].isalpha():
        base = 'Class_' + base
    return base

def load_document(file_path: str):
    """Dynamically load document using the appropriate LangChain loader."""
    extension = os.path.splitext(file_path)[1].lower()

    try:
        if extension == ".pdf":
            return PyPDFLoader(file_path).load()

        elif extension == ".txt":
            return TextLoader(file_path).load()

        elif extension == ".docx":
            return UnstructuredWordDocumentLoader(file_path).load()

        elif extension == ".csv":
            import pandas as pd
            df = pd.read_csv(file_path)
            processed_docs = []
            
            for _, row in df.iterrows():
                sentence = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                processed_docs.append(Document(page_content=sentence, metadata={"source": file_path}))

            # DEBUG
            if processed_docs:
                print("[DEBUG] Sample CSV chunk:")
                print(processed_docs[0].page_content[:1000])
                
                return processed_docs

        elif extension in [".xls", ".xlsx"]:
            import pandas as pd
            df = pd.read_excel(file_path)
            processed_docs = []
            
            for _, row in df.iterrows():
                sentence = ", ".join(f"{col}: {row[col]}" for col in df.columns)
                processed_docs.append(Document(page_content=sentence, metadata={"source": file_path}))

            # DEBUG
            if processed_docs:
                print("[DEBUG] Sample Excel chunk:")
                print(processed_docs[0].page_content[:1000])
                return processed_docs

        else:
            raise ValueError(f"Unsupported file format: {extension}")

    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")

def chunk_text(documents: List[Document], file_name: str, chunk_size=500, overlap=100):
    print("[INFO] Starting document chunking...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    all_chunks = splitter.split_documents(documents)
    print(f"[DEBUG] Total chunks created: {len(all_chunks)}")

    #Fix: Ensure all chunks have 'text' key for Weaviate
    for chunk in all_chunks:
        chunk.metadata["text"] = chunk.page_content

    print("[INFO] Connecting to Weaviate client...")
    client = Client(WEAVIATE_URL)

    sanitized_class_name = sanitize_class_name(file_name)
    print(f"[INFO] Using sanitized class name: {sanitized_class_name}")

    # Delete class if exists
    existing_classes = [c['class'].lower() for c in client.schema.get()['classes']]
    if sanitized_class_name.lower() in existing_classes:
        print(f"[INFO] Class `{sanitized_class_name}` already exists. Deleting it...")
        try:
            client.schema.delete_class(sanitized_class_name)
            print(f"[INFO] Deleted existing class `{sanitized_class_name}`.")
        except Exception as e:
            print(f"[ERROR] Failed to delete existing class: {e}")
            raise e

    # Create class
    print(f"[INFO] Creating new class `{sanitized_class_name}`...")
    schema = {
        "class": sanitized_class_name,
        "properties": [
            {
                "name": "text",
                "dataType": ["text"]
            }
        ],
        "vectorIndexConfig": {
            "distance": "cosine"
        },
        "vectorizer": "none"
    }
    client.schema.create_class(schema)

    # Add documents to Weaviate
    print("[INFO] Storing chunks in Weaviate with embeddings...")
    weaviate_store = Weaviate(
        client=client,
        index_name=sanitized_class_name,
        embedding=embedding_model,
        text_key="text"
    )

    weaviate_store.add_documents(all_chunks)
    print("[SUCCESS] Document chunks stored in Weaviate!")