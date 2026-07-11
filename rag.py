
"""
rag.py
Simple RAG Engine

Note: ChromaDB package install is blocked by Replit's package firewall
in this environment, so a lightweight local vector store (JSON + numpy
cosine similarity) is used instead. It is persisted to VECTOR_DB_PATH,
just like a real vector DB would be, and exposes the same
create_vector_db()/search_context() functions used by app.py.
"""

import os
import json
import uuid

import numpy as np

from pypdf import PdfReader

from google import genai

from config import (
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    VECTOR_DB_PATH,
    TOP_K,
)

# ---------------------------------
# Gemini Client
# ---------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ---------------------------------
# Local Vector Store
# ---------------------------------

os.makedirs(VECTOR_DB_PATH, exist_ok=True)

STORE_FILE = os.path.join(VECTOR_DB_PATH, "enterprise_rag.json")


def _load_store():
    if os.path.exists(STORE_FILE):
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_store(records):
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f)


# ---------------------------------
# PDF Reader
# ---------------------------------

def read_pdf(pdf_path: str):

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text + "\n"

    return text

# ---------------------------------
# Text Splitter
# ---------------------------------

def split_text(text: str):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunks.append(text[start:end])

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

# ---------------------------------
# Gemini Embeddings
# ---------------------------------

def create_embeddings(chunks):

    embeddings = []

    for chunk in chunks:

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=chunk,
        )

        vector = response.embeddings[0].values

        embeddings.append(
            {
                "id": str(uuid.uuid4()),
                "text": chunk,
                "embedding": vector,
            }
        )

    return embeddings


# ---------------------------------
# Store in Local Vector DB
# ---------------------------------

def create_vector_db(pdf_path: str):
        
    text = read_pdf(pdf_path)

    chunks = split_text(text)

    vectors = create_embeddings(chunks)

    # Fresh PDF replaces the old one, same as collection.delete + add.
    _save_store(vectors)

    return True


# ---------------------------------
# Search Context
# ---------------------------------

def search_context(question: str):

    records = _load_store()

    if not records:
        return ""

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=question,
    )

    query_embedding = np.array(response.embeddings[0].values)

    matrix = np.array([r["embedding"] for r in records])

    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)

    scores = matrix_norm @ query_norm

    top_indices = np.argsort(scores)[::-1][:TOP_K]

    documents = [records[i]["text"] for i in top_indices]

    context = "\n\n".join(documents)

    return context
