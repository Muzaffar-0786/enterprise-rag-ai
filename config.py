"""
config.py
Enterprise RAG AI Chatbot Configuration
"""

import os

# ============================
# Gemini API (Replit Secrets)
# ============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY नहीं मिला। Replit Secrets में जोड़ें।"
    )

# ============================
# Gemini Model
# ============================
MODEL_NAME = "gemini-2.5-flash"

# ============================
# Embedding Model
# ============================
EMBEDDING_MODEL = "models/gemini-embedding-001"

# ============================
# RAG Settings
# ============================
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4

# ============================
# Vector DB (local, file-based)
# ============================
CHROMA_DB_DIR = "chroma_db"
VECTOR_DB_PATH = CHROMA_DB_DIR

# ============================
# Streamlit
# ============================
APP_TITLE = "🤖 Enterprise RAG AI Chatbot"
PAGE_ICON = "🤖"
PAGE_LAYOUT = "wide"

# ============================
# File Upload
# ============================
MAX_FILE_SIZE_MB = 25

SUPPORTED_FILE_TYPES = [
    "pdf"
]
