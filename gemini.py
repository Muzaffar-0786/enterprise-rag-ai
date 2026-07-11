"""
gemini.py
Gemini AI Service
"""

from google import genai
from google.genai.types import GenerateContentConfig

from config import (
    GEMINI_API_KEY,
    MODEL_NAME,
)

# ----------------------------------
# Gemini Client
# ----------------------------------

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# ----------------------------------
# AI Function
# ----------------------------------

def ask_gemini(question: str, context: str) -> str:
    """
    PDF Context के आधार पर उत्तर देता है।
    """

    prompt = f"""
You are an Enterprise AI Assistant.

Answer ONLY using the given Context.

==========================
Context
==========================

{context}

==========================
Question
==========================

{question}

Rules:

1. Answer only from Context.
2. If answer is missing say:
   "Answer not found in uploaded PDF."
3. Never make up information.
4. Keep answer short and clear.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    return response.text.strip()
