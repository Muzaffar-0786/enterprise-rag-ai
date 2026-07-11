"""
app.py
Enterprise RAG AI Chatbot - Streamlit UI
"""

import os
import tempfile

import streamlit as st

from config import (
    APP_TITLE,
    PAGE_ICON,
    PAGE_LAYOUT,
    MAX_FILE_SIZE_MB,
    SUPPORTED_FILE_TYPES,
)
from gemini import ask_gemini
from rag import create_vector_db, search_context

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
)

st.title(APP_TITLE)

# ---------------------------------
# Session State
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# ---------------------------------
# Sidebar - PDF Upload
# ---------------------------------

with st.sidebar:

    st.header("📄 Upload PDF")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=SUPPORTED_FILE_TYPES,
    )

    if uploaded_file is not None:

        file_size_mb = uploaded_file.size / (1024 * 1024)

        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"⚠️ File too large. Max size is {MAX_FILE_SIZE_MB}MB.")
        else:
            if st.button("🚀 Process PDF"):

                with st.spinner("📚 Reading & indexing PDF..."):

                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".pdf"
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name

                        create_vector_db(tmp_path)

                        os.unlink(tmp_path)

                        st.session_state.pdf_ready = True
                        st.session_state.messages = []

                        st.success("✅ PDF processed! Ask your questions below.")

                    except Exception as e:
                        st.error(f"❌ Error processing PDF:\n\n{e}")

    if st.session_state.pdf_ready:
        st.info("✅ PDF is ready for questions.")
    else:
        st.warning("⚠️ Upload and process a PDF to start chatting.")

# ---------------------------------
# Chat History
# ---------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------
# Chat Input
# ---------------------------------

prompt = st.chat_input("Ask anything from your PDF...")

if prompt:

    if not st.session_state.pdf_ready:
        st.warning("⚠️ Please upload a PDF first.")
        st.stop()

    # User Message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant
    with st.chat_message("assistant"):

        with st.spinner("🤖 Thinking..."):

            try:

                context = search_context(prompt)

                answer = ask_gemini(
                    question=prompt,
                    context=context,
                )

            except Exception as e:

                answer = f"❌ Error:\n\n{e}"

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

# ---------------------------------
# Footer
# ---------------------------------

st.divider()

st.caption(
    "🚀 Enterprise RAG AI Chatbot | "
    "Google Gemini + Local Vector Search + Streamlit"
)
