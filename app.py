"""
app.py
------
Streamlit UI for the PDF RAG Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv

from app.pdf_processor import load_pdf_from_bytes, split_documents
from app.vector_store import build_vector_store
from app.rag_chain import build_rag_chain, ask_question

# ── Load environment variables from .env ──────────────────────────────────────
load_dotenv()

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 10px; padding: 8px; }
    .source-box {
        background: #f0f2f6;
        border-left: 3px solid #f55036;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ───────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_... (or set in .env)",
        help="Free API key from https://console.groq.com — no credit card needed.",
    )
    groq_api_key = api_key_input or os.getenv("GROQ_API_KEY", "")

    if not groq_api_key:
        st.warning("⚠️ Get a **free** API key at [console.groq.com](https://console.groq.com)")

    st.divider()

    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to chat with it.",
    )

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF… this may take a moment."):
                try:
                    pdf_bytes = uploaded_file.read()
                    documents = load_pdf_from_bytes(pdf_bytes, uploaded_file.name)
                    chunks = split_documents(documents)

                    vector_store = build_vector_store(chunks)
                    st.session_state.rag_chain = build_rag_chain(
                        vector_store,
                        groq_api_key=groq_api_key,
                        model_name="llama-3.3-70b-versatile",
                    )
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(f"✅ Processed **{uploaded_file.name}** ({len(chunks)} chunks)")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    elif uploaded_file and not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key above.")

    st.divider()

    st.markdown(
        """
        **How it works:**
        1. Upload any PDF
        2. PDF is chunked & embedded locally
        3. Embeddings stored in FAISS
        4. Ask questions — relevant chunks retrieved and answered by Llama 3.3
        """
    )

    if st.session_state.pdf_name:
        st.info(f"📖 Active document: **{st.session_state.pdf_name}**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain["chat_history"].clear()
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────
st.title("📄 PDF Q&A Chatbot")
st.caption("Powered by LangChain · FAISS · Llama 3.3 via Groq (Free) · RAG")

if not st.session_state.rag_chain:
    st.info(
        "👈 **Get started:** Upload a PDF in the sidebar and click **Process PDF**.",
        icon="ℹ️",
    )
    st.stop()

# Display chat history
for role, message, sources in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
        if sources:
            with st.expander("📎 Source chunks", expanded=False):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i} — Page {page}</strong><br>{doc.page_content[:300]}…</div>',
                        unsafe_allow_html=True,
                    )

# Chat input
user_input = st.chat_input("Ask a question about your PDF…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = ask_question(st.session_state.rag_chain, user_input)
                answer = result["answer"]
                sources = result["source_documents"]
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []

        st.markdown(answer)

        if sources:
            with st.expander("📎 Source chunks", expanded=False):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i} — Page {page}</strong><br>{doc.page_content[:300]}…</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append(("user", user_input, []))
    st.session_state.chat_history.append(("assistant", answer, sources))
