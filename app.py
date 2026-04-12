"""
app.py
------
Streamlit UI for the PDF RAG Chatbot — Self-RAG edition.
Run with: streamlit run app.py

New features in this version:
  • Full Self-RAG LangGraph pipeline (grade → rewrite → generate → hallucination check)
  • CrossEncoder re-ranking (keeps only the best chunks)
  • Agent reasoning panel shows every decision the AI made
  • Simple RAG toggle in sidebar (fewer Groq calls, faster)
"""

import streamlit as st
import os
from dotenv import load_dotenv

from app.pdf_processor import load_pdf_from_bytes, split_documents
from app.vector_store import build_vector_store
from app.rag_chain import build_rag_chain, ask_question

load_dotenv()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Q&A Chatbot — Self-RAG",
    page_icon="🤖",
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
    .reasoning-box {
        background: #eef6ff;
        border-left: 3px solid #2196F3;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-top: 4px;
    }
    .badge-grounded {
        background: #d4edda;
        color: #155724;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-warning {
        background: #fff3cd;
        color: #856404;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []   # list of (role, message, sources, steps, grounded)
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

    # Self-RAG toggle
    use_self_rag = st.toggle(
        "🤖 Self-RAG Mode (recommended)",
        value=True,
        help=(
            "ON: Full LangGraph pipeline — grades chunks, rewrites bad queries, "
            "checks for hallucinations. Uses ~3–5 Groq calls per question.\n\n"
            "OFF: Simple RAG — single Groq call, faster but less accurate."
        ),
    )

    if use_self_rag:
        st.success("✅ Self-RAG active: re-ranking + grading + hallucination check")
    else:
        st.info("ℹ️ Simple RAG mode — fast, 1 Groq call per question")

    st.divider()

    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to chat with it.",
    )

    if uploaded_file and groq_api_key:
        if st.button("🚀 Process PDF", use_container_width=True, type="primary"):
            with st.spinner("Processing PDF… building vector store…"):
                try:
                    pdf_bytes = uploaded_file.read()
                    documents = load_pdf_from_bytes(pdf_bytes, uploaded_file.name)
                    chunks = split_documents(documents)
                    vector_store = build_vector_store(chunks)

                    with st.spinner(
                        "Loading re-ranker model (first time ~30 sec)…"
                        if use_self_rag
                        else "Building chain…"
                    ):
                        st.session_state.rag_chain = build_rag_chain(
                            vector_store,
                            groq_api_key=groq_api_key,
                            model_name="llama-3.3-70b-versatile",
                            use_self_rag=use_self_rag,
                        )

                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.chat_history = []
                    st.success(
                        f"✅ Processed **{uploaded_file.name}** "
                        f"({len(chunks)} chunks)"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {e}")

    elif uploaded_file and not groq_api_key:
        st.warning("⚠️ Please enter your Groq API key above.")

    st.divider()

    st.markdown(
        """
        **How Self-RAG works:**
        1. Upload PDF → chunks embedded locally (no API)
        2. Question → FAISS retrieves top 6 chunks
        3. CrossEncoder **re-ranks** → keeps best 3
        4. Groq **grades** each chunk — filters irrelevant ones
        5. If no good chunks → **rewrites** query and retries (×2)
        6. Groq **generates** answer from only relevant chunks
        7. Groq **checks hallucination** — flags unsupported claims
        """
    )

    if st.session_state.pdf_name:
        st.info(f"📖 Active: **{st.session_state.pdf_name}**")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.rag_chain:
            st.session_state.rag_chain["chat_history"].clear()
        st.rerun()


# ── Main area ──────────────────────────────────────────────────────────────────
st.title("🤖 PDF Q&A Chatbot — Self-RAG")
st.caption(
    "Powered by LangGraph · CrossEncoder Re-ranking · Groq Llama 3.3 (Free) · FAISS · sentence-transformers"
)

if not st.session_state.rag_chain:
    st.info(
        "👈 **Get started:** Upload a PDF in the sidebar and click **Process PDF**.",
        icon="ℹ️",
    )

    # Architecture diagram
    with st.expander("🏗️ How Self-RAG works under the hood", expanded=False):
        st.markdown(
            """
```
User Question
      │
      ▼
  RETRIEVE (FAISS, top-6 chunks)
      │
      ▼
  RE-RANK (CrossEncoder scores each chunk)
  → keeps top 3 by true relevance
      │
      ▼
  GRADE (Groq asks: is each chunk relevant?)
  → filters out off-topic chunks
      │
      ├─ relevant chunks found ──────────────────────┐
      │                                              │
      └─ no chunks + rewrites left ──► REWRITE       │
                                       QUERY         │
                                          │          │
                                          └─► RETRIEVE (retry)
                                                     │
                                                     ▼
                                              GENERATE ANSWER
                                                     │
                                                     ▼
                                          HALLUCINATION CHECK
                                          (is answer grounded?)
                                                     │
                                                    END
```
            """
        )
    st.stop()


# ── Display chat history ───────────────────────────────────────────────────────
for role, message, sources, steps, grounded in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

        if role == "assistant":
            # Grounding badge
            if grounded:
                st.markdown(
                    '<span class="badge-grounded">✅ Grounded in document</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge-warning">⚠️ May contain unsupported claims</span>',
                    unsafe_allow_html=True,
                )

        # Agent reasoning steps
        if steps:
            with st.expander("🤖 Agent reasoning trace", expanded=False):
                for step in steps:
                    st.markdown(
                        f'<div class="reasoning-box">{step}</div>',
                        unsafe_allow_html=True,
                    )

        # Source chunks
        if sources:
            with st.expander("📎 Source chunks used", expanded=False):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<div class="source-box">'
                        f"<strong>Chunk {i} — Page {page}</strong><br>"
                        f"{doc.page_content[:300]}…"
                        f"</div>",
                        unsafe_allow_html=True,
                    )


# ── Chat input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your PDF…")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        is_self_rag = st.session_state.rag_chain.get("use_self_rag", False)
        spinner_msg = (
            "🤖 Self-RAG agent thinking… (retrieve → rerank → grade → generate → verify)"
            if is_self_rag
            else "Thinking…"
        )

        with st.spinner(spinner_msg):
            try:
                result = ask_question(st.session_state.rag_chain, user_input)
                answer = result["answer"]
                sources = result["source_documents"]
                steps = result.get("reasoning_steps", [])
                grounded = result.get("grounded", True)
                rewrites = result.get("rewrites_used", 0)
            except Exception as e:
                answer = f"❌ Error: {e}"
                sources = []
                steps = []
                grounded = True
                rewrites = 0

        st.markdown(answer)

        # Show rewrite count if any happened
        if rewrites > 0:
            st.caption(f"🔄 Query was rewritten {rewrites} time(s) to find better results")

        # Grounding badge
        if grounded:
            st.markdown(
                '<span class="badge-grounded">✅ Grounded in document</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge-warning">⚠️ May contain unsupported claims</span>',
                unsafe_allow_html=True,
            )

        # Agent reasoning trace
        if steps:
            with st.expander("🤖 Agent reasoning trace", expanded=True):
                for step in steps:
                    st.markdown(
                        f'<div class="reasoning-box">{step}</div>',
                        unsafe_allow_html=True,
                    )

        # Source chunks
        if sources:
            with st.expander("📎 Source chunks used", expanded=False):
                for i, doc in enumerate(sources, 1):
                    page = doc.metadata.get("page", "?")
                    st.markdown(
                        f'<div class="source-box">'
                        f"<strong>Chunk {i} — Page {page}</strong><br>"
                        f"{doc.page_content[:300]}…"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

    st.session_state.chat_history.append(("user", user_input, [], [], True))
    st.session_state.chat_history.append(
        ("assistant", answer, sources, steps, grounded)
    )
