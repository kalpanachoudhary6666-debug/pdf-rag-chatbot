"""
self_rag.py
-----------
Full Self-RAG pipeline using LangGraph with CrossEncoder re-ranking.

Graph flow:
  retrieve → rerank → grade_documents → (conditional)
      ├── relevant docs found      → generate → check_hallucination → END
      ├── no docs + rewrites left  → rewrite_query → retrieve (loop)
      └── no docs + no rewrites    → generate (returns "not found") → END

Each node appends a human-readable step to state["reasoning_steps"]
so the Streamlit UI can show the agent's thinking process live.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from groq import Groq
from langchain_core.documents import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

MAX_REWRITES = 2


# ─── State schema ─────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    """Shared state passed between every node in the graph."""
    question: str                  # original user question (never changes)
    rewritten_question: str        # updated by rewrite_query node (empty = not rewritten)
    chat_history: List[Dict]       # conversation memory list
    documents: List[Document]      # chunks retrieved + filtered
    generation: str                # final LLM answer
    rewrite_count: int             # how many rewrites have been done
    grounded: bool                 # did hallucination check pass?
    reasoning_steps: List[str]     # human-readable agent trace for UI


# ─── Reranker singleton ───────────────────────────────────────────────────────

_reranker_instance = None


def get_reranker():
    """
    Load CrossEncoder once and cache it.
    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Runs locally on CPU, no API key needed
    - ~80MB download on first use, then cached
    - Re-scores retrieved chunks by true relevance to the question
    """
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder
        _reranker_instance = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
    return _reranker_instance


# ─── Build the Self-RAG graph ─────────────────────────────────────────────────

def build_self_rag_graph(
    retriever,
    client: Groq,
    model_name: str,
    temperature: float,
    reranker=None,
):
    """
    Compile and return a LangGraph Self-RAG graph.

    Args:
        retriever:    FAISS retriever (from vector_store.as_retriever)
        client:       Groq client instance
        model_name:   Groq model to call for grading / generating / rewriting
        temperature:  LLM temperature for final answer generation
        reranker:     CrossEncoder instance (optional — skipped if None)

    Returns:
        Compiled LangGraph that accepts RAGState and returns final RAGState.
    """

    # ── Node: retrieve ────────────────────────────────────────────────────────
    def node_retrieve(state: RAGState) -> RAGState:
        """Fetch top-6 chunks from FAISS using current question (or rewritten)."""
        question = state["rewritten_question"] or state["question"]
        steps = list(state["reasoning_steps"])
        steps.append(f"🔍 **Retrieving** chunks for: *\"{question}\"*")

        docs = retriever.invoke(question)
        return {**state, "documents": docs, "reasoning_steps": steps}

    # ── Node: rerank ──────────────────────────────────────────────────────────
    def node_rerank(state: RAGState) -> RAGState:
        """
        Re-rank retrieved chunks using CrossEncoder.
        CrossEncoder scores each (question, chunk) pair directly —
        much more accurate than cosine similarity alone.
        Keeps only the top-3 highest-scoring chunks.
        """
        steps = list(state["reasoning_steps"])
        docs = state["documents"]

        if not docs:
            steps.append("📄 **No chunks** to re-rank")
            return {**state, "reasoning_steps": steps}

        if reranker is None:
            steps.append(f"📄 **Retrieved** {len(docs)} chunks (re-ranking disabled)")
            return {**state, "reasoning_steps": steps}

        question = state["rewritten_question"] or state["question"]
        pairs = [(question, doc.page_content) for doc in docs]
        scores = reranker.predict(pairs)

        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in ranked[:3]]

        steps.append(
            f"📊 **Re-ranked** {len(docs)} chunks → kept top **{len(top_docs)}** "
            f"(scores: {[round(float(s), 2) for s, _ in ranked[:3]]})"
        )
        return {**state, "documents": top_docs, "reasoning_steps": steps}

    # ── Node: grade_documents ─────────────────────────────────────────────────
    def node_grade_documents(state: RAGState) -> RAGState:
        """
        Ask Groq to grade each chunk as relevant / not relevant.
        Only keeps chunks that are truly useful for answering the question.
        This filters out FAISS false-positives (similar but off-topic chunks).
        """
        question = state["question"]
        docs = state["documents"]
        steps = list(state["reasoning_steps"])

        relevant = []
        for doc in docs:
            prompt = (
                "You are a strict relevance grader.\n"
                "Is this document chunk useful for answering the question?\n\n"
                f"Question: {question}\n\n"
                f"Chunk: {doc.page_content[:500]}\n\n"
                "Reply with ONLY one word: yes or no"
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )
            grade = resp.choices[0].message.content.strip().lower()
            if grade.startswith("yes"):
                relevant.append(doc)

        steps.append(
            f"✅ **Graded** {len(docs)} chunks → **{len(relevant)} relevant** "
            f"/ {len(docs) - len(relevant)} filtered out"
        )
        return {**state, "documents": relevant, "reasoning_steps": steps}

    # ── Node: rewrite_query ───────────────────────────────────────────────────
    def node_rewrite_query(state: RAGState) -> RAGState:
        """
        When retrieval fails, rewrite the question with better keywords.
        Example: "profit Q3" → "quarterly financial results third quarter earnings"
        Loops back to retrieve with the new query.
        """
        question = state["question"]
        rewrite_count = state["rewrite_count"]
        steps = list(state["reasoning_steps"])

        prompt = (
            f"The search query below did not find relevant document chunks.\n"
            f"Rewrite it using different vocabulary that might match the document better.\n\n"
            f"Original query: {question}\n\n"
            "Rewritten query (reply with ONLY the query, nothing else):"
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=80,
        )
        rewritten = resp.choices[0].message.content.strip()
        new_count = rewrite_count + 1

        steps.append(
            f"🔄 **Query rewrite {new_count}/{MAX_REWRITES}**: "
            f"*\"{question}\"* → *\"{rewritten}\"*"
        )
        return {
            **state,
            "rewritten_question": rewritten,
            "rewrite_count": new_count,
            "reasoning_steps": steps,
        }

    # ── Node: generate ────────────────────────────────────────────────────────
    def node_generate(state: RAGState) -> RAGState:
        """
        Generate final answer using only the graded, relevant chunks.
        If no relevant chunks exist, returns a safe "not found" message
        instead of hallucinating.
        """
        question = state["question"]
        docs = state["documents"]
        chat_history = state["chat_history"]
        steps = list(state["reasoning_steps"])

        if not docs:
            steps.append("❌ **No relevant chunks** — returning 'not found' message")
            return {
                **state,
                "generation": (
                    "I couldn't find relevant information in your PDF to answer this question. "
                    "Try rephrasing or check if your PDF covers this topic."
                ),
                "grounded": True,
                "reasoning_steps": steps,
            }

        context = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )

        history_str = ""
        for msg in chat_history:
            prefix = "Human" if msg["role"] == "human" else "Assistant"
            history_str += f"{prefix}: {msg['content']}\n"

        prompt = (
            "You are a helpful assistant. Answer the question using ONLY the context below.\n"
            "If the context doesn't fully answer the question, say so honestly.\n"
            "Do NOT use any knowledge outside the provided context.\n\n"
            f"Context from PDF:\n{context}\n\n"
            f"Chat History:\n{history_str}\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=2048,
        )
        generation = resp.choices[0].message.content
        steps.append(f"💡 **Generated answer** using {len(docs)} relevant chunks")

        return {**state, "generation": generation, "reasoning_steps": steps}

    # ── Node: check_hallucination ─────────────────────────────────────────────
    def node_check_hallucination(state: RAGState) -> RAGState:
        """
        Verify the generated answer is grounded in the retrieved chunks.
        Catches cases where the LLM added information not present in context.
        If not grounded and rewrites remain, loops back to try again.
        """
        docs = state["documents"]
        generation = state.get("generation", "")
        steps = list(state["reasoning_steps"])

        if not docs or not generation:
            return {**state, "grounded": True, "reasoning_steps": steps}

        context = "\n\n".join(doc.page_content[:400] for doc in docs)

        prompt = (
            "You are a hallucination detector.\n"
            "Does the answer below contain ONLY information that is supported by the context?\n\n"
            f"Context: {context}\n\n"
            f"Answer: {generation}\n\n"
            "Reply with ONLY: grounded  or  not_grounded"
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        result = resp.choices[0].message.content.strip().lower()
        grounded = "grounded" in result and "not_grounded" not in result

        if grounded:
            steps.append("🛡️ **Hallucination check passed** — answer is grounded in document")
        else:
            steps.append(
                "⚠️ **Hallucination detected** — answer may contain unsupported claims"
            )

        return {**state, "grounded": grounded, "reasoning_steps": steps}

    # ── Conditional routing ───────────────────────────────────────────────────

    def route_after_grading(state: RAGState) -> str:
        """After grading: go to generate if any relevant docs, else rewrite or give up."""
        if state["documents"]:
            return "generate"
        if state["rewrite_count"] < MAX_REWRITES:
            return "rewrite_query"
        return "generate"   # generate handles empty docs → "not found"

    def route_after_hallucination(state: RAGState) -> str:
        """After hallucination check: done if grounded, else rewrite if attempts left."""
        if state.get("grounded", True):
            return "end"
        if state["rewrite_count"] < MAX_REWRITES:
            return "rewrite_query"
        return "end"

    # ── Assemble graph ────────────────────────────────────────────────────────

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", node_retrieve)
    graph.add_node("rerank", node_rerank)
    graph.add_node("grade_documents", node_grade_documents)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generate", node_generate)
    graph.add_node("check_hallucination", node_check_hallucination)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade_documents")

    graph.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"},
    )

    graph.add_edge("rewrite_query", "retrieve")   # loop back for retry
    graph.add_edge("generate", "check_hallucination")

    graph.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {"end": END, "rewrite_query": "rewrite_query"},
    )

    return graph.compile()


# ─── Helper: run the compiled graph ───────────────────────────────────────────

def run_self_rag(
    compiled_graph,
    question: str,
    chat_history: List[Dict],
) -> Dict:
    """
    Invoke the compiled Self-RAG graph and return a clean result dict.

    Args:
        compiled_graph: Output of build_self_rag_graph(...)
        question:       User's question
        chat_history:   List of {"role": "human"/"assistant", "content": "..."}

    Returns:
        {
            "answer":          str,
            "reasoning_steps": List[str],   # for UI display
            "source_documents": List[Document],
            "grounded":        bool,
            "rewrites_used":   int,
        }
    """
    initial_state: RAGState = {
        "question": question,
        "rewritten_question": "",
        "chat_history": chat_history,
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
        "grounded": True,
        "reasoning_steps": [],
    }

    final_state = compiled_graph.invoke(initial_state)

    return {
        "answer": final_state["generation"],
        "reasoning_steps": final_state["reasoning_steps"],
        "source_documents": final_state["documents"],
        "grounded": final_state.get("grounded", True),
        "rewrites_used": final_state.get("rewrite_count", 0),
    }
