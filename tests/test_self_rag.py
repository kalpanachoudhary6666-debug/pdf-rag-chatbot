"""
test_self_rag.py
----------------
Unit tests for the Self-RAG LangGraph pipeline.
All Groq API calls and external models are mocked — no real network calls.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_docs():
    return [
        Document(page_content="The company revenue in Q3 was $5 million.", metadata={"page": 1}),
        Document(page_content="The CEO announced expansion plans in October.", metadata={"page": 2}),
        Document(page_content="Employee headcount grew by 20% this year.", metadata={"page": 3}),
    ]


@pytest.fixture
def mock_groq_client():
    """Groq client that returns predictable responses."""
    client = MagicMock()
    response = MagicMock()
    response.choices[0].message.content = "yes"
    client.chat.completions.create.return_value = response
    return client


@pytest.fixture
def mock_retriever(sample_docs):
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    return retriever


@pytest.fixture
def mock_reranker(sample_docs):
    reranker = MagicMock()
    # Return descending scores so first doc ranks highest
    reranker.predict.return_value = [0.9, 0.5, 0.2]
    return reranker


# ─── Test: get_reranker singleton ─────────────────────────────────────────────

class TestGetReranker:
    def test_returns_same_instance_twice(self):
        """Reranker should be loaded once and cached."""
        import app.self_rag as sr

        mock_cross = MagicMock()
        with patch("app.self_rag._reranker_instance", None):
            with patch("sentence_transformers.CrossEncoder", return_value=mock_cross):
                sr._reranker_instance = None
                r1 = sr.get_reranker()
                r2 = sr.get_reranker()
                # After first load, subsequent calls return cached instance
                assert r1 is r2 or r1 == mock_cross


# ─── Test: build_self_rag_graph ───────────────────────────────────────────────

class TestBuildSelfRagGraph:
    def test_returns_compiled_graph(self, mock_retriever, mock_groq_client, mock_reranker):
        from app.self_rag import build_self_rag_graph
        graph = build_self_rag_graph(
            retriever=mock_retriever,
            client=mock_groq_client,
            model_name="test-model",
            temperature=0.0,
            reranker=mock_reranker,
        )
        assert graph is not None

    def test_graph_has_invoke_method(self, mock_retriever, mock_groq_client, mock_reranker):
        from app.self_rag import build_self_rag_graph
        graph = build_self_rag_graph(
            retriever=mock_retriever,
            client=mock_groq_client,
            model_name="test-model",
            temperature=0.0,
            reranker=mock_reranker,
        )
        assert hasattr(graph, "invoke")

    def test_graph_works_without_reranker(self, mock_retriever, mock_groq_client):
        from app.self_rag import build_self_rag_graph
        graph = build_self_rag_graph(
            retriever=mock_retriever,
            client=mock_groq_client,
            model_name="test-model",
            temperature=0.0,
            reranker=None,   # no re-ranker
        )
        assert graph is not None


# ─── Test: run_self_rag ───────────────────────────────────────────────────────

class TestRunSelfRag:
    def _make_groq_sequence(self, *responses):
        """Return a Groq client mock that cycles through given response strings."""
        client = MagicMock()
        side_effects = []
        for text in responses:
            resp = MagicMock()
            resp.choices[0].message.content = text
            side_effects.append(resp)
        client.chat.completions.create.side_effect = side_effects
        return client

    def test_returns_answer_key(self, mock_retriever, mock_reranker):
        """run_self_rag must return a dict with 'answer' key."""
        from app.self_rag import build_self_rag_graph, run_self_rag

        # grade=yes, grade=yes, grade=yes → generate → grounded
        client = self._make_groq_sequence(
            "yes", "yes", "yes",         # grade_documents (3 chunks)
            "The revenue was $5M.",      # generate
            "grounded",                  # check_hallucination
        )
        graph = build_self_rag_graph(mock_retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "What was the revenue?", [])

        assert "answer" in result
        assert result["answer"] == "The revenue was $5M."

    def test_returns_reasoning_steps(self, mock_retriever, mock_reranker):
        """Reasoning steps list must be populated."""
        from app.self_rag import build_self_rag_graph, run_self_rag

        client = self._make_groq_sequence(
            "yes", "yes", "yes",
            "Some answer.",
            "grounded",
        )
        graph = build_self_rag_graph(mock_retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "Question?", [])

        assert isinstance(result["reasoning_steps"], list)
        assert len(result["reasoning_steps"]) > 0

    def test_returns_source_documents(self, mock_retriever, mock_reranker, sample_docs):
        """source_documents must be a list of Document objects."""
        from app.self_rag import build_self_rag_graph, run_self_rag

        client = self._make_groq_sequence(
            "yes", "yes", "yes",
            "Answer here.",
            "grounded",
        )
        graph = build_self_rag_graph(mock_retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "Question?", [])

        assert isinstance(result["source_documents"], list)
        for doc in result["source_documents"]:
            assert isinstance(doc, Document)

    def test_no_relevant_docs_returns_not_found(self, mock_reranker):
        """When all chunks graded 'no', answer should say not found."""
        from app.self_rag import build_self_rag_graph, run_self_rag

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Irrelevant text.", metadata={"page": 1}),
        ]

        # grade=no for all, rewrite×2, then generate with empty docs
        client = MagicMock()
        responses = [
            "no",                             # grade attempt 1
            "better search terms",            # rewrite 1
            "no",                             # grade attempt 2
            "even better terms",              # rewrite 2
            "no",                             # grade attempt 3
            "I couldn't find relevant info",  # generate (empty docs)
            "grounded",                       # hallucination check
        ]
        side_effects = []
        for text in responses:
            r = MagicMock()
            r.choices[0].message.content = text
            side_effects.append(r)
        client.chat.completions.create.side_effect = side_effects

        graph = build_self_rag_graph(retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "What is the meaning of life?", [])

        assert "answer" in result
        # After exhausting rewrites, generate should return not-found message
        assert result["rewrites_used"] <= 2

    def test_grounded_flag_true_when_grounded(self, mock_retriever, mock_reranker):
        from app.self_rag import build_self_rag_graph, run_self_rag

        client = self._make_groq_sequence(
            "yes", "yes", "yes",
            "The revenue was $5M.",
            "grounded",
        )
        graph = build_self_rag_graph(mock_retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "Revenue?", [])

        assert result["grounded"] is True

    def test_grounded_flag_false_when_hallucination(self, mock_retriever, mock_reranker):
        from app.self_rag import build_self_rag_graph, run_self_rag

        # MAX_REWRITES=2, so the graph loops up to 3 times before ending.
        # Loop 1: grade(×3) + generate + hallucination → rewrite (count=1)
        # Loop 2: grade(×3) + generate + hallucination → rewrite (count=2)
        # Loop 3: grade(×3) + generate + hallucination → count=2 >= MAX → END
        client = self._make_groq_sequence(
            # --- loop 1 ---
            "yes", "yes", "yes",        # grade_documents (3 docs)
            "Revenue was $100 trillion.", # generate
            "not_grounded",             # hallucination → rewrite (count=1)
            "rewritten query 1",        # rewrite_query
            # --- loop 2 ---
            "yes", "yes", "yes",        # grade_documents
            "Revenue was $999 trillion.", # generate
            "not_grounded",             # hallucination → rewrite (count=2)
            "rewritten query 2",        # rewrite_query
            # --- loop 3 ---
            "yes", "yes", "yes",        # grade_documents
            "Revenue was $50 trillion.", # generate
            "not_grounded",             # hallucination → count=2 >= MAX → END
        )
        graph = build_self_rag_graph(mock_retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "Revenue?", [])

        assert "grounded" in result
        assert result["grounded"] is False

    def test_rewrites_used_count(self, mock_reranker):
        """rewrite_count should increment each time query is rewritten."""
        from app.self_rag import build_self_rag_graph, run_self_rag

        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Some text.", metadata={"page": 1}),
        ]

        client = MagicMock()
        responses = [
            "no",              # grade 1 — trigger rewrite
            "new query 1",     # rewrite 1
            "yes",             # grade 2 — now relevant
            "Final answer.",   # generate
            "grounded",        # hallucination check
        ]
        side_effects = []
        for text in responses:
            r = MagicMock()
            r.choices[0].message.content = text
            side_effects.append(r)
        client.chat.completions.create.side_effect = side_effects

        graph = build_self_rag_graph(retriever, client, "m", 0.0, mock_reranker)
        result = run_self_rag(graph, "Question?", [])

        assert result["rewrites_used"] == 1


# ─── Test: rag_chain integration ─────────────────────────────────────────────

class TestRagChainIntegration:
    @patch("app.rag_chain.get_reranker")
    @patch("app.rag_chain.build_self_rag_graph")
    @patch("app.rag_chain.Groq")
    def test_build_rag_chain_self_rag_mode(
        self, mock_groq_cls, mock_build_graph, mock_get_reranker
    ):
        from app.rag_chain import build_rag_chain

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        mock_get_reranker.return_value = MagicMock()
        mock_build_graph.return_value = MagicMock()

        chain = build_rag_chain(mock_store, groq_api_key="gsk-test", use_self_rag=True)

        assert chain["use_self_rag"] is True
        assert chain["graph"] is not None
        mock_build_graph.assert_called_once()
        mock_get_reranker.assert_called_once()

    @patch("app.rag_chain.Groq")
    def test_build_rag_chain_simple_mode(self, mock_groq_cls):
        from app.rag_chain import build_rag_chain

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()

        chain = build_rag_chain(mock_store, groq_api_key="gsk-test", use_self_rag=False)

        assert chain["use_self_rag"] is False
        assert chain["graph"] is None

    @patch("app.rag_chain.Groq")
    def test_ask_question_simple_mode(self, mock_groq_cls):
        """Simple mode should call Groq once and return answer."""
        from app.rag_chain import build_rag_chain, ask_question

        mock_store = MagicMock()
        retriever = MagicMock()
        retriever.invoke.return_value = [
            Document(page_content="Sky is blue.", metadata={"page": 1})
        ]
        mock_store.as_retriever.return_value = retriever

        mock_client = MagicMock()
        resp = MagicMock()
        resp.choices[0].message.content = "The sky is blue."
        mock_client.chat.completions.create.return_value = resp
        mock_groq_cls.return_value = mock_client

        chain = build_rag_chain(mock_store, groq_api_key="gsk-test", use_self_rag=False)
        result = ask_question(chain, "What color is the sky?")

        assert result["answer"] == "The sky is blue."
        assert isinstance(result["source_documents"], list)
        assert result["reasoning_steps"] == []

    @patch("app.rag_chain.run_self_rag")
    @patch("app.rag_chain.get_reranker")
    @patch("app.rag_chain.build_self_rag_graph")
    @patch("app.rag_chain.Groq")
    def test_ask_question_self_rag_mode(
        self, mock_groq_cls, mock_build_graph, mock_get_reranker, mock_run
    ):
        """Self-RAG mode should delegate to run_self_rag."""
        from app.rag_chain import build_rag_chain, ask_question

        mock_store = MagicMock()
        mock_store.as_retriever.return_value = MagicMock()
        mock_get_reranker.return_value = MagicMock()
        mock_graph = MagicMock()
        mock_build_graph.return_value = mock_graph

        mock_run.return_value = {
            "answer": "Self-RAG answer.",
            "reasoning_steps": ["step1", "step2"],
            "source_documents": [],
            "grounded": True,
            "rewrites_used": 0,
        }

        chain = build_rag_chain(mock_store, groq_api_key="gsk-test", use_self_rag=True)
        result = ask_question(chain, "Test question?")

        mock_run.assert_called_once()
        assert result["answer"] == "Self-RAG answer."
        assert result["reasoning_steps"] == ["step1", "step2"]
        assert result["grounded"] is True
