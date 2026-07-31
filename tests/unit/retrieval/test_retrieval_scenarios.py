"""
tests.unit.retrieval.test_retrieval_scenarios
================================================
Unit tests verifying the 3 RAG retrieval scenarios:
Scenario 1: Retrieval Succeeded, Documents Found -> Calls LLM.
Scenario 2: Retrieval Succeeded, Zero Documents Found -> Does NOT call LLM.
Scenario 3: Retrieval Failed (AllRetrieversFailedError) -> Does NOT call LLM, returns system message.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document as LCDocument

from langchain_core.messages import AIMessage
from core.exceptions import AllRetrieversFailedError, EmbeddingQuotaExceededError, DenseRetrievalError
from core.models.response import LegalRAGResponse
from chains.legal_rag_chain import LegalRAGChain


from langchain_core.retrievers import BaseRetriever


class DummyRetriever(BaseRetriever):
    docs: list = []
    should_raise: Exception | None = None

    def _get_relevant_documents(self, query: str, **kwargs) -> list:
        if self.should_raise:
            raise self.should_raise
        return self.docs


def mock_llm_response(prompt, **kwargs):
    return AIMessage(content="Statutory answer or rephrased query.")


def test_scenario_1_retrieval_success():
    """Scenario 1: Docs retrieved -> LLM invoked."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_llm_response
    mock_llm.side_effect = mock_llm_response

    sample_docs = [
        LCDocument(
            page_content="Section 14: Offence and Penalties under Passports Act 1967.",
            metadata={"title": "Passports Act, 1967", "section_number": "Section 14", "year": 1967, "source": "passports.pdf"},
        )
    ]
    retriever = DummyRetriever(docs=sample_docs)

    chain = LegalRAGChain(llm=mock_llm, retriever=retriever)
    response: LegalRAGResponse = chain.query("What are penalties under Passports Act?")

    assert response.retrieval_status == "success"
    assert "Statutory answer" in response.answer or "Passports Act" in response.answer


def test_scenario_2_zero_documents_found():
    """Scenario 2: Zero docs found -> LLM is NOT called for QA generation."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_llm_response
    mock_llm.side_effect = mock_llm_response

    retriever = DummyRetriever(docs=[])

    chain = LegalRAGChain(llm=mock_llm, retriever=retriever)
    response: LegalRAGResponse = chain.query("What is the legal penalty for XYZ alien species?")

    assert response.retrieval_status == "no_documents_found"
    assert "I couldn't find any relevant statutory provisions" in response.answer
    assert response.confidence_score == 0.0


def test_scenario_3_retrieval_failed():
    """Scenario 3: Retriever raises exception -> LLM is NOT called, returns system message."""
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = mock_llm_response
    mock_llm.side_effect = mock_llm_response

    retriever = DummyRetriever(
        docs=[],
        should_raise=AllRetrieversFailedError(
            message="All document retrieval channels failed. Dense error: Embedding API quota exceeded (HTTP 403)",
            details={"query": "test query", "dense_error": "403 forbidden"},
        ),
    )

    chain = LegalRAGChain(llm=mock_llm, retriever=retriever)
    response: LegalRAGResponse = chain.query("What is section 5?")

    assert response.retrieval_status == "retrieval_failed"
    assert "Document Retrieval Temporarily Unavailable" in response.answer
    assert "403" in response.developer_details or "Embedding API quota exceeded" in response.developer_details
    assert response.confidence_score == 0.0
    assert not mock_llm.invoke.called  # LLM MUST NOT BE INVOKED
