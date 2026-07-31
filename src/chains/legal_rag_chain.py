"""
chains.legal_rag_chain
======================
Production LangChain LCEL Legal RAG Chain Assembly.

Assembles:
1. History Aware Retriever
2. Hybrid Retriever + Compression Retriever (BGE Reranker)
3. Statutory Prompt Template
4. LLM (OpenAI / Anthropic / Gemini / Groq / Ollama)
5. Citation Formatter & Grounding Parser
6. Conversation Memory Management
"""

from __future__ import annotations

import logging, warnings
from typing import Any, Sequence

warnings.filterwarnings("ignore", message=".*RunnableWithMessageHistory.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*KeyError.*")

logging.getLogger("langchain_core.tracers").setLevel(logging.CRITICAL)
logging.getLogger("langchain_core.tracers.root_listeners").setLevel(logging.CRITICAL)
logging.getLogger("langchain_core.callbacks").setLevel(logging.CRITICAL)

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory

from chains.citation_formatter import format_docs_for_prompt
from chains.history_aware_retriever import create_history_aware_legal_retriever
from chains.memory import get_session_history
from chains.output_parser import LegalRAGOutputParser
from chains.prompts import LEGAL_QA_PROMPT
from core.models.response import LegalRAGResponse
from retrieval.compression_retriever import LegalCompressionRetriever
from services.chat.llm_factory import get_llm

logger = logging.getLogger(__name__)


class LegalRAGChain:
    """
    End-to-End Statutory Legal RAG Pipeline using LangChain LCEL patterns.

    Args:
        retriever: LCEL BaseRetriever (defaults to LegalCompressionRetriever).
        llm: LCEL BaseChatModel (defaults to configured LLM via get_llm()).
    """

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.retriever = retriever or LegalCompressionRetriever()
        self.llm = llm or get_llm()
        self._chain: Runnable | None = None
        self._with_memory: RunnableWithMessageHistory | None = None
        self._build_chain()

    def _build_chain(self) -> None:
        """Construct the modular LCEL pipeline."""
        # 1. Create history-aware retriever
        history_aware_retriever = create_history_aware_legal_retriever(
            llm=self.llm,
            retriever=self.retriever,
        )

        # 2. Define document retriever step in LCEL
        def process_retrieval(input_dict: dict[str, Any]) -> dict[str, Any]:
            question = input_dict.get("input") or input_dict.get("question") or ""
            chat_history = input_dict.get("chat_history") or []
            filters = input_dict.get("filters")

            docs: list[LCDocument] = []
            retrieval_failed = False
            error_exc: Exception | None = None

            # Execute history-aware retrieval
            try:
                docs = history_aware_retriever.invoke(
                    {"input": question, "chat_history": chat_history, "filters": filters},
                    config={"callbacks": []},
                )
            except Exception as exc:
                retrieval_failed = True
                error_exc = exc
                logger.error("Legal RAG retrieval pipeline failed for query %r: %s", question[:50], exc, exc_info=True)

            # SCENARIO 3: Retrieval Failed (Do NOT call LLM)
            if retrieval_failed or error_exc is not None:
                return {
                    "input": question,
                    "output": "",
                    "chat_history": chat_history,
                    "retrieval_status": "retrieval_failed",
                    "retrieved_docs": [],
                    "answer_text": "",
                    "error_exc": error_exc,
                }

            # SCENARIO 2: Retrieval Succeeded, Zero Documents Found (Do NOT call LLM)
            if not docs:
                no_docs_msg = (
                    "I couldn't find any relevant statutory provisions in the indexed legal documents "
                    "for this query. Consider broadening the search or uploading additional statutory material."
                )
                return {
                    "input": question,
                    "output": no_docs_msg,
                    "chat_history": chat_history,
                    "retrieval_status": "no_documents_found",
                    "retrieved_docs": [],
                    "answer_text": no_docs_msg,
                }

            # SCENARIO 1: Retrieval Succeeded & Relevant Documents Found (Invoke LLM)
            formatted_context = format_docs_for_prompt(docs)
            prompt_value = LEGAL_QA_PROMPT.invoke(
                {"input": question, "chat_history": chat_history, "context": formatted_context},
                config={"callbacks": []},
            )
            llm_res = self.llm.invoke(prompt_value, config={"callbacks": []})
            answer_text = llm_res.content if hasattr(llm_res, "content") else str(llm_res)

            return {
                "input": question,
                "output": answer_text,
                "chat_history": chat_history,
                "retrieval_status": "success",
                "context": formatted_context,
                "retrieved_docs": docs,
                "answer_text": answer_text,
            }

        self._chain = RunnableLambda(process_retrieval)

        # 4. Attach conversation memory wrapper
        self._with_memory = RunnableWithMessageHistory(
            runnable=self._chain,
            get_session_history=get_session_history,
            input_messages_key="input",
            output_messages_key="output",
            history_messages_key="chat_history",
        )

        logger.info("LegalRAGChain LCEL pipeline built successfully")

    def query(
        self,
        question: str,
        session_id: str = "default_session",
        filters: dict[str, Any] | None = None,
    ) -> LegalRAGResponse:
        """
        Execute grounded legal RAG query through the LCEL pipeline with strict scenario handling.

        Args:
            question: User legal question.
            session_id: Conversation session identifier for memory.
            filters: Optional metadata filtering criteria.

        Returns:
            LegalRAGResponse object.
        """
        if not self._with_memory:
            self._build_chain()

        config = {"configurable": {"session_id": session_id}}
        input_payload = {"input": question, "filters": filters}

        logger.info("Executing LegalRAGChain for session %r query=%r", session_id, question[:50])

        result: dict[str, Any] = self._with_memory.invoke(input_payload, config=config)

        retrieval_status = result.get("retrieval_status", "success")

        # SCENARIO 3: Retrieval Failed
        if retrieval_status == "retrieval_failed":
            error_exc = result.get("error_exc")
            dev_details = str(error_exc) if error_exc else "Document retrieval service unhandled error."
            return LegalRAGResponse(
                question=question,
                answer=(
                    "### ⚠️ Document Retrieval Temporarily Unavailable\n\n"
                    "Our legal knowledge base could not be searched because the retrieval service is "
                    "currently unavailable or has exceeded its usage limit.\n\n"
                    "Please try again later. If the issue persists, contact the administrator."
                ),
                retrieval_status="retrieval_failed",
                error_title="Document Retrieval Temporarily Unavailable",
                error_message=(
                    "Our legal knowledge base could not be searched because the retrieval service is "
                    "currently unavailable or has exceeded its usage limit. Please try again later."
                ),
                developer_details=dev_details,
                confidence_score=0.0,
            )

        # SCENARIO 2: Zero Documents Found
        if retrieval_status == "no_documents_found":
            return LegalRAGResponse(
                question=question,
                answer=result.get("answer_text", ""),
                retrieval_status="no_documents_found",
                confidence_score=0.0,
            )

        # SCENARIO 1: Normal Grounded Generation
        answer_text = result.get("answer_text", "")
        retrieved_docs: Sequence[LCDocument] = result.get("retrieved_docs", [])

        response = LegalRAGOutputParser.parse(
            question=question,
            answer_text=answer_text,
            docs=retrieved_docs,
        )
        response.retrieval_status = "success"
        return response
