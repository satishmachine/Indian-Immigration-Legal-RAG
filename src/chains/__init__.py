"""
chains
======
LangChain LCEL Chains and Pipelines Package.

Exports:
- LegalRAGChain
- LegalCompressionRetriever
- create_history_aware_legal_retriever
- format_docs_for_prompt
- extract_citations_from_docs
- LegalRAGOutputParser
- get_session_history
- clear_session_history
"""

from __future__ import annotations

from chains.citation_formatter import extract_citations_from_docs, format_docs_for_prompt
from chains.history_aware_retriever import create_history_aware_legal_retriever
from chains.legal_rag_chain import LegalRAGChain
from chains.memory import clear_session_history, get_session_history
from chains.output_parser import LegalRAGOutputParser
from retrieval.compression_retriever import LegalCompressionRetriever

__all__: list[str] = [
    "LegalCompressionRetriever",
    "LegalRAGChain",
    "LegalRAGOutputParser",
    "clear_session_history",
    "create_history_aware_legal_retriever",
    "extract_citations_from_docs",
    "format_docs_for_prompt",
    "get_session_history",
]
