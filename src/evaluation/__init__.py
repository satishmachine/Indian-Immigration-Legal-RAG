"""
evaluation
==========
RAG Evaluation Framework Package.

Exports:
- EvaluationSample
- EvaluationMetrics
- EvaluationResult
- EvaluationReport
- LegalRAGEvaluator
- EvaluationPipeline
- compute_retrieval_recall
- compute_retrieval_precision
- compute_faithfulness
- compute_groundedness
- compute_hallucination_rate
- compute_citation_accuracy
"""

from __future__ import annotations

from evaluation.evaluator import LegalRAGEvaluator
from evaluation.metrics import (
    compute_citation_accuracy,
    compute_faithfulness,
    compute_groundedness,
    compute_hallucination_rate,
    compute_retrieval_precision,
    compute_retrieval_recall,
)
from evaluation.models import (
    EvaluationMetrics,
    EvaluationReport,
    EvaluationResult,
    EvaluationSample,
)
from evaluation.pipeline import EvaluationPipeline

__all__: list[str] = [
    "EvaluationMetrics",
    "EvaluationPipeline",
    "EvaluationReport",
    "EvaluationResult",
    "EvaluationSample",
    "LegalRAGEvaluator",
    "compute_citation_accuracy",
    "compute_faithfulness",
    "compute_groundedness",
    "compute_hallucination_rate",
    "compute_retrieval_precision",
    "compute_retrieval_recall",
]
