"""
evaluation.models
=================
Pydantic Data Models for RAG Evaluation Benchmark.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class EvaluationSample(BaseModel):
    """Ground-truth test case sample for RAG evaluation."""

    id: str = Field(description="Unique sample identifier")
    question: str = Field(description="User legal query")
    ground_truth_answer: str | None = Field(default=None, description="Reference ideal answer")
    expected_sections: list[str] = Field(default_factory=list, description="Expected section numbers (e.g. ['21', '5'])")
    expected_act: str | None = Field(default=None, description="Expected Act name")


class EvaluationMetrics(BaseModel):
    """Detailed evaluation scores for a single query sample."""

    retrieval_recall: float = Field(ge=0.0, le=1.0, description="Proportion of expected sections retrieved")
    retrieval_precision: float = Field(ge=0.0, le=1.0, description="Ratio of relevant retrieved passages")
    faithfulness: float = Field(ge=0.0, le=1.0, description="Alignment of answer with retrieved statutory context")
    groundedness: float = Field(ge=0.0, le=1.0, description="Ratio of claims supported by direct statutory citations")
    hallucination_rate: float = Field(ge=0.0, le=1.0, description="Proportion of unsupported or hallucinated claims")
    citation_accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy of section/page citations")
    latency_seconds: float = Field(ge=0.0, description="End-to-end execution latency in seconds")


class EvaluationResult(BaseModel):
    """Complete evaluation result for a single sample."""

    sample_id: str
    question: str
    generated_answer: str
    metrics: EvaluationMetrics
    citations_count: int
    retrieved_sections: list[str]


class EvaluationReport(BaseModel):
    """Aggregate evaluation benchmark report across all samples."""

    total_samples: int
    mean_recall: float
    mean_precision: float
    mean_faithfulness: float
    mean_groundedness: float
    mean_hallucination_rate: float
    mean_citation_accuracy: float
    mean_latency_seconds: float
    passed_samples: int
    failed_samples: int
    results: list[EvaluationResult] = Field(default_factory=list)
