"""
evaluation.pipeline
===================
Evaluation Pipeline Coordinator.

Runs evaluation benchmarks over test datasets and exports aggregate Markdown/JSON evaluation reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

from evaluation.evaluator import LegalRAGEvaluator
from evaluation.models import EvaluationReport, EvaluationResult, EvaluationSample

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Coordinator executing batch RAG evaluation benchmarks."""

    def __init__(
        self,
        evaluator: LegalRAGEvaluator | None = None,
        min_groundedness_threshold: float = 0.8,
    ) -> None:
        self.evaluator = evaluator or LegalRAGEvaluator()
        self.min_groundedness_threshold = min_groundedness_threshold

    def run_benchmark(self, samples: Sequence[EvaluationSample]) -> EvaluationReport:
        """
        Run batch evaluation over a dataset of EvaluationSample test cases.

        Args:
            samples: Sequence of EvaluationSample test cases.

        Returns:
            EvaluationReport containing mean metric scores across all samples.
        """
        results: list[EvaluationResult] = []
        passed = 0
        failed = 0

        logger.info("Starting EvaluationPipeline benchmark over %d samples...", len(samples))

        for sample in samples:
            res = self.evaluator.evaluate_sample(sample)
            results.append(res)

            if res.metrics.groundedness >= self.min_groundedness_threshold:
                passed += 1
            else:
                failed += 1

        total = max(len(samples), 1)
        report = EvaluationReport(
            total_samples=len(samples),
            mean_recall=round(sum(r.metrics.retrieval_recall for r in results) / total, 4),
            mean_precision=round(sum(r.metrics.retrieval_precision for r in results) / total, 4),
            mean_faithfulness=round(sum(r.metrics.faithfulness for r in results) / total, 4),
            mean_groundedness=round(sum(r.metrics.groundedness for r in results) / total, 4),
            mean_hallucination_rate=round(sum(r.metrics.hallucination_rate for r in results) / total, 4),
            mean_citation_accuracy=round(sum(r.metrics.citation_accuracy for r in results) / total, 4),
            mean_latency_seconds=round(sum(r.metrics.latency_seconds for r in results) / total, 4),
            passed_samples=passed,
            failed_samples=failed,
            results=results,
        )

        logger.info(
            "EvaluationPipeline completed: Mean Groundedness=%.2f, Mean Recall=%.2f, Passed=%d/%d",
            report.mean_groundedness,
            report.mean_recall,
            passed,
            len(samples),
        )
        return report

    def export_report_markdown(self, report: EvaluationReport, output_path: str | Path) -> None:
        """Export EvaluationReport as a formatted Markdown artifact."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        md = [
            "# Statutory Legal RAG Evaluation Benchmark Report\n",
            "## Summary Metrics\n",
            "| Metric | Mean Score / Value | Target Threshold | Status |",
            "|---|---|---|---|",
            f"| **Retrieval Recall** | `{report.mean_recall:.4f}` | `>= 0.80` | {'🟢 PASS' if report.mean_recall >= 0.8 else '🔴 FAIL'} |",
            f"| **Retrieval Precision** | `{report.mean_precision:.4f}` | `>= 0.75` | {'🟢 PASS' if report.mean_precision >= 0.75 else '🔴 FAIL'} |",
            f"| **Faithfulness** | `{report.mean_faithfulness:.4f}` | `>= 0.85` | {'🟢 PASS' if report.mean_faithfulness >= 0.85 else '🔴 FAIL'} |",
            f"| **Groundedness** | `{report.mean_groundedness:.4f}` | `>= 0.85` | {'🟢 PASS' if report.mean_groundedness >= 0.85 else '🔴 FAIL'} |",
            f"| **Hallucination Rate** | `{report.mean_hallucination_rate:.4f}` | `<= 0.15` | {'🟢 PASS' if report.mean_hallucination_rate <= 0.15 else '🔴 FAIL'} |",
            f"| **Citation Accuracy** | `{report.mean_citation_accuracy:.4f}` | `>= 0.90` | {'🟢 PASS' if report.mean_citation_accuracy >= 0.90 else '🔴 FAIL'} |",
            f"| **Mean Latency** | `{report.mean_latency_seconds:.4f}s` | `<= 3.00s` | {'🟢 PASS' if report.mean_latency_seconds <= 3.0 else '🟡 WARN'} |\n",
            f"**Total Benchmark Samples**: `{report.total_samples}` | **Passed**: `{report.passed_samples}` | **Failed**: `{report.failed_samples}`\n",
            "## Per-Sample Detailed Breakdown\n",
        ]

        for r in report.results:
            md.append(f"### Sample ID: `{r.sample_id}`")
            md.append(f"**Query**: {r.question}\n")
            md.append(f"- **Recall**: `{r.metrics.retrieval_recall}` | **Precision**: `{r.metrics.retrieval_precision}` | **Groundedness**: `{r.metrics.groundedness}`")
            md.append(f"- **Hallucination Rate**: `{r.metrics.hallucination_rate}` | **Citation Accuracy**: `{r.metrics.citation_accuracy}` | **Latency**: `{r.metrics.latency_seconds}s`")
            md.append(f"- **Referenced Sections**: `{r.retrieved_sections}`\n")

        path.write_text("\n".join(md), encoding="utf-8")
        logger.info("Exported evaluation markdown report to %s", path)
