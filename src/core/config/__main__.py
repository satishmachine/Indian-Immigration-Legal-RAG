"""
CLI smoke-test for the configuration layer.

Run with:
    uv run python -m src.core.config

Prints all non-sensitive configuration values to stdout.
"""

from __future__ import annotations

import sys

from core.config.settings import get_settings


def main() -> None:
    """Print a non-sensitive configuration summary to stdout."""
    cfg = get_settings()
    sep = "=" * 60
    print(sep)
    print(f"  {cfg.app.name}  v{cfg.app.version}")
    print(sep)
    print(f"  Environment   : {cfg.app.env.value}")
    print(f"  Active LLM    : {cfg.active_llm_provider.value}")
    print(f"  Qdrant URL    : {cfg.qdrant.url}")
    print(f"  Collection    : {cfg.qdrant.collection_name}")
    print(f"  Embedding     : {cfg.embedding.model_name} ({cfg.embedding.provider})")
    print(f"  Reranker      : {'on' if cfg.reranker.enabled else 'off'} — {cfg.reranker.model}")
    print(f"  Streamlit     : {cfg.streamlit.server_url}")
    print(f"  Document dir  : {cfg.directories.document_dir}")
    print(f"  Upload dir    : {cfg.directories.upload_dir}")
    print(f"  Eval dir      : {cfg.directories.evaluation_dir}")
    print(f"  LangSmith     : {'enabled' if cfg.langsmith.enabled else 'disabled'}")
    print(sep)


if __name__ == "__main__":
    main()
    sys.exit(0)
