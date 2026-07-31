import pathlib

dirs = [
    "src/core/config", "src/core/logging", "src/core/exceptions",
    "src/core/interfaces", "src/core/models", "src/core/utils",
    "src/ingestion/parsers", "src/ingestion/chunkers", "src/ingestion/pipeline",
    "src/retrieval/dense", "src/retrieval/sparse", "src/retrieval/hybrid",
    "src/retrieval/reranker",
    "src/services/chat", "src/services/embedding", "src/services/vector_store",
    "src/api/v1/endpoints", "src/api/v1/schemas",
    "src/api/middleware", "src/api/dependencies",
    "src/app/pages", "src/app/components",
    "tests/unit/core", "tests/unit/ingestion",
    "tests/unit/retrieval", "tests/unit/services",
    "tests/integration", "tests/e2e", "tests/fixtures",
    "configs", "scripts", "docs", "reports/coverage", "logs", "notebooks",
]

INIT_CONTENT = '"""Package init."""\n'

for d in dirs:
    p = pathlib.Path(d)
    p.mkdir(parents=True, exist_ok=True)
    init = p / "__init__.py"
    if not init.exists():
        init.write_text(INIT_CONTENT, encoding="utf-8")

print(f"Done: created {len(dirs)} directories with __init__.py files.")
