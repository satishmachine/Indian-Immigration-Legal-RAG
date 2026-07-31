# =============================================================================
# Indian Immigration & Emigration Legal Assistant – Makefile
# =============================================================================
.PHONY: help install install-dev setup-pre-commit lint format typecheck \
        test test-unit test-integration test-cov \
        run-api run-app run-all clean docker-qdrant

PYTHON   := python
PIP      := pip
UVICORN  := uvicorn
SRC_DIR  := src
TEST_DIR := tests

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  Indian Immigration Legal Assistant – Developer Commands"
	@echo "  ────────────────────────────────────────────────────────"
	@echo "  make install          Install production dependencies"
	@echo "  make install-dev      Install all dependencies (incl. dev)"
	@echo "  make setup-pre-commit Install and configure pre-commit hooks"
	@echo "  make lint             Run Ruff linter"
	@echo "  make format           Auto-format with Black + Ruff"
	@echo "  make typecheck        Run MyPy type checker"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run integration tests only"
	@echo "  make test-cov         Run tests with HTML coverage report"
	@echo "  make run-api          Start FastAPI development server"
	@echo "  make run-app          Start Streamlit frontend"
	@echo "  make docker-qdrant    Start Qdrant via Docker"
	@echo "  make clean            Remove build artifacts"
	@echo ""

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt --no-deps

install-dev:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup-pre-commit: install-dev
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅  pre-commit hooks installed"

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------
lint:
	ruff check $(SRC_DIR) $(TEST_DIR)

format:
	black $(SRC_DIR) $(TEST_DIR)
	ruff check --fix $(SRC_DIR) $(TEST_DIR)
	ruff format $(SRC_DIR) $(TEST_DIR)

typecheck:
	mypy $(SRC_DIR) --config-file pyproject.toml

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
test:
	pytest $(TEST_DIR) -v

test-unit:
	pytest $(TEST_DIR) -v -m unit

test-integration:
	pytest $(TEST_DIR) -v -m integration

test-cov:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) \
	  --cov-report=term-missing \
	  --cov-report=html:reports/coverage
	@echo "Coverage report: reports/coverage/index.html"

# ---------------------------------------------------------------------------
# Run Services
# ---------------------------------------------------------------------------
run-api:
	PYTHONPATH=src $(UVICORN) api.main:app \
	  --host 0.0.0.0 --port 8000 --reload

run-app:
	PYTHONPATH=src streamlit run src/app/main.py \
	  --server.port 8501

run-all:
	@echo "Start Qdrant first: make docker-qdrant"
	@echo "Then open two terminals: make run-api && make run-app"

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------
docker-qdrant:
	docker run -d --name qdrant \
	  -p 6333:6333 -p 6334:6334 \
	  -v $(PWD)/qdrant_storage:/qdrant/storage \
	  qdrant/qdrant:latest
	@echo "✅  Qdrant running at http://localhost:6333"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	rm -rf dist build *.egg-info
	rm -rf reports/coverage htmlcov
	@echo "🧹  Cleaned build artifacts"
