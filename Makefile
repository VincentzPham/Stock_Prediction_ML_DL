# ==============================================================================
# Stock Prediction ML/DL - Makefile
# ==============================================================================

.PHONY: help install dev clean test lint format run-api run-ui train docker-build docker-up docker-down docker-logs

# Default target
help:
	@echo "=============================================="
	@echo "Stock Prediction ML/DL - Available Commands"
	@echo "=============================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install dependencies using uv"
	@echo "  make dev           - Install with dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make run-api       - Run FastAPI backend (port 8000)"
	@echo "  make run-ui        - Run Streamlit frontend (port 8501)"
	@echo "  make run           - Run both API and UI"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Train all models (interactive)"
	@echo "  make train-ml      - Train ML models only (fast)"
	@echo "  make train-dl      - Train Deep Learning models"
	@echo "  make train-ts      - Train Time Series models"
	@echo "  make train-aapl    - Train all models for AAPL"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start all services (detached)"
	@echo "  make docker-down   - Stop all services"
	@echo "  make docker-logs   - View logs"
	@echo "  make docker-restart- Restart all services"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          - Run pytest tests"
	@echo "  make lint          - Run linter (ruff)"
	@echo "  make format        - Format code (ruff)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean cache and temp files"
	@echo "  make clean-models  - Remove all trained models"
	@echo ""

# ==============================================================================
# Setup
# ==============================================================================

install:
	uv sync

dev:
	uv sync --dev

# ==============================================================================
# Development - Run Services
# ==============================================================================

run-api:
	uv run uvicorn backend.api.app:app --reload --host 0.0.0.0 --port 8000

run-ui:
	uv run streamlit run frontend/app.py --server.port 8501

run:
	@echo "Starting API and UI..."
	@echo "API will be available at http://localhost:8000"
	@echo "UI will be available at http://localhost:8501"
	@make -j2 run-api run-ui

# ==============================================================================
# Training
# ==============================================================================

train:
	uv run python scripts/train_all.py

train-ml:
	uv run python scripts/train_all.py --ml-only

train-dl:
	uv run python scripts/train_all.py --dl-only

train-ts:
	uv run python scripts/train_all.py --ts-only

train-aapl:
	uv run python scripts/train_all.py -t AAPL

train-lstm:
	uv run python scripts/train_all.py -m LSTM

train-tune:
	uv run python scripts/train_all.py -t AAPL -m LSTM --tune --trials 20

# ==============================================================================
# Docker
# ==============================================================================

docker-build:
	docker compose build

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f

docker-restart:
	docker compose down && docker compose up --build -d

docker-shell-api:
	docker compose exec api /bin/bash

docker-shell-ui:
	docker compose exec ui /bin/bash

# ==============================================================================
# Testing & Quality
# ==============================================================================

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check backend/ frontend/ scripts/

format:
	uv run ruff format backend/ frontend/ scripts/

# ==============================================================================
# Cleanup
# ==============================================================================

clean:
	@echo "Cleaning cache and temp files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Done!"

clean-models:
	@echo "WARNING: This will delete all trained models!"
	@read -p "Are you sure? [y/N]: " confirm; \
	if [ "$$confirm" = "y" ]; then \
		rm -rf Models/*/*.pkl Models/*/*.keras; \
		echo "Models deleted."; \
	else \
		echo "Aborted."; \
	fi

# ==============================================================================
# CI/CD Helpers
# ==============================================================================

ci-test:
	uv run pytest tests/ -v --tb=short

ci-lint:
	uv run ruff check backend/ frontend/ scripts/ --exit-zero
