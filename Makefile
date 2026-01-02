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
	@echo "  make train-quick   - Quick test training"
	@echo ""
	@echo "Data:"
	@echo "  make download-data - Download latest stock data"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start all services (detached)"
	@echo "  make docker-down   - Stop all services"
	@echo "  make docker-logs   - View logs"
	@echo "  make docker-restart- Restart all services"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test          - Run quick tests"
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
	uv run python src/api/app.py

run-ui:
	uv run streamlit run src/ui/app.py --server.port 8501

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
	uv run python scripts/train_all.py --ml-only -y

train-dl:
	uv run python scripts/train_all.py --dl-only -y

train-ts:
	uv run python scripts/train_all.py --ts-only -y

train-quick:
	uv run python scripts/test_quick.py

train-ticker:
	@read -p "Enter ticker (e.g., AAPL): " ticker; \
	uv run python scripts/train_all.py --ticker $$ticker -y

# ==============================================================================
# Data
# ==============================================================================

download-data:
	uv run python -c "from src.data.downloader import DataDownloader; DataDownloader().download_all()"

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
	uv run python scripts/test_quick.py

lint:
	uv run ruff check src/ scripts/

format:
	uv run ruff format src/ scripts/

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
	uv run python scripts/train_all.py --ml-only --no-save -y
	uv run python scripts/train_all.py --ts-only --no-save -y

ci-lint:
	uv run ruff check src/ scripts/ --exit-zero
