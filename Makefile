.PHONY: help install install-dev test test-unit test-integration lint format clean \
        docker-build docker-run docker-stop docker-logs docker-shell \
        api train batch serve

# =============================================================================
# Variables
# =============================================================================
PYTHON := python
PIP := pip
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := h2-seep-detection
VERSION := $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")

# =============================================================================
# Help
# =============================================================================
help:
	@echo "H2 Seep Detection - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup:"
	@echo "  install          - Install package and dependencies"
	@echo "  install-dev      - Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test             - Run all tests"
	@echo "  test-unit        - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-cov         - Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint             - Run linting checks"
	@echo "  format           - Format code with black and isort"
	@echo "  type-check       - Run mypy type checking"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     - Build Docker image"
	@echo "  docker-build-gpu - Build GPU Docker image"
	@echo "  docker-run       - Run API in Docker"
	@echo "  docker-dev       - Run development environment"
	@echo "  docker-stop      - Stop Docker containers"
	@echo "  docker-logs      - View Docker logs"
	@echo "  docker-shell     - Open shell in container"
	@echo ""
	@echo "Application:"
	@echo "  serve            - Start API server locally"
	@echo "  train            - Train model (requires dataset)"
	@echo "  batch            - Run batch processing"
	@echo ""
	@echo "Utilities:"
	@echo "  clean            - Remove build artifacts and cache"
	@echo "  clean-docker     - Remove Docker images and volumes"

# =============================================================================
# Setup
# =============================================================================
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"
	$(PIP) install pytest pytest-cov pytest-asyncio black flake8 isort mypy httpx
	pre-commit install || true

# =============================================================================
# Testing
# =============================================================================
test:
	pytest tests/ -v -m "not slow and not gpu"

test-unit:
	pytest tests/ -v --ignore=tests/integration -m "not slow and not gpu"

test-integration:
	pytest tests/integration -v -m "not slow and not gpu"

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing -m "not slow and not gpu"
	@echo "Coverage report: htmlcov/index.html"

test-all:
	pytest tests/ -v --cov=src

# =============================================================================
# Code Quality
# =============================================================================
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	@echo "Linting passed!"

format:
	black src/ tests/ scripts/ examples/
	isort src/ tests/ scripts/ examples/
	@echo "Formatting complete!"

format-check:
	black --check --diff src/ tests/
	isort --check-only --diff src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

quality: format-check lint type-check
	@echo "All quality checks passed!"

# =============================================================================
# Docker
# =============================================================================
docker-build:
	docker build -t $(IMAGE_NAME):$(VERSION) -t $(IMAGE_NAME):latest --target production .

docker-build-gpu:
	docker build -t $(IMAGE_NAME):$(VERSION)-gpu -t $(IMAGE_NAME):latest-gpu --target gpu .

docker-build-dev:
	docker build -t $(IMAGE_NAME):dev --target development .

docker-run:
	$(DOCKER_COMPOSE) up -d api

docker-dev:
	$(DOCKER_COMPOSE) --profile dev up dev

docker-stop:
	$(DOCKER_COMPOSE) down

docker-logs:
	$(DOCKER_COMPOSE) logs -f

docker-shell:
	docker exec -it h2-seep-api /bin/bash

docker-clean:
	$(DOCKER_COMPOSE) down -v --rmi local
	docker image prune -f

# Production deployment
docker-prod:
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.prod.yml up -d

# GPU deployment
docker-gpu:
	$(DOCKER_COMPOSE) -f docker-compose.yml -f docker-compose.gpu.yml up -d

# =============================================================================
# Application
# =============================================================================
serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4

train:
	$(PYTHON) scripts/train_model.py --data data/dataset.yaml --output runs/train

batch:
	$(PYTHON) scripts/batch_process.py --input data/input --output outputs/results.json

predict:
	$(PYTHON) -m src.cli.main predict $(IMAGE)

export-model:
	$(PYTHON) scripts/export_model.py --weights models/weights/best.pt --format onnx

# =============================================================================
# Examples
# =============================================================================
run-example:
	$(PYTHON) examples/usage_example.py

# =============================================================================
# Utilities
# =============================================================================
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .coverage htmlcov/ .eggs/
	@echo "Cleaned!"

clean-all: clean clean-docker
	rm -rf runs/ outputs/ data/cache/
	@echo "Deep clean complete!"

# Version info
version:
	@echo "Version: $(VERSION)"
	@echo "Python: $(shell $(PYTHON) --version)"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'not installed')"
