.PHONY: help install install-dev test lint format clean run-example

help:
	@echo "H2 Seep Detection - Available Commands"
	@echo "======================================="
	@echo "install          - Install package and dependencies"
	@echo "install-dev      - Install with development dependencies"
	@echo "test             - Run tests"
	@echo "lint             - Run linting checks"
	@echo "format           - Format code with black"
	@echo "clean            - Remove build artifacts and cache"
	@echo "run-example      - Run example usage script"

install:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,notebooks]"

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/ example_usage.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/

run-example:
	python example_usage.py
