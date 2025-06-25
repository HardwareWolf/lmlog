.PHONY: help install install-dev test test-cov lint format check build clean

PYTHON := python3
PIP := pip3
SRC_DIR := src/lmlog
TEST_DIR := tests

help:
	@echo "Available commands:"
	@echo "  install     - Install package"
	@echo "  install-dev - Install package with dev dependencies"
	@echo "  test        - Run tests"
	@echo "  test-cov    - Run tests with coverage"
	@echo "  lint        - Run linting (ruff + pyrefly)"
	@echo "  format      - Format code (black + ruff)"
	@echo "  check       - Run all checks (lint + test)"
	@echo "  build       - Build package"
	@echo "  clean       - Clean build artifacts"

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev]"

test:
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term

test-cov:
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-fail-under=100

lint:
	ruff check --fix $(SRC_DIR) $(TEST_DIR)
	pyrefly check $(SRC_DIR)

format:
	black $(SRC_DIR) $(TEST_DIR) main.py
	ruff format $(SRC_DIR) $(TEST_DIR) main.py

check: lint test-cov

build:
	$(PYTHON) -m build

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f example_log.jsonl

dev-setup: install-dev
	@echo "Development environment setup complete"