.PHONY: all help install install-dev sync test test-cov lint format check build clean venv add add-dev update

UV := uv
PYTHON := $(UV) run python
SRC_DIR := src/lmlog
TEST_DIR := tests
VENV := .venv

GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

all: check

.DEFAULT_GOAL := help

help:
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  $(YELLOW)install$(NC)      - Install all dependencies (production + dev)"
	@echo "  $(YELLOW)install-prod$(NC) - Install production dependencies only"
	@echo "  $(YELLOW)sync$(NC)         - Sync environment with lockfile"
	@echo "  $(YELLOW)add$(NC)          - Add a new dependency"
	@echo "  $(YELLOW)add-dev$(NC)      - Add a new dev dependency"
	@echo "  $(YELLOW)update$(NC)       - Update all dependencies"
	@echo "  $(YELLOW)test$(NC)         - Run tests"
	@echo "  $(YELLOW)test-cov$(NC)     - Run tests with coverage"
	@echo "  $(YELLOW)lint$(NC)         - Run linting (ruff)"
	@echo "  $(YELLOW)format$(NC)       - Format code (black + ruff)"
	@echo "  $(YELLOW)check$(NC)        - Run all checks (lint + test)"
	@echo "  $(YELLOW)build$(NC)        - Build package"
	@echo "  $(YELLOW)clean$(NC)        - Clean build artifacts and environment"
	@echo "  $(YELLOW)venv$(NC)         - Create virtual environment"

# Check if uv is installed
check-uv:
	@command -v $(UV) >/dev/null 2>&1 || { \
		echo "$(RED)Error: uv is not installed$(NC)"; \
		echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	}

# Create virtual environment
venv: check-uv
	@echo "$(GREEN)Creating virtual environment...$(NC)"
	$(UV) venv

# Install all dependencies (production + dev)
install: check-uv
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	$(UV) sync
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

# Install production dependencies only
install-prod: check-uv
	@echo "$(YELLOW)Installing production dependencies only...$(NC)"
	$(UV) sync --no-dev

# Sync environment with lockfile
sync: check-uv
	@echo "$(GREEN)Syncing environment with lockfile...$(NC)"
	$(UV) sync

# Add a new dependency
add: check-uv
	@read -p "Enter package name: " package; \
	$(UV) add $$package

# Add a new dev dependency
add-dev: check-uv
	@read -p "Enter dev package name: " package; \
	$(UV) add --dev $$package

# Update all dependencies
update: check-uv
	@echo "$(YELLOW)Updating dependencies...$(NC)"
	$(UV) lock --upgrade
	$(UV) sync

# Run tests
test: check-uv
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(UV) run pytest -n auto $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-fail-under=95

testv: check-uv
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(UV) run pytest -v -n auto $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html --cov-fail-under=95

# Run linting
lint: check-uv
	@echo "$(YELLOW)Running linters...$(NC)"
	$(UV) run ruff check --fix $(SRC_DIR) $(TEST_DIR)
	pyrefly check $(SRC_DIR)

# Format code
format: check-uv
	@echo "$(YELLOW)Formatting code...$(NC)"
	$(UV) run black $(SRC_DIR) $(TEST_DIR) main.py

# Build package
build: check-uv
	@echo "$(GREEN)Building package...$(NC)"
	$(UV) build

# Show installed packages
show: check-uv
	@echo "$(GREEN)Installed packages:$(NC)"
	$(UV) pip list

# Generate requirements.txt
requirements: check-uv
	@echo "$(YELLOW)Generating requirements.txt...$(NC)"
	$(UV) pip compile pyproject.toml -o requirements.txt

# Clean build artifacts and environment
clean:
	@echo "$(RED)Cleaning build artifacts and environment...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f example_log.jsonl

# Clean everything including virtual environment
clean-all: clean
	@echo "$(RED)Removing virtual environment...$(NC)"
	rm -rf $(VENV)
	rm -f uv.lock
