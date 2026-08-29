# Makefile for the financial-ml project.
#
# Phase 0 wires up the install / lint / test / docker-up targets. The other
# targets (ingest / features / train / backtest / serve) are placeholders for
# later phases — they fail loudly with a TODO message so a wrong make target
# is obvious during the skeleton phase.

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

ASSET ?= SPY
MODEL ?= xgboost

.PHONY: help install install-tracking install-deep install-all lint lint-fix format test test-one ingest features \
        train backtest serve docker-up docker-down clean

help: ## Show this help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install project + dev dependencies into the active Python environment.
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

install-tracking: ## Add MLflow + pyarrow (Phase 6).
	$(PIP) install -e ".[tracking]"

install-deep: ## Add PyTorch (Phase 7).
	$(PIP) install -e ".[deep]"

install-all: ## Install every optional extra.
	$(PIP) install -e ".[dev,tracking,deep,notebook]"

lint: ## Run ruff (lint only) over the source and test trees.
	$(RUFF) check src tests

lint-fix: ## Run ruff lint with auto-fixes.
	$(RUFF) check --fix src tests

format: ## Format code with ruff.
	$(RUFF) format src tests

test: ## Run the full pytest suite.
	$(PYTEST)

test-one: ## Run a single test, e.g. `make test-one TEST=tests/unit/test_smoke.py::test_package_has_version`
	$(PYTEST) $(TEST)

ingest: ## Refresh raw data for an asset (Phase 1). Usage: make ingest ASSET=AAPL
	$(PYTHON) -m financial_ml.data.ingestion --asset $(ASSET)

ingest-all: ## Refresh raw data for ALL enabled assets.
	$(PYTHON) -m financial_ml.data.ingestion --all

features: ## Rebuild features from validated raw data (Phase 3).
	@echo "TODO: Phase 3 — implement feature pipeline" && exit 1

train: ## Run a tracked training run (Phase 6+). Usage: make train MODEL=xgboost
	@echo "TODO: Phase 6 — implement training entrypoint for $(MODEL)" && exit 1

backtest: ## Run the backtester on the registered model (Phase 9).
	@echo "TODO: Phase 9 — implement backtest entrypoint" && exit 1

serve: ## Start the FastAPI service locally (Phase 11).
	@echo "TODO: Phase 11 — implement uvicorn entrypoint" && exit 1

docker-up: ## Bring the full stack up via docker compose.
	docker compose up --build

docker-down: ## Tear down docker compose services.
	docker compose down

clean: ## Remove caches and build artifacts.
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
