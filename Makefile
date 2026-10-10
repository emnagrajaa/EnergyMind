# EnergyMind — task runner (Linux/macOS/CI).
# On Windows (no `make`), use the equivalent PowerShell runner: .\tasks.ps1 <task>

PYTHON ?= python
CONFIG ?= configs

.PHONY: help install config-check data inject bench demo test lint fmt env-up env-down clean

help:  ## List available tasks
	@echo "EnergyMind tasks:"
	@echo "  install       pip install -e .[dev]"
	@echo "  config-check  validate all configs/*.yaml"
	@echo "  data          load, clean, quality mask, features"
	@echo "  inject        generate the anomaly dataset and labels"
	@echo "  bench         run the benchmark, write results/runs.jsonl"
	@echo "  demo          ~20 cases through agents, confidence, routing, audit log"
	@echo "  test          run unit tests"
	@echo "  lint          ruff check + format check"
	@echo "  fmt           ruff autofix + format"
	@echo "  env-up        docker compose up -d (Postgres + pgvector)"
	@echo "  env-down      docker compose down"
	@echo "  clean         remove build/test caches"

install:  ## Install the package and dev extras
	$(PYTHON) -m pip install -e ".[dev]"

config-check:  ## Validate every config file
	$(PYTHON) -m energymind.config

env-up:  ## Start the memory + audit database
	docker compose up -d

env-down:  ## Stop the database
	docker compose down

data:  ## Step 1-7: load, clean, mask, features
	$(PYTHON) -m energymind.pipeline data

inject:  ## Step 5: generate the anomaly dataset + labels
	$(PYTHON) -m energymind.pipeline inject

bench:  ## Step 8: benchmark -> results/runs.jsonl
	$(PYTHON) -m energymind.pipeline bench

demo:  ## Step 9-12: decision demo on ~20 cases
	$(PYTHON) -m energymind.pipeline demo

test:  ## Run the unit tests
	$(PYTHON) -m pytest

lint:  ## Static checks
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

fmt:  ## Autofix and format
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

clean:  ## Remove caches
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['build','dist','.pytest_cache','.ruff_cache','.coverage','htmlcov']]"
