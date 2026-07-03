# =============================================================================
# ds_challenge_olist Makefile
# =============================================================================

VENV := .venv
VENV_PYTHON ?= $(shell for p in python3.14 python3.13 python3.12 python3.11; do command -v $$p >/dev/null 2>&1 && { echo $$p; exit; }; done; echo python3)
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
JUPYTER := $(VENV)/bin/jupyter

DATA_DIR ?= data
DOCKER_IMAGE ?= ds-challenge-olist
SHARE_ARCHIVE ?= dist/ds_challenge_olist_private_share.zip

# Prediction CLI parameters (make run CUSTOMER_ID=<ID> [TOP_K=5])
CUSTOMER_ID ?=
TOP_K ?= 5

# Notebooks executed in order by `make notebooks`
NOTEBOOKS := \
	notebooks/01_descriptive_analytics.ipynb \
	notebooks/02_delivery_feature_exploration.ipynb \
	notebooks/02_model.ipynb

.PHONY: help setup notebooks docker-build docker-notebooks run predict test share-archive clean

help:
	@echo "Usage: make <target>"
	@echo ""
	@echo "  setup      Create .venv and install requirements.txt"
	@echo "  notebooks  Execute all notebooks in order (jupyter nbconvert)"
	@echo "  docker-build      Build the Docker image"
	@echo "  docker-notebooks  Execute all notebooks inside Docker"
	@echo "  run        Score a customer (make run CUSTOMER_ID=<ID> [TOP_K=5])"
	@echo "  predict    Score a customer (make predict CUSTOMER_ID=<ID> [TOP_K=5])"
	@echo "  test       Run the test suite (pytest)"
	@echo "  share-archive  Build a private ZIP archive from the committed repo"
	@echo "  clean      Remove venv, caches, and __pycache__"

# =============================================================================
# SETUP
# =============================================================================

setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(VENV_PYTHON) -c 'import sys; sys.exit("Python >=3.11 is required; set VENV_PYTHON=/path/to/python3.11+") if sys.version_info < (3, 11) else None'
	$(VENV_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

# =============================================================================
# NOTEBOOKS
# =============================================================================

notebooks: setup
	$(JUPYTER) nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=-1 $(NOTEBOOKS)

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-notebooks: docker-build
	docker run --rm \
		-v "$(CURDIR):/app" \
		-w /app \
		-e DATA_DIR=/app/$(DATA_DIR) \
		$(DOCKER_IMAGE) \
		jupyter nbconvert --to notebook --execute --inplace \
			--ExecutePreprocessor.timeout=-1 $(NOTEBOOKS)

# =============================================================================
# RUN
# =============================================================================

run: setup
	@if [ -z "$(CUSTOMER_ID)" ]; then \
		echo "Error: CUSTOMER_ID is required. Usage: make run CUSTOMER_ID=<ID> [TOP_K=5]"; \
		exit 1; \
	fi
	DATA_DIR=$(DATA_DIR) $(PYTHON) -m src.main --customer_id $(CUSTOMER_ID) --top_k $(TOP_K)

predict: setup
	@if [ -z "$(CUSTOMER_ID)" ]; then \
		echo "Error: CUSTOMER_ID is required. Usage: make predict CUSTOMER_ID=<ID> [TOP_K=5]"; \
		exit 1; \
	fi
	DATA_DIR=$(DATA_DIR) $(PYTHON) -m src.main --customer_id $(CUSTOMER_ID) --top_k $(TOP_K)

test: setup
	$(PYTHON) -m pytest -q

# =============================================================================
# UTILITIES
# =============================================================================

share-archive:
	@if ! git diff --quiet || ! git diff --cached --quiet; then \
		echo "Error: commit or stash changes before building the share archive."; \
		exit 1; \
	fi
	mkdir -p $(dir $(SHARE_ARCHIVE))
	rm -f $(SHARE_ARCHIVE)
	git archive --format=zip --output=$(SHARE_ARCHIVE) --prefix=ds_challenge_olist/ HEAD
	@echo "Created $(SHARE_ARCHIVE)"
	@echo "Share this ZIP through a restricted file-sharing link."

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
