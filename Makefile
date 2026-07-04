# =============================================================================
# ds_challenge_olist Makefile
# =============================================================================

VENV := .venv
UV ?= uv
UV_CACHE_DIR ?= .uv-cache
PYTHON_VERSION ?= 3.11
PYTHON := $(VENV)/bin/python
JUPYTER := $(VENV)/bin/jupyter

export UV_CACHE_DIR

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
	@echo "  setup      Create .venv with Python $(PYTHON_VERSION) and install requirements.txt"
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

setup:
	@if ! command -v $(UV) >/dev/null 2>&1; then \
		echo "Error: uv is required so make setup can install/manage Python automatically."; \
		echo "Install uv first: https://docs.astral.sh/uv/getting-started/installation/"; \
		exit 1; \
	fi
	@if [ ! -x "$(PYTHON)" ] || ! "$(PYTHON)" -c 'import sys; expected = "$(PYTHON_VERSION)".split(".")[:2]; actual = [str(sys.version_info.major), str(sys.version_info.minor)]; sys.exit(0 if actual == expected else 1)' >/dev/null 2>&1; then \
		$(UV) venv --python $(PYTHON_VERSION) --clear $(VENV); \
	fi
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(PYTHON) -c 'import pandas, sys; print(f"Setup complete. Python {sys.version.split()[0]} with pandas {pandas.__version__}.")'

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
	rm -rf $(VENV) $(UV_CACHE_DIR) .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
