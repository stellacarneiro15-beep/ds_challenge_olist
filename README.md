# ds_challenge_olist

Data-science challenge on the Olist Brazilian e-commerce dataset.

The notebooks keep the exploratory analysis, model comparison, plots, and feature-selection experiments. The `src/` package is intentionally limited to the code needed to run:

```bash
python -m src.main --customer_id <ID> --top_k 5
```

## Public Repository

This repository is public. Share the GitHub repository URL directly with reviewers
or collaborators.

Notebooks can be viewed directly on GitHub. If GitHub's notebook preview is slow
or unavailable, reviewers can clone the repository and open the notebooks locally.

## Project Structure

```text
ds_challenge_olist/
├─ data/                     # Olist CSVs
├─ notebooks/                # EDA, feature exploration, and model experiments
├─ src/
│  ├─ data_loader.py         # Reads Olist CSVs and builds the modeling table
│  ├─ features.py            # Purchase-time feature engineering + late target
│  ├─ model.py               # Production model artifact, training, persistence, scoring
│  └─ main.py                # CLI entrypoint for customer scoring
├─ tests/
│  └─ test_model.py
├─ artifacts/
│  └─ late_delivery_model.pkl
├─ requirements.txt
├─ pyproject.toml
└─ Makefile
```

## Getting Started

First, download the project to your machine. You can either clone the repository:

```bash
git clone <repository-url>
cd ds_challenge_olist
```

Or download the repository as a ZIP file from GitHub, unzip it, and open a terminal in the extracted `ds_challenge_olist/` folder.

After that, choose one of the setup options below.

### Option 1: Local Setup With Make

Use this option if you want to run the project directly on your machine. The
project requires Python 3.11 or newer.

Recommended setup:

```bash
make setup
```

`make setup` checks required native dependencies, then uses `uv` to create or
repair `.venv` with Python 3.11 and install all packages from
`requirements.txt`, including `pandas`. If Python 3.11 is not already installed,
`uv` downloads it automatically.

On macOS, install `uv` and the OpenMP runtime used by XGBoost and LightGBM:

```bash
brew install uv libomp
```

On Linux and Windows, install `uv` with the official installer or package
manager for your platform:

```bash
https://docs.astral.sh/uv/getting-started/installation/
```

To use a different supported Python version:

```bash
make setup PYTHON_VERSION=3.12
```

Manual setup also works if you already have a supported Python command
installed. Replace `python3.14` with your installed version, such as
`python3.13`, `python3.12`, or `python3.11`:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a prediction:

```bash
make run CUSTOMER_ID=<ID> TOP_K=5
```

Run the tests:

```bash
make test
```

Execute all notebooks:

```bash
make notebooks
```

### Option 2: Docker

Use this option if you prefer not to manage a local Python environment. Make sure Docker is installed and running first.

Build the image:

```bash
make docker-build
```

Run a prediction inside Docker:

```bash
docker run --rm \
  -v "$(pwd):/app" \
  -w /app \
  ds-challenge-olist \
  python -m src.main --customer_id <ID> --top_k 5
```

The notebooks can also be executed in Docker:

```bash
make docker-notebooks
```

This builds the `ds-challenge-olist` image and runs all notebooks with
`jupyter nbconvert --execute`. The repository is mounted into `/app`, so the
executed notebooks are written back to `notebooks/` on the host. Override the
image name or data location if needed:

```bash
make docker-notebooks DOCKER_IMAGE=my-olist-image DATA_DIR=data
```

## Usage

Activate the virtual environment before running commands directly with `python`:

```bash
source .venv/bin/activate
```

```bash
python -m src.main --customer_id 9ef432eb6251297304e76186b10a928d --top_k 5
```

Optional flags:

```bash
python -m src.main --customer_id <ID> --top_k 5 --data_dir ./data
python -m src.main --customer_id <ID> --top_k 5 --retrain
python -m src.main --customer_id <ID> --top_k 5 --model_path artifacts/late_delivery_model.pkl
```

Via Make:

```bash
make run CUSTOMER_ID=<ID> TOP_K=5
make test
```

## Troubleshooting

If you see `ModuleNotFoundError: No module named 'pandas'`, the project
dependencies are not installed in the Python environment currently running the
code. From the project folder, run:

```bash
make setup
```

Then retry the command. If you are running commands directly with `python`,
activate the environment first with `source .venv/bin/activate`. If you are
using VS Code or Jupyter, select the interpreter/kernel from `.venv` so
notebooks and scripts use the same environment.

If you see `Library not loaded: @rpath/libomp.dylib` while importing XGBoost or
LightGBM on macOS, install the missing OpenMP runtime and rerun setup:

```bash
brew install libomp
make setup
```

## Production Flow

The CLI loads the raw Olist CSVs, builds one row per delivered order, loads `artifacts/late_delivery_model.pkl` when available, trains the production gradient-boosting model if needed, selects an F1-oriented operating threshold on a chronological validation slice, and prints the customer's highest-risk delivered orders.

Only purchase-time information is used as model input. Post-purchase timestamps such as carrier delivery and customer delivery dates are excluded from features; the customer delivery date is used only to create the historical `late` target.
