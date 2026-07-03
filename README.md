# ds_challenge_olist

Data-science challenge on the [Olist Brazilian e-commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The notebooks keep the exploratory analysis, model comparison, plots, and feature-selection experiments. The `src/` package is intentionally limited to the code needed to run:

```bash
python -m src.main --customer_id <ID> --top_k 5
```

## Portfolio and Recruiter Sharing

Use these links when sharing the project with recruiters, hiring managers, or portfolio reviewers:

| Service | Best For | Link |
| --- | --- | --- |
| GitHub repository | Source code, tests, Dockerfile, and project structure | <https://github.com/stellacarneiro15-beep/ds_challenge_olist> |
| nbviewer | Read-only notebook review without GitHub rendering issues | [EDA](https://nbviewer.org/github/stellacarneiro15-beep/ds_challenge_olist/blob/main/notebooks/01_descriptive_analytics.ipynb), [feature exploration](https://nbviewer.org/github/stellacarneiro15-beep/ds_challenge_olist/blob/main/notebooks/02_delivery_feature_exploration.ipynb), [modeling](https://nbviewer.org/github/stellacarneiro15-beep/ds_challenge_olist/blob/main/notebooks/02_model.ipynb) |
| Google Colab | Interactive notebook walkthroughs | [Open modeling notebook](https://colab.research.google.com/github/stellacarneiro15-beep/ds_challenge_olist/blob/main/notebooks/02_model.ipynb) |
| Kaggle | Dataset context and reproducible data source | [Olist dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) |

Suggested recruiter note:

```text
This repository contains an end-to-end data-science challenge using the public
Olist Brazilian e-commerce dataset. It includes exploratory notebooks,
purchase-time feature engineering, a late-delivery risk model, tests, Docker
support, and a CLI for scoring customer orders.

GitHub: https://github.com/stellacarneiro15-beep/ds_challenge_olist
Notebook preview: https://nbviewer.org/github/stellacarneiro15-beep/ds_challenge_olist/tree/main/notebooks/
```

If a reviewer only has a few minutes, recommend starting with the README,
then `notebooks/02_model.ipynb`, then `src/features.py` and `src/model.py`.

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

Use this option if you want to run the project directly on your machine. The project requires Python 3.11 or newer.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or:

```bash
make setup
```

`make setup` automatically picks an available `python3.14`, `python3.13`, `python3.12`, or `python3.11`. If your interpreter has a different name, run `make setup VENV_PYTHON=/path/to/python3.11`.

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

## Production Flow

The CLI loads the raw Olist CSVs, builds one row per delivered order, loads `artifacts/late_delivery_model.pkl` when available, trains the production gradient-boosting model if needed, selects an F1-oriented operating threshold on a chronological validation slice, and prints the customer's highest-risk delivered orders.

Only purchase-time information is used as model input. Post-purchase timestamps such as carrier delivery and customer delivery dates are excluded from features; the customer delivery date is used only to create the historical `late` target.
