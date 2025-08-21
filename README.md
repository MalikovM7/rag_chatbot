# rag_chatbot

![Build Status](https://github.com/MalikovM7/rag_chatbot/actions/workflows/ci-build.yaml/badge.svg)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![3.12](https://img.shields.io/badge/Python-3.12-green.svg)](https://shields.io/)

---

Reproducible ML project scaffold powered by uv

## Structure
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── uv.lock   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `uv lock > uv.lock`
    │
    ├── pyptoject.toml    <- makes project uv installable (uv installs) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py


--------


## Getting started (uv)
```bash
# create venv and sync (will create uv.lock)
uv sync

# add a runtime dependency
uv add numpy

# run code
uv run python -m src.models.train_model
```

## Code quality (ruff, isort, black via uvx)
### Run tools in ephemeral envs — no dev dependencies added to your project.

#### Lint (no changes)
```bash
# Lint entire repo
uvx ruff check .
```

#### Auto-fix
```bash
# 1) Sort imports
uvx isort .

# 2) Format code
uvx black .

# 3) Apply Ruff’s safe fixes (entire repo)
uvx ruff check --fix .
```
> Also remove unused imports/variables:
> ```bash
> uvx ruff check --fix --unsafe-fixes .
> ```



# RAG Chatbot – Deployment Guide

To deploy the project, open a terminal in Visual Studio (inside the folder where docker-compose.yml is located) and run:

docker compose up -d --build
## if your system uses the old binary:
## docker-compose up -d --build


This command builds and starts both the backend and frontend containers.

Once running, you can access the services in your browser:

Frontend (Streamlit UI): http://localhost:8501

Backend (FastAPI health check): http://localhost:8000/health

Useful commands:

Check running containers:

docker compose ps


View logs:

docker compose logs -f backend
docker compose logs -f frontend


Stop all containers:

docker compose down


Rebuild only one service:

docker compose up -d --build backend
docker compose up -d --build frontend


That’s it — start with docker compose up -d --build and then open the URLs above in your browser.
