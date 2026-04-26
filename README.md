# Football Analysis ML Project

## Overview
This project analyzes FIFA 18 player data and now includes a reproducible machine learning pipeline for:

- `role` classification: `attack`, `midfield`, `defense`, `goalkeeper`
- `position` classification: merged exact primary positions such as `CB`, `ST`, `CM`, `GK`

The training source of truth is the raw dataset at [players_18.csv](./FootballAnalysisML/players_18.csv). The older processed CSV files are kept as legacy artifacts for comparison only.

## What Changed
- Replaced the fragile notebook-only training flow with a reusable `scikit-learn` package that also powers the main Jupyter notebook.
- Removed the hard dependency on `paralytics`.
- Made labels deterministic by deriving targets from the first listed player position instead of using randomness.
- Added train/evaluate CLIs and smoke tests.

## Project Layout
- [football_analysis](./football_analysis): reusable preprocessing, training, and evaluation code used by the notebook and CLI
- [FootballAnalysisML/players_18.csv](./FootballAnalysisML/players_18.csv): raw dataset
- [FootballAnalysisML/Dataset Analysis.ipynb](./FootballAnalysisML/Dataset%20Analysis.ipynb): main end-to-end Jupyter Lab notebook to run cell by cell
- [FootballAnalysisML/Dataset Analysis .ipynb](./FootballAnalysisML/Dataset%20Analysis%20.ipynb): legacy notebook kept for reference
- [tests](./tests): reproducibility and smoke tests

## Requirements
Install the main dependencies:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn jupyterlab
```

## Jupyter Lab Workflow
From the project root, start Jupyter Lab:

```bash
jupyter lab
```

Then open:

- [FootballAnalysisML/Dataset Analysis.ipynb](./FootballAnalysisML/Dataset%20Analysis.ipynb)

This notebook is the main workflow now. It includes:

- dataset exploration
- the same style of graphs shown in the old notebook
- role-model training
- exact-position-model training
- confusion matrices and metric summaries

Run the notebook from top to bottom, cell by cell.

## Optional CLI Training
Train the 4-role classifier from the terminal:

```bash
python -m football_analysis.train_role
```

Train the exact-position classifier from the terminal:

```bash
python -m football_analysis.train_position
```

Both commands write artifacts under `artifacts/`.

Useful options:

```bash
python -m football_analysis.train_role --search-iterations 4 --cv-folds 5
python -m football_analysis.train_position --search-iterations 4 --cv-folds 5
python -m football_analysis.train_role --sample-size 1000 --search-iterations 1 --cv-folds 2
```

## Evaluation
Evaluate a saved role model:

```bash
python -m football_analysis.evaluate --task role
```

Evaluate a saved position model:

```bash
python -m football_analysis.evaluate --task position
```

This recomputes metrics from the saved artifact and raw CSV, then writes `evaluation.json` beside the model.

## Tests
Run the included checks:

```bash
python -m unittest discover -s tests -v
```

The test suite verifies:

- deterministic target generation
- numeric preprocessing output before model fit
- end-to-end smoke training for both tasks

## Notebook
Use [FootballAnalysisML/Dataset Analysis.ipynb](./FootballAnalysisML/Dataset%20Analysis.ipynb) as the primary notebook. The Python package remains in the repo to keep the training pipeline deterministic and reusable, but the notebook is now the intended Jupyter Lab interface.
