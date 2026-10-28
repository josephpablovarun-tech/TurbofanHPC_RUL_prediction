# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

Remaining Useful Life (RUL, in operating cycles) prediction for NASA C-MAPSS **FD001** turbofan data: a portfolio project with a trained model, a tool-using agent and a Streamlit dashboard. Educational demo on simulated public data, not a maintenance decision system.

Requirement IDs in docstrings and comments (`FR01`-`FR10`, `ML01`) refer to the FRD at `docs/Turbofan_RUL_Functional_Requirements.docx`. Tag new code with the requirement it implements.

## Status

- Built: data pipeline (`pipelines/`, `src/`), trained models (`model_save/`: XGBoost primary with 400 trees and depth 7, Random Forest support with 300 trees), the agent and Streamlit dashboard (`Agent/`), Slack alerting, fleet scoring, tests and agent evals.
- Planned, not built: FastAPI service (`api/`), MLflow tracking and registry, Docker Compose, CI, drift monitoring, retraining. `api/` and `app/` at the top level are empty placeholders. Do not describe these as done.
- Headline metrics (see `Agent/docs/model_card.md`): on the 100 official test engines at their last cycle, MAE 14.6, RMSE 21.9, R2 0.72. Quote them only with that qualification; across all test rows MAE is about 31.

## Commands

Run everything from the repo root (`configs/data.yaml` and `.streamlit/config.toml` are read relative to it).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

python -m pipelines.download_data            # or: --zip path/to/archive.zip
python -m pipelines.prepare_data             # writes to data/processed/
python -m pipelines.simulate_fleet --out data/simulated   # optional batch CSVs for Fleet scoring

python -m pytest Agent/tests -q              # agent, app, alerts, fleet, explorer tests
pytest tests -q                              # data pipeline tests
streamlit run Agent/app/app.py               # dashboard
```

Streamlit caches imported modules, so restart the app after code changes.

## Architecture

**Data layer (`src/`, `pipelines/`).** `download_data` writes `data/raw/CMAPSS/` (with sha256 checksums); `prepare_data` writes `data/processed/`. Both are git-ignored and raw data is immutable.
- `schema.py`: fixed 26-column layout (`engine_id`, `cycle`, `setting_1..3`, `sensor_1..21`).
- `data_loader.py`, `validation.py`: parse the headerless files and raise `DataValidationError` on hard failures. Gaps and constant columns are warnings. Rows are never dropped silently.
- `labels.py`: training RUL = max cycle minus current cycle, uncapped. For test data the last observed cycle is not failure; true RUL comes from `RUL_FD001.txt` and is for evaluation only.
- `splits.py`: train/validation split by whole engines, seeded from `configs/data.yaml`.
- `predictor.py`, `features.py`, `rul_models.py`: 97 causal features, square-root target, XGBoost and Random Forest.

**Agent layer (`Agent/`).**
- `orchestration/`: supervisor (regex routing), engine and document agents, tool allowlists, call budget, validator, deterministic mode and optional Ollama LLM mode (numbers checked against tools).
- `tools/`: `datastore.py` (data access with the cutoff enforced), `tools.py` (tool suite), `life_view.py` (train-engine life view), `explore.py`, `fleet_upload.py`, `alerts.py` (Slack).
- `app/`: `app.py` (RUL agent view), `explorer.py` (Data explorer), `fleet.py` (Fleet scoring), `charts.py`.
- `schemas/contracts.py`: typed contracts. `configs/agent.yaml`: agent, alert and data settings.

## Rules that must hold

- **No leakage past the cutoff.** The cutoff is enforced in the data layer and the feature builder. Tools take no cutoff argument, and the agent code must never reach the official test labels. Only train engines may be read past the cutoff, and then labelled retrospective.
- **Never commit or print the Slack webhook.** `SLACK_WEBHOOK_URL` is set only in the terminal environment that runs Streamlit. Without it, alerts run in dry-run mode.
- **Alerts fire once per crossing** of 80% estimated life used (re-arm below 75%), per split, engine and model version. Fleet scoring posts one summary per file and model version.
- **Describe sensor links as association, not cause.** The models give point estimates with no calibrated uncertainty.
- Add dataset policy to `configs/data.yaml` rather than hard-coding it.

## Tests

`Agent/tests/` uses synthetic fixtures from `conftest.py` (including a synthetic `RUL_FD001.txt`, with alerts disabled), so the suite runs without the NASA data. `tests/` holds the pipeline tests built on `make_history({engine_id: n_cycles})`. `Agent/evals/run.py` runs the 20 agent evals.

## Notebooks

`EDA/01_fd001_eda.ipynb` is self-contained and reimplements loading, validation and the split inline; its engine sets can differ from the pipeline's. `model/` holds tuning and evaluation notebooks.
