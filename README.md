# Turbofan Remaining Useful Life (RUL) — MLOps portfolio project

Estimate remaining useful life (in operating cycles) of turbofan engines from sensor history using
NASA C-MAPSS FD001. The goal is a reproducible MLOps workflow (validated data pipeline, engine-separated
evaluation, MLflow tracking and registry, FastAPI serving, a Streamlit UI); the status table below says what is built and what is still planned.

> Educational demonstration on simulated public data. Not an aircraft maintenance decision system.
> The models are trained on FD001 only. On the 100 official test engines, scored at each engine's last observed cycle, the primary XGBoost model has
> MAE 14.6 cycles (RMSE 21.9, R2 0.72); across every observed test row the MAE is about 31 cycles. `Agent/docs/model_card.md` says what these numbers do
> and do not mean (overestimation is the risky direction; short histories deserve caution; training engines are in-sample).

Requirements: see `docs/Turbofan_RUL_Functional_Requirements.docx` (FRD v1.0).

## Status

| Area | Status |
| --- | --- |
| Data pipeline: download, validation, labels, engine-separated split (`pipelines/`, `src/`) | Done |
| Models: XGBoost (primary) and Random Forest (support), saved in `model_save/` | Done |
| Streamlit dashboard and agent (`Agent/`): live RUL estimate with true RUL, train-engine life view, data explorer, fleet scoring, tool-using chat with guardrails | Done |
| Slack alert at 80% estimated life used, and one summary message for a fleet scan | Done |
| Automated tests (`pytest Agent/tests`) and 20 agent evals | Done |
| FastAPI service (`api/`), MLflow tracking and registry, Docker Compose, CI | Planned, not built |
| Drift monitoring, retraining candidates, promotion and rollback | Planned, not built |

## Setup (Apple Silicon / macOS)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Run the dashboard (from a fresh clone)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pipelines.download_data                         # NASA C-MAPSS into data/raw/CMAPSS (or: --zip path/to/archive.zip)
python -m pytest Agent/tests -q                           # synthetic fixtures, no data needed
python -m pipelines.simulate_fleet --out data/simulated   # optional: three simulated 100-engine batches for Fleet scoring
streamlit run Agent/app/app.py                            # run from the project root so .streamlit/config.toml applies
```

The trained models are in `model_save/`. The page has three views: **RUL agent**, **Data explorer** and **Fleet scoring**. Optional extras: `ollama pull qwen2.5:7b`
for the LLM mode, and a Slack incoming webhook for alerts (`export SLACK_WEBHOOK_URL=...` in the same terminal before starting; never commit it). Without the
webhook the alert logic runs in dry-run mode and sends nothing. Details: `Agent/README.md`.

## Data pipeline commands

```bash
python -m pipelines.download_data          # or: --zip path/to/archive.zip
python -m pipelines.prepare_data           # validation report, labels, split manifest
pytest -q
```

Raw data stays in `data/raw/` (git-ignored, immutable). Outputs go to `data/processed/`.

## Layout

`Agent/` agent layer and the Streamlit dashboard (`Agent/app/`) · `src/` shared data and model code · `pipelines/` data download, preparation and batch simulation ·
`model/` tuning and evaluation notebooks · `model_save/` trained models · `EDA/` exploration · `tests/` pipeline checks · `configs/` versioned policies · `docs/` requirements.
`app/` and `api/` at the top level are reserved for the planned FastAPI service and are empty.

## Data citation

NASA Prognostics Center of Excellence Data Set Repository, C-MAPSS turbofan engine degradation
simulation data. Retain the dataset README and cite as directed there.
