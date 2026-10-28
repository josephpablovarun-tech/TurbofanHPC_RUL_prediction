# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Remaining Useful Life (RUL, in operating cycles) prediction for NASA C-MAPSS **FD001** turbofan data, built as an MLOps portfolio project. Educational demo on simulated data. Do not make performance claims until a model has actually been trained and evaluated.

Requirement IDs in docstrings and comments (`FR01`–`FR10`, `ML01`) refer to the FRD at `docs/Turbofan_RUL_Functional_Requirements.docx`. Keep tagging new code with the requirement it implements.

Roadmap (from README): Day 1 data/labels/split/validation (in progress) → Day 2 features, baseline, Random Forest, MLflow → Day 3 FastAPI (`api/`) → Day 4 Streamlit (`app/`) → Day 5 Docker Compose, CI. `api/`, `app/`, `docs/` are empty placeholders right now.

## Commands

Run everything from the repo root. `configs/data.yaml` is read through a relative path, and the pipelines are run as modules.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                        # MLFLOW_TRACKING_URI, API_URL

python -m pipelines.download_data           # or: --zip path/to/archive.zip (when the NASA URL fails)
python -m pipelines.prepare_data            # writes to data/processed/

pytest -q                                   # all tests
pytest tests/test_labels.py -q              # one file
pytest tests/test_splits.py::test_name -q   # one test
```

No linter or formatter is configured yet.

## Architecture

Data flow: `download_data` → `data/raw/CMAPSS/` (FD001 files, readme, `checksums.json` with sha256) → `prepare_data` → `data/processed/` (`validation_report.json`, `split_manifest.json`, `train_labeled.parquet` or `.csv` depending on whether pyarrow is installed, `test_endpoints.csv`). Both `data/raw/` and `data/processed/` are git-ignored. Treat raw data as immutable.

`src/` holds the shared library that the pipelines and the future API/UI import:
- `schema.py`: the fixed 26-column layout (`engine_id`, `cycle`, `setting_1..3`, `sensor_1..21`). Every other module depends on these constants.
- `data_loader.py`: parses the whitespace-separated, headerless files. Trailing columns are dropped **only if they are entirely empty**. Otherwise it raises `DataValidationError`.
- `validation.py`: `validate_raw` raises `DataValidationError` (which carries an `.errors` list) on hard failures: missing columns, non-finite values, non-positive ids, duplicate `(engine_id, cycle)` keys, cycles that don't strictly increase. Cycle gaps and constant columns are reported as warnings, not errors. Rows are never dropped silently.
- `labels.py`: Training RUL = the engine's max cycle minus the current cycle, **uncapped** (FR06). For test data, the last observed cycle is **not** failure. `build_test_endpoint_labels` gives one labeled row per engine from `RUL_FD001.txt`. `add_test_trajectory_rul` produces per-row test truth for analysis only (FR10).
- `splits.py`: train/validation split by **whole engines** (never by row, which would leak), seeded by `configs/data.yaml` (`seed`, `val_fraction`), and persisted as a JSON manifest.

All dataset policy (paths, seed, val fraction, schema and parser versions, download URL) lives in `configs/data.yaml`. Add new policy there rather than hard-coding it.

Tests use synthetic histories from `tests/conftest.py::make_history({engine_id: n_cycles})`. The conftest also puts the repo root on `sys.path`, so they don't need the real NASA data.

## EDA notebook

`EDA/01_fd001_eda.ipynb` is self-contained. It reimplements loading, validation, labeling and the engine split inline instead of importing `src/`. Its split uses `N_VALIDATION_ENGINES`, not `split_manifest.json`, so its engine sets can differ from the pipeline's. It reads data from `FD001_DATA_DIR` or `data/raw/`. It needs matplotlib, which is not in `requirements.txt`. Its conclusions (constant columns to drop, sensors to keep, causal rolling features over windows of 5 and 10 with `min_periods=1` and `ddof=0`) are meant to feed into the Day 2 feature work.
