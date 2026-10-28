# Turbofan RUL multi-agent assistant

An agent layer on top of the saved models (XGBoost primary, Random Forest support) for NASA C-MAPSS FD001.
It answers questions about one selected engine at one selected cutoff cycle, using tools instead of free-form guesses.

## How it works
dashboard selection -> immutable `RequestContext` -> deterministic supervisor -> **engine-analysis agent** (`predict_rul`, `get_sensor_trends`, `get_model_metadata`)
and/or **documentation agent** (`search_documents`) -> harness -> validator -> templated response with trace.

* The engine, split and cutoff come from the sidebar only. Tool inputs have no such fields, so neither the user text nor the LLM can change them.
  The cutoff is enforced in the data layer (rows after the cutoff are never read) and again in the feature builder.
* The harness enforces per-agent tool allowlists, a 5-call budget, a 20 s per-tool timeout, a 120 s deadline, and rejects any result whose identifiers differ from the request. Every call is traced (`Agent/logs/traces.jsonl`).
* Two modes per agent: `deterministic` (fixed plan, no LLM) and `llm` (Ollama tool-calling + structured JSON). If the LLM is down or returns bad output, the agent falls back and says so (`deterministic_fallback`).
* Numbers shown come from tool fields. Model-written text is validated (numbers must exist in tool results, citations must be real chunk ids, no causal or maintenance wording) or dropped.
* Retrieval is lexical BM25 over `Agent/docs` (one chunk per section: `DG-01..08`, `MC-01..09`, `SG-01..05`). There are no embeddings.

## Setup
```bash
pip install -r requirements.txt
ollama pull qwen2.5:7b        # only needed for llm mode; set OLLAMA_MODEL / OLLAMA_HOST to override
```
Needs `data/raw/CMAPSS/train_FD001.txt`, `test_FD001.txt` and `model_save/xgboost.joblib`, `random_forest.joblib`. Settings: `Agent/configs/agent.yaml`.

## Run
```bash
pytest -q Agent/tests                 # synthetic fixtures; test_real_artifacts.py runs only if your real files exist
python -m Agent.evals.run             # 20 cases, deterministic; add --llm for Ollama
streamlit run Agent/app/app.py
```
Run `pytest Agent/tests/test_real_artifacts.py -v` first on your machine: it compares the agent's predictions with `artifacts/model/check_vs_RUL_FD001_per_engine_errors.csv`.

## Train-engine life view (retrospective)

When the dataset split is `train`, the page shows a panel with the engine's total life, its data role (in-sample or held out) and the sensors most associated with progression to failure (Spearman rank correlation with the cycle number over the full run). Charts are requested from the chat, for example `plot sensor_15 and the RUL` (that sensor over the full life, plus actual vs model-predicted RUL at every cycle) or `which sensors contributed to the degradation?`.

* It deliberately reads past the selected cutoff, so it is a separate path (`tools/life_view.py`), not one of the cutoff-bound agent tools. It is refused for the test split, labelled retrospective wherever shown, and ordinary answers keep the strict cutoff. An explicit "ignore the cutoff" request is still refused.
* "Associated with progression" is an observed correlation, not a cause. Sensor names and units are not verified.
* Predictions on train engines the models were fitted on are in-sample and optimistic. The predicted-RUL line is omitted, not invented, when the prediction outage toggle is on.

## Data explorer and true RUL (evaluation displays)

The page has two views: **RUL agent** and **Data explorer**. The explorer lets you pick one of the three FD001 files (`train_FD001`, `test_FD001`, `RUL_FD001`) and explore it with Overview, Statistics, Distributions, Trends, Correlations and Data quality sub-tabs (the label file has its own Overview, Distribution and Joined-with-test sub-tabs). The computations are pure pandas in `tools/explore.py`.

The estimate box shows the **true RUL** next to the prediction for both splits. For a train engine it is the last cycle minus the current cycle. For a test engine it comes from `RUL_FD001.txt`: the label is the true RUL at the last recorded cycle, and at an earlier cycle it is the label plus the cycles stepped back. This is an evaluation display only: the label file is read through `DataStore.rul_labels()` / `true_rul_at()`, which the agent tools, orchestration and chat never call (a test enforces it), so no label reaches the model or the agent.

## Slack alert at 80% of estimated life

Whenever the model predicts an RUL, the service computes **estimated life used = cycle / (cycle + predicted RUL)**. When that reaches 80% (`alerts.life_used_threshold`), it posts one message to Slack saying the engine is flagged for an inspection review (demo, not a maintenance decision). It alerts **once per crossing** per split, engine and model version, and re-arms only after the estimate falls below 75% (`alerts.rearm_below`), so a moving slider does not post repeatedly. For a test engine the total life is unknown, so this is always an estimate. Messages never contain the true RUL or the official labels.

* **Setup:** in Slack create an Incoming Webhook for the target channel, then start the app with the URL in the environment: `export SLACK_WEBHOOK_URL='https://hooks.slack.com/services/...'` followed by `streamlit run Agent/app/app.py`. Never put the URL in code, config or Git.
* **Without a webhook** the service runs in dry-run mode: the sidebar says so, the decision is shown on the page and logged to `Agent/logs/alerts.jsonl`, and nothing is sent.
* Evals and tests switch alerts off, so they never post to Slack even if the variable is set.
* State (which engines have alerted) is kept in `Agent/logs/alert_state.json`; delete it to re-arm every engine.

## Fleet scoring (batch import)

The **Fleet scoring** view imports readings for many engines (a CSV with a header, or 26 whitespace-separated columns like `test_FD001`), scores the **last cycle of each engine** with the same model, and flags engines at or above 80% estimated life used. Checks follow the requirements doc: a problem with the whole file (missing column, wrong column count, over 10 MB or 50,000 rows, non-whole engine or cycle numbers) rejects the file with a message; a problem with one engine (missing or non-finite values, duplicate or out-of-order cycles) rejects only that engine and the rest are scored. Results can be downloaded as CSV. Optional true-RUL labels (one number per engine, or a CSV with `engine_id` and `true_rul`) add an evaluation table (MAE, mean error, alert precision and recall); labels never reach the model or Slack. With alerts configured, **one summary message** listing the flagged engines is posted once per file and model version.

Simulated batches: `python -m pipelines.simulate_fleet --out data/simulated` writes `batch1.csv` (healthy fleet), `batch2.csv` (mixed) and `batch3.csv` (ageing fleet), 100 engines each, built from real train and test engines cut off at chosen points with small sensor noise, plus `truth/batchN_truth.csv` (true remaining life, optional labels). The output is deterministic. Engines derived from train engines may be ones the models were fitted on, so their results are optimistic; these are simulations of the same simulated fleet, not independent real data.

## Honest scope
* Sensors are generic identifiers (`sensor_1..21`); units and physical meaning are not verified.
* Point estimates only, no uncertainty. Model agreement and the Normal/Watch/Review bands are uncalibrated, illustrative heuristics.
* Endpoint MAE is about 14.6 cycles but about 31 cycles across arbitrary cutoffs; errors are largest with long remaining life and short history.
* Engines 1-70 style training engines give in-sample (optimistic) results; the app labels them.
* No maintenance, repair, diagnosis or airworthiness advice: such questions are declined with a pointer to `MC-09`.
* `joblib` files execute code on load; only load models you created.
* The test suite uses synthetic data and small stand-in models. Quality numbers come from `Agent/configs/model_metrics.json` (copied from your saved evaluation files), not from re-running anything.
