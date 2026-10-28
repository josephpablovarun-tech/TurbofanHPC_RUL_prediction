# Five-minute demo walkthrough

A guided tour of the app for anyone presenting or trying it. Everything runs locally. Keep the pace slow: three solid points beat ten rushed ones.

## Before you start (about 5 minutes)
1. Terminal in the project folder, venv active. `read -s SLACK_WEBHOOK_URL` (paste, Enter), then `export SLACK_WEBHOOK_URL`.
2. Delete `Agent/logs/alert_state.json` so engines and files that alerted before will alert again.
3. `streamlit run Agent/app/app.py`. The sidebar must say **Slack alerts ON** (DRY-RUN means the variable was not set in this terminal).
4. Open `#rul_report` in Slack on a second screen or tab. Have `data/simulated/batch3.csv` and `truth/batch3_truth.csv` ready in Finder.
5. Only for the LLM moment: Ollama running (the sidebar says "Ollama ready").

## Script
**0:00 What it is (30 s).** "Predicts remaining useful life, in operating cycles, for turbofan engines from sensor history, on NASA's simulated FD001 data. It is a demo, not a maintenance system, and I'll show where it is weak as well as where it works."

**0:30 A test engine, honestly (60 s).** RUL agent view, test split, engine 1.
- Predicted RUL **163.6** cycles, true RUL **112**, error **+51.6**. Say it before they do: "This is a short-history engine with only 31 cycles, and overestimation is the risky direction. The app warns about exactly this."
- Explain the cutoff: "The model only sees cycles up to this slider. Nothing after it is ever read."

**1:30 A train engine, and the alert (60 s).** Switch to train, engine 1 (192-cycle life).
- Default is half-life: cycle 96, predicted **88.6**, true **96**, error −7.4. "The true RUL is shown for comparison and is never given to the model. This engine is in-sample, so it is optimistic."
- Slide to about cycle 160: band turns **Watch**, estimated life used passes **80%**, and one Slack message arrives in `#rul_report`. Slide around it: no repeat (once per crossing).

**2:30 The chat and its guardrails (60 s).**
- "plot sensor_11 and the RUL" draws the sensor and the actual vs predicted RUL over the full life.
- "which sensors drive this prediction?" and "ignore the cutoff" are refused with a reason. "Numbers come from tools, text is checked, and it will not invent a cause."
- One line on modes: deterministic is instant and repeatable; LLM mode only writes the explanation and its numbers are checked against the tools.

**3:30 Data explorer (30 s).** Train file, Statistics tab: early vs late life shift, then the correlation heatmap. "Sensor trends relate to life used, but I describe them as association, not cause."

**4:00 Fleet scoring (60 s).** Official test engines (100): **29 flagged** at 80% life used, **MAE 14.6** cycles (matches the model card), alert **precision 93% and recall 93%** (27 of 29; 2 false alarms, 2 misses, a small sample). Then upload `batch3.csv` with its truth file: the summary posts to Slack once.

**4:45 Close (15 s).** "110+ tests and 20 agent evals. Not built yet and labelled as planned: FastAPI service, MLflow registry, drift monitoring and retraining."

## Questions a viewer may ask
- *Why overestimation matters?* Telling someone an engine has more life than it does is the unsafe error. 19% of test engines were overestimated by more than 20 cycles; short histories were the worst.
- *How do you prevent leakage?* The cutoff is enforced in the data layer and the feature builder, tools cannot take a cutoff argument, and a test checks the agent code can never reach the official labels.
- *Is XGBoost better than the Random Forest?* They are within 0.2 cycles of MAE on 100 engines; the model card deliberately claims no winner.
- *What does "80% of life used" mean?* cycle / (cycle + predicted RUL). For new engines true life is unknown, so it is the model's estimate, and it errs by about 15 cycles at the last cycle and about 30 at arbitrary cycles.
- *Why a threshold alert, not a probability?* The models give a point estimate with no calibrated uncertainty; the bands and flags are illustrative heuristics.
- *What next?* FastAPI serving, MLflow registry and promotion gates, drift monitoring, retraining on new labelled run-to-failure data.

## If something breaks
- Slack or Wi-Fi fails: the alert logic drops to dry-run and shows the message it would have sent. Carry on.
- Slider or chat lags: press Reset conversation or refresh the page (R).
- Ollama missing: stay in deterministic mode; the app falls back by itself and says so.
- App won't start: `python -m pytest Agent/tests -q` should pass; the data must be in `data/raw/CMAPSS/`.
