# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository status

This is a greenfield project. As of the current state, the only checked-in artifact is the planning document:

- `Docs/financial_time_series_ml_system_roadmap.docx` — full project roadmap, architecture, phased build plan, evaluation metrics, leakage rules, and definition of done.

All implementation decisions should be cross-checked against this roadmap. The roadmap explicitly says: implement incrementally, one phase at a time, and require tests along with commits to the github repository under 'main' branch before moving to the next phase.

## Project intent (from the roadmap)

A production-style financial time-series ML platform, not a forecasting notebook. End-to-end pipeline:

market data ingestion → validation/cleaning → feature engineering → leakage-safe dataset building → model training (baselines, XGBoost/LightGBM, PyTorch LSTM, optional Transformer) → walk-forward evaluation + MLflow tracking → backtesting with transaction costs → model registry → FastAPI service → PostgreSQL persistence → dashboard.

Primary prediction target is **next-trading-day return** (not price). Secondary outputs: direction, derived price, calibrated confidence. Multiple assets are configured, not hard-coded. Educational engineering project — not investment advice.

## High-level architecture

The pipeline flows top-down (see the diagram in section 4 of the roadmap). Layers and their responsibilities:

1. **Data ingestion** (`src/financial_ml/data/sources.py`, `ingestion.py`) — pulls from configured market-data source (API or CSV), applies retries/rate limits, stores raw data immutably.
2. **Validation & cleaning** (`data/validation.py`, `preprocessing.py`) — schema, duplicates, missing observations, timestamp/corporate-data normalization. Output is a canonical OHLCV table.
3. **Feature engineering** (`features/technical.py`, `pipeline.py`) — lagged returns, rolling stats, SMA/EMA, price-to-MA ratios, RSI, MACD, volatility, volume, calendar features. All features must use only information available at prediction time. Feature definitions are versioned.
4. **Dataset builder** (`datasets/builder.py`) — target = next-period return, chronological splits, leakage checks, walk-forward / expanding-window splits, scalers fit on training portion of each fold only, dataset/feature-schema metadata saved.
5. **Models** (`models/base.py`, `baseline.py`, `xgboost_model.py`, `lstm.py`) — all models implement a common interface so evaluation and serving treat them uniformly. Baselines first (naive return, linear, random forest), then XGBoost (Days 12–15), then LSTM (Days 15–19), optional Transformer (Days 19–22).
6. **Evaluation** (`evaluation/metrics.py`, `walk_forward.py`, `calibration.py`) — walk-forward CV, MAE/RMSE, directional accuracy, correlation, calibration, MLflow logging.
7. **Backtesting** (`backtesting/engine.py`, `costs.py`, `metrics.py`) — signals → position sizing → fees/slippage → equity curve → Sharpe, drawdown, vs. buy-and-hold. Backtest is implemented independently from model training.
8. **Model registry / artifacts** — best model, preprocessing, feature schema; reproducible promote workflow; training-data date range and version metadata.
9. **FastAPI service** (`api/main.py`, `api/routes/`) — endpoints for health, assets, prediction, backtest, metrics. Pydantic request validation. Loads artifacts at startup. Structured logs.
10. **Persistence** — PostgreSQL stores assets, ingestion runs, model runs, predictions, backtest summaries. Migrations + indexes. Raw market data lives separately from app metadata. Data-access layer, not inline SQL in routes.
11. **Dashboard** — asset selector, price/return charts, predicted vs actual, feature importance, model comparison, equity curve, drawdown, run history. Clearly label outputs as model results, not financial advice.
12. **Cross-cutting** — Git, tests, structured logging, configuration, Docker, CI/CD, monitoring.

## Target repository layout (from roadmap §5)

```
financial-ml-system/
├── pyproject.toml          # project + tooling config
├── .env.example            # documented env vars
├── docker-compose.yml
├── Dockerfile
├── Makefile                # common dev commands
├── configs/                # assets.yaml, features.yaml, models.yaml
├── data/{raw,processed,features}/
├── notebooks/exploration/  # EDA only — kept separate from training pipeline
├── src/financial_ml/
│   ├── config.py
│   ├── data/        # sources, ingestion, validation, preprocessing
│   ├── features/    # technical, pipeline
│   ├── datasets/    # builder (chronological splits, leakage checks)
│   ├── models/      # base, baseline, xgboost_model, lstm
│   ├── evaluation/  # metrics, walk_forward, calibration
│   ├── backtesting/ # engine, costs, metrics
│   └── services/    # prediction
├── api/             # main.py + routes/
├── dashboard/
├── models/          # serialized artifacts
├── mlruns/          # MLflow tracking
├── tests/{unit,integration}/
└── .github/workflows/ci.yml
```

## Intended development workflow

### Build order (mandatory — from roadmap §10)

Work one phase at a time. Do not skip ahead. Tests required before advancing.

1. Repo skeleton only (no ML).
2. Phase 1: data ingestion + validation, with tests.
3. Feature engineering + leakage tests.
4. Dataset building + walk-forward splits.
5. Baselines + common model interface.
6. XGBoost + MLflow tracking.
7. LSTM only after tabular pipeline is stable.
8. Backtesting independently from model training.
9. FastAPI + persistence.
10. Dashboard.
11. Docker, CI, monitoring, final docs.

**First milestone (roadmap §11):** download historical OHLCV → validate → leakage-safe features → chronological dataset → naive baseline + XGBoost → walk-forward predictions → metrics. End-to-end before anything else.

### Intended commands

The Makefile has not been written yet. When implementing the skeleton, expose (at minimum) these as `make` targets:

- `make install` — install dependencies from `pyproject.toml` into the project venv.
- `make lint` — run the configured linter/formatter (e.g. ruff format + ruff check, or black + flake8 — pick one and stick to it).
- `make test` — run the full pytest suite.
- `make test-one TEST=tests/unit/test_<name>.py::test_<case>` — run a single test (pytest node-id selection).
- `make ingest ASSET=<ticker>` — refresh raw data for an asset.
- `make features` — rebuild features from validated raw data.
- `make train MODEL=xgboost` — run a tracked training run (MLflow).
- `make backtest` — run the backtester on the registered model.
- `make serve` — start the FastAPI service locally (uvicorn).
- `make docker-up` — bring up the full stack via `docker compose up`.

Until the Makefile exists, invoke the underlying tools directly (`uv run …`, `pytest -k <name>`, `uvicorn api.main:app --reload`, `docker compose up`, etc.) — the roadmap does not yet lock a runner.

## Hard rules — data leakage (roadmap §8)

These are non-negotiable and apply to every PR touching data, features, datasets, or evaluation:

- Never randomly shuffle chronological observations for main evaluation.
- Never let a feature see future prices, future volume, or future-derived indicators.
- Fit scalers / transformations inside each training fold, not across the full dataset.
- Never tune hyperparameters on the final test period.
- Backtests must include realistic transaction costs and slippage where appropriate.
- Keep a final untouched test period for the last evaluation.
- Add explicit leakage tests for rolling/lag feature calculations.

## Definition of done (roadmap §9)

The system is "done" when **all** of the following hold:

- New asset added via configuration only — no core code changes.
- Pipeline can ingest, validate, and transform historical data reproducibly.
- At least three models trainable through the common interface.
- Walk-forward evaluation implemented and tested.
- Backtest reproducible from saved predictions + configuration.
- Selected model and preprocessing artifacts versioned.
- FastAPI serves predictions from the saved model.
- Dashboard visualizes predictions, metrics, and backtest results.
- Automated tests + CI pass.
- `docker compose` brings the app up locally.
- README covers setup, architecture, methodology, limitations, example usage.

## Evaluation metrics (roadmap §7)

- Regression: MAE, RMSE.
- Direction: directional accuracy, confusion matrix (only when a direction classifier is explicitly used).
- Forecast quality: correlation between predicted and realized returns; calibration for probabilistic outputs.
- Strategy: cumulative return, annualized volatility, Sharpe, max drawdown, turnover, transaction costs.
- System: prediction latency, API error rate, ingestion success rate, model version.

Do not use plain "accuracy" as the primary metric for a continuous return forecast.

## Operational notes for future Claude sessions

- Keep EDA in `notebooks/exploration/`; never let it leak into the reproducible training pipeline.
- Implement backtesting independently from model training — predictions + config should be sufficient to reproduce results.
- MLflow is the experiment tracker. Persist preprocessing + feature schema alongside the model artifact.
- The roadmap calls out that this is an educational engineering project, not an investment system — preserve that framing in any user-facing copy (API responses, dashboard labels).
- When in doubt about scope or ordering, defer to the roadmap document; it is the source of truth until superseded.
