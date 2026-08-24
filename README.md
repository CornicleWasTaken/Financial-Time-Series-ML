# financial-ml

Production-style financial time-series machine-learning platform — an
educational engineering project, not investment advice.

The end-to-end system ingests historical OHLCV data, builds leakage-safe
features, trains and compares multiple models (baselines, XGBoost, PyTorch
LSTM, optional Transformer), evaluates them with walk-forward validation,
backtests a strategy with realistic transaction costs, and serves predictions
through FastAPI — backed by PostgreSQL and visualised in a dashboard.

The full architecture, build plan, evaluation metrics, leakage rules and
definition of done live in
[`Docs/financial_time_series_ml_system_roadmap.docx`](Docs/financial_time_series_ml_system_roadmap.docx).
Read that document first; this README is just the operator's guide.

---

## Status

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Repo skeleton, tooling, configuration, Dockerfile, smoke test | ✅ landed |
| 1 | Market-data ingestion + validation | ⏳ next |
| 2 | Exploratory data analysis | ⏳ |
| 3 | Feature engineering | ⏳ |
| 4 | Dataset builder + walk-forward splits | ⏳ |
| 5 | Baselines + common model interface | ⏳ |
| 6 | XGBoost + MLflow tracking | ⏳ |
| 7 | LSTM | ⏳ |
| 8 | Optional Transformer | ⏳ |
| 9 | Backtesting | ⏳ |
| 10 | Model registry / promotion | ⏳ |
| 11 | FastAPI service | ⏳ |
| 12 | PostgreSQL persistence | ⏳ |
| 13 | Dashboard | ⏳ |
| 14 | MLOps / CI / monitoring | ⏳ |
| 15 | Final polish | ⏳ |

The roadmap calls for one phase at a time, with tests and a commit before
moving on. Phase 0 commits the skeleton.

---

## Goal & non-goals

**Goal.** Demonstrate a leakage-safe, time-aware ML platform that compares
classical and deep-learning forecasting models end-to-end against a
transaction-cost-aware backtest, tracks experiments in MLflow, serves
predictions through an API and presents results in a dashboard.

**Non-goals.**

- This is **not** an investment system and is **not** financial advice.
- The system does **not** aim to predict exact prices; the primary target is
  next-trading-day return. Direction, derived price and calibrated confidence
  are secondary outputs.
- Single-asset hard-coding is explicitly avoided — new assets are added by
  editing `configs/assets.yaml`, not core code.
- "Accuracy" is intentionally avoided as the primary metric for a continuous
  return forecast; see the evaluation section of the roadmap.

---

## Layout

```
financial-ml/
├── pyproject.toml          # Python project + tooling (ruff, pytest)
├── Dockerfile              # multi-stage build (Phase 0)
├── docker-compose.yml      # app + (future) postgres + mlflow
├── Makefile                # dev commands
├── .env.example            # documented env vars
├── configs/                # assets.yaml, features.yaml, models.yaml
├── data/{raw,processed,features}/
├── notebooks/exploration/  # EDA only — never part of the training pipeline
├── src/financial_ml/
│   ├── config.py           # env + YAML config loader (Phase 0)
│   ├── data/  features/  datasets/
│   ├── models/  evaluation/  backtesting/  services/
├── api/                    # FastAPI app (Phase 11)
├── dashboard/              # visualisation (Phase 13)
├── models/                 # serialised artifacts
├── mlruns/                 # MLflow tracking store
├── tests/{unit,integration}/
└── Docs/                   # roadmap (source of truth)
```

---

## Setup

Requires Python 3.11 or newer (3.12 is the Dockerfile baseline; 3.14 also
works once dependency wheels catch up — see *Notes* below).

```bash
# 1. Create + activate a virtualenv (any tool works; the Makefile is
#    tool-agnostic and uses the active `python`).
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 2. Install the project + dev tooling
make install

# 3. Copy and edit local env vars (optional — defaults are fine for Phase 0)
cp .env.example .env
```

If you prefer `uv`, the `pyproject.toml` is fully compatible:

```bash
uv sync --extra dev
```

---

## Common commands

| Command | What it does |
|---------|-------------|
| `make install` | Install deps into the active venv (`pip install -e ".[dev]"`). |
| `make lint` | Run ruff lint over `src/` and `tests/`. |
| `make format` | Auto-format with ruff. |
| `make test` | Run the full pytest suite. |
| `make test-one TEST=...` | Run a single test (pytest node-id). |
| `make ingest ASSET=AAPL` | *(Phase 1)* refresh raw data. |
| `make features` | *(Phase 3)* rebuild features. |
| `make train MODEL=xgboost` | *(Phase 6+)* tracked training run. |
| `make backtest` | *(Phase 9)* backtest entrypoint. |
| `make serve` | *(Phase 11)* FastAPI via uvicorn. |
| `make docker-up` | Build and run the stack. |

Targets marked *(Phase N)* exist as deliberate stubs that fail with a TODO
message — they become real as later phases land.

---

## Configuration

Two layers:

1. **Environment variables** — runtime settings (log level, data directory,
   MLflow tracking URI, future database URL). Loaded by Pydantic Settings.
   See `.env.example` for the full list.
2. **YAML files under `configs/`** — static configuration consumed by the
   pipeline:
   - `assets.yaml` — the tradable universe. Add a new asset by appending a
     row; no core code changes required (this is one of the definition-of-done
     properties).
   - `features.yaml` — feature engineering toggles. Populated in Phase 3.
   - `models.yaml` — model registry entries. Populated in Phase 5+.

The `financial_ml.config.load_config()` helper reads both layers, validates
them with Pydantic and memoises the result for the lifetime of the process.
Tests use the `app_config` fixture (see `tests/conftest.py`).

---

## Tests

```bash
make test                  # full suite
make test-one TEST=...     # single test
```

Phase 0 ships four smoke tests under `tests/unit/test_smoke.py` that verify
the package imports, the repo paths resolve, configuration loads end-to-end,
and at least one asset is enabled. Later phases add unit tests for every
module and integration tests for the API / database stack.

Tests are strict: any unconfigured warning is promoted to an error via
`filterwarnings = ["error", ...]` in `pyproject.toml`. If you genuinely need
to suppress one, scope it narrowly.

---

## Notes

- **Python 3.14.** Several ML libraries (notably `torch`) only ship wheels for
  older Python versions at any given moment. The Dockerfile uses 3.12 as the
  baseline; if you develop on 3.14 and a deep-learning dep fails to install,
  drop down to 3.12 until upstream catches up.
- **Educational framing.** API responses, dashboard labels and any
  user-facing copy must label outputs as model results, not advice.
- **EDAs vs pipeline.** Exploratory notebooks live in
  `notebooks/exploration/`. They are explicitly excluded from the
  reproducible training pipeline.
- **Roadmap is the source of truth.** When in doubt about scope or ordering,
  defer to `Docs/financial_time_series_ml_system_roadmap.docx`.

---

## License

MIT — see `LICENSE` (added in Phase 15).
