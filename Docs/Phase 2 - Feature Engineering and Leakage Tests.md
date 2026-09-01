# Phase 2 – Feature Engineering and Leakage Tests

## Overview
Phase 2 implements the **feature engineering** stage of the financial time‑series ML pipeline and adds **leakage tests** to ensure that no feature uses future information. This phase builds on the cleaned canonical OHLCV table produced in Phase 1 and prepares features for dataset construction and model training.

The work consists of:
1. **Feature engineering** – calculating lagged returns, rolling statistics, technical indicators (SMA, EMA, RSI, MACD), price‑to‑MA ratios, volatility, volume features, and calendar features.
2. **Leakage‑safe design** – ensuring every feature is computed using only information available at the prediction timestamp (no look‑ahead bias).
3. **Leakage tests** – automated tests that verify feature functions do not inadvertently leak future data.
4. **Feature versioning** – storing feature definitions and parameters so that experiments are reproducible.

All changes must be covered by automated tests and follow the project’s code‑quality standards.

## Goals
- **Generate** a wide set of predictive features from the canonical OHLCV table.
- **Guarantee** that each feature is leakage‑free (uses only past and present data relative to the prediction point).
- **Provide** a configurable feature pipeline (via `configs/features.yaml`) that enables/disables feature groups and sets hyperparameters (windows, lags, etc.).
- **Achieve** 100 % test coverage for Phase 2 (unit tests for feature functions and leakage tests) with no linting errors.
- **Document** the feature engineering logic, configuration, and how to run the feature rebuild.

## Scope
- **Input**: Canonical OHLCV table (`data/processed/`) for all assets.
- **Output**: Feature matrices stored under `data/features/` (or similar) in a format suitable for the dataset builder (e.g., Parquet or CSV with multi‑index [timestamp, asset]).
- **Feature groups** (as outlined in the roadmap):
  - Lagged returns (e.g., `return_lag_1`, `return_lag_5`)
  - Rolling statistics (mean, std, min, max over windows)
  - Technical indicators: SMA, EMA, RSI, MACD
  - Price‑to‑moving‑average ratios (e.g., `price_sma_ratio_10`)
  - Volatility (rolling standard deviation of returns)
  - Volume features (log volume, volume rolling mean, volume‑price trends)
  - Calendar features (day of week, month, quarter, holiday flags)
- **Exclusions**: No dataset building, model training, or backtesting in this phase.

## Requirements (Leakage Rules)
- **No future data** – For a prediction at time `t`, features may only use data from timestamps ≤ `t`.
- **Rolling windows** – Must be computed using only past observations (e.g., a 10‑day SMA at `t` uses `t‑9` to `t`, not `t+1`).
- **Lagged features** – A lag of `k` periods uses the value at `t‑k`.
- **Technical indicators** – Standard definitions that inherently use only past data (e.g., RSI uses average gains/losses over past window).
- **Leakage tests** – Will attempt to inject future values and confirm that feature outputs do not change.

## Tasks
| # | Task | Description |
|---|------|-------------|
| 1 | **Technical feature module** | Implement `src/financial_ml/features/technical.py` with functions for each indicator (SMA, EMA, RSI, MACD, etc.) that accept a pandas Series and window/lag parameters and return a Series aligned with the input index. |
| 2 | **Feature pipeline** | Implement `src/financial_ml/features/pipeline.py` that reads the canonical OHLCV table, applies selected feature functions from `configs/features.yaml`, and writes the feature matrix. |
| 3 | **Configuration** | Create `configs/features.yaml` to define feature groups, their parameters, and toggles. |
| 4 | **Unit tests** | Write tests for each feature function (correctness, edge cases) and for leakage (using future‑value injection). |
| 5 | **Integration test** | End‑to‑end test that runs the feature pipeline on processed data and verifies the output shape, column names, and leakage‑free property. |
| 6 | **Code quality** | Apply `ruff` linting/formatting, fix any issues. |
| 7 | **Documentation** | Update `README.md` (or create Phase 2 doc) describing how to run feature engineering (`make features`) and the expected output layout. |

## Deliverables
- `src/financial_ml/features/technical.py` – pure functions for each technical indicator.
- `src/financial_ml/features/pipeline.py` – orchestrates feature generation based on config.
- `configs/features.yaml` – feature toggles and hyperparameters.
- Feature matrices stored under `data/features/` (e.g., `data/features/features.parquet`).
- A comprehensive test suite (`tests/unit/` for features and `tests/integration/` for pipeline) with 100 % pass rate.
- Updated documentation describing Phase 2 workflow and usage.

## Definition of Done
- [x] All feature engineering code passes `ruff` linting with zero errors/warnings.
- [x] Unit tests cover all public feature functions and leakage tests; `make test` runs successfully.
- [x] `make features` successfully builds features from the canonical table and stores them in the expected location.
- [x] Leakage tests pass (no feature shows sensitivity to future data).
- [x] Configuration (`features.yaml`) is versioned and used by the pipeline.
- [x] Documentation reflects the current state and usage instructions.
- [x] Phase 2 is marked **completed** in the project roadmap and ready for Phase 3 (dataset building + walk‑forward splits) review.

---

*Prepared according to the Financial‑Time‑Series‑ML roadmap (Phase 2: feature engineering + leakage tests).*