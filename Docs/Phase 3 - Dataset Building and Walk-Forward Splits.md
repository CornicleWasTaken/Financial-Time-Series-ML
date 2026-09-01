# Phase 3 – Dataset Building and Walk-Forward Splits

## Overview
Phase 3 implements the **dataset building** and **walk-forward split** stages of the financial time‑series ML pipeline. Building on the feature matrix generated in Phase 2, this phase creates chronological, leakage‑safe training/validation/test splits and prepares the data for model training. The walk-forward approach ensures that predictions are always made using information available at the prediction time, eliminating look‑ahead bias.

The work consists of:
1. **Dataset Builder** – constructs a dataset from the feature matrix, applies scaling, and saves metadata (feature schema, split dates, etc.).
2. **Walk‑Forward Splits** – implements expanding‑window (expanding training set) or rolling‑origin splits to simulate realistic forecasting.
3. **Leakage‑Safe Scaling** – fits scalers only on training folds, never on the full dataset.
5. **Metadata Versioning** – stores dataset/feature schema alongside model artifacts for reproducibility.
6. **Automated Tests** – unit tests for split logic and leakage checks, plus an end‑to‑end integration test.

All changes must be covered by automated tests and follow the project’s code‑quality standards.

## Goals
- **Create** a reproducible dataset with chronological splits that respect leakage constraints.
- **Implement** walk‑forward (expanding‑window) and optionally rolling‑origin splits.
- **Fit** scalers only on training data within each fold, never on the full dataset.
- **Persist** dataset/feature schema metadata for later model training and serving.
- **Achieve** 100 % test coverage for Phase 3 (unit + integration) with no linting errors.
- **Document** the dataset building workflow, configuration, and usage.

## Scope
- **Input**: Feature matrix from Phase 2 (`data/features/`).
- **Output**: 
  - Scaled dataset files (e.g., Parquet or CSV) stored under `data/dataset/`.
  - Metadata file (`dataset_metadata.json` or similar) containing split dates, feature schema, scaler parameters.
  - Training/validation/test splits ready for model training.
- **Exclusions**: No model training, no backtesting, no evaluation metrics in this phase.

## Requirements (Leakage Rules)
- **Chronological Ordering** – Splits must respect time order; no shuffling of observations.
- **No Future‑Data Leak** – Features used in a fold must be computed using only data up to the split point.
- **Scaling Within Folds** – Scalers are fitted on the training portion of each fold and applied to validation/test folds; never fitted on the full dataset.
- **Reproducibility** – Dataset and split definitions are versioned; the same inputs always produce identical splits.

## Tasks
| # | Task | Description |
|---|------|-------------|
| 1 | **Dataset Builder Module** | Implement `src/financial_ml/datasets/builder.py` that reads the feature matrix, applies per‑fold scaling, and writes the dataset (e.g., `data/dataset/train.parquet`, `data/dataset/val.parquet`, `data/dataset/test.parquet`). |
| 2 | **Walk‑Forward Split Logic** | Implement a function that creates expanding‑window splits (e.g., train on days 1‑N, then train on days 1‑N+1, etc.) or rolling‑origin splits, respecting the configured window sizes and horizon. |
| 3 | **Scaling per Fold** | Ensure that StandardScaler/MinMaxScaler (or custom) is fit only on training data of each fold and applied to validation/test folds. |
| 4 | **Dataset Metadata** | Create a JSON/YAML file (`dataset_metadata.json`) that records: split dates, feature schema (names, types), scaler parameters per fold, and any other relevant configuration. |
| 5 | **Unit Tests** | Write tests for: <br>• Dataset loading and splitting correctness <br>• Leakage checks (ensuring no future data leaks into training) <br>• Scaling behavior (fit on train, transform on val/test). |
| 5 | **Integration Test** | End‑to‑end test that runs the full pipeline: ingest → validate → feature engineering → dataset building → splits → metadata saving. Verify that the pipeline can be rerun with identical inputs. |
| 6 | **Code Quality** | Apply `ruff` linting/formatting, fix any B905, B007, F821 issues. |
| 7 | **Documentation** | Update `README.md` (or create Phase 3 doc) describing how to run dataset building (`make dataset`), the expected output layout, and how to configure walk‑forward splits via `configs/dataset.yaml`. |

## Deliverables
- `src/financial_ml/datasets/builder.py` – core dataset building logic with per‑fold scaling and split generation.
- `src/financial_ml/datasets/split.py` (optional) – utilities for walk‑forward split generation.
- `configs/dataset.yaml` – configuration for split windows, horizon, and scaling options.
- Dataset files (`data/dataset/*.parquet` or CSV) with clear naming.
- `dataset_metadata.json` (or `.yaml`) containing split dates, feature schema, scaler parameters.
- Comprehensive test suite (`tests/unit/` for builder/split functions, `tests/integration/` for full pipeline) with 100 % pass rate.
- Updated documentation describing Phase 3 workflow and usage.

## Definition of Done
- [x] All dataset builder and split code passes `ruff` linting with zero errors/warnings.
- [x] Unit tests cover split logic, leakage checks, and scaling behavior; `make test` runs successfully.
- [x] `make dataset` successfully builds the dataset and splits, producing the expected files and metadata.
- [x] Leakage tests pass (no future data appears in training folds).
- [x] Configuration (`dataset.yaml`) is versioned and used by the pipeline.
- [x] Documentation reflects the current state and usage instructions.
- [x] Phase 3 is marked **completed** in the project roadmap and ready for Phase 4 (model training + MLflow tracking) review.

---

*Prepared according to the Financial‑Time‑Series‑ML roadmap (Phase 3: dataset building + walk‑forward splits).*