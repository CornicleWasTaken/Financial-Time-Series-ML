# Phase 1 – Data Ingestion and Validation

## Overview
Phase 1 implements the **data ingestion** and **validation/cleaning** stages of the financial time‑series ML pipeline. This foundation is required before any feature engineering, dataset building, or modeling can begin. The work consists of:

1. **Ingestion** – pulling historical OHLCV data from configured market‑data sources (CSV files or APIs) with retries, rate‑limit handling, and immutable storage of raw data.
2. **Validation & Cleaning** – schema enforcement, duplicate removal, missing‑value handling, timestamp/corporate‑data normalization, and production of a canonical OHLCV table ready for feature engineering.

All changes must be covered by automated tests and follow the project’s code‑quality standards (ruff linting, formatting, and strict typing).

## Goals
- **Ingest** historical OHLCV data for all configured assets into `data/raw/` immutably.
- **Validate** that raw data conforms to the expected schema (correct types, no duplicates, proper timestamps).
- **Clean** data to produce a canonical OHLCV table (`data/processed/`) that can be used downstream.
- **Achieve** 100 % test coverage for Phase 1 (unit + integration tests) with no linting errors.
- **Document** the ingestion and validation logic, including configuration via `configs/assets.yaml`.

## Scope
- **Supported Sources**: CSV files (as described in the roadmap) and any future API adapters.
- **Data Range**: Historical OHLCV for all assets listed in `configs/assets.yaml`.
- **Output**: 
  - Raw immutable files stored under `data/raw/`.
  - Cleaned canonical OHLCV table stored under `data/processed/`.
- **Exclusions**: No feature engineering, dataset building, or model training in this phase.

## Requirements (Leakage Rules)
- **Chronological Integrity** – No shuffling of observations; timestamps must be monotonic within each asset.
- **No Future‑Data Leak** – Features may only use information available at prediction time; validation must not expose future prices/volume.
- **Scoping** – Scalers and transformations are fit **inside each training fold** (not on the full dataset) – not applicable yet but the code structure should be prepared for later phases.
- **Backtest Reproducibility** – All ingested data must be versioned so that backtests can be rerun from the same raw files.

## Tasks
| # | Task | Description |
|---|------|-------------|
| 1 | **Ingestion Module** | Implement `src/financial_ml/data/ingestion.py` to read CSV files (and later API endpoints) with retry logic, rate‑limit handling, and write raw data to `data/raw/` without modification. |
| 2 | **Preprocessing Validation** | Extend `src/financial_ml/data/preprocessing.py` to enforce schema, remove duplicates, handle missing timestamps/values, and normalize corporate actions (splits, dividends). |
| 3 | **Canonical Table Generation** | Ensure the output of preprocessing is a clean, timestamp‑sorted OHLCV table saved to `data/processed/`. |
| 4 | **Unit Tests** | Write tests for ingestion (CSV parsing, API error handling) and preprocessing (schema validation, duplicate removal, missing‑value handling). |
| 5 | **Integration Tests** | End‑to‑end test that runs ingestion → validation → canonical table for a sample asset and verifies output correctness. |
| 6 | **Code Quality** | Apply `ruff` linting/formatting, fix any B905 (zip strict) and B007 (unused variable) issues identified in the Phase 1 fixes summary. |
| 7 | **Documentation** | Update `README.md` (or create Phase 1 doc) describing how to run ingestion (`make ingest ASSET=<ticker>`) and the expected data layout. |

## Deliverables
- `src/financial_ml/data/ingestion.py` (or equivalent) with robust CSV/API handling.
- `src/financial_ml/data/preprocessing.py` implementing schema checks, duplicate removal, timestamp/corporate‑data normalization, and production of a canonical OHLCV table.
- All raw data stored under `data/raw/` (immutable).
- All cleaned data under `data/processed/`.
- A comprehensive test suite (`tests/unit/` and `tests/integration/`) with 100 % pass rate.
- Updated documentation describing Phase 1 workflow and usage.

## Definition of Done
- [x] All ingestion and validation code passes `ruff` linting with zero errors/warnings.
- [x] Unit and integration tests cover all public functions and edge cases; `make test` runs successfully.
- [x] `make ingest ASSET=<ticker>` successfully ingests data for a configured ticker and produces a verified canonical table.
- [x] Raw data files are immutable (no post‑ingestion modifications allowed).
- [x] Documentation reflects the current state and usage instructions.
- [x] Phase 1 is marked **completed** in the project roadmap and ready for Phase 2 (feature engineering) review.

---

*Prepared according to the Financial‑Time‑Series‑ML roadmap (Phase 1: data ingestion + validation).*