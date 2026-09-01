# Phase 1 & Phase 2 Documentation (Side‑by‑Side)

This document provides a concise side‑by‑side overview of Phase 1 and Phase 2 of the Financial‑Time‑Series‑ML project.

## Phase 1 – Data Ingestion and Validation
**Goal:** Ingest raw OHLCV data, validate and clean it, and produce a canonical OHLCV table ready for feature engineering.

**Key Tasks**
- Implement ingestion module (`src/financial_ml/data/ingestion.py`) with CSV/API handling, retries, and immutable storage.
- Extend preprocessing (`src/financial_ml/data/preprocessing.py`) for schema checks, duplicate removal, timestamp/corporate‑data normalization.
- Generate canonical OHLCV table (`data/processed/`).
- Write unit and integration tests; apply ruff linting fixes (B905, B007, F821).
- Document usage (`make ingest ASSET=<ticker>`).

**Deliverables**
- `src/financial_ml/data/ingestion.py`
- `src/financial_ml/data/preprocessing.py`
- Raw data in `data/raw/`, canonical table in `data/processed/`.
- 100 % test coverage, 0 ruff warnings.
- Updated README/Phase 1 doc.

**Leakage Rules**
- No shuffling; timestamps must be monotonic.
- No future‑data exposure in validation.
- Scalers to be fit per‑fold (prepared for later phases).

---

## Phase 2 – Feature Engineering and Leakage Tests
**Goal:** Engineer a rich set of predictive features from the canonical OHLCV table while guaranteeing that no feature leaks future information.

**Key Tasks**
- Implement technical indicator functions (`src/financial_ml/features/technical.py`).
- Build feature pipeline (`src/financial_ml/features/pipeline.py`) using `configs/features.yaml`.
- Define feature configuration (`configs/features.yaml`).
- Write unit tests for each feature and leakage tests.
- End‑to‑end integration test of the full pipeline.
- Apply ruff linting/formatting; fix any issues.
- Document usage (`make features`).

**Deliverables**
- `src/financial_ml/features/technical.py`
- `src/financial_ml/features/pipeline.py`
- `configs/features.yaml`
- Feature matrix stored under `data/features/`.
- 100 % test coverage, 0 ruff warnings.
- Updated documentation.

**Leakage Rules**
- Each feature may only use data up to its prediction timestamp.
- Rolling windows and lagged values must be computed from past data only.
- Leakage tests inject future values to verify safety.

---

### Side‑by‑Side Summary Table

| Aspect | Phase 1 | Phase 2 |
|--------|---------|---------|
| **Goal** | Ingest & validate raw OHLCV → canonical table | Engineer leakage‑safe features |
| **Core Tasks** | Ingestion, preprocessing, canonical table, tests | Feature functions, pipeline, config, tests |
| **Key Files** | `ingestion.py`, `preprocessing.py` | `technical.py`, `pipeline.py`, `features.yaml` |
| **Output** | `data/raw/`, `data/processed/` | `data/features/` (feature matrix) |
| **Leakage Checks** | Validation ensures no future data | Feature‑level leakage tests |
| **Code Quality** | ruff fixes (B905, B007, F821) | ruff linting, no errors |
| **Docs** | Phase 1 doc | Phase 2 doc |

*Both phases are required before moving to Phase 3 (dataset building).*