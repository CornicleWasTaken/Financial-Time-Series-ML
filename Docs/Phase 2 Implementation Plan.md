# Phase 2 - Feature Engineering and Leakage Tests (Implementation Plan)

## Overview
This plan outlines the implementation of Phase 2: Feature Engineering and Leakage Tests. The goal is to engineer a rich set of predictive features from the canonical OHLCV table while guaranteeing that no feature leaks future information.

## Key Components to Implement

1. **Technical Features Module** (`src/financial_ml/features/technical.py`)
   - Implement pure functions for technical indicators:
     - Simple Moving Average (SMA)
     - Exponential Moving Average (EMA)
     - Relative Strength Index (RSI)
     - Moving Average Convergence Divergence (MACD)
     - Price-to-Moving Average ratios
     - Volatility measures (rolling standard deviation of returns)
     - Volume-based features (log volume, volume rolling statistics)
     - Calendar features (day of week, month, quarter, holiday flags)
   - Each function should accept a pandas Series and parameters, return a Series aligned with input index

2. **Feature Pipeline** (`src/financial_ml/features/pipeline.py`)
   - Read feature configuration from `configs/features.yaml`
   - Apply selected features to canonical OHLCV data
   - Handle leakage safety by ensuring features only use past/present data
   - Output feature matrix to `data/features/`

3. **Feature Configuration** (`configs/features.yaml`)
   - Define feature toggles and parameters
   - Structure: list of feature definitions with name, enabled flag, and parameters
   - Support enabling/disabling feature groups

4. **Leakage Tests**
   - Write unit tests for each feature function to verify leakage safety
   - Create test cases that inject future values and verify no leakage occurs
   - Test edge cases (empty data, missing values, boundary conditions)

5. **Integration Test**
   - End-to-end test that runs ingestion → preprocessing → feature engineering → feature matrix output
   - Verify feature matrix structure, column names, and leakage safety

## Implementation Approach

### 1. Technical Features Module (`src/financial_ml/features/technical.py`)
- Create pure functions for each indicator
- Each function should:
  - Accept a pandas Series (price/return/volume) and parameters
  - Return a Series with same index as input
  - Handle edge cases (NaN, insufficient data)
  - Be leakage-safe by design (only use past data)

### 2. Feature Pipeline (`src/financial_ml/features/pipeline.py`)
- Load feature configuration from `configs/features.yaml`
- For each enabled feature:
  - Determine the data column(s) it operates on
  - Call appropriate technical function with correct parameters
  - Ensure temporal integrity (features use only data up to current timestamp)
- Output: feature matrix with original timestamp column + all engineered features

### 3. Feature Configuration (`configs/features.yaml`)
- Define feature structure:
  ```yaml
  features:
    - name: "return_lag_1"
      enabled: true
      params: {}
    - name: "sma_10"
      enabled: true
      params: { "window": 10 }
    - name: "rsi_14"
      enabled: true
      params: { "period": 14 }
    # ... additional features
  ```

### 4. Testing Strategy
- Unit tests for each technical function (correctness, edge cases)
- Leakage tests that verify no future data influence
- Integration test for full pipeline
- All tests must pass with 0 ruff warnings

## Deliverables

- `src/financial_ml/features/technical.py` - Pure functions for technical indicators
- `src/financial_ml/features/pipeline.py` - Feature engineering orchestrator
- `configs/features.yaml` - Feature configuration definition
- Comprehensive test suite (`tests/unit/`, `tests/integration/`)
- Updated documentation (Phase 2 doc)

## Leakage Safety Requirements
- All features must be computable using only data available at prediction time
- No look-ahead bias in any feature calculation
- Rolling windows must use only past observations
- Lagged features must use values from previous timestamps only

## Dependencies
- pandas (for data manipulation)
- numpy (for mathematical operations)
- Existing data structure from Phase 1 (canonical OHLCV table)

## Success Criteria
- All features pass unit tests and leakage checks
- `make features` command successfully builds feature matrix
- 100% test coverage for Phase 2
- 0 ruff warnings after linting
- Feature matrix stored in `data/features/` with proper structure