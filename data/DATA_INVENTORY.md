# Dataset Inventory & Preprocessing Specification

**Module:** Data Engineering & Time-Series Preprocessing  
**Owner:** Person 1 (Data and Preprocessing Owner)  
**Branch:** `feature/data-preprocessing`  
**Governing Document:** [`SEMESTER_V_WORK_PLAN.md`](../SEMESTER_V_WORK_PLAN.md) (Revision: `2026-10-26 15:28:25 IST`)  
**Core Clone Reference:** [`PROJECT_EXPLANATION.md`](../PROJECT_EXPLANATION.md)  
**Status:** Pipeline Implementation & Sample Validation Complete; Full Universe Awaiting Team/Supervisor Confirmation  

---

## 1. Executive Summary & Legacy Isolation

This document records the current data engineering state for **Person 1** under Semester V. To resolve the timeline difference between the original paper's short benchmark and the SRS requirement, two distinct, isolated data contracts are defined:

1. **Contract A (Legacy Baseline Reproduction):** Preserves the exact 2024 monthly `Close` price and return CSVs in [`yfinance/`](../yfinance/) and model views in [`responses/`](../responses/) covering **June 1, 2024 to December 31, 2024**.
   * *Legacy Integrity:* [`responses/common_stocks.json`](../responses/common_stocks.json), all monthly model responses in `responses/`, and price files in `yfinance/` remain completely unchanged (identical to Git HEAD).
   * *Utility Safety:* [`responses/get_common_stocks.py`](../responses/get_common_stocks.py) was updated to resolve paths relative to the script directory and runs in check-only mode by default. It requires an explicit `--write` flag and validation thresholds, ensuring it will never rewrite or truncate legacy files during auditing.
2. **Contract B (SRS Multi-Year OHLCV Dataset):** Ingests daily Open, High, Low, Close, and Volume across **November 1, 2018 to December 31, 2025** stored in [`data/`](./) for feature engineering (Person 2) and continuous-control DRL agent training (Semester VI).
   * *Current Status:* The corrected pipeline has been verified and validated on the **5-ticker sample dataset** (`data/processed_ohlcv/sample/`). Execution of the full universe remains paused pending team/supervisor confirmation of the asset universe and date boundaries.

---

## 2. Environment & Provenance

* **Data Provider:** Yahoo Finance via the official `yfinance` Python library.
* **Software Dependencies:**
  * Python: `3.14.2`
  * `yfinance`: `1.7.0`
  * `pandas`: `2.3.3`
  * `numpy`: `2.3.5`
* **Provider Error Capture:** Verified on `yfinance==1.7.0` via batch download context (`yf.multi._DownloadCtx`), capturing exact provider diagnostics for failed tickers without relying on speculative external annotations.
* **Price Adjustment Convention:**
  * The pipeline defaults to `--auto-adjust` for Contract B (configurable via `--auto-adjust` or `--no-auto-adjust`).
  * *Technical Rationale:* In raw unadjusted series (`--no-auto-adjust`), stock splits (e.g. Nvidia 10:1 in June 2024, Apple 4:1 in August 2020) cause artificial -75% to -90% price drops in raw `Close`, which would corrupt daily returns and distort volatility. Furthermore, Person 2 requires internal OHLC geometric consistency ($High \ge Low$, $High \ge \max(Open, Close)$) across the multi-year timeline for technical indicators (ADX, ATR, MACD). Scaling Open, High, Low, and Close proportionally maintains stationarity and true economic returns.
  * *Legacy Note:* The legacy 2024 files in `yfinance/` were downloaded unadjusted and remain completely untouched.

---

## 3. Asset Universe Status & Prior Exploratory Findings

* **Legacy Response Universe:** [`responses/common_stocks.json`](../responses/common_stocks.json) contains **497 common stocks** intersecting across all 2024 model JSON files.
* **Documented Baseline Exclusions:** `['J', 'TSCO']` are excluded in the original paper's scripts due to empty/NaN return arrays in model outputs, leaving a candidate list of **495 tickers**.
* **Approval Status:** The 495-ticker universe is a **proposed candidate universe awaiting team and supervisor confirmation** (per `SEMESTER_V_WORK_PLAN.md` line 5). It has not been approved as a final project constant.
* **Prior Exploratory Run Notes (Unadjusted):**
  * An earlier exploratory download generated `data/raw_ohlcv/raw_ohlcv_panel_legacy_prior_unadjusted_20251230.csv` (~82 MB) using `auto_adjust=False` and ending on 2025-12-30.
  * In that exploratory run, **18 tickers** (`ANSS`, `AVB`, `BK`, `CTRA`, `DAY`, `DFS`, `EA`, `EQR`, `FI`, `HES`, `HOLX`, `IPG`, `JNPR`, `K`, `MMC`, `PARA`, `WBA`, `WBD`) returned zero usable observations from Yahoo Finance, and **15 tickers** returned partial observations due to historical IPO/spinoff dates.
  * *Action:* These 18 zero-observation tickers are explicitly documented as unverified upstream availability gaps. Their resolution (symbol mapping, alternate data sources, or exclusion) is an open decision for the team. The corrected pipeline has **not** been rerun on the full universe pending that decision.

---

## 4. Chronological Splits & Date Coverage (Sample Validation)

Yahoo Finance interprets the query parameter `end` as **exclusive** (`< end`). The pipeline increments the requested inclusive end date (`2025-12-31`) to query `< 2026-01-01`, capturing the final session on **December 31, 2025**.

The sample validation dataset covers **1,800 trading days**:

```
┌──────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────────────────────┐
│ Feature Warm-Up  │ In-Sample Training (4 Years)  │ Validation Period (1 Year)    │ Out-of-Sample Test (2 Years)  │
│ 40 Trading Days  │ 1,008 Trading Days            │ 250 Trading Days              │ 502 Trading Days              │
│ 2018-11-01 to    │ 2019-01-01 to 2022-12-31      │ 2023-01-01 to 2023-12-31      │ 2024-01-01 to 2025-12-31      │
│ 2018-12-31       │ (Pre-COVID, Crash, 2021 Bull, │ (Hyperparameter tuning &      │ (Matches SRS FR-10 1-2 Year   │
│ (Buffer >= 30d)  │  2022 Rate Hike Bear Market)  │  checkpoint selection)        │  OOS Backtest Requirement)    │
└──────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────────────────────┘
```

### Bias & Leakage Disclosures
* **Temporal Leakage Safeguards:** Chronological ordering is strictly maintained. The 40-day warm-up buffer (`2018-11-01` to `2018-12-31`) is reserved exclusively for indicator initialization and is excluded from training and evaluation. No future period statistics are used in preprocessing.
* **Survivorship Bias Qualification:** Backtesting a fixed 2024 S&P 500 candidate universe across 2018–2025 inherits **survivorship bias** (companies that went bankrupt or were delisted before 2024 are omitted). This is an explicit research limitation to document in the viva.

---

## 5. Concrete Quality Checks Performed by the Pipeline

The pipeline implements and executes automated validation checks documented in [`data/processed_ohlcv/sample/data_quality_report.json`](./processed_ohlcv/sample/data_quality_report.json):

1. **Duplicate Dates Check:** Asserts `not df.index.duplicated().any()`. Verified: 0 duplicate dates across all fields.
2. **Chronological Monotonicity:** Asserts `df.index.is_monotonic_increasing`. Verified: strictly increasing calendar trading dates.
3. **Price Positivity Check:** Asserts that all prices in `open`, `high`, `low`, and `close` are strictly $> 0$ where non-null. Verified: 0 non-positive prices.
4. **Non-negative Volume Check:** Asserts that all volume observations are $\ge 0$. Verified: 0 negative volumes.
5. **Internal OHLC Consistency Check:**
   * $High \ge Low$
   * $High \ge \max(Open, Close) - \epsilon$
   * $Low \le \min(Open, Close) + \epsilon$  
   Verified: 0 consistency violations across all trading days.
6. **Cross-Field Index Alignment:** Asserts that `open`, `high`, `low`, `close`, and `volume` share the exact identical date index, and `returns` matches `close.index[1:]`. Verified: 100% aligned.
7. **End-Date Verification:** Directly verifies whether the requested end date was returned as an active market trading session in the retrieved data (`target_final_session_included: true`, matching `2025-12-31` exactly) without relying on weekday heuristics that could mistake exchange holidays for trading sessions.

---

## 6. Directory Layout & Git Tracking Strategy

To prevent sample runs from overwriting full-universe outputs, processed directories are partitioned by universe mode:

```text
data/
├── raw_ohlcv/
│   ├── raw_ohlcv_panel_sample.csv                              # Tracked sample raw panel (~0.7 MB)
│   └── raw_ohlcv_panel_legacy_prior_unadjusted_20251230.csv     # Ignored prior exploratory panel (~82 MB)
└── processed_ohlcv/
    └── sample/                                                 # Tracked sample validation suite (~1.1 MB)
        ├── open.csv                                            (1,800 rows x 5 tickers)
        ├── high.csv                                            (1,800 rows x 5 tickers)
        ├── low.csv                                             (1,800 rows x 5 tickers)
        ├── close.csv                                           (1,800 rows x 5 tickers)
        ├── volume.csv                                          (1,800 rows x 5 tickers)
        ├── returns.csv                                         (1,799 rows x 5 tickers)
        ├── data_quality_report.json                            (Validation audit report)
        └── dataset_metadata.json                               (Metadata sheet)
```

### Git Tracking & Storage Rules:
* **Tracked in Git:**
  * Source code: [`src/data/ohlcv_pipeline.py`](../src/data/ohlcv_pipeline.py), [`responses/get_common_stocks.py`](../responses/get_common_stocks.py)
  * Documentation & reports: [`data/DATA_INVENTORY.md`](./DATA_INVENTORY.md), `data_quality_report.json`, `dataset_metadata.json`
  * Sample validation files: `data/raw_ohlcv/raw_ohlcv_panel_sample.csv` and `data/processed_ohlcv/sample/*.csv` (~1.8 MB total).
* **Excluded via `.gitignore`:**
  * Large raw panels (`data/raw_ohlcv/*legacy*.csv`, `*prior*.csv`)
  * Full-universe processed outputs (`data/processed_ohlcv/legacy/`, `data/processed_ohlcv/custom/`).
* **Teammate Data Sharing:**
  * Teammates have immediate functional access to the validated sample dataset in `data/processed_ohlcv/sample/`.
  * When the universe and dates are confirmed, the full dataset can be generated locally via the pipeline script or distributed as a compressed archive on shared storage.

---

## 7. Run Instructions

### Verify Legacy Common Stocks (Check-Only Mode)
```powershell
python responses/get_common_stocks.py
```

### Run Sample Pipeline (Fast Validation, 5 Tickers)
```powershell
python src/data/ohlcv_pipeline.py --universe sample --start 2018-11-01 --end 2025-12-31 --auto-adjust
```

### Run with Explicit Unadjusted Prices (Optional Mode)
```powershell
python src/data/ohlcv_pipeline.py --universe sample --start 2018-11-01 --end 2025-12-31 --no-auto-adjust
```

### Candidate Universe Pipeline (Full Download - Paused Awaiting Confirmation)
```powershell
python src/data/ohlcv_pipeline.py --universe legacy --batch-size 50 --start 2018-11-01 --end 2025-12-31 --auto-adjust
```
