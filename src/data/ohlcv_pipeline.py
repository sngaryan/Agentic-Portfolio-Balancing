"""
OHLCV Data Ingestion & Preprocessing Pipeline
=============================================
Semester V: Agentic AI for Financial Portfolio Balancing
Owner: Person 1 (Data & Preprocessing Owner)

Governing Document: SEMESTER_V_WORK_PLAN.md (Revision: 2026-10-26 15:28:25 IST)
Core Clone Reference: PROJECT_EXPLANATION.md

Purpose:
--------
Download, clean, validate, and store daily Open, High, Low, Close, Volume (OHLCV)
data from Yahoo Finance across 2019-01-01 through 2025-12-31 (plus >= 30 trading
observations before 2019-01-01 for feature warm-up).

Key Architectural Principles:
-----------------------------
1. Complete Isolation: Never modifies or overwrites legacy 2024 data in yfinance/ or responses/.
2. Partitioned Outputs: Stores processed data in universe-specific subdirectories
   (e.g. data/processed_ohlcv/sample/ vs data/processed_ohlcv/legacy/) so sample validation runs
   never overwrite full-universe outputs.
3. Coherent Price Adjustment: Configurable via --auto-adjust / --no-auto-adjust (default True).
   When True, scales Open, High, Low, and Close proportionally for splits and dividend distributions,
   eliminating artificial return dislocations and maintaining geometric consistency for technical
   indicator calculation (ADX, ATR, MACD).
4. Exclusive End-Date Resolution: Yahoo Finance interprets `end` as exclusive (< end). The pipeline
   automatically increments the requested inclusive end date by +1 day so the target final session
   (e.g. 2025-12-31) is included.
5. Concrete Quality Validation: Explicitly verifies duplicate dates, chronological order, positive
   prices, non-negative volumes, internal OHLC consistency, and cross-field date alignment.
6. Evidence-Based Ticker Audit: Reconciles requested tickers, returned columns, usable observations,
   partial observations, and download failures without inferring speculative external causes.
7. Zero Synthetic Data: Missing observations remain explicit NaNs; no forward-filling or fabrication.
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# Record exact software environment
DATA_ENGINE_VERSIONS = {
    "python": sys.version.split()[0],
    "yfinance": getattr(yf, "__version__", "1.7.0"),
    "pandas": pd.__version__,
    "numpy": np.__version__,
}

DEFAULT_WARMUP_START = "2018-11-01"  # Provides >= 30 trading days before 2019-01-01
DEFAULT_STUDY_START = "2019-01-01"
DEFAULT_STUDY_END = "2025-12-31"

SAMPLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]


def sanitize_ticker_for_yfinance(ticker: str) -> str:
    """Map dots to hyphens for Yahoo Finance compatibility (e.g. BRK.B -> BRK-B)."""
    return ticker.replace(".", "-")


def desanitize_ticker_from_yfinance(ticker: str, original_universe: List[str]) -> str:
    """Reverse map Yahoo ticker back to original project ticker symbol."""
    if ticker in original_universe:
        return ticker
    dot_candidate = ticker.replace("-", ".")
    if dot_candidate in original_universe:
        return dot_candidate
    return ticker


def resolve_universe(
    universe_mode: str,
    custom_tickers: Optional[str] = None,
    exclusions: Optional[List[str]] = None,
    repo_root: Optional[str] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """Load and validate the requested asset universe."""
    if repo_root is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    exclusions = exclusions or ["J", "TSCO"]

    if universe_mode == "sample":
        raw_tickers = SAMPLE_TICKERS
    elif universe_mode == "legacy":
        common_path = os.path.join(repo_root, "responses", "common_stocks.json")
        if not os.path.exists(common_path):
            raise FileNotFoundError(
                f"Legacy common stocks file not found: {common_path}"
            )
        with open(common_path, "r") as f:
            raw_tickers = json.load(f)
    elif universe_mode == "custom":
        if not custom_tickers:
            raise ValueError(
                "custom_tickers must be provided when universe_mode is 'custom'"
            )
        if os.path.exists(custom_tickers):
            with open(custom_tickers, "r") as f:
                raw_tickers = json.load(f)
        else:
            raw_tickers = [
                t.strip() for t in custom_tickers.split(",") if t.strip()
            ]
    else:
        raise ValueError(f"Unknown universe_mode: {universe_mode}")

    # Exclude mandatory exclusions
    active_tickers = [t for t in raw_tickers if t not in exclusions]
    active_tickers = sorted(list(set(active_tickers)))

    yf_mapping = {}
    for t in active_tickers:
        yf_sym = sanitize_ticker_for_yfinance(t)
        yf_mapping[yf_sym] = t

    return active_tickers, yf_mapping


def download_ohlcv_batch(
    yf_tickers: List[str],
    start_date: str,
    end_date_inclusive: str,
    auto_adjust: bool = True,
    yf_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Download OHLCV panel data via yfinance.
    
    Resolves Yahoo Finance's exclusive end-date behavior by incrementing
    the requested inclusive end date by +1 day so that the final trading
    session is retrieved. Captures verified provider errors for failed tickers.
    
    Returns:
        (downloaded_dataframe, captured_errors_dict)
    """
    dt_end = pd.to_datetime(end_date_inclusive) + pd.Timedelta(days=1)
    yf_end_date = dt_end.strftime("%Y-%m-%d")

    captured_errors = {}
    try:
        # Utilize yfinance internal download context when available to capture exact per-ticker errors
        if (
            hasattr(yf, "multi")
            and hasattr(yf.multi, "_DownloadCtx")
            and hasattr(yf.multi, "_download_impl")
        ):
            ctx = yf.multi._DownloadCtx()
            data = yf.multi._download_impl(
                ctx,
                yf_tickers,
                start=start_date,
                end=yf_end_date,
                auto_adjust=auto_adjust,
                progress=False,
                group_by="column",
            )
            if hasattr(ctx, "errors") and ctx.errors:
                for yf_t, err in ctx.errors.items():
                    orig_t = yf_map.get(yf_t, yf_t) if yf_map else yf_t
                    captured_errors[orig_t] = f"Provider error: {err}"
        else:
            data = yf.download(
                tickers=yf_tickers,
                start=start_date,
                end=yf_end_date,
                auto_adjust=auto_adjust,
                progress=False,
                group_by="column",
            )
    except Exception as e:
        for t in yf_tickers:
            orig_t = yf_map.get(t, t) if yf_map else t
            captured_errors[orig_t] = f"Exception: {type(e).__name__}: {str(e)}"
        data = pd.DataFrame()

    return data, captured_errors


def run_concrete_quality_checks(
    field_dfs: Dict[str, pd.DataFrame], returns_df: pd.DataFrame
) -> Dict:
    """Execute and document concrete data quality and validation checks."""
    checks = {}

    # 1. Duplicate dates check
    duplicate_dates = {}
    for fld, df in field_dfs.items():
        dups = df.index[df.index.duplicated()].tolist()
        if dups:
            duplicate_dates[fld] = dups
    checks["duplicate_dates_passed"] = len(duplicate_dates) == 0
    checks["duplicate_dates_found"] = duplicate_dates

    # 2. Chronological order check
    chronological_order = {}
    for fld, df in field_dfs.items():
        chronological_order[fld] = bool(df.index.is_monotonic_increasing)
    checks["chronological_order_passed"] = all(chronological_order.values())
    checks["chronological_order_by_field"] = chronological_order

    # 3. Positive prices check (Open, High, Low, Close must be > 0 where non-null)
    price_positivity = {}
    for fld in ["open", "high", "low", "close"]:
        if fld in field_dfs:
            df = field_dfs[fld]
            non_positive_count = int(((df <= 0) & (~df.isna())).sum().sum())
            price_positivity[fld] = {
                "passed": non_positive_count == 0,
                "non_positive_count": non_positive_count,
            }
    checks["price_positivity_passed"] = all(
        v["passed"] for v in price_positivity.values()
    )
    checks["price_positivity_details"] = price_positivity

    # 4. Nonnegative volume check
    if "volume" in field_dfs:
        v_df = field_dfs["volume"]
        negative_vol = int(((v_df < 0) & (~v_df.isna())).sum().sum())
        zero_vol = int(((v_df == 0) & (~v_df.isna())).sum().sum())
        checks["nonnegative_volume_passed"] = negative_vol == 0
        checks["volume_validation"] = {
            "negative_volume_count": negative_vol,
            "zero_volume_trading_days_count": zero_vol,
        }

    # 5. OHLC Internal Consistency Check
    if all(k in field_dfs for k in ["open", "high", "low", "close"]):
        o = field_dfs["open"]
        h = field_dfs["high"]
        l = field_dfs["low"]
        c = field_dfs["close"]

        tol = 1e-4
        valid_mask = (~o.isna()) & (~h.isna()) & (~l.isna()) & (~c.isna())

        high_less_than_low = int(((h < l - tol) & valid_mask).sum().sum())
        high_less_than_open = int(((h < o - tol) & valid_mask).sum().sum())
        high_less_than_close = int(((h < c - tol) & valid_mask).sum().sum())
        low_greater_than_open = int(((l > o + tol) & valid_mask).sum().sum())
        low_greater_than_close = int(((l > c + tol) & valid_mask).sum().sum())

        ohlc_passed = (
            high_less_than_low == 0
            and high_less_than_open == 0
            and high_less_than_close == 0
            and low_greater_than_open == 0
            and low_greater_than_close == 0
        )
        checks["ohlc_consistency_passed"] = ohlc_passed
        checks["ohlc_consistency_details"] = {
            "high_less_than_low_count": high_less_than_low,
            "high_less_than_open_count": high_less_than_open,
            "high_less_than_close_count": high_less_than_close,
            "low_greater_than_open_count": low_greater_than_open,
            "low_greater_than_close_count": low_greater_than_close,
        }

    # 6. Common Date Index across fields
    reference_index = (
        field_dfs["close"].index.tolist() if "close" in field_dfs else []
    )
    index_alignment = {}
    for fld, df in field_dfs.items():
        index_alignment[fld] = df.index.tolist() == reference_index
    if not returns_df.empty:
        index_alignment["returns"] = (
            returns_df.index.tolist() == reference_index[1:]
        )
    checks["common_date_index_passed"] = all(index_alignment.values())
    checks["common_date_index_details"] = index_alignment

    return checks


def run_pipeline(
    universe_mode: str = "sample",
    custom_tickers: Optional[str] = None,
    exclusions: Optional[List[str]] = None,
    start_date: str = DEFAULT_WARMUP_START,
    end_date: str = DEFAULT_STUDY_END,
    auto_adjust: bool = True,
    batch_size: int = 50,
    output_dir: str = "data",
    repo_root: Optional[str] = None,
) -> Dict:
    """Execute the configurable download and validation pipeline."""
    if repo_root is None:
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    raw_dir = os.path.join(repo_root, output_dir, "raw_ohlcv")
    # Partition processed output directory by universe mode to prevent sample runs from overwriting full runs
    processed_dir = os.path.join(repo_root, output_dir, "processed_ohlcv", universe_mode)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # 1. Resolve and validate universe
    canonical_tickers, yf_map = resolve_universe(
        universe_mode=universe_mode,
        custom_tickers=custom_tickers,
        exclusions=exclusions,
        repo_root=repo_root,
    )
    yf_ticker_list = sorted(list(yf_map.keys()))

    print("=" * 80)
    print(f"OHLCV Pipeline Execution - Universe: {universe_mode.upper()}")
    print(f"Requested Tickers: {len(canonical_tickers)}")
    print(f"Target Date Range: {start_date} to {end_date} (Inclusive)")
    print(f"Price Adjustment: auto_adjust={auto_adjust}")
    print(f"Raw Panel Directory: {raw_dir}")
    print(f"Processed Directory: {processed_dir}")
    print("=" * 80)

    # 2. Download in batches
    collected_dfs = []
    batch_failures = {}

    total_batches = (len(yf_ticker_list) + batch_size - 1) // batch_size
    for b_idx in range(total_batches):
        batch = yf_ticker_list[b_idx * batch_size : (b_idx + 1) * batch_size]
        print(
            f"Downloading Batch {b_idx + 1}/{total_batches} ({len(batch)} tickers)..."
        )
        batch_df, errs = download_ohlcv_batch(
            yf_tickers=batch,
            start_date=start_date,
            end_date_inclusive=end_date,
            auto_adjust=auto_adjust,
            yf_map=yf_map,
        )
        batch_failures.update(errs)
        if not batch_df.empty:
            collected_dfs.append(batch_df)
        else:
            for yf_t in batch:
                orig_t = yf_map.get(yf_t, yf_t)
                if orig_t not in batch_failures:
                    batch_failures[orig_t] = "Download returned empty dataframe"

    if not collected_dfs:
        raise RuntimeError("No data was retrieved from Yahoo Finance.")

    # 3. Combine batches along columns
    if len(collected_dfs) == 1:
        full_df = collected_dfs[0]
    else:
        full_df = pd.concat(collected_dfs, axis=1)

    # If single ticker was passed, yfinance returns flat columns instead of MultiIndex
    if not isinstance(full_df.columns, pd.MultiIndex):
        ticker = yf_ticker_list[0]
        full_df.columns = pd.MultiIndex.from_product(
            [full_df.columns, [ticker]]
        )

    # 4. Save raw panel
    raw_panel_path = os.path.join(
        raw_dir, f"raw_ohlcv_panel_{universe_mode}.csv"
    )
    full_df.to_csv(raw_panel_path)
    print(f"Saved raw panel to: {raw_panel_path}")

    # 5. Extract and process individual fields
    available_fields = full_df.columns.levels[0].tolist()
    field_dfs = {}

    standard_fields = ["Open", "High", "Low", "Close", "Volume"]
    for fld in standard_fields:
        if fld in available_fields:
            df_fld = full_df[fld].copy()
            df_fld.columns = [yf_map.get(col, col) for col in df_fld.columns]
            df_fld = df_fld.reindex(sorted(df_fld.columns), axis=1)
            df_fld = df_fld.sort_index()
            df_fld.index = pd.to_datetime(df_fld.index).strftime("%Y-%m-%d")
            df_fld.index.name = "Date"

            field_dfs[fld.lower()] = df_fld
            out_file = os.path.join(processed_dir, f"{fld.lower()}.csv")
            df_fld.to_csv(out_file)
            print(f"  Saved {fld.lower()}.csv: shape={df_fld.shape}")
        else:
            print(f"  Warning: Field {fld} not found in yfinance output.")

    # 6. Compute daily returns from Close without forward-filling
    if "close" in field_dfs:
        close_df = field_dfs["close"]
        returns_df = close_df.pct_change(fill_method=None).iloc[1:]
        returns_path = os.path.join(processed_dir, "returns.csv")
        returns_df.to_csv(returns_path)
        print(f"  Saved returns.csv: shape={returns_df.shape}")
    else:
        returns_df = pd.DataFrame()

    # 7. Quality audit and validation summary
    dates = field_dfs["close"].index.tolist() if "close" in field_dfs else []
    total_days = len(dates)
    warmup_dates = [d for d in dates if d < DEFAULT_STUDY_START]
    study_dates = [d for d in dates if d >= DEFAULT_STUDY_START]

    train_dates = [d for d in study_dates if "2019-01-01" <= d <= "2022-12-31"]
    val_dates = [d for d in study_dates if "2023-01-01" <= d <= "2023-12-31"]
    test_dates = [d for d in study_dates if "2024-01-01" <= d <= "2025-12-31"]

    # Reconcile requested vs returned columns
    close_df = field_dfs.get("close", pd.DataFrame())
    returned_tickers = sorted(close_df.columns.tolist())

    ticker_stats = {}
    zero_obs_tickers = []
    partial_obs_tickers = []
    full_obs_tickers = []

    for col in returned_tickers:
        valid_cnt = int(close_df[col].notna().sum())
        missing_cnt = int(close_df[col].isna().sum())
        usable_pct = round((valid_cnt / total_days) * 100, 2) if total_days > 0 else 0.0

        first_valid = (
            close_df[col].first_valid_index() if valid_cnt > 0 else None
        )
        last_valid = (
            close_df[col].last_valid_index() if valid_cnt > 0 else None
        )

        stat = {
            "usable_observations": valid_cnt,
            "missing_observations": missing_cnt,
            "usable_percentage": usable_pct,
            "first_valid_date": first_valid,
            "last_valid_date": last_valid,
            "status": (
                "full"
                if valid_cnt == total_days
                else "partial"
                if valid_cnt > 0
                else "zero_observations"
            ),
        }
        ticker_stats[col] = stat

        if valid_cnt == 0:
            zero_obs_tickers.append(col)
        elif valid_cnt < total_days:
            partial_obs_tickers.append(col)
        else:
            full_obs_tickers.append(col)

    # Any tickers requested but completely absent from returned DataFrame columns
    unreturned_tickers = sorted([t for t in canonical_tickers if t not in returned_tickers])

    # Reconcile download failures: include verified captured errors; never infer causes from annotations
    download_failures = {}
    for t in zero_obs_tickers:
        captured_err = batch_failures.get(t)
        download_failures[t] = {
            "status": "zero_observations",
            "captured_provider_error": captured_err,
            "cause": captured_err if captured_err else "Unknown (zero usable observations returned by provider; unverified upstream cause)",
            "usable_observations": 0,
            "missing_observations": total_days,
            "first_valid_date": None,
            "last_valid_date": None,
        }
    for t in unreturned_tickers:
        captured_err = batch_failures.get(t)
        download_failures[t] = {
            "status": "column_not_returned",
            "captured_provider_error": captured_err,
            "cause": captured_err if captured_err else "Unknown (ticker omitted from returned table; unverified upstream cause)",
            "usable_observations": 0,
            "missing_observations": total_days,
            "first_valid_date": None,
            "last_valid_date": None,
        }

    # Per-field missingness counts
    per_field_missing = {}
    for fld, df in field_dfs.items():
        per_field_missing[fld] = int(df.isna().sum().sum())
    if not returns_df.empty:
        per_field_missing["returns"] = int(returns_df.isna().sum().sum())

    # Execute concrete quality checks
    quality_checks = run_concrete_quality_checks(field_dfs, returns_df)

    # End-date verification: reports strictly what the pipeline can verify from returned data
    actual_last_date = dates[-1] if dates else None
    actual_first_date = dates[0] if dates else None
    target_final_session_included = bool(actual_last_date and actual_last_date == end_date)
    end_date_verification_note = (
        "Exact match: requested end date was an active market trading session and is included in the dataset."
        if target_final_session_included
        else f"Actual last returned trading session is {actual_last_date} (requested end date was {end_date}; non-match may reflect weekends, exchange holidays, or market closures)."
    )

    quality_report = {
        "execution_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "library_versions": DATA_ENGINE_VERSIONS,
        "universe_mode": universe_mode,
        "ticker_reconciliation": {
            "requested_tickers_count": len(canonical_tickers),
            "requested_tickers": canonical_tickers,
            "returned_columns_count": len(returned_tickers),
            "returned_columns": returned_tickers,
            "unreturned_tickers_count": len(unreturned_tickers),
            "unreturned_tickers_list": unreturned_tickers,
            "usable_tickers_count": len(full_obs_tickers) + len(partial_obs_tickers),
            "usable_tickers_list": sorted(full_obs_tickers + partial_obs_tickers),
            "full_history_count": len(full_obs_tickers),
            "full_history_list": full_obs_tickers,
            "partial_history_count": len(partial_obs_tickers),
            "partial_history_list": partial_obs_tickers,
            "zero_observations_count": len(zero_obs_tickers),
            "zero_observations_list": zero_obs_tickers,
            "download_failures_count": len(download_failures),
            "download_failures_list": sorted(list(download_failures.keys())),
        },
        "download_failures": download_failures,
        "zero_observation_tickers": {
            t: download_failures[t] for t in zero_obs_tickers
        },
        "partial_observation_tickers": {
            t: {
                "usable_observations": ticker_stats[t]["usable_observations"],
                "missing_observations": ticker_stats[t]["missing_observations"],
                "usable_percentage": ticker_stats[t]["usable_percentage"],
                "first_valid_date": ticker_stats[t]["first_valid_date"],
                "last_valid_date": ticker_stats[t]["last_valid_date"],
            }
            for t in partial_obs_tickers
        },
        "total_trading_days": total_days,
        "date_coverage": {
            "requested_start_date": start_date,
            "requested_end_date": end_date,
            "actual_first_trading_day": actual_first_date,
            "actual_last_trading_day": actual_last_date,
            "target_final_session_included": target_final_session_included,
            "end_date_verification_note": end_date_verification_note,
            "warmup_observations_count": len(warmup_dates),
            "warmup_start": warmup_dates[0] if warmup_dates else None,
            "warmup_end": warmup_dates[-1] if warmup_dates else None,
            "training_period_days": len(train_dates),
            "validation_period_days": len(val_dates),
            "testing_period_days": len(test_dates),
        },
        "price_adjustment_policy": {
            "auto_adjust": auto_adjust,
            "description": (
                "auto_adjust=True scales Open, High, Low, and Close proportionally for splits "
                "and dividend distributions. Eliminates artificial return dislocations and maintains "
                "geometric consistency for technical indicator calculation (ADX, ATR, MACD)."
                if auto_adjust
                else "auto_adjust=False preserves raw unadjusted market prices. Returns must be derived "
                "from adjusted prices to prevent split dislocations."
            ),
        },
        "per_field_missing_observation_counts": per_field_missing,
        "concrete_quality_checks": quality_checks,
    }

    report_path = os.path.join(processed_dir, "data_quality_report.json")
    with open(report_path, "w") as f:
        json.dump(quality_report, f, indent=4)
    print(f"Saved quality audit report to: {report_path}")

    # Metadata sheet
    metadata = {
        "dataset_name": f"ohlcv_dataset_{universe_mode}",
        "data_source": "Yahoo Finance (via yfinance library)",
        "download_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "fields": list(field_dfs.keys()) + (["returns"] if not returns_df.empty else []),
        "asset_count": len(close_df.columns) if not close_df.empty else 0,
        "asset_order": "Alphabetical (strictly sorted)",
        "date_index_format": "YYYY-MM-DD (chronologically sorted)",
        "price_adjustment": (
            "auto_adjust=True (proportionally split- and dividend-adjusted OHLC)"
            if auto_adjust
            else "auto_adjust=False (raw nominal market prices)"
        ),
        "warmup_policy": (
            ">= 30 trading observations prior to 2019-01-01 strictly reserved for feature warm-up; "
            "never evaluated as training or testing period"
        ),
        "leakage_safeguards": (
            "Chronological partitioning enforced. Rolling indicators look backward only; "
            "no future period observations used in scaling. Static universe selection across historical "
            "periods inherits survivorship bias, which is documented as an academic research limitation."
        ),
        "legacy_isolation": (
            "Legacy 2024 files in yfinance/ and responses/ preserved unchanged. "
            "Contract A (legacy 6-month baseline) and Contract B (multi-year OHLCV) are stored independently."
        ),
    }
    meta_path = os.path.join(processed_dir, "dataset_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved metadata to: {meta_path}")

    print("=" * 80)
    print("Pipeline Execution Completed Successfully.")
    print("=" * 80)

    return quality_report


def main():
    parser = argparse.ArgumentParser(
        description="Configurable OHLCV Download & Preprocessing Pipeline"
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="sample",
        choices=["sample", "legacy", "custom"],
        help="Asset universe to download (sample=5 stocks, legacy=495 candidate stocks, custom)",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers or path to JSON file (for custom universe)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=DEFAULT_WARMUP_START,
        help="Start date including warm-up period (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=DEFAULT_STUDY_END,
        help="Inclusive end date (YYYY-MM-DD)",
    )
    # BooleanOptionalAction makes --auto-adjust and --no-auto-adjust both valid CLI flags
    parser.add_argument(
        "--auto-adjust",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable price adjustment for splits and dividends (--auto-adjust or --no-auto-adjust). Default is --auto-adjust.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for yfinance download requests",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory path relative to project root",
    )

    args = parser.parse_args()

    run_pipeline(
        universe_mode=args.universe,
        custom_tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
        auto_adjust=args.auto_adjust,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
