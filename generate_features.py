"""
Feature Generation Pipeline
===========================
Processes raw/cleaned OHLCV market panels (from Person 1 data pipeline)
into standardized multi-asset technical feature panels, rolling risk vectors,
and shrinkage covariance matrices (SRS FR-03 & FR-04).

Outputs:
  data/features/{universe_mode}/
    ├── rsi.csv
    ├── macd.csv
    ├── macd_signal.csv
    ├── macd_histogram.csv
    ├── adx.csv
    ├── plus_di.csv
    ├── minus_di.csv
    ├── cci.csv
    ├── volatility_30d.csv
    ├── ledoit_wolf_covariance.csv
    ├── feature_metadata.json
    └── feature_quality_report.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from feature_engineering import (
    calculate_macd,
    calculate_rsi,
    calculate_adx,
    calculate_cci,
    calculate_rolling_volatility,
    compute_ledoit_wolf_covariance,
    extract_asset_features,
)


def generate_feature_panels(
    input_dir: str = "data/processed_ohlcv/sample",
    output_dir: str = "data/features/sample",
    warmup_period: int = 40,
) -> Dict[str, object]:
    """Generates standardized feature panels and audits quality metrics.

    Args:
        input_dir: Directory containing open.csv, high.csv, low.csv, close.csv,
                   volume.csv, and returns.csv.
        output_dir: Target output directory for feature CSVs and metadata.
        warmup_period: Warm-up trading days buffer reserved for initialization.

    Returns:
        Summary dictionary containing audit results and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[-] Loading processed OHLCV panel from: {input_dir}")
    open_df = pd.read_csv(os.path.join(input_dir, "open.csv"), parse_dates=["Date"]).set_index("Date")
    high_df = pd.read_csv(os.path.join(input_dir, "high.csv"), parse_dates=["Date"]).set_index("Date")
    low_df = pd.read_csv(os.path.join(input_dir, "low.csv"), parse_dates=["Date"]).set_index("Date")
    close_df = pd.read_csv(os.path.join(input_dir, "close.csv"), parse_dates=["Date"]).set_index("Date")
    vol_df = pd.read_csv(os.path.join(input_dir, "volume.csv"), parse_dates=["Date"]).set_index("Date")

    tickers = sorted(list(close_df.columns))
    dates = close_df.index
    n_days = len(dates)
    n_assets = len(tickers)
    print(f"    [+] Universe: {n_assets} assets ({', '.join(tickers)})")
    print(f"    [+] Date coverage: {n_days} sessions ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")

    # Initialize panel DataFrames with Date index
    panels = {
        "rsi": pd.DataFrame(index=dates),
        "macd": pd.DataFrame(index=dates),
        "macd_signal": pd.DataFrame(index=dates),
        "macd_histogram": pd.DataFrame(index=dates),
        "adx": pd.DataFrame(index=dates),
        "plus_di": pd.DataFrame(index=dates),
        "minus_di": pd.DataFrame(index=dates),
        "cci": pd.DataFrame(index=dates),
        "volatility_30d": pd.DataFrame(index=dates),
    }

    # Derive returns directly from aligned close prices
    ret_panel = close_df.pct_change()

    print("[-] Computing technical indicators per asset...")
    for ticker in tickers:
        c = close_df[ticker]
        h = high_df[ticker]
        l = low_df[ticker]
        r = ret_panel[ticker]

        # 1. RSI (14)
        panels["rsi"][ticker] = calculate_rsi(c, period=14)

        # 2. MACD (12, 26, 9)
        macd_df = calculate_macd(c, fast_period=12, slow_period=26, signal_period=9)
        panels["macd"][ticker] = macd_df["MACD"]
        panels["macd_signal"][ticker] = macd_df["Signal"]
        panels["macd_histogram"][ticker] = macd_df["Histogram"]

        # 3. ADX (14)
        adx_df = calculate_adx(h, l, c, period=14)
        panels["adx"][ticker] = adx_df["ADX"]
        panels["plus_di"][ticker] = adx_df["+DI"]
        panels["minus_di"][ticker] = adx_df["-DI"]

        # 4. CCI (20)
        panels["cci"][ticker] = calculate_cci(h, l, c, period=20)

        # 5. Volatility (30d)
        panels["volatility_30d"][ticker] = calculate_rolling_volatility(r, window=30)

    # 6. Ledoit-Wolf Shrinkage Covariance on latest trailing 30 days
    recent_returns = ret_panel[tickers].iloc[-30:].values
    shrunk_cov, shrinkage_delta = compute_ledoit_wolf_covariance(recent_returns)
    cov_df = pd.DataFrame(shrunk_cov, index=tickers, columns=tickers)

    # Save CSV outputs
    print(f"[-] Writing feature panel CSVs to: {output_dir}")
    for name, df in panels.items():
        out_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(out_path, index=True)
        print(f"    [+] Saved: {out_path} ({df.shape})")

    cov_path = os.path.join(output_dir, "ledoit_wolf_covariance.csv")
    cov_df.to_csv(cov_path)
    print(f"    [+] Saved: {cov_path} ({cov_df.shape})")

    # Data Quality Audit
    print("[-] Running post-warmup data quality audit...")
    audit_results = {}
    post_warmup_slice = slice(warmup_period, None)

    for name, df in panels.items():
        post_df = df.iloc[post_warmup_slice][tickers]
        nan_count = int(post_df.isna().sum().sum())
        inf_count = int(np.isinf(post_df.values).sum())
        min_val = float(post_df.min().min())
        max_val = float(post_df.max().max())

        audit_results[name] = {
            "post_warmup_nan_count": nan_count,
            "post_warmup_inf_count": inf_count,
            "min_value": round(min_val, 4),
            "max_value": round(max_val, 4),
            "status": "PASS" if nan_count == 0 and inf_count == 0 else "FAIL",
        }

    cov_eigenvalues = np.linalg.eigvalsh(shrunk_cov)
    audit_results["ledoit_wolf_covariance"] = {
        "shrinkage_intensity_delta": round(float(shrinkage_delta), 4),
        "min_eigenvalue": float(np.min(cov_eigenvalues)),
        "condition_number": round(float(np.linalg.cond(shrunk_cov)), 2),
        "is_positive_definite": bool((cov_eigenvalues > 0).all()),
        "status": "PASS" if (cov_eigenvalues > 0).all() else "FAIL",
    }

    # Save Quality Report JSON
    quality_report_path = os.path.join(output_dir, "feature_quality_report.json")
    with open(quality_report_path, "w") as f:
        json.dump(audit_results, f, indent=4)
    print(f"    [+] Saved: {quality_report_path}")

    # Save Metadata JSON
    metadata = {
        "dataset_name": "market_features_sample",
        "parent_ohlcv_dir": input_dir,
        "asset_count": n_assets,
        "asset_order": tickers,
        "date_range": {
            "start": dates[0].strftime("%Y-%m-%d"),
            "end": dates[-1].strftime("%Y-%m-%d"),
            "total_trading_days": n_days,
            "warmup_buffer_days": warmup_period,
            "effective_eval_days": n_days - warmup_period,
        },
        "indicators": {
            "RSI": {"period": 14, "smoothing": "Wilder", "bounds": [0, 100]},
            "MACD": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
            "ADX": {"period": 14, "components": ["ADX", "+DI", "-DI"], "bounds": [0, 100]},
            "CCI": {"period": 20, "constant": 0.015, "mad_scaling": True},
            "Rolling_Volatility": {"window": 30, "annualized": True, "scale": "sqrt(252)"},
            "Ledoit_Wolf_Covariance": {"window": 30, "shrinkage": "analytical", "target": "scaled_identity"},
        },
        "state_representation_spec": {
            "framework": "Gymnasium (Box space)",
            "dimension_formula": "7 * N + 2",
            "sample_dimension": 7 * n_assets + 2,
            "scaling_policy": "Normalized & bounded to [-5.0, 5.0]; fit strictly on training interval",
        },
    }

    metadata_path = os.path.join(output_dir, "feature_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"    [+] Saved: {metadata_path}")

    print("\n[+] Feature Generation & Quality Audit Complete.")
    return audit_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and audit technical feature panels.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed_ohlcv/sample",
        help="Directory with processed OHLCV CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features/sample",
        help="Directory to save feature panels",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=40,
        help="Warmup buffer days (default: 40)",
    )
    args = parser.parse_args()
    generate_feature_panels(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        warmup_period=args.warmup,
    )
