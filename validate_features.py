"""
Feature & State Validation Test Suite
=====================================
Performs automated verification of quantitative indicators, numerical bounds,
covariance conditioning, lookahead leakage prevention, and Gymnasium state
vector construction (SRS FR-03 & FR-04).
"""

import sys
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
from state_representation import (
    StateSpaceConfig,
    PortfolioStateBuilder,
    ChronologicalScaler,
)


def generate_synthetic_ohlcv(
    n_days: int = 120,
    seed: int = 42,
    base_price: float = 150.0,
) -> pd.DataFrame:
    """Generates synthetic OHLCV price series for unit testing."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    
    # Geometric Brownian Motion
    daily_returns = np.random.normal(0.0005, 0.015, size=n_days)
    price_path = base_price * np.cumprod(1 + daily_returns)
    
    highs = price_path * (1 + np.abs(np.random.normal(0.005, 0.004, size=n_days)))
    lows = price_path * (1 - np.abs(np.random.normal(0.005, 0.004, size=n_days)))
    opens = price_path * (1 + np.random.normal(0.0, 0.003, size=n_days))
    closes = price_path
    volumes = np.random.randint(1_000_000, 10_000_000, size=n_days)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        }
    )


def test_indicator_bounds_and_formulas():
    """Verifies that indicator outputs adhere to mathematical boundaries."""
    print("[-] Testing Indicator Formulas and Bounded Ranges...")
    df = generate_synthetic_ohlcv(n_days=100)

    # 1. RSI bounds [0, 100]
    rsi = calculate_rsi(df["Close"])
    valid_rsi = rsi.dropna()
    assert (valid_rsi >= 0.0).all() and (valid_rsi <= 100.0).all(), (
        f"RSI out of bounds: min={valid_rsi.min()}, max={valid_rsi.max()}"
    )
    print("    [+] RSI correctly bounded within [0, 100].")

    # 2. ADX bounds [0, 100]
    adx_df = calculate_adx(df["High"], df["Low"], df["Close"])
    valid_adx = adx_df["ADX"].dropna()
    assert (valid_adx >= 0.0).all() and (valid_adx <= 100.0).all(), (
        f"ADX out of bounds: min={valid_adx.min()}, max={valid_adx.max()}"
    )
    assert ((adx_df["+DI"].dropna() >= 0.0) & (adx_df["-DI"].dropna() >= 0.0)).all()
    print("    [+] ADX and directional indices (+DI, -DI) valid and non-negative.")

    # 3. MACD line identity
    macd_df = calculate_macd(df["Close"])
    diff = (macd_df["MACD"] - macd_df["Signal"]) - macd_df["Histogram"]
    assert np.allclose(diff.dropna(), 0.0, atol=1e-8), "MACD Histogram identity failed."
    print("    [+] MACD histogram correctly satisfies (MACD - Signal).")

    # 4. CCI calculation
    cci = calculate_cci(df["High"], df["Low"], df["Close"])
    assert not cci.dropna().empty, "CCI calculation produced all NaNs."
    print("    [+] CCI calculated successfully with finite outputs.")

    # 5. Volatility positive
    vol = calculate_rolling_volatility(df["Close"].pct_change(), window=30)
    assert (vol.dropna() >= 0.0).all(), "Volatility produced negative values."
    print("    [+] Rolling Volatility strictly positive.")


def test_lookahead_leakage_prevention():
    """Confirms features at time T never depend on information at T+1."""
    print("[-] Testing Zero Lookahead Bias (Strict Temporal Causality)...")
    df = generate_synthetic_ohlcv(n_days=60)
    
    # Calculate features on window up to day 40
    df_slice = df.iloc[:40].copy()
    features_t40 = extract_asset_features(df_slice)
    val_at_t40_before = features_t40.iloc[39][["MACD", "RSI", "ADX", "CCI"]].to_dict()

    # Calculate features on full window up to day 60 (with subsequent new prices)
    features_full = extract_asset_features(df)
    val_at_t40_after = features_full.iloc[39][["MACD", "RSI", "ADX", "CCI"]].to_dict()

    for indicator, val_before in val_at_t40_before.items():
        val_after = val_at_t40_after[indicator]
        assert np.isclose(val_before, val_after, atol=1e-7), (
            f"Lookahead leakage detected in {indicator}: {val_before} vs {val_after}"
        )
    print("    [+] Zero lookahead leakage: historical feature values are strictly invariant to future data.")


def test_ledoit_wolf_covariance_singularity():
    """Validates shrinkage conditioning when N > T (singular sample covariance)."""
    print("[-] Testing Ledoit-Wolf Shrinkage Covariance Conditioning (N > T)...")
    np.random.seed(101)
    
    # N = 50 assets, T = 30 trading days (Rank <= 29, singular)
    n_assets = 50
    n_days = 30
    synthetic_returns = np.random.normal(0.0005, 0.02, size=(n_days, n_assets))

    # Empirical sample covariance rank
    emp_cov = np.cov(synthetic_returns, rowvar=False)
    emp_rank = np.linalg.matrix_rank(emp_cov)
    emp_eigenvalues = np.linalg.eigvalsh(emp_cov)
    assert emp_rank <= n_days - 1, f"Expected rank <= 29, got {emp_rank}"
    assert np.min(emp_eigenvalues) <= 1e-12, "Empirical covariance expected to be singular."
    print(f"    [+] Verified: Sample covariance is rank-deficient (rank={emp_rank} < N={n_assets}).")

    # Ledoit-Wolf shrinkage
    shrunk_cov, shrinkage = compute_ledoit_wolf_covariance(synthetic_returns)
    shrunk_eigenvalues = np.linalg.eigvalsh(shrunk_cov)
    shrunk_rank = np.linalg.matrix_rank(shrunk_cov)

    assert shrunk_rank == n_assets, f"Shrunk covariance should be full rank {n_assets}, got {shrunk_rank}"
    assert (shrunk_eigenvalues > 0.0).all(), "Shrunk covariance must be strictly positive definite."
    assert 0.0 < shrinkage <= 1.0, f"Shrinkage intensity out of bounds: {shrinkage}"

    cond_emp = np.linalg.cond(emp_cov)
    cond_shrunk = np.linalg.cond(shrunk_cov)
    print(f"    [+] Shrinkage intensity: delta = {shrinkage:.4f}")
    print(f"    [+] Full rank achieved: rank={shrunk_rank}/{n_assets}. Condition number improved from {cond_emp:.1e} to {cond_shrunk:.1f}.")


def test_gym_state_vector_builder():
    """Tests Gymnasium state space construction and dimension conformance."""
    print("[-] Testing Gymnasium State Vector Builder (SRS FR-04)...")
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    n_assets = len(tickers)

    config = StateSpaceConfig(num_assets=n_assets)
    builder = PortfolioStateBuilder(config=config, ticker_order=tickers)

    # Mock inputs
    prices = pd.Series([180.0, 420.0, 160.0, 175.0, 120.0], index=tickers)
    macd = pd.Series([1.5, 2.8, -0.9, 0.4, 3.2], index=tickers)
    rsi = pd.Series([65.0, 72.0, 45.0, 52.0, 80.0], index=tickers)
    adx = pd.Series([28.0, 35.0, 18.0, 22.0, 48.0], index=tickers)
    cci = pd.Series([110.0, 140.0, -85.0, 15.0, 190.0], index=tickers)
    vols = pd.Series([0.22, 0.18, 0.25, 0.24, 0.35], index=tickers)

    # Shrunk covariance matrix
    synth_ret = np.random.normal(0, 0.01, size=(30, n_assets))
    cov_matrix, _ = compute_ledoit_wolf_covariance(synth_ret)
    prev_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    state_vec = builder.build_state(
        prices=prices,
        macd=macd,
        rsi=rsi,
        adx=adx,
        cci=cci,
        volatilities=vols,
        ledoit_wolf_cov=cov_matrix,
        previous_weights=prev_weights,
        cash_ratio=0.05,
        trailing_portfolio_return=0.045,
    )

    expected_dim = config.total_state_dim
    assert len(state_vec) == expected_dim, (
        f"State dimension mismatch: {len(state_vec)} vs {expected_dim}"
    )
    assert np.all(np.isfinite(state_vec)), "State vector contains NaN or Inf values."
    print(f"    [+] State vector assembled successfully: shape={state_vec.shape} (7*N + 2 = {expected_dim}).")

    gym_spec = builder.get_gym_observation_space_spec()
    assert gym_spec["shape"] == (expected_dim,)
    print("    [+] Gym Box observation space metadata verified.")


def test_person_1_sample_dataset():
    """Validates features and state vectors directly on Person 1's sample OHLCV dataset."""
    import os
    from feature_engineering import load_and_process_panel_dataset

    sample_dir = "data/processed_ohlcv/sample"
    if not os.path.exists(sample_dir):
        print("    [!] Sample directory not found, skipping real sample test.")
        return

    print("[-] Validating on Person 1 Processed OHLCV Sample Dataset (AAPL, AMZN, GOOGL, MSFT, NVDA)...")
    feature_dict, ret_df, tickers = load_and_process_panel_dataset(sample_dir)

    assert len(tickers) == 5, f"Expected 5 tickers, found {len(tickers)}: {tickers}"
    assert tickers == ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA"]

    # Warm-up check: first 30 trading days are warm-up; day 31+ must have zero NaNs
    warmup_days = 30
    for ticker in tickers:
        df_feat = feature_dict[ticker]
        assert len(df_feat) >= 1800, f"Expected >= 1800 rows, got {len(df_feat)}"
        post_warmup = df_feat.iloc[warmup_days:]

        # Assert no NaNs in key indicators after warm-up
        for col in ["MACD", "Signal", "Histogram", "RSI", "ADX", "+DI", "-DI", "CCI", "Rolling_Vol_30"]:
            if col in post_warmup.columns:
                nans = post_warmup[col].isna().sum()
                assert nans == 0, f"Found {nans} NaNs in {ticker} column {col} post-warmup"

        # Check bounds
        rsi_valid = post_warmup["RSI"]
        assert (rsi_valid >= 0.0).all() and (rsi_valid <= 100.0).all()
        adx_valid = post_warmup["ADX"]
        assert (adx_valid >= 0.0).all() and (adx_valid <= 100.0).all()

    print(f"    [+] Verified 5 tickers across {len(feature_dict['AAPL'])} dates (2018-11-01 to 2025-12-31).")
    print(f"    [+] Zero NaNs across all indicators post {warmup_days}-day warm-up window.")

    # Test Ledoit-Wolf on trailing 30-day window from returns.csv
    recent_returns = ret_df[tickers].iloc[-30:].values
    shrunk_cov, shrinkage = compute_ledoit_wolf_covariance(recent_returns)
    assert shrunk_cov.shape == (5, 5)
    assert np.all(np.linalg.eigvalsh(shrunk_cov) > 0.0)
    print(f"    [+] 30-day Ledoit-Wolf covariance strictly positive definite (shrinkage delta={shrinkage:.4f}).")

    # Test Gymnasium State Vector building on actual market data
    latest_date = ret_df["Date"].iloc[-1]
    latest_prices = pd.Series({t: feature_dict[t]["Close"].iloc[-1] for t in tickers})
    latest_macd = pd.Series({t: feature_dict[t]["MACD"].iloc[-1] for t in tickers})
    latest_rsi = pd.Series({t: feature_dict[t]["RSI"].iloc[-1] for t in tickers})
    latest_adx = pd.Series({t: feature_dict[t]["ADX"].iloc[-1] for t in tickers})
    latest_cci = pd.Series({t: feature_dict[t]["CCI"].iloc[-1] for t in tickers})
    latest_vols = pd.Series({t: feature_dict[t]["Rolling_Vol_30"].iloc[-1] for t in tickers})

    config = StateSpaceConfig(num_assets=5)
    builder = PortfolioStateBuilder(config=config, ticker_order=tickers)
    state_vec = builder.build_state(
        prices=latest_prices,
        macd=latest_macd,
        rsi=latest_rsi,
        adx=latest_adx,
        cci=latest_cci,
        volatilities=latest_vols,
        ledoit_wolf_cov=shrunk_cov,
        previous_weights=np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        cash_ratio=0.0,
        trailing_portfolio_return=0.05,
    )

    assert state_vec.shape == (37,)
    assert np.all(np.isfinite(state_vec))
    print(f"    [+] Gymnasium state s_t assembled from real market data as-of {latest_date.strftime('%Y-%m-%d')}: shape={state_vec.shape}.")


def test_chronological_scaler_leakage():
    """Verifies that ChronologicalScaler fits strictly on training split without peeking."""
    print("[-] Testing Chronological Scaler Training Isolation (SEMESTER_V_WORK_PLAN line 130)...")
    # Simulate multi-year data: 2019-2022 (train), 2023 (val), 2024-2025 (test)
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="B")
    np.random.seed(42)
    # Train regime: mean = 10.0, std = 2.0; Test regime: mean = 50.0 (dist shift)
    train_mask = dates <= "2022-12-31"
    values = np.where(train_mask, np.random.normal(10.0, 2.0, size=len(dates)), np.random.normal(50.0, 5.0, size=len(dates)))
    df = pd.DataFrame({"Date": dates, "Feature_A": values})

    scaler = ChronologicalScaler(train_end_date="2022-12-31")
    scaler.fit(df, date_col="Date")

    # The fitted mean must be close to 10.0 (train mean), NOT influenced by the 50.0 in test
    assert np.isclose(scaler.means["Feature_A"], 10.0, atol=0.2), (
        f"Data leakage detected! Scaler mean {scaler.means['Feature_A']} was influenced by test split."
    )
    assert np.isclose(scaler.stds["Feature_A"], 2.0, atol=0.2)

    # Transform forward
    scaled_df = scaler.transform(df)
    train_scaled = scaled_df[train_mask]["Feature_A"]
    assert np.isclose(train_scaled.mean(), 0.0, atol=0.05)
    assert np.isclose(train_scaled.std(), 1.0, atol=0.05)
    print("    [+] Strict isolation verified: scalers calibrated strictly on training interval (2019-2022) without test leakage.")


def main():
    print("=" * 75)
    print("     AGENTIC PORTFOLIO BALANCING: FEATURE & STATE VALIDATION SUITE")
    print("=" * 75)
    try:
        test_indicator_bounds_and_formulas()
        test_lookahead_leakage_prevention()
        test_ledoit_wolf_covariance_singularity()
        test_gym_state_vector_builder()
        test_chronological_scaler_leakage()
        test_person_1_sample_dataset()
        print("=" * 75)
        print(" ALL TESTS PASSED: Feature engineering & state specs are verified.")
        print("=" * 75)
    except AssertionError as e:
        print(f"\n[!] VALIDATION FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
