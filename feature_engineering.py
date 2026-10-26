"""
Feature Engineering Engine for Agentic Financial Portfolio Balancing
=====================================================================
Implements quantitative market indicators and compact risk representations
as specified in the System Requirements Specification (SRS FR-03 & FR-04).

Indicators implemented:
  1. MACD (Moving Average Convergence Divergence) - Trend & Momentum
  2. RSI (Relative Strength Index) - Momentum & Mean Reversion
  3. ADX (Average Directional Index) - Trend Strength (requires OHLC)
  4. CCI (Commodity Channel Index) - Cyclical Turning Points (requires OHLC)
  5. Rolling Volatility (30-day annualized) - Market Risk
  6. Ledoit-Wolf Shrinkage Covariance - Robust Risk Matrix for Singular regimes (N > T)

All calculations are strictly backwards-looking to guarantee zero lookahead bias.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def calculate_macd(
    prices: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """Computes the Moving Average Convergence Divergence (MACD).

    Args:
        prices: Series of asset closing prices sorted chronologically.
        fast_period: Short exponential moving average window (default: 12).
        slow_period: Long exponential moving average window (default: 26).
        signal_period: Exponential moving average of MACD line (default: 9).

    Returns:
        DataFrame with columns ['MACD', 'Signal', 'Histogram'].
    """
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame(
        {
            "MACD": macd_line,
            "Signal": signal_line,
            "Histogram": histogram,
        },
        index=prices.index,
    )


def calculate_rsi(
    prices: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Computes the Relative Strength Index (RSI) using Wilder's smoothing.

    Args:
        prices: Series of asset closing prices sorted chronologically.
        period: Lookback window (default: 14).

    Returns:
        Series of RSI values bounded in [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's exponential smoothing (alpha = 1 / period)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    # Handle edge cases: zero loss gives 100; zero gain gives 0; flat gives 50
    flat = (avg_gain == 0.0) & (avg_loss == 0.0)
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    rsi = rsi.where(~flat, 50.0)

    return pd.Series(rsi, index=prices.index, name="RSI")


def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.DataFrame:
    """Computes the Average Directional Index (ADX) measuring trend strength.

    Requires full High, Low, Close (OHLC) observations.

    Args:
        high: High prices series.
        low: Low prices series.
        close: Close prices series.
        period: Smoothing period (default: 14).

    Returns:
        DataFrame with columns ['+DI', '-DI', 'ADX'] bounded in [0, 100].
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index)
    minus_dm_s = pd.Series(minus_dm, index=close.index)

    # Wilder's smoothing
    smoothed_tr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    smoothed_plus_dm = plus_dm_s.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()
    smoothed_minus_dm = minus_dm_s.ewm(
        alpha=1.0 / period, min_periods=period, adjust=False
    ).mean()

    plus_di = 100.0 * (smoothed_plus_dm / smoothed_tr.replace(0.0, np.nan)).fillna(0.0)
    minus_di = 100.0 * (smoothed_minus_dm / smoothed_tr.replace(0.0, np.nan)).fillna(0.0)

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = (100.0 * (di_diff / di_sum.replace(0.0, np.nan))).fillna(0.0)

    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean().fillna(0.0)

    return pd.DataFrame(
        {
            "+DI": plus_di,
            "-DI": minus_di,
            "ADX": adx,
        },
        index=close.index,
    )


def calculate_cci(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 20,
    constant: float = 0.015,
) -> pd.Series:
    """Computes the Commodity Channel Index (CCI) for cyclical turning points.

    Args:
        high: High prices series.
        low: Low prices series.
        close: Close prices series.
        period: Moving average and deviation window (default: 20).
        constant: Lambert's constant scale factor (default: 0.015).

    Returns:
        Series of CCI values.
    """
    typical_price = (high + low + close) / 3.0
    sma = typical_price.rolling(window=period, min_periods=period).mean()

    # Rolling Mean Absolute Deviation (MAD)
    def mean_abs_dev(window: np.ndarray) -> float:
        return np.mean(np.abs(window - np.mean(window)))

    mad = typical_price.rolling(window=period, min_periods=period).apply(
        mean_abs_dev, raw=True
    )

    cci = ((typical_price - sma) / (constant * mad.replace(0.0, np.nan))).fillna(0.0)
    return pd.Series(cci, index=close.index, name="CCI")


def calculate_rolling_volatility(
    returns: Union[pd.Series, pd.DataFrame],
    window: int = 30,
    annualization_factor: float = np.sqrt(252),
) -> Union[pd.Series, pd.DataFrame]:
    """Computes rolling annualized volatility over a trailing window.

    Args:
        returns: Daily percentage returns series or asset DataFrame.
        window: Trailing trading days window (default: 30 days).
        annualization_factor: Scale to annual units (default: sqrt(252)).

    Returns:
        Series or DataFrame of rolling annualized volatility.
    """
    return returns.rolling(window=window, min_periods=window).std() * annualization_factor


def compute_ledoit_wolf_covariance(
    returns_matrix: Union[np.ndarray, pd.DataFrame],
    shrinkage_target: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Computes the Ledoit-Wolf analytical shrinkage covariance matrix.

    Addresses the sample covariance singularity problem when the asset count N
    exceeds the trailing observation window T (e.g. N ~ 450 stocks vs T = 30 days),
    where sample covariance rank is at most T - 1 = 29.

    Formula:
        Sigma_shrunk = (1 - shrinkage) * S + shrinkage * F
        where S is sample covariance and F is structured shrinkage target.

    Args:
        returns_matrix: (T x N) matrix of asset returns over a trailing window.
        shrinkage_target: Optional override for shrinkage intensity in [0, 1].
                         If None, uses optimal analytical Ledoit-Wolf shrinkage.

    Returns:
        Tuple of (shrunk_covariance_matrix (N x N), shrinkage_intensity (float)).
    """
    data = (
        returns_matrix.values
        if isinstance(returns_matrix, pd.DataFrame)
        else np.asarray(returns_matrix)
    )

    # Clean any potential non-finite values along asset columns
    if not np.all(np.isfinite(data)):
        raise ValueError("Returns matrix contains NaN or Inf values.")

    lw = LedoitWolf(store_precision=False, assume_centered=False)
    lw.fit(data)

    if shrinkage_target is not None:
        # User-specified shrinkage intensity
        s_target = np.clip(shrinkage_target, 0.0, 1.0)
        # Empirical covariance
        emp_cov = np.cov(data, rowvar=False)
        # Diagonal / trace target
        trace_target = np.eye(data.shape[1]) * np.trace(emp_cov) / data.shape[1]
        shrunk_cov = (1.0 - s_target) * emp_cov + s_target * trace_target
        return shrunk_cov, float(s_target)

    return lw.covariance_, float(lw.shrinkage_)


def extract_asset_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    open_col: str = "Open",
    high_col: str = "High",
    low_col: str = "Low",
    close_col: str = "Close",
    vol_col: Optional[str] = "Volume",
) -> pd.DataFrame:
    """Extracts all core technical indicators for a single asset time series.

    Input DataFrame must be chronologically sorted.

    Returns:
        DataFrame with columns:
        ['Close', 'Return', 'MACD', 'MACD_Signal', 'MACD_Hist',
         'RSI', 'ADX', 'CCI', 'Rolling_Vol_30']
    """
    sorted_df = df.sort_values(by=date_col).copy()
    close = sorted_df[close_col]
    high = sorted_df[high_col]
    low = sorted_df[low_col]

    daily_return = close.pct_change()
    macd_df = calculate_macd(close)
    rsi_s = calculate_rsi(close)
    adx_df = calculate_adx(high, low, close)
    cci_s = calculate_cci(high, low, close)
    vol_30 = calculate_rolling_volatility(daily_return, window=30)

    features = pd.DataFrame(
        {
            "Date": sorted_df[date_col],
            "Close": close,
            "Daily_Return": daily_return,
            "MACD": macd_df["MACD"],
            "MACD_Signal": macd_df["Signal"],
            "MACD_Hist": macd_df["Histogram"],
            "RSI": rsi_s,
            "ADX": adx_df["ADX"],
            "+DI": adx_df["+DI"],
            "-DI": adx_df["-DI"],
            "CCI": cci_s,
            "Rolling_Vol_30": vol_30,
        },
        index=sorted_df.index,
    )

    return features


def load_and_process_panel_dataset(
    processed_dir: str = "data/processed_ohlcv/sample",
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[str]]:
    """Loads standardized panel CSVs from Person 1 pipeline and computes features.

    Args:
        processed_dir: Path to directory containing open.csv, high.csv, low.csv,
                       close.csv, volume.csv, and returns.csv.

    Returns:
        Tuple of:
          - dict mapping ticker -> DataFrame of technical indicators
          - DataFrame of daily returns (Date indexed, tickers as columns)
          - List of tickers in stable sorted order
    """
    import os

    open_df = pd.read_csv(os.path.join(processed_dir, "open.csv"), parse_dates=["Date"])
    high_df = pd.read_csv(os.path.join(processed_dir, "high.csv"), parse_dates=["Date"])
    low_df = pd.read_csv(os.path.join(processed_dir, "low.csv"), parse_dates=["Date"])
    close_df = pd.read_csv(os.path.join(processed_dir, "close.csv"), parse_dates=["Date"])
    vol_df = pd.read_csv(os.path.join(processed_dir, "volume.csv"), parse_dates=["Date"])
    ret_df = pd.read_csv(os.path.join(processed_dir, "returns.csv"), parse_dates=["Date"])

    tickers = sorted([c for c in close_df.columns if c != "Date"])
    feature_dict: Dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        asset_df = pd.DataFrame(
            {
                "Date": close_df["Date"],
                "Open": open_df[ticker],
                "High": high_df[ticker],
                "Low": low_df[ticker],
                "Close": close_df[ticker],
                "Volume": vol_df[ticker],
            }
        )
        feature_dict[ticker] = extract_asset_features(
            asset_df,
            date_col="Date",
            open_col="Open",
            high_col="High",
            low_col="Low",
            close_col="Close",
            vol_col="Volume",
        )

    return feature_dict, ret_df, tickers
