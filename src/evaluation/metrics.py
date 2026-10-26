"""
Comprehensive Financial Metrics and Evaluation Utilities

Includes:
- Total / Cumulative Returns
- Annualized Return and Volatility
- Sharpe Ratio (with configurable risk-free rate)
- Sortino Ratio (downside risk-adjusted)
- Calmar Ratio and Maximum Drawdown
- Inter-period Portfolio Turnover
- Transaction Cost-adjusted Net Returns
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, Optional


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Computes cumulative compound returns series: (1 + r_t).cumprod() - 1.
    """
    return (1.0 + returns).cumprod() - 1.0


def compute_annualized_return(
    returns: pd.Series, 
    periods_per_year: int = 252
) -> float:
    """
    Computes compound annualized return:
    R_ann = (1 + R_total) ** (periods_per_year / N) - 1
    """
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    compounded = (1.0 + returns).prod()
    return float(compounded ** (periods_per_year / n_periods) - 1.0)


def compute_annualized_volatility(
    returns: pd.Series, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized volatility:
    sigma_ann = std(returns) * sqrt(periods_per_year)
    """
    if len(returns) < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def compute_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized Sharpe Ratio:
    Sharpe = (R_ann - R_f) / sigma_ann
    
    Parameters:
    - returns: Daily returns series.
    - risk_free_rate: Annualized risk-free rate (e.g., 0.02 = 2%).
    - periods_per_year: Number of trading periods in a year (252 for daily).
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)
    if ann_vol == 0.0 or np.isnan(ann_vol):
        return 0.0
    return float((ann_return - risk_free_rate) / ann_vol)


def compute_max_drawdown(returns: pd.Series) -> float:
    """
    Computes the maximum drawdown as a negative decimal (e.g., -0.15 for -15%).
    MDD = min_t ( (V_t - peak_t) / peak_t )
    """
    if len(returns) == 0:
        return 0.0
    wealth_index = (1.0 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return float(drawdowns.min())


def compute_sortino_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252
) -> float:
    """
    Computes annualized Sortino Ratio focusing on downside risk below target/zero:
    Sortino = (R_ann - R_f) / (downside_std * sqrt(periods_per_year))
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    # Downside deviation relative to 0
    negative_returns = returns[returns < 0.0]
    if len(negative_returns) == 0:
        return 0.0
    downside_std = np.sqrt(np.mean(negative_returns ** 2))
    ann_downside_vol = downside_std * np.sqrt(periods_per_year)
    if ann_downside_vol == 0.0 or np.isnan(ann_downside_vol):
        return 0.0
    return float((ann_return - risk_free_rate) / ann_downside_vol)


def compute_calmar_ratio(
    returns: pd.Series, 
    periods_per_year: int = 252
) -> float:
    """
    Computes Calmar Ratio:
    Calmar = R_ann / |Max Drawdown|
    """
    ann_return = compute_annualized_return(returns, periods_per_year)
    mdd = abs(compute_max_drawdown(returns))
    if mdd == 0.0:
        return 0.0
    return float(ann_return / mdd)


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """
    Computes two-way (or one-way) portfolio turnover across rebalancing periods:
    Turnover_t = sum_i |w_{i, t} - w_{i, t-1}|
    
    Parameters:
    - weights_df: DataFrame indexed by Date or period, columns = asset tickers.
    
    Returns:
    - pd.Series of turnover per rebalancing event.
    """
    numeric_weights = weights_df.select_dtypes(include=[np.number])
    # Compute absolute differences row by row
    weight_diffs = numeric_weights.diff().abs()
    # Sum over assets for each rebalance date
    turnovers = weight_diffs.sum(axis=1)
    # Drop first row because there is no prior weight
    return turnovers.dropna()


def apply_transaction_costs(
    daily_returns: pd.DataFrame, 
    turnovers: pd.Series, 
    cost_bps: float = 10.0
) -> pd.DataFrame:
    """
    Deducts transaction costs from daily portfolio returns on monthly rebalancing days.
    Identifies the first trading day of each month and applies:
    Cost = Turnover_m * (cost_bps / 10000.0)
    
    Parameters:
    - daily_returns: DataFrame with 'Date' and 'Portfolio_Return'
    - turnovers: Series of turnover values across rebalancing events
    - cost_bps: Transaction cost in basis points (e.g., 5 bps = 0.0005, 10 bps = 0.0010)
    
    Returns:
    - DataFrame with 'Date', 'Gross_Return', 'Cost_Deduction', 'Net_Return'
    """
    cost_rate = cost_bps / 10000.0  # Convert bps to decimal
    df = daily_returns.copy()
    df['Gross_Return'] = df['Portfolio_Return']
    df['Cost_Deduction'] = 0.0
    
    if cost_rate <= 0.0 or len(turnovers) == 0:
        df['Net_Return'] = df['Gross_Return']
        return df

    # Find the first trading day of each unique month in daily_returns
    df['YearMonth'] = pd.to_datetime(df['Date']).dt.to_period('M')
    unique_months = df['YearMonth'].unique()

    # Align turnovers to subsequent rebalanced months
    turnover_values = list(turnovers.values)
    
    for m_idx, ym in enumerate(unique_months[1:]):
        if m_idx < len(turnover_values):
            turnover_val = float(turnover_values[m_idx])
            first_day_idx = df[df['YearMonth'] == ym].index[0]
            df.loc[first_day_idx, 'Cost_Deduction'] = turnover_val * cost_rate

    df['Net_Return'] = df['Gross_Return'] - df['Cost_Deduction']
    df.drop(columns=['YearMonth'], inplace=True)
    return df


def compute_comprehensive_metrics(
    returns: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252,
    prefix: str = ""
) -> Dict[str, float]:
    """
    Computes all standard performance and risk metrics in one call.
    Returns percentages formatted as 0-100 values where standard.
    """
    tot_ret = float((1.0 + returns).prod() - 1.0)
    ann_ret = compute_annualized_return(returns, periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)
    sharpe = compute_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = compute_sortino_ratio(returns, risk_free_rate, periods_per_year)
    mdd = compute_max_drawdown(returns)
    calmar = compute_calmar_ratio(returns, periods_per_year)
    
    p = f"{prefix} " if prefix else ""
    return {
        f"{p}Total Return (%)": tot_ret * 100.0,
        f"{p}Annualized Return (%)": ann_ret * 100.0,
        f"{p}Annualized Volatility (%)": ann_vol * 100.0,
        f"{p}Sharpe Ratio": sharpe,
        f"{p}Sortino Ratio": sortino,
        f"{p}Max Drawdown (%)": mdd * 100.0,
        f"{p}Calmar Ratio": calmar
    }
