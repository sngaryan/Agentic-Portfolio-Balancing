"""
Feature Engineering and State Space Specification Package (SRS FR-03 & FR-04)
=============================================================================
Provides quantitative indicator computation, robust shrinkage covariance estimation,
and Gymnasium observation state representation for continuous-control DRL agents.
"""

from feature_engineering import (
    calculate_macd,
    calculate_rsi,
    calculate_adx,
    calculate_cci,
    calculate_rolling_volatility,
    compute_ledoit_wolf_covariance,
    extract_asset_features,
    load_and_process_panel_dataset,
)
from state_representation import (
    StateSpaceConfig,
    PortfolioStateBuilder,
)

__all__ = [
    "calculate_macd",
    "calculate_rsi",
    "calculate_adx",
    "calculate_cci",
    "calculate_rolling_volatility",
    "compute_ledoit_wolf_covariance",
    "extract_asset_features",
    "load_and_process_panel_dataset",
    "StateSpaceConfig",
    "PortfolioStateBuilder",
]
