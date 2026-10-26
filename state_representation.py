"""
State Representation & Gym Interface Specification (SRS FR-04)
==============================================================
Defines the continuous state space representation s_t for continuous-control
Deep Reinforcement Learning (PPO / SAC) portfolio balancing agents.

This module formalizes the interface handoff between Semester V feature
construction and Semester VI Gymnasium environment implementation.

State Vector Decomposition:
---------------------------
For an asset universe of N assets:
  1. Technical Indicators Matrix (N x 4):
     - MACD (normalized by price level)
     - RSI (min-max scaled to [-1, 1])
     - ADX (trend strength scaled to [0, 1])
     - CCI (robust z-score clipped to [-3, 3])
  2. Asset Volatility Vector (N x 1):
     - 30-day rolling annualized standard deviation
  3. Compact Risk Vector (N x 1):
     - Diagonal variance entries of the 30-day Ledoit-Wolf shrinkage matrix
  4. Previous Portfolio Allocation w_{t-1} (N x 1):
     - Current asset weights satisfying sum(w) = 1, w_i >= 0
  5. Cash / Portfolio Context (2 x 1):
     - Unallocated cash ratio
     - Trailing portfolio 30-day cumulative performance

Total Flattened State Dimension:
  D = N * 4 (tech) + N (vol) + N (risk) + N (weights) + 2 (context)
    = 7 * N + 2
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class StateSpaceConfig:
    """Configuration parameters for the DRL Observation Space."""
    num_assets: int
    include_macd: bool = True
    include_rsi: bool = True
    include_adx: bool = True
    include_cci: bool = True
    include_volatility: bool = True
    include_ledoit_wolf_diag: bool = True
    include_previous_weights: bool = True
    include_context: bool = True
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    @property
    def features_per_asset(self) -> int:
        count = 0
        if self.include_macd:
            count += 1
        if self.include_rsi:
            count += 1
        if self.include_adx:
            count += 1
        if self.include_cci:
            count += 1
        if self.include_volatility:
            count += 1
        if self.include_ledoit_wolf_diag:
            count += 1
        if self.include_previous_weights:
            count += 1
        return count

    @property
    def total_state_dim(self) -> int:
        base_dim = self.num_assets * self.features_per_asset
        if self.include_context:
            base_dim += 2
        return base_dim


class PortfolioStateBuilder:
    """Constructs normalized observation vectors s_t for DRL portfolio agents."""

    def __init__(self, config: StateSpaceConfig, ticker_order: List[str]):
        """
        Args:
            config: StateSpaceConfig configuration instance.
            ticker_order: Fixed, deterministic list of asset tickers.
        """
        self.config = config
        self.ticker_order = list(ticker_order)
        if len(self.ticker_order) != config.num_assets:
            raise ValueError(
                f"Ticker order length ({len(self.ticker_order)}) does not match "
                f"num_assets ({config.num_assets})."
            )

    def normalize_technical_features(
        self,
        macd: pd.Series,
        prices: pd.Series,
        rsi: pd.Series,
        adx: pd.Series,
        cci: pd.Series,
    ) -> np.ndarray:
        """Applies bounded scaling to technical features to stabilize RL gradients.

        Returns:
            (N x 4) array of scaled indicators ordered by self.ticker_order.
        """
        macd_aligned = macd.reindex(self.ticker_order).values
        price_aligned = prices.reindex(self.ticker_order).values
        rsi_aligned = rsi.reindex(self.ticker_order).values
        adx_aligned = adx.reindex(self.ticker_order).values
        cci_aligned = cci.reindex(self.ticker_order).values

        # 1. MACD normalized by price level (percentage difference)
        macd_norm = np.where(
            price_aligned > 0, macd_aligned / price_aligned, 0.0
        )
        macd_norm = np.clip(macd_norm * 100.0, -5.0, 5.0)

        # 2. RSI normalized from [0, 100] to [-1.0, 1.0]
        rsi_norm = (np.nan_to_num(rsi_aligned, nan=50.0) - 50.0) / 50.0
        rsi_norm = np.clip(rsi_norm, -1.0, 1.0)

        # 3. ADX normalized from [0, 100] to [0.0, 1.0]
        adx_norm = np.nan_to_num(adx_aligned, nan=20.0) / 100.0
        adx_norm = np.clip(adx_norm, 0.0, 1.0)

        # 4. CCI scaled (standard trading range -100 to +100 mapped to [-1, 1])
        cci_norm = np.nan_to_num(cci_aligned, nan=0.0) / 100.0
        cci_norm = np.clip(cci_norm, -3.0, 3.0)

        return np.column_stack([macd_norm, rsi_norm, adx_norm, cci_norm])

    def build_state(
        self,
        prices: pd.Series,
        macd: pd.Series,
        rsi: pd.Series,
        adx: pd.Series,
        cci: pd.Series,
        volatilities: pd.Series,
        ledoit_wolf_cov: np.ndarray,
        previous_weights: np.ndarray,
        cash_ratio: float = 0.0,
        trailing_portfolio_return: float = 0.0,
    ) -> np.ndarray:
        """Assembles and flattens the multi-factor market state vector s_t.

        Returns:
            1D numpy array of shape (total_state_dim,) bounded and finite.
        """
        features_list = []

        # 1. Technical Indicators (N x 4)
        tech_features = self.normalize_technical_features(
            macd=macd, prices=prices, rsi=rsi, adx=adx, cci=cci
        )
        features_list.append(tech_features.flatten())

        # 2. Rolling Volatilities (N x 1)
        vol_aligned = volatilities.reindex(self.ticker_order).values
        vol_clean = np.nan_to_num(vol_aligned, nan=0.20)
        vol_scaled = np.clip(vol_clean, 0.0, 2.0)
        features_list.append(vol_scaled)

        # 3. Compact Ledoit-Wolf Risk Vector (N x 1, diagonal variances)
        if ledoit_wolf_cov.shape != (self.config.num_assets, self.config.num_assets):
            raise ValueError(
                f"Ledoit-Wolf covariance shape {ledoit_wolf_cov.shape} does not match "
                f"num_assets ({self.config.num_assets}, {self.config.num_assets})."
            )
        diag_risk = np.diag(ledoit_wolf_cov)
        diag_risk_scaled = np.clip(np.sqrt(np.maximum(diag_risk, 0.0)) * np.sqrt(252), 0.0, 2.0)
        features_list.append(diag_risk_scaled)

        # 4. Previous Portfolio Allocation w_{t-1} (N x 1)
        w_prev = np.asarray(previous_weights, dtype=np.float64).flatten()
        if len(w_prev) != self.config.num_assets:
            raise ValueError(
                f"Previous weights length {len(w_prev)} does not match num_assets {self.config.num_assets}."
            )
        features_list.append(w_prev)

        # 5. Portfolio Context (2 x 1)
        if self.config.include_context:
            context = np.array([
                np.clip(cash_ratio, 0.0, 1.0),
                np.clip(trailing_portfolio_return, -1.0, 2.0),
            ], dtype=np.float64)
            features_list.append(context)

        # Flatten into final 1D continuous state vector s_t
        state_vector = np.concatenate(features_list).astype(np.float32)

        # Safety assert on dimension
        expected_dim = self.config.total_state_dim
        if len(state_vector) != expected_dim:
            raise ValueError(
                f"Constructed state dimension ({len(state_vector)}) does not match "
                f"configured dimension ({expected_dim})."
            )

        return state_vector

    def get_gym_observation_space_spec(self) -> Dict[str, object]:
        """Provides Gym / Gymnasium Box space metadata for Semester VI."""
        low_bound, high_bound = self.config.clip_range
        return {
            "type": "Box",
            "shape": (self.config.total_state_dim,),
            "dtype": "float32",
            "low": float(low_bound),
            "high": float(high_bound),
            "description": (
                f"Continuous multi-factor state for {self.config.num_assets} assets. "
                f"Includes technical momentum, volatility, Ledoit-Wolf risk, and prior allocation."
            ),
        }


class ChronologicalScaler:
    """Enforces strict zero lookahead bias by fitting transformation parameters

    exclusively on the in-sample training partition (e.g. 2019-01-01 to 2022-12-31)
    and projecting them forward onto validation and test periods without data leakage
    (SEMESTER_V_WORK_PLAN.md line 130).
    """

    def __init__(self, train_end_date: str = "2022-12-31"):
        self.train_end_date = pd.to_datetime(train_end_date)
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, date_col: str = "Date") -> "ChronologicalScaler":
        """Fits mean and std strictly on observations <= train_end_date."""
        if date_col in df.columns:
            train_mask = pd.to_datetime(df[date_col]) <= self.train_end_date
            train_data = df[train_mask].drop(columns=[date_col])
        else:
            train_mask = pd.to_datetime(df.index) <= self.train_end_date
            train_data = df[train_mask]

        if train_data.empty:
            raise ValueError(f"No observations found before train_end_date: {self.train_end_date}")

        for col in train_data.columns:
            self.means[col] = float(train_data[col].mean())
            s = float(train_data[col].std())
            self.stds[col] = s if s > 1e-8 else 1.0

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies saved in-sample parameters forward onto any partition."""
        if not self.is_fitted:
            raise ValueError("ChronologicalScaler must be fit on training data before transforming.")
        scaled = df.copy()
        for col, m in self.means.items():
            if col in scaled.columns:
                scaled[col] = (scaled[col] - m) / self.stds[col]
        return scaled
