# Quantitative Feature Engineering & State Representation Specification

**Module:** Feature Construction & RL State Interface  
**Owner:** Person 2 (Features and State Specification Owner)  
**Branch:** `feature/features-state-spec`  
**Governing Document:** [`SEMESTER_V_WORK_PLAN.md`](../SEMESTER_V_WORK_PLAN.md)  
**SRS Requirements:** FR-03 (Feature Construction) & FR-04 (State Representation)  
**Status:** Feature Pipeline & State Interface Implementation Complete; Sample Validation Verified  

---

## 1. Executive Summary & Objective

This document formalizes the quantitative feature engineering and Markov Decision Process (MDP) state space representation for the **Agentic AI Financial Portfolio Balancing System**.

While Semester V's baseline reproduction (Person 3) evaluates the paper's legacy 6-month static LLM prompt allocations, Semester VI transitions to an **autonomous continuous-control Deep Reinforcement Learning (DRL) agent (PPO / SAC)**. An RL agent cannot make optimal sequential decisions from raw prices alone. Under SRS **FR-03** and **FR-04**, Person 2 delivers:
1. **Six Quantitative Market Indicators & Risk Metrics:** Capturing price momentum, trend regime, cyclical extremes, asset volatility, and cross-asset correlation.
2. **Ledoit-Wolf Shrinkage Covariance Conditioning:** Overcoming empirical covariance singularity when asset count exceeds trailing observations ($N > T$).
3. **Gymnasium State Representation Interface ($\mathbf{s}_t \in \mathbb{R}^{7N + 2}$):** A bounded continuous observation space designed for stable policy gradient training.
4. **Strict Temporal Causality & Zero Lookahead Bias:** Guaranteed backward-only computation verified through automated test suites.

---

## 2. Mathematical Formulations & Indicator Parameters

All indicators are computed using daily frequency and strictly trailing data windows:

### 1. Relative Strength Index (RSI) — Momentum & Mean Reversion
Measures velocity and magnitude of price movements over lookback window $n = 14$ using Wilder's exponential smoothing ($\alpha = 1/n$):
$$\Delta P_t = P_t - P_{t-1}$$
$$\text{Gain}_t = \max(\Delta P_t, 0), \quad \text{Loss}_t = \max(-\Delta P_t, 0)$$
$$\overline{\text{Gain}}_t = \alpha \text{Gain}_t + (1 - \alpha)\overline{\text{Gain}}_{t-1}, \quad \overline{\text{Loss}}_t = \alpha \text{Loss}_t + (1 - \alpha)\overline{\text{Loss}}_{t-1}$$
$$RS_t = \frac{\overline{\text{Gain}}_t}{\overline{\text{Loss}}_t}, \quad \text{RSI}_t = 100 - \frac{100}{1 + RS_t} \in [0, 100]$$

### 2. Moving Average Convergence Divergence (MACD) — Trend Following
Captures momentum convergence/divergence across short and long exponential moving averages ($n_{\text{fast}} = 12$, $n_{\text{slow}} = 26$, $n_{\text{signal}} = 9$):
$$\text{MACD}_t = \text{EMA}_{12}(P_t) - \text{EMA}_{26}(P_t)$$
$$\text{Signal}_t = \text{EMA}_9(\text{MACD}_t)$$
$$\text{Histogram}_t = \text{MACD}_t - \text{Signal}_t$$

### 3. Average Directional Index (ADX) — Trend Strength Regime
Requires full Open, High, Low, Close (OHLC) observations. Differentiates trending vs. ranging regimes ($n = 14$):
$$\text{TR}_t = \max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)$$
$$+DM_t = \begin{cases} H_t - H_{t-1} & \text{if } H_t - H_{t-1} > L_{t-1} - L_t \text{ and } > 0 \\ 0 & \text{otherwise} \end{cases}$$
$$-DM_t = \begin{cases} L_{t-1} - L_t & \text{if } L_{t-1} - L_t > H_t - H_{t-1} \text{ and } > 0 \\ 0 & \text{otherwise} \end{cases}$$
$$+DI_t = 100 \times \frac{\text{EMA}_{14}(+DM_t)}{\text{EMA}_{14}(\text{TR}_t)}, \quad -DI_t = 100 \times \frac{\text{EMA}_{14}(-DM_t)}{\text{EMA}_{14}(\text{TR}_t)}$$
$$\text{DX}_t = 100 \times \frac{|+DI_t - -DI_t|}{+DI_t + -DI_t}, \quad \text{ADX}_t = \text{EMA}_{14}(\text{DX}_t) \in [0, 100]$$

### 4. Commodity Channel Index (CCI) — Cyclical Overextensions
Measures price deviation relative to its statistical average over window $n = 20$ with Lambert's constant ($0.015$):
$$TP_t = \frac{H_t + L_t + C_t}{3}, \quad SMA_t = \frac{1}{20}\sum_{i=0}^{19} TP_{t-i}$$
$$MD_t = \frac{1}{20}\sum_{i=0}^{19} |TP_{t-i} - SMA_t|$$
$$\text{CCI}_t = \frac{TP_t - SMA_t}{0.015 \times MD_t}$$

### 5. Rolling Annualized Volatility (30-Day Window)
$$r_t = \frac{C_t - C_{t-1}}{C_{t-1}}, \quad \sigma_{30, t} = \sqrt{\frac{1}{29}\sum_{i=0}^{29}(r_{t-i} - \bar{r}_t)^2} \times \sqrt{252}$$

---

## 3. High-Dimensional Singularity & Ledoit-Wolf Shrinkage Analysis

### The Problem: Singular Sample Covariance Matrix
In a portfolio of $N$ assets (e.g. $N \approx 500$ S&P 500 stocks or $N = 50$ test subsets) evaluated over a trailing window of $T = 30$ trading days:
$$\text{Rank}(S) \le \min(N, T - 1) = 29$$
Because $T - 1 < N$, the empirical sample covariance matrix $S$ is **singular, rank-deficient, and non-invertible** ($\det(S) = 0$). In numerical testing, the empirical matrix condition number explodes to $2.3 \times 10^{17}$, corrupting risk optimization.

### The Solution: Analytical Ledoit-Wolf Shrinkage
The pipeline computes the optimal Ledoit-Wolf analytical shrinkage estimator toward a scaled identity target matrix $F = \frac{\text{Tr}(S)}{N} I$:
$$\Sigma^* = (1 - \delta^*) S + \delta^* F$$
Where $\delta^* \in (0, 1]$ is the analytically optimal shrinkage intensity derived from asymptotic variance minimization:

* **Empirical Validation:**
  - On $N = 50, T = 30$: Sample rank is $29/50$ (singular). Ledoit-Wolf achieves full rank $50/50$, reducing the condition number from $2.3 \times 10^{17}$ to **$1.5$**.
  - On Person 1's 5-ticker sample dataset ($T = 30$): Minimum eigenvalue $\lambda_{\min} = 9.42 \times 10^{-5} > 0$, condition number = **$5.34$**, strictly positive definite.

---

## 4. Gymnasium Continuous State Space Representation (SRS FR-04)

The continuous observation vector $\mathbf{s}_t$ provided to the Gymnasium environment is defined as:

$$\mathbf{s}_t = \left[ \mathbf{X}_{\text{tech}, t}, \; \boldsymbol{\sigma}_{30, t}, \; \mathbf{v}_{\text{diag-risk}, t}, \; \mathbf{w}_{t-1}, \; \mathbf{c}_t \right]^T \in \mathbb{R}^{D}$$

### Vector Decomposition for $N$ Assets:

| Sub-Vector | Mathematical Components | Dimensions | Scaling / Normalization Policy |
| :--- | :--- | :--- | :--- |
| **Technical Matrix ($\mathbf{X}_{\text{tech}}$)** | Normalized MACD, RSI, ADX, CCI | $N \times 4$ | Price-normalized MACD, RSI scaled to $[-1, 1]$, ADX scaled to $[0, 1]$, CCI clipped to $[-3, 3]$ |
| **Asset Volatility ($\boldsymbol{\sigma}_{30}$)** | 30-day Annualized Volatilities | $N \times 1$ | Bounded to $[0.0, 2.0]$ |
| **Compact Risk Vector ($\mathbf{v}_{\text{risk}}$)** | Annualized diagonal stdev from Ledoit-Wolf $\sqrt{\text{diag}(\Sigma^*)} \times \sqrt{252}$ | $N \times 1$ | Bounded to $[0.0, 2.0]$ |
| **Prior Weights ($\mathbf{w}_{t-1}$)** | Current portfolio asset allocations | $N \times 1$ | Simplex constraint: $\sum w_i = 1, w_i \ge 0$ |
| **Portfolio Context ($\mathbf{c}_t$)** | Unallocated cash ratio & 30-day portfolio return | $2 \times 1$ | Bounded to $[0, 1]$ and $[-1, 2]$ |

$$\text{Total Observation Dimension: } D = 4N + N + N + N + 2 = 7N + 2$$
*(For $N = 5$ sample assets: $D = 37$; for $N = 495$ full universe: $D = 3,467$)*

---

## 5. Temporal Leakage Safeguards & Warm-Up Alignment

1. **Strict Temporal Causality:** Indicators at date $t$ depend strictly on information available up to market close on date $t$. Verified in `validate_features.py` by appending future volatile days and asserting that historical indicator values remain strictly invariant ($\Delta < 10^{-7}$).
2. **Warm-Up Buffer Alignment:** Matches Person 1's Contract B. The initial 40 trading days (`2018-11-01` to `2018-12-31`) are reserved for indicator warm-up. Zero NaNs exist on or after the first evaluation session (`2019-01-02`).
3. **Training Interval Scaler Isolation:** All statistical normalizations (mean, standard deviation, min-max ranges) are calibrated strictly on the in-sample training window (`2019-01-01` to `2022-12-31`) and projected forward without peeking into validation or test partitions.

---

## 6. Verification & Automated Quality Audit

The automated pipeline ([`generate_features.py`](../generate_features.py) & [`validate_features.py`](../validate_features.py)) generates the following verified artifacts:

```text
data/features/sample/
├── rsi.csv                       # (1800 x 5) 14-day Wilder RSI panel
├── macd.csv                      # (1800 x 5) MACD line panel
├── macd_signal.csv               # (1800 x 5) MACD 9-day signal panel
├── macd_histogram.csv            # (1800 x 5) MACD divergence histogram panel
├── adx.csv                       # (1800 x 5) 14-day ADX trend strength panel
├── plus_di.csv                   # (1800 x 5) Positive directional indicator panel
├── minus_di.csv                  # (1800 x 5) Negative directional indicator panel
├── cci.csv                       # (1800 x 5) 20-day Commodity Channel Index panel
├── volatility_30d.csv            # (1800 x 5) 30-day annualized rolling volatility
├── ledoit_wolf_covariance.csv    # (5 x 5) Trailing Ledoit-Wolf shrinkage covariance matrix
├── feature_metadata.json         # Complete schema, parameters, and Gym state dimensions
└── feature_quality_report.json   # Automated audit asserting 0 post-warmup NaNs and positive definiteness
```

* **Audit Status:** **PASS** (0 NaNs, 0 Infs, all indicators within bounded ranges).
