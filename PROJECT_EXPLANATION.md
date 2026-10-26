# Comprehensive Technical & Financial Guide: LLM-BLM

> **Integrating LLM-Generated Views into Mean-Variance Optimization Using the Black-Litterman Model**  
> *Official Implementation Guide for the ICLR 2025 Financial AI Workshop Paper*

---

## Table of Contents
1. [High-Level Overview](#1-high-level-overview)
2. [Financial & Mathematical Glossary](#2-financial--mathematical-glossary)
   - [Expected Return & Historical Return](#expected-return--historical-return)
   - [Variance & Covariance Matrix ($\Sigma$)](#variance--covariance-matrix-sigma)
   - [Beta ($\beta$) & CAPM Equilibrium Return ($\Pi$)](#beta-beta--capm-equilibrium-return-pi)
   - [Markowitz Mean-Variance Optimization (MVO)](#markowitz-mean-variance-optimization-mvo)
   - [The Black-Litterman Model (BLM) Formula & Parameters](#the-black-litterman-model-blm-formula--parameters)
   - [Portfolio Backtesting & Evaluation Theory](#portfolio-backtesting--evaluation-theory)
3. [End-to-End System Pipeline](#3-end-to-end-system-pipeline)
4. [In-Depth File Breakdown & Code Walkthrough](#4-in-depth-file-breakdown--code-walkthrough)
   - [1. `run.py` — Data Ingestion & LLM View Extraction](#1-runpy--data-ingestion--llm-view-extraction)
   - [2. `baselines.py` — Benchmark Strategies](#2-baselinespy--benchmark-strategies)
   - [3. `evaluate_multiple.py` — Black-Litterman Optimization Engine](#3-evaluate_multiplepy--black-litterman-optimization-engine)
   - [4. `calculate_llm_returns.py` — Detailed Walkthrough & Performance Engine](#4-calculate_llm_returnspy--detailed-walkthrough--performance-engine)
5. [Directory & Data Structure](#5-directory--data-structure)
6. [Step-by-Step Execution Guide](#6-step-by-step-execution-guide)

---

## 1. High-Level Overview

In classical quantitative finance, **Markowitz Mean-Variance Optimization (MVO)** computes optimal asset weights to maximize return for a given level of risk. However, MVO has a famous flaw: it is hypersensitive to small estimation errors in expected returns, often producing extreme, uninvestable portfolio allocations (e.g. 100% in one volatile stock).

The **Black-Litterman Model (BLM)** solves this by:
1. Starting with the **Market Equilibrium** (the broad market consensus) as a prior baseline.
2. Ingesting **Investor Views** (subjective return predictions) along with an explicit **Uncertainty/Confidence matrix**.
3. Applying **Bayesian updating** to blend the market baseline with the investor views, producing a stable, well-diversified set of **posterior expected returns**.

### The Innovation of this Codebase
Instead of human analysts providing the views, this project uses **Large Language Models (LLMs)** (LLaMA-3.1, Gemma-7B, Qwen2) to generate stock return forecasts. By querying each LLM **30 times per stock**, the framework measures the model's prediction consensus (mean $\rightarrow$ expected return view $Q$) and its uncertainty (variance $\rightarrow$ diagonal matrix $\Omega$).

```
[S&P 500 Historical Data] ──> [LLM Prompt (30x per stock)] ──> Views (Q) & Uncertainty (Ω)
                                                                           │
[Market Caps + History]   ──> [Market Equilibrium Returns (Π) & Cov (Σ)]   │
                                                   │                       │
                                                   ▼                       ▼
                                       [Black-Litterman Bayesian Updating]
                                                   │
                                                   ▼
                                     [Posterior Returns & Optimal Weights]
                                                   │
                                                   ▼
                                    [Out-of-Sample Performance Testing]
```

---

## 2. Financial & Mathematical Glossary

### Expected Return & Historical Return
* **Daily Percentage Return ($r_t$)**: The relative price change from trading day $t-1$ to day $t$:
  $$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
* **Expected Return ($\mu$)**: The anticipated mean return of an asset over an upcoming investment horizon.

---

### Variance & Covariance Matrix ($\Sigma$)
* **Variance ($\sigma^2$)**: Measures how much an individual stock's return fluctuates around its average. Higher variance indicates higher volatility and risk.
* **Covariance ($\sigma_{ij}$)**: Measures how two stocks move together:
  - **Positive covariance**: When Stock A rises, Stock B tends to rise.
  - **Negative covariance**: When Stock A rises, Stock B tends to fall.
  - **Zero covariance**: Stock movements are uncorrelated (ideal for diversification).
* **Covariance Matrix ($\Sigma$)**: An $N \times N$ matrix for $N$ stocks where the diagonal elements $\Sigma_{ii}$ are individual asset variances, and off-diagonal elements $\Sigma_{ij}$ are pairwise covariances:
  $$\Sigma = \text{cov}(R) \in \mathbb{R}^{N \times N}$$
  In code: `sigma = np.cov(returns.T)`

---

### Beta ($\beta$) & CAPM Equilibrium Return ($\Pi$)
* **Market Return ($R_m$)**: The weighted average return of the entire market, using market capitalizations:
  $$R_{m,t} = \sum_{i=1}^N w_{\text{mkt}, i} \cdot r_{i,t}$$
* **Market Beta ($\beta_i$)**: A measure of a stock's systematic risk relative to the overall market:
  $$\beta_i = \frac{\text{cov}(r_i, R_m)}{\text{var}(R_m)}$$
  - $\beta = 1.0$: The stock moves in lockstep with the market.
  - $\beta > 1.0$: The stock is more volatile than the market (e.g. high-growth tech).
  - $\beta < 1.0$: The stock is defensive/less volatile (e.g. utilities).
* **Market Equilibrium Return ($\Pi$)**: The expected return implied by the Capital Asset Pricing Model (CAPM):
  $$\Pi_i = \beta_i \cdot (R_m - R_f)$$
  where $R_f$ is the risk-free rate (set to 2% in `evaluate_multiple.py`).

---

### Markowitz Mean-Variance Optimization (MVO)
Given expected returns $\mu$ and covariance matrix $\Sigma$, MVO solves the quadratic optimization problem:
$$\min_w \left[ w^T \Sigma w - \lambda (w^T \mu) \right]$$
$$\text{subject to } \sum_{i=1}^N w_i = 1, \quad w_i \ge 0 \quad (\text{long-only})$$
* $w^T \Sigma w$: Portfolio variance (total risk).
* $w^T \mu$: Expected portfolio return.
* $\lambda$: Risk-aversion / trade-off parameter (set to `0.1`).

---

### The Black-Litterman Model (BLM) Formula & Parameters

The Black-Litterman master formula for **Posterior Expected Returns** ($\hat{\mu}$) is:

$$\hat{\mu} = M \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$
$$\text{where } M = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1}$$

#### Matrix-by-Matrix Breakdown:
| Symbol | Code Variable | Dimensions | Financial Meaning |
| :--- | :--- | :--- | :--- |
| $\Pi$ | `market_equilibrium_return` | $N \times 1$ | The CAPM prior equilibrium expected returns. |
| $\Sigma$ | `sigma` | $N \times N$ | Historical covariance matrix of asset returns. |
| $\tau$ (Tau) | `tau` | Scalar ($0.025$) | Uncertainty scalar on the market prior. Lower $\tau$ gives more weight to the market prior $\Pi$; higher $\tau$ trusts the views $Q$ more. |
| $Q$ | `Q` | $K \times 1$ | The LLM-predicted expected returns vector (mean of 30 inferences per stock). |
| $P$ | `P` | $K \times N$ | Identity pick matrix linking each view to its respective asset ($P = I$). |
| $\Omega$ (Omega)| `Omega` | $K \times K$ | Diagonal matrix representing the **uncertainty of the views**. In this project, $\Omega_{ii} = \text{var}(\text{LLM 30 predictions for stock } i)$. |
| $\hat{\mu}$ | `posterior_returns` | $N \times 1$ | The balanced, optimal expected return vector used in final portfolio construction. |

---

### Portfolio Backtesting & Evaluation Theory

To understand how returns are evaluated in `calculate_llm_returns.py`, several fundamental quantitative backtesting concepts are used:

#### 1. Out-of-Sample (OOS) Walk-Forward Testing
* **In-Sample (Training Window $T$)**: We use historical data up to month $T$ (e.g., June 2024) to train the model, query the LLM, and solve for portfolio weights $\mathbf{w}$.
* **Out-of-Sample (Testing Window $T+1$)**: We hold those weights $\mathbf{w}$ constant during the next, unseen month $T+1$ (e.g., July 2024) and observe actual performance.
* **Why it matters**: This mirrors real-world trading, where you must make decisions today without knowing tomorrow's stock prices.

#### 2. Prevention of Lookahead Bias (Data Leakage)
* **Lookahead Bias** happens when future information accidentally leaks into the decision-making process (e.g., using July prices to decide June weights).
* The codebase strictly isolates training windows from evaluation windows to guarantee zero lookahead bias.

#### 3. Linear Combination of Portfolio Returns (Dot Product)
For any given trading day $t$, if an investor holds weight vector $\mathbf{w} = [w_1, w_2, \dots, w_N]^T$ and assets realize returns $\mathbf{r}_t = [r_{1,t}, r_{2,t}, \dots, r_{N,t}]^T$, the daily portfolio return $R_{p,t}$ is the dot product:
$$R_{p,t} = \mathbf{w}^T \mathbf{r}_t = \sum_{i=1}^N w_i \cdot r_{i,t}$$

#### 4. Periodic Rebalancing (Monthly Frequency)
* In this strategy, asset weights are rebalanced **once per month** (at the start of each month). 
* During the month, the fixed weights represent the target allocation.

#### 5. Cumulative Compounded Returns
Daily returns compound over time. The cumulative return from day $1$ to day $T$ is:
$$\text{Cumulative Return}_T = \prod_{t=1}^T (1 + R_{p,t}) - 1$$
This is the metric plotted in [`figure/cumulative_returns2.png`](file:///c:/Users/aryan/project/LLM-BLM/figure/cumulative_returns2.png) to compare the total wealth trajectory of LLM portfolios against the S&P 500.

#### 6. Sharpe Ratio & Volatility
* **Daily Volatility ($\sigma_p$)**: $\sigma_p = \text{std}(R_{p,t})$
* **Annualized Sharpe Ratio ($SR$)**: Evaluates excess return earned per unit of risk:
  $$SR = \frac{\sqrt{252} \cdot \bar{R}_p}{\sigma_p}$$
  where 252 is the number of trading days in a standard calendar year.

---

## 3. End-to-End System Pipeline

```
Wikipedia Ticker List
         │
         ▼
yfinance Historical Prices
         │
         ▼
Daily Returns (yfinance/returns_*.csv)
   ├──────────────────────────────┬───────────────────────────────┐
   │                              │                               │
   ▼                              ▼                               │
run.py (LLM Views)          baselines.py (1/N & MVO)    market_caps.json
   │                              │                               │
   ▼                              ▼                               │
JSON Predictions (30x)      responses_portfolios/                 │
   │                              │                               │
   └──────────────────────────────┼───────────────────────────────┘
                                  │
                                  ▼
                    evaluate_multiple.py (Black-Litterman)
                                  │
                                  ▼
                    results/*_weights_tau_0.025.csv
                                  │
                                  ▼
                    calculate_llm_returns.py
                                  │
                                  ▼
                    Final Performance & Visualization (figure/)
```

---

## 4. In-Depth File Breakdown & Code Walkthrough

### 1. `run.py` — Data Ingestion & LLM View Extraction
* **Purpose**: Collects market price history for all S&P 500 stocks and queries the LLM 30 times per stock to generate predictions and confidence intervals.

#### Key Code Chunks:
1. **Model Selection & Client Initialization (`run.py:L17-L40`)**:
   Sets up the OpenAI SDK client pointing to the self-hosted or remote API endpoint for `llama`, `gemma`, or `qwen`.
2. **Scraping S&P 500 List & Downloading Prices (`run.py:L45-L70`)**:
   Uses `pd.read_html` on Wikipedia to get ticker symbols and sectors, then downloads monthly data through `yf.download()` into `yfinance/data_*.csv` and calculates daily percentage returns in `yfinance/returns_*.csv`.
3. **Structured Prompt Construction (`run.py:L83-L100`)**:
   Injects ticker name, GICS sector, sub-industry, and past daily return series into a structured prompt, enforcing strict JSON output via `Pydantic` schema (`ResearchPaperExtraction`).
4. **Sampling 30 Times for Uncertainty Estimation (`run.py:L116-L156`)**:
   Queries the model 30 times in a loop. Stores all 30 float outputs under `data_dict[ticker]['expected_return']` and saves the result to `responses/{model_name}_{start}_{end}.json`.

---

### 2. `baselines.py` — Benchmark Strategies
* **Purpose**: Constructs control portfolios without any LLM input to benchmark performance against the LLM-BLM model.

#### Key Code Chunks:
1. **Rolling Monthly Window (`baselines.py:L19-L46`)**:
   Uses month $T$ (`train_start` to `train_end`) for training and month $T+1$ (`test_start` to `test_end`) for forward testing.
2. **Equal-Weighted Portfolio ($1/N$) (`baselines.py:L74-L88`)**:
   Assigns uniform weight $w_i = \frac{1}{N}$ across all stocks and computes daily forward returns in the test period.
3. **Mean-Variance Optimized Portfolio (`baselines.py:L92-L126`)**:
   Calculates historical mean returns and the covariance matrix $\Sigma$. Solves the constrained minimization problem using `scipy.optimize.minimize` (method `'SLSQP'`).
4. **Aggregating Baseline Results (`baselines.py:L129-L135`)**:
   Concatenates monthly CSVs into `equal_weighted_portfolio.csv` and `optimized_portfolio.csv`.

---

### 3. `evaluate_multiple.py` — Black-Litterman Optimization Engine
* **Purpose**: Combines market capitalization equilibrium returns with LLM views to compute optimal portfolio weights for each month.

#### Key Code Chunks:
1. **Black-Litterman Math Function (`evaluate_multiple.py:L8-L39`)**:
   - Extracts $Q = \text{mean}(\text{predictions})$ and $\Omega = \text{diag}(\text{var}(\text{predictions}))$.
   - Computes Moore-Penrose pseudo-inverses `np.linalg.pinv` to handle singular or ill-conditioned covariance matrices safely.
   - Computes posterior expected returns vector `posterior_returns`.
   - Passes `posterior_returns` into quadratic optimizer with constraints $\sum w_i = 1$ and $w_i \in [0, 1]$.
2. **Market Equilibrium Derivation (`evaluate_multiple.py:L60-L72`)**:
   Computes market cap weights from `market_caps.json`, derives the market portfolio return, computes each asset's beta $\beta_i$, and determines the CAPM equilibrium $\Pi$.
3. **Multi-Period Runner (`evaluate_multiple.py:L94-L151`)**:
   Iterates through each month from June to December 2024 with parameter $\tau = 0.025$, producing portfolio weights saved to `results/{model}_black_litterman_weights_tau_0.025.csv`.

---

### 4. `calculate_llm_returns.py` — Detailed Walkthrough & Performance Engine

* **File Location**: [`calculate_llm_returns.py`](file:///c:/Users/aryan/project/LLM-BLM/calculate_llm_returns.py)
* **High-Level Purpose**: Takes the Black-Litterman portfolio weights generated by `evaluate_multiple.py` for each LLM (`llama`, `qwen`, `gemma`) and simulates the real-world financial performance by multiplying those weights by the **actual out-of-sample forward daily stock returns**.

```
[results/*_black_litterman_weights_tau_0.025.csv]  (Month T Weights)
                                   │
                                   ├─── Matrix Multiplication ───> Daily Portfolio Returns (Month T+1)
                                   │
[yfinance/returns_T+1.csv]                         (Month T+1 Returns)
```

#### Detailed Code Walkthrough:

#### Step 1: Helper Functions & Loading Weights (`Lines 6–18`)
```python
def get_last_day_of_month(year, month):
    return calendar.monthrange(year, month)[1]

def calculate_llm_returns(tau = 0.025):
    llama_results = pd.read_csv(f'results/llama_black_litterman_weights_tau_{tau}.csv')
    qwen_results = pd.read_csv(f'results/qwen_black_litterman_weights_tau_{tau}.csv')
    gemma_results = pd.read_csv(f'results/gemma_black_litterman_weights_tau_{tau}.csv')
```
* **Explanation**: 
  - `get_last_day_of_month`: Handles variable days in each calendar month (e.g., 30 vs 31 days).
  - Loads the precomputed Black-Litterman weight CSVs from `results/` for each LLM model corresponding to parameter $\tau$.

#### Step 2: Forward Monthly Alignment Loop (`Lines 19–32`)
```python
for month in range(6, 12):  # June (6) to November (11)
    weight_date = f'2024-{month:02d}-01'
    
    returns_month = month + 1
    returns_year = 2024
    returns_start = f'{returns_year}-{returns_month:02d}-01'
    returns_end = f'{returns_year}-{returns_month:02d}-{get_last_day_of_month(returns_year, returns_month):02d}'
    
    future_data = pd.read_csv(f'yfinance/returns_{returns_start}_{returns_end}.csv')
```
* **Explanation**: 
  - Iterates month by month from June (`month = 6`) to November (`month = 11`).
  - `weight_date`: The timestamp of the decision period (Month $T$).
  - `returns_start` to `returns_end`: The forward test period (Month $T+1$).
  - Loads `future_data`, which contains the actual historical daily percentage changes for all S&P 500 stocks during Month $T+1$.

#### Step 3: Weight Extraction & Daily Portfolio Return Dot Product (`Lines 33–45`)
```python
asset_columns = [col for col in llama_results.columns if col != 'Date']

llama_month_weights = llama_results[llama_results['Date'] == weight_date][asset_columns].iloc[0].values
qwen_month_weights  = qwen_results[qwen_results['Date'] == weight_date][asset_columns].iloc[0].values
gemma_month_weights = gemma_results[gemma_results['Date'] == weight_date][asset_columns].iloc[0].values

# Dot product: R_p = sum(w_i * r_i) for each trading day
llama_returns = (future_data[asset_columns] * llama_month_weights).sum(axis=1)
qwen_returns  = (future_data[asset_columns] * qwen_month_weights).sum(axis=1)
gemma_returns = (future_data[asset_columns] * gemma_month_weights).sum(axis=1)
```
* **Explanation**:
  - Extracts the 1D numpy array of weights $\mathbf{w}$ for the given month.
  - Performs row-wise matrix multiplication: multiplies every stock's daily return column by its portfolio weight and sums across columns (`axis=1`).
  - Yields the exact portfolio daily return $R_{p,t}$ for every trading day in Month $T+1$.

#### Step 4: Structuring DataFrames & Appending Monthly Results (`Lines 46–65`)
```python
llama_portfolio = pd.DataFrame({
    'Date': future_data['Date'],
    'Portfolio_Return': llama_returns
})
llama_monthly_returns.append(llama_portfolio)
```
* **Explanation**: Pairs each computed daily return with its calendar trading date and appends the monthly slice to an accumulator list.

#### Step 5: Multi-Month Concatenation & Export (`Lines 66–85`)
```python
llama_all_returns = pd.concat(llama_monthly_returns, ignore_index=True)
# Save final daily time series to CSV
llama_returns.to_csv(f'results/llama_black_litterman_returns_tau_{tau}.csv', index=False)
```
* **Explanation**: Glues all consecutive months together into a continuous daily return time series spanning the entire backtest window (July 2024 to December 2024) and exports the results to CSV files in [`results/`](file:///c:/Users/aryan/project/LLM-BLM/results).

---

## 5. Directory & Data Structure

```
LLM-BLM/
├── PROJECT_EXPLANATION.md          # Comprehensive technical & mathematical documentation
├── run.py                          # Data collection & LLM view inference
├── baselines.py                    # 1/N and Markowitz Mean-Variance baselines
├── evaluate_multiple.py             # Black-Litterman Bayesian optimization engine
├── calculate_llm_returns.py        # Out-of-sample realized return calculator
├── market_caps.json                # Market capitalization dictionary for S&P 500
├── responses/                      # Raw JSON predictions (30 samples per stock)
│   ├── common_stocks.json          # Intersection of tickers valid across all models
│   └── llama_2024-06-01_*.json     # Monthly prediction files
├── responses_portfolios/           # Baseline portfolio weights and returns
├── results/                        # Final LLM-BLM weights & return CSVs
├── yfinance/                       # Downloaded daily price and return CSVs
└── figure/                         # Performance plots & return visualizations
```

---

## 6. Step-by-Step Execution Guide

### Step 1: Query the LLM for Views
```bash
python run.py --model_name llama
```
*(Note: Ensure your API endpoint and credentials in `run.py` lines 25–40 are configured).*

### Step 2: Build Control / Baseline Portfolios
```bash
python baselines.py
```
Outputs baseline performance to `responses_portfolios/`.

### Step 3: Run Black-Litterman Portfolio Optimization
```bash
python evaluate_multiple.py
```
Computes Bayesian posterior returns and saves optimal allocations to `results/`.

### Step 4: Calculate Out-of-Sample Performance
```bash
python calculate_llm_returns.py
```
Computes realized daily returns across all forward testing months and logs final portfolio metrics.
