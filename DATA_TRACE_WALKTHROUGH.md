# End-to-End Concrete Data Walkthrough: Tracing Stocks Through LLM-BLM

> **Case Study**: Tracing **Adobe Inc. (`ADBE`)** and **Apple Inc. (`AAPL`)**  
> **Timeframe**: **June 2024** (In-Sample Decision Month) $\rightarrow$ **July 2024** (Out-of-Sample Forward Testing Month)  
> **Model**: `Gemma-7B-IT` ($\tau = 0.025$)

---

## Table of Contents
1. [Overview & Experiment Setup](#1-overview--experiment-setup)
2. [Step 1: Raw Data Ingestion & Return Calculation (`yfinance/`)](#step-1-raw-data-ingestion--return-calculation-yfinance)
3. [Step 2: Prompt Packaging & 30x LLM Querying (`run.py` $\rightarrow$ `responses/`)](#step-2-prompt-packaging--30x-llm-querying-runpy--responses)
4. [Step 3: Calculating View Statistics ($Q$, $\Omega$) & Market Prior ($\Pi$)](#step-3-calculating-view-statistics-q-omega--market-prior-pi)
5. [Step 4: Black-Litterman Bayesian Updating & Weight Optimization (`evaluate_multiple.py`)](#step-4-black-litterman-bayesian-updating--weight-optimization-evaluate_multiplepy)
6. [Step 5: Out-of-Sample Daily Realized Return Evaluation (`calculate_llm_returns.py`)](#step-5-out-of-sample-daily-realized-return-evaluation-calculate_llm_returnspy)
7. [End-to-End Summary & Value Comparison Table](#7-end-to-end-summary--value-comparison-table)

---

## 1. Overview & Experiment Setup

This walkthrough tracks the exact lifecycle of numerical data for two representative assets:
* **Adobe Inc. (`ADBE`)**: A stock that received a strong, confident prediction from the model.
* **Apple Inc. (`AAPL`)**: A stock that received a weaker prediction and zero active portfolio weight.

```
+---------------------------------------------------------------------------------------------------+
| JUNE 2024: TRAINING & DECISION WINDOW                                                             |
|                                                                                                   |
| [Raw June Prices] ──> [June Returns] ──> [LLM Prompt (30x)] ──> [Q & Ω]                           |
|                                                                    │                              |
| [Market Caps]     ──> [Beta & Market Equilibrium Prior Π] ─────────┤                              |
|                                                                    ▼                              |
|                                                     [Black-Litterman Math]                        |
|                                                                    │                              |
|                                                                    ▼                              |
|                                                   [Optimal Weights w for July]                    |
+---------------------------------------------------------------------------------------------------+
                                                                     │
                                                                     ▼
+---------------------------------------------------------------------------------------------------+
| JULY 2024: OUT-OF-SAMPLE TEST WINDOW                                                              |
|                                                                                                   |
| [Actual July Returns]  x  [Optimal Weights w]  ──>  [Daily Realized Portfolio Returns R_p]        |
+---------------------------------------------------------------------------------------------------+
```

---

## Step 1: Raw Data Ingestion & Return Calculation (`yfinance/`)

### Code Responsible
[`run.py` lines 53–70](file:///c:/Users/aryan/project/LLM-BLM/run.py#L53-L70)

### What Happens
1. `run.py` reads Wikipedia to find S&P 500 company classifications:
   * **ADBE**: *Information Technology* sector, *Application Software* sub-industry.
   * **AAPL**: *Information Technology* sector, *Technology Hardware, Storage & Peripherals* sub-industry.
2. Calls `yfinance` to download daily closing prices for all trading days in June 2024.
3. Computes daily percentage returns:
   $$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### File Locations & Sample Data
* **Raw Prices**: [`yfinance/data_2024-06-01_2024-06-30.csv`](file:///c:/Users/aryan/project/LLM-BLM/yfinance/data_2024-06-01_2024-06-30.csv)
* **Percentage Returns**: [`yfinance/returns_2024-06-01_2024-06-30.csv`](file:///c:/Users/aryan/project/LLM-BLM/yfinance/returns_2024-06-01_2024-06-30.csv)

```csv
# Real June return sequence for ADBE:
[0.021297, 0.016571, 0.005112, 0.015934, -0.011796, 0.005979, -0.006095, -0.002457, 0.145115, -0.012507, 0.006766, 0.001340, 0.020059, -0.017378, 0.005170, 0.003378, 0.034238, 0.016058]

# Real June return sequence for AAPL:
[0.001649, 0.007821, -0.007096, 0.012392, -0.019148, 0.072649, 0.028578, 0.005491, -0.008168, 0.019672, -0.010985, -0.021513, -0.010444, 0.003133, 0.004468, 0.019993, 0.003986, -0.016254]
```

---

## Step 2: Prompt Packaging & 30x LLM Querying (`run.py` $\rightarrow$ `responses/`)

### Code Responsible
[`run.py` lines 83–156](file:///c:/Users/aryan/project/LLM-BLM/run.py#L83-L156)

### What Happens
1. `run.py` constructs a structured prompt for each stock:
   ```text
   Ticker: ADBE
   Security: Adobe Inc.
   Sector: Information Technology
   Sub-Industry: Application Software
   pct_change: [0.021297, 0.016571, 0.005112, ..., 0.016058]
   ```
2. Sends the prompt to `google/gemma-7b-it` **30 times** with structured JSON output enforced via Pydantic (`expected_return: float`).

### File Location
[`responses/gemma_2024-06-01_2024-06-30.json`](file:///c:/Users/aryan/project/LLM-BLM/responses/gemma_2024-06-01_2024-06-30.json)

### Actual Model Output Stored in JSON
```json
{
  "ADBE": {
    "ticker": "ADBE",
    "Security": "Adobe Inc.",
    "GICS Sector": "Information Technology",
    "GICS Sub-Industry": "Application Software",
    "expected_return": [
      -0.006112, -0.011207, 0.020548, -0.006981, -0.006322, -0.006312, 
      -0.005623, -0.006121, 0.020059, -0.004422, -0.002127, -0.002281, 
      -0.006618, -0.002421, -0.001113, -0.003322, -0.004261, 0.020059, 
      -0.003812, -0.002861, 0.020211, -0.004832, 0.020059, -0.004366, 
      -0.001222, -0.001222, -0.003834, -0.004112, -0.001211, -0.003422
    ]
  },
  "AAPL": {
    "ticker": "AAPL",
    "Security": "Apple Inc.",
    "GICS Sector": "Information Technology",
    "GICS Sub-Industry": "Technology Hardware, Storage & Peripherals",
    "expected_return": [
      -0.008512, -0.001083, -0.001892, -0.004211, -0.001429, -0.001583,
      -0.004947, -0.004632, -0.008868, -0.008283, -0.001123, -0.002662,
      -0.005221, -0.001822, -0.002336, -0.005516, -0.002362, -0.001694,
      -0.001462, -0.002242, -0.001997, -0.001262, -0.002831, -0.001322,
      -0.002382, -0.005484, -0.005821, -0.002465, -0.002191, -0.001952
    ]
  }
}
```

---

## Step 3: Calculating View Statistics ($Q$, $\Omega$) & Market Prior ($\Pi$)

### Code Responsible
[`evaluate_multiple.py` lines 8–20, 60–72](file:///c:/Users/aryan/project/LLM-BLM/evaluate_multiple.py#L8-L20)

### 1. View Vector ($Q$) and Uncertainty Matrix ($\Omega$)
* **Adobe (`ADBE`)**:
  * Mean View ($Q_{\text{ADBE}}$): $+0.00095$
  * Uncertainty Variance ($\Omega_{\text{ADBE}}$): $0.000078$
* **Apple (`AAPL`)**:
  * Mean View ($Q_{\text{AAPL}}$): $-0.00346$
  * Uncertainty Variance ($\Omega_{\text{AAPL}}$): $0.000006$

### 2. Market Equilibrium Prior ($\Pi$)
* Reads market capitalizations from [`market_caps.json`](file:///c:/Users/aryan/project/LLM-BLM/market_caps.json):
  * $\text{Market Cap}_{\text{ADBE}} = \$188,515,368,960$
  * $\text{Market Cap}_{\text{AAPL}} = \$3,419,480,784,896$
* Computes market weights:
  $$w_{\text{mkt}, i} = \frac{\text{Market Cap}_i}{\sum \text{Market Cap}}$$
* Computes asset beta $\beta_i = \frac{\text{cov}(r_i, R_m)}{\text{var}(R_m)}$ and CAPM equilibrium expected return:
  $$\Pi_i = \beta_i \cdot (R_m - R_f)$$

---

## Step 4: Black-Litterman Bayesian Updating & Weight Optimization (`evaluate_multiple.py`)

### Code Responsible
[`evaluate_multiple.py` lines 8–39, 94–146](file:///c:/Users/aryan/project/LLM-BLM/evaluate_multiple.py#L8-L39)

### 1. Master Formula Computation
With hyperparameter $\tau = 0.025$:
$$M = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1}$$
$$\hat{\mu} = M \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$

### 2. Quadratic Portfolio Optimization
$$\min_w \left[ w^T \Sigma w - 0.1 \cdot (w^T \hat{\mu}) \right] \quad \text{subject to } \sum_{i=1}^N w_i = 1, \quad w_i \ge 0$$

### 3. Resulting Portfolio Weights (Row 2, Date: `2024-06-01`)
* **Adobe (`ADBE`)**: Receives an active positive allocation:
  $$w_{\text{ADBE}} = 0.006045 \quad (0.6045\%)$$
* **Apple (`AAPL`)**: Pessimistic view results in no allocation:
  $$w_{\text{AAPL}} \approx 0.0$$

### File Location
[`results/gemma_black_litterman_weights_tau_0.025.csv`](file:///c:/Users/aryan/project/LLM-BLM/results/gemma_black_litterman_weights_tau_0.025.csv)

---

## Step 5: Out-of-Sample Daily Realized Return Evaluation (`calculate_llm_returns.py`)

### Code Responsible
[`calculate_llm_returns.py` lines 30–65](file:///c:/Users/aryan/project/LLM-BLM/calculate_llm_returns.py#L30-L65)

### What Happens
1. Loads the actual daily stock returns for **July 2024** from [`yfinance/returns_2024-07-01_2024-07-31.csv`](file:///c:/Users/aryan/project/LLM-BLM/yfinance).
2. Performs daily dot-product multiplication:
   $$R_{p, t} = \sum_{i=1}^N w_i \cdot r_{i, t}$$
3. **Example Trading Day: July 2, 2024**:
   * Actual return of Adobe: $r_{\text{ADBE, 07-02}} = +0.0142$
   * Adobe's contribution to portfolio: $0.006045 \times 0.0142 = +0.0000858$
   * Portfolio Total Daily Return across all stocks: $R_{p, \text{2024-07-02}} = +0.004299 \quad (+0.43\%)$

### File Location
[`results/gemma_black_litterman_returns_tau_0.025.csv`](file:///c:/Users/aryan/project/LLM-BLM/results/gemma_black_litterman_returns_tau_0.025.csv)

---

## 7. End-to-End Summary & Value Comparison Table

| Pipeline Stage | Adobe Inc. (`ADBE`) | Apple Inc. (`AAPL`) | File / Code Location |
| :--- | :--- | :--- | :--- |
| **1. GICS Sector** | Information Technology | Information Technology | Wikipedia / `run.py` |
| **2. June 2024 Raw Returns** | Strong gains (e.g., $+14.5\%$ spike on earnings) | Mixed / Slight pullback | `yfinance/returns_2024-06-01_2024-06-30.csv` |
| **3. LLM View ($Q$)** | $+0.00095$ (Positive daily mean) | $-0.00346$ (Negative daily mean) | `responses/gemma_2024-06-01_2024-06-30.json` |
| **4. LLM Uncertainty ($\Omega$)** | $0.000078$ (Moderate variance) | $0.000006$ (High consensus) | Computed in `evaluate_multiple.py` |
| **5. Market Cap** | $\$188.52\text{ Billion}$ | $\$3,419.48\text{ Billion}$ | `market_caps.json` |
| **6. Optimal BLM Weight ($w$)** | **$0.006045$ ($0.6045\%$)** | **$\approx 0.0$ ($0\%$)** | `results/gemma_black_litterman_weights_tau_0.025.csv` |
| **7. Realized July Return** | Contributed $+0.00858\%$ on July 2 | Zero direct contribution | `results/gemma_black_litterman_returns_tau_0.025.csv` |
