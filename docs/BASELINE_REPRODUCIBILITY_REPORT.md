# Baseline Reproducibility and Performance Report

**Role Owner:** Person 3 (Baselines and Evaluation Owner)  
**Package:** Semester V Deliverables  
**Status:** Validated and Fully Reproducible  

---

## 1. Overview and Verification Statement

All baseline strategies described in [`SEMESTER_V_WORK_PLAN.md`](../SEMESTER_V_WORK_PLAN.md) and [`PROJECT_EXPLANATION.md`](../PROJECT_EXPLANATION.md) have been successfully audited, reproduced, and evaluated over the out-of-sample period (**July 2, 2024 to December 31, 2024**, 127 trading days).

The repository outputs have been verified against both the legacy scripts and the refactored, robust evaluation module [`src/evaluation/evaluate_baselines.py`](../src/evaluation/evaluate_baselines.py).

---

## 2. Comprehensive Performance Summary

### Table 1: Out-of-Sample Performance Comparison (Gross vs Net of Costs)

| Strategy | Cost Structure | Total Return (%) | Annualized Return (%) | Annualized Volatility (%) | Sharpe Ratio | Max Drawdown (%) | Mean Monthly Turnover |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Equal Weighted (1/N)** | **Gross (0 bps)** | **12.44%** | **28.45%** | **12.61%** | **2.10** | **-6.97%** | **0.00** |
| Equal Weighted (1/N) | Net (5 bps) | 12.44% | 28.45% | 12.61% | 2.10 | -6.97% | 0.00 |
| Equal Weighted (1/N) | Net (10 bps) | 12.44% | 28.45% | 12.61% | 2.10 | -6.97% | 0.00 |
| **Mean-Variance (MVO)** | **Gross (0 bps)** | **35.38%** | **90.96%** | **26.19%** | **3.40** | **-14.32%** | **1.65** |
| Mean-Variance (MVO) | Net (5 bps) | 34.85% | 89.38% | 26.18% | 3.34 | -14.32% | 1.65 |
| Mean-Variance (MVO) | Net (10 bps) | 34.33% | 87.80% | 26.18% | 3.28 | -14.32% | 1.65 |
| **Gemma-7B + BLM** | **Gross (0 bps)** | **13.75%** | **31.66%** | **22.47%** | **1.32** | **-9.05%** | **1.64** |
| Gemma-7B + BLM | Net (5 bps) | 13.32% | 30.61% | 22.48% | 1.27 | -9.22% | 1.64 |
| Gemma-7B + BLM | Net (10 bps) | 12.90% | 29.57% | 22.50% | 1.23 | -9.38% | 1.64 |
| **LLaMA-3.1 + BLM** | **Gross (0 bps)** | **-0.24%** | **-0.51%** | **21.58%** | **-0.12** | **-14.11%** | **1.71** |
| LLaMA-3.1 + BLM | Net (5 bps) | -0.64% | -1.37% | 21.65% | -0.16 | -14.28% | 1.71 |
| LLaMA-3.1 + BLM | Net (10 bps) | -1.05% | -2.22% | 21.73% | -0.19 | -14.46% | 1.71 |
| **Qwen2 + BLM** | **Gross (0 bps)** | **38.65%** | **100.95%** | **23.88%** | **4.14** | **-12.53%** | **1.65** |
| Qwen2 + BLM | Net (5 bps) | 38.11% | 99.28% | 23.88% | 4.07 | -12.53% | 1.65 |
| Qwen2 + BLM | Net (10 bps) | 37.57% | 97.62% | 23.88% | 4.00 | -12.53% | 1.65 |

---

## 3. In-Depth Strategy Analysis

### 3.1. Equal Weighted Benchmark ($1/N$)
- **Performance:** Achieved $+12.44\%$ total return with the lowest volatility ($12.61\%$) and lowest maximum drawdown ($-6.97\%$).
- **Characteristics:** By distributing capital equally across all 440+ assets, this strategy eliminates idiosyncratic stock risk through broad diversification. It incurs **zero rebalancing turnover**, making it completely immune to transaction costs.
- **Role in Project:** Serves as the primary benchmark to determine whether algorithmic or AI active management justifies its added complexity and trading friction.

### 3.2. Classical Markowitz Mean-Variance Optimization (MVO)
- **Performance:** Gross return $+35.38\%$ ($90.96\%$ annualized) with Sharpe ratio $3.40$.
- **Concentration Pathology:** Because sample mean returns are notorious for estimation error ("error maximizer"), MVO concentrated over $85\%$ of the portfolio weight into fewer than 5 assets in certain months.
- **Turnover Drag:** High monthly turnover ($1.65$ per rebalance) reduced annualized returns from $90.96\%$ down to $87.80\%$ under $10$ bps transaction costs.

### 3.3. Qwen2 + Black-Litterman Model
- **Performance:** The top-performing active strategy, delivering $+38.65\%$ gross ($+38.11\%$ net of $5$ bps costs) and a Sharpe ratio of $4.07$ net of costs.
- **Mechanism:** Qwen2's predictions effectively balanced market beta with high-conviction momentum names without driving weights into a single illiquid stock, showing high synergy with the Black-Litterman Bayesian blending process.

### 3.4. Gemma-7B + Black-Litterman Model
- **Performance:** Delivered $+13.75\%$ gross ($+13.32\%$ net of $5$ bps costs), with Sharpe ratio $1.32$.
- **Characteristics:** Gemma generated more conservative return forecasts, resulting in posterior returns that stayed closer to the market equilibrium prior. Consequently, it tracked closer to the market index with moderate risk.

### 3.5. LLaMA-3.1 + Black-Litterman Model
- **Performance:** Underperformed significantly, generating $-0.24\%$ gross and $-0.64\%$ net of $5$ bps costs.
- **Root Cause Analysis:** Weight inspection revealed severe corner solutions:
  - In October 2024, LLaMA-3.1 assigned $74.2\%$ weight to Air Products & Chemicals (`APD`).
  - In November 2024, LLaMA-3.1 assigned $37.5\%$ to Constellation Energy (`CEG`) and $38.6\%$ to Philip Morris (`PM`).
  - In December 2024, LLaMA-3.1 allocated $99.99\%$ to Coca-Cola (`KO`).
- **Conclusion:** Extreme over-concentration due to low forecast uncertainty ($\Omega$) caused LLaMA to lose the benefit of diversification, resulting in negative risk-adjusted performance.

---

## 4. Generated Artifacts and Visualizations

The reproduction generated the following publication-quality figures:
- [`figure/baseline_cumulative_returns_gross_net.png`](../figure/baseline_cumulative_returns_gross_net.png): Side-by-side comparison of out-of-sample cumulative equity curves under gross vs 5 bps friction.
- [`figure/baseline_risk_return_comparison.png`](../figure/baseline_risk_return_comparison.png): Bar comparison of annualized return vs annualized volatility.
- [`results/baseline_comparison_gross_and_net.csv`](../results/baseline_comparison_gross_and_net.csv): Machine-readable full metrics matrix.

---

## 5. How to Reproduce

Execute the following commands from the repository root:

```powershell
# 1. Run the unified baseline evaluation engine (Person 3 module):
python -m src.evaluation.evaluate_baselines

# 2. (Optional) Run individual legacy reproduction scripts:
python baselines.py
python calculate_llm_returns.py
python visualize_results.py
```
