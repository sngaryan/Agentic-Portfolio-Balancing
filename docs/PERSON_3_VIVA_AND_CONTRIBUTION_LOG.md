# Person 3: Contribution Log and Viva Defense Guide

**Role:** Baselines and Evaluation Owner  
**Owner:** Person 3  
**Project:** Agentic AI for Financial Portfolio Balancing  
**Semester:** Semester V (FR-09 Backtesting, FR-10 Baseline Evaluation)  

---

## Part 1: Individual Contribution Log

### 1.1 Summary of Deliverables Owned
- **Core Code Modules:**
  - [`src/evaluation/__init__.py`](../src/evaluation/__init__.py): Evaluation package initialization.
  - [`src/evaluation/metrics.py`](../src/evaluation/metrics.py): Financial mathematics engine (Cumulative return, Annualized return/volatility, Sharpe, Sortino, Calmar, Max Drawdown, Turnover, Transaction cost deduction).
  - [`src/evaluation/evaluate_baselines.py`](../src/evaluation/evaluate_baselines.py): Automated baseline evaluation, robust universe intersection, risk-free rate fix, and Gross vs Net comparison engine.
- **Documentation & Audit Reports:**
  - [`docs/BASELINE_AUDIT_AND_PROTOCOL.md`](BASELINE_AUDIT_AND_PROTOCOL.md): End-to-end workflow map, mathematical specifications, code discrepancy audit, and evaluation protocol.
  - [`docs/BASELINE_REPRODUCIBILITY_REPORT.md`](BASELINE_REPRODUCIBILITY_REPORT.md): Audit and verification report of all baseline results, turnover analysis, and strategy post-mortems.
  - [`docs/PERSON_3_VIVA_AND_CONTRIBUTION_LOG.md`](PERSON_3_VIVA_AND_CONTRIBUTION_LOG.md): Personal contribution log and comprehensive viva defense guide.
- **Generated Results & Visualizations:**
  - [`results/baseline_comparison_gross_and_net.csv`](../results/baseline_comparison_gross_and_net.csv)
  - [`figure/baseline_cumulative_returns_gross_net.png`](../figure/baseline_cumulative_returns_gross_net.png)
  - [`figure/baseline_risk_return_comparison.png`](../figure/baseline_risk_return_comparison.png)

### 1.2 Chronological Contribution History

| Date / Milestone | Action / Deliverable | Key Decisions Made |
|---|---|---|
| **Kickoff** | Checked out role branch `feature/baselines-evaluation` | Merged Person 1's data pipeline commit to align with cleaned data contracts. |
| **Code Audit** | Audited `baselines.py`, `evaluate_multiple.py`, and `calculate_llm_returns.py` | Discovered critical bug in `evaluate_multiple.py:67` where annual 2% $R_f$ was subtracted directly from daily returns. Discovered ticker drop misalignment in `baselines.py`. |
| **Architecture Design** | Created modular `src/evaluation/` package | Separated pure metric computation (`metrics.py`) from evaluation runner (`evaluate_baselines.py`) to ensure reuse in Semester VI. |
| **Reproducibility** | Executed and validated all 5 baseline strategies | Verified that Equal-Weight, MVO, Gemma, LLaMA, and Qwen reproduce the saved legacy performance over July–Dec 2024. |
| **Feature Enhancement** | Built Turnover & Transaction Cost engine | Added configurable basis points ($0, 5, 10$ bps) to evaluate net returns after portfolio rebalancing friction. |
| **Documentation** | Authored audit report, protocol, and viva guide | Established the official baseline benchmark against which future DRL agents will be evaluated. |

---

## Part 2: Viva Defense Preparation Guide

### Section A: Mean-Variance Optimization (Markowitz MVO)

#### Q1: What is Markowitz Mean-Variance Optimization and what are its key assumptions?
**Answer:**  
Proposed by Harry Markowitz in 1952, MVO formalizes the trade-off between expected return and variance (risk). It assumes:
1. Investors are rational, risk-averse, and utility-maximizing.
2. Asset returns are jointly normally distributed (or investors care only about the first two moments: mean and variance).
3. Markets are frictionless (no taxes or transaction costs in the basic model).
4. All information is reflected in expected returns ($\mu$) and the covariance matrix ($\Sigma$).

#### Q2: What is the objective function solved in our codebase for MVO?
**Answer:**  
We solve a quadratic programming problem using SciPy's SLSQP algorithm:
$$\min_w \quad w^T \Sigma w - \lambda w^T \mu$$
$$\text{subject to} \quad \sum_{i=1}^N w_i = 1, \quad w_i \ge 0 \; \forall i$$
where $\Sigma \in \mathbb{R}^{N \times N}$ is the sample covariance matrix, $\mu \in \mathbb{R}^N$ is the sample mean return vector, and $\lambda = 0.1$ is the investor's risk-tolerance parameter. Long-only constraints ($w_i \ge 0$) prevent short-selling.

#### Q3: Why is classical MVO often referred to as an "estimation error maximizer"?
**Answer:**  
Sample means ($\mu$) are notoriously noisy and difficult to estimate. MVO assigns large positive weights to assets that had high historical returns—frequently because of estimation error or transient anomalies—and large negative/zero weights to underperforming assets. As Michaud (1989) proved, MVO overweights assets with positive estimation error and underweights assets with negative estimation error, leading to extreme, unstable, and non-diversified portfolios.

---

### Section B: The Black-Litterman Model (BLM)

#### Q4: Why was the Black-Litterman model introduced in 1992?
**Answer:**  
Fischer Black and Robert Litterman at Goldman Sachs created the model to solve the instability and concentration pathologies of MVO. Instead of starting with noisy historical mean returns, Black-Litterman starts with the **market equilibrium prior** (derived via reverse optimization from the Capital Asset Pricing Model) and allows the investor to blend in subjective views with explicit uncertainty parameters.

#### Q5: What are the primary mathematical components of the Black-Litterman model?
**Answer:**  
1. **Market Equilibrium Return Vector ($\Pi$):**
   $$\Pi = \delta \Sigma w_{mkt}$$
   where $\delta$ is the market risk-aversion coefficient and $w_{mkt}$ is market-cap weighting. In our project, $\Pi = \beta \cdot \text{MRP}_{daily}$.
2. **View Vector ($Q \in \mathbb{R}^K$):** The expected returns of the investor's $K$ views.
3. **Pick Matrix ($P \in \mathbb{R}^{K \times N}$):** Maps which assets are involved in each view (in our case, $P = I_N$ because we have absolute views on each asset).
4. **View Uncertainty Covariance ($\Omega \in \mathbb{R}^{K \times K}$):** Diagonal matrix where $\Omega_{ii}$ represents the variance/uncertainty of view $i$.
5. **Scalar $\tau$:** Scales the uncertainty of the market prior (typically set to $0.025$ or $0.05$).
6. **Posterior Combined Return Vector ($E[R]$):**
   $$E[R] = \left[ (\tau \Sigma)^{-1} + P^T \Omega^{-1} P \right]^{-1} \left[ (\tau \Sigma)^{-1} \Pi + P^T \Omega^{-1} Q \right]$$

#### Q6: How does the model extract $Q$ and $\Omega$ from Large Language Models?
**Answer:**  
The LLMs (Gemma, LLaMA, Qwen) were queried with recent price trends, news, and financial indicators to generate repeated predictions of future expected returns for each stock.
- The **mean** of the LLM's predictions for stock $i$ forms the view $Q_i = \text{mean}(r_i)$.
- The **variance** of the LLM's predictions reflects model disagreement or dispersion, forming the diagonal element $\Omega_{ii} = \text{var}(r_i)$. If the model is confident and consistent, $\Omega_{ii}$ is small, giving the view higher weight in the posterior.

---

### Section C: Audit Findings and Bug Fixes

#### Q7: What was the risk-free rate scaling discrepancy you uncovered in `evaluate_multiple.py`?
**Answer:**  
In `evaluate_multiple.py` line 67, the script defined `risk_free_rate = 0.02` (annual 2%), but subtracted it directly from daily returns:
```python
market_risk_premium = (market_return_weighted - risk_free_rate).mean()
```
Since daily market returns are approximately $+0.05\%$, subtracting $2\%$ ($0.02$) yielded a daily market risk premium of $-1.95\%$. Annualized, this implied the market was expected to lose nearly $500\%$ per year.  
**Fix:** We converted the annual rate to its exact daily compound equivalent:
$$r_{f, daily} = (1 + 0.02)^{1/252} - 1 \approx 0.00007907$$
ensuring a realistic, positive daily equity risk premium.

#### Q8: What was the universe alignment bug in `baselines.py`?
**Answer:**  
In `baselines.py`, `data` (training) and `future_data` (testing) called `.dropna(axis=1)` independently. If an asset had valid data in the training month but experienced a trading halt or missing row in the forward test month, it was dropped from `future_data` but retained in `data`. When calculating portfolio returns `(future_data[asset_columns] * weights).sum()`, the script failed with KeyError or misaligned matrix multiplications.  
**Fix:** We created `get_aligned_universe()` in [`src/evaluation/evaluate_baselines.py`](../src/evaluation/evaluate_baselines.py), which computes a strict mutual intersection across all training, testing, market-cap, and model-response keys before any matrix operations.

---

### Section D: Portfolio Evaluation and Transaction Costs

#### Q9: What is portfolio turnover, and why must it be accounted for?
**Answer:**  
Turnover measures the fraction of portfolio value reallocated between periods:
$$\text{Turnover}_t = \frac{1}{2} \sum_{i=1}^N |w_{i, t} - w_{i, t-1}|$$
Active strategies that trade frequently incur brokerage commissions, bid-ask spreads, and market impact costs. Without transaction costs, an active strategy might appear profitable on paper while being completely unviable in practice.

#### Q10: How do transaction costs impact the baselines in our empirical results?
**Answer:**  
- **Equal-Weighted (1/N):** Rebalanced only as a buy-and-hold benchmark, incurring $0.00$ turnover and retaining $12.44\%$ net return.
- **MVO:** Incurred a high turnover of $1.65$ per rebalance. Under a 10 bps transaction cost, annualized return declined from $90.96\%$ (gross) to $87.80\%$ (net).
- **Qwen2 + BLM:** Despite $1.65$ turnover, its strong gross returns ($38.65\%$) comfortably absorbed 5 bps and 10 bps friction, finishing at $38.11\%$ and $37.57\%$ net respectively.
- **LLaMA-3.1 + BLM:** Extreme turnover ($1.71$) exacerbated its negative gross return ($-0.24\%$), pushing net return down to $-1.05\%$ under 10 bps costs.

---

### Section E: Connection to Future Work (Semester VI DRL)

#### Q11: Why are these baselines essential before developing Deep Reinforcement Learning (DRL) agents in Semester VI?
**Answer:**  
In financial machine learning, complex models like PPO (Proximal Policy Optimization) or SAC (Soft Actor-Critic) often suffer from overfitting, high variance, and sample inefficiency. Without rigorously audited classical and Bayesian benchmarks:
1. One cannot prove whether the DRL agent has learned genuine alpha or is merely capturing broad market beta.
2. The agent's reward function must be tuned against transaction costs; knowing baseline turnover establishes realistic friction penalties.
3. Baselines provide the lower bound (Equal-Weight $1/N$) and upper bound (Black-Litterman) for performance validation.
