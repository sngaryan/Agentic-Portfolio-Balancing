# Future Scope: Agentic AI for Financial Portfolio Balancing & Jev-BLM Integration

> **Project Reference**: *AI24300/CS24300 PROJECT-I — Agentic AI for Financial Portfolio Balancing*  
> **Authors**: Piyush Chaudhary, Aryan Singh, Manthan Khosla  
> **Supervisor**: Dr. Vishwambhar Pathak, Department of Computer Science & Engineering, Birla Institute of Technology, Mesra (Jaipur Campus)

---

## Executive Summary

This document outlines the future roadmap and architectural enhancements for the **LLM-BLM** framework. It bridges the current static Black-Litterman implementation with the comprehensive **Software Requirements Specification (SRS) & System Design Specification (SDS)** for the full **Agentic AI Portfolio Balancing System**, incorporating cutting-edge advancements such as **TypeSafe AI's Jev (System One Discriminative AI)**, **Continuous-Control Deep Reinforcement Learning (PPO/SAC)**, **Post-Hoc Explainability (SHAP/LIME)**, and **Production Microservices**.

---

## Table of Contents
1. [Integration of TypeSafe AI's Jev (System One Discriminative AI)](#1-integration-of-typesafe-ais-jev-system-one-discriminative-ai)
2. [Continuous-Control DRL Architecture (PPO & SAC)](#2-continuous-control-drl-architecture-ppo--sac)
3. [Friction-Aware Reward & Transaction Cost Modeling](#3-friction-aware-reward--transaction-cost-modeling)
4. [Advanced Feature Engineering & Compact Risk States](#4-advanced-feature-engineering--compact-risk-states)
5. [On-Demand Post-Hoc Explainability (SHAP & LIME)](#5-on-demand-post-hoc-explainability-shap--lime)
6. [Multi-Asset & Multi-Exchange Ingestion Pipelines](#6-multi-asset--multi-exchange-ingestion-pipelines)
7. [Enterprise Production Stack (FastAPI, PostgreSQL, Docker)](#7-enterprise-production-stack-fastapi-postgresql-docker)
8. [Roadmap & Milestone Timeline](#8-roadmap--milestone-timeline)

---

## 1. Integration of TypeSafe AI's Jev (System One Discriminative AI)

### The Current Limitation: Generative 30x Sampling
Currently, [`run.py`](file:///c:/Users/aryan/project/LLM-BLM/run.py) queries generative text models (LLaMA-3.1, Gemma-7B, Qwen2) 30 times per stock to fabricate an artificial uncertainty variance for $\Omega$. This results in:
* High computational overhead ($15,000$ LLM API requests per monthly rebalance).
* Token-generation hallucinations and arithmetic formatting fragility.
* Sampling temperature noise rather than true epistemic Bayesian confidence.

```
[Current Generative Approach]
Stock Data ──> 30x Token-by-Token Text Prompts ──> Regex Parsing ──> Sample Variance Ω (High Noise)

[Future Jev System One Approach]
Stock Data ──> Single Zero-Shot Forward Pass ──> Calibrated Probability & Score ──> True Bayesian Ω
```

### Proposed Jev-BLM Architecture
**Jev** (developed by TypeSafe AI) is a non-generative, discriminative **System One AI model** that returns typed continuous scores and calibrated probabilities directly:
1. **Calibrated View Uncertainty ($\Omega$)**: Jev directly outputs mathematical confidence distributions in a single forward pass ($O(1)$ instead of 30 generative loops), providing a rigorous Bayesian uncertainty matrix for Black-Litterman.
2. **Sub-Second Rebalancing Latency ($70\text{--}200\text{ ms}$)**: Enables transitioning from rigid monthly rebalancing to **daily or event-driven real-time portfolio rebalancing**.
3. **Zero Token Hallucinations**: Eliminates arithmetic parsing failures by providing strictly typed schema outputs.

---

## 2. Continuous-Control DRL Architecture (PPO & SAC)

As defined in the project SRS/SDS (FR-05, FR-06, FR-07), portfolio allocation is a **sequential decision-making problem** over continuous action spaces $\mathbf{w}_t = [w_1, w_2, \dots, w_n]^T$ where $\sum_{i=1}^n w_{i,t} = 1, w_{i,t} \ge 0$.

```
                            +-----------------------------------------+
                            |           Gymnasium Environment         |
                            |  - Asset Returns: r_t                   |
                            |  - Portfolio Value: V_t                 |
                            |  - Turnover: ||w_t - w_{t-1}||_1        |
                            +-----------------------------------------+
                                         ▲               │
                         Action: w_t     │               │  State: s_t
                   (Portfolio Weights)   │               │  Reward: R_t
                                         │               ▼
                            +-----------------------------------------+
                            |      Continuous-Control DRL Agent       |
                            |  Primary: PPO (Clipped Policy)          |
                            |  Secondary Benchmark: SAC (Entropy Reg) |
                            +-----------------------------------------+
```

### 1. Primary Agent: Proximal Policy Optimization (PPO)
* Utilizes a clipped surrogate policy objective:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t \right) \right]$$
* Ensures stable, monotonic policy improvements without disruptive weight oscillations.

### 2. Secondary Benchmark: Soft Actor-Critic (SAC)
* Implements an off-policy actor-critic framework with maximum-entropy regularization:
  $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t) \sim \rho_\pi} \left[ R(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$
* Encourages thorough exploration across diverse asset combinations in complex market regimes.

---

## 3. Friction-Aware Reward & Transaction Cost Modeling

Traditional allocation models ignore rebalancing frictions, leading to excessive portfolio turnover that erodes real-world alpha. As specified in SRS FR-08, the agent is trained on a **Friction-Aware Net Rolling Sharpe Reward**:

$$\text{Reward}_t = \text{Sharpe}_t - \text{TransactionCost}_t$$

$$\text{where } \text{TransactionCost}_t = c_{\text{fee}} \cdot \| \mathbf{w}_t - \mathbf{w}_{t-1} \|_1$$
* $\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$: $L_1$-norm measuring the total percentage of portfolio capital turned over.
* $c_{\text{fee}}$: Brokerage fee, bid-ask spread, and market impact cost (e.g. $0.10\%$).
* Directly incentivizes the agent to trade only when expected forward alpha exceeds execution costs.

---

## 4. Advanced Feature Engineering & Compact Risk States

Rather than feeding raw prices or unstable empirical covariance matrices $\Sigma$ into the state representation, the system implements a compact multi-factor market state $s_t$ (SRS FR-03 & FR-04):

```
State Vector s_t = [ Momentum Features | Trend & Regime Features | Compact Risk Features | Current Portfolio w_{t-1} ]
```

### Component Details:
1. **Momentum Indicators**:
   * **MACD (Moving Average Convergence Divergence)**: Captures short-term vs long-term price velocity.
   * **RSI (Relative Strength Index)**: Identifies overbought/oversold momentum extremes.
2. **Trend & Regime Detection**:
   * **ADX (Average Directional Index)**: Quantifies trend strength and non-trending sideways regimes.
   * **CCI (Commodity Channel Index)**: Detects cyclical turns and statistical divergence.
3. **Compact Risk & Ledoit-Wolf Shrinkage**:
   * Uses 30-day rolling volatility vectors.
   * **Ledoit-Wolf Shrinkage Covariance**: Eliminates ill-conditioned inversion singularities by shrinking the sample covariance matrix $\Sigma_{\text{sample}}$ toward a well-conditioned structured target $F$:
     $$\Sigma_{\text{LW}} = \delta F + (1 - \delta)\Sigma_{\text{sample}}$$

---

## 5. On-Demand Post-Hoc Explainability (SHAP & LIME)

Black-box portfolio allocations are unacceptable in institutional finance. As detailed in SDS Section 4.1.3 (SRS FR-11 & FR-12), the platform decouples fast execution from deep interpretability through an **asynchronous on-demand XAI engine**:

```
Client / API Request
        │
        ├───> /predict ──> DRL Policy Inference ──> Return Continuous Weights w_t (Immediate <50ms)
        │
        └───> /explain ──> On-Demand XAI Worker (Async) ──> SHAP & LIME Feature Attribution
```

* **SHAP (SHapley Additive exPlanations)**: Computes game-theoretic Shapley values to measure the global and local contribution of each technical indicator to asset weights.
* **LIME (Local Interpretable Model-agnostic Explanations)**: Fits interpretable sparse linear surrogate models around specific rebalancing events.
* **Non-Blocking Guarantee**: Decision inference returns immediately without waiting for compute-heavy explanation generation.

---

## 6. Multi-Asset & Multi-Exchange Ingestion Pipelines

Expanding data ingestion beyond US Equities (S&P 500) into global multi-asset markets (SRS FR-01):

| Asset Class / Region | Target Exchanges & Tickers | Ingestion Pipeline |
| :--- | :--- | :--- |
| **Indian Equities** | NSE (Nifty 50, Nifty Bank), BSE Sensex | Automated EOD & intraday historical feeds |
| **US Equities** | S&P 500, NASDAQ-100 | yfinance / polygon.io / Alpaca API |
| **Commodities** | Gold (XAU), Crude Oil (WTI), Silver (XAG) | Multi-exchange continuous futures feeds |
| **Forex / Macro** | USD/INR, EUR/USD, Treasury Yields (10Y) | Macroeconomic risk-factor integration |

---

## 7. Enterprise Production Stack (FastAPI, PostgreSQL, Docker)

To transition from research scripts into a production-grade algorithmic service (SRS FR-13, FR-14 & SDS Section 4.1.4):

```
+-----------------------------------------------------------------------------------+
|                              CONTAINERIZED ENVIRONMENT                            |
|                                                                                   |
|  +------------------------+      +--------------------+      +-----------------+  |
|  |     FastAPI Service    |      |  PostgreSQL / ORM  |      |   XAI Worker    |  |
|  | - /predict (DRL / BLM) | <──> | - SQLModel         | <──> | - SHAP Engine   |  |
|  | - /explain (SHAP/LIME) |      | - Alembic Schema   |      | - LIME Engine   |  |
|  | - /backtest (Sim)      |      | - Decision Logs    |      | - Audit Store   |  |
|  +------------------------+      +--------------------+      +-----------------+  |
|                                                                                   |
|                   Deployed via Docker & Docker Compose                            |
+-----------------------------------------------------------------------------------+
```

1. **FastAPI Endpoints**:
   * `POST /predict`: Ingests market state, triggers Jev-BLM or PPO policy, returns optimal portfolio weights.
   * `POST /explain`: Generates SHAP/LIME attributions for any historical decision ID.
   * `POST /backtest`: Runs multi-year chronological simulations with transaction costs and slippage.
2. **Persistence & Audit Logging**:
   * PostgreSQL with SQLModel schemas for storing timestamps, asset states, allocations, reward trajectories, model versions, and explanation artifacts.
   * Alembic for zero-downtime database migrations.
3. **Deployment**:
   * Reproducible multi-container orchestration with Docker Compose.

---

## 8. Roadmap & Milestone Timeline

```
+─────────────────────────────────────────────────────────────────────────────────────────+
| Phase 1: Jev Discriminative Model Integration                                          |
| ├── Replace 30x LLM sampling loop with single-shot Jev probability engine              |
| └── Construct full Bayesian uncertainty matrix Omega for high-frequency BLM             |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                            │
                                            ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| Phase 2: DRL Continuous-Control Engine & Friction Rewards                              |
| ├── Build Gymnasium portfolio environment with transaction cost penalty                 |
| ├── Train PPO continuous-control agent and SAC benchmark                                |
| └── Integrate Ledoit-Wolf shrinkage and technical state vectors                         |
+─────────────────────────────────────────────────────────────────────────────────────────+
                                            │
                                            ▼
+─────────────────────────────────────────────────────────────────────────────────────────+
| Phase 3: Post-Hoc XAI & Production Microservices                                       |
| ├── Implement asynchronous on-demand SHAP and LIME endpoints                            |
| ├── Build FastAPI backend with PostgreSQL decision logging                             |
| └── Containerize full pipeline with Docker Compose                                      |
+─────────────────────────────────────────────────────────────────────────────────────────+
```
