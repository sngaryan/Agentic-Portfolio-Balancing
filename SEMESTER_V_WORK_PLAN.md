# Semester V Work Plan

## Purpose and scope

This plan divides Semester V work for **Agentic AI for Financial Portfolio Balancing** among the three project members. [`PROJECT_EXPLANATION.md`](PROJECT_EXPLANATION.md) is the core description of the cloned LLM-BLM project and its current workflow. The revised SRS/SDS describes the larger intended system, including later reinforcement learning, explainability, API, database, and deployment work.

Semester V should establish a reliable research foundation: understand the clone, prepare data, construct and validate features, and reproduce or clearly audit the existing portfolio baselines. The literature review has already been completed. The small literature assignments below are for clear ownership and individual viva preparation; they are not a request to repeat the full review.

### Semester V requirement boundary

| SRS requirement | Semester V treatment |
|---|---|
| FR-01 Data Collection | Select and document the assets, source, fields, and date range; verify OHLCV coverage. |
| FR-02 Data Preparation | Build or document repeatable cleaning, alignment, validation, and chronological splits. |
| FR-03 Feature Construction | Construct and validate the specified market features where feasible. |
| FR-04 State Representation | Specify the proposed state and its feature interface; full agent integration is later work. |
| FR-09 Backtesting | Establish a reproducible baseline evaluation protocol and audit the existing backtest. |
| FR-10 Baseline Evaluation | Reproduce or document MVO and LLM-enhanced Black-Litterman comparisons. PPO/SAC comparison follows in a later semester. |
| FR-05–08, FR-11–14 | Record as future implementation requirements; do not let these displace the Semester V foundation unless the supervisor changes the scope. |

The SRS/SDS is the target system specification, while `PROJECT_EXPLANATION.md` explains the existing LLM-BLM codebase. Treat the latter as the project's core account of the clone, but verify every implementation claim against the scripts and saved data before reporting it as a result.

## Three-person work split

These are role-based work packages, not assignments to named team members. The team can choose who takes each package based on interest and progress, while keeping one accountable owner per deliverable.

1. **Data and preprocessing owner**
2. **Features and state specification owner**
3. **Baselines and evaluation owner**

Each owner should maintain a brief contribution log: files or sections owned, decisions made, commands or steps used to reproduce the work, and questions they can answer in a viva.

---

## Data and preprocessing owner

### Objective

Provide a documented, reproducible dataset that Persons 2 and 3 can use without making separate or inconsistent data choices.

### Detailed work

1. **Inventory existing project data**
   - Inspect `yfinance/`, `responses/`, and relevant result files.
   - Record the assets/tickers, date coverage, frequency, file formats, and columns in the existing data.
   - Confirm whether the current files contain Open, High, Low, Close, and Volume (OHLCV), or only Close and derived returns, as described in the current code walkthrough.
   - Note missing files, duplicate dates, missing observations, inconsistent ticker labels, and other quality issues.
   - Distinguish original downloaded data from derived returns, model responses, and generated portfolio outputs.

2. **Agree and document the experimental dataset**
   - Propose a feasible asset universe and date range that support the project’s experiments and later agent work.
   - Check that there are enough observations for the selected 30-day rolling features and chronological evaluation.
   - Record the data source, retrieval method/date where available, ticker symbols, currency/market, and known limitations.
   - If the existing dataset is unsuitable, document the minimum additional data required before fetching or regenerating anything.

3. **Prepare a repeatable pipeline**
   - Load raw data, standardize column names and date representation, sort chronologically, and validate types.
   - Handle duplicate records and missing values using a documented rule; avoid silently filling data in ways that create artificial observations.
   - Align trading dates across assets and explain the chosen alignment policy.
   - Generate returns consistently and preserve a clear separation between raw and processed inputs.
   - Save a configuration or concise run instructions so the same preparation can be repeated.

4. **Define chronological data periods**
   - Propose training, validation, and test intervals with no random shuffling across time.
   - Confirm these periods with Person 3 before baseline evaluation.
   - Provide Person 2 the prepared time series and date conventions before feature integration begins.

5. **Write the data documentation**
   - Describe the dataset, preprocessing decisions, exclusions, and known limitations.
   - Include a small validation summary: row/date coverage, number of assets, missing-value counts before and after cleaning, and OHLCV availability.

### Literature ownership for viva

Prepare a concise, non-duplicative refresher on **financial time-series data for portfolio research**: OHLCV fields, data quality and alignment, missing observations, chronological splits, and why future information must not leak into a historical decision. Link this to FR-01 and FR-02 and the data path in `PROJECT_EXPLANATION.md`.

### Deliverables

- Dataset inventory and source/coverage note.
- Repeatable data loading and preprocessing pipeline, or a clearly specified pipeline if the dataset cannot be retrieved locally.
- Prepared dataset or documented blocker and next action.
- Chronological split proposal agreed with Person 3.
- Data quality summary and Semester V report section.
- Viva notes and a short contribution log.

### Completion checks

- A teammate can identify which files are raw inputs and which are derived outputs.
- The selected dataset contains the fields required for the agreed work, or its limitations are stated plainly.
- The same preprocessing procedure yields the same date range, asset columns, and row counts.
- The training/validation/test boundaries are explicit and chronological.

---

## Feature engineering and state specification owner

### Objective

Deliver validated market features for Semester V and a precise interface specification for the future DRL state, without expanding this semester into PPO/SAC implementation.

### Detailed work

1. **Agree feature inputs and conventions**
   - Wait for Person 1's dataset schema and date conventions before finalizing implementation.
   - Document price/volume columns, return convention, feature frequency, rolling window length, and any scaling or normalization approach.
   - Confirm with Person 3 which dates are used for training and evaluation so feature calculations respect the same boundaries.

2. **Implement the required indicators**
   - Compute MACD, RSI, ADX, and CCI using the prepared OHLCV series.
   - Compute rolling volatility vectors using the documented 30-day window.
   - Investigate a 30-day Ledoit-Wolf shrinkage risk estimate. Implement it if the selected assets and available data support it; otherwise provide the technical rationale and exact blocker.
   - Record formulas or library conventions, parameter values, warm-up behavior, and output units for each feature.

3. **Validate feature outputs**
   - Check date alignment, missing or infinite values, expected ranges where meaningful, and warm-up rows.
   - Confirm feature values at each date are derived only from information available by that date.
   - Add small sanity checks or a reproducible validation notebook/script as appropriate.
   - Avoid fitting scalers or other transforms on future/test data; if scaling is used, fit it on training data and apply the saved transform forward.

4. **Specify the future agent state**
   - Propose a state layout that combines technical, trend/regime, compact risk, and current portfolio information, as described in FR-04 and the SDS.
   - Document feature ordering, dimensions, asset ordering, missing-value policy, and whether values are scaled.
   - State that this is an interface/design handoff for Semester VI; training PPO/SAC is not part of this owner’s Semester V deliverable.

5. **Hand features to baseline/evaluation owner**
   - Tell Person 3 what date range each feature supports after rolling-window warm-up.
   - Ensure any comparisons using features or risk estimates use only dates that are available consistently.

### Literature ownership for viva

Prepare a concise refresher on **technical indicators and compact risk representations**: what momentum, trend/regime, and volatility features summarize; why these may be useful to an agent state; what shrinkage covariance is intended to address; and what information a compact risk vector may omit. Connect the explanation to FR-03/FR-04 and the proposed state in the SRS/SDS.

### Deliverables

- Feature construction module or reproducible notebook/script.
- Feature definitions, parameters, units, and warm-up behavior.
- Validation and leakage-check summary.
- Proposed state-vector/interface specification for Semester VI.
- Semester V report section, viva notes, and contribution log.

### Completion checks

- A teammate can regenerate the features from the prepared inputs.
- Feature output dates and asset order are explicit and stable.
- Any dropped warm-up dates or unavailable Ledoit-Wolf results are explained.
- No future/test observations are used to compute earlier features or fit transformations.
- State representation is described clearly enough for a future RL module to consume.

---

## Baselines and evaluation owner

### Objective

Establish which existing portfolio results are reproducible, produce a fair baseline comparison protocol, and document the baseline evidence needed before DRL experiments begin.

### Detailed work

1. **Trace the cloned LLM-BLM workflow**
   - Use `PROJECT_EXPLANATION.md` as the conceptual guide.
   - Inspect `run.py`, `baselines.py`, `evaluate_multiple.py`, and `calculate_llm_returns.py` to verify the actual inputs, assumptions, date windows, and outputs.
   - Trace how returns, LLM views (`Q`), view uncertainty (`Omega`), market prior (`Pi`), posterior returns, and portfolio weights are represented in code and files.
   - Compare the written explanation with the implementation and note discrepancies rather than silently treating the explanation as proof.

2. **Audit the saved portfolio outputs**
   - Inspect `responses_portfolios/` and `results/` and classify generated files as equal-weight, MVO, LLM-BL, or evaluation output.
   - Check whether the saved outputs match the dates and inputs expected by the scripts.
   - Preserve existing outputs and report any unexplained or stale files; do not overwrite project data as part of an audit.

3. **Set the comparison protocol with Person 1**
   - Agree on the common asset universe, date range, chronological split, return definition, and rebalance cadence.
   - Include equal-weight, Classical MVO, and the LLM-Enhanced Black-Litterman baseline where reproducible.
   - Specify transaction-cost treatment. If costs are not implemented in an existing baseline, label results as gross of costs or add a clearly justified consistent cost calculation where feasible.
   - Keep the LLM-BL baseline at the fixed monthly rebalancing described in the SRS, and document the out-of-sample period chosen.

4. **Reproduce and summarize results**
   - Regenerate baseline results where the required data and configuration are available.
   - Record setup, inputs, settings, software assumptions, and any required model endpoint/API dependency.
   - If LLM prediction generation cannot run because credentials, endpoint configuration, or other dependencies are unavailable, do not fabricate results. Report which existing predictions can still be evaluated and what remains blocked.
   - Calculate or verify relevant metrics: cumulative/annualized return, volatility, Sharpe ratio, maximum drawdown, and turnover, where the data supports them.
   - Include metric definitions and state whether results include transaction costs.

5. **Prepare the next-stage evaluation plan**
   - Specify how later PPO and SAC results should be compared against the same Classical MVO and LLM-BL baselines.
   - Record what must be held constant (assets, dates, transaction costs, and metrics) for a fair comparison.

### Literature ownership for viva

Prepare a concise refresher on **portfolio optimization and the project's baselines**: mean-variance optimization, Black-Litterman market prior and views, the LLM-generated views/uncertainty described in `PROJECT_EXPLANATION.md`, and what each baseline contributes to the eventual comparison. Connect this to FR-09/FR-10 and the experiment design in the SRS/SDS.

### Deliverables

- Verified workflow map from input data through baseline weights and evaluated returns.
- Baseline reproducibility report and regenerated results where feasible.
- Documented evaluation protocol, metrics, transaction-cost assumptions, and limitations.
- Clear list of code/explanation discrepancies and any blockers.
- Semester V report section, viva notes, and contribution log.

### Completion checks

- A reader can tell which baseline produced each result file and which data period it covers.
- At least one baseline can be reproduced end to end, or blockers are evidenced and precisely described.
- Comparisons use agreed dates, assets, and metric conventions.
- Gross/net-of-cost status and rebalancing assumptions are explicit.
- No market or model output is invented to fill gaps.

---

## Checkpoints and inter-person handoffs

The schedule below is relative to the team's Semester V start date. Adjust the calendar weeks to your supervisor's deadlines, but keep the dependency gates.

### Checkpoint 0 — Kickoff and scope agreement (all three; first meeting)

**Agree together on:**
- Semester V boundary and the deliverables in this plan.
- Proposed asset universe and data period, subject to Person 1's feasibility check.
- Output format, folder naming, return convention, and how work will be shared.
- One common experiment assumptions sheet: assets, dates, rebalancing cadence, transaction-cost assumptions, and metrics.
- Which existing local outputs should be preserved during the audit.

**Exit condition:** each owner has a task list; no one has started an incompatible dataset or metric convention.

### Checkpoint 1 — Clone and data audit (Weeks 1–2; parallel work)

- Person 1 inventories available datasets and reports OHLCV/date coverage.
- Person 2 drafts the feature/state specification and literature viva notes, but waits to lock feature inputs until the data schema is confirmed.
- Person 3 traces scripts and saved outputs, compares the implementation with `PROJECT_EXPLANATION.md`, and drafts the baseline audit without regenerating outputs yet.

**Handoff:** Person 1 shares the dataset inventory/schema with Persons 2 and 3. Person 3 shares verified baseline inputs/date assumptions with Person 1.

**Exit condition:** all agree on the dataset source, assets, date range, and known feasibility gaps.

### Checkpoint 2 — Freeze dataset and experiment assumptions (end of Week 2)

- Person 1 publishes the agreed data schema and proposed chronological splits.
- Person 3 confirms that the splits and return windows support a fair baseline comparison.
- Person 2 confirms that the data supports the required indicators and rolling window.

**Wait gate:** Persons 2 and 3 should not finalize computations on an unapproved ticker set/date range. They may build generic code or prepare documentation while waiting.

**Exit condition:** one written configuration is approved by all three (and supervisor if required).

### Checkpoint 3 — Prepare data and implement in parallel (Weeks 3–4)

- Person 1 completes preprocessing and publishes versioned/clearly named prepared data plus the run instructions.
- Person 2 starts feature generation as soon as the prepared schema is stable; reports rolling-window warm-up dates to Person 3.
- Person 3 prepares baseline code/configuration against the agreed data contract and identifies any API or dependency blockers.

**Wait gate:** Person 2 must wait for Person 1's prepared data contract before final validation. Person 3 may audit and configure against the agreed contract, but must wait for the frozen dataset before producing final comparison metrics.

**Exit condition:** prepared data passes checks; feature and baseline inputs share the same asset order and date conventions.

### Checkpoint 4 — Integration and reproducibility review (Week 5)

- Person 1 reruns the data pipeline from documented inputs.
- Person 2 regenerates features from the prepared data and verifies date coverage/leakage checks.
- Person 3 regenerates baseline results using the frozen experiment configuration and records metrics or blockers.
- All three compare row counts, asset order, date ranges, and assumptions.

**Wait gate:** final metrics should wait until the data and feature warm-up periods are settled. If a baseline does not use the new features, it still must use the same agreed asset/date evaluation window where possible.

**Exit condition:** each reported table/figure can be traced to inputs and a reproducible run or is explicitly marked as non-reproducible with a reason.

### Checkpoint 5 — Report, viva, and supervisor review (Week 6)

- Each owner writes their report section and provides their evidence/contribution log.
- Each person reviews the other two sections for consistency.
- The team assembles a unified report with the chain: clone audit → dataset → features → baselines → results/limitations → Semester VI handoff.
- Conduct a short internal viva: each person explains their own work and answers basic questions about the other two modules.

**Exit condition:** no conflicting ticker/date/metric assumptions; all claims distinguish implemented results from proposed future work.

## Handoff contracts

Use these contracts to avoid silent dependency mismatches:

1. **Person 1 → Persons 2 and 3:** prepared data files, data dictionary, asset order, date index, return definition, split boundaries, missing-data policy, and known limitations.
2. **Person 2 → Persons 1 and 3:** feature names/formulas, asset and date alignment, warm-up window, scaling policy, validation evidence, and state-vector proposal.
3. **Person 3 → Persons 1 and 2:** baseline assumptions, rebalancing dates, metric definitions, transaction-cost convention, reproducibility status, and required evaluation-period coverage.
4. **All → final report:** source and configuration references, concise method description, evidence for results, limitations, and individual contribution statement.

## Integration risks to resolve early

- `PROJECT_EXPLANATION.md` describes the intended/explained LLM-BLM workflow; the current repository may contain edits and generated files. Verify actual code and outputs before claiming the workflow is reproducible.
- The existing LLM query script may require endpoint/API configuration. An API blocker should not prevent the team from documenting and evaluating already-saved responses where valid.
- The existing project may use monthly closing-price data while the SRS calls for daily OHLCV. Record whether you are extending the dataset or explicitly limiting a baseline to existing data.
- Feature warm-up periods can shorten the usable date range. Report the common evaluation interval rather than comparing strategies on unrelated periods.
- Transaction costs and turnover conventions must be fixed before comparing strategies; report whether each result is gross or net of costs.
- Ledoit-Wolf risk features, full PPO/SAC agents, SHAP/LIME, FastAPI, PostgreSQL, and Docker are wider system requirements. Keep Semester V centered on its assigned foundation unless the supervisor explicitly revises the milestone.

## Final Semester V package

The team should finish with:

- A verified description of what the LLM-BLM clone implements, using `PROJECT_EXPLANATION.md` as the core reference and noting discrepancies.
- A documented, reproducible dataset and preprocessing process.
- Validated feature outputs and a proposed state interface for Semester VI.
- Reproducible baseline results or an evidence-based explanation of remaining blockers.
- A consistent experiment protocol and limitations statement.
- Individually owned report sections, literature viva notes, and contribution logs.
- A Semester VI plan for Gymnasium environment design, PPO/SAC, friction-aware reward, XAI, and service/deployment requirements.

