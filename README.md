# CSI500 Stock Selection — LightGBM Portfolio Optimization

**Spring 2026 · CSCI-SHU 360 Machine Learning Competition**

A stock-selection and portfolio-construction pipeline built to select a **long-only portfolio of CSI500 constituents** and evaluate its short-horizon **excess return over the CSI500 index**. Starting from an **organizer-provided XGBoost baseline**, the project developed a LightGBM workflow with engineered market features, time-aware training, robust loss, portfolio-level model selection, and a multi-configuration self-ensemble.

> **Repository orientation:** The final submitted portfolio is [`results/final_self_ensemble.csv`](results/final_self_ensemble.csv). Start with [`generate_and_ensemble_candidates.py`](generate_and_ensemble_candidates.py) for the Phase 2 workflow, or [`model_v2_portfolio_aware_safe.py`](model_v2_portfolio_aware_safe.py) for a single portfolio. The full research write-up is in [`docs/ML Competition Report.docx`](docs/ML%20Competition%20Report.docx).

## Results at a glance

The table reports **four historical, time-ordered self-test windows** shared by all methods. Excess return means portfolio return minus the CSI500 benchmark return over the same holding period; it is not the portfolio's absolute return.

| Model/workflow | Average portfolio return | Average excess return | Median excess | Worst excess | Best excess | Positive windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Competition XGBoost baseline | +3.316% | +0.920% | +0.922% | −1.275% | +3.109% | 3/4 (75%) |
| **Phase 1 selected LightGBM** | **+3.602%** | **+1.207%** | +1.174% | +0.192% | +2.287% | **4/4 (100%)** |
| Phase 2 LightGBM self-ensemble | +3.435% | +1.039% | +1.032% | +0.122% | +1.970% | **4/4 (100%)** |

The Phase 1 live submission returned approximately **+6.99%**, versus approximately **+5.8%** for the class average and **+7.8%** for the class ensemble. Seeing the benefit of aggregation motivated the Phase 2 self-ensemble. Results are from small evaluation samples and are **not evidence of guaranteed future returns**.

## Repository structure

```text
csi500-stock-selection/
├── docs/
│   ├── ML Competition Report.docx         # Full report, experiments, methodology, limitations
│   └── REPRODUCIBILITY_README.pdf         # Original reproducibility handout (legacy paths)
├── intermediate/
│   ├── README.md                          # Historical model and generated scratch outputs
│   └── model_v2.py                        # Earlier LightGBM feature/ensemble development
│   # candidates/, sweep_portfolios/, diagnostics/ are created as needed
├── organizer/
│   ├── README.md                          # Source attribution and upstream links
│   ├── baseline_xgboost.py                # Organizer baseline with path adaptation
│   ├── score_submission.py                # Organizer scorer with path adaptation
│   └── validate_submission.py             # Unmodified organizer submission validator
├── results/
│   └── final_self_ensemble.csv            # Tracked, final Phase 2 submission (50 stocks)
├── README.md                              # Project overview and file guide
├── fetch_extra_data.py                    # Optional supplemental valuation/financial/macro data
├── generate_and_ensemble_candidates.py    # Phase 2 portfolio self-ensemble
├── model_v2_portfolio_aware_safe.py       # Final leakage-aware, single-portfolio model
├── self_test_portfolios.py                # Score already-generated portfolios by window
└── sweep_portfolio_aware_checked.py       # Historical portfolio-configuration sweep


# Local/generated rather than committed:
# data/ (prices.parquet, index.parquet, constituents.csv, optional extra data)
```

### Main scripts

| File | Purpose |
| --- | --- |
| [`model_v2_portfolio_aware_safe.py`](model_v2_portfolio_aware_safe.py) | Trains robust LightGBM models, supports Huber/LambdaRank objectives, date-safe historical portfolio generation, top-*k* selection, liquidity/risk filters, and alternative weighting schemes. |
| [`generate_and_ensemble_candidates.py`](generate_and_ensemble_candidates.py) | Generates and validates multiple strong candidate portfolios; averages their weights, chooses consensus holdings, applies the 10% cap, and writes the final ensemble CSV. |
| [`sweep_portfolio_aware_checked.py`](sweep_portfolio_aware_checked.py) | Tests combinations of portfolio size, loss, weighting, and filtering across specified historical windows, validates outputs, and records performance tables. |
| [`self_test_portfolios.py`](self_test_portfolios.py) | Scores **existing** portfolios over later historical windows; does not train or regenerate portfolios. |
| [`fetch_extra_data.py`](fetch_extra_data.py) | Optionally downloads valuation, quarterly financial, and macroeconomic indicators using AkShare. Financial information should only enter training with a realistic announcement lag. |

### Organizer-provided code and attribution

This project builds on the [NYUSH-ML Spring 2026 CSI500 competition starter repository](https://github.com/NYUSH-ML/ml-competition-sp26). The following **three files were byte-for-byte copies** of the organizer's `main` branch before reorganization (verified October 25, 2026). Two receive a minimal data-path change solely to support their new directory. They are **not original implementations** in this project:

| File | Source and use |
| --- | --- |
| [`organizer/baseline_xgboost.py`](organizer/baseline_xgboost.py) | [Original XGBoost baseline](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/baseline_xgboost.py). Starting-point benchmark; imports `features.py` from the upstream repository (not bundled here). |
| [`organizer/score_submission.py`](organizer/score_submission.py) | [Original scoring utility](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/score_submission.py). Computes realized portfolio, benchmark, and excess returns. |
| [`organizer/validate_submission.py`](organizer/validate_submission.py) | [Original submission validator](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/validate_submission.py). Checks holdings and portfolio-weight constraints. |

**What was developed for this project:** The LightGBM iterations, engineered feature sets and robust training workflow, historical configuration sweep, portfolio-aware selection, self-ensemble generator, and associated experiment analysis. Those project-specific scripts are listed above, alongside the earlier model preserved under `intermediate/`. The comparison establishes which files are upstream copies, not the authorship of every line of project-specific code.

See [`organizer/README.md`](organizer/README.md) for a provenance note and upstream supporting files such as `features.py`, `download_data.py`, and `requirements.txt`.

### Results, intermediate artifacts, and documentation

| Path | Contents |
| --- | --- |
| [`results/final_self_ensemble.csv`](results/final_self_ensemble.csv) | **Final submitted Phase 2 portfolio**: 50 CSI500 stock codes and associated normalized portfolio weights (`stock_code`, `weight`). This is the key output stored in Git. |
| `results/portfolio_sweep_checked_results.csv` | Generated detailed historical sweep results: per-window/per-configuration scores (not included in the current checkout). |
| `results/portfolio_sweep_checked_summary.csv` | Generated configuration-level performance summary from the sweep (not included). |
| `results/portfolio_sweep_checked_results.json` | Generated machine-readable sweep results (not included). |
| `results/self_test_portfolio_results.csv` and `*_summary.csv` | Generated self-test tables when running `self_test_portfolios.py` (not included). |
| [`intermediate/model_v2.py`](intermediate/model_v2.py) | Earlier LightGBM version with cross-sectional targets, robust loss, recency weighting, beta/regime features, and a five-seed ensemble; retained as development history, not the final model. |
| `intermediate/candidates/` | Temporary CSV portfolios produced for individual ensemble configurations; generated on demand and ignored by Git. |
| `intermediate/sweep_portfolios/` | Temporary portfolios from historical sweep runs; generated on demand and ignored by Git. |
| `intermediate/diagnostics/` | Feature importances and ensemble consensus diagnostics; generated on demand and ignored by Git. |
| [`docs/ML Competition Report.docx`](docs/ML%20Competition%20Report.docx) | Detailed technical report explaining the research progression, experimental decisions, test results, and caveats. |
| [`docs/REPRODUCIBILITY_README.pdf`](docs/REPRODUCIBILITY_README.pdf) | Archived competition reproducibility instructions; **some paths refer to the original folder layout**. Use the commands in this README for the reorganized structure. |

## Methodology

### Cross-sectional target and features

For each stock *i* on date *t*, define its five-trading-day return as

<p align="center">
  <b>r<sub>i,t</sub><sup>(5)</sup> = P<sub>i,t+5</sub> / P<sub>i,t</sub> - 1</b>
</p>

Then standardize those forward returns **within each trading date**:

<p align="center">
  <b>z<sub>i,t</sub> = (r<sub>i,t</sub><sup>(5)</sup> - μ<sub>t</sub>) / σ<sub>t</sub></b>
</p>

where:

- **r<sub>i,t</sub><sup>(5)</sup>** = 5-day forward return for stock *i* at time *t*
- **μ<sub>t</sub>** = mean 5-day forward return across all stocks on date *t*
- **σ<sub>t</sub>** = standard deviation of 5-day forward returns across all stocks on date *t*

This targets relative stock selection rather than simply forecasting whether the broad market will rise. Signals include multi-horizon momentum and reversal, volatility, drawdowns, volume and liquidity, moving-average distance, RSI/gap indicators, CSI500-relative alpha, rolling beta, and market-regime interactions.


### Training and leakage controls

- **LightGBM Huber regression** on clipped cross-sectional z-score targets reduced sensitivity to extreme observations; the sweep also evaluated LambdaRank.
- **Exponential recency weights** and an approximately **180-day training window** emphasized more recent trading regimes.
- **Five seeds (42–46)** reduced stochastic training variance through prediction averaging.
- **Time-based splits, a five-trading-day embargo, and historical as-of label cutoffs** prevent validation targets or training labels from reaching into information not available at prediction time.
- Historical portfolios are generated at the stated **as-of** date and evaluated only over **later** trading days. Scoring a portfolio generated with future data on an earlier window is invalid.

### Portfolio construction and experiments

The 240-run experiment swept five values of `top_k` (30, 40, 50, 60, 80), two training objectives (Huber and LambdaRank), three weighting methods (rank, equal, volatility-adjusted), two filter settings, and four historical evaluation windows. The highest mean in that sweep was **+1.643% excess** for the unfiltered top-30 Huber/rank strategy, but its worst window was **−0.662%**. The filtered variant achieved **+1.505% mean excess** with a smaller worst-window loss (**−0.260%**), so it was selected for Phase 1.

For Phase 2, the workflow generated several related LightGBM configurations, combined their portfolio weights, retained the **50 strongest aggregate positions**, capped each position at 10%, and renormalized the total to one. This intentionally traded some concentration for stronger consensus and diversification.

## Running the pipeline

**What is missing:** This public repository contains the project-specific code, three credited organizer scripts, and the final submission, **not** the competition's input market-data files. Provide the original data under `data/` before running inference, sweeps, or scoring. The organizer publishes [`download_data.py`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/download_data.py), [`features.py`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/features.py), and [`requirements.txt`](https://github.com/NYUSH-ML/ml-competition-sp26/blob/main/requirements.txt), but these are **not included** here. To run the unmodified organizer baseline, place the upstream `features.py` next to `organizer/baseline_xgboost.py` locally (or use the complete upstream checkout). Consequently, the checkout is an example of the research code and submitted artifact, not a complete one-command reproduction package.

1. From the repository root, install the applicable Python dependencies, including `numpy`, `pandas`, `pyarrow`, `scipy`, `lightgbm`, and (for optional extra-data fetching) `akshare`. The older XGBoost baseline additionally needs `xgboost` and its missing `features.py` module.
2. Place the competition-provided `prices.parquet`, `index.parquet`, and `constituents.csv` in `data/` in the formats expected by the source scripts.
3. Run any of the following examples **from the repository root**. Adjust `--as-of` and historical windows to match dates available in your own data.

```bash
# Recreate an individual Phase 1-style filtered portfolio
python model_v2_portfolio_aware_safe.py \
  --as-of 20260430 --top-k 30 --objective huber \
  --weight-method rank --filter-universe \
  --out results/phase1_recreated.csv
python organizer/validate_submission.py results/phase1_recreated.csv

# Create Phase 2 ensemble from multiple candidate portfolios
# Select an as-of date actually available in your local data.
python generate_and_ensemble_candidates.py --as-of 20260430 \
  --final-out results/ensemble_recreated.csv
python organizer/validate_submission.py results/ensemble_recreated.csv

# Re-run the original historical portfolio sweep (computationally intensive)
python sweep_portfolio_aware_checked.py \
  --windows 20260403:20260407:20260410 \
            20260410:20260414:20260417 \
            20260417:20260421:20260424 \
            20260424:20260427:20260430 \
  --top-k 30 40 50 60 80 \
  --objectives huber lambdarank \
  --weight-methods rank vol_adjusted equal \
  --filter-modes none filter --no-ensemble
```

The ensemble script's defaults now write **candidate CSVs to `intermediate/candidates/`** and its final output to `results/final_self_ensemble.csv`; if you want to preserve the tracked competition submission, specify a different `--final-out` as in the example above. Sweeps write temporary portfolios to `intermediate/sweep_portfolios/` and score tables to `results/`. The ensemble generator, sweep runner, and self-test wrapper default to calling `organizer/validate_submission.py` and/or `organizer/score_submission.py`. Invoke commands from the repository root.

## Limitations and Credits

Only four independent historical windows were used for the comparisons, and choosing among many tested configurations increases selection/overfitting risk. Real trading would also require explicit modeling of transaction costs, slippage, liquidity, and market impact. External financial data availability was inconsistent. This is a competition research project, **not trading or investment advice**.

Organizer-provided code is credited above. The prediction models were trained on historical market data, and portfolio weights were generated by the program rather than prompting an LLM to choose stocks.
