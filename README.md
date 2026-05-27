# Causal ML Coupon Campaign Replication

This project replicates the causal machine learning and optimal policy learning methodologies from the paper:
**"How causal machine learning can leverage marketing strategies: Assessing and improving the performance of a coupon campaign" (Langen & Huber, 2023)**.

The goal is to evaluate the causal impact of coupon campaigns on a retailer's sales using the AmExpert 2019 dataset, accounting for unconfoundedness (selection-on-observables).

## Project Structure

- `replicate_causal_ml.py`: The main Python script implementing the 7-step replication pipeline.
- `data/`: Directory containing the raw AmExpert 2019 CSV files.
- `MEMORY.md`: Private project notes.

## Methodology

The script implements the following steps:
1. **Data Preprocessing:** Aligns overlapping campaigns into 33 non-overlapping artificial periods.
2. **Feature Engineering:** Constructs covariates including socio-demographics, lagged spending behaviors, and lagged coupon history.
3. **Causal Forest Estimation:** Uses `econml.dml.CausalForestDML` with 2,000 honest trees and clustered standard errors.
4. **ATE & Validation:** Extracts Average Treatment Effects using doubly robust inference with propensity score trimming (0.01 - 0.99).
5. **Group Average Treatment Effects (GATE):** Analyzes treatment effect heterogeneity across customer archetypes using OLS.
6. **Optimal Policy Learning:** Fits a depth-3 policy tree to maximize empirical welfare through data-driven targeting.
7. **Robustness Check:** Validates results on a clean subset of socio-economic data.

## Installation

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy scikit-learn statsmodels econml
```

## Usage

To run the replication pipeline:

```bash
python replicate_causal_ml.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
