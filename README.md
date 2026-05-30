# Causal ML Coupon Campaign Replication

This project replicates the causal machine learning and optimal policy learning methodologies from the paper:
**"How causal machine learning can leverage marketing strategies: Assessing and improving the performance of a coupon campaign" (Langen & Huber, 2023)**.

The goal is to evaluate the causal impact of coupon campaigns on a retailer's sales using the AmExpert 2019 dataset, accounting for unconfoundedness (selection-on-observables).

## Project Structure

- `replicate_causal_ml_v2.py`: The main Python script implementing the replication pipeline using R's `grf` package via `rpy2`.
- `data_analysis.py`: Diagnostic script for analyzing panel data and treatment prevalence.
- `visualize_results.py`: Generates plots (CATE distributions, GATE heatmaps, DAGs) from the results.
- `data/`: Directory containing the raw AmExpert 2019 CSV files.
- `results/`: Directory for output CSV files (ATE, GATE, calibration, etc.).
- `plots/`: Directory for generated visualizations.

## Methodology

The replication pipeline follows these steps:
1. **Data Preprocessing:** Aligns overlapping campaigns into non-overlapping artificial periods (Paper Section 4).
2. **Feature Engineering:** Constructs covariates including socio-demographics, lagged spending behaviors, and lagged coupon history.
3. **Causal Forest Estimation:** Uses R's `grf::causal_forest` with 2,000 honest trees and clustered standard errors (Section 6.1).
4. **ATE & Validation:** Extracts Average Treatment Effects using doubly robust AIPW inference with propensity score trimming (0.01 - 0.99).
5. **Group Average Treatment Effects (GATE):** Analyzes treatment effect heterogeneity across subgroups (age, income, family size) using `best_linear_projection`.
6. **Robustness Checks:** 
    - **Double ML:** Validates results using Lasso-based Double ML via the `causalDML` R package.
    - **Reduced Sample:** Re-runs analysis on a subset with known socio-economic data.
7. **Visualization:** Produces CATE distribution histograms, violin plots, GATE subgroup error bars, heatmaps, and Structural Causal Models (DAGs).

## Installation

This project requires both Python and R.

### Python Dependencies
```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn rpy2
```

### R Dependencies
Ensure you have R installed, then install the required packages:
```bash
Rscript -e "install.packages(c('grf', 'causalDML'), repos='https://cloud.r-project.org')"
```

## Usage

1. **Run Replication:**
   ```bash
   python replicate_causal_ml_v2.py
   ```
   This will populate the `results/` folder with CSV files.

2. **Generate Visualizations:**
   ```bash
   python visualize_results.py
   ```
   This will create plots in the `plots/` folder.

3. **(Optional) Run Diagnostics:**
   ```bash
   python data_analysis.py
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
