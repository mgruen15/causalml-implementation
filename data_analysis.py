"""
data_analysis.py

Diagnostic script to analyze the merged panel data from replicate_causal_ml.py
and identify potential reasons for non-significant causal effects.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from replicate_causal_ml import preprocess_data

def run_analysis():
    print("=== Starting Data Analysis ===")
    
    # 1. Get the preprocessed panel data
    panel, treatment_cols = preprocess_data(full_run=True)
    
    print(f"\nPanel Shape: {panel.shape}")
    print(f"Treatment Columns: {treatment_cols}")
    
    # 2. Outcome Distribution Analysis
    outcomes = ['avg_daily_expenditure', 'avg_daily_expenditure_demeaned']
    print("\n--- Outcome Summary Statistics ---")
    print(panel[outcomes].describe())
    
    zero_exp_pct = (panel['avg_daily_expenditure'] == 0).mean() * 100
    print(f"Percentage of rows with zero expenditure: {zero_exp_pct:.2f}%")
    
    # 3. Treatment Prevalence
    print("\n--- Treatment Prevalence ---")
    prevalence = []
    for col in treatment_cols:
        count = panel[col].sum()
        pct = panel[col].mean() * 100
        prevalence.append({'Treatment': col, 'Count': count, 'Percentage': f"{pct:.2f}%"})
    
    prev_df = pd.DataFrame(prevalence)
    print(prev_df)
    
    # 4. Simple Comparison of Means (T-Tests)
    print("\n--- Simple Comparison of Means (T-tests) ---")
    results = []
    for t_col in treatment_cols:
        for o_col in outcomes:
            treated = panel[panel[t_col] == 1][o_col]
            untreated = panel[panel[t_col] == 0][o_col]
            
            if len(treated) > 0 and len(untreated) > 0:
                t_stat, p_val = stats.ttest_ind(treated, untreated, equal_var=False)
                results.append({
                    'Treatment': t_col,
                    'Outcome': o_col,
                    'Mean Treated': treated.mean(),
                    'Mean Untreated': untreated.mean(),
                    'Diff': treated.mean() - untreated.mean(),
                    'P-Value': p_val
                })
    
    comp_df = pd.DataFrame(results)
    print(comp_df.sort_values('P-Value'))
    
    # 5. Correlation Between Treatments
    print("\n--- Correlation Between Treatments ---")
    corr_matrix = panel[treatment_cols].corr()
    print(corr_matrix)
    
    # 5b. Overlap Analysis
    if 'treatment_drugstore items' in treatment_cols and 'treatment_other food' in treatment_cols:
        print("\n--- Overlap: Drugstore vs Other Food ---")
        overlap = pd.crosstab(panel['treatment_drugstore items'], panel['treatment_other food'])
        print(overlap)
        both = ((panel['treatment_drugstore items'] == 1) & (panel['treatment_other food'] == 1)).sum()
        only_drug = ((panel['treatment_drugstore items'] == 1) & (panel['treatment_other food'] == 0)).sum()
        only_food = ((panel['treatment_drugstore items'] == 0) & (panel['treatment_other food'] == 1)).sum()
        print(f"Both: {both}, Only Drugstore: {only_drug}, Only Food: {only_food}")
    
    # 6. Covariate Balance Check (Example: Drugstore Items)
    target_t = 'treatment_drugstore items'
    if target_t in treatment_cols:
        print(f"\n--- Covariate Balance Check for {target_t} ---")
        # Select some interesting covariates: lagged spending and some demographics
        lag_cols = [c for c in panel.columns if 'lagged_spend' in c]
        demo_cols = [c for c in panel.columns if 'age_range' in c or 'income' in c][:5] # first few
        
        balance_cols = lag_cols + demo_cols
        balance_results = []
        
        for b_col in balance_cols:
            treated_val = panel[panel[target_t] == 1][b_col].mean()
            untreated_val = panel[panel[target_t] == 0][b_col].mean()
            balance_results.append({
                'Covariate': b_col,
                'Mean Treated': treated_val,
                'Mean Untreated': untreated_val,
                'Diff': treated_val - untreated_val
            })
        
        balance_df = pd.DataFrame(balance_results)
        print(balance_df)

    # 7. Time Period Analysis
    print("\n--- Period Analysis ---")
    period_cols = [c for c in panel.columns if 'FE_period' in c]
    if period_cols:
        # Reconstruct period_id if possible or just use FE columns
        period_spending = []
        for p_col in period_cols:
            mean_spend = panel[panel[p_col] == 1]['avg_daily_expenditure'].mean()
            period_spending.append({'Period': p_col, 'Mean Spend': mean_spend})
        
        period_df = pd.DataFrame(period_spending)
        print("Mean spending across periods (first 5 and last 5):")
        print(period_df.head(5))
        print(period_df.tail(5))

if __name__ == '__main__':
    run_analysis()
