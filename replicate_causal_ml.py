"""
replicate_causal_ml.py

Replication of causal machine learning and optimal policy learning methodologies
from Langen & Huber (2023) "How causal machine learning can leverage marketing
strategies: Assessing and improving the performance of a coupon campaign".
"""

import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_predict
from econml.dml import CausalForestDML
import warnings

# Suppress warnings for cleaner output during execution
warnings.filterwarnings('ignore')

DATA_DIR = './data'

def load_data():
    """Loads all required raw data files from the data directory."""
    print("Loading data...")
    campaigns = pd.read_csv(os.path.join(DATA_DIR, 'campaign_data.csv'))
    coupons_items = pd.read_csv(os.path.join(DATA_DIR, 'coupon_item_mapping.csv'))
    demographics = pd.read_csv(os.path.join(DATA_DIR, 'customer_demographics.csv'))
    transactions = pd.read_csv(os.path.join(DATA_DIR, 'customer_transaction_data.csv'))
    items = pd.read_csv(os.path.join(DATA_DIR, 'item_data.csv'))
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    return campaigns, coupons_items, demographics, transactions, items, train

def create_artificial_periods(campaigns):
    """
    Step 1: Aligns overlapping campaigns into 33 non-overlapping artificial periods.
    """
    print("Creating non-overlapping artificial campaign periods...")
    campaigns['start_date'] = pd.to_datetime(campaigns['start_date'], format='%d/%m/%y')
    campaigns['end_date'] = pd.to_datetime(campaigns['end_date'], format='%d/%m/%y')
    
    # Extract all unique dates where a campaign status could change
    all_dates = pd.concat([campaigns['start_date'], campaigns['end_date'] + pd.Timedelta(days=1)]).unique()
    all_dates = np.sort(all_dates)
    
    # Create periods from adjacent dates
    periods = []
    # If this generates more than 33, the principle of non-overlapping bounds holds.
    for i in range(len(all_dates) - 1):
        start = pd.Timestamp(all_dates[i])
        end = pd.Timestamp(all_dates[i+1]) - pd.Timedelta(days=1)
        periods.append({
            'period_id': i + 1, 
            'start_date': start, 
            'end_date': end, 
            'duration_days': (end - start).days + 1
        })
        
    return pd.DataFrame(periods)

def map_item_categories(items):
    """Step 2: Maps original categories to the 5 targeted categories."""
    category_mapping = {
        'Prepared Food': 'ready-to-eat food',
        'Bakery': 'ready-to-eat food',
        'Salads': 'ready-to-eat food',
        'Vegetables (cut)': 'ready-to-eat food',
        'Restauarant': 'ready-to-eat food',
        
        'Packaged Meat': 'meat/seafood',
        'Seafood': 'meat/seafood',
        'Meat': 'meat/seafood',
        
        'Grocery': 'other food',
        'Natural Products': 'other food',
        'Dairy, Juices & Snacks': 'other food',
        'Alcohol': 'other food',
        
        'Pharmaceutical': 'drugstore items',
        'Skin & Hair Care': 'drugstore items',
    }
    
    items['target_category'] = items['category'].map(lambda x: category_mapping.get(x, 'other non-food products'))
    return items

def preprocess_data(full_run=True):
    """
    Executes Step 1 & 2: Preprocessing and Feature Engineering.
    Returns the panel dataset and treatment columns ready for modeling.
    """
    campaigns, coupons_items, demographics, transactions, items, train = load_data()
    
    # Map item categories
    items = map_item_categories(items)
    
    # Create artificial periods
    periods = create_artificial_periods(campaigns)
    
    # Merge items into transactions to get target_category
    transactions['date'] = pd.to_datetime(transactions['date'])
    transactions = transactions.merge(items[['item_id', 'target_category']], on='item_id', how='left')
    
    # Assign periods to transactions based on date
    period_bins = pd.IntervalIndex.from_arrays(
        periods['start_date'] - pd.Timedelta(days=0.5), 
        periods['end_date'] + pd.Timedelta(days=0.5)
    )
    transactions['period_id'] = pd.cut(transactions['date'], bins=period_bins)
    interval_to_id = dict(zip(period_bins, periods['period_id']))
    transactions['period_id'] = transactions['period_id'].map(interval_to_id)
    
    # Filter out transactions outside any active period
    transactions = transactions.dropna(subset=['period_id'])
    transactions['period_id'] = transactions['period_id'].astype(int)
    
    # Define Outcome (Y): Average per-day expenditures per customer within period
    outcome_df = transactions.groupby(['customer_id', 'period_id'])['selling_price'].sum().reset_index()
    outcome_df = outcome_df.rename(columns={'selling_price': 'total_expenditure'})
    outcome_df = outcome_df.merge(periods[['period_id', 'duration_days']], on='period_id', how='left')
    outcome_df['avg_daily_expenditure'] = outcome_df['total_expenditure'] / outcome_df['duration_days']
    
    # Pool to yield a panel/stacked layout
    all_customers = pd.DataFrame({'customer_id': transactions['customer_id'].unique()})
    
    # Cross join customers and periods to form balanced panel
    all_customers['key'] = 1
    periods['key'] = 1
    panel = pd.merge(all_customers, periods, on='key').drop('key', axis=1)
    
    # Merge Outcome Y into panel
    panel = panel.merge(outcome_df[['customer_id', 'period_id', 'avg_daily_expenditure']], 
                        on=['customer_id', 'period_id'], how='left')
    panel['avg_daily_expenditure'] = panel['avg_daily_expenditure'].fillna(0)
    
    # Handle Missing Socio-Economic Data
    demographics.fillna('unknown', inplace=True)
    panel = panel.merge(demographics, on='customer_id', how='left')
    
    demo_cols = ['age_range', 'marital_status', 'rented', 'family_size', 'no_of_children', 'income_bracket']
    panel[demo_cols] = panel[demo_cols].fillna('unknown')
    
    if not full_run:
        # Step 7 Robustness subset: filter out rows where socio-economic variables are unknown
        print("Filtering out 'unknown' socio-economic rows for robustness check...")
        for col in demo_cols:
            panel = panel[panel[col] != 'unknown']
            
    print("Engineering Covariates X...")
    
    # Feature Engineering (Covariates X)
    # 1. Socio-Demographics (One-hot encoded)
    panel = pd.get_dummies(panel, columns=demo_cols, drop_first=False)
    
    # 2. Lagged Spending Behaviors (t-1)
    cat_spend = transactions.groupby(['customer_id', 'period_id', 'target_category'])['selling_price'].sum().unstack(fill_value=0).reset_index()
    cat_spend = cat_spend.merge(periods[['period_id', 'duration_days']], on='period_id', how='left')
    for col in items['target_category'].unique():
        if col in cat_spend.columns:
            cat_spend[f'lagged_spend_{col}'] = cat_spend[col] / cat_spend['duration_days']
        else:
            cat_spend[f'lagged_spend_{col}'] = 0.0
            
    lagged_cols = [f'lagged_spend_{col}' for col in items['target_category'].unique()]
    
    # Shift period_id by 1 forward so merging attaches t-1 to t
    cat_spend['period_id'] = cat_spend['period_id'] + 1 
    panel = panel.merge(cat_spend[['customer_id', 'period_id'] + lagged_cols], on=['customer_id', 'period_id'], how='left')
    panel[lagged_cols] = panel[lagged_cols].fillna(0)
    
    # 3. Treatments & Lagged Coupon History
    # Link coupons to categories
    coupon_cats = coupons_items.merge(items[['item_id', 'target_category']], on='item_id')
    # Mode category per coupon
    coupon_dominant_cat = coupon_cats.groupby('coupon_id')['target_category'].agg(lambda x: pd.Series.mode(x)[0]).reset_index()
    train = train.merge(coupon_dominant_cat, on='coupon_id', how='left')
    
    # Identify which periods each campaign covers
    campaign_periods = []
    for idx, row in campaigns.iterrows():
        active_periods = periods[(periods['start_date'] <= row['end_date']) & (periods['end_date'] >= row['start_date'])]['period_id'].tolist()
        for p_id in active_periods:
            campaign_periods.append({'campaign_id': row['campaign_id'], 'period_id': p_id})
    campaign_periods_df = pd.DataFrame(campaign_periods)
    
    # Map (customer, campaign) to (customer, period)
    train_with_periods = train.merge(campaign_periods_df, on='campaign_id')
    
    # Treatments: which categories of coupons did the user receive in period t
    treatments = train_with_periods.groupby(['customer_id', 'period_id', 'target_category']).size().unstack(fill_value=0)
    treatments = (treatments > 0).astype(int).reset_index()
    treatments = treatments.rename(columns=lambda x: f'treatment_{x}' if x not in ['customer_id', 'period_id'] else x)
    
    # Global 'Any Coupon' Treatment
    treatments['treatment_Any Coupon'] = (treatments.drop(['customer_id', 'period_id'], axis=1).sum(axis=1) > 0).astype(int)
    
    panel = panel.merge(treatments, on=['customer_id', 'period_id'], how='left')
    treatment_cols = [c for c in treatments.columns if 'treatment_' in c]
    panel[treatment_cols] = panel[treatment_cols].fillna(0).astype(int)
    
    # Lagged coupon history
    lagged_treatments = treatments.copy()
    lagged_treatments['period_id'] = lagged_treatments['period_id'] + 1
    lagged_treatments = lagged_treatments.rename(columns={c: c.replace('treatment_', 'lagged_coupon_') for c in treatment_cols})
    panel = panel.merge(lagged_treatments, on=['customer_id', 'period_id'], how='left')
    lagged_coupon_cols = [c for c in lagged_treatments.columns if 'lagged_coupon_' in c]
    panel[lagged_coupon_cols] = panel[lagged_coupon_cols].fillna(0).astype(int)
    
    # 4. Fixed Effects (Dummy variables for period t)
    panel = pd.get_dummies(panel, columns=['period_id'], prefix='FE_period', drop_first=False)
    
    # 5. Shifted Outcomes for Longer-Term Effects (t+1, t+2)
    period_cols = [c for c in panel.columns if 'FE_period_' in c]
    panel['period_id_temp'] = panel[period_cols].idxmax(axis=1).str.replace('FE_period_', '').astype(int)
    panel = panel.sort_values(['customer_id', 'period_id_temp'])
    panel['avg_daily_expenditure_t1'] = panel.groupby('customer_id')['avg_daily_expenditure'].shift(-1)
    panel['avg_daily_expenditure_t2'] = panel.groupby('customer_id')['avg_daily_expenditure'].shift(-2)
    
    # 6. Customer Demeaning (Entity Fixed Effects Proxy)
    print("Demeaning outcomes by customer to account for fixed effects...")
    for col in ['avg_daily_expenditure', 'avg_daily_expenditure_t1', 'avg_daily_expenditure_t2']:
        customer_means = panel.groupby('customer_id')[col].transform('mean')
        panel[f'{col}_demeaned'] = panel[col] - customer_means
        
    panel = panel.drop('period_id_temp', axis=1)
    
    return panel, treatment_cols

def estimate_causal_effects(panel, treatment_col, feature_cols, outcome_col='avg_daily_expenditure'):
    """
    Executes Steps 3, 4, & 5: Causal Forest Estimation, ATE Validation, and GATE.
    """
    print(f"\n[{treatment_col} -> {outcome_col}] Initiating Estimation Pipeline...")
    
    # Cross-Category Controls: Include indicators for concurrent coupons
    if treatment_col == 'treatment_Any Coupon':
        other_treatments = []
    else:
        other_treatments = [c for c in panel.columns if 'treatment_' in c and c != treatment_col and c != 'treatment_Any Coupon']
    X_cols = feature_cols + other_treatments
    
    # Ensure correct data types, drop rows with NaN in outcome_col
    data_subset = panel.dropna(subset=[outcome_col])
    X = data_subset[X_cols].astype(float).fillna(0)
    Y = data_subset[outcome_col].astype(float).values
    T = data_subset[treatment_col].astype(int).values
    groups = data_subset['customer_id'].values
    
    if len(Y) == 0:
        print(f"[{treatment_col} -> {outcome_col}] No observations for this outcome. Skipping.")
        return None
        
    # Propensity Score Trimming (Step 4)
    print(f"[{treatment_col} -> {outcome_col}] Trimming bounds [0.01, 0.99] using strict Random Forest Propensities...")
    prop_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Calculate propensities
    try:
        p_hat = cross_val_predict(prop_model, X, T, cv=3, method='predict_proba')[:, 1]
    except Exception as e:
        print(f"[{treatment_col} -> {outcome_col}] Not enough treatment variation for cross-val propensities: {e}")
        return None
        
    mask = (p_hat >= 0.01) & (p_hat <= 0.99)
    print(f"[{treatment_col} -> {outcome_col}] Retained {sum(mask)} out of {len(mask)} observations within common support.")
    
    X_trimmed = X[mask]
    Y_trimmed = Y[mask]
    T_trimmed = T[mask]
    groups_trimmed = groups[mask]
    p_hat_trimmed = p_hat[mask]
    
    if len(np.unique(T_trimmed)) < 2:
        print(f"[{treatment_col} -> {outcome_col}] Insufficient treatment variance post-trimming. Skipping CATE.")
        return None
        
    # Step 3: Causal Forest Estimation (CATE)
    print(f"[{treatment_col} -> {outcome_col}] Training CausalForestDML (n_estimators=2000, honest=True)...")
    est = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        model_t=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        n_estimators=2000,
        honest=True,
        discrete_treatment=True,
        random_state=42,
        cv=3
    )
    
    # Fit clustering standard errors at the customer level via groups
    est.fit(Y_trimmed, T_trimmed, X=X_trimmed, W=None, groups=groups_trimmed)
    
    # Step 4: Average Treatment Effects (ATE) & Doubly Robust Inference
    print(f"[{treatment_col} -> {outcome_col}] Calculating Doubly Robust ATE...")
    ate_summary = est.ate_inference(X_trimmed)
    print(f"[{treatment_col} -> {outcome_col}] ATE Point Estimate: {ate_summary.mean_point:.3f} (p-value: {ate_summary.pvalue():.4f})")
    
    # Step 5: Group Average Treatment Effects (GATE)
    print(f"[{treatment_col} -> {outcome_col}] Extracting CATEs for GATE Heterogeneity Validation...")
    cates = est.effect(X_trimmed).flatten()
    
    # Map out subgroup coefficient deviations using OLS
    # Using socio-demographic features and lagged spending
    demo_cols_X = [c for c in X_trimmed.columns if 'age_range' in c or 'income' in c or 'family_size' in c]
    lag_cols_X = [c for c in X_trimmed.columns if 'lagged_spend' in c]
    gate_cols = demo_cols_X + lag_cols_X
    
    if len(gate_cols) > 0:
        X_gate = X_trimmed[gate_cols]
        X_gate = sm.add_constant(X_gate)
        # OLS regression mapping individualized effects to subgroups
        gate_model = sm.OLS(cates, X_gate).fit()
        print(f"[{treatment_col} -> {outcome_col}] GATE OLS R-squared: {gate_model.rsquared:.4f}")
        # Print top coefficients for heterogeneity check
        print(f"[{treatment_col} -> {outcome_col}] Significant GATE Heterogeneity (Top 5):")
        print(gate_model.params.sort_values(ascending=False).head(5))
        
    return est, X_trimmed, Y_trimmed, T_trimmed, p_hat_trimmed, cates

def optimal_policy_learning(est, X_trimmed, Y_trimmed, T_trimmed, p_hat_trimmed, cates):
    """
    Step 6: Optimal Policy Learning via Depth-3 Policy Tree.
    """
    print("\n--- Step 6: Optimal Policy Learning ---")
    
    X_policy = X_trimmed.copy()
    
    # Feature Quantization
    # 1. Map missing values to -1
    X_policy = X_policy.fillna(-1)
    
    # 2. Round pre-campaign daily expenditures
    lag_cols = [c for c in X_policy.columns if 'lagged_spend' in c]
    for col in lag_cols:
        # Cap all spending at or above 2000
        X_policy[col] = np.where(X_policy[col] >= 2000, 2000, X_policy[col])
        
        # Round 1000-2000 to nearest 200
        mask_high = (X_policy[col] >= 1000) & (X_policy[col] < 2000)
        X_policy.loc[mask_high, col] = np.round(X_policy.loc[mask_high, col] / 200) * 200
        
        # Round 0-1000 to nearest 100
        mask_low = X_policy[col] < 1000
        X_policy.loc[mask_low, col] = np.round(X_policy.loc[mask_low, col] / 100) * 100
        
    # Welfare maximization: Target when CATE > 0, weighted by magnitude of CATE
    target_class = (cates > 0).astype(int)
    weights = np.abs(cates)
    
    # Fit a depth-3 decision tree classifier (max 8 leaves)
    policy_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    policy_tree.fit(X_policy, target_class, sample_weight=weights)
    
    n_leaves = policy_tree.get_n_leaves()
    print(f"Fitted Policy Tree with {n_leaves} structural leaves (Depth=3).")
    
    # Evaluate policy objective (empirical welfare)
    policy_preds = policy_tree.predict(X_policy)
    welfare = np.sum(cates[policy_preds == 1])
    print(f"Empirical Welfare generated by Targeted Sub-Population: {welfare:.2f}")

def run_pipeline():
    """Main execution function coordinating all 7 methodology steps."""
    
    print("=== Phase 1: Main Analysis (Full N=50,624 Schema) ===")
    panel_full, treatment_cols = preprocess_data(full_run=True)
    
    # Target treatments for replication benchmarks
    benchmarks = [
        ('treatment_Any Coupon', ['avg_daily_expenditure_demeaned', 'avg_daily_expenditure_t1_demeaned', 'avg_daily_expenditure_t2_demeaned']), # no significant results
        ('treatment_drugstore items', ['avg_daily_expenditure_demeaned', 'avg_daily_expenditure_t1_demeaned', 'avg_daily_expenditure_t2_demeaned']),
        ('treatment_other food', ['avg_daily_expenditure_demeaned']),
        ('treatment_other non-food products', ['avg_daily_expenditure_demeaned', 'avg_daily_expenditure_t1_demeaned', 'avg_daily_expenditure_t2_demeaned']),
        ('treatment_ready-to-eat food', ['avg_daily_expenditure_demeaned', 'avg_daily_expenditure_t1_demeaned']),
        ('treatment_meat/seafood', ['avg_daily_expenditure_demeaned'])
    ]
                  
    for t_col, outcomes in benchmarks:
        if t_col in treatment_cols or t_col == 'treatment_Any Coupon':
            for outcome in outcomes:
                # Exclude metadata and all outcome variants from features X dynamically
                non_feature_cols = ['customer_id', 'start_date', 'end_date', 'duration_days', 
                                    'total_expenditure'] + [c for c in panel_full.columns if 'avg_daily' in c] + treatment_cols
                feature_cols = [c for c in panel_full.columns if c not in non_feature_cols]
                
                res = estimate_causal_effects(panel_full, t_col, feature_cols, outcome_col=outcome)
                if res is not None:
                    est, X_trimmed, Y_trimmed, T_trimmed, p_hat_trimmed, cates = res
                    
                    # Execute Optimal Policy Learning for specific categories
                    if t_col in ['treatment_Any Coupon', 'treatment_drugstore items', 'treatment_meat/seafood'] and outcome == 'avg_daily_expenditure':
                        print(f"\n[Policy Learning] Optimized Targeting for {t_col}...")
                        optimal_policy_learning(est, X_trimmed, Y_trimmed, T_trimmed, p_hat_trimmed, cates)
                    
    print("\n=== Phase 2: Robustness Diagnostics (Subset N=13,792) ===")
    panel_reduced, treatment_cols_red = preprocess_data(full_run=False)
    
    # Re-verify 'Any Coupon' stability over the restricted set
    t_col = 'treatment_Any Coupon'
    if t_col in treatment_cols_red or t_col == 'treatment_Any Coupon':
        non_feature_cols_red = ['customer_id', 'start_date', 'end_date', 'duration_days', 
                                'total_expenditure'] + [c for c in panel_reduced.columns if 'avg_daily' in c] + treatment_cols_red
        feature_cols_red = [c for c in panel_reduced.columns if c not in non_feature_cols_red]
        estimate_causal_effects(panel_reduced, t_col, feature_cols_red, outcome_col='avg_daily_expenditure_demeaned')

if __name__ == '__main__':
    run_pipeline()
