import re

with open('replicate_causal_ml_v2.py', 'r') as f:
    code = f.read()

# Fix the end of file
code = re.sub(r'if __name__ == "__main__":\n\s*run_pipeline\(\)\n.*', 'if __name__ == "__main__":\n    run_pipeline()\n', code, flags=re.DOTALL)

# Fix double_ml_ate
old_double_ml = """
    data = panel.dropna(subset=[outcome_col]).copy()
    X = data[feature_cols].astype(float).fillna(0)
    Y = data[outcome_col].astype(float).values
    W = data[treatment_col].astype(int).values

    r_X = _to_r_matrix(X)
    r_Y = ro.FloatVector(Y.tolist())
    r_W = ro.FloatVector(W.astype(float).tolist())
"""

new_double_ml = """
    data = panel.dropna(subset=[outcome_col]).copy()
    X_base = data[feature_cols].astype(float).fillna(0)
    
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_base)
    
    Y = data[outcome_col].astype(float).values
    W = data[treatment_col].astype(int).values

    # Convert X_poly to pandas DataFrame so _to_r_matrix can name columns
    import pandas as pd
    X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(X_base.columns))
    
    r_X = _to_r_matrix(X_poly_df)
    r_Y = ro.FloatVector(Y.tolist())
    r_W = ro.FloatVector(W.astype(float).tolist())
"""

code = code.replace(old_double_ml, new_double_ml)

with open('replicate_causal_ml_v2.py', 'w') as f:
    f.write(code)

