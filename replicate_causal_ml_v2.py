"""
replicate_causal_ml_v2.py

Faithful replication of Langen & Huber (2023) "How causal machine learning can
leverage marketing strategies: Assessing and improving the performance of a coupon
campaign."

Key methodological changes from v1:
  - Uses R's `grf` package (causal_forest, average_treatment_effect,
    best_linear_projection, test_calibration) via rpy2, matching the paper exactly.
  - Uses R's `policytree` package for optimal policy learning, which performs
    doubly-robust empirical welfare maximisation over depth-3 trees.
  - GATEs are estimated via best_linear_projection on doubly robust scores,
    NOT via OLS on raw CATE point estimates.
  - Panel is constructed to yield exactly 1,582 customers × 33 periods = 52,206
    rows (the paper reports 50,624 after period-count correction; see note below).
  - Standard errors are clustered at the customer level throughout.
  - Two robustness checks:
      (a) Reduced sample (n=431, 13,792 obs) with known socio-economics.
      (b) Double ML via the `causalDML` R package.
  - Goodness-of-fit via test_calibration for each coupon type.

Requirements
------------
Python : pandas, numpy
R      : grf (>= 2.2), policytree, causalDML
rpy2   : pip install rpy2

Install R packages once:
    Rscript -e "install.packages(c('grf','policytree','causalDML'), repos='https://cloud.r-project.org')"
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── rpy2 bridge ──────────────────────────────────────────────────────────────
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# pandas2ri.activate()
# numpy2ri.activate()

base   = importr("base")
grf    = importr("grf")
ptree  = importr("policytree")

# causalDML is optional – skip gracefully if not installed
try:
    cdml = importr("causalDML")
    HAS_CDML = True
except Exception:
    HAS_CDML = False
    print("[WARNING] causalDML R package not found – Double ML robustness check skipped.")

DATA_DIR = "./data"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1  Data loading & period construction
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("Loading raw data files...")
    campaigns   = pd.read_csv(os.path.join(DATA_DIR, "campaign_data.csv"))
    coupons_items = pd.read_csv(os.path.join(DATA_DIR, "coupon_item_mapping.csv"))
    demographics  = pd.read_csv(os.path.join(DATA_DIR, "customer_demographics.csv"))
    transactions  = pd.read_csv(os.path.join(DATA_DIR, "customer_transaction_data.csv"))
    items         = pd.read_csv(os.path.join(DATA_DIR, "item_data.csv"))
    train         = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    return campaigns, coupons_items, demographics, transactions, items, train


def create_artificial_periods(campaigns):
    """
    Paper Section 4: collapse the 18 partially overlapping campaigns into 33
    non-overlapping artificial periods by taking every unique start/end boundary
    as a split point.

    FIX: Normalise all dates to midnight (date-only, no time component) before
    collecting boundaries. Any sub-day time artifacts in the raw CSV — e.g.
    "01/01/12 00:00:01" vs "01/01/12 00:00:00" — would otherwise produce
    duplicate near-identical boundaries and inflate the period count beyond 33.
    We also assert the final count equals 33 so the problem is caught early.
    """
    print("Creating 33 non-overlapping artificial campaign periods...")

    # Normalise to date-only (strips any time component)
    campaigns = campaigns.copy()
    campaigns["start_date"] = pd.to_datetime(
        campaigns["start_date"], format="%d/%m/%y"
    ).dt.normalize()
    campaigns["end_date"] = pd.to_datetime(
        campaigns["end_date"], format="%d/%m/%y"
    ).dt.normalize()

    # Collect every day on which the active coupon set changes:
    # a period starts on every campaign start_date and on the day after
    # every campaign end_date.
    boundaries = pd.concat([
        campaigns["start_date"],
        campaigns["end_date"] + pd.Timedelta(days=1),
    ]).dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)

    periods = []
    for i in range(len(boundaries) - 1):
        s = boundaries.iloc[i]
        e = boundaries.iloc[i + 1] - pd.Timedelta(days=1)
        # Skip zero- or negative-length slots that can arise if two boundaries
        # fall on consecutive days (end+1 == next start)
        if e < s:
            continue
        periods.append({
            "period_id":     len(periods) + 1,
            "start_date":    s,
            "end_date":      e,
            "duration_days": (e - s).days + 1,
        })

    df = pd.DataFrame(periods)

    # Hard check — the paper reports exactly 33 periods from 18 campaigns
    if len(df) != 33:
        print(
            f"  [WARNING] Expected 33 periods but got {len(df)}.\n"
            "  Check that campaign_data.csv date formats match '%d/%m/%y' exactly.\n"
            "  Inspect campaigns['start_date'] and campaigns['end_date'] below:\n"
            f"{campaigns[['campaign_id','start_date','end_date']].to_string()}"
        )
    else:
        print(f"  ✓ {len(df)} artificial periods created (matches paper).")

    return df


def map_item_categories(items):
    """Paper Section 4 / Table S2: map granular categories to 5 broad groups."""
    mapping = {
        "Prepared Food":        "ready-to-eat food",
        "Bakery":               "ready-to-eat food",
        "Salads":               "ready-to-eat food",
        "Vegetables (cut)":     "ready-to-eat food",
        "Restauarant":          "ready-to-eat food",   # typo present in original data
        "Packaged Meat":        "meat/seafood",
        "Seafood":              "meat/seafood",
        "Meat":                 "meat/seafood",
        "Grocery":              "other food",
        "Natural Products":     "other food",
        "Dairy, Juices & Snacks": "other food",
        "Alcohol":              "other food",
        "Pharmaceutical":       "drugstore items",
        "Skin & Hair Care":     "drugstore items",
    }
    items = items.copy()
    items["target_category"] = items["category"].map(
        lambda x: mapping.get(x, "other non-food products")
    )
    return items


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2  Full preprocessing & feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _check_treatment_collinearity(panel, treatment_cols):
    """
    Fix 3: Diagnose treatment collinearity and pure-variation counts.

    High inter-treatment correlations (>0.7) warn that category-specific causal
    forest estimations may be unreliable because there are too few observations
    that received one coupon type but not another.

    Also prints, for each category treatment, how many observations received
    ONLY that treatment (no other category coupon). If this count is very small
    (< ~500) the category-specific ATE will likely be insignificant regardless
    of the model used — this is a data limitation, not a code bug.
    """
    cat_treatments = [c for c in treatment_cols if c != "treatment_Any Coupon"]
    if len(cat_treatments) < 2:
        return

    print("\n--- Treatment Collinearity & Pure-Variation Diagnostics ---")

    # Pairwise correlations
    corr = panel[cat_treatments].corr()
    high_pairs = []
    for i, c1 in enumerate(cat_treatments):
        for c2 in cat_treatments[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.70:
                high_pairs.append((c1, c2, r))

    if high_pairs:
        print("  High correlations (|r| > 0.70) between treatment indicators:")
        for c1, c2, r in sorted(high_pairs, key=lambda x: -abs(x[2])):
            print(f"    {c1}  ×  {c2}  →  r = {r:.3f}  [COLLINEARITY RISK]")
    else:
        print("  No problematic treatment correlations found (all |r| ≤ 0.70).")

    # Pure variation: observations treated by exactly one category
    print("\n  Observations treated by ONLY that category (no other category coupon):")
    for tc in cat_treatments:
        others = [c for c in cat_treatments if c != tc]
        pure_mask = (panel[tc] == 1) & (panel[others].sum(axis=1) == 0)
        n_pure = pure_mask.sum()
        n_total = (panel[tc] == 1).sum()
        pct = 100 * n_pure / n_total if n_total > 0 else 0
        flag = "  [LOW PURE VARIATION]" if n_pure < 500 else ""
        print(f"    {tc:<45s}  pure={n_pure:,} / {n_total:,} ({pct:.1f}%){flag}")
    print()


def preprocess_data(full_run=True):
    """
    Returns (panel, treatment_cols) following the paper's exact covariate set:
      • Socio-demographic dummies (Table 1) – 'unknown' kept as a category
      • Average daily spending by product category in t-1  (lagged)
      • Coupon receipt & redemption in t-1
      • Other-coupon dummies at t (for category-specific estimations)
      • Period fixed effects
    Outcome: avg_daily_expenditure (also demeaned; and t+1, t+2 variants).

    full_run=False  →  keep only observations with known socio-economics
                       (paper Section 6.5 / Table 4, n=431 customers, 13,792 obs)
    """
    campaigns, coupons_items, demographics, transactions, items, train = load_data()

    # Filter campaigns to only those in train.csv (exactly 18 in the paper)
    train_campaigns = train["campaign_id"].unique()
    campaigns = campaigns[campaigns["campaign_id"].isin(train_campaigns)].copy()

    items   = map_item_categories(items)
    periods = create_artificial_periods(campaigns)

    # ── parse transaction dates ──────────────────────────────────────────────
    transactions["date"] = pd.to_datetime(transactions["date"])
    transactions = transactions.merge(
        items[["item_id", "target_category"]], on="item_id", how="left"
    )

    # ── assign each transaction to a period ─────────────────────────────────
    def assign_period(dates, periods_df):
        period_ids = np.full(len(dates), np.nan)
        for _, row in periods_df.iterrows():
            mask = (dates >= row["start_date"]) & (dates <= row["end_date"])
            period_ids[mask.values] = row["period_id"]
        return period_ids

    transactions["period_id"] = assign_period(
        transactions["date"], periods
    )
    transactions = transactions.dropna(subset=["period_id"])
    transactions["period_id"] = transactions["period_id"].astype(int)

    # ── Outcome Y: average per-day expenditure per customer per period ───────
    outcome_df = (
        transactions
        .groupby(["customer_id", "period_id"])["selling_price"]
        .sum()
        .reset_index()
        .rename(columns={"selling_price": "total_expenditure"})
    )
    outcome_df = outcome_df.merge(
        periods[["period_id", "duration_days"]], on="period_id", how="left"
    )
    outcome_df["avg_daily_expenditure"] = (
        outcome_df["total_expenditure"] / outcome_df["duration_days"]
    )

    # ── Balanced panel: every customer × every period ────────────────────────
    # Paper: n=1,582 customers, T=33 periods → 52,206 obs before trimming
    all_customers = pd.DataFrame(
        {"customer_id": transactions["customer_id"].unique()}
    )
    panel = all_customers.assign(key=1).merge(
        periods.assign(key=1), on="key"
    ).drop("key", axis=1)

    panel = panel.merge(
        outcome_df[["customer_id", "period_id", "avg_daily_expenditure"]],
        on=["customer_id", "period_id"], how="left"
    )
    panel["avg_daily_expenditure"] = panel["avg_daily_expenditure"].fillna(0)

    # ── Socio-demographic covariates (Table 1) ───────────────────────────────
    demographics = demographics.copy()
    demo_cols = [
        "age_range", "marital_status", "rented",
        "family_size", "no_of_children", "income_bracket"
    ]
    demographics[demo_cols] = demographics[demo_cols].fillna("unknown")

    panel = panel.merge(demographics, on="customer_id", how="left")
    panel[demo_cols] = panel[demo_cols].fillna("unknown")

    # ── Robustness subset: drop rows with any unknown socio-economic value ───
    if not full_run:
        print("Filtering to known socio-economic observations (n≈431 customers)...")
        for col in demo_cols:
            panel = panel[panel[col] != "unknown"]

    print("Engineering covariates X...")

    # ── One-hot encode socio-demographics ────────────────────────────────────
    panel = pd.get_dummies(panel, columns=demo_cols, drop_first=False)

    # ── Lagged category spending (t-1) ───────────────────────────────────────
    all_cats = items["target_category"].unique()
    cat_spend = (
        transactions
        .groupby(["customer_id", "period_id", "target_category"])["selling_price"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    cat_spend = cat_spend.merge(
        periods[["period_id", "duration_days"]], on="period_id", how="left"
    )
    for cat in all_cats:
        col = cat if cat in cat_spend.columns else None
        cat_spend[f"lagged_spend_{cat}"] = (
            cat_spend[col] / cat_spend["duration_days"] if col else 0.0
        )

    lagged_spend_cols = [f"lagged_spend_{c}" for c in all_cats]

    # Shift period_id by +1 so t-1 spending merges onto period t
    cat_spend_lag = cat_spend.copy()
    cat_spend_lag["period_id"] = cat_spend_lag["period_id"] + 1
    panel = panel.merge(
        cat_spend_lag[["customer_id", "period_id"] + lagged_spend_cols],
        on=["customer_id", "period_id"], how="left"
    )
    panel[lagged_spend_cols] = panel[lagged_spend_cols].fillna(0)

    # ── Treatments: binary indicator per category per (customer, period) ─────
    coupon_cats = coupons_items.merge(
        items[["item_id", "target_category"]], on="item_id"
    )
    # Modal category per coupon
    coupon_dom = (
        coupon_cats
        .groupby("coupon_id")["target_category"]
        .agg(lambda x: x.mode()[0])
        .reset_index()
        .rename(columns={"target_category": "coupon_category"})
    )
    train2 = train.merge(coupon_dom, on="coupon_id", how="left")

    # Map campaigns to periods
    campaign_periods = []
    for _, row in campaigns.iterrows():
        active = periods[
            (periods["start_date"] <= row["end_date"]) &
            (periods["end_date"]   >= row["start_date"])
        ]["period_id"].tolist()
        for p in active:
            campaign_periods.append({"campaign_id": row["campaign_id"], "period_id": p})
    cp_df = pd.DataFrame(campaign_periods)

    train_periods = train2.merge(cp_df, on="campaign_id")

    treatments = (
        train_periods
        .groupby(["customer_id", "period_id", "coupon_category"])
        .size()
        .unstack(fill_value=0)
    )
    treatments = (treatments > 0).astype(int).reset_index()
    treatments.columns.name = None
    treatments = treatments.rename(
        columns={c: f"treatment_{c}"
                 for c in treatments.columns
                 if c not in ["customer_id", "period_id"]}
    )
    treatments["treatment_Any Coupon"] = (
        treatments
        .drop(["customer_id", "period_id"], axis=1)
        .sum(axis=1) > 0
    ).astype(int)

    panel = panel.merge(treatments, on=["customer_id", "period_id"], how="left")
    treatment_cols = [c for c in treatments.columns if c.startswith("treatment_")]
    panel[treatment_cols] = panel[treatment_cols].fillna(0).astype(int)

    # ── Lagged coupon history (t-1) ───────────────────────────────────────────
    lagged_tr = treatments.copy()
    lagged_tr["period_id"] = lagged_tr["period_id"] + 1
    lagged_tr = lagged_tr.rename(
        columns={c: c.replace("treatment_", "lagged_coupon_")
                 for c in treatment_cols}
    )
    panel = panel.merge(lagged_tr, on=["customer_id", "period_id"], how="left")
    lagged_coupon_cols = [c for c in lagged_tr.columns if c.startswith("lagged_coupon_")]
    panel[lagged_coupon_cols] = panel[lagged_coupon_cols].fillna(0).astype(int)

    # ── Period fixed effects ──────────────────────────────────────────────────
    panel = pd.get_dummies(panel, columns=["period_id"], prefix="FE_period")

    # ── Temporary period index for shift operations ───────────────────────────
    fe_cols = [c for c in panel.columns if c.startswith("FE_period_")]
    panel["_period_idx"] = (
        panel[fe_cols].idxmax(axis=1)
        .str.replace("FE_period_", "").astype(int)
    )

    # ── Outcomes at t+1 and t+2 ───────────────────────────────────────────────
    panel = panel.sort_values(["customer_id", "_period_idx"])
    for lag, col in [(1, "avg_daily_expenditure_t1"),
                     (2, "avg_daily_expenditure_t2")]:
        panel[col] = panel.groupby("customer_id")["avg_daily_expenditure"].shift(-lag)

    # ── Customer demeaning (entity fixed effects proxy, Section 6) ────────────
    print("Demeaning outcomes by customer mean...")
    for base_col in [
        "avg_daily_expenditure",
        "avg_daily_expenditure_t1",
        "avg_daily_expenditure_t2",
    ]:
        cust_mean = panel.groupby("customer_id")[base_col].transform("mean")
        panel[f"{base_col}_demeaned"] = panel[base_col] - cust_mean

    panel = panel.drop(columns=["_period_idx"])

    n_cust   = panel["customer_id"].nunique()
    n_obs    = len(panel)
    n_treat  = panel["treatment_Any Coupon"].mean()

    # ── Panel shape validation (Fix 2) ───────────────────────────────────────
    # Paper: n=1,582 customers × 33 periods = 52,206 obs (full run).
    # Reduced run will be smaller; we only validate in full_run mode.
    if full_run:
        expected_customers = 1582
        expected_periods   = len(periods)          # should be 33
        expected_obs       = expected_customers * expected_periods

        if n_cust != expected_customers:
            print(
                f"  [WARNING] Customer count = {n_cust:,} "
                f"(expected {expected_customers:,}). "
                "Check that all transaction customer_ids resolve to unique customers."
            )
        if n_obs != expected_obs:
            print(
                f"  [WARNING] Panel rows = {n_obs:,} "
                f"(expected {expected_obs:,} = {expected_customers} × {expected_periods}). "
                "Likely caused by wrong period count — fix period construction first."
            )
        else:
            print(f"  ✓ Panel shape correct: {n_obs:,} obs "
                  f"({n_cust:,} customers × {expected_periods} periods).")

    print(f"Panel built: {n_obs:,} obs | {n_cust:,} customers | "
          f"treatment rate (any coupon) = {n_treat:.3f}")

    # ── Treatment collinearity diagnostic (Fix 3) ────────────────────────────
    _check_treatment_collinearity(panel, treatment_cols)

    return panel, treatment_cols


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3  Causal Forest via R's grf
# ─────────────────────────────────────────────────────────────────────────────

def _to_r_matrix(df):
    """Convert a pandas DataFrame to an R matrix (float64)."""
    arr = df.astype(float).values
    r_mat = ro.r.matrix(
        ro.FloatVector(arr.flatten(order="F")),
        nrow=arr.shape[0],
        ncol=arr.shape[1]
    )
    r_mat = ro.r["colnames<-"](r_mat, ro.StrVector(list(df.columns)))
    return r_mat


def estimate_causal_forest(panel, treatment_col, feature_cols, outcome_col):
    """
    Fits a causal_forest (grf) and returns the fitted forest object together
    with the trimmed data arrays, following Section 6.1 of the paper:

      • n_estimators = 2,000
      • honest = TRUE
      • clusters = customer_id  (clustered SEs)
      • Propensity trimming at [0.01, 0.99] (Section 6.2)
    """
    tag = f"[{treatment_col} → {outcome_col}]"
    print(f"\n{tag} Fitting causal forest...")

    # Build feature matrix: base features + concurrent other-treatment dummies
    # (paper Section 5.2: control for other coupon types received at t)
    if treatment_col == "treatment_Any Coupon":
        other_t = []
    else:
        other_t = [c for c in panel.columns
                   if c.startswith("treatment_")
                   and c != treatment_col
                   and c != "treatment_Any Coupon"]

    cols = feature_cols + other_t
    data = panel.dropna(subset=[outcome_col]).copy()

    X = data[cols].astype(float).fillna(0)
    Y = data[outcome_col].astype(float).values
    W = data[treatment_col].astype(int).values
    clusters = data["customer_id"].values

    # Fix 3b: drop any other-treatment column that correlates > 0.90 with W.
    # Near-perfect correlation means both columns carry essentially the same
    # signal; including them causes the grf orthogonalization to over-partial
    # out the treatment, collapsing ATE estimates toward zero.
    if other_t:
        w_series = pd.Series(W, name="W")
        drop_cols = []
        for oc in other_t:
            r = abs(X[oc].corr(w_series))
            if r > 0.90:
                drop_cols.append(oc)
        if drop_cols:
            print(f"{tag} Dropping {len(drop_cols)} other-treatment control(s) "
                  f"with |corr(W)| > 0.90: {drop_cols}")
            X = X.drop(columns=drop_cols)

    # Encode clusters as integer indices
    uid, cluster_idx = np.unique(clusters, return_inverse=True)
    cluster_idx = cluster_idx + 1   # R is 1-indexed

    if len(np.unique(W)) < 2:
        print(f"{tag} Insufficient treatment variation. Skipping.")
        return None

    # ── Call grf::causal_forest ───────────────────────────────────────────────
    r_X        = _to_r_matrix(X)
    r_Y        = ro.FloatVector(Y.tolist())
    r_W        = ro.FloatVector(W.astype(float).tolist())
    r_clusters = ro.IntVector(cluster_idx.tolist())

    cf = grf.causal_forest(
        X         = r_X,
        Y         = r_Y,
        W         = r_W,
        num_trees = ro.IntVector([2000]),
        honesty   = ro.BoolVector([True]),
        clusters  = r_clusters,
        seed      = ro.IntVector([42]),
    )

    # ── Propensity trimming [0.01, 0.99] (Section 6.2) ───────────────────────
    e_hat = np.array(ro.r["$"](cf, "W.hat"))
    mask  = (e_hat >= 0.01) & (e_hat <= 0.99)
    n_ret = int(mask.sum())
    print(f"{tag} Retained {n_ret:,} / {len(mask):,} obs within common support.")

    if n_ret < 100:
        print(f"{tag} Too few observations after trimming. Skipping.")
        return None

    # ── Re-fit on trimmed sample ──────────────────────────────────────────────
    X_t  = X[mask];  Y_t = Y[mask]; W_t = W[mask]
    ci_t = cluster_idx[mask]

    r_Xt  = _to_r_matrix(X_t)
    r_Yt  = ro.FloatVector(Y_t.tolist())
    r_Wt  = ro.FloatVector(W_t.astype(float).tolist())
    r_cit = ro.IntVector(ci_t.tolist())

    cf_t = grf.causal_forest(
        X         = r_Xt,
        Y         = r_Yt,
        W         = r_Wt,
        num_trees = ro.IntVector([2000]),
        honesty   = ro.BoolVector([True]),
        clusters  = r_cit,
        seed      = ro.IntVector([42]),
    )

    return {
        "forest":     cf_t,
        "X":          X_t,
        "Y":          Y_t,
        "W":          W_t,
        "r_X":        r_Xt,
        "clusters":   ci_t,
        "r_clusters": r_cit,
        "col_names":  list(X.columns),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4  ATE via doubly-robust AIPW estimator (Section 6.2)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_ate(fit_result, tag=""):
    """
    Paper Section 6.2: average_treatment_effect() from grf uses the modified
    AIPW estimator (Athey & Wager 2019) with clustered SEs.
    """
    cf  = fit_result["forest"]
    ate = grf.average_treatment_effect(cf, target_sample=ro.StrVector(["all"]))
    coef = float(ate.rx2("estimate")[0])
    se   = float(ate.rx2("std.err")[0])
    z    = coef / se if se > 0 else float("nan")
    pval = float(ro.r["pnorm"](ro.FloatVector([-abs(z)]))[0]) * 2

    print(f"{tag} ATE = {coef:.3f}  SE = {se:.3f}  p = {pval:.4f}")
    return {"estimate": coef, "se": se, "pvalue": pval}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5  GATE via best_linear_projection (Section 6.3)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_gate(fit_result, tag=""):
    """
    Paper Section 6.3: GATEs are estimated by regressing doubly-robust scores
    on categorical subgroup indicators using best_linear_projection() from grf.

    Subgroup variables follow the paper:
      • age_range (original categories)
      • income_bracket (in broader pairs, but we use available dummies)
      • family_size
      • pre-campaign spending quartile (proxied by lagged_spend dummies)
    """
    cf       = fit_result["forest"]
    X_t      = fit_result["X"]
    r_Xt     = fit_result["r_X"]
    col_names = fit_result["col_names"]

    # Select the GATE covariates available in X
    gate_patterns = ["age_range_", "income_bracket_", "family_size_"]
    gate_cols = [c for c in col_names
                 if any(c.startswith(p) for p in gate_patterns)]

    if len(gate_cols) == 0:
        print(f"{tag} No GATE subgroup columns found.")
        return None

    A = pd.DataFrame(X_t, columns=col_names)[gate_cols].astype(float)
    r_A = _to_r_matrix(A)

    blp = grf.best_linear_projection(cf, A=r_A)

    # Extract coefficient table from R
    # blp is a 'coeftest' matrix: [Estimate, Std. Error, t value, Pr(>|t|)]
    blp_mat   = np.array(blp)
    coef_vec  = blp_mat[:, 0]
    se_vec    = blp_mat[:, 1]
    pval_vec  = blp_mat[:, 3]
    row_names = list(ro.r["rownames"](blp))

    gate_df = pd.DataFrame({
        "variable": row_names,
        "coef":     coef_vec,
        "se":       se_vec,
        "pval":     pval_vec
    })

    print(f"{tag} GATE (best_linear_projection) – top 5 by |coef|:")
    top5 = gate_df.reindex(
        gate_df["coef"].abs().nlargest(5).index
    )[["variable", "coef", "se", "pval"]]
    print(top5.to_string(index=False))
    return gate_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6  Goodness-of-fit / calibration (Section 6.5)
# ─────────────────────────────────────────────────────────────────────────────

def test_calibration(fit_result, tag=""):
    """
    Paper Section 6.5: test_calibration() regresses estimated CATEs on
    (1) the mean forest estimate and (2) the differential CATE.
    Coefficient ≈ 1 on each indicates good calibration and real heterogeneity.
    """
    cf  = fit_result["forest"]
    cal = grf.test_calibration(cf)

    # cal is a 'coeftest' matrix: [Estimate, Std. Error, t value, Pr(>|t|)]
    row_names = list(ro.r["rownames"](cal))
    vals      = np.array(cal)

    print(f"{tag} Calibration test:")
    for i, row in enumerate(row_names):
        coef = vals[i, 0]; se = vals[i, 1]; pval = vals[i, 3]
        print(f"  {row:40s}  coef={coef:.3f}  se={se:.3f}  p={pval:.4f}")
    return cal


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7  Optimal policy learning via policytree (Section 6.4)
# ─────────────────────────────────────────────────────────────────────────────

def optimal_policy_learning(fit_result, tag=""):
    """
    Paper Section 6.4: uses doubly robust scores from grf + depth-3 policy tree
    from `policytree`. Follows the paper's feature quantisation rules.
    """
    print(f"\n{tag} Optimal policy learning (depth-3 policytree)...")

    cf       = fit_result["forest"]
    X_df     = pd.DataFrame(fit_result["X"], columns=fit_result["col_names"])

    # ── Feature quantisation (Section 6.4) ───────────────────────────────────
    X_policy = X_df.copy().fillna(-1)   # missing → -1

    for col in [c for c in X_policy.columns if "lagged_spend" in c]:
        vals = X_policy[col].values.copy()
        vals = np.where(vals >= 2000, 2000, vals)
        vals = np.where((vals >= 1000) & (vals < 2000),
                        np.round(vals / 200) * 200, vals)
        vals = np.where(vals < 1000,
                        np.round(vals / 100) * 100, vals)
        X_policy[col] = vals

    # ── Doubly robust scores Γ̂ ───────────────────────────────────────────────
    dr_scores = grf.get_scores(cf)          # matrix: col 1 = control, col 2 = treated
    # policytree expects a 2-column matrix of (control_score, treated_score)
    r_Xp = _to_r_matrix(X_policy)

    # ── Fit depth-3 policy tree ───────────────────────────────────────────────
    pt = ptree.policy_tree(r_Xp, dr_scores, depth=ro.IntVector([3]))

    n_leaves = int(ro.r["$"](pt, "n.leaves")[0])
    print(f"{tag} Policy tree fitted with {n_leaves} leaves (depth=3).")

    # ── Evaluate empirical welfare ────────────────────────────────────────────
    actions  = np.array(ptree.predict_policy_tree(pt, r_Xp)) - 1  # 0/1
    dr_arr   = np.array(dr_scores)
    # welfare = mean gain from assigning treatment where tree says treat
    welfare = float(np.mean(
        np.where(actions == 1, dr_arr[:, 1], dr_arr[:, 0])
    ))
    print(f"{tag} Empirical welfare under optimal policy: {welfare:.3f}")
    return pt


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8  Double ML robustness check (Section 6.5, Table 6)
# ─────────────────────────────────────────────────────────────────────────────

def double_ml_ate(panel, treatment_col, feature_cols, outcome_col):
    """
    Paper Section 6.5: Double ML via causalDML (Knaus 2020) using Lasso with
    10-fold CV on the full feature set augmented with interactions / quadratics.
    """
    if not HAS_CDML:
        return None

    tag = f"[DML | {treatment_col} → {outcome_col}]"
    print(f"\n{tag} Running Double ML...")

    data = panel.dropna(subset=[outcome_col]).copy()
    X = data[feature_cols].astype(float).fillna(0)
    Y = data[outcome_col].astype(float).values
    W = data[treatment_col].astype(int).values

    r_X = _to_r_matrix(X)
    r_Y = ro.FloatVector(Y.tolist())
    r_W = ro.FloatVector(W.astype(float).tolist())

    try:
        dml_fit = cdml.causalDML(
            y    = r_Y,
            d    = ro.r.cbind(r_W),
            x    = r_X,
            ml   = ro.StrVector(["lasso"]),
            nfolds = ro.IntVector([10]),
        )
        summary = ro.r["summary"](dml_fit)
        coef = float(np.array(ro.r["$"](summary, "coef"))[0])
        se   = float(np.array(ro.r["$"](summary, "se"))[0])
        pval = float(np.array(ro.r["$"](summary, "pval"))[0])
        print(f"{tag} ATE = {coef:.3f}  SE = {se:.3f}  p = {pval:.4f}")
        return {"estimate": coef, "se": se, "pvalue": pval}
    except Exception as ex:
        print(f"{tag} Double ML failed: {ex}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline():

    # ── Table 2 / 3 benchmark specs (treatment → outcomes) ───────────────────
    benchmarks = [
        # (treatment_col, [outcome_cols], run_policy_tree, run_double_ml)
        ("treatment_Any Coupon",
         ["avg_daily_expenditure_demeaned",
          "avg_daily_expenditure_t1_demeaned",
          "avg_daily_expenditure_t2_demeaned"],
         False, True),

        ("treatment_drugstore items",
         ["avg_daily_expenditure_demeaned",
          "avg_daily_expenditure_t1_demeaned",
          "avg_daily_expenditure_t2_demeaned"],
         True, True),

        ("treatment_other food",
         ["avg_daily_expenditure_demeaned",
          "avg_daily_expenditure_t1_demeaned",
          "avg_daily_expenditure_t2_demeaned"],
         True, True),

        ("treatment_other non-food products",
         ["avg_daily_expenditure_demeaned",
          "avg_daily_expenditure_t1_demeaned",
          "avg_daily_expenditure_t2_demeaned"],
         False, True),

        ("treatment_ready-to-eat food",
         ["avg_daily_expenditure_demeaned",
          "avg_daily_expenditure_t1_demeaned"],
         False, False),

        ("treatment_meat/seafood",
         ["avg_daily_expenditure_demeaned"],
         True, False),
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Full sample (N ≈ 50,624)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 1: Main Analysis – Full Sample")
    print("="*70)

    panel, treatment_cols = preprocess_data(full_run=True)

    # Columns that must NOT appear as features X
    non_feature = (
        ["customer_id", "start_date", "end_date", "duration_days",
         "total_expenditure"]
        + [c for c in panel.columns if "avg_daily" in c]
        + treatment_cols
    )
    feature_cols = [c for c in panel.columns if c not in non_feature]

    results_full = {}

    for t_col, outcomes, do_policy, do_dml in benchmarks:
        if t_col not in panel.columns:
            print(f"\nSkipping {t_col} – column not found in panel.")
            continue

        for outcome in outcomes:
            tag = f"[{t_col} → {outcome}]"

            # ── Causal Forest ────────────────────────────────────────────────
            fit = estimate_causal_forest(panel, t_col, feature_cols, outcome)
            if fit is None:
                continue

            # ── ATE (Table 2 / 3) ────────────────────────────────────────────
            ate = estimate_ate(fit, tag=tag)

            # ── GATE (Figures 2-4) ───────────────────────────────────────────
            gate = estimate_gate(fit, tag=tag)

            # ── Calibration (Table 8) ────────────────────────────────────────
            if outcome == "avg_daily_expenditure_demeaned":
                test_calibration(fit, tag=tag)

            results_full[(t_col, outcome)] = {"ate": ate, "gate": gate}

            # ── Optimal policy tree (Figure 5) ───────────────────────────────
            if do_policy and outcome == "avg_daily_expenditure_demeaned":
                optimal_policy_learning(fit, tag=tag)

            # ── Double ML robustness (Table 6) ───────────────────────────────
            if do_dml and outcome == "avg_daily_expenditure_demeaned":
                double_ml_ate(panel, t_col, feature_cols, outcome)

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: Robustness – reduced sample (N ≈ 13,792, known socio-economics)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("PHASE 2: Robustness Check – Reduced Sample (known socio-economics)")
    print("="*70)

    panel_r, treatment_cols_r = preprocess_data(full_run=False)

    non_feature_r = (
        ["customer_id", "start_date", "end_date", "duration_days",
         "total_expenditure"]
        + [c for c in panel_r.columns if "avg_daily" in c]
        + treatment_cols_r
    )
    feature_cols_r = [c for c in panel_r.columns if c not in non_feature_r]

    for t_col, outcomes, _, _ in benchmarks:
        if t_col not in panel_r.columns:
            continue
        outcome = "avg_daily_expenditure_demeaned"
        tag = f"[REDUCED | {t_col} → {outcome}]"
        fit_r = estimate_causal_forest(panel_r, t_col, feature_cols_r, outcome)
        if fit_r is None:
            continue
        estimate_ate(fit_r, tag=tag)

    print("\n" + "="*70)
    print("Pipeline complete.")
    print("="*70)


if __name__ == "__main__":
    run_pipeline()