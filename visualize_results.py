import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory
os.makedirs("plots", exist_ok=True)
THINK_ALOUD_DIR = "plots/think_aloud"
os.makedirs(THINK_ALOUD_DIR, exist_ok=True)

# Configuration
LABEL_MAP = {
    "treatment_Any Coupon": "All Coupons",
    "treatment_ready-to-eat food": "Ready-to-Eat Food",
    "treatment_meat/seafood": "Meat and Seafood",
    "treatment_other food": "Other Food",
    "treatment_drugstore items": "Drugstore Items",
    "treatment_other non-food products": "Other Non-Food Items"
}

ORDER = [
    "treatment_Any Coupon",
    "treatment_ready-to-eat food",
    "treatment_meat/seafood",
    "treatment_drugstore items",
    "treatment_other food",
    "treatment_other non-food products"
]

def plot_fig1_cate_distributions():
    """Replicates Figure 1: Distribution of CATE by coupon type."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path):
        print(f"Skipping Fig 1: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # We only plot short-term expenditure
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))
    axes = axes.flatten()

    for i, t_col in enumerate(ORDER):
        if t_col not in df["treatment"].unique():
            axes[i].set_visible(False)
            continue
            
        data = df[df["treatment"] == t_col]["cate"]
        sns.histplot(data, bins=50, ax=axes[i], kde=True, color="skyblue")
        
        label = LABEL_MAP.get(t_col, t_col)
        axes[i].set_title(f"Coupons for {label}")
        axes[i].set_xlabel("Estimated CATE (monetary units)")
        axes[i].set_ylabel("Frequency")
        axes[i].axvline(0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("plots/fig1_cate_distributions.png")
    print("✓ Saved plots/fig1_cate_distributions.png")

def plot_gate_subgroups():
    """Replicates Figures 2-4: GATE estimates with 95% CI."""
    file_path = "results/gate_results.csv"
    if not os.path.exists(file_path):
        print(f"Skipping GATE plots: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    # Filter to Phase 1 and short-term outcome
    df = df[(df["phase"] == "Phase 1 - Full") & (df["outcome"] == "avg_daily_expenditure")]

    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col].copy()
        if data.empty:
            continue

        # Map variables to categories for grouping
        def categorize(var):
            if "age_range" in var: return "Age"
            if "income_bracket" in var: return "Income"
            if "family_size" in var: return "Family Size"
            return "Other"
        
        data["category"] = data["variable"].apply(categorize)
        
        # Plot each category
        for cat in ["Age", "Income", "Family Size"]:
            cat_data = data[data["category"] == cat]
            if cat_data.empty: continue
            
            plt.figure(figsize=(10, 6))
            cat_data = cat_data.copy()
            cat_data["ci"] = 1.96 * cat_data["se"]
            
            plt.errorbar(x=cat_data["variable"], y=cat_data["coef"], yerr=cat_data["ci"], 
                         fmt='o', color='black', ecolor='gray', capsize=5)
            
            plt.axhline(0, color='red', linestyle='--')
            plt.xticks(rotation=45, ha='right')
            plt.title(f"GATE: {LABEL_MAP.get(t_col, t_col)} - {cat}")
            plt.ylabel("Estimate (monetary units)")
            plt.grid(axis='y', linestyle=':', alpha=0.7)
            
            safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
            plt.tight_layout()
            plt.savefig(f"plots/gate_{safe_name}_{cat.lower()}.png")
            plt.close()
        
        print(f"✓ Saved GATE plots for {t_col}")

def plot_qini_curve():
    """Think Aloud #1: Cumulative Gain (Qini) Curve."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col].copy()
        # Sort by CATE descending
        data = data.sort_values("cate", ascending=False).reset_index(drop=True)
        
        # Calculate cumulative gain (sum of CATE)
        data['cumulative_gain'] = data['cate'].cumsum()
        
        # Calculate random baseline
        total_gain = data['cate'].sum()
        data['random_baseline'] = (data.index + 1) * (total_gain / len(data))
        
        plt.figure(figsize=(8, 6))
        plt.plot(data.index / len(data) * 100, data['cumulative_gain'], label="Model (CATE-based)", color="blue")
        plt.plot(data.index / len(data) * 100, data['random_baseline'], label="Random Targeting", linestyle="--", color="gray")
        
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Cumulative Gain Curve: {label}")
        plt.xlabel("% of Customers Targeted")
        plt.ylabel("Cumulative Expected Incremental Expenditure")
        plt.legend()
        plt.grid(linestyle=":", alpha=0.6)
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"{THINK_ALOUD_DIR}/qini_{safe_name}.png")
        plt.close()
        print(f"✓ Saved Qini curve for {t_col}")

def plot_waterfall_uncertainty():
    """Think Aloud #2: Ranked Waterfall Plot with Individual Uncertainty."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col].copy()
        
        # Sort by CATE and sample 100 representative customers
        data = data.sort_values("cate").reset_index(drop=True)
        indices = np.linspace(0, len(data) - 1, 100).astype(int)
        sample = data.iloc[indices].reset_index(drop=True)
        
        plt.figure(figsize=(12, 6))
        plt.bar(sample.index, sample['cate'], color='skyblue', alpha=0.6, label="Point Estimate")
        plt.errorbar(sample.index, sample['cate'], yerr=1.96 * sample['std_err'], 
                     fmt='none', ecolor='black', alpha=0.4, capsize=2, label="95% CI")
        
        plt.axhline(0, color="red", linestyle="--")
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Individual CATE Estimates with 95% CI (100 Repr. Customers)\n{label}")
        plt.xlabel("Customer Percentile (Ranked by CATE)")
        plt.ylabel("Estimated CATE (monetary units)")
        plt.xticks([])
        plt.legend()
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"{THINK_ALOUD_DIR}/waterfall_{safe_name}.png")
        plt.close()
        print(f"✓ Saved Waterfall plot for {t_col}")

def plot_personas():
    """Think Aloud #3: Pre-defined Persona Contrasts."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    # We focus on "All Coupons" for simplicity in persona demonstration
    t_col = "treatment_Any Coupon"
    if t_col not in df["treatment"].unique(): return
    data = df[df["treatment"] == t_col].copy()
    
    # Define Profiles
    profiles = [
        {"name": "Affluent & Large Family", "query": "income_bracket == '11.0' and family_size == '5+'"},
        {"name": "Budget Single", "query": "income_bracket == '1.0' and family_size == '1'"},
        {"name": "Elderly / Late Adopters", "query": "age_range == '70+'"}
    ]
    
    persona_results = []
    for p in profiles:
        subset = data.query(p["query"])
        if not subset.empty:
            persona_results.append({
                "name": p["name"],
                "cate": subset["cate"].mean(),
                "se": subset["std_err"].mean(),
                "count": len(subset)
            })
            
    if not persona_results: return
    
    fig, axes = plt.subplots(1, len(persona_results), figsize=(15, 6))
    if len(persona_results) == 1: axes = [axes]
    
    for i, res in enumerate(persona_results):
        ax = axes[i]
        desc = (
            f"Expected CATE: ${res['cate']:.2f}\n"
            f"Avg SE: ${res['se']:.2f}\n"
            f"Sample Size: {res['count']}\n"
            f"-------------------\n"
            f"Recommendation:\n"
            f"{'Target' if res['cate'] > 0 else 'Exclude'}"
        )
        ax.text(0.5, 0.5, desc, ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle="round,pad=1", facecolor="honeydew" if res['cate'] > 0 else "mistyrose", edgecolor="gray"))
        ax.set_title(res["name"], fontweight="bold")
        ax.axis('off')
        
    plt.suptitle("Strategic Persona Contrasts (Any Coupon)", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{THINK_ALOUD_DIR}/personas_Any_Coupon.png")
    plt.close()
    print(f"✓ Saved Persona plot")

def plot_profitability_threshold():
    """Think Aloud #4: Profitability Threshold Matrix (Dynamic Cost)."""
    dist_file = "results/cate_distributions.csv"
    cost_file = "results/dynamic_cost.txt"
    if not os.path.exists(dist_file) or not os.path.exists(cost_file): return
    
    df = pd.read_csv(dist_file)
    with open(cost_file, "r") as f:
        cost_threshold = float(f.read())
        
    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col].copy()
        
        # Aggregate by subgroup
        agg_df = data.groupby(['age_range', 'income_bracket']).agg(
            mean_cate=('cate', 'mean'),
            mean_se=('std_err', 'mean')
        ).reset_index()
        
        def get_action(row):
            ci_lower = row['mean_cate'] - 1.96 * row['mean_se']
            ci_upper = row['mean_cate'] + 1.96 * row['mean_se']
            if ci_lower > cost_threshold: return 1 # Green: Target
            if ci_upper < cost_threshold: return -1 # Red: Exclude
            return 0 # Gray: Uncertain
                
        agg_df['action'] = agg_df.apply(get_action, axis=1)
        pivot_df = agg_df.pivot(index="age_range", columns="income_bracket", values="action")
        
        plt.figure(figsize=(12, 8))
        cmap = sns.color_palette(["#ff9999", "#d3d3d3", "#99ff99"])
        sns.heatmap(pivot_df, annot=False, cmap=cmap, cbar=False, linewidths=.5)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#99ff99', label='Target (CATE > Cost)'),
            Patch(facecolor='#d3d3d3', label='Uncertain'),
            Patch(facecolor='#ff9999', label='Exclude (CATE < Cost)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Profitability Action Matrix (Cost = ${cost_threshold:.2f})\n{label}")
        plt.xlabel("Income Bracket")
        plt.ylabel("Age Range")
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"{THINK_ALOUD_DIR}/profitability_{safe_name}.png")
        plt.close()
        print(f"✓ Saved Profitability Matrix for {t_col}")

def plot_feature_importance():
    """Think Aloud #5: Feature Importance for Heterogeneity."""
    file_path = "results/feature_importance.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    
    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col].copy()
        top10 = data.sort_values("importance", ascending=False).head(10)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top10, x="importance", y="feature", palette="magma")
        
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Top 10 Drivers of Heterogeneity: {label}")
        plt.xlabel("Variable Importance (MWM Score)")
        plt.ylabel("Feature")
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"{THINK_ALOUD_DIR}/feature_importance_{safe_name}.png")
        plt.close()
        print(f"✓ Saved Feature Importance for {t_col}")

def plot_cate_violins():
    """Option A: Violin Plots for CATE distributions."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    df["treatment_label"] = df["treatment"].map(LABEL_MAP)
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=df, x="treatment_label", y="cate", 
        order=[LABEL_MAP[t] for t in ORDER if t in df["treatment"].unique()],
        inner="box", palette="muted"
    )
    plt.axhline(0, color="red", linestyle="--", alpha=0.5)
    plt.title("Comparison of CATE Distributions across Coupon Categories")
    plt.xlabel("Coupon Category")
    plt.ylabel("Estimated CATE (monetary units)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("plots/cate_violin_comparison.png")
    print("✓ Saved plots/cate_violin_comparison.png")

def plot_gate_heatmaps():
    """Option A: Heatmaps for Cross-Tabulated Subgroups."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path): return
    df = pd.read_csv(file_path)
    if "age_range" not in df.columns or "income_bracket" not in df.columns: return

    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col]
        pivot_df = data.pivot_table(values="cate", index="age_range", columns="income_bracket", aggfunc="mean")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Heatmap: Avg CATE for {label}")
        plt.xlabel("Income Bracket")
        plt.ylabel("Age Range")
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"plots/gate_heatmap_{safe_name}.png")
        plt.close()
        print(f"✓ Saved GATE heatmap for {t_col}")

def plot_causal_dag():
    """Option A: Structural Causal Model (DAG)."""
    ate_file = "results/ate_results.csv"
    conf_file = "results/confounder_strengths.csv"
    if not os.path.exists(ate_file) or not os.path.exists(conf_file): return
    ate_df = pd.read_csv(ate_file)
    conf_df = pd.read_csv(conf_file)
    ate_df = ate_df[(ate_df["method"] == "GRF") & (ate_df["phase"] == "Phase 1 - Full") & (ate_df["outcome"] == "avg_daily_expenditure")]

    for t_col in ate_df["treatment"].unique():
        ate_val = ate_df[ate_df["treatment"] == t_col]["estimate"].values[0]
        t_conf = conf_df[conf_df["treatment"] == t_col]
        plt.figure(figsize=(10, 7))
        pos = {"Demographics": (0.1, 0.8), "Baseline Habits": (0.1, 0.2), "Coupon (D)": (0.5, 0.5), "Expenditure (Y)": (0.9, 0.5)}
        for node, (x, y) in pos.items():
            color = "lightblue" if "(D)" in node else "lightgreen" if "(Y)" in node else "lightgrey"
            circle = plt.Circle((x, y), 0.08, color=color, ec="black", zorder=3)
            plt.gca().add_patch(circle)
            plt.text(x, y, node, ha="center", va="center", fontweight="bold", zorder=4)

        def draw_arrow(start, end, label, color="black", lw=2):
            plt.annotate("", xy=pos[end], xytext=pos[start], arrowprops=dict(arrowstyle="->", color=color, lw=lw, shrinkA=35, shrinkB=35, connectionstyle="arc3,rad=0.1"))
            lx, ly = (pos[start][0] + pos[end][0]) / 2, (pos[start][1] + pos[end][1]) / 2
            plt.text(lx, ly + 0.05, label, ha="center", color=color, fontweight="bold")

        for _, row in t_conf.iterrows():
            c_node = "Demographics" if row["confounder_group"] == "Demographics" else "Baseline Habits"
            draw_arrow(c_node, "Coupon (D)", f"r={row['strength_to_d']:.2f}", color="gray")
            draw_arrow(c_node, "Expenditure (Y)", f"r={row['strength_to_y']:.2f}", color="gray")
        draw_arrow("Coupon (D)", "Expenditure (Y)", f"ATE = {ate_val:+.2f}", color="darkblue", lw=3)
        plt.title(f"SCM: {LABEL_MAP.get(t_col, t_col)}", fontsize=14, pad=20)
        plt.xlim(0, 1); plt.ylim(0, 1); plt.axis("off")
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout(); plt.savefig(f"plots/causal_dag_{safe_name}.png"); plt.close()
        print(f"✓ Saved DAG for {t_col}")

if __name__ == "__main__":
    # Standard Replication Plots
    plot_fig1_cate_distributions()
    plot_cate_violins()
    plot_gate_subgroups()
    plot_gate_heatmaps()
    plot_causal_dag()
    
    # Think Aloud Study Plots
    print("\nGenerating Think Aloud Study Visualizations...")
    plot_qini_curve()
    plot_waterfall_uncertainty()
    plot_personas()
    plot_profitability_threshold()
    plot_feature_importance()
