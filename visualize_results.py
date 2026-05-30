import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory
os.makedirs("plots", exist_ok=True)

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
    
    # We only plot short-term expenditure (default in the cate saving logic)
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
            # 95% CI is approx 1.96 * SE
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

def plot_cate_violins():
    """Option A: Violin Plots with Integrated Box Plots for CATE distributions."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path):
        print(f"Skipping Violin Plot: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # Map column names to pretty labels
    df["treatment_label"] = df["treatment"].map(LABEL_MAP)
    
    plt.figure(figsize=(14, 8))
    # Use 'inner="box"' to include the integrated box plot
    sns.violinplot(
        data=df, 
        x="treatment_label", 
        y="cate", 
        order=[LABEL_MAP[t] for t in ORDER if t in df["treatment"].unique()],
        inner="box", 
        palette="muted"
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
    """Option A: Heatmaps for Cross-Tabulated Subgroups (Age vs Income)."""
    file_path = "results/cate_distributions.csv"
    if not os.path.exists(file_path):
        print(f"Skipping Heatmaps: {file_path} not found.")
        return

    df = pd.read_csv(file_path)
    
    # We only care about rows that have both age and income info
    if "age_range" not in df.columns or "income_bracket" not in df.columns:
        print("Skipping Heatmaps: age_range or income_bracket missing from data.")
        return

    # Filter out 'unknown' if it's too noisy, or keep it to see missingness effect
    # The paper mentions 'unknown' is common, so we keep it.

    for t_col in df["treatment"].unique():
        data = df[df["treatment"] == t_col]
        
        # Pivot the data to get average CATE per (Age, Income) cell
        # We handle potential duplicates by taking the mean (standard GATE logic)
        pivot_df = data.pivot_table(
            values="cate", 
            index="age_range", 
            columns="income_bracket", 
            aggfunc="mean"
        )

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_df, annot=True, fmt=".1f", cmap="RdYlGn", center=0)
        
        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Heatmap: Average CATE for {label}\n(Income Bracket vs Age Range)")
        plt.xlabel("Income Bracket")
        plt.ylabel("Age Range")
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"plots/gate_heatmap_{safe_name}.png")
        plt.close()
        
        print(f"✓ Saved GATE heatmap for {t_col}")

def plot_causal_dag():
    """Option A: Structural Causal Model (DAG) with Global Path Notation (ATE)."""
    ate_file = "results/ate_results.csv"
    conf_file = "results/confounder_strengths.csv"
    
    if not os.path.exists(ate_file) or not os.path.exists(conf_file):
        print("Skipping DAG: Result files not found.")
        return

    ate_df = pd.read_csv(ate_file)
    conf_df = pd.read_csv(conf_file)
    
    # Filter to GRF results in Phase 1 for the main short-term outcome
    ate_df = ate_df[
        (ate_df["method"] == "GRF") & 
        (ate_df["phase"] == "Phase 1 - Full") &
        (ate_df["outcome"] == "avg_daily_expenditure")
    ]

    for t_col in ate_df["treatment"].unique():
        ate_val = ate_df[ate_df["treatment"] == t_col]["estimate"].values[0]
        t_conf = conf_df[conf_df["treatment"] == t_col]
        
        plt.figure(figsize=(10, 7))
        
        # Define node positions (X, Y)
        # Layer 1: Confounders (Left Top)
        # Layer 2: Treatment (Middle)
        # Layer 3: Outcome (Right)
        pos = {
            "Demographics": (0.1, 0.8),
            "Baseline Habits": (0.1, 0.2),
            "Coupon (D)": (0.5, 0.5),
            "Expenditure (Y)": (0.9, 0.5)
        }
        
        # Draw Nodes
        for node, (x, y) in pos.items():
            color = "lightgrey"
            if "(D)" in node: color = "lightblue"
            if "(Y)" in node: color = "lightgreen"
            
            circle = plt.Circle((x, y), 0.08, color=color, ec="black", zorder=3)
            plt.gca().add_patch(circle)
            plt.text(x, y, node, ha="center", va="center", fontweight="bold", zorder=4)

        # Draw Edges and Labels
        def draw_arrow(start, end, label, color="black", style="->", lw=2):
            plt.annotate("", xy=pos[end], xytext=pos[start],
                         arrowprops=dict(arrowstyle=style, color=color, lw=lw, 
                                       shrinkA=35, shrinkB=35, connectionstyle="arc3,rad=0.1"))
            # Calculate midpoint for label
            lx = (pos[start][0] + pos[end][0]) / 2
            ly = (pos[start][1] + pos[end][1]) / 2
            plt.text(lx, ly + 0.05, label, ha="center", color=color, fontweight="bold")

        # X -> D and X -> Y
        for _, row in t_conf.iterrows():
            c_node = "Demographics" if row["confounder_group"] == "Demographics" else "Baseline Habits"
            draw_arrow(c_node, "Coupon (D)", f"r={row['strength_to_d']:.2f}", color="gray")
            draw_arrow(c_node, "Expenditure (Y)", f"r={row['strength_to_y']:.2f}", color="gray")

        # D -> Y (Main Causal Path)
        draw_arrow("Coupon (D)", "Expenditure (Y)", f"ATE = {ate_val:+.2f}", color="darkblue", lw=3)

        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Structural Causal Model: {label}", fontsize=14, pad=20)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis("off")
        
        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"plots/causal_dag_{safe_name}.png")
        plt.close()
        print(f"✓ Saved DAG for {t_col}")

if __name__ == "__main__":
    plot_fig1_cate_distributions()
    plot_cate_violins()
    plot_gate_subgroups()
    plot_gate_heatmaps()
    plot_causal_dag()
