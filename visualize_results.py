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

def plot_summary_table():
    """Think Aloud #6: Detailed Summary Tables per category."""
    ate_path = "results/ate_results.csv"
    cate_path = "results/cate_distributions.csv"
    gate_path = "results/gate_results.csv"
    if not all(os.path.exists(p) for p in [ate_path, cate_path, gate_path]): return

    ate_df = pd.read_csv(ate_path)
    cate_df = pd.read_csv(cate_path)
    gate_df = pd.read_csv(gate_path)

    # Filter ATE and GATE to main phase and outcome
    ate_df = ate_df[
        (ate_df["phase"] == "Phase 1 - Full") & 
        (ate_df["outcome"] == "avg_daily_expenditure") &
        (ate_df["method"] == "GRF")
    ]
    gate_df = gate_df[
        (gate_df["phase"] == "Phase 1 - Full") & 
        (gate_df["outcome"] == "avg_daily_expenditure")
    ]

    for t_col in ORDER:
        if t_col not in cate_df["treatment"].unique(): continue
        
        # 1. Global Metrics
        ate_row = ate_df[ate_df["treatment"] == t_col]
        if ate_row.empty: continue
        ate_val = ate_row["estimate"].values[0]
        ate_se = ate_row["se"].values[0]
        
        # 2. GATE Breakdown
        t_gate = gate_df[(gate_df["treatment"] == t_col) & (gate_df["variable"] != "(Intercept)")].copy()
        
        def parse_var(v):
            if "age_range" in v: 
                val = v.replace("age_range_", "").replace(".", "-")
                if val == "70-": val = "70+"
                return ("Age", val)
            if "income_bracket" in v: return ("Income", v.replace("income_bracket_", ""))
            if "family_size" in v: return ("Family Size", v.replace("family_size_", ""))
            return ("Other", v)

        rows = []
        # Section Header: Global
        rows.append(["GLOBAL CAUSAL METRICS", "", "", ""])
        rows.append(["Confounders Controlled", "Demographics, Baseline Habits", "", ""])
        rows.append(["Overall ATE", f"${ate_val:.2f} (±{1.96*ate_se:.2f})", "", ""])
        
        # Section Header: GATE
        rows.append(["", "", "", ""]) # Spacer
        rows.append(["SUBGROUP HETEROGENEITY (GATE)", "", "", ""])
        rows.append(["Subgroup Variable", "Value", "Point Estimate", "95% Confidence Interval"])
        
        for _, g_row in t_gate.iterrows():
            var_name, var_val = parse_var(g_row["variable"])
            ci_low = g_row["coef"] - 1.96 * g_row["se"]
            ci_high = g_row["coef"] + 1.96 * g_row["se"]
            rows.append([var_name, var_val, f"${g_row['coef']:.2f}", f"[${ci_low:.2f}, ${ci_high:.2f}]"])

        # Create Table Plot
        fig, ax = plt.subplots(figsize=(12, len(rows) * 0.4 + 2))
        ax.axis('off')
        
        table = ax.table(
            cellText=rows,
            cellLoc='left',
            loc='center',
            colWidths=[0.32, 0.32, 0.12, 0.24]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Styling: Bold headers and section titles
        for (i, j), cell in table.get_celld().items():
            content = rows[i][j]
            if content in ["GLOBAL CAUSAL METRICS", "SUBGROUP HETEROGENEITY (GATE)"]:
                cell.set_text_props(fontweight='bold', color='darkblue')
                cell.set_facecolor('#f0f0f8')
            if i == rows.index(["Subgroup Variable", "Value", "Point Estimate", "95% Confidence Interval"]):
                cell.set_text_props(fontweight='bold')
                cell.set_facecolor('#f2f2f2')
            
            # Hide borders for spacers
            if rows[i] == ["", "", "", ""]:
                cell.set_visible(False)

        label = LABEL_MAP.get(t_col, t_col)
        plt.title(f"Causal Effectiveness Summary: {label}", fontsize=14, pad=20, fontweight="bold")
        
        # Add explanation box
        explanation = (
            "Metrics Explanation:\n"
            "• ATE (Average Treatment Effect): The average change in expenditure caused by the coupon across all customers.\n"
            "• GATE (Group Average Treatment Effect): The estimated impact for specific demographic subgroups.\n"
            "  The 95% Confidence Interval shows the range where the true effect likely falls."
        )
        ax.text(0.5, -0.05, explanation, fontsize=8, ha='center', va='top', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="lightgray", alpha=0.9))

        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.tight_layout()
        plt.savefig(f"{THINK_ALOUD_DIR}/summary_table_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Think Aloud summary table for {label}")

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
    """Option A: Two-panel visualization (Structural DAG + CATE Forest Plot)."""
    ate_file = "results/ate_results.csv"
    conf_file = "results/confounder_strengths.csv"
    gate_file = "results/gate_results.csv"
    if not os.path.exists(ate_file) or not os.path.exists(conf_file) or not os.path.exists(gate_file): return
    
    ate_df = pd.read_csv(ate_file)
    conf_df = pd.read_csv(conf_file)
    gate_df = pd.read_csv(gate_file)
    
    ate_df = ate_df[(ate_df["method"] == "GRF") & (ate_df["phase"] == "Phase 1 - Full") & (ate_df["outcome"] == "avg_daily_expenditure")]

    for t_col in ate_df["treatment"].unique():
        ate_row = ate_df[ate_df["treatment"] == t_col]
        ate_val = ate_row["estimate"].values[0]
        ate_se = ate_row["se"].values[0]
        
        t_conf = conf_df[conf_df["treatment"] == t_col]
        t_gate = gate_df[(gate_df["treatment"] == t_col) & (gate_df["outcome"] == "avg_daily_expenditure") & (gate_df["phase"] == "Phase 1 - Full")]
        
        if t_gate.empty: continue

        # Prepare Forest Plot Data
        forest_data = t_gate[t_gate["variable"] != "(Intercept)"].copy()
        
        # Custom sorting logic
        def sort_key(var):
            if "age_range" in var:
                cat_order = 0
                val = float(var.split("_")[-1].split(".")[0]) if "." in var.split("_")[-1] else 0
            elif "income_bracket" in var:
                cat_order = 1
                val = float(var.split("_")[-1])
            elif "family_size" in var:
                cat_order = 2
                f_val = var.split("_")[-1]
                val = float(f_val.replace("+", ""))
            else:
                cat_order = 3
                val = 0
            return (cat_order, val)

        forest_data["sort_val"] = forest_data["variable"].apply(sort_key)
        forest_data = forest_data.sort_values("sort_val", ascending=False)
        forest_data["ci"] = 1.96 * forest_data["se"]

        # Create Figure
        fig, (ax_dag, ax_forest) = plt.subplots(1, 2, figsize=(20, 10), gridspec_kw={'width_ratios': [1, 1.2]})
        
        # --- LEFT PANEL: Structural DAG ---
        pos = {
            "Demographics": (0.1, 0.8),
            "Baseline Habits": (0.1, 0.2),
            "Coupon": (0.5, 0.5),
            "Expenditure": (0.9, 0.5)
        }
        
        node_patches = {}
        for node, (x, y) in pos.items():
            color = "lightblue" if "Coupon" in node else "lightgreen" if "Expenditure" in node else "lightgrey"
            circle = plt.Circle((x, y), 0.08, color=color, ec="black", zorder=3)
            ax_dag.add_patch(circle)
            node_patches[node] = circle
            ax_dag.text(x, y, node, ha="center", va="center", fontweight="bold", zorder=4, fontsize=11)

        def draw_arrow(start, end, label, color="black", lw=2, ax=ax_dag):
            ax.annotate("", xy=pos[end], xytext=pos[start], 
                         arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                         patchA=node_patches[start], patchB=node_patches[end],
                                         shrinkA=2, shrinkB=2))
            if label:
                p1, p2 = np.array(pos[start]), np.array(pos[end])
                mid = (p1 + p2) / 2
                ax.text(mid[0], mid[1], label, ha="center", va="center", color=color, 
                        fontweight="bold", fontsize=9, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))

        # Draw DAG Arrows
        for _, row in t_conf.iterrows():
            c_node = "Demographics" if row["confounder_group"] == "Demographics" else "Baseline Habits"
            draw_arrow(c_node, "Coupon", "", color="gray")
            draw_arrow(c_node, "Expenditure", "", color="gray")
        draw_arrow("Coupon", "Expenditure", f"ATE = {ate_val:+.2f}", color="darkblue", lw=3)

        ax_dag.set_title("Structural Causal Relationships", fontsize=14, fontweight="bold")
        ax_dag.set_xlim(0, 1); ax_dag.set_ylim(0, 1); ax_dag.axis("off")

        # --- RIGHT PANEL: Forest Plot ---
        # Add ATE to forest data for reference
        y_pos = np.arange(len(forest_data) + 1)
        ax_forest.errorbar(ate_val, len(forest_data), xerr=1.96*ate_se, fmt='o', color='darkblue', 
                           capsize=5, label="Overall ATE", markersize=8)
        
        ax_forest.errorbar(forest_data["coef"], np.arange(len(forest_data)), xerr=forest_data["ci"], 
                           fmt='o', color='black', ecolor='gray', capsize=3, alpha=0.7)
        
        ax_forest.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax_forest.set_yticks(y_pos)
        
        # Clean up labels
        def clean_label(v):
            if "age_range" in v:
                val = v.replace("age_range_", "").replace(".", "-")
                if val == "70-": val = "70+"
                return f"Age: {val}"
            return v.replace("income_bracket_", "Income: ").replace("family_size_", "Family: ")

        labels = [clean_label(v) for v in forest_data["variable"]] + ["OVERALL ATE"]
        ax_forest.set_yticklabels(labels)
        
        ax_forest.set_xlabel("Estimated Impact (monetary units)")
        ax_forest.set_title(f"Heterogeneity Details (GATE by Demographic-Subgroup)", fontsize=14, fontweight="bold")
        ax_forest.grid(axis='x', linestyle=':', alpha=0.6)
        
        # Styling
        plt.suptitle(f"Causal Analysis: {LABEL_MAP.get(t_col, t_col)}", fontsize=18, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Legend and Explanation at the bottom
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='darkblue', lw=2.5, label='Average Treatment Effect (ATE)'),
            Line2D([0], [0], color='black', marker='o', ls='none', label='Group Average Treatment Effect (GATE)')
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=True, fontsize=10)

        safe_name = t_col.replace("treatment_", "").replace("/", "_").replace(" ", "_")
        plt.savefig(f"plots/causal_dag_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved Two-Panel Analysis (GATE) for {t_col}")


if __name__ == "__main__":
    # Standard Replication Plots
    plot_fig1_cate_distributions()
    plot_cate_violins()
    plot_gate_subgroups()
    plot_gate_heatmaps()
    plot_causal_dag()
    
    # Think Aloud Study Plots
    print("\nGenerating Think Aloud Study Visualizations...")
    plot_summary_table()
    plot_qini_curve()
    plot_waterfall_uncertainty()
    plot_personas()
    plot_feature_importance()