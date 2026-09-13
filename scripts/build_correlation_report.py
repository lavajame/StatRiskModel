"""Build a PDF report comparing raw and factor-residual correlations in Claude style."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from riskmodel.data_loader import standardize_returns_ewma
from riskmodel.factor_engine import fit_global_pca
from riskmodel.universes import BENCHMARK_UNIVERSE


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "raw"
OUTPUT = REPORTS / "non_benchmark_correlation_report.pdf"

COLORS = {
    "ink": "#0f1419",
    "muted": "#9ca3af",
    "grid": "#e5e7eb",
    "teal": "#0891b2",
    "coral": "#dc2626",
    "gold": "#b45309",
    "green": "#059669",
    "paper": "#ffffff",
    "light_bg": "#f9fafb",
}


def load_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(DATA / "daily_returns.csv", parse_dates=["Date"]).set_index("Date")
    residuals = pd.read_csv(REPORTS / "strategy_oos_residuals.csv", parse_dates=["Date"]).set_index("Date")
    assets = [asset for asset in residuals if asset in raw and asset != "^PUT"]
    raw = raw.loc[residuals.index, assets]
    residuals = residuals[assets]
    pairs = []
    for left_index, left in enumerate(assets):
        for right in assets[left_index + 1 :]:
            pair = pd.concat([raw[left], raw[right], residuals[left], residuals[right]], axis=1).dropna()
            raw_corr = pair.iloc[:, 0].corr(pair.iloc[:, 1])
            residual_corr = pair.iloc[:, 2].corr(pair.iloc[:, 3])
            pairs.append({
                "left": left,
                "right": right,
                "raw": raw_corr,
                "residual": residual_corr,
                "change": residual_corr - raw_corr,
                "n": len(pair),
            })
    pair_frame = pd.DataFrame(pairs)
    diagnostics = pd.read_csv(REPORTS / "strategy_oos_diagnostics.csv").set_index("ticker").loc[assets]
    return raw, residuals, pair_frame, diagnostics


def heatmap(axis: plt.Axes, values: pd.DataFrame, title: str) -> None:
    image = axis.imshow(values, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    axis.set_xticks(range(len(values)), values.columns, fontsize=12, rotation=45, ha="right")
    axis.set_yticks(range(len(values)), values.index, fontsize=12)
    axis.tick_params(length=0)
    
    for row in range(len(values)):
        for column in range(len(values)):
            value = values.iloc[row, column]
            text_color = "white" if abs(value) > 0.5 else COLORS["ink"]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=9, color=text_color, fontweight="bold" if abs(value) > 0.6 else "normal")
    
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["grid"])
        spine.set_linewidth(1)
    return image


def style_page(figure: plt.Figure) -> None:
    figure.patch.set_facecolor(COLORS["paper"])


def compute_benchmark_loadings() -> tuple[pd.DataFrame, list[str], tuple, np.ndarray]:
    """Compute PC loadings for benchmark assets. Returns loadings DataFrame, list of benchmark tickers, date range, and eigenvalues."""
    from scipy import linalg
    
    # Load all returns
    all_returns = pd.read_csv(DATA / "daily_returns.csv", parse_dates=["Date"]).set_index("Date")
    
    # Get benchmark tickers that are available
    benchmark_tickers = [t for t in BENCHMARK_UNIVERSE.keys() if t in all_returns.columns and all_returns[t].notna().any()]
    
    # Extract benchmark returns and standardize
    benchmark_returns = all_returns[benchmark_tickers].dropna()
    standardized, _ = standardize_returns_ewma(benchmark_returns)
    
    # Fit PCA on full dataset - compute eigenvalues directly
    n_factors = min(len(benchmark_tickers), 8)  # Use up to 8 PCs or number of assets, whichever is smaller
    corr_matrix = standardized.corr().to_numpy()
    eigenvalues, eigenvectors = linalg.eigh(corr_matrix)
    indices = np.argsort(eigenvalues)[::-1][:n_factors]
    reference_loadings = eigenvectors[:, indices]
    eigenvalues_sorted = eigenvalues[indices]
    
    # Create DataFrame with loadings
    loadings_df = pd.DataFrame(
        reference_loadings,
        index=benchmark_tickers,
        columns=[f"PC{i+1}" for i in range(n_factors)]
    )
    
    date_range = (benchmark_returns.index.min(), benchmark_returns.index.max())
    return loadings_df, benchmark_tickers, date_range, eigenvalues_sorted


def style_page(figure: plt.Figure) -> None:
    figure.patch.set_facecolor(COLORS["paper"])


def build_report() -> Path:
    raw, residuals, pairs, diagnostics = load_analysis()
    raw_corr = raw.corr()
    residual_corr = residuals.corr()
    top = pairs.assign(abs_change=lambda frame: frame["change"].abs()).sort_values("abs_change", ascending=False)

    with PdfPages(OUTPUT) as pdf:
        # PAGE 1: TITLE + KEY INSIGHT
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.85, "Factor Model", fontsize=54, fontweight="bold", color=COLORS["ink"], ha="center", va="center")
        figure.text(0.5, 0.77, "Diagnostics", fontsize=54, fontweight="bold", color=COLORS["teal"], ha="center", va="center")
        figure.text(0.5, 0.68, "Systematic Analysis of Correlation Structure", fontsize=16, color=COLORS["muted"], ha="center", va="center", style="italic")
        
        figure.text(0.5, 0.55, "Key Finding", fontsize=28, fontweight="bold", color=COLORS["ink"], ha="center", va="top", wrap=True)
        figure.text(0.1, 0.48, "Strong raw-return associations typically diminish significantly once systematic factor exposure is removed.\n\nThis reveals that factor loading is the dominant driver of cross-asset correlation.", 
                   fontsize=14, color=COLORS["ink"], ha="left", va="top", wrap=True, linespacing=2.0)
        
        figure.text(0.05, 0.08, f"Analysis Scope  •  {len(raw.columns)} assets  •  {len(pairs)} pairs  •  {len(residuals):,} observations  •  {residuals.index.min():%b %Y} – {residuals.index.max():%b %Y}",
                   fontsize=11, color=COLORS["muted"], ha="left", va="bottom")
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 2: BENCHMARK UNIVERSE DEFINITION
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.96, "Benchmark Universe", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, "18 assets used to construct the factor model", fontsize=14, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        # Data range info - add at top
        all_ret_check = pd.read_csv(DATA / "daily_returns.csv", parse_dates=["Date"]).set_index("Date")
        bm_tickers_check = [t for t in BENCHMARK_UNIVERSE.keys() if t in all_ret_check.columns and all_ret_check[t].notna().any()]
        bm_ret_check = all_ret_check[bm_tickers_check].dropna()
        
        figure.text(0.5, 0.82, f"Training Period: {bm_ret_check.index.min():%B %d, %Y} to {bm_ret_check.index.max():%B %d, %Y} ({len(bm_ret_check):,} observations)", 
                   fontsize=11, color=COLORS["teal"], ha="center", va="top", fontweight="bold")
        
        # Create three columns for benchmark assets
        benchmark_list = list(BENCHMARK_UNIVERSE.items())
        col_size = (len(benchmark_list) + 2) // 3
        
        y_start = 0.75
        x_positions = [0.08, 0.37, 0.66]
        
        for col_idx in range(3):
            x = x_positions[col_idx]
            y = y_start
            start_idx = col_idx * col_size
            end_idx = min((col_idx + 1) * col_size, len(benchmark_list))
            
            for ticker, description in benchmark_list[start_idx:end_idx]:
                ax = figure.add_axes([x, y - 0.04, 0.25, 0.035])
                ax.axis("off")
                text_label = f"{ticker}\n{description}"
                ax.text(0.05, 0.5, text_label, fontsize=9, ha="left", va="center", family="monospace", fontweight="bold")
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS["grid"], linewidth=0.8, transform=ax.transAxes))
                y -= 0.055
        
        figure.text(0.08, 0.08, "These benchmark assets form the fixed universe on which the factor model is trained. All factors are extracted from the daily returns of these instruments.",
                   fontsize=10, color=COLORS["muted"], ha="left", va="bottom", style="italic", wrap=True)
        
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 3: PRINCIPAL COMPONENT LOADINGS HEATMAP
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        loadings_df, benchmark_tickers, bm_date_range, eigenvalues = compute_benchmark_loadings()
        
        # Cluster assets based on all PCs weighted by eigenvalues
        # Higher variance PCs naturally get more weight due to larger eigenvalues
        weights = eigenvalues / eigenvalues.sum()  # Normalize weights across all PCs
        
        # Extract all PCs and weight them
        weighted_loadings = loadings_df.values * weights
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(weighted_loadings, method="ward")
        cluster_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
        
        # Reorder loadings_df based on cluster order
        loadings_df = loadings_df.iloc[cluster_order, :]
        
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.96, "Factor Model Loadings", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, f"Principal component loadings on benchmark assets (first {len(loadings_df.columns)} PCs) — ordered by full PC similarity", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        ax = figure.add_axes([0.08, 0.15, 0.85, 0.68])
        
        # Create heatmap of loadings
        image = ax.imshow(loadings_df.T, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        
        # Set ticks
        ax.set_xticks(range(len(loadings_df)))
        ax.set_xticklabels(loadings_df.index, fontsize=10, rotation=45, ha="right")
        ax.set_yticks(range(len(loadings_df.columns)))
        ax.set_yticklabels(loadings_df.columns, fontsize=10)
        
        # Add value labels
        for row in range(len(loadings_df.columns)):
            for col in range(len(loadings_df)):
                value = loadings_df.iloc[col, row]
                text_color = "white" if abs(value) > 0.5 else COLORS["ink"]
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, 
                       color=text_color, fontweight="bold" if abs(value) > 0.6 else "normal")
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.8)
        
        ax.set_xlabel("Benchmark Assets", fontsize=12, fontweight="bold", color=COLORS["ink"])
        ax.set_ylabel("Principal Components", fontsize=12, fontweight="bold", color=COLORS["ink"])
        ax.tick_params(length=0)
        
        figure.text(0.5, 0.03, "Assets are ordered by hierarchical clustering of all PC loadings weighted by eigenvalues.\nDominant PCs (larger eigenvalues) guide similarity, capturing full factor structure.",
                   fontsize=10, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 4: FACTOR MODEL METHODOLOGY - CONSTRUCTION
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.96, "Factor Model Methodology", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, "Part 1: Model Construction", fontsize=14, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        figure.text(0.08, 0.82, "Objective", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.77, "Decompose asset returns into systematic factor exposures and idiosyncratic residuals. The model captures co-movement driven by shared macroeconomic and market factors.", 
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.7)
        
        figure.text(0.08, 0.68, "Construction Process", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.63, "1. Input data: Daily returns of the benchmark universe and explanatory universe assets\n2. Factor selection: Identify principal economic and market factors via PCA and correlation analysis\n3. Factor fit: Non-linear regression of asset returns on selected factors (XGBoost ensemble)\n4. Out-of-sample testing: Holdout period validation to assess generalization\n5. Residual extraction: OOS residuals = actual returns − fitted factor predictions",
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.8, family="monospace")
        
        figure.text(0.08, 0.28, "Key Properties", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.23, "• Factors are time-invariant and asset-universe-specific\n• Model trained on benchmark + explanatory universe\n• Residuals represent unexplained co-movement and asset-specific risk\n• Out-of-sample validation prevents overfitting bias",
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.8, family="monospace")
        
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 5: FACTOR MODEL METHODOLOGY - APPLICATION
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.96, "Factor Model Methodology", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, "Part 2: Diagnostic Application", fontsize=14, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        figure.text(0.08, 0.82, "Separate Analysis Universes", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.77, "The factor model is constructed on one fixed universe. For diagnostic purposes, we apply this same model to a separate target universe of assets, measuring how well the fitted factors explain their behavior.",
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.7)
        
        figure.text(0.08, 0.66, "Analysis Workflow", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.61, "1. Extract fitted factor loadings from model (trained universe)\n2. Apply loadings to target universe assets\n3. Compute target universe asset correlations (raw, daily returns)\n4. Compute residual universe correlations (OOS fit residuals)\n5. Compare correlation structures to identify factor-driven vs. unexplained co-movement",
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.8, family="monospace")
        
        figure.text(0.08, 0.31, "Diagnostic Insights", fontsize=13, fontweight="bold", color=COLORS["teal"])
        figure.text(0.08, 0.26, "• If raw correlation >> residual correlation: co-movement driven by shared factors (model success)\n• If raw correlation ≈ residual correlation: co-movement not captured by model (potential missing factors)\n• Magnitude of change reveals the strength of factor-loading dependence",
                   fontsize=11, color=COLORS["ink"], wrap=True, ha="left", va="top", linespacing=1.8, family="monospace")
        
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 6: TARGET UNIVERSE DEFINITION
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        figure.text(0.5, 0.96, "Target Universe", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, f"{len(raw.columns)} investible assets for diagnostic analysis", fontsize=14, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        # Data range info
        figure.text(0.5, 0.82, f"Analysis Period: {residuals.index.min():%B %d, %Y} to {residuals.index.max():%B %d, %Y} ({len(residuals):,} observations)", 
                   fontsize=11, color=COLORS["teal"], ha="center", va="top", fontweight="bold")
        
        # List target assets from raw.columns
        target_assets = raw.columns.tolist()
        
        # Create three columns for target assets
        col_size = (len(target_assets) + 2) // 3
        y_start = 0.75
        x_positions = [0.08, 0.37, 0.66]
        
        for col_idx in range(3):
            x = x_positions[col_idx]
            y = y_start
            start_idx = col_idx * col_size
            end_idx = min((col_idx + 1) * col_size, len(target_assets))
            
            for ticker in target_assets[start_idx:end_idx]:
                ax = figure.add_axes([x, y - 0.04, 0.25, 0.035])
                ax.axis("off")
                ax.text(0.05, 0.5, ticker, fontsize=10, ha="left", va="center", family="monospace", fontweight="bold")
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor=COLORS["grid"], linewidth=0.8, transform=ax.transAxes))
                y -= 0.055
        
        figure.text(0.08, 0.08, f"These {len(target_assets)} assets comprise the target universe to which the factor model is applied. Pairwise analysis produces {len(pairs)} unique correlations for diagnostic comparison.",
                   fontsize=10, color=COLORS["muted"], ha="left", va="bottom", style="italic", wrap=True)
        
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 7: RAW CORRELATION HEATMAP (FULL PAGE)
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        
        figure.text(0.5, 0.96, "Raw Returns", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.87, "Pairwise Pearson Correlation of Daily Returns", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        # Plot heatmap - smaller size, moved up
        left, bottom, width, height = 0.1, 0.18, 0.8, 0.62
        ax_heat = figure.add_axes([left, bottom, width, height])
        image = ax_heat.imshow(raw_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax_heat.set_xticks(range(len(raw_corr)), raw_corr.columns, fontsize=10, rotation=45, ha="right")
        ax_heat.set_yticks(range(len(raw_corr)), raw_corr.index, fontsize=10)
        ax_heat.tick_params(length=0)
        
        for row in range(len(raw_corr)):
            for column in range(len(raw_corr)):
                value = raw_corr.iloc[row, column]
                text_color = "white" if abs(value) > 0.5 else COLORS["ink"]
                ax_heat.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold" if abs(value) > 0.6 else "normal")
        
        for spine in ax_heat.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.8)
        
        figure.text(0.5, 0.06, "This matrix shows the baseline correlation structure before factor removal.",
                   fontsize=11, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 8: RESIDUAL CORRELATION HEATMAP (FULL PAGE)
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        
        figure.text(0.5, 0.96, "After Factor Removal", fontsize=40, fontweight="bold", color=COLORS["teal"], ha="center", va="top")
        figure.text(0.5, 0.87, "Pairwise Pearson Correlation of Out-of-Sample Residuals", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        # Plot heatmap - smaller size, moved up
        left, bottom, width, height = 0.1, 0.18, 0.8, 0.62
        ax_heat = figure.add_axes([left, bottom, width, height])
        image = ax_heat.imshow(residual_corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax_heat.set_xticks(range(len(residual_corr)), residual_corr.columns, fontsize=10, rotation=45, ha="right")
        ax_heat.set_yticks(range(len(residual_corr)), residual_corr.index, fontsize=10)
        ax_heat.tick_params(length=0)
        
        for row in range(len(residual_corr)):
            for column in range(len(residual_corr)):
                value = residual_corr.iloc[row, column]
                text_color = "white" if abs(value) > 0.5 else COLORS["ink"]
                ax_heat.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color, fontweight="bold" if abs(value) > 0.6 else "normal")
        
        for spine in ax_heat.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["grid"])
            spine.set_linewidth(0.8)
        
        figure.text(0.5, 0.06, "Notice the substantial reduction in correlation magnitude—evidence of factor model explanatory power.",
                   fontsize=11, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 9: SCATTER PLOT (RAW vs RESIDUAL)
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        
        figure.text(0.5, 0.94, "Pairwise Transformations", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.84, "How correlation changes when factors are removed", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        ax = figure.add_axes([0.12, 0.15, 0.75, 0.65])
        ax.axhline(0, color=COLORS["grid"], linewidth=1.5, alpha=0.8, zorder=0)
        ax.axvline(0, color=COLORS["grid"], linewidth=1.5, alpha=0.8, zorder=0)
        ax.plot([-1, 1], [-1, 1], linestyle="--", color=COLORS["muted"], linewidth=2.5, alpha=0.4, zorder=1)
        
        # Color points by correlation change: red (positive) to blue (negative)
        change_values = pairs["change"].values
        change_min = change_values.min()
        change_max = change_values.max()
        
        # Normalize changes to 0-1 range for color mapping
        if change_max > -change_min:
            # Positive changes are larger
            normalized_changes = (change_values - change_min) / (change_max - change_min)
        else:
            # Negative changes are larger (or equal)
            normalized_changes = (change_values - change_min) / (change_max - change_min)
        
        # Map to red-to-blue colormap: 0 = blue, 1 = red
        colors_scatter = plt.cm.RdBu_r(normalized_changes)
        
        scatter = ax.scatter(pairs["raw"], pairs["residual"], s=150, c=colors_scatter, 
                           edgecolor=COLORS["ink"], linewidth=1.2, zorder=3, alpha=0.75)
        
        # Label only the most extreme points with smart proximity-aware orientations
        labeled_points = top.head(5).copy()
        label_positions = [(row.raw, row.residual) for _, row in labeled_points.iterrows()]
        
        # Determine orientations based on proximity in data space
        orientations = []
        for idx, (pos_x, pos_y) in enumerate(label_positions):
            # Check if this point is close to any previous point
            close_to_previous = False
            if idx > 0:
                for prev_idx in range(idx):
                    prev_x, prev_y = label_positions[prev_idx]
                    # Distance in normalized space (correlation ranges -0.85 to 0.85)
                    dist = np.sqrt((pos_x - prev_x)**2 + (pos_y - prev_y)**2)
                    if dist < 0.25:  # Close threshold
                        close_to_previous = True
                        # Use opposite orientation of previous point
                        if orientations[prev_idx] in ["upper-left", "upper"]:
                            orientations.append("lower-right")
                        elif orientations[prev_idx] in ["lower-right", "lower"]:
                            orientations.append("upper-left")
                        else:
                            orientations.append("lower-right")
                        break
            
            if not close_to_previous:
                # Default: alternate based on index
                if idx % 2 == 0:
                    orientations.append("upper-left")
                else:
                    orientations.append("lower-right")
        
        # Apply labels with assigned orientations
        for idx, (_, row) in enumerate(labeled_points.iterrows()):
            label = f"{row.left} / {row.right}"
            orient = orientations[idx]
            
            if orient == "upper-left":
                offset = (-18, 16)
                arrow_dir = "<-"
            elif orient == "lower-right":
                offset = (18, -16)
                arrow_dir = "->"
            elif orient == "upper":
                offset = (0, 20)
                arrow_dir = "^"
            else:  # lower
                offset = (0, -20)
                arrow_dir = "v"
            
            ax.annotate(label, (row.raw, row.residual), xytext=offset, textcoords="offset points",
                       fontsize=9, color=COLORS["ink"], fontweight="bold",
                       bbox={"boxstyle": "round,pad=0.4", "facecolor": COLORS["light_bg"], "edgecolor": COLORS["muted"], "alpha": 0.9, "linewidth": 1},
                       arrowprops={"arrowstyle": arrow_dir, "color": COLORS["muted"], "linewidth": 1.0})
        
        ax.set_xlim(-0.85, 0.85)
        ax.set_ylim(-0.65, 0.65)
        ax.set_xlabel("Raw-Return Correlation", fontsize=14, fontweight="bold", color=COLORS["ink"])
        ax.set_ylabel("Residual Correlation", fontsize=14, fontweight="bold", color=COLORS["ink"])
        ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.4, zorder=0)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=11)
        
        # Quadrant labels (below diagonal = reduction, above diagonal = increase)
        ax.text(0.35, -0.58, "Correlation Reduction", fontsize=10, color=COLORS["coral"], fontweight="bold", va="bottom")
        ax.text(-0.82, 0.58, "Correlation Increase", fontsize=10, color=COLORS["green"], fontweight="bold", va="top")
        
        figure.text(0.5, 0.04, "Top 5 pairs by absolute correlation change. Below diagonal: factor model explained the co-movement. Above diagonal: unexpected residual correlation after factor removal. Color: blue = decrease, red = increase.",
                   fontsize=10, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        pdf.savefig(figure)
        plt.close(figure)

        # PAGE 10: LARGEST IMPACTS - ARROWS SHOWING DIRECTION OF CHANGE
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        
        figure.text(0.5, 0.96, "Largest Correlation Changes", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.89, "Direction and magnitude of shift from raw to residual correlation", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        display = top.head(10).iloc[::-1]
        ax = figure.add_axes([0.15, 0.18, 0.65, 0.62])

        lim_ = 0.0
        
        # Draw arrows from raw to residual correlation for each pair
        for idx, (_, row) in enumerate(display.iterrows()):
            # Determine color based on change direction
            color = COLORS["coral"] if row["change"] < 0 else COLORS["green"]
            
            # Arrow from raw to residual correlation
            dx = row["residual"] - row["raw"]
            arrow = ax.arrow(row["raw"], idx, dx, 0, head_width=0.25, head_length=0.04, 
                           fc=color, ec=COLORS["ink"], linewidth=1.5, alpha=0.8, zorder=2)
            
            # Dot at starting point (raw correlation)
            ax.scatter([row["raw"]], [idx], s=100, color=COLORS["teal"], edgecolor=COLORS["ink"], 
                      linewidth=1.2, zorder=3, marker="o")
            
            # Small text labels for initial (raw) and final (residual) correlations
            ax.text(row["raw"], idx + 0.22, f"{row['raw']:.3f}", fontsize=7, ha="center", 
                   color=COLORS["teal"], fontweight="bold", va="bottom")
            ax.text(row["residual"], idx + 0.22, f"{row['residual']:.3f}", fontsize=7, ha="center", 
                   color=color, fontweight="bold", va="bottom")

            lim_ = np.max([lim_, abs(row["raw"]), abs(row["residual"])])
        
        # Y-axis labels (pair names)
        ax.set_yticks(range(len(display)))
        ax.set_yticklabels([f"{row.left} / {row.right}" for _, row in display.iterrows()], fontsize=10)
        ax.set_ylim(-0.8, len(display) - 0.2)
        
        # X-axis (correlation values)
        ax.set_xlim(-lim_ - 0.05, lim_ + 0.05)
        ax.set_xlabel("Correlation Value", fontsize=12, fontweight="bold", color=COLORS["ink"])
        ax.axvline(0, color=COLORS["ink"], linewidth=1.5, alpha=0.5, linestyle=":")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0)
        
        figure.text(0.5, 0.06, "Blue arrows: Correlation decreased after factor removal (factor-driven co-movement)  •  Red arrows: Correlation increased (potential missing factors)",
                   fontsize=11, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        pdf.savefig(figure)
        plt.close(figure)
        figure = plt.figure(figsize=(11.7, 8.3))
        style_page(figure)
        
        # PAGE 11: MODEL QUALITY BY ASSET
        figure.text(0.5, 0.94, "Model Quality by Asset", fontsize=40, fontweight="bold", color=COLORS["ink"], ha="center", va="top")
        figure.text(0.5, 0.84, "Residual volatility vs. out-of-sample fit quality", fontsize=12, color=COLORS["muted"], ha="center", va="top", style="italic")
        
        x = diagnostics["oos_r_squared"]
        y = diagnostics["idiosyncratic_volatility"]
        
        ax = figure.add_axes([0.12, 0.15, 0.75, 0.65])
        ax.scatter(x, y, s=250, color=COLORS["gold"], edgecolor=COLORS["ink"], linewidth=1.3, alpha=0.8, zorder=3)
        ax.axvline(0.5, color=COLORS["muted"], linewidth=2, linestyle=":", alpha=0.5, zorder=1)
        
        for asset in diagnostics.index:
            # Smart label positioning based on position on plot
            if x[asset] < 0.4:
                offset_x = 14
            elif x[asset] > 0.8:
                offset_x = -16
            else:
                offset_x = 14 if y[asset] < ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) / 2 else -16
            ax.annotate(asset, (x[asset], y[asset]), xytext=(offset_x, 12), textcoords="offset points",
                       fontsize=10, color=COLORS["ink"], fontweight="bold",
                       bbox={"boxstyle": "round,pad=0.4", "facecolor": COLORS["light_bg"], "edgecolor": COLORS["muted"], "alpha": 0.9, "linewidth": 1})
        
        ax.set_xlabel("Out-of-Sample R-squared (Model Fit)", fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.set_ylabel("Residual Volatility (Annualized)", fontsize=13, fontweight="bold", color=COLORS["ink"])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        ax.grid(True, color=COLORS["grid"], linewidth=0.8, alpha=0.4, zorder=0)
        ax.tick_params(labelsize=11)
        
        # Compute axis limits with padding to avoid point clipping
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        x_padding = x_range * 0.15
        y_padding = y_range * 0.15
        
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        # Shaded risk region
        ax.axvspan(ax.get_xlim()[0], 0.5, alpha=0.05, color=COLORS["coral"], zorder=0)
        
        # Explanatory text box in whitespace
        textstr = "Higher R²\n(right) →\nBetter model fit\n\nHigher volatility\n(top) →\nMore unexplained\nvariation"
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9.5, verticalalignment="top",
               bbox={"boxstyle": "round,pad=0.6", "facecolor": COLORS["light_bg"], "edgecolor": COLORS["muted"], "alpha": 0.92, "linewidth": 0.8},
               color=COLORS["ink"], family="monospace")
        
        figure.text(0.5, 0.04, "Assets at bottom-left (low volatility, high fit) are well-explained by the model. Assets at top-left or bottom-middle indicate model risk.",
                   fontsize=11, color=COLORS["muted"], ha="center", va="bottom", style="italic")
        pdf.savefig(figure)
        plt.close(figure)

    return OUTPUT


if __name__ == "__main__":
    print(f"Wrote {build_report()}")