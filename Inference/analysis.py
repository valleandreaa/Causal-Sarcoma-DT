from __future__ import annotations

from typing import Tuple
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import wasserstein_distance, ks_2samp, entropy
from matplotlib import patches as mpatches

# ------------------------------------------------------------------ #
# Column names (edit here if your CSV uses different labels)
# ------------------------------------------------------------------ #
csv_path = 'counterfactual_scenarios_multiple_patients_no_embed.csv'
LOCAL_RECURRENCE_COL = "fake_episodes.treatments.fields.endpoint_1.0_prob"
METASTATIC_COL       = "fake_episodes.treatments.fields.endpoint_2.0_prob"
DEAD_OF_DESEASE     = "fake_episodes.diagnosis.fields.status_DOD_prob"
# 3-class status probabilities used for multi-class entropy/Gini badges
STATUS_AWD_COL       = "fake_episodes.diagnosis.fields.status_AWD_prob"
STATUS_NED_COL       = "fake_episodes.diagnosis.fields.status_NED_prob"
SURGERY_FLAG_COL     = "surgery"
RADIOTHERAPY_FLAG_COL= "radiotherapy"
CHEMOTHERAPY_FLAG_COL= "chemotherapy"
TIMESTEP_COL         = "timestep"
HISTOLOGICAL_DIAGNOSIS_COL = "tumor_characteristics.histological_diagnosis"
# Additional categorical columns for extended aggregation
LESION_SITE_COL = "tumor_characteristics.lesion_site"
FNCLCC_GRADING_COL = "tumor_characteristics.grading_fnclcc"
# Define which lesion_site values count as extremities (assumption based on provided uniques)
EXTREMITY_SITE_SET: set[str] = {
    # Lower limb / upper limb relevant bony or regional sites
    "femur", "tibia", "knee", "Acetabulum",  # pelvic socket counted as limb girdle
    # Shoulder girdle / upper limb proximal
    "Axilla/scapula",
}

def _map_extremity_site(val: object) -> str:
    """Map raw lesion_site to aggregated category: Extremity / Non-extremity / Unknown.
    Assumptions:
      - femur, tibia, knee, Acetabulum and Axilla/scapula are treated as 'Extremity'.
      - All other non-null values become 'Non-extremity'.
      - NaN -> 'Unknown'.
    Adjust EXTREMITY_SITE_SET above if domain definition differs.
    """
    if pd.isna(val):
        return "Unknown"
    s = str(val)
    return "Extremity" if s in EXTREMITY_SITE_SET else "Non-extremity"


# ------------------------------------------------------------------ #
# Utility helpers
# ------------------------------------------------------------------ #

def min_max_normalize(series: pd.Series) -> pd.Series:
    """Linearly scale values using quantiles to the [0, 1] range."""
    q_min, q_max = series.quantile(0.05), series.quantile(0.95)
    if q_max == q_min:  # avoid division by zero
        return pd.Series(np.zeros_like(series, dtype=float), index=series.index)
    # Clip values to quantile range, then normalize
    clipped = series.clip(lower=q_min, upper=q_max)
    return (clipped - q_min) / (q_max - q_min)


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load the CSV and add a column with the normalised local‑recurrence probability."""
    df = pd.read_csv(path)
    # Global normalization across the whole dataset for endpoints
    df["LR_prob_norm_global"]  = min_max_normalize(df[LOCAL_RECURRENCE_COL])
    df["MET_prob_norm_global"] = min_max_normalize(df[METASTATIC_COL])
    df["DOD_prob_norm_global"] = min_max_normalize(df[DEAD_OF_DESEASE])
    # Back-compat for functions using prob_norm (local recurrence normalized)
    df["prob_norm"] = df["LR_prob_norm_global"]
    return df


def _make_group_label(df: pd.DataFrame) -> pd.Series:
    """Return a Series labelling each row with its treatment group of interest."""
    conditions = [
        (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 0) & (df[CHEMOTHERAPY_FLAG_COL] == 0),
        (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 1) & (df[CHEMOTHERAPY_FLAG_COL] == 0),
        (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 0) & (df[CHEMOTHERAPY_FLAG_COL] == 1),
        (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 1) & (df[CHEMOTHERAPY_FLAG_COL] == 1),
    ]
    choices = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    return np.select(conditions, choices, default="Other")

# ------------------------------------------------------------------ #
# Overall analysis
# ------------------------------------------------------------------ #

def compare_groups_overall(df: pd.DataFrame, header: str = "Overall comparison") -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Compare all four treatment groups across *all* rows."""

    surgery_only_mask = (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 0) & (df[CHEMOTHERAPY_FLAG_COL] == 0)
    surgery_rt_mask   = (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 1) & (df[CHEMOTHERAPY_FLAG_COL] == 0)
    surgery_ct_mask   = (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 0) & (df[CHEMOTHERAPY_FLAG_COL] == 1)
    surgery_rt_ct_mask = (df[SURGERY_FLAG_COL] == 1) & (df[RADIOTHERAPY_FLAG_COL] == 1) & (df[CHEMOTHERAPY_FLAG_COL] == 1)

    grp_only = df.loc[surgery_only_mask, "prob_norm"]
    grp_rt   = df.loc[surgery_rt_mask,   "prob_norm"]
    grp_ct   = df.loc[surgery_ct_mask,   "prob_norm"]
    grp_rt_ct = df.loc[surgery_rt_ct_mask, "prob_norm"]

    _print_summary_stats_all_groups(grp_only, grp_rt, grp_ct, grp_rt_ct, header=header)
    return grp_only, grp_rt, grp_ct, grp_rt_ct

# ------------------------------------------------------------------ #
# Per‑timestep analysis
# ------------------------------------------------------------------ #

def compare_groups_by_timestep(df: pd.DataFrame, header: str = "Per‑timestep comparison") -> pd.DataFrame:
    """For each timestep, compare all four groups and report mean values + p‑values."""
    df = df.copy()
    df["group"] = _make_group_label(df)

    # Keep only rows belonging to the groups of interest
    df = df[df["group"].isin(["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"])]

    results = []
    for ts, chunk in df.groupby(TIMESTEP_COL):
        grp_only = chunk.loc[chunk["group"] == "Surgery only", "prob_norm"]
        grp_rt   = chunk.loc[chunk["group"] == "Surgery + RT", "prob_norm"]
        grp_ct   = chunk.loc[chunk["group"] == "Surgery + CT", "prob_norm"]
        grp_rt_ct = chunk.loc[chunk["group"] == "Surgery + RT + CT", "prob_norm"]

        # Skip timesteps without at least some groups
        groups_with_data = [g for g in [grp_only, grp_rt, grp_ct, grp_rt_ct] if not g.empty]
        if len(groups_with_data) < 2:
            continue

        # Perform pairwise comparisons where possible
        p_only_rt = None
        p_only_ct = None
        p_only_rt_ct = None
        p_rt_ct = None
        p_rt_rt_ct = None
        p_ct_rt_ct = None

        if not grp_only.empty and not grp_rt.empty and (grp_only.nunique() > 1 or grp_rt.nunique() > 1):
            _, p_only_rt = mannwhitneyu(grp_only, grp_rt, alternative="two-sided")
        if not grp_only.empty and not grp_ct.empty and (grp_only.nunique() > 1 or grp_ct.nunique() > 1):
            _, p_only_ct = mannwhitneyu(grp_only, grp_ct, alternative="two-sided")
        if not grp_only.empty and not grp_rt_ct.empty and (grp_only.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, p_only_rt_ct = mannwhitneyu(grp_only, grp_rt_ct, alternative="two-sided")
        if not grp_rt.empty and not grp_ct.empty and (grp_rt.nunique() > 1 or grp_ct.nunique() > 1):
            _, p_rt_ct = mannwhitneyu(grp_rt, grp_ct, alternative="two-sided")
        if not grp_rt.empty and not grp_rt_ct.empty and (grp_rt.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, p_rt_rt_ct = mannwhitneyu(grp_rt, grp_rt_ct, alternative="two-sided")
        if not grp_ct.empty and not grp_rt_ct.empty and (grp_ct.nunique() > 1 or grp_rt_ct.nunique() > 1):
            _, p_ct_rt_ct = mannwhitneyu(grp_ct, grp_rt_ct, alternative="two-sided")

        results.append({
            "timestep"       : ts,
            "n_only"         : len(grp_only),
            "n_rt"           : len(grp_rt),
            "n_ct"           : len(grp_ct),
            "n_rt_ct"        : len(grp_rt_ct),
            "mean_only"      : grp_only.mean() if not grp_only.empty else np.nan,
            "mean_rt"        : grp_rt.mean() if not grp_rt.empty else np.nan,
            "mean_ct"        : grp_ct.mean() if not grp_ct.empty else np.nan,
            "mean_rt_ct"     : grp_rt_ct.mean() if not grp_rt_ct.empty else np.nan,
            "std_only"       : grp_only.std() if not grp_only.empty else np.nan,
            "std_rt"         : grp_rt.std() if not grp_rt.empty else np.nan,
            "std_ct"         : grp_ct.std() if not grp_ct.empty else np.nan,
            "std_rt_ct"      : grp_rt_ct.std() if not grp_rt_ct.empty else np.nan,
            "median_only"    : grp_only.median() if not grp_only.empty else np.nan,
            "median_rt"      : grp_rt.median() if not grp_rt.empty else np.nan,
            "median_ct"      : grp_ct.median() if not grp_ct.empty else np.nan,
            "median_rt_ct"   : grp_rt_ct.median() if not grp_rt_ct.empty else np.nan,
            "p_only_rt"      : p_only_rt,
            "p_only_ct"      : p_only_ct,
            "p_only_rt_ct"   : p_only_rt_ct,
            "p_rt_ct"        : p_rt_ct,
            "p_rt_rt_ct"     : p_rt_rt_ct,
            "p_ct_rt_ct"     : p_ct_rt_ct,
        })

    res_df = pd.DataFrame(results).sort_values("timestep").reset_index(drop=True)

    print(f"\n=== {header} ===")
    if res_df.empty:
        print("No timesteps have treatment groups with sufficient data.")
    else:
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    return res_df

# ------------------------------------------------------------------ #
# Plotting helpers
# ------------------------------------------------------------------ #
def compute_individualized_treatment_effects(
    df: pd.DataFrame,
    reference_group: str = "Surgery only",
    patient_col: str = "patient_id",
    time_col: str = TIMESTEP_COL,
    endpoint_cols: dict[str, str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compute integrated individualized treatment effects for each patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing predicted endpoint risks for each treatment
        scenario at multiple time horizons. Must include columns identifying
        the patient, timestep and treatment flags (surgery, radiotherapy and
        chemotherapy).
    reference_group : str, default "Surgery only"
        Treatment scenario to compare against.
    patient_col : str, default "patient_id"
        Column containing patient identifiers.
    time_col : str, default TIMESTEP_COL
        Column with the prediction horizon values (e.g. months from baseline).
    endpoint_cols : dict[str, str], optional
        Mapping from endpoint names to the column in ``df`` holding its
        predicted risk. Defaults to the three endpoint probability columns used
        elsewhere in this module.
    weights : dict[str, float], optional
        Optional non-negative weights for computing a composite ITE across
        multiple endpoints. Keys should match those of ``endpoint_cols`` and
        sum to one.

    Returns
    -------
    pd.DataFrame
        Long-form table with one row per patient and treatment scenario
        containing the integrated ITE for each endpoint. If ``weights`` are
        supplied, an additional ``composite_ite`` column summarises the weighted
        effect across endpoints.
    """
    df = df.copy()
    df["group"] = _make_group_label(df)

    groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    df = df[df["group"].isin(groups)]

    if endpoint_cols is None:
        endpoint_cols = {
            "local_recurrence": LOCAL_RECURRENCE_COL,
            "metastasis": METASTATIC_COL,
            "death_of_disease": DEAD_OF_DESEASE,
        }

    results: list[pd.DataFrame] = []

    for endpoint, col in endpoint_cols.items():
        pivot = df.pivot_table(index=[patient_col, time_col], columns="group", values=col)
        if reference_group not in pivot.columns:
            continue
        ref_vals = pivot[reference_group]
        for treatment in groups:
            if treatment == reference_group or treatment not in pivot.columns:
                continue
            delta = pivot[treatment] - ref_vals
            delta = delta.dropna().reset_index(name="delta")

            def _integrate(group: pd.DataFrame) -> float:
                times = group[time_col].values
                vals = group["delta"].values
                if len(vals) == 1:
                    return float(vals[0])
                return float(np.trapz(vals, x=times) / (times[-1] - times[0]))

            per_patient = delta.groupby(patient_col).apply(_integrate).reset_index(name="ite")
            per_patient["treatment"] = treatment
            per_patient["endpoint"] = endpoint
            results.append(per_patient)

    if not results:
        return pd.DataFrame(columns=[patient_col, "treatment", "endpoint", "ite"])

    result = pd.concat(results, ignore_index=True)

    if weights:
        weight_series = pd.Series(weights, dtype=float)
        composite = (
            result.pivot_table(index=[patient_col, "treatment"], columns="endpoint", values="ite")
                  .reindex(columns=weight_series.index)
                  .mul(weight_series, axis=1)
                  .sum(axis=1)
                  .reset_index(name="composite_ite")
        )
        result = result.merge(composite, on=[patient_col, "treatment"], how="left")

    return result



def plot_composite_benefit_waterfall(
    ite_df: pd.DataFrame,
    treatment: str,
    histology: str | None = None,
    out_path: str = "composite_benefit_waterfall.png",
    patient_col: str = "patient_id",
) -> None:
    """Plot a waterfall of composite individualized treatment effects.

    Parameters
    ----------
    ite_df : pd.DataFrame
        Output of :func:`compute_individualized_treatment_effects` containing a
        ``composite_ite`` column.
    treatment : str
        Treatment scenario to visualise (e.g. "Surgery + RT").
    histology : str, optional
        Histology label used for the plot title.
    out_path : str, default "composite_benefit_waterfall.png"
        File path where the figure will be saved. If no directory is provided
        the figure is written to ``imgs/``.
    patient_col : str, default "patient_id"
        Column identifying patients in ``ite_df``.
    """
    data = ite_df[ite_df["treatment"] == treatment]
    if "composite_ite" not in data.columns or data.empty:
        print("plot_composite_benefit_waterfall: no composite_ite values to plot.")
        return

    data = (
        data[[patient_col, "composite_ite"]]
        .drop_duplicates(subset=[patient_col])
        .sort_values("composite_ite", ascending=False)
    )

    colors = data["composite_ite"].apply(
        lambda x: "#2ca02c" if x > 0 else "#d62728"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(data)), data["composite_ite"], color=colors, edgecolor="black")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Patients (sorted)")
    ax.set_ylabel("Composite benefit")
    title = f"Composite benefit: {treatment}"
    if histology:
        title += f" – {histology}"
    ax.set_title(title)

    n_pos = (data["composite_ite"] > 0).sum()
    n_total = len(data)
    frac = n_pos / n_total if n_total else 0.0
    ax.text(
        0.98,
        0.95,
        f"{n_pos}/{n_total} ({frac:.0%}) above zero",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    plt.tight_layout()
    d = os.path.dirname(out_path)
    if d == "":
        out_path = os.path.join("imgs", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nComposite benefit waterfall saved to {out_path}")

def plot_distribution(df: pd.DataFrame, out_path: str = "local_recurrence_distribution.png") -> None:
    """Histogram (original vs normalised) for a quick sanity‑check."""
    
    # Set up scientific plotting style
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Original distribution
    ax1.hist(df[LOCAL_RECURRENCE_COL], bins=30, edgecolor="black", alpha=0.7, color='#1f77b4')
    ax1.set_xlabel("Local Recurrence Probability (Original)", fontweight='bold')
    ax1.set_ylabel("Frequency", fontweight='bold')
    ax1.set_title("Original Distribution", fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.set_axisbelow(True)

    # Normalized distribution
    ax2.hist(df["prob_norm"], bins=30, edgecolor="black", alpha=0.7, color='#ff7f0e')
    ax2.set_xlabel("Local Recurrence Probability (Normalized)", fontweight='bold')
    ax2.set_ylabel("Frequency", fontweight='bold')
    ax2.set_title("Normalized Distribution", fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.set_axisbelow(True)

    plt.suptitle("Distribution of Local Recurrence Probabilities", fontweight='bold', y=1.02)
    plt.tight_layout()
    # ensure outputs go to imgs/ if no directory is provided
    def _ensure_imgs_path(p: str) -> str:
        d = os.path.dirname(p)
        if d == "":
            p = os.path.join("imgs", p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    out_path = _ensure_imgs_path(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    
    print(f"\nDistribution plot saved to {out_path}")


def plot_boxplot(grp_only: pd.Series, grp_rt: pd.Series, grp_ct: pd.Series, grp_rt_ct: pd.Series, out_path: str = "local_recurrence_boxplot.png") -> None:
    """Box‑plot comparing all four groups (overall)."""
    
    # Set up scientific plotting style
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 12,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    plt.figure(figsize=(8, 6))
    
    # Filter out empty groups
    groups = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Professional color palette
    group_colors = []
    
    if not grp_only.empty:
        groups.append(grp_only)
        labels.append("Surgery\nonly")
        group_colors.append(colors[0])
    if not grp_rt.empty:
        groups.append(grp_rt)
        labels.append("Surgery\n+ RT")
        group_colors.append(colors[1])
    if not grp_ct.empty:
        groups.append(grp_ct)
        labels.append("Surgery\n+ CT")
        group_colors.append(colors[2])
    if not grp_rt_ct.empty:
        groups.append(grp_rt_ct)
        labels.append("Surgery\n+ RT + CT")
        group_colors.append(colors[3])
    
    if groups:
        bp = plt.boxplot(groups, labels=labels, patch_artist=True, showmeans=True, meanline=True,
                        medianprops={'color': 'white', 'linewidth': 1.5},
                        meanprops={'color': 'black', 'linewidth': 1.5, 'linestyle': '--'},
                        flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6})
        
        # Color the boxes professionally
        for patch, color in zip(bp['boxes'], group_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)
        
        plt.ylabel("Normalized Local Recurrence Probability", fontweight='bold')
        plt.title("Local Recurrence Probability by Treatment Modality", fontweight='bold', pad=20)
        plt.xticks(rotation=0)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.gca().set_axisbelow(True)
        plt.ylim(-0.05, 1.05)
        
        plt.tight_layout()
        # Route to imgs/ by default
        d = os.path.dirname(out_path)
        if d == "":
            out_path = os.path.join("imgs", out_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
        
        print(f"\nBox‑plot saved to {out_path}")
    else:
        print("No data available for box plot")


def plot_timeseries(df: pd.DataFrame, out_path: str = "local_recurrence_timeseries.png") -> None:
    """Box‑plot of normalised probability over time for each group at each timestep."""
    if df.empty:
        print("Skipping timeseries plot – dataframe is empty.")
        return

    df = df.copy()
    # Choose normalized LR column if available
    prob_col = "LR_prob_norm_global" if "LR_prob_norm_global" in df.columns else "prob_norm"
    if prob_col not in df.columns and LOCAL_RECURRENCE_COL in df.columns:
        df["LR_prob_norm_global"] = min_max_normalize(df[LOCAL_RECURRENCE_COL])
        prob_col = "LR_prob_norm_global"

    df["group"] = _make_group_label(df)
    df = df[df["group"].isin(["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"])]

    if df.empty:
        print(f"Skipping timeseries plot for {out_path} – no data for the specified groups.")
        return
    
    timesteps = sorted(df[TIMESTEP_COL].dropna().unique())
    groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    # Professional color palette for scientific publications
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    group_labels = ["", "", "", ""]  # Empty labels since we'll use legend only
    
    # Calculate optimal figure size for A4 page (landscape)
    n_timesteps = len(timesteps)
    # A4 landscape: 11.7 x 8.3 inches, leave margins
    max_width = 10.5  # inches
    width_per_subplot = min(max_width / n_timesteps, 2.5)
    fig_width = width_per_subplot * n_timesteps
    fig_height = 4.5  # Compact height for A4
    
    fig, axes = plt.subplots(1, n_timesteps, figsize=(fig_width, fig_height), sharey=True)
    if n_timesteps == 1:
        axes = [axes]
    
    # Set up the plot style for scientific publication
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    for i, ts in enumerate(timesteps):
        ts_data = df[df[TIMESTEP_COL] == ts]
        
        box_data = []
        box_labels = []
        box_colors = []
        
        for j, group in enumerate(groups):
            group_data = ts_data[ts_data["group"] == group][prob_col]
            if not group_data.empty:
                box_data.append(group_data.values)
                box_labels.append(group_labels[j])
                box_colors.append(colors[j])
        
        if box_data:
            bp = axes[i].boxplot(box_data, labels=box_labels, patch_artist=True, 
                               showmeans=True, meanline=True, showfliers=True,
                               medianprops={'color': 'white', 'linewidth': 1.5},
                               meanprops={'color': 'black', 'linewidth': 1.5, 'linestyle': '--'},
                               flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6})
            
            # Color the boxes with professional styling
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.0)
        
        # Improve subplot formatting
        axes[i].set_title(f"t = {ts}", fontweight='bold', pad=10)
        axes[i].set_xticklabels([])  # Remove x-axis labels
        axes[i].grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        axes[i].set_axisbelow(True)
        
        # Remove y-axis label from all subplots
        
        # Set consistent y-axis limits across all subplots
        axes[i].set_ylim(-0.05, 1.05)
    
    # Add overall title with better positioning
    fig.suptitle("Local Recurrence Probability by Treatment Modality Over Time", 
                fontweight='bold', fontsize=14, y=0.95)
    
    # Create a fixed legend for treatment groups
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', alpha=0.7, label=label)
                      for color, label in zip(colors, ["S", "S + RT", "S + CT", "S + RT + CT"])]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.85),
               frameon=True, fancybox=True, shadow=True, ncol=1)
    
    # Optimize layout for A4 page
    plt.tight_layout(rect=[0, 0, 0.85, 0.92])
    
    # Save with high quality for publication
    d = os.path.dirname(out_path)
    if d == "":
        out_path = os.path.join("imgs", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white', 
                edgecolor='none', format='png')
    
    print(f"\nTimeseries boxplot saved to {out_path}")


def plot_three_endpoint_timeseries(
    df: pd.DataFrame,
    histology: str | None = None,
    out_path: str = "three_endpoint_timeseries.png"
) -> None:
    """Create a 3-row timeseries figure: LR, metastasis, and DOD over time.
    
    Each row shows all timesteps horizontally with grouped box plots by treatment arm.
    Uses global-normalized endpoint probabilities and consistent colors/legend.
    """
    if df.empty:
        print("plot_three_endpoint_timeseries: Empty DataFrame; nothing to plot.")
        return

    work = df.copy()
    
    # Ensure global-normalized columns exist
    if "LR_prob_norm_global" not in work.columns and LOCAL_RECURRENCE_COL in work.columns:
        work["LR_prob_norm_global"] = min_max_normalize(work[LOCAL_RECURRENCE_COL])
    if "MET_prob_norm_global" not in work.columns and METASTATIC_COL in work.columns:
        work["MET_prob_norm_global"] = min_max_normalize(work[METASTATIC_COL])
    if "DOD_prob_norm_global" not in work.columns and DEAD_OF_DESEASE in work.columns:
        work["DOD_prob_norm_global"] = min_max_normalize(work[DEAD_OF_DESEASE])

    # if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
    #     work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]
    
    work["group"] = _make_group_label(work)
    arms_full = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms_full)]
    
    if work.empty:
        print("plot_three_endpoint_timeseries: No rows for the specified groups/histology.")
        return

    timesteps = sorted(work[TIMESTEP_COL].dropna().unique())
    if not timesteps:
        print("plot_three_endpoint_timeseries: No timesteps available.")
        return

    # Endpoint definitions
    endpoints = [
        ("Local recurrence", "LR_prob_norm_global"),
        ("Metastasis", "MET_prob_norm_global"),
        ("Dead of Disease", "DOD_prob_norm_global"),
    ]
    
    # Color scheme and labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    group_labels = ["", "", "", ""]  # Empty labels for box plots
    
    # Calculate figure size
    n_timesteps = len(timesteps)
    max_width = 12.0  # inches
    width_per_subplot = min(max_width / n_timesteps, 2.0)
    fig_width = width_per_subplot * n_timesteps
    fig_height = 17.0  # 3 rows, ~4 inches each
    
    # Create 3 rows of subplots
    fig, axes = plt.subplots(3, n_timesteps, figsize=(fig_width, fig_height), 
                            sharex=True, sharey=True)
    if n_timesteps == 1:
        axes = axes.reshape(3, 1)
    
    # Set up plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    # Plot each endpoint
    for row_idx, (endpoint_name, endpoint_col) in enumerate(endpoints):
        if endpoint_col not in work.columns:
            print(f"Skipping {endpoint_name}: column {endpoint_col} not found.")
            continue
            
        for col_idx, ts in enumerate(timesteps):
            ax = axes[row_idx, col_idx]
            ts_data = work[work[TIMESTEP_COL] == ts]
            
            box_data = []
            box_colors = []
            
            for j, group in enumerate(arms_full):
                group_data = ts_data[ts_data["group"] == group][endpoint_col].dropna()
                if not group_data.empty:
                    box_data.append(group_data.values)
                    box_colors.append(colors[j])
            
            if box_data:
                bp = ax.boxplot(
                    box_data,
                    labels=group_labels[:len(box_data)],
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=True,
                    medianprops={'color': 'white', 'linewidth': 1.2},
                    meanprops={'color': 'black', 'linewidth': 1.2, 'linestyle': '--'},
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
                )
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.0)
            
            # Formatting - timestep label starts from 1
            if row_idx == 0:  # Top row gets timestep titles
                ax.set_title(f"t = {col_idx + 1}", fontweight='bold', pad=10)
            if col_idx == 0:  # Left column gets endpoint labels
                ax.set_ylabel(endpoint_name, fontweight='bold')
            
            # Endpoint-specific H/G badges (binary entropy/Gini averaged per arm)
            for j, group in enumerate(arms_full):
                group_series = ts_data[ts_data["group"] == group][endpoint_col].dropna()
                if group_series.empty:
                    continue
                s = pd.to_numeric(group_series, errors='coerce').dropna().clip(0.0, 1.0)
                if s.empty:
                    continue
                H_mean = float(_binary_entropy(s).mean())
                G_mean = float(_gini_impurity_binary(s).mean())
                H_norm = H_mean / np.log(2.0) if np.isfinite(H_mean) else np.nan
                G_norm = G_mean / 0.5 if np.isfinite(G_mean) else np.nan

                if np.isfinite(H_norm) and np.isfinite(G_norm):
                    # Position badges below the x-axis, stacked per arm
                    x_pos = 1
                    y_pos = -0.2 - j * 0.1
                    ax.add_patch(mpatches.Rectangle(
                        (x_pos, y_pos), 0.18, 0.06,
                        facecolor=colors[j], edgecolor='black',
                        transform=ax.get_xaxis_transform(), clip_on=False, lw=0.5
                    ))
                    h_text = f"$\\mathbf{{H}}$:{H_norm:.3f}"
                    g_text = f"$\\mathbf{{G}}$:{G_norm:.3f}"
                    ax.text(x_pos+0.2, y_pos+0.05, h_text,
                            ha='left', va='center', fontsize=6,
                            transform=ax.get_xaxis_transform(), color='black')
                    ax.text(x_pos+0.2, y_pos+0.01, g_text,
                            ha='left', va='center', fontsize=6,
                            transform=ax.get_xaxis_transform(), color='black')

            ax.set_xticklabels([])
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
    
    # Overall title
    title_hist = "All histologies" if histology is None else str(histology)
    fig.suptitle(f"", 
                fontweight='bold', fontsize=14, y=0.95)
    
    # Create legend below the plots
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', alpha=0.7, label=label)
                      for color, label in zip(colors, ["S", "S + RT", "S + CT", "S + RT + CT"])]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               frameon=True, fancybox=True, shadow=True, ncol=4)
    
    # Layout and save
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    # Route to imgs/
    d = os.path.dirname(out_path)
    if d == "":
        out_path = os.path.join("imgs", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"\nThree-endpoint timeseries saved to {out_path}")


def plot_three_endpoint_timeseries_no_badges(
    df: pd.DataFrame,
    histology: str | None = None,
    out_path: str = "three_endpoint_timeseries_no_badges.png"
) -> None:
    """Create a 3-row timeseries figure: LR, metastasis, and DOD over time without H/G badges.
    
    Each row shows all timesteps horizontally with grouped box plots by treatment arm.
    Uses global-normalized endpoint probabilities and consistent colors/legend.
    """
    if df.empty:
        print("plot_three_endpoint_timeseries_no_badges: Empty DataFrame; nothing to plot.")
        return

    work = df.copy()
    
    # Ensure global-normalized columns exist
    if "LR_prob_norm_global" not in work.columns and LOCAL_RECURRENCE_COL in work.columns:
        work["LR_prob_norm_global"] = min_max_normalize(work[LOCAL_RECURRENCE_COL])
    if "MET_prob_norm_global" not in work.columns and METASTATIC_COL in work.columns:
        work["MET_prob_norm_global"] = min_max_normalize(work[METASTATIC_COL])
    if "DOD_prob_norm_global" not in work.columns and DEAD_OF_DESEASE in work.columns:
        work["DOD_prob_norm_global"] = min_max_normalize(work[DEAD_OF_DESEASE])

    # if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
    #     work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]
    
    work["group"] = _make_group_label(work)
    arms_full = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms_full)]
    
    if work.empty:
        print("plot_three_endpoint_timeseries_no_badges: No rows for the specified groups/histology.")
        return

    timesteps = sorted(work[TIMESTEP_COL].dropna().unique())
    if not timesteps:
        print("plot_three_endpoint_timeseries_no_badges: No timesteps available.")
        return

    # Endpoint definitions
    endpoints = [
        ("Local recurrence", "LR_prob_norm_global"),
        ("Metastasis", "MET_prob_norm_global"),
        ("Dead of Disease", "DOD_prob_norm_global"),
    ]
    
    # Color scheme and labels
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    group_labels = ["", "", "", ""]  # Empty labels for box plots
    
    # Calculate figure size
    n_timesteps = len(timesteps)
    max_width = 12.0  # inches
    width_per_subplot = min(max_width / n_timesteps, 2.0)
    fig_width = width_per_subplot * n_timesteps
    fig_height = 17.0  # 3 rows, ~4 inches each
    
    # Create 3 rows of subplots
    fig, axes = plt.subplots(3, n_timesteps, figsize=(fig_width, fig_height), 
                            sharex=True, sharey=True)
    if n_timesteps == 1:
        axes = axes.reshape(3, 1)
    
    # Set up plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.3
    })
    
    # Plot each endpoint
    for row_idx, (endpoint_name, endpoint_col) in enumerate(endpoints):
        if endpoint_col not in work.columns:
            print(f"Skipping {endpoint_name}: column {endpoint_col} not found.")
            continue
            
        for col_idx, ts in enumerate(timesteps):
            ax = axes[row_idx, col_idx]
            ts_data = work[work[TIMESTEP_COL] == ts]
            
            box_data = []
            box_colors = []
            
            for j, group in enumerate(arms_full):
                group_data = ts_data[ts_data["group"] == group][endpoint_col].dropna()
                if not group_data.empty:
                    box_data.append(group_data.values)
                    box_colors.append(colors[j])
            
            if box_data:
                bp = ax.boxplot(
                    box_data,
                    labels=group_labels[:len(box_data)],
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    showfliers=True,
                    medianprops={'color': 'white', 'linewidth': 1.2},
                    meanprops={'color': 'black', 'linewidth': 1.2, 'linestyle': '--'},
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
                )
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                    patch.set_edgecolor('black')
                    patch.set_linewidth(1.0)
            
            # Formatting - timestep label starts from 1
            if row_idx == 0:  # Top row gets timestep titles
                ax.set_title(f"t = {col_idx + 1}", fontweight='bold', pad=10)
            if col_idx == 0:  # Left column gets endpoint labels
                ax.set_ylabel(endpoint_name, fontweight='bold')
            
            ax.set_xticklabels([])
            ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.3)
            ax.set_axisbelow(True)
            ax.set_ylim(0.0, 1.0)
    
    # Overall title
    title_hist = "All histologies" if histology is None else str(histology)
    fig.suptitle(f"", 
                fontweight='bold', fontsize=14, y=0.95)
    
    # Create legend below the plots
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, edgecolor='black', alpha=0.7, label=label)
                      for color, label in zip(colors, ["S", "S + RT", "S + CT", "S + RT + CT"])]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               frameon=True, fancybox=True, shadow=True, ncol=4)
    
    # Layout and save
    plt.tight_layout(rect=[0, 0.1, 1, 0.92])
    
    # Route to imgs/
    d = os.path.dirname(out_path)
    if d == "":
        out_path = os.path.join("imgs", out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
    plt.close()
    print(f"\nThree-endpoint timeseries (no badges) saved to {out_path}")


def plot_grouped_boxplots_per_timestep(
    df: pd.DataFrame,
    endpoint_col: str = "LR_prob_norm_global",
    histology: str | None = None,
    out_dir: str = "imgs/per_timestep",
    file_prefix: str = "lr",
) -> None:
    """Create a figure per timestep with grouped box plots by treatment arm, colored with legend.

    - endpoint_col should be one of the global-normalized columns (e.g., LR_prob_norm_global, MET_prob_norm_global, DOD_prob_norm_global)
    - Saves one PNG per timestep under out_dir.
    """
    if df.empty:
        print("plot_grouped_boxplots_per_timestep: Empty DataFrame; nothing to plot.")
        return

    work = df.copy()
    # Ensure endpoint column exists; if not and it's one of known globals, compute globally
    if endpoint_col not in work.columns:
        if endpoint_col == "LR_prob_norm_global" and LOCAL_RECURRENCE_COL in work.columns:
            work[endpoint_col] = min_max_normalize(work[LOCAL_RECURRENCE_COL])
        elif endpoint_col == "MET_prob_norm_global" and METASTATIC_COL in work.columns:
            work[endpoint_col] = min_max_normalize(work[METASTATIC_COL])
        elif endpoint_col == "DOD_prob_norm_global" and DEAD_OF_DESEASE in work.columns:
            work[endpoint_col] = min_max_normalize(work[DEAD_OF_DESEASE])
        else:
            print(f"plot_grouped_boxplots_per_timestep: Missing endpoint column {endpoint_col}.")
            return

    if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
        work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]
    work["group"] = _make_group_label(work)
    arms_full = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms_full)]
    if work.empty:
        print("plot_grouped_boxplots_per_timestep: No rows for the specified groups/histology.")
        return

    timesteps = sorted(pd.unique(work[TIMESTEP_COL].dropna()))
    if not timesteps:
        print("plot_grouped_boxplots_per_timestep: No timesteps available.")
        return

    # Color palette and legend mapping
    color_map = {
        "Surgery only": "#1f77b4",
        "Surgery + RT": "#ff7f0e",
        "Surgery + CT": "#2ca02c",
        "Surgery + RT + CT": "#d62728",
    }

    os.makedirs(out_dir, exist_ok=True)

    for ts in timesteps:
        sub = work[work[TIMESTEP_COL] == ts]
        if sub.empty:
            continue

        data = []
        labels = []
        colors = []
        ns = []
        for arm in arms_full:
            vals = sub.loc[sub["group"] == arm, endpoint_col].dropna().values
            if vals.size > 0:
                data.append(vals)
                labels.append(arm)
                colors.append(color_map[arm])
                ns.append(vals.size)

        if not data:
            continue

        plt.figure(figsize=(7.5, 5.0))
        bp = plt.boxplot(
            data,
            labels=["S", "S + RT", "S + CT", "S + RT + CT"][:len(data)],
            patch_artist=True,
            showmeans=True,
            meanline=True,
            medianprops={'color': 'white', 'linewidth': 1.4},
            meanprops={'color': 'black', 'linewidth': 1.4, 'linestyle': '--'},
            flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
        )
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.0)

        # n under each box
        for i, n_i in enumerate(ns, start=1):
            plt.gca().text(i, -0.08, f"n={n_i}", ha='center', va='top', transform=plt.gca().get_xaxis_transform(), fontsize=9, color='0.25')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color_map[arm], edgecolor='black', alpha=0.7, label=lbl)
                           for arm, lbl in zip(arms_full, ["S", "S + RT", "S + CT", "S + RT + CT"])]
        plt.legend(handles=legend_elements, loc='upper right', frameon=True, fontsize=9)

        # Aesthetics
        plt.ylabel("Predicted probability")
        plt.ylim(0.0, 1.0)
        title_ep = {"LR_prob_norm_global": "Local recurrence",
                    "MET_prob_norm_global": "Metastasis",
                    "DOD_prob_norm_global": "Disease-specific death"}.get(endpoint_col, endpoint_col)
        title_hist = "All histologies" if histology is None else str(histology)
        plt.title(f"{title_ep} at t = {int(ts)} mo – {title_hist}", fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.gca().set_axisbelow(True)

        fname = f"{file_prefix}_boxes_t{int(ts)}.png"
        out_path = os.path.join(out_dir, fname)
        plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Saved per-timestep boxplot to {out_path}")

# ------------------------------------------------------------------ #
# Composite summary figure (three rows of grouped box plots at horizon)
# ------------------------------------------------------------------ #

def _select_horizon_slice(df: pd.DataFrame, horizon_months: int) -> tuple[pd.DataFrame, int]:
    """Return rows at the requested timestep (or nearest if not present) and the chosen timestep.

    Assumes TIMESTEP_COL stores the horizon in months (or an integer index). If the exact
    value isn't present, the nearest available timestep is selected.
    """
    if df.empty or TIMESTEP_COL not in df.columns:
        return df, horizon_months
    ts_values = np.array(sorted(df[TIMESTEP_COL].dropna().unique()))
    if ts_values.size == 0:
        return df, horizon_months
    if horizon_months in ts_values:
        chosen = int(horizon_months)
    else:
        # nearest by absolute difference
        chosen = int(ts_values[np.argmin(np.abs(ts_values - horizon_months))])
    return df[df[TIMESTEP_COL] == chosen].copy(), chosen


def _normalize_prob_vector(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, None)
    s = p.sum()
    if s <= 0 or not np.isfinite(s):
        # fallback to uniform over 3 classes
        return np.full_like(p, 1.0 / len(p))
    return p / s


def _multiclass_entropy_gini(mean_vec: np.ndarray) -> tuple[float, float]:
    """Return normalized (H/ln 3, G/(2/3)) for a 3-class mean probability vector."""
    p = _normalize_prob_vector(mean_vec)
    # Shannon entropy (natural log), normalized by ln(3)
    H = -np.sum(np.where(p > 0, p * np.log(p), 0.0))
    H_norm = H / np.log(3.0)
    # Multi-class Gini impurity: 1 - sum p_i^2; max at uniform = 1 - 1/3 = 2/3
    G = 1.0 - np.sum(p ** 2)
    G_norm = G / (2.0 / 3.0)
    return float(H_norm), float(G_norm)


def plot_summary_figure(
    df: pd.DataFrame,
    horizon_months: int = 42,
    histology: str | None = None,
    out_path_png: str = "imgs/summary_figure_42mo.png",
    out_path_svg: str | None = None,
    include_jitter: bool = True,
) -> None:
    """Create a single-page grayscale summary figure with three stacked rows:
    1) Local recurrence probability
    2) Metastasis probability
    3) Disease-specific death probability

    Each row shows grouped box plots by treatment arm at a fixed horizon (default 42 mo).
    Overlaid: small dot for the mean, n under each box, and two dispersion badges to the right:
    a circle (H) for normalized entropy and a square (G) for normalized Gini, computed from the
    arm-level mean three-class distribution (NED/AWD/DOD) at that horizon.

    Notes:
    - Grayscale aesthetics; y-axis fixed to [0,1] for all rows.
    - Shared x-axis (arms); tick labels shown on the bottom row only.
    - If the requested horizon isn't present, the nearest timestep is used.
    - If histology is provided, the data are filtered to that category; otherwise, all histologies.
    """
    if df.empty:
        print("plot_summary_figure: Empty DataFrame; nothing to plot.")
        return

    # Ensure global-normalized columns exist on the full dataset first
    if "LR_prob_norm_global" not in df.columns:
        df["LR_prob_norm_global"]  = min_max_normalize(df[LOCAL_RECURRENCE_COL])
    if "MET_prob_norm_global" not in df.columns:
        df["MET_prob_norm_global"] = min_max_normalize(df[METASTATIC_COL])
    if "DOD_prob_norm_global" not in df.columns:
        df["DOD_prob_norm_global"] = min_max_normalize(df[DEAD_OF_DESEASE])

    work = df.copy()
    if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
        work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]
    # Label treatment arms and keep the four of interest
    work["group"] = _make_group_label(work)
    arms_full = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms_full)]
    if work.empty:
        print("plot_summary_figure: No rows for the specified groups/histology.")
        return

    # Choose the horizon slice
    slice_df, chosen_ts = _select_horizon_slice(work, horizon_months)
    if slice_df.empty:
        print("plot_summary_figure: No rows at or near the requested horizon.")
        return

    # Columns needed for endpoints and 3-class status
    required_cols = [
        LOCAL_RECURRENCE_COL, METASTATIC_COL, DEAD_OF_DESEASE,
        STATUS_AWD_COL, STATUS_NED_COL
    ]
    missing = [c for c in required_cols if c not in slice_df.columns]
    if missing:
        print(f"plot_summary_figure: Missing required columns: {missing}")
        return

    # Aggregate to per-patient per-arm at this horizon
    if "patient_id" in slice_df.columns:
        group_keys = ["patient_id", "group"]
    else:
        # fallback: per-row treated as a pseudo-patient
        group_keys = ["group"]

    # Mean per patient per arm at the horizon for endpoints and their global-normalized versions
    # Build a safe list of columns that actually exist
    cols_needed = [
        LOCAL_RECURRENCE_COL, METASTATIC_COL, DEAD_OF_DESEASE,
        "LR_prob_norm_global", "MET_prob_norm_global", "DOD_prob_norm_global",
        STATUS_NED_COL, STATUS_AWD_COL
    ]
    cols_present = [c for c in cols_needed if c in slice_df.columns]
    if not cols_present:
        print("plot_summary_figure: No required columns present at the selected horizon.")
        return
    try:
        agg = (slice_df
               .groupby(group_keys, dropna=False)[cols_present]
               .mean()
               .reset_index())
    except Exception as e:
        print(f"plot_summary_figure: Failed to aggregate per-patient/arm stats: {e}")
        return

    # Prepare data arrays for box plots per arm (using global-normalized probabilities)
    endpoints = [
            ("Local recurrence", "LR_prob_norm_global"),
            ("Metastasis", "MET_prob_norm_global"),
            ("Disease-specific death", "DOD_prob_norm_global"),
    ]

    # Compute arm-level mean 3-class distribution (NED, AWD, DOD) across patients
    arm_badges: dict[str, tuple[float, float]] = {}  # arm -> (H_norm, G_norm)
    for arm in arms_full:
        sub = agg[agg["group"] == arm]
        if sub.empty:
            arm_badges[arm] = (np.nan, np.nan)
            continue
        # mean across patients of each class prob at horizon - ensure scalar values
        ned_mean = float(sub[STATUS_NED_COL].mean()) if STATUS_NED_COL in sub.columns and not sub[STATUS_NED_COL].empty else 0.0
        awd_mean = float(sub[STATUS_AWD_COL].mean()) if STATUS_AWD_COL in sub.columns and not sub[STATUS_AWD_COL].empty else 0.0
        dod_mean = float(sub[DEAD_OF_DESEASE].mean()) if DEAD_OF_DESEASE in sub.columns and not sub[DEAD_OF_DESEASE].empty else 0.0
        mean_vec = np.array([ned_mean, awd_mean, dod_mean])
        arm_badges[arm] = _multiclass_entropy_gini(mean_vec)

    # Figure setup (grayscale, publication-friendly)
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 14,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.25,
    })

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 9.0), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # X positions for the four arms
    x_pos = np.arange(1, len(arms_full) + 1)
    xtick_labels = ["S", "S + RT", "S + CT", "S + RT + CT"]

    # grayscale facecolors for boxes
    face_colors = ['0.75', '0.6', '0.45', '0.3']

    for row_idx, (title, col) in enumerate(endpoints):
        ax = axes[row_idx]
        # collect data arrays (per-patient) for each arm
        data_arrays = []
        positions = []
        colors = []
        ns = []
        means = []
        for i, arm in enumerate(arms_full):
            if col not in agg.columns:
                vals = np.array([])
            else:
                vals = agg.loc[agg["group"] == arm, col].dropna().values
            n_i = vals.size
            ns.append(n_i)
            if n_i > 0:
                data_arrays.append(vals)
                positions.append(x_pos[i])
                colors.append(face_colors[i])
                means.append(float(np.mean(vals)))
            else:
                means.append(np.nan)

        # Jittered dots behind boxes
        if include_jitter:
            for i, arm in enumerate(arms_full):
                if col not in agg.columns:
                    vals = np.array([])
                else:
                    vals = agg.loc[agg["group"] == arm, col].dropna().values
                if vals.size == 0:
                    continue
                jitter = (np.random.rand(vals.size) - 0.5) * 0.20  # +/- 0.1 around the x
                ax.scatter(np.full(vals.size, x_pos[i]) + jitter, vals,
                           s=10, alpha=0.15, color='0.1', zorder=1)

        # Box plots
        if data_arrays:
            bp = ax.boxplot(
                data_arrays,
                positions=positions,
                widths=0.45,
                patch_artist=True,
                showmeans=False,  # we'll place mean dot ourselves
                medianprops={'color': 'black', 'linewidth': 1.2},
                whiskerprops={'color': '0.2', 'linewidth': 1.0},
                capprops={'color': '0.2', 'linewidth': 1.0},
                boxprops={'linewidth': 1.0, 'edgecolor': '0.2'},
                flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3, 'markerfacecolor': '0.2', 'markeredgecolor': '0.2'},
            )
            for patch, c in zip(bp['boxes'], colors):
                patch.set_facecolor(c)
        
        # Mean dots
        for i, m in enumerate(means):
            if np.isfinite(m):
                ax.scatter([x_pos[i]], [m], s=18, color='0.0', zorder=3)

        # n under each box (use axis transform to place below the axis)
        for i, n_i in enumerate(ns):
            ax.text(x_pos[i], -0.08, f"n={n_i}", ha='center', va='top',
                    transform=ax.get_xaxis_transform(), fontsize=8, color='0.2')

        # Dispersion badges (H and G) to the right of each position
        for i, arm in enumerate(arms_full):
            H_norm, G_norm = arm_badges.get(arm, (np.nan, np.nan))
            if not np.isfinite(H_norm) and not np.isfinite(G_norm):
                continue
            # x offsets to the right of the box
            xH = x_pos[i] + 0.38
            xG = x_pos[i] + 0.68
            yH = 0.92  # place high to avoid overlapping data; use axes fraction
            # Draw circle and square markers with labels; values next to them
            ax.scatter([xH], [yH], s=28, color='0.1', marker='o', transform=ax.get_yaxis_transform(), zorder=4)
            ax.text(xH + 0.03, yH, f"H {H_norm:.2f}", va='center', ha='left', fontsize=8,
                    transform=ax.get_yaxis_transform(), color='0.1')
            ax.scatter([xG], [yH], s=28, color='0.1', marker='s', transform=ax.get_yaxis_transform(), zorder=4)
            ax.text(xG + 0.03, yH, f"G {G_norm:.2f}", va='center', ha='left', fontsize=8,
                    transform=ax.get_yaxis_transform(), color='0.1')

        # Aesthetics
        ax.set_title(title, fontweight='bold', loc='left')
        ax.grid(True, axis='y', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(0.5, len(arms_full) + 1.1)
        # Small horizon label
        ax.text(0.01, 0.98, f"t = {chosen_ts} mo", transform=ax.transAxes, va='top', ha='left', fontsize=9, color='0.25')

        if row_idx < 2:
            ax.set_xticklabels([])
        else:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(xtick_labels)
            ax.set_xlabel("Treatment arm")

        if row_idx == 1:
            ax.set_ylabel("Predicted probability")

    # Caption
    caption = (
        ""
    ).format(mo=chosen_ts)
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9, color='0.25', wrap=True)

    # Optional overall title
    title_bits = ["Summary at t = {} mo".format(chosen_ts)]
    if histology is None:
        title_bits.append("All histologies")
    else:
        title_bits.append(str(histology))
    fig.suptitle(" – ".join(title_bits), y=0.995, fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    # Ensure outputs go to imgs/ if paths don't specify a directory
    def _ensure_imgs_path(p: str) -> str:
        d = os.path.dirname(p)
        if d == "":
            p = os.path.join("imgs", p)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p
    out_path_png = _ensure_imgs_path(out_path_png)
    fig.savefig(out_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    if out_path_svg:
        out_path_svg = _ensure_imgs_path(out_path_svg)
        fig.savefig(out_path_svg, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Summary figure saved to {out_path_png}{' and ' + out_path_svg if out_path_svg else ''}")

# ------------------------------------------------------------------ #
# Summary‑stats printer
# ------------------------------------------------------------------ #

def _print_summary_stats_all_groups(grp_only: pd.Series, grp_rt: pd.Series, grp_ct: pd.Series, grp_rt_ct: pd.Series, header: str = "") -> None:
    if header:
        print(f"\n=== {header} ===")
    
    groups = [
        (grp_only, "Surgery only"),
        (grp_rt, "Surgery + RT"),
        (grp_ct, "Surgery + CT"),
        (grp_rt_ct, "Surgery + RT + CT")
    ]
    
    # Filter out empty groups
    valid_groups = [(grp, name) for grp, name in groups if not grp.empty]
    
    if not valid_groups:
        print("No valid groups for comparison.")
        return
    
    summary_data = []
    for grp, name in valid_groups:
        summary_data.append({
            "group": name,
            "n": len(grp),
            "mean": grp.mean(),
            "std": grp.std(),
            "median": grp.median(),
            "IQR": grp.quantile(0.75) - grp.quantile(0.25)
        })
    
    summary = pd.DataFrame(summary_data)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Perform pairwise comparisons
    print("\nPairwise Mann‑Whitney U tests:")
    for i in range(len(valid_groups)):
        for j in range(i + 1, len(valid_groups)):
            grp1, name1 = valid_groups[i]
            grp2, name2 = valid_groups[j]
            u_stat, p_val = mannwhitneyu(grp1, grp2, alternative="two-sided")
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"  {name1} vs {name2}: U = {u_stat:.2f}, p = {p_val:.6f} {significance}")


def analyze_diagnosis(df: pd.DataFrame, diagnosis_name: str):
    """Analyzes and prints summary stats for a specific diagnosis."""
    header = f"Overall comparison for {diagnosis_name}"
    
    # Perform overall comparison for this subset
    grp_only, grp_rt, grp_ct, grp_rt_ct = compare_groups_overall(df, header=header)
    
    # Perform per-timestep comparison
    compare_groups_by_timestep(df, header=f"Per-timestep comparison for {diagnosis_name}")

    # Generate a boxplot for this diagnosis
    if any(not g.empty for g in [grp_only, grp_rt, grp_ct, grp_rt_ct]):
        diagnosis_str = "".join(c if c.isalnum() else "_" for c in str(diagnosis_name))
        out_path = f"local_recurrence_boxplot_{diagnosis_str}.png"
        plot_boxplot(grp_only, grp_rt, grp_ct, grp_rt_ct, out_path=out_path)

def evaluate_physiological_protocol_feasibility(df: pd.DataFrame) -> float:
    """
    Screen each generated row for:
      1) any negative numeric values (e.g. tumor volumes < 0)
      2) dosing schedules impossible under basic sarcoma rules
         (example rule: chemo or RT without surgery is invalid)
    Returns the fraction of rows that violate any rule.
    """
    # 1) negative‐value check
    numeric = df.select_dtypes(include=[np.number])
    neg_violation = (numeric < 0).any(axis=1)

    # 2) simple protocol check: no chemo/RT without surgery
    proto_violation = ((df[CHEMOTHERAPY_FLAG_COL] == 1) | (df[RADIOTHERAPY_FLAG_COL] == 1)) & (df[SURGERY_FLAG_COL] == 0)

    violations = neg_violation | proto_violation
    frac_fail = violations.mean()
    print(f"Physiological/protocol feasibility failures: {violations.sum()}/{len(df)} ({frac_fail:.2%})")
    return frac_fail


def evaluate_natural_experiment(real_df: pd.DataFrame,
                                synth_df: pd.DataFrame,
                                match_cols: list[str],
                                outcome_col: str) -> tuple[float, float]:
    """
    For patients in real_df who actually got the alternative therapy,
    find their nearest‐matched synth counterpart (on match_cols), then
    compare outcome_col via Wasserstein‐1 and two‐sample KS test.
    Returns (Wasserstein-1 distance, KS p-value).
    """
    # naive exact‐match merge
    merged = (real_df
              .rename(columns={outcome_col: outcome_col + "_real"})
              .merge(
                 synth_df.rename(columns={outcome_col: outcome_col + "_synth"}),
                 on=match_cols,
                 suffixes=("", "")
              )
             )
    y_real = merged[outcome_col + "_real"]
    y_synth = merged[outcome_col + "_synth"]

    wd = wasserstein_distance(y_real, y_synth)
    ks_stat, ks_p = ks_2samp(y_real, y_synth)
    print(f"Natural experiment – Wasserstein-1: {wd:.4f}, KS p-value: {ks_p:.4g}")
    return wd, ks_p


def evaluate_marginal_distribution_alignment(real_df: pd.DataFrame,
                                             synth_df: pd.DataFrame,
                                             treatment_col: str,
                                             outcome_col: str,
                                             num_bins: int = 20) -> float:
    """
    Compute the KL divergence between P_synth(T,Y) and P_real(T,Y),
    approximated by binning each outcome within each treatment stratum.
    Returns the (scalar) KL divergence.
    """
    # P_real(T)
    pT_real = real_df[treatment_col].value_counts(normalize=True)

    kl_total = 0.0
    for t, p_t in pT_real.items():
        real_y = real_df.loc[real_df[treatment_col] == t, outcome_col].dropna()
        synth_y = synth_df.loc[synth_df[treatment_col] == t, outcome_col].dropna()
        if len(real_y) < 2 or len(synth_y) < 2:
            continue

        # shared bin edges
        bins = np.histogram_bin_edges(
            np.concatenate([real_y, synth_y]), bins=num_bins
        )

        p_real, _ = np.histogram(real_y, bins=bins, density=True)
        p_synth, _ = np.histogram(synth_y, bins=bins, density=True)

        # avoid zeros
        eps = 1e-8
        p_real = (p_real + eps) / (p_real.sum() + eps * len(p_real))
        p_synth = (p_synth + eps) / (p_synth.sum() + eps * len(p_synth))

        kl_t = entropy(p_synth, p_real)
        kl_total += p_t * kl_t

    print(f"Marginal distribution KL divergence: {kl_total:.4f}")
    return kl_total



# ------------------------------------------------------------------ #
# Timeseries statistics (shifts, compression, certainty)
# ------------------------------------------------------------------ #

def _binary_entropy(p: pd.Series, eps: float = 1e-12) -> pd.Series:
    """Shannon entropy for binary probabilities p (and 1-p)."""
    p = p.clip(eps, 1 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _gini_impurity_binary(p: pd.Series) -> pd.Series:
    """Gini impurity for binary probabilities p (and 1-p)."""
    return 2 * p * (1 - p)


def compute_timeseries_statistics(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute per-timestep, per-treatment statistics for the time series.
    Returns:
      - stats_df: metrics per (timestep, group)
      - shifts_df: per timestep, difference vs 'Surgery only' (S) for mean/median, and ratios for std/IQR
      - deltas_df: within-group change from the first timestep (t=1 by order of TIMESTEP_COL)
    """
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    work["group"] = _make_group_label(work)
    keep_groups = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(keep_groups)]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Per (timestep, group) metrics
    rows = []
    for (ts, grp), chunk in work.groupby([TIMESTEP_COL, "group"], sort=True):
        s = chunk["prob_norm"].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        mean = s.mean()
        std = s.std(ddof=1)
        cv = (std / mean) if (np.isfinite(mean) and mean != 0) else np.nan
        rows.append({
            "timestep": ts,
            "group": grp,
            "n": int(s.size),
            "mean": mean,
            "median": s.median(),
            "std": std,
            "q1": q1,
            "q3": q3,
            "IQR": iqr,
            "CV": cv,
            "entropy_mean": _binary_entropy(s).mean(),
            "gini_mean": _gini_impurity_binary(s).mean(),
        })
    stats_df = pd.DataFrame(rows).sort_values(["timestep", "group"]).reset_index(drop=True)

    # Shifts vs Surgery only at each timestep
    shifts_rows = []
    if not stats_df.empty:
        for ts, sub in stats_df.groupby("timestep"):
            base_row = sub[sub["group"] == "Surgery only"]
            base = base_row.iloc[0] if not base_row.empty else None
            for _, r in sub.iterrows():
                if base is None or r["group"] == "Surgery only":
                    mean_diff = np.nan
                    median_diff = np.nan
                    std_ratio = np.nan
                    iqr_ratio = np.nan
                else:
                    mean_diff = r["mean"] - base["mean"]
                    median_diff = r["median"] - base["median"]
                    std_ratio = (r["std"] / base["std"]) if (np.isfinite(base["std"]) and base["std"] > 0) else np.nan
                    iqr_ratio = (r["IQR"] / base["IQR"]) if (np.isfinite(base["IQR"]) and base["IQR"] > 0) else np.nan
                shifts_rows.append({
                    "timestep": ts,
                    "group": r["group"],
                    "mean_diff_vs_S": mean_diff,
                    "median_diff_vs_S": median_diff,
                    "std_ratio_vs_S": std_ratio,
                    "IQR_ratio_vs_S": iqr_ratio,
                })
    shifts_df = pd.DataFrame(shifts_rows).sort_values(["timestep", "group"]).reset_index(drop=True)

    # Deltas from first timestep (by sorted order of TIMESTEP_COL) within each group
    deltas_rows = []
    if not stats_df.empty:
        ordered_ts = sorted(stats_df["timestep"].unique())
        if ordered_ts:
            t0 = ordered_ts[0]
            base_df = stats_df[stats_df["timestep"] == t0].set_index("group")
            for _, r in stats_df.iterrows():
                grp = r["group"]
                if grp in base_df.index:
                    base = base_df.loc[grp]
                    mean_delta = r["mean"] - base["mean"]
                    median_delta = r["median"] - base["median"]
                    iqr_ratio = (r["IQR"] / base["IQR"]) if (np.isfinite(base["IQR"]) and base["IQR"] > 0) else np.nan
                else:
                    mean_delta = np.nan
                    median_delta = np.nan
                    iqr_ratio = np.nan
                deltas_rows.append({
                    "timestep": r["timestep"],
                    "group": grp,
                    "mean_delta_from_t1": mean_delta,
                    "median_delta_from_t1": median_delta,
                    "IQR_ratio_vs_t1": iqr_ratio,
                })
    deltas_df = pd.DataFrame(deltas_rows).sort_values(["group", "timestep"]).reset_index(drop=True)

    return stats_df, shifts_df, deltas_df


# ------------------------------------------------------------------ #
# Endpoint uncertainty per (timestep, treatment arm)
# ------------------------------------------------------------------ #

def compute_endpoint_uncertainty_by_timestep(
    df: pd.DataFrame,
    endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute binary entropy and Gini per endpoint, arm (treatment group), and timestep.

    Returns a tidy DataFrame with columns:
      - timestep
      - group (one of the four treatment arms)
      - endpoint: one of {"LR", "MET", "DOD"}
      - n: number of patients contributing
      - entropy_mean: mean binary entropy (natural log)
      - entropy_mean_norm: mean binary entropy normalized by ln 2 (in [0,1])
      - gini_mean: mean binary Gini impurity 2 p (1-p)
      - gini_mean_norm: mean Gini normalized by 0.5 (in [0,1])

    Endpoint source columns (raw preferred, with global-normalized fallback):
      - LR:  LOCAL_RECURRENCE_COL -> LR_prob_norm_global
      - MET: METASTATIC_COL       -> MET_prob_norm_global
      - DOD: DEAD_OF_DESEASE      -> DOD_prob_norm_global
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "timestep", "group", "endpoint", "n",
            "entropy_mean", "entropy_mean_norm",
            "gini_mean", "gini_mean_norm",
        ])

    work = df.copy()
    work["group"] = _make_group_label(work)
    arms = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms)]
    if work.empty:
        return pd.DataFrame(columns=[
            "timestep", "group", "endpoint", "n",
            "entropy_mean", "entropy_mean_norm",
            "gini_mean", "gini_mean_norm",
        ])

    # Determine endpoints
    if endpoints is None:
        endpoints = ["LR", "MET", "DOD"]

    # Ensure global-normalized fallbacks exist when raw cols are present
    if LOCAL_RECURRENCE_COL in work.columns and "LR_prob_norm_global" not in work.columns:
        work["LR_prob_norm_global"] = min_max_normalize(work[LOCAL_RECURRENCE_COL])
    if METASTATIC_COL in work.columns and "MET_prob_norm_global" not in work.columns:
        work["MET_prob_norm_global"] = min_max_normalize(work[METASTATIC_COL])
    if DEAD_OF_DESEASE in work.columns and "DOD_prob_norm_global" not in work.columns:
        work["DOD_prob_norm_global"] = min_max_normalize(work[DEAD_OF_DESEASE])

    # Map endpoint tag -> best-available column
    ep_map: dict[str, str] = {}
    if "LR" in endpoints:
        if LOCAL_RECURRENCE_COL in work.columns:
            ep_map["LR"] = LOCAL_RECURRENCE_COL
        elif "LR_prob_norm_global" in work.columns:
            ep_map["LR"] = "LR_prob_norm_global"
    if "MET" in endpoints:
        if METASTATIC_COL in work.columns:
            ep_map["MET"] = METASTATIC_COL
        elif "MET_prob_norm_global" in work.columns:
            ep_map["MET"] = "MET_prob_norm_global"
    if "DOD" in endpoints:
        if DEAD_OF_DESEASE in work.columns:
            ep_map["DOD"] = DEAD_OF_DESEASE
        elif "DOD_prob_norm_global" in work.columns:
            ep_map["DOD"] = "DOD_prob_norm_global"

    rows: list[dict] = []
    for (ts, grp), chunk in work.groupby([TIMESTEP_COL, "group"], sort=True):
        for ep, col in ep_map.items():
            if col not in chunk.columns:
                continue
            s = pd.to_numeric(chunk[col], errors="coerce").dropna()
            if s.empty:
                continue
            s = s.clip(0.0, 1.0)
            H = _binary_entropy(s)
            G = _gini_impurity_binary(s)
            rows.append({
                "timestep": ts,
                "group": grp,
                "endpoint": ep,
                "n": int(s.size),
                "entropy_mean": float(H.mean()),
                "entropy_mean_norm": float((H / np.log(2.0)).mean()),
                "gini_mean": float(G.mean()),
                "gini_mean_norm": float((G / 0.5).mean()),
            })

    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return res[[
        "timestep", "group", "endpoint", "n",
        "entropy_mean", "entropy_mean_norm",
        "gini_mean", "gini_mean_norm",
    ]].sort_values(["timestep", "endpoint", "group"]).reset_index(drop=True)


def save_endpoint_uncertainty_csv(prefix: str, df: pd.DataFrame) -> None:
    """Save endpoint-uncertainty time series to CSV if not empty."""
    if not df.empty:
        df.to_csv(f"{prefix}_endpoint_uncertainty_timeseries.csv", index=False)


def save_timeseries_statistics_csv(prefix: str, stats_df: pd.DataFrame, shifts_df: pd.DataFrame, deltas_df: pd.DataFrame) -> None:
    """Save the three timeseries statistics DataFrames to CSV with a common prefix."""
    if not stats_df.empty:
        stats_df.to_csv(f"{prefix}_timeseries_stats.csv", index=False)
    if not shifts_df.empty:
        shifts_df.to_csv(f"{prefix}_timeseries_shifts_vs_S.csv", index=False)
    if not deltas_df.empty:
        deltas_df.to_csv(f"{prefix}_timeseries_deltas_from_t1.csv", index=False)

# ------------------------------------------------------------------ #
# Main analysis function
# ------------------------------------------------------------------ #

def main():
    """Main function to run the analysis."""
    try:
        df = load_and_prepare(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # print("\n--- Overall Analysis ---")
    # plot_distribution(df, out_path="local_recurrence_distribution_overall.png")
    # grp_only, grp_rt, grp_ct, grp_rt_ct = compare_groups_overall(df, header="Overall comparison")
    # plot_boxplot(grp_only, grp_rt, grp_ct, grp_rt_ct, out_path="local_recurrence_boxplot_overall.png")
    # res_df = compare_groups_by_timestep(df)
    # plot_timeseries(df, out_path="local_recurrence_timeseries_overall.png")

    # Timeseries statistics (overall)
    # stats_df, shifts_df, deltas_df = compute_timeseries_statistics(df)
    # save_timeseries_statistics_csv("overall", stats_df, shifts_df, deltas_df)

    # Endpoint uncertainty per timestep (overall)
    # ep_unc_df = compute_endpoint_uncertainty_by_timestep(df)
    # save_endpoint_uncertainty_csv("overall", ep_unc_df)

    # Composite summary figure (overall, default 42 mo)
    # plot_summary_figure(df, horizon_months=42, histology=None, out_path_png="summary_figure_42mo.png")

    # Per-timestep grouped box plots with colors and legend (LR by default)
    # plot_grouped_boxplots_per_timestep(df, endpoint_col="LR_prob_norm_global", out_dir="imgs/per_timestep", file_prefix="lr")
    
    # # Three-endpoint timeseries (LR, metastasis, DOD stacked vertically)
    # plot_three_endpoint_timeseries(df, histology=None, out_path="three_endpoint_timeseries_overall.png")
    # plot_three_endpoint_timeseries_no_badges(df, histology=None, out_path="three_endpoint_timeseries_no_badges_overall.png")

    # Call compute_individualized_treatment_effects
    compute_individualized_treatment_effects(df)
    if HISTOLOGICAL_DIAGNOSIS_COL in df.columns:
        for hist in df[HISTOLOGICAL_DIAGNOSIS_COL].dropna().unique():
            hist_df = df[df[HISTOLOGICAL_DIAGNOSIS_COL] == hist]
            if hist_df.empty:
                continue
            ite_res = compute_individualized_treatment_effects(
                hist_df,
                weights={
                    "local_recurrence": 1 / 3,
                    "metastasis": 1 / 3,
                    "death_of_disease": 1 / 3,
                },
            )
            for treat in ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]:
                plot_composite_benefit_waterfall(
                    ite_res,
                    treatment=treat,
                    histology=str(hist),
                    out_path=(
                        "composite_benefit_waterfall_"
                        f"{str(hist).replace(' ', '_')}_"
                        f"{treat.replace(' ', '_').replace('+', '')}.png"
                    ),
                )
    # if HISTOLOGICAL_DIAGNOSIS_COL in df.columns:
    #     unique_diagnoses = df[HISTOLOGICAL_DIAGNOSIS_COL].unique()
    #     print(f"\nFound {len(unique_diagnoses)} unique histological diagnoses. Generating tables and plots for each.")
    #     for diagnosis in unique_diagnoses:
    #         if pd.isna(diagnosis):
    #             filtered_df = df[df[HISTOLOGICAL_DIAGNOSIS_COL].isna()]
    #             diagnosis_str = "unknown_diagnosis"
    #             print_diagnosis = "Unknown"
    #         else:
    #             filtered_df = df[df[HISTOLOGICAL_DIAGNOSIS_COL] == diagnosis]
    #             diagnosis_str = "".join(c if c.isalnum() else "_" for c in str(diagnosis))
    #             print_diagnosis = diagnosis

    #         if not filtered_df.empty:
    #             print(f"\n--- Analysis for Histological Diagnosis: {print_diagnosis} ---")
    #             analyze_diagnosis(filtered_df, print_diagnosis)
    #             out_path = f"local_recurrence_timeseries_{diagnosis_str}.png"
    #             plot_timeseries(filtered_df, out_path=out_path)

    #             # Timeseries statistics (per diagnosis)
    #             s_df, sh_df, d_df = compute_timeseries_statistics(filtered_df)
    #             save_timeseries_statistics_csv(f"diagnosis_{diagnosis_str}", s_df, sh_df, d_df)

    #             # Composite summary figure per diagnosis
    #             # plot_summary_figure(
    #             #     filtered_df,
    #             #     horizon_months=42,
    #             #     histology=str(print_diagnosis),
    #             #     out_path_png=f"summary_figure_42mo_{diagnosis_str}.png",
    #             # )
    #             # # Per-timestep for diagnosis
    #             # plot_grouped_boxplots_per_timestep(
    #             #     filtered_df,
    #             #     endpoint_col="LR_prob_norm_global",
    #             #     histology=str(print_diagnosis),
    #             #     out_dir=os.path.join("imgs", "per_timestep", diagnosis_str),
    #             #     file_prefix=f"lr_{diagnosis_str}",
    #             # )
    #             # Three-endpoint timeseries per diagnosis
    #             plot_three_endpoint_timeseries(
    #                 filtered_df,
    #                 histology=str(print_diagnosis),
    #                 out_path=f"three_endpoint_timeseries_{diagnosis_str}.png"
    #             )
    #             plot_three_endpoint_timeseries_no_badges(
    #                 filtered_df,
    #                 histology=str(print_diagnosis),
    #                 out_path=f"three_endpoint_timeseries_no_badges_{diagnosis_str}.png"
    #             )
    #         else:
    #             print(f"\n--- No data for diagnosis: {print_diagnosis} ---")
    # else:
    #     print(f"\nColumn '{HISTOLOGICAL_DIAGNOSIS_COL}' not found in the CSV. Skipping per-diagnosis analysis.")

    # ------------------------------------------------------------------ #
    # Per-lesion site analysis (aggregated: Extremity vs Non-extremity)
    # ------------------------------------------------------------------ #
    if LESION_SITE_COL in df.columns:
        agg_col = "lesion_site_extremity_group"
        df[agg_col] = df[LESION_SITE_COL].apply(_map_extremity_site)
        categories = ["Extremity", "Non-extremity", "Unknown"]
        present = [c for c in categories if c in df[agg_col].unique()]
        print(f"\nLesion site aggregation present groups: {present}")
        for cat in present:
            filtered_df = df[df[agg_col] == cat]
            group_key = cat.lower().replace(" ", "_")
            if filtered_df.empty:
                print(f"\n--- No data for lesion site group: {cat} ---")
                continue
            print(f"\n--- Analysis for Lesion Site Group: {cat} ---")
            analyze_diagnosis(filtered_df, f"LesionSite:{cat}")
            plot_timeseries(filtered_df, out_path=f"local_recurrence_timeseries_lesion_site_group_{group_key}.png")
            s_df, sh_df, d_df = compute_timeseries_statistics(filtered_df)
            save_timeseries_statistics_csv(f"lesion_site_group_{group_key}", s_df, sh_df, d_df)
            plot_three_endpoint_timeseries(
                filtered_df,
                histology=f"LesionSite:{cat}",
                out_path=f"three_endpoint_timeseries_lesion_site_group_{group_key}.png"
            )
            plot_three_endpoint_timeseries_no_badges(
                filtered_df,
                histology=f"LesionSite:{cat}",
                out_path=f"three_endpoint_timeseries_no_badges_lesion_site_group_{group_key}.png"
            )
    else:
        print(f"\nColumn '{LESION_SITE_COL}' not found in the CSV. Skipping per-lesion site analysis (extremity vs non-extremity).")

    # ------------------------------------------------------------------ #
    # Per-FNCLCC grading analysis
    # ------------------------------------------------------------------ #
    if FNCLCC_GRADING_COL in df.columns:
        unique_grades = df[FNCLCC_GRADING_COL].unique()
        print(f"\nFound {len(unique_grades)} unique FNCLCC grades. Generating tables and plots for each.")
        for grade in unique_grades:
            if pd.isna(grade):
                filtered_df = df[df[FNCLCC_GRADING_COL].isna()]
                grade_str = "unknown_fnclcc_grade"
                print_grade = "Unknown"
            else:
                filtered_df = df[df[FNCLCC_GRADING_COL] == grade]
                grade_str = "".join(c if c.isalnum() else "_" for c in str(grade))
                print_grade = grade

            if not filtered_df.empty:
                print(f"\n--- Analysis for FNCLCC Grade: {print_grade} ---")
                analyze_diagnosis(filtered_df, str(print_grade))
                plot_timeseries(filtered_df, out_path=f"local_recurrence_timeseries_fnclcc_{grade_str}.png")
                s_df, sh_df, d_df = compute_timeseries_statistics(filtered_df)
                save_timeseries_statistics_csv(f"fnclcc_{grade_str}", s_df, sh_df, d_df)
                plot_three_endpoint_timeseries(
                    filtered_df,
                    histology=str(print_grade),
                    out_path=f"three_endpoint_timeseries_fnclcc_{grade_str}.png"
                )
                plot_three_endpoint_timeseries_no_badges(
                    filtered_df,
                    histology=str(print_grade),
                    out_path=f"three_endpoint_timeseries_no_badges_fnclcc_{grade_str}.png"
                )
            else:
                print(f"\n--- No data for FNCLCC grade: {print_grade} ---")
    else:
        print(f"\nColumn '{FNCLCC_GRADING_COL}' not found in the CSV. Skipping per-FNCLCC grading analysis.")


if __name__ == "__main__":
    main()
