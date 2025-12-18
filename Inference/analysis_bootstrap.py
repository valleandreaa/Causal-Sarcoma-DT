from __future__ import annotations

from typing import Tuple, Optional, Literal
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from matplotlib import patches as mpatches
from matplotlib.lines import Line2D

# Default directory for analysis images
DEFAULT_IMG_DIR = "imgs/analysis"
EPS = 1e-12

# ------------------------------------------------------------------ #
# Column names (edit here if your CSV uses different labels)
# ------------------------------------------------------------------ #
csv_path = 'counterfactual_scenarios_multiple_patients_no_embed.csv'
LOCAL_RECURRENCE_COL = "fake_episodes.treatments.fields.endpoint_1.0_prob"
METASTATIC_COL       = "fake_episodes.treatments.fields.endpoint_2.0_prob"
# Optional third endpoint (e.g., disease-specific death as endpoint_3.0)
ENDPOINT_DOD_COL     = "fake_episodes.treatments.fields.endpoint_3.0_prob"
DEAD_OF_DESEASE      = "fake_episodes.diagnosis.fields.status_DOD_prob"
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

def as_prob(series: pd.Series) -> pd.Series:
    """Coerce to numeric probabilities in [0,1] without re-scaling."""
    s = pd.to_numeric(series, errors="coerce")
    return s.clip(0.0, 1.0)


def min_max_normalize(series: pd.Series, q_low: float = 0.05, q_high: float = 0.95) -> pd.Series:
    """Quantile-based min-max normalization to [0,1].
    Clips to [q_low, q_high] quantiles before scaling.
    """
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.dropna().empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    q_min = s.quantile(q_low)
    q_max = s.quantile(q_high)
    if not np.isfinite(q_min) or not np.isfinite(q_max) or q_max == q_min:
        return pd.Series(np.zeros_like(s, dtype=float), index=s.index)
    clipped = s.clip(lower=q_min, upper=q_max)
    out = (clipped - q_min) / (q_max - q_min)
    return out


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load the CSV, min-max normalize selected numeric columns to [0,1],
    and add probability-like columns for LR/MET/DOD.

    Columns normalized independently using (x - min) / (max - min):
      - Endpoints: [endpoint_1.0_prob, endpoint_2.0_prob, endpoint_3.0_prob]
      - Diagnosis statuses: [status_AWD_prob, status_DOD_prob, status_NED_prob]
    """
    df = pd.read_csv(path)

    def _minmax_inplace(frame: pd.DataFrame, cols: list[str]) -> None:
        present = [c for c in cols if c in frame.columns]
        if not present:
            return
        for c in present:
            frame[c] = min_max_normalize(frame[c])

    # Min-max normalize endpoint and diagnosis columns if present
    _minmax_inplace(df, [LOCAL_RECURRENCE_COL, METASTATIC_COL, ENDPOINT_DOD_COL])
    _minmax_inplace(df, [STATUS_AWD_COL, DEAD_OF_DESEASE, STATUS_NED_COL])

    # Global normalization across the whole dataset for endpoints
    if LOCAL_RECURRENCE_COL in df.columns:
        df["LR_prob_norm_global"] = as_prob(df[LOCAL_RECURRENCE_COL])
    if METASTATIC_COL in df.columns:
        df["MET_prob_norm_global"] = as_prob(df[METASTATIC_COL])
    # For DOD, use diagnosis status DOD probability by design
    base_dod = pd.to_numeric(df.get(DEAD_OF_DESEASE, pd.Series(index=df.index, dtype=float)), errors="coerce")
    df["DOD_prob_norm_global"] = as_prob(base_dod) if base_dod.notna().any() else np.nan

    # Back-compat for functions using prob_norm (local recurrence normalized)
    if "LR_prob_norm_global" in df.columns:
        df["prob_norm"] = df["LR_prob_norm_global"]
    return df


def _resolve_out_path(out_path: str, default_dir: str = DEFAULT_IMG_DIR) -> str:
    """Return a path under default_dir if no directory is provided. Ensures parent exists."""
    d = os.path.dirname(out_path)
    if d == "":
        out_path = os.path.join(default_dir, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return out_path


# ------------------------------------------------------------------ #
# Risk-averse AUC comparison (median + GMD*)
# ------------------------------------------------------------------ #
def _gmd_norm(values: np.ndarray) -> float:
    """Normalized Gini Mean Difference GMD* in [0,1] for a 1D array.

    Computes the Gini mean difference on [0, 0.5] for probabilities and
    rescales by 2 to obtain GMD* in [0,1]. Returns 0.0 for <2 values.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n < 2:
        return 0.0
    # Weighted sum formula for GMD
    w = (2 * np.arange(1, n + 1) - n - 1)
    gmd = (2.0 / (n * (n - 1))) * float(np.sum(w * x))  # in [0, 0.5] on [0,1]
    return float(2.0 * gmd)


def _auc_trapz(t: np.ndarray, y: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size == 0 or y.size == 0:
        return float("nan")
    return float(np.trapz(y, t))


def _arm_curves(time_to_probs: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return times, median curve, GMD* curve from mapping time -> 1D probs."""
    times = np.array(sorted(time_to_probs.keys()), dtype=float)
    med, gmd = [], []
    for tt in times:
        p = np.asarray(time_to_probs[tt], dtype=float)
        p = np.clip(p, 0.0, 1.0)
        if p.size == 0:
            med.append(np.nan)
            gmd.append(np.nan)
        else:
            med.append(float(np.median(p)))
            gmd.append(_gmd_norm(p))
    return times, np.array(med, dtype=float), np.array(gmd, dtype=float)


def _arm_score(time_to_probs: dict, lam_gmd: float = 0.0) -> dict:
    """Compute AUC(median), AUC(GMD*), and risk-averse score for one arm."""
    t, med, gmd = _arm_curves(time_to_probs)
    auc_m = _auc_trapz(t, med)
    auc_g = _auc_trapz(t, gmd)
    return {
        "AUC_median": auc_m,
        "AUC_GMD*": auc_g,
        "risk_averse_score": (auc_m + lam_gmd * auc_g),
        "times": t,
        "median_curve": med,
        "gmd_curve": gmd,
    }


def _compare_arms(time_to_probs_A: dict, time_to_probs_B: dict,
                  lam_gmd: float = 0.0, n_boot: int = 2000, seed: int | None = 42,
                  paired: bool = True) -> dict:
    """Bootstrap percent change in score = (score_B − score_A) / score_A × 100.

    If paired=True, resamples same indices per time for both arms. When score_A is
    zero, the percent change is undefined and NaN is returned.
    """
    rng = np.random.default_rng(seed)

    sA = _arm_score(time_to_probs_A, lam_gmd)["risk_averse_score"]
    sB = _arm_score(time_to_probs_B, lam_gmd)["risk_averse_score"]
    def _percent_change(a: float, b: float) -> float:
        if not np.isfinite(a) or abs(a) < EPS:
            return float("nan")
        return float((b - a) / a * 100.0)

    delta = _percent_change(sA, sB)

    t = sorted(set(time_to_probs_A.keys()) & set(time_to_probs_B.keys()))
    A_arr = {tt: np.asarray(time_to_probs_A[tt], dtype=float) for tt in t}
    B_arr = {tt: np.asarray(time_to_probs_B[tt], dtype=float) for tt in t}
    n_by_t = {tt: min(A_arr[tt].size, B_arr[tt].size) for tt in t}

    def resample_once() -> float:
        A_s, B_s = {}, {}
        for tt in t:
            n = n_by_t[tt]
            if n == 0:
                A_s[tt] = np.array([])
                B_s[tt] = np.array([])
                continue
            if paired:
                idx = rng.integers(0, n, size=n)
                A_s[tt] = A_arr[tt][:n][idx]
                B_s[tt] = B_arr[tt][:n][idx]
            else:
                A_s[tt] = A_arr[tt][rng.integers(0, A_arr[tt].size, size=n)]
                B_s[tt] = B_arr[tt][rng.integers(0, B_arr[tt].size, size=n)]
        score_A = _arm_score(A_s, lam_gmd)["risk_averse_score"]
        score_B = _arm_score(B_s, lam_gmd)["risk_averse_score"]
        return _percent_change(score_A, score_B)

    boots = np.array([resample_once() for _ in range(n_boot)], dtype=float)
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        lo = hi = p_two = float("nan")
    else:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p_two = 2 * min(float((boots <= 0).mean()), float((boots >= 0).mean()))
    return {"percent_change": float(delta), "CI95": (float(lo), float(hi)), "p~": float(p_two)}


def _build_time_to_probs_from_df(df: pd.DataFrame, endpoint_col: str, arm_label: str) -> dict:
    """Build mapping time -> array of per-patient probabilities for a given arm.

    Aggregates duplicates per (patient_id, timestep) by mean. Requires
    TIMESTEP_COL and a patient_id column if present.
    """
    sub = df[df["group"] == arm_label]
    if sub.empty:
        return {}
    # Make sure numeric and clipped to [0,1]
    sub = sub[[TIMESTEP_COL, endpoint_col] + (["patient_id"] if "patient_id" in sub.columns else [])].copy()
    sub[endpoint_col] = pd.to_numeric(sub[endpoint_col], errors="coerce").astype(float).clip(0.0, 1.0)
    sub = sub.dropna(subset=[endpoint_col, TIMESTEP_COL])
    if sub.empty:
        return {}
    if "patient_id" in sub.columns:
        agg = sub.groupby([TIMESTEP_COL, "patient_id"], as_index=False)[endpoint_col].mean()
        grouped = agg.groupby(TIMESTEP_COL)
    else:
        grouped = sub.groupby(TIMESTEP_COL)
    return {int(t): g[endpoint_col].to_numpy(dtype=float) for t, g in grouped}


def compare_two_arms_df(
    df: pd.DataFrame,
    endpoint_col: str,
    arm_A: str = "Surgery only",
    arm_B: str = "Surgery + RT",
    lam_gmd: float = 0.0,
    n_boot: int = 2000,
    paired: bool = True,
    seed: int | None = 42,
) -> dict:
    """Compare two arms using risk-averse AUC scoring (percent change output).

    Returns dict with keys: percent_change, CI95(tuple), p~. With lam_gmd=0 this
    reports the percent change in AUC(median) for arm_B relative to arm_A.
    """
    df = df.copy()
    df["group"] = _make_group_label(df)
    if endpoint_col not in df.columns:
        # Try global-normalized fallbacks
        fallback = {
            LOCAL_RECURRENCE_COL: "LR_prob_norm_global",
            METASTATIC_COL: "MET_prob_norm_global",
            DEAD_OF_DESEASE: "DOD_prob_norm_global",
        }.get(endpoint_col)
        if fallback and fallback in df.columns:
            endpoint_col = fallback
        else:
            raise ValueError(f"Missing endpoint column: {endpoint_col}")

    A = _build_time_to_probs_from_df(df, endpoint_col, arm_A)
    B = _build_time_to_probs_from_df(df, endpoint_col, arm_B)
    if not A or not B:
        raise ValueError("Selected arms or endpoint have no data.")
    return _compare_arms(A, B, lam_gmd=lam_gmd, n_boot=n_boot, seed=seed, paired=paired)


def risk_averse_comparison_grid(
    df: pd.DataFrame,
    lam_gmd: float = 0.0,
    n_boot: int = 200,
    paired: bool = True,
    seed: int | None = 42,
    baseline: str = "Surgery only",
    arms: list[str] | None = None,
    endpoints: list[str] | None = None,
) -> pd.DataFrame:
    """Compute percent change (arm vs baseline) with CI across endpoints and arms.

    - endpoints: list of endpoint column names or tags; if None, uses
      ["LR_prob_norm_global", "MET_prob_norm_global", "DOD_prob_norm_global"].
    - arms: list of arm labels to compare against baseline; if None, compares
      ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"].
    Returns tidy DataFrame with columns:
      endpoint, baseline, arm, percent_change, ci_low, ci_high, p_tilde
    """
    df = df.copy()
    df["group"] = _make_group_label(df)
    if endpoints is None:
        endpoints = [
            "LR_prob_norm_global" if "LR_prob_norm_global" in df.columns else LOCAL_RECURRENCE_COL,
            "MET_prob_norm_global" if "MET_prob_norm_global" in df.columns else METASTATIC_COL,
            "DOD_prob_norm_global" if "DOD_prob_norm_global" in df.columns else DEAD_OF_DESEASE,
        ]
    if arms is None:
        arms = ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    rows = []
    for ep in endpoints:
        if ep not in df.columns and ep not in (LOCAL_RECURRENCE_COL, METASTATIC_COL, DEAD_OF_DESEASE):
            # skip unknown endpoints
            continue
        for arm in arms:
            try:
                res = compare_two_arms_df(
                    df, endpoint_col=ep, arm_A=baseline, arm_B=arm,
                    lam_gmd=lam_gmd, n_boot=n_boot, paired=paired, seed=seed,
                )
                rows.append({
                    "endpoint": ep,
                    "baseline": baseline,
                    "arm": arm,
                    "percent_change": res["percent_change"],
                    "ci_low": float(res["CI95"][0]),
                    "ci_high": float(res["CI95"][1]),
                    "p_tilde": res["p~"],
                })
            except Exception:
                # safely continue if an arm/endpoint has no data
                continue
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
# Patient-cluster bootstrap utilities (mirroring validation_bootstrap)
# ------------------------------------------------------------------ #
def _cluster_bootstrap_stat(
    values: pd.Series,
    pids: Optional[pd.Series] = None,
    agg: Literal["mean", "median"] = "mean",
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute a point estimate and 95% percentile CI for a statistic over a 1D
    vector of values in [0,1], using patient-cluster bootstrap if patient_ids
    are provided. If ``pids`` is None, falls back to IID row bootstrap.

    Returns (point_estimate, (ci_low, ci_high)). If resamples are ill-posed,
    CI bounds may be NaN.
    """
    s = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if s.empty:
        return np.nan, (np.nan, np.nan)
    # clip to probability range defensively
    s = s.clip(0.0, 1.0)

    if agg == "mean":
        point = float(s.mean())
        stat_fn = np.mean
    else:
        point = float(s.median())
        stat_fn = np.median

    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)

    if pids is None or pids.isna().all():
        m = s.shape[0]
        idx_all = np.arange(m)
        for b in range(n_boot):
            idx = rng.integers(0, m, size=m)
            boot[b] = float(stat_fn(s.iloc[idx])) if idx.size else np.nan
    else:
        # Align pids to s index, then factorize
        p = pids.loc[s.index]
        codes, uniques = pd.factorize(p, sort=False)
        clusters = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(clusters)
        if n_clusters == 0:
            return point, (np.nan, np.nan)
        for b in range(n_boot):
            draw = rng.integers(0, n_clusters, size=n_clusters)
            sel = np.concatenate([clusters[d] for d in draw], axis=0)
            boot[b] = float(stat_fn(s.iloc[sel])) if sel.size else np.nan

    boot = boot[~np.isnan(boot)]
    if boot.size == 0:
        return point, (np.nan, np.nan)
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return point, ci


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

    # Optional: report patient-cluster bootstrap CIs for group means/medians if patient_id is present
    if "patient_id" in df.columns:
        pid = df["patient_id"]
        for name, grp in (
            ("S only", grp_only),
            ("S+RT", grp_rt),
            ("S+CT", grp_ct),
            ("S+RT+CT", grp_rt_ct),
        ):
            if grp.empty:
                continue
            mean_pt, mean_ci = _cluster_bootstrap_stat(grp, pids=pid.loc[grp.index], agg="mean")
            med_pt, med_ci = _cluster_bootstrap_stat(grp, pids=pid.loc[grp.index], agg="median")
            print(
                f"  {name}: mean={mean_pt:.4f} [95% CI {mean_ci[0]:.4f}, {mean_ci[1]:.4f}]  "
                f"median={med_pt:.4f} [95% CI {med_ci[0]:.4f}, {med_ci[1]:.4f}]"
            )
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

        # Bootstrap CIs by patient if available
        pid_col = chunk["patient_id"] if "patient_id" in chunk.columns else None
        mean_only_ci = (np.nan, np.nan)
        mean_rt_ci = (np.nan, np.nan)
        mean_ct_ci = (np.nan, np.nan)
        mean_rt_ct_ci = (np.nan, np.nan)
        med_only_ci = (np.nan, np.nan)
        med_rt_ci = (np.nan, np.nan)
        med_ct_ci = (np.nan, np.nan)
        med_rt_ct_ci = (np.nan, np.nan)
        if pid_col is not None:
            if not grp_only.empty:
                _, mean_only_ci = _cluster_bootstrap_stat(grp_only, pids=pid_col.loc[grp_only.index], agg="mean")
                _, med_only_ci = _cluster_bootstrap_stat(grp_only, pids=pid_col.loc[grp_only.index], agg="median")
            if not grp_rt.empty:
                _, mean_rt_ci = _cluster_bootstrap_stat(grp_rt, pids=pid_col.loc[grp_rt.index], agg="mean")
                _, med_rt_ci = _cluster_bootstrap_stat(grp_rt, pids=pid_col.loc[grp_rt.index], agg="median")
            if not grp_ct.empty:
                _, mean_ct_ci = _cluster_bootstrap_stat(grp_ct, pids=pid_col.loc[grp_ct.index], agg="mean")
                _, med_ct_ci = _cluster_bootstrap_stat(grp_ct, pids=pid_col.loc[grp_ct.index], agg="median")
            if not grp_rt_ct.empty:
                _, mean_rt_ct_ci = _cluster_bootstrap_stat(grp_rt_ct, pids=pid_col.loc[grp_rt_ct.index], agg="mean")
                _, med_rt_ct_ci = _cluster_bootstrap_stat(grp_rt_ct, pids=pid_col.loc[grp_rt_ct.index], agg="median")

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
            "mean_only_ci_low": mean_only_ci[0],
            "mean_only_ci_high": mean_only_ci[1],
            "mean_rt_ci_low": mean_rt_ci[0],
            "mean_rt_ci_high": mean_rt_ci[1],
            "mean_ct_ci_low": mean_ct_ci[0],
            "mean_ct_ci_high": mean_ct_ci[1],
            "mean_rt_ct_ci_low": mean_rt_ct_ci[0],
            "mean_rt_ct_ci_high": mean_rt_ct_ci[1],
            "std_only"       : grp_only.std() if not grp_only.empty else np.nan,
            "std_rt"         : grp_rt.std() if not grp_rt.empty else np.nan,
            "std_ct"         : grp_ct.std() if not grp_ct.empty else np.nan,
            "std_rt_ct"      : grp_rt_ct.std() if not grp_rt_ct.empty else np.nan,
            "median_only"    : grp_only.median() if not grp_only.empty else np.nan,
            "median_rt"      : grp_rt.median() if not grp_rt.empty else np.nan,
            "median_ct"      : grp_ct.median() if not grp_ct.empty else np.nan,
            "median_rt_ct"   : grp_rt_ct.median() if not grp_rt_ct.empty else np.nan,
            "median_only_ci_low": med_only_ci[0],
            "median_only_ci_high": med_only_ci[1],
            "median_rt_ci_low": med_rt_ci[0],
            "median_rt_ci_high": med_rt_ci[1],
            "median_ct_ci_low": med_ct_ci[0],
            "median_ct_ci_high": med_ct_ci[1],
            "median_rt_ct_ci_low": med_rt_ct_ci[0],
            "median_rt_ct_ci_high": med_rt_ct_ci[1],
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
    n_boot: int = 0,
    seed: int | None = 42,
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
            # For DOD, explicitly use diagnosis DOD probability
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

    # Optional: bootstrap uncertainty for the composite ITE used in waterfalls.
    # We resample time points with replacement within each patient to obtain a
    # distribution for the integrated (trapezoidal) composite effect.
    if weights and n_boot and n_boot > 0:
        rng = np.random.default_rng(seed)
        weight_series = pd.Series(weights, dtype=float)

        # Pre-compute group labels to access probabilities per scenario
        dwork = df.copy()
        dwork["group"] = _make_group_label(dwork)

        comp_ci_rows: list[dict] = []
        groups = [
            "Surgery only",
            "Surgery + RT",
            "Surgery + CT",
            "Surgery + RT + CT",
        ]

        # Endpoint columns used for composite
        ep_cols = {
            k: endpoint_cols[k] if endpoint_cols and k in endpoint_cols else v
            for k, v in {
                "local_recurrence": LOCAL_RECURRENCE_COL,
                "metastasis": METASTATIC_COL,
                "death_of_disease": DEAD_OF_DESEASE,
            }.items()
            if (endpoint_cols or True)
        }

        for (pid, treat), _ in (
            result.dropna(subset=["composite_ite"]).drop_duplicates([patient_col, "treatment"]).groupby([patient_col, "treatment"]) 
        ):
            if treat == reference_group:
                continue
            # Build per-time composite delta = sum_w (p_treat - p_ref)
            ref = dwork[(dwork[patient_col] == pid) & (dwork["group"] == reference_group)]
            trt = dwork[(dwork[patient_col] == pid) & (dwork["group"] == treat)]
            if ref.empty or trt.empty:
                continue
            comp_df = None
            # align by time for each endpoint and accumulate
            for ep_name, col in ep_cols.items():
                if col not in dwork.columns or ep_name not in weight_series.index:
                    continue
                r = ref[[time_col, col]].rename(columns={col: "ref"})
                t = trt[[time_col, col]].rename(columns={col: "trt"})
                m = pd.merge(r, t, on=time_col, how="inner")
                if m.empty:
                    continue
                m[ep_name] = (m["trt"].astype(float) - m["ref"].astype(float)) * float(weight_series[ep_name])
                m = m[[time_col, ep_name]]
                comp_df = m if comp_df is None else pd.merge(comp_df, m, on=time_col, how="inner")
            if comp_df is None or comp_df.empty:
                continue
            comp_df = comp_df.sort_values(time_col)
            comp_df["comp"] = comp_df.drop(columns=[time_col]).sum(axis=1)
            times = comp_df[time_col].to_numpy()
            vals = comp_df["comp"].to_numpy()
            if len(vals) < 1:
                continue
            # point estimate using full grid
            if len(vals) == 1:
                point = float(vals[0])
            else:
                point = float(np.trapz(vals, x=times) / (times[-1] - times[0]))
            # bootstrap over time points
            boots = []
            if len(vals) >= 2:
                m = len(vals)
                for _ in range(n_boot):
                    idx = rng.integers(0, m, size=m)
                    tb = times[idx]
                    vb = vals[idx]
                    order = np.argsort(tb)
                    tb = tb[order]
                    vb = vb[order]
                    area = float(np.trapz(vb, x=tb) / (tb[-1] - tb[0])) if tb[-1] != tb[0] else float(vb.mean())
                    boots.append(area)
            if boots:
                low, high = np.percentile(boots, [2.5, 97.5])
            else:
                low = high = np.nan
            comp_ci_rows.append({patient_col: pid, "treatment": treat, "composite_ite_point": point, "composite_ci_low": low, "composite_ci_high": high})

        if comp_ci_rows:
            comp_ci_df = pd.DataFrame(comp_ci_rows)
            result = result.merge(comp_ci_df[[patient_col, "treatment", "composite_ci_low", "composite_ci_high"]],
                                  on=[patient_col, "treatment"], how="left")

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

    # Keep CI columns if present and sort by point estimate
    keep_cols = [c for c in [patient_col, "composite_ite", "composite_ci_low", "composite_ci_high"] if c in data.columns]
    data = data[keep_cols].drop_duplicates(subset=[patient_col]).sort_values("composite_ite", ascending=False)

    colors = data["composite_ite"].apply(
        lambda x: "#2ca02c" if x > 0 else "#d62728"
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(data))
    ax.bar(x, data["composite_ite"], color=colors, edgecolor="black")
    # Optional uncertainty whiskers
    if "composite_ci_low" in data.columns and "composite_ci_high" in data.columns:
        y = data["composite_ite"].to_numpy(dtype=float)
        yerr_low = np.clip(y - data["composite_ci_low"].to_numpy(dtype=float), 0, None)
        yerr_high = np.clip(data["composite_ci_high"].to_numpy(dtype=float) - y, 0, None)
        yerr = np.vstack([yerr_low, yerr_high])
        ax.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', elinewidth=0.8, capsize=2, alpha=0.9)
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
    out_path = _resolve_out_path(out_path)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nComposite benefit waterfall saved to {out_path}")


def plot_composite_benefit_fan_envelope(
    ite_df: pd.DataFrame,
    treatment: str,
    histology: str | None = None,
    out_path: str = "composite_benefit_fan_envelope.png",
    n_boot: int = 200,
    seed: int | None = 42,
    patient_col: str = "patient_id",
    sort_desc: bool = False,
) -> None:
    """Rank‑consistent quantile envelope (fan chart) of the waterfall curve.

    The procedure bootstraps patients, sorts each resample to produce a
    rank‑consistent curve, and then computes quantiles across rank positions.
    """
    data = ite_df[ite_df["treatment"] == treatment]
    if "composite_ite" not in data.columns or data.empty:
        print("plot_composite_benefit_fan_envelope: no composite_ite values to plot.")
        return

    vals = (
        data[[patient_col, "composite_ite"]]
        .drop_duplicates(subset=[patient_col])
        ["composite_ite"].astype(float).dropna().to_numpy()
    )
    N = vals.size
    if N == 0:
        print("plot_composite_benefit_fan_envelope: empty after filtering.")
        return

    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, N), dtype=float)
    for b in range(n_boot):
        smp = rng.choice(vals, size=N, replace=True)
        smp.sort()  # ascending
        if sort_desc:
            smp = smp[::-1]
        reps[b, :] = smp

    # Rank‑consistent quantiles across the bootstrap ensemble
    # Use 95% CI (2.5–97.5) and median only
    q2p5 = np.percentile(reps, 2.5, axis=0)
    q50 = np.percentile(reps, 50, axis=0)
    q97p5 = np.percentile(reps, 97.5, axis=0)

    ranks = np.arange(1, N + 1)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    ax.fill_between(ranks, q2p5, q97p5, color="#1f77b4", alpha=0.18, linewidth=0)
    ax.plot(ranks, q50, color="#1f77b4", lw=2)
    ax.axhline(0, color="black", lw=1, ls=":")
    ax.set_xlim(1, N)
    ax.set_xlabel("Rank (1 = best, N = worst)")
    ax.set_ylabel("Composite ITE (ordered)")
    title = f"Rank‑consistent envelope: {treatment}"
    if histology:
        title += f" – {histology}"
    ax.set_title(title)
    # Beneficial count (lower is better)
    n_benef = int(np.sum(vals < 0))
    ax.text(0.98, 0.95, f"{n_benef}/{N} ({n_benef / N:.0%}) beneficial",
            transform=ax.transAxes, ha='right', va='top', fontsize=10)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    out_path = _resolve_out_path(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nComposite benefit fan envelope (95% CI) saved to {out_path}")


def _rank_consistent_envelope(values: np.ndarray, n_boot: int = 200, seed: int | None = 42, sort_desc: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (q2.5, q50, q97.5) for the ordered waterfall using rank‑consistent bootstrap.

    values is a 1D array of per‑patient effects (no NaNs). The bootstrap
    resamples patients, sorts each sample to construct a monotone curve, and
    then computes rank‑wise quantiles.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    N = vals.size
    if N == 0:
        return np.array([]), np.array([]), np.array([])
    rng = np.random.default_rng(seed)
    reps = np.empty((n_boot, N), dtype=float)
    for b in range(n_boot):
        smp = rng.choice(vals, size=N, replace=True)
        smp.sort()  # ascending (best first when lower is better)
        if sort_desc:
            smp = smp[::-1]
        reps[b, :] = smp
    q2p5 = np.percentile(reps, 2.5, axis=0)
    q50 = np.percentile(reps, 50, axis=0)
    q97p5 = np.percentile(reps, 97.5, axis=0)
    return q2p5, q50, q97p5


def plot_endpoint_waterfall_fan_grid(
    ite_df: pd.DataFrame,
    treatments: list[str] | None = None,
    endpoints: list[str] | None = None,
    histology: str | None = None,
    out_path: str = "waterfall_fan_grid.png",
    n_boot: int = 200,
    seed: int | None = 42,
    patient_col: str = "patient_id",
    sort_desc: bool = False,
) -> None:
    """Create a grid of rank‑consistent 95% CI envelopes (fan charts).

    Rows correspond to endpoints, columns to treatments. Each cell shows the
    5–95% envelope and the median of the ordered per‑patient ITE values.
    """
    if endpoints is None:
        endpoints = ["local_recurrence", "metastasis", "death_of_disease"]
    if treatments is None:
        treatments = ["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]

    # Map pretty labels
    t_label = {
        "Surgery + RT": "S+RT",
        "Surgery + CT": "S+CT",
        "Surgery + RT + CT": "S+RT+CT",
    }
    e_label = {
        "local_recurrence": "Local Recurrence",
        "metastasis": "Metastasis",
        "death_of_disease": "DOD",
    }

    # Prepare data
    base = ite_df[[patient_col, "treatment", "endpoint", "ite"]].copy()
    if base.empty:
        print("plot_endpoint_waterfall_fan_grid: empty ite_df")
        return

    n_rows = len(endpoints)
    n_cols = len(treatments)
    fig_w = 3.8 * n_cols
    fig_h = 2.8 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=False, sharey=False)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Determine y‑limits per endpoint for consistent scaling across treatments
    ylims = {}
    for ep in endpoints:
        vals_ep = base.loc[base["endpoint"] == ep, "ite"].astype(float)
        if vals_ep.dropna().empty:
            ylims[ep] = (-0.01, 0.01)
        else:
            v = vals_ep.to_numpy()
            lo, hi = np.nanpercentile(v, [1, 99])
            rng = max(abs(lo), abs(hi))
            ylims[ep] = (-rng, rng)

    for r, ep in enumerate(endpoints):
        for c, tr in enumerate(treatments):
            ax = axes[r, c]
            vals = (
                base[(base["endpoint"] == ep) & (base["treatment"] == tr)]
                [[patient_col, "ite"]]
                .drop_duplicates(subset=[patient_col])
                ["ite"].astype(float).dropna().to_numpy()
            )
            if vals.size == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                ax.set_axis_off()
                continue
            q5, q50, q95 = _rank_consistent_envelope(vals, n_boot=n_boot, seed=seed, sort_desc=sort_desc)
            N = q50.size
            ranks = np.arange(1, N + 1)
            ax.fill_between(ranks, q5, q95, color="#1f77b4", alpha=0.2, linewidth=0)
            ax.plot(ranks, q50, color="#1f77b4", lw=1.8)
            ax.axhline(0, color="black", lw=0.8, ls=":")
            ax.set_xlim(1, N)
            ax.set_ylim(*ylims[ep])
            if r == n_rows - 1:
                ax.set_xlabel("Rank")
            if c == 0:
                ax.set_ylabel(e_label.get(ep, ep))
            ax.set_title(t_label.get(tr, tr), fontsize=10)
            # Beneficial count (lower is better)
            n_benef = int(np.sum(vals < 0))
            ax.text(0.98, 0.92, f"{n_benef}/{vals.size} ({(n_benef/vals.size):.0%})",
                    transform=ax.transAxes, ha='right', va='top', fontsize=8)
            ax.grid(True, axis='y', alpha=0.25)

    title = ""
    # if histology:
    #     title += f" – {histology}"
    fig.suptitle(title, y=0.995, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path = _resolve_out_path(out_path)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"waterfall fan grid saved to {out_path}")

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
        out_path = _resolve_out_path(out_path)
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor='white')
        
        print(f"\nBox‑plot saved to {out_path}")
    else:
        print("No data available for box plot")







# ------------------------------------------------------------------ #
# New: Three-endpoint trajectories with bootstrap ribbons
# ------------------------------------------------------------------ #
def _bootstrap_timecourse_by_group(
    df: pd.DataFrame,
    value_col: str,
    time_col: str = TIMESTEP_COL,
    patient_col: str = "patient_id",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    agg: Literal["median", "mean"] = "median",
    band_method: Literal["pointwise", "rcqe"] = "rcqe",
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Bootstrap a time-course summary (per-timestep) for a single treatment group.

    Returns (times, center, (low, high), counts) where center is the bootstrap
    median (or mean) at each time and (low, high) are 95% percentile bands.
    Counts are per-timestep unique-patient counts in the original data.
    """
    work = df[[time_col, value_col] + ([patient_col] if patient_col in df.columns else [])].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").clip(0.0, 1.0)
    work = work.dropna(subset=[value_col, time_col])
    if work.empty:
        return np.array([]), np.array([]), (np.array([]), np.array([])), np.array([])

    times = np.array(sorted(work[time_col].dropna().unique()))
    agg_fn = np.nanmedian if agg == "median" else np.nanmean

    # Counts per timestep (unique patients if available else rows)
    counts = []
    if patient_col in work.columns:
        for t in times:
            counts.append(int(work.loc[work[time_col] == t, patient_col].dropna().nunique()))
    else:
        for t in times:
            counts.append(int(work.loc[work[time_col] == t, value_col].notna().sum()))
    counts = np.array(counts, dtype=int)

    rng = np.random.default_rng(seed)

    # Prepare cluster units (patient-level) if possible
    use_cluster = patient_col in work.columns and work[patient_col].notna().any()
    if use_cluster:
        codes, uniques = pd.factorize(work[patient_col], sort=False)
        clusters = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(clusters)
        if n_clusters == 0:
            use_cluster = False

    boots = np.empty((n_boot, times.size), dtype=float)
    for b in range(n_boot):
        if use_cluster:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            sel_idx = np.concatenate([clusters[d] for d in draw], axis=0)
            sample = work.iloc[sel_idx]
        else:
            m = work.shape[0]
            sel_idx = rng.integers(0, m, size=m)
            sample = work.iloc[sel_idx]

        # per-time aggregation in the resample
        for i, t in enumerate(times):
            vals = pd.to_numeric(sample.loc[sample[time_col] == t, value_col], errors="coerce").to_numpy()
            vals = vals[(~np.isnan(vals)) & (vals >= 0.0) & (vals <= 1.0)]
            boots[b, i] = agg_fn(vals) if vals.size else np.nan

    # Compute bands and center line
    if band_method == "pointwise":
        low = np.nanpercentile(boots, 2.5, axis=0)
        high = np.nanpercentile(boots, 97.5, axis=0)
    else:
        # Rank‑consistent quantile envelope (RCQE):
        # pick entire bootstrap curves corresponding to lower/upper quantile ranks
        # using a curve-wise summary to order trajectories.
        # Compute curve summaries (mean over time ignoring NaNs)
        curve_score = np.nanmean(boots, axis=1)
        valid = np.isfinite(curve_score)
        if not np.any(valid):
            low = np.nanpercentile(boots, 2.5, axis=0)
            high = np.nanpercentile(boots, 97.5, axis=0)
        else:
            idx = np.argsort(curve_score[valid])
            boots_valid = boots[valid]
            n = boots_valid.shape[0]
            q = 0.025
            low_i = int(np.floor(q * (n - 1)))
            high_i = int(np.ceil((1.0 - q) * (n - 1)))
            low = boots_valid[idx[low_i], :]
            high = boots_valid[idx[high_i], :]
    center = np.nanmedian(boots, axis=0) if agg == "median" else np.nanmean(boots, axis=0)

    return times, center, (low, high), counts


def plot_three_endpoint_trajectories_with_bands(
    df: pd.DataFrame,
    histology: Optional[str] = None,
    out_path: str = "three_endpoint_trajectories_bands.png",
    n_boot: int = 200,
    seed: Optional[int] = 42,
    agg: Literal["median", "mean"] = "median",
    band_method: Literal["pointwise", "rcqe"] = "rcqe",
    use_color: bool = True,
    layout: Literal["rows", "cols"] = "rows",
    show_counts: bool = True,
    show_errorbars: bool = True,
    table_outside: bool = True,
    table_mode: Literal["counts", "summary", "both"] = "summary",
    side_titles: bool = True,
    annotate_arm_dispersion: bool = True,
    # Spacing controls (bigger panels, less compression)
    row_height: float = 5.0,               # per-panel height (rows layout)
    panel_hspace: float = 0.85,            # space between stacked panels
    table_gap: float = 0.10,               # gap between axis and table (outside tables)
    legend_y: float = 1.4,                # legend vertical anchor relative to top axis
    tight_top: float = 0.92,               # allow axes to use more vertical space
) -> None:
    """
    Draw lines + ribbons per endpoint (3 panels):
      - x: time, y: outcome probability
      - One panel per endpoint (LR, MET, DOD)
      - One line per treatment arm showing bootstrap median; 95% CI ribbon around it
      - Optional thin table with n at time t per treatment. By default the
        table is rendered just outside each axis to avoid overlap with the plot.
      - Optional sparse error bars on the center line to emphasize uncertainty.
      - Optional side titles per panel (left side) to improve spacing.
    """
    if df.empty:
        print("plot_three_endpoint_trajectories_with_bands: Empty DataFrame; nothing to plot.")
        return

    work = df.copy()
    # Ensure probability columns exist (no re-scaling)
    if "LR_prob_norm_global" not in work.columns and LOCAL_RECURRENCE_COL in work.columns:
        work["LR_prob_norm_global"] = as_prob(work[LOCAL_RECURRENCE_COL])
    if "MET_prob_norm_global" not in work.columns and METASTATIC_COL in work.columns:
        work["MET_prob_norm_global"] = as_prob(work[METASTATIC_COL])
    if "DOD_prob_norm_global" not in work.columns and DEAD_OF_DESEASE in work.columns:
        work["DOD_prob_norm_global"] = as_prob(work[DEAD_OF_DESEASE])

    # Optional histology filter
    if histology is not None and HISTOLOGICAL_DIAGNOSIS_COL in work.columns:
        work = work[work[HISTOLOGICAL_DIAGNOSIS_COL] == histology]

    # Label arms
    work["group"] = _make_group_label(work)
    arms = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
    work = work[work["group"].isin(arms)]
    if work.empty:
        print("plot_three_endpoint_trajectories_with_bands: No rows for the specified groups/histology.")
        return

    # Endpoint mapping (prefer normalized if present)
    endpoints = [
        ("Local Recurrence", "LR_prob_norm_global" if "LR_prob_norm_global" in work.columns else LOCAL_RECURRENCE_COL),
        ("Metastasis", "MET_prob_norm_global" if "MET_prob_norm_global" in work.columns else METASTATIC_COL),
        ("DOD", "DOD_prob_norm_global" if "DOD_prob_norm_global" in work.columns else DEAD_OF_DESEASE),
    ]

    # Style: color-blind friendly colors + distinct linestyles
    color_map = {
        "Surgery only": "#1f77b4",   # blue
        "Surgery + RT": "#ff7f0e",   # orange
        "Surgery + CT": "#2ca02c",   # green
        "Surgery + RT + CT": "#d62728",  # red
    }
    gray_line_map = {
        "Surgery only": "#111111",
        "Surgery + RT": "#333333",
        "Surgery + CT": "#555555",
        "Surgery + RT + CT": "#000000",
    }
    linestyles = {
        "Surgery only": "-",
        "Surgery + RT": "--",
        "Surgery + CT": "-.",
        "Surgery + RT + CT": ":",
    }

    # Determine dynamic width based on number of timesteps (approx; refine later per-ep)
    all_times = sorted(work[TIMESTEP_COL].dropna().unique()) if TIMESTEP_COL in work.columns else []
    n_steps = len(all_times)
    if layout == "rows":
        # 3 rows x 1 column; widen with more steps - increased width for better table fit
        fig_width = max(18.0, min(32.0, 0.7 * max(1, n_steps)))
        # add extra space when showing tables outside axes - increased height for legend and labels
        base_row_h = float(row_height)
        extra_bottom = (
            1.6 if (show_counts and table_outside and (table_mode in ("summary", "both")))
            else (1.2 if (show_counts and table_outside) else 0.5)
        )
        extra_top = 1.8  # More headroom above for bigger layout / legend
        fig_height = base_row_h * 3 + extra_bottom + extra_top
        fig, axes = plt.subplots(3, 1, figsize=(fig_width, fig_height), sharex=True, sharey=True)
    else:
        # 1 row x 3 columns
        fig_width = max(16.0, min(24.0, 4.5 + 0.35 * max(1, n_steps)))
        fig_height = 5.0
        fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
        'axes.linewidth': 1.2,
        'grid.alpha': 0.25,
    })

    # Compute and draw per endpoint
    any_drawn = False
    for ax_idx, (ax, (ep_name, ep_col)) in enumerate(zip(np.ravel(axes), endpoints)):
        if ep_col not in work.columns:
            ax.set_visible(False)
            continue

        # Collect per-arm trajectories and counts
        trajectories = {}
        counts_map = {}
        times_ref = None
        for arm in arms:
            sub = work[work["group"] == arm]
            if sub.empty or sub[ep_col].dropna().empty:
                continue
            times, center, (low, high), counts = _bootstrap_timecourse_by_group(
                sub, value_col=ep_col, time_col=TIMESTEP_COL, patient_col="patient_id",
                n_boot=n_boot, seed=seed, agg=agg, band_method=band_method,
            )
            if times.size == 0:
                continue
            # establish reference time grid
            if times_ref is None:
                times_ref = times
            else:
                # Align to reference if needed (subset intersection)
                common = np.intersect1d(times_ref, times)
                if common.size == 0:
                    continue
                # Reindex to common
                idx_ref = np.nonzero(np.isin(times_ref, common))[0]
                idx_t = np.nonzero(np.isin(times, common))[0]
                times_ref = common
                # Trim existing stored trajectories to common grid
                for k in list(trajectories.keys()):
                    c, l, h = trajectories[k]
                    trajectories[k] = (c[idx_ref], l[idx_ref], h[idx_ref])
                    counts_map[k] = counts_map[k][idx_ref]
                center, low, high, counts = center[idx_t], low[idx_t], high[idx_t], counts[idx_t]

            trajectories[arm] = (center, low, high)
            counts_map[arm] = counts

        if times_ref is None:
            ax.set_visible(False)
            continue

        # Draw ribbons then lines
        for arm in arms:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            # Choose ribbon color
            base_c = color_map[arm] if use_color else gray_line_map[arm]
            ax.fill_between(times_ref, low, high, color=base_c, alpha=0.18, linewidth=0)
        # Lines with distinct linestyles and optional markers
        step = max(1, int(len(times_ref) / 10))
        markers = {"Surgery only": "o", "Surgery + RT": "s", "Surgery + CT": "^", "Surgery + RT + CT": "D"}
        for arm in arms:
            if arm not in trajectories:
                continue
            center, low, high = trajectories[arm]
            line_c = color_map[arm] if use_color else gray_line_map[arm]
            ax.plot(times_ref, center, linestyle=linestyles[arm], color=line_c, linewidth=2.2, label=arm, zorder=3)
            # sparse markers to aid readability
            ax.plot(times_ref[::step], center[::step], linestyle='None', marker=markers[arm], color=line_c, markersize=4, alpha=0.8, zorder=4)
            if show_errorbars:
                # add error bars at the same sparse locations
                yerr_low = np.maximum(0, center[::step] - low[::step])
                yerr_high = np.maximum(0, high[::step] - center[::step])
                yerr = [yerr_low, yerr_high]
                ax.errorbar(
                    times_ref[::step], center[::step], yerr=yerr,
                    fmt='none', ecolor=line_c, elinewidth=1.0, capsize=2, alpha=0.9, zorder=5
                )

        # Panel title placement
        if side_titles:
            # Left-side vertical label to save vertical space - moved closer to plot
            ax.text(-0.05, 0.5, ep_name, transform=ax.transAxes, rotation=90,
                    va='center', ha='right', fontsize=12)
        else:
            ax.set_title(ep_name, pad=10)
        
        # Add x-axis ticks and labels to plots
        if layout == "rows":
            ax.xaxis.tick_top()
            # Set x-tick labels to start from 1 instead of 0
            if times_ref is not None and len(times_ref) > 0:
                # Create labels starting from 1
                x_labels = [str(i+1) for i in range(len(times_ref))]
                ax.set_xticks(times_ref)
                ax.set_xticklabels(x_labels)
                # Ensure ticks are visible at the top
                ax.tick_params(axis='x', which='both', top=True, bottom=False, labeltop=True, labelbottom=False)
            
            # Add x-axis label only on the first (top) plot to save space
            if ax_idx == 0:
                ax.xaxis.set_label_position('top')
                ax.set_xlabel("Time")
                ax.xaxis.labelpad = 2
            else:
                ax.set_xlabel("")
        
        ax.grid(True, alpha=0.25, linestyle='-')
        ax.set_ylim(0.0, 1.0)
        any_drawn = True

        # Arm-separation over time using binned distributions (mutual information and ΔGini)
        # This captures how distinct the arm distributions are at each timestep,
        # instead of relying on arm center magnitudes which can wash out differences.
        if annotate_arm_dispersion and TIMESTEP_COL in work.columns:
            nbins = 20
            bins = np.linspace(0.0, 1.0, nbins + 1)
            I_vals = []  # normalized mutual information per time (in [0,1])
            dG_vals = [] # normalized delta Gini per time
            for t in times_ref:
                values_by_arm: dict[str, np.ndarray] = {}
                ns: dict[str, int] = {}
                total_n = 0
                for arm in arms:
                    s = pd.to_numeric(
                        work.loc[(work["group"] == arm) & (work[TIMESTEP_COL] == t), ep_col],
                        errors="coerce",
                    ).dropna()
                    s = s[(s >= 0.0) & (s <= 1.0)]
                    if s.size >= 5:
                        arr = s.to_numpy()
                        values_by_arm[arm] = arr
                        ns[arm] = int(arr.size)
                        total_n += int(arr.size)
                # need at least 2 arms and enough samples
                if len(values_by_arm) < 2 or total_n < 10:
                    I_vals.append(np.nan)
                    dG_vals.append(np.nan)
                    continue
                # per-arm histograms to probabilities with eps smoothing
                eps = 1e-12
                P = {}
                for arm, arr in values_by_arm.items():
                    cnt, _ = np.histogram(arr, bins=bins)
                    cnt = cnt.astype(float) + eps
                    P[arm] = cnt / cnt.sum()
                # mixture distribution weighted by arm sample fractions
                weights = {arm: ns[arm] / total_n for arm in values_by_arm}
                p_mix = np.zeros(nbins, dtype=float)
                for arm, p in P.items():
                    p_mix += weights[arm] * p
                # entropy terms (natural log); normalize by ln(nbins)
                H_mix = -float(np.sum(p_mix * np.log(p_mix)))
                H_within = 0.0
                for arm, p in P.items():
                    H_within += weights[arm] * (-float(np.sum(p * np.log(p))))
                I = H_mix - H_within  # equals average KL to mixture; measures separability
                I_norm = I / np.log(nbins)
                # Gini impurity terms (normalized by 1 - 1/nbins)
                def gini(p: np.ndarray) -> float:
                    return 1.0 - float(np.sum(p ** 2))
                G_mix = gini(p_mix)
                G_within = 0.0
                for arm, p in P.items():
                    G_within += weights[arm] * gini(p)
                denom = 1.0 - 1.0 / nbins
                dG_norm = (G_mix - G_within) / denom
                I_vals.append(float(I_norm))
                dG_vals.append(float(dG_norm))

            # Annotate first and last valid values
            I_vals = np.asarray(I_vals, dtype=float)
            dG_vals = np.asarray(dG_vals, dtype=float)
            valid = np.where(np.isfinite(I_vals) & np.isfinite(dG_vals))[0]
            if valid.size >= 1:
                i0, i1 = int(valid[0]), int(valid[-1])
                txt = f"Separation H {I_vals[i0]:.2f}→{I_vals[i1]:.2f}\nΔG {dG_vals[i0]:.2f}→{dG_vals[i1]:.2f}"
                ax.text(
                    0.02, 0.98, txt,
                    transform=ax.transAxes,
                    ha='left', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.65, edgecolor='none'),
                    zorder=6,
                )

        # Build counts table under x-axis (optional)
        if show_counts:
            shown_arms = [arm for arm in arms if arm in counts_map]
            if shown_arms:
                # Show fewer column labels if many time steps
                col_labels = []
                for i, t in enumerate(times_ref):
                    if len(times_ref) <= 16 or i % max(1, len(times_ref) // 16) == 0:
                        col_labels.append(str(i + 1))  # Start from 1 instead of int(t)
                    else:
                        col_labels.append("")
                row_map = {
                    "Surgery only": "S",
                    "Surgery + RT": "S+RT",
                    "Surgery + CT": "S+CT",
                    "Surgery + RT + CT": "S+RT+CT",
                }
                disp_rows = [row_map[a] for a in shown_arms]
                if table_mode == "counts":
                    cell_text = [[str(int(n)) for n in counts_map[a]] for a in shown_arms]
                elif table_mode in ("summary", "both"):
                    # median/mean center with CI per timepoint
                    fmt = lambda c, l, h: ("–" if (np.isnan(c) or np.isnan(l) or np.isnan(h))
                                            else f"{c:.2f} [{l:.2f}–{h:.2f}]")
                    summary_rows = []
                    for a in shown_arms:
                        c, l, h = trajectories[a]
                        summary_rows.append([fmt(c[i], l[i], h[i]) for i in range(len(times_ref))])
                    if table_mode == "summary":
                        cell_text = summary_rows
                    else:  # both
                        cell_text = [[f"n={int(counts_map[a][i])}\n{summary_rows[j][i]}" for i in range(len(times_ref))]
                                     for j, a in enumerate(shown_arms)]

                # Render the table either inside the axes (shifted) or just outside
                if table_outside:
                    # bbox y set negative to place table just below the axis area
                    # increase height if including summary text
                    bbox_height = 0.48 if table_mode in ("summary", "both") else 0.34
                    tbl = ax.table(
                        cellText=cell_text,
                        rowLabels=disp_rows,
                        colLabels=col_labels,
                        cellLoc='center',
                        # table just below axis area; controlled via table_gap
                        bbox=[0.0, -bbox_height - float(table_gap), 1.0, bbox_height],
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(9)
                    # ax.set_xlabel("Time (t)")
                else:
                    tbl = ax.table(
                        cellText=cell_text,
                        rowLabels=disp_rows,
                        colLabels=col_labels,
                        loc='bottom',
                        cellLoc='center',
                    )
                    tbl.scale(1.0, 0.45)
                    # Push plot area up to make space for the table
                    ax.set_xlabel("Time (t)")
                    pos = ax.get_position()
                    ax.set_position([pos.x0, pos.y0 + 0.12, pos.width, pos.height - 0.12])

    # Shared y-label and legend
    if any_drawn:
        if layout == "rows":
            if not side_titles:
                axes[0].set_ylabel("Outcome probability")
            # Build robust legend handles with short labels and place at top center
            legend_colors = color_map if use_color else gray_line_map
            label_map = {
                "Surgery only": "S",
                "Surgery + RT": "S+RT",
                "Surgery + CT": "S+CT",
                "Surgery + RT + CT": "S+RT+CT",
            }
            order = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
            legend_handles = [
                Line2D([0], [0], color=legend_colors[a], linestyle=linestyles[a],
                       linewidth=2.2, label=label_map[a])
                for a in order
            ]
            # Place legend at top-center anchored to the first axis
            axes[0].legend(legend_handles, [h.get_label() for h in legend_handles],
                           loc='upper center', ncol=4, frameon=True,
                           bbox_to_anchor=(0.5, float(legend_y)), fancybox=True, shadow=False, fontsize=12)
            # Increase bottom margin if showing tables outside axes
            need_extra = (show_counts and table_outside)
            has_summary = table_mode in ("summary", "both")
            # Moderate bottom margin so tables have room but axes retain height
            bottom_margin = 0.14 if (need_extra and has_summary) else (0.10 if need_extra else (0.08 if show_counts else 0.05))
            left_margin = 0.08 if side_titles else 0.05  # Reduced since titles are closer
            # Reserve extra top space for the axis-anchored legend while
            # keeping generous vertical room for subplots
            # Reserve space for higher legend and tighten inter-axes spacing
            plt.tight_layout(rect=[left_margin, bottom_margin, 1, float(tight_top)])
            plt.subplots_adjust(hspace=float(panel_hspace))
        else:
            if not side_titles:
                axes[0].set_ylabel("Outcome probability")
            # Explicit legend handles (short labels) for column layout as well
            legend_colors = color_map if use_color else gray_line_map
            label_map = {
                "Surgery only": "S",
                "Surgery + RT": "S+RT",
                "Surgery + CT": "S+CT",
                "Surgery + RT + CT": "S+RT+CT",
            }
            order = ["Surgery only", "Surgery + RT", "Surgery + CT", "Surgery + RT + CT"]
            legend_handles = [
                Line2D([0], [0], color=legend_colors[a], linestyle=linestyles[a],
                       linewidth=2.2, label=label_map[a])
                for a in order
            ]
            # Anchor legend to the middle axis (top-center of the row)
            mid_ax = np.ravel(axes)[1] if len(np.ravel(axes)) >= 2 else np.ravel(axes)[0]
            mid_ax.legend(legend_handles, [h.get_label() for h in legend_handles],
                           loc='upper center', ncol=4, frameon=True,
                           bbox_to_anchor=(0.5, 1.70), fancybox=True, shadow=False, fontsize=11)
            need_extra = (show_counts and table_outside)
            has_summary = table_mode in ("summary", "both")
            bottom_margin = 0.36 if (need_extra and has_summary) else (0.26 if need_extra else (0.14 if show_counts else 0.06))
            left_margin = 0.14 if side_titles else 0.06
            plt.tight_layout(rect=[left_margin, bottom_margin, 1, 0.68])
            plt.subplots_adjust(wspace=0.25)
        out_path = _resolve_out_path(out_path)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"three-endpoint trajectories with bands saved to {out_path}")
    else:
        plt.close(fig)
        print("plot_three_endpoint_trajectories_with_bands: nothing drawn (missing columns/data).")

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



    # Call compute_individualized_treatment_effects
    compute_individualized_treatment_effects(df)

    # Example: risk-averse arm comparison (single-score with CI) on LR endpoint
    try:
        res_lr = compare_two_arms_df(
            df,
            endpoint_col="LR_prob_norm_global" if "LR_prob_norm_global" in df.columns else LOCAL_RECURRENCE_COL,
            arm_A="Surgery only",
            arm_B="Surgery + RT",
            lam_gmd=0.0,
            n_boot=200,
            paired=True,
            seed=42,
        )
        print("\nRisk-averse %Δscore (B vs A) on LR [B=S+RT, A=S]:",
              f"{res_lr['percent_change']:.2f}%  CI95={res_lr['CI95']}  p~={res_lr['p~']:.4f}")
    except Exception as e:
        print(f"Risk-averse arm comparison (LR) skipped: {e}")
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
                n_boot=200,
                seed=42,
            )
            # Trajectory plot with ribbons for this histology
            try:
                safe_hist = str(hist).replace(' ', '_')
                safe_hist = safe_hist.replace('/', '_')
                plot_three_endpoint_trajectories_with_bands(
                    hist_df,
                    histology=str(hist),
                    out_path=f"three_endpoint_trajectories_bands_{safe_hist}.png",
                    n_boot=200,
                    seed=42,
                    agg="median",
                )
                # Also compute risk-averse single-number comparisons for this histology
                grid = risk_averse_comparison_grid(
                    hist_df,
                    lam_gmd=0.0,
                    n_boot=200,
                    paired=True,
                    seed=42,
                    baseline="Surgery only",
                )
                if not grid.empty:
                    out_csv = os.path.join(DEFAULT_IMG_DIR, f"risk_averse_scores_{safe_hist}.csv")
                    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                    grid.to_csv(out_csv, index=False)
                    print(f"Saved risk-averse arm comparisons to {out_csv}")
            except Exception as e:
                print(f"three_endpoint_trajectories_bands (histology={hist}) skipped: {e}")

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
                # Fan‑chart (RCQE) envelope of the ordered composite ITEs
                plot_composite_benefit_fan_envelope(
                    ite_res,
                    treatment=treat,
                    histology=str(hist),
                    out_path=(
                        "composite_benefit_fan_envelope_"
                        f"{str(hist).replace(' ', '_')}_"
                        f"{treat.replace(' ', '_').replace('+', '')}.png"
                    ),
                    n_boot=200,
                    seed=42,
                )

            # Grid of endpoint‑specific fan envelopes across treatments
            plot_endpoint_waterfall_fan_grid(
                ite_res,
                treatments=["Surgery + RT", "Surgery + CT", "Surgery + RT + CT"],
                endpoints=["local_recurrence", "metastasis", "death_of_disease"],
                histology=str(hist),
                out_path=(
                    "waterfall_fan_grid_"
                    f"{str(hist).replace(' ', '_')}.png"
                ),
                n_boot=200,
                seed=42,
            )
                
    
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
                s_df, sh_df, d_df = compute_timeseries_statistics(filtered_df)
                save_timeseries_statistics_csv(f"fnclcc_{grade_str}", s_df, sh_df, d_df)
                # Trajectory plot with ribbons for this FNCLCC grade
                try:
                    plot_three_endpoint_trajectories_with_bands(
                        filtered_df,
                        histology=str(print_grade),
                        out_path=f"three_endpoint_trajectories_bands_fnclcc_{grade_str}.png",
                        n_boot=200,
                        seed=42,
                        agg="median",
                    )
                except Exception as e:
                    print(f"three_endpoint_trajectories_bands (FNCLCC={print_grade}) skipped: {e}")
               
                try:
                    grid = risk_averse_comparison_grid(
                        filtered_df,
                        lam_gmd=0.0,
                        n_boot=200,
                        paired=True,
                        seed=42,
                        baseline="Surgery only",
                    )
                    if not grid.empty:
                        out_csv = os.path.join(
                            DEFAULT_IMG_DIR,
                            f"risk_averse_scores_fnclcc_{grade_str}.csv",
                        )
                        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
                        grid.to_csv(out_csv, index=False)
                        print(
                            "Saved risk-averse arm comparisons "
                            f"(FNCLCC {print_grade}) to {out_csv}"
                        )
                except Exception as e:
                    print(f"Risk-averse FNCLCC comparison (grade={print_grade}) skipped: {e}")
            else:
                print(f"\n--- No data for FNCLCC grade: {print_grade} ---")
    else:
        print(f"\nColumn '{FNCLCC_GRADING_COL}' not found in the CSV. Skipping per-FNCLCC grading analysis.")


if __name__ == "__main__":
    main()
