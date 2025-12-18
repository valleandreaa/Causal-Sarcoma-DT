import os
from typing import Tuple, Literal, Optional, Dict, Sequence, Union, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from scipy.stats import ttest_ind
from scipy.stats import entropy
from scipy.stats import ks_2samp, wasserstein_distance

from Inference.analysis import load_and_prepare


# ------------------------------------------------------------------ #
# Constants and configuration
# ------------------------------------------------------------------ #
CSV_PATH = "counterfactual_scenarios_validation_patients.csv"
# CSV_PATH = "counterfactual_scenarios_multiple_patients_no_embed.csv"
REAL_OUTCOME_COL = "real_episodes.diagnosis.fields.status"
# REAL_OUTCOME_COL = "real_episodes.treatments.fields.endpoint"
HAS_CLIN_COL = "has_clinical_data"

FAKE_STATUS_PROB_COL_TPL = "fake_episodes.diagnosis.fields.status_{label}_prob"
# FAKE_STATUS_PROB_COL_TPL = "fake_episodes.treatments.fields.endpoint_{label}_prob"

def _ensure_dir(path: str) -> None:
    d = os.path.dirname(path) or "imgs"
    if d:
        os.makedirs(d, exist_ok=True)


def _select_labels_scores(
    df: pd.DataFrame,
    positive_label: str = "DOD",
    use_has_clinical: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract binary labels and corresponding predicted scores for the given
    class label from the combined real/fake dataframe.

    - Labels are 1 if REAL_OUTCOME_COL equals ``positive_label`` and 0 otherwise.
    - Scores are the fake predicted probability for ``positive_label``.
    - If ``use_has_clinical`` is True, only rows with real clinical data are used.
    """
    data = df.copy()
    if use_has_clinical and HAS_CLIN_COL in data.columns:
        data = data[data[HAS_CLIN_COL] == True]

    score_col = FAKE_STATUS_PROB_COL_TPL.format(label=positive_label)
    if score_col not in data.columns:
        raise KeyError(f"Missing score column: {score_col}")

    if "patient_id" not in data.columns:
        raise KeyError("Missing column: patient_id for cluster bootstrap")

    # Safer masking: drop rows with missing true label, score, or patient_id
    m = data[REAL_OUTCOME_COL].notna() & data[score_col].notna() & data["patient_id"].notna()
    data = data.loc[m]

    labels = (data[REAL_OUTCOME_COL] == positive_label).to_numpy(dtype=float)
    scores = data[score_col].to_numpy(dtype=float)
    pid_arr = data["patient_id"].to_numpy()
    if labels.size == 0:
        raise ValueError("No valid (label, score) pairs after filtering.")
    return labels, scores, pid_arr


# ------------------------------------------------------------------ #
# Bootstrap utilities
# ------------------------------------------------------------------ #
def bootstrap_auc(
    y: np.ndarray,
    s: np.ndarray,
    pids: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: Optional[int] = 42,
) -> Tuple[float, np.ndarray, Tuple[float, float]]:
    """
    Compute AUC and its bootstrap distribution + 95% CI via percentile method.
    Returns (auc, auc_samples, (ci_low, ci_high)).
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]
    auc0 = roc_auc_score(y, s)

    samples = np.empty(n_boot, dtype=float)
    if pids is None:
        # IID bootstrap over rows
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            try:
                samples[b] = roc_auc_score(y[idx], s[idx])
            except ValueError:
                samples[b] = np.nan
    else:
        # Cluster bootstrap: sample patients with replacement, carry all timesteps
        # Factorize patient IDs once for fast cluster mapping
        codes, uniques = pd.factorize(pids, sort=False)
        idx_lists = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(uniques)
        for b in range(n_boot):
            # Draw cluster indices with replacement
            draw = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([idx_lists[d] for d in draw], axis=0)
            yy, ss = y[idx], s[idx]
            if yy.size == 0 or yy.min() == yy.max():
                samples[b] = np.nan
                continue
            try:
                samples[b] = roc_auc_score(yy, ss)
            except ValueError:
                samples[b] = np.nan
    samples = samples[~np.isnan(samples)]

    ci = (np.percentile(samples, 2.5), np.percentile(samples, 97.5)) if samples.size else (np.nan, np.nan)
    return auc0, samples, ci


def _bootstrap_curve(
    y: np.ndarray,
    s: np.ndarray,
    kind: Literal["roc", "pr"],
    pids: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: Optional[int] = 42,
    grid_size: int = 200,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Bootstrap pointwise bands for ROC or PR curves.

    Returns (x_grid, mean_y, (low, high), (x_ref, y_ref)) where x/y refer to
    FPR/TPR for ROC or Recall/Precision for PR.
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]

    if kind == "roc":
        x_ref, y_ref, _ = roc_curve(y, s)
        x_grid = np.linspace(0.0, 1.0, grid_size)
        # Interpolate TPR at fixed FPR grid
        def curve(yy, ss):
            fpr, tpr, _ = roc_curve(yy, ss)
            # Ensure strictly increasing x for interpolation
            ufpr, idx = np.unique(fpr, return_index=True)
            tpr = tpr[idx]
            return np.interp(x_grid, ufpr, tpr, left=0.0, right=1.0)
    else:
        pr_recall, pr_precision, _ = precision_recall_curve(y, s)
        # precision_recall_curve returns recall increasing, precision values
        x_ref, y_ref = pr_recall, pr_precision
        x_grid = np.linspace(0.0, 1.0, grid_size)
        # Interpolate Precision at fixed Recall grid
        def curve(yy, ss):
            r, p, _ = precision_recall_curve(yy, ss)
            # Remove possible duplicate recall values before interp
            uniq_r, idx = np.unique(r, return_index=True)
            p = p[idx]
            return np.interp(x_grid, uniq_r, p, left=p[0], right=p[-1])

    # Precompute cluster mapping if needed
    if pids is not None:
        # Factorize patient IDs once for fast cluster mapping
        codes, uniques = pd.factorize(pids, sort=False)
        idx_lists = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(uniques)

    curves = []
    for b in range(n_boot):
        if pids is None:
            idx = rng.integers(0, n, size=n)
        else:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([idx_lists[d] for d in draw], axis=0)

        yy, ss = y[idx], s[idx]
        # Skip ill-posed resamples (only one class present)
        if yy.size == 0 or yy.min() == yy.max():
            continue
        curves.append(curve(yy, ss))

    if not curves:
        raise ValueError("All bootstrap resamples were ill-posed (single-class).")

    arr = np.vstack(curves)
    low = np.percentile(arr, 2.5, axis=0)
    high = np.percentile(arr, 97.5, axis=0)
    mean = np.nanmean(arr, axis=0)
    return x_grid, mean, (low, high), (x_ref, y_ref)


def _bootstrap_calibration(
    y: np.ndarray,
    s: np.ndarray,
    n_bins: int = 10,
    pids: Optional[np.ndarray] = None,
    n_boot: int = 2000,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Bootstrap bands for calibration. Uses fixed bin edges from full data
    (equal-frequency quantile bins) for consistent aggregation across resamples.

    Returns (bin_centers, mean_true, (low_true, high_true), mean_pred)
    where mean_pred is the mean predicted probability per bin from the full data
    (used for x-axis positions).
    """
    rng = np.random.default_rng(seed)
    n = y.shape[0]

    # Determine bin edges on full data (quantiles for roughly equal counts)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(s, quantiles)
    # Avoid zero-width bins due to ties by adding small jitter to edges
    edges[0], edges[-1] = 0.0, 1.0

    def bin_stats(yy: np.ndarray, ss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # digitize returns indices in 1..n_bins; map to 0..n_bins-1
        b = np.digitize(ss, edges[1:-1], right=True)
        true_rate = np.full(n_bins, np.nan, dtype=float)
        pred_mean = np.full(n_bins, np.nan, dtype=float)
        counts = np.zeros(n_bins, dtype=int)
        for k in range(n_bins):
            m = b == k
            counts[k] = int(m.sum())
            if counts[k] > 0:
                true_rate[k] = yy[m].mean()
                pred_mean[k] = ss[m].mean()
        return true_rate, pred_mean, counts

    # Full-data reference
    true_ref, pred_ref, _ = bin_stats(y, s)
    centers = pred_ref

    # Bootstrap bands for true_rate
    # Precompute cluster mapping if needed
    if pids is not None:
        # Factorize patient IDs once for fast cluster mapping
        codes, uniques = pd.factorize(pids, sort=False)
        idx_lists = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(uniques)

    mat = []
    for b in range(n_boot):
        if pids is None:
            idx = rng.integers(0, n, size=n)
        else:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([idx_lists[d] for d in draw], axis=0)

        yy, ss = y[idx], s[idx]
        if yy.size == 0 or yy.min() == yy.max():
            continue
        tr, _, _ = bin_stats(yy, ss)
        mat.append(tr)

    arr = np.vstack(mat)
    low = np.nanpercentile(arr, 2.5, axis=0)
    high = np.nanpercentile(arr, 97.5, axis=0)
    mean_true = np.nanmean(arr, axis=0)
    return centers, mean_true, (low, high), pred_ref


# ------------------------------------------------------------------ #
# Plotting
# ------------------------------------------------------------------ #
def plot_metric_distribution(
    samples: np.ndarray,
    point_estimate: float,
    ci: Tuple[float, float],
    out_path: str,
    title: str = "AUC bootstrap distribution",
) -> None:
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(samples, bins=40, color="#1f77b4", alpha=0.7, edgecolor="black")
    ax.axvline(point_estimate, color="#d62728", lw=2, label=f"AUC = {point_estimate:.3f}")
    if np.isfinite(ci[0]) and np.isfinite(ci[1]):
        ax.axvspan(ci[0], ci[1], color="#ff7f0e", alpha=0.2, label=f"95% CI [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set_xlabel("AUC")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_curve_with_bands(
    x_grid: np.ndarray,
    mean_y: np.ndarray,
    bands: Tuple[np.ndarray, np.ndarray],
    ref_curve: Tuple[np.ndarray, np.ndarray],
    out_path: str,
    kind: Literal["roc", "pr"] = "roc",
    title: Optional[str] = None,
) -> None:
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    low, high = bands
    ax.fill_between(x_grid, low, high, color="#1f77b4", alpha=0.2, label="95% band")
    ax.plot(x_grid, mean_y, color="#1f77b4", lw=2, label="Bootstrap mean")
    ax.plot(ref_curve[0], ref_curve[1], color="#d62728", lw=1.5, linestyle="--", label="Empirical curve")

    if kind == "roc":
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or "ROC with pointwise bands")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title or "PR with pointwise bands")

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_calibration_with_bands(
    centers: np.ndarray,
    mean_true: np.ndarray,
    bands: Tuple[np.ndarray, np.ndarray],
    out_path: str,
    title: str = "Calibration curve with bands",
) -> None:
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    low, high = bands
    # Diagonal
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":", label="Perfect calibration")

    # Bands and curve
    ax.fill_between(centers, low, high, color="#1f77b4", alpha=0.2, label="95% band")
    ax.plot(centers, mean_true, color="#1f77b4", lw=2, label="Bootstrap mean")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ #
# Natural experiment (with bootstrap uncertainty)
# ------------------------------------------------------------------ #
def evaluate_natural_experiment(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    match_cols: Sequence[str],
    outcome_col: str,
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[float, float, Tuple[float, float]]:
    """
    Natural experiment like Inference.analysis.evaluate_natural_experiment but with
    bootstrap uncertainty for Wasserstein-1.

    - Performs a naive exact-match merge on ``match_cols`` between real and synth.
    - Compares ``outcome_col`` across matched pairs using Wasserstein-1 and KS test.
    - Returns (w1_point, ks_pvalue, (w1_ci_low, w1_ci_high)).
    """
    # Prepare merged pairs
    if outcome_col not in real_df.columns:
        raise KeyError(f"real_df missing outcome column: {outcome_col}")
    if outcome_col not in synth_df.columns:
        raise KeyError(f"synth_df missing outcome column: {outcome_col}")

    # Select only needed columns to avoid overlap explosion
    keep_cols = list(match_cols) + [outcome_col]
    if "patient_id" in real_df.columns:
        keep_cols_real = keep_cols + ["patient_id"]
    else:
        keep_cols_real = keep_cols
    if "patient_id" in synth_df.columns:
        keep_cols_synth = keep_cols + ["patient_id"]
    else:
        keep_cols_synth = keep_cols

    merged = (
        real_df[keep_cols_real]
        .merge(
            synth_df[keep_cols_synth],
            on=list(match_cols),
            suffixes=("_real", "_synth"),
        )
    )

    if merged.empty:
        raise ValueError("No matched pairs after merging on match_cols.")

    y_real = pd.to_numeric(merged[outcome_col + "_real"], errors="coerce").dropna()
    y_synth = pd.to_numeric(merged[outcome_col + "_synth"], errors="coerce").dropna()

    n = min(len(y_real), len(y_synth))
    if n == 0:
        raise ValueError("No valid matched outcome pairs after dropping NaNs.")

    # Align lengths if dropna removed different rows
    # Use indices intersection to keep aligned pairs
    idx = y_real.index.intersection(y_synth.index)
    y_real = y_real.loc[idx].to_numpy()
    y_synth = y_synth.loc[idx].to_numpy()

    # Point estimates
    w1 = float(wasserstein_distance(y_real, y_synth))
    _, ks_p = ks_2samp(y_real, y_synth, alternative="two-sided", mode="asymp")
    ks_p = float(ks_p)

    # Bootstrap W1 over matched pairs; cluster by real patient if available
    rng = np.random.default_rng(seed)
    w1_samples = np.empty(n_boot, dtype=float)

    # Choose cluster IDs if present
    pid_cols = [c for c in merged.columns if c.startswith("patient_id")]
    real_pid_col = None
    for c in ("patient_id_real", "patient_id_x", "patient_id_real", "patient_id"):
        if c in merged.columns:
            real_pid_col = c
            break
    # Fallback: try any patient_id-like column
    if real_pid_col is None and pid_cols:
        real_pid_col = pid_cols[0]

    if real_pid_col is not None:
        # Factorize over aligned indices only
        pid_aligned = merged.loc[idx, real_pid_col]
        codes, uniques = pd.factorize(pid_aligned, sort=False)
        clusters = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(clusters)
        for b in range(n_boot):
            draw = rng.integers(0, n_clusters, size=n_clusters)
            sel = np.concatenate([clusters[d] for d in draw], axis=0)
            w1_samples[b] = wasserstein_distance(y_real[sel], y_synth[sel]) if sel.size else np.nan
    else:
        m = y_real.shape[0]
        for b in range(n_boot):
            sel = rng.integers(0, m, size=m)
            w1_samples[b] = wasserstein_distance(y_real[sel], y_synth[sel]) if sel.size else np.nan

    w1_samples = w1_samples[~np.isnan(w1_samples)]
    w1_ci = (
        float(np.percentile(w1_samples, 2.5)) if w1_samples.size else np.nan,
        float(np.percentile(w1_samples, 97.5)) if w1_samples.size else np.nan,
    )

    print(
        f"Natural experiment – W1={w1:.4f} (95% CI [{w1_ci[0]:.4f}, {w1_ci[1]:.4f}]), "
        f"KS p-value: {ks_p:.4g}"
    )
    return w1, ks_p, w1_ci


# ------------------------------------------------------------------ #
# Marginal distribution alignment with bootstrap uncertainty
# ------------------------------------------------------------------ #
def evaluate_marginal_distribution_alignment(
    df: pd.DataFrame,
    treatment_col: str = "treatment_scenario",
    real_outcome_col: str = "real_episodes.diagnosis.fields.status",
    prob_cols: Optional[Sequence[str]] = None,
    use_has_clinical: bool = True,
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap the KL divergence for marginal distribution alignment over treatments:
      KL_total = sum_T P_real(T) * KL( P_fake(Y|T) || P_real(Y|T) )

    - real_outcome_col: column with real outcomes (categorical labels).
    - prob_cols: fake class probability columns (must sum to 1 per row). If None,
      defaults to diagnosis: status_AWD_prob, status_DOD_prob, status_NED_prob.
    - Patient-cluster bootstrap (by patient_id) if available; otherwise IID row bootstrap.

    Returns (kl_point, (ci_low, ci_high)).
    """
    if prob_cols is None:
        default_prob_cols = [
            'fake_episodes.diagnosis.fields.status_AWD_prob',
            'fake_episodes.diagnosis.fields.status_DOD_prob',
            'fake_episodes.diagnosis.fields.status_NED_prob',
        ]
        prob_cols = [c for c in default_prob_cols if c in df.columns]

    # Validate required columns
    missing = [c for c in [treatment_col, real_outcome_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required column(s): {missing}")
    if not prob_cols:
        raise KeyError("No fake probability columns found for outcome inference.")

    # Helper to compute KL_total on provided dataframe sample
    def kl_on_sample(sample: pd.DataFrame) -> float:
        s = sample.copy()
        if use_has_clinical and HAS_CLIN_COL in s.columns:
            real_s = s[s[HAS_CLIN_COL] == True]
        else:
            real_s = s

        # Determine fake outcome label via argmax
        # Extract outcome names from column suffixes (_LABEL_prob)
        outcome_labels = [col.split('_')[-2] for col in prob_cols]
        # Use idxmax to get the winning full column name; then map to label
        argmax_col = s[prob_cols].idxmax(axis=1)
        s['fake_outcome'] = argmax_col.map({col: lab for col, lab in zip(prob_cols, outcome_labels)})

        # P_real(T)
        pT = real_s[treatment_col].value_counts(normalize=True)
        if pT.empty:
            return np.nan

        kl_total = 0.0
        # all possible outcomes observed in real data
        all_outcomes = sorted(s[real_outcome_col].dropna().unique())
        eps = 1e-10
        for t, p_t in pT.items():
            real_dist = real_s.loc[real_s[treatment_col] == t, real_outcome_col].value_counts(normalize=True)
            fake_dist = s.loc[s[treatment_col] == t, 'fake_outcome'].value_counts(normalize=True)
            if real_dist.empty or fake_dist.empty:
                # If either side has no support for this treatment in the sample, skip
                # contribution for this T (implicitly 0 weight or undefined); continue.
                continue
            real_dist = real_dist.reindex(all_outcomes, fill_value=0)
            fake_dist = fake_dist.reindex(all_outcomes, fill_value=0)

            pr = real_dist.values + eps
            pf = fake_dist.values + eps
            pr /= pr.sum()
            pf /= pf.sum()
            kl_t = float(np.sum(pf * (np.log(pf) - np.log(pr))))
            kl_total += float(p_t) * kl_t
        return float(kl_total)

    # Point estimate on full data
    kl_point = kl_on_sample(df)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    have_pid = "patient_id" in df.columns and df["patient_id"].notna().any()
    if have_pid:
        codes, uniques = pd.factorize(df["patient_id"], sort=False)
        idx_lists = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(uniques)
    else:
        n_rows = len(df)

    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if have_pid:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([idx_lists[d] for d in draw], axis=0)
        else:
            idx = rng.integers(0, n_rows, size=n_rows)
        s_df = df.iloc[idx]
        samples[b] = kl_on_sample(s_df)

    samples = samples[~np.isnan(samples)]
    kl_ci = (
        float(np.percentile(samples, 2.5)) if samples.size else np.nan,
        float(np.percentile(samples, 97.5)) if samples.size else np.nan,
    )
    boot_median = float(np.nanmedian(samples)) if samples.size else np.nan

    print(
        f"Marginal KL divergence: {kl_point:.4f}; "
        f"bootstrap median {boot_median:.4f}; "
        f"95% CI [{kl_ci[0]:.4f}, {kl_ci[1]:.4f}]"
    )
    return kl_point, kl_ci


# ------------------------------------------------------------------ #
# Joint distribution alignment with bootstrap uncertainty
# ------------------------------------------------------------------ #
def evaluate_joint_distribution_alignment(
    df: pd.DataFrame,
    treatment_col: str,
    real_outcome_col: str = "real_episodes.diagnosis.fields.status",
    prob_cols: Optional[Sequence[str]] = None,
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Tuple[float, Tuple[float, float]]:
    """
    Compute KL( P_real(T,Y) || P_fake(T,Y) ) comparing joint distributions of
    treatment and outcome, with 95% bootstrap CI (patient-cluster if available).

    - real_outcome_col: categorical real outcome column.
    - prob_cols: fake class probability columns; argmax gives fake outcome label.
      Defaults to diagnosis status prob columns if present.
    Returns (kl_point, (ci_low, ci_high)).
    """
    if prob_cols is None:
        default_prob_cols = [
            'fake_episodes.diagnosis.fields.status_AWD_prob',
            'fake_episodes.diagnosis.fields.status_DOD_prob',
            'fake_episodes.diagnosis.fields.status_NED_prob',
        ]
        prob_cols = [c for c in default_prob_cols if c in df.columns]
    if not prob_cols:
        raise KeyError("No fake probability columns found for joint alignment.")
    if treatment_col not in df.columns or real_outcome_col not in df.columns:
        raise KeyError("Missing treatment_col or real_outcome_col for joint alignment.")

    def kl_on_sample(s: pd.DataFrame) -> float:
        real_df = s.copy()
        fake_df = s.copy()

        # Real outcomes
        real_df = real_df.assign(outcome=real_df[real_outcome_col])

        # Fake outcomes via argmax
        argmax_col = fake_df[prob_cols].idxmax(axis=1)
        outcome_labels = [col.split('_')[-2] for col in prob_cols]
        label_map = {col: lab for col, lab in zip(prob_cols, outcome_labels)}
        fake_df = fake_df.assign(outcome=argmax_col.map(label_map))

        # Joint counts
        real_joint = real_df.groupby([treatment_col, 'outcome']).size().unstack(fill_value=0)
        fake_joint = fake_df.groupby([treatment_col, 'outcome']).size().unstack(fill_value=0)

        # Align axes
        all_outcomes = sorted(list(set(real_joint.columns) | set(fake_joint.columns)))
        real_joint = real_joint.reindex(columns=all_outcomes, fill_value=0)
        fake_joint = fake_joint.reindex(columns=all_outcomes, fill_value=0)

        all_treats = sorted(list(set(real_joint.index) | set(fake_joint.index)))
        real_joint = real_joint.reindex(index=all_treats, fill_value=0)
        fake_joint = fake_joint.reindex(index=all_treats, fill_value=0)

        # Normalize
        p_real = real_joint.values.flatten().astype(float)
        p_fake = fake_joint.values.flatten().astype(float)
        if p_real.sum() == 0 or p_fake.sum() == 0:
            return np.nan
        p_real /= p_real.sum()
        p_fake /= p_fake.sum()

        # Smooth and KL
        eps = 1e-10
        pr = p_real + eps
        pf = p_fake + eps
        pr /= pr.sum()
        pf /= pf.sum()
        return float(entropy(pr, pf))

    # Point estimate
    point = kl_on_sample(df)

    # Bootstrap
    rng = np.random.default_rng(seed)
    have_pid = "patient_id" in df.columns and df["patient_id"].notna().any()
    if have_pid:
        codes, uniques = pd.factorize(df["patient_id"], sort=False)
        idx_lists = [np.flatnonzero(codes == k) for k in range(len(uniques))]
        n_clusters = len(uniques)
    else:
        n_rows = len(df)

    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        if have_pid:
            draw = rng.integers(0, n_clusters, size=n_clusters)
            idx = np.concatenate([idx_lists[d] for d in draw], axis=0)
        else:
            idx = rng.integers(0, n_rows, size=n_rows)
        samples[b] = kl_on_sample(df.iloc[idx])

    samples = samples[~np.isnan(samples)]
    ci = (
        float(np.percentile(samples, 2.5)) if samples.size else np.nan,
        float(np.percentile(samples, 97.5)) if samples.size else np.nan,
    )
    boot_median = float(np.nanmedian(samples)) if samples.size else np.nan

    print(
        f"\nJoint Distribution KL Divergence: {point:.4f}; "
        f"bootstrap median {boot_median:.4f}; "
        f"95% CI [{ci[0]:.4f}, {ci[1]:.4f}]"
    )
    return point, ci


# ------------------------------------------------------------------ #
# Natural experiment by treatment (bootstrap uncertainty)
# ------------------------------------------------------------------ #
def evaluate_natural_experiment_by_treatment(
    df: pd.DataFrame,
    match_cols: Sequence[str],
    scenario_map: Dict[str, str],
    positive_label: str = 'AWD',
    n_boot: int = 200,
    seed: Optional[int] = 42,
) -> Dict[str, Tuple[float, float, Tuple[float, float]]]:
    """
    Compare real outcomes for a treatment scenario with counterfactual outcomes
    for a different, matched patient group. Adds bootstrap uncertainty for W1.

    For each (real_scenario, fake_scenario):
      1) Real patients who truly received real_scenario (via actual_* flags)
      2) Counterfactual predictions for patients under fake_scenario
      3) Match on match_cols, then compare real binary outcome vs fake prob

    Returns mapping: scenario_key -> (W1 point, KS p-value, (W1 CI low, high))
    where scenario_key is f"{real_scenario}->{fake_scenario}".
    """
    print("\n--- Evaluating Natural Experiments by Treatment Scenario (bootstrap) ---")
    results: Dict[str, Tuple[float, float, Tuple[float, float]]] = {}

    fake_prob_col = FAKE_STATUS_PROB_COL_TPL.format(label=positive_label)
    if fake_prob_col not in df.columns:
        raise KeyError(f"Missing fake probability column: {fake_prob_col}")

    def get_treatment_flags(scenario: str):
        if scenario == "S":
            return {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0}
        if scenario == "S_CT":
            return {"surgery": 1, "chemotherapy": 1, "radiotherapy": 0}
        if scenario == "S_RT":
            return {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1}
        if scenario == "S_RT_CT":
            return {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1}
        return None

    rng = np.random.default_rng(seed)

    for real_scenario, fake_scenario in scenario_map.items():
        print(f"\nComparing Real '{real_scenario}' vs. Fake '{fake_scenario}'")
        real_flags = get_treatment_flags(real_scenario)
        if real_flags is None:
            print(f"Unknown real scenario: {real_scenario}. Skipping.")
            continue

        # Group 1: real data rows with actual_* flags matching the scenario
        real_condition = (df[HAS_CLIN_COL] == True)
        for flag, value in real_flags.items():
            real_condition &= (df[f"actual_{flag}"] == value)
        real_group = df.loc[real_condition].copy()

        # Group 2: counterfactual rows for the fake scenario
        fake_group = df.loc[df['treatment_scenario'] == fake_scenario].copy()

        if real_group.empty:
            print(f"Warning: No real data for '{real_scenario}'. Skipping.")
            continue
        if fake_group.empty:
            print(f"Warning: No fake data for '{fake_scenario}'. Skipping.")
            continue

        # Prepare outcomes; carry patient_id if present for cluster bootstrap
        real_group['outcome_binary'] = (real_group[REAL_OUTCOME_COL] == positive_label).astype(float)
        real_cols = list(match_cols) + ['outcome_binary'] + (["patient_id"] if 'patient_id' in real_group.columns else [])
        fake_cols = list(match_cols) + [fake_prob_col] + (["patient_id"] if 'patient_id' in fake_group.columns else [])
        real_outcomes = real_group[real_cols].dropna(subset=['outcome_binary'])
        fake_outcomes = fake_group[fake_cols].dropna(subset=[fake_prob_col])

        # Inner join on match columns to form matched pairs
        merged = pd.merge(
            real_outcomes,
            fake_outcomes,
            on=list(match_cols),
            how='inner',
            suffixes=("_real", "_fake"),
        )

        if merged.empty:
            print(f"No matched patients between real '{real_scenario}' and fake '{fake_scenario}'. Skipping.")
            continue

        y_real = merged['outcome_binary'].astype(float).to_numpy()
        y_fake = merged[fake_prob_col].astype(float).to_numpy()
        if y_real.size < 2 or y_fake.size < 2:
            print(f"Not enough matched samples to compare ({y_real.size} real, {y_fake.size} fake). Skipping.")
            continue

        # Point metrics
        wd = float(wasserstein_distance(y_real, y_fake))  # true 1D W1/EMD
        try:
            ks_stat, ks_p = ks_2samp(y_real, y_fake, alternative='two-sided', mode='auto')
            ks_stat = float(ks_stat)
            ks_p = float(ks_p)
        except Exception:
            ks_stat = np.nan
            ks_p = np.nan

        # Bootstrap W1 over matched rows; cluster by real patient if available
        wd_samples = np.empty(n_boot, dtype=float)
        # Determine real patient_id column after merge
        pid_real_col = None
        for c in ("patient_id_real", "patient_id_x", "patient_id"):
            if c in merged.columns:
                pid_real_col = c
                break

        if pid_real_col is not None:
            codes, uniques = pd.factorize(merged[pid_real_col], sort=False)
            clusters = [np.flatnonzero(codes == k) for k in range(len(uniques))]
            n_clusters = len(clusters)
            for b in range(n_boot):
                draw = rng.integers(0, n_clusters, size=n_clusters)
                sel = np.concatenate([clusters[d] for d in draw], axis=0)
                wd_samples[b] = wasserstein_distance(y_real[sel], y_fake[sel]) if sel.size else np.nan
        else:
            m = y_real.shape[0]
            for b in range(n_boot):
                sel = rng.integers(0, m, size=m)
                wd_samples[b] = wasserstein_distance(y_real[sel], y_fake[sel]) if sel.size else np.nan

        wd_samples = wd_samples[~np.isnan(wd_samples)]
        wd_ci = (
            float(np.percentile(wd_samples, 2.5)) if wd_samples.size else np.nan,
            float(np.percentile(wd_samples, 97.5)) if wd_samples.size else np.nan,
        )
        wd_median = float(np.median(wd_samples)) if wd_samples.size else np.nan

        key = f"{real_scenario}->{fake_scenario}"
        results[key] = (wd, ks_p, wd_ci)

        print(f"Found {merged.shape[0]} matched pairs.")
        print(f"Wasserstein-1 distance: {wd:.4f} (bootstrap median: {wd_median:.4f}) (95% CI [{wd_ci[0]:.4f}, {wd_ci[1]:.4f}])")
        print(f"KS statistic: {ks_stat:.4f}, p-value: {ks_p:.4g}")

    return results


# ------------------------------------------------------------------ #
# Main entry
# ------------------------------------------------------------------ #
def run(
    csv_path: str = CSV_PATH,
    positive_label: str = "DOD",
    n_boot: int = 2000,
    seed: Optional[int] = 42,
) -> None:
    """
    Run bootstrap validation and generate plots:
    - Metric distribution + CI (AUC)
    - ROC with pointwise bands
    - PR with pointwise bands
    - Calibration curve with bands
    """
    df = load_and_prepare(csv_path)
    y, s, pids = _select_labels_scores(df, positive_label=positive_label, use_has_clinical=True)
    
    # 1) AUC bootstrap distribution
    auc0, auc_samples, auc_ci = bootstrap_auc(y, s, pids=pids, n_boot=max(2000, n_boot), seed=seed)
    print(f"AUC ({positive_label} vs rest): {auc0:.3f}; 95% CI [{auc_ci[0]:.3f}, {auc_ci[1]:.3f}] (percentile)")
    plot_metric_distribution(
        auc_samples,
        point_estimate=auc0,
        ci=auc_ci,
        out_path=f"imgs/bootstrap/{positive_label}_auc_bootstrap.png",
        title=f"AUC bootstrap distribution ({positive_label})",
    )

    # 2) ROC with bands
    xg, mean, bands, ref = _bootstrap_curve(y, s, kind="roc", pids=pids, n_boot=n_boot, seed=seed)
    plot_curve_with_bands(
        xg,
        mean,
        bands,
        ref,
        out_path=f"imgs/bootstrap/{positive_label}_roc_bands.png",
        kind="roc",
        title=f"ROC with bands ({positive_label})",
    )

    # 3) PR with bands
    xg, mean, bands, ref = _bootstrap_curve(y, s, kind="pr", pids=pids, n_boot=n_boot, seed=seed)
    plot_curve_with_bands(
        xg,
        mean,
        bands,
        ref,
        out_path=f"imgs/bootstrap/{positive_label}_pr_bands.png",
        kind="pr",
        title=f"PR with bands ({positive_label})",
    )

    # 4) Calibration with bands
    centers, mean_true, bands, _ = _bootstrap_calibration(y, s, n_bins=10, pids=pids, n_boot=n_boot, seed=seed)
    plot_calibration_with_bands(
        centers,
        mean_true,
        bands,
        out_path=f"imgs/bootstrap/{positive_label}_calibration_bands.png",
        title=f"Calibration with bands ({positive_label})",
    )

    # 5) Natural experiment with bootstrap uncertainty (endpoint-based)
    try:
        match_cols = ["general.age", "tumor_characteristics.histological_diagnosis"]
        # Build real and synthetic frames and align outcome column name
        outcome_col = "real_episodes.treatments.fields.endpoint"
        real_df = df[df[HAS_CLIN_COL] == True].copy()
        # Construct synth_df with only the synthetic endpoint, renamed to outcome_col
        synth_outcome_col = "fake_episodes.treatments.fields.endpoint"
        needed_cols = match_cols + [synth_outcome_col]
        if "patient_id" in df.columns:
            needed_cols = needed_cols + ["patient_id"]
        synth_df = df[needed_cols].rename(columns={synth_outcome_col: outcome_col}).copy()
        _ = evaluate_natural_experiment(
            real_df=real_df,
            synth_df=synth_df,
            match_cols=match_cols,
            outcome_col=outcome_col,
            n_boot=max(200, n_boot),
            seed=seed,
        )
    except Exception as e:
        print(f"Natural experiment evaluation skipped: {e}")

    # 6) Marginal distribution alignment with bootstrap (diagnosis/status based)
    try:
        _ = evaluate_marginal_distribution_alignment(
            df,
            treatment_col="treatment_scenario",
            real_outcome_col="real_episodes.diagnosis.fields.status",
            prob_cols=[
                'fake_episodes.diagnosis.fields.status_AWD_prob',
                'fake_episodes.diagnosis.fields.status_DOD_prob',
                'fake_episodes.diagnosis.fields.status_NED_prob',
            ],
            use_has_clinical=True,
            n_boot=max(200, n_boot),
            seed=seed,
        )
    except Exception as e:
        print(f"Marginal distribution alignment skipped: {e}")

    # 7) Joint distribution alignment with bootstrap (diagnosis/status based)
    try:
        _ = evaluate_joint_distribution_alignment(
            df,
            treatment_col="treatment_scenario",
            real_outcome_col="real_episodes.diagnosis.fields.status",
            prob_cols=[
                'fake_episodes.diagnosis.fields.status_AWD_prob',
                'fake_episodes.diagnosis.fields.status_DOD_prob',
                'fake_episodes.diagnosis.fields.status_NED_prob',
            ],
            n_boot=max(200, n_boot),
            seed=seed,
        )
    except Exception as e:
        print(f"Joint distribution alignment skipped: {e}")

    # 8) Natural experiment by treatment scenario (bootstrap, diagnosis/status positive_label)
    try:
        scenario_map = {
            "S_CT": "S_CT",
            "S_RT": "S_RT",
            "S_RT_CT": "S_RT_CT",
            "S": "S",
        }
        _ = evaluate_natural_experiment_by_treatment(
            df,
            match_cols=["general.age", "tumor_characteristics.histological_diagnosis"],
            scenario_map=scenario_map,
            positive_label=positive_label,
            n_boot=max(200, n_boot),
            seed=seed,
        )
    except Exception as e:
        print(f"Natural experiment by treatment skipped: {e}")


if __name__ == "__main__":
    # Simple CLI via environment variables (optional):
    #   POS_LABEL in {AWD,DOD,NED}
    #   CSV_PATH to override default path
    pos_label = os.environ.get("POS_LABEL", 'AWD')
    csv_path = os.environ.get("CSV_PATH", CSV_PATH)
    try:
        run(csv_path=csv_path, positive_label=pos_label, n_boot=2000, seed=42)
    except Exception as e:
        print(f"validation_bootstrap failed: {e}")
