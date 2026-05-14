"""CLI: Estimate treatment effects using do-intervention on a BN.

Supports multiple causal inference approaches:
1. Standard do-calculus using the Bayesian Network structure
2. IPTW (Inverse Probability of Treatment Weighting) to address confounding by indication

IPTW is the recommended approach when treatments are assigned based on patient
characteristics (e.g., sicker patients receiving more aggressive treatment).
It balances baseline covariates by weighting observations by 1/P(T|X), where
P(T|X) is the propensity score estimated from the BN structure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_auc_score, average_precision_score, log_loss, brier_score_loss,
    roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from dtcygan.bn import load_bayesian_network
from dtcygan.causal import average_treatment_effect, do_intervene
from pgmpy.inference import VariableElimination  # type: ignore
from pgmpy.factors.discrete import TabularCPD  # type: ignore
# Reuse data cleaning from fit_bn to ensure consistent encoding
from fit_bn import _apply_feature_selection, _validate_and_clean_data, _load_config

# Silence repetitive pgmpy CPD-replacement warnings triggered by virtual evidence queries.
logging.getLogger("pgmpy").setLevel(logging.ERROR)
import pickle
# When `last_status` is exploded into binaries, these are the nodes produced
STATUS_NODES = ["NED", "AWD", "DOD"]

# AJCC 8th edition staging survival probabilities (5-year OS for trunk/extremity STS)
AJCC_SURVIVAL_PROBS = {
    "IA": 0.90,    # Stage I overall ~90%
    "IB": 0.90,
    "II": 0.70,    # Stage II ~70%
    "IIIA": 0.62,  # Stage IIIA ~62%
    "IIIB": 0.50,  # Stage IIIB ~50%
    "IV": 0.15     # Stage IV ~10-20%
}

def calculate_ajcc_stage(row: pd.Series) -> str:
    """Calculate AJCC 8th edition stage for trunk/extremity soft tissue sarcoma.
    
    Based on TNM+Grade:
    - T (size): T1≤5cm, T2>5-10cm, T3>10-15cm, T4>15cm
    - N (nodes): Assumed N0 (not available in dataset)
    - M (metastasis): From metastasis_present_at_diagnosis
    - G (grade): From biopsy_grading (G1/G2/G3)
    
    Returns AJCC stage (IA, IB, II, IIIA, IIIB, IV) or 'Unknown'
    """
    # Get tumor size (continuous in cm)
    size = row.get('tumor_characteristics.initial_size', None)
    
    # Get grade (categorical: 1=G1, 2=G2, 3=G3)
    grade = row.get('tumor_characteristics.biopsy_grading', None)
    
    # Get metastasis (binary: 0=no, 1=yes)
    metastasis = row.get('tumor_characteristics.metastasis_present_at_diagnosis', 0)
    
    # Check if we have required data
    if pd.isna(size) or pd.isna(grade):
        return 'Unknown'
    
    # Map size to T category (AJCC 8th edition for trunk/extremity)
    if size <= 5:
        t_cat = 1
    elif size <= 10:
        t_cat = 2
    elif size <= 15:
        t_cat = 3
    else:
        t_cat = 4
    
    # Stage IV if metastasis present (M1)
    if metastasis == 1:
        return 'IV'
    
    # Assuming N0 (no nodal involvement) since not in dataset
    # Stage grouping for trunk/extremity (M0, N0):
    
    # Grade 1 (low grade)
    if grade == 1:
        if t_cat == 1:
            return 'IA'
        else:  # T2-T4
            return 'IB'
    
    # Grade 2-3 (high grade)
    elif grade in [2, 3]:
        if t_cat == 1:
            return 'II'
        elif t_cat == 2:
            return 'IIIA'
        else:  # T3-T4
            return 'IIIB'
    
    return 'Unknown'


def get_ajcc_mortality_risk(stage: str) -> float:
    """Get 5-year mortality risk (1 - survival) from AJCC stage."""
    if stage == 'Unknown':
        return 0.5  # Default 50% if unknown
    survival = AJCC_SURVIVAL_PROBS.get(stage, 0.5)
    return 1.0 - survival  # Convert survival to mortality (DOD risk)


def calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """Calculate calibration metrics: ECE, MCE, and calibration slope/intercept.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        n_bins: Number of bins for ECE/MCE calculation
        
    Returns:
        Dictionary with calibration metrics and binned data for plotting
    """
    # Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_indices = np.digitize(y_prob, bin_edges[:-1], right=False)
    bin_indices = np.clip(bin_indices, 1, n_bins)  # Handle edge cases
    
    ece = 0.0
    mce = 0.0
    bin_data = []
    
    for i in range(1, n_bins + 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            bin_size = np.sum(mask)
            bin_error = np.abs(bin_acc - bin_conf)
            
            ece += (bin_size / len(y_true)) * bin_error
            mce = max(mce, bin_error)
            
            bin_data.append({
                'center': bin_centers[i-1],
                'accuracy': bin_acc,
                'confidence': bin_conf,
                'size': bin_size,
                'error': bin_error
            })
        else:
            bin_data.append(None)
    
    # Calibration slope and intercept using logistic regression
    # Fit: logit(y_true) ~ logit(y_prob)
    # A well-calibrated model has slope=1, intercept=0
    try:
        # Convert probabilities to logits (with clipping to avoid infinities)
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        logit_prob = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        # Fit logistic regression
        lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
        lr.fit(logit_prob.reshape(-1, 1), y_true)
        
        calib_slope = lr.coef_[0][0]
        calib_intercept = lr.intercept_[0]
    except:
        calib_slope = np.nan
        calib_intercept = np.nan
    
    return {
        'ece': ece,
        'mce': mce,
        'calibration_slope': calib_slope,
        'calibration_intercept': calib_intercept,
        'bin_data': bin_data,
        'n_bins': n_bins
    }


def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                          title: str, output_path: str, 
                          calib_metrics: dict = None) -> None:
    """Plot reliability diagram (calibration curve) with ECE/MCE bands.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        title: Plot title
        output_path: Where to save the plot
        calib_metrics: Pre-computed calibration metrics (optional)
    """
    if calib_metrics is None:
        calib_metrics = calculate_calibration_metrics(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    
    # Plot binned calibration
    bin_data = calib_metrics['bin_data']
    valid_bins = [b for b in bin_data if b is not None]
    
    if valid_bins:
        confidences = [b['confidence'] for b in valid_bins]
        accuracies = [b['accuracy'] for b in valid_bins]
        sizes = [b['size'] for b in valid_bins]
        
        # Plot points with size proportional to bin size
        max_size = max(sizes)
        point_sizes = [300 * (s / max_size) for s in sizes]
        
        scatter = ax.scatter(confidences, accuracies, s=point_sizes, 
                           alpha=0.6, c='blue', edgecolors='black',
                           label='Observed frequency')
        
        # Connect points
        ax.plot(confidences, accuracies, 'b-', alpha=0.5, linewidth=1.5)
        
        # Add ECE bars
        for b in valid_bins:
            ax.plot([b['confidence'], b['confidence']], 
                   [b['accuracy'], b['confidence']], 
                   'r-', alpha=0.3, linewidth=1)
    
    # Add metrics text
    metrics_text = (
        f"ECE: {calib_metrics['ece']:.4f}\n"
        f"MCE: {calib_metrics['mce']:.4f}\n"
        f"Calib. Slope: {calib_metrics['calibration_slope']:.3f}\n"
        f"Calib. Intercept: {calib_metrics['calibration_intercept']:.3f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.8), fontsize=10)
    
    ax.set_xlabel('Predicted Probability', fontsize=12)
    ax.set_ylabel('Observed Frequency', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def calculate_probability_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Calculate probability-aware metrics: log loss, Brier score, AUROC, AUPRC.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_prob: Predicted probabilities
        
    Returns:
        Dictionary with all probability metrics
    """
    metrics = {}
    
    try:
        # Clip probabilities to avoid log(0)
        y_prob_clipped = np.clip(y_prob, 1e-7, 1 - 1e-7)
        
        # Log loss (cross-entropy)
        metrics['log_loss'] = log_loss(y_true, y_prob_clipped)
        
        # Brier score (mean squared error of probabilities)
        metrics['brier_score'] = brier_score_loss(y_true, y_prob)
        
        # AUROC (discrimination, threshold-free)
        if len(np.unique(y_true)) > 1:  # Need both classes
            metrics['auroc'] = roc_auc_score(y_true, y_prob)
            metrics['auprc'] = average_precision_score(y_true, y_prob)
        else:
            metrics['auroc'] = np.nan
            metrics['auprc'] = np.nan
            
    except Exception as e:
        print(f"Warning: Could not calculate some probability metrics: {e}")
        metrics['log_loss'] = np.nan
        metrics['brier_score'] = np.nan
        metrics['auroc'] = np.nan
        metrics['auprc'] = np.nan
    
    return metrics


def get_markov_blanket(model, node: str) -> dict:
    """Get the Markov blanket of a node in the BN.
    
    The Markov blanket consists of:
    - Parents of the node
    - Children of the node  
    - Other parents of the children (co-parents)
    
    Args:
        model: Bayesian Network model
        node: Node name
        
    Returns:
        Dictionary with parents, children, and co-parents
    """
    parents = set(model.get_parents(node))
    children = set(model.get_children(node))
    
    # Get co-parents (other parents of children)
    coparents = set()
    for child in children:
        child_parents = set(model.get_parents(child))
        coparents.update(child_parents - {node})
    
    markov_blanket = parents | children | coparents
    
    return {
        'parents': sorted(list(parents)),
        'children': sorted(list(children)),
        'coparents': sorted(list(coparents)),
        'markov_blanket': sorted(list(markov_blanket))
    }


def sensitivity_analysis(model, outcome: str, variable: str, 
                        baseline_evidence: dict, 
                        variable_values: list = None) -> list[dict]:
    """Perform sensitivity analysis: vary one variable while holding others fixed.
    
    Args:
        model: Bayesian Network model
        outcome: Outcome variable to query
        variable: Variable to vary
        baseline_evidence: Fixed evidence for other variables
        variable_values: List of values to test (if None, use all states)
        
    Returns:
        List of dicts with {value, risk} for each tested value
    """
    inference = VariableElimination(model)
    results = []
    
    # Get variable states if not provided
    if variable_values is None:
        cpd = model.get_cpds(variable)
        variable_values = cpd.state_names.get(variable, list(range(len(cpd.get_values()))))
    
    # Get outcome state for positive outcome (typically 1)
    outcome_cpd = model.get_cpds(outcome)
    outcome_states = outcome_cpd.state_names.get(outcome, list(range(len(outcome_cpd.get_values()))))
    
    for value in variable_values:
        # Create evidence with varied variable
        evidence = baseline_evidence.copy()
        evidence[variable] = value
        
        try:
            result = inference.query(variables=[outcome], evidence=evidence)
            
            # Get probability of positive outcome
            if len(outcome_states) == 2:
                try:
                    idx_one = outcome_states.index(1)
                except:
                    idx_one = 1
                risk = float(result.values[idx_one])
            else:
                # For multi-class, return full distribution
                risk = {str(state): float(prob) 
                       for state, prob in zip(outcome_states, result.values)}
            
            results.append({
                'value': value,
                'risk': risk
            })
        except Exception as e:
            print(f"Warning: Could not query {variable}={value}: {e}")
            results.append({
                'value': value,
                'risk': np.nan
            })
    
    return results


def print_markov_blanket_analysis(model, outcomes: list[str]) -> None:
    """Print Markov blanket analysis for outcome variables.
    
    Args:
        model: Bayesian Network model
        outcomes: List of outcome variable names
    """
    print(f"\\n{'='*80}")
    print(f"MARKOV BLANKET ANALYSIS (Model Interpretability)")
    print(f"{'='*80}\\n")
    
    for outcome in outcomes:
        if outcome not in model.nodes():
            continue
            
        print(f"\\n{'-'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'-'*80}")
        
        mb = get_markov_blanket(model, outcome)
        
        print(f"\\nParents (direct causes/predictors):")
        if mb['parents']:
            for parent in mb['parents']:
                print(f"  - {parent}")
        else:
            print(f"  (none)")
        
        print(f"\\nChildren (direct effects):")
        if mb['children']:
            for child in mb['children']:
                print(f"  - {child}")
        else:
            print(f"  (none)")
        
        print(f"\\nCo-parents (other influences on children):")
        if mb['coparents']:
            for coparent in mb['coparents']:
                print(f"  - {coparent}")
        else:
            print(f"  (none)")
        
        print(f"\\nComplete Markov Blanket (variables that matter for {outcome}):")
        print(f"  {', '.join(mb['markov_blanket']) if mb['markov_blanket'] else '(none)'}")
        print(f"  Size: {len(mb['markov_blanket'])} variables")
    
    print(f"\\n{'='*80}\\n")


def print_sensitivity_analysis(model, outcomes: list[str], 
                               key_variables: list[str],
                               baseline_evidence: dict = None) -> None:
    """Print sensitivity analysis for key variables.
    
    Args:
        model: Bayesian Network model
        outcomes: List of outcome variables
        key_variables: List of variables to analyze
        baseline_evidence: Baseline evidence dict (optional)
    """
    if baseline_evidence is None:
        baseline_evidence = {}
    
    print(f"\\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS (What-If Scenarios)")
    print(f"{'='*80}")
    
    if baseline_evidence:
        print(f"\\nBaseline Evidence: {baseline_evidence}\\n")
    else:
        print(f"\\nNo baseline evidence (marginal analysis)\\n")
    
    for outcome in outcomes:
        if outcome not in model.nodes():
            continue
        
        print(f"\\n{'-'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'-'*80}")
        
        for variable in key_variables:
            if variable not in model.nodes() or variable == outcome:
                continue
            
            print(f"\\n  Varying {variable}:")
            
            results = sensitivity_analysis(model, outcome, variable, baseline_evidence)
            
            # Print results
            for r in results:
                if isinstance(r['risk'], dict):
                    # Multi-class outcome
                    risk_str = ', '.join([f"{k}={v:.3f}" for k, v in r['risk'].items()])
                    print(f"    {variable}={r['value']:>5} → P({outcome}): {risk_str}")
                else:
                    # Binary outcome
                    print(f"    {variable}={r['value']:>5} → P({outcome}=1)={r['risk']:.4f}")
            
            # Calculate risk range
            if results and not isinstance(results[0]['risk'], dict):
                risks = [r['risk'] for r in results if not np.isnan(r['risk'])]
                if risks:
                    risk_range = max(risks) - min(risks)
                    print(f"    Risk range: {risk_range:.4f} (from {min(risks):.4f} to {max(risks):.4f})")
    
    print(f"\\n{'='*80}\\n")


# =============================================================================
# INVERSE PROBABILITY OF TREATMENT WEIGHTING (IPTW)
# =============================================================================
# IPTW addresses confounding by indication - when treatments are assigned based
# on patient characteristics. This is critical for observational sarcoma data where:
# - Sicker patients may receive more aggressive treatment (chemotherapy, RT)
# - Treatment assignment depends on tumor grade, size, metastasis status
#
# IPTW reweights observations by 1/P(T|X) to create a pseudo-population where
# treatment assignment is independent of confounders, enabling causal inference.
#
# The Bayesian Network provides P(T|X) naturally through its conditional
# probability distributions, making it ideal for propensity score estimation.
# =============================================================================

# Post-treatment variables that should NEVER be included in propensity models
# These occur after treatment and violate temporal ordering
POST_TREATMENT_VARIABLES = [
    'treatments.pathologist_margin_judgement',  # Happens after surgery
    'local_recurrence',  # Outcome, not confounder
    'metastasis',  # Can be outcome (distinct from baseline metastasis_present_at_diagnosis)
    'DOD', 'AWD', 'NED',  # Outcomes
    'last_status',  # Outcome
]


def _filter_confounders(confounders: list[str], treatment_vars: list[str] | None = None) -> list[str]:
    """Filter confounders to exclude treatment vars and known post-treatment vars."""
    treatment_vars = treatment_vars or []

    filtered: list[str] = []
    for conf in confounders:
        # Exclude treatment variables themselves
        if conf in treatment_vars:
            continue

        # Exclude post-treatment variables (exact match or flattened suffix match)
        is_post_treatment = False
        for post_var in POST_TREATMENT_VARIABLES:
            if conf == post_var or conf.endswith(f".{post_var}"):
                is_post_treatment = True
                break

        if not is_post_treatment:
            filtered.append(conf)

    return filtered

def _build_balance_covariates(
    model,
    data: pd.DataFrame,
    treatment_vars: list[str],
    outcome: str | None,
    base_confounders: list[str],
) -> list[str]:
    """Build a broader baseline covariate list for balance diagnostics."""
    model_nodes = set(model.nodes())
    candidate_cols = [c for c in data.columns if c in model_nodes]
    exclude_vars = list(treatment_vars)
    if outcome is not None:
        exclude_vars.append(outcome)

    baseline_candidates = _filter_confounders(candidate_cols, exclude_vars)
    return sorted(set(base_confounders).union(baseline_candidates))


def _effective_sample_size(weights: np.ndarray) -> float:
    """Compute effective sample size for weighted data."""
    sum_w = float(np.sum(weights))
    sum_w2 = float(np.sum(weights**2))
    if sum_w2 <= 0:
        return 0.0
    return (sum_w**2) / sum_w2


def _get_treated_state_index(state_names: list[Any]) -> int:
    """Best-effort index for the treated state in a binary treatment variable."""
    if not state_names:
        return 1

    # Prefer explicit "treated-like" encodings first.
    for candidate in (1, "1", True, "true", "True", "yes", "Yes", "treated", "Treated"):
        try:
            return state_names.index(candidate)
        except ValueError:
            continue

    # If we can find an explicit control encoding, treated is the other state.
    if len(state_names) == 2:
        for control_candidate in (0, "0", False, "false", "False", "no", "No", "control", "Control"):
            if control_candidate in state_names:
                return 1 - state_names.index(control_candidate)
        return 1

    # Degenerate or multi-state fallback.
    return 0 if len(state_names) == 1 else 1


def _bootstrap_weighted_ate(
    treatment_vals: np.ndarray,
    outcome_vals: np.ndarray,
    weights: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float | None]:
    """Bootstrap SE/CI for weighted ATE using row-wise resampling."""
    n = len(treatment_vals)
    if n == 0:
        return np.nan, np.nan, np.nan, None

    rng = np.random.default_rng(seed)
    ates = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t = treatment_vals[idx]
        y = outcome_vals[idx]
        w = weights[idx]

        treated_mask = t == 1
        control_mask = t == 0
        if not np.any(treated_mask) or not np.any(control_mask):
            continue

        try:
            mean_treated = np.average(y[treated_mask], weights=w[treated_mask])
            mean_control = np.average(y[control_mask], weights=w[control_mask])
            ates.append(float(mean_treated - mean_control))
        except Exception:
            continue

    if len(ates) < 2:
        return np.nan, np.nan, np.nan, None

    ate_samples = np.array(ates)
    se = float(np.std(ate_samples, ddof=1))
    ci_lower = float(np.quantile(ate_samples, 0.025))
    ci_upper = float(np.quantile(ate_samples, 0.975))

    # Two-sided p-value from empirical distribution around 0.
    p_left = float(np.mean(ate_samples <= 0))
    p_right = float(np.mean(ate_samples >= 0))
    p_value = float(min(1.0, 2 * min(p_left, p_right)))

    return se, ci_lower, ci_upper, p_value


def _bootstrap_iptw_ate(
    model,
    treatment: str,
    outcome: str,
    data: pd.DataFrame,
    confounders: list[str],
    stabilized: bool,
    truncate: float | None,
    min_ps: float | None,
    hard_evidence: dict | None,
    virtual_evidence: list[TabularCPD] | None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float | None]:
    """Bootstrap SE/CI for binary-treatment IPTW, recomputing PS and weights each replicate."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan, None

    rng = np.random.default_rng(seed)
    ates: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_data = data.iloc[idx].reset_index(drop=True)
        t = bs_data[treatment].values
        y = bs_data[outcome].values

        treated_mask = t == 1
        control_mask = t == 0
        if not np.any(treated_mask) or not np.any(control_mask):
            continue

        ps = calculate_propensity_scores(
            model,
            treatment,
            bs_data,
            confounders=confounders,
            hard_evidence=hard_evidence,
            virtual_evidence=virtual_evidence,
        )
        bs_data, ps = _apply_min_ps_filter(bs_data, ps, min_ps, label=f"{treatment} bootstrap")
        if len(bs_data) == 0:
            continue
        t = bs_data[treatment].values
        y = bs_data[outcome].values
        treated_mask = t == 1
        control_mask = t == 0
        if not np.any(treated_mask) or not np.any(control_mask):
            continue
        w = calculate_iptw_weights(t, ps, stabilized=stabilized, truncate=truncate)

        if float(np.sum(w[treated_mask])) <= 0 or float(np.sum(w[control_mask])) <= 0:
            continue

        mean_treated = np.average(y[treated_mask], weights=w[treated_mask])
        mean_control = np.average(y[control_mask], weights=w[control_mask])
        ates.append(float(mean_treated - mean_control))

    if len(ates) < 2:
        return np.nan, np.nan, np.nan, None

    ate_samples = np.array(ates)
    se = float(np.std(ate_samples, ddof=1))
    ci_lower = float(np.quantile(ate_samples, 0.025))
    ci_upper = float(np.quantile(ate_samples, 0.975))
    p_left = float(np.mean(ate_samples <= 0))
    p_right = float(np.mean(ate_samples >= 0))
    p_value = float(min(1.0, 2 * min(p_left, p_right)))
    return se, ci_lower, ci_upper, p_value


def _bootstrap_regime_iptw_ate(
    model,
    regime_vars: list[str],
    regime_a: dict,
    regime_b: dict,
    outcome: str,
    data: pd.DataFrame,
    stabilized: bool,
    truncate: float | None,
    min_ps: float | None,
    max_weight: float | None,
    hard_evidence: dict | None,
    virtual_evidence: list[TabularCPD] | None,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[float, float, float, float | None]:
    """Bootstrap SE/CI for regime IPTW, recomputing regime PS and weights each replicate."""
    n = len(data)
    if n == 0:
        return np.nan, np.nan, np.nan, None

    rng = np.random.default_rng(seed)
    ates: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        bs_data = data.iloc[idx].reset_index(drop=True)
        y = bs_data[outcome].values

        ps = calculate_regime_propensity_scores(
            model,
            regime_vars,
            bs_data,
            hard_evidence=hard_evidence,
            virtual_evidence=virtual_evidence,
        )
        bs_data, ps = _apply_min_ps_filter(bs_data, ps, min_ps, label="regime bootstrap")
        if len(bs_data) == 0:
            continue
        y = bs_data[outcome].values
        w_a, w_b = calculate_regime_iptw_weights(
            bs_data,
            regime_vars,
            regime_a,
            regime_b,
            ps,
            stabilized=stabilized,
            truncate=truncate,
            max_weight=max_weight,
        )

        mask_a = w_a > 0
        mask_b = w_b > 0
        if not np.any(mask_a) or not np.any(mask_b):
            continue
        if float(np.sum(w_a[mask_a])) <= 0 or float(np.sum(w_b[mask_b])) <= 0:
            continue

        mean_a = np.average(y[mask_a], weights=w_a[mask_a])
        mean_b = np.average(y[mask_b], weights=w_b[mask_b])
        ates.append(float(mean_b - mean_a))

    if len(ates) < 2:
        return np.nan, np.nan, np.nan, None

    ate_samples = np.array(ates)
    se = float(np.std(ate_samples, ddof=1))
    ci_lower = float(np.quantile(ate_samples, 0.025))
    ci_upper = float(np.quantile(ate_samples, 0.975))
    p_left = float(np.mean(ate_samples <= 0))
    p_right = float(np.mean(ate_samples >= 0))
    p_value = float(min(1.0, 2 * min(p_left, p_right)))
    return se, ci_lower, ci_upper, p_value


def _sanitize_propensity_scores(
    propensity_scores: np.ndarray,
    label: str,
    eps: float = 1e-8,
) -> np.ndarray:
    """Sanitize propensity scores to avoid exact 0/1, NaN, or Inf values."""
    ps = np.asarray(propensity_scores, dtype=float).copy()
    invalid_mask = ~np.isfinite(ps)
    n_invalid = int(np.sum(invalid_mask))
    if n_invalid > 0:
        ps[invalid_mask] = 0.5
        print(f"Warning: {label}: replaced {n_invalid} non-finite propensity scores with 0.5")

    n_zero = int(np.sum(ps <= 0))
    n_one = int(np.sum(ps >= 1))
    if n_zero > 0 or n_one > 0:
        print(
            f"Warning: {label}: detected {n_zero} scores <= 0 and {n_one} scores >= 1 "
            f"(positivity issue); clipping to [{eps:g}, {1 - eps:g}]"
        )

    return np.clip(ps, eps, 1 - eps)


def _apply_min_ps_filter(
    data: pd.DataFrame,
    propensity_scores: np.ndarray,
    min_ps: float | None,
    label: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop rows with propensity score below overlap threshold."""
    if min_ps is None or min_ps <= 0:
        return data.reset_index(drop=True), np.asarray(propensity_scores, dtype=float)

    ps = np.asarray(propensity_scores, dtype=float)
    keep_mask = ps >= float(min_ps)
    dropped = int(np.sum(~keep_mask))
    if dropped > 0:
        print(
            f"Overlap filter ({label}): dropped {dropped}/{len(ps)} rows with PS < {min_ps:g}"
        )
    filtered_data = data.loc[keep_mask].reset_index(drop=True)
    filtered_ps = ps[keep_mask]
    return filtered_data, filtered_ps


def calculate_propensity_scores(model, treatment: str, data: pd.DataFrame, 
                                confounders: list[str] = None,
                                hard_evidence: dict | None = None,
                                virtual_evidence: list[TabularCPD] | None = None) -> np.ndarray:
    """Calculate propensity scores P(T=1|X) from the Bayesian Network.
    
    Args:
        model: Fitted Bayesian Network
        treatment: Treatment variable name
        data: DataFrame with patient data
        confounders: List of confounder variables (if None, use treatment's parents)
        
    Returns:
        Array of propensity scores (probability of receiving treatment)
    """
    inference = VariableElimination(model)
    
    # If confounders not specified, use all available baseline covariates.
    if confounders is None:
        confounders = [c for c in data.columns if c in model.nodes()]
        confounders = _filter_confounders(confounders, [treatment])
    
    print(f"\n[Binary PS for {treatment}] Using confounders: {confounders[:10]}{'...' if len(confounders) > 10 else ''}")
    
    propensity_scores = []
    
    hard_evidence = hard_evidence or {}

    # Cache CPD state names for type alignment
    state_names: dict[str, list] = {}
    for var in confounders + [treatment]:
        if var in model.nodes():
            cpd = model.get_cpds(var)
            if cpd is not None and hasattr(cpd, 'state_names') and var in cpd.state_names:
                state_names[var] = list(cpd.state_names[var])

    for idx, row in data.iterrows():
        # Build evidence from confounders
        evidence = hard_evidence.copy()
        for conf in confounders:
            if conf in evidence:
                continue
            if conf not in row.index or conf not in model.nodes():
                continue
                
            value = row[conf]

            # Align type to CPD state names if present
            if conf in state_names:
                states = state_names[conf]
                
                # Direct match
                if value in states:
                    evidence[conf] = value
                elif str(value) in states:
                    evidence[conf] = str(value)
                elif isinstance(value, (int, float)) and not np.isnan(value):
                    # For binned numerical variables: try type conversions
                    matched = False
                    for candidate in [int(value), float(value), str(int(value))]:
                        if candidate in states:
                            evidence[conf] = candidate
                            matched = True
                            break
                    if not matched:
                        # Last resort: if states are numeric, find closest
                        if all(isinstance(s, (int, float)) or (isinstance(s, str) and s.replace('.','').replace('-','').isdigit()) for s in states):
                            numeric_states = [float(s) if isinstance(s, str) else s for s in states]
                            closest_state = states[np.argmin([abs(float(ns) - float(value)) for ns in numeric_states])]
                            evidence[conf] = closest_state
                        else:
                            # Cannot match - skip this confounder
                            continue
                else:
                    # Non-numeric, cannot match
                    continue
            else:
                # No state names: best-effort cast
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
                evidence[conf] = value
        
        try:
            # Query P(T|confounders)
            result = inference.query(
                variables=[treatment],
                evidence=evidence or None,
                virtual_evidence=virtual_evidence,
                show_progress=False,
            )
            state_names = result.state_names.get(treatment, list(range(result.cardinality[0])))

            # Robustly identify treated state index for binary encoding variants.
            idx_one = _get_treated_state_index(list(state_names))
            ps = float(result.values[idx_one])
            propensity_scores.append(ps)
        except Exception as e:
            # If query fails, use marginal treatment probability
            print(f"Warning: Could not calculate propensity score for row {idx}, using marginal: {e}")
            try:
                result = inference.query(
                    variables=[treatment],
                    evidence=hard_evidence or None,
                    virtual_evidence=virtual_evidence,
                    show_progress=False,
                )
                state_names = result.state_names.get(treatment, list(range(result.cardinality[0])))
                idx_one = _get_treated_state_index(list(state_names))
                ps = float(result.values[idx_one])
                propensity_scores.append(ps)
            except:
                propensity_scores.append(0.5)  # Default fallback
    
    return _sanitize_propensity_scores(
        np.array(propensity_scores),
        label=f"binary PS ({treatment})",
        eps=1e-8,
    )


def calculate_iptw_weights(treatment: np.ndarray, propensity_scores: np.ndarray,
                          stabilized: bool = True, truncate: float = None) -> np.ndarray:
    """Calculate Inverse Probability of Treatment Weighting (IPTW) weights.
    
    Args:
        treatment: Binary treatment indicator (0 or 1)
        propensity_scores: Propensity scores P(T=1|X)
        stabilized: If True, use stabilized weights (recommended)
        truncate: If provided, truncate propensity scores to [truncate, 1-truncate]
        
    Returns:
        Array of IPTW weights
    """
    ps = propensity_scores.copy()
    
    # Truncate propensity scores to avoid extreme weights
    if truncate is not None:
        ps = np.clip(ps, truncate, 1 - truncate)
    else:
        # Default: clip to avoid division by very small numbers
        ps = np.clip(ps, 0.01, 0.99)
    
    # Calculate weights
    if stabilized:
        # Stabilized weights: P(T) / P(T|X)
        p_treatment = np.mean(treatment)
        weights = np.where(treatment == 1, 
                          p_treatment / ps, 
                          (1 - p_treatment) / (1 - ps))
    else:
        # Unstabilized weights: 1 / P(T|X)
        weights = np.where(treatment == 1, 
                          1 / ps, 
                          1 / (1 - ps))
    
    return weights


def check_covariate_balance(data: pd.DataFrame, treatment: str, 
                           confounders: list[str], weights: np.ndarray = None) -> dict:
    """Check covariate balance before and after IPTW weighting.
    
    Args:
        data: DataFrame with patient data
        treatment: Treatment variable name
        confounders: List of confounder variables
        weights: IPTW weights (if None, checks unweighted balance)
        
    Returns:
        Dictionary with standardized mean differences for each confounder
    """
    treatment_vals = data[treatment].values
    balance = {}
    
    for conf in confounders:
        if conf not in data.columns:
            balance[conf] = {
                'mean_treated': np.nan,
                'mean_control': np.nan,
                'smd': np.nan,
                'ess_treated': np.nan,
                'ess_control': np.nan,
                'reason': 'missing_in_data',
            }
            continue
        conf_series = data[conf]
        valid_mask = conf_series.notna().values
        treated_mask = (treatment_vals == 1) & valid_mask
        control_mask = (treatment_vals == 0) & valid_mask

        if not np.any(treated_mask) or not np.any(control_mask):
            balance[conf] = {
                'mean_treated': np.nan,
                'mean_control': np.nan,
                'smd': np.nan,
                'ess_treated': _effective_sample_size(weights[treated_mask]) if weights is not None and np.any(treated_mask) else float(np.sum(treated_mask)),
                'ess_control': _effective_sample_size(weights[control_mask]) if weights is not None and np.any(control_mask) else float(np.sum(control_mask)),
                'reason': 'insufficient_group_data',
            }
            continue

        if weights is not None and (float(np.sum(weights[treated_mask])) <= 0 or float(np.sum(weights[control_mask])) <= 0):
            balance[conf] = {
                'mean_treated': np.nan,
                'mean_control': np.nan,
                'smd': np.nan,
                'ess_treated': _effective_sample_size(weights[treated_mask]),
                'ess_control': _effective_sample_size(weights[control_mask]),
                'reason': 'zero_weight_group',
            }
            continue

        # Treat int-coded/bool/object/category low-cardinality variables as categorical.
        nunique = int(conf_series[valid_mask].nunique())
        is_categorical = (
            pd.api.types.is_object_dtype(conf_series)
            or pd.api.types.is_categorical_dtype(conf_series)
            or pd.api.types.is_bool_dtype(conf_series)
            or (pd.api.types.is_integer_dtype(conf_series) and nunique <= 10)
        )

        if is_categorical:
            levels = list(pd.Series(conf_series[valid_mask]).unique())
            if len(levels) == 0:
                balance[conf] = {
                    'mean_treated': np.nan,
                    'mean_control': np.nan,
                    'smd': np.nan,
                    'ess_treated': _effective_sample_size(weights[treated_mask]) if weights is not None else float(np.sum(treated_mask)),
                    'ess_control': _effective_sample_size(weights[control_mask]) if weights is not None else float(np.sum(control_mask)),
                    'reason': 'insufficient_group_data',
                }
                continue

            best_level = None
            best_abs_smd = -1.0
            best_treated = np.nan
            best_control = np.nan
            for level in levels:
                indicator = (conf_series.values == level).astype(float)
                if weights is None:
                    p_t = float(np.mean(indicator[treated_mask]))
                    p_c = float(np.mean(indicator[control_mask]))
                    var_t = p_t * (1 - p_t)
                    var_c = p_c * (1 - p_c)
                    ess_treated = float(np.sum(treated_mask))
                    ess_control = float(np.sum(control_mask))
                else:
                    p_t = float(np.average(indicator[treated_mask], weights=weights[treated_mask]))
                    p_c = float(np.average(indicator[control_mask], weights=weights[control_mask]))
                    var_t = float(np.average((indicator[treated_mask] - p_t) ** 2, weights=weights[treated_mask]))
                    var_c = float(np.average((indicator[control_mask] - p_c) ** 2, weights=weights[control_mask]))
                    ess_treated = _effective_sample_size(weights[treated_mask])
                    ess_control = _effective_sample_size(weights[control_mask])

                pooled_std = np.sqrt((var_t + var_c) / 2)
                smd_level = (p_t - p_c) / pooled_std if pooled_std > 0 else 0.0
                if abs(smd_level) > best_abs_smd:
                    best_abs_smd = abs(smd_level)
                    best_level = level
                    best_treated = p_t
                    best_control = p_c
                    best_smd = smd_level

            balance[conf] = {
                'mean_treated': best_treated,
                'mean_control': best_control,
                'smd': best_smd,
                'ess_treated': ess_treated,
                'ess_control': ess_control,
                'reason': None,
                'type': 'categorical',
                'smd_level': best_level,
                'n_levels': len(levels),
            }
            continue

        # Continuous handling
        try:
            conf_vals = conf_series.astype(float).values
        except Exception:
            balance[conf] = {
                'mean_treated': np.nan,
                'mean_control': np.nan,
                'smd': np.nan,
                'ess_treated': np.nan,
                'ess_control': np.nan,
                'reason': 'non_numeric',
            }
            continue

        if weights is None:
            mean_treated = np.mean(conf_vals[treated_mask])
            mean_control = np.mean(conf_vals[control_mask])
            std_treated = np.std(conf_vals[treated_mask])
            std_control = np.std(conf_vals[control_mask])
            ess_treated = float(np.sum(treated_mask))
            ess_control = float(np.sum(control_mask))
        else:
            mean_treated = np.average(conf_vals[treated_mask], weights=weights[treated_mask])
            mean_control = np.average(conf_vals[control_mask], weights=weights[control_mask])
            var_treated = np.average((conf_vals[treated_mask] - mean_treated) ** 2, weights=weights[treated_mask])
            var_control = np.average((conf_vals[control_mask] - mean_control) ** 2, weights=weights[control_mask])
            std_treated = np.sqrt(var_treated)
            std_control = np.sqrt(var_control)
            ess_treated = _effective_sample_size(weights[treated_mask])
            ess_control = _effective_sample_size(weights[control_mask])

        pooled_std = np.sqrt((std_treated**2 + std_control**2) / 2)
        smd = (mean_treated - mean_control) / pooled_std if pooled_std > 0 else 0
        balance[conf] = {
            'mean_treated': mean_treated,
            'mean_control': mean_control,
            'smd': smd,
            'ess_treated': ess_treated,
            'ess_control': ess_control,
            'reason': None,
            'type': 'continuous',
        }
    
    return balance


def estimate_ate_with_iptw(model, treatment: str, outcome: str,
                           data: pd.DataFrame, confounders: list[str] = None,
                           stabilized: bool = True, truncate: float | None = 0.05,
                           min_ps: float | None = 0.0,
                           n_bootstrap: int = 500,
                           hard_evidence: dict | None = None,
                           virtual_evidence: list[TabularCPD] | None = None) -> dict:
    """Estimate Average Treatment Effect (ATE) using IPTW.
    
    This addresses confounding by indication by weighting observations to balance
    baseline covariates across treatment groups.
    
    Args:
        model: Fitted Bayesian Network
        treatment: Treatment variable name
        outcome: Outcome variable name  
        data: DataFrame with patient data
        confounders: List of confounder variables (if None, use all baseline covariates)
        stabilized: Use stabilized weights (recommended)
        truncate: Truncate propensity scores to [truncate, 1-truncate]
        
    Returns:
        Dictionary with ATE, propensity scores, weights, and balance diagnostics
    """
    if confounders is None:
        confounders = [c for c in data.columns if c in model.nodes()]
        confounders = _filter_confounders(confounders, [treatment])

    # Calculate propensity scores
    propensity_scores = calculate_propensity_scores(
        model,
        treatment,
        data,
        confounders,
        hard_evidence=hard_evidence,
        virtual_evidence=virtual_evidence,
    )
    data, propensity_scores = _apply_min_ps_filter(
        data, propensity_scores, min_ps, label=f"{treatment} analysis"
    )
    if len(data) == 0:
        raise ValueError(f"No samples remain after min-ps filtering for treatment '{treatment}'.")
    
    # Get treatment and outcome values
    treatment_vals = data[treatment].values
    outcome_vals = data[outcome].values
    
    # Calculate IPTW weights
    weights = calculate_iptw_weights(treatment_vals, propensity_scores, 
                                     stabilized=stabilized, truncate=truncate)
    
    # For diagnostics, report balance on all baseline covariates available.
    balance_covariates = _build_balance_covariates(
        model=model,
        data=data,
        treatment_vars=[treatment],
        outcome=outcome,
        base_confounders=confounders,
    )
    
    balance_before = check_covariate_balance(data, treatment, balance_covariates, weights=None)
    balance_after = check_covariate_balance(data, treatment, balance_covariates, weights=weights)
    
    # Calculate weighted outcome means
    treated_mask = treatment_vals == 1
    control_mask = treatment_vals == 0
    if not np.any(treated_mask) or not np.any(control_mask):
        raise ValueError(
            f"Insufficient overlap after min-ps filtering for treatment '{treatment}': "
            f"n_treated={int(np.sum(treated_mask))}, n_control={int(np.sum(control_mask))}"
        )
    
    mean_outcome_treated = np.average(outcome_vals[treated_mask], 
                                     weights=weights[treated_mask])
    mean_outcome_control = np.average(outcome_vals[control_mask], 
                                     weights=weights[control_mask])
    
    ate = mean_outcome_treated - mean_outcome_control
    
    # Bootstrap SE/CI for IPTW ATE, recomputing PS and weights each replicate.
    se_ate, ci_lower, ci_upper, p_value = _bootstrap_iptw_ate(
        model=model,
        treatment=treatment,
        outcome=outcome,
        data=data,
        confounders=confounders,
        stabilized=stabilized,
        truncate=truncate,
        min_ps=min_ps,
        hard_evidence=hard_evidence,
        virtual_evidence=virtual_evidence,
        n_bootstrap=n_bootstrap,
    )
    
    return {
        'ate': ate,
        'se': se_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'mean_outcome_treated': mean_outcome_treated,
        'mean_outcome_control': mean_outcome_control,
        'propensity_scores': propensity_scores,
        'weights': weights,
        'weight_summary': {
            'mean': np.mean(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'effective_sample_size': _effective_sample_size(weights)
        },
        'balance_before': balance_before,
        'balance_after': balance_after,
        'confounders': balance_covariates,
        'ps_confounders': confounders,
    }


def print_iptw_diagnostics(iptw_result: dict) -> None:
    """Print IPTW diagnostics including balance checks and weight distribution.
    
    Args:
        iptw_result: Result dictionary from estimate_ate_with_iptw
    """
    print(f"\n{'─'*80}")
    print(f"IPTW DIAGNOSTICS")
    print(f"{'─'*80}")
    
    # Weight distribution
    ws = iptw_result['weight_summary']
    print(f"\nWeight Distribution:")
    print(f"  Mean:                {ws['mean']:.3f}")
    print(f"  Range:               [{ws['min']:.3f}, {ws['max']:.3f}]")
    print(f"  Effective Sample:    {ws['effective_sample_size']:.1f}")
    
    # Covariate balance
    print(f"\nCovariate Balance (Standardized Mean Difference):")
    print(f"  {'Variable':<35} {'Before':>10} {'After':>10} {'ESS T/C':>16} {'Status':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*16} {'-'*10}")
    
    for conf in iptw_result['confounders']:
        before = iptw_result['balance_before'].get(conf, {})
        after = iptw_result['balance_after'].get(conf, {})

        smd_before = before.get('smd', np.nan)
        smd_after = after.get('smd', np.nan)
        ess_t = after.get('ess_treated', np.nan)
        ess_c = after.get('ess_control', np.nan)
        reason = after.get('reason') or before.get('reason')

        before_str = f"{smd_before:.3f}" if np.isfinite(smd_before) else "n/a"
        after_str = f"{smd_after:.3f}" if np.isfinite(smd_after) else "n/a"
        ess_str = f"{ess_t:.1f}/{ess_c:.1f}" if np.isfinite(ess_t) and np.isfinite(ess_c) else "n/a"

        if np.isfinite(smd_after):
            status = "✓" if abs(smd_after) < 0.1 else "~" if abs(smd_after) < 0.2 else "✗"
        else:
            status = "?"

        print(f"  {conf:<35} {before_str:>10} {after_str:>10} {ess_str:>16} {status:>10}")
        if reason:
            print(f"    reason: {reason}")
    
    print(f"\n  Legend: ✓ = balanced (|SMD| < 0.1), ~ = acceptable (|SMD| < 0.2), ✗ = imbalanced")


def print_regime_iptw_diagnostics(iptw_result: dict, regime_a: dict, regime_b: dict) -> None:
    """Print IPTW diagnostics for regime comparisons.
    
    Args:
        iptw_result: Result dictionary from estimate_regime_ate_with_iptw
        regime_a: Regime A specification
        regime_b: Regime B specification
    """
    print(f"\n{'─'*80}")
    print(f"REGIME IPTW DIAGNOSTICS")
    print(f"{'─'*80}")
    
    # Regime definitions
    print(f"\nRegime A: {regime_a}")
    print(f"Regime B: {regime_b}")
    
    # Weight distribution for both regimes
    print(f"\nWeight Distribution:")
    ws_a = iptw_result['weight_summary_a']
    ws_b = iptw_result['weight_summary_b']
    
    print(f"\n  Regime A (n={iptw_result['n_a']}):")
    print(f"    Mean:                {ws_a['mean']:.3f}")
    print(f"    Range:               [{ws_a['min']:.3f}, {ws_a['max']:.3f}]")
    print(f"    Effective Sample:    {ws_a['ess']:.1f}")
    
    print(f"\n  Regime B (n={iptw_result['n_b']}):")
    print(f"    Mean:                {ws_b['mean']:.3f}")
    print(f"    Range:               [{ws_b['min']:.3f}, {ws_b['max']:.3f}]")
    print(f"    Effective Sample:    {ws_b['ess']:.1f}")
    
    # Covariate balance
    if 'balance_before' in iptw_result and 'balance_after' in iptw_result:
        print(f"\nCovariate Balance (Standardized Mean Difference):")
        print(f"  {'Variable':<35} {'Before':>10} {'After':>10} {'ESS T/C':>16} {'Status':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*16} {'-'*10}")
        
        confounders = iptw_result.get('confounders', [])
        for conf in confounders:
            before = iptw_result['balance_before'].get(conf, {})
            after = iptw_result['balance_after'].get(conf, {})
            smd_before = before.get('smd', np.nan)
            smd_after = after.get('smd', np.nan)
            ess_t = after.get('ess_treated', np.nan)
            ess_c = after.get('ess_control', np.nan)
            reason = after.get('reason') or before.get('reason')

            before_str = f"{smd_before:.3f}" if np.isfinite(smd_before) else "n/a"
            after_str = f"{smd_after:.3f}" if np.isfinite(smd_after) else "n/a"
            ess_str = f"{ess_t:.1f}/{ess_c:.1f}" if np.isfinite(ess_t) and np.isfinite(ess_c) else "n/a"

            if np.isfinite(smd_after):
                status = "✓" if abs(smd_after) < 0.1 else "~" if abs(smd_after) < 0.2 else "✗"
            else:
                status = "?"

            print(f"  {conf:<35} {before_str:>10} {after_str:>10} {ess_str:>16} {status:>10}")
            if reason:
                print(f"    reason: {reason}")
        
        print(f"\n  Legend: ✓ = balanced (|SMD| < 0.1), ~ = acceptable (|SMD| < 0.2), ✗ = imbalanced")
    else:
        print(f"\n  [Balance diagnostics not available]")


def calculate_regime_propensity_scores(
    model: Any,
    regime_vars: list[str],
    data: pd.DataFrame,
    hard_evidence: dict | None = None,
    virtual_evidence: list[TabularCPD] | None = None,
) -> np.ndarray:
    """Generalized propensity score for observed joint regime assignment.

    For each patient i:
        ps_i = P(T = t_i | X_i, hard_evidence, virtual_evidence)

    where:
      - t_i is the observed joint assignment over regime_vars for that patient
      - X_i is the HARDCODED set of baseline confounders (tumor/patient characteristics)
    """
    inference = VariableElimination(model)
    hard_evidence = hard_evidence or {}

    # HARDCODED confounders: baseline tumor and patient characteristics
    # These are the clinically relevant variables that determine treatment assignment
    hardcoded_confounders = [
        "general.age",
        "tumor_characteristics.biopsy_grading",
        "tumor_characteristics.extremity_tumor",
        "tumor_characteristics.metastasis_present_at_diagnosis",
        "tumor_characteristics.initial_size",
    ]
    
    # Keep only confounders that exist in both the data and the model
    confounders = [c for c in hardcoded_confounders if c in data.columns and c in model.nodes()]
    
    # Filter out any post-treatment variables and treatment variables themselves
    confounders = _filter_confounders(confounders, regime_vars)
    
    # print(f"\n[Regime PS] Using hardcoded confounders: {confounders}")
    
    # Show which confounders have state names (categorical/binned) vs continuous
    categorical_confounders = []
    for c in confounders:
        cpd = model.get_cpds(c)
        if cpd and hasattr(cpd, 'state_names') and c in cpd.state_names:
            states = cpd.state_names[c]
            categorical_confounders.append(f"{c} ({len(states)} states)")
    # if categorical_confounders:
        # print(f"[Regime PS] Categorical/binned: {', '.join(categorical_confounders)}")

    # 2) Cache CPD state names for type alignment (prevents silent evidence dropping)
    state_names: dict[str, list] = {}
    for var in confounders + [v for v in regime_vars if v in model.nodes()]:
        cpd = model.get_cpds(var) if var in model.nodes() else None
        if cpd is not None and getattr(cpd, "state_names", None) and var in cpd.state_names:
            state_names[var] = list(cpd.state_names[var])

    propensity_scores: list[float] = []

    for _, row in data.iterrows():
        # Build evidence from confounders
        evidence = dict(hard_evidence)

        for conf in confounders:
            if conf in evidence:
                continue

            value = row[conf]

            # Align type to CPD state names if present
            if conf in state_names:
                states = state_names[conf]
                
                # Direct match
                if value in states:
                    evidence[conf] = value
                elif str(value) in states:
                    evidence[conf] = str(value)
                elif isinstance(value, (int, float)) and not np.isnan(value):
                    # For binned numerical variables: use the binned value directly from data
                    # The data should already have binned values matching state names
                    # If not matching, try common type conversions
                    matched = False
                    for candidate in [int(value), float(value), str(int(value))]:
                        if candidate in states:
                            evidence[conf] = candidate
                            matched = True
                            break
                    if not matched:
                        # Last resort: if states are numeric, find closest
                        if all(isinstance(s, (int, float)) or (isinstance(s, str) and s.replace('.','').replace('-','').isdigit()) for s in states):
                            numeric_states = [float(s) if isinstance(s, str) else s for s in states]
                            closest_state = states[np.argmin([abs(float(ns) - float(value)) for ns in numeric_states])]
                            evidence[conf] = closest_state
                        else:
                            # Cannot match - skip this confounder for this row
                            print(f"Warning: Cannot match {conf}={value} to states {states[:5]}{'...' if len(states) > 5 else ''}")
                            continue
                else:
                    # Non-numeric, cannot match
                    continue
            else:
                # No state names: use raw value (best-effort cast)
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
                evidence[conf] = value

        # Observed joint assignment for the treatment regime
        assignment: dict[str, Any] = {}
        query_vars: list[str] = []
        for tx in regime_vars:
            if tx not in model.nodes() or tx not in row.index:
                continue

            tx_value = row[tx]
            if tx in state_names:
                states = state_names[tx]
                if tx_value in states:
                    pass
                elif str(tx_value) in states:
                    tx_value = str(tx_value)
                else:
                    raise ValueError(f"Invalid state for {tx}: {tx_value!r}")

            assignment[tx] = tx_value
            query_vars.append(tx)

        if not query_vars:
            propensity_scores.append(0.5)
            continue

        try:
            # P(observed regime assignment | confounders, (optional) evidence)
            result = inference.query(
                variables=query_vars,
                evidence=evidence or None,
                virtual_evidence=virtual_evidence,
                show_progress=False,
            )
            ps = float(result.get_value(**assignment))
            propensity_scores.append(ps)
        except Exception:
            # Conservative fallback (unconditional on row-specific confounders)
            try:
                result = inference.query(
                    variables=query_vars,
                    evidence=hard_evidence or None,
                    virtual_evidence=virtual_evidence,
                    show_progress=False,
                )
                propensity_scores.append(float(result.get_value(**assignment)))
            except Exception:
                propensity_scores.append(0.5)

    return _sanitize_propensity_scores(
        np.array(propensity_scores, dtype=float),
        label=f"regime PS ({'+'.join(regime_vars)})",
        eps=1e-10,
    )

def calculate_regime_iptw_weights(data: pd.DataFrame, regime_vars: list[str],
                                 regime_a: dict, regime_b: dict,
                                 propensity_scores: np.ndarray,
                                 stabilized: bool = True,
                                 truncate: float | None = None,
                                 max_weight: float | None = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Calculate IPTW weights for comparing two treatment regimes.
    
    Args:
        data: DataFrame with patient data
        regime_vars: List of treatment variables
        regime_a: Dictionary defining regime A (e.g., {'surgery': 1, 'RT': 0})
        regime_b: Dictionary defining regime B (e.g., {'surgery': 1, 'RT': 1})
        propensity_scores: Propensity scores for observed regimes
        stabilized: Use stabilized weights
        truncate: Truncate propensity scores
        
    Returns:
        Tuple of (weights_a, weights_b) for each regime
    """
    ps = np.maximum(propensity_scores.astype(float), 1e-10)

    mask_a = np.ones(len(data), dtype=bool)
    mask_b = np.ones(len(data), dtype=bool)
    for var, val in regime_a.items():
        mask_a &= (data[var] == val)
    for var, val in regime_b.items():
        mask_b &= (data[var] == val)

    # Only A/B rows are relevant
    in_ab = mask_a | mask_b

    # Stabilization constants (don’t affect weighted means, but OK for variance/ESS)
    p_a = mask_a[in_ab].mean()
    p_b = mask_b[in_ab].mean()

    if stabilized:
        w_a = np.where(mask_a, p_a / ps, 0.0)
        w_b = np.where(mask_b, p_b / ps, 0.0)
    else:
        w_a = np.where(mask_a, 1.0 / ps, 0.0)
        w_b = np.where(mask_b, 1.0 / ps, 0.0)

    if max_weight is not None and max_weight > 0:
        w_a = np.where(mask_a, np.minimum(w_a, max_weight), 0.0)
        w_b = np.where(mask_b, np.minimum(w_b, max_weight), 0.0)

    return w_a, w_b


def estimate_regime_ate_with_iptw(model, regime_vars: list[str],
                                 regime_a: dict, regime_b: dict,
                                 outcome: str, data: pd.DataFrame,
                                 stabilized: bool = True,
                                 truncate: float | None = 0.005,
                                 min_ps: float | None = 0.0,
                                 max_weight: float | None = 20.0,
                                 n_bootstrap: int = 500,
                                 hard_evidence: dict | None = None,
                                 virtual_evidence: list[TabularCPD] | None = None) -> dict:
    """Estimate ATE between two treatment regimes using IPTW.
    
    Args:
        model: Fitted Bayesian Network
        regime_vars: List of treatment variables (e.g., ['surgery', 'radiotherapy', 'chemotherapy'])
        regime_a: Regime A specification (e.g., {'surgery': 1, 'radiotherapy': 0, 'chemotherapy': 0})
        regime_b: Regime B specification (e.g., {'surgery': 1, 'radiotherapy': 1, 'chemotherapy': 0})
        outcome: Outcome variable
        data: Patient data
        stabilized: Use stabilized weights
        truncate: Truncate propensity scores
        
    Returns:
        Dictionary with ATE, weights, and diagnostics including covariate balance
    """
    # Use all available baseline covariates from data/model for regime PS.
    confounders_raw = [c for c in data.columns if c in model.nodes()]
    confounders = sorted(_filter_confounders(confounders_raw, regime_vars))
    
    # Calculate propensity scores for observed regimes
    propensity_scores = calculate_regime_propensity_scores(
        model,
        regime_vars,
        data,
        hard_evidence=hard_evidence,
        virtual_evidence=virtual_evidence,
    )
    data, propensity_scores = _apply_min_ps_filter(
        data, propensity_scores, min_ps, label="regime analysis"
    )
    if len(data) == 0:
        return {
            'ate': None,
            'status': 'insufficient_data',
            'error': 'No observations remain after min-ps overlap filtering'
        }
    
    # Calculate weights for each regime
    weights_a, weights_b = calculate_regime_iptw_weights(
        data, regime_vars, regime_a, regime_b, propensity_scores,
        stabilized=stabilized, truncate=truncate, max_weight=max_weight
    )
    
    # Get outcome values
    outcome_vals = data[outcome].values
    
    # Calculate weighted means for each regime
    mask_a = weights_a > 0
    mask_b = weights_b > 0
    
    if not np.any(mask_a) or not np.any(mask_b):
        return {
            'ate': None,
            'status': 'insufficient_data',
            'error': 'No observations matching one or both regimes'
        }
    
    mean_outcome_a = np.average(outcome_vals[mask_a], weights=weights_a[mask_a])
    mean_outcome_b = np.average(outcome_vals[mask_b], weights=weights_b[mask_b])
    
    ate = mean_outcome_b - mean_outcome_a
    
    # Bootstrap uncertainty for regime ATE, recomputing PS and weights each replicate.
    se_ate, ci_lower, ci_upper, _ = _bootstrap_regime_iptw_ate(
        model=model,
        regime_vars=regime_vars,
        regime_a=regime_a,
        regime_b=regime_b,
        outcome=outcome,
        data=data,
        stabilized=stabilized,
        truncate=truncate,
        min_ps=min_ps,
        max_weight=max_weight,
        hard_evidence=hard_evidence,
        virtual_evidence=virtual_evidence,
        n_bootstrap=n_bootstrap,
    )
    
    # Calculate effective sample sizes
    n_a = np.sum(mask_a)
    n_b = np.sum(mask_b)
    ess_a = _effective_sample_size(weights_a[mask_a]) if n_a > 0 else 0
    ess_b = _effective_sample_size(weights_b[mask_b]) if n_b > 0 else 0
    
    pseudo_treatment = np.zeros(len(data))
    pseudo_treatment[mask_b] = 1
    combined_weights = np.zeros(len(data))
    combined_weights[mask_a] = weights_a[mask_a]
    combined_weights[mask_b] = weights_b[mask_b]

    # Create temporary dataframe for balance checking
    temp_data = data.copy()
    temp_data['_pseudo_treatment_'] = pseudo_treatment
    
    # Only check balance for observations in regime A or B
    regime_mask = mask_a | mask_b
    balance_data = temp_data[regime_mask].copy()
    balance_weights = combined_weights[regime_mask]
    
    # For diagnostics, report all available baseline covariates.
    balance_covariates = _build_balance_covariates(
        model=model,
        data=data,
        treatment_vars=regime_vars,
        outcome=outcome,
        base_confounders=confounders,
    )

    balance_before = check_covariate_balance(
        balance_data, '_pseudo_treatment_', balance_covariates, weights=None
    )
    balance_after = check_covariate_balance(
        balance_data, '_pseudo_treatment_', balance_covariates, weights=balance_weights
    )
    
    return {
        'ate': ate,
        'se': se_ate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'mean_outcome_a': mean_outcome_a,
        'mean_outcome_b': mean_outcome_b,
        'n_a': int(n_a),
        'n_b': int(n_b),
        'weight_summary_a': {
            'mean': float(np.mean(weights_a[mask_a])),
            'min': float(np.min(weights_a[mask_a])),
            'max': float(np.max(weights_a[mask_a])),
            'ess': float(ess_a),
        },
        'weight_summary_b': {
            'mean': float(np.mean(weights_b[mask_b])),
            'min': float(np.min(weights_b[mask_b])),
            'max': float(np.max(weights_b[mask_b])),
            'ess': float(ess_b),
        },
        'balance_before': balance_before,
        'balance_after': balance_after,
        'confounders': balance_covariates,
        'ps_confounders': confounders,
        'status': 'success'
    }


def _parse_value(raw: str) -> Any:
    raw = raw.strip()
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    if raw.isdigit():
        return int(raw)
    return raw


def _preprocess_evidence(evidence: dict, model, config: dict) -> dict:
    """Preprocess evidence values to match BN state encodings.

    Important: we do not run single-row binning here because quantile binning on a
    single row is not meaningful and can produce invalid mappings.
    """
    if not evidence:
        return evidence
    
    scalar_evidence: dict[str, Any] = {}
    non_scalar_evidence: dict[str, Any] = {}
    for key, value in evidence.items():
        if isinstance(value, (dict, list, tuple)):
            non_scalar_evidence[key] = value
        else:
            scalar_evidence[key] = value

    processed_evidence: dict[str, Any] = dict(non_scalar_evidence)
    if not scalar_evidence:
        return processed_evidence

    binning_config = (config or {}).get('binning', {})
    for key, value in scalar_evidence.items():
        if key not in model.nodes():
            processed_evidence[key] = value
            continue

        cpd = model.get_cpds(key)
        if cpd is None:
            processed_evidence[key] = value
            continue

        state_names = list(cpd.state_names.get(key, list(range(cpd.cardinality[0]))))

        # Direct exact match first.
        if value in state_names:
            processed_evidence[key] = value
            continue

        # Try common scalar conversions.
        candidates: list[Any] = []
        if isinstance(value, str):
            parsed = _parse_value(value)
            candidates.extend([parsed, str(parsed)])
        else:
            candidates.extend([value, str(value)])
            try:
                as_float = float(value)
                candidates.append(int(as_float) if as_float.is_integer() else as_float)
            except (TypeError, ValueError):
                pass

        matched = False
        for cand in candidates:
            if cand in state_names:
                processed_evidence[key] = cand
                matched = True
                break

        if matched:
            continue

        # If variable is configured for binning, require explicit bin/state value.
        if key in binning_config:
            raise ValueError(
                f"Evidence '{key}={value}' is not a valid BN state. "
                f"This variable is binned; pass an explicit bin/state from {state_names}."
            )

        # Fallback: keep raw value; downstream virtual-evidence builder may still reject it.
        processed_evidence[key] = value

    return processed_evidence


def _build_hard_and_virtual_evidence(model, evidence: dict | None) -> tuple[dict, list[TabularCPD]]:
    """Build virtual-evidence CPDs only (hard evidence intentionally disabled)."""
    if not evidence:
        return {}, []

    virtual_evidence: list[TabularCPD] = []

    for var, value in evidence.items():
        if var not in model.nodes():
            continue

        cpd = model.get_cpds(var)
        if cpd is None:
            continue

        state_names = list(cpd.state_names.get(var, list(range(cpd.cardinality[0]))))

        if isinstance(value, dict):
            probs = np.zeros(len(state_names), dtype=float)
            for state_key, prob in value.items():
                state = _parse_value(str(state_key)) if isinstance(state_key, str) else state_key
                if state in state_names:
                    probs[state_names.index(state)] = float(prob)
            if probs.sum() > 0:
                probs = probs / probs.sum()
                virtual_evidence.append(
                    TabularCPD(
                        variable=var,
                        variable_card=len(state_names),
                        values=[[float(p)] for p in probs.tolist()],
                        state_names={var: state_names},
                    )
                )
            continue

        if isinstance(value, (list, tuple)):
            if len(value) != len(state_names):
                raise ValueError(
                    f"Virtual evidence for '{var}' has {len(value)} probs but cardinality is {len(state_names)}."
                )
            probs = np.array([float(p) for p in value], dtype=float)
            if probs.sum() <= 0:
                raise ValueError(f"Virtual evidence probabilities for '{var}' must sum to > 0.")
            probs = probs / probs.sum()
            virtual_evidence.append(
                TabularCPD(
                    variable=var,
                    variable_card=len(state_names),
                    values=[[float(p)] for p in probs.tolist()],
                    state_names={var: state_names},
                )
            )
            continue

        # Scalar evidence is encoded as point-mass virtual evidence.
        # This keeps conditioning entirely in virtual evidence mode.
        state_value = value
        if state_value not in state_names:
            parsed = _parse_value(str(state_value)) if isinstance(state_value, str) else state_value
            if parsed in state_names:
                state_value = parsed
            elif str(state_value) in state_names:
                state_value = str(state_value)
            else:
                print(f"Warning: evidence state {value!r} not found for '{var}', skipping.")
                continue

        probs = np.zeros(len(state_names), dtype=float)
        probs[state_names.index(state_value)] = 1.0
        virtual_evidence.append(
            TabularCPD(
                variable=var,
                variable_card=len(state_names),
                values=[[float(p)] for p in probs.tolist()],
                state_names={var: state_names},
            )
        )

    return {}, virtual_evidence


def _parse_evidence(pairs: list[str]) -> dict[str, Any]:
    evidence = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Evidence must be in key=value form. Got: {pair}")
        key, value = pair.split("=", 1)
        evidence[key.strip()] = _parse_value(value.strip())
    return evidence


def _extract_scalar_evidence_constraints(evidence: dict | None) -> dict[str, Any]:
    """Extract scalar equality constraints for IPTW subgroup conditioning."""
    if not evidence:
        return {}
    constraints: dict[str, Any] = {}
    for k, v in evidence.items():
        if isinstance(v, (dict, list, tuple)):
            continue
        constraints[k] = v
    return constraints


def _print_binning_summary(data: pd.DataFrame, binning_config: dict, header: str) -> None:
    """Print compact summary of configured binned variables after cleaning."""
    if not binning_config:
        return

    print(f"\n{header}")
    printed = 0
    for col, cfg in binning_config.items():
        if col not in data.columns:
            continue
        vals = data[col].dropna()
        if vals.empty:
            print(f"  {col}: no data")
            printed += 1
            continue
        vc = vals.value_counts().sort_index()
        counts_str = ", ".join([f"{int(idx)}:{int(cnt)}" for idx, cnt in vc.items()])
        method = cfg.get("method", "quantile")
        n_bins = cfg.get("n_bins", "?")
        print(
            f"  {col} (method={method}, n_bins={n_bins}) -> "
            f"states={sorted(vals.unique().tolist())}; counts=[{counts_str}]"
        )
        printed += 1

    if printed == 0:
        print("  [No configured binned columns present in current dataset]")


def _print_binning_intervals(raw_data: pd.DataFrame, dag_nodes: list[str], binning_config: dict, header: str) -> None:
    """Print bin intervals implied by configured binning on pre-clean data."""
    if not binning_config:
        return

    print(f"\n{header}")
    available_cols = [c for c in dag_nodes if c in raw_data.columns]
    if not available_cols:
        print("  [No DAG columns available to infer intervals]")
        return

    # Match cleaning behavior that drops rows with any null in DAG columns.
    base = raw_data[available_cols].dropna()
    printed = 0
    for col, cfg in binning_config.items():
        if col not in base.columns:
            continue
        numeric = pd.to_numeric(base[col], errors="coerce").dropna()
        if numeric.empty:
            print(f"  {col}: interval inference unavailable (non-numeric/empty)")
            printed += 1
            continue

        n_bins = int(cfg.get("n_bins", 5))
        method = cfg.get("method", "quantile")
        try:
            if method == "quantile":
                _, bins = pd.qcut(numeric, q=n_bins, retbins=True, duplicates="drop")
            else:
                _, bins = pd.cut(numeric, bins=n_bins, retbins=True, duplicates="drop")
        except Exception as e:
            print(f"  {col}: interval inference failed ({e})")
            printed += 1
            continue

        intervals = []
        for i in range(len(bins) - 1):
            left = float(bins[i])
            right = float(bins[i + 1])
            intervals.append(f"{i}:[{left:.4g}, {right:.4g}]")

        print(
            f"  {col} (method={method}, requested_bins={n_bins}, actual_bins={len(bins)-1}) -> "
            + ", ".join(intervals)
        )
        printed += 1

    if printed == 0:
        print("  [No configured binned columns present in current dataset]")


def run_multiple_treatment_effects(model, outcomes: list[str], treatments: list[str], evidence: dict = None) -> list[dict]:
    """Run treatment effect calculations for multiple treatment-outcome pairs."""
    results = []
    
    if evidence:
        print(f"Conditioning on: {evidence}")
    
    print(f"\n{'='*80}")
    print(f"TREATMENT EFFECT ANALYSIS")
    print(f"{'='*80}\n")
    
    for outcome in outcomes:
        print(f"\n{'-'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'-'*80}\n")
        
        for treatment in treatments:
            # Skip if treatment is the same as outcome
            if treatment == outcome:
                continue
            
            # Determine treatment values (0 = no treatment, 1 = treatment)
            control_value = 0
            treated_value = 1
            
            # For outcomes, typically we want to measure P(outcome=1)
            outcome_state = 1
            
            try:
                ate = average_treatment_effect(
                    model,
                    treatment,
                    outcome,
                    (control_value, treated_value),
                    outcome_state,
                    evidence=evidence,
                )
                
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "control_value": control_value,
                    "treated_value": treated_value,
                    "outcome_state": outcome_state,
                    "ate": ate,
                    "status": "success"
                }
                
                print(f"  {treatment} (0→1) → {outcome}(={outcome_state})")
                print(f"    ATE: {ate:+.6f}")
                
                if ate > 0:
                    print(f"    ↑ Treatment INCREASES outcome by {abs(ate):.2%}")
                elif ate < 0:
                    print(f"    ↓ Treatment DECREASES outcome by {abs(ate):.2%}")
                else:
                    print(f"    → No effect")
                print()
                
            except Exception as e:
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "control_value": control_value,
                    "treated_value": treated_value,
                    "outcome_state": outcome_state,
                    "ate": None,
                    "status": "error",
                    "error": str(e)
                }
                print(f"  {treatment} → {outcome}: ERROR - {e}\n")
            
            results.append(result)
    
    return results


def run_gcomputation_regime_analysis(model, config_path: str, outcomes: list[str],
                                    regime_comparisons: list[tuple] | list[dict],
                                    test_data_path: str = None,
                                    evidence: dict = None,
                                    stratify_by: list[str] = None) -> list[dict]:
    """Run g-computation regime analysis with sensitivity analysis support.
    
    This approach uses the stratified BN (with IPW-adjusted CPDs) to:
    1. Load real patient data from test set
    2. For each patient, use their baseline characteristics as evidence
    3. Simulate both treatment regimes using do-calculus
    4. Average individual treatment effects (ITEs) to get population ATE
    5. Optionally run sensitivity analysis across baseline condition combinations
    
    Args:
        model: Fitted Bayesian Network (preferably with stratified CPDs)
        config_path: Path to config YAML
        outcomes: List of outcome variables
        regime_comparisons: List of tuples (regime_a, regime_b, label) or
                          dicts with 'regime_a', 'regime_b', 'label', and
                          optional 'sensitivity_conditions' (dict of var: [values])
        test_data_path: Path to test data pickle (REQUIRED)
        evidence: Optional global filtering conditions
        stratify_by: List of variables to stratify results by
        
    Returns:
        List of result dictionaries for each regime comparison and stratum
    """
    from itertools import product
    
    print(f"\n{'='*80}")
    print(f"G-COMPUTATION REGIME ANALYSIS")
    print(f"Using Stratified BN with Patient-Level Evidence")
    print(f"{'='*80}\n")
    
    if not test_data_path:
        raise ValueError("test_data_path is REQUIRED for g-computation (use --test-data-path)")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load test data
    print(f"Loading patient data from: {test_data_path}")
    full_data = pd.read_pickle(test_data_path)
    print(f"Loaded {len(full_data)} patients")
    
    # Filter to model nodes only
    full_data = full_data[[col for col in full_data.columns if col in model.nodes()]]
    print(f"Using {len(full_data.columns)} variables from BN model\n")
    
    # Normalize regime_comparisons to dict format
    normalized_comparisons = []
    for comp in regime_comparisons:
        if isinstance(comp, tuple):
            # Old format: (regime_a, regime_b, label)
            normalized_comparisons.append({
                'regime_a': comp[0],
                'regime_b': comp[1],
                'label': comp[2],
                'sensitivity_conditions': None
            })
        else:
            # Dict format with optional sensitivity_conditions
            normalized_comparisons.append(comp)
    
    # Identify baseline confounders (exclude treatments and outcomes)
    # Get regime vars from first comparison
    first_comp = normalized_comparisons[0]
    regime_vars = list(first_comp['regime_a'].keys())
    all_outcomes = set(outcomes)
    
    # Baseline confounders are all variables except treatments and outcomes
    baseline_vars = [v for v in full_data.columns if v in model.nodes() 
                    and v not in regime_vars and v not in all_outcomes]
    baseline_vars = _filter_confounders(baseline_vars, regime_vars)
    
    print(f"Data columns available: {list(full_data.columns)[:10]}{'...' if len(full_data.columns) > 10 else ''}")
    print(f"Regime variables: {regime_vars}")
    print(f"Outcomes: {list(all_outcomes)}")
    print(f"Baseline confounders ({len(baseline_vars)}): {baseline_vars[:10]}{'...' if len(baseline_vars) > 10 else ''}")
    print(f"Sample patient data (first row):")
    if len(full_data) > 0:
        first_patient = full_data.iloc[0]
        for var in baseline_vars[:5]:
            if var in first_patient.index:
                print(f"  {var}: {first_patient[var]}")
    print()
    
    results = []
    
    # Helper function to compute statistics for a group of ITEs
    def compute_ate_stats(ites_list, risks_a_list, risks_b_list, label=""):
        """Compute ATE and statistics from lists of ITEs and risks."""
        if len(ites_list) == 0:
            return None
            
        ites_array = np.array(ites_list)
        risks_a_array = np.array(risks_a_list)
        risks_b_array = np.array(risks_b_list)
        
        ate = float(np.mean(ites_array))
        se = float(np.std(ites_array, ddof=1) / np.sqrt(len(ites_array))) if len(ites_array) > 1 else 0.0
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        
        z_stat = ate / se if se > 0 else 0
        from scipy import stats
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        return {
            'ate': ate,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value': p_value,
            'n_patients': len(ites_list),
            'mean_risk_a': float(np.mean(risks_a_array)),
            'mean_risk_b': float(np.mean(risks_b_array)),
            'sd_risk_a': float(np.std(risks_a_array, ddof=1)) if len(risks_a_array) > 1 else 0.0,
            'sd_risk_b': float(np.std(risks_b_array, ddof=1)) if len(risks_b_array) > 1 else 0.0
        }
    
    # Helper function to generate all combinations of sensitivity conditions
    def generate_sensitivity_combinations(sensitivity_conditions):
        """Generate all combinations from sensitivity conditions dict.
        
        Args:
            sensitivity_conditions: Dict of {var: [val1, val2, ...]} or None
            
        Returns:
            List of tuples: [(condition_dict, condition_label), ...]
        """
        if not sensitivity_conditions:
            return [(None, "All patients")]
        
        # Get variables and their value lists
        variables = list(sensitivity_conditions.keys())
        value_lists = [sensitivity_conditions[var] for var in variables]
        
        # Generate Cartesian product
        combinations = []
        for value_combo in product(*value_lists):
            condition_dict = dict(zip(variables, value_combo))
            condition_label = ", ".join([f"{var.split('.')[-1]}={val}" for var, val in condition_dict.items()])
            combinations.append((condition_dict, condition_label))
        
        return combinations
    
    # Filter outcomes to only those in model
    valid_outcomes = [o for o in outcomes if o in model.nodes()]
    if len(valid_outcomes) < len(outcomes):
        missing = [o for o in outcomes if o not in model.nodes()]
        print(f"Warning: Outcomes not in model (skipping): {missing}")
    
    if len(valid_outcomes) == 0:
        print("ERROR: No valid outcomes in model")
        return []
    
    print(f"\nQuerying {len(valid_outcomes)} outcomes jointly: {valid_outcomes}")
    print("(Joint query captures correlations between outcomes for better distinction)\n")
    
    # Loop over regime comparisons and sensitivity conditions
    for comparison in normalized_comparisons:
        regime_a = comparison['regime_a']
        regime_b = comparison['regime_b']
        comparison_label = comparison['label']
        sensitivity_conditions = comparison.get('sensitivity_conditions', None)
        
        # Generate all sensitivity condition combinations
        sensitivity_combos = generate_sensitivity_combinations(sensitivity_conditions)
        
        if len(sensitivity_combos) > 1:
            print(f"\n{'='*80}")
            print(f"Comparison: {comparison_label}")
            print(f"  Regime A: {regime_a}")
            print(f"  Regime B: {regime_b}")
            print(f"  Sensitivity Analysis: {len(sensitivity_combos)} subgroups")
            print(f"{'='*80}\n")
        
        # Loop over each sensitivity condition combination
        for condition_dict, condition_label in sensitivity_combos:
            # Generate pseudo-patients or use full dataset
            if condition_dict:
                # Check all condition variables exist in data
                missing_vars = [var for var in condition_dict.keys() if var not in full_data.columns]
                if missing_vars:
                    print(f"  ⚠ Warning: Condition variables not in data: {missing_vars}, skipping this combination")
                    continue
                
                print(f"\n{'='*80}")
                print(f"SENSITIVITY CONDITION: {condition_label}")
                print(f"{'='*80}")
                print(f"  Comparison: {comparison_label}")
                print(f"  Fixed baseline characteristics: {condition_dict}")
                print(f"  Generating {len(full_data)} pseudo-patients with these characteristics")
                print(f"{'='*80}")
                
                # Generate pseudo-patients: use full dataset but override specific variables
                # This creates counterfactual patients with controlled baseline characteristics
                data = full_data.copy()
                for var, val in condition_dict.items():
                    data[var] = val  # Override with fixed sensitivity value
            else:
                # No sensitivity conditions - use real patient data
                data = full_data
                if len(sensitivity_combos) == 1:
                    print(f"\n{'='*80}")
                    print(f"Comparison: {comparison_label}")
                    print(f"  Regime A: {regime_a}")
                    print(f"  Regime B: {regime_b}")
                    print(f"{'='*80}\n")
        
            # For each patient, compute joint query for all outcomes
            # Store results by outcome
            patient_results_by_outcome = {outcome: [] for outcome in valid_outcomes}
            errors = []
            
            for patient_num, (idx, row) in enumerate(data.iterrows()):
                # Build patient-specific evidence from baseline characteristics
                patient_evidence = {}
                
                for var in baseline_vars:
                    if var in row.index and pd.notna(row[var]):
                        patient_evidence[var] = row[var]
                
                # Diagnostic: print first 3 patients' evidence
                if patient_num < 3:
                    print(f"  Patient {idx} evidence ({len(patient_evidence)} vars):")
                    evidence_preview = list(patient_evidence.items())[:5]
                    for k, v in evidence_preview:
                        print(f"    {k}: {v}")
                    if len(patient_evidence) > 5:
                        print(f"    ... and {len(patient_evidence) - 5} more variables")
                
                try:
                    # Simulate Regime A for this patient - QUERY ALL OUTCOMES JOINTLY
                    do_model_a = do_intervene(model, regime_a)
                    infer_a = VariableElimination(do_model_a)
                    result_a = infer_a.query(
                        variables=valid_outcomes,  # Joint query for all outcomes
                        evidence=patient_evidence if patient_evidence else None,
                        show_progress=False
                    )
                    
                    # Simulate Regime B for this patient - QUERY ALL OUTCOMES JOINTLY                    do_model_b = do_intervene(model, regime_b)
                    infer_b = VariableElimination(do_model_b)
                    result_b = infer_b.query(
                        variables=valid_outcomes,  # Joint query for all outcomes
                        evidence=patient_evidence if patient_evidence else None,
                        show_progress=False
                    )
                    
                    # Diagnostic: print first patient's joint distribution info
                    if patient_num == 0:
                        print(f"    Joint query returned: {result_a.variables}, shape={result_a.values.shape}")
                    
                    # Extract marginal probabilities for each outcome
                    # For each outcome, marginalize over all other outcomes
                    for outcome in valid_outcomes:
                        # Marginalize to get P(outcome=1 | evidence)
                        marginal_a = result_a.marginalize([o for o in valid_outcomes if o != outcome], inplace=False)
                        marginal_b = result_b.marginalize([o for o in valid_outcomes if o != outcome], inplace=False)
                        
                        # Extract probability of positive outcome (state=1)
                        risk_a = marginal_a.values[1] if len(marginal_a.values) > 1 else np.nan
                        risk_b = marginal_b.values[1] if len(marginal_b.values) > 1 else np.nan
                        
                        # Diagnostic: print first 3 patients' risks for first outcome only
                        if patient_num < 3 and outcome == valid_outcomes[0]:
                            print(f"    {outcome}: Risk(A)={risk_a:.4f}, Risk(B)={risk_b:.4f}, ITE={risk_b-risk_a:.4f}")
                        
                        # Store individual treatment effect
                        if not np.isnan(risk_a) and not np.isnan(risk_b):
                            ite = risk_b - risk_a
                            patient_results_by_outcome[outcome].append({
                                'patient_idx': idx,
                                'ite': ite,
                                'risk_a': risk_a,
                                'risk_b': risk_b,
                                'baseline': {var: row[var] for var in baseline_vars if var in row.index and pd.notna(row[var])}
                            })
                        
                except Exception as e:
                    # Collect error information
                    if len(errors) < 3:  # Only store first 3 errors for reporting
                        errors.append({
                            'patient_idx': idx,
                            'error': str(e),
                            'evidence_vars': list(patient_evidence.keys())
                        })
                    continue
            
            print()  # Blank line after patient loop
            
            # Check for computation errors
            computed_any = any(len(patient_results_by_outcome[o]) > 0 for o in valid_outcomes)
            if not computed_any:
                print(f"  ERROR: No valid ITEs computed for any outcome")
                print(f"  Attempted: {len(data)} patients, Failures: {len(errors)}")
                if errors:
                    print(f"\n  Sample errors (first {len(errors)}):")
                    for err in errors:
                        print(f"    Patient {err['patient_idx']}: {err['error']}")
                        print(f"      Evidence variables: {err['evidence_vars'][:5]}{'...' if len(err['evidence_vars']) > 5 else ''}")
                for outcome in valid_outcomes:
                    results.append({
                        'outcome': outcome,
                        'comparison': comparison_label,
                        'regime_a': regime_a,
                        'regime_b': regime_b,
                        'sensitivity_condition': condition_dict,
                        'sensitivity_label': condition_label,
                        'ate': None,
                        'status': 'error',
                        'error': 'No valid ITEs',
                        'sample_errors': errors
                    })
                continue
            
            # Process results for each outcome
            for outcome in valid_outcomes:
                patient_results = patient_results_by_outcome[outcome]
                
                if len(patient_results) == 0:
                    print(f"\n  WARNING: No valid ITEs for outcome '{outcome}'")
                    results.append({
                        'outcome': outcome,
                        'comparison': comparison_label,
                        'regime_a': regime_a,
                        'regime_b': regime_b,
                        'sensitivity_condition': condition_dict,
                        'sensitivity_label': condition_label,
                        'ate': None,
                        'status': 'error',
                        'error': f'No valid ITEs for {outcome}'
                    })
                    continue
                
                print(f"\n  {'-'*78}")
                print(f"  OUTCOME: {outcome}")
                print(f"  {'-'*78}")
                
                # If stratification requested, group by baseline characteristics
                stratify_vars = stratify_by  # Use the parameter passed to function
                if stratify_vars:
                    # Check all stratification variables are available
                    available_stratify_vars = [v for v in stratify_vars if v in baseline_vars]
                    if len(available_stratify_vars) < len(stratify_vars):
                        missing = [v for v in stratify_vars if v not in baseline_vars]
                        print(f"    ⚠ Warning: Stratification variables not available: {missing}")
                    
                    if len(available_stratify_vars) == 0:
                        print(f"    ⚠ No stratification variables available, computing overall ATE only")
                        stratify_vars = None  # Fall back to overall analysis
                    else:
                        stratify_vars = available_stratify_vars
                        print(f"    Stratifying by: {stratify_vars}")
                
                # Compute overall ATE (population-level)
                all_ites = [p['ite'] for p in patient_results]
                all_risks_a = [p['risk_a'] for p in patient_results]
                all_risks_b = [p['risk_b'] for p in patient_results]
                
                overall_stats = compute_ate_stats(all_ites, all_risks_a, all_risks_b)
                
                print(f"\n    Overall Results (N={overall_stats['n_patients']}):")
                print(f"      Mean risk (Regime A):    {overall_stats['mean_risk_a']:.4f} (SD={overall_stats['sd_risk_a']:.4f})")
                print(f"      Mean risk (Regime B):    {overall_stats['mean_risk_b']:.4f} (SD={overall_stats['sd_risk_b']:.4f})")
                print(f"      ATE (B - A):             {overall_stats['ate']:+.4f} ({overall_stats['ate']*100:+.2f} pp)")
                print(f"      Standard Error:          {overall_stats['se']:.4f}")
                print(f"      95% CI:                  [{overall_stats['ci_lower']:.4f}, {overall_stats['ci_upper']:.4f}]")
                print(f"      P-value:                 {overall_stats['p_value']:.4f}")
                
                if overall_stats['p_value'] < 0.05:
                    if overall_stats['ate'] > 0:
                        print(f"      ✓ Regime B INCREASES {outcome} (p={overall_stats['p_value']:.4f})")
                    else:
                        print(f"      ✓ Regime B DECREASES {outcome} (p={overall_stats['p_value']:.4f})")
                else:
                    print(f"      ~ No significant difference (CI includes 0)")
                
                # Store overall result
                result_entry = {
                    'outcome': outcome,
                    'comparison': comparison_label,
                    'regime_a': regime_a,
                    'regime_b': regime_b,
                    'sensitivity_condition': condition_dict,
                    'sensitivity_label': condition_label,
                    'stratum': 'overall',
                    **overall_stats,
                    'status': 'success'
                }
                results.append(result_entry)
                
                # Stratified analysis if requested
                if stratify_vars:
                    print(f"\n    Stratified Results:")
                    print(f"    {'-'*76}")
                    
                    # Group patients by stratification variables
                    strata = {}
                    for p in patient_results:
                        # Create stratum key from baseline values
                        stratum_key = tuple((var, p['baseline'].get(var)) for var in stratify_vars if var in p['baseline'])
                        if stratum_key not in strata:
                            strata[stratum_key] = []
                        strata[stratum_key].append(p)
                    
                    # Sort strata by key for consistent output
                    for stratum_key in sorted(strata.keys()):
                        patients_in_stratum = strata[stratum_key]
                        
                        # Create readable stratum label
                        stratum_label = ', '.join([f"{var}={val}" for var, val in stratum_key])
                        
                        # Compute ATE for this stratum
                        strat_ites = [p['ite'] for p in patients_in_stratum]
                        strat_risks_a = [p['risk_a'] for p in patients_in_stratum]
                        strat_risks_b = [p['risk_b'] for p in patients_in_stratum]
                        
                        strat_stats = compute_ate_stats(strat_ites, strat_risks_a, strat_risks_b, stratum_label)
                        
                        if strat_stats is None:
                            continue
                        
                        print(f"\n      Stratum: {stratum_label} (N={strat_stats['n_patients']})")
                        print(f"        ATE:      {strat_stats['ate']:+.4f} ({strat_stats['ate']*100:+.2f} pp)")
                        print(f"        SE:       {strat_stats['se']:.4f}")
                        print(f"        95% CI:   [{strat_stats['ci_lower']:.4f}, {strat_stats['ci_upper']:.4f}]")
                        print(f"        P-value:  {strat_stats['p_value']:.4f}")
                        
                        if strat_stats['p_value'] < 0.05:
                            sig_marker = "✓"
                        else:
                            sig_marker = "~"
                        print(f"        {sig_marker} {'Significant' if strat_stats['p_value'] < 0.05 else 'Not significant'}")
                        
                        # Store stratified result
                        results.append({
                            'outcome': outcome,
                            'comparison': comparison_label,
                            'regime_a': regime_a,
                            'regime_b': regime_b,
                            'sensitivity_condition': condition_dict,
                            'sensitivity_label': condition_label,
                            'stratum': stratum_label,
                            'stratum_vars': dict(stratum_key),
                            **strat_stats,
                            'status': 'success'
                        })
    
    # Print summary table of all sensitivity combinations
    print(f"\n{'='*80}")
    print(f"SENSITIVITY ANALYSIS SUMMARY")
    print(f"{'='*80}\n")
    
    # Group results by comparison and outcome
    for comparison in normalized_comparisons:
        comp_label = comparison['label']
        comp_results = [r for r in results if r['comparison'] == comp_label and r.get('stratum') == 'overall']
        
        if not comp_results:
            continue
        
        print(f"\nComparison: {comp_label}")
        print(f"{'-'*80}")
        
        # Group by outcome
        for outcome in valid_outcomes:
            outcome_results = [r for r in comp_results if r['outcome'] == outcome]
            if not outcome_results:
                continue

            
            print(f"\n  Outcome: {outcome}")
            print(f"  {'Sensitivity Condition':<50} {'ATE':>10} {'95% CI':>20} {'P-value':>10}")
            print(f"  {'-'*50} {'-'*10} {'-'*20} {'-'*10}")
            
            for r in outcome_results:
                sens_label = r['sensitivity_label']
                ate = r['ate']
                ci_lower = r['ci_lower']
                ci_upper = r['ci_upper']
                p_value = r['p_value']
                regime_a = r['regime_a']
                regime_b = r['regime_b']
                
                # Significance marker
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                print(f"  Regime A: {regime_a}")
                print(f"  Regime B: {regime_b}")
                print(f"  {sens_label:<50} {ate:+10.4f} [{ci_lower:7.4f}, {ci_upper:7.4f}] {p_value:10.4f} {sig}")
    
    # Export to CSV
    if results:
        import csv
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        
        csv_path = output_dir / "sensitivity_analysis_results.csv"
        
        print(f"\n{'='*80}")
        print(f"Exporting results to CSV: {csv_path}")
        print(f"{'='*80}\n")
        
        # Prepare flattened results for CSV
        csv_rows = []
        for r in results:
            # Flatten regime dicts to strings
            regime_a_str = ', '.join([f"{k}={v}" for k, v in r['regime_a'].items()])
            regime_b_str = ', '.join([f"{k}={v}" for k, v in r['regime_b'].items()])
            
            # Flatten sensitivity condition dict
            if r.get('sensitivity_condition'):
                sens_cond_str = ', '.join([f"{k}={v}" for k, v in r['sensitivity_condition'].items()])
            else:
                sens_cond_str = "All patients"
            
            row = {
                'comparison': r['comparison'],
                'outcome': r['outcome'],
                'regime_a': regime_a_str,
                'regime_b': regime_b_str,
                'sensitivity_condition': sens_cond_str,
                'sensitivity_label': r['sensitivity_label'],
                'stratum': r.get('stratum', 'overall'),
                'n_patients': r.get('n_patients', ''),
                'ate': r.get('ate', ''),
                'se': r.get('se', ''),
                'ci_lower': r.get('ci_lower', ''),
                'ci_upper': r.get('ci_upper', ''),
                'p_value': r.get('p_value', ''),
                'mean_risk_a': r.get('mean_risk_a', ''),
                'mean_risk_b': r.get('mean_risk_b', ''),
                'sd_risk_a': r.get('sd_risk_a', ''),
                'sd_risk_b': r.get('sd_risk_b', ''),
                'status': r.get('status', '')
            }
            csv_rows.append(row)
        
        # Write CSV
        fieldnames = ['comparison', 'outcome', 'regime_a', 'regime_b', 
                     'sensitivity_condition', 'sensitivity_label', 'stratum',
                     'n_patients', 'ate', 'se', 'ci_lower', 'ci_upper', 'p_value',
                     'mean_risk_a', 'mean_risk_b', 'sd_risk_a', 'sd_risk_b', 'status']
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"  ✓ Exported {len(csv_rows)} rows to {csv_path}")
        print(f"  Columns: {', '.join(fieldnames[:5])}... ({len(fieldnames)} total)")
    
    return results


def run_iptw_regime_analysis(model, config_path: str, outcomes: list[str],
                           regime_comparisons: list[tuple],
                           test_data_path: str = None,
                           stabilized: bool = True,
                           truncate: float | None = 0.005,
                           min_ps: float | None = 0.0,
                           max_weight: float | None = 20.0,
                           n_bootstrap: int = 500,
                           evidence: dict = None) -> list[dict]:
    """Run IPTW-based regime comparison analysis.
    
    Args:
        model: Fitted Bayesian Network
        config_path: Path to config YAML
        outcomes: List of outcome variables
        regime_comparisons: List of (regime_a, regime_b, label) tuples
        test_data_path: Path to test data
        stabilized: Use stabilized weights
        truncate: Truncate propensity scores
        evidence: Optional dict of variables to condition on (e.g., {'biopsy_grading': 2})
        
    Returns:
        List of result dictionaries for each regime comparison
    """
    print(f"\n{'='*80}")
    print(f"IPTW-BASED REGIME COMPARISON ANALYSIS")
    print(f"{'='*80}")
    print(f"\nComparing treatment regimes using Inverse Probability Weighting")
    print(f"Stabilized weights: {stabilized}")
    if truncate is None:
        print("Propensity score truncation: none")
    else:
        print(f"Propensity score truncation: [{truncate:.3f}, {1-truncate:.3f}]")
    print(f"Minimum propensity threshold: {min_ps if min_ps is not None else 'none'}")
    print(f"Regime max weight cap: {max_weight if max_weight is not None else 'none'}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    binning_config = config.get('binning', {})
    
    # Load data
    if test_data_path:
        print(f"Loading data from: {test_data_path}")
        data = pd.read_pickle(test_data_path)
        raw_for_bins = data.copy()
        data = _validate_and_clean_data(data, list(model.nodes()), binning_config)
        _print_binning_intervals(raw_for_bins, list(model.nodes()), binning_config, "Binning intervals:")
        _print_binning_summary(data, binning_config, "Binning summary (after cleaning):")
        print(f"Loaded {len(data)} samples")
    else:
        print("Loading data from MongoDB...")
        load_dotenv()
        
        mongo_uri = config.get("mongodb", {}).get("uri") or os.getenv("MONGO_URI")
        db_name = config.get("mongodb", {}).get("database") or os.getenv("MONGO_DB")
        collection_name = config.get("mongodb", {}).get("collection") or os.getenv("MONGO_COLLECTION")
        
        if not all([mongo_uri, db_name, collection_name]):
            raise ValueError("MongoDB connection details missing")
        
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        pipeline = config['mongodb']['pipeline']
        cursor = collection.aggregate(pipeline)
        raw_data = pd.json_normalize(cursor)
        data = raw_data.copy()
        
        if config.get('features'):
            data = _apply_feature_selection(data, config)
        
        raw_for_bins = data.copy()
        data = _validate_and_clean_data(data, list(model.nodes()), binning_config)
        _print_binning_intervals(raw_for_bins, list(model.nodes()), binning_config, "Binning intervals:")
        _print_binning_summary(data, binning_config, "Binning summary (after cleaning):")
        print(f"Loaded {len(data)} samples")
    
    hard_evidence, virtual_evidence = _build_hard_and_virtual_evidence(model, evidence)
    scalar_constraints = _extract_scalar_evidence_constraints(evidence)

    # Subgroup estimand: keep only rows matching scalar evidence values.
    if scalar_constraints:
        original_n = len(data)
        for var, value in scalar_constraints.items():
            if var in data.columns:
                data = data[data[var] == value]
        if len(data) == 0:
            raise ValueError(f"No samples remain after IPTW scalar-evidence filtering: {scalar_constraints}")
        print(
            f"IPTW sample conditioning: {original_n} -> {len(data)} "
            f"({len(data)/original_n*100:.1f}% retained)"
        )

    # Apply BN conditioning details
    if evidence:
        print(f"\n{'='*80}")
        print(f"CONDITIONING ON EVIDENCE")
        print(f"{'='*80}")
        print(f"Raw evidence: {evidence}")
        print("Hard evidence: {} (disabled; using virtual evidence only)")
        print(f"IPTW sample constraints: {scalar_constraints if scalar_constraints else '{}'}")
        if virtual_evidence:
            print("Virtual evidence:")
            for ve in virtual_evidence:
                states = ve.state_names.get(ve.variable, list(range(ve.variable_card)))
                probs = [float(p) for p in np.asarray(ve.values).reshape(-1).tolist()]
                pretty = ", ".join([f"{s}={p:.3f}" for s, p in zip(states, probs)])
                print(f"  {ve.variable}: {pretty}")
        else:
            print("Virtual evidence: none")
        print("\nNote: IPTW is computed on the evidence-matched subgroup; virtual evidence is also used in BN inference.")
        print(f"{'='*80}\n")
    
    # Extract regime variables from first comparison
    regime_vars = list(regime_comparisons[0][0].keys())
    print(f"Regime variables: {regime_vars}\n")
    
    results = []
    
    for outcome in outcomes:
        if outcome not in data.columns or outcome not in model.nodes():
            print(f"\nSkipping {outcome}: not in data or model")
            continue
        
        print(f"\n{'='*80}")
        print(f"Outcome: {outcome}")
        print(f"{'='*80}")
        
        for regime_a, regime_b, label in regime_comparisons:
            print(f"\n{'-'*80}")
            print(f"Comparison: {label}")
            print(f"  Regime A: {regime_a}")
            print(f"  Regime B: {regime_b}")
            print(f"{'-'*80}")
            
            try:
                iptw_result = estimate_regime_ate_with_iptw(
                    model, regime_vars, regime_a, regime_b, outcome, data,
                    stabilized=stabilized,
                    truncate=truncate,
                    min_ps=min_ps,
                    max_weight=max_weight,
                    n_bootstrap=n_bootstrap,
                    hard_evidence=None,
                    virtual_evidence=virtual_evidence,
                )
                
                if iptw_result['status'] != 'success':
                    print(f"  ✗ {iptw_result.get('error', 'Unknown error')}")
                    results.append({
                        'outcome': outcome,
                        'regime_a': regime_a,
                        'regime_b': regime_b,
                        'label': label,
                        'status': 'error',
                        'error': iptw_result.get('error')
                    })
                    continue
                
                ate = iptw_result['ate']
                se = iptw_result['se']
                ci_lower = iptw_result['ci_lower']
                ci_upper = iptw_result['ci_upper']
                
                print(f"\nResults:")
                print(f"  Mean outcome (Regime A): {iptw_result['mean_outcome_a']:.4f}")
                print(f"  Mean outcome (Regime B): {iptw_result['mean_outcome_b']:.4f}")
                print(f"  ATE (B - A):             {ate:+.4f} ({ate*100:+.2f} pp)")
                print(f"  Standard Error:          {se:.4f}")
                print(f"  95% CI:                  [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"  N (Regime A):            {iptw_result['n_a']}")
                print(f"  N (Regime B):            {iptw_result['n_b']}")
                
                # Interpret effect
                if ci_lower > 0:
                    print(f"  ✓ Regime B SIGNIFICANTLY INCREASES {outcome} (p < 0.05)")
                elif ci_upper < 0:
                    print(f"  ✓ Regime B SIGNIFICANTLY DECREASES {outcome} (p < 0.05)")
                else:
                    print(f"  ~ No significant difference (CI includes 0)")
                
                print(f"\n  Weight Summary (Regime A): mean={iptw_result['weight_summary_a']['mean']:.2f}, "
                      f"range=[{iptw_result['weight_summary_a']['min']:.2f}, {iptw_result['weight_summary_a']['max']:.2f}], "
                      f"ESS={iptw_result['weight_summary_a']['ess']:.1f}")
                print(f"  Weight Summary (Regime B): mean={iptw_result['weight_summary_b']['mean']:.2f}, "
                      f"range=[{iptw_result['weight_summary_b']['min']:.2f}, {iptw_result['weight_summary_b']['max']:.2f}], "
                      f"ESS={iptw_result['weight_summary_b']['ess']:.1f}")
                
                # Print detailed diagnostics
                print_regime_iptw_diagnostics(iptw_result, regime_a, regime_b)
                
                results.append({
                    'outcome': outcome,
                    'regime_a': regime_a,
                    'regime_b': regime_b,
                    'label': label,
                    'ate': ate,
                    'se': se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'mean_a': iptw_result['mean_outcome_a'],
                    'mean_b': iptw_result['mean_outcome_b'],
                    'n_a': iptw_result['n_a'],
                    'n_b': iptw_result['n_b'],
                    'weight_summary_a': iptw_result['weight_summary_a'],
                    'weight_summary_b': iptw_result['weight_summary_b'],
                    'balance_before': iptw_result.get('balance_before', {}),
                    'balance_after': iptw_result.get('balance_after', {}),
                    'confounders': iptw_result.get('confounders', []),
                    'status': 'success'
                })
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                results.append({
                    'outcome': outcome,
                    'regime_a': regime_a,
                    'regime_b': regime_b,
                    'label': label,
                    'status': 'error',
                    'error': str(e)
                })
    
    return results


def run_iptw_analysis(model, config_path: str, outcomes: list[str], treatments: list[str],
                     test_data_path: str = None, stabilized: bool = True, 
                     truncate: float | None = 0.05, min_ps: float | None = 0.0,
                     n_bootstrap: int = 500,
                     evidence: dict = None) -> list[dict]:
    """Run IPTW-based treatment effect analysis on real patient data.
    
    This addresses confounding by indication using Inverse Probability of Treatment Weighting.
    
    Args:
        model: Fitted Bayesian Network
        config_path: Path to config YAML
        outcomes: List of outcome variables
        treatments: List of treatment variables
        test_data_path: Path to test/validation data (if None, loads from MongoDB)
        stabilized: Use stabilized IPTW weights
        truncate: Truncate propensity scores to [truncate, 1-truncate]
        
    Returns:
        List of result dictionaries for each treatment-outcome pair
    """
    print(f"\n{'='*80}")
    print(f"IPTW-BASED TREATMENT EFFECT ANALYSIS")
    print(f"{'='*80}")
    print(f"\nAddressing confounding by indication using Inverse Probability of Treatment Weighting")
    print(f"Stabilized weights: {stabilized}")
    if truncate is None:
        print("Propensity score truncation: none")
    else:
        print(f"Propensity score truncation: [{truncate:.3f}, {1-truncate:.3f}]")
    print(f"Minimum propensity threshold: {min_ps if min_ps is not None else 'none'}")
    print(f"Bootstrap samples: {n_bootstrap}")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    binning_config = config.get('binning', {})
    
    # Load data
    if test_data_path:
        print(f"Loading data from: {test_data_path}")
        data = pd.read_pickle(test_data_path)
        raw_for_bins = data.copy()
        data = _validate_and_clean_data(data, list(model.nodes()), binning_config)
        _print_binning_intervals(raw_for_bins, list(model.nodes()), binning_config, "Binning intervals:")
        _print_binning_summary(data, binning_config, "Binning summary (after cleaning):")
        print(f"Loaded {len(data)} samples")
    else:
        print("Loading data from MongoDB...")
        load_dotenv()
        
        mongo_uri = config.get("mongodb", {}).get("uri") or os.getenv("MONGO_URI")
        db_name = config.get("mongodb", {}).get("database") or os.getenv("MONGO_DB")
        collection_name = config.get("mongodb", {}).get("collection") or os.getenv("MONGO_COLLECTION")
        
        if not all([mongo_uri, db_name, collection_name]):
            raise ValueError("MongoDB connection details missing")
        
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        pipeline = config['mongodb']['pipeline']
        cursor = collection.aggregate(pipeline)
        raw_data = pd.json_normalize(cursor)
        data = raw_data.copy()
        
        if config.get('features'):
            data = _apply_feature_selection(data, config)
        
        raw_for_bins = data.copy()
        data = _validate_and_clean_data(data, list(model.nodes()), binning_config)
        _print_binning_intervals(raw_for_bins, list(model.nodes()), binning_config, "Binning intervals:")
        _print_binning_summary(data, binning_config, "Binning summary (after cleaning):")
        print(f"Loaded {len(data)} samples")
    
    results = []
    hard_evidence, virtual_evidence = _build_hard_and_virtual_evidence(model, evidence)
    scalar_constraints = _extract_scalar_evidence_constraints(evidence)

    # Subgroup estimand: keep only rows matching scalar evidence values.
    if scalar_constraints:
        original_n = len(data)
        for var, value in scalar_constraints.items():
            if var in data.columns:
                data = data[data[var] == value]
        if len(data) == 0:
            raise ValueError(f"No samples remain after IPTW scalar-evidence filtering: {scalar_constraints}")
        print(
            f"IPTW sample conditioning: {original_n} -> {len(data)} "
            f"({len(data)/original_n*100:.1f}% retained)"
        )

    # Run IPTW analysis for each treatment-outcome pair
    for outcome in outcomes:
        if outcome not in data.columns or outcome not in model.nodes():
            print(f"\nSkipping {outcome}: not in data or model")
            continue
        
        print(f"\n{'='*80}")
        print(f"Outcome: {outcome}")
        print(f"{'='*80}")
        
        for treatment in treatments:
            if treatment not in data.columns or treatment not in model.nodes():
                print(f"\nSkipping treatment {treatment}: not in data or model")
                continue
            
            if treatment == outcome:
                continue
            
            print(f"\n{'-'*80}")
            print(f"Treatment: {treatment} → Outcome: {outcome}")
            print(f"{'-'*80}")
            
            try:
                # Estimate ATE using IPTW
                iptw_result = estimate_ate_with_iptw(
                    model, treatment, outcome, data,
                    confounders=None,  # Will use treatment's parents from DAG
                    stabilized=stabilized,
                    truncate=truncate,
                    min_ps=min_ps,
                    n_bootstrap=n_bootstrap,
                    hard_evidence=None,
                    virtual_evidence=None,
                )
                
                ate = iptw_result['ate']
                se = iptw_result['se']
                ci_lower = iptw_result['ci_lower']
                ci_upper = iptw_result['ci_upper']
                
                print(f"\nResults:")
                print(f"  Mean outcome (treated):    {iptw_result['mean_outcome_treated']:.4f}")
                print(f"  Mean outcome (control):    {iptw_result['mean_outcome_control']:.4f}")
                print(f"  ATE:                       {ate:+.4f} ({ate*100:+.2f} pp)")
                print(f"  Standard Error:            {se:.4f}")
                print(f"  95% CI:                    [{ci_lower:.4f}, {ci_upper:.4f}]")
                
                # Interpret effect
                if ci_lower > 0:
                    print(f"  ✓ Significant INCREASE in {outcome} (p < 0.05)")
                elif ci_upper < 0:
                    print(f"  ✓ Significant DECREASE in {outcome} (p < 0.05)")
                else:
                    print(f"  ~ No significant effect (CI includes 0)")
                
                # Print diagnostics
                print_iptw_diagnostics(iptw_result)
                
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "ate": ate,
                    "se": se,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "mean_treated": iptw_result['mean_outcome_treated'],
                    "mean_control": iptw_result['mean_outcome_control'],
                    "confounders": iptw_result['confounders'],
                    "weight_summary": iptw_result['weight_summary'],
                    "status": "success"
                }
                
            except Exception as e:
                print(f"\nERROR: {e}")
                import traceback
                traceback.print_exc()
                
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "ate": None,
                    "status": "error",
                    "error": str(e)
                }
            
            results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"IPTW ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total analyses: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
    
    # Summary table of significant effects
    significant = [r for r in results if r['status'] == 'success' and 
                  (r['ci_lower'] > 0 or r['ci_upper'] < 0)]
    
    if significant:
        print(f"\nSignificant Treatment Effects (p < 0.05):")
        print(f"  {'Treatment':<20} {'→':<3} {'Outcome':<20} {'ATE':>10} {'95% CI':>20}")
        print(f"  {'-'*20} {'-'*3} {'-'*20} {'-'*10} {'-'*20}")
        for r in significant:
            ci_str = f"[{r['ci_lower']:.3f}, {r['ci_upper']:.3f}]"
            print(f"  {r['treatment']:<20} {'→':<3} {r['outcome']:<20} {r['ate']:>+10.4f} {ci_str:>20}")
    else:
        print(f"\nNo significant treatment effects detected at p < 0.05")
    
    print(f"{'='*80}\n")
    
    return results


def validate_predictions(model, config_path: str, sample_size: int = 20, use_full_dataset: bool = False, plot_dir: str = None, test_data_path: str = None) -> None:
    """Load real patient data and compare actual vs predicted outcomes.
    
    Args:
        model: Fitted Bayesian Network
        config_path: Path to config YAML
        sample_size: Number of patients to display (default 20)
        use_full_dataset: If True, calculate metrics on entire dataset (default False)
        plot_dir: Directory to save distribution plots (optional)
        test_data_path: Path to test set pickle file (if None, loads from MongoDB)
    """
    
    print(f"\n{'='*80}")
    print(f"MODEL VALIDATION - ACTUAL vs PREDICTED")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    binning_config = config.get('binning', {})
    
    # Load data from test set pickle or MongoDB
    if test_data_path:
        print(f"Loading test set from: {test_data_path}")
        data = pd.read_pickle(test_data_path)
        print(f"Loaded {len(data)} test samples")
        # Keep a copy for raw display (already cleaned)
        raw_data = data.copy()
    else:
        print("Warning: No test data path provided, loading from MongoDB")
        print("This may contaminate validation with training data!")
        
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection details from config or environment
        mongo_uri = config.get("mongodb", {}).get("uri") or os.getenv("MONGO_URI")
        db_name = config.get("mongodb", {}).get("database") or os.getenv("MONGO_DB")
        collection_name = config.get("mongodb", {}).get("collection") or os.getenv("MONGO_COLLECTION")
        
        if not all([mongo_uri, db_name, collection_name]):
            raise ValueError("MongoDB connection details missing in config or environment variables")
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {db_name}.{collection_name}")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Execute aggregation pipeline and keep raw copy for readable history
        pipeline = config['mongodb']['pipeline']
        cursor = collection.aggregate(pipeline)
        raw_data = pd.json_normalize(cursor)
        data = raw_data.copy()
        
        # Apply feature selection then reuse fit_bn's cleaning to ensure
        # the same categorical encodings used when fitting the BN.
        if config.get('features'):
            # Apply same selection to raw and working copy
            raw_data = _apply_feature_selection(raw_data, config)
            data = _apply_feature_selection(data, config)

        # Validate and clean using the same routine as fit_bn (this will
        # restrict to DAG nodes and convert types/codes consistently).
        data = _validate_and_clean_data(data, list(model.nodes()), binning_config)

        # Align raw_data to the cleaned rows (cleaning may have dropped rows)
        raw_data = raw_data.loc[data.index]

    # Sample patients
    sample_data = data.sample(n=min(sample_size, len(data)), random_state=42)
    sample_raw = raw_data.loc[sample_data.index]
    
    print(f"Loaded {len(data)} patients, showing {len(sample_data)} samples\n")
    
    # Get inference engine
    inference = VariableElimination(model)
    
    # Outcomes to validate: adapt to new encoding where 'last_status'
    # may have been converted into three binary indicators: 'NED','AWD','DOD'.
    outcomes = ['metastasis', 'local_recurrence']
    # Prefer the single 'last_status' node if present in the model/data,
    # otherwise include binary indicators if available.
    model_nodes = list(model.nodes())
    if 'last_status' in model_nodes or 'last_status' in sample_data.columns:
        outcomes.append('last_status')
    else:
        for status_node in ['NED', 'AWD', 'DOD']:
            if status_node in model_nodes or status_node in sample_data.columns:
                outcomes.append(status_node)
    
    # Display header (include patient history and AJCC)
    print(f"{'ID':<5} {'History':<40} {'Surgery':>7} {'RT':>7} {'Chemo':>7} {'AJCC':>7} ", end="")
    for outcome in outcomes:
        if outcome in sample_data.columns:
            if outcome in ('metastasis', 'local_recurrence') or outcome in STATUS_NODES:
                print(f"{'Actual':>7} {'PredProb':>7} ", end="")
            else:
                print(f"{'Actual':>7} {'Pred':>7} ", end="")
    print("\n" + "-" * 80)
    
    # Process each patient
    for idx, ix in enumerate(sample_data.index, 1):
        row = sample_data.loc[ix]
        raw_row = sample_raw.loc[ix]
        print(f"{idx:<5} ", end="")

        # Patient history: prefer explicit history column from raw data, else build from raw tumor fields
        history_val = ""
        for hc in ['history', 'patient_history', 'clinical_history', 'notes', 'last_episode_summary']:
            if hc in raw_row.index and pd.notna(raw_row.get(hc)):
                history_val = str(raw_row.get(hc))
                break
        if not history_val:
            parts = []
            for fld in ['tumor_characteristics.who_diagnosis_code', 'tumor_characteristics.anatomic_region_grouping', 'tumor_characteristics.biopsy_grading', 'tumor_characteristics.initial_size']:
                if fld in raw_row.index and pd.notna(raw_row.get(fld)):
                    parts.append(str(raw_row.get(fld)))
            history_val = ' | '.join(parts)

        # Print full history field (no truncation)
        print(f"{history_val:<40} ", end="")

        # Print treatment values (from cleaned row)
        print(f"{int(row.get('surgery', 0)):>7} ", end="")
        print(f"{int(row.get('radiotherapy', 0)):>7} ", end="")
        print(f"{int(row.get('chemotherapy', 0)):>7} ", end="")
        
        # Calculate and print AJCC stage
        ajcc_stage = calculate_ajcc_stage(row)
        print(f"{ajcc_stage:>7} ", end="")
        
        # Build evidence from patient data (exclude outcomes)
        evidence = {}
        for col in row.index:
            if col not in outcomes and col in model.nodes():
                val = row[col]
                # Try to convert to int, if it fails, use the value as-is
                try:
                    evidence[col] = int(val)
                except (ValueError, TypeError):
                    # Keep as string or original type
                    evidence[col] = val
        
        # Predict each outcome
        for outcome in outcomes:
            if outcome not in sample_data.columns:
                continue

            actual_val = row[outcome]
            # Handle both numeric and string actual values
            if outcome in STATUS_NODES:
                # Status nodes are probabilities (0-1), keep as float
                try:
                    actual = float(actual_val)
                except (ValueError, TypeError):
                    actual = 0.0
            else:
                try:
                    actual = int(actual_val)
                except (ValueError, TypeError):
                    # If we have a single multi-class 'last_status', map known labels
                    if outcome == 'last_status':
                        status_map = {'AWD': 0, 'DOD': 1, 'NED': 2}
                        actual = status_map.get(actual_val, 0)
                    else:
                        # For binary indicators or other string-valued vars, try to map
                        try:
                            actual = int(pd.Categorical([actual_val]).codes[0])
                        except Exception:
                            actual = 0

            try:
                # Query marginal given evidence
                result = inference.query(variables=[outcome], evidence=evidence)
                state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))

                # If binary outcome, print probability of state=1
                if outcome in ('metastasis', 'local_recurrence') or outcome in STATUS_NODES or len(state_names) == 2:
                    # find index for state '1'
                    try:
                        idx_one = state_names.index(1)
                    except Exception:
                        try:
                            idx_one = state_names.index('1')
                        except Exception:
                            idx_one = 1
                    prob_one = float(result.values[idx_one])
                    # Display actual and predicted probability
                    if outcome in STATUS_NODES:
                        # For status probabilities, show both as floats
                        print(f"{actual:>7.3f} {prob_one:7.3f}", end=" ")
                    else:
                        # For binary outcomes, show actual as int and predicted as probability
                        print(f"{int(actual):>7} {prob_one:7.3f}", end=" ")
                else:
                    # Multi-class: use argmax index as predicted class code
                    predicted_idx = int(result.values.argmax())
                    match_symbol = "✓" if predicted_idx == actual else "✗"
                    print(f"{actual:>7} {predicted_idx:>7}{match_symbol}", end=" ")

            except Exception as e:
                print(f"{actual:>7} {'ERR':>7} ", end="")
        
        print()
    
    # Calculate accuracy on sample
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY (Sample)")
    print("=" * 80)
    
    for outcome in outcomes:
        if outcome not in sample_data.columns:
            continue
        
        # Skip accuracy calculation for probabilistic status nodes
        if outcome in STATUS_NODES:
            continue
            
        correct = 0
        total = 0
        
        for _, row in sample_data.iterrows():
            evidence = {}
            for col in row.index:
                if col not in outcomes and col in model.nodes():
                    val = row[col]
                    # Try to convert to int, if it fails, use the value as-is
                    try:
                        evidence[col] = int(val)
                    except (ValueError, TypeError):
                        evidence[col] = val
            
            try:
                result = inference.query(variables=[outcome], evidence=evidence)
                state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                predicted_idx = int(result.values.argmax())
                
                actual_val = row[outcome]
                # Handle both numeric and string values
                try:
                    actual = int(actual_val)
                except (ValueError, TypeError):
                    if outcome == 'last_status':
                        status_map = {'AWD': 0, 'DOD': 1, 'NED': 2}
                        actual = status_map.get(actual_val, 0)
                    elif outcome in STATUS_NODES:
                        # Binary status indicators
                        try:
                            actual = int(actual_val)
                        except (ValueError, TypeError):
                            actual = 1 if str(actual_val).strip() in ("1", "True", "true", "Y", "y") else 0
                    else:
                        actual = 0
                
                if predicted_idx == actual:
                    correct += 1
                total += 1
            except:
                pass
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"{outcome}: {correct}/{total} correct ({accuracy:.1f}%)")
    
    print()
    
    # Evaluate on full dataset if requested
    if use_full_dataset:
        print("\n" + "=" * 80)
        print("FULL DATASET EVALUATION")
        print("=" * 80)
        print(f"Evaluating on {len(data)} patients...\n")
        
        # Get inference engine
        inference = VariableElimination(model)
        
        # Store predictions and actuals for each outcome
        predictions = {outcome: [] for outcome in outcomes if outcome not in STATUS_NODES}
        actuals = {outcome: [] for outcome in outcomes if outcome not in STATUS_NODES}
        probabilities = {outcome: [] for outcome in outcomes}  # For all outcomes including status nodes
        ajcc_dod_predictions = []  # AJCC-based DOD risk predictions
        
        for idx, row in data.iterrows():
            # Calculate AJCC stage and predicted DOD risk
            ajcc_stage = calculate_ajcc_stage(row)
            ajcc_dod_risk = get_ajcc_mortality_risk(ajcc_stage)
            ajcc_dod_predictions.append(ajcc_dod_risk)
            
            evidence = {}
            for col in row.index:
                if col not in outcomes and col in model.nodes():
                    try:
                        evidence[col] = int(row[col])
                    except (ValueError, TypeError):
                        evidence[col] = row[col]
            
            for outcome in outcomes:
                if outcome not in data.columns:
                    continue
                
                try:
                    result = inference.query(variables=[outcome], evidence=evidence)
                    
                    if outcome in STATUS_NODES:
                        # For status probabilities, just store predicted probability
                        state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                        try:
                            idx_one = state_names.index(1)
                        except:
                            idx_one = 1
                        prob = float(result.values[idx_one])
                        probabilities[outcome].append((float(row[outcome]), prob))
                    else:
                        # For binary outcomes, store predictions
                        predicted_idx = int(result.values.argmax())
                        actual_val = int(row[outcome])
                        predictions[outcome].append(predicted_idx)
                        actuals[outcome].append(actual_val)
                        
                        # Also store probabilities
                        state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                        try:
                            idx_one = state_names.index(1)
                        except:
                            idx_one = 1
                        prob = float(result.values[idx_one])
                        probabilities[outcome].append((actual_val, prob))
                except Exception as e:
                    pass
        
        # Calculate and display metrics for binary outcomes
        for outcome in outcomes:
            if outcome in STATUS_NODES:
                continue
            
            if outcome not in predictions or len(predictions[outcome]) == 0:
                continue
            
            y_true = actuals[outcome]
            y_pred = predictions[outcome]
            
            acc = accuracy_score(y_true, y_pred)
            
            # Handle cases where outcome class is not present
            try:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
            except:
                prec = rec = f1 = 0.0
                cm = None
            
            print(f"\n{outcome.upper()}:")
            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall:    {rec:.3f}")
            print(f"  F1 Score:  {f1:.3f}")
            
            if cm is not None:
                print(f"  Confusion Matrix:")
                print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
                print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
            
            # Add probability-aware metrics
            if outcome in probabilities and len(probabilities[outcome]) > 0:
                outcome_actuals = np.array([a for a, p in probabilities[outcome]])
                outcome_probs = np.array([p for a, p in probabilities[outcome]])
                
                prob_metrics = calculate_probability_metrics(outcome_actuals, outcome_probs)
                
                print(f"\n  Probability-Aware Metrics:")
                print(f"    Log Loss:      {prob_metrics['log_loss']:.4f}")
                print(f"    Brier Score:   {prob_metrics['brier_score']:.4f}")
                print(f"    AUROC:         {prob_metrics['auroc']:.4f}")
                print(f"    AUPRC:         {prob_metrics['auprc']:.4f}")
                
                # Calculate calibration metrics
                calib_metrics = calculate_calibration_metrics(outcome_actuals, outcome_probs)
                
                print(f"\n  Calibration Metrics:")
                print(f"    ECE:               {calib_metrics['ece']:.4f}")
                print(f"    MCE:               {calib_metrics['mce']:.4f}")
                print(f"    Calib. Slope:      {calib_metrics['calibration_slope']:.3f} (ideal=1.0)")
                print(f"    Calib. Intercept:  {calib_metrics['calibration_intercept']:.3f} (ideal=0.0)")
                
                # Save calibration plot if plot directory provided
                if plot_dir:
                    plot_path = Path(plot_dir)
                    plot_path.mkdir(parents=True, exist_ok=True)
                    calib_plot_path = plot_path / f"{outcome}_calibration.png"
                    plot_calibration_curve(
                        outcome_actuals, outcome_probs, 
                        f"{outcome.upper()} - Calibration Curve",
                        str(calib_plot_path),
                        calib_metrics
                    )
                    print(f"    Calibration plot saved: {calib_plot_path}")
        
        # AJCC vs BN comparison for DOD
        if 'DOD' in probabilities and len(ajcc_dod_predictions) == len(probabilities['DOD']):
            print(f"\n{'='*80}")
            print(f"AJCC STAGING vs BAYESIAN NETWORK COMPARISON (DOD Prediction)")
            print(f"{'='*80}")
            
            # Extract actual and predicted from tuples
            dod_actual = np.array([actual for actual, pred in probabilities['DOD']])
            bn_dod_probs = np.array([pred for actual, pred in probabilities['DOD']])
            ajcc_probs = np.array(ajcc_dod_predictions[:len(probabilities['DOD'])])
            
            # Diagnostic: Check actual value distribution
            print(f"\nDiagnostic - DOD actual values:")
            print(f"  Unique values: {np.unique(dod_actual)}")
            print(f"  Value counts: 0={np.sum(dod_actual==0)}, 1={np.sum(dod_actual==1)}")
            print(f"  Min={dod_actual.min()}, Max={dod_actual.max()}, Mean={dod_actual.mean():.3f}")
            if len(np.unique(dod_actual)) > 2:
                print(f"  WARNING: DOD has more than 2 unique values - should be binary (0 or 1)!")
                print(f"  All unique values: {sorted(np.unique(dod_actual))}")
            print()
            
            # Mean absolute error for both methods
            bn_mae = np.mean(np.abs(bn_dod_probs - dod_actual))
            ajcc_mae = np.mean(np.abs(ajcc_probs - dod_actual))
            
            # Mean squared error
            bn_mse = np.mean((bn_dod_probs - dod_actual) ** 2)
            ajcc_mse = np.mean((ajcc_probs - dod_actual) ** 2)
            
            # Correlation with actual outcomes
            bn_corr = np.corrcoef(bn_dod_probs, dod_actual)[0, 1]
            ajcc_corr = np.corrcoef(ajcc_probs, dod_actual)[0, 1]
            
            print(f"\nMean Absolute Error:")
            print(f"  Bayesian Network: {bn_mae:.4f}")
            print(f"  AJCC Staging:     {ajcc_mae:.4f}")
            print(f"  {'✓ BN wins' if bn_mae < ajcc_mae else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nMean Squared Error:")
            print(f"  Bayesian Network: {bn_mse:.4f}")
            print(f"  AJCC Staging:     {ajcc_mse:.4f}")
            print(f"  {'✓ BN wins' if bn_mse < ajcc_mse else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nCorrelation with Actual DOD:")
            print(f"  Bayesian Network: {bn_corr:.4f}")
            print(f"  AJCC Staging:     {ajcc_corr:.4f}")
            print(f"  {'✓ BN wins' if bn_corr > ajcc_corr else '✓ AJCC wins'} (higher is better)")
            
            # Add probability-aware metrics for both methods
            bn_prob_metrics = calculate_probability_metrics(dod_actual, bn_dod_probs)
            ajcc_prob_metrics = calculate_probability_metrics(dod_actual, ajcc_probs)
            
            print(f"\nLog Loss (Cross-Entropy):")
            print(f"  Bayesian Network: {bn_prob_metrics['log_loss']:.4f}")
            print(f"  AJCC Staging:     {ajcc_prob_metrics['log_loss']:.4f}")
            print(f"  {'✓ BN wins' if bn_prob_metrics['log_loss'] < ajcc_prob_metrics['log_loss'] else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nBrier Score:")
            print(f"  Bayesian Network: {bn_prob_metrics['brier_score']:.4f}")
            print(f"  AJCC Staging:     {ajcc_prob_metrics['brier_score']:.4f}")
            print(f"  {'✓ BN wins' if bn_prob_metrics['brier_score'] < ajcc_prob_metrics['brier_score'] else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nAUROC (Discrimination):")
            print(f"  Bayesian Network: {bn_prob_metrics['auroc']:.4f}")
            print(f"  AJCC Staging:     {ajcc_prob_metrics['auroc']:.4f}")
            print(f"  {'✓ BN wins' if bn_prob_metrics['auroc'] > ajcc_prob_metrics['auroc'] else '✓ AJCC wins'} (higher is better)")
            
            print(f"\nAUPRC (Precision-Recall):")
            print(f"  Bayesian Network: {bn_prob_metrics['auprc']:.4f}")
            print(f"  AJCC Staging:     {ajcc_prob_metrics['auprc']:.4f}")
            print(f"  {'✓ BN wins' if bn_prob_metrics['auprc'] > ajcc_prob_metrics['auprc'] else '✓ AJCC wins'} (higher is better)")
            
            # Calibration metrics
            bn_calib = calculate_calibration_metrics(dod_actual, bn_dod_probs)
            ajcc_calib = calculate_calibration_metrics(dod_actual, ajcc_probs)
            
            print(f"\nCalibration Metrics:")
            print(f"  Expected Calibration Error (ECE):")
            print(f"    Bayesian Network: {bn_calib['ece']:.4f}")
            print(f"    AJCC Staging:     {ajcc_calib['ece']:.4f}")
            print(f"    {'✓ BN wins' if bn_calib['ece'] < ajcc_calib['ece'] else '✓ AJCC wins'} (lower is better)")
            
            print(f"  Maximum Calibration Error (MCE):")
            print(f"    Bayesian Network: {bn_calib['mce']:.4f}")
            print(f"    AJCC Staging:     {ajcc_calib['mce']:.4f}")
            print(f"    {'✓ BN wins' if bn_calib['mce'] < ajcc_calib['mce'] else '✓ AJCC wins'} (lower is better)")
            
            print(f"  Calibration Slope (ideal=1.0):")
            print(f"    Bayesian Network: {bn_calib['calibration_slope']:.3f}")
            print(f"    AJCC Staging:     {ajcc_calib['calibration_slope']:.3f}")
            
            print(f"  Calibration Intercept (ideal=0.0):")
            print(f"    Bayesian Network: {bn_calib['calibration_intercept']:.3f}")
            print(f"    AJCC Staging:     {ajcc_calib['calibration_intercept']:.3f}")
            
            # Count stage distribution
            stage_counts = {}
            for idx, row in data.iterrows():
                stage = calculate_ajcc_stage(row)
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            print(f"\nAJCC Stage Distribution in Test Set:")
            for stage in sorted(stage_counts.keys()):
                count = stage_counts[stage]
                pct = 100.0 * count / len(data)
                expected_mort = get_ajcc_mortality_risk(stage)
                print(f"  Stage {stage:8s}: {count:4d} patients ({pct:5.1f}%), Expected DOD risk: {expected_mort:.1%}")
            
            print(f"\n{'='*80}")
        
        # Plot distributions if directory provided
        if plot_dir:
            plot_path = Path(plot_dir)
            plot_path.mkdir(parents=True, exist_ok=True)
            
            # AJCC vs BN comparison plot for DOD
            if 'DOD' in probabilities and len(ajcc_dod_predictions) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                
                # Extract actual and predicted from tuples
                dod_actual = np.array([actual for actual, pred in probabilities['DOD']])
                bn_dod_probs = np.array([pred for actual, pred in probabilities['DOD']])
                ajcc_probs = np.array(ajcc_dod_predictions[:len(probabilities['DOD'])])
                
                # Plot 1: BN - Distribution by actual class
                actual_0_bn = bn_dod_probs[dod_actual == 0]
                actual_1_bn = bn_dod_probs[dod_actual == 1]
                
                axes[0, 0].hist(actual_0_bn, bins=20, alpha=0.6, label=f'DOD=0 (n={len(actual_0_bn)})', color='blue')
                axes[0, 0].hist(actual_1_bn, bins=20, alpha=0.6, label=f'DOD=1 (n={len(actual_1_bn)})', color='red')
                axes[0, 0].set_xlabel('BN Predicted P(DOD=1)')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Bayesian Network - Predicted Probabilities by Actual Class')
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
                
                # Plot 2: BN - Calibration curve
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_counts_bn = []
                bin_means_bn = []
                
                for i in range(len(bins)-1):
                    mask = (bn_dod_probs >= bins[i]) & (bn_dod_probs < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts_bn.append(mask.sum())
                        bin_means_bn.append(dod_actual[mask].mean())
                    else:
                        bin_counts_bn.append(0)
                        bin_means_bn.append(np.nan)
                
                axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
                valid_bn = ~np.isnan(bin_means_bn)
                axes[0, 1].scatter(bin_centers[valid_bn], np.array(bin_means_bn)[valid_bn], 
                                  s=[c*5 for c, v in zip(bin_counts_bn, valid_bn) if v], 
                                  alpha=0.6, color='blue', label='BN observed')
                axes[0, 1].plot(bin_centers[valid_bn], np.array(bin_means_bn)[valid_bn], 
                               'b-', alpha=0.5, linewidth=2)
                axes[0, 1].set_xlabel('BN Predicted P(DOD=1)')
                axes[0, 1].set_ylabel('Actual fraction of DOD=1')
                axes[0, 1].set_title(f'Bayesian Network Calibration\nMAE: {np.mean(np.abs(bn_dod_probs - dod_actual)):.4f}')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
                axes[0, 1].set_xlim(0, 1)
                axes[0, 1].set_ylim(0, 1)
                
                # Plot 3: AJCC - Distribution by actual class
                actual_0_ajcc = ajcc_probs[dod_actual == 0]
                actual_1_ajcc = ajcc_probs[dod_actual == 1]
                
                axes[1, 0].hist(actual_0_ajcc, bins=20, alpha=0.6, label=f'DOD=0 (n={len(actual_0_ajcc)})', color='blue')
                axes[1, 0].hist(actual_1_ajcc, bins=20, alpha=0.6, label=f'DOD=1 (n={len(actual_1_ajcc)})', color='red')
                axes[1, 0].set_xlabel('AJCC Predicted P(DOD=1)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('AJCC Staging - Predicted Probabilities by Actual Class')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
                
                # Plot 4: AJCC - Calibration curve
                bin_counts_ajcc = []
                bin_means_ajcc = []
                
                for i in range(len(bins)-1):
                    mask = (ajcc_probs >= bins[i]) & (ajcc_probs < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts_ajcc.append(mask.sum())
                        bin_means_ajcc.append(dod_actual[mask].mean())
                    else:
                        bin_counts_ajcc.append(0)
                        bin_means_ajcc.append(np.nan)
                
                axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
                valid_ajcc = ~np.isnan(bin_means_ajcc)
                axes[1, 1].scatter(bin_centers[valid_ajcc], np.array(bin_means_ajcc)[valid_ajcc], 
                                  s=[c*5 for c, v in zip(bin_counts_ajcc, valid_ajcc) if v], 
                                  alpha=0.6, color='orange', label='AJCC observed')
                axes[1, 1].plot(bin_centers[valid_ajcc], np.array(bin_means_ajcc)[valid_ajcc], 
                               'orange', alpha=0.5, linewidth=2, linestyle='-')
                axes[1, 1].set_xlabel('AJCC Predicted P(DOD=1)')
                axes[1, 1].set_ylabel('Actual fraction of DOD=1')
                axes[1, 1].set_title(f'AJCC Staging Calibration\nMAE: {np.mean(np.abs(ajcc_probs - dod_actual)):.4f}')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                
                plt.tight_layout()
                comparison_plot_path = plot_path / 'ajcc_vs_bn_comparison.png'
                plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\nAJCC comparison plot saved to: {comparison_plot_path}")
                
                # Calculate calibration metrics for plotting (may not have been calculated yet)
                bn_calib_for_plot = calculate_calibration_metrics(dod_actual, bn_dod_probs)
                ajcc_calib_for_plot = calculate_calibration_metrics(dod_actual, ajcc_probs)
                
                # Save individual calibration curves with full metrics
                bn_calib_path = plot_path / 'DOD_BN_calibration.png'
                plot_calibration_curve(
                    dod_actual, bn_dod_probs,
                    "Bayesian Network - DOD Calibration",
                    str(bn_calib_path),
                    bn_calib_for_plot
                )
                print(f"BN calibration plot saved to: {bn_calib_path}")
                
                ajcc_calib_path = plot_path / 'DOD_AJCC_calibration.png'
                plot_calibration_curve(
                    dod_actual, ajcc_probs,
                    "AJCC Staging - DOD Calibration",
                    str(ajcc_calib_path),
                    ajcc_calib_for_plot
                )
                print(f"AJCC calibration plot saved to: {ajcc_calib_path}")
            print(f"\nSaving distribution plots to: {plot_dir}")
            
            for outcome in outcomes:
                if outcome not in probabilities or len(probabilities[outcome]) == 0:
                    continue
                
                actual_vals, pred_probs = zip(*probabilities[outcome])
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle(f'{outcome} - Predicted vs Actual', fontsize=14, fontweight='bold')
                
                # Plot 1: Distribution of predicted probabilities by actual class
                actual_0 = [p for a, p in probabilities[outcome] if a == 0]
                actual_1 = [p for a, p in probabilities[outcome] if a == 1]
                
                axes[0].hist(actual_0, bins=20, alpha=0.6, label=f'{outcome}=0 (n={len(actual_0)})', color='blue')
                axes[0].hist(actual_1, bins=20, alpha=0.6, label=f'{outcome}=1 (n={len(actual_1)})', color='red')
                axes[0].set_xlabel('Predicted P(outcome=1)')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Predicted Probabilities by Actual Class')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # Plot 2: Calibration curve (binned)
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_counts = []
                bin_means = []
                
                for i in range(len(bins)-1):
                    mask = (np.array(pred_probs) >= bins[i]) & (np.array(pred_probs) < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts.append(mask.sum())
                        bin_means.append(np.array(actual_vals)[mask].mean())
                    else:
                        bin_counts.append(0)
                        bin_means.append(0)
                
                axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[1].scatter(bin_centers, bin_means, s=[c*5 for c in bin_counts], alpha=0.6, label='Observed')
                axes[1].set_xlabel('Predicted P(outcome=1)')
                axes[1].set_ylabel('Actual fraction of positive')
                axes[1].set_title('Calibration Curve')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                axes[1].set_xlim(0, 1)
                axes[1].set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(plot_path / f'{outcome}_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {outcome}_distribution.png")
    
    print()


def run_regime_comparisons(
    model, 
    outcomes: list[str], 
    treatments: list[str],
    comparisons: list[Tuple[Dict, Dict, str]],
    evidence: dict = None
) -> list[dict]:
    """Run treatment regime comparisons (e.g., surgery+RT vs surgery only)."""
    
    hard_evidence, virtual_evidence = _build_hard_and_virtual_evidence(model, evidence)

    if evidence:
        print(f"\nConditioning all comparisons on: {evidence}\n")
    results = []
    
    print(f"\n{'='*80}")
    print(f"TREATMENT REGIME COMPARISON ANALYSIS")
    print(f"{'='*80}\n")
    
    # Print DAG structure for outcomes
    print(f"DAG STRUCTURE CHECK:")
    print(f"{'-'*80}")
    for outcome in outcomes:
        if outcome in model.nodes():
            parents = list(model.get_parents(outcome))
            print(f"{outcome} <- {parents if parents else 'NO PARENTS'}")
    print(f"{'-'*80}\n")
    
    for outcome in outcomes:
        print(f"\n{'='*80}")
        print(f"Outcome: {outcome}")
        print(f"{'='*80}\n")
        
        # Print outcome variable information
        if outcome in model.nodes():
            cpd = model.get_cpds(outcome)
            state_names = cpd.state_names.get(outcome, list(range(len(cpd.get_values()))))
            print(f"Variable '{outcome}' has {len(state_names)} states: {state_names}")
            print(f"State mapping: {dict(enumerate(state_names))}")
            
            # Clinical interpretation for last_status
            # Note: pandas converts strings to codes alphabetically: AWD->0, DOD->1, NED->2
            if outcome == "last_status" and len(state_names) == 3:
                print(f"Clinical meaning (alphabetical encoding):")
                print(f"  0 = AWD (Alive With Disease)")
                print(f"  1 = DOD (Dead of Disease)")
                print(f"  2 = NED (No Evidence of Disease)")
            print()
        
        # For binary outcomes (metastasis, local_recurrence), we measure P(outcome=1)
        # For multi-class (last_status), we'll show distribution across all states
        
        for regime_a, regime_b, label in comparisons:
            regime_a_str = " + ".join([f"{k}={v}" for k, v in regime_a.items()])
            regime_b_str = " + ".join([f"{k}={v}" for k, v in regime_b.items()])
            
            print(f"\n{'-'*80}")
            print(f"Comparison: {label}")
            print(f"  Regime A: {regime_a_str}")
            print(f"  Regime B: {regime_b_str}")
            print(f"{'-'*80}")
            
            try:
                # Compute P(outcome | do(regime_a), evidence)
                do_model_a = do_intervene(model, regime_a)
                infer_a = VariableElimination(do_model_a)
                result_a = infer_a.query(
                    variables=[outcome],
                    evidence=hard_evidence or None,
                    virtual_evidence=virtual_evidence or None,
                    show_progress=False,
                )
                state_names_a = result_a.state_names.get(outcome, list(range(result_a.cardinality[0])))
                
                # Compute P(outcome | do(regime_b), evidence)
                do_model_b = do_intervene(model, regime_b)
                infer_b = VariableElimination(do_model_b)
                result_b = infer_b.query(
                    variables=[outcome],
                    evidence=hard_evidence or None,
                    virtual_evidence=virtual_evidence or None,
                    show_progress=False,
                )
                state_names_b = result_b.state_names.get(outcome, list(range(result_b.cardinality[0])))
                
                # DIAGNOSTIC: Check marginal probabilities
                print(f"\n  DIAGNOSTIC - Marginal P({outcome}) in original model" + (f" | {evidence}" if evidence else "") + ":")
                infer_orig = VariableElimination(model)
                result_orig = infer_orig.query(
                    variables=[outcome],
                    evidence=hard_evidence or None,
                    virtual_evidence=virtual_evidence or None,
                    show_progress=False,
                )
                for state, prob in zip(state_names_a, result_orig.values):
                    print(f"    P({outcome}={state}): {prob:.4f}")
                
                # Check if binary or multi-class
                if len(state_names_a) == 2:
                    # Binary outcome: report P(outcome=1)
                    outcome_state = 1
                    outcome_state_idx = state_names_a.index(outcome_state) if outcome_state in state_names_a else 1
                    prob_a = float(result_a.values[outcome_state_idx])
                    prob_b = float(result_b.values[outcome_state_idx])
                    
                    ate = prob_b - prob_a
                    
                    result = {
                        "outcome": outcome,
                        "comparison": label,
                        "regime_a": regime_a,
                        "regime_b": regime_b,
                        "regime_a_str": regime_a_str,
                        "regime_b_str": regime_b_str,
                        "outcome_type": "binary",
                        "risk_a": prob_a,
                        "risk_b": prob_b,
                        "ate_pp": ate * 100,
                        "relative_change": (ate / prob_a * 100) if prob_a > 0 else None,
                        "outcome_state": outcome_state,
                        "status": "success"
                    }
                    
                    print(f"\n  Binary Outcome - P({outcome}=1):")
                    print(f"    Risk under A: {prob_a:.4f} ({prob_a*100:.2f}%)")
                    print(f"    Risk under B: {prob_b:.4f} ({prob_b*100:.2f}%)")
                    print(f"    ATE (risk diff): {ate:+.4f} ({ate*100:+.2f} pp)")
                    
                    if prob_a > 0:
                        rel_change = (ate / prob_a) * 100
                        print(f"    Relative change: {rel_change:+.2f}%")
                    
                    if ate > 0.01:
                        print(f"    ↑ Regime B INCREASES risk by {abs(ate*100):.2f} percentage points")
                    elif ate < -0.01:
                        print(f"    ↓ Regime B DECREASES risk by {abs(ate*100):.2f} percentage points")
                    else:
                        print(f"    → No substantial effect")
                else:
                    # Multi-class outcome: report full distribution
                    result = {
                        "outcome": outcome,
                        "comparison": label,
                        "regime_a": regime_a,
                        "regime_b": regime_b,
                        "regime_a_str": regime_a_str,
                        "regime_b_str": regime_b_str,
                        "outcome_type": "multiclass",
                        "distribution_a": {str(state): float(prob) for state, prob in zip(state_names_a, result_a.values)},
                        "distribution_b": {str(state): float(prob) for state, prob in zip(state_names_b, result_b.values)},
                        "status": "success"
                    }
                    
                    print(f"\n  Multi-class Outcome - Distribution:")
                    print(f"\n    Regime A:")
                    for state, prob in zip(state_names_a, result_a.values):
                        print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                    
                    print(f"\n    Regime B:")
                    for state, prob in zip(state_names_b, result_b.values):
                        print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                    
                    print(f"\n    Changes (B - A):")
                    for state in state_names_a:
                        idx = state_names_a.index(state)
                        change = result_b.values[idx] - result_a.values[idx]
                        print(f"      Δ P({outcome}={state}): {change:+.4f} ({change*100:+.2f} pp)")
                
            except Exception as e:
                result = {
                    "outcome": outcome,
                    "comparison": label,
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "regime_a_str": regime_a_str,
                    "regime_b_str": regime_b_str,
                    "status": "error",
                    "error": str(e)
                }
                print(f"\n  ERROR: {e}")
            
            results.append(result)
    
    return results


def print_regime_table(results: list[dict]) -> None:
    """Print results in a clinical-style table format."""
    print(f"\n{'='*80}")
    print(f"TREATMENT REGIME EFFECTS TABLE")
    print(f"{'='*80}\n")
    
    # Group by outcome
    outcomes = {}
    for r in results:
        if r['status'] == 'success':
            outcome = r['outcome']
            if outcome not in outcomes:
                outcomes[outcome] = []
            outcomes[outcome].append(r)
    
    for outcome, outcome_results in outcomes.items():
        print(f"\n{'─'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'─'*80}")
        
        # Check if binary or multi-class
        if outcome_results and outcome_results[0].get('outcome_type') == 'binary':
            # Binary outcome table
            print(f"\n{'Comparison':<40} {'Risk A':>8} {'Risk B':>8} {'ATE(pp)':>9} {'RelChg%':>9} {'Interpretation':<20}")
            print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*20}")
            
            for r in outcome_results:
                comparison = r['comparison'][:39]
                risk_a = f"{r['risk_a']:.4f}" if r.get('risk_a') is not None else "N/A"
                risk_b = f"{r['risk_b']:.4f}" if r.get('risk_b') is not None else "N/A"
                ate = f"{r['ate_pp']:+.2f}" if r.get('ate_pp') is not None else "N/A"
                rel = f"{r['relative_change']:+.1f}" if r.get('relative_change') is not None else "N/A"
                
                if r.get('ate_pp') and abs(r['ate_pp']) > 1:
                    if r['ate_pp'] > 0:
                        interp = "↑ Increases risk"
                    else:
                        interp = "↓ Decreases risk"
                else:
                    interp = "→ No effect"
                
                print(f"{comparison:<40} {risk_a:>8} {risk_b:>8} {ate:>9} {rel:>9} {interp:<20}")
        else:
            # Multi-class outcome - show distributions
            for r in outcome_results:
                print(f"\n  Comparison: {r['comparison']}")
                print(f"    Regime A: {r['regime_a_str']}")
                dist_a = r.get('distribution_a', {})
                for state, prob in dist_a.items():
                    print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                
                print(f"    Regime B: {r['regime_b_str']}")
                dist_b = r.get('distribution_b', {})
                for state, prob in dist_b.items():
                    print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                
                print(f"    Changes (B - A):")
                for state in dist_a.keys():
                    change = dist_b.get(state, 0) - dist_a.get(state, 0)
                    print(f"      Δ P({outcome}={state}): {change:+.4f} ({change*100:+.2f} pp)")
        
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ATE using do-interventions.")
    parser.add_argument("--model", required=True, help="Path to BN pickle.")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", action="store_true", help="Single treatment-outcome calculation.")
    mode_group.add_argument("--multiple", action="store_true", help="Multiple treatment-outcome calculations.")
    mode_group.add_argument("--regimes", action="store_true", help="Compare treatment regimes.")
    mode_group.add_argument("--validate", action="store_true", help="Validate model predictions vs actual data.")
    mode_group.add_argument("--interpret", action="store_true", help="Model interpretability: Markov blanket and sensitivity analysis.")
    mode_group.add_argument("--iptw", action="store_true", help="IPTW-based treatment effect analysis (addresses confounding by indication).")
    
    # Single mode arguments
    parser.add_argument("--treatment", help="Treatment node name (required for --single).")
    parser.add_argument("--outcome", help="Outcome node name (required for --single).")
    parser.add_argument("--treatment-values", nargs=2, help="Control and treated values (required for --single).")
    parser.add_argument("--outcome-state", help="Outcome state to query (required for --single).")
    parser.add_argument("--evidence", nargs="*", default=[], help="Optional evidence as key=value pairs.")
    
    # Multiple mode arguments
    parser.add_argument("--treatments", nargs="+", help="List of treatment variables for --multiple mode.")
    parser.add_argument("--outcomes", nargs="+", help="List of outcome variables for --multiple and --regimes mode.")
    parser.add_argument("--output", help="Path to save results JSON (optional).")
    
    # Regime mode arguments
    parser.add_argument("--regime-comparisons", help="Path to JSON file defining regime comparisons.")
    
    # IPTW mode arguments
    parser.add_argument("--gcomputation", action="store_true", help="Use g-computation/standardization instead of IPTW for regime analysis (leverages BN structure, better for small samples).")
    parser.add_argument("--stratify-by", nargs="+", help="List of baseline variables to stratify results by (e.g., age biopsy_grading). Only for g-computation mode.")
    parser.add_argument("--no-stabilized", action="store_true", help="Disable stabilized IPTW weights (default: stabilized weights enabled).")
    parser.add_argument(
        "--truncate",
        type=float,
        default=None,
        help="Propensity truncation floor. Default is mode-specific: 0.05 (single treatment IPTW), 0.005 (regime IPTW).",
    )
    parser.add_argument(
        "--regime-max-weight",
        type=float,
        default=20.0,
        help="Max cap for regime IPTW weights (default: 20.0, set <=0 to disable).",
    )
    parser.add_argument(
        "--min-ps",
        type=float,
        default=0.0,
        help="Drop observations with propensity score below this threshold (default: 0 = disabled).",
    )
    parser.add_argument(
        "--bootstrap-samples",
        "--bootstrap-sample",
        dest="bootstrap_samples",
        type=int,
        default=500,
        help="Number of bootstrap resamples for IPTW SE/CI (default: 500).",
    )
    
    # Validation mode arguments
    parser.add_argument("--config", help="Path to config YAML (required for --validate and --iptw).")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of patients to display in sample table.")
    parser.add_argument("--full-dataset", action="store_true", help="Evaluate metrics on entire dataset (not just sample).")
    parser.add_argument("--plot-dir", help="Directory to save distribution plots (e.g., output/validation_plots).")
    parser.add_argument("--test-data-path", help="Path to test set pickle file (for uncontaminated validation).")

    args = parser.parse_args()
    if args.bootstrap_samples < 2:
        parser.error("--bootstrap-samples must be >= 2")
    if args.truncate is not None and not (0 <= args.truncate < 0.5):
        parser.error("--truncate must be in [0, 0.5)")
    if args.min_ps < 0 or args.min_ps >= 1:
        parser.error("--min-ps must be in [0, 1)")

    model = load_bayesian_network(args.model)
    
    if args.single:
        # Validate single mode arguments
        if not all([args.treatment, args.outcome, args.treatment_values, args.outcome_state]):
            parser.error("--single mode requires --treatment, --outcome, --treatment-values, and --outcome-state")
        
        # Single treatment-outcome calculation
        raw_evidence = _parse_evidence(args.evidence)
        # Load config if available for preprocessing
        config = _load_config(args.config) if args.config else {}
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        ate = average_treatment_effect(
            model,
            args.treatment,
            args.outcome,
            tuple(_parse_value(val) for val in args.treatment_values),
            _parse_value(args.outcome_state),
            evidence=evidence,
        )
        print(f"ATE({args.treatment} -> {args.outcome}): {ate:.6f}")
    
    elif args.multiple:
        # Validate multiple mode arguments
        if not args.treatments or not args.outcomes:
            parser.error("--multiple mode requires --treatments and --outcomes")
        
        # Load config if provided (for evidence preprocessing)
        config = {}
        if args.config:
            config = _load_config(args.config)
        
        # Adapt outcomes: if last_status is requested but not in model, use NED/AWD/DOD instead
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                # Replace with binary status nodes if they exist in the model
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Parse evidence if provided
        raw_evidence = _parse_evidence(args.evidence) if args.evidence else None
        # Preprocess evidence to match BN discretization
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        # Multiple treatment-outcome calculations
        results = run_multiple_treatment_effects(model, adapted_outcomes, args.treatments, evidence)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total calculations: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"{'='*80}\n")
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}\n")
    
    elif args.regimes:
        # Validate regime mode arguments
        if not args.outcomes:
            parser.error("--regimes mode requires --outcomes")
        
        # Load config if provided (for evidence preprocessing)
        config = {}
        if args.config:
            config = _load_config(args.config)
        
        # Adapt outcomes: if last_status is requested but not in model, use NED/AWD/DOD instead
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                # Replace with binary status nodes if they exist in the model
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Parse evidence if provided
        raw_evidence = _parse_evidence(args.evidence) if args.evidence else None
        # Preprocess evidence to match BN discretization
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        # Load regime comparisons from file or use defaults
        if args.regime_comparisons:
            with open(args.regime_comparisons, 'r') as f:
                regime_config = json.load(f)
                comparisons = regime_config['comparisons']  # Pass full dict including sensitivity_conditions
        else:
            # Default clinically meaningful comparisons (no sensitivity analysis)
            comparisons = [
                {
                    'regime_a': {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0},
                    'regime_b': {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1},
                    'label': "Add RT to surgery"
                },
                {
                    'regime_a': {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1},
                    'regime_b': {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1},
                    'label': "Add chemo to surgery+RT"
                },
                {
                    'regime_a': {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0},
                    'regime_b': {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1},
                    'label': "Surgery only vs full treatment"
                },
            ]
        
        # Run regime comparisons
        treatments = ["surgery", "chemotherapy", "radiotherapy"]
        results = run_regime_comparisons(model, adapted_outcomes, treatments, comparisons, evidence)
        
        # Print tabular output
        print_regime_table(results)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total comparisons: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"{'='*80}\n")
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}\n")
    
    elif args.interpret:
        # Interpret mode - model interpretability analysis
        if not args.outcomes:
            parser.error("--interpret mode requires --outcomes")
        
        # Adapt outcomes
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Markov blanket analysis
        print_markov_blanket_analysis(model, adapted_outcomes)
        
        # Sensitivity analysis with key clinical variables
        key_variables = []
        for var in ['tumor_characteristics.biopsy_grading', 'tumor_characteristics.initial_size', 
                   'tumor_characteristics.metastasis_present_at_diagnosis', 'general.age',
                   'surgery', 'radiotherapy', 'chemotherapy']:
            if var in model_nodes:
                key_variables.append(var)
        
        # Also add any treatment variables from args if provided
        if args.treatments:
            for var in args.treatments:
                if var in model_nodes and var not in key_variables:
                    key_variables.append(var)
        
        # Parse baseline evidence if provided
        baseline_evidence = {}
        if args.evidence:
            raw_evidence = _parse_evidence(args.evidence)
            # Load config for preprocessing if available
            config = {}
            if args.config:
                config = _load_config(args.config)
            baseline_evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else {}
        
        print_sensitivity_analysis(model, adapted_outcomes, key_variables, baseline_evidence)
        
        # Print DAG structure info
        print(f"\n{'='*80}")
        print(f"DAG STRUCTURE")
        print(f"{'='*80}\n")
        print(f"Total nodes: {len(model.nodes())}")
        print(f"Total edges: {len(model.edges())}")
        print(f"\nNodes: {', '.join(sorted(model.nodes()))}")
        print(f"\nEdges:")
        for parent, child in sorted(model.edges()):
            print(f"  {parent} → {child}")
        print(f"\n{'='*80}\n")
    
    elif args.iptw:
        # IPTW mode - treatment effect analysis with inverse probability weighting
        if not args.config:
            parser.error("--iptw mode requires --config")
        if not args.outcomes:
            parser.error("--iptw mode requires --outcomes")
        
        # Adapt outcomes
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Load config for evidence preprocessing
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Parse evidence if provided
        raw_evidence = _parse_evidence(args.evidence) if args.evidence else None
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        
        # Check if regime comparisons are provided
        if args.regime_comparisons:
            regime_truncate = 0.005 if args.truncate is None else args.truncate
            regime_max_weight = None if args.regime_max_weight is None or args.regime_max_weight <= 0 else args.regime_max_weight
            # Regime comparison mode
            with open(args.regime_comparisons, 'r') as f:
                regime_config = json.load(f)
                comparisons = regime_config['comparisons']  # Pass full dict including sensitivity_conditions
            
            # Choose method: g-computation or IPTW
            if args.gcomputation:
                print("Using g-computation/standardization method (leverages BN structure, no overlap required)...")
                stratify_vars = getattr(args, 'stratify_by', None)
                if stratify_vars:
                    print(f"Results will be stratified by: {stratify_vars}")
                results = run_gcomputation_regime_analysis(
                    model,
                    args.config,
                    adapted_outcomes,
                    comparisons,
                    test_data_path=args.test_data_path,
                    evidence=evidence,
                    stratify_by=stratify_vars
                )
            else:
                print("Using IPTW method (inverse probability of treatment weighting)...")
                # Convert to tuple format for IPTW (doesn't support sensitivity analysis yet)
                iptw_comparisons = [
                    (comp['regime_a'], comp['regime_b'], comp['label'])
                    if isinstance(comp, dict) else comp
                    for comp in comparisons
                ]
                results = run_iptw_regime_analysis(
                    model,
                    args.config,
                    adapted_outcomes,
                    iptw_comparisons,  # Use tuple format
                    test_data_path=args.test_data_path,
                    stabilized=not args.no_stabilized,
                    truncate=regime_truncate,
                    min_ps=args.min_ps,
                    max_weight=regime_max_weight,
                    n_bootstrap=args.bootstrap_samples,
                    evidence=evidence
                )
        else:
            single_truncate = 0.05 if args.truncate is None else args.truncate
            # Individual treatment analysis with IPTW
            if not args.treatments:
                parser.error("--iptw mode requires --treatments (or --regime-comparisons for regime analysis)")
            
            print(f"Analyzing treatments: {args.treatments}")
            
            results = run_iptw_analysis(
                model, 
                args.config, 
                adapted_outcomes, 
                args.treatments,
                test_data_path=args.test_data_path,
                stabilized=not args.no_stabilized,
                truncate=single_truncate,
                min_ps=args.min_ps,
                n_bootstrap=args.bootstrap_samples,
                evidence=evidence,
            )
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Determine file format based on extension
            if output_path.suffix.lower() == '.csv':
            # Save as CSV
                df = pd.DataFrame(results)
                df.to_csv(output_path, index=False)
                print(f"Results saved to CSV: {output_path}")
        else:
            # Default to JSON
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to: {args.output}\n")
            print(f"Results saved to: {args.output}\n")
    
    elif args.validate:
        # Validate mode - compare predictions to actual data
        if not args.config:
            parser.error("--validate mode requires --config")
        
        validate_predictions(model, args.config, args.sample_size, args.full_dataset, args.plot_dir, args.test_data_path)


if __name__ == "__main__":
    main()
