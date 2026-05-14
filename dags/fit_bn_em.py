"""CLI: Fit a Bayesian network using Expectation-Maximisation (EM).

Unlike fit_bn.py, which drops every row that contains any missing value
(complete-case / listwise deletion), this script keeps ALL rows and uses
pgmpy's ExpectationMaximization estimator to handle missing values as latent
variables via the EM algorithm.

Why EM?
-------
Complete-case analysis is only unbiased under MCAR (Missing Completely At
Random), which is an unrealistic assumption in clinical sarcoma data where
missingness is often related to disease severity or treatment decisions
(MAR or MNAR).  EM treats unobserved values as latent variables and
iterates between:
  E-step: impute the expected sufficient statistics (fractional counts)
          for missing entries using the current CPDs
  M-step: re-estimate CPDs from the completed expected counts (equivalent
          to MLE on the imputed data)

This gives Maximum Marginal Likelihood estimates:
  θ* = argmax_θ  P(D_obs | G, θ)
where D_obs is the observed (possibly incomplete) data and G is the fixed DAG.

All preprocessing (type coercion, binning, train/test split) is identical to
fit_bn.py so the two approaches remain comparable.  The key difference is that
NaN values are left in the data handed to pgmpy instead of dropping rows.

Usage example
-------------
python fit_bn_em.py \
    --mongodb \
    --config dags/config/learn_structure_mongodb.yaml \
    --dag output/learned_dag_mongodb.json \
    --output output/fitted_bn_mongodb_em.pkl \
    --max-iter 200 \
    --atol 1e-6 \
    --test-size 0.3 \
    --random-seed 42 \
    --save-test dags/test_data_em.pkl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pgmpy.estimators import ExpectationMaximization
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

from dtcygan.bn import BNFitResult, save_bayesian_network
from dtcygan.io import DAGSerializer

# Reuse helpers that are already tested in fit_bn
from fit_bn import (
    _apply_feature_selection,
    _compute_bin_edges,
    _export_cpd_parameters_csv,
    _load_config,
    _load_data_from_mongodb,
)


# ---------------------------------------------------------------------------
# Data preparation (EM-aware: preserves NaN instead of dropping rows)
# ---------------------------------------------------------------------------

def _validate_and_clean_data_em(
    df: pd.DataFrame,
    dag_nodes: list[str],
    binning_config: dict | None = None,
    fit_binning: bool = True,
    bin_edges_map: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Validate, type-coerce and discretise data while KEEPING rows with NaN.

    This is the EM-aware counterpart of fit_bn._validate_and_clean_data.
    The only behavioural difference is that missing values are preserved as
    NaN so that pgmpy's ExpectationMaximization can handle them during the
    M-step instead of silently discarding those patients.

    Returns
    -------
    df : cleaned DataFrame with NaN preserved for missing entries
    learned_bin_edges : dict mapping column name → np.ndarray of bin edges
                        (populated only when fit_binning=True)
    """
    print(f"\nValidating and cleaning data (EM mode – NaN preserved)...")
    print(f"Initial data shape: {df.shape}")

    # ── column presence check ──────────────────────────────────────────────
    missing_cols = [n for n in dag_nodes if n not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")

    df = df[dag_nodes].copy()

    # ── report missingness per column ─────────────────────────────────────
    print(f"\nColumn data types and missingness:")
    total_missing = 0
    for col in df.columns:
        n_null = df[col].isnull().sum()
        total_missing += n_null
        pct = 100 * n_null / len(df) if len(df) else 0
        print(f"  {col}: dtype={df[col].dtype}, unique={df[col].nunique()}, "
              f"null={n_null} ({pct:.1f}%) ← kept for EM")

    print(f"\nTotal missing cells: {total_missing} "
          f"({100 * total_missing / (len(df) * len(df.columns)):.1f}% of all entries)")
    print(f"Rows with ≥1 missing value: "
          f"{df.isnull().any(axis=1).sum()} / {len(df)} "
          f"({100 * df.isnull().any(axis=1).mean():.1f}%) — RETAINED for EM")

    # ── type coercion & discretisation ────────────────────────────────────
    print(f"\nConverting columns to discrete types for pgmpy...")
    learned_bin_edges: dict = {} if fit_binning else (bin_edges_map or {})

    for col in df.columns:
        # Keep the NaN mask so we can restore it after numeric operations
        nan_mask = df[col].isnull()

        # Object columns: try numeric first, then categorical codes
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"  {col}: converted object → numeric")
            except (ValueError, TypeError):
                # Map categories to int codes; NaN stays NaN in Categorical
                df[col] = pd.Categorical(df[col]).codes.astype("float")
                # pd.Categorical gives -1 for NaN → restore NaN
                df.loc[df[col] == -1, col] = np.nan
                print(f"  {col}: converted object → categorical codes")

        explicit_binning = binning_config and col in binning_config

        # Boolean → int
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype("float")
            # Restore NaN
            df.loc[nan_mask, col] = np.nan
            print(f"  {col}: converted bool → int (0/1)")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_col = pd.to_numeric(df[col], errors="coerce")

            should_discretize = explicit_binning or not pd.api.types.is_integer_dtype(df[col])

            if should_discretize:
                n_bins = binning_config[col]["n_bins"] if explicit_binning else 5
                method = binning_config[col].get("method", "quantile") if explicit_binning else "quantile"
                src = "train" if fit_binning else "saved"
                print(f"  {col}: discretising into {n_bins} bins (method={method}, edges from {src})")

                if fit_binning:
                    # Compute edges only from non-NaN values
                    learned_bin_edges[col] = _compute_bin_edges(
                        numeric_col.dropna(), n_bins, method
                    )

                edges = learned_bin_edges.get(col)
                if edges is None:
                    # Constant column – map to state 0, preserve NaN
                    df[col] = np.where(nan_mask, np.nan, 0.0)
                    print(f"  {col}: constant after binning → state 0")
                else:
                    discretized = pd.cut(
                        numeric_col,
                        bins=edges,
                        labels=False,
                        include_lowest=True,
                    ).astype("float")
                    # pd.cut returns NaN for original NaN → already correct
                    df[col] = discretized
            else:
                # Integer-like float: round and keep NaN
                rounded = numeric_col.round()
                df[col] = np.where(nan_mask, np.nan, rounded)
        else:
            # Fallback categorical
            codes = pd.Categorical(df[col]).codes.astype("float")
            codes[codes == -1] = np.nan
            df[col] = codes
            print(f"  {col}: fallback categorical codes")

    # ── final report ──────────────────────────────────────────────────────
    print(f"\nPost-cleaning data shape: {df.shape}")
    remaining_missing = df.isnull().sum().sum()
    print(f"Missing cells after cleaning: {remaining_missing} "
          f"(all retained for EM)")

    # Validate: columns must be numeric (float OK for NaN) or already int
    for col in df.columns:
        if not (pd.api.types.is_numeric_dtype(df[col]) or
                pd.api.types.is_float_dtype(df[col])):
            raise ValueError(
                f"Column '{col}' is {df[col].dtype} after cleaning – "
                "must be numeric (int or float with NaN)."
            )

    print(f"\nSample of cleaned data (first 3 rows):")
    print(df.head(3).to_string())

    return df, learned_bin_edges


# ---------------------------------------------------------------------------
# EM fitting
# ---------------------------------------------------------------------------

def _smooth_cpd_laplace(cpd: TabularCPD, pseudo_counts: float) -> TabularCPD:
    """Apply post-EM Dirichlet/Laplace smoothing to one CPD.

    Adds pseudo-count mass to every state in each parent configuration, then
    re-normalizes each CPD column. This avoids exact zero probabilities and
    stabilizes sparse configurations.
    """
    if pseudo_counts <= 0:
        return cpd

    values = np.asarray(cpd.get_values(), dtype=float)
    smoothed = values + float(pseudo_counts)
    col_sums = smoothed.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0.0] = 1.0
    smoothed = smoothed / col_sums

    evidence = list(cpd.variables[1:]) if len(cpd.variables) > 1 else None
    evidence_card = list(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else None
    return TabularCPD(
        variable=cpd.variable,
        variable_card=cpd.variable_card,
        values=smoothed,
        evidence=evidence,
        evidence_card=evidence_card,
        state_names=cpd.state_names,
    )


def _apply_bayesian_smoothing(cpds: list[TabularCPD], pseudo_counts: float) -> list[TabularCPD]:
    """Apply post-EM Bayesian smoothing to all learned CPDs."""
    return [_smooth_cpd_laplace(cpd, pseudo_counts) for cpd in cpds]

def fit_bn_em(
    data: pd.DataFrame,
    dag,
    max_iter: int = 100,
    atol: float = 1e-8,
    n_jobs: int = 1,
    seed: int | None = None,
    show_progress: bool = True,
    bayes_smoothing: bool = False,
    pseudo_counts: float = 1.0,
) -> BNFitResult:
    """Fit a Discrete BN using Expectation-Maximisation on (possibly) incomplete data.

    Parameters
    ----------
    data : pd.DataFrame
        Cleaned, discretised training data.  NaN marks missing values.
    dag : DAGSpec
        The fixed DAG structure.
    max_iter : int
        Maximum number of EM iterations.
    atol : float
        Absolute tolerance for convergence (change in log-likelihood).
    n_jobs : int
        Number of parallel workers for the M-step (-1 = all cores).
    seed : int | None
        Random seed for CPD initialisation in pgmpy's EM.
    show_progress : bool
        Show tqdm progress bar per iteration.
    bayes_smoothing : bool
        If True, apply post-EM Dirichlet/Laplace smoothing to CPDs.
    pseudo_counts : float
        Pseudo-counts added per state when bayes_smoothing=True.

    Returns
    -------
    BNFitResult
    """
    import networkx as nx

    print(f"\nValidating DAG structure for EM fitting...")
    G = nx.DiGraph()
    G.add_nodes_from(dag.nodes)
    G.add_edges_from(dag.edges)
    if not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        raise ValueError(
            f"DAG contains {len(cycles)} cycle(s) – cannot fit BN. "
            f"First cycle: {cycles[0]}"
        )
    print("✓ DAG is acyclic")

    # Build the DiscreteBayesianNetwork shell (structure only, no CPDs yet)
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(dag.nodes)
    model.add_edges_from(dag.edges)

    # pgmpy's EM implementation in this environment cannot index CPDs with NaN
    # during likelihood evaluation. Encode missing entries as explicit per-column
    # states so we can keep all rows without crashing.
    em_data = data.copy()
    state_names: dict[str, list] = {}
    missing_state_map: dict[str, int] = {}
    for col in data.columns:
        if col in model.nodes():
            observed = data[col].dropna()
            states = sorted({int(s) for s in observed.unique().tolist()})

            # Degenerate edge case: if a column is fully missing, create a single
            # state so pgmpy still has a valid finite cardinality.
            if not states:
                states = [0]

            if data[col].isnull().any():
                missing_state = max(states) + 1
                missing_state_map[col] = missing_state
                em_data[col] = data[col].apply(
                    lambda x: missing_state if pd.isna(x) else int(x)
                )
                state_names[col] = states + [missing_state]
            else:
                em_data[col] = data[col].apply(int)
                state_names[col] = states

    print(f"\nFitting BN via EM (max_iter={max_iter}, atol={atol:.2e}, "
          f"n_jobs={n_jobs}, seed={seed})...")
    print(f"  Rows in training set : {len(data)}")
    print(f"  Rows with any NaN    : {data.isnull().any(axis=1).sum()}")
    print(f"  Total missing cells  : {data.isnull().sum().sum()}")
    if missing_state_map:
        print(f"  Missing-state encoding applied for {len(missing_state_map)} columns")

    em = ExpectationMaximization(model, em_data, state_names=state_names)

    cpds = em.get_parameters(
        max_iter=max_iter,
        atol=atol,
        n_jobs=n_jobs,
        seed=seed,
        show_progress=show_progress,
    )

    if bayes_smoothing:
        print(f"Applying Bayesian smoothing to CPDs (pseudo_counts={pseudo_counts})...")
        cpds = _apply_bayesian_smoothing(cpds, pseudo_counts)

    for cpd in cpds:
        model.add_cpds(cpd)

    if not model.check_model():
        raise ValueError("EM-fitted model failed pgmpy consistency check.")

    print("✓ EM converged and model is consistent")

    metadata = {
        "n_samples": len(data),
        "n_samples_with_missing": int(data.isnull().any(axis=1).sum()),
        "n_nodes": len(dag.nodes),
        "estimator": "em",
        "max_iter": max_iter,
        "atol": atol,
        "seed": seed,
        "bayes_smoothing": bayes_smoothing,
        "pseudo_counts": pseudo_counts if bayes_smoothing else None,
        "missing_state_map": missing_state_map,
    }
    return BNFitResult(model=model, estimator="em", metadata=metadata)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a Bayesian network via Expectation-Maximisation (EM), "
            "which handles missing values as latent variables instead of dropping rows."
        )
    )

    # ── input ──────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data", help="Path to CSV data file.")
    input_group.add_argument(
        "--mongodb", action="store_true",
        help="Load data from MongoDB (requires --config)."
    )

    parser.add_argument(
        "--config",
        help="Path to YAML/JSON config (required for --mongodb; "
             "also used for binning config when using --data)."
    )
    parser.add_argument("--dag", required=True, help="Path to DAG JSON.")
    parser.add_argument("--output", required=True,
                        help="Path to save the fitted BN pickle.")

    # ── EM hyper-parameters ────────────────────────────────────────────────
    parser.add_argument(
        "--max-iter", type=int, default=100,
        help="Maximum number of EM iterations (default: 100)."
    )
    parser.add_argument(
        "--atol", type=float, default=1e-8,
        help="Convergence tolerance for log-likelihood change (default: 1e-8)."
    )
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel workers for M-step (-1 = all cores, default: 1)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for CPD initialisation (default: None = random)."
    )
    parser.add_argument(
        "--no-progress", action="store_true",
        help="Suppress the per-iteration tqdm progress bar."
    )
    parser.add_argument(
        "--bayes-smoothing", action="store_true",
        help="Apply post-EM Bayesian smoothing (Dirichlet/Laplace) to CPDs."
    )
    parser.add_argument(
        "--pseudo-counts", type=float, default=1.0,
        help="Pseudo-counts for post-EM Bayesian smoothing (default: 1.0)."
    )

    # ── train / test split ─────────────────────────────────────────────────
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out as test set (default: 0.2)."
    )
    parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed for the train/test split (default: 42)."
    )
    parser.add_argument(
        "--save-test",
        help="Optional path to save the test set pickle for later validation."
    )

    args = parser.parse_args()

    if args.bayes_smoothing and args.pseudo_counts <= 0:
        parser.error("--pseudo-counts must be > 0 when --bayes-smoothing is enabled")

    # ── load config ────────────────────────────────────────────────────────
    config: dict = {}
    if args.config:
        config = _load_config(args.config)
        print(f"Loaded config from: {args.config}")

    # ── load data ──────────────────────────────────────────────────────────
    if args.mongodb:
        if not config:
            parser.error("--config is required when using --mongodb")
        data = _load_data_from_mongodb(config)
        if config.get("features"):
            data = _apply_feature_selection(data, config)
    else:
        print(f"Loading data from CSV: {args.data}")
        data = pd.read_csv(args.data)
        print(f"Loaded {len(data)} rows")

    # ── load DAG ───────────────────────────────────────────────────────────
    dag = DAGSerializer.load_json(args.dag)
    dag_nodes = list(dag.nodes)

    missing_cols = [n for n in dag_nodes if n not in data.columns]
    if missing_cols:
        raise ValueError(f"DAG nodes absent from data: {missing_cols}")

    # Restrict to DAG columns (NaN preserved – do NOT dropna here)
    data = data[dag_nodes].copy()

    print(f"\nData before split: {len(data)} rows, "
          f"{data.isnull().any(axis=1).sum()} with missing values")

    # ── train / test split ─────────────────────────────────────────────────
    stratify_col = "local_recurrence" if "local_recurrence" in data.columns else None
    # For stratified split we need non-NaN labels; use only complete-case rows
    # for the split key but retain all rows in the resulting sets.
    if stratify_col and data[stratify_col].isnull().any():
        stratify_labels = None
        print(f"Note: '{stratify_col}' has missing values – "
              "falling back to unstratified split.")
    else:
        stratify_labels = data[stratify_col] if stratify_col else None

    train_data, test_data = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=stratify_labels,
    )
    print(f"\nTrain / test split: {len(train_data)} / {len(test_data)}")

    # ── discretise (NaN-preserving) ────────────────────────────────────────
    binning_config = config.get("binning", {})

    train_data, learned_bin_edges = _validate_and_clean_data_em(
        train_data,
        dag_nodes,
        binning_config=binning_config,
        fit_binning=True,
    )
    test_data, _ = _validate_and_clean_data_em(
        test_data,
        dag_nodes,
        binning_config=binning_config,
        fit_binning=False,
        bin_edges_map=learned_bin_edges,
    )

    # ── optionally save test set ───────────────────────────────────────────
    if args.save_test:
        test_path = Path(args.save_test)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_pickle(str(test_path))
        print(f"Test set saved to: {test_path}")

    # ── EM fitting ─────────────────────────────────────────────────────────
    result = fit_bn_em(
        train_data,
        dag,
        max_iter=args.max_iter,
        atol=args.atol,
        n_jobs=args.n_jobs,
        seed=args.seed,
        show_progress=not args.no_progress,
        bayes_smoothing=args.bayes_smoothing,
        pseudo_counts=args.pseudo_counts,
    )

    # ── save model ─────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_bayesian_network(result.model, args.output)

    print(f"\n{'='*60}")
    print(f"✓ EM-fitted Bayesian network saved!")
    print(f"{'='*60}")
    print(f"Output      : {args.output}")
    print(f"Nodes       : {len(result.model.nodes())}")
    print(f"Edges       : {len(result.model.edges())}")
    print(f"CPDs        : {len(result.model.get_cpds())}")
    if result.metadata.get("bayes_smoothing"):
        print(f"Smoothing   : Bayesian (pseudo_counts={result.metadata.get('pseudo_counts')})")
    print(f"Train rows  : {result.metadata['n_samples']} "
          f"(of which {result.metadata['n_samples_with_missing']} had missing values)")

    cpd_csv_path = output_path.with_name(f"{output_path.stem}_cpd_parameters.csv")
    _export_cpd_parameters_csv(result.model, cpd_csv_path)
    print(f"CPD CSV     : {cpd_csv_path}")

    print("\nLearned CPDs:")
    for cpd in result.model.get_cpds():
        print(cpd)
        print("-" * 80)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
