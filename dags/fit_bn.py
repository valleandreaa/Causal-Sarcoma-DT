"""CLI: Fit a Bayesian network given a learned DAG."""

from __future__ import annotations

import argparse
import inspect
import itertools
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

from dtcygan.bn import fit_bayesian_network, save_bayesian_network
from dtcygan.data_schema import DataSchema
from dtcygan.io import DAGSerializer
import numpy as np
from sklearn.linear_model import LogisticRegression
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.factors.discrete import TabularCPD


def _load_config(config_path: str) -> dict:
    """Load configuration from YAML or JSON file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with config_file.open("r", encoding="utf-8") as f:
        if config_file.suffix.lower() in [".yaml", ".yml"]:
            return yaml.safe_load(f)
        elif config_file.suffix.lower() == ".json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_file.suffix}")


def _load_data_from_mongodb(config: dict) -> pd.DataFrame:
    """Load data from MongoDB using configuration."""
    load_dotenv()
    
    # Get MongoDB connection details from config or environment
    mongo_uri = config.get("mongodb", {}).get("uri") or os.getenv("MONGO_URI")
    db_name = config.get("mongodb", {}).get("database") or os.getenv("MONGO_DB")
    collection_name = config.get("mongodb", {}).get("collection") or os.getenv("MONGO_COLLECTION")
    
    if not all([mongo_uri, db_name, collection_name]):
        raise ValueError("MongoDB connection details missing in config or environment variables")
    
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    mongodb_config = config.get("mongodb", {})
    pipeline = mongodb_config.get("pipeline")
    
    print(f"Connecting to MongoDB: {db_name}.{collection_name}")
    
    # Check if using aggregation pipeline or simple query
    if pipeline:
        print(f"Using aggregation pipeline with {len(pipeline)} stages")
        cursor = collection.aggregate(pipeline)
    else:
        # Get query and projection for simple find
        query = mongodb_config.get("query", {})
        projection = mongodb_config.get("projection")
        print(f"Query: {query}")
        cursor = collection.find(query, projection)
    
    # Fetch data
    data = list(cursor)
    client.close()
    
    if not data:
        raise ValueError("No data returned from MongoDB query")
    
    print(f"Loaded {len(data)} documents from MongoDB")
    
    # Convert to DataFrame
    df = pd.json_normalize(data)
    
    # Remove MongoDB _id if present and not needed
    if "_id" in df.columns and not mongodb_config.get("include_id", False):
        df = df.drop(columns=["_id"])
    
    return df


def _apply_feature_selection(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Apply feature selection based on config."""
    features = config.get("features", {})
    
    # Include specific features if specified
    include_features = features.get("include")
    if include_features:
        missing_features = [f for f in include_features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")
        df = df[include_features]
        print(f"Selected {len(include_features)} features: {include_features}")
    
    # Exclude specific features if specified
    exclude_features = features.get("exclude", [])
    if exclude_features:
        df = df.drop(columns=[f for f in exclude_features if f in df.columns])
        print(f"Excluded features: {exclude_features}")
    
    return df


def _apply_rebalancing(df: pd.DataFrame, target_column: str = 'local_recurrence', random_seed: int = 42) -> pd.DataFrame:
    """Apply stratified oversampling to balance classes in target column.
    
    Uses stratified resampling to duplicate minority class samples until balanced.
    This is preferred over SMOTE for causal inference as it uses real patient data.
    
    Args:
        df: Input dataframe
        target_column: Column to use for balancing (default: local_recurrence)
        random_seed: Random seed for reproducibility
    
    Returns:
        Rebalanced dataframe
    """
    if target_column not in df.columns:
        print(f"Warning: {target_column} not found in data, skipping rebalancing")
        return df
    
    print(f"\nApplying stratified rebalancing on {target_column}...")
    print(f"Class distribution before rebalancing:")
    print(df[target_column].value_counts())
    
    # Separate by class
    df_majority = df[df[target_column] == 0]
    df_minority = df[df[target_column] == 1]
    
    # Upsample minority class to match majority
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=random_seed)
    
    # Combine and shuffle
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Class distribution after rebalancing:")
    print(df_balanced[target_column].value_counts())
    print(f"Total samples: {len(df)} → {len(df_balanced)}")
    
    return df_balanced


def _compute_bin_edges(series: pd.Series, n_bins: int, method: str) -> np.ndarray | None:
    """Compute reusable bin edges from training data."""
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.isnull().any():
        raise ValueError(f"Cannot bin non-numeric column: {series.name}")

    min_val = float(numeric.min())
    max_val = float(numeric.max())
    if min_val == max_val:
        return None

    if method == "uniform":
        edges = np.linspace(min_val, max_val, n_bins + 1)
    else:
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(numeric, quantiles)

    edges = np.unique(edges)
    if len(edges) < 2:
        return None

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _validate_and_clean_data(
    df: pd.DataFrame,
    dag_nodes: list[str],
    binning_config: dict | None = None,
    fit_binning: bool = True,
    bin_edges_map: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Validate and clean data for Bayesian network fitting.
    
    Args:
        df: Input dataframe
        dag_nodes: List of nodes in the DAG
        binning_config: Dictionary mapping column names to binning parameters
                       e.g., {'column_name': {'n_bins': 5, 'method': 'quantile'}}
    """
    print(f"\nValidating and cleaning data...")
    print(f"Initial data shape: {df.shape}")
    print(f"Data columns: {list(df.columns)}")
    print(f"DAG nodes: {dag_nodes}")
    
    # Check for missing columns
    missing_cols = [node for node in dag_nodes if node not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")
    
    # Keep only columns that are in the DAG
    df = df[dag_nodes].copy()
    
    # Check data types and handle problematic columns
    print(f"\nColumn data types:")
    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        n_null = df[col].isnull().sum()
        print(f"  {col}: dtype={dtype}, unique={n_unique}, null={n_null}")
        
        # Handle null values
        if n_null > 0:
            print(f"    Warning: {col} has {n_null} null values - dropping rows with nulls")
    
    # Drop rows with any null values (pgmpy doesn't handle them well)
    initial_len = len(df)
    df = df.dropna()
    if len(df) < initial_len:
        print(f"  Dropped {initial_len - len(df)} rows with null values")
    
    if len(df) == 0:
        raise ValueError("No data remaining after dropping null values")
    
    # Convert all columns to discrete/categorical types for pgmpy
    print(f"\nConverting columns to discrete types for pgmpy...")
    learned_bin_edges = {} if fit_binning else (bin_edges_map or {})
    for col in df.columns:
        # Check for mixed types or objects
        if df[col].dtype == 'object':
            # Try to convert to numeric first
            try:
                df[col] = pd.to_numeric(df[col])
                print(f"  {col}: converted object to numeric")
            except (ValueError, TypeError):
                # If that fails, convert to categorical codes
                df[col] = pd.Categorical(df[col]).codes
                print(f"  {col}: converted object to categorical codes")
        
        # Apply explicit binning config first (even if values are integer-like).
        explicit_binning = binning_config and col in binning_config

        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            print(f"  {col}: converted bool to int")
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_col = pd.to_numeric(df[col], errors='coerce')
            if numeric_col.isnull().any():
                raise ValueError(f"Column {col} contains non-numeric values after conversion")

            should_discretize = explicit_binning or not pd.api.types.is_integer_dtype(df[col])

            if should_discretize:
                if explicit_binning:
                    n_bins = binning_config[col].get('n_bins', 5)
                    method = binning_config[col].get('method', 'quantile')
                    print(f"  {col}: discretizing into {n_bins} bins (method: {method}) from {'train' if fit_binning else 'saved'} edges")
                else:
                    n_bins = 5
                    method = 'quantile'
                    print(f"  Warning: {col} has non-integer values, discretizing into {n_bins} bins (method: {method})...")

                if fit_binning:
                    learned_bin_edges[col] = _compute_bin_edges(numeric_col, n_bins, method)

                edges = learned_bin_edges.get(col)
                if edges is None:
                    df[col] = 0
                    print(f"  {col}: constant after binning, mapped to single state 0")
                else:
                    discretized = pd.cut(
                        numeric_col,
                        bins=edges,
                        labels=False,
                        include_lowest=True,
                    )
                    df[col] = discretized.fillna(0).astype(int)
            else:
                if (numeric_col == numeric_col.round()).all():
                    df[col] = numeric_col.round().astype(int)
                    print(f"  {col}: converted float to int")
                else:
                    raise ValueError(f"Column {col} has unexpected non-integer numeric values")
        else:
            # Fallback: convert to categorical codes
            df[col] = pd.Categorical(df[col]).codes
            print(f"  {col}: converted to categorical codes (fallback)")

        if pd.api.types.is_integer_dtype(df[col]):
            print(f"  {col}: already integer")
    
    # Final validation: ensure all columns are integers with no nulls
    print(f"\nFinal validation:")
    for col in df.columns:
        if not pd.api.types.is_integer_dtype(df[col]):
            raise ValueError(f"Column {col} is not integer type after conversion: {df[col].dtype}")
        if df[col].isnull().any():
            raise ValueError(f"Column {col} still has null values after cleaning")
        print(f"  {col}: dtype={df[col].dtype}, range=[{df[col].min()}, {df[col].max()}], unique={df[col].nunique()}")
    
    print(f"\nFinal data shape: {df.shape}")
    print(f"Sample of cleaned data:")
    print(df.head())
    print(f"\nData types after cleaning:")
    print(df.dtypes)
    
    return df, learned_bin_edges


def _compute_standardized_mean_difference(data: pd.DataFrame, treatment_regime: pd.Series, 
                                         confounder_vars: list[str], weights: pd.Series = None) -> pd.DataFrame:
    """Compute Standardized Mean Difference (SMD) for each confounder across treatment regimes.
    
    Args:
        data: DataFrame containing confounders
        treatment_regime: Series indicating treatment regime for each sample
        confounder_vars: List of confounder variable names
        weights: Optional IPW weights (if None, unweighted comparison)
        
    Returns:
        DataFrame with SMD for each confounder and regime pair
    """
    smd_results = []
    unique_regimes = np.unique(treatment_regime)
    
    for conf in confounder_vars:
        if conf not in data.columns:
            continue
            
        for i, regime1 in enumerate(unique_regimes):
            for regime2 in unique_regimes[i+1:]:
                mask1 = treatment_regime == regime1
                mask2 = treatment_regime == regime2
                
                x1 = data.loc[mask1, conf]
                x2 = data.loc[mask2, conf]
                
                if weights is not None:
                    w1 = weights[mask1]
                    w2 = weights[mask2]
                    # Weighted means and variances
                    mean1 = np.average(x1, weights=w1)
                    mean2 = np.average(x2, weights=w2)
                    var1 = np.average((x1 - mean1)**2, weights=w1)
                    var2 = np.average((x2 - mean2)**2, weights=w2)
                else:
                    # Unweighted
                    mean1, mean2 = x1.mean(), x2.mean()
                    var1, var2 = x1.var(), x2.var()
                
                # Pooled standard deviation
                pooled_sd = np.sqrt((var1 + var2) / 2)
                smd = (mean1 - mean2) / pooled_sd if pooled_sd > 0 else 0.0
                
                smd_results.append({
                    'confounder': conf,
                    'regime_1': int(regime1),
                    'regime_2': int(regime2),
                    'smd': abs(smd),
                    'mean_1': mean1,
                    'mean_2': mean2,
                    'weighted': weights is not None
                })
    
    return pd.DataFrame(smd_results)


def _create_ipw_diagnostic_plots(diagnostics: dict, output_dir: Path):
    """Create diagnostic plots for IPW quality assessment.
    
    Args:
        diagnostics: Dictionary with weight statistics and SMD results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Weight distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Weight histogram
    weights = diagnostics['weights']
    axes[0, 0].hist(weights, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(weights.mean(), color='red', linestyle='--', label=f'Mean: {weights.mean():.2f}')
    axes[0, 0].axvline(weights.median(), color='blue', linestyle='--', label=f'Median: {weights.median():.2f}')
    axes[0, 0].set_xlabel('IPW Weight')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Weight boxplot by regime
    regime_data = diagnostics['regime_data']
    regime_data_sorted = regime_data.sort_values('regime')
    axes[0, 1].boxplot([regime_data[regime_data['regime']==r]['weight'].values 
                        for r in regime_data_sorted['regime'].unique()],
                       labels=[f"R{int(r)}" for r in regime_data_sorted['regime'].unique()])
    axes[0, 1].set_xlabel('Treatment Regime')
    axes[0, 1].set_ylabel('IPW Weight')
    axes[0, 1].set_title('')
    axes[0, 1].grid(alpha=0.3)
    
    # ESS by regime
    ess_by_regime = regime_data.groupby('regime').apply(
        lambda g: (g['weight'].sum()**2) / (g['weight']**2).sum()
    )
    axes[1, 0].bar(range(len(ess_by_regime)), ess_by_regime.values, color='steelblue', alpha=0.7)
    axes[1, 0].set_xlabel('Treatment Regime')
    axes[1, 0].set_ylabel('Effective Sample Size (ESS)')
    axes[1, 0].set_title('')
    axes[1, 0].set_xticks(range(len(ess_by_regime)))
    axes[1, 0].set_xticklabels([f"R{int(r)}" for r in ess_by_regime.index])
    axes[1, 0].grid(alpha=0.3, axis='y')
    
    # Sample size comparison
    n_by_regime = regime_data['regime'].value_counts().sort_index()
    x_pos = np.arange(len(n_by_regime))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, n_by_regime.values, width, label='Actual N', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, ess_by_regime.values, width, label='Effective N', alpha=0.7)
    axes[1, 1].set_xlabel('Treatment Regime')
    axes[1, 1].set_ylabel('Sample Size')
    axes[1, 1].set_title('')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f"R{int(r)}" for r in n_by_regime.index])
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ipw_weight_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. SMD balance plot (before and after weighting)
    if 'smd_before' in diagnostics and 'smd_after' in diagnostics:
        smd_before = diagnostics['smd_before']
        smd_after = diagnostics['smd_after']
        
        # Merge SMD results
        smd_comparison = pd.merge(
            smd_before[['confounder', 'regime_1', 'regime_2', 'smd']],
            smd_after[['confounder', 'regime_1', 'regime_2', 'smd']],
            on=['confounder', 'regime_1', 'regime_2'],
            suffixes=('_before', '_after')
        )
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Love plot (before vs after)
        for idx, row in smd_comparison.iterrows():
            axes[0].plot([row['smd_before'], row['smd_after']], [idx, idx], 'o-', alpha=0.6)
        axes[0].axvline(0.1, color='red', linestyle='--', alpha=0.5, label='SMD=0.1 threshold')
        axes[0].set_xlabel('Absolute Standardized Mean Difference')
        axes[0].set_ylabel('Confounder-Regime Pair')
        axes[0].set_title('SMD Balance: Before vs After IPW')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Heatmap of SMD after weighting
        smd_pivot = smd_after.pivot_table(
            index='confounder', 
            columns=['regime_1', 'regime_2'], 
            values='smd'
        )
        if not smd_pivot.empty:
            sns.heatmap(smd_pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                       center=0.1, vmin=0, vmax=0.3, ax=axes[1], cbar_kws={'label': 'Abs(SMD)'})
            axes[1].set_title('SMD Heatmap After IPW (Balanced if < 0.1)')
            axes[1].set_xlabel('Regime Pair')
            axes[1].set_ylabel('Confounder')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'ipw_smd_balance.png', dpi=300, bbox_inches='tight')
        plt.close()


def _compute_propensity_weights(
    data: pd.DataFrame,
    treatment_vars: list[str],
    confounder_vars: list[str],
    min_ps: float = 0.01,
    max_weight: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    """Compute inverse propensity weights using a joint treatment propensity model.
    
    Args:
        data: Training data
        treatment_vars: List of treatment variable names
        confounder_vars: List of confounder variable names that predict treatment
        min_ps: Minimum propensity score (clip below this)
        max_weight: Maximum weight value (truncate above this)
        
    Returns:
        Tuple of (DataFrame with 'ipw_weight' column, diagnostics dictionary)
    """
    print(f"\n{'='*60}")
    print(f"Computing Inverse Propensity Weights for Stratified CPDs")
    print(f"{'='*60}")
    print(f"Treatment variables: {treatment_vars}")
    print(f"Confounder variables: {confounder_vars}")
    
    data_with_weights = data.copy()
    data_with_weights['ipw_weight'] = 1.0

    available_treatments = [t for t in treatment_vars if t in data.columns]
    available_confounders = [c for c in confounder_vars if c in data.columns]
    print(f"Available treatment variables: {available_treatments}")
    print(f"Available confounder variables: {available_confounders}")

    # Enforce logical constraints: if margin_judgement variable exists, surgery must be 1
    # (margin judgement can only be recorded if surgery was performed)
    if 'surgery' in data_with_weights.columns and 'treatments.pathologist_margin_judgement' in data_with_weights.columns:
        # Filter to only samples where surgery=1 (margin judgement requires surgery)
        no_surgery_mask = data_with_weights['surgery'] == 0
        n_excluded = no_surgery_mask.sum()
        if n_excluded > 0:
            print(f"\n  ⚠ Data filtering: Excluding {n_excluded} samples with surgery=0")
            print(f"     (margin_judgement variable requires surgery=1 to be meaningful)")
            data_with_weights = data_with_weights[~no_surgery_mask].copy()
            print(f"     Remaining samples: {len(data_with_weights)}")

    # Initialize diagnostics
    diagnostics = {
        'weights': data_with_weights['ipw_weight'],
        'regime_data': pd.DataFrame({'regime': 0, 'weight': 1.0}, index=data.index),
        'ess_overall': len(data),
        'ess_by_regime': {},
        'smd_before': pd.DataFrame(),
        'smd_after': pd.DataFrame(),
        'reliability_flags': []
    }

    if not available_treatments or not available_confounders:
        print("  Warning: Missing treatments or confounders; using uniform weights")
        return data_with_weights, diagnostics

    treatment_regime = (
        data_with_weights[available_treatments]
        .astype(str)
        .agg("|".join, axis=1)
        .astype("category")
    )
    y = treatment_regime.cat.codes.to_numpy()
    n_regimes = int(np.unique(y).size)
    
    # Create regime descriptions (map code -> treatment combinations)
    regime_descriptions = {}
    for regime_code in np.unique(y):
        regime_mask = y == regime_code
        # Get the first sample of this regime to show the treatment combination
        sample_idx = np.where(regime_mask)[0][0]
        treatment_combo = {}
        for treat_var in available_treatments:
            treatment_combo[treat_var] = data_with_weights[treat_var].iloc[sample_idx]
        regime_descriptions[int(regime_code)] = treatment_combo
    
    # Print regime descriptions
    print(f"\nJoint treatment regimes observed: {n_regimes}")
    print(f"\nRegime Descriptions:")
    print(f"  {'Code':<6} {'N':<6} {'Treatment Combination'}")
    print(f"  {'-'*70}")
    for regime_code in sorted(regime_descriptions.keys()):
        n_in_regime = (y == regime_code).sum()
        combo_str = ", ".join([f"{k}={v}" for k, v in regime_descriptions[regime_code].items()])
        print(f"  R{regime_code:<5} {n_in_regime:<6} {combo_str}")

    if n_regimes < 2:
        print("  Warning: Only one observed treatment regime; using uniform weights")
        return data_with_weights, diagnostics

    X = data_with_weights[available_confounders].apply(pd.to_numeric, errors='coerce')
    if X.isnull().any().any():
        raise ValueError("Confounder matrix contains non-numeric values after conversion")

    # sklearn API compatibility: newer versions may remove `multi_class`.
    ps_model_kwargs = {
        "max_iter": 2000,
        "random_state": 42,
        "solver": "lbfgs",
    }
    if "multi_class" in inspect.signature(LogisticRegression.__init__).parameters:
        ps_model_kwargs["multi_class"] = "multinomial"

    ps_model = LogisticRegression(**ps_model_kwargs)
    ps_model.fit(X.values, y)
    ps_matrix = ps_model.predict_proba(X.values)
    observed_ps = ps_matrix[np.arange(len(y)), y]

    clipped_ps = np.clip(observed_ps, min_ps, 1.0)
    marginal_regime_probs = np.bincount(y) / len(y)
    stabilized_weights = marginal_regime_probs[y] / clipped_ps
    stabilized_weights = np.clip(stabilized_weights, 0.1, max_weight)
    data_with_weights['ipw_weight'] = stabilized_weights

    # Normalize weights to sum to n (preserve sample size)
    total_weight = data_with_weights['ipw_weight'].sum()
    data_with_weights['ipw_weight'] *= len(data_with_weights) / total_weight

    near_positivity = float((observed_ps < 0.05).mean())
    if observed_ps.min() < 0.05:
        print(
            f"  Warning: positivity concern detected; {near_positivity:.1%} "
            f"of observed regimes have PS < 0.05"
        )
    
    # Compute diagnostics
    weights = data_with_weights['ipw_weight']
    ess_overall = (weights.sum()**2) / (weights**2).sum()
    
    print(f"\n  {'='*50}")
    print(f"  IPW DIAGNOSTICS")
    print(f"  {'='*50}")
    print(f"  Joint PS range: [{observed_ps.min():.3f}, {observed_ps.max():.3f}]")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Weight mean: {weights.mean():.3f}")
    print(f"  Weight std: {weights.std():.3f}")
    print(f"  Overall ESS: {ess_overall:.1f} ({ess_overall/len(data)*100:.1f}% of sample)")
    
    # ESS by regime
    regime_data = pd.DataFrame({
        'regime': y,
        'weight': weights,
        'ps': observed_ps
    })
    
    print(f"\n  ESS by Treatment Regime:")
    ess_by_regime = {}
    for regime_code in np.unique(y):
        regime_mask = y == regime_code
        regime_weights = weights[regime_mask]
        n_regime = regime_mask.sum()
        ess_regime = (regime_weights.sum()**2) / (regime_weights**2).sum()
        ess_by_regime[int(regime_code)] = ess_regime
        reliability = "✓ RELIABLE" if ess_regime >= 30 else "⚠ UNRELIABLE (ESS < 30)"
        
        # Get treatment combination for this regime
        combo_str = ", ".join([f"{k}={v}" for k, v in regime_descriptions[int(regime_code)].items()])
        print(f"    Regime {regime_code} [{combo_str}]:")
        print(f"      N={n_regime}, ESS={ess_regime:.1f}, ESS/N={ess_regime/n_regime*100:.1f}% - {reliability}")
    
    # Compute SMD before and after weighting
    print(f"\n  SMD Balance Assessment:")
    smd_before = _compute_standardized_mean_difference(data_with_weights, y, available_confounders, weights=None)
    smd_after = _compute_standardized_mean_difference(data_with_weights, y, available_confounders, weights=weights)
    
    # Check balance (SMD < 0.1 is considered balanced)
    max_smd_before = smd_before['smd'].max() if len(smd_before) > 0 else 0
    max_smd_after = smd_after['smd'].max() if len(smd_after) > 0 else 0
    mean_smd_before = smd_before['smd'].mean() if len(smd_before) > 0 else 0
    mean_smd_after = smd_after['smd'].mean() if len(smd_after) > 0 else 0
    
    print(f"    Before weighting: Max SMD={max_smd_before:.3f}, Mean SMD={mean_smd_before:.3f}")
    print(f"    After weighting:  Max SMD={max_smd_after:.3f}, Mean SMD={mean_smd_after:.3f}")
    
    # Count balanced confounders
    n_balanced_after = (smd_after['smd'] < 0.1).sum()
    n_total_comparisons = len(smd_after)
    if n_total_comparisons > 0:
        balance_pct = n_balanced_after / n_total_comparisons * 100
        print(f"    Balance achieved: {n_balanced_after}/{n_total_comparisons} comparisons " +
              f"({balance_pct:.1f}%) have SMD < 0.1")
        
        if max_smd_after > 0.2:
            print(f"    ⚠ WARNING: Some confounders remain imbalanced (max SMD > 0.2)")
        elif max_smd_after > 0.1:
            print(f"    ⚠ CAUTION: Moderate imbalance remains (max SMD > 0.1)")
        else:
            print(f"    ✓ GOOD: All confounders well-balanced (max SMD < 0.1)")
    
    # Reliability flags
    reliability_flags = []
    for regime_code, ess in ess_by_regime.items():
        if ess < 30:
            reliability_flags.append({
                'issue': 'Low ESS',
                'regime': regime_code,
                'ess': ess,
                'severity': 'HIGH' if ess < 10 else 'MODERATE'
            })
    
    if max_smd_after > 0.2:
        reliability_flags.append({
            'issue': 'Poor balance',
            'max_smd': max_smd_after,
            'severity': 'HIGH'
        })
    elif max_smd_after > 0.1:
        reliability_flags.append({
            'issue': 'Moderate imbalance',
            'max_smd': max_smd_after,
            'severity': 'MODERATE'
        })
    
    if reliability_flags:
        print(f"\n  ⚠ RELIABILITY CONCERNS:")
        for flag in reliability_flags:
            print(f"    - {flag['issue']}: {flag}")
    else:
        print(f"\n  ✓ NO MAJOR RELIABILITY CONCERNS")
    
    print(f"  {'='*50}\n")
    
    # Package diagnostics
    diagnostics = {
        'weights': weights,
        'regime_data': regime_data,
        'ess_overall': ess_overall,
        'ess_by_regime': ess_by_regime,
        'regime_descriptions': regime_descriptions,
        'smd_before': smd_before,
        'smd_after': smd_after,
        'reliability_flags': reliability_flags,
        'ps_range': (float(observed_ps.min()), float(observed_ps.max())),
        'weight_range': (float(weights.min()), float(weights.max())),
        'n_regimes': n_regimes,
        'max_smd_before': float(max_smd_before),
        'max_smd_after': float(max_smd_after),
        'mean_smd_before': float(mean_smd_before),
        'mean_smd_after': float(mean_smd_after)
    }
    
    return data_with_weights, diagnostics


def fit_bn_with_stratified_cpds(data: pd.DataFrame, dag, treatment_vars: list[str],
                                 outcome_vars: list[str], confounder_vars: list[str],
                                 estimator: str = "ml", prior_type: str = "BDeu",
                                 pseudo_counts: float = 1.0, output_dir: Path = None):
    """Fit Bayesian Network with IPW-adjusted CPDs for outcome variables.
    
    This addresses confounding by indication by:
    1. Computing propensity weights based on confounders
    2. Fitting outcome node CPDs using weighted frequencies
    3. Fitting other nodes with standard MLE
    
    Args:
        data: Training data
        dag: DAG specification
        treatment_vars: Treatment variable names (e.g., ['surgery', 'radiotherapy'])
        outcome_vars: Outcome variable names (e.g., ['metastasis', 'DOD'])
        confounder_vars: Baseline confounders (e.g., ['age', 'grading', 'size'])
        estimator: 'ml' or 'bayes'
        prior_type: Prior for Bayesian estimation
        pseudo_counts: Pseudo-counts for Bayesian estimation
        output_dir: Directory for saving diagnostic outputs
        
    Returns:
        BNFitResult with adjusted model
    """
    from dtcygan.bn import BNFitResult
    import networkx as nx
    
    print(f"\n{'='*60}")
    print(f"Fitting BN with Stratified CPDs (IPW-adjusted)")
    print(f"{'='*60}")
    
    # Compute propensity weights and get diagnostics
    data_weighted, diagnostics = _compute_propensity_weights(data, treatment_vars, confounder_vars)
    
    # Save diagnostics if output directory provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save diagnostic plots
        _create_ipw_diagnostic_plots(diagnostics, output_dir)
        print(f"\n✓ Diagnostic plots saved to: {output_dir}")
        
        # Save SMD tables
        if not diagnostics['smd_before'].empty:
            diagnostics['smd_before'].to_csv(output_dir / 'ipw_smd_before.csv', index=False)
            print(f"  - SMD before weighting: {output_dir / 'ipw_smd_before.csv'}")
        
        if not diagnostics['smd_after'].empty:
            diagnostics['smd_after'].to_csv(output_dir / 'ipw_smd_after.csv', index=False)
            print(f"  - SMD after weighting: {output_dir / 'ipw_smd_after.csv'}")
        
        # Save comprehensive diagnostics JSON
        diagnostics_json = {
            'ess_overall': float(diagnostics['ess_overall']),
            'ess_by_regime': {int(k): float(v) for k, v in diagnostics['ess_by_regime'].items()},
            'regime_descriptions': {
                int(k): {str(treat): int(val) if isinstance(val, (int, np.integer)) else float(val) if isinstance(val, (float, np.floating)) else str(val)
                         for treat, val in v.items()}
                for k, v in diagnostics['regime_descriptions'].items()
            },
            'weight_range': diagnostics['weight_range'],
            'ps_range': diagnostics['ps_range'],
            'n_regimes': diagnostics['n_regimes'],
            'max_smd_before': diagnostics['max_smd_before'],
            'max_smd_after': diagnostics['max_smd_after'],
            'mean_smd_before': diagnostics['mean_smd_before'],
            'mean_smd_after': diagnostics['mean_smd_after'],
            'reliability_flags': diagnostics['reliability_flags']
        }
        
        with open(output_dir / 'ipw_diagnostics.json', 'w') as f:
            json.dump(diagnostics_json, f, indent=2)
        print(f"  - Diagnostic summary: {output_dir / 'ipw_diagnostics.json'}")
        
        # Save regime descriptions as readable CSV
        regime_desc_rows = []
        for regime_code, treatments in diagnostics['regime_descriptions'].items():
            n_in_regime = (diagnostics['regime_data']['regime'] == regime_code).sum()
            ess = diagnostics['ess_by_regime'].get(regime_code, 0)
            row = {
                'regime_code': regime_code,
                'n_samples': n_in_regime,
                'ess': ess,
                **treatments
            }
            regime_desc_rows.append(row)
        pd.DataFrame(regime_desc_rows).to_csv(output_dir / 'regime_descriptions.csv', index=False)
        print(f"  - Regime descriptions: {output_dir / 'regime_descriptions.csv'}")
        
        # Save detailed weight table
        weight_detail = diagnostics['regime_data'].copy()
        weight_detail.to_csv(output_dir / 'ipw_weights_by_sample.csv', index=True)
        print(f"  - Sample-level weights: {output_dir / 'ipw_weights_by_sample.csv'}")
    
    # Validate DAG structure
    G = nx.DiGraph()
    G.add_nodes_from(dag.nodes)
    G.add_edges_from(dag.edges)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("DAG contains cycles")
    
    # Create BN model
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(dag.nodes)
    model.add_edges_from(dag.edges)
    
    # Cast to categorical for pgmpy so variables are treated as discrete states.
    # Use data_weighted (filtered data) for ALL nodes to maintain consistency
    data_for_estimation = data_weighted.copy()
    for col in data_for_estimation.columns:
        if col != 'ipw_weight':  # Don't convert the weight column
            data_for_estimation[col] = data_for_estimation[col].astype("category")

    # Fit CPDs node by node
    print(f"\nFitting CPDs:")
    for node in model.nodes():
        parents = model.get_parents(node)
        
        # Use IPW for all requested outcomes.
        use_ipw = node in outcome_vars
        
        if use_ipw:
            print(f"  {node}: IPW-adjusted (parents: {parents})")
            # Fit weighted CPD
            cpd = _fit_weighted_cpd(node, parents, data_weighted, data_weighted['ipw_weight'])
        else:
            print(f"  {node}: Standard {'MLE' if estimator == 'ml' else 'Bayesian'} (parents: {parents})")
            # Standard MLE or Bayesian fit on filtered data
            if estimator == "ml":
                cpd_estimator = MaximumLikelihoodEstimator(model, data_for_estimation)
                cpd = cpd_estimator.estimate_cpd(node)
            else:
                cpd_estimator = BayesianEstimator(model, data_for_estimation)
                cpd = cpd_estimator.estimate_cpd(node, prior_type=prior_type, 
                                                pseudo_counts=pseudo_counts)
        
        model.add_cpds(cpd)
    
    # Validate model
    if not model.check_model():
        raise ValueError("Model CPDs do not satisfy consistency checks")
    
    print(f"✓ All CPDs fitted and validated")
    print(f"{'='*60}\n")
    
    metadata = {
        "n_samples": len(data_weighted),
        "n_nodes": len(dag.nodes),
        "estimator": "stratified_ipw",
        "treatment_vars": treatment_vars,
        "outcome_vars": outcome_vars,
        "confounder_vars": confounder_vars,
    }
    
    return BNFitResult(model=model, estimator="stratified_ipw", metadata=metadata)


def _fit_weighted_cpd(node: str, parents: list, data: pd.DataFrame, weights: pd.Series):
    """Fit a single CPD using weighted frequencies.
    
    Args:
        node: Node name
        parents: List of parent node names
        data: DataFrame with all variables
        weights: Series with sample weights
        
    Returns:
        TabularCPD for the node
    """
    # Get cardinalities and explicit state names for discrete variables.
    node_states = sorted(pd.unique(data[node]))
    node_card = len(node_states)
    parent_states_map = {p: sorted(pd.unique(data[p])) for p in parents}
    parent_cards = [len(parent_states_map[p]) for p in parents] if parents else []
    
    # Create frequency table with weights
    if not parents:
        # No parents: simple weighted frequency
        values = np.zeros(node_card)
        for idx, state in enumerate(node_states):
            mask = data[node] == state
            values[idx] = weights[mask].sum()
        # Normalize
        values /= values.sum()
        values_reshaped = values.reshape(-1, 1)
    else:
        # With parents: conditional weighted frequency
        import itertools
        parent_states = [parent_states_map[p] for p in parents]
        n_parent_configs = int(np.prod(parent_cards))
        
        values = np.zeros((node_card, n_parent_configs))
        
        for config_idx, parent_config in enumerate(itertools.product(*parent_states)):
            # Filter to rows matching this parent configuration
            mask = np.ones(len(data), dtype=bool)
            for p_idx, (parent, parent_state) in enumerate(zip(parents, parent_config)):
                mask &= (data[parent] == parent_state)
            
            # Compute weighted frequencies for this configuration
            if mask.sum() > 0:
                for node_idx, node_state in enumerate(node_states):
                    node_mask = mask & (data[node] == node_state)
                    values[node_idx, config_idx] = weights[node_mask].sum()
                
                # Normalize column
                col_sum = values[:, config_idx].sum()
                if col_sum > 0:
                    values[:, config_idx] /= col_sum
                else:
                    # Uniform if no data
                    values[:, config_idx] = 1.0 / node_card
            else:
                # Uniform if no data for this parent configuration
                values[:, config_idx] = 1.0 / node_card
        
        values_reshaped = values
    
    # Create CPD
    cpd = TabularCPD(
        variable=node,
        variable_card=node_card,
        values=values_reshaped,
        evidence=parents if parents else None,
        evidence_card=parent_cards if parents else None,
        state_names={
            node: node_states,
            **({p: parent_states_map[p] for p in parents} if parents else {}),
        },
    )
    
    return cpd


def _export_cpd_parameters_csv(model, csv_path: Path) -> None:
    """Export all CPD parameters as a flat CSV table."""
    rows = []
    for cpd in model.get_cpds():
        child = cpd.variable
        parents = list(cpd.variables[1:])
        state_names = cpd.state_names or {}
        child_states = state_names.get(child, list(range(cpd.variable_card)))
        values = cpd.get_values()

        if parents:
            parent_states = [
                state_names.get(parent, list(range(cpd.cardinality[idx + 1])))
                for idx, parent in enumerate(parents)
            ]
            for col_idx, parent_config in enumerate(itertools.product(*parent_states)):
                for row_idx, child_state in enumerate(child_states):
                    row = {
                        "node": child,
                        "state": child_state,
                        "probability": float(values[row_idx, col_idx]),
                    }
                    for parent, parent_state in zip(parents, parent_config):
                        row[parent] = parent_state
                    rows.append(row)
        else:
            for row_idx, child_state in enumerate(child_states):
                rows.append(
                    {
                        "node": child,
                        "state": child_state,
                        "probability": float(values[row_idx, 0]),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)



def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a Bayesian network from CSV data or MongoDB.")
    
    # Input mode arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--data", help="Path to CSV data file.")
    input_group.add_argument("--mongodb", action="store_true", help="Load data from MongoDB using config.")
    
    # Configuration and other arguments
    parser.add_argument("--config", help="Path to configuration file (YAML or JSON). Required for --mongodb.")
    parser.add_argument("--dag", required=True, help="Path to DAG JSON.")
    parser.add_argument("--output", required=True, help="Path to save the fitted BN (pickle).")
    parser.add_argument("--estimator", choices=["ml", "bayes"], default="ml")
    parser.add_argument("--prior", default="BDeu", help="Bayesian prior type (if estimator=bayes).")
    parser.add_argument("--pseudo-counts", type=float, default=1.0, help="Pseudo-counts for Bayesian estimator.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction of data to use for test set (default: 0.2).")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for train/test split and any optional rebalancing (default: 42).")
    parser.add_argument("--rebalance", action="store_true", help="Apply stratified rebalancing on local_recurrence.")
    parser.add_argument("--save-test", help="Path to save test set (optional, for validation use).")
    parser.add_argument("--stratified-cpds", action="store_true", help="Use IPW-adjusted CPDs for outcome nodes to handle confounding by indication.")
    parser.add_argument("--treatment-vars", nargs="+", default=["surgery", "chemotherapy", "radiotherapy"], help="Treatment variable names for stratified CPDs.")
    parser.add_argument("--outcome-vars", nargs="+", default=["metastasis", "local_recurrence", "DOD", "AWD", "NED"], help="Outcome variable names for stratified CPDs.")
    parser.add_argument("--confounder-vars", nargs="+", default=["general.age", "tumor_characteristics.biopsy_grading", "tumor_characteristics.initial_size", "tumor_characteristics.extremity_tumor", "tumor_characteristics.metastasis_present_at_diagnosis"], help="Baseline confounder variables for propensity weighting.")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        config = _load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

    # Load data based on input mode
    if args.mongodb:
        if not config:
            parser.error("--config is required when using --mongodb")
        data = _load_data_from_mongodb(config)
        # Apply feature selection from config if available
        if config.get("features"):
            data = _apply_feature_selection(data, config)
    elif args.data:
        print(f"Loading data from CSV: {args.data}")
        data = pd.read_csv(args.data)
        print(f"Loaded {len(data)} rows from CSV")
    
    # Load DAG
    dag = DAGSerializer.load_json(args.dag)
    dag_nodes = list(dag.nodes)
    missing_cols = [node for node in dag_nodes if node not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in data: {missing_cols}")

    # Keep only DAG columns and drop nulls before split.
    data = data[dag_nodes].copy()
    initial_len = len(data)
    data = data.dropna()
    if len(data) < initial_len:
        print(f"Dropped {initial_len - len(data)} rows with null values before split")
    
    # Split into train and test sets BEFORE rebalancing
    print(f"\n{'='*60}")
    print(f"Splitting data into train/test sets...")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"Test size: {args.test_size:.1%}")
    print(f"Random seed: {args.random_seed}")
    
    train_data, test_data = train_test_split(
        data,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=data['local_recurrence'] if 'local_recurrence' in data.columns else None
    )
    
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Clean/discretize using train-fitted bin edges to avoid test leakage.
    binning_config = config.get('binning', {})
    train_data, learned_bin_edges = _validate_and_clean_data(
        train_data,
        dag_nodes,
        binning_config=binning_config,
        fit_binning=True,
    )
    test_data, _ = _validate_and_clean_data(
        test_data,
        dag_nodes,
        binning_config=binning_config,
        fit_binning=False,
        bin_edges_map=learned_bin_edges,
    )

    # Apply rebalancing ONLY to training set (never to test set).
    if args.rebalance and args.stratified_cpds:
        raise ValueError(
            "Do not use --rebalance with --stratified-cpds for causal/IPW BN fitting."
        )
    if args.rebalance:
        train_data = _apply_rebalancing(train_data, target_column='local_recurrence', random_seed=args.random_seed)
    
    # Save test set if path provided
    if args.save_test:
        test_path = Path(args.save_test)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data.to_pickle(args.save_test)
        print(f"\nTest set saved to: {args.save_test}")
    
    # Fit Bayesian network on TRAINING set only
    print(f"\n{'='*60}")
    print(f"Fitting Bayesian network with {args.estimator} estimator...")
    if args.stratified_cpds:
        print(f"Using STRATIFIED CPDs with IPW adjustment for outcomes")
    print(f"{'='*60}")
    print(f"Training data shape: {train_data.shape}")
    print(f"DAG nodes: {sorted(dag.nodes)}")
    print(f"DAG edges: {len(dag.edges)}")
    
    if args.stratified_cpds:
        # Use stratified CPD fitting with IPW
        # Prepare output directory for diagnostics
        output_path = Path(args.output)
        diagnostics_dir = output_path.parent / 'ipw_diagnostics'
        
        result = fit_bn_with_stratified_cpds(
            train_data,
            dag,
            treatment_vars=args.treatment_vars,
            outcome_vars=args.outcome_vars,
            confounder_vars=args.confounder_vars,
            estimator=args.estimator,
            prior_type=args.prior,
            pseudo_counts=args.pseudo_counts,
            output_dir=diagnostics_dir,
        )
    else:
        # Standard fitting
        result = fit_bayesian_network(
            train_data,
            dag,
            estimator=args.estimator,
            prior_type=args.prior,
            pseudo_counts=args.pseudo_counts,
        )
    
    # Save the fitted model
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_bayesian_network(result.model, args.output)
    
    print(f"\n{'='*60}")
    print(f"✓ Bayesian network successfully fitted and saved!")
    print(f"{'='*60}")
    print(f"Output: {args.output}")
    print(f"Nodes: {len(result.model.nodes())}")
    print(f"Edges: {len(result.model.edges())}")
    print(f"CPDs: {len(result.model.get_cpds())}")

    cpd_csv_path = output_path.with_name(f"{output_path.stem}_cpd_parameters.csv")
    _export_cpd_parameters_csv(result.model, cpd_csv_path)
    print(f"CPD parameter table CSV: {cpd_csv_path}")

    # Print learned CPD parameters for inspection/debugging.
    print("\nLearned CPDs:")
    for cpd in result.model.get_cpds():
        print(cpd)
        print("-" * 80)

    cpd_dod = result.model.get_cpds("DOD")
    if cpd_dod is not None:
        print("\nCPD for DOD:")
        print(cpd_dod)
        vals = cpd_dod.get_values()
        print(vals)
        print("evidence:", cpd_dod.variables[1:])
        print("state_names:", cpd_dod.state_names)
    else:
        print("\nCPD for DOD not found in model.")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
