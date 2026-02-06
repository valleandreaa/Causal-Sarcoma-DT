"""CLI: Fit a Bayesian network given a learned DAG."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from dtcygan.bn import fit_bayesian_network, save_bayesian_network
from dtcygan.data_schema import DataSchema
from dtcygan.io import DAGSerializer


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


def _validate_and_clean_data(df: pd.DataFrame, dag_nodes: list[str]) -> pd.DataFrame:
    """Validate and clean data for Bayesian network fitting."""
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
        
        # Now ensure everything is an integer type
        if pd.api.types.is_numeric_dtype(df[col]):
            if not pd.api.types.is_integer_dtype(df[col]):
                # Convert floats to ints (ensure they're whole numbers first)
                if (df[col] == df[col].round()).all():
                    df[col] = df[col].round().astype(int)
                    print(f"  {col}: converted float to int")
                else:
                    # If not whole numbers, discretize
                    print(f"  Warning: {col} has non-integer values, discretizing...")
                    df[col] = pd.cut(df[col], bins=10, labels=False, duplicates='drop')
            else:
                print(f"  {col}: already integer")
        elif pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
            print(f"  {col}: converted bool to int")
        else:
            # Fallback: convert to categorical codes
            df[col] = pd.Categorical(df[col]).codes
            print(f"  {col}: converted to categorical codes (fallback)")
    
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
    
    return df


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
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for train/test split and rebalancing (default: 42).")
    parser.add_argument("--rebalance", action="store_true", help="Apply stratified rebalancing on local_recurrence.")
    parser.add_argument("--save-test", help="Path to save test set (optional, for validation use).")

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
    
    # Validate and clean data to match DAG nodes
    data = _validate_and_clean_data(data, list(dag.nodes))
    
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
    
    # Apply rebalancing ONLY to training set (never to test set)
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
    print(f"{'='*60}")
    print(f"Training data shape: {train_data.shape}")
    print(f"DAG nodes: {sorted(dag.nodes)}")
    print(f"DAG edges: {len(dag.edges)}")
    
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
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()