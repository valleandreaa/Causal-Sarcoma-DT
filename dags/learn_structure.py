"""CLI: Learn a DAG structure with pgmpy and optional constraints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from pgmpy.estimators import PC, ExpertKnowledge
from pgmpy.models import BayesianNetwork
from pymongo import MongoClient

from dtcygan.constraints import ConstraintParser, ConstraintSet, EdgeConstraints, TierConstraints
from dtcygan.data_bundle import DataBundle
from dtcygan.data_schema import DataSchema
from dtcygan.io import DAGSerializer
from dtcygan.learning import PCStableLearner
from dtcygan.structure import StructureLearningResult, learn_dag


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def generate_random_dataframe(rows: int, seed: int) -> pd.DataFrame:
    """Generate synthetic medical data with causal relationships."""
    rng = np.random.default_rng(seed)

    gender = rng.integers(0, 2, size=rows)
    institution = rng.integers(0, 3, size=rows)

    biopsy_grading = np.clip(
        1 + gender + (institution == 2).astype(int) + rng.normal(0, 0.6, size=rows),
        1,
        4,
    ).round()

    who_diagnosis_code = np.clip(
        100 + 10 * biopsy_grading + rng.normal(0, 4, size=rows),
        90,
        160,
    ).round()

    anatomic_region_code = np.clip(
        200 + 4 * institution + rng.normal(0, 3, size=rows),
        190,
        220,
    ).round()
    anatomic_region_grouping = (anatomic_region_code // 5).astype(int)
    anatomic_region_side = rng.integers(0, 2, size=rows)

    metastasis_logit = (
        -2.0
        + 0.6 * biopsy_grading
        + 0.03 * (who_diagnosis_code - 100)
        + 0.2 * (anatomic_region_grouping - anatomic_region_grouping.mean())
        + rng.normal(0, 0.3, size=rows)
    )
    metastasis_present = (rng.random(size=rows) < _sigmoid(metastasis_logit)).astype(int)

    return pd.DataFrame(
        {
            "gender": gender,
            "institution": institution,
            "biopsy_grading": biopsy_grading,
            "who_diagnosis_code": who_diagnosis_code,
            "anatomic_region_code": anatomic_region_code,
            "anatomic_region_grouping": anatomic_region_grouping,
            "anatomic_region_side": anatomic_region_side,
            "metastasis_present_at_diagnosis": metastasis_present,
        }
    )


def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


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


def _load_data_from_csv(csv_path: str, config: dict) -> pd.DataFrame:
    """Load data from CSV file."""
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    print(f"Loading data from CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
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


def _build_constraints_from_config(config: dict) -> ConstraintSet | None:
    """Build constraint set from configuration."""
    constraints_config = config.get("constraints", {})
    
    if not constraints_config:
        return None
    
    # Build edge constraints
    edge_config = constraints_config.get("edges", {})
    required_edges = set()
    forbidden_edges = set()
    
    for edge in edge_config.get("required", []):
        if isinstance(edge, list) and len(edge) == 2:
            required_edges.add(tuple(edge))
    
    for edge in edge_config.get("forbidden", []):
        if isinstance(edge, list) and len(edge) == 2:
            forbidden_edges.add(tuple(edge))
    
    edge_constraints = EdgeConstraints(
        required_edges=required_edges,
        forbidden_edges=forbidden_edges
    ) if (required_edges or forbidden_edges) else None
    
    # Build tier constraints
    tier_config = constraints_config.get("tiers", {})
    tier_constraints = None
    
    if tier_config.get("node_to_tier"):
        tier_constraints = TierConstraints(
            node_to_tier=tier_config["node_to_tier"],
            allow_intra_tier_edges=tier_config.get("allow_intra_tier_edges", False)
        )
    
    if edge_constraints or tier_constraints:
        return ConstraintSet(
            edge_constraints=edge_constraints,
            tier_constraints=tier_constraints
        )
    
    return None


def _write_records(path: str, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".json":
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    else:
        pd.DataFrame(records).to_csv(output_path, index=False)


def _constraint_conflicts(dag, constraints) -> list[dict]:
    conflicts = []
    for src, dst in sorted(constraints.edge_constraints.required_edges):
        if not dag.has_edge(src, dst):
            conflicts.append(
                {
                    "test_type": "constraint_conflict",
                    "x": src,
                    "y": dst,
                    "conditioning_set": [],
                    "expected": "required_edge",
                    "decision": "missing_required_edge",
                }
            )
    for src, dst in sorted(constraints.edge_constraints.forbidden_edges):
        if dag.has_edge(src, dst):
            conflicts.append(
                {
                    "test_type": "constraint_conflict",
                    "x": src,
                    "y": dst,
                    "conditioning_set": [],
                    "expected": "forbidden_edge",
                    "decision": "forbidden_edge_present",
                }
            )
    if constraints.tier_constraints is not None:
        for src, dst in sorted(dag.edges):
            if not constraints.tier_constraints.is_edge_allowed(src, dst):
                conflicts.append(
                    {
                        "test_type": "constraint_conflict",
                        "x": src,
                        "y": dst,
                        "conditioning_set": [],
                        "expected": "tier_order",
                        "decision": "tier_violation",
                    }
                )
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn a DAG structure from CSV data or MongoDB.")
    
    # Input mode arguments
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--data", help="Path to CSV data file.")
    input_group.add_argument("--mongodb", action="store_true", help="Load data from MongoDB using config.")
    input_group.add_argument("--generate-rows", type=int, help="Generate random synthetic data with this many rows.")
    
    # Configuration file
    parser.add_argument("--config", help="Path to configuration file (YAML or JSON) containing features, constraints, and settings.")
    
    # Legacy arguments (kept for backward compatibility)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data generation (default: 42).")
    parser.add_argument("--output", required=True, help="Path to save DAG JSON.")
    parser.add_argument("--image", help="Path to save DAG visualization (e.g., dag.png).")
    parser.add_argument("--method", default="pc", choices=["pc", "pc_stable", "pc_pgmpy", "ges"], help="Structure learning algorithm.")
    parser.add_argument("--edge-constraints", help="YAML/JSON path with required/forbidden edges (overrides config).")
    parser.add_argument("--tier-constraints", help="YAML/JSON path with tier constraints (overrides config).")
    parser.add_argument("--whitelist", help="Comma-separated list of nodes to include (overrides config).")
    parser.add_argument("--blacklist", help="Comma-separated list of nodes to exclude (overrides config).")
    parser.add_argument("--variant", help="PC variant (e.g., stable).")
    parser.add_argument("--ci-test", dest="ci_test", help="Conditional independence test for PC.")
    parser.add_argument("--significance", type=float, help="Significance level for PC.")
    parser.add_argument("--score", help="Score function for GES.")
    parser.add_argument("--pc-audit-log", help="Path to save all CI tests evaluated during PC (CSV or JSON).")
    parser.add_argument("--pc-validation-log", help="Path to save post-hoc validation tests (CSV or JSON).")
    parser.add_argument("--pc-edge-strength", action="store_true", help="Include edge strength summaries in validation log.")
    parser.add_argument("--pc-no-collider-tests", action="store_true", help="Disable collider support tests in validation log.")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    if args.config:
        config = _load_config(args.config)
        print(f"Loaded configuration from: {args.config}")

    # Load data based on input mode
    if args.generate_rows:
        data = generate_random_dataframe(args.generate_rows, args.seed)
        print(f"Generated {args.generate_rows} rows of synthetic data with seed {args.seed}")
    elif args.mongodb:
        if not config:
            parser.error("--config is required when using --mongodb")
        data = _load_data_from_mongodb(config)
    elif args.data:
        data = _load_data_from_csv(args.data, config)
    else:
        parser.error("One of --data, --mongodb, or --generate-rows must be specified.")
    
    # Apply feature selection from config
    if config.get("features"):
        data = _apply_feature_selection(data, config)
    
    # Convert categorical string columns to numeric codes (for both pgmpy and causal-learn)
    variable_types = config.get("variable_types") if config else None
    
    # Apply custom mappings from config
    mappings = config.get("mappings", {})
    if mappings:
        print("\nApplying custom mappings:")
        for col, mapping in mappings.items():
            if col in data.columns:
                unique_before = data[col].unique()
                print(f"  Column '{col}' unique values before mapping: {unique_before}")
                data[col] = data[col].map(mapping)
                
                # Check for unmapped values (NaN after mapping)
                unmapped_mask = data[col].isna()
                if unmapped_mask.any():
                    unmapped_values = data.loc[unmapped_mask, col].unique()
                    print(f"    WARNING: Unmapped values found: {unmapped_values}")
                    print(f"    Please add these values to the mapping in config file")
                
                print(f"  Mapped column '{col}': {mapping}")
    
    # First, convert ALL string/object columns to numeric codes (safety measure)
    print("\nChecking for string columns to convert:")
    for col in data.columns:
        if data[col].dtype == 'object' or data[col].dtype.name == 'category':
            print(f"  Converting column '{col}' (dtype: {data[col].dtype}) to numeric codes")
            print(f"    Unique values: {data[col].unique()}")
            data[col] = pd.Categorical(data[col]).codes
    
    # Then apply variable_types if specified (for additional control)
    if variable_types:
        for col, dtype in variable_types.items():
            if col in data.columns and dtype in ['categorical', 'binary']:
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    print(f"  Additional conversion for '{col}' specified in variable_types")
                    data[col] = pd.Categorical(data[col]).codes
    
    print(f"\nFinal data types:\n{data.dtypes}\n")
    
    # Build constraints from config
    constraints = _build_constraints_from_config(config) if config else None
    
    # Override with command-line constraints if provided (legacy support)
    if args.edge_constraints or args.tier_constraints or args.whitelist or args.blacklist:
        print("Using command-line constraints (overriding config constraints)")
        constraints = ConstraintParser.load_constraint_set(
            args.edge_constraints,
            args.tier_constraints,
            node_whitelist=_parse_list(args.whitelist),
            node_blacklist=_parse_list(args.blacklist),
        )
    
    # Use algorithm settings from config if available
    algorithm_config = config.get("algorithm", {})
    method = algorithm_config.get("method", args.method)
    
    # Prepare algorithm kwargs
    kwargs = {}
    if args.variant or algorithm_config.get("variant"):
        kwargs["variant"] = args.variant or algorithm_config.get("variant")
    if args.ci_test or algorithm_config.get("ci_test"):
        kwargs["ci_test"] = args.ci_test or algorithm_config.get("ci_test")
    if args.significance is not None or algorithm_config.get("significance_level") is not None:
        kwargs["significance_level"] = args.significance if args.significance is not None else algorithm_config.get("significance_level")
    if args.score or algorithm_config.get("scoring_method"):
        kwargs["scoring_method"] = args.score or algorithm_config.get("scoring_method")

    use_pc_stable = method == "pc_stable" or (
        method == "pc" and (args.pc_audit_log or args.pc_validation_log)
    )
    
    use_pc_pgmpy = method == "pc_pgmpy"
    
    if use_pc_stable and method == "pc":
        print("Using PC-Stable implementation to capture CI audit trail.")
    
    if use_pc_pgmpy:
        print("Using pgmpy.estimators.PC for structure learning.")
        
        # Apply node filtering if constraints exist
        if constraints is not None:
            data = data.loc[:, constraints.filter_nodes(list(data.columns))]
        
        # Prepare PC estimator arguments
        ci_test = kwargs.get("ci_test", "chi_square")
        significance_level = kwargs.get("significance_level", 0.05)
        
        print(f"CI test: {ci_test}, significance level: {significance_level}")
        
        # Build ExpertKnowledge object for pgmpy using the repo helper
        # (avoids calling version-specific ExpertKnowledge methods directly)
        from dtcygan.structure import _expert_knowledge as _build_expert
        nodes_in_data = list(data.columns)
        expert_knowledge = _build_expert(constraints, nodes_in_data)
        if constraints is not None:
            if expert_knowledge is None:
                print("\nNo expert knowledge produced (no required/forbidden/temporal constraints found)")
            else:
                print("\nExpertKnowledge object prepared from config constraints")
        
        # Create PC estimator
        pc_estimator = PC(data=data)
        
        print(f"\nRunning PC algorithm with ExpertKnowledge...")
        
        # Estimate structure with expert knowledge
        learned_model = pc_estimator.estimate(
            variant="stable",
            ci_test=ci_test,
            significance_level=significance_level,
            expert_knowledge=expert_knowledge if constraints else None,
        )
        
        # Convert pgmpy model to dtcygan DAG structure
        from dtcygan.dag_spec import DAGSpec
        dag = DAGSpec(
            nodes=list(learned_model.nodes()),
            edges=set(learned_model.edges())  # Convert to set, not list
        )
        
        # Verify constraints were respected
        print(f"\nVerifying constraint compliance in learned DAG...")
        violations_found = False
        
        if constraints is not None:
            # Check for tier violations
            if constraints.tier_constraints:
                tier_map = constraints.tier_constraints.node_to_tier
                for src, dst in dag.edges:
                    src_tier = tier_map.get(src)
                    dst_tier = tier_map.get(dst)
                    if src_tier is not None and dst_tier is not None and src_tier > dst_tier:
                        print(f"  ⚠ TIER VIOLATION: {src} (tier {src_tier}) → {dst} (tier {dst_tier})")
                        violations_found = True
            
            # Check for forbidden edges
            if constraints.edge_constraints and constraints.edge_constraints.forbidden_edges:
                for src, dst in dag.edges:
                    if (src, dst) in constraints.edge_constraints.forbidden_edges:
                        print(f"  ⚠ FORBIDDEN EDGE: {src} → {dst}")
                        violations_found = True
            
            # Check for missing required edges
            if constraints.edge_constraints and constraints.edge_constraints.required_edges:
                for src, dst in constraints.edge_constraints.required_edges:
                    if not dag.has_edge(src, dst):
                        print(f"  ⚠ MISSING REQUIRED EDGE: {src} → {dst}")
                        violations_found = True
        
        if violations_found:
            print(f"\n⚠ pgmpy did not fully respect constraints - applying post-processing...")
            
            # Post-process: Remove forbidden edges
            removed_count = 0
            if constraints.tier_constraints:
                tier_map = constraints.tier_constraints.node_to_tier
                for src, dst in list(dag.edges):
                    src_tier = tier_map.get(src)
                    dst_tier = tier_map.get(dst)
                    if src_tier is not None and dst_tier is not None and src_tier > dst_tier:
                        dag.remove_edge(src, dst)
                        removed_count += 1
                        print(f"  Removed tier violation: {src} (tier {src_tier}) → {dst} (tier {dst_tier})")
            
            if constraints.edge_constraints and constraints.edge_constraints.forbidden_edges:
                for src, dst in list(dag.edges):
                    if (src, dst) in constraints.edge_constraints.forbidden_edges:
                        if dag.has_edge(src, dst):
                            dag.remove_edge(src, dst)
                            removed_count += 1
                            print(f"  Removed forbidden: {src} → {dst}")
            
            # Post-process: Add missing required edges
            added_count = 0
            if constraints.edge_constraints and constraints.edge_constraints.required_edges:
                for src, dst in constraints.edge_constraints.required_edges:
                    if not dag.has_edge(src, dst):
                        dag.add_edge(src, dst)
                        added_count += 1
                        print(f"  Added required: {src} → {dst}")
            
            print(f"\n✓ Post-processing complete: removed {removed_count}, added {added_count} edges")
        else:
            print(f"✓ All constraints respected during learning")
        
        result = StructureLearningResult(
            dag=dag,
            estimator="pc_pgmpy",
            metadata={
                "method": "pc_pgmpy",
                "ci_test": ci_test,
                "significance_level": significance_level,
                "n_edges": len(dag.edges),
                "n_nodes": len(dag.nodes),
            }
        )

    elif use_pc_stable:
        if constraints is not None:
            data = data.loc[:, constraints.filter_nodes(list(data.columns))]
        schema = DataSchema.from_dataframe(data)
        bundle = DataBundle(df=data, schema=schema)
        learner = PCStableLearner(variable_types=variable_types)
        if kwargs.get("ci_test"):
            learner.set_ci_test(kwargs["ci_test"])
        if kwargs.get("significance_level") is not None:
            learner.set_alpha(kwargs["significance_level"])
        dag = learner.fit(bundle, constraints)
        result = StructureLearningResult(dag=dag, estimator="pc_stable", metadata=dag.provenance)

        if args.pc_audit_log:
            _write_records(args.pc_audit_log, learner.get_ci_test_log())
            print(f"PC audit log saved to: {args.pc_audit_log}")
        if args.pc_validation_log:
            validation = learner.build_validation_tests(
                dag,
                include_edge_strength_summary=args.pc_edge_strength,
                include_collider_support=not args.pc_no_collider_tests,
            )
            if constraints is not None:
                validation.extend(_constraint_conflicts(dag, constraints))
            _write_records(args.pc_validation_log, validation)
            print(f"PC validation tests saved to: {args.pc_validation_log}")
    else:
        result = learn_dag(data, method=method, constraints=constraints, **kwargs)
    
    # Print learned structure
    print(f"\n{'='*60}")
    print(f"Learned DAG Structure ({args.method.upper()} algorithm)")
    print(f"{'='*60}")
    print(f"Nodes: {sorted(result.dag.nodes)}")
    print(f"Number of edges: {len(result.dag.edges)}")
    print(f"\nEdges (parent → child):")
    for parent, child in sorted(result.dag.edges):
        print(f"  {parent} → {child}")
    print(f"{'='*60}\n")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    DAGSerializer.save_json(result.dag, str(output_path))
    print(f"DAG saved to: {output_path}")
    
    # Generate visualization if requested
    if args.image:
        G = nx.DiGraph()
        G.add_nodes_from(result.dag.nodes)
        G.add_edges_from(result.dag.edges)
        
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50, seed=args.seed)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                node_size=3000, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                arrows=True, arrowsize=20, 
                                arrowstyle='->', width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, 
                                font_weight='bold')
        
        plt.title(f"Learned DAG Structure ({args.method.upper()} algorithm)", 
                    fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        image_path = Path(args.image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(image_path), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"DAG visualization saved to: {image_path}")


if __name__ == "__main__":
    main()
