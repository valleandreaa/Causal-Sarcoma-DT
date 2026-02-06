"""Demo script for structure learning + Bayesian network fitting with visualization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dtcygan.bn import fit_bayesian_network, save_bayesian_network
from dtcygan.causal_pipeline import CausalDiscoveryPipeline
from dtcygan.constraints import ConstraintSet, EdgeConstraints, TierConstraints
from dtcygan.data_bundle import DataBundle
from dtcygan.data_schema import DataSchema
from dtcygan.dag_plot import DAGPlotter
from dtcygan.learning.pc_stable import PCStableLearner


FEATURES: List[str] = [
    "gender",
    "institution",
    "biopsy_grading",
    "who_diagnosis_code",
    "anatomic_region_code",
    "anatomic_region_grouping",
    "anatomic_region_side",
    "metastasis_present_at_diagnosis",
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


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


def build_constraints() -> ConstraintSet:
    """Define domain knowledge constraints for causal structure."""
    edge_constraints = EdgeConstraints(
        required_edges={
            ("biopsy_grading", "who_diagnosis_code"),
            ("who_diagnosis_code", "metastasis_present_at_diagnosis"),
        },
        forbidden_edges={
            ("metastasis_present_at_diagnosis", "gender"),
            ("metastasis_present_at_diagnosis", "institution"),
        },
    )
    tiers: Dict[str, int] = {
        "gender": 0,
        "institution": 0,
        "biopsy_grading": 1,
        "who_diagnosis_code": 1,
        "anatomic_region_code": 2,
        "anatomic_region_grouping": 2,
        "anatomic_region_side": 2,
        "metastasis_present_at_diagnosis": 3,
    }
    tier_constraints = TierConstraints(node_to_tier=tiers, allow_intra_tier_edges=False)
    return ConstraintSet(edge_constraints=edge_constraints, tier_constraints=tier_constraints)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Demo: Learn structure + fit Bayesian network with visualization."
    )
    parser.add_argument("--rows", type=int, default=500, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap DAGs to sample.")
    parser.add_argument("--estimator", choices=["ml", "bayes"], default="ml", help="BN estimator type.")
    parser.add_argument("--prior", default="BDeu", help="Bayesian prior type (if estimator=bayes).")
    parser.add_argument("--pseudo-counts", type=float, default=1.0, help="Pseudo-counts for Bayesian estimator.")
    parser.add_argument("--output-dir", default="./output", help="Directory to save outputs.")
    parser.add_argument("--no-plot", action="store_true", help="Skip DAG visualization (plots by default).")
    
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data
    print("== Generating synthetic data ==")
    df = generate_random_dataframe(rows=args.rows, seed=args.seed)
    schema = DataSchema.from_dataframe(df)
    bundle = DataBundle(df=df, schema=schema)
    print(f"Generated {len(df)} rows with {len(df.columns)} features")
    print("\nData preview:")
    print(df.head())

    # Save generated data
    data_path = output_dir / "synthetic_data.csv"
    df.to_csv(data_path, index=False)
    print(f"\nSaved data to: {data_path}")

    # Build constraints
    constraints = build_constraints()
    print("\n== Constraint summary ==")
    print("Required edges:", sorted(constraints.edge_constraints.required_edges))
    print("Forbidden edges:", sorted(constraints.edge_constraints.forbidden_edges))
    if constraints.tier_constraints:
        print("Tier ordering:", constraints.tier_constraints.node_to_tier)

    # Learn structure
    print("\n== Learning causal structure ==")
    learner = PCStableLearner(alpha=0.05)
    pipeline = CausalDiscoveryPipeline(
        learner=learner,
        constraints=constraints,
        n_boot=args.bootstrap,
    )
    ensemble = pipeline.fit(bundle)

    print(f"\nLearned {len(ensemble.dags)} DAG(s)")
    for idx, dag in enumerate(ensemble.dags, start=1):
        print(f"\nGraph {idx}:")
        print(f"  Nodes: {len(dag.nodes)}")
        print(f"  Edges: {len(dag.edges)}")
        print(f"  Edge list: {sorted(dag.edges)}")

    if ensemble.edge_frequencies:
        print("\n== Bootstrap edge frequencies ==")
        for edge, freq in sorted(ensemble.edge_frequencies.items()):
            print(f"  {edge}: {freq:.2f}")

    # Fit Bayesian Network on the learned DAG
    print("\n== Fitting Bayesian Network ==")
    primary_dag = ensemble.dags[0]
    
    bn_result = fit_bayesian_network(
        df,
        primary_dag,
        estimator=args.estimator,
        prior_type=args.prior,
        pseudo_counts=args.pseudo_counts,
    )
    
    print(f"Fitted BN with estimator: {args.estimator}")
    print(f"Model nodes: {sorted(bn_result.model.nodes())}")
    print(f"Model edges: {sorted(bn_result.model.edges())}")
    
    # Save the fitted Bayesian Network
    bn_path = output_dir / "fitted_bn.pkl"
    save_bayesian_network(bn_result.model, str(bn_path))
    print(f"\nSaved Bayesian Network to: {bn_path}")

    # Display CPD information
    print("\n== Conditional Probability Distributions ==")
    for node in sorted(bn_result.model.nodes()):
        cpd = bn_result.model.get_cpds(node)
        if cpd is not None:
            parents = cpd.variables[1:] if len(cpd.variables) > 1 else []
            print(f"\n{node} | {parents if parents else 'no parents'}")
            print(f"  Shape: {cpd.values.shape}")
            if cpd.values.size <= 20:  # Only print small CPDs
                print(f"  Values:\n{cpd.values}")

    # Plot DAG
    if not args.no_plot:
        print("\n== Generating DAG visualization ==")
        try:
            plotter = DAGPlotter()
            plot_path = output_dir / "dag_plot"
            plotter.plot(
                primary_dag, 
                output_path=plot_path, 
                title="Learned Causal Structure + Fitted BN",
                figsize=(12, 8)
            )
            print(f"Saved DAG plot to: {plot_path}.png and {plot_path}.pdf")
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")

    print("\n== Demo complete ==")
    print(f"All outputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
