"""Demo script for constraint-aware causal discovery on random data."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

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


def format_dag_edges(dag, constraints: ConstraintSet) -> List[str]:
    lines: List[str] = []
    for src, dst in sorted(dag.edges):
        tags: List[str] = []
        if (src, dst) in constraints.edge_constraints.required_edges:
            tags.append("required")
        if (src, dst) in constraints.edge_constraints.forbidden_edges:
            tags.append("forbidden")
        if constraints.tier_constraints and not constraints.tier_constraints.is_edge_allowed(src, dst):
            tags.append("tier_blocked")
        tag_str = f" ({', '.join(tags)})" if tags else ""
        lines.append(f"{src} -> {dst}{tag_str}")
    return lines


def save_dag_images(dags, output_dir: str | Path, layout: str) -> List[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images: List[Path] = []
    for idx, dag in enumerate(dags, start=1):
        base_path = output_path / f"causal_dag_{idx}"
        try:
            png_path, _ = DAGPlotter.plot_to_file(dag, str(base_path), layout=layout)
            images.append(png_path)
        except ImportError as exc:
            print(f"Skipping DAG image export: {exc}")
            break
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a causal discovery demo with constraints.")
    parser.add_argument("--rows", type=int, default=500, help="Number of synthetic rows to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--bootstrap", type=int, default=0, help="Number of bootstrap DAGs to sample.")
    parser.add_argument("--output-dir", type=str, default="imgs/causal_demo", help="Output directory for DAG images.")
    parser.add_argument("--layout", type=str, default="hierarchical", help="DAG layout: hierarchical or spring.")
    args = parser.parse_args()

    df = generate_random_dataframe(rows=args.rows, seed=args.seed)
    schema = DataSchema.from_dataframe(df)
    bundle = DataBundle(df=df, schema=schema)

    constraints = build_constraints()
    learner = PCStableLearner(alpha=0.05)
    pipeline = CausalDiscoveryPipeline(
        learner=learner,
        constraints=constraints,
        n_boot=args.bootstrap,
    )
    ensemble = pipeline.fit(bundle)

    print("== Demo data preview ==")
    print(df.head())
    print("\n== Constraint summary ==")
    print("Required edges:", sorted(constraints.edge_constraints.required_edges))
    print("Forbidden edges:", sorted(constraints.edge_constraints.forbidden_edges))
    print("Tier ordering:", constraints.tier_constraints.node_to_tier if constraints.tier_constraints else {})

    print("\n== Learned DAGs ==")
    for idx, dag in enumerate(ensemble.dags, start=1):
        print(f"Graph {idx}:")
        print("  edges:")
        for line in format_dag_edges(dag, constraints):
            print(f"    - {line}")
        if dag.edge_metadata:
            print("  edge metadata:", dag.edge_metadata)
        if dag.provenance:
            print("  provenance:", dag.provenance)

    if ensemble.edge_frequencies:
        print("\n== Bootstrap edge frequencies ==")
        for edge, freq in sorted(ensemble.edge_frequencies.items()):
            print(f"{edge}: {freq:.2f}")

    image_paths = save_dag_images(ensemble.dags, args.output_dir, args.layout)
    if image_paths:
        print("\n== DAG images saved ==")
        for path in image_paths:
            print(path)


if __name__ == "__main__":
    main()