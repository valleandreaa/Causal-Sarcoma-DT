"""CLI: Fit Surgery+RT Bayesian Network and estimate adjusted effects with DoWhy.

This script starts from a learned DAG JSON (e.g., surgery_rt_dag.json), fits a
Discrete Bayesian Network with pgmpy, and then estimates treatment effects with
DoWhy using propensity-score weighting to address confounding by indication.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
from dotenv import load_dotenv
from dowhy import CausalModel
from pgmpy.estimators import BayesianEstimator
from pgmpy.models import DiscreteBayesianNetwork

from dtcygan.data_processing.data_processing import DataProcessing
from matplotlib.patches import Patch


def _write_json(path: str, payload: dict | list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_dag(path: str) -> tuple[list[str], list[tuple[str, str]], nx.DiGraph]:
    dag_path = Path(path)
    with dag_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    nodes = payload.get("nodes", [])
    edges = [tuple(edge) for edge in payload.get("edges", [])]

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"Input graph is not a DAG: {path}")

    return nodes, edges, graph


def _to_dot_graph(nodes: list[str], edges: list[tuple[str, str]]) -> str:
    lines = ["digraph {"]
    for node in nodes:
        lines.append(f'  "{node}";')
    for src, dst in edges:
        lines.append(f'  "{src}" -> "{dst}";')
    lines.append("}")
    return "\n".join(lines)


def _safe_discretize(df: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    data = df.copy()
    missing_nodes = [n for n in nodes if n not in data.columns]
    if missing_nodes:
        raise ValueError(f"Missing required DAG columns in data: {missing_nodes}")

    data = data[nodes].copy()

    for col in data.columns:
        if data[col].dtype == "bool":
            data[col] = data[col].astype(int)
            continue

        if data[col].dtype == "object" or str(data[col].dtype).startswith("category"):
            data[col] = pd.Categorical(data[col]).codes

    # pgmpy discrete models assume complete, finite states.
    data = data.dropna().reset_index(drop=True)

    if data.empty:
        raise ValueError("No rows available after dropping missing values.")

    return data


def _fit_bn(
    data: pd.DataFrame,
    nodes: list[str],
    edges: list[tuple[str, str]],
    model_path: str,
    cpd_summary_path: str | None,
    equivalent_sample_size: float,
) -> DiscreteBayesianNetwork:
    bn = DiscreteBayesianNetwork()
    bn.add_nodes_from(nodes)
    bn.add_edges_from(edges)

    bn.fit(
        data[nodes],
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=equivalent_sample_size,
    )

    out_path = Path(model_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(bn, f)

    if cpd_summary_path:
        rows: list[dict] = []
        for cpd in bn.get_cpds():
            evidence = list(cpd.variables[1:])
            state_names = dict(cpd.state_names) if cpd.state_names else {}
            cardinality = {
                var: int(card)
                for var, card in zip(cpd.variables, cpd.cardinality)
            }
            rows.append(
                {
                    "variable": cpd.variable,
                    "evidence": evidence,
                    "cardinality": cardinality,
                    "state_names": state_names,
                }
            )
        Path(cpd_summary_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(cpd_summary_path, index=False)

    return bn


def _is_binary(series: pd.Series) -> bool:
    values = set(series.dropna().unique().tolist())
    return values.issubset({0, 1})


def _estimate_with_ps_weighting(
    model: CausalModel,
    estimand: Any,
    target_units: str,
    min_ps: float,
) -> Any:
    try:
        return model.estimate_effect(
            estimand,
            method_name="backdoor.propensity_score_weighting",
            target_units=target_units,
            method_params={
                "min_ps_score": min_ps,
                "max_ps_score": 1.0 - min_ps,
                "weighting_scheme": "ips_stabilized_weight",
            },
        )
    except TypeError:
        # Compatibility fallback for DoWhy versions with different method params.
        return model.estimate_effect(
            estimand,
            method_name="backdoor.propensity_score_weighting",
            target_units=target_units,
        )


def _bootstrap_ci(values: list[float], ci_level: float = 0.95) -> tuple[float, float] | None:
    if not values:
        return None

    alpha = 1.0 - ci_level
    series = pd.Series(values)
    lower = float(series.quantile(alpha / 2.0))
    upper = float(series.quantile(1.0 - alpha / 2.0))
    return lower, upper


def _apply_deep_only_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    col = "tumor_characteristics.anatomic_region_grouping"
    if col not in df.columns:
        raise ValueError(
            f"Deep-group filtering requires missing column: {col}"
        )

    deep_label = "deep_soft_tissue"
    deep_value: Any = deep_label

    mappings = config.get("mappings", {}) if isinstance(config, dict) else {}
    col_map = mappings.get(col, {}) if isinstance(mappings, dict) else {}
    if isinstance(col_map, dict) and deep_label in col_map:
        deep_value = col_map[deep_label]

    filtered = df.loc[df[col] == deep_value].copy()
    if filtered.empty:
        observed = sorted(df[col].dropna().unique().tolist())
        raise ValueError(
            "No rows after deep-group filtering. "
            f"Expected value: {deep_value}. Observed values: {observed}"
        )

    print(
        "Applied deep-only filter: "
        f"{len(filtered)} / {len(df)} rows retained"
    )
    return filtered


def _apply_not_deep_filter(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    col = "tumor_characteristics.anatomic_region_grouping"
    if col not in df.columns:
        raise ValueError(
            f"--compare-deep requested but required column is missing: {col}"
        )

    deep_label = "deep_soft_tissue"
    deep_value: Any = deep_label

    mappings = config.get("mappings", {}) if isinstance(config, dict) else {}
    col_map = mappings.get(col, {}) if isinstance(mappings, dict) else {}
    if isinstance(col_map, dict) and deep_label in col_map:
        deep_value = col_map[deep_label]

    filtered = df.loc[df[col] != deep_value].copy()
    if filtered.empty:
        observed = sorted(df[col].dropna().unique().tolist())
        raise ValueError(
            "No rows after non-deep filtering. "
            f"Excluded deep value: {deep_value}. Observed values: {observed}"
        )

    print(
        "Applied non-deep filter: "
        f"{len(filtered)} / {len(df)} rows retained"
    )
    return filtered


def _apply_min_initial_size_filter(df: pd.DataFrame, min_initial_size: float) -> pd.DataFrame:
    col = "tumor_characteristics.initial_size"
    if col not in df.columns:
        raise ValueError(
            f"--min-initial-size requested but required column is missing: {col}"
        )

    filtered = df.loc[df[col].notna() & (df[col] >= min_initial_size)].copy()
    if filtered.empty:
        observed_min = float(df[col].min()) if df[col].notna().any() else None
        observed_max = float(df[col].max()) if df[col].notna().any() else None
        raise ValueError(
            "No rows after size filtering. "
            f"Requested {col} >= {min_initial_size}, "
            f"observed range: [{observed_min}, {observed_max}]"
        )

    print(
        "Applied minimum initial-size filter: "
        f"{len(filtered)} / {len(df)} rows retained "
        f"({col} >= {min_initial_size})"
    )
    return filtered


def _attach_subgroup(records: list[dict], subgroup: str) -> list[dict]:
    return [{**r, "subgroup": subgroup} for r in records]


def _build_deep_comparison(deep_records: list[dict], non_deep_records: list[dict]) -> list[dict]:
    deep_by_outcome = {r["outcome"]: r for r in deep_records}
    non_deep_by_outcome = {r["outcome"]: r for r in non_deep_records}

    comparison: list[dict] = []
    for outcome in sorted(set(deep_by_outcome) & set(non_deep_by_outcome)):
        deep_effect = float(deep_by_outcome[outcome]["effect"])
        non_deep_effect = float(non_deep_by_outcome[outcome]["effect"])
        comparison.append(
            {
                "outcome": outcome,
                "deep_effect": deep_effect,
                "non_deep_effect": non_deep_effect,
                "effect_difference_deep_minus_non_deep": deep_effect - non_deep_effect,
                "deep_ci_95": deep_by_outcome[outcome].get("ci_95"),
                "non_deep_ci_95": non_deep_by_outcome[outcome].get("ci_95"),
                "deep_n_rows": deep_by_outcome[outcome].get("n_rows"),
                "non_deep_n_rows": non_deep_by_outcome[outcome].get("n_rows"),
                "target_units": deep_by_outcome[outcome].get("target_units"),
            }
        )

    return comparison


def _select_outcome_records(records: list[dict], outcome_terms: list[str]) -> dict[str, dict]:
    selected: dict[str, dict] = {}
    for term in outcome_terms:
        match = next(
            (
                r
                for r in records
                if term in str(r.get("outcome", "")).lower()
            ),
            None,
        )
        if match is not None:
            selected[term] = match
    return selected


def _save_bootstrap_histograms(
    output_path: str,
    target_units: str,
    records_by_group: dict[str, dict[str, dict]],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Skipping local recurrence plot: matplotlib is not installed. "
            f"Intended output path: {output_path}"
        )
        return

    outcome_labels = {
        "local_recurrence": "Local recurrence",
        "metastasis": "Metastasis",
        "dod": "DOD",
    }

    available_terms = [
        term
        for term in outcome_labels
        if any(term in group_map for group_map in records_by_group.values())
    ]
    if not available_terms:
        print("Skipping histogram plot: no requested outcomes found in records.")
        return

    plot_path = Path(output_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        len(available_terms),
        1,
        figsize=(9, 3.2 * len(available_terms)),
        squeeze=False,
    )

    group_names = list(records_by_group.keys())
    colors = {
        "deep": "#1f77b4",
        "non_deep": "#ff7f0e",
        "overall": "#2ca02c",
    }

    for row_idx, term in enumerate(available_terms):
        ax = axes[row_idx][0]
        plotted_any = False

        for group_name in group_names:
            rec = records_by_group[group_name].get(term)
            if rec is None:
                continue

            values = rec.get("bootstrap_effect_values")
            if isinstance(values, list) and len(values) > 0:
                ax.hist(
                    values,
                    bins=20,
                    alpha=0.45,
                    color=colors.get(group_name, None),
                    density=True,
                )
                plotted_any = True

            effect = float(rec.get("effect", 0.0))
            ax.axvline(
                effect,
                linestyle="--",
                linewidth=2,
                color=colors.get(group_name, "#333333"),
            )
            plotted_any = True

        if not plotted_any:
            ax.text(0.5, 0.5, "No bootstrap data", ha="center", va="center")

        ax.set_title(f"{outcome_labels[term]}")
        ax.set_xlabel("Estimated causal effect")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.2, linestyle=":")

        # Simplified legend: color only for deep/non-deep
        legend_handles = []
        if "deep" in records_by_group:
            legend_handles.append(Patch(facecolor=colors["deep"], alpha=0.45, label="deep"))
        if "non_deep" in records_by_group:
            legend_handles.append(Patch(facecolor=colors["non_deep"], alpha=0.45, label="non-deep"))
        if legend_handles:
            ax.legend(handles=legend_handles, loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    print(f"Bootstrap histogram plot saved to: {plot_path}")


def _default_histogram_plot_path(output_effects_path: str, target_units: str) -> str:
    p = Path(output_effects_path)
    return str(p.with_name(f"{p.stem}_histograms_{target_units}.png"))


def _estimate_dowhy_effects(
    data: pd.DataFrame,
    treatment: str,
    outcomes: list[str],
    nodes: list[str],
    edges: list[tuple[str, str]],
    graph: nx.DiGraph,
    min_ps: float,
    target_units: str,
    bootstrap_samples: int,
    seed: int,
) -> list[dict]:
    if treatment not in data.columns:
        raise ValueError(f"Treatment column not in data: {treatment}")

    if not _is_binary(data[treatment]):
        raise ValueError(
            f"Treatment {treatment} must be binary coded as 0/1 for propensity weighting."
        )
    if data[treatment].nunique() < 2:
        raise ValueError(
            f"Treatment {treatment} has no treated/control variation in this subset."
        )

    dot_graph = _to_dot_graph(nodes, edges)

    confounders = sorted(nx.ancestors(graph, treatment))
    print(f"Identified potential confounders for {treatment}: {confounders}")

    records: list[dict] = []

    for outcome in outcomes:
        if outcome not in data.columns:
            print(f"Skipping outcome not present in data: {outcome}")
            continue

        if not _is_binary(data[outcome]):
            print(f"Skipping non-binary outcome (expected 0/1): {outcome}")
            continue

        print(f"Estimating adjusted causal effect: {treatment} -> {outcome}")

        model = CausalModel(
            data=data,
            treatment=treatment,
            outcome=outcome,
            graph=dot_graph,
            common_causes=confounders,
        )

        estimand = model.identify_effect(proceed_when_unidentifiable=True)

        estimate = _estimate_with_ps_weighting(
            model=model,
            estimand=estimand,
            target_units=target_units,
            min_ps=min_ps,
        )

        bootstrap_effects: list[float] = []
        if bootstrap_samples > 0:
            for b in range(bootstrap_samples):
                sample_df = data.sample(
                    n=len(data),
                    replace=True,
                    random_state=seed + b,
                )

                # Skip resamples without treatment variation.
                if sample_df[treatment].nunique() < 2:
                    continue

                try:
                    sample_model = CausalModel(
                        data=sample_df,
                        treatment=treatment,
                        outcome=outcome,
                        graph=dot_graph,
                        common_causes=confounders,
                    )
                    sample_estimand = sample_model.identify_effect(
                        proceed_when_unidentifiable=True
                    )
                    sample_estimate = _estimate_with_ps_weighting(
                        model=sample_model,
                        estimand=sample_estimand,
                        target_units=target_units,
                        min_ps=min_ps,
                    )
                    bootstrap_effects.append(float(sample_estimate.value))
                except Exception:
                    continue

        ci_95 = _bootstrap_ci(bootstrap_effects, ci_level=0.95)

        records.append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "method": "backdoor.propensity_score_weighting",
                "estimand": str(estimand),
                "effect": float(estimate.value),
                "confounders": confounders,
                "min_ps": min_ps,
                "target_units": target_units,
                "n_rows": int(len(data)),
                "n_treated": int((data[treatment] == 1).sum()),
                "n_control": int((data[treatment] == 0).sum()),
                "bootstrap_samples_requested": bootstrap_samples,
                "bootstrap_samples_used": len(bootstrap_effects),
                "bootstrap_effect_values": bootstrap_effects,
                "ci_95": (
                    {"lower": ci_95[0], "upper": ci_95[1]} if ci_95 is not None else None
                ),
            }
        )

    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit Surgery+RT Bayesian Network from DAG JSON and estimate DoWhy "
            "effects adjusted for confounding by indication."
        )
    )

    parser.add_argument(
        "--mongodb",
        action="store_true",
        help="Load data from MongoDB using the provided config.",
    )

    parser.add_argument(
        "--config",
        required=True,
        help="YAML/JSON config used for feature selection and preprocessing.",
    )
    parser.add_argument(
        "--dag",
        required=True,
        help="Path to learned DAG JSON (e.g., output/surgery_rt_dag.json).",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Path to save fitted Bayesian network (.pkl).",
    )
    parser.add_argument(
        "--output-effects",
        required=True,
        help="Path to save DoWhy estimated effects (JSON).",
    )
    parser.add_argument(
        "--cpd-summary",
        help="Optional CSV path to save CPD summary metadata.",
    )
    parser.add_argument(
        "--treatment",
        default="treatments.radiotherapy_preoperative",
        help="Binary treatment node (0/1), default is preoperative RT.",
    )
    parser.add_argument(
        "--outcomes",
        nargs="+",
        default=[
            "treatments.any_local_recurrence",
            "treatments.any_metastasis",
            "treatments.dod",
        ],
        help="Binary outcomes to estimate treatment effects for.",
    )
    parser.add_argument(
        "--min-ps",
        type=float,
        default=0.05,
        help="Lower propensity bound for trimming in DoWhy weighting.",
    )
    parser.add_argument(
        "--target-units",
        choices=["ate", "att", "atc"],
        default="att",
        help="Causal estimand target units for weighting (default: att).",
    )
    parser.add_argument(
        "--compare-deep",
        action="store_true",
        help="Estimate effects in deep and non-deep groups and save both plus their differences.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Number of bootstrap resamples for 95% CI (default: 0 = disabled).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--data-summary",
        help="Optional CSV path for data summary after preprocessing.",
    )
    parser.add_argument(
        "--min-initial-size",
        type=float,
        help=(
            "Optional lower bound for tumor_characteristics.initial_size. "
            "Example: --min-initial-size 84. In --compare-deep mode, "
            "this threshold is applied to the deep group only."
        ),
    )
    args = parser.parse_args()

    if not 0.0 < args.min_ps < 0.5:
        raise ValueError("--min-ps must be in (0, 0.5).")
    if not args.mongodb:
        raise ValueError("This launcher-specific script requires --mongodb.")
    if not args.compare_deep:
        raise ValueError("This launcher-specific script requires --compare-deep.")

    processor = DataProcessing(args.config)

    load_dotenv()
    data = processor._load_data_from_mongodb()

    data = processor._apply_feature_selection(data)
    data = processor._mapping_converter(data)
    data = processor._variable_type_converter(data)

    # Keep a pre-binning copy so compare mode can threshold deep tumors by raw size.
    pre_binning_data = data.copy()

    data, binning_info = processor.binning(data)

    raw_data = data.copy()

    if args.data_summary:
        summary_df = processor._generate_data_summary(data, binning_info)
        summary_path = Path(args.data_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Data summary saved to: {summary_path}")

    nodes, edges, graph = _load_dag(args.dag)
    data = _safe_discretize(data, nodes)

    print(f"Fitting Bayesian Network on {len(data)} rows, {len(nodes)} nodes, {len(edges)} edges")
    _fit_bn(
        data=data,
        nodes=nodes,
        edges=edges,
        model_path=args.output_model,
        cpd_summary_path=args.cpd_summary,
        equivalent_sample_size=10.0,
    )
    print(f"Bayesian Network saved to: {args.output_model}")

    deep_subset = _apply_deep_only_filter(raw_data, processor.config)
    if args.min_initial_size is not None:
        eligible_idx = _apply_min_initial_size_filter(
            pre_binning_data,
            args.min_initial_size,
        ).index
        deep_subset = deep_subset.loc[deep_subset.index.intersection(eligible_idx)].copy()
        if deep_subset.empty:
            raise ValueError(
                "No rows left for deep group after applying --min-initial-size."
            )
        print(
            "Applied deep-group size threshold in compare mode: "
            f"{len(deep_subset)} deep rows retained (initial_size >= {args.min_initial_size})"
        )

    deep_data = _safe_discretize(deep_subset, nodes)
    non_deep_data = _safe_discretize(_apply_not_deep_filter(raw_data, processor.config), nodes)

    deep_effects = _estimate_dowhy_effects(
        data=deep_data,
        treatment=args.treatment,
        outcomes=args.outcomes,
        nodes=nodes,
        edges=edges,
        graph=graph,
        min_ps=args.min_ps,
        target_units=args.target_units,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    non_deep_effects = _estimate_dowhy_effects(
        data=non_deep_data,
        treatment=args.treatment,
        outcomes=args.outcomes,
        nodes=nodes,
        edges=edges,
        graph=graph,
        min_ps=args.min_ps,
        target_units=args.target_units,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )

    payload = {
        "mode": "deep_vs_non_deep",
        "treatment": args.treatment,
        "target_units": args.target_units,
        "groups": {
            "deep": _attach_subgroup(deep_effects, "deep"),
            "non_deep": _attach_subgroup(non_deep_effects, "non_deep"),
        },
        "comparison": _build_deep_comparison(deep_effects, non_deep_effects),
    }
    _write_json(args.output_effects, payload)

    plot_path = _default_histogram_plot_path(
        args.output_effects,
        args.target_units,
    )
    outcome_terms = ["local_recurrence", "metastasis", "dod"]
    deep_selected = _select_outcome_records(deep_effects, outcome_terms)
    non_deep_selected = _select_outcome_records(non_deep_effects, outcome_terms)
    if deep_selected or non_deep_selected:
        _save_bootstrap_histograms(
            output_path=plot_path,
            target_units=args.target_units,
            records_by_group={"deep": deep_selected, "non_deep": non_deep_selected},
        )
    else:
        print(
            "Skipping histogram plot in compare mode: no requested outcomes "
            "(local recurrence, metastasis, DoD) were estimated."
        )

    print(f"DoWhy deep-vs-non-deep effect estimates saved to: {args.output_effects}")


if __name__ == "__main__":
    main()
