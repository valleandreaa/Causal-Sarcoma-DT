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

from dtcygan.constraints import ConstraintParser, _build_constraints_from_config
from dtcygan.data_bundle import DataBundle
from dtcygan.data_schema import DataSchema
from dtcygan.io import DAGSerializer
from dtcygan.learning import PCStableLearner
from dtcygan.structure import StructureLearningResult, learn_dag
from dtcygan.data_processing.data_processing import DataProcessing

class CITestLogger:
    """Wrapper for pgmpy CI tests that logs all test results."""
    
    def __init__(self, ci_test_name, data, significance_level=0.05, debug=False):
        self.ci_test_name = ci_test_name
        self.data = data
        self.significance_level = significance_level
        self.debug = debug
        self.test_log = []
        self.test_count = 0
        
        # Import the actual CI test function that returns full statistics
        if ci_test_name == "chi_square":
            from pgmpy.estimators.CITests import chi_square
            self.ci_test_func = chi_square
        elif ci_test_name == "g_sq":
            from pgmpy.estimators.CITests import g_sq
            self.ci_test_func = g_sq
        elif ci_test_name == "log_likelihood":
            from pgmpy.estimators.CITests import log_likelihood
            self.ci_test_func = log_likelihood
        elif ci_test_name == "freeman_tuckey":
            from pgmpy.estimators.CITests import freeman_tuckey
            self.ci_test_func = freeman_tuckey
        elif ci_test_name == "modified_log_likelihood":
            from pgmpy.estimators.CITests import modified_log_likelihood
            self.ci_test_func = modified_log_likelihood
        elif ci_test_name == "neyman":
            from pgmpy.estimators.CITests import neyman
            self.ci_test_func = neyman
        elif ci_test_name == "cressie_read":
            from pgmpy.estimators.CITests import cressie_read
            self.ci_test_func = cressie_read
        else:
            self.ci_test_func = None
    
    def __call__(self, X, Y, Z, data, **kwargs):
        """Execute CI test and log the result."""
        self.test_count += 1
        
        # Call the original CI test function to get full statistics
        test_stat = None
        p_value = None
        dof = None
        
        if self.ci_test_func is not None:
            try:
                # Call the CI test function directly to get statistics
                # Pass significance_level and boolean=False to get raw statistics
                raw_result = self.ci_test_func(
                    X, Y, Z, self.data, 
                    boolean=False,  # Request raw statistics instead of boolean
                    significance_level=self.significance_level
                )
                
                if self.debug and self.test_count <= 5:
                    print(f"  [DEBUG CI Test #{self.test_count}] {X} _||_ {Y} | {Z}")
                    print(f"    Raw result type: {type(raw_result)}, Value: {raw_result}")
                
                # Parse based on return format
                if isinstance(raw_result, tuple) and len(raw_result) >= 2:
                    test_stat = float(raw_result[0]) if raw_result[0] is not None else None
                    p_value = float(raw_result[1]) if raw_result[1] is not None else None
                    if len(raw_result) >= 3:
                        dof = int(raw_result[2]) if raw_result[2] is not None else None
                elif isinstance(raw_result, (float, int)) and not isinstance(raw_result, bool):
                    p_value = float(raw_result)
            except Exception as e:
                if self.debug and self.test_count <= 5:
                    print(f"    Error calling CI test function: {e}")
        
        # Determine independence
        if p_value is not None:
            independent = p_value > self.significance_level
        else:
            # Fallback: call the wrapped function as pgmpy would
            result = kwargs.get('_pgmpy_result')
            independent = bool(result) if result is not None else None
        
        self.test_log.append({
            'X': X,
            'Y': Y,
            'Z': sorted(list(Z)) if Z else [],
            'conditioning_set_size': len(Z) if Z else 0,
            'test_statistic': test_stat,
            'p_value': p_value,
            'dof': dof,
            'significance_level': self.significance_level,
            'independent': independent,
            'decision': 'independent' if independent else 'dependent'
        })
        
        # Return boolean as pgmpy expects
        return independent if independent is not None else False
    
    def get_log(self):
        """Return all logged CI tests."""
        return self.test_log


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))




def _parse_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]










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
    parser.add_argument("--pgmpy-validation-log", help="Path to save pgmpy constraint violations and post-processing actions (CSV or JSON).")
    parser.add_argument("--pgmpy-ci-tests-log", help="Path to save pgmpy conditional independence test results (CSV or JSON).")
    parser.add_argument("--debug-ci-tests", action="store_true", help="Print debug info about CI test return values.")
    parser.add_argument("--data-summary", help="Path to save data summary table with variable distributions and statistics (CSV).")

    args = parser.parse_args()

    # Load configuration if provided
    config = {}
    display_names = {}

    DataProc = DataProcessing(args.config)

    if args.config:
        
        print(f"Loaded configuration from: {args.config}")
        # Extract display names for visualization
        display_names = config.get('display_names', {})
        load_dotenv()
        data = DataProc._load_data_from_mongodb()
    else:
        parser.error("One of --data, --mongodb, or --generate-rows must be specified.")
    
    # Apply feature selection from config
    
    data = DataProc._apply_feature_selection(data)
    

    # Apply custom mappings from config
    data = DataProc._mapping_converter(data)
    
    
    # Then apply variable_types if specified (for additional control)
    data = DataProc._variable_type_converter(data)
    
    # Apply binning to numerical columns based on config
    data, binning_info = DataProc.binning(data)
    
    # Generate data summary if requested
    if args.data_summary:
        print("Generating data summary table...")
        summary_df = DataProc._generate_data_summary(data, binning_info)
        summary_path = Path(args.data_summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"Data summary saved to: {summary_path}")
        print(f"  Variables: {len(summary_df)}, Total observations: {len(data)}\n")
    
    # Build constraints from config
    constraints = _build_constraints_from_config(DataProc.config)
    
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
    algorithm_config = DataProc.config.get("algorithm", {})
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
        
        # Wrap CI test function to capture test results if logging is requested
        ci_test_logger = None
        if args.pgmpy_ci_tests_log:
            print(f"\nCapturing CI test results for logging...")
            
            # Create logger wrapper that calls CI test directly
            ci_test_logger = CITestLogger(
                ci_test_name=ci_test,
                data=data,
                significance_level=significance_level,
                debug=args.debug_ci_tests
            )
            ci_test = ci_test_logger
        
        print(f"\nRunning PC algorithm with ExpertKnowledge...")
        
        # Estimate structure with expert knowledge
        learned_model = pc_estimator.estimate(
            variant="stable",
            ci_test=ci_test,
            significance_level=significance_level,
            expert_knowledge=expert_knowledge if constraints else None,
            # NOTE: pgmpy 1.0 can raise KeyError in orient_colliders when
            # enforce_expert_knowledge=True removes edges before sep-sets exist.
            # Keep False and enforce hard constraints after estimation.
            enforce_expert_knowledge=False,
        )
        
        # Save CI test log if requested
        if args.pgmpy_ci_tests_log and ci_test_logger:
            ci_test_records = ci_test_logger.get_log()
            if ci_test_records:
                _write_records(args.pgmpy_ci_tests_log, ci_test_records)
                print(f"pgmpy CI test log saved to: {args.pgmpy_ci_tests_log} ({len(ci_test_records)} tests)")
            else:
                print(f"Warning: No CI tests were captured in the log")
        
        # Convert pgmpy model to dtcygan DAG structure
        from dtcygan.dag_spec import DAGSpec
        dag = DAGSpec(
            nodes=list(learned_model.nodes()),
            edges=set(learned_model.edges())  # Convert to set, not list
        )

        # Apply strict hard constraints post-learning (forbidden/tier + required).
        # This avoids pgmpy enforce_expert_knowledge KeyError while still
        # guaranteeing constraints in the final exported DAG.
        if constraints is not None:
            dag = constraints.apply_hard_constraints(dag)
        
        # Verify constraints were respected
        print(f"\nVerifying constraint compliance in learned DAG...")
        violations_found = False
        validation_records = []
        
        if constraints is not None:
            # Check for tier violations
            if constraints.tier_constraints:
                tier_map = constraints.tier_constraints.node_to_tier
                for src, dst in dag.edges:
                    src_tier = tier_map.get(src)
                    dst_tier = tier_map.get(dst)
                    if src_tier is not None and dst_tier is not None and src_tier > dst_tier:
                        print(f"TIER VIOLATION: {src} (tier {src_tier}) → {dst} (tier {dst_tier})")
                        violations_found = True
                        validation_records.append({
                            "stage": "verification",
                            "type": "tier_violation",
                            "source": src,
                            "target": dst,
                            "source_tier": src_tier,
                            "target_tier": dst_tier,
                            "action": "detected",
                            "description": f"Edge violates tier ordering: {src} (tier {src_tier}) → {dst} (tier {dst_tier})"
                        })
            
            # Check for forbidden edges
            if constraints.edge_constraints and constraints.edge_constraints.forbidden_edges:
                for src, dst in dag.edges:
                    if (src, dst) in constraints.edge_constraints.forbidden_edges:
                        print(f"  ⚠ FORBIDDEN EDGE: {src} → {dst}")
                        violations_found = True
                        validation_records.append({
                            "stage": "verification",
                            "type": "forbidden_edge",
                            "source": src,
                            "target": dst,
                            "source_tier": None,
                            "target_tier": None,
                            "action": "detected",
                            "description": f"Forbidden edge present: {src} → {dst}"
                        })
            
            # Check for missing required edges
            if constraints.edge_constraints and constraints.edge_constraints.required_edges:
                for src, dst in constraints.edge_constraints.required_edges:
                    if not dag.has_edge(src, dst):
                        print(f"  ⚠ MISSING REQUIRED EDGE: {src} → {dst}")
                        violations_found = True
                        validation_records.append({
                            "stage": "verification",
                            "type": "missing_required_edge",
                            "source": src,
                            "target": dst,
                            "source_tier": None,
                            "target_tier": None,
                            "action": "detected",
                            "description": f"Required edge missing: {src} → {dst}"
                        })
        
        if violations_found:
            print("\n⚠ pgmpy did not fully respect constraints - no automatic fixes applied.")
            # Summarize detected violations for user review
            violation_counts = {}
            for rec in validation_records:
                vtype = rec.get("type", "unknown")
                violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
            for vtype, count in sorted(violation_counts.items(), key=lambda x: x[0]):
                print(f"  {count} x {vtype}")
            print("  Please review the validation log and apply fixes manually if desired.")
        else:
            print("✓ All constraints respected during learning")
        
        # Save validation log if requested
        if args.pgmpy_validation_log and validation_records:
            _write_records(args.pgmpy_validation_log, validation_records)
            print(f"pgmpy validation log saved to: {args.pgmpy_validation_log}")
        
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
        learner = PCStableLearner(variable_types=DataProc.config.get("variable_types", {}))
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
    # print(f"\n{'='*60}")
    # print(f"Learned DAG Structure ({args.method.upper()} algorithm)")
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
        
        # Check if we have tier information from constraints
        tier_map = None
        if constraints is not None and constraints.tier_constraints is not None:
            tier_map = constraints.tier_constraints.node_to_tier
        
        # Create tier-based layout if available
        if tier_map:
            # Group nodes by tier
            tiers = {}
            for node in G.nodes():
                tier = tier_map.get(node, -1)  # -1 for nodes without tier
                if tier not in tiers:
                    tiers[tier] = []
                tiers[tier].append(node)
            
            # Sort tiers
            sorted_tiers = sorted(tiers.keys())
            
            # Use multipartite layout for better hierarchical arrangement
            for node in G.nodes():
                G.nodes[node]['subset'] = tier_map.get(node, -1)
            
            try:
                # Try using multipartite_layout for cleaner hierarchy
                pos = nx.multipartite_layout(G, subset_key='subset', align='horizontal', scale=3)
            except:
                # Fallback to manual positioning
                pos = {}
                for tier_idx, tier in enumerate(sorted_tiers):
                    nodes_in_tier = tiers[tier]
                    y_positions = np.linspace(-len(nodes_in_tier)/2, len(nodes_in_tier)/2, len(nodes_in_tier))
                    for node_idx, node in enumerate(sorted(nodes_in_tier)):
                        pos[node] = (tier_idx * 3, y_positions[node_idx])
            
            # Adjust figure size based on number of tiers and nodes
            max_nodes_in_tier = max(len(tiers[t]) for t in tiers)
            fig_width = max(16, len(sorted_tiers) * 5)
            fig_height = max(10, max_nodes_in_tier * 1.5)
            plt.figure(figsize=(fig_width, fig_height))
            
            # Color nodes by tier with better color scheme
            tier_colors_map = {
                0: '#FFB6C1',  # Light pink - baseline
                1: '#87CEEB',  # Sky blue - treatments
                2: '#FFA07A',  # Light salmon - follow-up
                3: '#90EE90',  # Light green - outcomes
            }
            node_colors = [tier_colors_map.get(tier_map.get(node, -1), 'lightgray') 
                          for node in G.nodes()]
            
            # Categorize edges by type (intra-tier vs inter-tier)
            intra_tier_edges = []
            inter_tier_edges = []
            for src, dst in G.edges():
                src_tier = tier_map.get(src, -1)
                dst_tier = tier_map.get(dst, -1)
                if src_tier == dst_tier:
                    intra_tier_edges.append((src, dst))
                else:
                    inter_tier_edges.append((src, dst))
        else:
            # Fall back to spring layout
            plt.figure(figsize=(14, 10))
            pos = nx.spring_layout(G, k=2, iterations=50, seed=args.seed)
            node_colors = 'lightblue'
            inter_tier_edges = list(G.edges())
            intra_tier_edges = []
        
        # Draw nodes with larger size and better styling
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                node_size=4000, alpha=0.95, edgecolors='black', linewidths=2)
        
        # Create label mapping using display names from config
        labels = {node: display_names.get(node, node) for node in G.nodes()}
        
        # Draw labels with better font
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, 
                                font_weight='bold', font_family='sans-serif')
        
        # Draw edges AFTER nodes and labels so arrows are visible on top
        # Draw inter-tier edges (main causal flow) with emphasis
        if inter_tier_edges:
            nx.draw_networkx_edges(G, pos, edgelist=inter_tier_edges,
                                    edge_color='#2E4057', 
                                    arrows=True, arrowsize=30, 
                                    arrowstyle='-|>', width=3.5, alpha=0.9,
                                    connectionstyle='arc3,rad=0.1',
                                    node_size=4000, min_source_margin=15, min_target_margin=15)
        
        # Draw intra-tier edges (correlations within tier) with lighter style
        if intra_tier_edges:
            nx.draw_networkx_edges(G, pos, edgelist=intra_tier_edges,
                                    edge_color='gray', 
                                    arrows=True, arrowsize=22, 
                                    arrowstyle='-|>', width=2, alpha=0.5,
                                    connectionstyle='arc3,rad=0.2', style='dashed',
                                    node_size=4000, min_source_margin=15, min_target_margin=15)
        
        # Add tier labels if using tier layout
        if tier_map:
            tier_names = {
                0: 'Baseline/Diagnosis',
                1: 'Primary Treatments',
                2: 'Follow-up Events',
                3: 'Patient Outcomes'
            }
            for tier_idx, tier in enumerate(sorted_tiers):
                tier_nodes_y = [pos[n][1] for n in tiers[tier]]
                y_pos = max(tier_nodes_y) + (max(tier_nodes_y) - min(tier_nodes_y)) * 0.15 + 0.3
                x_pos = np.mean([pos[n][0] for n in tiers[tier]])
                
                tier_label = tier_names.get(tier, f'Tier {tier}')
                plt.text(x_pos, y_pos, 
                        tier_label, 
                        ha='center', fontsize=13, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.7', facecolor='yellow', 
                                 alpha=0.5, edgecolor='black', linewidth=1.5))
        
        # Add legend for edge types if we have both
        # if tier_map and intra_tier_edges and inter_tier_edges:
        #     from matplotlib.lines import Line2D
        #     legend_elements = [
        #         Line2D([0], [0], color='#2E4057', linewidth=3, label='Causal flow (inter-tier)'),
        #         Line2D([0], [0], color='gray', linewidth=1.5, linestyle='--', label='Within-tier correlation')
        #     ]
        #     plt.legend(handles=legend_elements, loc='upper right', fontsize=11)
        
        # plt.title(f"Learned DAG Structure - {method.upper()} Algorithm" + 
        #          (" (Hierarchical Layout)" if tier_map else ""), 
        #             fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        image_path = Path(args.image)
        image_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(image_path), dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"DAG visualization saved to: {image_path}" + 
              (f" (using {len(sorted_tiers)} tiers, {len(inter_tier_edges)} inter-tier edges)" if tier_map else ""))


if __name__ == "__main__":
    main()
