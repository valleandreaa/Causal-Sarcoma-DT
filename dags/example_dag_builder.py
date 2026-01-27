"""
Example script demonstrating how to use the DAG builder
with sample data (without requiring MongoDB connection).

This example shows two approaches:
1. Manual DAG specification using edge lists
2. Data-driven DAG learning using causal discovery algorithms

This uses synthetic data instead of querying MongoDB with MongoExtractor.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

from dtcygan.manual_graph import ManualGraphBuilder
from dtcygan.dag_spec import DAGSpec
from dtcygan.data_schema import DataSchema
from dtcygan.data_bundle import DataBundle
from dtcygan.io import DAGSerializer, GraphExporter
from dtcygan.pipelines import DAGBuilder
from dtcygan.learning.pc_stable import PCStableLearner
from dtcygan.dag_plot import DAGPlotter

def create_sample_data(n_samples=100):
    """Create sample sarcoma patient data for testing.
    
    This simulates the kind of data that would be returned by
    MongoExtractor.get_dataframe() after querying MongoDB.
    """
    np.random.seed(42)
    
    # Generate synthetic patient data
    data = {
        'patient_id': [f'P{i:04d}' for i in range(n_samples)],
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice(['male', 'female'], n_samples),
        'tumor_size': np.random.uniform(1.0, 15.0, n_samples),
        'tumor_grade': np.random.choice(['low', 'intermediate', 'high'], n_samples),
        'histological_subtype': np.random.choice(['liposarcoma', 'leiomyosarcoma', 'synovial'], n_samples),
        'treatment_type': np.random.choice(['surgery_only', 'multimodal', 'palliative'], n_samples),
        'surgery_performed': np.random.choice([0, 1], n_samples),
        'chemotherapy': np.random.choice([0, 1], n_samples),
        'radiotherapy': np.random.choice([0, 1], n_samples),
        'metastasis': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'recurrence': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'survival_status': np.random.choice([0, 1], n_samples, p=[0.65, 0.35]),
        'survival_time': np.random.uniform(6, 120, n_samples)
    }
    
    return pd.DataFrame(data)


def build_example_dag_from_data(df):
    """Build a DAG by learning from data using causal discovery."""
    
    print("=" * 60)
    print("Learning DAG from Data using PC-Stable Algorithm")
    print("=" * 60)
    print()
    
    # Create schema from data
    schema = DataSchema.from_dataframe(df)
    print(f"✓ Created schema with {len(schema.columns)} columns")
    
    # Create data bundle
    data_bundle = DataBundle(df=df, schema=schema)
    print(f"✓ Created data bundle (fingerprint: {data_bundle.fingerprint()[:16]}...)")
    print()
    
    # Initialize learner
    learner = PCStableLearner(
        ci_test="partial_correlation",
        alpha=0.05
    )
    print(f"✓ Initialized {learner.name()} with alpha={learner.alpha}")
    print()
    
    # Initialize DAG builder
    builder = DAGBuilder()
    
    print("Learning DAG structure from data...")
    print("(Note: This example uses PC-Stable which may require implementation)")
    print()
    
    try:
        # Learn DAG from data
        dag = builder.build_from_data(
            learner=learner,
            data=data_bundle,
            constraints=None,
            postprocess=True
        )
        print(f"✓ Successfully learned DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
        
        # Add provenance
        dag.provenance = {
            'source': 'data_driven_learning',
            'algorithm': learner.name(),
            'parameters': learner.get_params(),
            'data_fingerprint': data_bundle.fingerprint()
        }
        
        return dag
        
    except NotImplementedError as e:
        print(f"⚠ Learning algorithm not fully implemented: {e}")
        print("Falling back to manual DAG specification...")
        print()
        return None


def build_example_dag():
    """Build an example DAG for sarcoma patient data using manual specification."""
    
    print("=" * 60)
    print("Building Manual DAG for Sarcoma Patient Data")
    print("=" * 60)
    print()
    
    # Define nodes (variables)
    nodes = [
        'age',
        'gender',
        'tumor_size',
        'tumor_grade',
        'histological_subtype',
        'treatment_type',
        'surgery_performed',
        'chemotherapy',
        'radiotherapy',
        'metastasis',
        'recurrence',
        'survival_status',
        'survival_time'
    ]
    
    # Define edges (causal relationships)
    edges = [
        # Baseline characteristics affect tumor characteristics
        ('age', 'tumor_grade'),
        ('age', 'treatment_type'),
        ('gender', 'tumor_size'),
        
        # Tumor characteristics influence treatment decisions
        ('tumor_size', 'treatment_type'),
        ('tumor_size', 'surgery_performed'),
        ('tumor_grade', 'treatment_type'),
        ('histological_subtype', 'treatment_type'),
        
        # Treatment type determines specific treatments
        ('treatment_type', 'surgery_performed'),
        ('treatment_type', 'chemotherapy'),
        ('treatment_type', 'radiotherapy'),
        
        # Tumor characteristics affect disease progression
        ('tumor_size', 'metastasis'),
        ('tumor_grade', 'metastasis'),
        
        # Treatments affect disease progression
        ('surgery_performed', 'metastasis'),
        ('surgery_performed', 'recurrence'),
        ('chemotherapy', 'metastasis'),
        ('radiotherapy', 'recurrence'),
        
        # Disease progression affects survival
        ('metastasis', 'survival_status'),
        ('metastasis', 'survival_time'),
        ('recurrence', 'survival_status'),
        ('recurrence', 'survival_time'),
        
        # Treatments affect survival
        ('chemotherapy', 'survival_status'),
        ('surgery_performed', 'survival_status'),
        
        # Age affects survival
        ('age', 'survival_time'),
        ('tumor_grade', 'survival_time'),
        
        # Survival status affects survival time
        ('survival_status', 'survival_time')
    ]
    
    print(f"Creating DAG with {len(nodes)} nodes and {len(edges)} edges...")
    
    # Build the DAG
    dag = ManualGraphBuilder.from_edge_list(
        nodes=nodes,
        edges=edges,
        graph_type="dag"
    )
    
    print(f"✓ DAG created successfully")
    print()
    
    # Add node metadata
    node_metadata = {
        'age': {'type': 'continuous', 'unit': 'years'},
        'gender': {'type': 'binary', 'categories': ['male', 'female']},
        'tumor_size': {'type': 'continuous', 'unit': 'cm'},
        'tumor_grade': {'type': 'ordinal', 'categories': ['low', 'intermediate', 'high']},
        'histological_subtype': {'type': 'categorical'},
        'treatment_type': {'type': 'categorical'},
        'surgery_performed': {'type': 'binary'},
        'chemotherapy': {'type': 'binary'},
        'radiotherapy': {'type': 'binary'},
        'metastasis': {'type': 'binary'},
        'recurrence': {'type': 'binary'},
        'survival_status': {'type': 'binary'},
        'survival_time': {'type': 'continuous', 'unit': 'months'}
    }
    
    # Add edge metadata
    edge_metadata = {
        ('tumor_size', 'metastasis'): {'strength': 'strong', 'evidence': 'clinical'},
        ('tumor_grade', 'metastasis'): {'strength': 'strong', 'evidence': 'clinical'},
        ('chemotherapy', 'survival_status'): {'strength': 'moderate', 'evidence': 'observational'},
    }
    
    dag = ManualGraphBuilder.attach_metadata(dag, node_metadata, edge_metadata)
    print(f"✓ Added metadata to nodes and edges")
    print()
    
    # Add provenance
    dag.provenance = {
        'source': 'manual_specification',
        'creation_method': 'example_script',
        'description': 'Example DAG for sarcoma patient outcomes'
    }
    
    return dag


def validate_dag_with_data(dag, df):
    """Validate the DAG against the data."""
    
    print("Validating DAG...")
    print("-" * 60)
    
    # Basic validation
    try:
        dag.validate_basic()
        print("✓ Basic validation passed")
    except ValueError as e:
        print(f"✗ Basic validation failed: {e}")
        return False
    
    # Check if acyclic
    if dag.is_acyclic():
        print("✓ DAG is acyclic")
    else:
        print("✗ DAG contains cycles")
        return False
    
    # Create schema from data
    schema = DataSchema.from_dataframe(df)
    print(f"✓ Created schema from data ({len(schema.columns)} columns)")
    
    # Validate nodes exist in schema
    try:
        ManualGraphBuilder.validate_against_schema(dag, schema)
        print("✓ All DAG nodes exist in data")
    except ValueError as e:
        print(f"✗ Schema validation failed: {e}")
        return False
    
    # Get topological order
    try:
        topo_order = dag.topological_sort()
        print(f"✓ Topological order: {' → '.join(topo_order[:5])}... (showing first 5)")
    except ValueError as e:
        print(f"✗ Failed to compute topological order: {e}")
    
    print()
    return True


def plot_dag(dag, output_dir="output/example_output"):
    """Plot and visualize the DAG using DAGPlotter.
    
    This creates an interactive visualization of the DAG structure
    showing nodes, edges, and optionally node metadata.
    """
    try:
        output_path = Path(__file__).parent / output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("Plotting DAG visualization...")
        print("-" * 60)
        
        # Use DAGPlotter to create visualization
        plot_path = output_path / "dag_visualization"
        png_path, pdf_path = DAGPlotter.plot_to_file(dag, str(plot_path))
        
        print(f"✓ Saved visualization: {png_path}")
        print(f"✓ Saved PDF: {pdf_path}")
        print()
        
    except ImportError as e:
        print(f"⚠ Could not create plot: {e}")
        print("  Install required packages: pip install matplotlib networkx")
        print()


def save_dag_outputs(dag, output_dir="output/example_output"):
    """Save the DAG in various formats."""
    
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Saving DAG outputs...")
    print("-" * 60)
    
    # Save as JSON
    json_path = output_path / "example_dag.json"
    DAGSerializer.save_json(dag, str(json_path))
    print(f"✓ Saved JSON: {json_path}")
    
    # Save as GraphML
    graphml_path = output_path / "example_dag.graphml"
    GraphExporter.to_graphml(dag, str(graphml_path))
    print(f"✓ Saved GraphML: {graphml_path}")
    
    # Save as DOT
    dot_path = output_path / "example_dag.dot"
    GraphExporter.to_dot(dag, str(dot_path))
    print(f"✓ Saved DOT: {dot_path}")
    
    # Try to save as PNG (requires graphviz)
    try:
        png_path = output_path / "example_dag.png"
        GraphExporter.to_png(dag, str(png_path))
        print(f"✓ Saved PNG: {png_path}")
    except (ImportError, Exception) as e:
        print(f"⚠ Could not save PNG (graphviz may not be installed): {e}")
    
    print()


def print_dag_summary(dag):
    """Print a summary of the DAG structure."""
    
    print("DAG Summary")
    print("=" * 60)
    print(f"Graph Type: {dag.graph_type}")
    print(f"Number of Nodes: {len(dag.nodes)}")
    print(f"Number of Edges: {len(dag.edges)}")
    print()
    
    print("Nodes:")
    for i, node in enumerate(dag.nodes, 1):
        metadata = dag.node_metadata.get(node, {})
        node_type = metadata.get('type', 'unknown')
        print(f"  {i:2d}. {node:25s} [{node_type}]")
    print()
    
    print("Edges (first 10):")
    for i, (src, dst) in enumerate(sorted(dag.edges)[:10], 1):
        metadata = dag.edge_metadata.get((src, dst), {})
        strength = metadata.get('strength', '')
        print(f"  {i:2d}. {src:20s} → {dst:20s} {strength}")
    if len(dag.edges) > 10:
        print(f"  ... and {len(dag.edges) - 10} more edges")
    print()
    


def main():
    """Main execution function."""
    
    print()
    print("=" * 60)
    print("DAG Builder Example - Data-Driven Learning")
    print("=" * 60)
    print()
    
    # Step 1: Create sample data
    print("Step 1: Creating sample data...")
    df = create_sample_data(n_samples=100)
    print(f"✓ Created sample dataset with {len(df)} patients")
    print(f"  Columns: {', '.join(df.columns[:5])}...")
    print()
    
    # Step 2: Try to build DAG from data
    print("Step 2: Learning DAG from data...")
    dag = build_example_dag_from_data(df)
    
    # If learning fails, fall back to manual specification
    if dag is None:
        print("Step 2b: Using manual DAG specification as fallback...")
        dag = build_example_dag()
    
    # Step 3: Print summary
    print_dag_summary(dag)
    
    # Step 4: Validate
    print("Step 3: Validating DAG...")
    if validate_dag_with_data(dag, df):
        print("✓ All validations passed!")
    else:
        print("✗ Validation failed")
        return
    print()
    
    # Step 5: Plot the DAG
    print("Step 4: Plotting DAG...")
    plot_dag(dag, output_dir="output/learned_dag")
    
    # Step 6: Save outputs
    print("Step 5: Saving outputs...")
    save_dag_outputs(dag, output_dir="output/learned_dag")
    
    print("=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    print()
    print("Approaches demonstrated:")
    print("  1. Data-driven learning: Uses PC-Stable or similar algorithms")
    print("  2. Manual specification: Explicitly defines edges based on domain knowledge")
    print()
    print("Next steps:")
    print("  1. Review the generated files in the 'learned_dag/' directory")
    print("  2. Implement full causal discovery algorithms (PC-Stable, GES, NOTEARS)")
    print("  3. Modify config/config_manual_dag.yaml for your actual data")
    print("  4. Set MongoDB environment variables")
    print("  5. Run: python create_manual_dag.py")
    print()
    print("Note: create_manual_dag.py uses MongoExtractor from Models/utils/helpers.py")
    print("      to handle MongoDB connection, aggregation, and data flattening.")
    print()


if __name__ == '__main__':
    main()


