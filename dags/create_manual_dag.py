#!/usr/bin/env python3
"""Script to create a manual DAG from MongoDB data using causal-dtcygan library.

This script:
1. Loads configuration from YAML file
2. Connects to MongoDB and queries data
3. Builds a DAG using the ManualGraphBuilder from causal-dtcygan
4. Validates the DAG structure
5. Saves the DAG in multiple formats
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
import pandas as pd
import numpy as np

# # Add the causal-dtcygan package to path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent / "causal-dtcygan" / "src"))

# # Add the Models utils package to path for MongoExtractor
# sys.path.insert(0, str(Path(__file__).parent.parent / "etl_sdt" / "utils"))

from dtcygan.manual_graph import ManualGraphBuilder
from dtcygan.dag_spec import DAGSpec
from dtcygan.data_schema import DataSchema
from dtcygan.io import DAGSerializer, GraphExporter
from dtcygan.graph_validation import GraphValidator
from Models.utils.helpers import MongoExtractor


class DAGFromMongoBuilder:
    """Builder class for creating DAGs from MongoDB data."""
    
    def __init__(self, config_path: str):
        """Initialize the builder with configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # MongoDB connection using MongoExtractor
        self.mongo_extractor = None
        
        # Data and DAG
        self.df = None
        self.dag = None
        self.schema = None
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Replace environment variables
        config = self._replace_env_vars(config)
        return config
    
    def _replace_env_vars(self, obj):
        """Recursively replace environment variable placeholders."""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
            env_var = obj[2:-1]
            return os.getenv(env_var, obj)
        return obj
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_file = log_config.get('log_file')
        
        # Create log directory if needed
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def connect_mongodb(self):
        """Establish connection to MongoDB using MongoExtractor."""
        mongo_config = self.config['mongodb']
        query_config = self.config['query']
        
        try:
            self.logger.info("Connecting to MongoDB using MongoExtractor...")
            
            # Create a custom config for MongoExtractor with the query structure
            custom_config = {
                'aggregator': query_config.get('aggregator', []),
                'fields': query_config.get('fields', [])
            }
            
            # Initialize MongoExtractor with custom config
            self.mongo_extractor = MongoExtractor(
                connection_string=mongo_config['connection_string'],
                database_name=mongo_config['database_name'],
                collection_name=mongo_config['collection_name'],
                custom_config=custom_config
            )
            
            # Test connection by accessing the collection
            _ = self.mongo_extractor.collection.count_documents({})
            self.logger.info(f"Connected to MongoDB: {mongo_config['database_name']}.{mongo_config['collection_name']}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    def query_data(self) -> pd.DataFrame:
        """Query data from MongoDB using MongoExtractor.
        
        Returns:
            DataFrame with queried data
        """
        if self.mongo_extractor is None:
            raise ValueError("MongoDB connection not established. Call connect_mongodb() first.")
        
        query_config = self.config['query']
        
        try:
            self.logger.info("Executing MongoDB aggregation pipeline using MongoExtractor...")
            
            # Use MongoExtractor's fetch_data method which handles aggregation and flattening
            self.df = self.mongo_extractor.get_dataframe()
            
            self.logger.info(f"Retrieved {len(self.df)} documents from MongoDB")
            
            if self.df.empty:
                self.logger.warning("No data retrieved from MongoDB")
                return pd.DataFrame()
            
            # Keep only specified fields if provided
            fields = query_config.get('fields', [])
            if fields:
                available_fields = [f for f in fields if f in self.df.columns]
                if available_fields:
                    self.df = self.df[available_fields]
                else:
                    self.logger.warning("None of the specified fields found in data, keeping all columns")
                
                missing_fields = set(fields) - set(available_fields)
                if missing_fields:
                    self.logger.warning(f"Fields not found in data: {missing_fields}")
            
            self.logger.info(f"DataFrame shape: {self.df.shape}")
            self.logger.info(f"Columns: {list(self.df.columns)}")
            
            return self.df
            
        except Exception as e:
            self.logger.error(f"MongoDB query error: {e}")
            raise
    
    def process_data(self):
        """Process and clean the data."""
        if self.df is None or self.df.empty:
            self.logger.warning("No data to process")
            return
        
        processing_config = self.config.get('data_processing', {})
        
        self.logger.info("Processing data...")
        
        # Handle missing values
        if processing_config.get('handle_missing', True):
            strategy = processing_config.get('missing_strategy', 'drop')
            
            if strategy == 'drop':
                original_size = len(self.df)
                self.df = self.df.dropna()
                self.logger.info(f"Dropped {original_size - len(self.df)} rows with missing values")
            
            elif strategy == 'impute':
                # Simple imputation: median for numeric, mode for categorical
                for col in self.df.columns:
                    if self.df[col].dtype in ['int64', 'float64']:
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    else:
                        self.df[col].fillna(self.df[col].mode()[0] if not self.df[col].mode().empty else 'unknown', inplace=True)
                self.logger.info("Imputed missing values")
        
        # Encode categorical variables
        if processing_config.get('encode_categoricals', True):
            method = processing_config.get('encoding_method', 'label')
            
            for col in self.df.columns:
                if self.df[col].dtype == 'object':
                    if method == 'label':
                        self.df[col] = pd.Categorical(self.df[col]).codes
                        self.logger.debug(f"Label encoded column: {col}")
        
        # Normalize numeric columns
        if processing_config.get('normalize', False):
            method = processing_config.get('normalization_method', 'standard')
            
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if method == 'standard':
                self.df[numeric_cols] = (self.df[numeric_cols] - self.df[numeric_cols].mean()) / self.df[numeric_cols].std()
                self.logger.info("Applied standard normalization")
            
            elif method == 'minmax':
                self.df[numeric_cols] = (self.df[numeric_cols] - self.df[numeric_cols].min()) / (self.df[numeric_cols].max() - self.df[numeric_cols].min())
                self.logger.info("Applied min-max normalization")
        
        self.logger.info(f"Processed data shape: {self.df.shape}")
    
    def create_schema(self) -> DataSchema:
        """Create a DataSchema from the processed data.
        
        Returns:
            DataSchema object
        """
        if self.df is None or self.df.empty:
            raise ValueError("No data available to create schema")
        
        self.logger.info("Creating data schema...")
        self.schema = DataSchema.from_dataframe(self.df)
        self.logger.info(f"Schema created with {len(self.schema.columns)} columns")
        
        return self.schema
    
    def build_dag(self) -> DAGSpec:
        """Build the DAG from configuration.
        
        Returns:
            DAGSpec object
        """
        dag_config = self.config['dag']
        
        self.logger.info("Building DAG...")
        
        # Extract nodes and edges
        nodes = dag_config['nodes']
        edges = [tuple(edge) for edge in dag_config['edges']]
        graph_type = dag_config.get('graph_type', 'dag')
        
        # Build the DAG using ManualGraphBuilder
        self.dag = ManualGraphBuilder.from_edge_list(
            nodes=nodes,
            edges=edges,
            graph_type=graph_type
        )
        
        self.logger.info(f"DAG created with {len(self.dag.nodes)} nodes and {len(self.dag.edges)} edges")
        
        # Attach metadata if provided
        node_metadata = dag_config.get('node_metadata')
        edge_metadata = dag_config.get('edge_metadata')
        
        if node_metadata or edge_metadata:
            # Convert edge_metadata keys to tuples if needed
            if edge_metadata:
                edge_metadata_tuples = {}
                for key, value in edge_metadata.items():
                    if isinstance(key, str) and '->' in key:
                        src, dst = key.split('->')
                        edge_metadata_tuples[(src, dst)] = value
                    else:
                        edge_metadata_tuples[key] = value
                edge_metadata = edge_metadata_tuples
            
            self.dag = ManualGraphBuilder.attach_metadata(
                self.dag,
                node_metadata,
                edge_metadata
            )
            self.logger.info("Attached metadata to DAG")
        
        # Add provenance information
        self.dag.provenance = {
            'source': 'mongodb',
            'config_file': str(self.config_path),
            'creation_method': 'manual_specification',
            'n_samples': len(self.df) if self.df is not None else 0
        }
        
        return self.dag
    
    def validate_dag(self):
        """Validate the DAG structure and against data schema."""
        if self.dag is None:
            raise ValueError("No DAG to validate")
        
        validation_config = self.config.get('validation', {})
        
        self.logger.info("Validating DAG...")
        
        # Basic validation
        try:
            self.dag.validate_basic()
            self.logger.info("✓ Basic DAG validation passed")
        except ValueError as e:
            self.logger.error(f"✗ Basic DAG validation failed: {e}")
            raise
        
        # Check acyclic
        if validation_config.get('check_acyclic', True):
            if self.dag.is_acyclic():
                self.logger.info("✓ DAG is acyclic")
            else:
                self.logger.error("✗ DAG contains cycles")
                raise ValueError("DAG contains cycles")
        
        # Validate against schema
        if validation_config.get('validate_schema', True) and self.schema:
            try:
                ManualGraphBuilder.validate_against_schema(self.dag, self.schema)
                self.logger.info("✓ All DAG nodes exist in data schema")
            except ValueError as e:
                self.logger.error(f"✗ Schema validation failed: {e}")
                raise
        
        # Check minimum samples
        min_samples = validation_config.get('min_samples', 0)
        if self.df is not None and len(self.df) < min_samples:
            self.logger.warning(f"Data has only {len(self.df)} samples, minimum is {min_samples}")
        
        self.logger.info("All validations passed")
    
    def save_outputs(self):
        """Save the DAG and data to various output formats."""
        if self.dag is None:
            raise ValueError("No DAG to save")
        
        output_config = self.config.get('output', {})
        
        self.logger.info("Saving outputs...")
        
        # Create output directory
        output_dir = Path(self.config_path).parent.parent / 'output'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DAG in JSON format
        if 'dag_json' in output_config:
            json_path = Path(self.config_path).parent.parent / output_config['dag_json']
            json_path.parent.mkdir(parents=True, exist_ok=True)
            DAGSerializer.save_json(self.dag, str(json_path))
            self.logger.info(f"Saved DAG to JSON: {json_path}")
        
        # Save DAG in GraphML format
        if 'dag_graphml' in output_config:
            graphml_path = Path(self.config_path).parent.parent / output_config['dag_graphml']
            graphml_path.parent.mkdir(parents=True, exist_ok=True)
            GraphExporter.to_graphml(self.dag, str(graphml_path))
            self.logger.info(f"Saved DAG to GraphML: {graphml_path}")
        
        # Save DAG in DOT format
        if 'dag_dot' in output_config:
            dot_path = Path(self.config_path).parent.parent / output_config['dag_dot']
            dot_path.parent.mkdir(parents=True, exist_ok=True)
            GraphExporter.to_dot(self.dag, str(dot_path))
            self.logger.info(f"Saved DAG to DOT: {dot_path}")
        
        # Save DAG as PNG (requires graphviz)
        if 'dag_png' in output_config:
            try:
                png_path = Path(self.config_path).parent.parent / output_config['dag_png']
                png_path.parent.mkdir(parents=True, exist_ok=True)
                GraphExporter.to_png(self.dag, str(png_path))
                self.logger.info(f"Saved DAG to PNG: {png_path}")
            except ImportError:
                self.logger.warning("graphviz not installed, skipping PNG export")
        
        # Save processed data
        if 'processed_data' in output_config and self.df is not None:
            data_path = Path(self.config_path).parent.parent / output_config['processed_data']
            data_path.parent.mkdir(parents=True, exist_ok=True)
            self.df.to_csv(data_path, index=False)
            self.logger.info(f"Saved processed data: {data_path}")
        
        # Save validation report
        if 'validation_report' in output_config:
            report_path = Path(self.config_path).parent.parent / output_config['validation_report']
            report_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_validation_report(report_path)
            self.logger.info(f"Saved validation report: {report_path}")
    
    def _save_validation_report(self, path: Path):
        """Save a validation report with DAG statistics."""
        with open(path, 'w', encoding='utf-8') as f:
            f.write("DAG Validation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Configuration file: {self.config_path}\n")
            f.write(f"Graph type: {self.dag.graph_type}\n")
            f.write(f"Number of nodes: {len(self.dag.nodes)}\n")
            f.write(f"Number of edges: {len(self.dag.edges)}\n")
            f.write(f"Is acyclic: {self.dag.is_acyclic()}\n\n")
            
            if self.df is not None:
                f.write(f"Data samples: {len(self.df)}\n")
                f.write(f"Data features: {len(self.df.columns)}\n\n")
            
            f.write("Nodes:\n")
            for node in self.dag.nodes:
                f.write(f"  - {node}\n")
            
            f.write("\nEdges:\n")
            for src, dst in sorted(self.dag.edges):
                f.write(f"  - {src} -> {dst}\n")
            
            if self.dag.node_metadata:
                f.write("\nNode Metadata:\n")
                for node, meta in self.dag.node_metadata.items():
                    f.write(f"  {node}: {meta}\n")
            
            if self.dag.edge_metadata:
                f.write("\nEdge Metadata:\n")
                for edge, meta in self.dag.edge_metadata.items():
                    f.write(f"  {edge[0]} -> {edge[1]}: {meta}\n")
    
    def run(self):
        """Execute the full pipeline."""
        try:
            self.logger.info("Starting DAG creation pipeline...")
            
            # Step 1: Connect to MongoDB
            self.connect_mongodb()
            
            # Step 2: Query data
            self.query_data()
            
            # Step 3: Process data
            self.process_data()
            
            # Step 4: Create schema
            self.create_schema()
            
            # Step 5: Build DAG
            self.build_dag()
            
            # Step 6: Validate DAG
            self.validate_dag()
            
            # Step 7: Save outputs
            self.save_outputs()
            
            self.logger.info("DAG creation pipeline completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            # Close MongoDB connection
            if self.mongo_extractor:
                self.mongo_extractor.close_connection()
                self.logger.info("Closed MongoDB connection")


def main():
    """Main entry point."""
    # Get config file path
    config_path = Path(__file__).parent / 'config' / 'config_manual_dag.yaml'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Create and run the builder
    builder = DAGFromMongoBuilder(str(config_path))
    builder.run()


if __name__ == '__main__':
    main()