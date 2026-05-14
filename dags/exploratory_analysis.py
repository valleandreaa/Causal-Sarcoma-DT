"""
Exploratory Data Analysis for Treatment Distributions and Patient Characteristics.

This script performs comprehensive exploratory analysis of sarcoma patient data,
focusing on treatment patterns combined with patient demographics and tumor characteristics.

WORKFLOW AND DATA QUALITY ASSURANCE:
=====================================

1. DATA LOADING:
   - Loads patient records from MongoDB
   
2. FILTERING (CRITICAL - prevents contamination):
   - REMOVES patients with non-soft tissue anatomic regions
   - REMOVES patients with non-malignant gradings:
     * benign
     * not_a_sarcoma
     * suspicious_of_malignancy
   - These patients are COMPLETELY EXCLUDED before any analysis
   
3. EXTRACTION:
   - Extracts characteristics from filtered dataset
   - Validates no excluded gradings remain
   
4. MISSING DATA HANDLING:
   - AFTER filtering, None/null values represent TRULY MISSING data
   - All analyses use _fill_missing_values() helper to label nulls as 'Missing'
   - 'Missing' in plots = data not recorded in database
   - 'Missing' ≠ non-malignant cases (those were filtered out in step 2)
   
5. ANALYSES:
   - All subsequent analyses work on clean, filtered data
   - Missing values shown explicitly for data quality assessment

This ensures:
- Benign cases never appear in any analysis (not even as "Missing")
- True missing data is visible for quality evaluation
- Filtering happens BEFORE aggregations
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
from scipy import stats


class ExploratoryAnalyzer:
    """Main class for performing exploratory data analysis on patient treatment data."""
    
    def __init__(self, config_path: str):
        """Initialize analyzer with configuration."""
        self.config = self._load_config(config_path)
        self.data = None
        self.processed_data = None
        self.output_dir = Path(self.config['output']['base_dir'])
        self.plots_dir = self.output_dir / self.config['output']['plots_dir']
        self.tables_dir = self.output_dir / self.config['output']['tables_dir']
        self.reports_dir = self.output_dir / self.config['output']['reports_dir']
        
        # Create output directories
        for directory in [self.output_dir, self.plots_dir, self.tables_dir, self.reports_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set visualization defaults
        plt.style.use('seaborn-v0_8-' + self.config['visualization']['style'])
        sns.set_palette(self.config['visualization']['palette'])
        plt.rcParams['font.size'] = self.config['visualization']['font_size']
    
    def _load_config(self, config_path: str) -> dict:
        """Load YAML configuration file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data_from_mongodb(self) -> None:
        """Load patient data from MongoDB."""
        load_dotenv()
        
        # Get MongoDB connection details
        mongo_config = self.config['mongodb']
        mongo_uri =  os.getenv('MONGO_URI')
        db_name =  os.getenv('MONGO_DB')
        collection_name = os.getenv('MONGO_COLLECTION')
        
        if not all([mongo_uri, db_name, collection_name]):
            raise ValueError("MongoDB connection details missing")
        
        print(f"Connecting to MongoDB: {db_name}.{collection_name}")
        
        # Connect and fetch data
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        query = mongo_config.get('query', {})
        cursor = collection.find(query)
        self.data = list(cursor)
        client.close()
        
        print(f"Loaded {len(self.data)} patient records from MongoDB")
        
        # Apply soft tissue filter if configured
        if self.config.get('filtering', {}).get('soft_tissue_only', False):
            self._filter_soft_tissue_sarcomas()
    
    def _filter_soft_tissue_sarcomas(self) -> None:
        """
        STEP 1: Filter data to include ONLY soft tissue sarcomas.
        Completely removes patients with:
        - Non-soft tissue anatomic regions
        - Non-malignant biopsy grading (benign, not_a_sarcoma, suspicious_of_malignancy)
        
        After filtering, None/null values in biopsy_grading represent TRULY MISSING data.
        """
        allowed_regions = self.config.get('filtering', {}).get('anatomic_regions_included', [
            'deep_soft_tissue', 'superficial_soft_tissue'
        ])
        
        excluded_gradings = self.config.get('filtering', {}).get('exclude_biopsy_grading', [
            'benign', 'not_a_sarcoma', 'suspicious'
        ])
        
        exclude_missing_grading = self.config.get('filtering', {}).get('exclude_missing_grading', False)
        
        initial_count = len(self.data)
        
        print("\n" + "="*70)
        print("STEP 1: FILTERING DATA - REMOVING NON-MALIGNANT CASES")
        print("="*70)
        
        # Show unique biopsy_grading values BEFORE filtering
        grading_values_before = {}
        region_values_before = {}
        for patient in self.data:
            grading = patient.get('tumor_characteristics', {}).get('biopsy_grading')
            region = patient.get('tumor_characteristics', {}).get('anatomic_region_grouping')
            
            grading_key = grading if grading else '<None/Missing>'
            region_key = region if region else '<None/Missing>'
            
            grading_values_before[grading_key] = grading_values_before.get(grading_key, 0) + 1
            region_values_before[region_key] = region_values_before.get(region_key, 0) + 1
        
        print(f"\nBEFORE FILTERING (n={initial_count}):")
        print("\nAnatomic regions:")
        for region, count in sorted(region_values_before.items(), key=lambda x: -x[1]):
            print(f"  {region}: {count}")
        
        print("\nBiopsy grading:")
        for grading, count in sorted(grading_values_before.items(), key=lambda x: -x[1]):
            print(f"  {grading}: {count}")
        
        # Apply filters with detailed tracking
        filtered_data = []
        excluded_by_region = 0
        excluded_by_grading = 0
        excluded_by_missing_grading = 0
        excluded_details = {'by_region': [], 'by_grading': [], 'by_missing': []}
        
        for patient in self.data:
            region = patient.get('tumor_characteristics', {}).get('anatomic_region_grouping')
            grading = patient.get('tumor_characteristics', {}).get('biopsy_grading')
            
            # Normalize region (handle None, empty string, whitespace)
            region_normalized = None
            if region:
                region_normalized = region.strip().lower()
                if region_normalized == '':
                    region_normalized = None
            
            # Normalize grading (handle None, empty string, whitespace)
            grading_normalized = None
            if grading:
                grading_normalized = grading.strip().lower()
                if grading_normalized == '':
                    grading_normalized = None
            
            # FILTER 1: Check anatomic region - must be in allowed list
            if region_normalized is None or region_normalized not in [r.strip().lower() for r in allowed_regions]:
                excluded_by_region += 1
                if region not in excluded_details['by_region']:
                    excluded_details['by_region'].append(region if region else '<None>')
                continue
            
            # FILTER 2: Check grading exclusions - REMOVE non-malignant cases
            if grading_normalized and grading_normalized in [g.strip().lower() for g in excluded_gradings]:
                excluded_by_grading += 1
                if grading not in excluded_details['by_grading']:
                    excluded_details['by_grading'].append(grading)
                continue
            
            # FILTER 3: Optionally exclude missing grading
            if exclude_missing_grading and grading_normalized is None:
                excluded_by_missing_grading += 1
                continue
            
            # Patient passes all filters - KEEP IT
            filtered_data.append(patient)
        
        self.data = filtered_data
        
        print(f"\n" + "-"*70)
        print(f"FILTERING RESULTS:")
        print(f"  Initial patients: {initial_count}")
        print(f"  Excluded by region: {excluded_by_region} (kept only: {', '.join(allowed_regions)})")
        if excluded_details['by_region']:
            print(f"    Excluded regions: {', '.join(set(excluded_details['by_region']))}")
        print(f"  Excluded by non-malignant grading: {excluded_by_grading}")
        if excluded_details['by_grading']:
            print(f"    Excluded gradings: {', '.join(set(excluded_details['by_grading']))}")
        if exclude_missing_grading:
            print(f"  Excluded by missing grading: {excluded_by_missing_grading}")
        print(f"  REMAINING PATIENTS: {len(self.data)}")
        print("-"*70)
        
        # Validate: Show what's LEFT after filtering
        grading_values_after = {}
        region_values_after = {}
        for patient in self.data:
            grading = patient.get('tumor_characteristics', {}).get('biopsy_grading')
            region = patient.get('tumor_characteristics', {}).get('anatomic_region_grouping')
            
            grading_key = grading if grading else '<None/Missing>'
            region_key = region if region else '<None/Missing>'
            
            grading_values_after[grading_key] = grading_values_after.get(grading_key, 0) + 1
            region_values_after[region_key] = region_values_after.get(region_key, 0) + 1
        
        print(f"\nAFTER FILTERING (n={len(self.data)}):")
        print("\nAnatomic regions (should only be allowed regions):")
        for region, count in sorted(region_values_after.items(), key=lambda x: -x[1]):
            print(f"  {region}: {count}")
        
        print("\nBiopsy grading (benign/not_a_sarcoma/suspicious should be GONE):")
        for grading, count in sorted(grading_values_after.items(), key=lambda x: -x[1]):
            print(f"  {grading}: {count}")
        
        # VALIDATION: Ensure no excluded gradings remain
        remaining_excluded = []
        for patient in self.data:
            grading = patient.get('tumor_characteristics', {}).get('biopsy_grading')
            if grading:
                grading_normalized = grading.strip().lower()
                if grading_normalized in [g.strip().lower() for g in excluded_gradings]:
                    remaining_excluded.append(grading)
        
        if remaining_excluded:
            print(f"\n⚠️  WARNING: Found {len(remaining_excluded)} patients with excluded gradings still in dataset!")
            print(f"   This should not happen! Gradings: {set(remaining_excluded)}")
        else:
            print(f"\n✓ VALIDATION PASSED: No non-malignant cases remain in dataset")
        
        print("\n" + "="*70 + "\n")
    
    def extract_patient_characteristics(self) -> pd.DataFrame:
        """
        STEP 2: Extract patient characteristics into a flat DataFrame.
        
        At this point, filtering has already removed:
        - Non-soft tissue cases
        - Benign, not_a_sarcoma, suspicious_of_malignancy cases
        
        Any None/null values now represent TRULY MISSING data and will be labeled as 'Missing' in analyses.
        """
        print("\n" + "="*70)
        print("STEP 2: EXTRACTING PATIENT CHARACTERISTICS")
        print("="*70)
        print(f"Extracting characteristics from {len(self.data)} filtered patients...")
        print("Note: None/null values will be treated as 'Missing' in subsequent analyses")
        
        records = []
        
        for patient in self.data:
            record = {'patient_id': patient.get('_id')}
            
            # Extract demographic characteristics
            for field in self.config['patient_characteristics']['demographic']:
                value = self._get_nested_field(patient, field)
                field_name = field.split('.')[-1]
                record[field_name] = value
            
            # Extract tumor characteristics
            for field in self.config['patient_characteristics']['tumor']:
                value = self._get_nested_field(patient, field)
                field_name = field.split('.')[-1]
                record[field_name] = value
            
            # Extract treatment summary
            for field in self.config['patient_characteristics']['treatment_summary']:
                value = self._get_nested_field(patient, field)
                field_name = field.split('.')[-1]
                record[field_name] = value
            
            # Extract episode-level treatment counts
            episodes = patient.get('episodes', [])
            record['n_episodes'] = len(episodes)
            record['total_surgeries'] = sum(ep.get('surgery', 0) for ep in episodes)
            record['total_chemotherapy'] = sum(ep.get('chemotherapy', 0) for ep in episodes)
            record['total_radiotherapy'] = sum(ep.get('radiotherapy', 0) for ep in episodes)
            
            # Check for metastasis and local recurrence in any episode
            record['had_metastasis'] = 1 if any(ep.get('metastasis', 0) == 1 for ep in episodes) else 0
            record['had_local_recurrence'] = 1 if any(ep.get('local_recurrence', 0) == 1 for ep in episodes) else 0
            
            # Check for WHOOPS in any episode's treatments
            had_whoops = 0
            for episode in episodes:
                for treatment in episode.get('treatments', []):
                    if treatment.get('whoops', 0) == 1:
                        had_whoops = 1
                        break
                if had_whoops:
                    break
            record['whoops'] = had_whoops
            
            # Get patient status from episode with final_status = 1
            patient_status = None
            for episode in episodes:
                if episode.get('final_status') == 1:
                    # Look for last_status in diagnosis section of this episode
                    for diag in episode.get('diagnosis', []):
                        if diag.get('section') == 'follow_up' and 'last_status' in diag:
                            patient_status = diag['last_status']
                            break
                    break  # Found the final status episode
            
            record['patient_status'] = patient_status
            
            # Treatment regime (combination of treatments)
            treatments = []
            if record['total_surgeries'] > 0:
                treatments.append('S')
            if record['total_chemotherapy'] > 0:
                treatments.append('C')
            if record['total_radiotherapy'] > 0:
                treatments.append('R')
            record['treatment_regime'] = '+'.join(treatments) if treatments else 'None'
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Apply binning
        if 'age' in df.columns and 'age_bins' in self.config:
            bins_config = self.config['age_bins']
            df['age_group'] = pd.cut(df['age'], bins=bins_config['bins'], 
                                      labels=bins_config['labels'], right=False)
        
        if 'tumor_max_size_before_surgery' in df.columns and 'size_bins' in self.config:
            bins_config = self.config['size_bins']
            df['size_group'] = pd.cut(df['tumor_max_size_before_surgery'], 
                                       bins=bins_config['bins'],
                                       labels=bins_config['labels'], right=False)
        
        self.processed_data = df
        
        # Show missing data summary
        print(f"\nExtracted {len(df)} patient records with {len(df.columns)} features")
        print(f"✓ Patient count: {len(df)} (should be ~700 after filtering)")
        print(f"✓ Unique patient IDs: {df['patient_id'].nunique()}")
        
        # DETAILED BIOPSY GRADING ANALYSIS
        if 'biopsy_grading' in df.columns:
            print(f"\n  BIOPSY GRADING BREAKDOWN (POST-FILTERING):")
            print(f"    Total records: {len(df)}")
            grading_counts = df['biopsy_grading'].value_counts(dropna=False)
            for grade, count in sorted(grading_counts.items(), key=lambda x: -x[1]):
                grade_label = grade if pd.notna(grade) else '<NaN/null in database>'
                print(f"      {grade_label}: {count} ({count/len(df)*100:.1f}%)")
            
            null_count = df['biopsy_grading'].isna().sum()
            print(f"    Explicit NaN/null values: {null_count} ({null_count/len(df)*100:.1f}%)")
            print(f"    Non-null values: {df['biopsy_grading'].notna().sum()}")
        
        print("\nMissing data summary (these will show as 'Missing' in analyses):")
        missing_summary = []
        key_columns = ['biopsy_grading', 'who_diagnosis_code', 'gender', 'anatomic_region_grouping', 
                      'pathologist_margin_judgement', 'patient_status', 'extremity_tumor', 'whoops']
        for col in key_columns:
            if col in df.columns:
                n_missing = df[col].isna().sum()
                pct_missing = (n_missing / len(df)) * 100
                if n_missing > 0:
                    missing_summary.append(f"  {col}: {n_missing} ({pct_missing:.1f}%)")
        
        if missing_summary:
            print('\n'.join(missing_summary))
        else:
            print("  No missing values detected in key columns")
        
        # FINAL VALIDATION: Ensure no excluded grading values in processed data
        if 'biopsy_grading' in df.columns:
            excluded_gradings = self.config.get('filtering', {}).get('exclude_biopsy_grading', [
                'benign', 'not_a_sarcoma', 'suspicious'
            ])
            
            # Check if any excluded gradings are present (should be ZERO)
            excluded_found = df['biopsy_grading'].dropna().str.lower().isin(
                [g.lower() for g in excluded_gradings]
            ).sum()
            
            if excluded_found > 0:
                print(f"\n⚠️  VALIDATION FAILED: Found {excluded_found} records with excluded gradings!")
                print("   These should have been filtered out. Checking which gradings...")
                problem_grades = df[df['biopsy_grading'].str.lower().isin([g.lower() for g in excluded_gradings])]['biopsy_grading'].value_counts()
                print(f"   Problem gradings found: {problem_grades.to_dict()}")
            else:
                print(f"\n✓ VALIDATION PASSED: No excluded gradings (benign/not_a_sarcoma/suspicious) found")
                print(f"  All {len(df)} patients have valid grading or null/NaN (legitimately missing from database)")
        
        print("="*70 + "\n")
        
        # Export if requested
        if self.config['output']['export_processed_data']:
            output_file = self.output_dir / self.config['output']['processed_data_file']
            df.to_csv(output_file, index=False)
            print(f"Exported processed data to: {output_file}")
        
        return df
    
    def _get_nested_field(self, doc: dict, field_path: str) -> Any:
        """Get nested field value from document using dot notation."""
        parts = field_path.split('.')
        value = doc
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None
        return value
    
    def _fill_missing_values(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Helper function to consistently fill missing values with 'Missing' label.
        
        This should be called AFTER filtering has removed non-malignant cases.
        Any None/NaN values at this point represent truly missing data.
        
        Args:
            df: DataFrame to process
            columns: List of column names to fill missing values for
            
        Returns:
            DataFrame with missing values filled
        """
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns:
                # Check if column is categorical (from pd.cut or similar)
                if pd.api.types.is_categorical_dtype(df_copy[col]):
                    # Add 'Missing' as a valid category first
                    if 'Missing' not in df_copy[col].cat.categories:
                        df_copy[col] = df_copy[col].cat.add_categories(['Missing'])
                    # Now fill NaN with 'Missing'
                    df_copy[col] = df_copy[col].fillna('Missing')
                # Check if column is numeric (int or float) - keep as numeric, don't convert to string
                elif pd.api.types.is_numeric_dtype(df_copy[col]):
                    # For binary columns (0/1), keep them as-is, don't fill NaN with 'Missing'
                    # Only fill NaN for truly missing values, but keep the numeric type
                    pass  # Keep numeric columns as numeric
                else:
                    # For non-categorical, non-numeric columns (strings), handle None, NaN, empty strings
                    df_copy[col] = df_copy[col].fillna('Missing')
                    df_copy[col] = df_copy[col].astype(str).replace({'nan': 'Missing', '': 'Missing'})
        return df_copy
    
    def generate_distribution_plots(self) -> None:
        """Generate distribution plots for patient characteristics and treatments."""
        if self.processed_data is None:
            raise ValueError("No processed data available. Run extract_patient_characteristics first.")
        
        print("\nGenerating distribution plots...")
        
        # Fill missing values for categorical variables using helper function
        df = self._fill_missing_values(
            self.processed_data,
            columns=['gender', 'age_group', 'biopsy_grading', 'who_diagnosis_code', 
                    'anatomic_region_grouping', 'metastasis_present_at_diagnosis', 'size_group']
        )
        
        figsize = tuple(self.config['visualization']['figsize'])
        dpi = self.config['visualization']['dpi']
        
        # 1. Demographics distributions
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Patient Demographics Distribution', fontsize=14, fontweight='bold')
        
        # Gender
        if 'gender' in df.columns:
            gender_counts = df['gender'].value_counts()
            axes[0, 0].bar(gender_counts.index, gender_counts.values)
            axes[0, 0].set_title('Gender Distribution')
            axes[0, 0].set_ylabel('Count')
        
        # Age distribution
        if 'age' in df.columns:
            axes[0, 1].hist(df['age'].dropna(), bins=20, edgecolor='black')
            axes[0, 1].set_title('Age Distribution')
            axes[0, 1].set_xlabel('Age')
            axes[0, 1].set_ylabel('Frequency')
        
        # Age groups
        if 'age_group' in df.columns:
            age_group_counts = df['age_group'].value_counts().sort_index()
            axes[1, 0].bar(range(len(age_group_counts)), age_group_counts.values)
            axes[1, 0].set_xticks(range(len(age_group_counts)))
            axes[1, 0].set_xticklabels(age_group_counts.index, rotation=45)
            axes[1, 0].set_title('Age Group Distribution')
            axes[1, 0].set_ylabel('Count')
        
        # Institution
        if 'institution' in df.columns:
            # Convert to string to handle mixed types (int, list, etc.)
            inst_data = df['institution'].astype(str)
            inst_counts = inst_data.value_counts()
            axes[1, 1].bar(range(len(inst_counts)), inst_counts.values)
            axes[1, 1].set_xticks(range(len(inst_counts)))
            axes[1, 1].set_xticklabels(inst_counts.index, rotation=45)
            axes[1, 1].set_title('Institution Distribution')
            axes[1, 1].set_xlabel('Institution')
            axes[1, 1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'demographics_distribution.{self.config["output"]["plot_format"]}', 
                   dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 2. Tumor characteristics
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Tumor Characteristics Distribution', fontsize=14, fontweight='bold')
        
        # Biopsy grading
        if 'biopsy_grading' in df.columns:
            # Convert to string and handle mixed types
            grading_data = df['biopsy_grading'].astype(str)
            grading_counts = grading_data.value_counts()
            axes[0, 0].bar(range(len(grading_counts)), grading_counts.values)
            axes[0, 0].set_xticks(range(len(grading_counts)))
            axes[0, 0].set_xticklabels(grading_counts.index, rotation=45)
            axes[0, 0].set_title('Biopsy Grading')
            axes[0, 0].set_ylabel('Count')
        
        # WHO diagnosis
        if 'who_diagnosis_code' in df.columns:
            # Convert to string to handle mixed types
            who_data = df['who_diagnosis_code'].astype(str)
            who_counts = who_data.value_counts().head(10)
            axes[0, 1].barh(range(len(who_counts)), who_counts.values)
            axes[0, 1].set_yticks(range(len(who_counts)))
            axes[0, 1].set_yticklabels(who_counts.index, fontsize=8)
            axes[0, 1].set_title('Top 10 WHO Diagnoses')
            axes[0, 1].set_xlabel('Count')
        
        # Anatomic region
        if 'anatomic_region_grouping' in df.columns:
            # Convert to string to handle mixed types
            region_data = df['anatomic_region_grouping'].astype(str)
            region_counts = region_data.value_counts()
            axes[0, 2].barh(range(len(region_counts)), region_counts.values)
            axes[0, 2].set_yticks(range(len(region_counts)))
            axes[0, 2].set_yticklabels(region_counts.index, fontsize=8)
            axes[0, 2].set_title('Anatomic Region Grouping')
            axes[0, 2].set_xlabel('Count')
        
        # Metastasis at diagnosis
        if 'metastasis_present_at_diagnosis' in df.columns:
            met_counts = df['metastasis_present_at_diagnosis'].value_counts()
            axes[1, 0].pie(met_counts.values, labels=met_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Metastasis at Diagnosis')
        
        # Tumor size distribution
        if 'tumor_max_size_before_surgery' in df.columns:
            axes[1, 1].hist(df['tumor_max_size_before_surgery'].dropna(), bins=30, edgecolor='black')
            axes[1, 1].set_title('Tumor Size Distribution')
            axes[1, 1].set_xlabel('Size (mm)')
            axes[1, 1].set_ylabel('Frequency')
        
        # Tumor size groups
        if 'size_group' in df.columns:
            size_counts = df['size_group'].value_counts().sort_index()
            axes[1, 2].bar(range(len(size_counts)), size_counts.values)
            axes[1, 2].set_xticks(range(len(size_counts)))
            axes[1, 2].set_xticklabels(size_counts.index, rotation=45)
            axes[1, 2].set_title('Tumor Size Groups')
            axes[1, 2].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'tumor_characteristics.{self.config["output"]["plot_format"]}',
                   dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # 3. Treatment distributions
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Treatment Distributions', fontsize=14, fontweight='bold')
        
        # Treatment counts
        treatment_vars = ['total_surgeries', 'total_chemotherapy', 'total_radiotherapy']
        for idx, var in enumerate(treatment_vars):
            if var in df.columns:
                counts = df[var].value_counts().sort_index()
                axes[0, idx].bar(counts.index.astype(str), counts.values)
                axes[0, idx].set_title(f'{var.replace("total_", "").title()} Count')
                axes[0, idx].set_xlabel('Number of treatments')
                axes[0, idx].set_ylabel('Patients')
        
        # Treatment regime combinations
        if 'treatment_regime' in df.columns:
            regime_counts = df['treatment_regime'].value_counts().head(10)
            axes[1, 0].barh(range(len(regime_counts)), regime_counts.values)
            axes[1, 0].set_yticks(range(len(regime_counts)))
            axes[1, 0].set_yticklabels(regime_counts.index)
            axes[1, 0].set_title('Treatment Regime Combinations')
            axes[1, 0].set_xlabel('Count')
        
        # Outcomes
        if 'had_metastasis' in df.columns:
            met_counts = df['had_metastasis'].value_counts()
            axes[1, 1].pie(met_counts.values, labels=['No', 'Yes'], autopct='%1.1f%%')
            axes[1, 1].set_title('Patients with Metastasis')
        
        if 'had_local_recurrence' in df.columns:
            rec_counts = df['had_local_recurrence'].value_counts()
            axes[1, 2].pie(rec_counts.values, labels=['No', 'Yes'], autopct='%1.1f%%')
            axes[1, 2].set_title('Patients with Local Recurrence')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / f'treatment_distributions.{self.config["output"]["plot_format"]}',
                   dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"Saved distribution plots to: {self.plots_dir}")
    
    def generate_treatment_by_characteristics(self) -> None:
        """Generate cross-tabulation plots of treatments by patient characteristics."""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        print("\nGenerating treatment by characteristics plots...")
        
        # Fill missing values for categorical variables using helper function
        df = self._fill_missing_values(
            self.processed_data,
            columns=['age_group', 'gender', 'size_group', 'treatment_regime', 'biopsy_grading']
        )
        
        figsize = tuple(self.config['visualization']['figsize'])
        dpi = self.config['visualization']['dpi']
        
        # Treatment regime by age group
        if 'treatment_regime' in df.columns and 'age_group' in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            crosstab = pd.crosstab(df['age_group'], df['treatment_regime'])
            crosstab.plot(kind='bar', stacked=False, ax=ax)
            ax.set_title('Treatment Regimes by Age Group', fontsize=14, fontweight='bold')
            ax.set_xlabel('Age Group')
            ax.set_ylabel('Number of Patients')
            ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'treatment_by_age.{self.config["output"]["plot_format"]}',
                       dpi=dpi, bbox_inches='tight')
            plt.close()
            
            # Save crosstab
            crosstab.to_csv(self.tables_dir / 'treatment_by_age_crosstab.csv')
        
        # Treatment regime by gender
        if 'treatment_regime' in df.columns and 'gender' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            crosstab = pd.crosstab(df['gender'], df['treatment_regime'])
            crosstab.plot(kind='bar', ax=ax)
            ax.set_title('Treatment Regimes by Gender', fontsize=14, fontweight='bold')
            ax.set_xlabel('Gender')
            ax.set_ylabel('Number of Patients')
            ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=0)
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'treatment_by_gender.{self.config["output"]["plot_format"]}',
                       dpi=dpi, bbox_inches='tight')
            plt.close()
            
            crosstab.to_csv(self.tables_dir / 'treatment_by_gender_crosstab.csv')
        
        # Treatment regime by biopsy grading
        if 'treatment_regime' in df.columns and 'biopsy_grading' in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            crosstab = pd.crosstab(df['biopsy_grading'], df['treatment_regime'])
            crosstab.plot(kind='bar', ax=ax)
            ax.set_title('Treatment Regimes by Biopsy Grading', fontsize=14, fontweight='bold')
            ax.set_xlabel('Biopsy Grading')
            ax.set_ylabel('Number of Patients')
            ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'treatment_by_grading.{self.config["output"]["plot_format"]}',
                       dpi=dpi, bbox_inches='tight')
            plt.close()
            
            crosstab.to_csv(self.tables_dir / 'treatment_by_grading_crosstab.csv')
        
        # Treatment regime by tumor size group
        if 'treatment_regime' in df.columns and 'size_group' in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            crosstab = pd.crosstab(df['size_group'], df['treatment_regime'])
            crosstab.plot(kind='bar', ax=ax)
            ax.set_title('Treatment Regimes by Tumor Size', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tumor Size Group')
            ax.set_ylabel('Number of Patients')
            ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'treatment_by_size.{self.config["output"]["plot_format"]}',
                       dpi=dpi, bbox_inches='tight')
            plt.close()
            
            crosstab.to_csv(self.tables_dir / 'treatment_by_size_crosstab.csv')
        
        # Metastasis by treatment regime
        if 'had_metastasis' in df.columns and 'treatment_regime' in df.columns:
            fig, ax = plt.subplots(figsize=figsize)
            crosstab = pd.crosstab(df['treatment_regime'], df['had_metastasis'])
            crosstab.plot(kind='bar', stacked=True, ax=ax)
            ax.set_title('Metastasis Rate by Treatment Regime', fontsize=14, fontweight='bold')
            ax.set_xlabel('Treatment Regime')
            ax.set_ylabel('Number of Patients')
            ax.legend(title='Had Metastasis', labels=['No', 'Yes'])
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'metastasis_by_treatment.{self.config["output"]["plot_format"]}',
                       dpi=dpi, bbox_inches='tight')
            plt.close()
        
        print(f"Saved treatment by characteristics plots to: {self.plots_dir}")
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics for treatments and patient characteristics."""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        df = self.processed_data
        
        print("\nGenerating summary statistics...")
        
        # Overall statistics
        summary_stats = {
            'Total Patients': len(df),
            'Mean Age': df['age'].mean() if 'age' in df.columns else None,
            'Median Age': df['age'].median() if 'age' in df.columns else None,
            'Female Patients (%)': (df['gender'] == 'female').sum() / len(df) * 100 if 'gender' in df.columns else None,
            'Patients with Surgery (%)': (df['total_surgeries'] > 0).sum() / len(df) * 100 if 'total_surgeries' in df.columns else None,
            'Patients with Chemotherapy (%)': (df['total_chemotherapy'] > 0).sum() / len(df) * 100 if 'total_chemotherapy' in df.columns else None,
            'Patients with Radiotherapy (%)': (df['total_radiotherapy'] > 0).sum() / len(df) * 100 if 'total_radiotherapy' in df.columns else None,
            'Patients with Metastasis (%)': df['had_metastasis'].sum() / len(df) * 100 if 'had_metastasis' in df.columns else None,
            'Patients with Recurrence (%)': df['had_local_recurrence'].sum() / len(df) * 100 if 'had_local_recurrence' in df.columns else None,
        }
        
        summary_df = pd.DataFrame(summary_stats.items(), columns=['Metric', 'Value'])
        summary_df.to_csv(self.tables_dir / 'overall_statistics.csv', index=False)
        
        # Treatment regime statistics
        if 'treatment_regime' in df.columns:
            regime_stats = df['treatment_regime'].value_counts().reset_index()
            regime_stats.columns = ['Treatment Regime', 'Count']
            regime_stats['Percentage'] = regime_stats['Count'] / len(df) * 100
            regime_stats.to_csv(self.tables_dir / 'treatment_regime_statistics.csv', index=False)
        
        # Statistics by age group
        if 'age_group' in df.columns:
            age_group_stats = df.groupby('age_group').agg({
                'patient_id': 'count',
                'total_surgeries': 'mean',
                'total_chemotherapy': 'mean',
                'total_radiotherapy': 'mean',
                'had_metastasis': 'mean',
                'had_local_recurrence': 'mean'
            }).round(2)
            age_group_stats.columns = ['N Patients', 'Mean Surgeries', 'Mean Chemo', 
                                        'Mean Radio', 'Metastasis Rate', 'Recurrence Rate']
            age_group_stats.to_csv(self.tables_dir / 'statistics_by_age_group.csv')
        
        print(f"Saved summary statistics to: {self.tables_dir}")
        
        return summary_df
    
    def analyze_treatment_sequences(self) -> None:
        """Analyze treatment sequences from diagnosis_treatment_sequence field."""
        if self.data is None:
            raise ValueError("No data loaded.")
        
        print("\nAnalyzing treatment sequences...")
        
        sequence_patterns = []
        
        for patient in self.data:
            patient_id = patient.get('_id')
            seq = patient.get('diagnosis_treatment_sequence', [])
            
            # Extract sequence of treatment sections
            treatment_seq = []
            for item in seq:
                section = item.get('section')
                if section in ['surgery', 'radiotherapy', 'chemotherapy']:
                    treatment_seq.append(section)
            
            # Create pattern string
            if treatment_seq:
                pattern = ' -> '.join(treatment_seq)
                sequence_patterns.append({
                    'patient_id': patient_id,
                    'sequence': pattern,
                    'n_treatments': len(treatment_seq)
                })
        
        seq_df = pd.DataFrame(sequence_patterns)
        
        # Most common sequences
        if not seq_df.empty:
            top_sequences = seq_df['sequence'].value_counts().head(20).reset_index()
            top_sequences.columns = ['Sequence', 'Count']
            top_sequences['Percentage'] = top_sequences['Count'] / len(seq_df) * 100
            top_sequences.to_csv(self.tables_dir / 'top_treatment_sequences.csv', index=False)
            
            # Visualize top sequences
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(range(len(top_sequences)), top_sequences['Count'])
            ax.set_yticks(range(len(top_sequences)))
            ax.set_yticklabels(top_sequences['Sequence'], fontsize=8)
            ax.set_xlabel('Number of Patients')
            ax.set_title('Top 20 Treatment Sequences', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'treatment_sequences.{self.config["output"]["plot_format"]}',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"Analyzed {len(seq_df)} treatment sequences")
    
    def analyze_treatment_indications(self) -> None:
        """Analyze surgery, radiotherapy, and chemotherapy indications at patient level (first occurrence only)."""
        if self.data is None:
            raise ValueError("No data loaded.")
        
        print("\nAnalyzing treatment indications...")
        
        indication_records = []
        
        for patient in self.data:
            patient_id = patient.get('_id')
            episodes = patient.get('episodes', [])
            
            # Track which treatment types we've seen for this patient
            seen_treatments = {
                'surgery': False,
                'radiotherapy': False,
                'chemotherapy': False
            }
            
            # Process episodes in order to get first occurrence
            for ep_idx, episode in enumerate(episodes):
                # Iterate through treatments within each episode
                treatments = episode.get('treatments', [])
                
                for treatment in treatments:
                    section = treatment.get('section')
                    
                    # Extract surgery indication (first occurrence only)
                    if section == 'surgery' and not seen_treatments['surgery']:
                        surgery_ind = treatment.get('surgery_indication')
                        if surgery_ind and surgery_ind != 'no surgery':
                            indication_records.append({
                                'patient_id': patient_id,
                                'episode': ep_idx + 1,
                                'treatment_type': 'Surgery',
                                'indication': surgery_ind if surgery_ind else 'Missing'
                            })
                            seen_treatments['surgery'] = True
                        elif surgery_ind is None or surgery_ind == 'no surgery':
                            # Add missing indication record
                            indication_records.append({
                                'patient_id': patient_id,
                                'episode': ep_idx + 1,
                                'treatment_type': 'Surgery',
                                'indication': 'Missing'
                            })
                            seen_treatments['surgery'] = True
                    
                    # Extract radiotherapy indication (first occurrence only)
                    if section == 'radiotherapy' and not seen_treatments['radiotherapy']:
                        radio_ind = treatment.get('radiotherapy_indication')
                        if radio_ind and radio_ind != 'no radiotherapy':
                            indication_records.append({
                                'patient_id': patient_id,
                                'episode': ep_idx + 1,
                                'treatment_type': 'Radiotherapy',
                                'indication': radio_ind if radio_ind else 'Missing'
                            })
                            seen_treatments['radiotherapy'] = True
                        elif radio_ind is None or radio_ind == 'no radiotherapy':
                            # Add missing indication record
                            indication_records.append({
                                'patient_id': patient_id,
                                'episode': ep_idx + 1,
                                'treatment_type': 'Radiotherapy',
                                'indication': 'Missing'
                            })
                            seen_treatments['radiotherapy'] = True
                    
                    # Extract chemotherapy/systemic therapy indication (first occurrence only)
                    if section == 'systemic_therapy' and not seen_treatments['chemotherapy']:
                        chemo_ind = treatment.get('systemic_treatment_reason')
                        indication_records.append({
                            'patient_id': patient_id,
                            'episode': ep_idx + 1,
                            'treatment_type': 'Chemotherapy',
                            'indication': chemo_ind if chemo_ind else 'Missing'
                        })
                        seen_treatments['chemotherapy'] = True
        
        if not indication_records:
            print("No indication data found")
            return
        
        ind_df = pd.DataFrame(indication_records)
        
        # Save detailed indication data
        ind_df.to_csv(self.tables_dir / 'treatment_indications_detail.csv', index=False)
        
        # Surgery indications distribution
        surgery_ind = ind_df[ind_df['treatment_type'] == 'Surgery']
        if not surgery_ind.empty:
            surgery_counts = surgery_ind['indication'].value_counts().reset_index()
            surgery_counts.columns = ['Indication', 'Count']
            surgery_counts['Percentage'] = surgery_counts['Count'] / len(surgery_ind) * 100
            surgery_counts.to_csv(self.tables_dir / 'surgery_indications_distribution.csv', index=False)
            
            # Plot surgery indications
            fig, ax = plt.subplots(figsize=(12, 8))
            top_surgery = surgery_counts.head(15)
            ax.barh(range(len(top_surgery)), top_surgery['Count'])
            ax.set_yticks(range(len(top_surgery)))
            ax.set_yticklabels(top_surgery['Indication'], fontsize=9)
            ax.set_xlabel('Number of Cases')
            ax.set_title('Surgery Indications Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'surgery_indications.{self.config["output"]["plot_format"]}',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  Surgery indications: {len(surgery_counts)} unique values")
        
        # Radiotherapy indications distribution
        radio_ind = ind_df[ind_df['treatment_type'] == 'Radiotherapy']
        if not radio_ind.empty:
            radio_counts = radio_ind['indication'].value_counts().reset_index()
            radio_counts.columns = ['Indication', 'Count']
            radio_counts['Percentage'] = radio_counts['Count'] / len(radio_ind) * 100
            radio_counts.to_csv(self.tables_dir / 'radiotherapy_indications_distribution.csv', index=False)
            
            # Plot radiotherapy indications
            fig, ax = plt.subplots(figsize=(12, 8))
            top_radio = radio_counts.head(15)
            ax.barh(range(len(top_radio)), top_radio['Count'])
            ax.set_yticks(range(len(top_radio)))
            ax.set_yticklabels(top_radio['Indication'], fontsize=9)
            ax.set_xlabel('Number of Cases')
            ax.set_title('Radiotherapy Indications Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'radiotherapy_indications.{self.config["output"]["plot_format"]}',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  Radiotherapy indications: {len(radio_counts)} unique values")
        
        # Chemotherapy indications distribution
        chemo_ind = ind_df[ind_df['treatment_type'] == 'Chemotherapy']
        if not chemo_ind.empty:
            chemo_counts = chemo_ind['indication'].value_counts().reset_index()
            chemo_counts.columns = ['Indication', 'Count']
            chemo_counts['Percentage'] = chemo_counts['Count'] / len(chemo_ind) * 100
            chemo_counts.to_csv(self.tables_dir / 'chemotherapy_indications_distribution.csv', index=False)
            
            # Plot chemotherapy indications
            fig, ax = plt.subplots(figsize=(12, 8))
            top_chemo = chemo_counts.head(15)
            ax.barh(range(len(top_chemo)), top_chemo['Count'])
            ax.set_yticks(range(len(top_chemo)))
            ax.set_yticklabels(top_chemo['Indication'], fontsize=9)
            ax.set_xlabel('Number of Cases')
            ax.set_title('Chemotherapy/Systemic Therapy Indications Distribution', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'chemotherapy_indications.{self.config["output"]["plot_format"]}',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  Chemotherapy indications: {len(chemo_counts)} unique values")
        
        # Combined indication overview
        indication_summary = ind_df.groupby('treatment_type')['indication'].agg(['count', 'nunique']).reset_index()
        indication_summary.columns = ['Treatment Type', 'Total Cases', 'Unique Indications']
        indication_summary.to_csv(self.tables_dir / 'indication_summary.csv', index=False)
        
        print(f"Saved indication analysis to: {self.tables_dir}")
    
    def analyze_treatment_regimes_with_timing(self) -> None:
        """Analyze treatment regimes with radiotherapy timing (preoperative vs postoperative)."""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        print("\nAnalyzing treatment regimes with radiotherapy timing...")
        
        # Create enhanced regime classification
        regime_records = []
        
        for patient in self.data:
            patient_id = patient.get('_id')
            episodes = patient.get('episodes', [])
            
            # Track treatments and timing
            has_surgery = False
            has_radiotherapy = False
            has_chemotherapy = False
            radio_timing = None
            
            # Process episodes in order
            for episode in episodes:
                treatments = episode.get('treatments', [])
                
                for treatment in treatments:
                    section = treatment.get('section')
                    
                    if section == 'surgery':
                        has_surgery = True
                    elif section == 'radiotherapy' and not has_radiotherapy:
                        has_radiotherapy = True
                        radio_ind = treatment.get('radiotherapy_indication')
                        # Determine timing
                        if radio_ind in ['preoperative', 'neoadjuvant']:
                            radio_timing = 'preoperative'
                        elif radio_ind in ['postoperative', 'adjuvant']:
                            radio_timing = 'postoperative'
                        else:
                            radio_timing = 'other'
                    elif section == 'systemic_therapy':
                        has_chemotherapy = True
            
            # Classify regime with timing
            if has_surgery and has_radiotherapy and radio_timing:
                regime = f"S+R_{radio_timing}"
            elif has_surgery and not has_radiotherapy and not has_chemotherapy:
                regime = "S_alone"
            elif has_surgery and has_radiotherapy:
                regime = "S+R"
            elif has_surgery and has_chemotherapy and has_radiotherapy:
                regime = "S+C+R"
            elif has_surgery and has_chemotherapy:
                regime = "S+C"
            elif has_surgery:
                regime = "S"
            else:
                regime = "Other"
            
            regime_records.append({
                'patient_id': patient_id,
                'regime_with_timing': regime
            })
        
        # Merge with processed data
        regime_df = pd.DataFrame(regime_records)
        enhanced_df = self.processed_data.merge(regime_df, on='patient_id', how='left')
        
        # Store regime timing in processed data for later use
        self.processed_data = enhanced_df
        
        # Overall distribution
        timing_dist = enhanced_df['regime_with_timing'].value_counts().reset_index()
        timing_dist.columns = ['Treatment Regime', 'Count']
        timing_dist['Percentage'] = timing_dist['Count'] / len(enhanced_df) * 100
        timing_dist.to_csv(self.tables_dir / 'treatment_regime_with_timing.csv', index=False)
        
        print(f"  Treatment regimes with timing: {len(timing_dist)} categories")
        
        # Analyze by moderators
        moderators = []
        if 'age_group' in enhanced_df.columns:
            moderators.append(('age_group', 'Age Group'))
        if 'gender' in enhanced_df.columns:
            moderators.append(('gender', 'Gender'))
        if 'biopsy_grading' in enhanced_df.columns:
            moderators.append(('biopsy_grading', 'Biopsy Grading'))
        if 'size_group' in enhanced_df.columns:
            moderators.append(('size_group', 'Tumor Size'))
        if 'anatomic_region_grouping' in enhanced_df.columns:
            moderators.append(('anatomic_region_grouping', 'Anatomic Region'))
        
        for mod_col, mod_name in moderators:
            # Filter for main regimes of interest
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            enhanced_df['regime_with_timing'] = enhanced_df['regime_with_timing'].fillna('Missing')
            filtered_df = enhanced_df[enhanced_df['regime_with_timing'].isin(main_regimes)]
            
            if not filtered_df.empty:
                # Create crosstab
                crosstab = pd.crosstab(
                    filtered_df[mod_col], 
                    filtered_df['regime_with_timing']
                )
                
                # Save crosstab
                crosstab.to_csv(self.tables_dir / f'regime_timing_by_{mod_col}.csv')
                
                # Generate plot only if crosstab has data
                if not crosstab.empty and crosstab.sum().sum() > 0:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    crosstab.plot(kind='bar', ax=ax, rot=45)
                    ax.set_title(f'Treatment Regimes with Radiotherapy Timing by {mod_name}', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel(mod_name)
                    ax.set_ylabel('Number of Patients')
                    ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'regime_timing_by_{mod_col}.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                
                print(f"  Analyzed regime timing by {mod_name}")
        
        # Special analysis: Age group with grading (reduced age bins)
        if 'age' in enhanced_df.columns and 'biopsy_grading' in enhanced_df.columns:
            print("\n  Creating treatment regime analysis by age (reduced bins) and grading...")
            
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            main_gradings = ['G1', 'G2', 'G3', 'Missing']
            
            # Filter for main regimes
            age_grading_df = enhanced_df[enhanced_df['regime_with_timing'].isin(main_regimes)].copy()
            
            if not age_grading_df.empty:
                # Create reduced age bins (3 bins: ≤50, 51-70, >70)
                age_grading_df['age_group_reduced'] = pd.cut(
                    age_grading_df['age'],
                    bins=[0, 50, 70, float('inf')],
                    labels=['≤50 years', '51-70 years', '>70 years'],
                    include_lowest=True
                )
                age_grading_df['age_group_reduced'] = age_grading_df['age_group_reduced'].astype(str).replace('nan', 'Missing')
                
                # Fill missing grading
                age_grading_df['biopsy_grading'] = age_grading_df['biopsy_grading'].fillna('Missing')
                
                # Filter for main gradings
                age_grading_filtered = age_grading_df[
                    age_grading_df['biopsy_grading'].isin(main_gradings)
                ]
                
                if not age_grading_filtered.empty:
                    # Create multi-level crosstab: (age_group × grading) vs regime
                    age_grading_crosstab = pd.crosstab(
                        [age_grading_filtered['age_group_reduced'], age_grading_filtered['biopsy_grading']],
                        age_grading_filtered['regime_with_timing'],
                        margins=True
                    )
                    
                    # Save crosstab
                    age_grading_crosstab.to_csv(
                        self.tables_dir / 'regime_timing_by_age_group_and_grading.csv'
                    )
                    
                    # Create visualization (without margins)
                    age_grading_plot = pd.crosstab(
                        [age_grading_filtered['age_group_reduced'], age_grading_filtered['biopsy_grading']],
                        age_grading_filtered['regime_with_timing'],
                        margins=False
                    )
                    
                    # Reorder columns
                    available_regimes = [r for r in main_regimes if r in age_grading_plot.columns]
                    if available_regimes:
                        age_grading_plot = age_grading_plot[available_regimes]
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    age_grading_plot.plot(kind='bar', ax=ax, rot=45,
                                         color=['#2ecc71', '#e74c3c', '#3498db'])
                    ax.set_title('Treatment Regimes with RT Timing by Age Group and Grading', 
                               fontsize=14, fontweight='bold')
                    ax.set_xlabel('Age Group and Grading', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'regime_timing_by_age_group.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Created regime timing by age group (reduced bins) and grading")
                    print(f"    Age bins: ≤50 years, 51-70 years, >70 years")
                    print(f"    Total patients: {len(age_grading_filtered)}")
        
        print(f"Saved treatment regime timing analysis to: {self.tables_dir}")
    
    def analyze_clinical_outcomes(self) -> None:
        """Analyze key clinical outcomes: metastasis, recurrence, margins, WHO diagnosis, extremity, and survival."""
        if self.processed_data is None:
            raise ValueError("No processed data available.")
        
        print("\nAnalyzing key clinical outcomes...")
        print(f"\n  INITIAL DATA STATE:")
        print(f"    Total rows in processed_data: {len(self.processed_data)}")
        print(f"    Unique patient_ids: {self.processed_data['patient_id'].nunique() if 'patient_id' in self.processed_data.columns else 'N/A'}")
        
        if 'biopsy_grading' in self.processed_data.columns:
            print(f"    Biopsy grading BEFORE _fill_missing_values:")
            grading_before = self.processed_data['biopsy_grading'].value_counts(dropna=False)
            print(f"      {grading_before.to_dict()}")
            print(f"      NaN count: {self.processed_data['biopsy_grading'].isna().sum()}")
        
        # Fill missing values for analysis using helper function
        df = self._fill_missing_values(
            self.processed_data,
            columns=['metastasis_present_at_diagnosis', 'extremity_tumor', 'biopsy_grading', 
                    'who_diagnosis_code', 'pathologist_margin_judgement', 'patient_status', 
                    'regime_with_timing', 'treatment_regime', 'anatomic_region_grouping', 'whoops']
        )
        
        print(f"\n  AFTER _fill_missing_values:")
        print(f"    Total rows in df: {len(df)}")
        if 'biopsy_grading' in df.columns:
            print(f"    Biopsy grading AFTER _fill_missing_values:")
            grading_after = df['biopsy_grading'].value_counts(dropna=False)
            print(f"      {grading_after.to_dict()}")
        
        # 1. Overall outcome statistics
        outcome_stats = {}
        
        if 'metastasis_present_at_diagnosis' in df.columns:
            met_at_dx = (df['metastasis_present_at_diagnosis'] == 'yes').sum()
            outcome_stats['Metastasis at Diagnosis (n)'] = met_at_dx
            outcome_stats['Metastasis at Diagnosis (%)'] = met_at_dx / len(df) * 100
        
        if 'had_metastasis' in df.columns:
            outcome_stats['Developed Metastasis (n)'] = df['had_metastasis'].sum()
            outcome_stats['Developed Metastasis (%)'] = df['had_metastasis'].sum() / len(df) * 100
        
        if 'had_local_recurrence' in df.columns:
            outcome_stats['Local Recurrence (n)'] = df['had_local_recurrence'].sum()
            outcome_stats['Local Recurrence (%)'] = df['had_local_recurrence'].sum() / len(df) * 100
        
        if 'patient_status' in df.columns:
            dod_count = (df['patient_status'] == 'DOD').sum()
            awd_count = (df['patient_status'] == 'AWD').sum()
            ned_count = (df['patient_status'] == 'NED').sum()
            outcome_stats['Dead of Disease (n)'] = dod_count
            outcome_stats['Dead of Disease (%)'] = dod_count / len(df) * 100
            outcome_stats['Alive with Disease (n)'] = awd_count
            outcome_stats['Alive with Disease (%)'] = awd_count / len(df) * 100
            outcome_stats['No Evidence of Disease (n)'] = ned_count
            outcome_stats['No Evidence of Disease (%)'] = ned_count / len(df) * 100
        
        if 'pathologist_margin_judgement' in df.columns:
            r0_count = (df['pathologist_margin_judgement'] == 'R0').sum()
            outcome_stats['R0 Margins (n)'] = r0_count
            outcome_stats['R0 Margins (%)'] = r0_count / len(df) * 100
        
        if 'extremity_tumor' in df.columns:
            extremity_count = (df['extremity_tumor'] == 1).sum()
            outcome_stats['Extremity Tumors (n)'] = extremity_count
            outcome_stats['Extremity Tumors (%)'] = extremity_count / len(df) * 100
        
        outcome_df = pd.DataFrame(outcome_stats.items(), columns=['Clinical Outcome', 'Value'])
        outcome_df.to_csv(self.tables_dir / 'clinical_outcomes_summary.csv', index=False)
        
        # 1b. Patient counts and outcomes by treatment regime
        if 'regime_with_timing' in df.columns:
            print("\n  Creating patient counts and outcomes by treatment regime...")
            
            # Focus on main regimes
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            regime_outcome_records = []
            
            for regime in main_regimes:
                regime_df = df[df['regime_with_timing'] == regime]
                n_patients = len(regime_df)
                
                if n_patients > 0:
                    record = {
                        'Treatment Regime': regime,
                        'Total Patients': n_patients
                    }
                    
                    # Metastasis
                    if 'had_metastasis' in df.columns:
                        met_count = regime_df['had_metastasis'].sum()
                        record['Metastasis (n)'] = met_count
                        record['Metastasis (%)'] = round(met_count / n_patients * 100, 1)
                    
                    # Local recurrence
                    if 'had_local_recurrence' in df.columns:
                        rec_count = regime_df['had_local_recurrence'].sum()
                        record['Local Recurrence (n)'] = rec_count
                        record['Local Recurrence (%)'] = round(rec_count / n_patients * 100, 1)
                    
                    # Patient status: DOD
                    if 'patient_status' in df.columns:
                        dod_count = (regime_df['patient_status'] == 'DOD').sum()
                        record['DOD (n)'] = dod_count
                        record['DOD (%)'] = round(dod_count / n_patients * 100, 1)
                        
                        # AWD
                        awd_count = (regime_df['patient_status'] == 'AWD').sum()
                        record['AWD (n)'] = awd_count
                        record['AWD (%)'] = round(awd_count / n_patients * 100, 1)
                        
                        # NED
                        ned_count = (regime_df['patient_status'] == 'NED').sum()
                        record['NED (n)'] = ned_count
                        record['NED (%)'] = round(ned_count / n_patients * 100, 1)
                    
                    regime_outcome_records.append(record)
            
            # Create DataFrame
            if regime_outcome_records:
                regime_outcomes_df = pd.DataFrame(regime_outcome_records)
                regime_outcomes_df.to_csv(self.tables_dir / 'outcomes_by_regime.csv', index=False)
                
                print(f"  ✓ Created outcomes by regime table")
                print(f"    Regimes analyzed: {', '.join(main_regimes)}")
                print(f"\n  Summary:")
                for record in regime_outcome_records:
                    print(f"    {record['Treatment Regime']}: {record['Total Patients']} patients")
        
        # 1c. Patient counts and outcomes by treatment regime AND grading
        if 'regime_with_timing' in df.columns and 'biopsy_grading' in df.columns:
            print("\n  Creating patient counts and outcomes by treatment regime AND grading...")
            
            # Focus on main regimes and gradings
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            main_gradings = ['G1', 'G2', 'G3', 'Missing']
            
            regime_grading_outcome_records = []
            
            for regime in main_regimes:
                for grading in main_gradings:
                    regime_grading_df = df[
                        (df['regime_with_timing'] == regime) & 
                        (df['biopsy_grading'] == grading)
                    ]
                    n_patients = len(regime_grading_df)
                    
                    if n_patients > 0:
                        record = {
                            'Treatment Regime': regime,
                            'Grading': grading,
                            'Total Patients': n_patients
                        }
                        
                        # Metastasis
                        if 'had_metastasis' in df.columns:
                            met_count = regime_grading_df['had_metastasis'].sum()
                            record['Metastasis (n)'] = met_count
                            record['Metastasis (%)'] = round(met_count / n_patients * 100, 1) if n_patients > 0 else 0
                        
                        # Local recurrence
                        if 'had_local_recurrence' in df.columns:
                            rec_count = regime_grading_df['had_local_recurrence'].sum()
                            record['Local Recurrence (n)'] = rec_count
                            record['Local Recurrence (%)'] = round(rec_count / n_patients * 100, 1) if n_patients > 0 else 0
                        
                        # Patient status: DOD
                        if 'patient_status' in df.columns:
                            dod_count = (regime_grading_df['patient_status'] == 'DOD').sum()
                            record['DOD (n)'] = dod_count
                            record['DOD (%)'] = round(dod_count / n_patients * 100, 1) if n_patients > 0 else 0
                            
                            # AWD
                            awd_count = (regime_grading_df['patient_status'] == 'AWD').sum()
                            record['AWD (n)'] = awd_count
                            record['AWD (%)'] = round(awd_count / n_patients * 100, 1) if n_patients > 0 else 0
                            
                            # NED
                            ned_count = (regime_grading_df['patient_status'] == 'NED').sum()
                            record['NED (n)'] = ned_count
                            record['NED (%)'] = round(ned_count / n_patients * 100, 1) if n_patients > 0 else 0
                        
                        regime_grading_outcome_records.append(record)
            
            # Create DataFrame
            if regime_grading_outcome_records:
                regime_grading_outcomes_df = pd.DataFrame(regime_grading_outcome_records)
                regime_grading_outcomes_df.to_csv(self.tables_dir / 'outcomes_by_regime_and_grading.csv', index=False)
                
                print(f"  ✓ Created outcomes by regime and grading table")
                print(f"    Total regime × grading combinations: {len(regime_grading_outcome_records)}")
                print(f"    Saved to: outcomes_by_regime_and_grading.csv")
        
        # 2. WHO diagnosis distribution
        if 'who_diagnosis_code' in df.columns:
            who_counts = df['who_diagnosis_code'].value_counts().reset_index()
            who_counts.columns = ['WHO Diagnosis', 'Count']
            who_counts['Percentage'] = who_counts['Count'] / len(df) * 100
            who_counts.to_csv(self.tables_dir / 'who_diagnosis_distribution.csv', index=False)
            
            # Plot top 15 WHO diagnoses
            fig, ax = plt.subplots(figsize=(12, 10))
            top_who = who_counts.head(15)
            ax.barh(range(len(top_who)), top_who['Count'])
            ax.set_yticks(range(len(top_who)))
            ax.set_yticklabels(top_who['WHO Diagnosis'], fontsize=9)
            ax.set_xlabel('Number of Patients')
            ax.set_title('WHO Diagnosis Distribution (Top 15)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.plots_dir / f'who_diagnosis_distribution.{self.config["output"]["plot_format"]}',
                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
            plt.close()
            
            print(f"  WHO diagnoses: {len(who_counts)} unique subtypes")
        
        # 2a. WHO diagnosis by biopsy grading and treatment regime for key subtypes
        if 'who_diagnosis_code' in df.columns and 'biopsy_grading' in df.columns and 'regime_with_timing' in df.columns:
            # Key subtypes to analyze (corrected names)
            key_subtypes = ['liposarcoma', 'fibroblastic_tumor', 'undifferentiated_sarcoma', 'leiomyosarcoma']
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            
            # Filter for key subtypes and main regimes
            who_analysis_df = df[
                df['who_diagnosis_code'].isin(key_subtypes) & 
                df['regime_with_timing'].isin(main_regimes)
            ].copy()
            
            if not who_analysis_df.empty:
                # 1. WHO by grading (overall)
                who_grading_crosstab = pd.crosstab(
                    who_analysis_df['who_diagnosis_code'],
                    who_analysis_df['biopsy_grading']
                )
                who_grading_crosstab.to_csv(self.tables_dir / 'who_by_grading_key_subtypes.csv')
                
                # 2. WHO by regime (overall)
                who_regime_crosstab = pd.crosstab(
                    who_analysis_df['who_diagnosis_code'],
                    who_analysis_df['regime_with_timing']
                )
                who_regime_crosstab.to_csv(self.tables_dir / 'who_by_regime_key_subtypes.csv')
                
                # 3. Create combined visualization: 4 subtypes showing grading by regime
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                axes = axes.flatten()
                
                for idx, subtype in enumerate(key_subtypes):
                    subtype_df = who_analysis_df[who_analysis_df['who_diagnosis_code'] == subtype]
                    
                    if not subtype_df.empty:
                        # Create crosstab: grading x regime
                        grading_regime_crosstab = pd.crosstab(
                            subtype_df['biopsy_grading'],
                            subtype_df['regime_with_timing']
                        )
                        
                        # Reorder columns to main regimes
                        available_regimes = [r for r in main_regimes if r in grading_regime_crosstab.columns]
                        if available_regimes:
                            grading_regime_crosstab = grading_regime_crosstab[available_regimes]
                        
                        # Save individual crosstab
                        grading_regime_crosstab.to_csv(
                            self.tables_dir / f'who_{subtype}_grading_by_regime.csv'
                        )
                        
                        # Plot
                        if not grading_regime_crosstab.empty and grading_regime_crosstab.sum().sum() > 0:
                            grading_regime_crosstab.plot(kind='bar', ax=axes[idx], rot=45)
                            axes[idx].set_title(f'{subtype.replace("_", " ").title()}', 
                                              fontsize=12, fontweight='bold')
                            axes[idx].set_xlabel('Biopsy Grading', fontsize=10)
                            axes[idx].set_ylabel('Number of Patients', fontsize=10)
                            axes[idx].legend(title='Treatment Regime', 
                                           labels=[l.replace('S+R_', 'S+RT ').replace('_', ' ').title() 
                                                  for l in available_regimes],
                                           fontsize=8)
                        else:
                            axes[idx].text(0.5, 0.5, f'No data for {subtype}', 
                                         ha='center', va='center', transform=axes[idx].transAxes)
                            axes[idx].set_title(f'{subtype.replace("_", " ").title()}', 
                                              fontsize=12, fontweight='bold')
                
                fig.suptitle('Key WHO Diagnoses: Biopsy Grading by Treatment Regime', 
                           fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'who_grading_by_regime_key_subtypes.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  Analyzed WHO diagnosis by grading and regime for {len(key_subtypes)} key subtypes")
        
        # 2b. Metastasis at diagnosis by treatment regime timing
        if 'metastasis_present_at_diagnosis' in df.columns and 'regime_with_timing' in df.columns:
            # Filter for main surgical regimes
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            met_regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
            
            if not met_regime_df.empty:
                # Create crosstab
                met_crosstab = pd.crosstab(
                    met_regime_df['metastasis_present_at_diagnosis'],
                    met_regime_df['regime_with_timing']
                )
                
                # Reorder columns if they exist
                available_regimes = [r for r in main_regimes if r in met_crosstab.columns]
                if available_regimes:
                    met_crosstab = met_crosstab[available_regimes]
                
                # Save crosstab
                met_crosstab.to_csv(self.tables_dir / 'metastasis_at_diagnosis_by_regime_timing.csv')
                
                # Plot only if crosstab has data
                if not met_crosstab.empty and met_crosstab.sum().sum() > 0:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    met_crosstab.T.plot(kind='bar', ax=ax, rot=45, stacked=False,
                                        color=['#3498db', '#e74c3c', '#95a5a6'])
                    ax.set_xlabel('Treatment Regime', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.set_title('Metastasis Present at Diagnosis by Treatment Regime', 
                               fontsize=14, fontweight='bold')
                    ax.legend(title='Metastasis at Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Rename x-axis labels for clarity
                    labels = [label.get_text().replace('S+R_', 'S+RT ').replace('_', ' ').title() 
                             for label in ax.get_xticklabels()]
                    ax.set_xticklabels(labels)
                    
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'metastasis_at_diagnosis_by_regime.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Analyzed metastasis at diagnosis by treatment regime timing")
        
        # 2c. WHOOPS by treatment regime timing
        if 'whoops' in df.columns and 'regime_with_timing' in df.columns:
            # Filter for main surgical regimes
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            whoops_regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
            
            if not whoops_regime_df.empty:
                # Convert whoops to readable labels (whoops is 0/1 as integers)
                whoops_regime_df['whoops_label'] = whoops_regime_df['whoops'].map({
                    0: 'No WHOOPS',
                    1: 'WHOOPS'
                })
                
                # Fill any unmapped values (NaN) as Missing
                whoops_regime_df['whoops_label'] = whoops_regime_df['whoops_label'].fillna('Missing')
                
                # Create crosstab
                whoops_crosstab = pd.crosstab(
                    whoops_regime_df['whoops_label'],
                    whoops_regime_df['regime_with_timing']
                )
                
                # Reorder columns if they exist
                available_regimes = [r for r in main_regimes if r in whoops_crosstab.columns]
                if available_regimes:
                    whoops_crosstab = whoops_crosstab[available_regimes]
                
                # Save crosstab
                whoops_crosstab.to_csv(self.tables_dir / 'whoops_by_regime_timing.csv')
                
                print(f"  WHOOPS distribution: {dict(whoops_regime_df['whoops'].value_counts())}")
                print(f"  WHOOPS label distribution: {dict(whoops_regime_df['whoops_label'].value_counts())}")
                
                # Plot only if crosstab has data
                if not whoops_crosstab.empty and whoops_crosstab.sum().sum() > 0:
                    fig, ax = plt.subplots(figsize=(12, 7))
                    whoops_crosstab.T.plot(kind='bar', ax=ax, rot=45, stacked=False)
                    ax.set_xlabel('Treatment Regime', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.set_title('WHOOPS (Unplanned Excision) by Treatment Regime', 
                               fontsize=14, fontweight='bold')
                    ax.legend(title='WHOOPS Status', bbox_to_anchor=(1.05, 1), loc='upper left')
                    
                    # Rename x-axis labels for clarity
                    labels = [label.get_text().replace('S+R_', 'S+RT ').replace('_', ' ').title() 
                             for label in ax.get_xticklabels()]
                    ax.set_xticklabels(labels)
                    
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'whoops_by_regime.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Analyzed WHOOPS by treatment regime timing")
        
        # 2d. Top 15 WHO diagnosis by treatment regime timing
        if 'who_diagnosis_code' in df.columns and 'regime_with_timing' in df.columns:
            # Filter for main surgical regimes
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            who_regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
            
            if not who_regime_df.empty and who_regime_df['who_diagnosis_code'].notna().sum() > 0:
                # Get top 15 WHO diagnoses overall
                top_15_who = who_regime_df['who_diagnosis_code'].value_counts().head(15).index.tolist()
                
                # Filter for top 15 only
                who_top_df = who_regime_df[who_regime_df['who_diagnosis_code'].isin(top_15_who)].copy()
                
                if not who_top_df.empty:
                    # Create crosstab
                    who_crosstab = pd.crosstab(
                        who_top_df['who_diagnosis_code'],
                        who_top_df['regime_with_timing']
                    )
                    
                    # Reorder columns if they exist
                    available_regimes = [r for r in main_regimes if r in who_crosstab.columns]
                    if available_regimes:
                        who_crosstab = who_crosstab[available_regimes]
                    
                    # Sort by total count descending
                    who_crosstab['Total'] = who_crosstab.sum(axis=1)
                    who_crosstab = who_crosstab.sort_values('Total', ascending=False)
                    who_crosstab = who_crosstab.drop('Total', axis=1)
                    
                    # Save crosstab
                    who_crosstab.to_csv(self.tables_dir / 'who_top15_by_regime_timing.csv')
                    
                    # Plot only if crosstab has data
                    if not who_crosstab.empty and who_crosstab.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(14, 10))
                        who_crosstab.plot(kind='barh', ax=ax, stacked=False)
                        ax.set_ylabel('WHO Diagnosis Code', fontsize=12)
                        ax.set_xlabel('Number of Patients', fontsize=12)
                        ax.set_title('Top 15 WHO Diagnoses by Treatment Regime', 
                                   fontsize=14, fontweight='bold')
                        ax.legend(title='Treatment Regime', 
                                 labels=[l.replace('S+R_', 'S+RT ').replace('_', ' ').title() for l in available_regimes],
                                 bbox_to_anchor=(1.05, 1), loc='upper left')
                        
                        plt.tight_layout()
                        plt.savefig(self.plots_dir / f'who_top15_by_regime.{self.config["output"]["plot_format"]}',
                                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                        plt.close()
                        
                        print(f"  Analyzed top 15 WHO diagnoses by treatment regime timing")
        
        # 3. Margin status distribution
        if 'pathologist_margin_judgement' in df.columns:
            margin_counts = df['pathologist_margin_judgement'].value_counts().reset_index()
            margin_counts.columns = ['Margin Status', 'Count']
            margin_counts['Percentage'] = margin_counts['Count'] / len(df) * 100
            margin_counts.to_csv(self.tables_dir / 'margin_status_distribution.csv', index=False)
            
            # Plot margin distribution by treatment regime timing
            if 'regime_with_timing' in df.columns:
                # Filter for main surgical regimes
                main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
                margin_regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
                
                if not margin_regime_df.empty:
                    # Create crosstab
                    margin_crosstab = pd.crosstab(
                        margin_regime_df['pathologist_margin_judgement'],
                        margin_regime_df['regime_with_timing']
                    )
                    
                    # Reorder columns if they exist
                    available_regimes = [r for r in main_regimes if r in margin_crosstab.columns]
                    margin_crosstab = margin_crosstab[available_regimes]
                    
                    # Save crosstab
                    margin_crosstab.to_csv(self.tables_dir / 'margin_status_by_regime_timing.csv')
                    
                    # Plot only if crosstab has data
                    if not margin_crosstab.empty and margin_crosstab.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(12, 7))
                        margin_crosstab.plot(kind='bar', ax=ax, rot=45, 
                                            color=['#2ecc71', '#e74c3c', '#3498db'])
                        ax.set_xlabel('Margin Status', fontsize=12)
                        ax.set_ylabel('Number of Patients', fontsize=12)
                        ax.set_title('Surgical Margin Status by Treatment Regime Timing', 
                                   fontsize=14, fontweight='bold')
                        ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                        plt.tight_layout()
                        plt.savefig(self.plots_dir / f'margin_status_distribution.{self.config["output"]["plot_format"]}',
                                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                        plt.close()
                    
                    print(f"  Analyzed margin status by regime timing")
            
            # Margin status grouped by BOTH regime AND grading
            if 'regime_with_timing' in df.columns and 'biopsy_grading' in df.columns:
                print("\n  Creating margin status distribution grouped by regime AND grading...")
                
                main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
                main_gradings = ['G1', 'G2', 'G3', 'Missing']
                
                # Filter for main regimes and gradings
                margin_combined_df = df[
                    df['regime_with_timing'].isin(main_regimes) & 
                    df['biopsy_grading'].isin(main_gradings)
                ].copy()
                
                print(f"    Filtered dataframe size: {len(margin_combined_df)}")
                print(f"    Non-null margin values: {margin_combined_df['pathologist_margin_judgement'].notna().sum()}")
                
                if not margin_combined_df.empty and margin_combined_df['pathologist_margin_judgement'].notna().sum() > 0:
                    # Create multi-level crosstab: (regime × grading) vs margin status
                    margin_regime_grading_crosstab = pd.crosstab(
                        [margin_combined_df['regime_with_timing'], margin_combined_df['biopsy_grading']],
                        margin_combined_df['pathologist_margin_judgement'],
                        margins=True
                    )
                    
                    # Save crosstab
                    margin_regime_grading_crosstab.to_csv(
                        self.tables_dir / 'margin_status_by_regime_and_grading.csv'
                    )
                    
                    # Create visualization - grouped bar plot
                    margin_plot_data = pd.crosstab(
                        [margin_combined_df['regime_with_timing'], margin_combined_df['biopsy_grading']],
                        margin_combined_df['pathologist_margin_judgement'],
                        margins=False
                    )
                    
                    fig, ax = plt.subplots(figsize=(16, 8))
                    
                    if not margin_plot_data.empty:
                        margin_plot_data.plot(kind='bar', ax=ax, rot=45)
                        ax.set_title('Surgical Margin Status by Treatment Regime and Grading', 
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('Treatment Regime and Grading', fontsize=12)
                        ax.set_ylabel('Number of Patients', fontsize=12)
                        ax.legend(title='Margin Status', bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'margin_status_by_regime_and_grading.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Created margin status distribution grouped by regime AND grading")
                    print(f"    Total patients in analysis: {len(margin_combined_df)}")
                    print(f"    Saved to: margin_status_by_regime_and_grading.csv")
                else:
                    print(f"  ⚠ Skipping margin combined analysis: insufficient data")
            
            else:
                # Fallback to simple bar plot if regime timing not available
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(margin_counts['Margin Status'], margin_counts['Count'])
                ax.set_xlabel('Margin Status')
                ax.set_ylabel('Number of Patients')
                ax.set_title('Surgical Margin Status Distribution', fontsize=14, fontweight='bold')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'margin_status_distribution.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
        
        # 3b. Extremity tumor distribution by treatment regime and grading
        if 'extremity_tumor' in df.columns:
            # Convert extremity_tumor to readable labels (handle NaN properly)
            extremity_labeled = df['extremity_tumor'].apply(
                lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
            )
            
            # Overall distribution
            extremity_counts = extremity_labeled.value_counts().reset_index()
            extremity_counts.columns = ['Extremity Location', 'Count']
            extremity_counts['Percentage'] = extremity_counts['Count'] / len(df) * 100
            extremity_counts.to_csv(self.tables_dir / 'extremity_tumor_distribution.csv', index=False)
            
            # Extremity by treatment regime timing
            if 'regime_with_timing' in df.columns:
                # Filter for main surgical regimes
                main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
                extremity_regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
                extremity_regime_df['extremity_location'] = extremity_regime_df['extremity_tumor'].apply(
                    lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
                )
                
                if not extremity_regime_df.empty:
                    # Create crosstab
                    extremity_regime_crosstab = pd.crosstab(
                        extremity_regime_df['extremity_location'],
                        extremity_regime_df['regime_with_timing']
                    )
                    
                    # Reorder columns if they exist
                    available_regimes = [r for r in main_regimes if r in extremity_regime_crosstab.columns]
                    extremity_regime_crosstab = extremity_regime_crosstab[available_regimes]
                    
                    # Save crosstab
                    extremity_regime_crosstab.to_csv(self.tables_dir / 'extremity_tumor_by_regime_timing.csv')
                    
                    print(f"  Extremity by regime timing crosstab shape: {extremity_regime_crosstab.shape}")
                    print(f"  Total count: {extremity_regime_crosstab.sum().sum()}")
            
            # Extremity by biopsy grading
            if 'biopsy_grading' in df.columns:
                print(f"\n  DEBUG: Extremity by grading analysis")
                print(f"    Total patients in df: {len(df)}")
                print(f"    Unique patients: {df['patient_id'].nunique() if 'patient_id' in df.columns else 'N/A'}")
                
                # Check if df has correct number of patients (~700)
                if len(df) != df['patient_id'].nunique():
                    print(f"    ⚠️  WARNING: Row count ({len(df)}) != unique patients ({df['patient_id'].nunique()})")
                    print(f"    This suggests duplicated patient records!")
                
                print(f"    Biopsy grading value counts (including NaN):")
                grading_value_counts = df['biopsy_grading'].value_counts(dropna=False)
                for grade, count in grading_value_counts.items():
                    grade_display = grade if pd.notna(grade) else '<NaN/null>'
                    print(f"      {grade_display}: {count}")
                print(f"    Sum of all grading counts: {grading_value_counts.sum()}")
                
                extremity_grading_df = df.copy()
                extremity_grading_df['extremity_location'] = extremity_grading_df['extremity_tumor'].apply(
                    lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
                )
                
                print(f"    Extremity location value counts:")
                extremity_loc_counts = extremity_grading_df['extremity_location'].value_counts()
                for loc, count in extremity_loc_counts.items():
                    print(f"      {loc}: {count}")
                print(f"    Sum of all extremity location counts: {extremity_loc_counts.sum()}")
                
                # Create crosstab
                extremity_grading_crosstab = pd.crosstab(
                    extremity_grading_df['extremity_location'],
                    extremity_grading_df['biopsy_grading']
                )
                
                # Save crosstab
                extremity_grading_crosstab.to_csv(self.tables_dir / 'extremity_tumor_by_grading.csv')
                
                print(f"    Crosstab shape: {extremity_grading_crosstab.shape}")
                print(f"    Total count in crosstab: {extremity_grading_crosstab.sum().sum()}")
                print(f"    Crosstab should equal total patients ({len(df)} rows)")
                print(f"    Crosstab breakdown:")
                print(extremity_grading_crosstab)
            
            # Combined plot: Extremity by regime and grading
            if 'regime_with_timing' in df.columns and 'biopsy_grading' in df.columns:
                main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
                main_gradings = ['G1', 'G2', 'G3']
                
                combined_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
                combined_df['extremity_location'] = combined_df['extremity_tumor'].apply(
                    lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
                )
                
                # Filter for main gradings (optional - include Missing if present)
                combined_df_filtered = combined_df[
                    combined_df['biopsy_grading'].isin(main_gradings + ['Missing'])
                ].copy()
                
                # Create filtered crosstab for grading (from regime-filtered data)
                extremity_grading_crosstab_filtered = pd.crosstab(
                    combined_df['extremity_location'],
                    combined_df['biopsy_grading']
                )
                
                if not combined_df_filtered.empty:
                    # Create figure with 2 subplots side by side
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                    
                    # Left plot: Extremity by regime timing
                    if not extremity_regime_df.empty and extremity_regime_crosstab.sum().sum() > 0:
                        extremity_regime_crosstab.plot(kind='bar', ax=ax1, rot=45,
                                                       color=['#2ecc71', '#e74c3c', '#3498db'])
                        ax1.set_xlabel('Extremity Location', fontsize=12)
                        ax1.set_ylabel('Number of Patients', fontsize=12)
                        ax1.set_title('Extremity Tumor by Treatment Regime Timing',
                                     fontsize=13, fontweight='bold')
                        ax1.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax1.text(0.5, 0.5, 'No data available', 
                                ha='center', va='center', transform=ax1.transAxes)
                        ax1.set_title('Extremity Tumor by Treatment Regime Timing',
                                     fontsize=13, fontweight='bold')
                    
                    # Right plot: Extremity by biopsy grading (filtered by main regimes)
                    if not extremity_grading_crosstab_filtered.empty and extremity_grading_crosstab_filtered.sum().sum() > 0:
                        extremity_grading_crosstab_filtered.plot(kind='bar', ax=ax2, rot=45)
                        ax2.set_xlabel('Extremity Location', fontsize=12)
                        ax2.set_ylabel('Number of Patients', fontsize=12)
                        ax2.set_title('Extremity Tumor by Biopsy Grading\n(Main Regimes Only)',
                                     fontsize=13, fontweight='bold')
                        ax2.legend(title='Biopsy Grading', bbox_to_anchor=(1.05, 1), loc='upper left')
                    else:
                        ax2.text(0.5, 0.5, 'No data available',
                                ha='center', va='center', transform=ax2.transAxes)
                        ax2.set_title('Extremity Tumor by Biopsy Grading\n(Main Regimes Only)',
                                     fontsize=13, fontweight='bold')
                    
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'extremity_tumor_by_regime_and_grading.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  Saved extremity tumor analysis plots")
            
            # Extremity grouped by BOTH regime AND grading (combined crosstab)
            if 'regime_with_timing' in df.columns and 'biopsy_grading' in df.columns:
                print("\n  Creating extremity distribution grouped by regime AND grading...")
                
                main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
                main_gradings = ['G1', 'G2', 'G3', 'Missing']
                
                # Filter for main regimes and gradings (including Missing)
                extremity_combined_df = df[
                    df['regime_with_timing'].isin(main_regimes) & 
                    df['biopsy_grading'].isin(main_gradings)
                ].copy()
                
                print(f"    Filtered dataframe size: {len(extremity_combined_df)}")
                print(f"    Non-null extremity values: {extremity_combined_df['extremity_tumor'].notna().sum()}")
                
                if not extremity_combined_df.empty:
                    extremity_combined_df['extremity_location'] = extremity_combined_df['extremity_tumor'].apply(
                        lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
                    )
                    
                    # Create multi-level crosstab: (regime × grading) vs extremity location
                    extremity_regime_grading_crosstab = pd.crosstab(
                        [extremity_combined_df['regime_with_timing'], extremity_combined_df['biopsy_grading']],
                        extremity_combined_df['extremity_location'],
                        margins=True
                    )
                    
                    # Save crosstab
                    extremity_regime_grading_crosstab.to_csv(
                        self.tables_dir / 'extremity_by_regime_and_grading_combined.csv'
                    )
                    
                    # Create visualization - grouped bar plot
                    extremity_plot_data = pd.crosstab(
                        [extremity_combined_df['regime_with_timing'], extremity_combined_df['biopsy_grading']],
                        extremity_combined_df['extremity_location'],
                        margins=False
                    )
                    
                    fig, ax = plt.subplots(figsize=(14, 8))
                    
                    if not extremity_plot_data.empty:
                        extremity_plot_data.plot(kind='bar', ax=ax, rot=45,
                                               color=['#e74c3c', '#2ecc71', '#95a5a6'])
                        ax.set_title('Extremity Location by Treatment Regime and Grading', 
                                    fontsize=14, fontweight='bold')
                        ax.set_xlabel('Treatment Regime and Grading', fontsize=12)
                        ax.set_ylabel('Number of Patients', fontsize=12)
                        ax.legend(title='Location', bbox_to_anchor=(1.05, 1), loc='upper left')
                        ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(self.plots_dir / f'extremity_by_regime_and_grading_combined.{self.config["output"]["plot_format"]}',
                               dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                    plt.close()
                    
                    print(f"  ✓ Created extremity distribution grouped by regime AND grading")
                    print(f"    Total patients in analysis: {len(extremity_combined_df)}")
                    print(f"    Saved to: extremity_by_regime_and_grading_combined.csv")
                else:
                    print(f"  ⚠ Skipping extremity combined analysis: no data after filtering")
        
        # 4. Outcomes by key moderators
        moderators = []
        if 'extremity_tumor' in df.columns:
            # Convert extremity_tumor to readable labels (handle NaN properly)
            df['extremity_location'] = df['extremity_tumor'].apply(
                lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
            )
            moderators.append(('extremity_location', 'Extremity vs Non-Extremity'))
        if 'biopsy_grading' in df.columns:
            moderators.append(('biopsy_grading', 'Biopsy Grading'))
        if 'who_diagnosis_code' in df.columns:
            moderators.append(('who_diagnosis_code', 'WHO Diagnosis'))
        if 'pathologist_margin_judgement' in df.columns:
            moderators.append(('pathologist_margin_judgement', 'Margin Status'))
        
        outcomes = []
        if 'had_metastasis' in df.columns:
            outcomes.append(('had_metastasis', 'Developed Metastasis'))
        if 'had_local_recurrence' in df.columns:
            outcomes.append(('had_local_recurrence', 'Local Recurrence'))
        if 'patient_status' in df.columns:
            # Create DOD indicator from status field (already filled with 'Missing' at top)
            df['died_of_disease'] = (df['patient_status'] == 'DOD').astype(int)
            outcomes.append(('died_of_disease', 'Dead of Disease'))
        
        # Generate crosstabs for key combinations
        for mod_col, mod_name in moderators[:2]:  # Limit to first 2 moderators to avoid too many plots
            for outcome_col, outcome_name in outcomes:
                if mod_col in df.columns and outcome_col in df.columns:
                    # Create crosstab
                    crosstab = pd.crosstab(df[mod_col], df[outcome_col])
                    # Rename columns based on actual values present
                    if len(crosstab.columns) == 2:
                        crosstab.columns = ['No', 'Yes']
                    elif len(crosstab.columns) == 1:
                        crosstab.columns = ['Yes'] if crosstab.columns[0] == 1 else ['No']
                    
                    # Save crosstab
                    crosstab.to_csv(self.tables_dir / f'{outcome_col}_by_{mod_col}.csv')
                    
                    # Plot only if crosstab has data
                    if not crosstab.empty and crosstab.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        crosstab.plot(kind='bar', ax=ax, rot=45)
                        ax.set_title(f'{outcome_name} by {mod_name}', fontsize=14, fontweight='bold')
                        ax.set_xlabel(mod_name)
                        ax.set_ylabel('Number of Patients')
                        ax.legend(title=outcome_name)
                        plt.tight_layout()
                        plt.savefig(self.plots_dir / f'{outcome_col}_by_{mod_col}.{self.config["output"]["plot_format"]}',
                                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                        plt.close()
        
        # 5. Treatment regime by clinical outcomes
        if 'treatment_regime' in df.columns:
            for outcome_col, outcome_name in outcomes:
                if outcome_col in df.columns:
                    # Create crosstab
                    crosstab = pd.crosstab(df['treatment_regime'], df[outcome_col])
                    # Rename columns based on actual values present
                    if len(crosstab.columns) == 2:
                        crosstab.columns = ['No', 'Yes']
                    elif len(crosstab.columns) == 1:
                        crosstab.columns = ['Yes'] if crosstab.columns[0] == 1 else ['No']
                    
                    # Save crosstab
                    crosstab.to_csv(self.tables_dir / f'{outcome_col}_by_treatment_regime.csv')
                    
                    # Plot only if crosstab has data
                    if not crosstab.empty and crosstab.sum().sum() > 0:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        crosstab.plot(kind='bar', ax=ax, rot=45)
                        ax.set_title(f'{outcome_name} by Treatment Regime', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Treatment Regime')
                        ax.set_ylabel('Number of Patients')
                        ax.legend(title=outcome_name)
                        plt.tight_layout()
                        plt.savefig(self.plots_dir / f'{outcome_col}_by_treatment_regime.{self.config["output"]["plot_format"]}',
                                   dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                        plt.close()
                    
                    print(f"  Analyzed {outcome_name} by treatment regime")
        
        # 6. Treatment regime TIMING by clinical outcomes
        if 'regime_with_timing' in df.columns:
            # Filter for main regimes with timing detail (already filled with 'Missing' at top)
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            timing_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
            
            if not timing_df.empty:
                for outcome_col, outcome_name in outcomes:
                    if outcome_col in timing_df.columns:
                        # Create crosstab with counts
                        crosstab = pd.crosstab(timing_df['regime_with_timing'], timing_df[outcome_col])
                        # Rename columns based on actual values present
                        if len(crosstab.columns) == 2:
                            crosstab.columns = ['No', 'Yes']
                        elif len(crosstab.columns) == 1:
                            crosstab.columns = ['Yes'] if crosstab.columns[0] == 1 else ['No']
                        
                        # Save crosstab
                        crosstab.to_csv(self.tables_dir / f'{outcome_col}_by_regime_timing.csv')
                        
                        # Plot only if crosstab has data and 'Yes' column exists
                        if not crosstab.empty and 'Yes' in crosstab.columns and crosstab['Yes'].sum() > 0:
                            fig, ax = plt.subplots(figsize=(10, 6))
                            crosstab['Yes'].plot(kind='bar', ax=ax, rot=45, color=['#2ecc71', '#e74c3c', '#3498db'])
                            ax.set_title(f'{outcome_name} by Treatment Regime Timing', fontsize=14, fontweight='bold')
                            ax.set_xlabel('Treatment Regime')
                            ax.set_ylabel('Number of Patients with Outcome')
                            plt.tight_layout()
                            plt.savefig(self.plots_dir / f'{outcome_col}_by_regime_timing.{self.config["output"]["plot_format"]}',
                                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                            plt.close()
                        
                        print(f"  Analyzed {outcome_name} by regime timing")
                
                # Create combined plot with all outcomes for each regime timing
                # Prepare data for combined visualization
                combined_data = []
                for regime in main_regimes:
                    regime_subset = timing_df[timing_df['regime_with_timing'] == regime]
                    total_patients = len(regime_subset)
                    
                    row_data = {'Regime': regime, 'Total Patients': total_patients}
                    
                    # Add outcome counts
                    if 'had_metastasis' in regime_subset.columns:
                        row_data['Metastasis'] = regime_subset['had_metastasis'].sum()
                    if 'had_local_recurrence' in regime_subset.columns:
                        row_data['Local Recurrence'] = regime_subset['had_local_recurrence'].sum()
                    if 'died_of_disease' in regime_subset.columns:
                        row_data['DOD'] = regime_subset['died_of_disease'].sum()
                    
                    # Add patient status counts
                    if 'patient_status' in regime_subset.columns:
                        row_data['NED'] = (regime_subset['patient_status'] == 'NED').sum()
                        row_data['AWD'] = (regime_subset['patient_status'] == 'AWD').sum()
                    
                    combined_data.append(row_data)
                
                combined_df = pd.DataFrame(combined_data)
                combined_df.set_index('Regime', inplace=True)
                
                # Save combined data
                combined_df.to_csv(self.tables_dir / 'regime_timing_outcomes_combined.csv')
                
                # Create grouped bar plot
                fig, ax = plt.subplots(figsize=(14, 8))
                combined_df.plot(kind='bar', ax=ax, rot=45, 
                               color=['#95a5a6', '#e74c3c', '#f39c12', '#c0392b', '#2ecc71', '#3498db'])
                ax.set_title('Treatment Regime Timing: Patient Volume and Clinical Outcomes', 
                           fontsize=14, fontweight='bold')
                ax.set_xlabel('Treatment Regime', fontsize=12)
                ax.set_ylabel('Number of Patients', fontsize=12)
                ax.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'regime_timing_outcomes_combined.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
                
                print("  Created combined regime timing outcomes visualization")
        
        # 7. Combined analysis: Outcomes by regime AND moderators
        if 'regime_with_timing' in df.columns:
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            regime_df = df[df['regime_with_timing'].isin(main_regimes)].copy()
            
            # Define outcomes for combined analysis
            combined_outcomes = []
            if 'had_metastasis' in regime_df.columns:
                combined_outcomes.append(('had_metastasis', 'Developed Metastasis'))
            if 'had_local_recurrence' in regime_df.columns:
                combined_outcomes.append(('had_local_recurrence', 'Local Recurrence'))
            
            # Define moderators for combined analysis
            combined_moderators = []
            if 'extremity_tumor' in regime_df.columns:
                regime_df['extremity_location'] = regime_df['extremity_tumor'].apply(
                    lambda x: 'Missing' if pd.isna(x) else ('Extremity' if x == 1 else 'Non-Extremity')
                )
                combined_moderators.append(('extremity_location', 'Extremity Location'))
            if 'biopsy_grading' in regime_df.columns:
                combined_moderators.append(('biopsy_grading', 'Biopsy Grading'))
            
            # Add WHO diagnosis (subtypes) - filter for key sarcoma types and aggregate by grading
            if 'who_diagnosis_code' in regime_df.columns and 'biopsy_grading' in regime_df.columns:
                # Define key sarcoma subtypes
                key_subtypes = [
                    'liposarcoma',
                    'fibrosarcoma', 
                    'undifferentiated_sarcoma',
                    'leiomyosarcoma'
                ]
                
                # Filter and combine WHO diagnosis with grading
                def combine_who_grading(row):
                    who = row.get('who_diagnosis_code')
                    grade = row.get('biopsy_grading')
                    
                    # Check if WHO is one of the key subtypes
                    if pd.notna(who) and who in key_subtypes:
                        # Combine with grading
                        if pd.notna(grade) and grade in ['G1', 'G2', 'G3']:
                            # Format: "Liposarcoma G1"
                            who_formatted = who.replace('_', ' ').title()
                            return f"{who_formatted} {grade}"
                        else:
                            # WHO known but grading missing
                            who_formatted = who.replace('_', ' ').title()
                            return f"{who_formatted} (Grade Missing)"
                    else:
                        # WHO not in key subtypes or missing
                        return 'Other/Missing'
                
                regime_df['who_grading_combined'] = regime_df.apply(combine_who_grading, axis=1)
                combined_moderators.append(('who_grading_combined', 'Key Subtypes by Grade'))
            
            
            # Add tumor size groups
            if 'initial_size' in regime_df.columns:
                regime_df['size_group'] = pd.cut(
                    regime_df['initial_size'],
                    bins=[0, 50, 100, 150, float('inf')],
                    labels=['≤50mm', '51-100mm', '101-150mm', '>150mm'],
                    include_lowest=True
                )
                regime_df['size_group'] = regime_df['size_group'].astype(str).replace('nan', 'Missing')
                combined_moderators.append(('size_group', 'Tumor Size'))
            
            # Add WHOOPS
            if 'whoops' in regime_df.columns:
                regime_df['whoops_label'] = regime_df['whoops'].apply(
                    lambda x: 'Missing' if pd.isna(x) else ('WHOOPS' if x == 1 else 'Planned Surgery')
                )
                combined_moderators.append(('whoops_label', 'WHOOPS Status'))
            
            # Add patient status
            if 'patient_status' in regime_df.columns:
                regime_df['status_clean'] = regime_df['patient_status'].fillna('Missing')
                combined_moderators.append(('status_clean', 'Patient Status'))
            
            # Add age groups
            if 'age_at_admission' in regime_df.columns:
                regime_df['age_group'] = pd.cut(
                    regime_df['age_at_admission'],
                    bins=[0, 40, 60, 80, float('inf')],
                    labels=['≤40', '41-60', '61-80', '>80'],
                    include_lowest=True
                )
                regime_df['age_group'] = regime_df['age_group'].astype(str).replace('nan', 'Missing')
                combined_moderators.append(('age_group', 'Age Group'))
            
            
            # Generate combined crosstabs and visualizations
            for outcome_col, outcome_name in combined_outcomes:
                for mod_col, mod_name in combined_moderators:
                    if outcome_col in regime_df.columns and mod_col in regime_df.columns:
                        # Create multi-level crosstab: regime x moderator showing outcome
                        crosstab = pd.crosstab(
                            [regime_df['regime_with_timing'], regime_df[mod_col]], 
                            regime_df[outcome_col],
                            margins=False
                        )
                        
                        # Rename columns based on actual values present
                        if len(crosstab.columns) == 2:
                            crosstab.columns = ['No', 'Yes']
                        elif len(crosstab.columns) == 1:
                            crosstab.columns = ['Yes'] if crosstab.columns[0] == 1 else ['No']
                        
                        # Save crosstab
                        crosstab.to_csv(self.tables_dir / f'{outcome_col}_by_regime_and_{mod_col}.csv')
                        
                        # Create visualization if data exists
                        if not crosstab.empty and 'Yes' in crosstab.columns and crosstab['Yes'].sum() > 0:
                            fig, ax = plt.subplots(figsize=(14, 7))
                            
                            # Prepare data for grouped bar plot
                            plot_data = crosstab['Yes'].unstack(level=0)
                            plot_data.plot(kind='bar', ax=ax, rot=45,
                                         color=['#2ecc71', '#e74c3c', '#3498db'])
                            
                            ax.set_title(f'{outcome_name} by Treatment Regime and {mod_name}',
                                       fontsize=14, fontweight='bold')
                            ax.set_xlabel(mod_name, fontsize=12)
                            ax.set_ylabel('Number of Patients with Outcome', fontsize=12)
                            ax.legend(title='Treatment Regime', bbox_to_anchor=(1.05, 1), loc='upper left')
                            ax.grid(axis='y', alpha=0.3)
                            
                            plt.tight_layout()
                            plt.savefig(self.plots_dir / f'{outcome_col}_by_regime_and_{mod_col}.{self.config["output"]["plot_format"]}',
                                       dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                            plt.close()
                            
                            print(f"  Created combined analysis: {outcome_name} by regime and {mod_name}")
            
            # 8. Comprehensive grid plot: All outcomes x moderators by regime
            if combined_outcomes and combined_moderators:
                n_outcomes = len(combined_outcomes)
                n_moderators = len(combined_moderators)
                
                # Create grid: rows = outcomes, columns = moderators
                # Adjust size based on number of subplots
                fig, axes = plt.subplots(n_outcomes, n_moderators, 
                                        figsize=(6 * n_moderators, 5 * n_outcomes),
                                        squeeze=False)  # Always return 2D array
                
                # Plot each combination
                for i, (outcome_col, outcome_name) in enumerate(combined_outcomes):
                    for j, (mod_col, mod_name) in enumerate(combined_moderators):
                        ax = axes[i, j]
                        
                        if outcome_col in regime_df.columns and mod_col in regime_df.columns:
                            # Create crosstab
                            crosstab = pd.crosstab(
                                [regime_df['regime_with_timing'], regime_df[mod_col]], 
                                regime_df[outcome_col],
                                margins=False
                            )
                            
                            # Rename columns
                            if len(crosstab.columns) == 2:
                                crosstab.columns = ['No', 'Yes']
                            elif len(crosstab.columns) == 1:
                                crosstab.columns = ['Yes'] if crosstab.columns[0] == 1 else ['No']
                            
                            # Plot if Yes column exists
                            if 'Yes' in crosstab.columns and crosstab['Yes'].sum() > 0:
                                plot_data = crosstab['Yes'].unstack(level=0)
                                plot_data.plot(kind='bar', ax=ax, rot=45,
                                             color=['#2ecc71', '#e74c3c', '#3498db'])
                                ax.set_title(f'{outcome_name} by {mod_name}', 
                                           fontsize=10, fontweight='bold', pad=10)
                                ax.set_xlabel(mod_name, fontsize=9)
                                ax.set_ylabel('Number of Patients', fontsize=9)
                                ax.tick_params(axis='both', labelsize=8)
                                ax.grid(axis='y', alpha=0.3)
                                
                                # Only show legend on top-right plot
                                if i == 0 and j == n_moderators - 1:
                                    ax.legend(title='Regime', fontsize=8, title_fontsize=9,
                                            bbox_to_anchor=(1.05, 1), loc='upper left')
                                else:
                                    ax.legend().set_visible(False)
                            else:
                                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                                       transform=ax.transAxes, fontsize=10, color='gray')
                                ax.set_title(f'{outcome_name} by {mod_name}', 
                                           fontsize=10, fontweight='bold', pad=10)
                                ax.set_xlabel('')
                                ax.set_ylabel('')
                        else:
                            ax.text(0.5, 0.5, 'Data not available', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=10, color='gray')
                            ax.set_title(f'{outcome_name} by {mod_name}', 
                                       fontsize=10, fontweight='bold', pad=10)
                            ax.set_xlabel('')
                            ax.set_ylabel('')
                
                plt.suptitle('Clinical Outcomes by Treatment Regime and Patient Characteristics',
                           fontsize=14, fontweight='bold', y=0.998)
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'outcomes_by_regime_grid.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  Created comprehensive grid plot: All outcomes by regime and moderators ({n_outcomes}x{n_moderators} grid)")
        
        # 9. Age analysis by treatment regime and grading
        if 'regime_with_timing' in df.columns and 'age_at_admission' in df.columns and 'biopsy_grading' in df.columns:
            print("\n  Creating age distribution analysis by regime and grading...")
            
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            main_gradings = ['G1', 'G2', 'G3', 'Missing']
            
            # Filter for main regimes and valid grading (including Missing)
            age_analysis_df = df[
                df['regime_with_timing'].isin(main_regimes) & 
                df['biopsy_grading'].isin(main_gradings)
            ].copy()
            
            print(f"    Filtered dataframe size: {len(age_analysis_df)}")
            print(f"    Non-null age values: {age_analysis_df['age_at_admission'].notna().sum()}")
            
            if not age_analysis_df.empty and age_analysis_df['age_at_admission'].notna().sum() > 0:
                # Create age bins
                age_analysis_df['age_bin'] = pd.cut(
                    age_analysis_df['age_at_admission'],
                    bins=[0, 40, 60, 80, float('inf')],
                    labels=['≤40', '41-60', '61-80', '>80'],
                    include_lowest=True
                )
                age_analysis_df['age_bin'] = age_analysis_df['age_bin'].astype(str).replace('nan', 'Missing')
                
                # Create crosstab: regime × grading showing age bin distribution
                age_by_regime_grading = pd.crosstab(
                    [age_analysis_df['regime_with_timing'], age_analysis_df['biopsy_grading']], 
                    age_analysis_df['age_bin'],
                    margins=True
                )
                
                # Save to CSV
                age_by_regime_grading.to_csv(self.tables_dir / 'age_by_regime_and_grading.csv')
                
                # Create visualization - grouped bar plot (without margins for plotting)
                age_plot_data = pd.crosstab(
                    [age_analysis_df['regime_with_timing'], age_analysis_df['biopsy_grading']], 
                    age_analysis_df['age_bin'],
                    margins=False
                )
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                if not age_plot_data.empty:
                    # Plot grouped bars
                    age_plot_data.plot(kind='bar', ax=ax, rot=45,
                                      color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
                    ax.set_title('Age Distribution by Treatment Regime and Grading', 
                                fontsize=14, fontweight='bold')
                    ax.set_xlabel('Treatment Regime and Grading', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.legend(title='Age Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'age_by_regime_and_grading.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Created age distribution analysis by regime and grading")
                print(f"    Total patients in analysis: {len(age_analysis_df)}")
                print(f"    Age distribution saved to: age_by_regime_and_grading.csv")
            else:
                print(f"  ⚠ Skipping age analysis: insufficient data (filtered patients={len(age_analysis_df)}, with age={age_analysis_df['age_at_admission'].notna().sum()})")
        else:
            missing_cols = []
            if 'regime_with_timing' not in df.columns:
                missing_cols.append('regime_with_timing')
            if 'age_at_admission' not in df.columns:
                missing_cols.append('age_at_admission')
            if 'biopsy_grading' not in df.columns:
                missing_cols.append('biopsy_grading')
            print(f"  ⚠ Skipping age analysis: missing columns: {', '.join(missing_cols)}")
        
        # 10. Tumor size analysis by treatment regime and grading
        if 'regime_with_timing' in df.columns and 'initial_size' in df.columns and 'biopsy_grading' in df.columns:
            print("\n  Creating tumor size distribution analysis by regime and grading...")
            
            main_regimes = ['S_alone', 'S+R_preoperative', 'S+R_postoperative']
            main_gradings = ['G1', 'G2', 'G3', 'Missing']
            
            # Filter for main regimes and valid grading (including Missing)
            size_analysis_df = df[
                df['regime_with_timing'].isin(main_regimes) & 
                df['biopsy_grading'].isin(main_gradings)
            ].copy()
            
            print(f"    Filtered dataframe size: {len(size_analysis_df)}")
            print(f"    Non-null size values: {size_analysis_df['initial_size'].notna().sum()}")
            
            if not size_analysis_df.empty and size_analysis_df['initial_size'].notna().sum() > 0:
                # Create size bins
                size_analysis_df['size_bin'] = pd.cut(
                    size_analysis_df['initial_size'],
                    bins=[0, 50, 100, 150, float('inf')],
                    labels=['≤50mm', '51-100mm', '101-150mm', '>150mm'],
                    include_lowest=True
                )
                size_analysis_df['size_bin'] = size_analysis_df['size_bin'].astype(str).replace('nan', 'Missing')
                
                # Create crosstab: regime × grading showing size bin distribution
                size_by_regime_grading = pd.crosstab(
                    [size_analysis_df['regime_with_timing'], size_analysis_df['biopsy_grading']], 
                    size_analysis_df['size_bin'],
                    margins=True
                )
                
                # Save to CSV
                size_by_regime_grading.to_csv(self.tables_dir / 'size_by_regime_and_grading.csv')
                
                # Create visualization - grouped bar plot (without margins for plotting)
                size_plot_data = pd.crosstab(
                    [size_analysis_df['regime_with_timing'], size_analysis_df['biopsy_grading']], 
                    size_analysis_df['size_bin'],
                    margins=False
                )
                
                fig, ax = plt.subplots(figsize=(14, 8))
                
                if not size_plot_data.empty:
                    # Plot grouped bars
                    size_plot_data.plot(kind='bar', ax=ax, rot=45,
                                       color=['#27ae60', '#f39c12', '#e67e22', '#c0392b'])
                    ax.set_title('Tumor Size Distribution by Treatment Regime and Grading', 
                                fontsize=14, fontweight='bold')
                    ax.set_xlabel('Treatment Regime and Grading', fontsize=12)
                    ax.set_ylabel('Number of Patients', fontsize=12)
                    ax.legend(title='Size Category', bbox_to_anchor=(1.05, 1), loc='upper left')
                    ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(self.plots_dir / f'size_by_regime_and_grading.{self.config["output"]["plot_format"]}',
                           dpi=self.config['visualization']['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"  ✓ Created tumor size distribution analysis by regime and grading")
                print(f"    Total patients in analysis: {len(size_analysis_df)}")
                print(f"    Size distribution saved to: size_by_regime_and_grading.csv")
            else:
                print(f"  ⚠ Skipping size analysis: insufficient data (filtered patients={len(size_analysis_df)}, with size={size_analysis_df['initial_size'].notna().sum()})")
        else:
            missing_cols = []
            if 'regime_with_timing' not in df.columns:
                missing_cols.append('regime_with_timing')
            if 'initial_size' not in df.columns:
                missing_cols.append('initial_size')
            if 'biopsy_grading' not in df.columns:
                missing_cols.append('biopsy_grading')
            print(f"  ⚠ Skipping size analysis: missing columns: {', '.join(missing_cols)}")
        
        print(f"Saved clinical outcomes analysis to: {self.tables_dir}")
    
    def generate_html_report(self) -> None:
        """Generate comprehensive HTML summary report."""
        print("\nGenerating HTML summary report...")
        
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Treatment Distribution Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 30px; }}
                .summary-box {{ background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
                .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            </style>
        </head>
        <body>
            <h1>Exploratory Data Analysis: Treatment Distributions and Patient Characteristics</h1>
            <p><strong>Generated:</strong> {date}</p>
        """
        
        from datetime import datetime
        html_content = html_content.format(date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add summary statistics
        if (self.tables_dir / 'overall_statistics.csv').exists():
            stats_df = pd.read_csv(self.tables_dir / 'overall_statistics.csv')
            html_content += """
            <div class="summary-box">
                <h2>Overall Statistics</h2>
                <div style="text-align: center;">
            """
            for _, row in stats_df.iterrows():
                if pd.notna(row['Value']):
                    value = f"{row['Value']:.2f}" if isinstance(row['Value'], float) else str(row['Value'])
                    html_content += f"""
                    <div class="metric">
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{row['Metric']}</div>
                    </div>
                    """
            html_content += """
                </div>
            </div>
            """
        
        # Add outcomes by regime table
        if (self.tables_dir / 'outcomes_by_regime.csv').exists():
            outcomes_regime_df = pd.read_csv(self.tables_dir / 'outcomes_by_regime.csv')
            html_content += """
            <div class="summary-box">
                <h2>Patient Counts and Clinical Outcomes by Treatment Regime</h2>
                <p>Summary of patient counts and key clinical outcomes (metastasis, local recurrence, patient status) by main treatment regimes.</p>
                """ + outcomes_regime_df.to_html(index=False) + """
            </div>
            """
        
        # Add outcomes by regime and grading table
        if (self.tables_dir / 'outcomes_by_regime_and_grading.csv').exists():
            outcomes_regime_grading_df = pd.read_csv(self.tables_dir / 'outcomes_by_regime_and_grading.csv')
            html_content += """
            <div class="summary-box">
                <h2>Patient Counts and Clinical Outcomes by Treatment Regime and Grading</h2>
                <p>Detailed breakdown of patient counts and key clinical outcomes by treatment regime and tumor grading (G1, G2, G3, Missing).</p>
                """ + outcomes_regime_grading_df.to_html(index=False) + """
            </div>
            """
        
        # Add plots
        html_content += '<div class="summary-box"><h2>Visualizations</h2>'
        
        plot_titles = {
            'demographics_distribution': 'Patient Demographics',
            'tumor_characteristics': 'Tumor Characteristics',
            'treatment_distributions': 'Treatment Distributions',
            'treatment_by_age': 'Treatments by Age Group',
            'treatment_by_gender': 'Treatments by Gender',
            'treatment_by_grading': 'Treatments by Biopsy Grading',
            'treatment_by_size': 'Treatments by Tumor Size',
            'treatment_sequences': 'Common Treatment Sequences',
            'surgery_indications': 'Surgery Indications Distribution',
            'radiotherapy_indications': 'Radiotherapy Indications Distribution',
            'chemotherapy_indications': 'Chemotherapy/Systemic Therapy Indications Distribution',
            'regime_timing_by_age_group': 'Treatment Regimes with RT Timing by Age',
            'regime_timing_by_gender': 'Treatment Regimes with RT Timing by Gender',
            'regime_timing_by_biopsy_grading': 'Treatment Regimes with RT Timing by Grading',
            'regime_timing_by_size_group': 'Treatment Regimes with RT Timing by Tumor Size',
            'regime_timing_by_anatomic_region_grouping': 'Treatment Regimes with RT Timing by Region',
            'who_diagnosis_distribution': 'WHO Diagnosis Distribution',
            'who_grading_by_regime_key_subtypes': 'Key WHO Diagnoses: Grading by Treatment Regime (4 Subtypes)',
            'metastasis_at_diagnosis_by_regime': 'Metastasis Present at Diagnosis by Treatment Regime',
            'whoops_by_regime': 'WHOOPS (Unplanned Excision) by Treatment Regime',
            'who_top15_by_regime': 'Top 15 WHO Diagnoses by Treatment Regime',
            'margin_status_distribution': 'Surgical Margin Status by Treatment Regime Timing',
            'margin_status_by_regime_and_grading': 'Surgical Margin Status by Treatment Regime and Grading',
            'extremity_tumor_by_regime_and_grading': 'Extremity Tumor Distribution by Treatment Regime and Biopsy Grading',
            'extremity_by_regime_and_grading_combined': 'Extremity Location by Treatment Regime and Grading (Combined)',
            'had_metastasis_by_extremity_location': 'Metastasis by Extremity Location',
            'had_metastasis_by_biopsy_grading': 'Metastasis by Tumor Grade',
            'had_local_recurrence_by_extremity_location': 'Local Recurrence by Extremity Location',
            'had_local_recurrence_by_biopsy_grading': 'Local Recurrence by Tumor Grade',
            'died_of_disease_by_extremity_location': 'Survival (DOD) by Extremity Location',
            'died_of_disease_by_biopsy_grading': 'Survival (DOD) by Tumor Grade',
            'had_metastasis_by_treatment_regime': 'Metastasis by Treatment Regime',
            'had_local_recurrence_by_treatment_regime': 'Local Recurrence by Treatment Regime',
            'died_of_disease_by_treatment_regime': 'Survival (DOD) by Treatment Regime',
            'had_metastasis_by_regime_timing': 'Metastasis by Regime Timing (S alone vs S+R pre/post)',
            'had_local_recurrence_by_regime_timing': 'Local Recurrence by Regime Timing (S alone vs S+R pre/post)',
            'died_of_disease_by_regime_timing': 'Survival (DOD) by Regime Timing (S alone vs S+R pre/post)',
            'regime_timing_outcomes_combined': 'Combined View: Patient Volume and Outcomes by Regime Timing',
            'had_metastasis_by_regime_and_extremity_location': 'Metastasis by Treatment Regime and Extremity Location',
            'had_metastasis_by_regime_and_biopsy_grading': 'Metastasis by Treatment Regime and Tumor Grade',
            'had_local_recurrence_by_regime_and_extremity_location': 'Local Recurrence by Treatment Regime and Extremity Location',
            'had_local_recurrence_by_regime_and_biopsy_grading': 'Local Recurrence by Treatment Regime and Tumor Grade',
            'outcomes_by_regime_grid': 'Comprehensive Grid: All Outcomes by Regime and Patient Characteristics',
            'age_by_regime_and_grading': 'Age Distribution by Treatment Regime and Tumor Grading',
            'size_by_regime_and_grading': 'Tumor Size Distribution by Treatment Regime and Grading'
        }
        
        for plot_name, title in plot_titles.items():
            plot_file = self.plots_dir / f'{plot_name}.{self.config["output"]["plot_format"]}'
            if plot_file.exists():
                rel_path = plot_file.relative_to(self.reports_dir.parent)
                html_content += f"""
                <div class="plot-container">
                    <h3>{title}</h3>
                    <img src="../{rel_path}" alt="{title}">
                </div>
                """
        
        html_content += '</div>'
        
        # Add treatment regime table
        if (self.tables_dir / 'treatment_regime_statistics.csv').exists():
            regime_df = pd.read_csv(self.tables_dir / 'treatment_regime_statistics.csv')
            html_content += """
            <div class="summary-box">
                <h2>Treatment Regime Statistics</h2>
                """ + regime_df.to_html(index=False) + """
            </div>
            """
        
        # Add treatment regime with timing table
        if (self.tables_dir / 'treatment_regime_with_timing.csv').exists():
            regime_timing_df = pd.read_csv(self.tables_dir / 'treatment_regime_with_timing.csv')
            html_content += """
            <div class="summary-box">
                <h2>Treatment Regimes with Radiotherapy Timing</h2>
                """ + regime_timing_df.to_html(index=False) + """
            </div>
            """
        
        # Add indication summary table
        if (self.tables_dir / 'indication_summary.csv').exists():
            indication_summary_df = pd.read_csv(self.tables_dir / 'indication_summary.csv')
            html_content += """
            <div class="summary-box">
                <h2>Treatment Indication Summary</h2>
                """ + indication_summary_df.to_html(index=False) + """
            </div>
            """
        
        # Add surgery indications table
        if (self.tables_dir / 'surgery_indications_distribution.csv').exists():
            surgery_ind_df = pd.read_csv(self.tables_dir / 'surgery_indications_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>Surgery Indications Distribution</h2>
                """ + surgery_ind_df.head(20).to_html(index=False) + """
            </div>
            """
        
        # Add radiotherapy indications table
        if (self.tables_dir / 'radiotherapy_indications_distribution.csv').exists():
            radio_ind_df = pd.read_csv(self.tables_dir / 'radiotherapy_indications_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>Radiotherapy Indications Distribution</h2>
                """ + radio_ind_df.head(20).to_html(index=False) + """
            </div>
            """
        
        # Add chemotherapy indications table
        if (self.tables_dir / 'chemotherapy_indications_distribution.csv').exists():
            chemo_ind_df = pd.read_csv(self.tables_dir / 'chemotherapy_indications_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>Chemotherapy/Systemic Therapy Indications Distribution</h2>
                """ + chemo_ind_df.head(20).to_html(index=False) + """
            </div>
            """
        
        # Add clinical outcomes summary table
        if (self.tables_dir / 'clinical_outcomes_summary.csv').exists():
            outcomes_df = pd.read_csv(self.tables_dir / 'clinical_outcomes_summary.csv')
            html_content += """
            <div class="summary-box">
                <h2>Clinical Outcomes Summary</h2>
                """ + outcomes_df.to_html(index=False) + """
            </div>
            """
        
        # Add WHO diagnosis distribution table
        if (self.tables_dir / 'who_diagnosis_distribution.csv').exists():
            who_df = pd.read_csv(self.tables_dir / 'who_diagnosis_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>WHO Diagnosis Distribution (Top 20)</h2>
                """ + who_df.head(20).to_html(index=False) + """
            </div>
            """
        
        # Add WHO by grading and regime for key subtypes tables
        if (self.tables_dir / 'who_by_grading_key_subtypes.csv').exists():
            who_grading_df = pd.read_csv(self.tables_dir / 'who_by_grading_key_subtypes.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Key WHO Diagnoses by Biopsy Grading (Overall)</h2>
                <p>Subtypes: liposarcoma, fibroblastic_tumor, undifferentiated_sarcoma, leiomyosarcoma</p>
                """ + who_grading_df.to_html() + """
            </div>
            """
        
        if (self.tables_dir / 'who_by_regime_key_subtypes.csv').exists():
            who_regime_df = pd.read_csv(self.tables_dir / 'who_by_regime_key_subtypes.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Key WHO Diagnoses by Treatment Regime (Overall)</h2>
                """ + who_regime_df.to_html() + """
            </div>
            """
        
        # Add individual WHO subtype grading by regime tables
        key_subtypes = ['liposarcoma', 'fibroblastic_tumor', 'undifferentiated_sarcoma', 'leiomyosarcoma']
        for subtype in key_subtypes:
            subtype_file = self.tables_dir / f'who_{subtype}_grading_by_regime.csv'
            if subtype_file.exists():
                subtype_df = pd.read_csv(subtype_file, index_col=0)
                html_content += f"""
                <div class="summary-box">
                    <h2>{subtype.replace('_', ' ').title()}: Grading by Treatment Regime</h2>
                    """ + subtype_df.to_html() + """
                </div>
                """
        
        # Add metastasis at diagnosis by regime timing table
        if (self.tables_dir / 'metastasis_at_diagnosis_by_regime_timing.csv').exists():
            met_regime_df = pd.read_csv(self.tables_dir / 'metastasis_at_diagnosis_by_regime_timing.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Metastasis Present at Diagnosis by Treatment Regime</h2>
                """ + met_regime_df.to_html() + """
            </div>
            """
        
        # Add WHOOPS by regime timing table
        if (self.tables_dir / 'whoops_by_regime_timing.csv').exists():
            whoops_regime_df = pd.read_csv(self.tables_dir / 'whoops_by_regime_timing.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>WHOOPS (Unplanned Excision) by Treatment Regime</h2>
                """ + whoops_regime_df.to_html() + """
            </div>
            """
        
        # Add WHO top 15 by regime timing table
        if (self.tables_dir / 'who_top15_by_regime_timing.csv').exists():
            who_top15_df = pd.read_csv(self.tables_dir / 'who_top15_by_regime_timing.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Top 15 WHO Diagnoses by Treatment Regime</h2>
                """ + who_top15_df.to_html() + """
            </div>
            """
        
        # Add margin status distribution table
        if (self.tables_dir / 'margin_status_distribution.csv').exists():
            margin_df = pd.read_csv(self.tables_dir / 'margin_status_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>Surgical Margin Status Distribution (Overall)</h2>
                """ + margin_df.to_html(index=False) + """
            </div>
            """
        
        # Add margin status by regime timing table
        if (self.tables_dir / 'margin_status_by_regime_timing.csv').exists():
            margin_regime_df = pd.read_csv(self.tables_dir / 'margin_status_by_regime_timing.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Surgical Margin Status by Treatment Regime Timing</h2>
                """ + margin_regime_df.to_html() + """
            </div>
            """
        
        # Add extremity tumor distribution tables
        if (self.tables_dir / 'extremity_tumor_distribution.csv').exists():
            extremity_dist_df = pd.read_csv(self.tables_dir / 'extremity_tumor_distribution.csv')
            html_content += """
            <div class="summary-box">
                <h2>Extremity Tumor Distribution</h2>
                """ + extremity_dist_df.to_html(index=False) + """
            </div>
            """
        
        if (self.tables_dir / 'extremity_tumor_by_regime_timing.csv').exists():
            extremity_regime_df = pd.read_csv(self.tables_dir / 'extremity_tumor_by_regime_timing.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Extremity Tumor by Treatment Regime Timing</h2>
                """ + extremity_regime_df.to_html() + """
            </div>
            """
        
        if (self.tables_dir / 'extremity_tumor_by_grading.csv').exists():
            extremity_grading_df = pd.read_csv(self.tables_dir / 'extremity_tumor_by_grading.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Extremity Tumor by Biopsy Grading</h2>
                """ + extremity_grading_df.to_html() + """
            </div>
            """
        
        # Add treatment regime by outcomes crosstabs
        outcome_tables = [
            ('had_metastasis_by_treatment_regime.csv', 'Metastasis Rate by Treatment Regime'),
            ('had_local_recurrence_by_treatment_regime.csv', 'Local Recurrence Rate by Treatment Regime'),
            ('died_of_disease_by_treatment_regime.csv', 'Mortality (DOD) by Treatment Regime')
        ]
        
        for table_file, table_title in outcome_tables:
            if (self.tables_dir / table_file).exists():
                outcome_regime_df = pd.read_csv(self.tables_dir / table_file, index_col=0)
                html_content += f"""
                <div class="summary-box">
                    <h2>{table_title}</h2>
                    """ + outcome_regime_df.to_html() + """
                </div>
                """
        
        # Add regime TIMING by outcomes crosstabs
        timing_outcome_tables = [
            ('had_metastasis_by_regime_timing.csv', 'Metastasis Rate by Regime Timing'),
            ('had_local_recurrence_by_regime_timing.csv', 'Local Recurrence Rate by Regime Timing'),
            ('died_of_disease_by_regime_timing.csv', 'Mortality (DOD) by Regime Timing')
        ]
        
        for table_file, table_title in timing_outcome_tables:
            if (self.tables_dir / table_file).exists():
                timing_outcome_df = pd.read_csv(self.tables_dir / table_file, index_col=0)
                html_content += f"""
                <div class="summary-box">
                    <h2>{table_title}</h2>
                    """ + timing_outcome_df.to_html() + """
                </div>
                """
        
        # Add combined regime timing outcomes table
        if (self.tables_dir / 'regime_timing_outcomes_combined.csv').exists():
            combined_timing_df = pd.read_csv(self.tables_dir / 'regime_timing_outcomes_combined.csv', index_col=0)
            html_content += """
            <div class="summary-box">
                <h2>Combined Regime Timing: Patient Volume and Outcomes</h2>
                """ + combined_timing_df.to_html() + """
            </div>
            """
        
        # Add combined regime AND moderator crosstabs
        combined_tables = [
            ('had_metastasis_by_regime_and_extremity_location.csv', 'Metastasis by Treatment Regime and Extremity Location'),
            ('had_metastasis_by_regime_and_biopsy_grading.csv', 'Metastasis by Treatment Regime and Tumor Grade'),
            ('had_local_recurrence_by_regime_and_extremity_location.csv', 'Local Recurrence by Treatment Regime and Extremity Location'),
            ('had_local_recurrence_by_regime_and_biopsy_grading.csv', 'Local Recurrence by Treatment Regime and Tumor Grade')
        ]
        
        for table_file, table_title in combined_tables:
            if (self.tables_dir / table_file).exists():
                combined_df = pd.read_csv(self.tables_dir / table_file, index_col=[0, 1])
                html_content += f"""
                <div class="summary-box">
                    <h2>{table_title}</h2>
                    """ + combined_df.to_html() + """
                </div>
                """
        
        # Add age statistics by regime and grading table
        if (self.tables_dir / 'age_by_regime_and_grading.csv').exists():
            age_stats_df = pd.read_csv(self.tables_dir / 'age_by_regime_and_grading.csv', index_col=[0, 1])
            html_content += """
            <div class="summary-box">
                <h2>Age Distribution by Treatment Regime and Tumor Grading</h2>
                <p>Distribution of patients across age categories (≤40, 41-60, 61-80, >80) by treatment regime and tumor grade (G1, G2, G3):</p>
                """ + age_stats_df.to_html() + """
            </div>
            """
        
        # Add tumor size statistics by regime and grading table
        if (self.tables_dir / 'size_by_regime_and_grading.csv').exists():
            size_stats_df = pd.read_csv(self.tables_dir / 'size_by_regime_and_grading.csv', index_col=[0, 1])
            html_content += """
            <div class="summary-box">
                <h2>Tumor Size Distribution by Treatment Regime and Tumor Grading</h2>
                <p>Distribution of patients across tumor size categories (≤50mm, 51-100mm, 101-150mm, >150mm) by treatment regime and tumor grade (G1, G2, G3):</p>
                """ + size_stats_df.to_html() + """
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save report
        report_path = self.reports_dir / self.config['output']['summary_report']
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML report saved to: {report_path}")
    
    def run_analysis(self) -> None:
        """
        Run the complete exploratory analysis pipeline.
        
        EXECUTION ORDER:
        1. Load data from MongoDB
        2. FILTER to remove non-malignant cases (benign, not_a_sarcoma, suspicious)
        3. Extract patient characteristics from filtered data
        4. Run all analyses with None/null values shown as 'Missing'
        """
        print("\n" + "=" * 80)
        print(" " * 20 + "EXPLORATORY DATA ANALYSIS PIPELINE")
        print("=" * 80)
        
        # Step 1: Load raw data from MongoDB
        print("\nStep 1: Loading data from MongoDB...")
        self.load_data_from_mongodb()
        
        # Step 2: Extract patient characteristics (filtering already applied in load step)
        print("\nStep 2: Extracting patient characteristics...")
        self.extract_patient_characteristics()
        
        print("\n" + "-" * 80)
        print("DATA PREPARATION COMPLETE - Starting analyses...")
        print("-" * 80 + "\n")
        
        # Generate analyses based on config
        if self.config['analysis']['generate_plots']:
            self.generate_distribution_plots()
        
        if self.config['analysis']['generate_crosstabs']:
            self.generate_treatment_by_characteristics()
        
        if self.config['analysis']['generate_statistics']:
            self.generate_summary_statistics()
        
        if self.config['analysis']['analyze_sequences']:
            self.analyze_treatment_sequences()
        
        if self.config['analysis'].get('analyze_indications', False):
            self.analyze_treatment_indications()
        
        if self.config['analysis'].get('analyze_regime_timing', False):
            self.analyze_treatment_regimes_with_timing()
        
        if self.config['analysis'].get('analyze_clinical_outcomes', False):
            self.analyze_clinical_outcomes()
        
        # Generate HTML report
        self.generate_html_report()
        
        print("\n" + "=" * 80)
        print(" " * 30 + "ANALYSIS COMPLETE!")
        print(f" Results saved to: {self.output_dir}")
        print("=" * 80 + "\n")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Perform exploratory data analysis on sarcoma patient treatment data"
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to YAML configuration file'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = ExploratoryAnalyzer(args.config)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()
