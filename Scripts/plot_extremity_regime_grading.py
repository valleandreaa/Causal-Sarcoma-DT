#!/usr/bin/env python3
"""
Plot Extremity Tumor Distribution by Treatment Regime and Grading

This script creates visualizations showing the relationship between:
- Extremity tumor location (yes/no)
- Treatment regimes (Surgery, Chemotherapy, Radiotherapy combinations)
- Tumor grading (G1, G2, G3)
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def load_data_from_mongodb():
    """Load patient data from MongoDB."""
    load_dotenv()
    
    # Get MongoDB connection details
    mongo_uri = os.getenv('MONGO_URI')
    db_name = os.getenv('MONGO_DB')
    collection_name = os.getenv('MONGO_COLLECTION')
    
    if not all([mongo_uri, db_name, collection_name]):
        raise ValueError("MongoDB connection details missing from .env file")
    
    print(f"Connecting to MongoDB: {db_name}.{collection_name}")
    
    # Connect to MongoDB
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Fetch all patient records
    data = list(collection.find({}))
    print(f"Loaded {len(data)} patient records")
    
    return data


def extract_features(data):
    """Extract relevant features from patient records."""
    records = []
    
    for patient in data:
        # Get patient ID
        patient_id = patient.get('_id')
        
        # Get static characteristics
        general = patient.get('general', {})
        tumor_chars = patient.get('tumor_characteristics', {})
        
        age = general.get('age')
        gender = general.get('gender')
        
        # Extremity tumor (0 or 1)
        extremity_tumor = tumor_chars.get('extremity_tumor')
        
        # Biopsy grading
        biopsy_grading = tumor_chars.get('biopsy_grading')
        
        # Count treatments across all episodes
        episodes = patient.get('episodes', [])
        total_surgeries = 0
        total_chemotherapy = 0
        total_radiotherapy = 0
        
        for episode in episodes:
            # Count surgery
            if episode.get('surgery', {}).get('surgery_flag') == 1:
                total_surgeries += 1
            
            # Count chemotherapy
            if episode.get('systemic_therapy', {}).get('chemotherapy_flag') == 1:
                total_chemotherapy += 1
            
            # Count radiotherapy
            if episode.get('radiotherapy', {}).get('radiation_oncology_flag') == 1:
                total_radiotherapy += 1
        
        # Create treatment regime string
        treatments = []
        if total_surgeries > 0:
            treatments.append('S')
        if total_chemotherapy > 0:
            treatments.append('C')
        if total_radiotherapy > 0:
            treatments.append('R')
        treatment_regime = '+'.join(treatments) if treatments else 'None'
        
        record = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'extremity_tumor': extremity_tumor,
            'biopsy_grading': biopsy_grading,
            'treatment_regime': treatment_regime,
            'total_surgeries': total_surgeries,
            'total_chemotherapy': total_chemotherapy,
            'total_radiotherapy': total_radiotherapy
        }
        
        records.append(record)
    
    df = pd.DataFrame(records)
    print(f"\nExtracted {len(df)} patient records")
    return df


def create_plots(df, output_dir='output/extremity_analysis'):
    """Create visualization plots."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map extremity_tumor to labels
    df['extremity_label'] = df['extremity_tumor'].map({0: 'Non-Extremity', 1: 'Extremity'})
    
    # Filter out None/NaN values for analysis
    df_clean = df.dropna(subset=['extremity_tumor', 'treatment_regime', 'biopsy_grading'])
    
    print(f"\nData Summary:")
    print(f"Total patients: {len(df)}")
    print(f"Patients with complete data: {len(df_clean)}")
    print(f"\nExtremity tumor distribution:")
    print(df['extremity_label'].value_counts())
    print(f"\nTreatment regime distribution:")
    print(df['treatment_regime'].value_counts())
    print(f"\nBiopsy grading distribution:")
    print(df['biopsy_grading'].value_counts())
    
    # --- Plot 1: Extremity by Treatment Regime ---
    fig, ax = plt.subplots(figsize=(12, 6))
    crosstab1 = pd.crosstab(df_clean['treatment_regime'], df_clean['extremity_label'])
    crosstab1.plot(kind='bar', ax=ax, color=['#E74C3C', '#3498DB'])
    ax.set_title('Extremity Tumor Distribution by Treatment Regime', fontsize=14, fontweight='bold')
    ax.set_xlabel('Treatment Regime', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.legend(title='Tumor Location', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path / 'extremity_by_regime.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved: {output_path / 'extremity_by_regime.png'}")
    plt.close()
    
    # Save crosstab
    crosstab1.to_csv(output_path / 'extremity_by_regime_crosstab.csv')
    
    # --- Plot 2: Extremity by Grading ---
    fig, ax = plt.subplots(figsize=(10, 6))
    crosstab2 = pd.crosstab(df_clean['biopsy_grading'], df_clean['extremity_label'])
    crosstab2.plot(kind='bar', ax=ax, color=['#E74C3C', '#3498DB'])
    ax.set_title('Extremity Tumor Distribution by Grading', fontsize=14, fontweight='bold')
    ax.set_xlabel('Biopsy Grading', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.legend(title='Tumor Location', fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path / 'extremity_by_grading.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'extremity_by_grading.png'}")
    plt.close()
    
    # Save crosstab
    crosstab2.to_csv(output_path / 'extremity_by_grading_crosstab.csv')
    
    # --- Plot 3: Treatment Regime by Grading (Heatmap) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    crosstab3 = pd.crosstab(df_clean['treatment_regime'], df_clean['biopsy_grading'])
    sns.heatmap(crosstab3, annot=True, fmt='d', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Treatment Regime by Grading (All Patients)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Biopsy Grading', fontsize=12)
    ax.set_ylabel('Treatment Regime', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path / 'regime_by_grading_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'regime_by_grading_heatmap.png'}")
    plt.close()
    
    # Save crosstab
    crosstab3.to_csv(output_path / 'regime_by_grading_crosstab.csv')
    
    # --- Plot 4: Combined 3-way plot (Extremity + Regime + Grading) ---
    # Separate data by extremity
    df_extremity = df_clean[df_clean['extremity_tumor'] == 1]
    df_non_extremity = df_clean[df_clean['extremity_tumor'] == 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Extremity tumors
    if len(df_extremity) > 0:
        crosstab_ext = pd.crosstab(df_extremity['treatment_regime'], df_extremity['biopsy_grading'])
        sns.heatmap(crosstab_ext, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
                   cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Extremity Tumors (n={len(df_extremity)})', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Biopsy Grading', fontsize=11)
        axes[0].set_ylabel('Treatment Regime', fontsize=11)
    
    # Non-extremity tumors
    if len(df_non_extremity) > 0:
        crosstab_nonext = pd.crosstab(df_non_extremity['treatment_regime'], df_non_extremity['biopsy_grading'])
        sns.heatmap(crosstab_nonext, annot=True, fmt='d', cmap='Reds', ax=axes[1], 
                   cbar_kws={'label': 'Count'})
        axes[1].set_title(f'Non-Extremity Tumors (n={len(df_non_extremity)})', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Biopsy Grading', fontsize=11)
        axes[1].set_ylabel('Treatment Regime', fontsize=11)
    
    plt.suptitle('Treatment Regime by Grading: Extremity vs Non-Extremity', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path / 'regime_by_grading_split_by_extremity.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'regime_by_grading_split_by_extremity.png'}")
    plt.close()
    
    # Save crosstabs
    if len(df_extremity) > 0:
        crosstab_ext.to_csv(output_path / 'extremity_regime_by_grading_crosstab.csv')
    if len(df_non_extremity) > 0:
        crosstab_nonext.to_csv(output_path / 'non_extremity_regime_by_grading_crosstab.csv')
    
    # --- Plot 5: Grouped bar chart by grading ---
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Prepare data for grouped bar chart
    grading_order = ['G1', 'G2', 'G3']
    regime_labels = sorted(df_clean['treatment_regime'].unique())
    
    x = np.arange(len(grading_order))
    width = 0.15
    
    for i, regime in enumerate(regime_labels):
        regime_data = df_clean[df_clean['treatment_regime'] == regime]
        extremity_counts = []
        non_extremity_counts = []
        
        for grade in grading_order:
            grade_data = regime_data[regime_data['biopsy_grading'] == grade]
            extremity_counts.append(len(grade_data[grade_data['extremity_tumor'] == 1]))
            non_extremity_counts.append(len(grade_data[grade_data['extremity_tumor'] == 0]))
        
        # Plot extremity
        ax.bar(x + i * width * 2, extremity_counts, width, 
               label=f'{regime} (Extremity)', alpha=0.8)
        # Plot non-extremity
        ax.bar(x + i * width * 2 + width, non_extremity_counts, width, 
               label=f'{regime} (Non-Extremity)', alpha=0.5)
    
    ax.set_xlabel('Biopsy Grading', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Extremity Distribution by Treatment Regime and Grading', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(regime_labels) - 1))
    ax.set_xticklabels(grading_order)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path / 'grouped_extremity_regime_grading.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path / 'grouped_extremity_regime_grading.png'}")
    plt.close()
    
    print(f"\nAll plots saved to: {output_path.absolute()}")


def main():
    """Main function."""
    print("=" * 60)
    print("Extremity Tumor Analysis by Regime and Grading")
    print("=" * 60)
    
    # Load data
    data = load_data_from_mongodb()
    
    # Extract features
    df = extract_features(data)
    
    # Create plots
    create_plots(df)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
