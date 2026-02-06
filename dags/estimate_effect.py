"""CLI: Estimate treatment effects using do-intervention on a BN."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from dotenv import load_dotenv
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from dtcygan.bn import load_bayesian_network
from dtcygan.causal import average_treatment_effect, do_intervene
from pgmpy.inference import VariableElimination  # type: ignore
# Reuse data cleaning from fit_bn to ensure consistent encoding
from fit_bn import _apply_feature_selection, _validate_and_clean_data, _load_config

# When `last_status` is exploded into binaries, these are the nodes produced
STATUS_NODES = ["NED", "AWD", "DOD"]

# AJCC 8th edition staging survival probabilities (5-year OS for trunk/extremity STS)
AJCC_SURVIVAL_PROBS = {
    "IA": 0.90,    # Stage I overall ~90%
    "IB": 0.90,
    "II": 0.70,    # Stage II ~70%
    "IIIA": 0.62,  # Stage IIIA ~62%
    "IIIB": 0.50,  # Stage IIIB ~50%
    "IV": 0.15     # Stage IV ~10-20%
}

def calculate_ajcc_stage(row: pd.Series) -> str:
    """Calculate AJCC 8th edition stage for trunk/extremity soft tissue sarcoma.
    
    Based on TNM+Grade:
    - T (size): T1≤5cm, T2>5-10cm, T3>10-15cm, T4>15cm
    - N (nodes): Assumed N0 (not available in dataset)
    - M (metastasis): From metastasis_present_at_diagnosis
    - G (grade): From biopsy_grading (G1/G2/G3)
    
    Returns AJCC stage (IA, IB, II, IIIA, IIIB, IV) or 'Unknown'
    """
    # Get tumor size (continuous in cm)
    size = row.get('tumor_characteristics.initial_size', None)
    
    # Get grade (categorical: 1=G1, 2=G2, 3=G3)
    grade = row.get('tumor_characteristics.biopsy_grading', None)
    
    # Get metastasis (binary: 0=no, 1=yes)
    metastasis = row.get('tumor_characteristics.metastasis_present_at_diagnosis', 0)
    
    # Check if we have required data
    if pd.isna(size) or pd.isna(grade):
        return 'Unknown'
    
    # Map size to T category (AJCC 8th edition for trunk/extremity)
    if size <= 5:
        t_cat = 1
    elif size <= 10:
        t_cat = 2
    elif size <= 15:
        t_cat = 3
    else:
        t_cat = 4
    
    # Stage IV if metastasis present (M1)
    if metastasis == 1:
        return 'IV'
    
    # Assuming N0 (no nodal involvement) since not in dataset
    # Stage grouping for trunk/extremity (M0, N0):
    
    # Grade 1 (low grade)
    if grade == 1:
        if t_cat == 1:
            return 'IA'
        else:  # T2-T4
            return 'IB'
    
    # Grade 2-3 (high grade)
    elif grade in [2, 3]:
        if t_cat == 1:
            return 'II'
        elif t_cat == 2:
            return 'IIIA'
        else:  # T3-T4
            return 'IIIB'
    
    return 'Unknown'


def get_ajcc_mortality_risk(stage: str) -> float:
    """Get 5-year mortality risk (1 - survival) from AJCC stage."""
    if stage == 'Unknown':
        return 0.5  # Default 50% if unknown
    survival = AJCC_SURVIVAL_PROBS.get(stage, 0.5)
    return 1.0 - survival  # Convert survival to mortality (DOD risk)


def _parse_value(raw: str) -> str | int:
    if raw.isdigit():
        return int(raw)
    return raw


def _preprocess_evidence(evidence: dict, model, config: dict) -> dict:
    """Preprocess evidence values to match the discretization used during BN fitting.
    
    This ensures that continuous values like age are discretized into the same bins
    that were used when fitting the Bayesian network.
    """
    if not evidence:
        return evidence
    
    # Create a dummy dataframe with all model nodes set to 0, then overlay evidence
    dummy_data = pd.DataFrame([{node: 0 for node in model.nodes()}])
    
    # Overlay the evidence values
    for key, value in evidence.items():
        if key in dummy_data.columns:
            dummy_data[key] = value
    
    # Apply the same cleaning pipeline used in fit_bn
    cleaned = _validate_and_clean_data(dummy_data, list(model.nodes()))
    
    # Extract only the evidence variables that were provided
    processed_evidence = {key: int(cleaned[key].iloc[0]) for key in evidence.keys() if key in cleaned.columns}
    
    return processed_evidence


def _parse_evidence(pairs: list[str]) -> dict[str, str | int]:
    evidence = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Evidence must be in key=value form. Got: {pair}")
        key, value = pair.split("=", 1)
        evidence[key.strip()] = _parse_value(value.strip())
    return evidence


def run_multiple_treatment_effects(model, outcomes: list[str], treatments: list[str], evidence: dict = None) -> list[dict]:
    """Run treatment effect calculations for multiple treatment-outcome pairs."""
    results = []
    
    if evidence:
        print(f"Conditioning on: {evidence}")
    
    print(f"\n{'='*80}")
    print(f"TREATMENT EFFECT ANALYSIS")
    print(f"{'='*80}\n")
    
    for outcome in outcomes:
        print(f"\n{'-'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'-'*80}\n")
        
        for treatment in treatments:
            # Skip if treatment is the same as outcome
            if treatment == outcome:
                continue
            
            # Determine treatment values (0 = no treatment, 1 = treatment)
            control_value = 0
            treated_value = 1
            
            # For outcomes, typically we want to measure P(outcome=1)
            outcome_state = 1
            
            try:
                ate = average_treatment_effect(
                    model,
                    treatment,
                    outcome,
                    (control_value, treated_value),
                    outcome_state,
                    evidence=evidence,
                )
                
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "control_value": control_value,
                    "treated_value": treated_value,
                    "outcome_state": outcome_state,
                    "ate": ate,
                    "status": "success"
                }
                
                print(f"  {treatment} (0→1) → {outcome}(={outcome_state})")
                print(f"    ATE: {ate:+.6f}")
                
                if ate > 0:
                    print(f"    ↑ Treatment INCREASES outcome by {abs(ate):.2%}")
                elif ate < 0:
                    print(f"    ↓ Treatment DECREASES outcome by {abs(ate):.2%}")
                else:
                    print(f"    → No effect")
                print()
                
            except Exception as e:
                result = {
                    "treatment": treatment,
                    "outcome": outcome,
                    "control_value": control_value,
                    "treated_value": treated_value,
                    "outcome_state": outcome_state,
                    "ate": None,
                    "status": "error",
                    "error": str(e)
                }
                print(f"  {treatment} → {outcome}: ERROR - {e}\n")
            
            results.append(result)
    
    return results


def validate_predictions(model, config_path: str, sample_size: int = 20, use_full_dataset: bool = False, plot_dir: str = None, test_data_path: str = None) -> None:
    """Load real patient data and compare actual vs predicted outcomes.
    
    Args:
        model: Fitted Bayesian Network
        config_path: Path to config YAML
        sample_size: Number of patients to display (default 20)
        use_full_dataset: If True, calculate metrics on entire dataset (default False)
        plot_dir: Directory to save distribution plots (optional)
        test_data_path: Path to test set pickle file (if None, loads from MongoDB)
    """
    
    print(f"\n{'='*80}")
    print(f"MODEL VALIDATION - ACTUAL vs PREDICTED")
    print(f"{'='*80}\n")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data from test set pickle or MongoDB
    if test_data_path:
        print(f"Loading test set from: {test_data_path}")
        data = pd.read_pickle(test_data_path)
        print(f"Loaded {len(data)} test samples")
        # Keep a copy for raw display (already cleaned)
        raw_data = data.copy()
    else:
        print("Warning: No test data path provided, loading from MongoDB")
        print("This may contaminate validation with training data!")
        
        # Load environment variables
        load_dotenv()
        
        # Get MongoDB connection details from config or environment
        mongo_uri = config.get("mongodb", {}).get("uri") or os.getenv("MONGO_URI")
        db_name = config.get("mongodb", {}).get("database") or os.getenv("MONGO_DB")
        collection_name = config.get("mongodb", {}).get("collection") or os.getenv("MONGO_COLLECTION")
        
        if not all([mongo_uri, db_name, collection_name]):
            raise ValueError("MongoDB connection details missing in config or environment variables")
        
        # Connect to MongoDB
        print(f"Connecting to MongoDB: {db_name}.{collection_name}")
        client = MongoClient(mongo_uri)
        db = client[db_name]
        collection = db[collection_name]
        
        # Execute aggregation pipeline and keep raw copy for readable history
        pipeline = config['mongodb']['pipeline']
        cursor = collection.aggregate(pipeline)
        raw_data = pd.json_normalize(cursor)
        data = raw_data.copy()
        
        # Apply feature selection then reuse fit_bn's cleaning to ensure
        # the same categorical encodings used when fitting the BN.
        if config.get('features'):
            # Apply same selection to raw and working copy
            raw_data = _apply_feature_selection(raw_data, config)
            data = _apply_feature_selection(data, config)

        # Validate and clean using the same routine as fit_bn (this will
        # restrict to DAG nodes and convert types/codes consistently).
        data = _validate_and_clean_data(data, list(model.nodes()))

        # Align raw_data to the cleaned rows (cleaning may have dropped rows)
        raw_data = raw_data.loc[data.index]

    # Sample patients
    sample_data = data.sample(n=min(sample_size, len(data)), random_state=42)
    sample_raw = raw_data.loc[sample_data.index]
    
    print(f"Loaded {len(data)} patients, showing {len(sample_data)} samples\n")
    
    # Get inference engine
    inference = VariableElimination(model)
    
    # Outcomes to validate: adapt to new encoding where 'last_status'
    # may have been converted into three binary indicators: 'NED','AWD','DOD'.
    outcomes = ['metastasis', 'local_recurrence']
    # Prefer the single 'last_status' node if present in the model/data,
    # otherwise include binary indicators if available.
    model_nodes = list(model.nodes())
    if 'last_status' in model_nodes or 'last_status' in sample_data.columns:
        outcomes.append('last_status')
    else:
        for status_node in ['NED', 'AWD', 'DOD']:
            if status_node in model_nodes or status_node in sample_data.columns:
                outcomes.append(status_node)
    
    # Display header (include patient history and AJCC)
    print(f"{'ID':<5} {'History':<40} {'Surgery':>7} {'RT':>7} {'Chemo':>7} {'AJCC':>7} ", end="")
    for outcome in outcomes:
        if outcome in sample_data.columns:
            if outcome in ('metastasis', 'local_recurrence') or outcome in STATUS_NODES:
                print(f"{'Actual':>7} {'PredProb':>7} ", end="")
            else:
                print(f"{'Actual':>7} {'Pred':>7} ", end="")
    print("\n" + "-" * 80)
    
    # Process each patient
    for idx, ix in enumerate(sample_data.index, 1):
        row = sample_data.loc[ix]
        raw_row = sample_raw.loc[ix]
        print(f"{idx:<5} ", end="")

        # Patient history: prefer explicit history column from raw data, else build from raw tumor fields
        history_val = ""
        for hc in ['history', 'patient_history', 'clinical_history', 'notes', 'last_episode_summary']:
            if hc in raw_row.index and pd.notna(raw_row.get(hc)):
                history_val = str(raw_row.get(hc))
                break
        if not history_val:
            parts = []
            for fld in ['tumor_characteristics.who_diagnosis_code', 'tumor_characteristics.anatomic_region_grouping', 'tumor_characteristics.biopsy_grading', 'tumor_characteristics.initial_size']:
                if fld in raw_row.index and pd.notna(raw_row.get(fld)):
                    parts.append(str(raw_row.get(fld)))
            history_val = ' | '.join(parts)

        # Print full history field (no truncation)
        print(f"{history_val:<40} ", end="")

        # Print treatment values (from cleaned row)
        print(f"{int(row.get('surgery', 0)):>7} ", end="")
        print(f"{int(row.get('radiotherapy', 0)):>7} ", end="")
        print(f"{int(row.get('chemotherapy', 0)):>7} ", end="")
        
        # Calculate and print AJCC stage
        ajcc_stage = calculate_ajcc_stage(row)
        print(f"{ajcc_stage:>7} ", end="")
        
        # Build evidence from patient data (exclude outcomes)
        evidence = {}
        for col in row.index:
            if col not in outcomes and col in model.nodes():
                val = row[col]
                # Try to convert to int, if it fails, use the value as-is
                try:
                    evidence[col] = int(val)
                except (ValueError, TypeError):
                    # Keep as string or original type
                    evidence[col] = val
        
        # Predict each outcome
        for outcome in outcomes:
            if outcome not in sample_data.columns:
                continue

            actual_val = row[outcome]
            # Handle both numeric and string actual values
            if outcome in STATUS_NODES:
                # Status nodes are probabilities (0-1), keep as float
                try:
                    actual = float(actual_val)
                except (ValueError, TypeError):
                    actual = 0.0
            else:
                try:
                    actual = int(actual_val)
                except (ValueError, TypeError):
                    # If we have a single multi-class 'last_status', map known labels
                    if outcome == 'last_status':
                        status_map = {'AWD': 0, 'DOD': 1, 'NED': 2}
                        actual = status_map.get(actual_val, 0)
                    else:
                        # For binary indicators or other string-valued vars, try to map
                        try:
                            actual = int(pd.Categorical([actual_val]).codes[0])
                        except Exception:
                            actual = 0

            try:
                # Query marginal given evidence
                result = inference.query(variables=[outcome], evidence=evidence)
                state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))

                # If binary outcome, print probability of state=1
                if outcome in ('metastasis', 'local_recurrence') or outcome in STATUS_NODES or len(state_names) == 2:
                    # find index for state '1'
                    try:
                        idx_one = state_names.index(1)
                    except Exception:
                        try:
                            idx_one = state_names.index('1')
                        except Exception:
                            idx_one = 1
                    prob_one = float(result.values[idx_one])
                    # Display actual and predicted probability
                    if outcome in STATUS_NODES:
                        # For status probabilities, show both as floats
                        print(f"{actual:>7.3f} {prob_one:7.3f}", end=" ")
                    else:
                        # For binary outcomes, show actual as int and predicted as probability
                        print(f"{int(actual):>7} {prob_one:7.3f}", end=" ")
                else:
                    # Multi-class: use argmax index as predicted class code
                    predicted_idx = int(result.values.argmax())
                    match_symbol = "✓" if predicted_idx == actual else "✗"
                    print(f"{actual:>7} {predicted_idx:>7}{match_symbol}", end=" ")

            except Exception as e:
                print(f"{actual:>7} {'ERR':>7} ", end="")
        
        print()
    
    # Calculate accuracy on sample
    print("\n" + "=" * 80)
    print("ACCURACY SUMMARY (Sample)")
    print("=" * 80)
    
    for outcome in outcomes:
        if outcome not in sample_data.columns:
            continue
        
        # Skip accuracy calculation for probabilistic status nodes
        if outcome in STATUS_NODES:
            continue
            
        correct = 0
        total = 0
        
        for _, row in sample_data.iterrows():
            evidence = {}
            for col in row.index:
                if col not in outcomes and col in model.nodes():
                    val = row[col]
                    # Try to convert to int, if it fails, use the value as-is
                    try:
                        evidence[col] = int(val)
                    except (ValueError, TypeError):
                        evidence[col] = val
            
            try:
                result = inference.query(variables=[outcome], evidence=evidence)
                state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                predicted_idx = int(result.values.argmax())
                
                actual_val = row[outcome]
                # Handle both numeric and string values
                try:
                    actual = int(actual_val)
                except (ValueError, TypeError):
                    if outcome == 'last_status':
                        status_map = {'AWD': 0, 'DOD': 1, 'NED': 2}
                        actual = status_map.get(actual_val, 0)
                    elif outcome in STATUS_NODES:
                        # Binary status indicators
                        try:
                            actual = int(actual_val)
                        except (ValueError, TypeError):
                            actual = 1 if str(actual_val).strip() in ("1", "True", "true", "Y", "y") else 0
                    else:
                        actual = 0
                
                if predicted_idx == actual:
                    correct += 1
                total += 1
            except:
                pass
        
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"{outcome}: {correct}/{total} correct ({accuracy:.1f}%)")
    
    print()
    
    # Evaluate on full dataset if requested
    if use_full_dataset:
        print("\n" + "=" * 80)
        print("FULL DATASET EVALUATION")
        print("=" * 80)
        print(f"Evaluating on {len(data)} patients...\n")
        
        # Get inference engine
        inference = VariableElimination(model)
        
        # Store predictions and actuals for each outcome
        predictions = {outcome: [] for outcome in outcomes if outcome not in STATUS_NODES}
        actuals = {outcome: [] for outcome in outcomes if outcome not in STATUS_NODES}
        probabilities = {outcome: [] for outcome in outcomes}  # For all outcomes including status nodes
        ajcc_dod_predictions = []  # AJCC-based DOD risk predictions
        
        for idx, row in data.iterrows():
            # Calculate AJCC stage and predicted DOD risk
            ajcc_stage = calculate_ajcc_stage(row)
            ajcc_dod_risk = get_ajcc_mortality_risk(ajcc_stage)
            ajcc_dod_predictions.append(ajcc_dod_risk)
            
            evidence = {}
            for col in row.index:
                if col not in outcomes and col in model.nodes():
                    try:
                        evidence[col] = int(row[col])
                    except (ValueError, TypeError):
                        evidence[col] = row[col]
            
            for outcome in outcomes:
                if outcome not in data.columns:
                    continue
                
                try:
                    result = inference.query(variables=[outcome], evidence=evidence)
                    
                    if outcome in STATUS_NODES:
                        # For status probabilities, just store predicted probability
                        state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                        try:
                            idx_one = state_names.index(1)
                        except:
                            idx_one = 1
                        prob = float(result.values[idx_one])
                        probabilities[outcome].append((float(row[outcome]), prob))
                    else:
                        # For binary outcomes, store predictions
                        predicted_idx = int(result.values.argmax())
                        actual_val = int(row[outcome])
                        predictions[outcome].append(predicted_idx)
                        actuals[outcome].append(actual_val)
                        
                        # Also store probabilities
                        state_names = result.state_names.get(outcome, list(range(result.cardinality[0])))
                        try:
                            idx_one = state_names.index(1)
                        except:
                            idx_one = 1
                        prob = float(result.values[idx_one])
                        probabilities[outcome].append((actual_val, prob))
                except Exception as e:
                    pass
        
        # Calculate and display metrics for binary outcomes
        for outcome in outcomes:
            if outcome in STATUS_NODES:
                continue
            
            if outcome not in predictions or len(predictions[outcome]) == 0:
                continue
            
            y_true = actuals[outcome]
            y_pred = predictions[outcome]
            
            acc = accuracy_score(y_true, y_pred)
            
            # Handle cases where outcome class is not present
            try:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                cm = confusion_matrix(y_true, y_pred)
            except:
                prec = rec = f1 = 0.0
                cm = None
            
            print(f"\n{outcome.upper()}:")
            print(f"  Accuracy:  {acc:.3f}")
            print(f"  Precision: {prec:.3f}")
            print(f"  Recall:    {rec:.3f}")
            print(f"  F1 Score:  {f1:.3f}")
            
            if cm is not None:
                print(f"  Confusion Matrix:")
                print(f"    TN={cm[0,0]}, FP={cm[0,1]}")
                print(f"    FN={cm[1,0]}, TP={cm[1,1]}")
        
        # AJCC vs BN comparison for DOD
        if 'DOD' in probabilities and len(ajcc_dod_predictions) == len(probabilities['DOD']):
            print(f"\n{'='*80}")
            print(f"AJCC STAGING vs BAYESIAN NETWORK COMPARISON (DOD Prediction)")
            print(f"{'='*80}")
            
            # Extract actual and predicted from tuples
            dod_actual = np.array([actual for actual, pred in probabilities['DOD']])
            bn_dod_probs = np.array([pred for actual, pred in probabilities['DOD']])
            ajcc_probs = np.array(ajcc_dod_predictions[:len(probabilities['DOD'])])
            
            # Diagnostic: Check actual value distribution
            print(f"\nDiagnostic - DOD actual values:")
            print(f"  Unique values: {np.unique(dod_actual)}")
            print(f"  Value counts: 0={np.sum(dod_actual==0)}, 1={np.sum(dod_actual==1)}")
            print(f"  Min={dod_actual.min()}, Max={dod_actual.max()}, Mean={dod_actual.mean():.3f}")
            if len(np.unique(dod_actual)) > 2:
                print(f"  WARNING: DOD has more than 2 unique values - should be binary (0 or 1)!")
                print(f"  All unique values: {sorted(np.unique(dod_actual))}")
            print()
            
            # Mean absolute error for both methods
            bn_mae = np.mean(np.abs(bn_dod_probs - dod_actual))
            ajcc_mae = np.mean(np.abs(ajcc_probs - dod_actual))
            
            # Mean squared error
            bn_mse = np.mean((bn_dod_probs - dod_actual) ** 2)
            ajcc_mse = np.mean((ajcc_probs - dod_actual) ** 2)
            
            # Correlation with actual outcomes
            bn_corr = np.corrcoef(bn_dod_probs, dod_actual)[0, 1]
            ajcc_corr = np.corrcoef(ajcc_probs, dod_actual)[0, 1]
            
            print(f"\nMean Absolute Error:")
            print(f"  Bayesian Network: {bn_mae:.4f}")
            print(f"  AJCC Staging:     {ajcc_mae:.4f}")
            print(f"  {'✓ BN wins' if bn_mae < ajcc_mae else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nMean Squared Error:")
            print(f"  Bayesian Network: {bn_mse:.4f}")
            print(f"  AJCC Staging:     {ajcc_mse:.4f}")
            print(f"  {'✓ BN wins' if bn_mse < ajcc_mse else '✓ AJCC wins'} (lower is better)")
            
            print(f"\nCorrelation with Actual DOD:")
            print(f"  Bayesian Network: {bn_corr:.4f}")
            print(f"  AJCC Staging:     {ajcc_corr:.4f}")
            print(f"  {'✓ BN wins' if bn_corr > ajcc_corr else '✓ AJCC wins'} (higher is better)")
            
            # Count stage distribution
            stage_counts = {}
            for idx, row in data.iterrows():
                stage = calculate_ajcc_stage(row)
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
            
            print(f"\nAJCC Stage Distribution in Test Set:")
            for stage in sorted(stage_counts.keys()):
                count = stage_counts[stage]
                pct = 100.0 * count / len(data)
                expected_mort = get_ajcc_mortality_risk(stage)
                print(f"  Stage {stage:8s}: {count:4d} patients ({pct:5.1f}%), Expected DOD risk: {expected_mort:.1%}")
            
            print(f"\n{'='*80}")
        
        # Plot distributions if directory provided
        if plot_dir:
            plot_path = Path(plot_dir)
            plot_path.mkdir(parents=True, exist_ok=True)
            
            # AJCC vs BN comparison plot for DOD
            if 'DOD' in probabilities and len(ajcc_dod_predictions) > 0:
                fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                
                # Extract actual and predicted from tuples
                dod_actual = np.array([actual for actual, pred in probabilities['DOD']])
                bn_dod_probs = np.array([pred for actual, pred in probabilities['DOD']])
                ajcc_probs = np.array(ajcc_dod_predictions[:len(probabilities['DOD'])])
                
                # Plot 1: BN - Distribution by actual class
                actual_0_bn = bn_dod_probs[dod_actual == 0]
                actual_1_bn = bn_dod_probs[dod_actual == 1]
                
                axes[0, 0].hist(actual_0_bn, bins=20, alpha=0.6, label=f'DOD=0 (n={len(actual_0_bn)})', color='blue')
                axes[0, 0].hist(actual_1_bn, bins=20, alpha=0.6, label=f'DOD=1 (n={len(actual_1_bn)})', color='red')
                axes[0, 0].set_xlabel('BN Predicted P(DOD=1)')
                axes[0, 0].set_ylabel('Count')
                axes[0, 0].set_title('Bayesian Network - Predicted Probabilities by Actual Class')
                axes[0, 0].legend()
                axes[0, 0].grid(alpha=0.3)
                
                # Plot 2: BN - Calibration curve
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_counts_bn = []
                bin_means_bn = []
                
                for i in range(len(bins)-1):
                    mask = (bn_dod_probs >= bins[i]) & (bn_dod_probs < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts_bn.append(mask.sum())
                        bin_means_bn.append(dod_actual[mask].mean())
                    else:
                        bin_counts_bn.append(0)
                        bin_means_bn.append(np.nan)
                
                axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
                valid_bn = ~np.isnan(bin_means_bn)
                axes[0, 1].scatter(bin_centers[valid_bn], np.array(bin_means_bn)[valid_bn], 
                                  s=[c*5 for c, v in zip(bin_counts_bn, valid_bn) if v], 
                                  alpha=0.6, color='blue', label='BN observed')
                axes[0, 1].plot(bin_centers[valid_bn], np.array(bin_means_bn)[valid_bn], 
                               'b-', alpha=0.5, linewidth=2)
                axes[0, 1].set_xlabel('BN Predicted P(DOD=1)')
                axes[0, 1].set_ylabel('Actual fraction of DOD=1')
                axes[0, 1].set_title(f'Bayesian Network Calibration\nMAE: {np.mean(np.abs(bn_dod_probs - dod_actual)):.4f}')
                axes[0, 1].legend()
                axes[0, 1].grid(alpha=0.3)
                axes[0, 1].set_xlim(0, 1)
                axes[0, 1].set_ylim(0, 1)
                
                # Plot 3: AJCC - Distribution by actual class
                actual_0_ajcc = ajcc_probs[dod_actual == 0]
                actual_1_ajcc = ajcc_probs[dod_actual == 1]
                
                axes[1, 0].hist(actual_0_ajcc, bins=20, alpha=0.6, label=f'DOD=0 (n={len(actual_0_ajcc)})', color='blue')
                axes[1, 0].hist(actual_1_ajcc, bins=20, alpha=0.6, label=f'DOD=1 (n={len(actual_1_ajcc)})', color='red')
                axes[1, 0].set_xlabel('AJCC Predicted P(DOD=1)')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('AJCC Staging - Predicted Probabilities by Actual Class')
                axes[1, 0].legend()
                axes[1, 0].grid(alpha=0.3)
                
                # Plot 4: AJCC - Calibration curve
                bin_counts_ajcc = []
                bin_means_ajcc = []
                
                for i in range(len(bins)-1):
                    mask = (ajcc_probs >= bins[i]) & (ajcc_probs < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts_ajcc.append(mask.sum())
                        bin_means_ajcc.append(dod_actual[mask].mean())
                    else:
                        bin_counts_ajcc.append(0)
                        bin_means_ajcc.append(np.nan)
                
                axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
                valid_ajcc = ~np.isnan(bin_means_ajcc)
                axes[1, 1].scatter(bin_centers[valid_ajcc], np.array(bin_means_ajcc)[valid_ajcc], 
                                  s=[c*5 for c, v in zip(bin_counts_ajcc, valid_ajcc) if v], 
                                  alpha=0.6, color='orange', label='AJCC observed')
                axes[1, 1].plot(bin_centers[valid_ajcc], np.array(bin_means_ajcc)[valid_ajcc], 
                               'orange', alpha=0.5, linewidth=2, linestyle='-')
                axes[1, 1].set_xlabel('AJCC Predicted P(DOD=1)')
                axes[1, 1].set_ylabel('Actual fraction of DOD=1')
                axes[1, 1].set_title(f'AJCC Staging Calibration\nMAE: {np.mean(np.abs(ajcc_probs - dod_actual)):.4f}')
                axes[1, 1].legend()
                axes[1, 1].grid(alpha=0.3)
                axes[1, 1].set_xlim(0, 1)
                axes[1, 1].set_ylim(0, 1)
                
                plt.tight_layout()
                comparison_plot_path = plot_path / 'ajcc_vs_bn_comparison.png'
                plt.savefig(comparison_plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"\nAJCC comparison plot saved to: {comparison_plot_path}")
            print(f"\nSaving distribution plots to: {plot_dir}")
            
            for outcome in outcomes:
                if outcome not in probabilities or len(probabilities[outcome]) == 0:
                    continue
                
                actual_vals, pred_probs = zip(*probabilities[outcome])
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.suptitle(f'{outcome} - Predicted vs Actual', fontsize=14, fontweight='bold')
                
                # Plot 1: Distribution of predicted probabilities by actual class
                actual_0 = [p for a, p in probabilities[outcome] if a == 0]
                actual_1 = [p for a, p in probabilities[outcome] if a == 1]
                
                axes[0].hist(actual_0, bins=20, alpha=0.6, label=f'{outcome}=0 (n={len(actual_0)})', color='blue')
                axes[0].hist(actual_1, bins=20, alpha=0.6, label=f'{outcome}=1 (n={len(actual_1)})', color='red')
                axes[0].set_xlabel('Predicted P(outcome=1)')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Predicted Probabilities by Actual Class')
                axes[0].legend()
                axes[0].grid(alpha=0.3)
                
                # Plot 2: Calibration curve (binned)
                bins = np.linspace(0, 1, 11)
                bin_centers = (bins[:-1] + bins[1:]) / 2
                bin_counts = []
                bin_means = []
                
                for i in range(len(bins)-1):
                    mask = (np.array(pred_probs) >= bins[i]) & (np.array(pred_probs) < bins[i+1])
                    if mask.sum() > 0:
                        bin_counts.append(mask.sum())
                        bin_means.append(np.array(actual_vals)[mask].mean())
                    else:
                        bin_counts.append(0)
                        bin_means.append(0)
                
                axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
                axes[1].scatter(bin_centers, bin_means, s=[c*5 for c in bin_counts], alpha=0.6, label='Observed')
                axes[1].set_xlabel('Predicted P(outcome=1)')
                axes[1].set_ylabel('Actual fraction of positive')
                axes[1].set_title('Calibration Curve')
                axes[1].legend()
                axes[1].grid(alpha=0.3)
                axes[1].set_xlim(0, 1)
                axes[1].set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(plot_path / f'{outcome}_distribution.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved: {outcome}_distribution.png")
    
    print()


def run_regime_comparisons(
    model, 
    outcomes: list[str], 
    treatments: list[str],
    comparisons: list[Tuple[Dict, Dict, str]],
    evidence: dict = None
) -> list[dict]:
    """Run treatment regime comparisons (e.g., surgery+RT vs surgery only)."""
    
    if evidence:
        print(f"\nConditioning all comparisons on: {evidence}\n")
    results = []
    
    print(f"\n{'='*80}")
    print(f"TREATMENT REGIME COMPARISON ANALYSIS")
    print(f"{'='*80}\n")
    
    # Print DAG structure for outcomes
    print(f"DAG STRUCTURE CHECK:")
    print(f"{'-'*80}")
    for outcome in outcomes:
        if outcome in model.nodes():
            parents = list(model.get_parents(outcome))
            print(f"{outcome} <- {parents if parents else 'NO PARENTS'}")
    print(f"{'-'*80}\n")
    
    for outcome in outcomes:
        print(f"\n{'='*80}")
        print(f"Outcome: {outcome}")
        print(f"{'='*80}\n")
        
        # Print outcome variable information
        if outcome in model.nodes():
            cpd = model.get_cpds(outcome)
            state_names = cpd.state_names.get(outcome, list(range(len(cpd.get_values()))))
            print(f"Variable '{outcome}' has {len(state_names)} states: {state_names}")
            print(f"State mapping: {dict(enumerate(state_names))}")
            
            # Clinical interpretation for last_status
            # Note: pandas converts strings to codes alphabetically: AWD->0, DOD->1, NED->2
            if outcome == "last_status" and len(state_names) == 3:
                print(f"Clinical meaning (alphabetical encoding):")
                print(f"  0 = AWD (Alive With Disease)")
                print(f"  1 = DOD (Dead of Disease)")
                print(f"  2 = NED (No Evidence of Disease)")
            print()
        
        # For binary outcomes (metastasis, local_recurrence), we measure P(outcome=1)
        # For multi-class (last_status), we'll show distribution across all states
        
        for regime_a, regime_b, label in comparisons:
            regime_a_str = " + ".join([f"{k}={v}" for k, v in regime_a.items()])
            regime_b_str = " + ".join([f"{k}={v}" for k, v in regime_b.items()])
            
            print(f"\n{'-'*80}")
            print(f"Comparison: {label}")
            print(f"  Regime A: {regime_a_str}")
            print(f"  Regime B: {regime_b_str}")
            print(f"{'-'*80}")
            
            try:
                # Compute P(outcome | do(regime_a), evidence)
                do_model_a = do_intervene(model, regime_a)
                infer_a = VariableElimination(do_model_a)
                result_a = infer_a.query(variables=[outcome], evidence=evidence) if evidence else infer_a.query(variables=[outcome])
                state_names_a = result_a.state_names.get(outcome, list(range(result_a.cardinality[0])))
                
                # Compute P(outcome | do(regime_b), evidence)
                do_model_b = do_intervene(model, regime_b)
                infer_b = VariableElimination(do_model_b)
                result_b = infer_b.query(variables=[outcome], evidence=evidence) if evidence else infer_b.query(variables=[outcome])
                state_names_b = result_b.state_names.get(outcome, list(range(result_b.cardinality[0])))
                
                # DIAGNOSTIC: Check marginal probabilities
                print(f"\n  DIAGNOSTIC - Marginal P({outcome}) in original model" + (f" | {evidence}" if evidence else "") + ":")
                infer_orig = VariableElimination(model)
                result_orig = infer_orig.query(variables=[outcome], evidence=evidence) if evidence else infer_orig.query(variables=[outcome])
                for state, prob in zip(state_names_a, result_orig.values):
                    print(f"    P({outcome}={state}): {prob:.4f}")
                
                # Check if binary or multi-class
                if len(state_names_a) == 2:
                    # Binary outcome: report P(outcome=1)
                    outcome_state = 1
                    outcome_state_idx = state_names_a.index(outcome_state) if outcome_state in state_names_a else 1
                    prob_a = float(result_a.values[outcome_state_idx])
                    prob_b = float(result_b.values[outcome_state_idx])
                    
                    ate = prob_b - prob_a
                    
                    result = {
                        "outcome": outcome,
                        "comparison": label,
                        "regime_a": regime_a,
                        "regime_b": regime_b,
                        "regime_a_str": regime_a_str,
                        "regime_b_str": regime_b_str,
                        "outcome_type": "binary",
                        "risk_a": prob_a,
                        "risk_b": prob_b,
                        "ate_pp": ate * 100,
                        "relative_change": (ate / prob_a * 100) if prob_a > 0 else None,
                        "outcome_state": outcome_state,
                        "status": "success"
                    }
                    
                    print(f"\n  Binary Outcome - P({outcome}=1):")
                    print(f"    Risk under A: {prob_a:.4f} ({prob_a*100:.2f}%)")
                    print(f"    Risk under B: {prob_b:.4f} ({prob_b*100:.2f}%)")
                    print(f"    ATE (risk diff): {ate:+.4f} ({ate*100:+.2f} pp)")
                    
                    if prob_a > 0:
                        rel_change = (ate / prob_a) * 100
                        print(f"    Relative change: {rel_change:+.2f}%")
                    
                    if ate > 0.01:
                        print(f"    ↑ Regime B INCREASES risk by {abs(ate*100):.2f} percentage points")
                    elif ate < -0.01:
                        print(f"    ↓ Regime B DECREASES risk by {abs(ate*100):.2f} percentage points")
                    else:
                        print(f"    → No substantial effect")
                else:
                    # Multi-class outcome: report full distribution
                    result = {
                        "outcome": outcome,
                        "comparison": label,
                        "regime_a": regime_a,
                        "regime_b": regime_b,
                        "regime_a_str": regime_a_str,
                        "regime_b_str": regime_b_str,
                        "outcome_type": "multiclass",
                        "distribution_a": {str(state): float(prob) for state, prob in zip(state_names_a, result_a.values)},
                        "distribution_b": {str(state): float(prob) for state, prob in zip(state_names_b, result_b.values)},
                        "status": "success"
                    }
                    
                    print(f"\n  Multi-class Outcome - Distribution:")
                    print(f"\n    Regime A:")
                    for state, prob in zip(state_names_a, result_a.values):
                        print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                    
                    print(f"\n    Regime B:")
                    for state, prob in zip(state_names_b, result_b.values):
                        print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                    
                    print(f"\n    Changes (B - A):")
                    for state in state_names_a:
                        idx = state_names_a.index(state)
                        change = result_b.values[idx] - result_a.values[idx]
                        print(f"      Δ P({outcome}={state}): {change:+.4f} ({change*100:+.2f} pp)")
                
            except Exception as e:
                result = {
                    "outcome": outcome,
                    "comparison": label,
                    "regime_a": regime_a,
                    "regime_b": regime_b,
                    "regime_a_str": regime_a_str,
                    "regime_b_str": regime_b_str,
                    "status": "error",
                    "error": str(e)
                }
                print(f"\n  ERROR: {e}")
            
            results.append(result)
    
    return results


def print_regime_table(results: list[dict]) -> None:
    """Print results in a clinical-style table format."""
    print(f"\n{'='*80}")
    print(f"TREATMENT REGIME EFFECTS TABLE")
    print(f"{'='*80}\n")
    
    # Group by outcome
    outcomes = {}
    for r in results:
        if r['status'] == 'success':
            outcome = r['outcome']
            if outcome not in outcomes:
                outcomes[outcome] = []
            outcomes[outcome].append(r)
    
    for outcome, outcome_results in outcomes.items():
        print(f"\n{'─'*80}")
        print(f"Outcome: {outcome}")
        print(f"{'─'*80}")
        
        # Check if binary or multi-class
        if outcome_results and outcome_results[0].get('outcome_type') == 'binary':
            # Binary outcome table
            print(f"\n{'Comparison':<40} {'Risk A':>8} {'Risk B':>8} {'ATE(pp)':>9} {'RelChg%':>9} {'Interpretation':<20}")
            print(f"{'-'*40} {'-'*8} {'-'*8} {'-'*9} {'-'*9} {'-'*20}")
            
            for r in outcome_results:
                comparison = r['comparison'][:39]
                risk_a = f"{r['risk_a']:.4f}" if r.get('risk_a') is not None else "N/A"
                risk_b = f"{r['risk_b']:.4f}" if r.get('risk_b') is not None else "N/A"
                ate = f"{r['ate_pp']:+.2f}" if r.get('ate_pp') is not None else "N/A"
                rel = f"{r['relative_change']:+.1f}" if r.get('relative_change') is not None else "N/A"
                
                if r.get('ate_pp') and abs(r['ate_pp']) > 1:
                    if r['ate_pp'] > 0:
                        interp = "↑ Increases risk"
                    else:
                        interp = "↓ Decreases risk"
                else:
                    interp = "→ No effect"
                
                print(f"{comparison:<40} {risk_a:>8} {risk_b:>8} {ate:>9} {rel:>9} {interp:<20}")
        else:
            # Multi-class outcome - show distributions
            for r in outcome_results:
                print(f"\n  Comparison: {r['comparison']}")
                print(f"    Regime A: {r['regime_a_str']}")
                dist_a = r.get('distribution_a', {})
                for state, prob in dist_a.items():
                    print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                
                print(f"    Regime B: {r['regime_b_str']}")
                dist_b = r.get('distribution_b', {})
                for state, prob in dist_b.items():
                    print(f"      P({outcome}={state}): {prob:.4f} ({prob*100:.2f}%)")
                
                print(f"    Changes (B - A):")
                for state in dist_a.keys():
                    change = dist_b.get(state, 0) - dist_a.get(state, 0)
                    print(f"      Δ P({outcome}={state}): {change:+.4f} ({change*100:+.2f} pp)")
        
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate ATE using do-interventions.")
    parser.add_argument("--model", required=True, help="Path to BN pickle.")
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--single", action="store_true", help="Single treatment-outcome calculation.")
    mode_group.add_argument("--multiple", action="store_true", help="Multiple treatment-outcome calculations.")
    mode_group.add_argument("--regimes", action="store_true", help="Compare treatment regimes.")
    mode_group.add_argument("--validate", action="store_true", help="Validate model predictions vs actual data.")
    
    # Single mode arguments
    parser.add_argument("--treatment", help="Treatment node name (required for --single).")
    parser.add_argument("--outcome", help="Outcome node name (required for --single).")
    parser.add_argument("--treatment-values", nargs=2, help="Control and treated values (required for --single).")
    parser.add_argument("--outcome-state", help="Outcome state to query (required for --single).")
    parser.add_argument("--evidence", nargs="*", default=[], help="Optional evidence as key=value pairs.")
    
    # Multiple mode arguments
    parser.add_argument("--treatments", nargs="+", help="List of treatment variables for --multiple mode.")
    parser.add_argument("--outcomes", nargs="+", help="List of outcome variables for --multiple and --regimes mode.")
    parser.add_argument("--output", help="Path to save results JSON (optional).")
    
    # Regime mode arguments
    parser.add_argument("--regime-comparisons", help="Path to JSON file defining regime comparisons.")
    
    # Validation mode arguments
    parser.add_argument("--config", help="Path to config YAML (required for --validate).")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of patients to display in sample table.")
    parser.add_argument("--full-dataset", action="store_true", help="Evaluate metrics on entire dataset (not just sample).")
    parser.add_argument("--plot-dir", help="Directory to save distribution plots (e.g., output/validation_plots).")
    parser.add_argument("--test-data-path", help="Path to test set pickle file (for uncontaminated validation).")

    args = parser.parse_args()

    model = load_bayesian_network(args.model)
    
    if args.single:
        # Validate single mode arguments
        if not all([args.treatment, args.outcome, args.treatment_values, args.outcome_state]):
            parser.error("--single mode requires --treatment, --outcome, --treatment-values, and --outcome-state")
        
        # Single treatment-outcome calculation
        raw_evidence = _parse_evidence(args.evidence)
        # Load config if available for preprocessing
        config = _load_config(args.config) if args.config else {}
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        ate = average_treatment_effect(
            model,
            args.treatment,
            args.outcome,
            tuple(_parse_value(val) for val in args.treatment_values),
            _parse_value(args.outcome_state),
            evidence=evidence,
        )
        print(f"ATE({args.treatment} -> {args.outcome}): {ate:.6f}")
    
    elif args.multiple:
        # Validate multiple mode arguments
        if not args.treatments or not args.outcomes:
            parser.error("--multiple mode requires --treatments and --outcomes")
        
        # Adapt outcomes: if last_status is requested but not in model, use NED/AWD/DOD instead
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                # Replace with binary status nodes if they exist in the model
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Parse evidence if provided
        raw_evidence = _parse_evidence(args.evidence) if args.evidence else None
        # Preprocess evidence to match BN discretization
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        # Multiple treatment-outcome calculations
        results = run_multiple_treatment_effects(model, adapted_outcomes, args.treatments, evidence)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total calculations: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"{'='*80}\n")
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}\n")
    
    elif args.regimes:
        # Validate regime mode arguments
        if not args.outcomes:
            parser.error("--regimes mode requires --outcomes")
        
        # Adapt outcomes: if last_status is requested but not in model, use NED/AWD/DOD instead
        model_nodes = set(model.nodes())
        adapted_outcomes = []
        for outcome in args.outcomes:
            if outcome == 'last_status' and outcome not in model_nodes:
                # Replace with binary status nodes if they exist in the model
                for status_node in STATUS_NODES:
                    if status_node in model_nodes:
                        adapted_outcomes.append(status_node)
            elif outcome in model_nodes:
                adapted_outcomes.append(outcome)
        
        if not adapted_outcomes:
            parser.error(f"None of the requested outcomes are in the model. Model nodes: {sorted(model_nodes)}")
        
        print(f"Analyzing outcomes: {adapted_outcomes}")
        
        # Parse evidence if provided
        raw_evidence = _parse_evidence(args.evidence) if args.evidence else None
        # Preprocess evidence to match BN discretization
        evidence = _preprocess_evidence(raw_evidence, model, config) if raw_evidence else None
        if raw_evidence and evidence:
            print(f"Evidence preprocessed: {raw_evidence} → {evidence}")
        
        # Load regime comparisons from file or use defaults
        if args.regime_comparisons:
            with open(args.regime_comparisons, 'r') as f:
                config = json.load(f)
                comparisons = []
                for comp in config['comparisons']:
                    comparisons.append((
                        comp['regime_a'],
                        comp['regime_b'],
                        comp['label']
                    ))
        else:
            # Default clinically meaningful comparisons
            comparisons = [
                (
                    {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0},
                    {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1},
                    "Add RT to surgery"
                ),
                (
                    {"surgery": 1, "chemotherapy": 0, "radiotherapy": 1},
                    {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1},
                    "Add chemo to surgery+RT"
                ),
                (
                    {"surgery": 1, "chemotherapy": 0, "radiotherapy": 0},
                    {"surgery": 1, "chemotherapy": 1, "radiotherapy": 1},
                    "Surgery only vs full treatment"
                ),
            ]
        
        # Run regime comparisons
        treatments = ["surgery", "chemotherapy", "radiotherapy"]
        results = run_regime_comparisons(model, adapted_outcomes, treatments, comparisons, evidence)
        
        # Print tabular output
        print_regime_table(results)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"Total comparisons: {len(results)}")
        print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"Errors: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"{'='*80}\n")
        
        # Save results if output path provided
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}\n")
    
    elif args.validate:
        # Validate mode - compare predictions to actual data
        if not args.config:
            parser.error("--validate mode requires --config")
        
        validate_predictions(model, args.config, args.sample_size, args.full_dataset, args.plot_dir, args.test_data_path)


if __name__ == "__main__":
    main()