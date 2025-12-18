import os
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import ks_2samp
from tabulate import tabulate

from Models.models.GANs import LSTMGenerator
from Models.models.decoder import Decoder
from Models.models.utils import MetadataHandler, randomize_multiple_one_hot

# Number of additional ones to add per row when randomizing multi-hot
EXTRA_ONES = 1

from Models.utils.helpers import MongoExtractor
from Models.dataset.tabular_dataset import TabularDatasetPID
from Models.models.decoder_embedders import decode_embedding
from Inference.refined_evidence import compute_survival_columns
from Inference.influence_function_validator import (
    validate_cyclegan_with_if, 
    print_if_validation_summary
)


def load_temporal_cycle_gan(meta_path: Path, device: torch.device):
    """Load generator and metadata."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    params = meta.get("model_params", {})
    embedding_dim = params.get("embedding_dim", 192)
    hidden_dim = params.get("hidden_dim", 128)
    num_layers = params.get("num_layers", 1)

    gx = LSTMGenerator(
        input_dim=embedding_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)
    gx.load_state_dict(torch.load(meta["Gx_path"], map_location=device))
    gx.eval()
    return gx, meta


def load_decoder(meta_file: str, decoder_path: str, device: torch.device):
    """Load decoder network using parameters stored in metadata."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta.get("model_params", {})
    hidden_dim = params.get("hidden_dim")
    output_dim = params.get("input_dim")
    dec = Decoder(hidden_dim, output_dim).to(device)
    dec.load_state_dict(torch.load(decoder_path, map_location=device))
    dec.eval()
    feature_spec = meta.get("feature_spec")
    one_hot_mappings = meta.get("one_hot_mappings", {})
    ord_mappings = meta.get("ord_mappings", {})
    numeric_ranges = meta.get("numeric_ranges", {})
    return dec, feature_spec, one_hot_mappings, ord_mappings, numeric_ranges


def generate_fake_dataset(
    loader: DataLoader,
    gx,
    decoder_treat,
    feat_spec: Dict[str, Any],
    oh_map: Dict[str, Any],
    ord_map: Dict[str, Any],
    num_ranges: Dict[str, Any],
    device: torch.device,
) -> pd.DataFrame:
    """Generate decoded treatment records for a loader batch."""
    from Models.models.utils import randomize_one_hot
    
    fake_rows: List[pd.DataFrame] = []
    gx.eval()
    with torch.no_grad():
        for treat, clin, actual in loader:
            treat = treat.to(device).float()
            clin = clin.to(device).float()
            actual = actual.to(device)
            
            # Create counterfactual conditioning from actual treatments
            tr_counter = randomize_one_hot(actual)
            
            # Generate fake treatments: Gx(clinical_embeddings, counterfactual_conditioning)
            fake = gx(clin, tr_counter)
            
            # Decode each timestep for each batch
            for b in range(fake.size(0)):
                for t in range(fake.size(1)):
                    decoded = decode_embedding(
                        decoder_treat(fake[b, t]), feat_spec, oh_map, ord_map, num_ranges
                    )
                    fake_rows.append(decoded)
    if fake_rows:
        return pd.concat(fake_rows, ignore_index=True)
    return pd.DataFrame()


def evaluate_clinical_plausibility(
    df: pd.DataFrame, numeric_ranges: Dict[str, Tuple[float, float]]
) -> Dict[str, float]:
    """Return fraction of values within allowed numeric ranges."""
    results: Dict[str, float] = {}
    for col, bounds in numeric_ranges.items():
        if col not in df.columns:
            continue
        mn, mx = bounds
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        within = ((vals >= mn) & (vals <= mx)).mean()
        results[col] = float(within)
    return results


def evaluate_sub_population_comparison(
    df: pd.DataFrame, treatment_col: str, outcome_col: str
) -> Dict[str, float]:
    """Compare outcome distribution for treated vs untreated cohorts."""
    if treatment_col not in df.columns or outcome_col not in df.columns:
        return {}
    t_vals = df[df[treatment_col] == 1][outcome_col].dropna()
    c_vals = df[df[treatment_col] == 0][outcome_col].dropna()
    if len(t_vals) == 0 or len(c_vals) == 0:
        return {}
    stat, pval = ks_2samp(t_vals, c_vals)
    return {
        "mean_treat": float(t_vals.mean()),
        "mean_control": float(c_vals.mean()),
        "ks_pvalue": float(pval),
    }


def evaluate_intra_generator_consistency(
    loader: DataLoader, gx, device: torch.device, draws: int = 3
) -> float:
    """Measure output variance across repeated generation draws."""
    from Models.models.utils import randomize_one_hot
    
    gx.eval()
    variances: List[float] = []
    with torch.no_grad():
        for treat, clin, actual in loader:
            treat = treat.to(device).float()
            clin = clin.to(device).float()
            actual = actual.to(device)
            
            samples = []
            for _ in range(draws):
                # Create different counterfactual conditioning for each draw
                tr_counter = randomize_one_hot(actual)
                samples.append(gx(clin, tr_counter))
            stacked = torch.stack(samples)
            variances.append(stacked.var(dim=0).mean().item())
    return float(np.mean(variances)) if variances else float("nan")


def evaluate_distribution_shift(
    real_df: pd.DataFrame, fake_df: pd.DataFrame, feature_spec: Dict[str, Any]
) -> Dict[str, float]:
    """Compute KS statistic for numeric columns between real and fake data."""
    numeric_cols = [c for c, t in feature_spec.items() if t == "numeric"]
    results: Dict[str, float] = {}
    for col in numeric_cols:
        if col in real_df.columns and col in fake_df.columns:
            r = real_df[col].dropna()
            f = fake_df[col].dropna()
            if len(r) > 0 and len(f) > 0:
                stat, _ = ks_2samp(r, f)
                results[col] = float(stat)
    return results


def evaluate_treatment_effect_variability(
    loader: DataLoader,
    gx,
    decoder_treat,
    feat_spec: Dict[str, Any],
    oh_map: Dict[str, Any],
    ord_map: Dict[str, Any],
    num_ranges: Dict[str, Any],
    device: torch.device,
    num_treatment_scenarios: int = 5
) -> Dict[str, float]:
    """Check if same patient with different treatments produces different outputs.
    
    This validates that the model is learning treatment effects by generating
    multiple counterfactual scenarios for the same patient and measuring
    the variability in outputs.
    
    Parameters
    ----------
    loader : DataLoader
        DataLoader containing patient data
    gx : torch.nn.Module
        Generator model
    decoder_treat : torch.nn.Module
        Treatment decoder
    feat_spec, oh_map, ord_map, num_ranges : Dict
        Feature specifications and mappings
    device : torch.device
        Computing device
    num_treatment_scenarios : int
        Number of different treatment scenarios to generate per patient
        
    Returns
    -------
    Dict[str, float]
        Statistics about treatment effect variability
    """
    from Models.models.utils import randomize_one_hot
    
    gx.eval()
    patient_variances = []
    feature_variances = {col: [] for col in feat_spec.keys()}
    
    with torch.no_grad():
        for treat, clin, actual in loader:
            treat = treat.to(device).float()
            clin = clin.to(device).float()
            actual = actual.to(device)
            
            # Generate multiple treatment scenarios for all patients at once
            scenario_outputs = []
            for scenario in range(num_treatment_scenarios):
                # Create different counterfactual treatment conditioning for all patients
                tr_counter = randomize_multiple_one_hot(actual, extra_ones=EXTRA_ONES)
                
                # Generate fake treatments for this scenario (all patients at once)
                fake_embeddings = gx(clin, tr_counter)
                
                # Decode all timesteps for all patients in this scenario
                scenario_rows = []
                for b in range(fake_embeddings.size(0)):  # For each patient
                    for t in range(fake_embeddings.size(1)):  # For each timestep
                        decoded = decode_embedding(
                            decoder_treat(fake_embeddings[b, t]), 
                            feat_spec, oh_map, ord_map, num_ranges
                        )
                        # Add patient identifier to track which patient this belongs to
                        decoded['patient_idx'] = b
                        decoded['timestep'] = t
                        scenario_rows.append(decoded)
                
                if scenario_rows:
                    scenario_df = pd.concat(scenario_rows, ignore_index=True)
                    scenario_outputs.append(scenario_df)
            
            # Calculate variance across scenarios for each patient
            if len(scenario_outputs) >= 2:
                # Get unique patient indices
                num_patients = clin.size(0)
                
                for patient_idx in range(num_patients):
                    # Extract data for this specific patient across all scenarios
                    patient_scenario_data = []
                    for scenario_df in scenario_outputs:
                        patient_data = scenario_df[scenario_df['patient_idx'] == patient_idx]
                        if not patient_data.empty:
                            patient_scenario_data.append(patient_data)
                    
                    if len(patient_scenario_data) >= 2:
                        # For each numeric feature, calculate variance across scenarios
                        numeric_cols = [c for c, t in feat_spec.items() if t == "numeric"]
                        patient_feature_vars = []
                        
                        for col in numeric_cols:
                            if col in patient_scenario_data[0].columns:
                                col_values = []
                                for patient_df in patient_scenario_data:
                                    # Take mean of this feature across timesteps for this scenario
                                    col_mean = patient_df[col].dropna().mean()
                                    if not pd.isna(col_mean):
                                        col_values.append(col_mean)
                                
                                if len(col_values) >= 2:
                                    col_var = np.var(col_values)
                                    patient_feature_vars.append(col_var)
                                    feature_variances[col].append(col_var)
                        
                        if patient_feature_vars:
                            # Average variance across features for this patient
                            patient_variances.append(np.mean(patient_feature_vars))
    
    # Calculate summary statistics
    results = {}
    
    if patient_variances:
        results["mean_patient_variance"] = float(np.mean(patient_variances))
        results["std_patient_variance"] = float(np.std(patient_variances))
        results["num_patients_evaluated"] = len(patient_variances)
        
        # Check what fraction of patients show meaningful treatment effects
        # (variance above a small threshold to account for numerical precision)
        meaningful_threshold = 1e-6
        meaningful_effects = np.array(patient_variances) > meaningful_threshold
        results["fraction_with_treatment_effects"] = float(meaningful_effects.mean())
    else:
        results["mean_patient_variance"] = 0.0
        results["std_patient_variance"] = 0.0
        results["num_patients_evaluated"] = 0
        results["fraction_with_treatment_effects"] = 0.0
    
    # Feature-specific variances
    for col, variances in feature_variances.items():
        if variances:
            results[f"mean_variance_{col}"] = float(np.mean(variances))
    
    return results


def collect_and_filter_validation_data(extractor: Any, meta: Dict[str, Any], 
                                     meta_clin: Any, meta_treat: Any) -> Tuple[List[Tuple[str, Any]], List[Any], Dict[str, Any]]:
    """Step 1: Collect all data and filter for validation patients from metadata.
    
    Parameters
    ----------
    extractor : Any
        Data extractor instance
    meta : Dict[str, Any]
        Model metadata containing validation patient IDs
    meta_clin : Any
        Clinical metadata handler
    meta_treat : Any
        Treatment metadata handler
        
    Returns
    -------
    Tuple[List[Tuple[str, Any]], List[ValidationPatient], Dict[str, Any]]
        Raw patient data, processed validation patients, and collection stats
    """
    print("="*60)
    print("STEP 1: DATA COLLECTION AND FILTERING")
    print("="*60)
    
    # Get all data from extractor
    print("Collecting all patient data...")
    all_data = extractor.get_dataframe()
    print(f"Total patients in dataset: {len(all_data['_id'].unique())}")
    
    # Get validation patient IDs from metadata
    val_ids = meta.get("val_patient_ids")
    if not val_ids:
        print("ERROR: No validation patient IDs found in metadata")
        return [], [], {"error": "No validation patient IDs in metadata"}
    
    print(f"Validation patient IDs from metadata: {len(val_ids)}")
    
    # Filter for available validation patients
    available_patient_ids = set(all_data['_id'].unique())
    available_val_ids = [pid for pid in val_ids if pid in available_patient_ids]
    
    if not available_val_ids:
        print("ERROR: No validation patients found in the dataset")
        return [], [], {"error": "No validation patients found in dataset"}
    
    if len(available_val_ids) < len(val_ids):
        print(f"WARNING: Only {len(available_val_ids)} of {len(val_ids)} validation patients found in dataset")
    
    print(f"Available validation patients: {len(available_val_ids)}")
    
    # Collect validation patient data in batch
    print("Collecting validation patient data...")
    batch_patient_data = []
    failed_patient_ids = []
    
    for pid in available_val_ids:
        try:
            df = all_data[all_data['_id'] == pid]
            if df.empty:
                failed_patient_ids.append(pid)
                continue
            batch_patient_data.append((pid, df))
        except Exception as e:
            print(f"Data collection failed for patient {pid}: {e}")
            failed_patient_ids.append(pid)
    
    print(f"Successfully collected: {len(batch_patient_data)} patients")
    print(f"Failed to collect: {len(failed_patient_ids)} patients")
    
    # Process patient data for embeddings in batch
    print("Processing patient data for embeddings in batch...")
    
    # Combine all patient data into a single DataFrame
    if not batch_patient_data:
        print("ERROR: No patient data collected")
        return None, None
    
    print(f"Combining data from {len(batch_patient_data)} patients...")
    combined_df_list = []
    for pid, patient_df in batch_patient_data:
        combined_df_list.append(patient_df)
    
    # Concatenate all patient DataFrames
    try:
        combined_df = pd.concat(combined_df_list, ignore_index=True)
        print(f"Combined DataFrame shape: {combined_df.shape}")
    except Exception as e:
        print(f"ERROR: Failed to combine patient data: {e}")
        return None, None
    
    # Create dataset from all collected patient data at once
    print(f"Creating TabularDatasetPID with combined data...")
    try:
        dataset = TabularDatasetPID(combined_df, meta_clin, meta_treat, seq_len=8)
        
        # Use batch size equal to the number of patients for single batch processing
        batch_size = len(batch_patient_data)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        print(f"Created DataLoader with batch_size={batch_size}")
        
        return dataset, loader
        
    except Exception as e:
        print(f"ERROR: Failed to create dataset and loader: {e}")
        return None, None

def main() -> None:
    """Command line entry point for IF validation."""
    parser = argparse.ArgumentParser(
        description="Temporal inference validation for high-risk STS (AJCC Stage III patients)"
    )
    parser.add_argument(
        "--meta_file",
        type=str,
        default="saved_models/temporal_cycle_gan_20250722_204311/metadata.json",
        help="Path to the CycleGAN metadata JSON file.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="Models/configs/temporal_inference.yaml",
        help="Path to the Mongo configuration YAML file.",
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gx, meta = load_temporal_cycle_gan(Path(args.meta_file), device)
    meta_clin = MetadataHandler(meta["metadata_clinical_file"])
    meta_treat = MetadataHandler(meta["metadata_treatment_file"])
    decoder_treat, feat_spec, oh_map, ord_map, num_ranges = load_decoder(
        meta["metadata_treatment_file"], meta["decoder_treatment"], device
    )

    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=args.config_file,
    )

    # Step 1: Collect all data and filter for validation patients
    dataset, loader = collect_and_filter_validation_data(
        extractor, meta, meta_clin, meta_treat
    )
    

    fake_df = generate_fake_dataset(
        loader,
        gx,
        decoder_treat,
        feat_spec,
        oh_map,
        ord_map,
        num_ranges,
        device,
    )

    real_df = extractor.get_dataframe()
    real_df = compute_survival_columns(real_df)

    results = {
        "clinical_plausibility": evaluate_clinical_plausibility(fake_df, num_ranges),
        "sub_population_comparison": evaluate_sub_population_comparison(
            real_df, "episodes.chemotherapy", "event"
        ),
        "intra_generator_consistency": evaluate_intra_generator_consistency(
            loader, gx, device
        ),
        "treatment_effect_variability": evaluate_treatment_effect_variability(
            loader, gx, decoder_treat, feat_spec, oh_map, ord_map, num_ranges, device
        ),
        "distribution_shift": evaluate_distribution_shift(real_df, fake_df, feat_spec),
    }

    # Add Influence Function Validation
    print("\n" + "="*80)
    print("RUNNING INFLUENCE FUNCTION VALIDATION")
    print("="*80)
    
    if_results = validate_cyclegan_with_if(
        loader, gx, decoder_treat, feat_spec, oh_map, ord_map, num_ranges, device
    )
    
    results["influence_function_validation"] = if_results

    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    
    # Prepare data for tabulated output with shortened names
    table_data = []
    
    def shorten_metric_name(full_name):
        """Shorten long metric names for better display."""
        # Remove common prefixes to make names shorter
        name = full_name.replace("episodes.treatments.fields.", "")
        name = name.replace("clinical_plausibility.", "plausibility.")
        name = name.replace("sub_population_comparison.", "subpop.")
        name = name.replace("treatment_effect_variability.", "treatment_var.")
        name = name.replace("distribution_shift.", "dist_shift.")
        name = name.replace("intra_generator_consistency", "generator_consistency")
        name = name.replace("influence_function_validation.", "if_validation.")
        
        # Further shorten specific field names
        name = name.replace("number_surgery_reoperations", "surgery_reops")
        name = name.replace("duration_chemotherapy", "chemo_duration")
        name = name.replace("radiotherapy_total_dose", "radio_dose")
        name = name.replace("redonc_fractions_gray", "redonc_gray")
        name = name.replace("redonc_fractions", "redonc_fractions")
        name = name.replace("tumor_max_size", "tumor_size")
        name = name.replace("general.prom_score", "prom_score")
        
        return name
    
    def format_value(value):
        """Format a value for display in the table."""
        if isinstance(value, float):
            if np.isnan(value):
                return "NaN"
            elif abs(value) < 1e-10:  # Very small numbers
                return "~0.000"
            elif value == 0:
                return "0.000"
            elif value == 1:
                return "1.000"
            else:
                return f"{value:.3f}"
        elif isinstance(value, bool):
            return "True" if value else "False"
        elif isinstance(value, (int, np.integer)):
            return str(value)
        elif isinstance(value, (list, np.ndarray)):
            try:
                return f"[{len(value)} items]"
            except:
                return "[array]"
        elif isinstance(value, dict):
            return f"{{dict with {len(value)} keys}}"
        else:
            return str(value)
    
    for metric_name, metric_results in results.items():
        if isinstance(metric_results, dict):
            # For nested dictionaries, create sub-rows
            for sub_metric, value in metric_results.items():
                full_metric_name = f"{metric_name}.{sub_metric}"
                short_name = shorten_metric_name(full_metric_name)
                formatted_value = format_value(value)
                table_data.append([short_name, formatted_value])
        else:
            # For simple values
            short_name = shorten_metric_name(metric_name)
            formatted_value = format_value(metric_results)
            table_data.append([short_name, formatted_value])
    
    # Print the table with simple formatting to avoid expandtabs errors
    print("┌" + "─" * 52 + "┬" + "─" * 17 + "┐")
    print(f"│ {'Metric':<50} │ {'Value':>15} │")
    print("├" + "─" * 52 + "┼" + "─" * 17 + "┤")
    
    for row in table_data:
        metric = str(row[0])[:50]  # Truncate if too long
        value = str(row[1])[:15]   # Truncate if too long
        print(f"│ {metric:<50} │ {value:>15} │")
    
    print("└" + "─" * 52 + "┴" + "─" * 17 + "┘")
    
    # Print summary interpretation
    print("\n" + "="*80)
    print("INTERPRETATION SUMMARY")
    print("="*80)
    
    # Clinical Plausibility Summary
    if "clinical_plausibility" in results:
        cp_results = results["clinical_plausibility"]
        print("\n📊 CLINICAL PLAUSIBILITY:")
        good_features = [k for k, v in cp_results.items() if v > 0.8]
        poor_features = [k for k, v in cp_results.items() if v < 0.5]
        print(f"   ✅ Good (>80%): {len(good_features)} features")
        print(f"   ❌ Poor (<50%): {len(poor_features)} features")
        if poor_features:
            print(f"   🔍 Needs attention: {', '.join(poor_features[:3])}{'...' if len(poor_features) > 3 else ''}")
    
    # Treatment Effect Variability Summary
    if "treatment_effect_variability" in results:
        tev_results = results["treatment_effect_variability"]
        print("\n🔄 TREATMENT EFFECT VARIABILITY:")
        if "fraction_with_treatment_effects" in tev_results:
            fraction = tev_results["fraction_with_treatment_effects"]
            if fraction > 0.7:
                print(f"   ✅ Strong treatment effects: {fraction:.1%} of patients show variability")
            elif fraction > 0.3:
                print(f"   ⚠️  Moderate treatment effects: {fraction:.1%} of patients show variability")
            else:
                print(f"   ❌ Weak treatment effects: {fraction:.1%} of patients show variability")
    
    # Distribution Shift Summary
    if "distribution_shift" in results:
        ds_results = results["distribution_shift"]
        print("\n📈 DISTRIBUTION SHIFT:")
        high_shift = [k for k, v in ds_results.items() if v > 0.7]
        low_shift = [k for k, v in ds_results.items() if v < 0.3]
        print(f"   ❌ High shift (>0.7): {len(high_shift)} features")
        print(f"   ✅ Low shift (<0.3): {len(low_shift)} features")
    
    # Influence Function Validation Summary
    if "influence_function_validation" in results:
        print_if_validation_summary(results["influence_function_validation"])
        
    print("\n" + "="*80)


if __name__ == "__main__":
    main()