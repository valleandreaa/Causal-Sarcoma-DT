import os
import json

from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
import pandas as pd
from torch.utils.data import DataLoader

from Models.models.GANs import LSTMGenerator
from Models.models.decoder import Decoder
from Models.models.utils import MetadataHandler
from Models.utils.helpers import MongoExtractor
from Models.dataset.tabular_dataset import TabularDatasetPID
import argparse
import numpy as np
from Models.models.decoder_embedders import decode_embedding
from sklearn.metrics import (
    mean_absolute_error,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import proportional_hazard_test
import matplotlib.pyplot as plt
from Inference.refined_evidence import (
    generate_refined_evidence_package,
    plot_sarculator_by_scenario,
    validate_model,
    plot_kaplan_meier,
    analyze_local_recurrence,
    estimate_ate,
    survival_rate_at,
    compute_survival_metrics,
    compute_survival_columns,
    generate_counterfactual_recurrence_outcomes,
)
def load_temporal_cycle_gan(meta_path: Path, device: torch.device):
    """Load Gx generator from saved metadata."""
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

def parse_treatment(treat: str) -> List[int]:
    parts = [int(p) for p in treat.split(",")]
    if len(parts) != 3:
        raise ValueError("Treatment must contain three comma separated integers")
    return parts

def show_applied_treatments(decoded: List[pd.DataFrame], actual_treat: torch.Tensor) -> None:
    """Display decoded steps only for which a treatment was applied."""
    applied_mask = (actual_treat.sum(dim=1) > 0).cpu()
    for idx, (df_out, applied) in enumerate(zip(decoded, applied_mask)):
        if applied:
            print(f"\n--- Decoded step {idx} ---")
            print(df_out.to_string(index=False))


def compare_neoadjuvant_adjuvant(df: pd.DataFrame, output: Path = Path("neoadjuvant_vs_adjuvant_km.png")) -> Dict[str, Any]:
    """Compare outcomes for neoadjuvant vs. adjuvant chemotherapy."""
    neo = df[df.get('episodes.treatments.fields.chemo_reason', "").str.contains("neo", case=False, na=False)]
    adj = df[df.get('episodes.treatments.fields.chemo_reason', "").str.contains("adjuvant", case=False, na=False)]

    if neo.empty or adj.empty:
        print("Insufficient data for comparison")
        return {}
    neo  = compute_survival_columns(neo)
    adj = compute_survival_columns(adj)
    t_times = neo["time_difference_years"].to_numpy()
    t_events = neo["event"].to_numpy()
    c_times = adj["time_difference_years"].to_numpy()
    c_events = adj["event"].to_numpy()

    plot_kaplan_meier(t_times, t_events, c_times, c_events, output)

    metrics: Dict[str, Any] = {
        "ate": estimate_ate(t_times, c_times),
        "survival_5yr_neo": survival_rate_at(5.0, t_times, t_events),
        "survival_5yr_adjuvant": survival_rate_at(5.0, c_times, c_events),
    }
    metrics.update(
        {
            "c_index_neo": compute_survival_metrics(t_times, t_events, t_times).get("c_index"),
            "c_index_adjuvant": compute_survival_metrics(c_times, c_events, c_times).get("c_index"),
        }
    )

    return metrics

def evaluate_research_question(
    rq: int,
    extractor: MongoExtractor,
    meta_clin: MetadataHandler,
    meta_treat: MetadataHandler,
    gx,
    decoder_treat,
    feat_spec,
    oh_map,
    ord_map,
    num_ranges,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate the requested research question."""
    df = extractor.get_dataframe()

    if rq == 1:
        raise ValueError(
            "Research Question 1 has moved to temporal_inference_high_risk.py")
        
    elif rq == 2:
        # Research Question 2: Local recurrence analysis - Surgery vs Surgery + Radiotherapy
        print("=== Research Question 2: Local Recurrence Analysis ===")
        print("Comparing Surgery alone vs Surgery + Radiotherapy")
        
        # Run comprehensive local recurrence analysis with inference components
        recurrence_results = analyze_local_recurrence(
            df,
            extractor=extractor,
            meta_clin=meta_clin,
            meta_treat=meta_treat,
            gx=gx,
            decoder_treat=decoder_treat,
            feat_spec=feat_spec,
            oh_map=oh_map,
            ord_map=ord_map,
            num_ranges=num_ranges,
            device=device
        )
        
        return recurrence_results
    elif rq == 3:
        chemo_scenario = {"chemo": [0, 1, 0]}
        df_with_counterfactuals = generate_counterfactual_recurrence_outcomes(
            df,
            extractor=extractor,
            meta_clin=meta_clin,
            meta_treat=meta_treat,
            gx=gx,
            decoder_treat=decoder_treat,
            feat_spec=feat_spec,
            oh_map=oh_map,
            ord_map=ord_map,
            num_ranges=num_ranges,
            device=device,
            treatment_scenarios=chemo_scenario,
        )
        
        # Merge original clinical data with counterfactual results
        combined_df = df.merge(df_with_counterfactuals, on='_id', how='left', suffixes=('_original', '_counterfactual'))
        combined_df.to_csv('counterfactual_results_3q.csv', index=False)
        return compare_neoadjuvant_adjuvant(df)


def main():
    # For a Jupyter Notebook, preset parameters instead of using argparse
    parser = argparse.ArgumentParser(description="Temporal Inference Script Arguments")
    parser.add_argument(
        "--meta_file", type=str,
        default="saved_models/temporal_cycle_gan_20250712_111007/metadata.json",
        help="Path to the metadata file for the temporal cycle gan."
    )
    parser.add_argument(
        "--config_file", type=str,
        default="Models/configs/temporal_inference.yaml",
        help="Path to the Mongo configuration file."
    )
    parser.add_argument(
        "--patient_id", type=int,
        default=7095159,
        help="Patient id to retrieve."
    )

    parser.add_argument(
        "--next_treatment",
        type=str,
        default="1,0,0",
        help="Potential next treatment as comma separated values (surgery, chemotherapy, radiotherapy)",
    )
    parser.add_argument(
        "--show_only_applied",
        action="store_true",
        help="Display only steps where a treatment was applied",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation metrics on the validation cohort",
    )

    parser.add_argument(
        "--research_question",
        type=int,
        choices=[1, 2, 3],
        help="Evaluate a specific research question (1, 2 or 3)",
    )

    args = parser.parse_args()

    meta_file = args.meta_file
    config_file = args.config_file
    patient_id = args.patient_id
    next_treatment = args.next_treatment
    show_only_applied = args.show_only_applied
    run_validation = args.validate
    research_question = args.research_question

    treat_vec = parse_treatment(next_treatment)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gx, meta = load_temporal_cycle_gan(Path(meta_file), device)

    meta_clin = MetadataHandler(meta["metadata_clinical_file"])
    meta_treat = MetadataHandler(meta["metadata_treatment_file"])

    decoder_treat, feat_spec, oh_map, ord_map, num_ranges = load_decoder(
        meta["metadata_treatment_file"], meta["decoder_treatment"], device
    )

    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=config_file,
    )

    df = extractor.get_patient_by_id(patient_id)


    dataset = TabularDatasetPID(
        df,
        meta_clin,
        meta_treat,
        seq_len=None,
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    _, clin_emb, actual_treat = next(iter(loader))
    clin_emb = clin_emb.to(device)
    actual_treat = actual_treat.to(device).float()
    with torch.no_grad():
        fake_actual = gx(clin_emb, actual_treat)

    decoded_actual = [
        decode_embedding(
            decoder_treat(fake_actual[0, i]),
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
        )
        for i in range(fake_actual.size(1))
    ]
    
    treat_tensor = torch.tensor(treat_vec, dtype=torch.float32, device=device).unsqueeze(0)
    treat_tensor = treat_tensor.unsqueeze(1).repeat(1, clin_emb.size(1), 1)

    with torch.no_grad():
        fake_treat = gx(clin_emb, treat_tensor)

    decoded = [
        decode_embedding(
            decoder_treat(fake_treat[0, i]),
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
        )
        for i in range(fake_treat.size(1))
    ]
    
    for idx, df_out in enumerate(decoded):
        print(f"\n--- Decoded step {idx} ---")
        print(df_out.to_string(index=False))

    next_tensor = torch.tensor(treat_vec, dtype=torch.float32, device=device).view(1, 1, -1)
    clin_next = torch.cat([clin_emb, clin_emb[:, -1:, :]], dim=1)
    treat_next = torch.cat([actual_treat, next_tensor], dim=1)

    with torch.no_grad():
        fake_next = gx(clin_next, treat_next)

    decoded_next = decode_embedding(
        decoder_treat(fake_next[0, -1]),
        feat_spec,
        oh_map,
        ord_map,
        num_ranges,
    )

    if show_only_applied:
        show_applied_treatments(decoded_actual, actual_treat[0])
    else:
        for idx, df_out in enumerate(decoded_actual):
            print(f"\n--- Decoded step {idx} ---")
            for _, row in df_out.iterrows():
                print(" \n ".join([f"{col}: {row[col]}" for col in df_out.columns]))

        print("\n--- Potential next treatment ---")
        for _, row in decoded_next.iterrows():
            print(" \n ".join([f"{col}: {row[col]}" for col in decoded_next.columns]))

    if run_validation:
        print("\n=== Running Model Validation ===")
        metrics = validate_model(
            gx,
            meta,
            extractor,
            meta_clin,
            meta_treat,
            decoder_treat,
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
            device,
        )
        print("\nValidation metrics:")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        
        # Display CSV export status
        if metrics.get("validation_csv_saved"):
            print(f"\nValidation data exported to: {metrics.get('validation_csv_filename')}")
            print(f"Records: {metrics.get('validation_records')}")
            print(f"Patients: {metrics.get('validation_patients')}")
        else:
            print("\nWarning: No validation data was saved to CSV")

    if research_question:
        extractor = MongoExtractor(
                connection_string=os.getenv("MONGO_URI"),
                database_name=os.getenv("MONGO_DB"),
                collection_name=os.getenv("MONGO_COLLECTION"),
                config_file=config_file,
            )
        rq_metrics = evaluate_research_question(
            research_question,
            extractor,
            meta_clin,
            meta_treat,
            gx,
            decoder_treat,
            feat_spec,
            oh_map,
            ord_map,
            num_ranges,
            device,
        )
        if rq_metrics:
            print(f"\nResearch question {research_question} metrics:")
            for k, v in rq_metrics.items():
                print(f"{k}: {v}")

def plot_sarculator_by_scenario(stage_iii_df_generated: pd.DataFrame, filename: str = "sarculator_comparison_stage_iii.png") -> None:
    """Create box plots and bar charts comparing sarculator scores across treatment scenarios."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Define treatment scenario labels for better visualization
        scenario_labels = {
            'S': 'Surgery Only',
            'S_CT': 'Surgery + Chemo',
            'S_RT': 'Surgery + RT',
            'S_RT_CT': 'Surgery + RT + Chemo'
        }
        
        # 5-year sarculator box plot
        if 'general.sarculator_five' in stage_iii_df_generated.columns and 'treatment_scenario' in stage_iii_df_generated.columns:
            valid_data = stage_iii_df_generated.dropna(subset=['general.sarculator_five', 'treatment_scenario'])
            
            if len(valid_data) > 0:
                # Box plot
                scenarios = []
                scores = []
                for scenario in valid_data['treatment_scenario'].unique():
                    scenario_data = valid_data[valid_data['treatment_scenario'] == scenario]['general.sarculator_five']
                    scenarios.extend([scenario_labels.get(scenario, scenario)] * len(scenario_data))
                    scores.extend(scenario_data.values)
                
                if scenarios:
                    import pandas as pd
                    box_data = pd.DataFrame({'Scenario': scenarios, 'Sarculator_5yr': scores})
                    box_data.boxplot(column='Sarculator_5yr', by='Scenario', ax=axes[0,0])
                    axes[0,0].set_title('5-year Sarculator Scores by Treatment')
                    axes[0,0].set_ylabel('Sarculator 5-year Score')
                    axes[0,0].tick_params(axis='x', rotation=45)
                
                # Mean comparison bar chart
                scenario_means = {}
                for scenario in valid_data['treatment_scenario'].unique():
                    scenario_data = valid_data[valid_data['treatment_scenario'] == scenario]['general.sarculator_five']
                    scenario_means[scenario_labels.get(scenario, scenario)] = scenario_data.mean()
                
                if scenario_means:
                    scenarios_list = list(scenario_means.keys())
                    means_list = list(scenario_means.values())
                    colors = ['blue', 'green', 'orange', 'red'][:len(scenarios_list)]
                    
                    bars = axes[0,1].bar(scenarios_list, means_list, color=colors, alpha=0.7)
                    axes[0,1].set_title('Mean 5-year Sarculator Scores')
                    axes[0,1].set_ylabel('Mean Sarculator 5-year Score')
                    axes[0,1].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, means_list):
                        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 10-year sarculator box plot
        if 'general.sarculator_ten' in stage_iii_df_generated.columns and 'treatment_scenario' in stage_iii_df_generated.columns:
            valid_data = stage_iii_df_generated.dropna(subset=['general.sarculator_ten', 'treatment_scenario'])
            
            if len(valid_data) > 0:
                # Box plot
                scenarios = []
                scores = []
                for scenario in valid_data['treatment_scenario'].unique():
                    scenario_data = valid_data[valid_data['treatment_scenario'] == scenario]['general.sarculator_ten']
                    scenarios.extend([scenario_labels.get(scenario, scenario)] * len(scenario_data))
                    scores.extend(scenario_data.values)
                
                if scenarios:
                    box_data = pd.DataFrame({'Scenario': scenarios, 'Sarculator_10yr': scores})
                    box_data.boxplot(column='Sarculator_10yr', by='Scenario', ax=axes[1,0])
                    axes[1,0].set_title('10-year Sarculator Scores by Treatment')
                    axes[1,0].set_ylabel('Sarculator 10-year Score')
                    axes[1,0].tick_params(axis='x', rotation=45)
                
                # Mean comparison bar chart
                scenario_means = {}
                for scenario in valid_data['treatment_scenario'].unique():
                    scenario_data = valid_data[valid_data['treatment_scenario'] == scenario]['general.sarculator_ten']
                    scenario_means[scenario_labels.get(scenario, scenario)] = scenario_data.mean()
                
                if scenario_means:
                    scenarios_list = list(scenario_means.keys())
                    means_list = list(scenario_means.values())
                    colors = ['blue', 'green', 'orange', 'red'][:len(scenarios_list)]
                    
                    bars = axes[1,1].bar(scenarios_list, means_list, color=colors, alpha=0.7)
                    axes[1,1].set_title('Mean 10-year Sarculator Scores')
                    axes[1,1].set_ylabel('Mean Sarculator 10-year Score')
                    axes[1,1].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, val in zip(bars, means_list):
                        axes[1,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                     f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('AJCC Stage III STS: Sarculator Score Comparison by Treatment Scenario', fontsize=14)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sarculator treatment comparison plot saved: {filename}")
        
    except Exception as exc:
        print(f"Sarculator treatment comparison plot failed: {exc}")

def plot_sarculator_comparison(stage_iii_df_generated: pd.DataFrame, filename: str = "sarculator_comparison.png") -> None:
    """Create scatter plots comparing original vs generated sarculator scores."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 5-year sarculator comparison
        if 'general.sarculator_five' in stage_iii_df_generated.columns and 'original_sarculator_five' in stage_iii_df_generated.columns:
            valid_data = stage_iii_df_generated.dropna(subset=['general.sarculator_five', 'original_sarculator_five'])
            
            if len(valid_data) > 0:
                original_five = valid_data['original_sarculator_five'].values
                generated_five = valid_data['general.sarculator_five'].values
                
                axes[0].scatter(original_five, generated_five, alpha=0.6, s=30)
                axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect agreement')
                axes[0].set_xlabel('Original Sarculator 5-year')
                axes[0].set_ylabel('Generated Sarculator 5-year')
                axes[0].set_title(f'5-year Sarculator Comparison\n(n={len(valid_data)})')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(original_five, generated_five)[0, 1] if len(original_five) > 1 else np.nan
                mae = np.mean(np.abs(generated_five - original_five))
                axes[0].text(0.05, 0.95, f'r = {corr:.3f}\nMAE = {mae:.3f}', 
                           transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 10-year sarculator comparison  
        if 'general.sarculator_ten' in stage_iii_df_generated.columns and 'original_sarculator_ten' in stage_iii_df_generated.columns:
            valid_data = stage_iii_df_generated.dropna(subset=['general.sarculator_ten', 'original_sarculator_ten'])
            
            if len(valid_data) > 0:
                original_ten = valid_data['original_sarculator_ten'].values
                generated_ten = valid_data['general.sarculator_ten'].values
                
                axes[1].scatter(original_ten, generated_ten, alpha=0.6, s=30, color='orange')
                axes[1].plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect agreement')
                axes[1].set_xlabel('Original Sarculator 10-year')
                axes[1].set_ylabel('Generated Sarculator 10-year')
                axes[1].set_title(f'10-year Sarculator Comparison\n(n={len(valid_data)})')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Add correlation coefficient
                corr = np.corrcoef(original_ten, generated_ten)[0, 1] if len(original_ten) > 1 else np.nan
                mae = np.mean(np.abs(generated_ten - original_ten))
                axes[1].text(0.05, 0.95, f'r = {corr:.3f}\nMAE = {mae:.3f}', 
                           transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Sarculator comparison plot saved: {filename}")
        
    except Exception as exc:
        print(f"Sarculator comparison plot failed: {exc}")


if __name__ == "__main__":
    main()