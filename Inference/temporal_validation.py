import os
import json

from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

from tabulate import tabulate
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
from sklearn.metrics import mean_absolute_error, accuracy_score, f1_score, brier_score_loss
from Models.monitoring.loss_logger import plot_train_cycle_gan


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

def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load full patient dataset from JSON file."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_dataset_characteristics(df: pd.DataFrame, val_ids: List[int]) -> None:
    """Print dataset characteristics for training and validation cohorts."""
    # Split data based on validation IDs
    val_set = df[df['_id'].isin(val_ids)]
    train_set = df[~df['_id'].isin(val_ids)]

    def summarize(data: pd.DataFrame) -> Dict[str, Any]:
        n_patients = len(data)
        
        # Get age statistics
        age_col = None
        for col in data.columns:
            if 'age' in col.lower() and pd.api.types.is_numeric_dtype(data[col]):
                age_col = col
                break
        
        if age_col:
            ages = data[age_col].dropna()
            mean_age = float(np.mean(ages)) if len(ages) > 0 else float("nan")
            std_age = float(np.std(ages)) if len(ages) > 0 else float("nan")
        else:
            mean_age = std_age = float("nan")
        
        # Get gender statistics
        gender_col = None
        for col in data.columns:
            if 'gender' in col.lower():
                gender_col = col
                break
        
        if gender_col:
            genders = data[gender_col].value_counts().to_dict()
            # Map numeric codes to readable names if needed
            female_count = genders.get(2, 0) + genders.get('female', 0)  # 2 often represents female
            male_count = genders.get(1, 0) + genders.get('male', 0)      # 1 often represents male
        else:
            female_count = male_count = 0
        
        # Count treatment types based on available columns
        surgery_count = 0
        chemo_count = 0
        radio_count = 0
        
        # Look for treatment-related columns
        for col in data.columns:
            if 'surgery' in col.lower() or 'operation' in col.lower():
                surgery_count += data[col].notna().sum()
            elif 'chemo' in col.lower() or 'chemotherapy' in col.lower():
                chemo_count += data[col].notna().sum()
            elif 'radio' in col.lower() or 'radiation' in col.lower():
                radio_count += data[col].notna().sum()
        
        # Follow-up times
        follow_up_col = None
        for col in data.columns:
            if 'follow_up' in col.lower() and 'days' in col.lower():
                follow_up_col = col
                break
        
        if follow_up_col:
            follow_up_times = data[follow_up_col].dropna()
            mean_follow_up = float(np.mean(follow_up_times)) if len(follow_up_times) > 0 else float("nan")
        else:
            mean_follow_up = float("nan")
        
        # Survival status
        status_col = None
        for col in data.columns:
            if 'status' in col.lower():
                status_col = col
                break
        
        if status_col:
            status_counts = data[status_col].value_counts().to_dict()
            status_alive = status_counts.get('NED', 0) + status_counts.get(3, 0)  # 3 often represents alive
            status_dod = status_counts.get('DOD', 0) + status_counts.get(1, 0)      # 1 often represents DOD
            status_dod_other = status_counts.get('AWD', 0) + status_counts.get(2, 0)  # 2 often represents DOD_other
            status_unknown = status_counts.get('unknown', 0)
        else:
            status_alive = status_dod = status_dod_other = status_unknown = 0
        
        return {
            "n_patients": n_patients,
            "mean_age": round(mean_age, 1) if not np.isnan(mean_age) else "N/A",
            "std_age": round(std_age, 1) if not np.isnan(std_age) else "N/A",
            "surgery": surgery_count,
            "chemotherapy": chemo_count,
            "radiotherapy": radio_count,
            "female": female_count,
            "male": male_count,
            "mean_follow_up_days": round(mean_follow_up, 1) if not np.isnan(mean_follow_up) else "N/A",
            "status_no_evidence": status_alive,
            "status_dod": status_dod,
            "status_alive_with_disease": status_dod_other,
            "status_unknown": status_unknown,
        }

    train_stats = summarize(train_set)
    val_stats = summarize(val_set)
    summary_df = pd.DataFrame({"Training": train_stats, "Validation": val_stats})
    print("\nDataset Characteristics:")
    print(tabulate(summary_df, headers="keys", tablefmt="grid"))


def check_sarculator_consistency(df: pd.DataFrame) -> None:
    """Report patients whose sarculator scores conflict with survival status and provide accuracy statistics based on time intervals."""
    from datetime import datetime
    
    inconsistent = []
    
    # Find sarculator columns
    sarculator_five_col = None
    sarculator_ten_col = None
    status_col = None
    date_diagnosis_col = None
    date_follow_up_col = None
    
    for col in df.columns:
        if 'sarculator_five' in col.lower():
            sarculator_five_col = col
        elif 'sarculator_ten' in col.lower():
            sarculator_ten_col = col
        elif 'status' in col.lower():
            status_col = col
        elif 'date_of_diagnosis' in col.lower():
            date_diagnosis_col = col
        elif 'date_follow_up' in col.lower():
            date_follow_up_col = col
    
    if sarculator_five_col is None and sarculator_ten_col is None:
        print("\nNo sarculator columns found in the dataset.")
        return
    
    if status_col is None:
        print("\nNo status column found in the dataset.")
        return
    
    if date_diagnosis_col is None or date_follow_up_col is None:
        print("\nMissing date columns. Required: date_of_diagnosis and date_follow_up")
        return
    
    # Initialize counters for accuracy calculation
    s5_stats = {'correct': 0, 'incorrect': 0, 'total': 0, 'excluded': 0}
    s10_stats = {'correct': 0, 'incorrect': 0, 'total': 0, 'excluded': 0}
    
    for _, row in df.iterrows():
        patient_id = row.get("_id")
        s5 = row.get(sarculator_five_col) if sarculator_five_col else None
        s10 = row.get(sarculator_ten_col) if sarculator_ten_col else None
        status = row.get(status_col)
        date_diagnosis = row.get(date_diagnosis_col)
        date_follow_up = row.get(date_follow_up_col)
        
        if s5 is None and s10 is None:
            continue
            
        # Check if sarculator values are in valid range
        if s5 is not None and not (0 <= s5 <= 1):
            inconsistent.append(patient_id)
            continue
        if s10 is not None and not (0 <= s10 <= 1):
            inconsistent.append(patient_id)
            continue
        
        # Parse dates
        try:
            if isinstance(date_diagnosis, str):
                diagnosis_date = datetime.fromisoformat(date_diagnosis.replace('Z', '+00:00'))
            elif isinstance(date_diagnosis, dict) and '$date' in date_diagnosis:
                diagnosis_date = datetime.fromisoformat(date_diagnosis['$date'].replace('Z', '+00:00'))
            elif hasattr(date_diagnosis, "to_pydatetime"):  # handles pd.Timestamp and similar types
                diagnosis_date = date_diagnosis.to_pydatetime()
            else:
                continue

            if isinstance(date_follow_up, str):
                follow_up_date = datetime.fromisoformat(date_follow_up.replace('Z', '+00:00'))
            elif isinstance(date_follow_up, dict) and '$date' in date_follow_up:
                follow_up_date = datetime.fromisoformat(date_follow_up['$date'].replace('Z', '+00:00'))
            elif hasattr(date_follow_up, "to_pydatetime"):  # handles pd.Timestamp and similar types
                follow_up_date = date_follow_up.to_pydatetime()
            else:
                continue
        except (ValueError, TypeError):
            continue
        # Calculate time difference in years
        time_diff_years = (follow_up_date - diagnosis_date).days / 365.25
        
        # For patients who are not DOD, check if enough time has passed
        if status != "DOD":
            current_date = datetime.now()
            time_from_diagnosis_to_now = (current_date - diagnosis_date).days / 365.25
            
            # Only evaluate if enough time has passed for the respective sarculator predictions
            if s5 is not None and time_from_diagnosis_to_now >= 5:
                s5_stats['total'] += 1
                # High sarculator score (>0.5) predicts survival
                predicted_survival = s5 > 0.5
                # Patient is actually alive and 5+ years have passed
                actual_survival = True
                
                if predicted_survival == actual_survival:
                    s5_stats['correct'] += 1
                else:
                    s5_stats['incorrect'] += 1
            elif s5 is not None:
                s5_stats['excluded'] += 1
            
            if s10 is not None and time_from_diagnosis_to_now >= 10:
                s10_stats['total'] += 1
                # High sarculator score (>0.5) predicts survival
                predicted_survival = s10 > 0.5
                # Patient is actually alive and 10+ years have passed
                actual_survival = True
                
                if predicted_survival == actual_survival:
                    s10_stats['correct'] += 1
                else:
                    s10_stats['incorrect'] += 1
            elif s10 is not None:
                s10_stats['excluded'] += 1
        
        # For patients who died (DOD), check if they died within the prediction timeframe
        else:  # status == "DOD"
            if s5 is not None:
                s5_stats['total'] += 1
                # High sarculator score (>0.5) predicts survival
                predicted_survival = s5 > 0.5
                # Patient died within 5 years if time_diff_years < 5
                actual_survival = time_diff_years >= 5
                
                if predicted_survival == actual_survival:
                    s5_stats['correct'] += 1
                else:
                    s5_stats['incorrect'] += 1
                    # This is an inconsistency if high score with death within 5 years
                    if predicted_survival and not actual_survival:
                        inconsistent.append(patient_id)
            
            if s10 is not None:
                s10_stats['total'] += 1
                # High sarculator score (>0.5) predicts survival
                predicted_survival = s10 > 0.5
                # Patient died within 10 years if time_diff_years < 10
                actual_survival = time_diff_years >= 10
                
                if predicted_survival == actual_survival:
                    s10_stats['correct'] += 1
                else:
                    s10_stats['incorrect'] += 1
                    # This is an inconsistency if high score with death within 10 years
                    if predicted_survival and not actual_survival:
                        if patient_id not in inconsistent:  # Avoid duplicate entries
                            inconsistent.append(patient_id)

    # Print statistics
    print("\n=== Sarculator Performance Analysis (Time-based) ===")
    
    if s5_stats['total'] > 0:
        s5_accuracy = s5_stats['correct'] / s5_stats['total'] * 100
        print(f"Sarculator 5-year predictions:")
        print(f"  - Total patients evaluated: {s5_stats['total']}")
        print(f"  - Correct predictions: {s5_stats['correct']}")
        print(f"  - Incorrect predictions: {s5_stats['incorrect']}")
        print(f"  - Accuracy: {s5_accuracy:.1f}%")
        if s5_stats['excluded'] > 0:
            print(f"  - Excluded (insufficient follow-up): {s5_stats['excluded']}")
    
    if s10_stats['total'] > 0:
        s10_accuracy = s10_stats['correct'] / s10_stats['total'] * 100
        print(f"Sarculator 10-year predictions:")
        print(f"  - Total patients evaluated: {s10_stats['total']}")
        print(f"  - Correct predictions: {s10_stats['correct']}")
        print(f"  - Incorrect predictions: {s10_stats['incorrect']}")
        print(f"  - Accuracy: {s10_accuracy:.1f}%")
        if s10_stats['excluded'] > 0:
            print(f"  - Excluded (insufficient follow-up): {s10_stats['excluded']}")

    if inconsistent:
        print(f"\nWarning: {len(inconsistent)} patients with inconsistent sarculator values (showing up to 10): {inconsistent[:10]}")
    else:
        print("\nSarculator values are compatible with patient status for all patients.")

def show_applied_treatments(decoded: List[pd.DataFrame], actual_treat: torch.Tensor) -> None:
    """Display decoded steps only for which a treatment was applied."""
    applied_mask = (actual_treat.sum(dim=1) > 0).cpu()
    for idx, (df_out, applied) in enumerate(zip(decoded, applied_mask)):
        if applied:
            print(f"\n--- Decoded step {idx} ---")
            print(df_out.to_string(index=False))

def compute_pointwise_metrics(real: List[pd.DataFrame], generated: List[pd.DataFrame]) -> Dict[str, Any]:
    """Compute MAE for numeric columns and accuracy/F1 for binary events."""
    metrics: Dict[str, Any] = {}
    if not real or not generated:
        return metrics

    num_cols = [c for c in real[0].columns if pd.api.types.is_numeric_dtype(real[0][c])]
    bin_cols = [c for c in real[0].columns if set(real[0][c].dropna().unique()).issubset({0, 1})]

    if num_cols:
        real_num = np.concatenate([df[num_cols].to_numpy() for df in real])
        gen_num = np.concatenate([df[num_cols].to_numpy() for df in generated])
        metrics["mae"] = mean_absolute_error(real_num, gen_num)

    if bin_cols:
        real_bin = np.concatenate([df[bin_cols].to_numpy() for df in real]).flatten()
        gen_bin = np.rint(np.concatenate([df[bin_cols].to_numpy() for df in generated])).astype(int).flatten()
        metrics["accuracy"] = accuracy_score(real_bin, gen_bin)
        metrics["f1"] = f1_score(real_bin, gen_bin, zero_division=0)

    return metrics


def plot_kaplan_meier(real_times: np.ndarray, real_events: np.ndarray,
                       gen_times: np.ndarray, gen_events: np.ndarray, save_path: Path = None) -> None:
    """Plot Kaplan-Meier curves for real and generated cohorts."""
    try:
        from lifelines import KaplanMeierFitter
        import matplotlib.pyplot as plt

        km_real = KaplanMeierFitter()
        km_gen = KaplanMeierFitter()
        km_real.fit(real_times, real_events, label="Real")
        km_gen.fit(gen_times, gen_events, label="Generated")
        ax = km_real.plot()
        km_gen.plot(ax=ax)
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival")
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    except ImportError:
        print("lifelines package not installed. Kaplan-Meier plot skipped.")
    except Exception as exc:  # pragma: no cover - plotting optional
        print(f"KM plot failed: {exc}")


def compute_survival_metrics(times: np.ndarray, events: np.ndarray, pred_times: np.ndarray) -> Dict[str, Any]:
    """Compute C-index and Brier score for survival predictions."""
    metrics: Dict[str, Any] = {}
    try:
        # Concordance index using standard definition
        n = len(times)
        assert len(pred_times) == n
        concordant = permissible = 0
        for i in range(n):
            for j in range(i + 1, n):
                if events[i] == events[j] == 0:
                    continue
                if times[i] == times[j]:
                    continue
                permissible += 1
                if (pred_times[i] < pred_times[j] and times[i] < times[j]) or (
                    pred_times[i] > pred_times[j] and times[i] > times[j]
                ):
                    concordant += 1
        metrics["c_index"] = concordant / permissible if permissible > 0 else float("nan")
    except Exception:
        metrics["c_index"] = float("nan")

    try:
        metrics["brier"] = brier_score_loss(events, pred_times)
    except Exception:
        metrics["brier"] = float("nan")
    return metrics

def survival_rate_at(threshold: float, times: np.ndarray, events: np.ndarray) -> float:
    """Return survival rate at a given time threshold."""
    survived = (times >= threshold) | (events == 0)
    return float(np.mean(survived))


def estimate_ate(treated: np.ndarray, control: np.ndarray) -> float:
    """Estimate average treatment effect as difference in mean survival time."""
    return float(np.mean(treated) - np.mean(control))


def validate_model(gx, meta: Dict[str, Any], extractor: MongoExtractor,
                   meta_clin: MetadataHandler, meta_treat: MetadataHandler,
                   decoder_treat, feat_spec, oh_map, ord_map, num_ranges,
                   device: torch.device) -> Dict[str, Any]:
    """Run validation over patient ids stored in metadata."""
    val_ids = meta.get("val_patient_ids")
    if not val_ids:
        print("No validation ids found in metadata")
        return {}

    real_decoded: List[pd.DataFrame] = []
    gen_decoded: List[pd.DataFrame] = []

    for pid in val_ids:
        df = extractor.get_patient_by_id(pid)
        if df.empty:
            continue
        dataset = TabularDatasetPID(df, meta_clin, meta_treat, seq_len=None)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        treat_real, clin_emb, actual_treat = next(iter(loader))
        clin_emb = clin_emb.to(device)
        actual_treat = actual_treat.to(device).float()
        treat_real = treat_real.to(device)
        with torch.no_grad():
            fake = gx(clin_emb, actual_treat)

        for t_real, t_fake in zip(treat_real[0], fake[0]):
            real_decoded.append(
                decode_embedding(decoder_treat(t_real), feat_spec, oh_map, ord_map, num_ranges)
            )
            gen_decoded.append(
                decode_embedding(decoder_treat(t_fake), feat_spec, oh_map, ord_map, num_ranges)
            )

    metrics = compute_pointwise_metrics(real_decoded, gen_decoded)
    return metrics



def plot_model_losses(meta: Dict[str, Any], save_path: Path = None) -> None:
    """Plot the cycle GAN loss curves from training."""
    loss_log_path = meta.get("loss_log_path")
    if loss_log_path and Path(loss_log_path).exists():
        print(f"\nPlotting loss curves from: {loss_log_path}")
        try:
            plot_train_cycle_gan(loss_log_path, save_path)
            print("Loss curves plotted successfully")
        except Exception as e:
            print(f"Error plotting loss curves: {e}")
    else:
        print("Loss log file not found or not specified in metadata")

def main():
    # For a Jupyter Notebook, preset parameters instead of using argparse
    parser = argparse.ArgumentParser(description="Temporal Inference Script Arguments")
    parser.add_argument(
        "--meta_file", type=str,
        default="saved_models/temporal_cycle_gan_20250703_085434/metadata.json",
        help="Path to the metadata file for the temporal cycle gan."
    )
    parser.add_argument(
        "--config_file", type=str,
        default="Models/configs/mock_config_cycle_gan.yaml",
        help="Path to the Mongo configuration file."
    )
    parser.add_argument(
        "--treatment", type=str,
        default="1,0,0",
        help="Treatment vector as comma-separated values (surgery,chemo,radio)."
    )
    parser.add_argument(
        "--show_only_applied", action="store_true",
        help="Show only treatment steps where treatment was applied."
    )
    parser.add_argument(
        "--save_loss_plot", type=str,
        default="loss_plot.png",
        help="Path to save the loss plot image."
    )

    args = parser.parse_args()

    meta_file = args.meta_file
    config_file = args.config_file
    treat_vec = parse_treatment(args.treatment)
    show_only_applied = args.show_only_applied
    save_loss_plot = args.save_loss_plot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and metadata
    gx, meta = load_temporal_cycle_gan(Path(meta_file), device)
    print(f"Loaded model from: {meta_file}")

    # Plot loss curves
    plot_model_losses(meta, Path(save_loss_plot) if save_loss_plot else None)

    meta_clin = MetadataHandler(meta["metadata_clinical_file"])
    meta_treat = MetadataHandler(meta["metadata_treatment_file"])

    # Load data
    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=config_file,
    )
    df = extractor.get_dataframe()
    
    # Print dataset characteristics
    print_dataset_characteristics(df, meta.get("val_patient_ids", []))
    
    # Check sarculator consistency
    check_sarculator_consistency(df)
    
    # Load decoder
    decoder_treat, feat_spec, oh_map, ord_map, num_ranges = load_decoder(
        meta["metadata_treatment_file"], meta["decoder_treatment"], device
    )

    # Create dataset and dataloader
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
    
    # Generate with actual treatments
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
    
    # Generate with specified treatment
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
    
    print(f"\n=== Results for treatment: {args.treatment} ===")
    for idx, df_out in enumerate(decoded):
        print(f"\n--- Decoded step {idx} ---")
        print(df_out.to_string(index=False))

    # Generate next treatment step
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

    print("\n=== Patient's Actual Treatment Sequence ===")
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

    # Run validation
    print("\n=== Model Validation ===")
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
        print(f"{k}: {v:.4f}")



if __name__ == "__main__":
    main()