import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
import pandas as pd
from Models.models.GANs import LSTMGenerator
from Models.models.decoder import Decoder
from Models.models.utils import MetadataHandler
from Models.utils.helpers import MongoExtractor
from Inference.refined_evidence import filter_ajcc_stage_iii
import argparse
from Inference.refined_evidence_package import EvidencePackageGenerator, EvidenceComponents
from Models.dataset.tabular_dataset import TabularDatasetPID
from torch.utils.data import DataLoader
from Models.models.decoder_embedders import decode_embedding

def calculate_output_dim(feature_spec, one_hot_mappings, ord_mappings):
    """Calculate the correct output dimension from feature specifications."""
    output_dim = 0
    
    for col, spec in feature_spec.items():
        if spec == "numeric":
            output_dim += 1
        elif spec == "one_hot":
            if col in one_hot_mappings:
                output_dim += len(one_hot_mappings[col])
            else:
                # Fallback - assume binary if no mapping found
                output_dim += 1
        elif isinstance(spec, dict) or spec == "ordinal":
            # Ordinal feature
            if isinstance(spec, dict):
                output_dim += 1  # Single ordinal value
            elif col in ord_mappings:
                output_dim += 1  # Single ordinal value
            else:
                output_dim += 1  # Fallback
    
    return output_dim

def load_decoder(meta_file: str, decoder_path: str, device: torch.device):
    """Load decoder using parameters stored in metadata and calculate correct output dimension."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta.get("model_params", {})
    hidden_dim = params.get("hidden_dim")
    
    # Get feature specifications
    feature_spec = meta.get("feature_spec")
    one_hot_mappings = meta.get("one_hot_mappings", {})
    ord_mappings = meta.get("ord_mappings", {})
    numeric_ranges = meta.get("numeric_ranges", {})
    
    # Calculate the correct output dimension from feature specifications
    output_dim = calculate_output_dim(feature_spec, one_hot_mappings, ord_mappings)
    
    # Create decoder with correct dimensions
    dec = Decoder(hidden_dim, output_dim).to(device)
    dec.load_state_dict(torch.load(decoder_path, map_location=device))
    dec.eval()
    
    return dec, feature_spec, one_hot_mappings, ord_mappings, numeric_ranges

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

def initialize_models(meta_file: Path, device: torch.device):
    """Load generator, decoder, and metadata handlers - same as training script."""
    gx, meta = load_temporal_cycle_gan(meta_file, device)
    
    # Load metadata handlers exactly like in training
    meta_clin = MetadataHandler(meta["metadata_clinical_file"])
    meta_treat = MetadataHandler(meta["metadata_treatment_file"])
    
    # Load decoders
    decoder_treat, feat_spec_treat, oh_map_treat, ord_map_treat, num_ranges_treat = load_decoder(
        meta["metadata_treatment_file"], meta["decoder_treatment"], device
    )
    decoder_clin, feat_spec_clin, oh_map_clin, ord_map_clin, num_ranges_clin = load_decoder(
        meta["metadata_clinical_file"], meta["decoder_clinical"], device
    )
    return (
        gx,
        meta_clin,
        meta_treat,
        decoder_treat,
        decoder_clin,
        feat_spec_treat,
        feat_spec_clin,
        oh_map_treat,
        oh_map_clin,
        ord_map_treat,
        ord_map_clin,
        num_ranges_treat,
        num_ranges_clin,
    )

def create_single_patient_dataset(patient_id: int, config_file: str, meta_clin, meta_treat, seq_len: int = 5):
    """Create dataset for single patient using same approach as training script."""
    
    # Create extractors exactly like in temporal_cycle_gan.py
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Clinical data extractor
    extractor_clin = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=None,
        custom_config=config["clinical_data_config"],
    )
    
    # Treatment data extractor
    extractor_treat = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=None,
        custom_config=config["treatment_data_config"],
    )
    
    # Actual treatment extractor (for conditioning)
    actual_treatment_config = {
        "aggregator": [
            {"$match": {
                "tumor_characteristics.histological_diagnosis": {
                    "$in": [
                        "Angiosarcoma of soft tissues",
                        "Atypical lipomatous tumor",
                        "Dedifferentiated liposarcoma",
                        "Desmoid‐type fibromatosis",
                        "GIST",
                        "Intramuscular myxoma",
                        "Leiomyosarcoma",
                        "Myxofibrosarcoma",
                        "Myxoid liposarcoma",
                        "Pleomorphic liposarcoma",
                        "Solitary fibrous tumor",
                        "Synovial sarcoma",
                        "Tenosynovial giant cell tumor",
                        "Undifferentiated / unclassified sarcoma"
                    ]
                }
            }},
            {"$unwind": {"path": "$episodes", "preserveNullAndEmptyArrays": True}},
            {"$project": {
                "_id": 1,
                "episodes.surgery": {"$ifNull": ["$episodes.surgery", 0]},
                "episodes.chemotherapy": {"$ifNull": ["$episodes.chemotherapy", 0]},
                "episodes.radiotherapy": {"$ifNull": ["$episodes.radiotherapy", 0]}
            }}
        ]
    }
    
    extractor_actual_treat = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        custom_config=actual_treatment_config
    )
    
    # Get dataframes
    df_clin = extractor_clin.get_dataframe()
    df_treat = extractor_treat.get_dataframe()
    df_actual_treat = extractor_actual_treat.get_dataframe()
    
    # CRITICAL FIX: Reorder DataFrame columns to match feature_spec order
    # MongoDB returns columns in alphabetical order, but our encoding/decoding
    # logic assumes they're in the same order as the YAML feature specification
    df_clin = MongoExtractor.reorder_dataframe_columns(df_clin, config["clinical_data_config"]["features"])
    df_treat = MongoExtractor.reorder_dataframe_columns(df_treat, config["treatment_data_config"]["features"])
    
    # Filter for the specific patient
    df_clin = df_clin[df_clin['_id'] == patient_id].copy()
    df_treat = df_treat[df_treat['_id'] == patient_id].copy()
    df_actual_treat = df_actual_treat[df_actual_treat['_id'] == patient_id].copy()
    
    if df_clin.empty:
        raise ValueError(f"No clinical data found for patient {patient_id}")
    if df_treat.empty:
        raise ValueError(f"No treatment data found for patient {patient_id}")
    if df_actual_treat.empty:
        raise ValueError(f"No actual treatment data found for patient {patient_id}")
    
    # Reset indices exactly like in temporal_cycle_gan.py
    df_clin.reset_index(inplace=True)
    df_treat.reset_index(inplace=True)
    df_actual_treat.reset_index(inplace=True)
    
    # Create dataset exactly like in training
    dataset = TabularDatasetPID(
        df_clin=df_clin,
        df_treat=df_treat,
        df_actual_treat=df_actual_treat,
        meta_clin=meta_clin,
        meta_treat=meta_treat,
        seq_len=seq_len,
        id_field="_id",
        return_mask=True,
    )
    
    return dataset, df_clin, df_treat, df_actual_treat

def main():
    patient_id = 12586
    meta_file = "saved_models/temporal_cycle_gan_20250802_165825/metadata.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config_file = "Models/configs/temporal_inference.yaml"
    
    # Initialize models exactly like training script
    (
        gx,
        meta_clin,
        meta_treat,
        decoder_treat,
        decoder_clin,
        feat_spec_treat,
        feat_spec_clin,
        oh_map_treat,
        oh_map_clin,
        ord_map_treat,
        ord_map_clin,
        num_ranges_treat,
        num_ranges_clin,
    ) = initialize_models(Path(meta_file), device)

    # Create dataset for single patient
    dataset, df_clin, df_treat, df_actual_treat = create_single_patient_dataset(
        patient_id, config_file, meta_clin, meta_treat, seq_len=5
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Dataset created successfully")
    print(f"Clinical data shape: {df_clin.shape}")
    print(f"Treatment data shape: {df_treat.shape}")
    print(f"Actual treatment data shape: {df_actual_treat.shape}")
    print(f"Clinical feature spec: {list(feat_spec_clin.keys())}")
    print(f"Treatment feature spec: {list(feat_spec_treat.keys())}")
    
    # Generate counterfactual scenarios
    scenarios = {
        "S": [1, 0, 0],
        "S_CT": [1, 1, 0], 
        "S_RT": [1, 0, 1],
        "S_RT_CT": [1, 1, 1],
    }
    
    # Get embeddings exactly like in training loop
    embeddings = []
    lengths_list = []
    
    for batch in loader:
        # Extract data from batch dictionary exactly like temporal_cycle_gan.py
        tr_real = batch["x_treat"]
        cl_real = batch["x_clin"] 
        tr_actual = batch["actual_treatment"]
        tr_mask = batch["mask_treat"]
        cl_mask = batch["mask_clin"]
        
        cl_real = cl_real.to(device)
        tr_real = tr_real.to(device)
        tr_actual = tr_actual.to(device)
        tr_mask = tr_mask.to(device)
        cl_mask = cl_mask.to(device)
        
        # Create per-timestep mask exactly like temporal_cycle_gan.py
        step_mask = (
            (tr_mask.sum(dim=2, keepdim=True) + cl_mask.sum(dim=2, keepdim=True))
            > 0
        ).float()
        
        # Apply masking to embeddings
        cl_real = cl_real * step_mask
        tr_real = tr_real * step_mask
        
        # Compute sequence lengths for LSTM
        lengths = step_mask.squeeze(-1).sum(dim=1).long()
        
        embeddings.append(cl_real)
        lengths_list.append(lengths)
    
    clin_emb_all = torch.cat(embeddings, dim=0)
    lengths_all = torch.cat(lengths_list, dim=0)
    
    print(f"Clinical embeddings shape: {clin_emb_all.shape}")
    print(f"Sequence lengths: {lengths_all}")
    
    # Generate counterfactuals and decode both clinical and treatment for comparison
    generated = []
    
    for name, t_vec in scenarios.items():
        # Create treatment conditioning tensor exactly like training
        treat_tensor = torch.zeros(
            (clin_emb_all.shape[0], clin_emb_all.shape[1], 3),
            dtype=torch.float32,
            device=device,
        )
        treat_tensor[:, :, :] = torch.tensor(t_vec, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            # Generate counterfactual exactly like training (using Gx with lengths)
            fake = gx(clin_emb_all, treat_tensor, lengths=lengths_all)
            
        # Decode both clinical and treatment embeddings (first timestep)
        fake_first_step = fake[0, 0].unsqueeze(0)  # Keep batch dimension
        clin_first_step = clin_emb_all[0, 0].unsqueeze(0)  # Original clinical embedding
        
        # Decode treatment embedding
        decoded_treatment = decode_embedding(
            decoder_treat(fake_first_step),
            feat_spec_treat,
            oh_map_treat,
            ord_map_treat,
            num_ranges_treat,
            one_hot_argmax=True
        )
        
        # Decode clinical embedding for comparison
        decoded_clinical = decode_embedding(
            decoder_clin(clin_first_step),
            feat_spec_clin,
            oh_map_clin,
            ord_map_clin,
            num_ranges_clin,
            one_hot_argmax=True
        )
        
        # Get original data for comparison
        patient_first_row = df_clin.iloc[0]
        record = {}
        
        # Add both original and decoded clinical features for comparison
        for col in feat_spec_clin.keys():
            # Original clinical data
            if col in df_clin.columns:
                record[f"clinical_{col}_original"] = patient_first_row[col]
            else:
                record[f"clinical_{col}_original"] = None
            
            # Decoded clinical data
            if col in decoded_clinical.columns:
                record[f"clinical_{col}_decoded"] = decoded_clinical.iloc[0][col]
            else:
                record[f"clinical_{col}_decoded"] = None
        
        # Add decoded treatment features
        for col in feat_spec_treat.keys():
            if col in decoded_treatment.columns:
                record[f"treatment_{col}"] = decoded_treatment.iloc[0][col]
            else:
                record[f"treatment_{col}"] = None
        
        # Add metadata
        record.update({
            "_id": patient_id,
            "treatment_scenario": name,
            "surgery": t_vec[0],
            "chemotherapy": t_vec[1],
            "radiotherapy": t_vec[2],
        })
        generated.append(record)
        
        # Print comparison for this scenario
        print(f"\n--- Scenario: {name} ---")
        print("Clinical feature comparison (original vs decoded):")
        for col in feat_spec_clin.keys():
            if col in df_clin.columns and col in decoded_clinical.columns:
                orig_val = patient_first_row[col]
                decoded_val = decoded_clinical.iloc[0][col]
                match = "✓" if orig_val == decoded_val else "✗"
                print(f"  {col}: {orig_val} vs {decoded_val} {match}")
        
        print("Treatment features:")
        for col in feat_spec_treat.keys():
            if col in decoded_treatment.columns:
                print(f"  {col}: {decoded_treatment.iloc[0][col]}")
        print(f"  Expected treatment vector: {t_vec}")
    
    # Save results
    results_df = pd.DataFrame(generated)
    results_df.to_csv(f"counterfactual_scenarios_patient_{patient_id}.csv", index=False)
    print(f"Generated {len(results_df)} counterfactual scenarios for patient {patient_id}")
    print("Results saved to CSV file")

if __name__ == "__main__":
    main()