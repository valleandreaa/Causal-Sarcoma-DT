import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import pandas as pd
import numpy as np
from Models.models.GANs import LSTMGenerator
from Models.utils.helpers import MongoExtractor
import argparse

class TabularDataset:
    """Exact same TabularDataset as in temporal_cycle_gan_no_embed.py - must use saved metadata for consistency."""
    
    def __init__(self, df: pd.DataFrame, feature_spec: Dict[str, str], saved_metadata: Dict = None):
        self.spec = feature_spec
        self.num_cols = [c for c, t in self.spec.items() if t == "numeric"]
        self.oh_cols = [c for c, t in self.spec.items() if t == "one_hot"]
        self.ord_cols = [c for c, t in self.spec.items() if t == "ordinal" or isinstance(t, dict)]

        # Extract ordinal mappings directly from feature_spec (YAML config)
        self.ord_maps = self._build_ordinal_maps_from_spec()

        # Use saved metadata if provided, otherwise build fresh (for training consistency)
        if saved_metadata:
            self.one_hot_mappings = saved_metadata.get("one_hot_mappings", {})
            self.numeric_ranges = saved_metadata.get("numeric_ranges", {})
            # Ordinal mappings come from YAML config, not saved metadata
        else:
            # Build one_hot mappings for dummy conversion (fresh - only for training)
            self.one_hot_mappings: Dict[str, list[str]] = {}
            
            # Store numeric ranges for later reconstruction (fresh - only for training)
            self.numeric_ranges: Dict[str, Tuple[float, float]] = {
                col: (float(df[col].min()), float(df[col].max())) for col in self.num_cols
            }
        
        # Pre‑process
        self.x_features, self.x_ord, self.mask, self.feature_dim = self._preprocess(df)

    def _build_ordinal_maps_from_spec(self):
        """Build ordinal mappings directly from the feature specification (YAML config)."""
        ord_maps = []
        for col in self.ord_cols:
            spec_val = self.spec[col]
            if isinstance(spec_val, dict):
                # Use the mapping directly from YAML config - values are the ordinal integers
                shifted = {v: i+1 for i, v in enumerate(sorted(spec_val.keys()))}
                ord_maps.append(shifted)
            else:
                # For 'ordinal' string spec, we would need to infer from data, but this should be avoided in inference
                raise ValueError(f"Ordinal column '{col}' must have explicit mapping in YAML config for inference consistency")
        return ord_maps

    def _encode_ord(self, s: pd.Series, mp: Dict[str, int]) -> np.ndarray:
        return s.map(mp).fillna(0).astype("int64").to_numpy()

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # numeric -------------------------------------------------------------
        x_num = df[self.num_cols].to_numpy(dtype="float32", copy=True) if self.num_cols else np.empty((len(df), 0), dtype="float32")
        num_mask = ~np.isnan(x_num) if self.num_cols else np.empty_like(x_num, dtype=bool)
        
        # Normalize numeric features using saved ranges (CRITICAL for inference consistency)
        for i, col in enumerate(self.num_cols):
            if col in self.numeric_ranges:
                col_min, col_max = self.numeric_ranges[col]
                if col_max > col_min:
                    x_num[:, i] = (x_num[:, i] - col_min) / (col_max - col_min)
        
        x_num[np.isnan(x_num)] = 0.0

        # one‑hot -------------------------------------------------------------
        if self.oh_cols:
            x_oh_parts, mask_parts = [], []
            for col in self.oh_cols:
                # Use saved one-hot mappings for consistency (CRITICAL)
                if col in self.one_hot_mappings:
                    categories = self.one_hot_mappings[col]
                    # Create categorical dtype to handle unseen categories
                    cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=False)
                    try:
                        dummies = pd.get_dummies(df[col].astype(cat_dtype), prefix=col).astype("float32")
                    except:
                        # Fallback: create zero matrix with correct columns
                        dummy_cols = [f"{col}_{cat}" for cat in categories]
                        dummies = pd.DataFrame(0.0, index=df.index, columns=dummy_cols, dtype="float32")
                        # Fill in actual values where present
                        for idx, val in df[col].items():
                            if pd.notna(val) and f"{col}_{val}" in dummy_cols:
                                dummies.loc[idx, f"{col}_{val}"] = 1.0
                else:
                    # Fresh processing (only during training)
                    dummies = pd.get_dummies(df[col], dummy_na=False, prefix=col).astype("float32")
                    self.one_hot_mappings[col] = [col_name.split('_', 1)[1] for col_name in dummies.columns]
                
                x_oh_parts.append(dummies.to_numpy())
                col_mask = (~pd.isna(df[col])).to_numpy(dtype="float32")[:, None]
                mask_parts.append(np.repeat(col_mask, dummies.shape[1], axis=1))
            x_oh = np.concatenate(x_oh_parts, axis=1)
            oh_mask = np.concatenate(mask_parts, axis=1)
            x_features = np.concatenate([x_num, x_oh], axis=1)
            mask_features = np.concatenate([num_mask, oh_mask], axis=1).astype("float32")
        else:
            x_features = x_num
            mask_features = num_mask.astype("float32")
        
        # ordinal -------------------------------------------------------------
        if self.ord_cols:
            ord_arrays = []
            ord_mask_arrays = []
            for col, mp in zip(self.ord_cols, self.ord_maps):
                # Create mask based on original missing values, not encoded values
                orig_mask = (~pd.isna(df[col])).to_numpy().astype("float32")
                ord_mask_arrays.append(orig_mask[:, None])
                
                # Encode ordinals (missing values become 0)
                encoded = self._encode_ord(df[col], mp)
                ord_arrays.append(encoded[:, None])
            
            x_ord = np.concatenate(ord_arrays, axis=1)
            mask_ord = np.concatenate(ord_mask_arrays, axis=1)
            
            # Add ordinals as features (converted to float)
            x_features = np.concatenate([x_features, x_ord.astype("float32")], axis=1)
            mask_features = np.concatenate([mask_features, mask_ord], axis=1)
            x_ord_tensor = torch.from_numpy(x_ord).long()
        else:
            x_ord_tensor = None

        feature_dim = x_features.shape[1]
        
        return (
            torch.from_numpy(x_features).float(),
            x_ord_tensor,
            torch.from_numpy(mask_features).float(),
            feature_dim,
        )

class TemporalTabularDatasetPID:
    """Temporal dataset that uses saved metadata for exact preprocessing consistency."""
    
    def __init__(
        self,
        df_clin: pd.DataFrame,
        df_treat: pd.DataFrame,
        df_actual_treat: pd.DataFrame,
        feature_spec_clin: Dict[str, str],
        feature_spec_treat: Dict[str, str],
        seq_len: int,
        id_field: str,
        return_mask: bool = True,
        saved_metadata: Dict = None,
    ):
        self.seq_len = seq_len
        self.id_field = id_field
        self.return_mask = return_mask
        
        # Extract metadata for each domain
        clinical_metadata = {}
        treatment_metadata = {}
        
        if saved_metadata:
            clinical_metadata = {
                "one_hot_mappings": saved_metadata.get("clinical_one_hot_mappings", {}),
                "numeric_ranges": saved_metadata.get("clinical_numeric_ranges", {}),
                "ord_mappings": saved_metadata.get("clinical_ord_mappings", []),
            }
            treatment_metadata = {
                "one_hot_mappings": saved_metadata.get("treatment_one_hot_mappings", {}),
                "numeric_ranges": saved_metadata.get("treatment_numeric_ranges", {}),
                "ord_mappings": saved_metadata.get("treatment_ord_mappings", []),
            }
        
        # Create TabularDatasets for preprocessing with saved metadata
        self.clinical_dataset = TabularDataset(df_clin, feature_spec_clin, clinical_metadata)
        self.treatment_dataset = TabularDataset(df_treat, feature_spec_treat, treatment_metadata)
        
        # Store feature dimensions
        self.clin_feature_dim = self.clinical_dataset.feature_dim
        self.treat_feature_dim = self.treatment_dataset.feature_dim
        
        # Process actual treatment data (simple numeric)
        actual_treat_cols = ["episodes.surgery", "episodes.chemotherapy", "episodes.radiotherapy"]
        self.actual_treatment = df_actual_treat[actual_treat_cols].fillna(0).to_numpy(dtype="float32")
        
        # Group rows by patient id
        groups = df_clin.groupby(id_field).indices
        self.patient_ids = list(groups.keys())
        self.group_indices = [np.asarray(idx_list) for idx_list in groups.values()]
        self.num_patients = len(self.group_indices)

    def get_patient_data(self, patient_idx=0):
        """Get data for a specific patient (usually 0 for single patient dataset)."""
        indices = self.group_indices[patient_idx]
        
        # Get sequences for this patient
        x_clin = self.clinical_dataset.x_features[indices]
        x_treat = self.treatment_dataset.x_features[indices]
        mask_clin = self.clinical_dataset.mask[indices]
        mask_treat = self.treatment_dataset.mask[indices]
        actual_treat = self.actual_treatment[indices]
        
        # Pad or truncate to seq_len
        seq_actual_len = len(indices)
        if seq_actual_len < self.seq_len:
            # Pad with zeros
            pad_len = self.seq_len - seq_actual_len
            x_clin = torch.cat([x_clin, torch.zeros(pad_len, self.clin_feature_dim)], dim=0)
            x_treat = torch.cat([x_treat, torch.zeros(pad_len, self.treat_feature_dim)], dim=0)
            mask_clin = torch.cat([mask_clin, torch.zeros(pad_len, self.clin_feature_dim)], dim=0)
            mask_treat = torch.cat([mask_treat, torch.zeros(pad_len, self.treat_feature_dim)], dim=0)
            actual_treat = np.pad(actual_treat, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
        else:
            # Truncate to seq_len
            x_clin = x_clin[:self.seq_len]
            x_treat = x_treat[:self.seq_len]
            mask_clin = mask_clin[:self.seq_len]
            mask_treat = mask_treat[:self.seq_len]
            actual_treat = actual_treat[:self.seq_len]
        
        return {
            "x_clin": x_clin.unsqueeze(0),  # Add batch dimension
            "x_treat": x_treat.unsqueeze(0),
            "actual_treatment": torch.from_numpy(actual_treat).float().unsqueeze(0),
            "mask_clin": mask_clin.unsqueeze(0),
            "mask_treat": mask_treat.unsqueeze(0),
        }

def load_temporal_cycle_gan_no_embed(meta_path: Path, device: torch.device):
    """Load Gx generator from saved no-embed metadata."""
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    params = meta.get("model_params", {})
    clin_dim = params.get("clin_dim")
    treat_dim = params.get("treat_dim")
    hidden_dim = params.get("hidden_dim", 128)
    num_layers = params.get("num_layers", 1)

    gx = LSTMGenerator(
        input_dim=clin_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=treat_dim,
        num_layers=num_layers,
    ).to(device)
    gx.load_state_dict(torch.load(meta["Gx_path"], map_location=device))
    gx.eval()
    return gx, meta

def create_single_patient_dataset_no_embed(patient_id: int, config_file: str, seq_len: int = 5, saved_metadata: Dict = None):
    """Create dataset for single patient using raw features approach."""
    
    # Load config
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
        "features": {
            "episodes.surgery": "numeric",
            "episodes.chemotherapy": "numeric", 
            "episodes.radiotherapy": "numeric"
        },
        "aggregator": [
            {"$match": {
                "tumor_characteristics.histological_diagnosis": {
                    "$in": [
                        "Angiosarcoma of soft tissues", "Atypical lipomatous tumor",
                        "Dedifferentiated liposarcoma", "Desmoid‐type fibromatosis",
                        "GIST", "Intramuscular myxoma", "Leiomyosarcoma",
                        "Myxofibrosarcoma", "Myxoid liposarcoma", "Pleomorphic liposarcoma",
                        "Solitary fibrous tumor", "Synovial sarcoma",
                        "Tenosynovial giant cell tumor", "Undifferentiated / unclassified sarcoma"
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
    
    # Reorder columns to match feature_spec order
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
    
    # Reset indices
    df_clin.reset_index(inplace=True)
    df_treat.reset_index(inplace=True)
    df_actual_treat.reset_index(inplace=True)
    
    # Create dataset with saved metadata for exact preprocessing consistency
    dataset = TemporalTabularDatasetPID(
        df_clin=df_clin,
        df_treat=df_treat,
        df_actual_treat=df_actual_treat,
        feature_spec_clin=config["clinical_data_config"]["features"],
        feature_spec_treat=config["treatment_data_config"]["features"],
        seq_len=seq_len,
        id_field="_id",
        return_mask=True,
        saved_metadata=saved_metadata,
    )
    
    return dataset, df_clin, df_treat, df_actual_treat

def decode_raw_features_to_df(feature_vector, dataset, mask_vector=None, prefix="", return_all=False):
    """Convert raw feature vector back to interpretable DataFrame format.
    
    Args:
        feature_vector: Raw feature vector to decode
        dataset: TabularDataset object with metadata
        mask_vector: Optional mask vector to identify missing values
        prefix: Prefix to add to column names
        return_all: If True, return all features including probabilities and raw values
    """
    # Ensure inputs are numpy arrays
    if hasattr(feature_vector, 'cpu'):
        feature_vector = feature_vector.cpu().numpy()
    elif hasattr(feature_vector, 'numpy'):
        feature_vector = feature_vector.numpy()
    
    if mask_vector is not None:
        if hasattr(mask_vector, 'cpu'):
            mask_vector = mask_vector.cpu().numpy()
        elif hasattr(mask_vector, 'numpy'):
            mask_vector = mask_vector.numpy()
    
    data = {}
    idx = 0
    
    # Numeric features
    for col in dataset.num_cols:
        is_missing = mask_vector is not None and float(mask_vector[idx]) == 0.0
        
        if is_missing:
            data[f"{prefix}{col}"] = np.nan
            if return_all:
                data[f"{prefix}{col}_raw"] = np.nan
        else:
            value = float(feature_vector[idx])
            if return_all:
                # Store raw normalized value
                data[f"{prefix}{col}_raw"] = value
            
            # Denormalize
            if col in dataset.numeric_ranges:
                col_min, col_max = dataset.numeric_ranges[col]
                if col_max > col_min:
                    value = value * (col_max - col_min) + col_min
            data[f"{prefix}{col}"] = value
        idx += 1
    
    # One-hot features
    for col in dataset.oh_cols:
        if col in dataset.one_hot_mappings:
            categories = dataset.one_hot_mappings[col]
            oh_values = feature_vector[idx:idx+len(categories)]
            
            # Check mask for this feature group
            is_missing = mask_vector is not None and np.all(mask_vector[idx:idx+len(categories)] == 0)

            # If all values are zero OR mask indicates missing, it's a missing category
            if is_missing or np.all(oh_values == 0):
                data[f"{prefix}{col}"] = np.nan
            else:
                # Find the category with highest probability
                max_idx = np.argmax(oh_values)
                data[f"{prefix}{col}"] = categories[max_idx]
            
            if return_all:
                # Store individual probabilities for analysis
                for i, cat in enumerate(categories):
                    data[f"{prefix}{col}_{cat}_prob"] = float(oh_values[i]) if not is_missing else np.nan
                
                # Store raw one-hot vector
                data[f"{prefix}{col}_raw_vector"] = oh_values.tolist() if not is_missing else []
            
            idx += len(categories)
        else:
            # Missing mapping - this indicates an inconsistency between data spec and model metadata
            raise ValueError(f"One-hot column '{col}' is missing from saved mappings. "
                           f"This indicates an inconsistency between your data specification "
                           f"and the trained model's metadata.")
    
    # Ordinal features
    for i, col in enumerate(dataset.ord_cols):
        is_missing = mask_vector is not None and float(mask_vector[idx]) == 0.0
        
        if is_missing:
            data[f"{prefix}{col}"] = np.nan
            if return_all:
                data[f"{prefix}{col}_raw"] = np.nan
        else:
            raw_value = float(feature_vector[idx])
            ord_value = int(np.round(raw_value))  # Round to nearest integer instead of truncating
            ord_map = dataset.ord_maps[i]
            
            if return_all:
                # Store raw floating-point value
                data[f"{prefix}{col}_raw"] = raw_value
            
            # Reverse lookup to get original value
            reverse_map = {v: k for k, v in ord_map.items()}
            data[f"{prefix}{col}"] = reverse_map.get(ord_value, 'Unknown')
        idx += 1
    
    return pd.DataFrame([data])

def process_single_patient(patient_id, gx, meta, device, config_file):
        """Process a single patient and return generated records."""
        print(f"\n=== Processing Patient {patient_id} ===")
        
        seq_len=8
        # Create dataset for single patient using saved metadata
        dataset, df_clin, df_treat, df_actual_treat = create_single_patient_dataset_no_embed(
            patient_id, config_file, seq_len=seq_len, saved_metadata=meta
        )
        
        print(f"Clinical data shape: {df_clin.shape}")
        print(f"Treatment data shape: {df_treat.shape}")
        print(f"Clinical feature dimension: {dataset.clin_feature_dim}")
        print(f"Treatment feature dimension: {dataset.treat_feature_dim}")
        
        # Get patient data
        batch = dataset.get_patient_data(0)
        
        # Extract and process data
        tr_real = batch["x_treat"].to(device)
        cl_real = batch["x_clin"].to(device)
        tr_actual = batch["actual_treatment"].to(device)
        tr_mask = batch["mask_treat"].to(device)
        cl_mask = batch["mask_clin"].to(device)
        
        # Apply feature-level masking
        cl_real_masked = cl_real * cl_mask
        tr_real_masked = tr_real * tr_mask
        
        # Generate counterfactual scenarios
        scenarios = {
            "S": [1, 0, 0],
            "S_CT": [1, 1, 0], 
            "S_RT": [1, 0, 1],
            "S_RT_CT": [1, 1, 1],
        }
        
        # Generate counterfactuals
        generated = []
        
        for name, t_vec in scenarios.items():
            print(f"Processing scenario: {name}")
            
            # Create treatment conditioning tensor
            treat_tensor = torch.zeros(
                (cl_real_masked.shape[0], cl_real_masked.shape[1], 3),
                dtype=torch.float32,
                device=device,
            )
            treat_tensor[:, :, :] = torch.tensor(t_vec, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                # Generate counterfactual using Gx with new mask-aware interface
                # Create conditioning mask (all ones for actual treatment which is always present)
                cond_mask = torch.ones_like(treat_tensor)
                fake_treatment, _  = gx(cl_real, cl_mask, treat_tensor, cond_mask)
                # fake_treatment_masked = fake_treatment * tr_mask
            
            # Store timestep records for this scenario
            scenario_records = []
            
            # Ensure we get Python int, not tensor
            # seq_len = int(fake_treatment.shape[1])

            for timestep in range(seq_len):
                # Always process all timesteps up to seq_len for consistent output
                # Extract data for this timestep
                fake_step = fake_treatment[0, timestep].cpu().numpy()
                clinical_step = cl_real_masked[0, timestep].cpu().numpy()
                real_treatment_step = tr_real_masked[0, timestep].cpu().numpy()
                
                # Check if this timestep has valid clinical data
                clinical_timestep_mask = cl_mask[0, timestep].cpu().numpy()
                treatment_timestep_mask = tr_mask[0, timestep].cpu().numpy()
                has_clinical_data = clinical_timestep_mask.sum() > 0
                
                # Decode features for this timestep (will handle padded/masked data gracefully)
                decoded_fake_treatment = decode_raw_features_to_df(fake_step, dataset.treatment_dataset, None, "fake_", return_all=True)
                decoded_clinical = decode_raw_features_to_df(clinical_step, dataset.clinical_dataset, clinical_timestep_mask, "", return_all=False)
                decoded_real_treatment = decode_raw_features_to_df(real_treatment_step, dataset.treatment_dataset, treatment_timestep_mask, "real_", return_all=False)

                # Create record for this timestep
                record = {}
                record.update(decoded_clinical.iloc[0].to_dict())
                record.update(decoded_fake_treatment.iloc[0].to_dict())
                record.update(decoded_real_treatment.iloc[0].to_dict())
                
                # Add actual treatment flags for this timestep
                actual_treatment_step = tr_actual[0, timestep].cpu().numpy()
                record.update({
                    "actual_surgery": actual_treatment_step[0],
                    "actual_chemotherapy": actual_treatment_step[1],
                    "actual_radiotherapy": actual_treatment_step[2],
                })
                
                # Add metadata including whether this timestep has real data
                record.update({
                    "patient_id": patient_id,
                    "treatment_scenario": name,
                    "timestep": timestep,
                    "surgery": t_vec[0],
                    "chemotherapy": t_vec[1],
                    "radiotherapy": t_vec[2],
                    "has_clinical_data": has_clinical_data,  # Flag to indicate if this is real or padded data
                })
                scenario_records.append(record)
            
            # Add all timestep records for this scenario
            generated.extend(scenario_records)
        return generated



def process_multiple_patients(patient_ids, meta_file, config_file, output_file):
    """Process multiple patients and save all results to a single CSV file."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading no-embed model...")
    print(f"Model metadata: {meta_file}")
    print(f"Device: {device}")
    
    # Load no-embed model

    gx, meta = load_temporal_cycle_gan_no_embed(Path(meta_file), device)

    
    all_generated = []
    
    for patient_id in patient_ids:
        patient_records = process_single_patient(patient_id, gx, meta, device, config_file)
        all_generated.extend(patient_records)
        print(f"Processed patient {patient_id}: {len(patient_records)} records")
    
    # Save all results to CSV
    if all_generated:
        results_df = pd.DataFrame(all_generated)
        results_df.to_csv(output_file, index=False)
        
        print(f"\n=== Results Summary ===")
        print(f"Total records generated: {len(results_df)}")
        print(f"Patients processed: {len(patient_ids)}")
        print(f"Unique scenarios: {results_df['treatment_scenario'].nunique()}")
        print(f"Results saved to: {output_file}")
        print(f"Columns saved: {len(results_df.columns)}")
        
        # Show breakdown by patient
        patient_counts = results_df.groupby('patient_id').size()
        print(f"\nRecords per patient:")
        for patient_id, count in patient_counts.items():
            print(f"  Patient {patient_id}: {count} records")
            
    else:
        print("No records generated!")

def main():
    # Configuration
    meta_file = "saved_models/temporal_cycle_gan_no_embed_20250804_171903/metadata.json"
    config_file = "Models/configs/temporal_inference.yaml"
    
    # Multiple patients processing - EDIT THIS LIST WITH YOUR PATIENT IDs
    patient_ids = [156590, 215635, 242834, 323244, 345717, 526227, 527406, 541208,
       543042, 554253, 557218, 596485, 597316, 620309, 623524, 630105,
       643779, 644919, 666586, 685885, 722597, 729882, 731534, 737027,
       740578, 746581, 754544, 758485, 772731, 786252, 788512, 799525,
       806949, 883980, 1004685, 1009921, 1019291, 1024164, 1024170,
       1024802, 1027820, 1039395, 1040820, 1041181, 1043175, 1043203,
       1045021, 1048981, 1051298, 1059922, 1062642, 1062817, 1066772,
       1067388, 1068383, 1068420, 1069056, 1073393, 1075614, 1076361,
       1082196, 1083297, 1089543, 1093501, 1108596, 1121411, 1129171,
       1129883, 1135797, 1135959, 1137744, 1150095, 1156225, 1163146,
       1164292, 1164583, 1166965, 1168780, 1170493, 1172118, 1179808,
       1182831, 1196929, 1201406, 1210516, 1218316, 1227669, 1234070,
       1367536, 1482920, 1623575, 1864637, 2062965, 2965739, 3009605,
       3046004, 3070279, 3132164, 3218244, 3713571, 3931935, 3998282,
       4072073, 4133915, 4136202, 4173960, 4258568, 4391969, 4597575,
       5124433, 7005393, 7027290, 7041863, 7042017, 7063452, 7077825,
       7089896, 7094437, 7095159, 7127671, 7189917, 7193543, 7212730,
       7220739, 7224489, 7270384, 7290020, 7302677, 7355593, 7389708,
       7390813, 7396520, 7397603, 7398851, 7530399, 8041667, 8043280,
       8056530, 8099932, 8113981, 8136245, 8142960, 8243730, 8329971,
       8404797, 8502025, 8553685, 8563359, 8574708, 11145886, 11188681,
       11199316, 11203520, 100661653, '1051442ksw', '1051612ksw',
       '1093406ksw', '1106490ksw', '1124750ksw', '1130473ksw',
       '453800ksw'] 
    output_file = "counterfactual_scenarios_multiple_patients_no_embed.csv"
    
    # Process multiple patients
    process_multiple_patients(patient_ids, meta_file, config_file, output_file)

if __name__ == "__main__":
    main()
