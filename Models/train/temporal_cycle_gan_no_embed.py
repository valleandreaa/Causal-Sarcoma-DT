#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
temporal_cycle_gan_no_embed.py
A temporal CycleGAN that works directly with raw preprocessed features instead of embeddings.
This solves missing value handling issues by maintaining direct correspondence between masks and features.
Author: <you> – 2025‑08‑02
"""

from __future__ import annotations

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from dataclasses import dataclass
from pathlib import Path
import argparse, yaml, random, time, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils.helpers import MongoExtractor
import json
from models.utils import MetadataHandler, save_temporal_cycle_gan, randomize_multiple_one_hot
import math
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from models.GANs import LSTMGenerator, LSTMDiscriminator
from monitoring.losses import CycleGANLossMonitor
from monitoring.loss_logger import LossLogger
from datetime import datetime
from typing import Dict, List, Tuple

# --------------------------------------------------------------------- #
# 1. Configuration via YAML
# --------------------------------------------------------------------- #
@dataclass
class Config:
    # ---------- DATA ----------
    seq_len:          int
    id_field:         str
    # ---------- MODEL ----------
    x_dim:  int  # Will be overridden by actual feature dimensions
    y_dim:  int  # Will be overridden by actual feature dimensions
    treat_dim:  int
    g_hidden:   int
    d_hidden:   int
    num_layers: int
    bidir:      bool

    # ---------- TRAIN ----------
    batch_size:  int
    lr_g:        float
    lr_d:        float
    beta1:       float
    beta2:       float
    lambda_cyc:  float
    lambda_id:   float
    epochs:      int
    seed:        int
    device:      str | None = None
    adv_loss:    str = "bce"
    opt_gen:     str = "adam"
    opt_disc:    str = "adam"
    criterion:   str = "bce"

    # ---------- DATA CONFIG ----------
    data_config: dict = None
    mode: str = "train"

def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # If data_config is present, pass it along
    if "clinical_data_config" in d and "treatment_data_config" in d:
        d["data_config"] = {
            "clinical": d.pop("clinical_data_config"),
            "treatment": d.pop("treatment_data_config"),
        }
    return Config(**d)

def get_optimizer(name: str, params, lr: float, betas=(0.5, 0.999)):
    """Return optimizer instance based on name."""
    name = name.lower()
    if name == "adam":
        return optim.Adam(params, lr=lr, betas=betas)
    if name == "sgd":
        return optim.SGD(params, lr=lr)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {name}")

def get_criterion(name: str):
    """Return PyTorch loss given its name."""
    name = name.lower()
    if name in {"bce", "bcelogits", "binary_cross_entropy"}:
        return nn.BCEWithLogitsLoss()
    if name in {"mse", "mse_loss"}:
        return nn.MSELoss()
    if name in {"l1", "mae"}:
        return nn.L1Loss()
    if name in {"smooth_l1", "huber"}:
        return nn.SmoothL1Loss()
    if name in {"cosine", "cosine_embedding"}:
        return nn.CosineEmbeddingLoss()
    raise ValueError(f"Unsupported criterion: {name}")

# --------------------------------------------------------------------- #
# 2. TabularDataset for raw features (adapted from train_encoder.py)
# --------------------------------------------------------------------- #
class TabularDataset(Dataset):
    """Prepare mixed‑type tabular tensors and missing‑value masks for raw features."""

    def __init__(self, df: pd.DataFrame, feature_spec: Dict[str, str]):
        super().__init__()
        self.spec = feature_spec
        self.num_cols = [c for c, t in self.spec.items() if t == "numeric"]
        self.oh_cols = [c for c, t in self.spec.items() if t == "one_hot"]
        self.ord_cols = [c for c, t in self.spec.items() if t == "ordinal" or isinstance(t, dict)]

        # Build maps for ordinal columns
        self.ord_maps: List[Dict] = []
        for col in self.ord_cols:
            spec_val = self.spec[col]
            if isinstance(spec_val, dict):
                shifted = {v: i+1 for i, v in enumerate(sorted(spec_val.keys()))}
                self.ord_maps.append(shifted)
            else:
                uniq = sorted(df[col].dropna().unique())
                self.ord_maps.append({v: i+1 for i, v in enumerate(uniq)})

        # Build one_hot mappings for dummy conversion
        self.one_hot_mappings: Dict[str, List[str]] = {}
        
        # Store numeric ranges for later reconstruction
        self.numeric_ranges: Dict[str, Tuple[float, float]] = {
            col: (float(df[col].min()), float(df[col].max())) for col in self.num_cols
        }
        
        # Pre‑process
        self.x_features, self.x_ord, self.mask, self.feature_dim = self._preprocess(df)

    def _encode_ord(self, s: pd.Series, mp: Dict[str, int]) -> np.ndarray:
        return s.map(mp).fillna(0).astype("int64").to_numpy()

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int]:
        # numeric -------------------------------------------------------------
        x_num = df[self.num_cols].to_numpy(dtype="float32", copy=True) if self.num_cols else np.empty((len(df), 0), dtype="float32")
        num_mask = ~np.isnan(x_num) if self.num_cols else np.empty_like(x_num, dtype=bool)
        
        # Normalize numeric features
        for i, col in enumerate(self.num_cols):
            col_min, col_max = self.numeric_ranges[col]
            if col_max > col_min:
                x_num[:, i] = (x_num[:, i] - col_min) / (col_max - col_min)
        
        x_num[np.isnan(x_num)] = 0.0

        # one‑hot -------------------------------------------------------------
        if self.oh_cols:
            x_oh_parts, mask_parts = [], []
            for col in self.oh_cols:
                # Get unique categories excluding NaN
                unique_cats = sorted(df[col].dropna().unique())
                self.one_hot_mappings[col] = unique_cats
                
                # Create one-hot encoding manually
                oh_matrix = np.zeros((len(df), len(unique_cats)), dtype="float32")
                for i, cat in enumerate(unique_cats):
                    oh_matrix[:, i] = (df[col] == cat).astype("float32")
                
                x_oh_parts.append(oh_matrix)
                
                # Create mask: 1 where value is not NaN, 0 where NaN
                col_mask = (~pd.isna(df[col])).to_numpy(dtype="float32")[:, None]
                mask_parts.append(np.repeat(col_mask, len(unique_cats), axis=1))
            
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

    def __len__(self) -> int:
        return self.x_features.shape[0]

    def __getitem__(self, idx: int):
        if self.x_ord is not None:
            return self.x_features[idx], self.x_ord[idx], self.mask[idx]
        return self.x_features[idx], torch.zeros(0, dtype=torch.long), self.mask[idx]

# --------------------------------------------------------------------- #
# 3. Temporal Dataset for CycleGAN (raw features)
# --------------------------------------------------------------------- #
class TemporalTabularDatasetPID(Dataset):
    """Temporal dataset that works with raw preprocessed features instead of embeddings."""
    
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
    ):
        self.seq_len = seq_len
        self.id_field = id_field
        self.return_mask = return_mask
        
        # Create TabularDatasets for preprocessing
        self.clinical_dataset = TabularDataset(df_clin, feature_spec_clin)
        self.treatment_dataset = TabularDataset(df_treat, feature_spec_treat)
        
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
        
        print(f"Dataset initialized with {self.num_patients} patients")
        print(f"Clinical feature dimension: {self.clin_feature_dim}")
        print(f"Treatment feature dimension: {self.treat_feature_dim}")

    def __len__(self):
        return self.num_patients

    def __getitem__(self, patient_idx):
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
        
        result = {
            "x_clin": x_clin,
            "x_treat": x_treat,
            "actual_treatment": torch.from_numpy(actual_treat).float(),
        }
        
        if self.return_mask:
            result["mask_clin"] = mask_clin
            result["mask_treat"] = mask_treat
            
        return result

# --------------------------------------------------------------------- #
# 4. Evaluation functions
# --------------------------------------------------------------------- #
def evaluate_counterfactual_distributions(
    Gx,
    loader,
    treatment_dataset,
    device,
    output_path,
):
    """Evaluate distribution of generated counterfactual treatments using raw features."""
    prev_mode = Gx.training
    Gx.eval()
    real_rows = []
    fake_rows = []
    
    with torch.no_grad():
        for batch in loader:
            tr_real = batch["x_treat"].to(device)
            cl_real = batch["x_clin"].to(device)
            tr_actual = batch["actual_treatment"].to(device)
            tr_mask = batch["mask_treat"].to(device)
            cl_mask = batch["mask_clin"].to(device)
            
            # Apply feature-level masking
            cl_real_masked = cl_real * cl_mask
            tr_real_masked = tr_real * tr_mask
            tr_counter = randomize_multiple_one_hot(tr_actual,extra_ones=1)
            
            # Generate counterfactual treatments
            fake_y, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
            fake_y = fake_y * tr_mask

            # Convert raw features back to interpretable format
            for b in range(fake_y.size(0)):
                for t in range(fake_y.size(1)):
                    # Skip empty timesteps
                    if tr_mask[b, t].sum() == 0:
                        continue
                    
                    # Convert real treatment features to dataframe
                    real_df = decode_raw_features(
                        tr_real_masked[b, t].cpu().numpy(),
                        treatment_dataset
                    )
                    
                    # Convert fake treatment features to dataframe  
                    fake_df = decode_raw_features(
                        fake_y[b, t].cpu().numpy(), 
                        treatment_dataset
                    )
                    
                    real_rows.append(real_df)
                    fake_rows.append(fake_df)

    if prev_mode:
        Gx.train()

    if not real_rows:
        return

    real_df = pd.concat(real_rows, ignore_index=True)
    fake_df = pd.concat(fake_rows, ignore_index=True)

    cols = real_df.columns
    n_cols = len(cols)
    
    # Categorize columns based on dataset feature mappings
    numeric_cols = treatment_dataset.num_cols
    categorical_cols = []
    ordinal_cols = treatment_dataset.ord_cols
    
    # One-hot columns need special handling
    for oh_col in treatment_dataset.oh_cols:
        if oh_col in treatment_dataset.one_hot_mappings:
            categorical_cols.extend([f"{oh_col}_{cat}" for cat in treatment_dataset.one_hot_mappings[oh_col]])
    
    # Calculate total subplots needed
    total_plots = len(numeric_cols) + 2 * len(categorical_cols) + 2 * len(ordinal_cols)
    cols_per_row = 3
    n_rows = math.ceil(total_plots / cols_per_row)
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = axes.flatten()

    plot_idx = 0
    
    # Plot numeric columns
    for col in numeric_cols:
        if col in real_df.columns and col in fake_df.columns:
            ax = axes[plot_idx]
            r = real_df[col].dropna()
            f = fake_df[col].dropna()
            ax.hist(r, bins=20, alpha=0.5, label="real")
            ax.hist(f, bins=20, alpha=0.5, label="fake")
            ax.set_title(f"{col} (Numeric)")
            ax.legend()
            plot_idx += 1
    
    # Plot categorical columns (one-hot encoded)
    for col in categorical_cols:
        if col in real_df.columns and col in fake_df.columns:
            r = real_df[col].dropna()
            f = fake_df[col].dropna()
            
            r_counts = r.value_counts()
            f_counts = f.value_counts()
            categories = sorted(set(r_counts.index).union(f_counts.index))
            r_vals = r_counts.reindex(categories).fillna(0)
            f_vals = f_counts.reindex(categories).fillna(0)
            idxs = np.arange(len(categories))
            
            # Real distribution subplot
            ax_real = axes[plot_idx]
            ax_real.bar(idxs, r_vals.values, width=0.6, color='blue', alpha=0.7)
            ax_real.set_xticks(idxs)
            ax_real.set_xticklabels(categories, rotation=90, fontsize=8)
            ax_real.set_title(f"{col} - Real (Categorical)")
            plot_idx += 1
            
            # Fake distribution subplot
            ax_fake = axes[plot_idx]
            ax_fake.bar(idxs, f_vals.values, width=0.6, color='orange', alpha=0.7)
            ax_fake.set_xticks(idxs)
            ax_fake.set_xticklabels(categories, rotation=90, fontsize=8)
            ax_fake.set_title(f"{col} - Fake (Categorical)")
            plot_idx += 1
    
    # Plot ordinal columns
    for col in ordinal_cols:
        if col in real_df.columns and col in fake_df.columns:
            r = real_df[col].dropna()
            f = fake_df[col].dropna()
            
            r_counts = r.value_counts()
            f_counts = f.value_counts()
            
            # Use the ordering from dataset's ordinal mappings
            ord_idx = treatment_dataset.ord_cols.index(col)
            ord_map = treatment_dataset.ord_maps[ord_idx]
            ordered_categories = sorted(ord_map.keys(), key=lambda x: ord_map[x])
            
            r_vals = r_counts.reindex(ordered_categories).fillna(0)
            f_vals = f_counts.reindex(ordered_categories).fillna(0)
            idxs = np.arange(len(ordered_categories))
            
            # Real distribution subplot
            ax_real = axes[plot_idx]
            ax_real.bar(idxs, r_vals.values, width=0.6, color='green', alpha=0.7)
            ax_real.set_xticks(idxs)
            ax_real.set_xticklabels(ordered_categories, rotation=90, fontsize=8)
            ax_real.set_title(f"{col} - Real (Ordinal)")
            plot_idx += 1
            
            # Fake distribution subplot
            ax_fake = axes[plot_idx]
            ax_fake.bar(idxs, f_vals.values, width=0.6, color='red', alpha=0.7)
            ax_fake.set_xticks(idxs)
            ax_fake.set_xticklabels(ordered_categories, rotation=90, fontsize=8)
            ax_fake.set_title(f"{col} - Fake (Ordinal)")
            plot_idx += 1

    # Turn off remaining empty subplots
    for ax in axes[plot_idx:]:
        ax.axis("off")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def decode_raw_features(feature_vector, dataset):
    """Convert raw feature vector back to interpretable DataFrame format."""
    data = {}
    idx = 0
    
    # Numeric features
    for col in dataset.num_cols:
        value = feature_vector[idx]
        # Denormalize
        col_min, col_max = dataset.numeric_ranges[col]
        if col_max > col_min:
            value = value * (col_max - col_min) + col_min
        data[col] = value
        idx += 1
    
    # One-hot features
    for col in dataset.oh_cols:
        if col in dataset.one_hot_mappings:
            categories = dataset.one_hot_mappings[col]
            oh_values = feature_vector[idx:idx+len(categories)]
            
            # Find the category with highest probability
            max_idx = np.argmax(oh_values)
            data[col] = categories[max_idx]
            
            # Also store individual one-hot columns for plotting
            for i, cat in enumerate(categories):
                data[f"{col}_{cat}"] = oh_values[i]
            
            idx += len(categories)
    
    # Ordinal features
    for i, col in enumerate(dataset.ord_cols):
        ord_value = int(feature_vector[idx])
        ord_map = dataset.ord_maps[i]
        # Reverse lookup to get original value
        reverse_map = {v: k for k, v in ord_map.items()}
        data[col] = reverse_map.get(ord_value, 'Unknown')
        idx += 1
    
    return pd.DataFrame([data])


def evaluate(
    Gx, Gy, Dx, Dy, loader, criterion_adv, criterion_cycle, criterion_id,
    lambda_cycle, lambda_id, device,
):
    """Evaluate model on validation data."""
    prev_states = (Gx.training, Gy.training, Dx.training, Dy.training)
    Gx.eval(); Gy.eval(); Dx.eval(); Dy.eval()
    g_loss = d_loss = cyc_loss = id_loss = 0.0
    gx_loss = gy_loss = dx_loss = dy_loss = 0.0
    
    with torch.no_grad():
        for batch in loader:
            tr_real = batch["x_treat"].to(device)
            cl_real = batch["x_clin"].to(device)
            tr_actual = batch["actual_treatment"].to(device)
            tr_mask = batch["mask_treat"].to(device)
            cl_mask = batch["mask_clin"].to(device)
            
            # Apply masking to features
            cl_real_masked = cl_real * cl_mask
            tr_real_masked = tr_real * tr_mask
            tr_counter = randomize_multiple_one_hot(tr_actual, extra_ones=1)

            fake_y, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
            fake_y = fake_y * tr_mask
            rec_x, _ = Gy(fake_y, tr_mask, tr_actual, torch.ones_like(tr_actual))
            rec_x = rec_x * tr_mask

            # Create main sequences for discriminators
            main_real = torch.cat((cl_real_masked, tr_real_masked), dim=2)
            main_fake_x = torch.cat((cl_real_masked, fake_y), dim=2)
            main_fake_y = torch.cat((cl_real_masked, rec_x), dim=2)
            main_mask = torch.cat((cl_mask, tr_mask), dim=2)
            cond_mask = torch.ones_like(tr_actual)

            pred_fake_x = Dx(main_fake_x, main_mask, tr_counter, cond_mask)
            pred_real_x = Dx(main_real, main_mask, tr_actual, cond_mask)
            pred_fake_y = Dy(main_fake_y, main_mask, tr_actual, cond_mask)
            pred_real_y = Dy(main_real, main_mask, tr_actual, cond_mask)

            loss_Dx = 0.5*(criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x)) +
                           criterion_adv(pred_real_x, torch.ones_like(pred_real_x)))
            loss_Dy = 0.5*(criterion_adv(pred_real_y, torch.ones_like(pred_real_y)) +
                           criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y)))

            loss_Gx_adv = criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_Gy_adv = criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y))

            cycle_l = criterion_cycle(rec_x * tr_mask, tr_real_masked)
            id_l = torch.tensor(0.0, device=device)
            if lambda_id > 0:
                id_y, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
                id_y = id_y * tr_mask
                id_x, _ = Gy(id_y, tr_mask, tr_actual, torch.ones_like(tr_actual))
                id_x = id_x * tr_mask
                id_l = criterion_id(id_x, tr_real_masked)

            loss_G = loss_Gx_adv + loss_Gy_adv + lambda_cycle * cycle_l + lambda_id * id_l

            g_loss += loss_G.item()
            d_loss += (loss_Dx + loss_Dy).item()
            cyc_loss += cycle_l.item()
            id_loss += id_l.item()
            gx_loss += loss_Gx_adv.item()
            gy_loss += loss_Gy_adv.item()
            dx_loss += loss_Dx.item()
            dy_loss += loss_Dy.item()
    
    n = len(loader)

    # Restore original training/eval state
    if prev_states[0]: Gx.train()
    if prev_states[1]: Gy.train()
    if prev_states[2]: Dx.train()
    if prev_states[3]: Dy.train()

    return (g_loss / n, d_loss / n, cyc_loss / n, id_loss / n,
            gx_loss / n, gy_loss / n, dx_loss / n, dy_loss / n)

# --------------------------------------------------------------------- #
# 5. Main training function
# --------------------------------------------------------------------- #
def train(cfg, args):
    # Load data
    if not os.getenv("MONGO_URI"):
        raise RuntimeError("Environment variable MONGO_URI is required to fetch data from MongoDB.")
    
    # Clinical data
    extractor_clin = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        custom_config=cfg.data_config['clinical']
    )
    
    # Treatment data
    extractor_treat = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        custom_config=cfg.data_config['treatment']
    )
    
    # Actual treatment labels
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
    
    df_clin = extractor_clin.get_dataframe()
    df_treat = extractor_treat.get_dataframe()
    df_actual_treat = extractor_actual_treat.get_dataframe()
    
    # Reorder columns to match feature_spec order
    df_clin = MongoExtractor.reorder_dataframe_columns(df_clin, cfg.data_config["clinical"]["features"])
    df_treat = MongoExtractor.reorder_dataframe_columns(df_treat, cfg.data_config["treatment"]["features"])
    
    df_clin.reset_index(inplace=True)
    df_treat.reset_index(inplace=True)
    df_actual_treat.reset_index(inplace=True)
    
    # Create dataset
    base_dataset = TemporalTabularDatasetPID(
        df_clin=df_clin,
        df_treat=df_treat,
        df_actual_treat=df_actual_treat,
        feature_spec_clin=cfg.data_config["clinical"]["features"],
        feature_spec_treat=cfg.data_config["treatment"]["features"],
        seq_len=cfg.seq_len,
        id_field=cfg.id_field,
        return_mask=True,
    )

    # Split patients into train/validation
    rng = random.Random(cfg.seed)
    indices = list(range(base_dataset.num_patients))
    rng.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = torch.utils.data.Subset(base_dataset, train_indices)
    val_subset = torch.utils.data.Subset(base_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg.batch_size, shuffle=False)

    print(f"Training on {len(train_indices)} patients, validating on {len(val_indices)} patients")

    # Model parameters
    clin_dim = base_dataset.clin_feature_dim
    treat_dim = base_dataset.treat_feature_dim
    hidden_dim = cfg.g_hidden
    num_layers = cfg.num_layers
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Clinical features: {clin_dim}, Treatment features: {treat_dim}")

    # Instantiate networks
    Gx = LSTMGenerator(
        input_dim=clin_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=treat_dim,  # Generate treatment features
        num_layers=num_layers,
    ).to(device)
    
    Gy = LSTMGenerator(
        input_dim=treat_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=treat_dim,  # Reconstruct treatment features
        num_layers=num_layers,
    ).to(device)
    
    # For discriminators, input_dim is for main sequence, cond_dim is for conditioning
    Dx = LSTMDiscriminator(
        input_dim=clin_dim + treat_dim,  # Main sequence (clinical + treatment)
        cond_dim=3,  # Conditioning (actual treatment)
        hidden_dim=cfg.d_hidden,
        output_dim=1,  # Add missing output_dim parameter
        num_layers=num_layers,
    ).to(device)
    
    Dy = LSTMDiscriminator(
        input_dim=clin_dim + treat_dim,  # Main sequence (clinical + treatment)
        cond_dim=3,  # Conditioning (actual treatment)
        hidden_dim=cfg.d_hidden,
        output_dim=1,  # Add missing output_dim parameter
        num_layers=num_layers,
    ).to(device)

    # Loss functions
    from models.losses import get_adversarial_loss
    criterion_adv = get_adversarial_loss(cfg.adv_loss)
    criterion_cycle = get_criterion(cfg.criterion)
    criterion_id = get_criterion(cfg.criterion)

    # Optimizers
    optimizer_G = get_optimizer(cfg.opt_gen, list(Gx.parameters()) + list(Gy.parameters()), 
                               cfg.lr_g, (cfg.beta1, cfg.beta2))
    optimizer_Dx = get_optimizer(cfg.opt_disc, Dx.parameters(), cfg.lr_d, (cfg.beta1, cfg.beta2))
    optimizer_Dy = get_optimizer(cfg.opt_disc, Dy.parameters(), cfg.lr_d, (cfg.beta1, cfg.beta2))
    
    # Monitoring
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"saved_models/temporal_cycle_gan_no_embed_{timestamp}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    loss_plot_path = save_dir / "cyclegan_losses.png"
    loss_monitor = CycleGANLossMonitor(path=loss_plot_path)
    loss_logger = LossLogger(save_dir / "losses.log")
    
    print(f"Loss plots will be saved to: {loss_plot_path}")
    print(f"Loss logs will be saved to: {save_dir / 'losses.log'}")
    if args.check_counterfactuals:
        print(f"Counterfactual plots will be saved to: {save_dir}")

    # Training loop
    for epoch in range(cfg.epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_cycle_losses = []
        epoch_id_losses = []
        epoch_gx_losses = []
        epoch_gy_losses = []
        epoch_dx_losses = []
        epoch_dy_losses = []

        for i, batch in enumerate(train_loader):
            # Extract data from batch dictionary
            tr_real = batch["x_treat"].to(device)    # [B, T, F_treat]
            cl_real = batch["x_clin"].to(device)    # [B, T, F_clin]
            # tr_actual = batch["actual_treatment"].to(device)  # [B, T, 3]
            tr_mask = batch["mask_treat"].to(device)  # [B, T, F_treat]
            cl_mask = batch["mask_clin"].to(device)  # [B, T, F_clin]
            # Repeat the first row of tr_actual across all timesteps
            tr_actual_expanded = batch["actual_treatment"][:, 0:1, :].expand(-1, batch["actual_treatment"].size(1), -1)
            tr_actual = tr_actual_expanded.to(device)  # [B, T, 3]
            # Apply feature-level masking (zeros out missing features)
            cl_real_masked = cl_real * cl_mask
            tr_real_masked = tr_real * tr_mask
            
            # Create counterfactual treatment
            tr_counter = randomize_multiple_one_hot(tr_actual, extra_ones=1)

            # ——— 1) Discriminator update ———
            optimizer_Dx.zero_grad()
            optimizer_Dy.zero_grad()

            # Generate fakes and reconstructions, then detach for D
            fake_y_det, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
            fake_y_det = fake_y_det.detach()
            rec_x_det, _ = Gy(fake_y_det, tr_mask, tr_actual, torch.ones_like(tr_actual))
            rec_x_det = rec_x_det.detach()
            
            # Apply treatment mask to generated outputs
            fake_y_det = fake_y_det * tr_mask
            rec_x_det = rec_x_det * tr_mask

            # D's predictions on real vs. fake (concatenate clinical + treatment as main sequence)
            main_real = torch.cat((cl_real_masked, tr_real_masked), dim=2)
            main_fake_x = torch.cat((cl_real_masked, fake_y_det), dim=2)
            main_fake_y = torch.cat((cl_real_masked, fake_y_det), dim=2)
            
            # Create combined masks for main sequences
            main_mask_real = torch.cat((cl_mask, tr_mask), dim=2)
            main_mask_fake_x = torch.cat((cl_mask, tr_mask), dim=2)
            main_mask_fake_y = torch.cat((cl_mask, tr_mask), dim=2)
            
            # Conditioning mask (all ones for actual treatment which is always present)
            cond_mask = torch.ones_like(tr_actual)
            
            pred_real_x = Dx(main_real, main_mask_real, tr_actual, cond_mask)
            pred_fake_x = Dx(main_fake_x, main_mask_fake_x, tr_counter, cond_mask)
            pred_real_y = Dy(main_real, main_mask_real, tr_actual, cond_mask)
            pred_fake_y = Dy(main_fake_y, main_mask_fake_y, tr_counter, cond_mask)

            # D losses
            loss_Dx = 0.5*(criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x)) + 
                           criterion_adv(pred_real_x, torch.ones_like(pred_real_x)))
            loss_Dy = 0.5*(criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y)) +
                           criterion_adv(pred_real_y, torch.ones_like(pred_real_y)))

            (loss_Dx + loss_Dy).backward()
            optimizer_Dx.step()
            optimizer_Dy.step()

            # ——— 2) Generator update ———
            for p in Dx.parameters(): p.requires_grad_(False)
            for p in Dy.parameters(): p.requires_grad_(False)

            optimizer_G.zero_grad()

            # Fresh forward through G
            fake_y, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
            rec_x, _ = Gy(fake_y, tr_mask, tr_actual, torch.ones_like(tr_actual))
            
            # Apply treatment mask to generated outputs
            fake_y = fake_y * tr_mask
            rec_x = rec_x * tr_mask

            # Create main sequences for discriminators
            main_fake_x = torch.cat((cl_real_masked, fake_y), dim=2)
            main_fake_y = torch.cat((cl_real_masked, fake_y), dim=2)
            main_mask_fake_x = torch.cat((cl_mask, tr_mask), dim=2)
            main_mask_fake_y = torch.cat((cl_mask, tr_mask), dim=2)
            cond_mask = torch.ones_like(tr_actual)

            pred_fake_x = Dx(main_fake_x, main_mask_fake_x, tr_counter, cond_mask)
            pred_fake_y = Dy(main_fake_y, main_mask_fake_y, tr_actual, cond_mask)

            # G losses
            loss_Gx_adv = criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_Gy_adv = criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y))

            # Cycle loss (only on masked regions)
            loss_cycle = criterion_cycle(rec_x * tr_mask, tr_real_masked)

            # Identity loss
            loss_id = torch.tensor(0.0, device=device)
            if cfg.lambda_id > 0:
                id_y, _ = Gx(cl_real, cl_mask, tr_counter, torch.ones_like(tr_counter))
                id_y = id_y * tr_mask
                id_x, _ = Gy(id_y, tr_mask, tr_actual, torch.ones_like(tr_actual))
                id_x = id_x * tr_mask
                loss_id = criterion_id(id_x, tr_real_masked)

            # Total G loss
            loss_G = (loss_Gx_adv + loss_Gy_adv + 
                     cfg.lambda_cyc * loss_cycle + cfg.lambda_id * loss_id)
            
            loss_G.backward()
            optimizer_G.step()

            # Unfreeze D's weights
            for p in Dx.parameters(): p.requires_grad_(True)
            for p in Dy.parameters(): p.requires_grad_(True)

            # Store losses
            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append((loss_Dx + loss_Dy).item())
            epoch_cycle_losses.append(loss_cycle.item())
            epoch_id_losses.append(loss_id.item())
            epoch_gx_losses.append(loss_Gx_adv.item())
            epoch_gy_losses.append(loss_Gy_adv.item())
            epoch_dx_losses.append(loss_Dx.item())
            epoch_dy_losses.append(loss_Dy.item())

        # Validation
        val_losses = evaluate(Gx, Gy, Dx, Dy, val_loader, criterion_adv, criterion_cycle, 
                             criterion_id, cfg.lambda_cyc, cfg.lambda_id, device)
        val_g_loss, val_d_loss, val_cycle_loss, val_id_loss, val_gx_loss, val_gy_loss, val_dx_loss, val_dy_loss = val_losses

        # Average losses
        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_cycle_loss = sum(epoch_cycle_losses) / len(epoch_cycle_losses)
        avg_id_loss = sum(epoch_id_losses) / len(epoch_id_losses)
        avg_gx_loss = sum(epoch_gx_losses) / len(epoch_gx_losses)
        avg_gy_loss = sum(epoch_gy_losses) / len(epoch_gy_losses)
        avg_dx_loss = sum(epoch_dx_losses) / len(epoch_dx_losses)
        avg_dy_loss = sum(epoch_dy_losses) / len(epoch_dy_losses)

        print(f"Epoch [{epoch+1}/{cfg.epochs}] "
              f"G: {avg_g_loss:.4f}, D: {avg_d_loss:.4f}, "
              f"Cycle: {avg_cycle_loss:.4f}, Val_G: {val_g_loss:.4f}")

        # Log losses
        loss_monitor.update(
            epoch, avg_g_loss, avg_d_loss, cycle_loss=avg_cycle_loss,
            identity_loss=avg_id_loss, val_gen_loss=val_g_loss, val_disc_loss=val_d_loss,
            val_cycle_loss=val_cycle_loss, val_identity_loss=val_id_loss,
            gx_loss=avg_gx_loss, gy_loss=avg_gy_loss, dx_loss=avg_dx_loss, dy_loss=avg_dy_loss,
            val_gx_loss=val_gx_loss, val_gy_loss=val_gy_loss, val_dx_loss=val_dx_loss, val_dy_loss=val_dy_loss,
        )

        loss_logger.log(
            epoch, avg_g_loss, avg_d_loss, cycle_loss=avg_cycle_loss,
            identity_loss=avg_id_loss, val_gen_loss=val_g_loss, val_disc_loss=val_d_loss,
            val_cycle_loss=val_cycle_loss, val_identity_loss=val_id_loss,
            gx_loss=avg_gx_loss, gy_loss=avg_gy_loss, dx_loss=avg_dx_loss, dy_loss=avg_dy_loss,
            val_gx_loss=val_gx_loss, val_gy_loss=val_gy_loss, val_dx_loss=val_dx_loss, val_dy_loss=val_dy_loss,
        )

        # Evaluate counterfactual distributions if requested
        if args.check_counterfactuals:
            dist_path = save_dir / f"counterfactuals_epoch_{epoch+1}.png"
            evaluate_counterfactual_distributions(
                Gx,
                val_loader,
                base_dataset.treatment_dataset,
                device,
                dist_path,
            )

    # Save final loss plot
    if loss_monitor.path:
        print(f"Saving final loss plot to: {loss_monitor.path}")
        loss_monitor.fig.savefig(loss_monitor.path, dpi=150, bbox_inches='tight')
        plt.close(loss_monitor.fig)

    # Save trained models
    meta = {
        "clinical_feature_spec": cfg.data_config["clinical"]["features"],
        "treatment_feature_spec": cfg.data_config["treatment"]["features"],
        "clinical_feature_dim": clin_dim,
        "treatment_feature_dim": treat_dim,
        "clinical_one_hot_mappings": base_dataset.clinical_dataset.one_hot_mappings,
        "treatment_one_hot_mappings": base_dataset.treatment_dataset.one_hot_mappings,
        "clinical_numeric_ranges": base_dataset.clinical_dataset.numeric_ranges,
        "treatment_numeric_ranges": base_dataset.treatment_dataset.numeric_ranges,
        "val_patient_ids": [base_dataset.patient_ids[i] for i in val_indices],
        "loss_log_path": str(loss_logger.filepath),
    }

    params = {
        "clin_dim": clin_dim,
        "treat_dim": treat_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_len": cfg.seq_len,
        "batch_size": cfg.batch_size,
        "num_epochs": cfg.epochs,
        "lr_g": cfg.lr_g,
        "lr_d": cfg.lr_d,
        "beta1": cfg.beta1,
        "lambda_cycle": cfg.lambda_cyc,
        "lambda_id": cfg.lambda_id,
    }

    save_temporal_cycle_gan(Gx, Gy, Dx, Dy, save_dir, metadata=meta, model_params=params, timestamp=timestamp)
    print(f"Models saved to {save_dir}")

# --------------------------------------------------------------------- #
# 6. Entry point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train temporal CycleGAN with raw features")
    parser.add_argument("--config_file", default="configs/mock_config_cycle_gan_split.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument(
        "--check_counterfactuals",
        action="store_true",
        help="Evaluate counterfactual distributions each epoch",
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config_file)
    
    # Set seed
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    
    train(cfg, args)
