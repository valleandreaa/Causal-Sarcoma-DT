import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class TabularDataset(Dataset):
    """Prepare mixed‑type tabular tensors and missing‑value masks for clinical and treatment data."""
    def __init__(self, df: pd.DataFrame, meta_clin, meta_treat):
        super().__init__()
        # Extract separate feature specifications
        clinical_spec = meta_clin.get_feature_spec()
        treatment_spec = meta_treat.get_feature_spec()
        self.meta_clin = meta_clin
        self.meta_treat = meta_treat
        # Preprocess clinical data
        self.clinical = self._preprocess_spec(df, clinical_spec, self.meta_clin)
        # Preprocess treatment data
        self.treatment = self._preprocess_spec(df, treatment_spec, self.meta_treat)
        self.actual_treat = self.get_actual_treatment(df)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_actual_treatment(self, df: pd.DataFrame) -> torch.Tensor:
        df_actual_treatemnt = df[["episodes.surgery", "episodes.chemotherapy", "episodes.radiotherapy"]]
        return torch.from_numpy(df_actual_treatemnt.to_numpy(dtype="int64").reshape(-1, 3))
    
    def _preprocess_spec(self, df: pd.DataFrame, spec: Dict, meta) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        # Derive columns from spec
        num_cols = [c for c, t in spec.items() if t == "numeric"]
        oh_cols = [c for c, t in spec.items() if t == "one_hot"]
        ord_cols = [c for c, t in spec.items() if t == "ordinal" or isinstance(t, dict)]
    
        # Get metadata for consistency
        meta_json = meta.import_metadata()
        saved_numeric_ranges = meta_json.get("numeric_ranges", {})
        saved_one_hot_mappings = meta_json.get("one_hot_mappings", {})
        saved_ord_mappings = meta_json.get("ord_mappings", {})

        # Build ordinal maps
        ord_maps = []
        for col in ord_cols:
            t_val = spec[col]
            if isinstance(t_val, dict):
                shifted = {v: i+1 for i, v in enumerate(sorted(t_val.keys()))}
                ord_maps.append(shifted)
            else:
                if col in saved_ord_mappings:
                    ord_maps.append(saved_ord_mappings[col])
                else:
                    uniq = sorted(df[col].dropna().unique())
                    ord_maps.append({v: i+1 for i, v in enumerate(uniq)})

        # numeric -------------------------------------------------------------
        x_num = df[num_cols].to_numpy(dtype="float32", copy=True) if num_cols else np.empty((len(df), 0), dtype="float32")
        num_mask = ~np.isnan(x_num) if num_cols else np.empty((len(df), 0), dtype=bool)
        
        # Apply normalization using saved ranges
        for i, col in enumerate(num_cols):
            if col in saved_numeric_ranges:
                col_min, col_max = saved_numeric_ranges[col]
                if col_max > col_min:
                    x_num[:, i] = (x_num[:, i] - col_min) / (col_max - col_min)
        
        x_num[np.isnan(x_num)] = 0.0

        # one‑hot -------------------------------------------------------------
        if oh_cols:
            x_oh_parts, mask_parts = [], []
            for col in oh_cols:
                if col in saved_one_hot_mappings:
                    categories = saved_one_hot_mappings[col]
                    cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
                    dummies = pd.get_dummies(df[col].astype(cat_dtype), prefix=col).astype("float32")
                else:
                    dummies = pd.get_dummies(df[col], dummy_na=False, prefix=col).astype("float32")

                x_oh_parts.append(dummies.to_numpy())
                col_mask = (~pd.isna(df[col])).to_numpy(dtype="float32")[:, None]
                mask_parts.append(np.repeat(col_mask, dummies.shape[1], axis=1))
            x_oh = np.concatenate(x_oh_parts, axis=1)
            oh_mask = np.concatenate(mask_parts, axis=1)
            x_numeric_oh = np.concatenate([x_num, x_oh], axis=1)
            mask_numeric = np.concatenate([num_mask, oh_mask], axis=1).astype("float32")
        else:
            x_numeric_oh = x_num
            mask_numeric  = num_mask.astype("float32")
        
        # ordinal -------------------------------------------------------------
        if ord_cols:
            ord_arrays = []
            ord_mask_arrays = []
            for col, mp in zip(ord_cols, ord_maps):
                # Create mask based on original missing values, not encoded values
                orig_mask = (~pd.isna(df[col])).to_numpy().astype("float32")
                ord_mask_arrays.append(orig_mask[:, None])
                
                # Encode ordinals (missing values become 0)
                encoded = df[col].map(mp).fillna(0).astype("int64").to_numpy()
                ord_arrays.append(encoded[:, None])
            
            x_ord = np.concatenate(ord_arrays, axis=1)
            mask_ord = np.concatenate(ord_mask_arrays, axis=1)
            x_target = np.concatenate([x_numeric_oh, x_ord.astype("float32")], axis=1)
            mask_target = np.concatenate([mask_numeric, mask_ord], axis=1)
        else:
            x_target = x_numeric_oh
            mask_target = mask_numeric
            x_ord = np.empty((len(df), 0), dtype="int64")

        total_dim = x_target.shape[1]
        return (
            torch.from_numpy(x_target).float(),
            torch.from_numpy(x_ord).long(),
            torch.from_numpy(mask_target).float(),
            total_dim,
        )

    def __len__(self) -> int:
        # Assuming clinical and treatment data have the same number of rows.
        return self.clinical[0].shape[0]

    def __getitem__(self, idx: int):
        clin_raw = (self.clinical[0], self.clinical[1], self.clinical[2])
        # Move treatment and clinical raw tensors to self.device
        treat_raw = (self.treatment[0].to(self.device), self.treatment[1].to(self.device), self.treatment[2].to(self.device))
        clin_raw = (self.clinical[0].to(self.device), self.clinical[1].to(self.device), self.clinical[2].to(self.device))
        
        actual_treat = self.actual_treat.to(self.device)
        # Generate treatment tensor on the device (adjust as needed for proper treatment encoding)
        # actual_treat = torch.zeros((self.treatment[0].shape[0], 3), device=self.device)
        # indices = torch.randint(0, 3, (self.treatment[0].shape[0],), device=self.device)
        # actual_treat[torch.arange(self.treatment[0].shape[0]), indices] = 1

        enc_clin = self.meta_clin.get_encoded_dataset(clin_raw, self.device)
        enc_treat = self.meta_treat.get_encoded_dataset(treat_raw, self.device)
        encoded_clin = enc_clin[0] if isinstance(enc_clin, tuple) else enc_clin
        encoded_treat = enc_treat[0] if isinstance(enc_treat, tuple) else enc_treat

        self.meta_clin.encoder.eval()
        for p in self.meta_clin.encoder.parameters():
            p.requires_grad_(False)

        # Freeze treatment-encoder
        self.meta_treat.encoder.eval()
        for p in self.meta_treat.encoder.parameters():
            p.requires_grad_(False)
        return encoded_treat, encoded_clin, actual_treat



class TabularDatasetPID(Dataset):
    """Prepare mixed‑type tabular tensors and missing‑value masks for clinical and treatment data.
    """

    def __init__(
        self,
        df_clin: pd.DataFrame,
        df_treat: pd.DataFrame,
        df_actual_treat: pd.DataFrame,
        meta_clin,
        meta_treat,
        seq_len: int = 1,
        id_field: str = "_id",
        return_mask: bool = False,
    ):
        self.df_clin = df_clin
        self.df_treat = df_treat
        self.df_actual_treat = df_actual_treat
        self.meta_clin = meta_clin
        self.meta_treat = meta_treat
        self.seq_len = seq_len
        self.id_field = id_field
        self.return_mask = return_mask

        # Freeze encoders
        for param in self.meta_clin.encoder.parameters():
            param.requires_grad = False
        for param in self.meta_treat.encoder.parameters():
            param.requires_grad = False
            
        self.meta_clin.encoder.eval()
        self.meta_treat.encoder.eval()
        
        # Get device from encoder
        device = next(self.meta_clin.encoder.parameters()).device
        
        with torch.no_grad():
            self.x_clin_num, self.x_clin_ord, self.mask_clin_raw, _ = self._preprocess_spec(self.df_clin, self.meta_clin.get_feature_spec(), self.meta_clin)
            self.x_treat_num, self.x_treat_ord, self.mask_treat_raw, _ = self._preprocess_spec(self.df_treat, self.meta_treat.get_feature_spec(), self.meta_treat)
            
            # Move tensors to the same device as encoders
            self.x_clin_num = self.x_clin_num.to(device)
            self.x_clin_ord = self.x_clin_ord.to(device)
            self.x_treat_num = self.x_treat_num.to(device)
            self.x_treat_ord = self.x_treat_ord.to(device)
            
            self.x_clin_emb = self.meta_clin.encoder(self.x_clin_num, self.x_clin_ord)
            self.x_treat_emb = self.meta_treat.encoder(self.x_treat_num, self.x_treat_ord)
            
            # Use the detailed feature-level masks directly instead of converting to embedding-level masks
            # This preserves the correct masking information for missing features
            self.mask_clin = self.mask_clin_raw.to(device)
            self.mask_treat = self.mask_treat_raw.to(device)

        # Group rows by patient id
        groups = self.df_clin.groupby(self.id_field).indices
        self.patient_ids = list(groups.keys())
        self.group_indices = [np.asarray(idx_list) for idx_list in groups.values()]
        self.num_patients = len(self.group_indices)

    def get_actual_treatment(self, df: pd.DataFrame) -> torch.Tensor:
        df_actual_treatemnt = df[["episodes.surgery", "episodes.chemotherapy", "episodes.radiotherapy"]]
        return torch.from_numpy(df_actual_treatemnt.to_numpy(dtype="int64").reshape(-1, 3))
    
    def _preprocess_spec(self, df: pd.DataFrame, spec: Dict, meta) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Use the EXACT same preprocessing logic as train_encoder.py TabularDataset
        to ensure perfect alignment between training and inference.
        """
        # Derive columns from spec (same as train_encoder.py)
        num_cols = [c for c, t in spec.items() if t == "numeric"]
        oh_cols = [c for c, t in spec.items() if t == "one_hot"]
        ord_cols = [c for c, t in spec.items() if t == "ordinal" or isinstance(t, dict)]

        # Get metadata for consistency
        meta_json = meta.import_metadata()
        saved_numeric_ranges = meta_json.get("numeric_ranges", {})
        saved_one_hot_mappings = meta_json.get("one_hot_mappings", {})
        saved_ord_mappings = meta_json.get("ord_mappings", {})

        # Build ordinal maps (same logic as train_encoder.py)
        ord_maps = []
        for col in ord_cols:
            t_val = spec[col]
            if isinstance(t_val, dict):
                # Use the same sorting logic as train_encoder.py
                shifted = {v: i+1 for i, v in enumerate(sorted(t_val.keys()))}
                ord_maps.append(shifted)
            else:
                # Use saved mapping if available, otherwise build fresh
                if col in saved_ord_mappings:
                    ord_maps.append(saved_ord_mappings[col])
                else:
                    uniq = sorted(df[col].dropna().unique())
                    ord_maps.append({v: i+1 for i, v in enumerate(uniq)})

        # NUMERIC PROCESSING (same as train_encoder.py with normalization)
        x_num = df[num_cols].to_numpy(dtype="float32", copy=True) if num_cols else np.empty((len(df), 0), dtype="float32")
        num_mask = ~np.isnan(x_num) if num_cols else np.empty_like(x_num, dtype=bool)
        
        # Apply normalization using saved ranges (CRITICAL for alignment)
        for i, col in enumerate(num_cols):
            if col in saved_numeric_ranges:
                col_min, col_max = saved_numeric_ranges[col]
                if col_max > col_min:
                    x_num[:, i] = (x_num[:, i] - col_min) / (col_max - col_min)
        
        x_num[np.isnan(x_num)] = 0.0

        # ONE-HOT PROCESSING (same as train_encoder.py)
        if oh_cols:
            x_oh_parts, mask_parts = [], []
            for col in oh_cols:
                # Use saved one-hot mappings for consistency
                if col in saved_one_hot_mappings:
                    categories = saved_one_hot_mappings[col]
                    # Use CategoricalDtype to ensure order and handle unseen categories
                    cat_dtype = pd.api.types.CategoricalDtype(categories=categories, ordered=True)
                    dummies = pd.get_dummies(df[col].astype(cat_dtype), prefix=col).astype("float32")
                else:
                    # Fallback to standard processing if no saved mapping
                    dummies = pd.get_dummies(df[col], dummy_na=False, prefix=col).astype("float32")
                
                x_oh_parts.append(dummies.to_numpy())
                col_mask = (~pd.isna(df[col])).to_numpy(dtype="float32")[:, None]
                mask_parts.append(np.repeat(col_mask, dummies.shape[1], axis=1))
            
            x_oh = np.concatenate(x_oh_parts, axis=1)
            oh_mask = np.concatenate(mask_parts, axis=1)
            x_numeric_oh = np.concatenate([x_num, x_oh], axis=1)
            mask_numeric = np.concatenate([num_mask, oh_mask], axis=1).astype("float32")
        else:
            x_numeric_oh = x_num
            mask_numeric = num_mask.astype("float32")
                
        # ordinal -------------------------------------------------------------
        if ord_cols:
            ord_arrays = []
            ord_mask_arrays = []
            for col, mp in zip(ord_cols, ord_maps):
                # Create mask based on original missing values, not encoded values
                orig_mask = (~pd.isna(df[col])).to_numpy().astype("float32")
                ord_mask_arrays.append(orig_mask[:, None])
                
                # Encode ordinals (missing values become 0)
                encoded = df[col].map(mp).fillna(0).astype("int64").to_numpy()
                ord_arrays.append(encoded[:, None])
            
            x_ord = np.concatenate(ord_arrays, axis=1)
            mask_ord = np.concatenate(ord_mask_arrays, axis=1)
            # The full target includes numeric, one-hot, and ordinal features
            x_target = np.concatenate([x_numeric_oh, x_ord.astype("float32")], axis=1)
            mask_target = np.concatenate([mask_numeric, mask_ord], axis=1)
        else:
            x_target = x_numeric_oh
            mask_target = mask_numeric
            x_ord = np.empty((len(df), 0), dtype="int64")

        # Return format for encoder: (numeric+one-hot, ordinals, full_mask, input_dim)
        num_oh_dim = x_numeric_oh.shape[1]
        return (
            torch.from_numpy(x_numeric_oh).float(),
            torch.from_numpy(x_ord).long(),
            torch.from_numpy(mask_target).float(),
            num_oh_dim,
        )

    def __len__(self) -> int:
        # Assuming clinical and treatment data have the same number of rows.
        return self.num_patients

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a dict of tensors for a single patient sequence."""
        indices = self.group_indices[idx]
        
        # Pad or truncate sequence
        if len(indices) > self.seq_len:
            indices = indices[:self.seq_len]
        elif len(indices) < self.seq_len:
            pad_len = self.seq_len - len(indices)
            indices = np.pad(indices, (0, pad_len), 'edge')

        # Clinical data
        x_clin_emb = self.x_clin_emb[indices]
        
        # Treatment data
        x_treat_emb = self.x_treat_emb[indices]
        
        # Get actual treatment for the sequence
        actual_treatment = self.get_actual_treatment(self.df_actual_treat.iloc[indices])

        result = {
            "x_clin": x_clin_emb,
            "x_treat": x_treat_emb,
            "actual_treatment": actual_treatment,
        }
        
        if self.return_mask:
            # Add masks for clinical and treatment data
            result["mask_clin"] = self.mask_clin[indices]
            result["mask_treat"] = self.mask_treat[indices]
        
        return result

