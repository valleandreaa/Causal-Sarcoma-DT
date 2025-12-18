import numpy as np
import pandas as pd
import torch

ONE_HOT_MIN_PROB = 0.1

def decode_embedding(
    t: torch.Tensor,
    feature_spec: dict,
    one_hot_map: dict,
    ord_map: dict,
    numeric_ranges=None,
    one_hot_argmax: bool = True
) -> pd.DataFrame:
    """Decode an embedding tensor into a pandas DataFrame of original features in the correct order."""
    # Ensure the tensor is 2D for DataFrame creation
    if t.dim() == 3:
        # If tensor is (batch_size, seq_len, features), reshape to (batch_size * seq_len, features)
        recon = t.detach().cpu().view(-1, t.size(-1)).numpy()
    else:
        recon = t.detach().cpu().numpy()
        if recon.ndim == 1:
            recon = np.expand_dims(recon, axis=0)

    # Get feature lists in the original order
    numeric_cols = [c for c, k in feature_spec.items() if k == "numeric"]
    one_hot_features = [c for c, k in feature_spec.items() if k == "one_hot"]
    ord_features = [c for c, k in feature_spec.items() if k == "ordinal" or isinstance(k, dict)]
    
    n_numeric = len(numeric_cols)
    n_onehot_categories = sum(len(one_hot_map.get(f, [])) for f in one_hot_features)
    total_numeric_onehot = n_numeric + n_onehot_categories
    ord_count = len(ord_features)

    # Split the reconstruction tensor by type (as stored in the encoder output)
    recon_numeric = recon[:, :total_numeric_onehot]
    if recon.shape[1] >= total_numeric_onehot + ord_count:
        recon_ordinal = recon[:, total_numeric_onehot : total_numeric_onehot + ord_count]
    else:
        recon_ordinal = np.empty((recon.shape[0], 0))

    # Process numeric features
    df_numeric_values = recon_numeric[:, :n_numeric]
    numeric_data = {}
    for i, col in enumerate(numeric_cols):
        vals = df_numeric_values[:, i]
        if numeric_ranges and col in numeric_ranges:
            mn, mx = numeric_ranges[col]
            vals = vals * (mx - mn) + mn
        numeric_data[col] = vals

    # Process one-hot features
    one_hot_array = recon_numeric[:, n_numeric:]
    onehot_data = {}
    if one_hot_array.shape[1] != 0:
        idx = 0
        for feat in one_hot_features:
            mapping = one_hot_map.get(feat)
            if mapping is None:
                onehot_data[feat] = ["UNK"] * recon.shape[0]
                continue
            width = len(mapping)
            slice_ = one_hot_array[:, idx : idx + width]
            if slice_.size != 0:
                if one_hot_argmax:
                    # Aggregate: select single best category per variable
                    indices = slice_.argmax(axis=1)
                    max_probs = slice_.max(axis=1)
                    vals = []
                    for i_idx, p in zip(indices, max_probs):
                        if np.isnan(p) or p < ONE_HOT_MIN_PROB or i_idx >= len(mapping):
                            vals.append("UNK")
                        else:
                            vals.append(mapping[i_idx])
                    onehot_data[feat] = vals
                else:
                    # Expanded: create separate binary columns for each category
                    for i, category in enumerate(mapping):
                        category_probs = slice_[:, i]
                        binary_vals = []
                        for prob in category_probs:
                            if np.isnan(prob):
                                binary_vals.append('UNK')
                            else:
                                binary_vals.append(prob)
                        onehot_data[f"{feat}_{category}"] = binary_vals
            else:
                onehot_data[feat] = ["UNK"] * recon.shape[0]
            idx += width

    # Process ordinal features
    ordinal_data = {}
    if ord_features and recon_ordinal.shape[1] >= len(ord_features) and recon_ordinal.shape[1] > 0:
        for i, feat in enumerate(ord_features):
            spec_val = feature_spec.get(feat)
            if isinstance(spec_val, dict):
                rev_map = {v: k for k, v in spec_val.items()}
            else:
                mapping = ord_map.get(feat)
                if mapping is None:
                    mapping = {j: f"cat_{j}" for j in range(1, 10)}
                try:
                    rev_map = {int(v): k for k, v in mapping.items()}
                except ValueError:
                    rev_map = mapping
            preds = recon_ordinal[:, i]
            indices = np.rint(preds).astype(int)
            ordinal_data[feat] = [rev_map.get(idx, "UNK") for idx in indices]
    else:
        # If no ordinal features or dimension mismatch, create empty values for each ordinal feature
        for feat in ord_features:
            ordinal_data[feat] = ["UNK"] * recon.shape[0]

    # Reconstruct DataFrame in the ORIGINAL feature specification order
    all_data = {}
    all_data.update(numeric_data)
    all_data.update(onehot_data)
    all_data.update(ordinal_data)
    
    # Create DataFrame with columns in the original feature_spec order
    ordered_columns = list(feature_spec.keys())
    df_result = pd.DataFrame(all_data)[ordered_columns]
    
    return df_result