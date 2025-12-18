import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from models.encoder import Encoder  
from models.decoder import Decoder  # type: ignore
from models.decoder_embedders import decode_embedding  # type: ignore
from utils.helpers import MongoExtractor  # type: ignore
import math
from encoder.train_encoder import TabularDataset, set_seed  # type: ignore
ONE_HOT_MIN_PROB = 0.5
def load_yaml(fp: os.PathLike) -> dict:
    with open(fp, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)

# New function: replace elements with np.nan when the mask is 0
def apply_mask_with_none(recon: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
    recon_masked = recon.clone()
    recon_masked[x_mask == 0] = float('nan')
    return recon_masked


def evaluate_quality(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    dataset: TabularDataset,
    loader: DataLoader,
    device: torch.device,
):
    """Return reconstruction MSE for numeric features, one-hot accuracy, and ordinal accuracy."""
    # Calculate dimensions for different feature types
    n_numeric = len(dataset.num_cols)
    n_onehot_categories = sum(len(dataset.one_hot_mappings.get(col, [])) for col in dataset.oh_cols)
    ord_count = len(dataset.ord_cols) if hasattr(dataset, 'ord_cols') else 0
    total_numeric_onehot = n_numeric + n_onehot_categories
    
    # Initialize accumulators
    numeric_mse_total, numeric_mask_total = 0.0, 0.0
    onehot_correct, onehot_total = 0.0, 0.0
    ord_mse_total, ord_mask_total = 0.0, 0.0

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for x_target, x_ord, mask in loader:
            x_target = x_target.to(device)
            x_ord = x_ord.to(device) if x_ord is not None else None
            mask = mask.to(device)  # This is the full mask now

            # Extract numeric+one-hot part for encoder input
            n_numeric = len(dataset.num_cols)
            n_onehot = sum(len(dataset.one_hot_mappings.get(col, [])) for col in dataset.oh_cols)
            x_num_oh = x_target[:, :n_numeric + n_onehot]

            recon = decoder(encoder(x_num_oh, x_ord))
            
            # Evaluate numeric features (MSE)
            if n_numeric > 0:
                recon_numeric = recon[:, :n_numeric]
                target_numeric = x_target[:, :n_numeric]  # Extract from full target
                numeric_mask = mask[:, :n_numeric]
                numeric_mse_total += ((recon_numeric - target_numeric) ** 2 * numeric_mask).sum().item()
                numeric_mask_total += numeric_mask.sum().item()
            
            # Evaluate one-hot features (categorical accuracy)
            if n_onehot_categories > 0:
                recon_onehot = recon[:, n_numeric:total_numeric_onehot]
                target_onehot = x_target[:, n_numeric:total_numeric_onehot]  # Extract from full target
                onehot_mask = mask[:, n_numeric:total_numeric_onehot]
                
                # Convert to categorical predictions by finding argmax within each one-hot group
                current_idx = 0
                for col in dataset.oh_cols:
                    n_categories = len(dataset.one_hot_mappings.get(col, []))
                    if n_categories > 0:
                        
                        recon_slice = recon_onehot[:, current_idx:current_idx + n_categories]
                        target_slice = target_onehot[:, current_idx:current_idx + n_categories]
                        mask_slice = onehot_mask[:, current_idx:current_idx + n_categories]
                        
                        valid_samples = mask_slice.sum(dim=1) > 0
                        if valid_samples.sum() > 0:
                            pred_categories = recon_slice[valid_samples].argmax(dim=1)
                            true_categories = target_slice[valid_samples].argmax(dim=1)
                            onehot_correct += (pred_categories == true_categories).sum().item()
                            onehot_total += valid_samples.sum().item()
                        
                        current_idx += n_categories

            # Evaluate ordinal features (MSE on float values)
            if ord_count > 0 and recon.shape[1] > total_numeric_onehot:
                recon_ord = recon[:, total_numeric_onehot:total_numeric_onehot + ord_count]
                target_ord = x_target[:, total_numeric_onehot:total_numeric_onehot + ord_count]  # Extract from full target
                # Use ordinal mask from full mask
                ord_mask = mask[:, total_numeric_onehot:total_numeric_onehot + ord_count]
                ord_mse_total += ((recon_ord - target_ord) ** 2 * ord_mask).sum().item()
                ord_mask_total += ord_mask.sum().item()

    # Calculate final metrics
    numeric_mse = numeric_mse_total / numeric_mask_total if numeric_mask_total > 0 else float('nan')
    onehot_acc = onehot_correct / onehot_total if onehot_total > 0 else float('nan')
    ord_mse = ord_mse_total / ord_mask_total if ord_mask_total > 0 else float('nan')
    
    return numeric_mse, onehot_acc, ord_mse

def main():
    parser = argparse.ArgumentParser("Reconstruct original dataset using trained autoencoder")
    parser.add_argument("--config_file", required=True, help="Path to YAML config file")
    parser.add_argument("--encoder_file", required=True, help="Path to the saved encoder model file")
    parser.add_argument("--decoder_file", required=True, help="Path to the saved decoder model file")
    parser.add_argument("--metadata_file", required=True, help="Path to metadata JSON file with feature_spec")
    parser.add_argument("--output", default="reconstructed_data.csv", help="Path to save reconstructed data")
    args = parser.parse_args()

    cfg = load_yaml(args.config_file)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load metadata to obtain the feature specification
    with open(args.metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    feature_spec = metadata.get("feature_spec")
    if feature_spec is None:
        # Fallback to configuration if not found in metadata
        feature_spec = cfg["data_config"]["features"]

    # Load data using MongoExtractor
    if os.getenv("MONGO_URI"):
        extractor = MongoExtractor(
            connection_string=os.getenv("MONGO_URI"),
            database_name=os.getenv("MONGO_DB"),
            collection_name=os.getenv("MONGO_COLLECTION"),
            config_file=args.config_file,
        )
        df = extractor.get_dataframe()
    else:
        raise RuntimeError("Environment variable MONGO_URI is required.")

    # CRITICAL FIX: Reorder DataFrame columns to match feature_spec order
    # MongoDB returns columns in alphabetical order, but our encoding/decoding
    # logic assumes they're in the same order as the YAML feature specification
    df = MongoExtractor.reorder_dataframe_columns(df, feature_spec)

    dataset = TabularDataset(df, feature_spec)
    batch_size = cfg.get("batch_size", 256)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Initialize models using dataset dimensions
    ord_cards = [len(mp) + 1 for mp in dataset.ord_maps] 
    hidden_dim = cfg.get("hidden_dim", 384)
    encoder = Encoder(dataset.input_dim, hidden_dim, ord_cards).to(device)
    decoder = Decoder(hidden_dim, dataset.output_dim).to(device)

    # Load model state from provided files
    encoder.load_state_dict(torch.load(args.encoder_file, map_location=device))
    decoder.load_state_dict(torch.load(args.decoder_file, map_location=device))
    encoder.eval(); decoder.eval()

    # Evaluate reconstruction quality
    numeric_mse, onehot_acc, ord_mse = evaluate_quality(encoder, decoder, dataset, dl, device)
    print(f"Numeric reconstruction MSE: {numeric_mse:.8f}")
    if not math.isnan(onehot_acc):
        print(f"One-hot reconstruction accuracy: {onehot_acc:.8f}")
    else:
        print("No one-hot features to evaluate")
    if not math.isnan(ord_mse):
        print(f"Ordinal reconstruction MSE: {ord_mse:.8f}")
    else:
        print("No ordinal features to evaluate")

    # Reconstruct dataset
    recon_list = []
    target_ord_list = []  # Store target ordinal values for masking
    mask_list = []   # Store masks for proper masking
    
    with torch.no_grad():
        for x_target, x_ord, x_mask in dl:
            x_target = x_target.to(device)
            x_ord = x_ord.to(device) if x_ord is not None else None
            x_mask = x_mask.to(device) if x_mask is not None else None

            # Extract numeric+one-hot part for encoder input
            n_numeric = len([c for c, t in feature_spec.items() if t == "numeric"])
            n_onehot = sum(len(metadata.get("one_hot_mappings", {}).get(f, [])) for f in [c for c, t in feature_spec.items() if t == "one_hot"])
            x_num_oh = x_target[:, :n_numeric + n_onehot]

            # Encode and decode (no masking of inputs during reconstruction)
            z = encoder(x_num_oh, x_ord)
            recon = decoder(z)
            
            recon_list.append(recon.cpu())
            
            # Extract ordinal values from target for masking purposes
            ord_count = len([c for c, t in feature_spec.items() if t == "ordinal" or isinstance(t, dict)])
            if ord_count > 0 and x_target.shape[1] > n_numeric + n_onehot:
                target_ord = x_target[:, n_numeric + n_onehot:n_numeric + n_onehot + ord_count]
                target_ord_list.append(target_ord.cpu())
            
            if x_mask is not None:
                mask_list.append(x_mask.cpu())
    
    recon_tensor = torch.cat(recon_list, dim=0)
    target_ord_tensor = torch.cat(target_ord_list, dim=0) if target_ord_list else None
    mask_tensor = torch.cat(mask_list, dim=0) if mask_list else None
    
    # Apply masking to reconstruction
    if mask_tensor is not None:
        recon_tensor = apply_mask_with_none(recon_tensor, mask_tensor)
    
    # Use the corrected decode_embedding function to maintain feature order
    print("Using decode_embedding function for proper feature ordering...")
    
    # Get mappings from metadata
    one_hot_mappings = metadata.get("one_hot_mappings", {})
    ord_mappings = metadata.get("ord_mappings", {})
    numeric_ranges = metadata.get("numeric_ranges", {})
    
    # Decode each row using decode_embedding
    decoded_rows = []
    for i in range(recon_tensor.shape[0]):
        row_tensor = recon_tensor[i:i+1]  # Keep batch dimension
        df_decoded = decode_embedding(
            row_tensor,
            feature_spec,
            one_hot_mappings,
            ord_mappings,
            numeric_ranges,
            one_hot_argmax=True
        )
        decoded_rows.append(df_decoded)
    
    # Concatenate all decoded rows
    final_df = pd.concat(decoded_rows, ignore_index=True)
    
    # Apply masking to set missing values to None
    if mask_tensor is not None:
        for i, feature in enumerate(feature_spec.keys()):
            if feature in final_df.columns:
                for row_idx in range(len(final_df)):
                    # Find the corresponding mask position for this feature
                    # This is complex since features are stored differently in the tensor
                    # For now, we'll rely on the decode_embedding function's NaN handling
                    pass
    
    final_df.to_csv(args.output, index=False)
    print(f"Reconstructed data saved to {args.output}")
    print(f"Final DataFrame shape: {final_df.shape}")
    print(f"Features in correct order: {list(final_df.columns)}")

if __name__ == "__main__":
    main()
