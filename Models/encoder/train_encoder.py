# coding: utf-8
"""
Train a tabular auto‑encoder with **per‑column encodings** declared in YAML.

Supported column types
----------------------
* **numeric**   – kept as float (NaN ⇒ 0) + mask
* **one_hot**   – pandas `get_dummies` + mask per original cell
* **ordinal**   – integer index fed through learnable embeddings (+ mask)

Minimal YAML example
--------------------
```yaml
seed: 42
features:
  country: one_hot      # categorical → one‑hot
  segment: ordinal      # categorical → embedding
  gender: ordinal
  age: numeric
  income: numeric
  spend_last_month: numeric
batch_size: 256
hidden_dim: 384
learning_rate: 1.0e-3
epochs: 40
patience: 6
save_dir: "saved_models"
log_every: 1
```
Run with:
```bash
python train_encoder_decoder.py --config_file configs/config.yaml
```
The model trains on **all** rows; validation is reconstruction error on those
rows (use a separate dataset if you need a true hold‑out test).
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import shutil
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))
# ╔══════════════════════════════════════════════════════════════════════════╗
# ┃ Optional project‑local imports                                          ┃
# ╚══════════════════════════════════════════════════════════════════════════╝



from utils.helpers import MongoExtractor  # type: ignore

# ---------------------------------------------------------------------------
# Fallback Encoder/Decoder (remove if you have real models in models.*)      
# ---------------------------------------------------------------------------

from models.encoder import Encoder  # type: ignore
from models.decoder import Decoder  # type: ignore

    



# ---------------------------------------------------------------------------
# Utility to save checkpoints                                                
# ---------------------------------------------------------------------------

from models.utils import save_encoder_decoder  # type: ignore

# ╔══════════════════════════════════════════════════════════════════════════╗
# ┃ Dataset                                                                  ┃
# ╚══════════════════════════════════════════════════════════════════════════╝
class TabularDataset(Dataset):
    """Prepare mixed‑type tabular tensors and missing‑value masks."""

    def __init__(self, df: pd.DataFrame, feature_spec: Dict[str, str]):
        super().__init__()
        self.spec = feature_spec
        self.num_cols = [c for c, t in self.spec.items() if t == "numeric"]
        self.oh_cols = [c for c, t in self.spec.items() if t == "one_hot"]
        # Now ord_cols include features whose spec is "ordinal" or a dict (explicit mapping)
        self.ord_cols = [c for c, t in self.spec.items() if t == "ordinal" or isinstance(t, dict)]

        # Build maps for ordinal columns: use provided mapping if dict; otherwise,
        # build mapping that reserves 0 for missing and maps categories to 1,...,N.
        self.ord_maps: List[Dict] = []
        for col in self.ord_cols:
            spec_val = self.spec[col]
            if isinstance(spec_val, dict):
                # Shift provided mapping: sort keys to have deterministic order.
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
        # Pre‑process: update one_hot loop to store mappings
        self.x_num_oh, self.x_ord, self.mask, self.input_dim, self.output_dim = self._preprocess(df)

    # ------------- helpers --------------------------------------------------
    def _encode_ord(self, s: pd.Series, mp: Dict[str, int]) -> np.ndarray:

        return s.map(mp).fillna(0).astype("int64").to_numpy()

    def _preprocess(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, int, int]:
        # numeric -------------------------------------------------------------
        x_num = df[self.num_cols].to_numpy(dtype="float32", copy=True) if self.num_cols else np.empty((len(df), 0), dtype="float32")
        num_mask = ~np.isnan(x_num) if self.num_cols else np.empty_like(x_num, dtype=bool)
        for i, col in enumerate(self.num_cols):
            col_min, col_max = self.numeric_ranges[col]
            if col_max > col_min:
                x_num[:, i] = (x_num[:, i] - col_min) / (col_max - col_min)
        
        x_num[np.isnan(x_num)] = 0.0

        # one‑hot -------------------------------------------------------------
        if self.oh_cols:
            x_oh_parts, mask_parts = [], []
            for col in self.oh_cols:
                dummies = pd.get_dummies(df[col], dummy_na=False, prefix=col).astype("float32")
                x_oh_parts.append(dummies.to_numpy())
                self.one_hot_mappings[col] = [col_name.split('_', 1)[1] for col_name in dummies.columns]
                col_mask = (~pd.isna(df[col])).to_numpy(dtype="float32")[:, None]
                mask_parts.append(np.repeat(col_mask, dummies.shape[1], axis=1))
            x_oh    = np.concatenate(x_oh_parts, axis=1)
            oh_mask = np.concatenate(mask_parts, axis=1)
            x_numeric_oh = np.concatenate([x_num, x_oh], axis=1)
            mask_numeric  = np.concatenate([num_mask, oh_mask], axis=1).astype("float32")
        else:
            x_numeric_oh = x_num
            mask_numeric  = num_mask.astype("float32")
        
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
            # Our reconstruction target now is numeric+one_hot and ordinal (as float)
            x_target = np.concatenate([x_numeric_oh, x_ord.astype("float32")], axis=1)
            mask_target = np.concatenate([mask_numeric, mask_ord], axis=1)
        else:
            x_target = x_numeric_oh
            mask_target = mask_numeric
            x_ord = None

        # Return the full target and mask if ordinals should be reconstructed
        if self.ord_cols:
            # Include ordinals in reconstruction
            encoder_input_dim = x_numeric_oh.shape[1]  # numeric + one-hot only (for encoder)
            decoder_output_dim = x_target.shape[1]     # numeric + one-hot + ordinal (for decoder)
            return (
                torch.from_numpy(x_target).float(),  # Full target: numeric + one-hot + ordinal
                x_ord if x_ord is None else torch.from_numpy(x_ord).long(),  # x_ord for embeddings
                torch.from_numpy(mask_target).float(),  # Full mask: numeric + one-hot + ordinal
                encoder_input_dim,   # Encoder input dim (numeric + one-hot)
                decoder_output_dim,  # Decoder output dim (numeric + one-hot + ordinal)
            )
        else:
            # Only numeric + one-hot features
            num_oh = x_numeric_oh
            mask_num_oh = mask_numeric
            num_oh_dim = num_oh.shape[1]
            return (
                torch.from_numpy(num_oh).float(),  # x_num_oh: numeric + one-hot inputs
                x_ord if x_ord is None else torch.from_numpy(x_ord).long(),  # x_ord for embeddings
                torch.from_numpy(mask_num_oh).float(),  # mask for numeric + one-hot
                num_oh_dim,   # Encoder input dim
                num_oh_dim,   # Decoder output dim (same when no ordinals)
            )

    # dataset protocol -------------------------------------------------------
    def __len__(self) -> int:
        return self.x_num_oh.shape[0]

    def __getitem__(self, idx: int):
        if self.x_ord is not None:
            return self.x_num_oh[idx], self.x_ord[idx], self.mask[idx]

        return self.x_num_oh[idx], torch.zeros(0, dtype=torch.long), self.mask[idx]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ┃ Training utilities                                                      ┃
# ╚══════════════════════════════════════════════════════════════════════════╝

def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    return (diff ** 2).sum() / mask.sum()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# evaluation loop -----------------------------------------------------------

def evaluate(enc: nn.Module, dec: nn.Module, loader: DataLoader, device: torch.device) -> float:
    enc.eval(); dec.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x_target, x_ord, mask in loader:
            x_target, mask = x_target.to(device), mask.to(device)
            x_ord = x_ord.to(device) if x_ord is not None else None
            
            # Extract numeric+one-hot part for encoder input
            n_numeric = len(loader.dataset.num_cols)
            n_onehot = sum(len(loader.dataset.one_hot_mappings.get(col, [])) for col in loader.dataset.oh_cols)
            x_num_oh = x_target[:, :n_numeric + n_onehot]
            
            recon = dec(enc(x_num_oh, x_ord))
            total += masked_mse(recon, x_target, mask).item() * x_target.size(0)
            n += x_target.size(0)
    return total / n

# training loop -------------------------------------------------------------

def train(
    enc: nn.Module,
    dec: nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    epochs: int,
    device: torch.device,
    lr: float,
    patience: int,
    log_every: int,
    save_dir: Path,
    feature_spec: Dict[str, str],
    model_params: Dict,
    disable_early_stopping: bool = False,  # added parameter
) -> None:
    opt = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=lr)
    best_val, wait = float("inf"), 0
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt = save_dir / "encoder_decoder.pt"

    for epoch in range(1, epochs + 1):
        enc.train(); dec.train(); running, seen = 0.0, 0
        for x_target, x_ord, mask in train_loader:
            x_target, mask = x_target.to(device), mask.to(device)
            x_ord = x_ord.to(device) if x_ord is not None else None
            
            # Extract numeric+one-hot part for encoder input
            n_numeric = len(train_loader.dataset.num_cols)
            n_onehot = sum(len(train_loader.dataset.one_hot_mappings.get(col, [])) for col in train_loader.dataset.oh_cols)
            x_num_oh = x_target[:, :n_numeric + n_onehot]
            
            opt.zero_grad()
            loss = masked_mse(dec(enc(x_num_oh, x_ord)), x_target, mask)
            loss.backward(); opt.step()
            running += loss.item() * x_target.size(0); seen += x_target.size(0)
        train_loss = running / seen
        val_loss = evaluate(enc, dec, eval_loader, device)
        if epoch % log_every == 0:
            print(f"Epoch {epoch:>3d}: train={train_loss:.8f} │ val={val_loss:.8f}")
        # early‑stop & checkpoint
        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0

            save_encoder_decoder(
                enc,
                dec,
                ckpt,
                feature_spec=feature_spec,
                one_hot_mappings=train_loader.dataset.one_hot_mappings,
                ord_mappings={c: mp for c, mp in zip(train_loader.dataset.ord_cols, train_loader.dataset.ord_maps)},
                numeric_ranges=getattr(train_loader.dataset, "numeric_ranges", None),
                model_params=model_params,
            )
        else:
            wait += 1
            if not disable_early_stopping and wait >= patience:  # modified condition
                print(f"Early stopping (best val={best_val:.4f})")
                break

# ╔══════════════════════════════════════════════════════════════════════════╗
# ┃ CLI                                                                     ┃
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_yaml(fp: os.PathLike) -> Dict:
    with open(fp, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> None:
    parser = argparse.ArgumentParser("Train a mixed‑type auto‑encoder")
    parser.add_argument("--config_file", required=True, help="Path to YAML config file")
    parser.add_argument("--resume", type=str, default=None, help="Existing checkpoint to resume from")
    parser.add_argument("--disable_early_stopping", action="store_true", help="Disable early stopping")  # added argument
    parser.add_argument("--clean_output", action="store_true", help="Delete all files in output folder before training")  # new argument
    args = parser.parse_args()

    # ---------------- configuration & RNG ----------------------------------
    cfg = load_yaml(args.config_file)
    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------- data acquisition -------------------------------------
    if os.getenv("MONGO_URI"):
        extractor = MongoExtractor(
            connection_string=os.getenv("MONGO_URI"),
            database_name=os.getenv("MONGO_DB"),
            collection_name=os.getenv("MONGO_COLLECTION"),
            config_file=args.config_file,
        )
        df = extractor.get_dataframe()
    else:
        raise RuntimeError("Environment variable MONGO_URI is required to fetch data from MongoDB.")

    # ---------------- dataset / loaders ------------------------------------
    feature_spec: Dict[str, str] = cfg["data_config"]["features"]
    
    # CRITICAL FIX: Reorder DataFrame columns to match feature_spec order
    # MongoDB returns columns in alphabetical order, but our encoding/decoding
    # logic assumes they're in the same order as the YAML feature specification
    df = MongoExtractor.reorder_dataframe_columns(df, feature_spec)
    
    dataset = TabularDataset(df, feature_spec)
    batch = cfg.get("batch_size", 256)
    dl_train = DataLoader(dataset, batch_size=batch, shuffle=True, drop_last=True)
    dl_eval = DataLoader(dataset, batch_size=batch * 4, shuffle=False)

    # ---------------- model init -------------------------------------------
    ord_cards = [len(mp) + 1 for mp in dataset.ord_maps]
    hidden_dim = cfg.get("hidden_dim", 384)

    encoder = Encoder(dataset.input_dim, hidden_dim, ord_cards).to(device)
    decoder = Decoder(hidden_dim, dataset.output_dim).to(device)

    # ---------------- resume? ----------------------------------------------
    if args.resume and Path(args.resume).exists():
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        print(f"Resumed from {args.resume}")

    # ---------------- training ---------------------------------------------
    save_dir = Path(cfg.get("save_dir", "saved_models"))
    if args.clean_output:
        if save_dir.exists():
            for f in save_dir.iterdir():
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            print(f"Cleaned output folder: {save_dir}")
        else:
            print(f"Output folder {save_dir} does not exist. Nothing to clean.")
            
    train(
        enc=encoder,
        dec=decoder,
        train_loader=dl_train,
        eval_loader=dl_eval,
        epochs=cfg.get("epochs", 40),
        device=device,
        lr=cfg.get("learning_rate", 1.0e-3),
        patience=cfg.get("patience", 6),
        log_every=cfg.get("log_every", 1),
        save_dir=save_dir,
        feature_spec=feature_spec,
        model_params={"hidden_dim": hidden_dim, 
                      "input_dim": dataset.input_dim, 
                      "batch_size": batch, 
                      "patience": cfg.get("patience", 6), 
                      "seed": cfg.get("seed", 42), 
                      "learning_rate": cfg.get("learning_rate", 1.0e-3),
                      "epochs": cfg.get("epochs", 40),
                      "device": str(device),
                       "ord_cards": ord_cards,
                      },
        disable_early_stopping=args.disable_early_stopping 
    )


if __name__ == "__main__":
    main()
