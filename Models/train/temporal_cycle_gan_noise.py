#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
temporal_cycle_gan.py
A two‑domain temporal CycleGAN for clinical → treatment‑output embeddings.
Author: <you> – 2025‑05‑12
"""
# --------------------------------------------------------------------- #
# 0. Imports
# --------------------------------------------------------------------- #


from __future__ import annotations

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from dataclasses import dataclass
from pathlib import Path
import argparse, yaml, random, time, os
from models.noise import generate_sequence_noise
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils.helpers import MongoExtractor
from dataset.tabular_dataset import TabularDatasetPID
import json
from models.utils  import MetadataHandler, save_temporal_cycle_gan, randomize_one_hot
from models.decoder import Decoder
from models.decoder_embedders import decode_embedding
import pandas as pd
from models.GANs import LSTMGenerator, LSTMDiscriminator
from monitoring.losses import CycleGANLossMonitor
from monitoring.loss_logger import LossLogger
from datetime import datetime

# --------------------------------------------------------------------- #
# 1. Configuration via YAML
# --------------------------------------------------------------------- #
@dataclass
class Config:
    # ---------- DATA ----------
    query_pipeline:   list
    seq_len:          int
    id_field:         str
    # ---------- MODEL ----------
    x_dim:  int
    y_dim:  int
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
    # ---------- NOISE ----------
    noise_dim:  int = 0
    noise_type: str = "gaussian"

    # ---------- DATA CONFIG ----------
    data_config: dict = None
    mode: str = "train"

def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # If data_config is present, pass it along
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
    raise ValueError(f"Unsupported criterion: {name}")


def load_decoder(meta_file: str, decoder_path: str, device: torch.device):
    """Load decoder using parameters stored in metadata."""
    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    params = meta.get("model_params", {})
    hidden_dim = params.get("hidden_dim")
    output_dim = params.get("input_dim")
    dec = Decoder(hidden_dim, output_dim).to(device)
    dec.load_state_dict(torch.load(decoder_path, map_location=device))
    dec.eval()
    feat_spec = meta.get("feature_spec")
    oh_map = meta.get("one_hot_mappings", {})
    ord_map = meta.get("ord_mappings", {})
    num_ranges = meta.get("numeric_ranges", {})
    return dec, feat_spec, oh_map, ord_map, num_ranges

def evaluate(
    Gx,
    Gy,
    Dx,
    Dy,
    loader,
    criterion_adv,
    criterion_cycle,
    criterion_id,
    lambda_cycle,
    lambda_id,
    device,
    noise_dim,
    noise_type,
):
    """Evaluate model on validation data."""
    prev_states = (Gx.training, Gy.training, Dx.training, Dy.training)
    Gx.eval(); Gy.eval(); Dx.eval(); Dy.eval()
    g_loss = d_loss = cyc_loss = id_loss = 0.0
    gx_loss = gy_loss = dx_loss = dy_loss = 0.0
    with torch.no_grad():
        for tr_real, cl_real, tr_actual in loader:
            cl_real = cl_real.to(device)
            tr_real = tr_real.to(device)
            tr_actual = tr_actual.to(device)

            tr_counter = randomize_one_hot(tr_actual)

            noise = generate_sequence_noise(
                cl_real.size(0),
                cl_real.size(1),
                noise_dim,
                noise_type=noise_type,
                device=device,
            )

            fake_y = Gx(torch.cat((cl_real, noise), dim=2), tr_counter)
            rec_x = Gy(fake_y, tr_actual)

            # Discriminator combines clinical embedding, treatment (real/fake), and treatment one-hot
            pred_fake_x = Dx(torch.cat((cl_real, fake_y, tr_counter), dim=2))
            pred_real_x = Dx(torch.cat((cl_real, tr_real, tr_actual), dim=2))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x, tr_actual), dim=2))
            pred_real_y = Dy(torch.cat((cl_real, tr_real, tr_actual), dim=2))

            loss_Dx = 0.5*(criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x)) +
                           criterion_adv(pred_real_x, torch.ones_like(pred_real_x)))
            loss_Dy = 0.5*(criterion_adv(pred_real_y, torch.ones_like(pred_real_y)) +
                           criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y)))

            loss_Gx_adv = criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_Gy_adv = criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y))

            cycle_l = criterion_cycle(rec_x, tr_real)
            id_l = torch.tensor(0.0, device=device)
            if lambda_id > 0:
                id_y = Gx(torch.cat((cl_real, noise), dim=2), tr_counter)
                id_x = Gy(id_y, tr_actual)
                id_l = criterion_id(id_x, tr_real)

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
    if prev_states[0]:
        Gx.train()
    if prev_states[1]:
        Gy.train()
    if prev_states[2]:
        Dx.train()
    if prev_states[3]:
        Dy.train()

    return (
        g_loss / n,
        d_loss / n,
        cyc_loss / n,
        id_loss / n,
        gx_loss / n,
        gy_loss / n,
        dx_loss / n,
        dy_loss / n,
    )

# --------------------------------------------------------------------- #
# 3. Dataset that embeds *two* token streams with frozen encoders
# --------------------------------------------------------------------- #
class MongoTemporalDataset(Dataset):
    """
    Mongo document schema (example):
      {
        clinical_tokens   : list[str]     # domain A
        treatment_tokens  : list[str]     # domain B
        treatment_id      : int           # conditioning 'c'
        split             : "train" | "val" | ...
      }
    """
    def __init__(self, cfg: Config, split: str, args):
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.args = args
        self.df = self.load_dataframe()
        self.meta_clin, self.meta_treat = self.load_metadata()
        self.train_loader = self.create_dataloader()

    def load_dataframe(self):
        # New function to load dataframe from MongoDB
        if os.getenv("MONGO_URI"):
            extractor = MongoExtractor(
                connection_string=os.getenv("MONGO_URI"),
                database_name=os.getenv("MONGO_DB"),
                collection_name=os.getenv("MONGO_COLLECTION"),
                config_file=self.args.config_file
            )
            df = extractor.get_dataframe()
            return df
        else:
            raise RuntimeError("Environment variable MONGO_URI is required to fetch data from MongoDB.")

    def load_metadata(self):
        meta_clin = MetadataHandler(self.args.metadata_clinical_file)
        meta_treat = MetadataHandler(self.args.metadata_treatment_file)
        return meta_clin, meta_treat

    def create_dataloader(self):
        # New function to create the dataloader using TabularDataset and metadata
        dataset = TabularDatasetPID(
            self.df,
            self.meta_clin,
            self.meta_treat,
            seq_len=self.cfg.seq_len,
            id_field=self.cfg.id_field,
        )

        return DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

# --------------------------------------------------------------------- #
# New train function (splitting training from the dataset class)
# --------------------------------------------------------------------- #
torch.autograd.set_detect_anomaly(True)
def train(cfg, args):
    ds = MongoTemporalDataset(cfg, split="train", args=args)
    base_dataset = TabularDatasetPID(
        ds.df,
        ds.meta_clin,
        ds.meta_treat,
        seq_len=cfg.seq_len,
        id_field=cfg.id_field,
    )

    # ---- split patients into train/validation -------------------------------
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

    val_patient_ids = [base_dataset.patient_ids[i] for i in val_indices]

    # Hyperparameters from configuration
    # Derive embedding dimension from encoded clinical embeddings to ensure consistency
    # encoded_clin has shape [n_samples, embed_dim]
    embedding_dim = base_dataset.encoded_clin.shape[1]
    hidden_dim = cfg.g_hidden
    num_layers = cfg.num_layers
    T = cfg.seq_len
    batch_size = cfg.batch_size
    num_epochs = cfg.epochs
    lr_g = cfg.lr_g
    lr_d = cfg.lr_d
    beta1 = cfg.beta1
    lambda_cycle = cfg.lambda_cyc
    lambda_id = cfg.lambda_id
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Network building blocks (as befor
    # Instantiate networks and loss functions
    # Instantiate generators with dynamic embed size and treat_dim for conditioning
    Gx = LSTMGenerator(
        input_dim=embedding_dim + cfg.noise_dim,
        cond_input_dim=cfg.treat_dim,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)
    Gy = LSTMGenerator(
        input_dim=embedding_dim,
        cond_input_dim=cfg.treat_dim,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers
    ).to(device)
    # Discriminator input dimension: two embeddings + treatment one-hot
    disc_input_dim = embedding_dim * 2 + cfg.treat_dim
    Dx = LSTMDiscriminator(input_dim=disc_input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    Dy = LSTMDiscriminator(input_dim=disc_input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    # Loss functions
    from models.losses import get_adversarial_loss

    criterion_adv = get_adversarial_loss(cfg.adv_loss)
    criterion_cycle = get_criterion(cfg.criterion)
    criterion_id = get_criterion(cfg.criterion)

    # Optimizers for generators and discriminators
    optimizer_G = get_optimizer(cfg.opt_gen, list(Gx.parameters()) + list(Gy.parameters()), lr_g, (beta1, cfg.beta2))
    optimizer_Dx = get_optimizer(cfg.opt_disc, Dx.parameters(), lr_d, (beta1, cfg.beta2))
    optimizer_Dy = get_optimizer(cfg.opt_disc, Dy.parameters(), lr_d, (beta1, cfg.beta2))
    
    timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_monitor = CycleGANLossMonitor(path= Path(f"saved_models/temporal_cycle_gan_noise_{timestemp}") / "cyclegan_losses.png")
    loss_logger = LossLogger(
        Path(f"saved_models/temporal_cycle_gan_noise_{timestemp}") / "losses.log"
    )

    # Load decoders for debug or counterfactual evaluation
    if cfg.mode == "debug" or args.check_counterfactuals:
        dec_clin, spec_clin, oh_clin, ord_clin, ranges_clin = load_decoder(
            args.metadata_clinical_file,
            args.decoder_clinical,
            device,
        )
        dec_treat, spec_treat, oh_treat, ord_treat, ranges_treat = load_decoder(
            args.metadata_treatment_file,
            args.decoder_treatment,
            device,
        )
    # Training loop (as before)
    for epoch in range(num_epochs):
        epoch_g_losses = []
        epoch_d_losses = []
        epoch_cycle_losses = []
        epoch_id_losses = []
        epoch_gx_losses = []
        epoch_gy_losses = []
        epoch_dx_losses = []
        epoch_dy_losses = []
        for i, (tr_real, cl_real, tr_actual) in enumerate(train_loader):

            # ——— 0) Move everything to device ———
            cl_real   = cl_real.to(device).detach()    # [B, T, E]
            tr_real   = tr_real.to(device).detach()    # [B, T, E]
            tr_actual = tr_actual.to(device).detach()  # [B, T, C]
            
            noise = generate_sequence_noise(
                cl_real.size(0),
                cl_real.size(1),
                cfg.noise_dim,
                noise_type=cfg.noise_type,
                device=device,
            )
            # Build a counterfactual one-hot
            tr_counter = randomize_one_hot(tr_actual)  # still on device

            # ——— 1) Discriminator update ———
            optimizer_Dx.zero_grad()
            optimizer_Dy.zero_grad()

            # Generate fakes and reconstructions, then detach for D
            fake_y_det = Gx(torch.cat((cl_real, noise), dim=2), tr_counter).detach()
            rec_x_det  = Gy(fake_y_det, tr_actual).detach()




            # def hook(module, grad_in, grad_out):
            #     print(f"[HOOK] {module}: grad_in = {tuple(g.shape if g is not None else None for g in grad_in)}")
            # for m in Gx.modules():
            #     m.register_full_backward_hook(hook)

            # D’s predictions on real vs. fake
            # Discriminator combines clinical embedding, treatment and one-hot
            pred_real_x = Dx(torch.cat((cl_real, tr_real, tr_actual), dim=2))
            pred_fake_x = Dx(torch.cat((cl_real, fake_y_det, tr_counter), dim=2))
            
            pred_real_y = Dy(torch.cat((cl_real, tr_real, tr_actual), dim=2))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x_det, tr_actual), dim=2))

            # D losses
            loss_Dx = 0.5*(criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x))+ 
                           criterion_adv(pred_real_x, torch.ones_like(pred_real_x)))
    
            loss_Dy = 0.5*(criterion_adv(pred_real_y, torch.ones_like(pred_real_y)) +
                            criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y)))
                        
        

            # Backward & step
            # disable cuDNN for this backward so RNN backward works outside of training mode
            
            (loss_Dx + loss_Dy).backward()
            optimizer_Dx.step()
            optimizer_Dy.step()

            # ——— 2) Generator update ———
            # (optional) freeze D’s weights to skip computing grads through them
            for p in Dx.parameters(): p.requires_grad_(False)
            for p in Dy.parameters(): p.requires_grad_(False)

            optimizer_G.zero_grad()

            # Fresh forward through G
            fake_y = Gx(torch.cat((cl_real, noise), dim=2), tr_counter)
            rec_x  = Gy(fake_y, tr_actual)


            # Discriminator opinions for generator loss, include one-hot
            pred_fake_x = Dx(torch.cat((cl_real, fake_y, tr_counter), dim=2))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x, tr_actual), dim=2))

            loss_Gx_adv = criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_Gy_adv = criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y))


            loss_cycle = criterion_cycle(rec_x, tr_real)

            # Identity loss (if used)
            loss_id = torch.tensor(0.0, device=device)
            if lambda_id > 0:
                id_y = Gx(torch.cat((cl_real, noise), dim=2), tr_counter)   # A→A
                id_x = Gy(id_y, tr_actual)    # B→B
                loss_id = criterion_id(id_x, tr_real)

            # Total G loss, single backward
            loss_G = (
                loss_Gx_adv
            + loss_Gy_adv
            + lambda_cycle * loss_cycle
            + lambda_id    * loss_id
            )
            # print(">>> loss_G.grad_fn:", loss_G.grad_fn)
            with torch.autograd.detect_anomaly():
                loss_G.backward()
            optimizer_G.step()


            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append((loss_Dx + loss_Dy).item())
            epoch_cycle_losses.append(loss_cycle.item())
            epoch_id_losses.append(loss_id.item())
            epoch_gx_losses.append(loss_Gx_adv.item())
            epoch_gy_losses.append(loss_Gy_adv.item())
            epoch_dx_losses.append(loss_Dx.item())
            epoch_dy_losses.append(loss_Dy.item())

            # # Unfreeze D’s weights
            for p in Dx.parameters():
                p.requires_grad_(True)
            for p in Dy.parameters():
                p.requires_grad_(True)



            # ——— 3) Logging ———
            
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}]  "
            f"Loss_Dx: {loss_Dx.item():.4f}, Loss_Dy: {loss_Dy.item():.4f}, "
            f"Loss_G:  {loss_G.item():.4f}"
        )


        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_cycle_loss = sum(epoch_cycle_losses) / len(epoch_cycle_losses)
        avg_id_loss = sum(epoch_id_losses) / len(epoch_id_losses)
        avg_gx_loss = sum(epoch_gx_losses) / len(epoch_gx_losses)
        avg_gy_loss = sum(epoch_gy_losses) / len(epoch_gy_losses)
        avg_dx_loss = sum(epoch_dx_losses) / len(epoch_dx_losses)
        avg_dy_loss = sum(epoch_dy_losses) / len(epoch_dy_losses)

        val_g_loss, val_d_loss, val_cycle_loss, val_id_loss, val_gx_loss, val_gy_loss, val_dx_loss, val_dy_loss = evaluate(
            Gx,
            Gy,
            Dx,
            Dy,
            val_loader,
            criterion_adv,
            criterion_cycle,
            criterion_id,
            lambda_cycle,
            lambda_id,
            device,
            cfg.noise_dim,
            cfg.noise_type,
        )

        loss_monitor.update(
            epoch,
            avg_g_loss,
            avg_d_loss,
            cycle_loss=avg_cycle_loss,
            identity_loss=avg_id_loss,
            val_gen_loss=val_g_loss,
            val_disc_loss=val_d_loss,
            val_cycle_loss=val_cycle_loss,
            val_identity_loss=val_id_loss,
            gx_loss=avg_gx_loss,
            gy_loss=avg_gy_loss,
            dx_loss=avg_dx_loss,
            dy_loss=avg_dy_loss,
            val_gx_loss=val_gx_loss,
            val_gy_loss=val_gy_loss,
            val_dx_loss=val_dx_loss,
            val_dy_loss=val_dy_loss,
        )

        loss_logger.log(
            epoch,
            avg_g_loss,
            avg_d_loss,
            cycle_loss=avg_cycle_loss,
            identity_loss=avg_id_loss,
            val_gen_loss=val_g_loss,
            val_disc_loss=val_d_loss,
            val_cycle_loss=val_cycle_loss,
            val_identity_loss=val_id_loss,
            gx_loss=avg_gx_loss,
            gy_loss=avg_gy_loss,
            dx_loss=avg_dx_loss,
            dy_loss=avg_dy_loss,
            val_gx_loss=val_gx_loss,
            val_gy_loss=val_gy_loss,
            val_dx_loss=val_dx_loss,
            val_dy_loss=val_dy_loss,
        )
        # Counterfactual distribution plots
        if args.check_counterfactuals:
            cf_path = Path(f"saved_models/temporal_cycle_gan_noise_{timestemp}") \
                / f"counterfactuals_epoch_{epoch+1}.png"
            evaluate_counterfactual_distributions(
                Gx,
                val_loader,
                dec_treat,
                spec_treat,
                oh_treat,
                ord_treat,
                ranges_treat,
                device,
                cf_path,
                cfg.noise_dim,
                cfg.noise_type,
            )
        if cfg.mode == "debug":
            tr_ex, cl_ex, act_ex = next(iter(val_loader))
            tr_ex = tr_ex.to(device)
            cl_ex = cl_ex.to(device)
            act_ex = act_ex.to(device)
            with torch.no_grad():
                # Generate noise for debugging feed into generator
                counter = randomize_one_hot(act_ex)
                noise_ex = generate_sequence_noise(
                    cl_ex.size(0),
                    cl_ex.size(1),
                    cfg.noise_dim,
                    noise_type=cfg.noise_type,
                    device=device,
                )
                fake_ex = Gx(torch.cat((cl_ex, noise_ex), dim=2), counter)
            print("\n[DEBUG] Decoded sample:\nClinical:")
            for t in range(min(3, cl_ex.size(1))):  # Show first 3 timesteps
                print(f"\n--- Timestep {t} ---")
                df_out = decode_embedding(dec_clin(cl_ex[0, t]), spec_clin, oh_clin, ord_clin, ranges_clin)
                for _, row in df_out.iterrows():
                    print(" \n ".join([f"{col}: {row[col]}" for col in df_out.columns]))
                
                print("\nReal Treatment:")
                df_out = decode_embedding(dec_treat(tr_ex[0, t]), spec_treat, oh_treat, ord_treat, ranges_treat)
                for _, row in df_out.iterrows():
                    print(" \n ".join([f"{col}: {row[col]}" for col in df_out.columns]))
                
                print("\nGenerated Treatment:")
                df_out = decode_embedding(dec_treat(fake_ex[0, t]), spec_treat, oh_treat, ord_treat, ranges_treat)
                for _, row in df_out.iterrows():
                    print(" \n ".join([f"{col}: {row[col]}" for col in df_out.columns]))
    # ---- save trained models -------------------------------------------------
    meta = {
        "metadata_clinical_file": args.metadata_clinical_file,
        "metadata_treatment_file": args.metadata_treatment_file,
        "encoder_clinical": args.encoder_clinical,
        "encoder_treatment": args.encoder_treatment,
        "decoder_clinical": args.decoder_clinical,
        "decoder_treatment": args.decoder_treatment,
        "numeric_ranges_clinical": ds.meta_clin.numeric_ranges,
        "numeric_ranges_treatment": ds.meta_treat.numeric_ranges,
        "val_patient_ids": val_patient_ids,
        "loss_log_path": str(loss_logger.filepath),
    }

    params = {
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "num_layers": num_layers,
        "seq_len": T,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "beta1": beta1,
        "lambda_cycle": lambda_cycle,
        "lambda_id": lambda_id,
        "noise_dim": cfg.noise_dim,
        "noise_type": cfg.noise_type,
    }

    save_temporal_cycle_gan(
        Gx,
        Gy,
        Dx,
        Dy,
        Path(f"saved_models/temporal_cycle_gan_noise_{timestemp}"),
        metadata=meta,
        model_params=params,
        timestamp=timestemp
    )            

def evaluate_counterfactual_distributions(
    Gx,
    loader,
    decoder_treat,
    feat_spec,
    oh_map,
    ord_map,
    num_ranges,
    device,
    output_path,
    noise_dim,
    noise_type
):
    """Evaluate distribution of generated counterfactual treatments."""
    prev_mode = Gx.training
    Gx.eval()
    real_rows = []
    fake_rows = []
    with torch.no_grad():
        for tr_real, cl_real, tr_actual in loader:
            cl_real = cl_real.to(device)
            tr_real = tr_real.to(device)
            tr_actual = tr_actual.to(device)
            # generate counterfactual one-hot and noise
            tr_counter = randomize_one_hot(tr_actual)
            noise = generate_sequence_noise(
                cl_real.size(0), cl_real.size(1), noise_dim, noise_type=noise_type, device=device
            )
            fake_y = Gx(torch.cat((cl_real, noise), dim=2), tr_counter)
            # collect records timestep by timestep
            for b in range(fake_y.size(0)):
                for t in range(fake_y.size(1)):
                    real_df = decode_embedding(
                        decoder_treat(tr_real[b, t]), feat_spec, oh_map, ord_map, num_ranges
                    )
                    fake_df = decode_embedding(
                        decoder_treat(fake_y[b, t]), feat_spec, oh_map, ord_map, num_ranges
                    )
                    real_df['batch'] = b; real_df['timestep'] = t
                    fake_df['batch'] = b; fake_df['timestep'] = t
                    real_rows.append(real_df)
                    fake_rows.append(fake_df)
    if prev_mode:
        Gx.train()
    if not real_rows:
        return
    real_df = pd.concat(real_rows, ignore_index=True)
    fake_df = pd.concat(fake_rows, ignore_index=True)
    # plotting logic (histograms, bar charts) as in temporal_cycle_gan
    import matplotlib.pyplot as plt, math
    cols = real_df.columns.drop(['batch','timestep'])
    # categorize feature columns (exclude batch/timestep)
    cols = [c for c in real_df.columns if c not in ['batch','timestep']]
    numeric_cols, categorical_cols, ordinal_cols = [], [], []
    for col in cols:
        spec = feat_spec.get(col)
        if spec == 'numeric':
            numeric_cols.append(col)
        elif spec == 'one_hot':
            categorical_cols.append(col)
        elif spec == 'ordinal' or isinstance(spec, dict):
            ordinal_cols.append(col)
        else:
            # fallback by dtype
            if pd.api.types.is_numeric_dtype(real_df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
    # determine subplot grid
    total_plots = len(numeric_cols) + 2*len(categorical_cols) + 2*len(ordinal_cols)
    cols_per_row = 3
    n_rows = math.ceil(total_plots / cols_per_row) if total_plots > 0 else 1
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5*cols_per_row, 4*n_rows))
    axes = axes.flatten()
    plot_idx = 0
    # plot numeric
    for col in numeric_cols:
        ax = axes[plot_idx]
        r = real_df[col].dropna(); f = fake_df[col].dropna()
        ax.hist(r, bins=20, alpha=0.5, label='real')
        ax.hist(f, bins=20, alpha=0.5, label='fake')
        ax.set_title(f"{col} (Numeric)")
        ax.legend()
        plot_idx += 1
    # plot categorical
    for col in categorical_cols:
        ax_real = axes[plot_idx]
        r = real_df[col].dropna(); f = fake_df[col].dropna()
        r_counts = r.value_counts(); f_counts = f.value_counts()
        categories = sorted(set(r_counts.index).union(f_counts.index))
        idxs = np.arange(len(categories))
        r_vals = r_counts.reindex(categories).fillna(0)
        f_vals = f_counts.reindex(categories).fillna(0)
        ax_real.bar(idxs, r_vals.values, width=0.6, color='blue', alpha=0.7)
        ax_real.set_xticks(idxs); ax_real.set_xticklabels(categories, rotation=90, fontsize=8)
        ax_real.set_title(f"{col} - Real (Categorical)")
        plot_idx += 1
        ax_fake = axes[plot_idx]
        ax_fake.bar(idxs, f_vals.values, width=0.6, color='orange', alpha=0.7)
        ax_fake.set_xticks(idxs); ax_fake.set_xticklabels(categories, rotation=90, fontsize=8)
        ax_fake.set_title(f"{col} - Fake (Categorical)")
        plot_idx += 1
    # plot ordinal
    for col in ordinal_cols:
        r = real_df[col].dropna(); f = fake_df[col].dropna()
        # preserve order
        if isinstance(feat_spec.get(col), dict):
            ordered = sorted(feat_spec[col].keys(), key=lambda x: feat_spec[col][x])
        elif col in ord_map:
            ordered = sorted(ord_map[col].keys(), key=lambda x: ord_map[col][x])
        else:
            all_cats = set(r.index).union(f.index)
            try: ordered = sorted(all_cats)
            except: ordered = list(all_cats)
        r_counts = r.value_counts(); f_counts = f.value_counts()
        r_vals = r_counts.reindex(ordered).fillna(0)
        f_vals = f_counts.reindex(ordered).fillna(0)
        idxs = np.arange(len(ordered))
        ax = axes[plot_idx]
        ax.bar(idxs, r_vals.values, width=0.6, color='green', alpha=0.7)
        ax.set_xticks(idxs); ax.set_xticklabels(ordered, rotation=90, fontsize=8)
        ax.set_title(f"{col} - Real (Ordinal)")
        plot_idx += 1
        ax = axes[plot_idx]
        ax.bar(idxs, f_vals.values, width=0.6, color='red', alpha=0.7)
        ax.set_xticks(idxs); ax.set_xticklabels(ordered, rotation=90, fontsize=8)
        ax.set_title(f"{col} - Fake (Ordinal)")
        plot_idx += 1
    # disable empty plots
    for ax in axes[plot_idx:]: ax.axis('off')
    fig.tight_layout()
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp)
    plt.close(fig)

# --------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--config_file", default="configs/mock_config_cycle_gan_noise.yaml",
                    help="Path to YAML configuration file")
    ap.add_argument("--metadata_clinical_file", default="saved_models/clinical_data_encoder/encoder_decoder.pt/metadata_20250727_125941.json")
    ap.add_argument("--metadata_treatment_file", default="saved_models/treatments_data_output/encoder_decoder.pt/metadata_20250727_121153.json")

    ap.add_argument("--encoder_clinical", default="saved_models/clinical_data_encoder/encoder_decoder.pt/encoder_20250727_125941.pt")
    ap.add_argument("--encoder_treatment", default="saved_models/treatments_data_output/encoder_decoder.pt/encoder_20250727_121153.pt")

    ap.add_argument("--decoder_clinical", default="saved_models/clinical_data_encoder/encoder_decoder.pt/decoder_20250727_125941.pt")
    ap.add_argument("--decoder_treatment", default="saved_models/treatments_data_output/encoder_decoder.pt/decoder_20250727_121153.pt")

    ap.add_argument(
        "--check_counterfactuals",
        action="store_true",
        help="Evaluate counterfactual distributions each epoch",
    )
    args = ap.parse_args()
    cfg  = load_config(args.config_file)
    train(cfg, args)