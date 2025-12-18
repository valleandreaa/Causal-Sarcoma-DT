#!/usr/bin/env python
"""
temporal_cycle_gan_training_strategy.py
Training script for Temporal CycleGAN using a curriculum
strategy over the episode sequence length.
Each epoch increases the number of episodes fed to the model.
"""

from __future__ import annotations

import sys
from pathlib import Path
import random
from datetime import datetime
import argparse

# Allow importing modules from this directory and the main project
CURR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURR_DIR))
sys.path.append(str(CURR_DIR.parent))  # Models directory for `models` package

# Import utilities from the original training script
from temporal_cycle_gan import (
    Config,
    load_config,
    get_optimizer,
    get_criterion,
    load_decoder,
    evaluate,
    evaluate_counterfactual_distributions,
    MongoTemporalDataset,
)

import torch
from torch.utils.data import DataLoader, Subset
from models.GANs import LSTMGenerator, LSTMDiscriminator
from models.utils import save_temporal_cycle_gan, randomize_one_hot
from dataset.tabular_dataset import TabularDatasetPID
from monitoring.losses import CycleGANLossMonitor
from monitoring.loss_logger import LossLogger
from models.decoder_embedders import decode_embedding


def train(cfg: Config, args) -> None:
    ds = MongoTemporalDataset(cfg, split="train", args=args)
    base_dataset = TabularDatasetPID(
        ds.df,
        ds.meta_clin,
        ds.meta_treat,
        seq_len=cfg.seq_len,
        id_field=cfg.id_field,
    )

    rng = random.Random(cfg.seed)
    indices = list(range(base_dataset.num_patients))
    rng.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)

    val_patient_ids = [base_dataset.patient_ids[i] for i in val_indices]

    embedding_dim = cfg.x_dim
    hidden_dim = cfg.g_hidden
    num_layers = cfg.num_layers
    batch_size = cfg.batch_size
    num_epochs = cfg.epochs
    lr_g = cfg.lr_g
    lr_d = cfg.lr_d
    beta1 = cfg.beta1
    lambda_cycle = cfg.lambda_cyc
    lambda_id = cfg.lambda_id
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    Gx = LSTMGenerator(
        input_dim=embedding_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)
    Gy = LSTMGenerator(
        input_dim=embedding_dim,
        cond_input_dim=3,
        hidden_dim=hidden_dim,
        output_dim=embedding_dim,
        num_layers=num_layers,
    ).to(device)
    Dx = LSTMDiscriminator(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    Dy = LSTMDiscriminator(
        input_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    from models.losses import get_adversarial_loss

    criterion_adv = get_adversarial_loss(cfg.adv_loss)
    criterion_cycle = get_criterion(cfg.criterion)
    criterion_id = get_criterion(cfg.criterion)

    optimizer_G = get_optimizer(
        cfg.opt_gen,
        list(Gx.parameters()) + list(Gy.parameters()),
        lr_g,
        (beta1, cfg.beta2),
    )
    optimizer_Dx = get_optimizer(cfg.opt_disc, Dx.parameters(), lr_d, (beta1, cfg.beta2))
    optimizer_Dy = get_optimizer(cfg.opt_disc, Dy.parameters(), lr_d, (beta1, cfg.beta2))

    timestemp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loss_monitor = CycleGANLossMonitor(
        path=Path(f"saved_models/temporal_cycle_gan_{timestemp}") / "cyclegan_losses.png"
    )
    loss_logger = LossLogger(Path(f"saved_models/temporal_cycle_gan_{timestemp}") / "losses.log")

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

    for epoch in range(num_epochs):
        curr_len = min(cfg.seq_len, epoch + 1)
        base_dataset.seq_len = curr_len
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        epoch_g_losses = []
        epoch_d_losses = []
        epoch_cycle_losses = []
        epoch_id_losses = []
        epoch_gx_losses = []
        epoch_gy_losses = []
        epoch_dx_losses = []
        epoch_dy_losses = []

        for tr_real, cl_real, tr_actual in train_loader:
            cl_real = cl_real.to(device).detach()
            tr_real = tr_real.to(device).detach()
            tr_actual = tr_actual.to(device).detach()
            tr_counter = randomize_one_hot(tr_actual)

            optimizer_Dx.zero_grad()
            optimizer_Dy.zero_grad()
            fake_y_det = Gx(cl_real, tr_counter).detach()
            rec_x_det = Gy(fake_y_det, tr_actual).detach()
            pred_real_x = Dx(torch.cat((cl_real, tr_real), dim=1))
            pred_fake_x = Dx(torch.cat((cl_real, fake_y_det), dim=1))
            pred_real_y = Dy(torch.cat((cl_real, tr_real), dim=1))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x_det), dim=1))
            loss_Dx = 0.5 * (
                criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x))
                + criterion_adv(pred_real_x, torch.ones_like(pred_real_x))
            )
            loss_Dy = 0.5 * (
                criterion_adv(pred_real_y, torch.ones_like(pred_real_y))
                + criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y))
            )
            (loss_Dx + loss_Dy).backward()
            optimizer_Dx.step()
            optimizer_Dy.step()

            for p in Dx.parameters():
                p.requires_grad_(False)
            for p in Dy.parameters():
                p.requires_grad_(False)
            optimizer_G.zero_grad()
            fake_y = Gx(cl_real, tr_counter)
            rec_x = Gy(fake_y, tr_actual)
            pred_fake_x = Dx(torch.cat((cl_real, fake_y), dim=1))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x), dim=1))
            loss_Gx_adv = criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_Gy_adv = criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y))
            loss_cycle = criterion_cycle(rec_x, tr_real)
            loss_id = torch.tensor(0.0, device=device)
            if lambda_id > 0:
                id_y = Gx(cl_real, tr_counter)
                id_x = Gy(id_y, tr_actual)
                loss_id = criterion_id(id_x, tr_real)
            loss_G = loss_Gx_adv + loss_Gy_adv + lambda_cycle * loss_cycle + lambda_id * loss_id
            loss_G.backward()
            optimizer_G.step()
            for p in Dx.parameters():
                p.requires_grad_(True)
            for p in Dy.parameters():
                p.requires_grad_(True)

            epoch_g_losses.append(loss_G.item())
            epoch_d_losses.append((loss_Dx + loss_Dy).item())
            epoch_cycle_losses.append(loss_cycle.item())
            epoch_id_losses.append(loss_id.item())
            epoch_gx_losses.append(loss_Gx_adv.item())
            epoch_gy_losses.append(loss_Gy_adv.item())
            epoch_dx_losses.append(loss_Dx.item())
            epoch_dy_losses.append(loss_Dy.item())

        avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
        avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
        avg_cycle_loss = sum(epoch_cycle_losses) / len(epoch_cycle_losses)
        avg_id_loss = sum(epoch_id_losses) / len(epoch_id_losses)
        avg_gx_loss = sum(epoch_gx_losses) / len(epoch_gx_losses)
        avg_gy_loss = sum(epoch_gy_losses) / len(epoch_gy_losses)
        avg_dx_loss = sum(epoch_dx_losses) / len(epoch_dx_losses)
        avg_dy_loss = sum(epoch_dy_losses) / len(epoch_dy_losses)

        (
            val_g_loss,
            val_d_loss,
            val_cycle_loss,
            val_id_loss,
            val_gx_loss,
            val_gy_loss,
            val_dx_loss,
            val_dy_loss,
        ) = evaluate(
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

        if args.check_counterfactuals:
            dist_path = (
                Path(f"saved_models/temporal_cycle_gan_{timestemp}")
                / f"counterfactuals_epoch_{epoch+1}.png"
            )
            evaluate_counterfactual_distributions(
                Gx,
                val_loader,
                dec_treat,
                spec_treat,
                oh_treat,
                ord_treat,
                ranges_treat,
                device,
                dist_path,
            )

        if cfg.mode == "debug":
            tr_ex, cl_ex, act_ex = next(iter(val_loader))
            tr_ex = tr_ex.to(device)
            cl_ex = cl_ex.to(device)
            act_ex = act_ex.to(device)
            with torch.no_grad():
                counter = randomize_one_hot(act_ex)
                fake_ex = Gx(cl_ex, counter)
            print("\n[DEBUG] Decoded sample:\nClinical:")
            for t in range(min(3, cl_ex.size(1))):
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
        "seq_len": cfg.seq_len,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "lr_g": lr_g,
        "lr_d": lr_d,
        "beta1": beta1,
        "lambda_cycle": lambda_cycle,
        "lambda_id": lambda_id,
    }

    save_temporal_cycle_gan(
        Gx,
        Gy,
        Dx,
        Dy,
        Path(f"saved_models/temporal_cycle_gan_{timestemp}"),
        metadata=meta,
        model_params=params,
        timestamp=timestemp,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--config_file",
        default="configs/mock_config_cycle_gan.yaml",
        help="Path to YAML configuration file",
    )
    ap.add_argument(
        "--metadata_clinical_file",
        default="saved_models/clinical_encoder/encoder_decoder.pt/metadata_20250722_203711.json",
    )
    ap.add_argument(
        "--metadata_treatment_file",
        default="saved_models/treatments_output/encoder_decoder.pt/metadata_20250722_203134.json",
    )
    ap.add_argument(
        "--encoder_clinical",
        default="saved_models/clinical_encoder/encoder_decoder.pt/encoder_20250722_203711.pt",
    )
    ap.add_argument(
        "--encoder_treatment",
        default="saved_models/treatments_output/encoder_decoder.pt/encoder_20250722_203134.pt",
    )
    ap.add_argument(
        "--decoder_clinical",
        default="saved_models/clinical_encoder/encoder_decoder.pt/decoder_20250722_203711.pt",
    )
    ap.add_argument(
        "--decoder_treatment",
        default="saved_models/treatments_output/encoder_decoder.pt/decoder_20250722_203134.pt",
    )
    ap.add_argument(
        "--check_counterfactuals",
        action="store_true",
        help="Evaluate counterfactual distributions each epoch",
    )
    args = ap.parse_args()
    cfg = load_config(args.config_file)
    train(cfg, args)
