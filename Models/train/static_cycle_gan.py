import argparse
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.GANs import CycleGANGenerator, CycleGANDiscriminator
from models.losses import get_adversarial_loss
from utils.helpers import MongoExtractor
from models.utils import MetadataHandler, save_temporal_cycle_gan
from dataset.tabular_dataset import TabularDataset
from datetime import datetime


def load_dataset(args):
    extractor = MongoExtractor(
        connection_string=os.getenv("MONGO_URI"),
        database_name=os.getenv("MONGO_DB"),
        collection_name=os.getenv("MONGO_COLLECTION"),
        config_file=args.config_file,
    )
    df = extractor.get_dataframe()
    meta_clin = MetadataHandler(args.metadata_clinical_file)
    meta_treat = MetadataHandler(args.metadata_treatment_file)
    dataset = TabularDataset(df, meta_clin, meta_treat)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loader, meta_clin.hidden_dim


def train(args):
    loader, emb_dim = load_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Gx = CycleGANGenerator(input_dim=emb_dim, output_dim=emb_dim).to(device)
    Gy = CycleGANGenerator(input_dim=emb_dim, output_dim=emb_dim).to(device)
    Dx = CycleGANDiscriminator(input_dim=emb_dim * 2).to(device)
    Dy = CycleGANDiscriminator(input_dim=emb_dim * 2).to(device)
    criterion_adv = get_adversarial_loss("bce")
    criterion_cycle = nn.L1Loss()
    optimizer_G = torch.optim.Adam(list(Gx.parameters()) + list(Gy.parameters()), lr=args.lr)
    optimizer_Dx = torch.optim.Adam(Dx.parameters(), lr=args.lr)
    optimizer_Dy = torch.optim.Adam(Dy.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        for tr_real, cl_real, _ in loader:
            cl_real = cl_real.to(device)
            tr_real = tr_real.to(device)

            optimizer_Dx.zero_grad(); optimizer_Dy.zero_grad()
            fake_y = Gx(cl_real).detach()
            rec_x = Gy(fake_y).detach()
            pred_real_x = Dx(torch.cat((cl_real, tr_real), dim=1))
            pred_fake_x = Dx(torch.cat((cl_real, fake_y), dim=1))
            pred_real_y = Dy(torch.cat((cl_real, tr_real), dim=1))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x), dim=1))
            loss_Dx = 0.5 * (criterion_adv(pred_real_x, torch.ones_like(pred_real_x)) +
                             criterion_adv(pred_fake_x, torch.zeros_like(pred_fake_x)))
            loss_Dy = 0.5 * (criterion_adv(pred_real_y, torch.ones_like(pred_real_y)) +
                             criterion_adv(pred_fake_y, torch.zeros_like(pred_fake_y)))
            (loss_Dx + loss_Dy).backward()
            optimizer_Dx.step(); optimizer_Dy.step()

            for p in Dx.parameters(): p.requires_grad_(False)
            for p in Dy.parameters(): p.requires_grad_(False)
            optimizer_G.zero_grad()
            fake_y = Gx(cl_real)
            rec_x = Gy(fake_y)
            pred_fake_x = Dx(torch.cat((cl_real, fake_y), dim=1))
            pred_fake_y = Dy(torch.cat((cl_real, rec_x), dim=1))
            loss_G = (criterion_adv(pred_fake_x, torch.ones_like(pred_fake_x)) +
                      criterion_adv(pred_fake_y, torch.ones_like(pred_fake_y)) +
                      args.lambda_cyc * criterion_cycle(rec_x, tr_real))
            loss_G.backward()
            optimizer_G.step()
            for p in Dx.parameters(): p.requires_grad_(True)
            for p in Dy.parameters(): p.requires_grad_(True)
        print(f"Epoch {epoch+1}/{args.epochs} completed")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_temporal_cycle_gan(Gx, Gy, Dx, Dy, os.path.join(args.out_dir, f"static_cycle_gan_{ts}"), timestamp=ts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", default="configs/mock_config_cycle_gan.yaml")
    args = ap.parse_args()
    train(args)