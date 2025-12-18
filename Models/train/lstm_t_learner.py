import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.GANs import LSTMGenerator
from utils.helpers import MongoExtractor
from models.utils import MetadataHandler
from dataset.tabular_dataset import TabularDatasetPID

class TLearner(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.net = LSTMGenerator(input_dim, cond_dim, hidden_dim, input_dim, num_layers)

    def forward(self, x, t):
        return self.net(x, t)


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
    dataset = TabularDatasetPID(df, meta_clin, meta_treat, seq_len=args.seq_len, id_field=args.id_field)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return loader, meta_clin.hidden_dim


def train(args):
    loader, emb_dim = load_dataset(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TLearner(emb_dim, 3, args.hidden_dim, args.num_layers).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    for epoch in range(args.epochs):
        for tr, cl, act in loader:
            cl = cl.to(device)
            act = act.float().to(device)
            tr = tr.to(device)
            opt.zero_grad()
            pred = model(cl, act)
            loss = crit(pred, tr)
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1}/{args.epochs} - loss {loss.item():.4f}")
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "lstm_t_learner.pt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", default="configs/mock_config_cycle_gan.yaml")
    ap.add_argument("--metadata_clinical_file")
    ap.add_argument("--metadata_treatment_file")
    ap.add_argument("--seq_len", type=int, default=5)
    ap.add_argument("--id_field", default="patient_id")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out_dir", default="saved_models")
    args = ap.parse_args()
    train(args)