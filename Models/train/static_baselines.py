import argparse
import os
import numpy as np
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils.helpers import MongoExtractor
from models.utils import MetadataHandler
from dataset.tabular_dataset import TabularDataset

class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, x):
        return self.net(x)

def load_config(config_file):
    """Load configuration from YAML file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config

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
    X = []
    Y = []
    for tr, cl, act in loader:
        X.append(cl.cpu())
        Y.append(act.cpu())
    X = torch.cat(X).numpy()
    Y = torch.cat(Y).numpy()
    return X, Y, meta_clin

def train_fnn(X, Y, input_dim, args):
    model = FeedForwardNet(input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optim_fnn = optim.Adam(model.parameters(), lr=args.lr)
    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    for _ in range(args.epochs):
        optim_fnn.zero_grad()
        out = model(X_t)
        loss = criterion(out, Y_t)
        loss.backward()
        optim_fnn.step()
    torch.save(model.state_dict(), os.path.join(args.out_dir, "fnn.pt"))


def train_gbt(X, Y, args):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = MultiOutputClassifier(GradientBoostingClassifier()).fit(X_train, y_train)
    acc = clf.score(X_val, y_val)
    print(f"GBT validation accuracy: {acc:.3f}")
    from joblib import dump
    dump(clf, os.path.join(args.out_dir, "gbt.joblib"))


def train_bart(X, Y, args):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = MultiOutputClassifier(HistGradientBoostingClassifier()).fit(X_train, y_train)
    acc = clf.score(X_val, y_val)
    print(f"BART (hist GB) validation accuracy: {acc:.3f}")
    from joblib import dump
    dump(clf, os.path.join(args.out_dir, "bart.joblib"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", default="configs/config_static_baselines.yaml")
    args = ap.parse_args()

    # Load configuration from file
    config = load_config(args.config_file)
    training_config = config.get('training', {})
    
    # Get parameters from config or use defaults
    metadata_clinical_file = training_config.get('metadata_clinical_file')
    metadata_treatment_file = training_config.get('metadata_treatment_file')
    batch_size = training_config.get('batch_size', 64)
    lr = training_config.get('learning_rate', 1e-3)
    epochs = training_config.get('epochs', 10)
    out_dir = training_config.get('out_dir', 'saved_models/static_baselines')

    # Create a simple args object to pass to other functions
    class TrainingArgs:
        def __init__(self):
            self.config_file = args.config_file
            self.metadata_clinical_file = metadata_clinical_file
            self.metadata_treatment_file = metadata_treatment_file
            self.batch_size = batch_size
            self.lr = lr
            self.epochs = epochs
            self.out_dir = out_dir
    
    training_args = TrainingArgs()
    
    os.makedirs(training_args.out_dir, exist_ok=True)
    X, Y, meta_clin = load_dataset(training_args)
    input_dim = meta_clin.hidden_dim
    train_fnn(X, Y, input_dim, training_args)
    train_gbt(X, Y, training_args)
    train_bart(X, Y, training_args)

if __name__ == "__main__":
    main()