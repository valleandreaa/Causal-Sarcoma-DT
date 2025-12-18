import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.original_df = dataframe.copy()

        self.num_cols = self.original_df.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols = self.original_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        # Build mask: 1 for observed, 0 for missing
        self.mask_df = self.original_df[self.num_cols + self.cat_cols].notnull().astype(float)

        # Fill missing numerical values
        self.original_df[self.num_cols] = self.original_df[self.num_cols].fillna(self.original_df[self.num_cols].mean())

        # Fill missing categorical with placeholder
        self.original_df[self.cat_cols] = self.original_df[self.cat_cols].fillna("missing")

        # One-hot encode categorical
        self.original_df = pd.get_dummies(self.original_df, columns=self.cat_cols)

        # Align the mask to the one-hot encoded dataframe
        self.mask_df = pd.get_dummies(self.mask_df, columns=self.cat_cols)
        self.mask_df = self.mask_df.reindex(columns=self.original_df.columns, fill_value=1.0)

        # Normalize numerical columns
        self.scaler = StandardScaler()
        self.original_df[self.num_cols] = self.scaler.fit_transform(self.original_df[self.num_cols])

        # Convert to tensors
        self.data = torch.tensor(self.original_df.values, dtype=torch.float32)
        self.mask = torch.tensor(self.mask_df.values, dtype=torch.float32)

        self.input_dim = self.data.shape[1]
        self.output_dim = self.data.shape[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        m = self.mask[idx]
        return x, x, m  # x_input, x_target, mask
