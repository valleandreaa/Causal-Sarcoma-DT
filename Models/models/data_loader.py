from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch

class PatientDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.data = self.preprocess_data(self.dataframe)
    
    def preprocess_data(self, dataframe):
        # Identify numerical and categorical columns
        numerical_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = dataframe.select_dtypes(include=['object']).columns

        # Define the preprocessing steps for numerical and categorical data
        numerical_transformer = MinMaxScaler()
        categorical_transformer = OneHotEncoder()

        # Create a ColumnTransformer to apply the transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

        # Fit and transform the data
        preprocessed_data = preprocessor.fit_transform(dataframe)
        return preprocessed_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)