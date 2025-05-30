import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.values.astype(float)
        self.y = y.values.astype(float)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

def df_to_dataset_new(X, y, batch_size=32, shuffle=False):
    dataset = CustomDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)