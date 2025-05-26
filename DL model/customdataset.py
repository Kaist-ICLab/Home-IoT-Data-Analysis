import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, df, label):
        self.df = df
        self.label = label
        self.to_drop = ['phq2_result', 'gad2_result',
                        'stress_result', 'posNeg_result', 'arousal_result',
                        'phq2_result_binary', 'gad2_result_binary', 
                        'stress_result_binary', 'posNeg_result_binary', 'arousal_result_binary',
                        'uid', 'timestamp']
        if 'timestamp' not in df.columns:
            self.to_drop.remove('timestamp')

        self.X = df.drop(columns=self.to_drop).values.astype(float)
        self.y = df[label].values.astype(float)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

def df_to_dataset(df, label, batch_size=32, shuffle=False, splitter='loso',):

    dataset = CustomDataset(df, label)

    if splitter!='kfold':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # elif splitter == 'kfold':
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return data_loader

# 2024-11-05 추가

class CustomDataset_new(Dataset):
    def __init__(self, X, y):
        self.X = X.values.astype(float)
        self.y = y.values.astype(float)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return X, y

def df_to_dataset_new(X, y, batch_size=32, shuffle=False, splitter='loso',):

    dataset = CustomDataset_new(X, y)

    if splitter!='kfold':
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    # elif splitter == 'kfold':
    else:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    return data_loader