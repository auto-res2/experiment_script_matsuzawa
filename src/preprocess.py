"""Data preprocessing utilities."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def prepare_data(config):
    """Generate and preprocess synthetic data for the experiment."""
    # Generate synthetic data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_classes=2, random_state=config.random_seed
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=config.train_size, random_state=config.random_seed
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = SyntheticDataset(X_train, y_train)
    test_dataset = SyntheticDataset(X_test, y_test)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    return train_loader, test_loader
