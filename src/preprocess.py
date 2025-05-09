"""
Data preprocessing for G-DS3 Transformer experiments.
"""

import torch
import torch.utils.data

class CopyMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, seq_len=50, n_digits=10, num_samples=200):
        """
        Synthetic dataset for copy-memory task.
        
        Args:
            seq_len: Length of the sequence
            n_digits: Number of unique digits/classes
            num_samples: Number of samples in the dataset
        """
        self.seq_len = seq_len
        self.n_digits = n_digits
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        seq = torch.randint(0, self.n_digits, (self.seq_len,))
        label = seq[0]
        return seq, label
