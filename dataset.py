### dataset.py ###
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

class MemoryAccessDataset(Dataset):
    def __init__(self, access_sequences: List[int], sequence_length: int):
        self.sequence_length = sequence_length
        self.sequences = []
        self.targets = []
        
        # Calculate deltas (differences between consecutive addresses)
        deltas = np.diff(access_sequences)
        access_array = np.array(deltas).reshape(-1, 1)
        
        # Store last absolute address for reconstruction
        self.last_absolute_address = access_sequences[-1]
        
        # Standardize the deltas
        self.scaler = StandardScaler()
        normalized_deltas = self.scaler.fit_transform(access_array).flatten()
        
        # Create sequences and targets
        for i in range(len(normalized_deltas) - sequence_length):
            seq = normalized_deltas[i:i + sequence_length]
            target = normalized_deltas[i + sequence_length]
            self.sequences.append(seq)
            self.targets.append(target)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor([self.targets[idx]], dtype=torch.float32)
    
    def inverse_transform_delta(self, normalized_delta: float) -> int:
        return int(self.scaler.inverse_transform([[normalized_delta]])[0][0])