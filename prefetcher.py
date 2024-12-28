### prefetcher.py ###
from collections import deque, defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from model import LSTMPrefetcher
from dataset import MemoryAccessDataset

class PatternDetector:
    def __init__(self, window_size: int = 32):
        self.window_size = window_size
        self.stride_history = deque(maxlen=window_size)
        self.pattern_counts = defaultdict(int)
        self.last_stride = None
        
    def detect_stride(self, addresses: deque) -> Optional[int]:
        if len(addresses) < 2:
            return None
        
        # Calculate recent strides
        strides = [addresses[i] - addresses[i-1] for i in range(1, len(addresses))]
        
        # Count stride frequencies
        stride_counts = defaultdict(int)
        for stride in strides[-self.window_size:]:
            stride_counts[stride] += 1
        
        # Find most common stride
        if stride_counts:
            most_common_stride = max(stride_counts.items(), key=lambda x: x[1])
            if most_common_stride[1] >= len(strides) * 0.6:  # 60% threshold
                return most_common_stride[0]
        return None
    
    def detect_loop(self, addresses: deque, max_loop_size: int = 50) -> Optional[Tuple[int, int]]:
        if len(addresses) < 4:
            return None
        
        # Try different loop sizes
        for size in range(2, min(max_loop_size, len(addresses) // 2)):
            potential_loop = list(addresses)[-size:]
            next_iter = list(addresses)[-2*size:-size]
            if potential_loop == next_iter:
                return size, potential_loop[0]
        return None
    
    def update(self, addresses: deque) -> Dict:
        patterns = {}
        
        # Detect stride pattern
        stride = self.detect_stride(addresses)
        if stride is not None:
            patterns['stride'] = stride
            self.pattern_counts['stride'] += 1
            self.last_stride = stride
        
        # Detect loop pattern
        loop = self.detect_loop(addresses)
        if loop is not None:
            patterns['loop'] = loop
            self.pattern_counts['loop'] += 1
        
        return patterns

class PredictivePrefetcher:
    def __init__(self, sequence_length: int = 32, hidden_size: int = 256, num_layers: int = 3):
        self.sequence_length = sequence_length
        self.model = LSTMPrefetcher(1, hidden_size, num_layers)
        self.recent_accesses = deque(maxlen=sequence_length + 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dataset = None
        self.pattern_detector = PatternDetector()
        
    def train(self, access_sequences: List[int], epochs: int = 100, batch_size: int = 128,
              learning_rate: float = 0.001) -> List[float]:
        self.dataset = MemoryAccessDataset(access_sequences, self.sequence_length)
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        losses = []
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 10
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sequences, targets in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            # Learning rate scheduling
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience_limit:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if epoch % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        return losses
    
    def predict_next_access(self) -> Optional[int]:
        if len(self.recent_accesses) <= self.sequence_length or self.dataset is None:
            return None
        
        # Detect patterns
        patterns = self.pattern_detector.update(self.recent_accesses)
        
        # If strong stride pattern detected, use stride prediction
        if 'stride' in patterns:
            stride = patterns['stride']
            return self.recent_accesses[-1] + stride
        
        # If loop pattern detected, use loop prediction
        if 'loop' in patterns:
            loop_size, loop_start = patterns['loop']
            position_in_loop = (len(self.recent_accesses) - 1) % loop_size
            return self.recent_accesses[-loop_size + ((position_in_loop + 1) % loop_size)]
        
        # Fall back to LSTM prediction
        self.model.eval()
        with torch.no_grad():
            deltas = []
            for i in range(len(self.recent_accesses) - 1):
                delta = self.recent_accesses[i + 1] - self.recent_accesses[i]
                normalized_delta = self.dataset.scaler.transform([[delta]])[0][0]
                deltas.append(normalized_delta)
            
            sequence = torch.tensor(deltas[-self.sequence_length:], dtype=torch.float32)
            sequence = sequence.to(self.device)
            prediction = self.model(sequence.unsqueeze(0))
            
            predicted_delta = self.dataset.inverse_transform_delta(prediction.item())
            return self.recent_accesses[-1] + predicted_delta
    
    def record_access(self, address: int):
        self.recent_accesses.append(address)