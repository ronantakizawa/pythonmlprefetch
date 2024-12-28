### utils.py ###
import numpy as np
from typing import List, Tuple
from prefetcher import PredictivePrefetcher

def generate_realistic_access_pattern(size: int = 1000) -> List[int]:
    patterns = []
    np.random.seed(42)
    
    def add_noise(pattern, noise_level=0.1):
        noise = np.random.normal(0, noise_level, len(pattern))
        return [max(0, int(p + n)) for p, n in zip(pattern, noise * size)]
    
    # Sequential accesses with varying strides
    strides = [8, 16, 32]
    for stride in strides:
        seq_pattern = list(range(0, size * stride, stride))
        patterns.extend(add_noise(seq_pattern))
    
    # Loop patterns of different sizes
    loop_sizes = [50, 100, 200]
    for loop_size in loop_sizes:
        base = np.random.randint(0, size)
        loop_pattern = list(range(base, base + loop_size)) * 3
        patterns.extend(add_noise(loop_pattern))
    
    # Strided loop patterns (common in matrix operations)
    matrix_size = 50
    for i in range(matrix_size):
        for j in range(matrix_size):
            patterns.append(i * matrix_size + j)  # Row-major
            patterns.append(j * matrix_size + i)  # Column-major
    
    # Linked list traversal simulation
    current = np.random.randint(0, size)
    for _ in range(100):
        patterns.append(current)
        current = (current * 17 + 31) % size  # Simple hash function
    
    # Random accesses with temporal and spatial locality
    for _ in range(5):
        base = np.random.randint(0, size)
        region_size = 100
        # Temporal locality: repeat recent accesses
        recent = []
        for _ in range(50):
            if recent and np.random.random() < 0.3:
                patterns.append(np.random.choice(recent))
            else:
                addr = base + np.random.randint(0, region_size)
                patterns.append(addr)
                recent = [addr] + recent[:5]
    
    # Ensure all patterns are within bounds
    patterns = [min(max(0, p), size - 1) for p in patterns]
    
    # Add some controlled randomness while preserving local patterns
    final_patterns = []
    window_size = 30
    overlap = 10
    for i in range(0, len(patterns) - window_size + 1, window_size - overlap):
        window = patterns[i:i + window_size]
        if np.random.random() < 0.2:  # 20% chance to shuffle window
            np.random.shuffle(window)
        final_patterns.extend(window)
    
    return final_patterns

def evaluate_prefetcher(prefetcher: PredictivePrefetcher, test_sequence: List[int], 
                       tolerance: int = 16) -> Tuple[float, List[Tuple[int, int]]]:
    correct_predictions = 0
    total_predictions = 0
    prediction_pairs = []
    
    for address in test_sequence:
        prediction = prefetcher.predict_next_access()
        if prediction is not None:
            if abs(prediction - address) <= tolerance:
                correct_predictions += 1
            total_predictions += 1
            prediction_pairs.append((prediction, address))
        prefetcher.record_access(address)
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, prediction_pairs