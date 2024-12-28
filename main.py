### main.py ###
import numpy as np
from typing import List, Tuple
from prefetcher import PredictivePrefetcher

def generate_realistic_access_pattern(size: int = 1000) -> List[int]:
    patterns = []
    np.random.seed(42)
    
    def add_pattern(pattern: List[int]):
        if len(pattern) > 1:
            noise = np.random.normal(0, 0.1, len(pattern))
            noisy_pattern = [max(0, int(p + n * size)) for p, n in zip(pattern, noise)]
            patterns.extend(noisy_pattern)
    
    # Sequential access with stride
    for stride in [1, 2, 4, 8, 16]:
        seq = list(range(0, size * stride, stride))[:100]
        add_pattern(seq)
    
    # Loop pattern
    for loop_size in [10, 20, 50]:
        base = np.random.randint(0, size)
        loop = list(range(base, base + loop_size)) * 5
        add_pattern(loop)
    
    # Array traversal (row-major)
    array_dim = 32
    for i in range(array_dim):
        for j in range(array_dim):
            patterns.append(i * array_dim + j)
    
    return patterns[:1000]  # Truncate to reasonable size

def evaluate_prefetcher(prefetcher: PredictivePrefetcher, test_sequence: List[int], 
                       tolerance: int = 32) -> Tuple[float, List[Tuple[int, int]]]:
    correct_predictions = 0
    total_predictions = 0
    prediction_pairs = []
    
    for i in range(len(test_sequence) - 1):
        current = test_sequence[i]
        next_actual = test_sequence[i + 1]
        
        prefetcher.record_access(current)
        prediction = prefetcher.predict_next_access()
        
        if prediction is not None:
            # Check if prediction matches within tolerance
            if abs(prediction - next_actual) <= tolerance:
                correct_predictions += 1
            total_predictions += 1
            prediction_pairs.append((prediction, next_actual))
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return accuracy, prediction_pairs

def main():
    # Generate access pattern
    access_pattern = generate_realistic_access_pattern()
    
    # Split into train and test
    train_ratio = 0.8
    split_idx = int(len(access_pattern) * train_ratio)
    train_sequence = access_pattern[:split_idx]
    test_sequence = access_pattern[split_idx:]
    
    # Initialize and train prefetcher
    prefetcher = PredictivePrefetcher(
        sequence_length=20,
        hidden_size=128,
        num_layers=3
    )
    
    # Train
    losses = prefetcher.train(
        train_sequence,
        epochs=100,
        batch_size=128,
        learning_rate=0.002
    )
    
    # Evaluate
    accuracy, predictions = evaluate_prefetcher(prefetcher, test_sequence)
    print(f"\nFinal prediction accuracy: {accuracy:.2%}")
    
    # Print some example predictions
    print("\nSample predictions (Predicted -> Actual):")
    for pred, actual in predictions[:10]:
        print(f"{pred:5d} -> {actual:5d} (diff: {abs(pred - actual):4d})")
    
    # Print pattern statistics
    diffs = [abs(p - a) for p, a in predictions]
    print(f"\nPrediction statistics:")
    print(f"Average difference: {np.mean(diffs):.2f}")
    print(f"Median difference: {np.median(diffs):.2f}")
    print(f"Max difference: {max(diffs)}")
    print(f"Min difference: {min(diffs)}")

if __name__ == "__main__":
    main()