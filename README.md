# LSTM-based Memory Access Predictor

A deep learning-based memory prefetcher that uses LSTM (Long Short-Term Memory) networks combined with pattern detection to predict future memory access patterns.

## Features

- **Hybrid Prediction Strategy**
  - LSTM-based neural network for complex patterns
  - Pattern detection for regular access patterns
  - Automatic strategy selection based on detected patterns

- **Pattern Detection**
  - Stride pattern detection
  - Loop pattern detection
  - Adaptive pattern confidence scoring

- **Advanced Architecture**
  - Bidirectional LSTM layers
  - Attention mechanism
  - Delta-based prediction
  - Standardized input processing

## Project Structure

```
pythonprefetcher/
├── model.py         # LSTM neural network implementation
├── dataset.py       # Data preprocessing and loading
├── prefetcher.py    # Main prefetcher logic and training
├── utils.py         # Helper functions
├── main.py         # Entry point and execution
└── requirements.txt # Project dependencies
```

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy 1.19.2+
- scikit-learn 0.24.0+

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Basic usage:
```python
from prefetcher import PredictivePrefetcher

# Initialize prefetcher
prefetcher = PredictivePrefetcher(
    sequence_length=32,
    hidden_size=256,
    num_layers=3
)

# Train on your access pattern
prefetcher.train(access_sequences, epochs=100)

# Make predictions
next_access = prefetcher.predict_next_access()
```

2. Run the demo:
```bash
python main.py
```

## Architecture Details

### Pattern Detection

The system uses a hybrid approach combining:

1. **Stride Detection**
   - Identifies regular stride patterns in memory accesses
   - Calculates stride frequencies and confidence scores
   - Uses majority voting within a sliding window

2. **Loop Detection**
   - Identifies repeating patterns
   - Supports variable loop sizes
   - Handles nested loops

3. **LSTM Prediction**
   - Falls back to neural network for complex patterns
   - Uses delta-based prediction for better accuracy
   - Employs attention mechanism for pattern recognition

### Neural Network Architecture

- Input layer with normalization
- Bidirectional LSTM layers
- Attention mechanism
- Fully connected output layers
- Residual connections
- Layer normalization

## Performance

Current performance metrics:
- Prediction accuracy: ~63%
- Median prediction error: 5 addresses
- Average prediction error: ~45 addresses
- Handles various access patterns:
  - Sequential access
  - Strided access
  - Loop patterns
  - Random access
  - Matrix traversals

## Contributing

Contributions are welcome! Some areas for improvement:

1. Additional pattern detection algorithms
2. Performance optimizations
3. Support for more complex memory access patterns
4. Integration with hardware simulators
5. Enhanced evaluation metrics

## Implementation Details

### Dataset Processing

The system processes memory access patterns by:
1. Calculating address deltas
2. Normalizing the data
3. Creating sequence-target pairs
4. Handling variable-length sequences

### Training Process

The training pipeline includes:
1. Batch processing with variable sizes
2. Learning rate scheduling
3. Early stopping
4. Gradient clipping
5. Loss monitoring

### Prediction Strategy

The prediction workflow:
1. Pattern detection attempt
2. Strategy selection
3. Prediction generation
4. Confidence scoring
5. Result verification

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This implementation draws inspiration from:
- Hardware prefetching techniques
- LSTM applications in sequence prediction
- Pattern recognition algorithms in computer architecture