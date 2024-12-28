### model.py ###
import torch
import torch.nn as nn

class LSTMPrefetcher(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super(LSTMPrefetcher, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Initial feature extraction
        self.feature_extract = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, seq_len]
        batch_size = x.size(0)
        
        # Feature extraction
        x = self.feature_extract(x.unsqueeze(-1))
        
        # LSTM
        h0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Attention
        attention_weights = self.attention(lstm_out)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Output
        out = self.fc_layers(context)
        return out