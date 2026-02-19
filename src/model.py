
import torch
import torch.nn as nn

class MicrogridLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MicrogridLSTM, self).__init__()
        
        # Define the LSTM layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer to map LSTM output to our energy prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # Initial hidden and cell states are handled by PyTorch if not provided (zeros)
        # out shape: (batch_size, sequence_length, hidden_size)
        out, _ = self.lstm(x)
        
        # We only care about the output of the VERY LAST time step (the 24th hour)
        # We use out[:, -1, :] to get that last step's hidden state
        out = self.fc(out[:, -1, :])
        
        return out