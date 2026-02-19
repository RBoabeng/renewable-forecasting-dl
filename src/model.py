
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class MicrogridLSTM(nn.Module):
    # Added a default dropout of 0.2 (20% of neurons turned off during training)
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(MicrogridLSTM, self).__init__()
        
        # We pass the dropout directly into the PyTorch LSTM module
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out