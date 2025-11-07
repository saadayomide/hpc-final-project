"""
Diffusion Convolutional Recurrent Neural Network (DCRNN) Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DCGRUCell(nn.Module):
    """Diffusion Convolutional Gated Recurrent Unit Cell"""
    
    def __init__(self, input_dim, hidden_dim, num_nodes, num_supports=2):
        super(DCGRUCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_supports = num_supports
        
        # Input gate
        self.input_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # Reset gate
        self.reset_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        # New gate
        self.new_gate = nn.Linear(input_dim + hidden_dim, hidden_dim)
        
        # Diffusion convolution for gates
        self.diff_conv = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_supports)
        ])
        
    def forward(self, x, h, supports):
        """
        Args:
            x: Input features [batch, num_nodes, input_dim]
            h: Hidden state [batch, num_nodes, hidden_dim]
            supports: List of support matrices [batch, num_nodes, num_nodes]
        """
        xh = torch.cat([x, h], dim=-1)  # [batch, num_nodes, input_dim + hidden_dim]
        
        # Gates
        r = torch.sigmoid(self.reset_gate(xh))  # Reset gate
        u = torch.sigmoid(self.input_gate(xh))  # Update gate
        c = torch.tanh(self.new_gate(xh))       # Candidate
        
        # Apply diffusion convolution to hidden state
        h_diff = h
        for i, support in enumerate(supports):
            h_diff = h_diff + self.diff_conv[i](torch.matmul(support, h))
        
        # New hidden state
        h_new = u * h + (1 - u) * c
        return h_new


class DCRNN(nn.Module):
    """
    Diffusion Convolutional Recurrent Neural Network
    """
    
    def __init__(self, input_dim, hidden_dim, num_nodes, num_layers=2, output_dim=1, 
                 num_supports=2, dropout=0.0):
        super(DCRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Build RNN layers
        self.cells = nn.ModuleList()
        self.cells.append(DCGRUCell(input_dim, hidden_dim, num_nodes, num_supports))
        for _ in range(num_layers - 1):
            self.cells.append(DCGRUCell(hidden_dim, hidden_dim, num_nodes, num_supports))
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, supports, h0=None):
        """
        Args:
            x: Input sequence [batch, seq_len, num_nodes, input_dim]
            supports: List of support matrices [num_supports, num_nodes, num_nodes]
            h0: Initial hidden states [num_layers, batch, num_nodes, hidden_dim]
        Returns:
            output: Output predictions [batch, num_nodes, output_dim]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # Initialize hidden states
        if h0 is None:
            h = [torch.zeros(batch_size, num_nodes, self.hidden_dim, 
                           device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        else:
            h = [h0[i] for i in range(self.num_layers)]
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, num_nodes, input_dim]
            
            # Process through layers
            for i, cell in enumerate(self.cells):
                h[i] = cell(x_t, h[i], supports)
                x_t = h[i]  # Input to next layer is output of current layer
                if i < len(self.cells) - 1:
                    h[i] = self.dropout(h[i])
        
        # Output layer
        output = self.output_layer(h[-1])  # [batch, num_nodes, output_dim]
        return output
