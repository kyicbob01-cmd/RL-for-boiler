"""
V3.0 RCP Policy Network
Return-Conditioned Policy: (state, target_cost) -> action
"""

import torch
import torch.nn as nn

class RCPolicy(nn.Module):
    """
    Return-Conditioned Policy Network
    Input: 7D = 6D state + 1D target_cost
    Output: 1D action (power %)
    """
    
    def __init__(self, state_dim=6, hidden_dim=512):
        super().__init__()
        
        # Input: state (6D) + target_cost (1D) = 7D
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output 0-1 for power percent
        )
    
    def forward(self, state, target_cost):
        """
        Forward pass
        Args:
            state: (batch, 6) normalized state vector
            target_cost: (batch, 1) target cost (normalized by /50)
        Returns:
            action: (batch, 1) power percent [0, 1]
        """
        cost_norm = target_cost / 50.0  # Normalize cost
        x = torch.cat([state, cost_norm], dim=-1)
        return self.net(x)
