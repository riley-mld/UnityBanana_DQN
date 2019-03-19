# Import packages
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Q Network."""
    
    def __init__(self, state_size, action_size, seed, use_dueling):
        """Initilize the network."""
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Initialize parameters
        self.state_size = state_size
        self.action_size = action_size
        self.drop_prob = 0.5
        hidden_layers = [512, 128]
        self.use_dueling = use_dueling
        
        # Initialize layers
        self.hidden_layers = nn.ModuleList([nn.Linear(self.state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.action_value = nn.Linear(hidden_layers[-1], self.action_size)
        self.state_value = nn.Linear(hidden_layers[-1], 1)
        
    def forward(self, state):
        """Define the forward pass."""
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        
        if self.use_dueling:
            return self.action_value(state) + self.state_value(state)
        
        else:
            return self.action_value(state)
            