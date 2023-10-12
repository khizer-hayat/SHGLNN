import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SHGLNN_Task2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SHGLNN_Task2, self).__init__()
        
        # Initial GCN layer
        self.initial_gcn = GCNConv(in_channels=in_channels, out_channels=hidden_channels)
        
        # Attention Mechanism
        self.attention_layer = torch.nn.Linear(hidden_channels, 1)
        
        # Hyperedge Aggregation
        self.hyperedge_weight = torch.nn.Parameter(torch.ones(size=(hidden_channels, 1), dtype=torch.float32))
        
        # Hidden GCN layers (assuming 2 for this demonstration)
        self.hidden_gcn1 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
        self.hidden_gcn2 = GCNConv(in_channels=hidden_channels, out_channels=hidden_channels)
        
        # Output layer
        self.out = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Initial GCN layer
        x = self.initial_gcn(x, edge_index)
        x = F.relu(x)
        
        # Attention Mechanism (assuming you're using it here)
        attn_weights = F.softmax(self.attention_layer(x), dim=0)
        x = x * attn_weights
        
        # Hyperedge Aggregation (assuming you're using it here)
        x = torch.mm(x, self.hyperedge_weight)
        
        # Hidden GCN layers
        x = self.hidden_gcn1(x, edge_index)
        x = F.relu(x)
        x = self.hidden_gcn2(x, edge_index)
        x = F.relu(x)
        
        # Node-level average pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.out(x)
        
        return F.log_softmax(x, dim=1)

# If you want to test or instantiate this model
# Make sure to pass the necessary arguments like in_channels, hidden_channels, and out_channels.
