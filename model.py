import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import NodeConv, HyperedgeConv

class SHGLNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SHGLNN, self).__init__()
        self.node_conv = NodeConv(in_dim, hidden_dim)
        self.hyperedge_conv = HyperedgeConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_dim, out_dim)

    def forward(self, x, H, hyperedge_weights):
        x = self.node_conv(x, H)
        x = self.hyperedge_conv(x, H, hyperedge_weights)
        x = self.dropout(x)
        return self.fc(x)
