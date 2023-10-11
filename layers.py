import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeConv, self).__init__()
        self.W_v = nn.Linear(in_dim, out_dim)
        self.W_e = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x, H):
        out = self.activation(self.W_v(H @ x) + self.W_e(H.T @ x))
        return out

class HyperedgeConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HyperedgeConv, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()

    def forward(self, x, H, hyperedge_weights):
        out = self.activation(self.W(H @ x))
        return out * hyperedge_weights
