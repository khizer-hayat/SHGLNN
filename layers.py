import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NodeConvolution, self).__init__()
        self.W_v = nn.Linear(in_channels, out_channels)
        self.W_e = nn.Linear(in_channels, out_channels)
        self.P = nn.Parameter(torch.FloatTensor(out_channels, out_channels))
        self.activation = nn.ReLU()

    def forward(self, x, H, D_v_inv, D_e_inv):
        out_1 = self.W_v(x)
        out_2 = self.W_e(x)
        out = self.activation(self.P @ (D_v_inv @ H @ out_1 @ D_e_inv @ H.T @ x + D_v_inv @ H @ out_2 @ D_e_inv @ H.T @ x))
        return out

class HyperedgeConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HyperedgeConvolution, self).__init__()
        self.W_q = nn.Linear(in_channels, out_channels)
        self.W_o = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x, M, K, alpha):
        out = self.activation((1/(M.size(0)+1)) * (torch.sum(self.W_q(x) * alpha, dim=0) + self.W_o(x)))
        return out

class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.attention_vector = nn.Linear(in_channels, out_channels)

    def forward(self, v, K):
        attention_score = F.softmax(self.attention_vector(v).mm(K.T), dim=-1)
        return attention_score
