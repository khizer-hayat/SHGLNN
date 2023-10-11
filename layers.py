import torch
import torch.nn.functional as F
from torch.nn import Linear, Parameter

class NodeConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NodeConvolution, self).__init__()
        self.W_v = Parameter(torch.Tensor(in_dim, out_dim))
        self.W_e = Parameter(torch.Tensor(in_dim, out_dim))
        self.P = Parameter(torch.Tensor(out_dim, out_dim))
        
    def forward(self, x, H, D_v_inv, D_e_inv):
        XW_v = torch.mm(x, self.W_v)
        XW_e = torch.mm(x, self.W_e)
        out = torch.mm(D_v_inv, H) @ XW_v + torch.mm(D_v_inv, H) @ XW_e
        out = torch.mm(out, self.P)
        return F.relu(out)

class HyperedgeConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(HyperedgeConvolution, self).__init__()
        self.W_q = Parameter(torch.Tensor(in_dim, out_dim))
        self.W_o = Parameter(torch.Tensor(in_dim, out_dim))

    def forward(self, x, M, K, alpha):
        z = torch.mm(M, x) @ self.W_q
        z = torch.div(z, len(K))
        return F.relu(z + torch.mm(x, self.W_o))

class GraphAttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super(GraphAttentionPooling, self).__init__()
        self.weight = Parameter(torch.Tensor(in_dim, 1))

    def forward(self, x, hyperedge_types):
        attn_weights = F.softmax(self.weight(x), dim=0)
        return torch.mm(attn_weights.T, x)

class Attention(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Attention, self).__init__()
        self.W_K = Parameter(torch.Tensor(in_dim, out_dim))

    def forward(self, x, K):
        xK = torch.mm(x, self.W_K)
        alpha = F.softmax(torch.mm(xK, K.T), dim=1)
        return alpha

class ContextAwarePooling(nn.Module):
    def __init__(self, in_dim):
        super(ContextAwarePooling, self).__init__()
        self.W_intra = Parameter(torch.Tensor(in_dim, 1))
        self.W_inter = Parameter(torch.Tensor(in_dim, 1))

    def forward(self, x, E_intra, E_inter):
        alpha = F.softmax(self.W_intra(x), dim=0)
        beta = F.softmax(self.W_inter(x), dim=0)
        intra_rep = torch.mm(alpha.T, E_intra)
        inter_rep = torch.mm(beta.T, E_inter)
        return intra_rep + inter_rep
