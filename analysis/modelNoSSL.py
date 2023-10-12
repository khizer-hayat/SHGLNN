import torch.nn as nn
from layers import NodeConvolution, HyperedgeConvolution, Attention, ContextAwarePooling

class SHGLNN_NoSL(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(SHGLNN_NoSL, self).__init__()
        self.node_conv = NodeConvolution(in_dim, hidden_dim)
        self.hyperedge_conv = HyperedgeConvolution(hidden_dim, hidden_dim)
        self.attention = Attention(hidden_dim, hidden_dim)
        self.context_pooling = ContextAwarePooling(hidden_dim)

    def forward(self, x, H, K, M, D_v_inv, D_e_inv, E_intra, E_inter):
        # NOTE: This will be similar to your original model but without the specific SSL augmentations.
        x = self.node_conv(x, H, D_v_inv, D_e_inv)
        alpha = self.attention(x, K)
        x = self.hyperedge_conv(x, M, K, alpha)
        graph_embedding = self.context_pooling(x, E_intra, E_inter)
        return graph_embedding
