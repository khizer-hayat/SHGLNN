import torch
import torch_geometric
from torch_geometric.datasets import TUDataset
from model import SHGLNN

# Load data (for now MUTAG as an example)
dataset_name = 'MUTAG'
dataset = TUDataset(root=f"../datasets/{dataset_name}", name=dataset_name)
data = dataset[0]  # Using only one graph for simplicity

# Hyperparameters
input_dim = data.num_node_features
hidden_dim = 128
output_dim = dataset.num_classes
learning_rate = 0.005

# Initialize model and optimizer
model = SHGLNN(input_dim, hidden_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = nn.CrossEntropyLoss()

# Sample training loop
for epoch in range(100):  # Placeholder epoch value
    optimizer.zero_grad()
    out = model(data.x, data.edge_index, hyperedge_weights)  # Placeholder hyperedge_weights
    loss = loss_function(out, data.y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
