import torch.optim as optim
from model import SHGLNN
from dataset_loader import CustomDatasetLoader

def contrastive_loss(output_original, output_augmented, temperature=0.5):
    numerator = torch.exp(F.cosine_similarity(output_original, output_augmented) / temperature)
    denominator = numerator + torch.sum(torch.exp(output_original.mm(output_augmented.T) / temperature))
    loss = -torch.log(numerator / denominator)
    return loss

def train(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        outputs_original = model(data.nodes, data.edges, data.hyperedge_weights, data.D_v_inv, data.D_e_inv, data.E_intra, data.E_inter)
        outputs_augmented = model(data.nodes_augmented, data.edges_augmented, data.hyperedge_weights_augmented, data.D_v_inv, data.D_e_inv, data.E_intra, data.E_inter)
        loss = contrastive_loss(outputs_original, outputs_augmented)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Define model, optimizer, and other configurations
model = SHGLNN(128, 256)
optimizer = optim.Adam(model.parameters(), lr=0.005)

# Load dataset
name = 'MUTAG'
dataset_loader = CustomDatasetLoader(name, f"../datasets/{name}")
data_loader = dataset_loader.get_dataloader()

# train the model
for epoch in range(250):
    loss = train(model, data_loader, optimizer)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
