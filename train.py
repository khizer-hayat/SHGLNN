import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

from model import SHGLNN
from layers import NodeConv, HyperedgeConv, Attention
from dataset_loader import GraphDataset

# Hyperparameters and settings
EPOCHS = 250
BATCH_SIZE = 32
LEARNING_RATE = 0.005
DATASETS = ['MUTAG', 'NCI1', 'PROTEINS', 'ENZYMES', 'IMDB-BINARY', 'COLLAB']
NUM_RUNS = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_one_epoch(model, loader, optimizer):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        out1, out2 = model(data)
        loss = model.compute_loss(out1, out2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data in loader:
            out1, _ = model(data)
            pred = out1.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return accuracy, f1

def main():
    for dataset_name in DATASETS:
        accuracies = []
        f1_scores = []
        losses = []

        for _ in range(NUM_RUNS):
            dataset = GraphDataset(name=dataset_name)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

            model = SHGLNN().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

            for epoch in range(EPOCHS):
                loss = train_one_epoch(model, loader, optimizer)

            accuracy, f1 = evaluate(model, loader)
            accuracies.append(accuracy)
            f1_scores.append(f1)
            losses.append(loss)

        print(f"Dataset: {dataset_name}")
        print(f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f}")
        print(f"F1-Score: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}")
        print(f"Loss: {np.mean(losses):.4f} +/- {np.std(losses):.4f}")
        print("-------------------------------------------------")

if __name__ == "__main__":
    main()
