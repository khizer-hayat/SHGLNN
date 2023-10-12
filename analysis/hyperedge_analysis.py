import hypernetx as hnx
import numpy as np
import torch
from torch_geometric.data import Data, DataLoader
from model import SHGLNN
from torch_geometric.datasets import TUDataset
from sklearn.metrics import f1_score

# Define other necessary functions and global variables here
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEARNING_RATE = 0.005
EPOCHS = 250
BATCH_SIZE = 32
criterion = torch.nn.CrossEntropyLoss()

# ... [Other global settings and functions]

def matrix_to_dict(matrix):
    connections = {}
    num_rows, num_cols = matrix.shape
    for i in range(num_rows):
        connected_nodes = [j for j in range(num_cols) if matrix[i, j] == 1]
        if connected_nodes:
            connections[f"E{i}"] = connected_nodes
    return connections

def hypergraph_to_data(H):
    edge_index = []
    for edge, nodes in H.edges.items():
        for node in nodes:
            edge_index.append([node, edge])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    data = Data(edge_index=edge_index)
    return data

def process_dataset_for_analysis(dataset_name):
    k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')  
    s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')
    k_connections = matrix_to_dict(k_matrix)
    s_connections = matrix_to_dict(s_matrix)
    hypergraph_connections = {**k_connections, **s_connections}
    H = hnx.Hypergraph(hypergraph_connections)
    data = hypergraph_to_data(H)
    return data


def train_and_evaluate(dataset_name, data):
 
    dataset = TUDataset(root='./data', name=dataset_name)
    model = SHGLNN(in_channels=dataset.num_node_features, hidden_channels=256, out_channels=dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Split the dataset
    train_dataset = dataset[:int(0.3 * len(dataset))]
    val_dataset = dataset[int(0.3 * len(dataset)):int(0.4 * len(dataset))]
    test_dataset = dataset[int(0.4 * len(dataset)):]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    for run in range(NUM_RUNS):
        for epoch in range(EPOCHS):
            # --------- Training ---------
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            train_predictions = []
            train_labels = []

            for batch in train_loader:
                optimizer.zero_grad()
                data = batch.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += data.y.size(0)
                train_correct += (predicted == data.y).sum().item()
                train_predictions.extend(predicted.cpu().numpy())
                train_labels.extend(data.y.cpu().numpy())

            train_accuracy = 100 * train_correct / train_total
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f} %, Train F1: {train_f1:.4f}')

            # --------- Validation ---------
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_labels = []
            
            for data in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += data.y.size(0)
                val_correct += (predicted == data.y).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                val_labels.extend(data.y.cpu().numpy())

            val_accuracy = 100 * val_correct / val_total
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            print(f'Run [{run+1}/{NUM_RUNS}], Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}, Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f} %, Validation F1: {val_f1:.4f}')

        # --------- Testing ---------
        test_loss = 0
        test_correct = 0
        test_total = 0
        test_predictions = []
        test_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data.y)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += data.y.size(0)
                test_correct += (predicted == data.y).sum().item()
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(data.y.cpu().numpy())

        test_accuracy = 100 * test_correct / test_total
        test_f1 = f1_score(test_labels, test_predictions, average='weighted')
        print(f'Run [{run+1}/{NUM_RUNS}], Testing Loss: {test_loss/len(test_loader):.4f}, Testing Accuracy: {test_accuracy:.2f} %, Testing F1: {test_f1:.4f}')

def main():
    datasets = ["MUTAG", "PROTEINS", "IMDB-BINARY"]
    for dataset in datasets:
        data = process_dataset_for_analysis(dataset)
        train_and_evaluate(dataset, data)

if __name__ == "__main__":
    main()
