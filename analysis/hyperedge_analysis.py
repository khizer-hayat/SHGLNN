import hypernetx as hnx
import numpy as np
import torch
from torch_geometric.data import Data
from train import train_and_evaluate

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

def main():
    datasets = ["MUTAG", "PROTEINS", "IMDB-BINARY"]
    for dataset in datasets:
        data = process_dataset_for_analysis(dataset)
        train_and_evaluate(dataset, data)  # Adjusted to pass the processed data

if __name__ == "__main__":
    main()
