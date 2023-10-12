import hypernetx as hnx
import numpy as np
import random

def matrix_to_dict(matrix, dropout_prob=0.0):
    """Convert a matrix (K or S) to a dictionary format for hypergraph generation with dropout."""
    connections = {}
    num_rows, num_cols = matrix.shape

    for i in range(num_rows):
        connected_nodes = [j for j in range(num_cols) if matrix[i, j] == 1]
        if connected_nodes:
            if dropout_prob > 0:
                # Drop nodes with given probability
                connected_nodes = [node for node in connected_nodes if random.uniform(0, 1) > dropout_prob]
            connections[f"E{i}"] = connected_nodes

    return connections

def process_dataset(dataset_name, intra_dropout=0.0, inter_dropout=0.0):
    # Load K and S matrices from saved files.
    k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')  # Assuming K matrix filename remains the same for all datasets
    s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')

    # Convert matrices to dictionary format with dropout.
    k_connections = matrix_to_dict(k_matrix, dropout_prob=intra_dropout)
    s_connections = matrix_to_dict(s_matrix, dropout_prob=inter_dropout)

    # Combine both dictionaries (assuming no key overlaps).
    hypergraph_connections = {**k_connections, **s_connections}

    # Construct the hypergraph using HyperNetX
    H = hnx.Hypergraph(hypergraph_connections)
    return H
