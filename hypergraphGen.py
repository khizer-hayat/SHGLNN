import hypernetx as hnx
import numpy as np

def matrix_to_dict(matrix):
    """Convert a matrix (K or S) to a dictionary format for hypergraph generation."""
    connections = {}
    num_rows, num_cols = matrix.shape

    for i in range(num_rows):
        connected_nodes = [j for j in range(num_cols) if matrix[i, j] == 1]
        if connected_nodes:
            connections[f"E{i}"] = connected_nodes

    return connections

def process_dataset(dataset_name):
    # Load K and S matrices from saved files.
    k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')  # Assuming K matrix filename remains the same for all datasets
    s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')

    # Convert matrices to dictionary format.
    k_connections = matrix_to_dict(k_matrix)
    s_connections = matrix_to_dict(s_matrix)

    # Combine both dictionaries (assuming no key overlaps).
    hypergraph_connections = {**k_connections, **s_connections}

    # Construct the hypergraph using HyperNetX
    H = hnx.Hypergraph(hypergraph_connections)

def main():
    datasets = ["MUTAG", "NCI1", "PROTEINS", "ENZYMES", "IMDB-BINARY", "COLLAB"]

    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
