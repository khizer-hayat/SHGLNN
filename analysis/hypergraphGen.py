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

class HypergraphGen:
    def __init__(self, edge_type='both'):
        self.edge_type = edge_type

    def generate_hypergraph(self, dataset_name):
        # Load K and S matrices from saved files.
        k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')
        s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')

        if self.edge_type == 'intra':
            # Convert K matrix to dictionary format.
            hypergraph_connections = matrix_to_dict(k_matrix)
        elif self.edge_type == 'inter':
            # Convert S matrix to dictionary format.
            hypergraph_connections = matrix_to_dict(s_matrix)
        else:  # both
            k_connections = matrix_to_dict(k_matrix)
            s_connections = matrix_to_dict(s_matrix)
            hypergraph_connections = {**k_connections, **s_connections}

        # Construct the hypergraph using HyperNetX
        H = hnx.Hypergraph(hypergraph_connections)
        return H  # Return the hypergraph.
