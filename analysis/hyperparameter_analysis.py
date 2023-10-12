import numpy as np
import hypernetx as hnx
from train import train_and_evaluate  # Assuming train_and_evaluate is in train.py

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
    k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')  # Assuming K matrix filename remains the same for all datasets
    s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')

    k_connections = matrix_to_dict(k_matrix)
    s_connections = matrix_to_dict(s_matrix)

    hypergraph_connections = {**k_connections, **s_connections}
    H = hnx.Hypergraph(hypergraph_connections)

    return H

def compute_alpha(v_i, v_j, K_i_star):
    """
    Compute attention value alpha for nodes v_i and v_j using Eq. 3.
    For simplicity, we'll use a dot product as a representation of the attention mechanism.
    """
    # Vectors for v_i and v_j are the embeddings from a prior layer or input features
    v_i_embedding = K_i_star[v_i]
    v_j_embedding = K_i_star[v_j]

    # Dot product as a basic representation of attention
    attention_value = np.dot(v_i_embedding, v_j_embedding)
    
    # Normalizing the attention value
    normalized_attention_value = np.tanh(attention_value)

    return normalized_attention_value

def calculate_drop_probability(alpha, deg_h, rho_lambda, rho_eta):
    """Calculate the dropout probability for a node based on Eq. 13."""
    return (1 - alpha) / (1 + deg_h) * rho_lambda + rho_eta

def update_hyperedge_weight(w, e, alpha, R):
    """Update the weight of a hyperedge based on Eq. 14."""
    sum_val = sum([alpha[i, r] for r in R for i in e])
    return w + 1/len(e) * sum_val

def augment_hypergraph(H, alpha, rho_lambda, rho_eta):
    """Generate augmented hypergraph based on dropout probabilities and update weights."""
    R = set()  # Set of removed nodes
    for e, nodes in H.items():
        w = H.get_edge_data(e)["weight"]
        for node in nodes:
            deg_h = len(H.incident_edges(node))  # get the hyperdegree
            drop_prob = calculate_drop_probability(alpha[node], deg_h, rho_lambda, rho_eta)
            if np.random.rand() < drop_prob:
                nodes.remove(node)
                R.add(node)
        # Update hyperedge weight
        H.set_edge_data(e, {"weight": update_hyperedge_weight(w, nodes, alpha, R)})
    return H

def main():
    H_original = process_dataset('COLLAB')
    
    # Assuming a matrix representation for your hypergraph
    number_of_nodes = H_original.number_of_nodes()
    alpha_matrix = np.zeros((number_of_nodes, number_of_nodes))

    for i in range(number_of_nodes):
        for j in H_original[i]:  # j represents neighbors of i
            alpha_matrix[i, j] = compute_alpha(i, j, H_original[i])  # i's row in K matrix

    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    for rho_lambda in values:
        for rho_eta in values:
            H_augmented = augment_hypergraph(H_original, alpha_matrix, rho_lambda, rho_eta)
            accuracy, f1 = train_and_evaluate(H_augmented)
            print(f"Rho_lambda: {rho_lambda}, Rho_eta: {rho_eta}, Accuracy: {accuracy}, F1-score: {f1}")

if __name__ == "__main__":
    main()
