def matrix_to_edge_index(matrix):
    """Convert a matrix to edge_index format."""
    src, dst = matrix.nonzero()
    return torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0)

def process_dataset_for_hyperedges(dataset_name, edge_type):
    # Load original dataset
    dataset = TUDataset(root='./data', name=dataset_name)[0]  # Assuming single graph in dataset
    
    if edge_type == "intra":
        k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')
        edge_index = matrix_to_edge_index(k_matrix)
    
    elif edge_type == "inter":
        s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')
        edge_index = matrix_to_edge_index(s_matrix)
    
    elif edge_type == "both":
        k_matrix = np.loadtxt("kMatrix_data.csv", delimiter=',')
        s_matrix = np.loadtxt(f'{dataset_name}_sMatrix_data.csv', delimiter=',')
        
        k_edge_index = matrix_to_edge_index(k_matrix)
        s_edge_index = matrix_to_edge_index(s_matrix)
        
        # Adjust node indices to ensure no overlaps between K and S representations
        s_edge_index[0] += k_matrix.shape[0]
        s_edge_index[1] += k_matrix.shape[1]

        edge_index = torch.cat([k_edge_index, s_edge_index], dim=1)

    dataset.edge_index = edge_index
    return dataset
