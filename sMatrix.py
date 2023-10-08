import torch_geometric
import numpy as np
from t_nodes_graph import total_nodes_in_a_graph
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type = str, default="", help = 'name of the dataset')
    parser.add_argument('--save', type = str, default="", help = 'save dir for the dataset')
    opt = parser.parse_args()
    return opt

def generate_s_matrix(name, save):
    """
    parameters
    ----------
    name: str --> name of the dataset
    save: str --> save directory

    :return
    -------
    S matrix of size (num_nodes, num_nodes) while
    num_nodes are the total nodes in the dataset
    """
    nodes, data = total_nodes_in_a_graph(dataset_name= name, save_dir = save)
    S_matrix = np.zeros((nodes, nodes))
    height, width = 0, 0
    for i in range(len(data)):
        dataset = data[i]
        rows, cols = dataset['x'].shape
        for row in range(rows):
            if row != 0 or i != 0:
                height +=1
            query_node_encoding = dataset['x'][row]
            for j in range(len(data)):
                sub_graph = data[j]
                m, n = sub_graph['x'].shape
                for k in range(m):
                    if k != 0 or j != 0:
                        width +=1
                    value_node_encoding = sub_graph['x'][k]
                    if width >= nodes:
                        width = 0
                    if np.array_equal(query_node_encoding.numpy(), value_node_encoding.numpy()):
                        # print(height, width)
                        S_matrix[height, width] = 1
                    else:
                        print(height, width)
                        # S_matrix[height, width] = 0
    return S_matrix

if __name__ == "__main__":
    args = read_args()
    name = args.name
    save_dir = args.save
    print('Generating S matrix...')
    S_matrix = generate_s_matrix(name, save_dir)
    np.savetxt(f'{name}_s_matrix.csv', S_matrix, delimiter=',')
    print('done!!!')
