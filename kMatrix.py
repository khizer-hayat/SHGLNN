# Generate K matrix from the following datasets
# - COLLAB
# - IMDB
# - PROTEINS
# - MUTAG
# The K matrix calculates the degree between the different nodes in the graph such as source and target node.

import pandas as pd
import numpy as np
import torch_geometric
from get_dataset import Dataset
import networkx as nx
import argparse

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type = str, default = "MUTAG", help = "the dataset name")
    parser.add_argument("--khops", type = int, default = 2, help = "K hope value for generating k matrix")
    parser.add_argument("--graph_index", type = int, default = 0, help = "sub graph index")
    opt = parser.parse_args()
    return opt

def tensor_to_list(tensor):
    array = tensor.numpy()
    array2list = list(array)
    return array2list

def get_src_target_nodes(graph):
    """
    Params:
    graph: torch_geometric.datasets.TUDataset
    return:
    source_nodes: list
    target_nodes: list
    """
    source_nodes = graph["edge_index"][0, :]
    target_nodes = graph["edge_index"][1, :]
    source_nodes = tensor_to_list(source_nodes)
    target_nodes = tensor_to_list(target_nodes)
    return source_nodes, target_nodes

def create_graph(graph, source_nodes, target_nodes):
    """
    Parameters
    ----------
    graph: torch_geometric
    source_nodes: list
    target_nodes: list

    Return
    ------
    spl: nx.Graph.shortest_path
    new_graph: nx.Graph
    k_matrix: np.ndarray
    """
    unique_nodes = list(np.unique(graph["edge_index"]))
    num_nodes = len(unique_nodes)
    k_matrix = np.zeros((num_nodes, num_nodes))
    new_graph = nx.Graph()
    new_graph.add_nodes_from(unique_nodes)
    edge_info = list(zip(source_nodes, target_nodes))
    new_graph.add_edges_from(edge_info)
    spl = dict(nx.all_pairs_shortest_path_length(new_graph))
    return spl, k_matrix, unique_nodes

def calculateLength(a, b, spl):
    try:
        return spl[a][b]
    except KeyError:
        return 0

def save_data_csv(K, nodes_list):
    '''convert to graph data to a csv file for graph generation
    Args: K (numpy array of graph relations) 2708x2708
          nodes_list: (list) of nodes
    '''
    Kmatrix = {}
    for i, cols in enumerate(nodes_list):
        Kmatrix[f"{cols}"] = K[:, i].tolist()

    graph_data = pd.DataFrame(Kmatrix)
    drop_col = graph_data.columns.tolist()[0]
    graph_data = graph_data.drop(drop_col, axis=1)
    graph_data.to_csv("grah_data.csv", sep=',')

def generate_kmatrix(initial_k_matrix,
                     unique_nodes,
                     spl,
                     k_hops = None):
    kmatrix = initial_k_matrix
    unique_nodes.sort()
    if k_hops is not None:
        for i, row in enumerate(unique_nodes):
            for j, col in enumerate(unique_nodes):
                length = calculateLength(row, col, spl)
                if length <= k_hops and length != 0:
                    kmatrix[i, j] = 1
                else:
                    kmatrix[i, j] = 0
    # saving the k matrix into a csv file...
    save_data_csv(kmatrix, unique_nodes)
    return kmatrix

if __name__ == "__main__":
    args = read_args()
    NAME = args.name.upper()
    dataset = Dataset(f"{NAME}", save_dir= f"../datasets/{NAME}")
    data = dataset.return_dataset()
    first_graph = data[args.graph_index]
    src_nodes, target_nodes = get_src_target_nodes(first_graph)
    spl, kmatrix, unique_nodes = create_graph(first_graph, src_nodes, target_nodes)
    kmatrix = generate_kmatrix(kmatrix, unique_nodes, spl, args.khops)
    print(f"K matrix generated: \n{kmatrix}")
    print(f'K matrix shape: {kmatrix.shape}')
